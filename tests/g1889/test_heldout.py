"""The held-out loss (1889/v1-design.md §4): each epoch, before it trains, a
sample of its new rows scored by the weights that have yet to train on them.

- A run's ``heldout/`` tags equal value MSE and policy KL computed here, from the
  model's outputs on the epoch's rows, overall and for each split; the rows are
  that epoch's alone; started games' rows are apart. The same run without the
  held-out loss writes no tag and ends with the same weights.
- A sample short of the epoch draws only its rows, the same ones each time, and
  ``heldout_rows`` 0 scores nothing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from games import RELATION_RECORD_WIDTH, get_game
from nn import create_model, get_model_input_spec
from nn.model_contract import flatten_relation_records
from train import main as main_module
from train.checkpoint import load_checkpoint
from train.config import TrainingConfig
from train.replay_buffer import FLAG_FROM_POSITION, ReplayBuffer
from train.tb_reader import read_tb_scalars
from train.trainer import Trainer

from tests.g1889.test_capacity_probe import _play
from tests.g1889.test_training_run import _smoke_config, _train

MAX_PLAYERS = 5


def _buffer(config: TrainingConfig) -> ReplayBuffer:
    game = get_game(config)
    return ReplayBuffer(
        game, config.buffer_capacity, game.state_size(MAX_PLAYERS), MAX_PLAYERS,
        min_players=3, max_players=MAX_PLAYERS,
    )


def _direct(model: torch.nn.Module, config: TrainingConfig, rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Each row's policy KL(target || model) and squared value error summed over
    its real players, from the model's outputs in float64."""
    game = get_game(config)
    spec = get_model_input_spec(config)
    model.eval()
    kls, sqerrs = [], []
    for start in range(0, len(rows["states"]), 128):
        part = {name: column[start:start + 128] for name, column in rows.items()}
        states = [state for state in part["states"]]
        tokens = np.empty((len(states), game.num_tokens(MAX_PLAYERS), spec.token_dim), np.float32)
        game.fill_tokens_batch(states, tokens, max_players=MAX_PLAYERS, layout_version=spec.layout_version)
        records = np.empty((len(states), game.max_relation_edges, RELATION_RECORD_WIDTH), np.uint8)
        game.fill_relations_sparse_batch(states, records, max_players=MAX_PLAYERS)
        with torch.no_grad():
            logits, values = model(
                torch.from_numpy(tokens), torch.from_numpy(part["legal_masks"]).bool(),
                torch.from_numpy(flatten_relation_records(records)),
            )
        log_policy = torch.log_softmax(logits.double(), dim=-1).numpy()
        target = part["policy_targets"].astype(np.float64)
        positive = target > 0
        terms = np.zeros_like(target)
        terms[positive] = target[positive] * (np.log(target[positive]) - log_policy[positive])
        kls.append(terms.sum(axis=1))
        real = np.arange(MAX_PLAYERS)[None] < part["player_counts"][:, None]
        sqerrs.append(((values.double().numpy() - part["value_targets"]) ** 2 * real).sum(axis=1))
    return np.concatenate(kls), np.concatenate(sqerrs)


def _expected(
    model: torch.nn.Module, config: TrainingConfig, rows: dict[str, np.ndarray],
) -> dict[str, float]:
    """The held-out tags of ``rows``, one epoch's, oldest first."""
    kl, sqerr = _direct(model, config, rows)
    players = rows["player_counts"].astype(np.float64)
    started = (rows["flags"] & FLAG_FROM_POSITION) != 0
    # Each row's quarter of its game: rows of a game run together under its id.
    quarter = np.empty(len(kl), dtype=np.int64)
    for game_id in np.unique(rows["game_ids"]):
        of_game = np.flatnonzero(rows["game_ids"] == game_id)
        quarter[of_game] = np.arange(len(of_game)) * 4 // len(of_game)
    groups = {"": np.ones(len(kl), dtype=bool), "_opening": ~started, "_started": started}
    groups |= {f"_q{q + 1}": ~started & (quarter == q) for q in range(4)}
    phase_names = get_game(config).policy.phase_names
    groups |= {f"_{name}": ~started & (rows["phase_ids"] == i) for i, name in enumerate(phase_names)}
    expected = {"heldout/rows": float(len(kl)), "heldout/rows_started": float(started.sum())}
    for suffix, rows_in in groups.items():
        if rows_in.any():
            expected[f"heldout/value_mse{suffix}"] = sqerr[rows_in].sum() / players[rows_in].sum()
            expected[f"heldout/policy_kl{suffix}"] = kl[rows_in].mean()
    return expected


def test_a_run_logs_each_epochs_heldout_loss_and_trains_as_without_it(tmp_path, monkeypatch, capsys):
    """Two epochs of four games, one started from a position, every row of an
    epoch held out: each epoch's tags are the weights it began with, scored on
    its own rows."""
    def run(name: str, heldout_rows: int) -> tuple[Path, dict[str, list[tuple[int, float]]]]:
        out = tmp_path / name
        fields = {
            **_smoke_config(out), "games_per_epoch": 4, "num_epochs": 2, "buffer_capacity": 65_536,
            "start_positions_path": "g1889/data/human_starts.npz", "human_start_fraction": 0.25,
            "heldout_rows": heldout_rows,
        }
        config_path = out / "config.json"
        out.mkdir()
        config_path.write_text(json.dumps(fields))
        _train(monkeypatch, "--config", str(config_path))
        capsys.readouterr()
        return out / "checkpoints", read_tb_scalars(str(out / "runs"))

    checkpoints, scalars = run("heldout", 1_000_000)
    final = load_checkpoint(checkpoints / "checkpoint_epoch_0002.pt", torch.device("cpu"))
    config = TrainingConfig.from_json(final["config_json"])
    buffer = _buffer(config)
    assert buffer.load(checkpoints / "replay_buffer") > 0
    rows = buffer.rows_oldest_first()
    second_epoch = buffer.epochs[1][1]  # its first game id

    # The first epoch's weights are the run's initial ones; the second's, the first's checkpoint.
    torch.manual_seed(config.seed)
    initial = create_model(config)
    trained = create_model(config)
    trained.load_state_dict(load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", torch.device("cpu"))["model_state_dict"])
    for epoch, (model, in_epoch) in enumerate((
        (initial, rows["game_ids"] < second_epoch), (trained, rows["game_ids"] >= second_epoch),
    ), start=1):
        epoch_rows = {name: column[in_epoch] for name, column in rows.items()}
        expected = _expected(model, config, epoch_rows)
        assert expected["heldout/rows_started"] > 0 and "heldout/value_mse_q4" in expected
        logged = {
            tag: by_epoch[epoch] for tag, values in scalars.items()
            if tag.startswith("heldout/") and epoch in (by_epoch := dict(values))
        }
        assert logged.pop("heldout/value_mse_over_training") == pytest.approx(
            expected["heldout/value_mse"] / dict(scalars["epoch/value_loss_avg"])[epoch], rel=1e-4,
        )
        assert logged.keys() == expected.keys()
        for tag, value in expected.items():
            assert logged[tag] == pytest.approx(value, rel=1e-4, abs=1e-6), (epoch, tag)

    # Without it: no tag, and training as it was.
    plain_checkpoints, plain_scalars = run("plain", 0)
    assert not any(tag.startswith("heldout/") for tag in plain_scalars)
    plain = load_checkpoint(plain_checkpoints / "checkpoint_epoch_0002.pt", torch.device("cpu"))["model_state_dict"]
    for name, weights in final["model_state_dict"].items():
        assert torch.equal(weights, plain[name]), name


def test_a_sample_draws_the_epochs_rows_alone_and_none_without_heldout_rows(monkeypatch):
    config = dataclasses.replace(
        TrainingConfig.from_json(json.dumps(_smoke_config(Path("unused")))),
        batch_size=16, min_buffer_size=16, buffer_capacity=4096, replay_priority_fraction=0.0,
    )
    buffer = _buffer(config)
    games = [_play(seed, 3 + seed % 3, 20 + 7 * seed) for seed in range(7)]
    for epoch, seeds in enumerate((range(0, 4), range(4, 7))):
        buffer.begin_epoch(epoch)
        for seed in seeds:
            game = games[seed]
            buffer.add_stacked(
                game["states"], game["phase_ids"], game["legal_masks"], game["policy_targets"],
                game["value_targets"], player_counts=game["player_counts"],
                flags=FLAG_FROM_POSITION if seed == 5 else 0,
            )
    torch.manual_seed(0)
    trainer = Trainer(create_model(config), config, torch.device("cpu"))
    # Each scored row's game, by the seed ``_play`` puts in its value target, and its state.
    scored: list[tuple[float, bytes]] = []
    real_score_rows = Trainer.score_rows

    def score_rows(self, states, legal_masks, policy_targets, value_targets, player_counts):
        scored.extend(zip(value_targets[:, 0].tolist(), (state.tobytes() for state in states)))
        return real_score_rows(self, states, legal_masks, policy_targets, value_targets, player_counts)

    monkeypatch.setattr(Trainer, "score_rows", score_rows)
    phases = get_game(config).policy.phase_names

    assert main_module._heldout_scalars(trainer, buffer, config, 1, phases) == {}
    assert scored == []

    sampled = dataclasses.replace(config, heldout_rows=25)
    trainer.model.train()
    first = main_module._heldout_scalars(trainer, buffer, sampled, 1, phases)
    assert trainer.model.training
    assert first["heldout/rows"] == 25 and len(scored) == 25
    assert {seed for seed, _ in scored} <= {4.0, 5.0, 6.0}  # the newest epoch's games
    assert len(set(scored)) == 25  # without replacement
    assert first["heldout/rows_started"] > 0 and "heldout/value_mse_started" in first
    # The draw is the seed's and the epoch's: the same rows again.
    again = list(scored)
    scored.clear()
    assert main_module._heldout_scalars(trainer, buffer, sampled, 1, phases) == first
    assert scored == again
