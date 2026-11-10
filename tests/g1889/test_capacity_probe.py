"""The capacity probe (1889/model-design.md §5.7): its held-out split, its KL
measure, its verdict, and a run end to end on the CPU at small sizes, interrupted
and resumed, with its weights warm-starting ``train.main``.

The GPU's run is the runbook's (§11).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import g1889.constants as K
from games import STATUS_GAME_OVER, STATUS_OK, get_game
from nn import create_model
from train import capacity_probe as probe
from train import main as main_module
from train.checkpoint import load_checkpoint, load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer, policy_row_losses

from tests.g1889.test_training_run import _smoke_config

MAX_PLAYERS = 5
CONFIG = TrainingConfig.from_json(Path("train_configs/1889-v0-smoke.json").read_text())
GAME = get_game(CONFIG)
assert GAME.game_start_rows is not None
GAME_STARTS = GAME.game_start_rows


def _play(seed: int, players: int, max_decisions: int | None = None) -> dict[str, np.ndarray]:
    """One random game's rows as self-play stores them, value slot 0 holding ``seed``."""
    rng = np.random.default_rng(seed)
    state = GAME.new_game(players, MAX_PLAYERS, seed)
    policy = GAME.policy
    actions = np.empty(policy.max_action_size, np.uint16)
    lut = policy.build_action_lut().numpy()
    rows: list[dict[str, Any]] = []
    while max_decisions is None or len(rows) < max_decisions:
        n = GAME.enumerate_policy_actions(state, actions)
        phase = int(GAME.decision_phase(state))
        legal = np.zeros(policy.unified_dim, np.uint8)
        legal[lut[phase, actions[:n]]] = 1
        value = np.zeros(MAX_PLAYERS, np.float32)
        value[0] = seed
        rows.append({
            "states": state._array.copy(),
            "phase_ids": np.int8(phase),
            "legal_masks": legal,
            "policy_targets": legal / legal.sum(),
            "value_targets": value,
            "player_counts": np.uint8(players),
            "priorities": np.float32(seed + 1),
        })
        status = GAME.apply_action(state, int(actions[int(rng.integers(n))]))
        if status == STATUS_GAME_OVER:
            break
        assert status == STATUS_OK
    return {
        name: np.stack([row[name] for row in rows]).astype(
            np.float32 if name == "policy_targets" else rows[0][name].dtype,
        )
        for name in rows[0]
    }


def _games(count: int, max_decisions: int | None = 30, first_seed: int = 0) -> dict[str, np.ndarray]:
    games = [_play(first_seed + i, 3 + i % 3, max_decisions) for i in range(count)]
    return {name: np.concatenate([game[name] for game in games]) for name in games[0]}


def _buffer(capacity: int, rows: dict[str, np.ndarray], *, fraction: float = 0.0) -> ReplayBuffer:
    buffer = ReplayBuffer(
        GAME, capacity, GAME.state_size(MAX_PLAYERS), MAX_PLAYERS, min_players=3, max_players=MAX_PLAYERS,
        priority_fraction=fraction,
    )
    # A few rows an add, as games arrive, so a small ring wraps.
    for start in range(0, len(rows["states"]), 10):
        part = {name: array[start:start + 10] for name, array in rows.items()}
        buffer.add_stacked(
            part["states"], part["phase_ids"], part["legal_masks"], part["policy_targets"], part["value_targets"],
            player_counts=part["player_counts"], priorities=part["priorities"],
        )
    return buffer


# ---------------------------------------------------------------------------
# The held-out split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("players", [3, 4, 5])
def test_each_game_has_one_start_row_at_its_first_decision(players: int) -> None:
    games = [_play(seed, players) for seed in (11, 12, 13)]
    states = np.concatenate([game["states"] for game in games])
    starts = GAME_STARTS(states, MAX_PLAYERS)
    expected = np.cumsum([0, *(len(game["states"]) for game in games[:-1])])
    np.testing.assert_array_equal(starts, expected)


@pytest.mark.parametrize("players", [3, 5])
def test_a_game_whose_auction_opens_with_passes_starts_only_once(players: int) -> None:
    state = GAME.new_game(players, MAX_PLAYERS, 3)
    rows = [state._array.copy()]
    # Two full rounds of passes: flags set, flags cleared with a discount, and on.
    for _ in range(2 * players):
        assert int(GAME.decision_phase(state)) == K.DPHASE_AUCTION
        assert GAME.apply_action(state, K.AUCTION_PASS) == STATUS_OK
        rows.append(state._array.copy())
    np.testing.assert_array_equal(GAME_STARTS(np.stack(rows), MAX_PLAYERS), [0])


def test_held_out_rows_are_the_newest_games_of_a_wrapped_ring_with_no_leak() -> None:
    rows = _games(12)
    lengths = [int((rows["value_targets"][:, 0] == seed).sum()) for seed in range(12)]
    total = sum(lengths)
    # The ring has overwritten the first game and part of the second.
    capacity = total - lengths[0] - lengths[1] // 2
    ring = _buffer(capacity, rows)
    assert ring._index != 0
    ordered = ring.rows_oldest_first()
    np.testing.assert_array_equal(ordered["states"], rows["states"][-capacity:])

    starts = GAME_STARTS(ordered["states"], MAX_PLAYERS)
    assert len(starts) == 10  # the second game's start is gone
    split = probe.choose_split(starts, capacity, games_per_epoch=2, holdout_epochs=2, logged_rows=None)
    assert split.holdout_rows == sum(lengths[-4:])
    assert split.train_rows == capacity - split.holdout_rows

    train = probe._train_buffer(dataclasses.replace(CONFIG, replay_priority_fraction=0.5), ordered, split.train_rows)
    assert len(train) == split.train_rows
    train_games = set(train.rows_oldest_first()["value_targets"][:, 0].tolist())
    held_out_games = set(ordered["value_targets"][split.train_rows:, 0].tolist())
    assert held_out_games == {8.0, 9.0, 10.0, 11.0}
    assert train_games == set(range(1, 8))
    # Priorities came along, and the sampler can reach only the training rows.
    np.testing.assert_array_equal(train.rows_oldest_first()["priorities"], ordered["priorities"][:split.train_rows])
    indices, _ = train._sample_indices(32, np.random.default_rng(0))
    assert indices.max() < split.train_rows


def test_the_split_reports_whether_tensorboard_agrees() -> None:
    starts = np.array([0, 10, 25, 33, 50])
    agree = probe.choose_split(starts, 60, games_per_epoch=1, holdout_epochs=2, logged_rows=27)
    assert (agree.train_rows, agree.holdout_rows, agree.agrees_with_tensorboard) == (33, 27, True)
    assert probe.choose_split(starts, 60, games_per_epoch=1, holdout_epochs=2, logged_rows=30).agrees_with_tensorboard is False
    silent = probe.choose_split(starts, 60, games_per_epoch=1, holdout_epochs=2, logged_rows=None)
    assert silent.agrees_with_tensorboard is None
    assert silent.train_rows == 33
    with pytest.raises(probe.ProbeError, match="lower --holdout-epochs"):
        probe.choose_split(starts, 60, games_per_epoch=3, holdout_epochs=2, logged_rows=None)


def _recorded_buffer(capacity: int, epochs: dict[int, list[dict[str, np.ndarray]]]) -> ReplayBuffer:
    """A buffer as a run fills it: each epoch begun, then its games added whole."""
    buffer = ReplayBuffer(
        GAME, capacity, GAME.state_size(MAX_PLAYERS), MAX_PLAYERS, min_players=3, max_players=MAX_PLAYERS,
    )
    for epoch, games in epochs.items():
        buffer.begin_epoch(epoch)
        for game in games:
            buffer.add_stacked(
                game["states"], game["phase_ids"], game["legal_masks"], game["policy_targets"],
                game["value_targets"], player_counts=game["player_counts"], priorities=game["priorities"],
            )
    return buffer


def test_the_epoch_record_gives_the_split_that_the_rows_states_give(tmp_path: Path) -> None:
    """On a run's saved buffer, 3 games an epoch in a ring that has wrapped, the
    cut read from the record is the cut reconstructed from the rows."""
    games = [_play(seed, 3 + seed % 3, 18 + 5 * (seed % 4)) for seed in range(15)]
    total = sum(len(game["states"]) for game in games)
    capacity = total - len(games[0]["states"]) - len(games[1]["states"]) // 2
    ring = _recorded_buffer(capacity, {epoch: games[3 * epoch:3 * epoch + 3] for epoch in range(5)})
    assert ring._index != 0
    ring.save(tmp_path)

    rows, epochs = probe._load_rows(CONFIG, tmp_path, capacity=capacity)
    assert epochs == [(epoch, 3 * epoch) for epoch in range(5)]
    starts = GAME_STARTS(rows["states"], MAX_PLAYERS)
    for holdout_epochs in (1, 2, 3, 4):
        recorded = probe.record_split(
            rows["game_ids"], epochs, last_epoch=5, holdout_epochs=holdout_epochs, logged_rows=None,
        )
        assert recorded == probe.choose_split(
            starts, capacity, games_per_epoch=3, holdout_epochs=holdout_epochs, logged_rows=None,
        )
        assert recorded is not None and recorded.games == 3 * holdout_epochs
        held_out = set(rows["value_targets"][recorded.train_rows:, 0].tolist())
        assert held_out == set(range(15 - 3 * holdout_epochs, 15))
    # A record without the epoch, or no record, leaves the cut to the rows.
    for record, last_epoch in ((epochs, 9), ([], 5)):
        assert probe.record_split(
            rows["game_ids"], record, last_epoch=last_epoch, holdout_epochs=2, logged_rows=None,
        ) is None


def test_the_record_split_cuts_between_epochs_of_any_size() -> None:
    game_ids = np.repeat(np.arange(4, 12), [5, 3, 4, 6, 2, 7, 3, 4])
    # Epochs 0-indexed, as the run counts them; 4 is the partial epoch a
    # graceful shutdown left after the checkpoint at epoch 4 (1-indexed).
    epochs = [(1, 3), (2, 6), (3, 9), (4, 11)]
    split = probe.record_split(game_ids, epochs, last_epoch=4, holdout_epochs=2, logged_rows=22)
    assert split is not None
    assert (split.train_rows, split.holdout_rows, split.games) == (8, 26, 6)
    assert split.agrees_with_tensorboard is False
    one = probe.record_split(game_ids, epochs, last_epoch=4, holdout_epochs=1, logged_rows=None)
    assert one is not None and (one.train_rows, one.games, one.agrees_with_tensorboard) == (20, 3, None)
    # Its oldest epoch's first games have left the ring: no rows are older.
    three = probe.record_split(game_ids, epochs, last_epoch=4, holdout_epochs=3, logged_rows=None)
    assert three is not None and three.train_rows == 0


def test_tensorboard_rows_need_every_epoch() -> None:
    examples = [(1, 100.0), (2, 110.0), (3, 120.0), (5, 130.0)]
    assert probe.tensorboard_rows(examples, 3, 2) == 230
    assert probe.tensorboard_rows(examples, 5, 2) is None  # epoch 4 is missing
    assert probe.tensorboard_rows(examples, 9, 2) is None
    assert probe.tensorboard_rows([], 3, 2) is None


# ---------------------------------------------------------------------------
# The measures
# ---------------------------------------------------------------------------

def test_policy_kl_is_cross_entropy_minus_target_entropy() -> None:
    logits = torch.tensor([[math.log(0.25), math.log(0.75), -1e9], [0.0, 0.0, -1e9]])
    targets = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    cross_entropy, entropy = policy_row_losses(logits, targets)
    exact = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert (cross_entropy - entropy)[0].item() == pytest.approx(exact, abs=1e-6)
    # A policy that is its target has no KL, however much entropy they share.
    assert (cross_entropy - entropy)[1].item() == pytest.approx(0.0, abs=1e-6)
    assert entropy[1].item() == pytest.approx(math.log(2), abs=1e-6)


def test_scores_group_by_player_count_and_phase() -> None:
    cross_entropy = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    entropy = np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)
    value_sqerr = np.array([3.0, 6.0, 10.0, 0.0], dtype=np.float32)
    players = np.array([3, 3, 5, 5], dtype=np.uint8)
    phase_names = GAME.policy.phase_names
    phases = np.array([phase_names.index(name) for name in ("track", "track", "tile", "stock")], dtype=np.int8)
    scores = probe.summarize_scores(
        cross_entropy, entropy, value_sqerr, players, probe.group_rows(players, phases, tuple(phase_names)),
    )
    assert scores["all"]["policy_kl"] == pytest.approx((0.5 + 1.5 + 2.0 + 3.0) / 4)
    assert scores["all"]["value_mse"] == pytest.approx(19.0 / 16)
    assert scores["3p"]["value_mse"] == pytest.approx(9.0 / 6)
    assert scores["5p"]["policy_kl"] == pytest.approx(2.5)
    assert scores["TRACK"]["rows"] == 2
    assert scores["TRACK"]["policy_kl"] == pytest.approx(1.0)
    assert scores["TILE"]["value_mse"] == pytest.approx(10.0 / 5)
    assert scores["STOCK"]["policy_kl"] == pytest.approx(3.0)
    assert "BUY_TRAIN" not in scores  # no rows


def test_held_out_scores_are_the_trainers_own_loss() -> None:
    """On the same weights and rows, the probe's measures are what a training
    step reports before its update, uniform sampling over every row."""
    rows = _games(6)
    config = dataclasses.replace(
        CONFIG, d_model=32, num_layers=1, batch_size=32, min_buffer_size=32, replay_priority_fraction=0.0,
    )
    torch.manual_seed(0)
    model = create_model(config)
    trainer = Trainer(model, config, torch.device("cpu"))
    scores = probe.evaluate_holdout(trainer, rows, 32, tuple(GAME.policy.phase_names))
    total = len(rows["states"])
    losses = trainer.train_step(_buffer(total, rows), total, np.random.default_rng(0))
    assert scores["all"]["rows"] == total
    assert scores["all"]["policy_kl"] == pytest.approx(losses["policy_kl"], abs=1e-5)
    assert scores["all"]["value_mse"] == pytest.approx(losses["value_loss"], abs=1e-5)
    for n in (3, 4, 5):
        assert scores[f"{n}p"]["value_mse"] == pytest.approx(losses[f"value_loss_{n}p"], abs=1e-5)
    assert scores["STOCK"]["policy_kl"] == pytest.approx(losses["policy_kl_stock"], abs=1e-5)


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------

def _results(
    *, large: tuple[float, float] = (1.0, 1.0), small: tuple[float, float] = (1.5, 1.5),
    heads: tuple[float, float] = (1.0, 1.0), seeds: tuple[float, float] = (1.0, 1.0),
) -> dict[str, dict[str, float]]:
    """Models' (value MSE, policy KL), with one seed at each ``seeds`` for both."""
    def row(pair: tuple[float, float]) -> dict[str, float]:
        return {"value_mse": pair[0], "policy_kl": pair[1]}
    return {
        "base_seed0": row((seeds[0], seeds[0])),
        "base_seed1": row((seeds[1], seeds[1])),
        "small": row(small),
        "large": row(large),
        "heads8": row(heads),
    }


@pytest.mark.parametrize(
    ("results", "epoch", "verdict"),
    [
        # Clear winners and no winner.
        (_results(large=(0.9, 0.9)), 50, "grow"),
        (_results(small=(1.0, 1.0)), 150, "shrink"),
        (_results(heads=(0.9, 0.9)), 50, "8 heads"),
        (_results(), 50, "keep"),
        # The first that holds.
        (_results(large=(0.9, 0.9), heads=(0.9, 0.9), small=(1.0, 1.0)), 150, "grow"),
        (_results(heads=(0.9, 0.9), small=(1.0, 1.0)), 150, "shrink"),
        # Better on one measure only.
        (_results(large=(0.9, 1.0)), 50, "keep"),
        (_results(large=(1.0, 0.9)), 50, "keep"),
        (_results(heads=(0.9, 1.01)), 50, "keep"),
        (_results(small=(1.0, 1.01)), 150, "keep"),
        # The 3% floor, with no seed gap: at it and just under it.
        (_results(large=(0.97, 0.97)), 50, "grow"),
        (_results(large=(0.9701, 0.97)), 50, "keep"),
        (_results(heads=(0.97, 0.97)), 50, "8 heads"),
        (_results(heads=(0.9701, 0.97)), 50, "keep"),
        # Twice the seed gap: the mean is 1.0 and the gap 0.04, so more than 0.08.
        (_results(large=(0.91, 0.91), seeds=(0.98, 1.02)), 50, "grow"),
        (_results(large=(0.92, 0.91), seeds=(0.98, 1.02)), 50, "keep"),
        (_results(large=(0.93, 0.93), seeds=(0.98, 1.02)), 50, "keep"),
        # A wide gap swamps a 10% gain, on the mean of 1.0.
        (_results(large=(0.9, 0.9), seeds=(0.9, 1.1)), 50, "keep"),
        # Within the seed gap: shrink only at epoch 150, and only within it.
        (_results(small=(1.0, 1.0)), 50, "keep"),
        (_results(small=(1.0, 1.0)), 149, "keep"),
        (_results(small=(1.0, 1.0)), 151, "shrink"),
        (_results(small=(1.02, 1.02), seeds=(0.98, 1.02)), 150, "shrink"),
        (_results(small=(1.04, 1.04), seeds=(0.98, 1.02)), 150, "shrink"),
        (_results(small=(1.0401, 1.04), seeds=(0.98, 1.02)), 150, "keep"),
        (_results(small=(1.0001, 1.0)), 150, "keep"),  # no gap: nothing to be within
        # Smaller and better still shows v0 is larger than its data needs.
        (_results(small=(0.8, 0.8), seeds=(0.98, 1.02)), 150, "shrink"),
    ],
)
def test_verdict(results: dict[str, dict[str, float]], epoch: int, verdict: str) -> None:
    decided = probe.decide_verdict(results, epoch)
    assert decided["verdict"] == verdict, decided


def test_verdict_reports_what_else_held_and_what_is_missing() -> None:
    results = _results(large=(0.9, 0.9), heads=(0.9, 0.9), small=(1.0, 1.0))
    decided = probe.decide_verdict(results, 150)
    assert decided["conditions"] == {"grow": True, "shrink": True, "8 heads": True}
    assert any("8 heads" in note for note in decided["notes"])
    assert any("192 x 12" in note for note in decided["notes"])
    assert decided["comparisons"]["large"]["policy_kl"]["margin_share"] == pytest.approx(0.1)

    early = probe.decide_verdict(_results(small=(1.0, 1.0)), 50)
    assert any("only at epoch 150" in note for note in early["notes"])

    results.pop("large")
    assert probe.decide_verdict(results, 150)["verdict"] == "incomplete"


# ---------------------------------------------------------------------------
# The schedule and the whole run
# ---------------------------------------------------------------------------

def test_every_model_gets_the_one_schedule_across_the_probes_steps() -> None:
    steps, warmup = 200, 20
    schedules = []
    for model_spec in probe.SMOKE_MODELS:
        config = probe.probe_config(CONFIG, model_spec, steps=steps, batch_size=32, warmup_steps=warmup)
        assert (config.d_model, config.num_layers, config.num_heads) == (
            model_spec.d_model, model_spec.num_layers, model_spec.num_heads,
        )
        assert config.optimizer == CONFIG.optimizer
        trainer = Trainer(create_model(config), config, torch.device("cpu"))
        schedules.append([trainer.scheduler.lr_lambdas[0](step) for step in range(0, steps + 1, 10)])  # type: ignore[attr-defined]
        assert trainer.scheduler.lr_lambdas[0](steps) == pytest.approx(CONFIG.lr_min / CONFIG.learning_rate)  # type: ignore[attr-defined]
        assert trainer.scheduler.lr_lambdas[0](warmup) == pytest.approx(1.0)  # type: ignore[attr-defined]
    assert all(schedule == schedules[0] for schedule in schedules)


def _saved_run(tmp_path: Path) -> Path:
    """A checkpoint and buffer as a run leaves them: 3 games an epoch, 8 epochs' worth."""
    fields = {**_smoke_config(tmp_path), "games_per_epoch": 3, "training_steps_per_epoch": 700}
    config = TrainingConfig.from_json(json.dumps(fields))
    rows = _games(24, max_decisions=25)
    run_dir = Path(config.checkpoint_dir)
    # A wrapped ring, as a run's would be by the probes.
    _buffer(len(rows["states"]) - 7, rows).save(run_dir / "replay_buffer")
    torch.manual_seed(1)
    model = create_model(config)
    trainer = Trainer(model, config, torch.device("cpu"))
    trainer._global_step = 5_000
    checkpoint = run_dir / "checkpoint_epoch_0050.pt"
    save_checkpoint(
        checkpoint, 49, model, trainer.state_dict(), config, {},
        {"size": 0, "capacity": config.buffer_capacity},
        rng_state=main_module._capture_rng_state(np.random.default_rng(77)),
    )
    return checkpoint


def _probe_args(checkpoint: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--checkpoint", str(checkpoint), "--out", str(out), "--smoke", "--device", "cpu",
        "--steps", "6", "--batch-size", "8", "--warmup-steps", "1", "--eval-batch-size", "32",
        "--save-every", "2", "--eval-every", "3", "--log-every", "3", *extra,
    ]


def test_a_probe_runs_skips_finished_models_and_warm_starts_training(tmp_path, monkeypatch, capsys):
    checkpoint = _saved_run(tmp_path)
    out = tmp_path / "probe"
    probe.main(_probe_args(checkpoint, out))
    output = capsys.readouterr().out
    assert "Verdict:" in output
    assert "TensorBoard has no" in output

    report = json.loads((out / "probe.json").read_text())
    assert report["verdict"]["verdict"] in ("grow", "shrink", "8 heads", "keep")
    assert set(report["results"]) == {m.role for m in probe.SMOKE_MODELS}
    assert report["split"]["games"] == 6
    assert report["split"]["holdout_rows"] == 6 * 25
    for result in report["results"].values():
        assert result["steps"] == 6
        assert set(result["holdout"]) >= {"all", "3p", "4p", "5p", "STOCK"}
        assert result["history"][0]["step"] == 3
        assert result["steps_per_second"] > 0
    assert report["run_epoch"] == 50

    # Each model's weights, in the run's checkpoint format, at its own size.
    for model_spec in probe.SMOKE_MODELS:
        path = out / model_spec.role / "checkpoint_epoch_0050.pt"
        model, config, cp = load_model_from_checkpoint(path, torch.device("cpu"))
        assert model.relation_bias_mult.shape[:2] == (model_spec.num_layers, model_spec.num_heads)  # type: ignore[union-attr]
        assert (config.d_model, config.training_steps_per_epoch, config.lr_decay_end_epoch) == (
            model_spec.d_model, 700, CONFIG.lr_decay_end_epoch,
        )
        assert cp["epoch"] == 49
        assert cp["trainer_state"]["global_step"] == 5_000
        assert not (out / model_spec.role / "progress.pt").exists()

    # A rerun trains nothing.
    capsys.readouterr()
    monkeypatch.setattr(Trainer, "train_step", lambda *_: pytest.fail("a finished model trained"))
    probe.main(_probe_args(checkpoint, out))
    output = capsys.readouterr().out
    assert output.count("already done, skipping") == len(probe.SMOKE_MODELS)
    assert json.loads((out / "probe.json").read_text())["verdict"] == report["verdict"]
    monkeypatch.undo()

    # Other settings don't pass for a finished model's.
    with pytest.raises(SystemExit, match="came from other settings"):
        probe.main(_probe_args(checkpoint, out, "--holdout-epochs", "1"))

    # python -m train --resume, on the grown model's weights: the next epoch, at
    # the run's step on the run's schedule, in a directory of its own.
    grown = out / "large" / "checkpoint_epoch_0050.pt"
    new_dir, new_logs = tmp_path / "grown", tmp_path / "grown_runs"
    monkeypatch.setattr(sys, "argv", [
        "train", "--device", "cpu", "--no-compile", "--resume", str(grown),
        "--checkpoint-dir", str(new_dir), "--tensorboard-dir", str(new_logs),
        "--num-epochs", "51", "--games-per-epoch", "3",
    ])
    main_module.main()
    resumed = load_checkpoint(new_dir / "checkpoint_epoch_0051.pt", torch.device("cpu"))
    assert resumed["epoch"] == 50
    step = resumed["trainer_state"]["global_step"]
    assert isinstance(step, int) and step > 5_000
    assert TrainingConfig.from_json(resumed["config_json"]).d_model == probe.SMOKE_MODELS[3].d_model
    assert "Restored RNG state from checkpoint" in capsys.readouterr().out


def test_a_probe_cuts_at_the_epoch_record_of_a_buffer_that_has_one(tmp_path, capsys):
    checkpoint = _saved_run(tmp_path)  # epoch 50 (1-indexed), 3 games an epoch
    # Its buffer, as a run that records epochs leaves it: the newest two hold 7
    # games, where the rows alone would give 2 x 3.
    sizes = {44: 3, 45: 3, 46: 2, 47: 4, 48: 3, 49: 4}
    seeds = iter(range(100))
    epochs = {epoch: [_play(next(seeds), 3 + epoch % 3, 25) for _ in range(count)] for epoch, count in sizes.items()}
    _recorded_buffer(19 * 25 - 9, epochs).save(checkpoint.parent / "replay_buffer")

    out = tmp_path / "probe"
    probe.main(_probe_args(checkpoint, out, "--models", "base_seed0"))
    assert "Verdict: INCOMPLETE" in capsys.readouterr().out
    split = json.loads((out / "probe.json").read_text())["split"]
    assert (split["games"], split["holdout_rows"], split["train_rows"]) == (7, 7 * 25, 12 * 25 - 9)


def test_an_interrupted_model_continues_to_the_same_weights(tmp_path, monkeypatch):
    checkpoint = _saved_run(tmp_path)
    whole, broken = tmp_path / "whole", tmp_path / "broken"
    probe.main(_probe_args(checkpoint, whole, "--models", "base_seed0"))

    real_step = Trainer.train_step

    def stop_at_five(self: Trainer, *args: object) -> dict[str, float]:
        if self.global_step == 5:
            raise KeyboardInterrupt
        return real_step(self, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(Trainer, "train_step", stop_at_five)
    with pytest.raises(SystemExit, match="Rerun the same command"):
        probe.main(_probe_args(checkpoint, broken, "--models", "base_seed0"))
    assert (broken / "base_seed0" / "progress.pt").exists()
    assert not (broken / "base_seed0" / "result.json").exists()
    monkeypatch.undo()

    probe.main(_probe_args(checkpoint, broken, "--models", "base_seed0"))
    report = json.loads((broken / "probe.json").read_text())
    assert report["verdict"]["verdict"] == "incomplete"
    first = load_checkpoint(whole / "base_seed0" / "checkpoint_epoch_0050.pt", torch.device("cpu"))
    second = load_checkpoint(broken / "base_seed0" / "checkpoint_epoch_0050.pt", torch.device("cpu"))
    for name, weight in first["model_state_dict"].items():
        assert torch.equal(weight, second["model_state_dict"][name]), name
    assert report["results"]["base_seed0"]["holdout"] == (
        json.loads((whole / "probe.json").read_text())["results"]["base_seed0"]["holdout"]
    )


def test_a_probe_says_what_is_missing(tmp_path, capsys):
    with pytest.raises(SystemExit, match="no checkpoint"):
        probe.main(["--checkpoint", str(tmp_path / "none.pt")])
    checkpoint = _saved_run(tmp_path)
    with pytest.raises(SystemExit, match="no replay buffer"):
        probe.main(_probe_args(checkpoint, tmp_path / "out", "--buffer", str(tmp_path / "nowhere")))
    with pytest.raises(SystemExit, match="lower --holdout-epochs"):
        probe.main(_probe_args(checkpoint, tmp_path / "out", "--holdout-epochs", "9"))
    with pytest.raises(SystemExit, match="unknown --models"):
        probe.main(_probe_args(checkpoint, tmp_path / "out", "--models", "nope"))
