"""The pipeline on token layout 2 with nn/transformer-1889-v1.py (1889/v1-design.md
§7; 1889/model-design.md §12), on the CPU at a small size, and on the GPU
where there is one.

- **Transport.** The trainer, ``NNEvaluator``, the eval server with real
  worker processes, and its compiled warmup, by
  ``tests/g1889/test_relation_transport.py``'s own tests with v1 as their
  model: tokens 109 wide, E = 4,096, and a trunk of 80 rows under an input of
  95.
- **A run.** ``train.main`` plays an epoch with a quarter of its games started
  from the human positions, trains by ``replay_ratio``, scores its held-out
  rows, and checkpoints; ``--resume latest`` runs the next epoch. Each epoch's
  held-out tags are what the checkpoint before it, loaded through
  ``load_model_from_checkpoint``, gives on that epoch's rows, and the loaded
  model's outputs are those of the weights it was saved from.

The local smoke run covers a full-size epoch on the GPU
(``train_configs/1889-v1-smoke.json``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from games import RELATION_RECORD_WIDTH, get_game
from games.g1889_starts import load_start_positions
from nn import create_model, get_model_input_spec
from nn.model_contract import flatten_relation_records
from train.checkpoint import load_checkpoint, load_model_from_checkpoint
from train.config import TrainingConfig
from train.eval_server import _flat_filler_rows, _resolve_flat_launch_rows
from train.replay_buffer import FLAG_FROM_POSITION
from train.tb_reader import read_tb_scalars

import tests.g1889.test_relation_transport as transport_tests
from tests.g1889.test_heldout import _buffer, _expected
from tests.g1889.test_relation_transport import samples  # the fixture
from tests.g1889.test_training_run import SMALL, _train

MODEL_PATH = "nn/transformer-1889-v1.py"
MAX_PLAYERS = 5
TOKEN_DIM = 109
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


@pytest.fixture(autouse=True)
def few_math_threads():
    """Models this small gain nothing from a thread a core, and on a shared
    machine the threads' contention multiplies these tests' time."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 4))
    yield
    torch.set_num_threads(before)


@pytest.fixture
def transport(monkeypatch) -> Any:
    """``test_relation_transport.py`` with v1 as its model, so that its tests,
    called from here, run the pipeline on layout 2."""
    monkeypatch.setattr(transport_tests, "CONFIG", replace(transport_tests.CONFIG, model_path=MODEL_PATH))
    monkeypatch.setattr(transport_tests, "SMOKE_CONFIG", replace(transport_tests.SMOKE_CONFIG, model_path=MODEL_PATH))
    monkeypatch.setattr(transport_tests, "TOKEN_DIM", TOKEN_DIM)
    spec = get_model_input_spec(transport_tests.CONFIG)
    assert (spec.layout_version, spec.token_dim, transport_tests.EDGES) == (2, TOKEN_DIM, 4096)
    model: Any = transport_tests._model()
    assert model.cfg.num_trunk_tokens == transport_tests.NUM_TOKENS - 15 == 80
    return transport_tests


# =============================================================================
# TRANSPORT
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_trainer_steps_on_records_match_steps_on_planes(transport, monkeypatch, samples, device):
    transport.test_trainer_steps_on_records_match_steps_on_planes(monkeypatch, samples, device)


@pytest.mark.parametrize("device", DEVICES)
def test_trainer_overfits_a_fixed_batch(transport, monkeypatch, samples, device):
    transport.test_trainer_overfits_a_fixed_batch(monkeypatch, samples, device)


@pytest.mark.parametrize("device", DEVICES)
def test_nn_evaluator_gives_the_same_outputs_on_records_as_on_planes(transport, samples, device):
    transport.test_nn_evaluator_gives_the_same_outputs_on_records_as_on_planes(samples, device)


@pytest.mark.parametrize(
    ("device", "mode", "leaves"),
    [("cpu", "dynamic", 8), ("cpu", "bucketed", 8), pytest.param("cuda", "bucketed", 40, marks=CUDA)],
)
def test_eval_server_round_trip_gives_nn_evaluator_s_outputs(transport, samples, device, mode, leaves):
    """Real workers fill shared buffers 109 wide, and launches carry the
    candidates' records, which the model gives no weight."""
    transport.test_eval_server_round_trip_gives_nn_evaluator_s_outputs(samples, device, mode, leaves)


def test_warmup_leaves_n_dynamic_in_each_bucket(transport, samples):
    """Compiled, v1's forward takes every launch shape on the warmup's graphs."""
    transport.test_warmup_leaves_n_dynamic_in_each_bucket(samples)


def test_one_model_plays_with_nn_evaluator_and_trains(transport):
    transport.test_one_model_plays_with_nn_evaluator_and_trains()


def test_a_worker_plays_through_the_eval_server_and_its_rows_train(transport):
    transport.test_a_worker_plays_through_the_eval_server_and_its_rows_train()


def test_filler_leaves_the_forward_exactly_unchanged(transport, samples):
    """A launch with a scratch row and filler, against the same launch without:
    the same bits. Filler names every row, the candidates' among them."""
    game, edges, num_tokens = transport.GAME, transport.EDGES, transport.NUM_TOKENS
    leaves = samples[-3:]
    model = transport._model().eval()
    rows = list(transport._stack_rows(leaves))
    tokens = torch.zeros(4, num_tokens, TOKEN_DIM)
    game.fill_tokens_batch(rows, tokens.numpy(), max_players=MAX_PLAYERS, layout_version=2)
    mask = torch.zeros(4, transport.UNIFIED, dtype=torch.bool)
    mask[:3] = torch.from_numpy(transport._masks(leaves)).bool()
    records = np.empty((3, edges, RELATION_RECORD_WIDTH), np.uint8)
    game.fill_relations_sparse_batch(rows, records, max_players=MAX_PLAYERS)
    flat = flatten_relation_records(records)
    launch_rows = _resolve_flat_launch_rows(real_rows=len(flat), launch_n=4, batch_shape_mode="bucketed")
    filler = _flat_filler_rows(launch_rows - len(flat), edges, num_tokens)
    assert len(filler) > 0 and ((filler[:, 2:4] >= 75) & (filler[:, 2:4] < 90)).any()
    with torch.inference_mode():
        logits, values = model(tokens, mask, torch.from_numpy(np.concatenate([flat, filler])))
        plain_logits, plain_values = model(tokens, mask, torch.from_numpy(flat))
    assert torch.equal(logits[:3], plain_logits[:3]) and torch.equal(values[:3], plain_values[:3])


# =============================================================================
# A RUN
# =============================================================================

def _fields(tmp_path: Path) -> dict[str, object]:
    """v1's smoke config at a size for the CPU, writing under ``tmp_path``:
    twelve games an epoch, every row of an epoch held out."""
    fields = json.loads(Path("train_configs/1889-v1-smoke.json").read_text())
    assert fields["model_path"] == MODEL_PATH and fields["human_start_fraction"] == 0.25
    assert fields["start_positions_path"] == "g1889/data/human_starts.npz" and fields["replay_ratio"] == 1.1
    return {
        **fields,
        **SMALL,
        "games_per_epoch": 12,
        "buffer_capacity": 65_536,
        "heldout_rows": 1_000_000,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tensorboard_dir": str(tmp_path / "runs"),
    }


def _outputs(model: torch.nn.Module, config: TrainingConfig, states: np.ndarray, masks: np.ndarray):
    game, spec = get_game(config), get_model_input_spec(config)
    tokens = np.empty((len(states), game.num_tokens(MAX_PLAYERS), spec.token_dim), np.float32)
    game.fill_tokens_batch(list(states), tokens, max_players=MAX_PLAYERS, layout_version=spec.layout_version)
    records = np.empty((len(states), game.max_relation_edges, RELATION_RECORD_WIDTH), np.uint8)
    game.fill_relations_sparse_batch(list(states), records, max_players=MAX_PLAYERS)
    with torch.inference_mode():
        return model.eval()(
            torch.from_numpy(tokens), torch.from_numpy(masks).bool(), torch.from_numpy(flatten_relation_records(records)),
        )


def test_a_v1_run_plays_trains_scores_checkpoints_resumes_and_loads(tmp_path, monkeypatch, capsys):
    fields = _fields(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")

    _train(monkeypatch, "--config", str(config_path))
    output = capsys.readouterr().out
    assert "Start positions: g1889/data/human_starts.npz" in output and "25% of each" in output
    _train(monkeypatch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "2")
    assert "Resumed from epoch 0" in capsys.readouterr().out

    paths = [checkpoints / f"checkpoint_epoch_{epoch:04d}.pt" for epoch in (1, 2)]
    first, second = (load_checkpoint(path, device) for path in paths)
    config = TrainingConfig.from_json(second["config_json"])
    spec = get_model_input_spec(config)
    assert config.model_path == MODEL_PATH
    assert (spec.layout_version, spec.token_dim, get_game(config).max_relation_edges) == (2, TOKEN_DIM, 4096)
    steps: list[Any] = [checkpoint["trainer_state"]["global_step"] for checkpoint in (first, second)]
    assert 0 < steps[0] < steps[1]

    # --- Replay: both epochs' rows, a quarter of each player count's games started ---
    buffer = _buffer(config)
    assert buffer.load(checkpoints / "replay_buffer") == second["buffer_stats"]["size"] > 0
    rows = buffer.rows_oldest_first()
    second_epoch = buffer.epochs[1][1]  # its first game id
    assert set(rows["game_ids"].tolist()) == set(range(24)) and second_epoch == 12
    started = sorted(set(rows["game_ids"][(rows["flags"] & FLAG_FROM_POSITION) != 0].tolist()))
    assert [sum(game < 12 for game in started), len(started)] == [3, 6]
    positions = {row.tobytes() for row in load_start_positions(Path("g1889/data/human_starts.npz")).rows}
    for game in started:
        of_game = rows["game_ids"] == game
        assert rows["states"][of_game][0].tobytes() in positions
    assert {int(rows["player_counts"][rows["game_ids"] == game][0]) for game in started[:3]} == {3, 4, 5}

    # --- TensorBoard: the run trained by its rows, and scored each epoch before training on it ---
    scalars = read_tb_scalars(str(tmp_path / "runs"))
    assert scalars["self_play_started/games"] == [(1, 3), (2, 3)]
    played = [int(value) for _, value in scalars["self_play_aggregate/total_examples"]]
    begun = [int(value) for _, value in scalars["self_play_started/total_examples"]]
    in_epoch = [int((rows["game_ids"] < second_epoch).sum()), int((rows["game_ids"] >= second_epoch).sum())]
    assert [a + b for a, b in zip(played, begun, strict=True)] == in_epoch
    # replay_ratio 1.1 by the share of the 3 epochs held: a third, then two thirds.
    expected_steps = [round(1.1 * count / 64 * share) for count, share in zip(in_epoch, (1 / 3, 2 / 3), strict=True)]
    assert [value for _, value in scalars["epoch/training_steps"]] == expected_steps == [steps[0], steps[1] - steps[0]]
    for tag in ("epoch/total_loss_avg", "epoch/policy_loss_avg", "epoch/value_loss_avg", "epoch/policy_loss_tile_avg",
                "epoch/policy_kl_tile_avg", "epoch/policy_loss_div_avg", "epoch/policy_loss_train_avg"):
        assert [step for step, _ in scalars[tag]] == [1, 2] and all(np.isfinite(v) for _, v in scalars[tag]), tag

    # --- Checkpoints: loaded back, each is the model that scored the next epoch ---
    loaded: Any
    loaded, loaded_config, _ = load_model_from_checkpoint(paths[0], device)
    assert loaded_config.model_path == MODEL_PATH and loaded.cfg.layout_version == 2
    assert (loaded.cfg.num_tokens, loaded.cfg.num_trunk_tokens, loaded.cfg.token_dim) == (95, 80, TOKEN_DIM)
    torch.manual_seed(config.seed)
    initial = create_model(config)
    for epoch, (model, of_epoch) in enumerate((
        (initial, rows["game_ids"] < second_epoch), (loaded, rows["game_ids"] >= second_epoch),
    ), start=1):
        expected = _expected(model, config, {name: column[of_epoch] for name, column in rows.items()})
        assert expected["heldout/rows"] == in_epoch[epoch - 1] and expected["heldout/rows_started"] > 0
        assert "heldout/policy_kl_tile" in expected and "heldout/policy_kl_er_bonus" in expected
        logged = {
            tag: by_epoch[epoch] for tag, values in scalars.items()
            if tag.startswith("heldout/") and tag != "heldout/value_mse_over_training"
            and epoch in (by_epoch := dict(values))
        }
        assert logged.keys() == expected.keys()
        for tag, value in expected.items():
            assert logged[tag] == pytest.approx(value, rel=1e-4, abs=1e-6), (epoch, tag)

    # --- The loaded model is the saved weights' ---
    final, final_config, _ = load_model_from_checkpoint(paths[1], device)
    rebuilt = create_model(final_config)
    rebuilt.load_state_dict(second["model_state_dict"])
    newest = slice(len(rows["states"]) - 256, None)
    states, masks = rows["states"][newest], rows["legal_masks"][newest]
    logits, values = _outputs(final, final_config, states, masks)
    rebuilt_logits, rebuilt_values = _outputs(rebuilt, final_config, states, masks)
    assert torch.equal(logits, rebuilt_logits) and torch.equal(values, rebuilt_values)
    legal = torch.from_numpy(masks).bool()
    assert torch.isfinite(logits[legal]).all() and (logits[~legal] == -1e9).all()
    earlier_logits, _ = _outputs(loaded, loaded_config, states, masks)
    assert (earlier_logits - logits)[legal].abs().max() > 1e-4  # the second epoch trained

    # --- Training reached what v1 adds: the candidate head and the new relations' channels ---
    before, after = first["model_state_dict"], second["model_state_dict"]
    for name in ("candidate_head.0.weight", "candidate_head.4.weight", "candidate_norm.weight", "exit_proj.weight",
                 "exit_side_proj.0.weight", "par_head.0.weight", "train_price_head.0.weight"):
        assert not torch.equal(before[name], after[name]), name
    # Channels 21-24 are the SR's sales and the route hexes (relations 25-28).
    assert after["relation_bias_mult"].shape[-1] == 35
    assert after["relation_bias_mult"][:, :, 21:25].abs().sum((0, 1)).gt(0).all()
