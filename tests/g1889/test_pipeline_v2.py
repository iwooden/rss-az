"""The pipeline on layout 3 with nn/transformer-1889-v2.py (1889/v2-design.md §7
and §10, step 4), on the CPU at a small size. ``test_packed_transport.py`` has
the transport's own tests, ``test_tournament.py`` v1 against v2 in one game.

- **A run.** ``train.main`` on the v2 smoke config: an epoch of self-play by
  two worker processes through an eval server, a quarter of its games started
  from the human positions, training by ``replay_ratio``, its held-out rows
  scored, and a checkpoint; ``--resume latest`` runs the next epoch in-process
  (``NNEvaluator``). Each epoch's held-out tags are what the checkpoint before
  it gives on that epoch's rows, the loaded model's outputs are its saved
  weights', and the second epoch trained v2's own parameters. The eval
  server's tags split its time by stage and count launches by bucket.
- **The game log.** ``utils_18xx.g1889_selfplay_log``, 1889's game analysis,
  plays a game from the run's checkpoint, and Ruby replays it;
  ``train.analyze_game``, RSS's, refuses the checkpoint.
- **The configs.** ``1889-v2.json`` and its smoke config are v1's but for the
  model and the run's directories.

The local smoke run covers full-size epochs on the GPU
(``train_configs/1889-v2-smoke.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from games import get_game
from nn import create_model, get_model_input_spec
from train import analyze_game as analyze_game_module
from train.checkpoint import load_checkpoint, load_model_from_checkpoint
from train.config import TrainingConfig
from train.profile_stats import STAGES
from train.replay_buffer import FLAG_FROM_POSITION
from train.tb_reader import read_tb_scalars
from utils_18xx.g1889_selfplay_log import compose, play, redrive

import tests.g1889.test_heldout as heldout_tests
from tests.g1889.replay.harness import check_export
from tests.g1889.test_heldout import _buffer, _expected
from tests.g1889.test_training_run import SMALL, _train

MODEL_PATH = "nn/transformer-1889-v2.py"
MAX_PLAYERS = 5
INPUT_SIZE = 5108


@pytest.fixture(autouse=True)
def few_math_threads():
    """Models this small gain nothing from a thread a core, and on a shared
    machine the threads' contention multiplies these tests' time."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(before, 4))
    yield
    torch.set_num_threads(before)


def _vectors(states: np.ndarray) -> np.ndarray:
    game = get_game(TrainingConfig(game="1889"))
    assert game.fill_inputs_batch is not None
    out = np.empty((len(states), INPUT_SIZE), np.float32)
    game.fill_inputs_batch([row.copy() for row in states], out, MAX_PLAYERS, layout_version=3)
    return out


def _outputs(model: torch.nn.Module, states: np.ndarray, masks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        return model.eval()(torch.from_numpy(_vectors(states)), torch.from_numpy(masks).bool())


def _direct(model: torch.nn.Module, config: TrainingConfig, rows: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``test_heldout._direct`` on layout 3: each row's policy KL and squared
    value error summed over its real players, in float64."""
    assert get_model_input_spec(config).packed
    model.eval()
    kls, sqerrs = [], []
    for start in range(0, len(rows["states"]), 128):
        part = {name: column[start:start + 128] for name, column in rows.items()}
        with torch.no_grad():
            logits, values = model(torch.from_numpy(_vectors(part["states"])), torch.from_numpy(part["legal_masks"]).bool())
        log_policy = torch.log_softmax(logits.double(), dim=-1).numpy()
        target = part["policy_targets"].astype(np.float64)
        positive = target > 0
        terms = np.zeros_like(target)
        terms[positive] = target[positive] * (np.log(target[positive]) - log_policy[positive])
        kls.append(terms.sum(axis=1))
        real = np.arange(MAX_PLAYERS)[None] < part["player_counts"][:, None]
        sqerrs.append(((values.double().numpy() - part["value_targets"]) ** 2 * real).sum(axis=1))
    return np.concatenate(kls), np.concatenate(sqerrs)


def _fields(tmp_path: Path) -> dict[str, object]:
    """v2's smoke config at a size for the CPU, writing under ``tmp_path``:
    twelve games an epoch by two workers, every row of an epoch held out."""
    fields = json.loads(Path("train_configs/1889-v2-smoke.json").read_text())
    assert fields["model_path"] == MODEL_PATH and fields["human_start_fraction"] == 0.25
    assert fields["eval_batch_shape_mode"] == "bucketed" and fields["replay_ratio"] == 1.1
    return {
        **fields,
        **SMALL,
        "games_per_epoch": 12,
        "num_workers": 2,
        "eval_min_batch_size": 2,
        "eval_max_batch_size": 4,  # the partition's: 2 workers' 2 leaves
        "buffer_capacity": 65_536,
        "heldout_rows": 1_000_000,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tensorboard_dir": str(tmp_path / "runs"),
    }


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    """The run: an epoch with workers, then a resume in-process."""
    tmp_path = tmp_path_factory.mktemp("v2_run")
    fields = _fields(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields))
    with pytest.MonkeyPatch.context() as monkeypatch:
        _train(monkeypatch, "--config", str(config_path))
        _train(monkeypatch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "2",
               "--num-workers", "0")
    return tmp_path


def test_a_v2_run_plays_trains_scores_checkpoints_resumes_and_loads(run, monkeypatch):
    checkpoints = run / "checkpoints"
    device = torch.device("cpu")
    paths = [checkpoints / f"checkpoint_epoch_{epoch:04d}.pt" for epoch in (1, 2)]
    first, second = (load_checkpoint(path, device) for path in paths)
    config = TrainingConfig.from_json(second["config_json"])
    spec = get_model_input_spec(config)
    assert config.model_path == MODEL_PATH and config.num_workers == 0
    assert TrainingConfig.from_json(first["config_json"]).num_workers == 2
    assert spec.packed and (spec.layout_version, spec.input_size) == (3, INPUT_SIZE)
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
    assert set(rows["player_counts"].tolist()) == {3, 4, 5}

    # --- TensorBoard: the run trained by its rows, and scored each epoch before training on it ---
    scalars = read_tb_scalars(str(run / "runs"))
    assert scalars["self_play_started/games"] == [(1, 3), (2, 3)]
    played = [int(value) for _, value in scalars["self_play_aggregate/total_examples"]]
    begun = [int(value) for _, value in scalars["self_play_started/total_examples"]]
    in_epoch = [int((rows["game_ids"] < second_epoch).sum()), int((rows["game_ids"] >= second_epoch).sum())]
    assert [a + b for a, b in zip(played, begun, strict=True)] == in_epoch
    expected_steps = [round(1.1 * count / 64 * share) for count, share in zip(in_epoch, (1 / 3, 2 / 3), strict=True)]
    assert [value for _, value in scalars["epoch/training_steps"]] == expected_steps == [steps[0], steps[1] - steps[0]]
    for tag in ("epoch/total_loss_avg", "epoch/policy_loss_avg", "epoch/value_loss_avg", "epoch/policy_loss_tile_avg",
                "epoch/policy_kl_track_avg", "epoch/policy_loss_div_avg", "epoch/policy_loss_train_avg"):
        assert [step for step, _ in scalars[tag]] == [1, 2] and all(np.isfinite(v) for _, v in scalars[tag]), tag

    # --- The eval server's epoch: bucketed launches of real states, its time by stage ---
    assert [step for step, _ in scalars["eval_server/launches"]] == [1]  # the resume plays in-process
    (_, real), = scalars["eval_server/real_rows"]
    (_, launched), = scalars["eval_server/launched_rows"]
    assert 0 < real <= launched
    sizes = {int(tag.rsplit("_", 1)[1]) for tag in scalars if tag.startswith("eval_server/launches_at_")}
    assert sizes and sizes <= {1, 2, 4}
    shares = [scalars[f"eval_server/{stage}_share"][0][1] for stage in STAGES]
    assert all(share >= 0 for share in shares) and sum(shares) == pytest.approx(1.0, abs=0.02)

    # --- Checkpoints: loaded back, each is the model that scored the next epoch ---
    monkeypatch.setattr(heldout_tests, "_direct", _direct)
    loaded: Any
    loaded, loaded_config, _ = load_model_from_checkpoint(paths[0], device)
    assert loaded_config.model_path == MODEL_PATH and loaded.cfg.layout_version == 3
    assert (loaded.cfg.input_size, loaded.cfg.num_trunk_rows) == (INPUT_SIZE, 45)
    torch.manual_seed(config.seed)
    initial = create_model(config)
    for epoch, (model, of_epoch) in enumerate((
        (initial, rows["game_ids"] < second_epoch), (loaded, rows["game_ids"] >= second_epoch),
    ), start=1):
        expected = _expected(model, config, {name: column[of_epoch] for name, column in rows.items()})
        assert expected["heldout/rows"] == in_epoch[epoch - 1] and expected["heldout/rows_started"] > 0
        assert "heldout/policy_kl_track" in expected and "heldout/policy_kl_stock" in expected
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
    logits, values = _outputs(final, states, masks)
    rebuilt_logits, rebuilt_values = _outputs(rebuilt, states, masks)
    assert torch.equal(logits, rebuilt_logits) and torch.equal(values, rebuilt_values)
    legal = torch.from_numpy(masks).bool()
    assert torch.isfinite(logits[legal]).all() and (logits[~legal] == -1e9).all()
    earlier_logits, _ = _outputs(loaded, states, masks)
    assert (earlier_logits - logits)[legal].abs().max() > 1e-4  # the second epoch trained

    # --- Training reached v2's own parameters: the choice head, the pair-block bias ---
    before, after = first["model_state_dict"], second["model_state_dict"]
    for name in ("slot_embed.weight", "choice_phase_weight", "choice_hidden.weight", "choice_out.weight",
                 "named_proj.weight", "hex_embed.weight", "tile_proj.weight", "relation_bias_mult"):
        assert not torch.equal(before[name], after[name]), name


def test_the_game_log_plays_a_v2_checkpoint_and_ruby_replays_it(run, monkeypatch):
    path = run / "checkpoints" / "checkpoint_epoch_0002.pt"
    model, config, cp = load_model_from_checkpoint(path, torch.device("cpu"))
    played = play(model, config, num_players=3, seed=7, epoch=int(cp["epoch"]) + 1, simulations=2,
                  device=torch.device("cpu"))
    assert len(played.actions) == len(played.evals) > 100
    log, _notes = compose(played, created_at=1)
    assert check_export("selfplay-v2", log, played.net_worths, redrive(played)[1]) == []

    # train.analyze_game renders RSS games, and says where 1889's are.
    monkeypatch.setattr(sys, "argv", ["analyze_game", str(path), "--device", "cpu"])
    with pytest.raises(SystemExit):
        analyze_game_module.main()
    with pytest.raises(ValueError, match="utils_18xx.g1889_selfplay_log"):
        analyze_game_module.analyze_game(model, torch.device("cpu"), config, 0, 2, checkpoint_path=path)


@pytest.mark.parametrize("name", ["1889-v2.json", "1889-v2-smoke.json"])
def test_v2_s_configs_are_v1_s_but_for_the_model_and_the_run_s_directories(name):
    v2 = json.loads(Path("train_configs", name).read_text())
    v1 = json.loads(Path("train_configs", name.replace("v2", "v1")).read_text())
    differ = {key for key in v1.keys() | v2.keys() if v1.get(key) != v2.get(key)}
    assert differ == {"model_path", "checkpoint_dir", "tensorboard_dir"}
    assert v2["model_path"] == MODEL_PATH
    run_name = name.removesuffix(".json")
    assert (v2["checkpoint_dir"], v2["tensorboard_dir"]) == (f"checkpoints/{run_name}", f"runs/{run_name}")
    config = TrainingConfig.from_json(json.dumps(v2))
    assert get_model_input_spec(config).packed
