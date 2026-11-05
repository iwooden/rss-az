"""A 1889 training run and its resume (1889/model-design.md §8 rows 8.3-8.4), on
the CPU at a small size.

- ``train.main.main`` plays an epoch at 3-5 players through ``NNEvaluator``, trains
  on mixed-player batches, and checkpoints; ``--resume latest`` then loads the
  checkpoint and the buffer and runs the next epoch. The stats hook's scalars and
  the per-phase policy metrics reach TensorBoard, each group's, for both epochs.
- A trainer rebuilt from a checkpoint and a saved buffer continues exactly as the
  original: the same batches, losses, and weights.

The local smoke run covers the eval server and the GPU
(``train_configs/1889-v0-smoke.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from games import get_game
from nn import create_model
from train import main as main_module
from train.checkpoint import load_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.tb_reader import read_tb_scalars
from train.trainer import Trainer

from tests.g1889.phases.test_random_playouts import _choose

MAX_PLAYERS = 5
SMALL = {
    "d_model": 32,
    "num_layers": 1,
    "games_per_epoch": 3,
    "num_epochs": 1,
    "training_steps_per_epoch": 40,
    "num_simulations": 2,
    "search_batch_size": 2,
    "num_workers": 0,
    "buffer_capacity": 8192,
    "min_buffer_size": 64,
    "batch_size": 64,
    "profile": False,
}


def _smoke_config(tmp_path: Path) -> dict[str, object]:
    """The smoke config at a size for the CPU, writing under ``tmp_path``."""
    fields = json.loads(Path("train_configs/1889-v0-smoke.json").read_text())
    return {
        **fields,
        **SMALL,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "tensorboard_dir": str(tmp_path / "runs"),
    }


def _train(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--device", "cpu", "--no-compile", *args])
    main_module.main()


def test_a_run_checkpoints_and_resumes_with_its_scalars(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_smoke_config(tmp_path)))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")

    _train(monkeypatch, "--config", str(config_path))
    first = load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", device)
    assert first["epoch"] == 0
    first_rows = json.loads((checkpoints / "replay_buffer" / "metadata.json").read_text())["size"]
    assert first_rows == first["buffer_stats"]["size"] > 0
    capsys.readouterr()

    _train(monkeypatch, "--config", str(config_path), "--resume", "latest", "--num-epochs", "2")
    output = capsys.readouterr().out
    step = first["trainer_state"]["global_step"]
    assert isinstance(step, int)
    assert f"Resumed from epoch 0, step {step}" in output
    assert f"Loaded {first_rows:,} examples into replay buffer" in output
    assert "Restored RNG state" in output

    second = load_checkpoint(checkpoints / "checkpoint_epoch_0002.pt", device)
    assert second["epoch"] == 1
    second_step = second["trainer_state"]["global_step"]
    assert isinstance(second_step, int) and second_step > step
    assert any(
        not torch.equal(before, second["model_state_dict"][name])
        for name, before in first["model_state_dict"].items()
    )

    scalars = read_tb_scalars(str(tmp_path / "runs"))
    # The second epoch's rows joined the first's.
    new_rows = scalars["self_play_aggregate/total_examples"][1][1]
    assert [value for _, value in scalars["buffer/size"]] == [first_rows, first_rows + new_rows]

    for epoch_tags in (
        *(f"self_play_{group}/{name}" for group in ("aggregate", "3p", "4p", "5p") for name in (
            "end_bank", "end_bankruptcy", "ors_played", "track_pass_share", "corps_floated",
            "last_phase_D", "trains_bought_2", "trains_bought_D",
        )),
        "self_play_aggregate/policy/phase/track/decision_count",
        "self_play_aggregate/policy/phase/stock/target_kl_to_prior_mean",
        "epoch/policy_loss_track_avg",
        "epoch/policy_kl_track_avg",
        *(f"epoch/value_loss_{n}p_avg" for n in (3, 4, 5)),
    ):
        assert [step for step, _ in scalars[epoch_tags]] == [1, 2], epoch_tags
    # Each game ended one way.
    for group in ("aggregate", "3p", "4p", "5p"):
        for (_, bank), (_, bankrupt) in zip(
            scalars[f"self_play_{group}/end_bank"], scalars[f"self_play_{group}/end_bankruptcy"], strict=True,
        ):
            assert bank + bankrupt == pytest.approx(1.0)
    for name in ("total_loss_avg", "policy_loss_avg", "value_loss_avg"):
        assert all(np.isfinite(value) for _, value in scalars[f"epoch/{name}"])


def _buffer(config: TrainingConfig) -> ReplayBuffer:
    game = get_game(config)
    return ReplayBuffer(
        game, config.buffer_capacity, game.state_size(MAX_PLAYERS), MAX_PLAYERS,
        min_players=3, max_players=MAX_PLAYERS,
        priority_fraction=config.replay_priority_fraction, priority_exponent=config.replay_priority_exponent,
    )


def _random_rows(config: TrainingConfig) -> ReplayBuffer:
    """A buffer of random-playout rows at 3-5 players with priorities."""
    game = get_game(config)
    buffer = _buffer(config)
    scratch = np.empty(game.policy.max_action_size, dtype=np.uint16)
    lut = game.policy.build_action_lut().numpy()
    rng = np.random.default_rng(0)
    for num_players in (3, 4, 5):
        state = game.new_game(num_players, MAX_PLAYERS, num_players)
        while True:
            phase = game.decision_phase(state)
            count = game.enumerate_policy_actions(state, scratch)
            slots = lut[phase, scratch[:count]]
            mask = np.zeros((1, game.policy.unified_dim), np.uint8)
            target = np.zeros((1, game.policy.unified_dim), np.float32)
            mask[0, slots] = 1
            target[0, slots] = rng.dirichlet(np.ones(count)).astype(np.float32)
            value = rng.uniform(-1, 1, (1, num_players)).astype(np.float32)
            buffer.add_stacked(
                state._array[None].copy(), np.array([phase], np.int8), mask, target, value,
                num_players=num_players, priorities=rng.uniform(0.01, 1.0, 1).astype(np.float32),
            )
            if game.apply_action(state, _choose(state, rng, True)) == 1:
                break
    return buffer


def test_a_resumed_trainer_continues_exactly(tmp_path):
    config = TrainingConfig.from_json(json.dumps(_smoke_config(tmp_path)))
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = create_model(config)
    trainer = Trainer(model, config, device)
    buffer = _random_rows(config)
    rng = np.random.default_rng(1)
    for _ in range(3):
        trainer.train_step(buffer, config.batch_size, rng)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, 0, model, trainer.state_dict(), config, {}, {"size": len(buffer), "capacity": config.buffer_capacity},
        rng_state=main_module._capture_rng_state(rng),
    )
    buffer.save(tmp_path / "replay_buffer")

    resumed = load_checkpoint(path, device)
    resumed_config = TrainingConfig.from_json(resumed["config_json"])
    resumed_model = create_model(resumed_config)
    resumed_model.load_state_dict(resumed["model_state_dict"])
    resumed_trainer = Trainer(resumed_model, resumed_config, device)
    resumed_trainer.load_state_dict(resumed["trainer_state"])
    resumed_buffer = _buffer(resumed_config)
    assert resumed_buffer.load(tmp_path / "replay_buffer") == len(buffer)
    resumed_rng = np.random.default_rng(99)
    assert "rng_state" in resumed
    main_module._restore_rng_state(resumed_rng, resumed["rng_state"])
    assert resumed_trainer.global_step == trainer.global_step and resumed_trainer.lr == trainer.lr

    for _ in range(3):
        losses = trainer.train_step(buffer, config.batch_size, rng)
        resumed_losses = resumed_trainer.train_step(resumed_buffer, config.batch_size, resumed_rng)
        assert resumed_losses == losses
    for name, weights in model.state_dict().items():
        assert torch.equal(weights, resumed_model.state_dict()[name]), name
