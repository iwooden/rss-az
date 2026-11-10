"""Trainer learning-rate schedule behavior."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from tests.test_replay_priority import U_DIM, _add_rows, _buffer
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer
from train.trainer import Trainer


def _make_trainer(training_steps_per_epoch: int) -> Trainer:
    model = torch.nn.Linear(1, 1)
    cfg = TrainingConfig(
        model_path="nn/transformer-v2.py",
        optimizer="adamw",
        learning_rate=1.0,
        lr_min=0.1,
        warmup_epochs=1.0,
        num_epochs=4,
        lr_decay_end_epoch=4,
        training_steps_per_epoch=training_steps_per_epoch,
    )
    return Trainer(model, cfg, torch.device("cpu"))


@pytest.mark.parametrize("training_steps_per_epoch", [10, 20])
def test_warmup_epochs_scale_with_training_steps_per_epoch(
    training_steps_per_epoch: int,
) -> None:
    trainer = _make_trainer(training_steps_per_epoch)
    lr_lambda = trainer.scheduler.lr_lambdas[0]

    assert lr_lambda(training_steps_per_epoch - 1) == pytest.approx(1.0)
    assert lr_lambda(training_steps_per_epoch) == pytest.approx(1.0)
    assert lr_lambda(training_steps_per_epoch // 2 - 1) == pytest.approx(0.5)


def test_warmup_epochs_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="warmup_epochs"):
        TrainingConfig(warmup_epochs=-0.5)


# ---------------------------------------------------------------------------
# The epoch-based schedule (replay_ratio): 1889/v1-design.md 3.4
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """Constant logits and values, with a matrix for Muon to hold."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(U_DIM))
        self.hidden = torch.nn.Linear(2, 2)

    def input_output_layers(self) -> list[torch.nn.Module]:
        return []

    def forward(self, tokens, masks, relations):
        logits = (self.logits + self.hidden.weight.sum()).expand(len(tokens), -1)
        return logits.masked_fill(~masks, -1e9), torch.zeros(len(tokens), 3)


def _config(*, replay_ratio: float = 0.0, **fields: Any) -> TrainingConfig:
    return TrainingConfig(**{
        "model_path": "nn/transformer-v2.py", "optimizer": "adamw", "learning_rate": 1e-2, "lr_min": 1e-3,
        "num_epochs": 6, "lr_decay_end_epoch": 4, "replay_ratio": replay_ratio,
        "buffer_epochs": 2 if replay_ratio else 0, **fields,
    })


def _rows() -> ReplayBuffer:
    buffer = _buffer(8)
    _add_rows(buffer, list(range(8)))
    return buffer


def _rates(trainer: Trainer, epochs: list[tuple[int, int]], seed: int = 0) -> list[tuple[float, float | None]]:
    """The rates each step trains at, over ``(epoch, steps)`` epochs run as the
    main loop runs them: an epoch without steps never reaches the trainer."""
    buffer, rng = _rows(), np.random.default_rng(seed)
    rates = []
    for epoch, steps in epochs:
        if steps:
            trainer.begin_epoch(epoch, steps)
        for _ in range(steps):
            rates.append((trainer.lr, trainer.aux_lr))
            trainer.train_step(buffer, 8, rng)
    return rates


def _step_schedule(config: TrainingConfig, step: int, steps_per_epoch: int) -> float:
    """The rate at an optimizer step as the trainer set it before the epoch
    schedule existed, ``steps_per_epoch`` being ``training_steps_per_epoch``."""
    total_steps = (config.lr_decay_end_epoch or config.num_epochs) * steps_per_epoch
    warmup_steps = math.ceil(config.warmup_epochs * steps_per_epoch)
    if step < warmup_steps:
        return config.learning_rate * ((step + 1) / max(warmup_steps, 1))
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    floor = config.lr_min / config.learning_rate
    return config.learning_rate * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress)))


@pytest.mark.parametrize("fields", [
    {"training_steps_per_epoch": 10},
    {"training_steps_per_epoch": 7, "warmup_epochs": 0.5},  # the warmup ends mid-step
    {"training_steps_per_epoch": 6, "warmup_epochs": 0.0, "lr_decay_end_epoch": None, "num_epochs": 5},
    {"training_steps_per_epoch": 5, "warmup_epochs": 2.0, "optimizer": "muon"},
])
def test_at_constant_steps_the_epoch_schedule_is_the_step_schedule(fields: dict[str, Any]) -> None:
    """With every epoch the same steps, the two schedules train each step at
    the same rate, on the auxiliary optimizer too, through the warmup, the
    cosine, and the floor past its end. Off, the schedule is what it was."""
    steps = fields["training_steps_per_epoch"]
    epochs = [(epoch, steps) for epoch in range(6)]
    by_step = _config(**fields)
    # The epoch schedule never reads training_steps_per_epoch.
    by_epoch = _config(replay_ratio=1.1, **{**fields, "training_steps_per_epoch": 999})

    step_rates = _rates(Trainer(_TinyModel(), by_step, torch.device("cpu")), epochs)
    epoch_rates = _rates(Trainer(_TinyModel(), by_epoch, torch.device("cpu")), epochs)

    before = [_step_schedule(by_step, step, steps) for step in range(6 * steps)]
    assert [rate for rate, _ in step_rates] == before
    assert [rate for rate, _ in epoch_rates] == pytest.approx(before, rel=1e-12)
    if by_step.optimizer == "muon":
        assert all(aux == rate for rate, aux in step_rates + epoch_rates)
    assert len(set(before)) > steps and before[-1] == pytest.approx(by_step.lr_min)


def test_the_epoch_schedule_reads_progress_in_epochs_whatever_their_steps() -> None:
    """Epochs of any steps: the rate at a step is the schedule's at (epochs
    finished + the share of this epoch's steps done), an untrained epoch
    counting as finished."""
    config = _config(replay_ratio=1.1)
    epochs = [(0, 3), (1, 8), (2, 0), (3, 5), (4, 2), (5, 6)]
    rates = [rate for rate, _ in _rates(Trainer(_TinyModel(), config, torch.device("cpu")), epochs)]

    def at(progress: float, step_share: float) -> float:
        if progress < config.warmup_epochs:  # 1 epoch, reaching the full rate on its last step
            return config.learning_rate * (progress + step_share)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min((progress - 1) / (4 - 1), 1.0)))
        return config.lr_min + (config.learning_rate - config.lr_min) * cosine

    expected = [at(epoch + step / steps, 1 / steps) for epoch, steps in epochs for step in range(steps)]
    assert rates == pytest.approx(expected, rel=1e-12)
    # Epoch 3 begins at the same rate whether or not epoch 2 trained.
    trained = [rate for rate, _ in _rates(
        Trainer(_TinyModel(), config, torch.device("cpu")), [(0, 3), (1, 8), (2, 4), (3, 5)],
    )]
    assert trained[-5:] == rates[11:16]


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("resumed_steps", [5, 9])
def test_a_resumed_epoch_schedule_continues_exactly(tmp_path, optimizer: str, resumed_steps: int) -> None:
    """A trainer rebuilt from saved state trains its next steps at the rates
    the uninterrupted one does: at an epoch boundary, and when the epoch it
    resumes into has other steps than the last, as the replayed epoch of a
    graceful mid-epoch shutdown does with the rows it already held."""
    config = _config(replay_ratio=1.1, optimizer=optimizer, warmup_epochs=1.5, lr_decay_end_epoch=6)
    before, after = [(0, 4), (1, 0), (2, 6)], [(3, resumed_steps), (4, 5)]
    whole = _rates(Trainer(_TinyModel(), config, torch.device("cpu")), before + after)

    torch.manual_seed(0)
    first = Trainer(_TinyModel(), config, torch.device("cpu"))
    rates = _rates(first, before)
    torch.save(first.state_dict(), tmp_path / "trainer.pt")
    resumed = Trainer(_TinyModel(), config, torch.device("cpu"))
    resumed.load_state_dict(torch.load(tmp_path / "trainer.pt", weights_only=False))
    assert resumed.global_step == 10 and resumed.lr == first.lr

    assert rates + _rates(resumed, after) == whole
    assert len({rate for rate, _ in whole}) == len(whole)  # every step's rate differs: none could stand in


def test_the_epoch_schedule_needs_each_epochs_steps() -> None:
    """Without ``begin_epoch`` the schedule has no progress to read: training
    fails rather than run at some rate. The step schedule ignores it."""
    trainer = Trainer(_TinyModel(), _config(replay_ratio=1.1), torch.device("cpu"))
    buffer, rng = _rows(), np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="begin_epoch"):
        trainer.train_step(buffer, 8, rng)
    trainer.begin_epoch(0, 2)
    trainer.train_step(buffer, 8, rng)
    trainer.train_step(buffer, 8, rng)
    with pytest.raises(RuntimeError, match="begin_epoch"):
        trainer.train_step(buffer, 8, rng)

    by_step = _config(training_steps_per_epoch=4)
    plain = _rates(Trainer(_TinyModel(), by_step, torch.device("cpu")), [(0, 12)])
    # Steps and epochs that the step count would not give: nothing changes.
    told = _rates(Trainer(_TinyModel(), by_step, torch.device("cpu")), [(7, 5), (2, 7)])
    assert plain == told
