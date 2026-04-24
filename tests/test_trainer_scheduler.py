"""Trainer learning-rate schedule behavior."""

from __future__ import annotations

import pytest
import torch

from train.config import TrainingConfig
from train.trainer import Trainer


def _make_trainer(training_steps_per_epoch: int) -> Trainer:
    model = torch.nn.Linear(1, 1)
    cfg = TrainingConfig(
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
