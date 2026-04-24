"""Replay-buffer fullness scaling for per-epoch training updates."""

from __future__ import annotations

from train.config import TrainingConfig
from train.main import _apply_overrides, _build_parser, _scaled_training_steps


def _config(training_steps_per_epoch: int = 1000) -> TrainingConfig:
    return TrainingConfig(
        buffer_capacity=500_000,
        min_buffer_size=10_000,
        training_steps_per_epoch=training_steps_per_epoch,
    )


def test_scaled_training_steps_skip_below_min_buffer_size() -> None:
    assert _scaled_training_steps(_config(), 9_999) == 0


def test_scaled_training_steps_follow_buffer_fullness() -> None:
    assert _scaled_training_steps(_config(), 100_000) == 200
    assert _scaled_training_steps(_config(), 250_000) == 500


def test_scaled_training_steps_cap_at_configured_step_count() -> None:
    assert _scaled_training_steps(_config(), 500_000) == 1000
    assert _scaled_training_steps(_config(), 750_000) == 1000


def test_scaled_training_steps_keep_at_least_one_step_when_trainable() -> None:
    assert (
        _scaled_training_steps(_config(training_steps_per_epoch=10), 10_000)
        == 1
    )


def test_cli_overrides_training_steps_per_epoch() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--training-steps-per-epoch", "2000"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.training_steps_per_epoch == 2000
