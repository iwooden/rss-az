"""Replay-buffer fullness scaling for per-epoch training updates."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_replay_epochs import _buffer, _Writer
from train.config import TrainingConfig
from train.main import _apply_overrides, _build_parser, _epoch_training_steps, _scaled_training_steps


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


def test_scaled_training_steps_follow_the_share_of_buffer_epochs_held() -> None:
    """With ``buffer_epochs``, fullness is the window's epochs over
    ``buffer_epochs``, whatever share of the capacity its rows are."""
    config = _config()
    config.buffer_epochs = 4
    assert _scaled_training_steps(config, 9_999, window_epochs=4) == 0
    assert _scaled_training_steps(config, 20_000, window_epochs=1) == 250
    assert _scaled_training_steps(config, 20_000, window_epochs=3) == 750
    assert _scaled_training_steps(config, 20_000, window_epochs=4) == 1000
    assert _scaled_training_steps(config, 499_999, window_epochs=1) == 250
    # Off, the epochs held change nothing.
    assert _scaled_training_steps(_config(), 100_000, window_epochs=4) == 200
    with pytest.raises(ValueError, match="buffer_epochs must be >= 0"):
        TrainingConfig(buffer_epochs=-1).validate()


# ---------------------------------------------------------------------------
# Steps from a replay ratio (1889/v1-design.md 3.3)
# ---------------------------------------------------------------------------


def _ratio_config(**fields) -> TrainingConfig:
    return TrainingConfig(**{
        "buffer_capacity": 8192, "min_buffer_size": 64, "batch_size": 16,
        "buffer_epochs": 3, "replay_ratio": 1.1, **fields,
    })


def test_replay_ratio_steps_follow_the_epochs_rows_and_the_windows_share() -> None:
    config = _ratio_config()
    buffer = _buffer(config.buffer_capacity, buffer_epochs=3)
    writer = _Writer(buffer)
    steps = []
    for epoch, game_rows in enumerate([(30,), (300, 340), (640,), (200, 440), (100, 60), (1,)]):
        writer.add_epoch(epoch, game_rows)
        steps.append(_epoch_training_steps(config, buffer))
    # Under min_buffer_size nothing trains. Then 1.1 * rows / 16, times the
    # window's share of 3 epochs, whatever the game sizes; never under a step.
    assert steps == [0, round(44 * 2 / 3), 44, 44, 11, 1]
    assert buffer.epoch_rows == 1 and buffer.window_epochs == 3

    # training_steps_per_epoch and the capacity are not read.
    other = _ratio_config(training_steps_per_epoch=7, buffer_capacity=100_000)
    assert _epoch_training_steps(other, buffer) == 1
    with pytest.raises(ValueError, match="replay_ratio > 0 requires buffer_epochs > 0"):
        TrainingConfig(replay_ratio=1.1)
    with pytest.raises(ValueError, match="replay_ratio must be >= 0"):
        TrainingConfig(replay_ratio=-1.0, buffer_epochs=2)


def test_replay_ratio_counts_the_rows_a_replayed_epoch_already_held(tmp_path) -> None:
    """A graceful shutdown mid-epoch leaves the epoch its games; the resume
    replays the epoch in full, and its steps cover both."""
    config = _ratio_config()
    buffer = _buffer(config.buffer_capacity, buffer_epochs=3)
    writer = _Writer(buffer)
    for epoch in range(3):
        writer.add_epoch(epoch, (320,))
    writer.add_epoch(3, (160,))  # then the shutdown
    buffer.save(tmp_path)

    resumed = _buffer(config.buffer_capacity, buffer_epochs=3)
    assert resumed.load(tmp_path) == len(buffer)
    _Writer(resumed).add_epoch(3, (320,))
    assert resumed.epoch_rows == 480
    assert _epoch_training_steps(config, resumed) == 33  # 1.1 * 480 / 16

    # A resume at an epoch boundary begins the next epoch with its own rows.
    boundary = _buffer(config.buffer_capacity, buffer_epochs=3)
    boundary.load(tmp_path)
    _Writer(boundary).add_epoch(4, (320,))
    assert boundary.epoch_rows == 320 and _epoch_training_steps(config, boundary) == 22


def test_replay_ratio_draws_each_row_about_that_often_over_its_epochs() -> None:
    """Equal epochs of 640 rows through a window of 3: the steps each epoch
    trains, sampled as the trainer samples, draw every epoch's rows about
    ``replay_ratio`` times over the 3 epochs they are in the window, the first
    epochs' as the rest. Without the window's share the first epoch's rows
    would be drawn 1 + 1/2 + 1/3 times as often."""
    config = _ratio_config()
    buffer = _buffer(config.buffer_capacity, buffer_epochs=3)
    writer = _Writer(buffer)
    rng = np.random.default_rng(3)
    expected = np.zeros(7)
    drawn = np.zeros(7 * 640)
    for epoch in range(7):
        writer.add_epoch(epoch, (640,))
        steps = _epoch_training_steps(config, buffer)
        in_window = range(max(0, epoch - 2), epoch + 1)
        assert buffer.window_size == 640 * len(in_window)
        expected[list(in_window)] += steps * config.batch_size / buffer.window_size
        for _ in range(steps):
            tags = buffer.sample(config.batch_size, rng)["value_targets"][:, 0].numpy().astype(int)
            np.add.at(drawn, tags, 1)
    # Epochs 0-4 have left the window, or are about to.
    assert expected[:5] == pytest.approx(1.1, rel=0.01)
    assert drawn.reshape(7, 640).mean(axis=1)[:5] == pytest.approx(1.1, rel=0.1)


def test_without_a_replay_ratio_steps_are_scaled_by_fullness() -> None:
    for buffer_epochs in (0, 2):
        config = TrainingConfig(
            buffer_capacity=4096, min_buffer_size=64, batch_size=16, training_steps_per_epoch=50,
            buffer_epochs=buffer_epochs,
        )
        buffer = _buffer(config.buffer_capacity, buffer_epochs=buffer_epochs)
        writer = _Writer(buffer)
        steps = []
        for epoch, rows in enumerate((40, 60, 924, 1024)):
            writer.add_epoch(epoch, (rows,))
            steps.append(_epoch_training_steps(config, buffer))
        # Today's rule: the share of 2 epochs held, or of the capacity filled.
        assert steps == ([0, 50, 50, 50] if buffer_epochs else [0, 1, 12, 25])


def test_cli_overrides_training_steps_per_epoch() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--training-steps-per-epoch", "2000"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.training_steps_per_epoch == 2000


def test_cli_overrides_phase_conditioning() -> None:
    parser = _build_parser()

    args = parser.parse_args(["--no-phase-conditioning"])
    config = TrainingConfig(phase_conditioning=True)
    _apply_overrides(config, args)
    config.validate()
    assert config.phase_conditioning is False

    args = parser.parse_args(["--phase-conditioning"])
    config = TrainingConfig(phase_conditioning=False)
    _apply_overrides(config, args)
    config.validate()
    assert config.phase_conditioning is True


def test_json_config_threads_phase_conditioning() -> None:
    config = TrainingConfig.from_json('{"phase_conditioning": false}')

    assert config.phase_conditioning is False
    assert '"phase_conditioning": false' in config.to_json()


def test_relation_input_mixing_config_and_cli_round_trip() -> None:
    assert TrainingConfig.from_json('{}').relation_input_mixing is True
    config = TrainingConfig.from_json('{"relation_input_mixing": false}')
    assert config.relation_input_mixing is False
    assert TrainingConfig.from_json(config.to_json()).relation_input_mixing is False
    parser = _build_parser()
    _apply_overrides(config, parser.parse_args([]))
    assert config.relation_input_mixing is False
    for option, enabled in (("--relation-input-mixing", True), ("--no-relation-input-mixing", False)):
        _apply_overrides(config, parser.parse_args([option]))
        config.validate()
        assert config.relation_input_mixing is enabled
    for invalid in ('1', '"false"', 'null'):
        with pytest.raises(ValueError, match="relation_input_mixing must be bool"):
            TrainingConfig.from_json('{"relation_input_mixing":' + invalid + '}')


def test_zero_sum_values_config_and_cli_round_trip() -> None:
    assert TrainingConfig.from_json('{}').zero_sum_values is False
    config = TrainingConfig.from_json('{"zero_sum_values": true}')
    assert config.zero_sum_values is True
    assert TrainingConfig.from_json(config.to_json()).zero_sum_values is True
    parser = _build_parser()
    _apply_overrides(config, parser.parse_args([]))
    assert config.zero_sum_values is True
    for option, enabled in (("--no-zero-sum-values", False), ("--zero-sum-values", True)):
        _apply_overrides(config, parser.parse_args([option]))
        config.validate()
        assert config.zero_sum_values is enabled
    for invalid in ('1', '"true"', 'null'):
        with pytest.raises(ValueError, match="zero_sum_values must be bool"):
            TrainingConfig.from_json('{"zero_sum_values":' + invalid + '}')


def test_cli_overrides_model_path() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--model-path", "nn/transformer-v2.py"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.model_path == "nn/transformer-v2.py"


def test_json_config_threads_model_path() -> None:
    config = TrainingConfig.from_json('{"model_path": " nn/transformer-v2.py "}')

    assert config.model_path == "nn/transformer-v2.py"
    assert '"model_path": "nn/transformer-v2.py"' in config.to_json()


def test_cli_overrides_transformer_size_hyperparameters() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--d-model", "320",
        "--d-proj", "96",
        "--num-heads", "5",
        "--num-layers", "18",
        "--ff-mult", "2.67",
    ])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.d_model == 320
    assert config.d_proj == 96
    assert config.num_heads == 5
    assert config.num_layers == 18
    assert config.ff_mult == 2.67


def test_json_config_threads_transformer_size_hyperparameters() -> None:
    config = TrainingConfig.from_json(
        '{"d_model": 320, "d_proj": 96, "num_heads": 5, '
        '"num_layers": 18, "ff_mult": 2.67}'
    )

    assert config.d_model == 320
    assert config.d_proj == 96
    assert config.num_heads == 5
    assert config.num_layers == 18
    assert config.ff_mult == 2.67
    assert '"d_model": 320' in config.to_json()
