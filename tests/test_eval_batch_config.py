from __future__ import annotations

import pytest
import torch

from train.config import TrainingConfig
from train.main import _apply_overrides, _build_parser, _resolve_eval_devices


def test_training_config_defaults_to_dynamic_eval_batch_shapes() -> None:
    config = TrainingConfig()

    assert config.eval_batch_shape_mode == "dynamic"
    assert config.eval_max_batch_size == 0


def test_cli_overrides_eval_batch_shape_mode_and_eval_max_batch_size() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--eval-batch-shape-mode", "bucketed",
        "--eval-max-batch-size", "16",
    ])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.eval_batch_shape_mode == "bucketed"
    assert config.eval_max_batch_size == 16


def test_cli_overrides_eval_devices() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--num-eval-servers", "2",
        "--eval-devices", "cuda:0,cuda:1",
    ])
    config = TrainingConfig(num_workers=8)

    _apply_overrides(config, args)
    config.validate()

    assert config.eval_devices == ["cuda:0", "cuda:1"]


def test_resolve_eval_devices_defaults_to_training_device() -> None:
    config = TrainingConfig(num_workers=4, num_eval_servers=2)

    devices = _resolve_eval_devices(config, torch.device("cpu"))

    assert devices == [torch.device("cpu"), torch.device("cpu")]


def test_resolve_eval_devices_uses_explicit_mapping() -> None:
    config = TrainingConfig(
        num_workers=4,
        num_eval_servers=2,
        eval_devices=["cuda:0", "cuda:1"],
    )

    devices = _resolve_eval_devices(config, torch.device("cpu"))

    assert devices == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_cli_overrides_policy_target_temperature_schedule() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--policy-target-temp-initial", "1.0",
        "--policy-target-temp-final", "0.25",
        "--policy-target-temp-anneal-start", "40",
        "--policy-target-temp-anneal-end", "100",
    ])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.policy_target_temp_initial == 1.0
    assert config.policy_target_temp_final == 0.25
    assert config.policy_target_temp_anneal_start == 40
    assert config.policy_target_temp_anneal_end == 100


def test_scalar_temperature_windows_expand_to_player_count_arrays() -> None:
    config = TrainingConfig(
        num_players=0,
        min_players=3,
        max_players=5,
        temp_anneal_start=40,
        temp_anneal_end=80,
        policy_target_temp_anneal_start=50,
        policy_target_temp_anneal_end=100,
    )

    assert config.temp_anneal_starts == [40, 40, 40]
    assert config.temp_anneal_ends == [80, 80, 80]
    assert config.policy_target_temp_anneal_starts == [50, 50, 50]
    assert config.policy_target_temp_anneal_ends == [100, 100, 100]


def test_per_player_temperature_windows_used_when_scalar_zero() -> None:
    config = TrainingConfig(
        num_players=0,
        min_players=3,
        max_players=5,
        temp_anneal_start=0,
        temp_anneal_end=0,
        temp_anneal_starts=[30, 45, 60],
        temp_anneal_ends=[60, 90, 120],
        policy_target_temp_anneal_start=0,
        policy_target_temp_anneal_end=0,
        policy_target_temp_anneal_starts=[20, 35, 50],
        policy_target_temp_anneal_ends=[40, 70, 100],
    )

    assert config.temp_anneal_window(3) == (30, 60)
    assert config.temp_anneal_window(4) == (45, 90)
    assert config.temp_anneal_window(5) == (60, 120)
    assert config.policy_target_temp_anneal_window(3) == (20, 40)
    assert config.policy_target_temp_anneal_window(4) == (35, 70)
    assert config.policy_target_temp_anneal_window(5) == (50, 100)


def test_per_player_temperature_windows_require_full_range() -> None:
    with pytest.raises(ValueError, match="temp_anneal_starts"):
        TrainingConfig(
            num_players=0,
            min_players=3,
            max_players=5,
            temp_anneal_start=0,
            temp_anneal_end=0,
            temp_anneal_starts=[30, 45],
            temp_anneal_ends=[60, 90],
        )


def test_cli_overrides_per_player_temperature_windows() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--temp-anneal-start", "0",
        "--temp-anneal-end", "0",
        "--temp-anneal-starts", "30,45,60",
        "--temp-anneal-ends", "60,90,120",
    ])
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    _apply_overrides(config, args)
    config.validate()

    assert config.temp_anneal_starts == [30, 45, 60]
    assert config.temp_anneal_ends == [60, 90, 120]


def test_cli_overrides_dirichlet_epsilon() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--dirichlet-epsilon", "0.15"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.dirichlet_epsilon == 0.15


def test_cli_overrides_buffer_capacity() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--buffer-capacity", "250000"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.buffer_capacity == 250_000


def test_cli_overrides_weight_decay() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--weight-decay", "0.02"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.weight_decay == 0.02


def test_training_config_rejects_unknown_eval_batch_shape_mode() -> None:
    with pytest.raises(ValueError, match="eval_batch_shape_mode"):
        TrainingConfig(eval_batch_shape_mode="wrong")


def test_training_config_rejects_eval_devices_count_mismatch() -> None:
    with pytest.raises(ValueError, match="eval_devices length"):
        TrainingConfig(
            num_workers=8,
            num_eval_servers=2,
            eval_devices=["cuda:0"],
        )


def test_training_config_rejects_inverted_policy_target_temp_schedule() -> None:
    with pytest.raises(ValueError, match="policy_target_temp_anneal_start"):
        TrainingConfig(
            policy_target_temp_anneal_start=120,
            policy_target_temp_anneal_end=60,
        )


def test_training_config_rejects_price_slot_residual_blend_out_of_range() -> None:
    with pytest.raises(ValueError, match="price_slot_residual_scale"):
        TrainingConfig(price_slot_residual_scale=-0.1)

    with pytest.raises(ValueError, match="price_slot_residual_scale"):
        TrainingConfig(price_slot_residual_scale=1.1)


def test_training_config_validates_max_acq_price_actions() -> None:
    config = TrainingConfig(max_acq_price_actions=10)
    assert config.to_mcts_config().max_acq_price_actions == 10

    with pytest.raises(ValueError, match="max_acq_price_actions"):
        TrainingConfig(max_acq_price_actions=-2)
    with pytest.raises(ValueError, match="divisible by 2"):
        TrainingConfig(max_acq_price_actions=9)


def test_training_config_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="model_type"):
        TrainingConfig(model_type="mlp")


def test_training_config_rejects_invalid_resnet_hyperparameters() -> None:
    with pytest.raises(ValueError, match="resnet_hidden_dim"):
        TrainingConfig(model_type="resnet", resnet_hidden_dim=0)

    with pytest.raises(ValueError, match="resnet_num_blocks"):
        TrainingConfig(model_type="resnet", resnet_num_blocks=-1)


def test_training_config_rejects_eval_max_batch_size_in_dynamic_mode() -> None:
    with pytest.raises(ValueError, match="dynamic"):
        TrainingConfig(eval_batch_shape_mode="dynamic", eval_max_batch_size=16)


def test_training_config_rejects_non_power_of_two_bucketed_eval_max_batch_size() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        TrainingConfig(eval_batch_shape_mode="bucketed", eval_max_batch_size=30)


def test_training_config_rejects_bucketed_eval_max_batch_size_above_partition_cap() -> None:
    with pytest.raises(ValueError, match=r"partition_size \* search_batch_size"):
        TrainingConfig(
            num_workers=8,
            num_eval_servers=2,
            search_batch_size=8,
            eval_batch_shape_mode="bucketed",
            eval_max_batch_size=64,
        )


def test_training_config_rejects_bucketed_eval_max_batch_size_below_min_batch_size() -> None:
    with pytest.raises(ValueError, match="eval_min_batch_size"):
        TrainingConfig(
            num_workers=8,
            num_eval_servers=1,
            search_batch_size=8,
            eval_min_batch_size=32,
            eval_batch_shape_mode="bucketed",
            eval_max_batch_size=16,
        )


def test_training_config_accepts_power_of_two_bucketed_eval_max_batch_size_at_partition_cap() -> None:
    config = TrainingConfig(
        num_workers=8,
        num_eval_servers=2,
        search_batch_size=8,
        eval_batch_shape_mode="bucketed",
        eval_max_batch_size=32,
    )

    assert config.eval_batch_shape_mode == "bucketed"
    assert config.eval_max_batch_size == 32
