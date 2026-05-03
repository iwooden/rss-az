from __future__ import annotations

import pytest

from train.config import TrainingConfig
from train.main import _apply_overrides, _build_parser


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


def test_training_config_rejects_unknown_eval_batch_shape_mode() -> None:
    with pytest.raises(ValueError, match="eval_batch_shape_mode"):
        TrainingConfig(eval_batch_shape_mode="wrong")


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
