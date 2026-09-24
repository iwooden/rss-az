from __future__ import annotations

import pytest

from train.config import TrainingConfig
from train.logging import _format_player_range
from train.main import _apply_overrides, _build_parser


def test_v3_behavior_is_explicit_validated_and_serialized():
    assert TrainingConfig().v3_behavior is False
    # Model selection does not silently change engine behavior.
    assert TrainingConfig(model_path="nn/transformer-v3.py").v3_behavior is False
    config = TrainingConfig.from_json('{"v3_behavior": true}')
    assert config.v3_behavior is True
    assert TrainingConfig.from_json(config.to_json()).v3_behavior is True
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="v3_behavior must be bool"):
            TrainingConfig(v3_behavior=invalid)  # type: ignore[arg-type]
    for option, expected in (("--v3-behavior", True), ("--no-v3-behavior", False)):
        args = _build_parser().parse_args([option])
        _apply_overrides(config, args)
        assert config.v3_behavior is expected


def test_cross_player_acquisition_config_and_cli_round_trip():
    assert TrainingConfig().acq_same_president is True
    config = TrainingConfig.from_json('{"v3_behavior": true, "acq_same_president": false}')
    assert TrainingConfig.from_json(config.to_json()).acq_same_president is False
    for invalid in (1, "false", None):
        with pytest.raises(ValueError, match="acq_same_president must be bool"):
            TrainingConfig(acq_same_president=invalid)  # type: ignore[arg-type]
    for option, expected in (("--acq-same-president", True), ("--no-acq-same-president", False)):
        args = _build_parser().parse_args([option])
        _apply_overrides(config, args)
        assert config.acq_same_president is expected


def test_legacy_num_players_config_validates_with_effective_single_range() -> None:
    config = TrainingConfig.from_json('{"num_players": 3}')

    assert config.num_players == 3
    assert config.min_players == 0
    assert config.max_players == 0
    assert config.effective_min_players == 3
    assert config.effective_max_players == 3
    assert not config.is_mixed_player_training
    assert list(config.iter_player_counts()) == [3]


def test_acquisition_price_cap_config_and_cli_round_trip():
    config = TrainingConfig.from_json('{"max_acq_price_actions": 8}')
    assert TrainingConfig.from_json(config.to_json()).max_acq_price_actions == 8
    assert config.to_mcts_config().max_acq_price_actions == 8
    for invalid in (-2, 9, 52, True, 8.0, "8"):
        with pytest.raises(ValueError, match="max_acq_price_actions"):
            TrainingConfig(max_acq_price_actions=invalid)  # type: ignore[arg-type]
    args = _build_parser().parse_args(["--max-acq-price-actions", "0"])
    _apply_overrides(config, args)
    assert config.max_acq_price_actions == 0


def test_mixed_player_config_validates_with_effective_range() -> None:
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    assert config.effective_min_players == 3
    assert config.effective_max_players == 5
    assert config.is_mixed_player_training
    assert list(config.iter_player_counts()) == [3, 4, 5]


def test_startup_player_logging_formats_single_and_mixed_ranges() -> None:
    assert _format_player_range(TrainingConfig(num_players=3)) == "3"
    assert (
        _format_player_range(
            TrainingConfig(num_players=0, min_players=3, max_players=5)
        )
        == "3-5"
    )


def test_training_config_rejects_all_zero_player_counts() -> None:
    with pytest.raises(ValueError, match="num_players or both min_players and max_players"):
        TrainingConfig(num_players=0, min_players=0, max_players=0)


def test_training_config_rejects_mutually_exclusive_player_modes() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(num_players=3, min_players=3, max_players=5)


@pytest.mark.parametrize(
    "min_players,max_players",
    [
        (3, 0),
        (0, 5),
    ],
)
def test_training_config_rejects_partial_player_ranges(
    min_players: int,
    max_players: int,
) -> None:
    with pytest.raises(ValueError, match="both be set"):
        TrainingConfig(
            num_players=0,
            min_players=min_players,
            max_players=max_players,
        )


@pytest.mark.parametrize(
    "min_players,max_players,match",
    [
        (2, 5, "min_players"),
        (3, 6, "max_players"),
        (5, 5, "min_players must be < max_players"),
        (5, 3, "min_players must be < max_players"),
    ],
)
def test_training_config_rejects_invalid_player_ranges(
    min_players: int,
    max_players: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TrainingConfig(
            num_players=0,
            min_players=min_players,
            max_players=max_players,
        )


def test_to_mcts_config_requires_actual_num_players_in_mixed_mode() -> None:
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    with pytest.raises(ValueError, match="num_players must be passed"):
        config.to_mcts_config()

    mcts_config = config.to_mcts_config(num_players=4)

    assert mcts_config.num_players == 4


def test_to_mcts_config_carries_nonfinite_check_flag() -> None:
    config = TrainingConfig(check_nonfinite_mcts=False)

    mcts_config = config.to_mcts_config()

    assert mcts_config.check_nonfinite is False


def test_to_mcts_config_rejects_actual_num_players_outside_configured_range() -> None:
    config = TrainingConfig(num_players=0, min_players=3, max_players=5)

    with pytest.raises(ValueError, match="configured player range"):
        config.to_mcts_config(num_players=2)


def test_cli_overrides_mixed_player_range() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "--num-players", "0",
        "--min-players", "3",
        "--max-players", "5",
    ])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.num_players == 0
    assert config.min_players == 3
    assert config.max_players == 5
    assert config.is_mixed_player_training


def test_cli_can_disable_mcts_nonfinite_checks() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--no-check-nonfinite-mcts"])
    config = TrainingConfig()

    _apply_overrides(config, args)
    config.validate()

    assert config.check_nonfinite_mcts is False
