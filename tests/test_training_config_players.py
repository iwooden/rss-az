from __future__ import annotations

from pathlib import Path

import pytest

from train.config import TrainingConfig
from train.logging import _format_player_range
from train.main import _apply_overrides, _build_parser


def test_legacy_num_players_config_validates_with_effective_single_range() -> None:
    config = TrainingConfig.from_json('{"num_players": 3}')

    assert config.num_players == 3
    assert config.min_players == 0
    assert config.max_players == 0
    assert config.effective_min_players == 3
    assert config.effective_max_players == 3
    assert not config.is_mixed_player_training
    assert list(config.iter_player_counts()) == [3]


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


def test_existing_3p_training_config_still_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "train_configs" / "3p.json"

    config = TrainingConfig.from_json(config_path.read_text())

    assert config.num_players == 3
    assert config.effective_min_players == 3
    assert config.effective_max_players == 3
