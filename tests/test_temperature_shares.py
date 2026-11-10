"""The action temperature's window as a share of game length
(1889/v1-design.md 2.4).

- ``temp_anneal_start_pcts`` / ``_end_pcts`` become move numbers each epoch
  from the last finished epoch's mean game length at each player count, and
  reach ``play_game`` in ``EpochConfig``.
- A count without a length, as every count in a run's first epoch, plays at
  the initial temperature throughout.
- Lengths are over the games played from the opening.
- Unset, the config's move numbers apply as before, and the policy-target
  temperature keeps its own move numbers either way.

The run through ``train.main`` with a resume is in
``tests/g1889/test_training_run.py``.
"""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace
from typing import Any

import pytest

from train.config import EpochConfig, TrainingConfig
from train.main import _OpeningGameLengths
from train.replay_buffer import FLAG_FROM_POSITION
from train.self_play import _compute_policy_target_temperature, _compute_temperature

MIXED: dict[str, Any] = {"num_players": 0, "min_players": 3, "max_players": 5}


def _mixed(**fields) -> TrainingConfig:
    return TrainingConfig(**{
        **MIXED,
        "temp_anneal_start_pcts": [0.3, 0.3, 0.25], "temp_anneal_end_pcts": [0.6, 0.6, 0.75],
        **fields,
    })


def _temperatures(config: TrainingConfig, epoch_config: EpochConfig | None, num_players: int, moves) -> list[float]:
    return [_compute_temperature(move, config, num_players, epoch_config) for move in moves]


def test_shares_of_the_last_epochs_length_become_the_window() -> None:
    """30% and 60% at 3 players, after an epoch whose 3-player games averaged
    500 moves: the temperature anneals from move 150 to move 300."""
    config = TrainingConfig(temp_anneal_start_pcts=[0.3], temp_anneal_end_pcts=[0.6])
    epoch_config = config.compute_epoch_config(7, {3: 500.0})
    assert epoch_config.temp_anneal_windows == {3: (150, 300)}
    assert _temperatures(config, epoch_config, 3, (0, 150, 225, 300, 900)) == pytest.approx(
        [1.0, 1.0, 0.75, 0.5, 0.5]
    )
    # The config's own move numbers, 60-120, are not read; the target's are.
    assert config.temp_anneal_window(3) == (60, 120)
    assert _compute_temperature(100, config, 3, epoch_config) == 1.0
    assert _compute_policy_target_temperature(90, config, 3) == pytest.approx(0.75)


def test_each_player_count_reads_its_own_length_and_none_means_the_initial_temperature() -> None:
    config = _mixed(temp_initial=1.2, temp_final=0.4)
    # The first epoch has no lengths: every count plays at the initial temperature.
    for first in (config.compute_epoch_config(0), config.compute_epoch_config(0, {})):
        assert first.temp_anneal_windows == {}
        for num_players in (3, 4, 5):
            assert _temperatures(config, first, num_players, (0, 50, 5000)) == [1.2, 1.2, 1.2]

    # 4 players has had no game from the opening yet.
    epoch_config = config.compute_epoch_config(3, {3: 400.0, 5: 801.0})
    assert epoch_config.temp_anneal_windows == {3: (120, 240), 5: (200, 601)}
    assert _temperatures(config, epoch_config, 3, (120, 180, 240)) == pytest.approx([1.2, 0.8, 0.4])
    assert _temperatures(config, epoch_config, 4, (0, 180, 5000)) == [1.2, 1.2, 1.2]
    assert _temperatures(config, epoch_config, 5, (200, 601)) == pytest.approx([1.2, 0.4])
    # The windows survive the trip to a worker.
    assert pickle.loads(pickle.dumps(epoch_config)) == epoch_config


def test_a_window_may_be_empty_or_start_at_the_first_move() -> None:
    config = TrainingConfig(temp_anneal_start_pcts=[0.0], temp_anneal_end_pcts=[0.0])
    epoch_config = config.compute_epoch_config(1, {3: 300.0})
    # As with move numbers, the initial temperature holds through the start move.
    assert _temperatures(config, epoch_config, 3, (0, 1)) == [1.0, 0.5]
    config = TrainingConfig(temp_anneal_start_pcts=[0.5], temp_anneal_end_pcts=[0.5])
    epoch_config = config.compute_epoch_config(1, {3: 300.0})
    assert _temperatures(config, epoch_config, 3, (149, 150, 151)) == [1.0, 1.0, 0.5]
    config = TrainingConfig(temp_anneal_start_pcts=[0.0], temp_anneal_end_pcts=[1.0])
    epoch_config = config.compute_epoch_config(1, {3: 300.0})
    assert _temperatures(config, epoch_config, 3, (0, 150, 300)) == pytest.approx([1.0, 0.75, 0.5])


def test_unset_shares_leave_the_move_number_windows() -> None:
    config = TrainingConfig(
        **MIXED, temp_anneal_start=0, temp_anneal_end=0,
        temp_anneal_starts=[10, 20, 30], temp_anneal_ends=[20, 40, 60],
        policy_target_temp_anneal_start=0, policy_target_temp_anneal_end=0,
        policy_target_temp_anneal_starts=[5, 10, 15], policy_target_temp_anneal_ends=[15, 30, 45],
    )
    assert config.temp_anneal_start_pcts == config.temp_anneal_end_pcts == []
    # Lengths, were the main loop to pass them, change nothing.
    epoch_config = config.compute_epoch_config(5, {3: 500.0, 4: 600.0, 5: 700.0})
    assert epoch_config.temp_anneal_windows is None
    assert epoch_config == config.compute_epoch_config(5)
    for num_players, moves in ((3, (10, 15, 20)), (5, (30, 45, 60))):
        expected = [1.0, 0.75, 0.5]
        assert _temperatures(config, epoch_config, num_players, moves) == pytest.approx(expected)
        assert _temperatures(config, None, num_players, moves) == pytest.approx(expected)
        assert [_compute_temperature(move, config, num_players) for move in moves] == pytest.approx(expected)


@pytest.mark.parametrize(("fields", "message"), [
    ({"temp_anneal_start_pcts": [0.3, 0.3, 0.3]}, "must both be set"),
    ({"temp_anneal_end_pcts": [0.6, 0.6, 0.6]}, "must both be set"),
    ({"temp_anneal_start_pcts": [0.3, 0.3], "temp_anneal_end_pcts": [0.6, 0.6]}, "3 entries for player counts 3-5"),
    ({"temp_anneal_start_pcts": [0.3, 0.3, 0.3], "temp_anneal_end_pcts": [0.6, 0.6, 0.6, 0.6]}, "3 entries"),
    ({"temp_anneal_start_pcts": [0.3, 0.7, 0.3], "temp_anneal_end_pcts": [0.6, 0.6, 0.6]}, "4p has 0.7-0.6"),
    ({"temp_anneal_start_pcts": [-0.1, 0.3, 0.3], "temp_anneal_end_pcts": [0.6, 0.6, 0.6]}, "3p has -0.1-0.6"),
    ({"temp_anneal_start_pcts": [0.3, 0.3, 0.3], "temp_anneal_end_pcts": [0.6, 0.6, 60]}, "5p has 0.3-60"),
    ({"temp_anneal_start_pcts": [0.3, 0.3, "0.3"], "temp_anneal_end_pcts": [0.6, 0.6, 0.6]}, "must be numbers"),
    ({"temp_anneal_start_pcts": 0.3, "temp_anneal_end_pcts": 0.6}, "must both be set"),
])
def test_shares_are_validated(fields, message) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**MIXED, **fields)


def test_shares_round_trip_through_a_checkpoints_config() -> None:
    config = _mixed()
    stored = TrainingConfig.from_json(config.to_json())
    assert stored.temp_anneal_start_pcts == [0.3, 0.3, 0.25] and stored.temp_anneal_end_pcts == [0.6, 0.6, 0.75]
    # A resume's config file can turn them on.
    plain = TrainingConfig(**MIXED)
    plain.apply_json_overrides(json.dumps({
        "temp_anneal_start_pcts": [0.3, 0.3, 0.3], "temp_anneal_end_pcts": [0.6, 0.6, 0.6],
    }))
    plain.validate()
    assert plain.compute_epoch_config(1, {4: 100.0}).temp_anneal_windows == {4: (30, 60)}


def _record(num_players: int, moves: int, flags: int = 0) -> SimpleNamespace:
    return SimpleNamespace(num_players=num_players, total_moves=moves, flags=flags)


def test_lengths_are_of_the_games_played_from_the_opening() -> None:
    """A game started from a position measures only its remainder: it is left
    out, and a count with no other game has no length this epoch, so the run
    keeps the one it had."""
    lengths = _OpeningGameLengths()
    assert lengths.means() == {}
    for record in (
        _record(3, 400), _record(3, 500), _record(3, 90, FLAG_FROM_POSITION),
        _record(4, 120, FLAG_FROM_POSITION), _record(4, 80, FLAG_FROM_POSITION),
        _record(5, 701, FLAG_FROM_POSITION | 2), _record(5, 700, 2),  # another flag is not a start
    ):
        lengths.add_record(record)
    assert lengths.means() == {3: 450.0, 5: 700.0}

    held = {3: 300.0, 4: 610.0}
    held.update(lengths.means())
    assert held == {3: 450.0, 4: 610.0, 5: 700.0}
    assert _mixed().compute_epoch_config(2, held).temp_anneal_windows == {
        3: (135, 270), 4: (183, 366), 5: (175, 525),
    }
