"""g1889.driver ahead of the phases (1889/engine-design.md §6): status codes,
terminal and automated states, and the game's end at an automated step.
tests/g1889/phases/test_stock_round.py covers the forced rule."""

import numpy as np
import pytest

import g1889.constants as K
import g1889.driver as Dr
from g1889.entities.bank import BANK
from g1889.entities.turn import TURN
from g1889.state import GameState


@pytest.fixture
def state():
    s = GameState(3)
    s.initialize_game(3, seed=5)
    return s


def test_status_codes():
    assert (Dr.STATUS_OK, Dr.STATUS_GAME_OVER, Dr.STATUS_INVALID) == (0, 1, 2)


def test_game_over_is_reported_without_changes(state):
    TURN.set_round(state, K.ROUND_GAME_OVER)
    TURN.set_step(state, K.STEP_GAME_OVER)
    row = state._array.copy()
    history = []
    assert Dr.DRIVER.apply_action(state, 0, history) == Dr.STATUS_GAME_OVER
    np.testing.assert_array_equal(state._array, row)
    assert history == []


def test_automated_steps_reject_actions(state):
    TURN.set_round(state, K.ROUND_STOCK)
    TURN.set_step(state, K.STEP_SR_END)
    assert Dr.DRIVER.apply_action(state, 0) == Dr.STATUS_INVALID


def test_the_game_ends_after_a_broken_banks_set(state):
    """OR_END after the set's last OR, with the bank broken, ends the game:
    the auto-chain reports it, and apply_action changes nothing after."""
    TURN.set_round(state, K.ROUND_OPERATING)
    TURN.set_step(state, K.STEP_OR_END)
    TURN.set_or_count(state, 1)
    BANK.set_broken(state)
    history = []
    assert Dr.DRIVER.advance(state, history) == Dr.STATUS_GAME_OVER
    assert [step for _, _, step in history] == [K.STEP_OR_END]
    assert (TURN.round(state), TURN.step(state), TURN.end_reason(state)) == \
        (K.ROUND_GAME_OVER, K.STEP_GAME_OVER, K.END_REASON_BANK)
    row = state._array.copy()
    assert Dr.DRIVER.apply_action(state, 0) == Dr.STATUS_GAME_OVER
    np.testing.assert_array_equal(state._array, row)
