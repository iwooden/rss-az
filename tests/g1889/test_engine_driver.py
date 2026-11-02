"""g1889.driver ahead of the phases (1889/engine-design.md §6): status codes,
terminal and automated states, and loud failures at unimplemented phases."""

import numpy as np
import pytest

import g1889.constants as K
import g1889.driver as Dr
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


def test_unimplemented_phases_fail_loudly(state):
    with pytest.raises(NotImplementedError, match='AUCTION'):
        Dr.DRIVER.apply_action(state, K.AUCTION_PASS)
