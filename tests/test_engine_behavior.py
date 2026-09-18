"""Engine behavior mode survives state wrappers and actual turn transitions."""

import numpy as np
import pytest

from core.state import GameState
from entities.deck import DECK
from entities.player import PLAYERS
from entities.turn import TURN
from core.driver import DRIVER
from core.data import GamePhases
from utils_18xx.live import _clone_live_state


@pytest.mark.parametrize('enabled', [False, True])
def test_engine_mode_on_state_reconstruction_rebind_and_determinization(enabled):
    state = GameState(3, max_players=5, v3_behavior=enabled)
    state.initialize_game(3, seed=42, max_players=5)
    assert state.v3_behavior is enabled
    for construct in (GameState.from_array, GameState.from_buffer):
        restored = construct(state._array.copy(), 3, max_players=5, v3_behavior=enabled)
        assert restored.v3_behavior is enabled
        restored.rebind(state._array.copy(), 3, max_players=5)
        assert restored.v3_behavior is enabled
    clone = DECK.determinize_remaining(state, np.random.default_rng(1))
    assert clone.v3_behavior is enabled
    assert _clone_live_state(state, 3, 5).v3_behavior is enabled


@pytest.mark.parametrize('enabled', [False, True])
def test_invest_exit_resets_all_players_all_corps_only_in_legacy_mode(enabled):
    state = GameState(3, v3_behavior=enabled)
    state.initialize_game(3, seed=42)
    state.step_mode = True
    for p in range(3):
        for c in range(8):
            PLAYERS[p].increment_share_buys(state, c)
            PLAYERS[p].increment_share_sells(state, c)
    for _ in range(3):
        DRIVER.apply_action(state, 0)
    assert TURN.get_phase(state) == GamePhases.PHASE_WRAP_UP
    for p in range(3):
        for c in range(8):
            assert PLAYERS[p].get_share_buys(state, c) == int(enabled)
            assert PLAYERS[p].get_share_sells(state, c) == int(enabled)
