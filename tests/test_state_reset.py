import numpy as np
import pytest

from core.state import GameState, get_layout, get_turn_fields
from entities import DECK, TURN, PLAYERS


def test_initialize_game_true_reset_same_seed():
    state = GameState(4)
    state.initialize_game(4, seed=42)
    first_order = DECK.get_order(state)
    first_remaining = DECK.get_remaining_count(state)

    PLAYERS[0].add_cash(state, 99)
    state.initialize_game(4, seed=42)

    assert PLAYERS[0].get_cash(state) == 30
    assert DECK.get_order(state) == first_order
    assert DECK.get_remaining_count(state) == first_remaining
    assert TURN.get_cards_remaining(state) == first_remaining


def test_initialize_game_can_change_player_count():
    state = GameState(2)
    state.initialize_game(4, seed=42)

    layout = get_layout(4)
    assert len(state._array) == layout.total_size
    assert TURN.get_num_players(state) == 4
    for player_id in range(4):
        assert PLAYERS[player_id].get_cash(state) == 30


def test_from_buffer_rejects_num_players_mismatch():
    layout = get_layout(4)
    turn_fields = get_turn_fields()
    buffer = np.zeros(layout.total_size, dtype=np.int16)
    buffer[layout.turn_offset + turn_fields.num_players] = 6

    with pytest.raises(AssertionError, match="canonical num_players"):
        GameState.from_buffer(buffer, 4)


def test_from_buffer_rejects_readonly_buffer():
    layout = get_layout(4)
    turn_fields = get_turn_fields()
    buffer = np.zeros(layout.total_size, dtype=np.int16)
    buffer[layout.turn_offset + turn_fields.num_players] = 4
    buffer.setflags(write=False)

    with pytest.raises(AssertionError, match="writeable"):
        GameState.from_buffer(buffer, 4)


def test_from_array_accepts_noncontiguous_view():
    state = GameState(4)
    state.initialize_game(4, seed=42)

    doubled = np.zeros(len(state._array) * 2, dtype=np.int16)
    doubled[::2] = state._array
    view = doubled[::2]

    cloned = GameState.from_array(view, 4)

    assert np.array_equal(cloned._array, state._array)
    assert TURN.get_num_players(cloned) == 4
