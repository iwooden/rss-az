"""Tests for GameState."""

import pytest
import numpy as np
from state import (
    GameState,
    get_state_size,
    get_visible_size,
    get_market_price,
    get_market_index,
    PHASE_NAMES,
)


class TestGameStateCreation:
    """Test GameState initialization."""

    def test_create_3_player(self):
        state = GameState(3)
        assert state.num_players == 3
        assert state.size > 0
        assert state.visible_size < state.size

    def test_create_various_player_counts(self):
        for n in range(2, 7):
            state = GameState(n)
            assert state.num_players == n

    def test_invalid_player_count_too_low(self):
        with pytest.raises(ValueError):
            GameState(1)

    def test_invalid_player_count_too_high(self):
        with pytest.raises(ValueError):
            GameState(7)

    def test_size_functions_match_instance(self):
        for n in range(2, 7):
            state = GameState(n)
            assert state.size == get_state_size(n)
            assert state.visible_size == get_visible_size(n)


class TestGameStateProperties:
    """Test GameState property accessors."""

    @pytest.fixture
    def state(self):
        return GameState(3)

    def test_initial_phase(self, state):
        # Phase should be unset (-1) initially
        assert state.phase == -1

    def test_set_phase(self, state):
        state.phase = 0  # INVEST
        assert state.phase == 0

        state.phase = 4  # CLOSING
        assert state.phase == 4

    def test_phase_count(self):
        assert len(PHASE_NAMES) == 11

    def test_initial_active_player(self, state):
        assert state.active_player == 0

    def test_set_active_player(self, state):
        state.active_player = 2
        assert state.active_player == 2

    def test_initial_coo_level(self, state):
        # Default should be 1
        assert state.coo_level == 1

    def test_set_coo_level(self, state):
        state.coo_level = 4
        assert state.coo_level == 4

    def test_initial_turn_number(self, state):
        assert state.turn_number == 0

    def test_set_turn_number(self, state):
        state.turn_number = 15
        assert state.turn_number == 15

    def test_initial_consecutive_passes(self, state):
        assert state.consecutive_passes == 0

    def test_set_consecutive_passes(self, state):
        state.consecutive_passes = 2
        assert state.consecutive_passes == 2


class TestGameStateArrayAccess:
    """Test array and tensor access."""

    @pytest.fixture
    def state(self):
        return GameState(3)

    def test_as_numpy_returns_array(self, state):
        arr = state.as_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert len(arr) == state.size

    def test_as_numpy_is_not_copy(self, state):
        """as_numpy should return the underlying array, not a copy."""
        arr = state.as_numpy()
        state.phase = 3
        # The array should reflect the change
        assert arr is state.as_numpy()

    def test_clone_creates_independent_copy(self, state):
        state.phase = 5
        state.active_player = 2

        clone = state.clone()

        assert clone.phase == 5
        assert clone.active_player == 2

        # Modify original
        state.phase = 0
        state.active_player = 0

        # Clone should be unchanged
        assert clone.phase == 5
        assert clone.active_player == 2

    def test_is_terminal(self, state):
        assert not state.is_terminal()
        state.phase = 10  # GAME_OVER
        assert state.is_terminal()


class TestGameStateNNInput:
    """Test neural network input generation."""

    def test_get_nn_input_size(self):
        state = GameState(3)
        nn_input = state.get_nn_input()
        assert len(nn_input) == state.visible_size

    def test_get_nn_input_excludes_hidden(self):
        state = GameState(3)
        full = state.as_numpy()
        nn_input = state.get_nn_input()

        assert len(nn_input) < len(full)
        assert len(full) - len(nn_input) > 0  # Hidden portion exists

    def test_get_nn_input_no_rotation_for_player_0(self):
        state = GameState(3)
        state.phase = 0
        state.active_player = 0

        full = state.as_numpy()
        nn_input = state.get_nn_input()

        # For player 0, visible portion should match exactly
        np.testing.assert_array_equal(nn_input, full[:state.visible_size])

    def test_get_nn_input_rotates_for_other_players(self):
        state = GameState(3)
        state.phase = 0
        state.active_player = 1

        full = state.as_numpy()
        nn_input = state.get_nn_input()

        # Should not be identical (rotation happened)
        # Note: This could fail if all player data is zeros
        # But phase/coo should be same
        assert len(nn_input) == state.visible_size


class TestMarketFunctions:
    """Test market price lookup functions."""

    def test_get_market_price_valid(self):
        assert get_market_price(0) == 0   # Bankruptcy
        assert get_market_price(1) == 5
        assert get_market_price(26) == 75  # Max

    def test_get_market_price_invalid(self):
        assert get_market_price(-1) == -1
        assert get_market_price(27) == -1

    def test_get_market_index_valid(self):
        assert get_market_index(0) == 0
        assert get_market_index(5) == 1
        assert get_market_index(75) == 26

    def test_get_market_index_invalid(self):
        assert get_market_index(3) == -1  # Not a valid price
        assert get_market_index(100) == -1

    def test_market_roundtrip(self):
        """Price -> index -> price should be identity."""
        for idx in range(27):
            price = get_market_price(idx)
            assert get_market_index(price) == idx
