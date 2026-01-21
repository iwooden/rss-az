"""Tests for GameDriver action dispatch and validation."""

import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER, GameDriver
from core.actions import get_valid_action_mask, get_action_layout, decode_action_py
from core.data import GamePhases

# Status codes (match core/driver.pxd)
STATUS_OK = 0
STATUS_INVALID = 1
STATUS_GAME_OVER = 2


class TestGameDriverBasics:
    """Test GameDriver instantiation and basic interface."""

    def test_driver_singleton_exists(self):
        """DRIVER singleton should be available at module level."""
        assert DRIVER is not None
        assert isinstance(DRIVER, GameDriver)

    def test_driver_has_apply_action(self):
        """GameDriver should have apply_action method."""
        assert hasattr(DRIVER, 'apply_action')
        assert callable(DRIVER.apply_action)

    def test_driver_has_get_legal_moves(self):
        """GameDriver should have get_legal_moves method."""
        assert hasattr(DRIVER, 'get_legal_moves')
        assert callable(DRIVER.get_legal_moves)


class TestGetLegalMoves:
    """Test GameDriver.get_legal_moves() method."""

    @pytest.fixture
    def game_state(self):
        """Create initialized game state for testing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        return state

    def test_get_legal_moves_returns_numpy_array(self, game_state):
        """get_legal_moves should return numpy float32 array."""
        mask = DRIVER.get_legal_moves(game_state)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.float32

    def test_get_legal_moves_matches_action_mask(self, game_state):
        """get_legal_moves should return same result as get_valid_action_mask."""
        driver_mask = DRIVER.get_legal_moves(game_state)
        direct_mask = get_valid_action_mask(game_state)
        np.testing.assert_array_equal(driver_mask, direct_mask)

    def test_get_legal_moves_correct_size(self, game_state):
        """Mask size should match action layout total_size."""
        mask = DRIVER.get_legal_moves(game_state)
        layout = get_action_layout(3)  # 3 players
        assert len(mask) == layout['total_size']

    def test_get_legal_moves_has_valid_actions(self, game_state):
        """Initial state should have at least one valid action."""
        mask = DRIVER.get_legal_moves(game_state)
        assert np.sum(mask) > 0, "Should have at least one valid action"


class TestApplyActionValidation:
    """Test action validation in apply_action."""

    @pytest.fixture
    def game_state(self):
        """Create initialized game state for testing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        return state

    def test_invalid_action_index_negative(self, game_state):
        """Negative action index should return STATUS_INVALID."""
        result = DRIVER.apply_action(game_state, -1)
        assert result == STATUS_INVALID

    def test_invalid_action_index_too_large(self, game_state):
        """Action index beyond total_size should return STATUS_INVALID."""
        layout = get_action_layout(3)
        result = DRIVER.apply_action(game_state, layout['total_size'])
        assert result == STATUS_INVALID

    def test_invalid_action_not_in_mask(self, game_state):
        """Action with mask[idx]=0 should return STATUS_INVALID."""
        mask = DRIVER.get_legal_moves(game_state)
        # Find an invalid action (mask = 0)
        invalid_idx = None
        for i in range(len(mask)):
            if mask[i] == 0.0:
                invalid_idx = i
                break
        assert invalid_idx is not None, "Test requires at least one invalid action"
        result = DRIVER.apply_action(game_state, invalid_idx)
        assert result == STATUS_INVALID


class TestApplyActionInvestPhase:
    """Test apply_action dispatch for INVEST phase."""

    @pytest.fixture
    def invest_state(self):
        """Create game state in INVEST phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        # Game starts in INVEST phase
        assert state.get_phase() == GamePhases.PHASE_INVEST
        return state

    def test_pass_action_returns_ok(self, invest_state):
        """Pass action in INVEST phase should return STATUS_OK."""
        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']
        # Verify pass is valid
        mask = DRIVER.get_legal_moves(invest_state)
        assert mask[pass_idx] == 1.0, "Pass should be valid in INVEST"
        result = DRIVER.apply_action(invest_state, pass_idx)
        assert result == STATUS_OK

    def test_valid_auction_action_returns_ok(self, invest_state):
        """Valid auction action should return STATUS_OK."""
        mask = DRIVER.get_legal_moves(invest_state)
        layout = get_action_layout(3)
        # Find first valid auction action
        auction_idx = None
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                auction_idx = i
                break
        if auction_idx is not None:
            result = DRIVER.apply_action(invest_state, auction_idx)
            assert result == STATUS_OK


class TestApplyActionBidPhase:
    """Test apply_action dispatch for BID_IN_AUCTION phase."""

    @pytest.fixture
    def bid_state(self):
        """Create game state in BID_IN_AUCTION phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        # Manually set phase to BID_IN_AUCTION for testing
        # (In real game, this happens via start auction action)
        state.set_phase(GamePhases.PHASE_BID_IN_AUCTION)
        # Set up minimal auction state so mask has valid actions
        # Get first available company
        for company_id in range(36):
            if state.is_company_for_auction(company_id):
                state.set_auction_company(company_id)
                state.set_auction_price(1)  # Face value
                break
        return state

    def test_leave_auction_returns_ok(self, bid_state):
        """Leave auction action should return STATUS_OK."""
        layout = get_action_layout(3)
        leave_idx = layout['leave_auction']
        mask = DRIVER.get_legal_moves(bid_state)
        if mask[leave_idx] == 1.0:
            result = DRIVER.apply_action(bid_state, leave_idx)
            assert result == STATUS_OK

    def test_valid_raise_bid_returns_ok(self, bid_state):
        """Valid raise bid action should return STATUS_OK."""
        mask = DRIVER.get_legal_moves(bid_state)
        layout = get_action_layout(3)
        # Find first valid raise bid action
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                result = DRIVER.apply_action(bid_state, i)
                assert result == STATUS_OK
                break


class TestPhaseDispatch:
    """Test that actions dispatch to correct phase handlers."""

    def test_invest_action_dispatches_correctly(self):
        """INVEST phase actions should dispatch to invest handler."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        assert state.get_phase() == GamePhases.PHASE_INVEST

        # Pass action is always valid in INVEST
        layout = get_action_layout(3)
        result = DRIVER.apply_action(state, layout['pass_invest'])
        # Stub returns 0 (STATUS_OK) for valid action types
        assert result == STATUS_OK

    def test_bid_action_dispatches_correctly(self):
        """BID phase actions should dispatch to bid handler."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.set_phase(GamePhases.PHASE_BID_IN_AUCTION)

        # Set up auction state
        for company_id in range(36):
            if state.is_company_for_auction(company_id):
                state.set_auction_company(company_id)
                state.set_auction_price(1)
                break

        # Leave auction is always valid
        layout = get_action_layout(3)
        mask = DRIVER.get_legal_moves(state)
        if mask[layout['leave_auction']] == 1.0:
            result = DRIVER.apply_action(state, layout['leave_auction'])
            assert result == STATUS_OK


class TestMultiplePlayerCounts:
    """Test driver works correctly for different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_get_legal_moves_correct_size_per_player_count(self, num_players):
        """Mask size should be correct for each player count."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)
        mask = DRIVER.get_legal_moves(state)
        layout = get_action_layout(num_players)
        assert len(mask) == layout['total_size']

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_pass_action_works_for_all_player_counts(self, num_players):
        """Pass action should work for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)
        layout = get_action_layout(num_players)
        result = DRIVER.apply_action(state, layout['pass_invest'])
        assert result == STATUS_OK
