"""Tests for Wrap Up phase."""

import pytest
from cython_core.state import GameState
from cython_core.phases.wrapup import WrapUpPhase, get_phase_handler, handle_wrap_up
from cython_core.data import py_get_company_face_value

from tests.test_common import StateBuilder, PHASE_WRAP_UP


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in WRAP_UP phase."""
    s = GameState(3)
    s.phase = PHASE_WRAP_UP
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get wrap up phase handler for 3 players."""
    return get_phase_handler(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# BASIC HANDLER TESTS
# =============================================================================

class TestWrapUpPhaseHandler:
    """Test WrapUpPhase handler basics."""

    def test_get_phase_handler_creates_handler(self):
        handler = get_phase_handler(3)
        assert isinstance(handler, WrapUpPhase)

    def test_get_phase_handler_caches(self):
        h1 = get_phase_handler(3)
        h2 = get_phase_handler(3)
        assert h1 is h2

    def test_wrong_phase_raises(self, state, handler):
        state.phase = 0  # INVEST, not WRAP_UP
        with pytest.raises(ValueError):
            handler.execute(state)


# =============================================================================
# PLAYER ORDER TESTS
# =============================================================================

class TestPlayerOrderSorting:
    """Test player order reordering by cash."""

    def test_order_by_cash_descending(self, state, handler, builder):
        # Set up different cash amounts
        builder.set_player_cash(0, 20)
        builder.set_player_cash(1, 50)
        builder.set_player_cash(2, 30)

        # Initial order: 0, 1, 2
        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        # New order should be: 1 (50), 2 (30), 0 (20)
        assert builder.get_player_turn_order(1) == 0  # Player 1 is first
        assert builder.get_player_turn_order(2) == 1  # Player 2 is second
        assert builder.get_player_turn_order(0) == 2  # Player 0 is third

    def test_ties_broken_by_old_order(self, state, handler, builder):
        # Same cash for all
        builder.set_player_cash(0, 30)
        builder.set_player_cash(1, 30)
        builder.set_player_cash(2, 30)

        # Initial order: 2, 0, 1
        builder.set_player_turn_order(0, 1)
        builder.set_player_turn_order(1, 2)
        builder.set_player_turn_order(2, 0)

        handler.execute(state)

        # Order should stay: 2, 0, 1 (ties broken by old order)
        assert builder.get_player_turn_order(2) == 0
        assert builder.get_player_turn_order(0) == 1
        assert builder.get_player_turn_order(1) == 2

    def test_partial_ties(self, state, handler, builder):
        # Two players tied at top
        builder.set_player_cash(0, 50)
        builder.set_player_cash(1, 50)
        builder.set_player_cash(2, 20)

        builder.set_player_turn_order(0, 1)  # Player 0 was second
        builder.set_player_turn_order(1, 0)  # Player 1 was first
        builder.set_player_turn_order(2, 2)  # Player 2 was third

        handler.execute(state)

        # Player 1 should be first (was ahead in old order)
        # Player 0 should be second (tied but was behind)
        # Player 2 should be third
        assert builder.get_player_turn_order(1) == 0
        assert builder.get_player_turn_order(0) == 1
        assert builder.get_player_turn_order(2) == 2


# =============================================================================
# FOREIGN INVESTOR BUYING TESTS
# =============================================================================

class TestForeignInvestorBuying:
    """Test FI buying companies at face value."""

    def test_fi_buys_cheapest_first(self, state, handler, builder):
        # FI has enough for only cheapest company
        # Company 0 has face value 1, Company 2 has face value 5
        builder.set_fi_cash(3)

        builder.set_company_for_auction(0, True)  # Face value 1
        builder.set_company_for_auction(2, True)  # Face value 5

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        # FI should buy company 0 (cheapest, can afford)
        # but not company 2 (too expensive after buying 0)
        assert builder.fi_owns_company(0)
        assert not builder.fi_owns_company(2)
        assert not builder.has_company_for_auction(0)
        assert builder.has_company_for_auction(2)
        assert builder.get_fi_cash() == 2  # 3 - 1

    def test_fi_buys_multiple_companies(self, state, handler, builder):
        # FI has enough for multiple
        builder.set_fi_cash(10)

        # Companies 0 (1), 1 (2), 2 (5) available
        builder.set_company_for_auction(0, True)
        builder.set_company_for_auction(1, True)
        builder.set_company_for_auction(2, True)

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        # FI should buy 0 (1) + 1 (2) + 2 (5) = 8 total
        assert builder.fi_owns_company(0)
        assert builder.fi_owns_company(1)
        assert builder.fi_owns_company(2)
        assert builder.get_fi_cash() == 2  # 10 - 1 - 2 - 5 = 2

    def test_fi_stops_when_too_poor(self, state, handler, builder):
        builder.set_fi_cash(4)

        # Company 2 (face=5) available - too expensive
        builder.set_company_for_auction(2, True)

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        assert not builder.fi_owns_company(2)
        assert builder.has_company_for_auction(2)
        assert builder.get_fi_cash() == 4  # Unchanged

    def test_fi_draws_new_cards(self, state, handler, builder):
        builder.set_fi_cash(5)
        builder.set_company_for_auction(0, True)  # Face value 1
        builder.setup_deck([10, 11])  # Company 10 on top

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        # FI bought company 0, drew company 10 to revealed
        assert builder.fi_owns_company(0)
        # Company 10 should now be available (revealed moved to auction)
        assert builder.has_company_for_auction(10)


# =============================================================================
# REVEALED TO AUCTION TESTS
# =============================================================================

class TestRevealedToAuction:
    """Test moving revealed companies to auction."""

    def test_revealed_becomes_available(self, state, handler, builder):
        # Some companies revealed (from previous turn's auctions)
        builder.set_company_revealed(5, True)
        builder.set_company_revealed(6, True)

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        # Should now be available for auction
        assert builder.has_company_for_auction(5)
        assert builder.has_company_for_auction(6)
        # And no longer revealed
        assert not builder.is_company_revealed(5)
        assert not builder.is_company_revealed(6)


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransition:
    """Test phase transition to Acquisition."""

    def test_transitions_to_acquisition(self, state, handler, builder):
        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        assert state.phase == 3  # ACQUISITION

    def test_active_player_reset(self, state, handler, builder):
        state.active_player = 2

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handler.execute(state)

        assert state.active_player == 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_handle_wrap_up(self, state, builder):
        builder.set_player_cash(0, 10)
        builder.set_player_cash(1, 20)
        builder.set_player_cash(2, 15)

        builder.set_player_turn_order(0, 0)
        builder.set_player_turn_order(1, 1)
        builder.set_player_turn_order(2, 2)

        handle_wrap_up(state)

        assert state.phase == 3  # ACQUISITION
        # Player 1 should be first (highest cash)
        assert builder.get_player_turn_order(1) == 0
