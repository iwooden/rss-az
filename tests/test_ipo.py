"""Tests for IPO phase."""

import pytest
from cython_core.state import GameState
from cython_core.phases.ipo import IPOPhase, get_constants
from cython_core.data import (
    py_get_company_face_value, py_get_company_stars,
    py_get_market_price, py_get_par_price, py_is_valid_par_price,
    py_get_corp_share_count
)

from tests.test_common import (
    StateBuilder, PHASE_IPO, PHASE_INVEST,
    NUM_CORPS, NUM_COMPANIES, NUM_PAR_PRICES
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in IPO phase."""
    s = GameState(3)
    s.phase = PHASE_IPO
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get IPO phase handler for 3 players."""
    return IPOPhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_market_index_for_par_price(par_index: int) -> int:
    """Get market index for a par price index."""
    par_price = py_get_par_price(par_index)
    # Map par price to market index
    market_prices = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
                     18, 20, 22, 24, 27, 30, 33, 37, 41, 45,
                     50, 55, 61, 68, 75]
    return market_prices.index(par_price)


# =============================================================================
# BASIC FLOW TESTS
# =============================================================================

class TestIPOPhaseFlow:
    """Test basic IPO phase flow."""

    def test_processes_companies_in_descending_face_value(self, state, handler, builder):
        """Companies are processed in descending face value order."""
        # Player 0 owns company 10 (OL, face value 15) and company 5 (MHE, face value 8)
        builder.set_player_owns_company(0, 10, True)
        builder.set_player_owns_company(0, 5, True)
        builder.set_turn_ipo_remaining(10, 1.0)
        builder.set_turn_ipo_remaining(5, 1.0)
        builder.set_player_cash(0, 100)

        handler.advance_to_next_company(state)

        # Company 10 (higher face value) should be selected first
        assert handler.get_current_company(state) == 10

    def test_skips_companies_player_cannot_afford(self, state, handler, builder):
        """Companies are skipped if owner cannot afford any IPO."""
        # Company 35 (CDG) has face value 60, requires at least par 30
        # Par 30 at index 11, player needs (1*30 - 60) = -30 (can afford with 0 cash)
        # Actually, if par < face_value, player needs (2*30 - 60) = 0
        # So player with 0 cash should be able to afford company 35 at par 30

        # Use a scenario where player truly can't afford:
        # Company 29 (HH, face value 45, star 5)
        # Valid pars for star 5: 30, 33, 37
        # At par 30: player pays (2*30 - 45) = 15
        # At par 33: player pays (2*33 - 45) = 21
        # At par 37: player pays (2*37 - 45) = 29
        # Player with 10 cash cannot afford any

        builder.set_player_owns_company(0, 29, True)  # HH, face value 45
        builder.set_turn_ipo_remaining(29, 1.0)
        builder.set_player_cash(0, 10)

        # Also need at least one inactive corp
        # All corps start inactive by default

        handler.advance_to_next_company(state)

        # Company should be skipped since player can't afford, transitions to INVEST
        assert state.phase == PHASE_INVEST

    def test_transitions_to_invest_when_no_inactive_corps(self, state, handler, builder):
        """Transitions to INVEST if no inactive corps available."""
        # Make all corps active
        for i in range(NUM_CORPS):
            builder.set_corp_active(i, True)
            builder.set_corp_price_index(i, i + 1)

        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 100)

        handler.advance_to_next_company(state)

        assert state.phase == PHASE_INVEST

    def test_transitions_to_invest_when_no_companies_left(self, state, handler, builder):
        """Transitions to INVEST when no companies in ipo_remaining."""
        # No companies marked for IPO
        for i in range(NUM_COMPANIES):
            builder.set_turn_ipo_remaining(i, 0.0)

        handler.advance_to_next_company(state)

        assert state.phase == PHASE_INVEST


# =============================================================================
# IPO ACTION VALIDATION
# =============================================================================

class TestIPOValidation:
    """Test IPO action validation."""

    def test_can_ipo_with_valid_corp_and_par(self, state, handler, builder):
        """Can IPO with valid corp and par price."""
        # Company 0 (BME, face value 1, star 1)
        # Valid pars for star 1: indices 0-4 (prices 10, 11, 12, 13, 14)
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)

        # Par index 0 = price 10, cost = (1*10 - 1) = 9
        assert handler.can_ipo(state, 0, 0)

    def test_cannot_ipo_with_active_corp(self, state, handler, builder):
        """Cannot IPO with already active corp."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)

        handler.advance_to_next_company(state)

        assert not handler.can_ipo(state, 0, 0)

    def test_cannot_ipo_with_invalid_par_for_star_tier(self, state, handler, builder):
        """Cannot IPO with par price invalid for company's star tier."""
        # Company 0 (BME, star 1) - valid pars are indices 0-4
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 50)

        handler.advance_to_next_company(state)

        # Par index 10 = price 27, not valid for star 1
        assert not handler.can_ipo(state, 0, 10)

    def test_cannot_ipo_if_market_space_taken(self, state, handler, builder):
        """Cannot IPO if the market space is already taken."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        # Mark market space for par index 0 (price 10 = market index 6) as taken
        builder.set_market_available(6, False)

        handler.advance_to_next_company(state)

        assert not handler.can_ipo(state, 0, 0)

    def test_cannot_ipo_if_insufficient_cash(self, state, handler, builder):
        """Cannot IPO if player doesn't have enough cash."""
        # Company 0 (BME, face value 1)
        # Par index 0 = price 10, cost = (1*10 - 1) = 9
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 5)  # Not enough

        handler.advance_to_next_company(state)

        assert not handler.can_ipo(state, 0, 0)


# =============================================================================
# IPO ACTION EXECUTION
# =============================================================================

class TestIPOExecution:
    """Test IPO action execution."""

    def test_ipo_activates_corp(self, state, handler, builder):
        """IPO activates the corporation."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.is_corp_active(0)

    def test_ipo_sets_corp_price(self, state, handler, builder):
        """IPO sets corporation's share price."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        # Par index 2 = price 12 = market index 8
        handler.do_ipo(state, 0, 2)

        assert builder.get_corp_price_index(0) == 8

    def test_ipo_marks_market_space_taken(self, state, handler, builder):
        """IPO marks the market space as taken."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        # Par index 0 = price 10 = market index 6
        handler.do_ipo(state, 0, 0)

        assert not builder.is_market_available(6)

    def test_ipo_transfers_company_to_corp(self, state, handler, builder):
        """IPO transfers company from player to corp."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert not builder.player_owns_company(0, 0)
        assert builder.corp_owns_company(0, 0)

    def test_ipo_gives_player_shares(self, state, handler, builder):
        """IPO gives player their shares."""
        # Company 0 (BME, face value 1)
        # Par index 0 = price 10, face_value=1 < 10, so player gets 1 share
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.get_player_shares(0, 0) == 1

    def test_ipo_gives_2_shares_when_par_less_than_face_value(self, state, handler, builder):
        """IPO gives player 2 shares when par < face_value."""
        # Company 6 (WT, face value 11, star 2)
        # Valid pars for star 2: indices 0-7 (prices 10, 11, 12, 13, 14, 16, 18, 20)
        # Par index 0 = price 10 < 11, so player gets 2 shares
        builder.set_player_owns_company(0, 6, True)
        builder.set_turn_ipo_remaining(6, 1.0)
        # Cost = (2*10 - 11) = 9
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.get_player_shares(0, 0) == 2

    def test_ipo_sets_player_as_president(self, state, handler, builder):
        """IPO sets player as corporation president."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.is_player_president(0, 0)

    def test_ipo_deducts_player_cash(self, state, handler, builder):
        """IPO deducts correct amount from player."""
        # Company 0 (BME, face value 1)
        # Par index 0 = price 10, cost = (1*10 - 1) = 9
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.get_player_cash(0) == 11

    def test_ipo_gives_corp_correct_cash(self, state, handler, builder):
        """IPO gives corporation correct cash."""
        # Company 0 (BME, face value 1)
        # Par index 0 = price 10
        # 1 share each to player and bank
        # Player pays: (1*10 - 1) = 9
        # Bank pays: 1*10 = 10
        # Corp gets: 9 + 10 = 19
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.get_corp_cash(0) == 19

    def test_ipo_sets_issued_shares(self, state, handler, builder):
        """IPO sets correct issued shares count."""
        # Par >= face_value: 2 shares issued
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert state.get_corp_issued_shares_py(0) == 2

    def test_ipo_sets_4_shares_when_par_less_than_face(self, state, handler, builder):
        """IPO issues 4 shares when par < face_value."""
        # Company 6 (WT, face value 11)
        # Par index 0 = price 10 < 11
        builder.set_player_owns_company(0, 6, True)
        builder.set_turn_ipo_remaining(6, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert state.get_corp_issued_shares_py(0) == 4

    def test_ipo_sets_bank_shares(self, state, handler, builder):
        """IPO sets correct bank shares count."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert state.get_corp_bank_shares_py(0) == 1

    def test_ipo_sets_unissued_shares(self, state, handler, builder):
        """IPO sets correct unissued shares count."""
        # Corp 0 (JS) has 7 total shares
        # After IPO with par >= face_value: 2 issued, 5 unissued
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert state.get_corp_unissued_shares_py(0) == 5


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test pass action."""

    def test_can_always_pass(self, state, handler, builder):
        """Player can always pass on IPO."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 100)

        handler.advance_to_next_company(state)

        assert handler.can_pass(state)

    def test_pass_advances_to_next_company(self, state, handler, builder):
        """Passing advances to next company."""
        # Player 0 owns companies 10 (face 15) and 5 (face 8)
        builder.set_player_owns_company(0, 10, True)
        builder.set_player_owns_company(0, 5, True)
        builder.set_turn_ipo_remaining(10, 1.0)
        builder.set_turn_ipo_remaining(5, 1.0)
        builder.set_player_cash(0, 100)

        handler.advance_to_next_company(state)
        assert handler.get_current_company(state) == 10

        handler.do_pass(state)

        assert handler.get_current_company(state) == 5

    def test_pass_removes_from_ipo_remaining(self, state, handler, builder):
        """Passing removes company from ipo_remaining."""
        # Set up two companies so passing one doesn't end the phase
        builder.set_player_owns_company(0, 5, True)  # Company 5 (higher face value)
        builder.set_player_owns_company(0, 0, True)  # Company 0 (lower face value)
        builder.set_turn_ipo_remaining(5, 1.0)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 100)

        handler.advance_to_next_company(state)
        # Should be on company 5 (higher face value)
        assert handler.get_current_company(state) == 5

        handler.do_pass(state)

        # Company 5 should be removed from ipo_remaining
        assert builder.get_turn_ipo_remaining(5) == 0.0
        # Company 0 should still be in ipo_remaining
        assert builder.get_turn_ipo_remaining(0) == 1.0


# =============================================================================
# NET WORTH TESTS
# =============================================================================

class TestNetWorthUpdate:
    """Test net worth updates after IPO."""

    def test_net_worth_updated_after_ipo(self, state, handler, builder):
        """Player net worth is updated after IPO."""
        # Company 0 (BME, face value 1)
        # Par index 0 = price 10
        # Player starts with 20 cash, pays 9, ends with 11 cash
        # Player gets 1 share worth 10
        # Net worth = 11 + 10 = 21
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        handler.do_ipo(state, 0, 0)

        assert builder.get_player_net_worth(0) == 21

    def test_net_worth_includes_remaining_companies(self, state, handler, builder):
        """Net worth includes face value of remaining companies."""
        # Player owns company 0 (face 1) and company 1 (face 2)
        # IPO company 1, keep company 0
        builder.set_player_owns_company(0, 0, True)
        builder.set_player_owns_company(0, 1, True)
        builder.set_turn_ipo_remaining(1, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        # Par index 0 = price 10, cost = (1*10 - 2) = 8
        handler.do_ipo(state, 0, 0)

        # Cash: 20 - 8 = 12
        # Company 0: face value 1
        # 1 share of corp 0 at price 10
        # Net worth = 12 + 1 + 10 = 23
        assert builder.get_player_net_worth(0) == 23


# =============================================================================
# VALID OPTIONS TESTS
# =============================================================================

class TestValidOptions:
    """Test get_valid_ipo_options method."""

    def test_returns_valid_corp_par_pairs(self, state, handler, builder):
        """Returns list of valid (corp_id, par_index) pairs."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)

        handler.advance_to_next_company(state)
        options = handler.get_valid_ipo_options(state)

        # Should have multiple options (8 corps * valid pars for star 1)
        assert len(options) > 0

        # Each option should be a tuple of (corp_id, par_index)
        for corp_id, par_index in options:
            assert 0 <= corp_id < NUM_CORPS
            assert 0 <= par_index < NUM_PAR_PRICES

    def test_excludes_active_corps(self, state, handler, builder):
        """Excludes active corporations from options."""
        builder.set_player_owns_company(0, 0, True)
        builder.set_turn_ipo_remaining(0, 1.0)
        builder.set_player_cash(0, 20)
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)

        handler.advance_to_next_company(state)
        options = handler.get_valid_ipo_options(state)

        # Corp 0 should not appear in options
        for corp_id, _ in options:
            assert corp_id != 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_handles_no_current_company(self, state, handler, builder):
        """Handles case where there's no current company."""
        # Don't set up any companies
        assert handler.get_current_company(state) == -1
        assert not handler.can_pass(state)
        assert handler.get_valid_ipo_options(state) == []

    def test_multiple_players_companies(self, state, handler, builder):
        """Handles companies owned by different players."""
        # Player 0 owns company 5 (face 8)
        # Player 1 owns company 10 (face 15)
        builder.set_player_owns_company(0, 5, True)
        builder.set_player_owns_company(1, 10, True)
        builder.set_turn_ipo_remaining(5, 1.0)
        builder.set_turn_ipo_remaining(10, 1.0)
        builder.set_player_cash(0, 100)
        builder.set_player_cash(1, 100)

        handler.advance_to_next_company(state)

        # Company 10 (higher face value) should be first
        assert handler.get_current_company(state) == 10
        # Active player should be player 1 (owner of company 10)
        assert state.active_player == 1


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test that constants are accessible."""

    def test_get_constants(self):
        """Can access phase constants."""
        constants = get_constants()
        assert constants['NUM_PAR_PRICES'] == NUM_PAR_PRICES
        assert constants['NUM_CORPS'] == NUM_CORPS
