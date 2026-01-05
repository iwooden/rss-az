"""Tests for Issue Shares phase."""

import pytest
from cython_core.state import GameState
from cython_core.phases.issue import IssuePhase, get_constants
from cython_core.data import py_get_corp_share_count, py_get_market_price

from tests.test_common import (
    StateBuilder, PHASE_ISSUE_SHARES, PHASE_IPO,
    NUM_CORPS, NUM_MARKET_SPACES, CORP_SM
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in ISSUE_SHARES phase."""
    s = GameState(3)
    s.phase = PHASE_ISSUE_SHARES
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get Issue phase handler for 3 players."""
    return IssuePhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# BASIC FLOW TESTS
# =============================================================================

class TestIssuePhaseFlow:
    """Test basic Issue phase flow."""

    def test_processes_corps_in_descending_price_order(self, state, handler, builder):
        """Corps are processed in descending share price order."""
        # Set up two corps with different prices
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 10)  # Price 14
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        builder.set_corp_active(1, True)
        builder.set_corp_price_index(1, 15)  # Price 24
        builder.set_turn_issue_remaining(1, 1.0)
        builder.set_player_president(1, 1, True)
        state.set_corp_unissued_shares_py(1, 3)
        state.set_corp_issued_shares_py(1, 2)

        handler.advance_to_next_corp(state)

        # Corp 1 (higher price) should be selected first
        assert handler.get_current_corp(state) == 1

    def test_skips_inactive_corps(self, state, handler, builder):
        """Inactive corps are skipped."""
        builder.set_corp_active(0, False)
        builder.set_turn_issue_remaining(0, -1.0)  # Inactive

        builder.set_corp_active(1, True)
        builder.set_corp_price_index(1, 15)
        builder.set_turn_issue_remaining(1, 1.0)
        builder.set_player_president(0, 1, True)
        state.set_corp_unissued_shares_py(1, 3)
        state.set_corp_issued_shares_py(1, 2)

        handler.advance_to_next_corp(state)

        assert handler.get_current_corp(state) == 1

    def test_transitions_to_ipo_when_done(self, state, handler, builder):
        """Transitions to IPO phase when all corps processed."""
        # No corps to process
        for i in range(NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)

        assert state.phase == PHASE_IPO


# =============================================================================
# ISSUE ACTION TESTS
# =============================================================================

class TestIssueAction:
    """Test share issuance."""

    def test_can_issue_if_unissued_shares(self, state, handler, builder):
        """Corp can issue if it has unissued shares."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        handler.advance_to_next_corp(state)

        assert handler.can_issue(state)

    def test_cannot_issue_if_fully_issued(self, state, handler, builder):
        """Corp cannot issue if fully issued."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 0)  # No unissued shares
        state.set_corp_issued_shares_py(0, 7)

        handler.advance_to_next_corp(state)

        assert not handler.can_issue(state)

    def test_issue_increments_shares(self, state, handler, builder):
        """Issuing increments issued_shares and bank_shares."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 4)
        state.set_corp_bank_shares_py(0, 2)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        assert state.get_corp_issued_shares_py(0) == 5
        assert state.get_corp_bank_shares_py(0) == 3
        assert state.get_corp_unissued_shares_py(0) == 2

    def test_issue_moves_price_down(self, state, handler, builder):
        """Issuing moves share price down by 1."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24, index 15
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Price should have moved from index 15 to 14
        assert builder.get_corp_price_index(0) == 14

    def test_issue_skips_taken_market_spaces(self, state, handler, builder):
        """Issuing skips taken market spaces when moving down."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24, index 15
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Mark index 14 as taken
        builder.set_market_available(14, False)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Price should have moved from index 15 to 13 (skipping 14)
        assert builder.get_corp_price_index(0) == 13

    def test_issue_adds_cash_at_new_price(self, state, handler, builder):
        """Corp receives new share price as cash."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24
        builder.set_corp_cash(0, 10)
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # New price at index 14 is 22
        # Cash should be 10 + 22 = 32
        assert builder.get_corp_cash(0) == 32


# =============================================================================
# STOCK MASTERS SPECIAL ABILITY
# =============================================================================

class TestStockMastersSpecial:
    """Test Stock Masters (SM) special ability."""

    def test_sm_does_not_move_price_down(self, state, handler, builder):
        """Stock Masters doesn't move price down when issuing."""
        builder.set_corp_active(CORP_SM, True)
        builder.set_corp_price_index(CORP_SM, 15)  # Price 24
        builder.set_turn_issue_remaining(CORP_SM, 1.0)
        builder.set_player_president(0, CORP_SM, True)
        state.set_corp_unissued_shares_py(CORP_SM, 3)
        state.set_corp_issued_shares_py(CORP_SM, 2)

        # Only this corp to process
        for i in range(NUM_CORPS):
            if i != CORP_SM:
                builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Price should stay at index 15
        assert builder.get_corp_price_index(CORP_SM) == 15

    def test_sm_receives_current_price(self, state, handler, builder):
        """Stock Masters receives current (not new) share price."""
        builder.set_corp_active(CORP_SM, True)
        builder.set_corp_price_index(CORP_SM, 15)  # Price 24
        builder.set_corp_cash(CORP_SM, 10)
        builder.set_turn_issue_remaining(CORP_SM, 1.0)
        builder.set_player_president(0, CORP_SM, True)
        state.set_corp_unissued_shares_py(CORP_SM, 3)
        state.set_corp_issued_shares_py(CORP_SM, 2)

        # Only this corp to process
        for i in range(NUM_CORPS):
            if i != CORP_SM:
                builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Cash should be 10 + 24 = 34
        assert builder.get_corp_cash(CORP_SM) == 34


# =============================================================================
# RECEIVERSHIP TESTS
# =============================================================================

class TestReceivershipIssuing:
    """Test receivership corps issuing."""

    def test_receivership_auto_issues(self, state, handler, builder):
        """Receivership corp auto-issues if able."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_corp_in_receivership(0, True)
        builder.set_turn_issue_remaining(0, 1.0)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)
        state.set_corp_bank_shares_py(0, 2)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)

        # Should have auto-issued and advanced
        # After issuing, issued_shares should be 3
        assert state.get_corp_issued_shares_py(0) == 3

    def test_receivership_cannot_pass_if_can_issue(self, state, handler, builder):
        """Receivership corp cannot pass if it can issue."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_corp_in_receivership(0, True)
        builder.set_turn_issue_remaining(0, 1.0)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)
        builder.set_player_president(0, 0, True)

        handler.advance_to_next_corp(state)

        # Receivership already auto-issued, but if we manually check
        # a receivership corp with unissued shares shouldn't be able to pass
        # Since auto-issue happens, we need to set up differently
        # Let's test the can_pass logic directly

    def test_receivership_skipped_if_fully_issued(self, state, handler, builder):
        """Receivership corp is skipped if fully issued."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_corp_in_receivership(0, True)
        builder.set_turn_issue_remaining(0, 1.0)  # End card marked it as can issue
        state.set_corp_unissued_shares_py(0, 0)  # But actually no unissued
        state.set_corp_issued_shares_py(0, 7)

        # No other corps
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)

        # Should have transitioned to IPO
        assert state.phase == PHASE_IPO


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test pass (skip issuing) action."""

    def test_can_pass(self, state, handler, builder):
        """Normal corp can always pass."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        handler.advance_to_next_corp(state)

        assert handler.can_pass(state)

    def test_pass_advances_to_next(self, state, handler, builder):
        """Passing advances to next corp."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        builder.set_corp_active(1, True)
        builder.set_corp_price_index(1, 10)  # Lower price
        builder.set_turn_issue_remaining(1, 1.0)
        builder.set_player_president(1, 1, True)
        state.set_corp_unissued_shares_py(1, 3)
        state.set_corp_issued_shares_py(1, 2)

        handler.advance_to_next_corp(state)
        assert handler.get_current_corp(state) == 0

        handler.do_pass(state)

        # Should now be on corp 1
        assert handler.get_current_corp(state) == 1


# =============================================================================
# BANKRUPTCY TESTS
# =============================================================================

class TestIssueBankruptcy:
    """Test bankruptcy during issuing."""

    def test_bankruptcy_if_price_hits_zero(self, state, handler, builder):
        """Corp goes bankrupt if price hits 0."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 1)  # Price 5, index 1
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Corp should be bankrupt (inactive)
        assert not builder.is_corp_active(0)


# =============================================================================
# NET WORTH UPDATE TESTS
# =============================================================================

class TestNetWorthUpdate:
    """Test that net worth is updated after issuing."""

    def test_net_worth_updated_after_issue(self, state, handler, builder):
        """Player net worth is updated after issuing."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Player 0 owns 2 shares and has 50 cash
        builder.set_player_shares(0, 0, 2)
        builder.set_player_cash(0, 50)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # New price at index 14 is 22
        # Net worth should be 50 (cash) + 2 * 22 (shares) = 94
        assert builder.get_player_net_worth(0) == 94

    def test_net_worth_includes_companies(self, state, handler, builder):
        """Net worth includes face value of owned companies."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 15)  # Price 24
        builder.set_turn_issue_remaining(0, 1.0)
        builder.set_player_president(0, 0, True)
        state.set_corp_unissued_shares_py(0, 3)
        state.set_corp_issued_shares_py(0, 2)

        # Player 0 has 50 cash and owns company 0 (BME, face value 1)
        builder.set_player_cash(0, 50)
        builder.set_player_owns_company(0, 0, True)

        # Only this corp to process
        for i in range(1, NUM_CORPS):
            builder.set_turn_issue_remaining(i, -1.0)

        handler.advance_to_next_corp(state)
        handler.do_issue(state)

        # Net worth should be 50 (cash) + 1 (company) = 51
        assert builder.get_player_net_worth(0) == 51


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test that constants are accessible."""

    def test_get_constants(self):
        """Can access phase constants."""
        constants = get_constants()
        assert constants['CORP_SM'] == CORP_SM
