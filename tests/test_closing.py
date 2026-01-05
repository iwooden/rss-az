"""Tests for Closing phase."""

import pytest
from state import GameState
from phases.closing import ClosingPhase, get_action_constants
from data import (
    py_get_company_income, py_get_company_stars,
    py_get_cost_of_ownership, COMPANY_NAME_TO_ID
)

from tests.test_common import (
    StateBuilder, PHASE_CLOSING, PHASE_INCOME,
    CORP_JS, CORP_S, CORP_OS, CORP_SM, CORP_PR, CORP_DA, CORP_VM, CORP_SI
)

# Get action constants
_constants = get_action_constants()
CLOSING_ACTION_PASS = _constants['CLOSING_ACTION_PASS']
CLOSING_ACTION_MAX = _constants['CLOSING_ACTION_MAX']


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in CLOSING phase."""
    s = GameState(3)
    s.phase = PHASE_CLOSING
    s.coo_level = 1
    s.active_player = 0
    # Set up turn order
    s.set_player_turn_order_py(0, 0)
    s.set_player_turn_order_py(1, 1)
    s.set_player_turn_order_py(2, 2)
    return s


@pytest.fixture
def handler():
    """Get closing phase handler for 3 players."""
    return ClosingPhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# BASIC HANDLER TESTS
# =============================================================================

class TestClosingPhaseHandler:
    """Test ClosingPhase handler basics."""

    def test_handler_creates_correctly(self):
        handler = ClosingPhase(3)
        assert handler is not None

    def test_no_closeable_transitions_to_income(self, state, handler, builder):
        """If no closeable companies, transitions directly to Income."""
        handler.setup_closing(state)
        assert state.phase == PHASE_INCOME

    def test_pass_action_is_always_valid(self, state, handler, builder):
        """Pass action should always be valid when waiting for action."""
        # Give player a negative income company at high CoO
        state.coo_level = 5  # Red companies have CoO of 4 at level 5
        builder.set_player_owns_company(0, 0, True)  # BME (company 0, red, income=1)

        handler.setup_closing(state)

        # Pass should be valid
        assert handler.can_do_action(state, CLOSING_ACTION_PASS)


class TestPlayerClosing:
    """Test player closing private companies."""

    def test_player_can_close_negative_income_company(self, state, handler, builder):
        """Player can close their own negative-income company."""
        # Set CoO level 5 where red companies have CoO of 4
        state.coo_level = 5
        builder.set_player_owns_company(0, 0, True)  # BME (income=1, CoO=4 -> adjusted=-3)

        handler.setup_closing(state)

        # Should be waiting for action with an offer
        assert handler.is_waiting_for_action(state)
        # Should be able to close (action 0) or pass
        assert handler.can_do_action(state, 0)  # Close

    def test_player_cannot_close_positive_income_company(self, state, handler, builder):
        """Player cannot close a positive-income company."""
        # At CoO level 1, company 0 has income=1, CoO=0, adjusted=1
        state.coo_level = 1
        builder.set_player_owns_company(0, 0, True)

        handler.setup_closing(state)

        # No closeable companies - should transition directly to Income phase
        assert state.phase == PHASE_INCOME

    def test_closing_removes_company(self, state, handler, builder):
        """Closing a company removes it from game."""
        state.coo_level = 5
        builder.set_player_owns_company(0, 0, True)

        handler.setup_closing(state)
        handler.do_action(state, 0)  # Close company 0

        assert not builder.player_owns_company(0, 0)
        assert builder.is_company_removed(0)

    def test_player_can_close_multiple_companies(self, state, handler, builder):
        """Player can close multiple companies before passing."""
        state.coo_level = 5
        builder.set_player_owns_company(0, 0, True)  # BME
        builder.set_player_owns_company(0, 1, True)  # BSE

        handler.setup_closing(state)

        # Close first offered company (action 0 = close)
        handler.do_action(state, 0)
        assert builder.is_company_removed(0)

        # Next company is now offered - close it too (action 0)
        assert handler.is_waiting_for_action(state)
        assert handler.can_do_action(state, 0)  # Close action
        handler.do_action(state, 0)
        assert builder.is_company_removed(1)


class TestCorpClosing:
    """Test closing corp subsidiary companies."""

    def test_corp_can_close_with_two_plus_companies(self, state, handler, builder):
        """Corp with 2+ companies can close one."""
        state.coo_level = 5
        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 100)
        builder.set_player_president(0, CORP_PR, True)
        builder.set_corp_owns_company(CORP_PR, 0, True)  # BME
        builder.set_corp_owns_company(CORP_PR, 1, True)  # BSE

        handler.setup_closing(state)

        # Should have an offer for one of the corp's companies
        assert handler.is_waiting_for_action(state)
        assert handler.can_do_action(state, 0)  # Close is valid

    def test_corp_cannot_close_last_company(self, state, handler, builder):
        """Corp with only 1 company cannot close it."""
        state.coo_level = 5
        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 100)
        builder.set_player_president(0, CORP_PR, True)
        builder.set_corp_owns_company(CORP_PR, 0, True)  # Only one company

        handler.setup_closing(state)

        # Corp can't close its only company, so no offer
        # Should go straight to Income (or auto-close) phase
        assert state.phase == PHASE_INCOME

    def test_js_gets_bonus_when_closing(self, state, handler, builder):
        """Junkyard Scrappers gets 2x printed income when closing."""
        state.coo_level = 5
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (income=1)
        builder.set_corp_owns_company(CORP_JS, 1, True)  # BSE (income=1)

        handler.setup_closing(state)
        handler.do_action(state, 0)  # Close BME

        # JS should have 2 cash (2 * income of 1)
        assert builder.get_corp_cash(CORP_JS) == 2


class TestAutoClosing:
    """Test automatic closing at end of phase."""

    def test_fi_auto_closes_negative_income(self, state, handler, builder):
        """FI automatically closes negative-income companies."""
        state.coo_level = 5
        builder.set_fi_owns_company(0, True)  # BME (income=1, CoO=4 -> -3)

        # No players have closeable, so auto-close triggers immediately
        handler.setup_closing(state)

        # FI company should be removed
        assert not builder.fi_owns_company(0)
        assert builder.is_company_removed(0)

    def test_receivership_auto_closes_red_at_level_5(self, state, handler, builder):
        """Receivership corp closes red companies when CoO >= 4."""
        state.coo_level = 5
        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_in_receivership(CORP_PR, True)
        builder.set_corp_owns_company(CORP_PR, 0, True)  # BME (red)
        builder.set_corp_owns_company(CORP_PR, 6, True)  # WT (orange, company 6)

        handler.setup_closing(state)

        # Red company should be closed, orange kept
        assert not builder.corp_owns_company(CORP_PR, 0)
        assert builder.is_company_removed(0)
        assert builder.corp_owns_company(CORP_PR, 6)

    def test_receivership_keeps_highest_face_value(self, state, handler, builder):
        """Receivership keeps the highest face value company."""
        state.coo_level = 5
        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_in_receivership(CORP_PR, True)
        # All red companies with increasing face values
        builder.set_corp_owns_company(CORP_PR, 0, True)  # BME (face=1)
        builder.set_corp_owns_company(CORP_PR, 1, True)  # BSE (face=2)
        builder.set_corp_owns_company(CORP_PR, 2, True)  # KME (face=5)

        handler.setup_closing(state)

        # Should keep company 2 (highest face value), close others
        assert builder.corp_owns_company(CORP_PR, 2)
        assert builder.is_company_removed(0)
        assert builder.is_company_removed(1)

    def test_forced_player_closing_prevents_bankruptcy(self, state, handler, builder):
        """Players force-close companies to prevent bankruptcy."""
        state.coo_level = 5
        builder.set_player_cash(0, 0)  # Player has $0
        # Give player companies with negative adjusted income
        builder.set_player_owns_company(0, 0, True)  # BME (adjusted=-3)
        builder.set_player_owns_company(0, 1, True)  # BSE (adjusted=-3)

        # Player has no cash, will get -6 income, so would go to -6
        # System should force-close companies

        # With offer-based closing, player must pass on each offered company
        handler.setup_closing(state)
        handler.do_action(state, CLOSING_ACTION_PASS)  # Pass on company 0
        handler.do_action(state, CLOSING_ACTION_PASS)  # Pass on company 1
        # After all offers exhausted, auto_close_and_transition runs

        # Both companies should be force-closed to prevent bankruptcy
        assert builder.is_company_removed(0)
        assert builder.is_company_removed(1)


class TestPhaseTransition:
    """Test transition to Income phase."""

    def test_transitions_to_income_after_all_pass(self, state, handler, builder):
        """Transitions to Income when all players pass."""
        state.coo_level = 5
        builder.set_player_owns_company(0, 0, True)

        handler.setup_closing(state)
        handler.do_action(state, CLOSING_ACTION_PASS)

        assert state.phase == PHASE_INCOME
