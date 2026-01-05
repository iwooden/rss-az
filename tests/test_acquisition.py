"""Tests for Acquisition phase."""

import pytest
from state import GameState
from phases.acquisition import (
    AcquisitionPhase, get_phase_handler, handle_acquisition_phase,
    get_action_constants
)
from data import py_get_company_face_value, py_get_company_low_price, py_get_company_high_price

from tests.test_common import (
    StateBuilder, PHASE_ACQUISITION, PHASE_CLOSING,
    CORP_JS, CORP_S, CORP_OS, CORP_SM, CORP_PR, CORP_DA, CORP_VM, CORP_SI
)

# Get action constants
_constants = get_action_constants()
ACQ_ACTION_PASS = _constants['ACQ_ACTION_PASS']
ACQ_FI_ACTION_BUY = _constants['ACQ_FI_ACTION_BUY']
ACQ_FI_ACTION_PASS = _constants['ACQ_FI_ACTION_PASS']


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in ACQUISITION phase."""
    s = GameState(3)
    s.phase = PHASE_ACQUISITION
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get acquisition phase handler for 3 players."""
    return get_phase_handler(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# BASIC HANDLER TESTS
# =============================================================================

class TestAcquisitionPhaseHandler:
    """Test AcquisitionPhase handler basics."""

    def test_get_phase_handler_creates_handler(self):
        handler = get_phase_handler(3)
        assert isinstance(handler, AcquisitionPhase)

    def test_get_phase_handler_caches(self):
        h1 = get_phase_handler(3)
        h2 = get_phase_handler(3)
        assert h1 is h2

    def test_no_offers_when_no_corps(self, state, handler, builder):
        """No offers if no corps are active."""
        result = handler.setup_next_offer(state)
        assert result is False

    def test_no_offers_when_no_targets(self, state, handler, builder):
        """No offers if corps have no valid targets."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 100)
        builder.set_corp_price_index(CORP_JS, 15)  # Some price
        builder.set_player_president(0, CORP_JS, True)

        result = handler.setup_next_offer(state)
        assert result is False


# =============================================================================
# FI OFFER TESTS
# =============================================================================

class TestFIOffers:
    """Test FI company offers."""

    def test_fi_offer_to_os_first(self, state, handler, builder):
        """OS gets first offer on FI companies."""
        # Set up OS and another corp
        builder.set_corp_active(CORP_OS, True)
        builder.set_corp_cash(CORP_OS, 10)  # Enough for company 0 (face=1)
        builder.set_corp_price_index(CORP_OS, 10)
        builder.set_player_president(0, CORP_OS, True)

        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 15)  # Higher price
        builder.set_player_president(1, CORP_JS, True)

        # FI owns company 0
        builder.set_fi_owns_company(0, True)

        result = handler.setup_next_offer(state)
        assert result is True
        assert state.is_acq_fi_offer_py()

        # Should be offered to OS (highest priority)
        assert state.get_acq_active_corp_py() == CORP_OS

    def test_fi_offer_by_share_price_order(self, state, handler, builder):
        """After OS, FI offers go by share price descending."""
        # Set up two corps (not OS), different share prices
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)  # Price 14
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 10)
        builder.set_corp_price_index(CORP_PR, 15)  # Price 24 (higher)
        builder.set_player_president(1, CORP_PR, True)

        # FI owns company 0 (face=1, high=2)
        builder.set_fi_owns_company(0, True)

        result = handler.setup_next_offer(state)
        assert result is True

        # Should be offered to PR (higher share price)
        assert state.get_acq_active_corp_py() == CORP_PR

    def test_fi_offer_skips_corp_without_cash(self, state, handler, builder):
        """Corps without enough cash are skipped."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)  # No cash
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 10)
        builder.set_corp_price_index(CORP_PR, 10)  # Lower price but has cash
        builder.set_player_president(1, CORP_PR, True)

        builder.set_fi_owns_company(0, True)

        result = handler.setup_next_offer(state)
        assert result is True

        # Should be offered to PR (JS has no cash)
        assert state.get_acq_active_corp_py() == CORP_PR

    def test_os_pays_face_value(self, state, handler, builder):
        """OS pays face value for FI companies."""
        builder.set_corp_active(CORP_OS, True)
        builder.set_corp_cash(CORP_OS, 5)  # Enough for face value, not high price
        builder.set_corp_price_index(CORP_OS, 10)
        builder.set_player_president(0, CORP_OS, True)

        # Company 2 (KME): face=5, high=7
        builder.set_fi_owns_company(2, True)
        builder.set_fi_cash(0)

        handler.setup_next_offer(state)
        handler.do_action(state, ACQ_FI_ACTION_BUY)

        # OS should have paid 5 (face value)
        assert builder.get_corp_cash(CORP_OS) == 0
        assert builder.get_fi_cash() == 5
        assert builder.corp_has_acquisition_company(CORP_OS, 2)
        assert not builder.fi_owns_company(2)

    def test_non_os_pays_high_price(self, state, handler, builder):
        """Non-OS corps pay high price for FI companies."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)

        # Company 2 (KME): face=5, high=7
        builder.set_fi_owns_company(2, True)
        builder.set_fi_cash(0)

        handler.setup_next_offer(state)
        handler.do_action(state, ACQ_FI_ACTION_BUY)

        # JS should have paid 7 (high price)
        assert builder.get_corp_cash(CORP_JS) == 3
        assert builder.get_fi_cash() == 7

    def test_receivership_auto_buys(self, state, handler, builder):
        """Receivership corps auto-buy from FI."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_corp_in_receivership(CORP_JS, True)
        # No president for receivership corp

        builder.set_fi_owns_company(0, True)  # face=1, high=2
        builder.set_fi_cash(0)

        # setup_next_offer should auto-execute the purchase
        result = handler.setup_next_offer(state)

        # Should have already bought (no offer pending)
        assert builder.get_corp_cash(CORP_JS) == 8  # 10 - 2 (high price)
        assert builder.get_fi_cash() == 2
        assert builder.corp_has_acquisition_company(CORP_JS, 0)

    def test_fi_companies_offered_by_face_value_desc(self, state, handler, builder):
        """FI companies are offered by face value descending."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 100)
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        # FI owns companies 0 (face=1) and 5 (face=8)
        builder.set_fi_owns_company(0, True)
        builder.set_fi_owns_company(5, True)

        handler.setup_next_offer(state)

        # Should offer company 5 first (higher face value)
        assert state.get_acq_target_company_py() == 5


class TestFIOffersBehavior:
    """Test FI offers through observable behavior."""

    def test_fi_buy_action_transfers_company(self, state, handler, builder):
        """Buying from FI transfers company to acquisition pile."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_fi_owns_company(0, True)
        builder.set_fi_cash(0)

        handler.setup_next_offer(state)
        handler.do_action(state, ACQ_FI_ACTION_BUY)

        assert not builder.fi_owns_company(0)
        assert builder.corp_has_acquisition_company(CORP_JS, 0)

    def test_fi_pass_action_leaves_company(self, state, handler, builder):
        """Passing on FI offer leaves company with FI."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_fi_owns_company(0, True)

        handler.setup_next_offer(state)
        initial_cash = builder.get_corp_cash(CORP_JS)
        handler.do_action(state, ACQ_FI_ACTION_PASS)

        assert builder.fi_owns_company(0)
        assert builder.get_corp_cash(CORP_JS) == initial_cash


# =============================================================================
# GENERAL ACQUISITION TESTS
# =============================================================================

class TestGeneralAcquisitions:
    """Test general (non-FI) acquisitions."""

    def test_corp_can_buy_presidents_company(self, state, handler, builder):
        """Corp can buy company from its president."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)

        # Player 0 owns company 0 (low=1, high=2)
        builder.set_player_owns_company(0, 0, True)

        result = handler.setup_next_offer(state)
        assert result is True
        assert not state.is_acq_fi_offer_py()

    def test_price_offset_determines_price(self, state, handler, builder):
        """Price offset from low_price determines actual price."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_owns_company(0, 0, True)  # Company 0: low=1, high=2
        builder.set_player_cash(0, 0)

        handler.setup_next_offer(state)

        # Action 0 = low_price (1), Action 1 = low_price + 1 (2)
        handler.do_action(state, 1)  # Pay 2

        assert builder.get_corp_cash(CORP_JS) == 8  # 10 - 2
        assert builder.get_player_cash(0) == 2

    def test_pass_leaves_company(self, state, handler, builder):
        """Passing leaves company with owner."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_owns_company(0, 0, True)

        handler.setup_next_offer(state)
        handler.do_action(state, ACQ_ACTION_PASS)

        assert builder.player_owns_company(0, 0)

    def test_corp_can_buy_from_sibling_corp(self, state, handler, builder):
        """Corp can buy from another corp with same president."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_price_index(CORP_JS, 15)  # Higher price
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 10)
        builder.set_corp_price_index(CORP_PR, 10)
        builder.set_player_president(0, CORP_PR, True)  # Same president

        # PR owns companies 0 and 1 (needs 2+ to sell)
        builder.set_corp_owns_company(CORP_PR, 0, True)
        builder.set_corp_owns_company(CORP_PR, 1, True)

        result = handler.setup_next_offer(state)
        assert result is True

    def test_cannot_buy_from_corp_with_one_company(self, state, handler, builder):
        """Cannot buy from corp that only has one company."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 10)
        builder.set_corp_price_index(CORP_PR, 10)
        builder.set_player_president(0, CORP_PR, True)

        # PR only owns one company
        builder.set_corp_owns_company(CORP_PR, 0, True)

        result = handler.setup_next_offer(state)
        # Should be no offers (can't take last company)
        assert result is False

    def test_cannot_buy_from_different_president(self, state, handler, builder):
        """Cannot buy from corp with different president."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 10)
        builder.set_corp_price_index(CORP_PR, 10)
        builder.set_player_president(1, CORP_PR, True)  # Different president!

        builder.set_corp_owns_company(CORP_PR, 0, True)
        builder.set_corp_owns_company(CORP_PR, 1, True)

        result = handler.setup_next_offer(state)
        # Should be no offers
        assert result is False

    def test_corp_to_corp_proceeds_are_pending(self, state, handler, builder):
        """Selling corp gets acquisition_proceeds, not cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 0)
        builder.set_corp_price_index(CORP_PR, 10)
        builder.set_player_president(0, CORP_PR, True)

        # Company 0: low=1, high=2
        builder.set_corp_owns_company(CORP_PR, 0, True)
        builder.set_corp_owns_company(CORP_PR, 1, True)

        handler.setup_next_offer(state)
        handler.do_action(state, 0)  # Buy at low_price (1)

        # PR should have 1 in acquisition_proceeds, not cash
        assert builder.get_corp_cash(CORP_PR) == 0
        assert builder.get_corp_acquisition_proceeds(CORP_PR) == 1


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransition:
    """Test phase transition and finalization."""

    def test_finalize_moves_acquisition_companies(self, state, handler, builder):
        """Finalization moves acquisition companies to owned."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_fi_owns_company(0, True)
        builder.set_fi_cash(0)

        # Buy the company
        handler.setup_next_offer(state)
        handler.do_action(state, ACQ_FI_ACTION_BUY)

        # Should be in acquisition pile, not owned
        assert builder.corp_has_acquisition_company(CORP_JS, 0)
        assert not builder.corp_owns_company(CORP_JS, 0)

        # Finalize
        handler.transition_to_closing(state)

        # Now should be owned
        assert builder.corp_owns_company(CORP_JS, 0)
        assert not builder.corp_has_acquisition_company(CORP_JS, 0)

    def test_finalize_moves_proceeds_to_cash(self, state, handler, builder):
        """Finalization moves acquisition proceeds to cash."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 50)
        builder.set_corp_price_index(CORP_JS, 15)
        builder.set_player_president(0, CORP_JS, True)

        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 0)
        builder.set_corp_price_index(CORP_PR, 10)
        builder.set_player_president(0, CORP_PR, True)

        builder.set_corp_owns_company(CORP_PR, 0, True)
        builder.set_corp_owns_company(CORP_PR, 1, True)

        handler.setup_next_offer(state)
        handler.do_action(state, 1)  # Buy at low_price + 1 = 2

        assert builder.get_corp_acquisition_proceeds(CORP_PR) == 2

        handler.transition_to_closing(state)

        assert builder.get_corp_cash(CORP_PR) == 2
        assert builder.get_corp_acquisition_proceeds(CORP_PR) == 0

    def test_transitions_to_closing(self, state, handler, builder):
        """Phase transitions to CLOSING."""
        handler.transition_to_closing(state)
        assert state.phase == PHASE_CLOSING


# =============================================================================
# VALID ACTIONS TESTS
# =============================================================================

class TestValidActions:
    """Test get_valid_actions returns correct actions."""

    def test_fi_offer_has_buy_and_pass(self, state, handler, builder):
        """FI offer has buy and pass actions."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_fi_owns_company(0, True)

        handler.setup_next_offer(state)
        actions = handler.get_valid_actions(state)

        assert ACQ_FI_ACTION_BUY in actions
        assert ACQ_FI_ACTION_PASS in actions
        assert len(actions) == 2

    def test_general_has_valid_prices_and_pass(self, state, handler, builder):
        """General offer has valid price offsets and pass."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 5)  # Enough for low and high
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        # Company 0: low=1, high=2
        builder.set_player_owns_company(0, 0, True)

        handler.setup_next_offer(state)
        actions = handler.get_valid_actions(state)

        # Should have actions 0 (price=1), 1 (price=2), and 50 (pass)
        assert 0 in actions  # low_price
        assert 1 in actions  # high_price
        assert ACQ_ACTION_PASS in actions
        assert 2 not in actions  # Would be > high_price

    def test_limited_cash_limits_price_actions(self, state, handler, builder):
        """Corp with limited cash has fewer price options."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 1)  # Only enough for low_price
        builder.set_corp_price_index(CORP_JS, 10)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_player_owns_company(0, 0, True)  # Company 0: low=1, high=2

        handler.setup_next_offer(state)
        actions = handler.get_valid_actions(state)

        assert 0 in actions  # low_price (1)
        assert 1 not in actions  # Can't afford 2
        assert ACQ_ACTION_PASS in actions
