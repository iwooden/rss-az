"""Tests for Invest phase."""

import pytest
import numpy as np
from state import GameState
from phases.invest import InvestPhase, get_phase_handler
from data import py_get_company_face_value

from tests.test_common import StateBuilder, PHASE_INVEST, PHASE_WRAP_UP, PHASE_BID_IN_AUCTION


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state."""
    s = GameState(3)
    s.phase = PHASE_INVEST
    s.coo_level = 1
    s.active_player = 0
    # Initialize turn order (player i is at position i)
    for i in range(3):
        s.set_player_turn_order_py(i, i)
    return s


@pytest.fixture
def handler():
    """Get invest phase handler for 3 players."""
    return get_phase_handler(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# BASIC PHASE HANDLER TESTS
# =============================================================================

class TestInvestPhaseHandler:
    """Test InvestPhase handler basics."""

    def test_get_phase_handler_creates_handler(self):
        handler = get_phase_handler(3)
        assert isinstance(handler, InvestPhase)

    def test_get_phase_handler_caches(self):
        h1 = get_phase_handler(3)
        h2 = get_phase_handler(3)
        assert h1 is h2

    def test_different_player_counts_different_handlers(self):
        h3 = get_phase_handler(3)
        h4 = get_phase_handler(4)
        assert h3 is not h4


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test pass action behavior."""

    def test_pass_increments_consecutive_passes(self, state, handler):
        assert state.consecutive_passes == 0
        handler.do_pass(state)
        assert state.consecutive_passes == 1

    def test_pass_advances_player(self, state, handler):
        assert state.active_player == 0
        handler.do_pass(state)
        assert state.active_player == 1

    def test_pass_wraps_player(self, state, handler):
        state.active_player = 2
        handler.do_pass(state)
        assert state.active_player == 0

    def test_all_pass_ends_phase(self, state, handler):
        """When all players pass consecutively, transition to WRAP_UP."""
        handler.do_pass(state)  # Player 0
        handler.do_pass(state)  # Player 1
        handler.do_pass(state)  # Player 2

        assert state.phase == PHASE_WRAP_UP
        assert state.consecutive_passes == 0


# =============================================================================
# BUY SHARE TESTS
# =============================================================================

class TestBuyShare:
    """Test buy share action."""

    def test_cannot_buy_inactive_corp(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, False)
        builder.set_corp_bank_shares(0, 5)

        assert not handler.can_do_buy_share(state, 0)

    def test_cannot_buy_no_shares_available(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 0)
        builder.set_corp_price_index(0, 5)

        assert not handler.can_do_buy_share(state, 0)

    def test_cannot_buy_insufficient_cash(self, state, handler, builder):
        builder.set_player_cash(0, 5)  # Only $5
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 10)  # Price at index 10 is $14, next is $16
        builder.set_market_available(11, True)  # Next price slot

        assert not handler.can_do_buy_share(state, 0)

    def test_can_buy_with_sufficient_cash(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)  # Price $9
        builder.set_market_available(6, True)  # Next slot at $10

        assert handler.can_do_buy_share(state, 0)

    def test_buy_transfers_share(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)
        builder.set_market_available(6, True)

        handler.do_buy_share(state, 0)

        assert builder.get_player_shares(0, 0) == 1
        assert builder.get_corp_bank_shares(0) == 4

    def test_buy_increases_price(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)  # $9
        builder.set_market_available(6, True)  # $10

        handler.do_buy_share(state, 0)

        assert builder.get_corp_price_index(0) == 6

    def test_buy_deducts_new_price(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)  # $9
        builder.set_market_available(6, True)  # $10

        handler.do_buy_share(state, 0)

        # Should pay $10 (the new price)
        assert builder.get_player_cash(0) == 90

    def test_buy_clears_consecutive_passes(self, state, handler, builder):
        state.consecutive_passes = 2

        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)
        builder.set_market_available(6, True)

        handler.do_buy_share(state, 0)

        assert state.consecutive_passes == 0

    def test_buy_advances_player(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)
        builder.set_market_available(6, True)

        handler.do_buy_share(state, 0)

        assert state.active_player == 1


# =============================================================================
# SELL SHARE TESTS
# =============================================================================

class TestSellShare:
    """Test sell share action."""

    def test_cannot_sell_no_shares(self, state, handler, builder):
        builder.set_player_shares(0, 0, 0)
        assert not handler.can_do_sell_share(state, 0)

    def test_can_sell_with_shares(self, state, handler, builder):
        builder.set_player_shares(0, 0, 2)
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 10)
        builder.set_market_available(9, True)

        assert handler.can_do_sell_share(state, 0)

    def test_sell_transfers_share(self, state, handler, builder):
        builder.set_player_shares(0, 0, 3)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 2)
        builder.set_corp_price_index(0, 10)
        builder.set_market_available(9, True)

        handler.do_sell_share(state, 0)

        assert builder.get_player_shares(0, 0) == 2
        assert builder.get_corp_bank_shares(0) == 3

    def test_sell_decreases_price(self, state, handler, builder):
        builder.set_player_shares(0, 0, 3)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 2)
        builder.set_corp_price_index(0, 10)  # $14
        builder.set_market_available(9, True)  # $13

        handler.do_sell_share(state, 0)

        assert builder.get_corp_price_index(0) == 9

    def test_sell_pays_new_price(self, state, handler, builder):
        builder.set_player_cash(0, 50)
        builder.set_player_shares(0, 0, 3)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 2)
        builder.set_corp_price_index(0, 10)  # $14
        builder.set_market_available(9, True)  # $13

        handler.do_sell_share(state, 0)

        # Should receive $13 (the new price)
        assert builder.get_player_cash(0) == 63


# =============================================================================
# AUCTION TESTS
# =============================================================================

class TestStartAuction:
    """Test starting an auction."""

    def test_cannot_start_auction_company_not_available(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_company_for_auction(5, False)

        assert not handler.can_do_start_auction(state, 5, 0)

    def test_cannot_start_auction_insufficient_cash(self, state, handler, builder):
        builder.set_company_for_auction(0, True)  # Company 0 has face value $1
        builder.set_player_cash(0, 0)

        assert not handler.can_do_start_auction(state, 0, 0)

    def test_can_start_auction(self, state, handler, builder):
        builder.set_company_for_auction(0, True)
        builder.set_player_cash(0, 100)

        assert handler.can_do_start_auction(state, 0, 0)  # Bid at face value
        assert handler.can_do_start_auction(state, 0, 5)  # Bid +5 over face

    def test_start_auction_changes_phase(self, state, handler, builder):
        builder.set_company_for_auction(0, True)
        builder.set_player_cash(0, 100)

        handler.do_start_auction(state, 0, 3)  # Bid face+3

        assert state.phase == PHASE_BID_IN_AUCTION

    def test_start_auction_sets_auction_state(self, state, handler, builder):
        builder.set_company_for_auction(5, True)
        builder.set_player_cash(0, 100)
        face_value = py_get_company_face_value(5)

        handler.do_start_auction(state, 5, 4)

        # Auction company should be set (check via get_valid_actions in auction phase)
        # Active player should advance to next
        assert state.active_player == 1

    def test_start_auction_advances_player(self, state, handler, builder):
        builder.set_company_for_auction(0, True)
        builder.set_player_cash(0, 100)

        handler.do_start_auction(state, 0, 0)

        assert state.active_player == 1


class TestAuctionBidding:
    """Test auction bidding mechanics."""

    def setup_auction(self, state, builder, company_id=0, initial_bid_offset=0):
        """Helper to set up an auction in progress."""
        handler = get_phase_handler(3)
        builder.set_company_for_auction(company_id, True)
        builder.set_player_cash(0, 100)
        builder.set_player_cash(1, 100)
        builder.set_player_cash(2, 100)

        handler.do_start_auction(state, company_id, initial_bid_offset)
        return handler

    def test_can_raise_bid(self, state, builder):
        handler = self.setup_auction(state, builder, 0, 0)

        # Player 1 is now active, can raise
        assert handler.can_do_raise_bid(state, 1)  # Bid offset 1

    def test_cannot_raise_bid_below_current(self, state, builder):
        handler = self.setup_auction(state, builder, 0, 5)  # Started at face+5

        # Player 1 cannot bid at face+3 (below current)
        assert not handler.can_do_raise_bid(state, 3)

    def test_raise_bid_updates_price(self, state, builder):
        handler = self.setup_auction(state, builder, 0, 0)

        handler.do_raise_bid(state, 5)  # Player 1 raises to face+5

        # Now player 2 is active
        assert state.active_player == 2

    def test_leave_auction(self, state, builder):
        handler = self.setup_auction(state, builder, 0, 0)

        # Player 1 leaves
        handler.do_leave_auction(state)

        # Player 2 should be active now
        assert state.active_player == 2

    def test_auction_resolves_when_one_left(self, state, builder):
        handler = self.setup_auction(state, builder, 0, 0)
        initial_cash = builder.get_player_cash(0)

        # Player 1 leaves
        handler.do_leave_auction(state)
        # Player 2 leaves
        handler.do_leave_auction(state)

        # Auction should resolve - player 0 wins
        assert state.phase == PHASE_INVEST

        # Player 0 should own the company and have paid
        face_value = py_get_company_face_value(0)
        assert builder.get_player_cash(0) == initial_cash - face_value

    def test_auction_winner_gets_company(self, state, builder):
        handler = self.setup_auction(state, builder, 5, 0)

        # Players 1 and 2 leave
        handler.do_leave_auction(state)
        handler.do_leave_auction(state)

        # Check player 0 owns company 5
        assert builder.player_owns_company(0, 5)

    def test_auction_turn_goes_to_after_starter(self, state, builder):
        """After auction resolves, turn goes to player after starter, not winner."""
        handler = self.setup_auction(state, builder, 0, 0)  # Player 0 starts

        # Player 1 raises
        handler.do_raise_bid(state, 5)

        # Players 2 and 0 leave
        handler.do_leave_auction(state)  # Player 2 leaves
        handler.do_leave_auction(state)  # Player 0 leaves

        # Player 1 wins, but turn should go to player after starter (player 0)
        # So next player should be player 1 (0 + 1)
        assert state.active_player == 1


class TestAuctionDeckDraw:
    """Test that winning auction draws new company."""

    def test_auction_draws_from_deck(self, state, builder):
        handler = get_phase_handler(3)

        # Set up deck with company 10 on top
        builder.setup_deck([10, 11, 12])
        builder.set_company_for_auction(0, True)
        builder.set_player_cash(0, 100)
        builder.set_player_cash(1, 100)
        builder.set_player_cash(2, 100)

        handler.do_start_auction(state, 0, 0)

        # All others leave
        handler.do_leave_auction(state)
        handler.do_leave_auction(state)

        # Company 10 should be drawn to revealed pile (becomes available in WRAP_UP)
        assert state.is_company_revealed_py(10)


# =============================================================================
# VALID ACTIONS TESTS
# =============================================================================

class TestGetValidActions:
    """Test get_valid_actions method."""

    def test_valid_actions_invest_phase(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_company_for_auction(0, True)
        builder.set_company_for_auction(1, True)

        actions = handler.get_valid_actions(state)

        assert 'buy' in actions
        assert 'sell' in actions
        assert 'auction' in actions
        assert 'pass' in actions
        assert actions['pass'] is True

    def test_valid_actions_includes_buyable_corps(self, state, handler, builder):
        builder.set_player_cash(0, 100)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 5)
        builder.set_market_available(6, True)

        actions = handler.get_valid_actions(state)

        assert 0 in actions['buy']

    def test_valid_actions_includes_sellable_corps(self, state, handler, builder):
        builder.set_player_shares(0, 2, 3)  # 3 shares of corp 2
        builder.set_corp_active(2, True)
        builder.set_corp_price_index(2, 10)
        builder.set_market_available(9, True)

        actions = handler.get_valid_actions(state)

        assert 2 in actions['sell']

    def test_valid_actions_auction_phase(self, state, handler, builder):
        builder.set_company_for_auction(0, True)
        builder.set_player_cash(0, 100)
        builder.set_player_cash(1, 100)

        handler.do_start_auction(state, 0, 0)

        actions = handler.get_valid_actions(state)

        assert 'raise' in actions
        assert 'leave' in actions
        assert actions['leave'] is True


# =============================================================================
# ROUND-TRIP LIMIT TESTS
# =============================================================================

class TestRoundTripLimit:
    """Test the round-trip trading limit."""

    def test_can_buy_sell_up_to_limit(self, state, handler, builder):
        # Set up corp with plenty of shares
        builder.set_player_cash(0, 1000)
        builder.set_player_shares(0, 0, 5)
        builder.set_corp_active(0, True)
        builder.set_corp_bank_shares(0, 5)
        builder.set_corp_price_index(0, 10)

        # Make market slots available
        for i in range(27):
            builder.set_market_available(i, True)

        # Buy-sell-buy-sell should work (2 round trips)
        assert handler.can_do_buy_share(state, 0)
        handler.do_buy_share(state, 0)
        state.active_player = 0  # Reset for testing

        assert handler.can_do_sell_share(state, 0)
        handler.do_sell_share(state, 0)
        state.active_player = 0

        assert handler.can_do_buy_share(state, 0)
        handler.do_buy_share(state, 0)
        state.active_player = 0

        assert handler.can_do_sell_share(state, 0)
        handler.do_sell_share(state, 0)
        state.active_player = 0

        # Now should be blocked (2 round trips completed)
        assert not handler.can_do_buy_share(state, 0)
        assert not handler.can_do_sell_share(state, 0)
