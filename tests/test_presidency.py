"""Consolidated tests for presidency recalculation.

Presidency is automatically derived from share ownership via
_recalculate_presidency() in entities/player.pyx, called by set_shares().

Tests organized by trigger mechanism:
1. Direct share changes (unit tests of recalculation logic)
2. Buy/sell in INVEST phase (presidency changes via trading)
3. IPO (initial president assignment)
4. Bankruptcy (president flags cleared)
5. Receivership (entry/exit, no-president state)
"""
import pytest
from core.state import GameState
from core.data import GamePhases, GameConstants
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.turn import TURN
from entities.market import MARKET
from phases.ipo import setup_ipo_phase_py, apply_ipo_action_py
from tests.phases.conftest import (
    STATUS_OK, float_corp_for_test, assert_invariants,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def game_state():
    """Base initialized game state in INVEST phase."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    return state


@pytest.fixture
def trade_state():
    """State with active corp for buy/sell testing."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 (JS) with 2 shares each to player and bank
    # This sets up: unissued(3), bank(2), issued(4), player 0 has 2 shares, price index 10
    float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)

    # Set up player cash for trading
    PLAYERS[0].set_cash(state, 100)

    return state


@pytest.fixture
def ipo_state_with_company(game_state):
    """3-player game with one player-owned company ready for IPO."""
    state = game_state

    # Initialize and transfer company 14 (FV=20, stars=3) to player 0
    company = COMPANIES[14]
    company.initialize(state)
    company.transfer_to_player(state, 0)

    # Player 0 has plenty of cash
    PLAYERS[0].set_cash(state, 100)

    # Initialize market (all spaces available)
    MARKET.initialize(state)

    # Initialize corps (all inactive)
    for corp_id in range(int(GameConstants.NUM_CORPS)):
        CORPS[corp_id].initialize(state)

    # Set up IPO phase
    TURN.set_phase(state, GamePhases.PHASE_IPO)
    setup_ipo_phase_py(state)

    return state


@pytest.fixture
def bankruptcy_state():
    """State where one sell triggers bankruptcy (price index 1 -> 0)."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 and then move to bankruptcy-prone position
    float_corp_for_test(state, corp_id=0, par_index=1, float_shares=2)
    # float_shares=2 gives: player=2, bank=2, issued=4

    PLAYERS[0].set_cash(state, 100)

    return state


# =============================================================================
# PRESIDENCY RECALCULATION TESTS
# =============================================================================

class TestPresidencyRecalculation:
    """Test presidency transfer via share changes (INV-18, INV-19, INV-20)."""

    def test_presidency_transfers_to_most_shares(self, trade_state):
        """INV-18: Player with most shares becomes president."""
        # Player 0 has 2 shares, is president
        # Give player 1 more shares
        PLAYERS[1].set_shares(trade_state, 0, 3)

        # Sell a share (triggers presidency check)
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # Player 1 should now be president (3 shares > 1 share)
        assert PLAYERS[1].is_president_of(trade_state, 0)
        assert not PLAYERS[0].is_president_of(trade_state, 0)

    def test_presidency_incumbent_keeps_on_tie(self, trade_state):
        """INV-19: Incumbent keeps presidency when shares are equal."""
        # Player 0 has 2 shares, is president
        # Give player 1 same shares
        PLAYERS[1].set_shares(trade_state, 0, 2)

        # Sell a share (triggers presidency check)
        # After sell: P0 has 1, P1 has 2 -> P1 wins
        # But if we set P1 to 1 share, then after sell both have 1
        PLAYERS[1].set_shares(trade_state, 0, 1)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # P0 now has 1 share, P1 has 1 share -> tie, incumbent (P0) keeps it
        assert PLAYERS[0].is_president_of(trade_state, 0)

    def test_presidency_maintained_after_buy(self, trade_state):
        """Presidency checks happen after buy transactions."""
        corp = CORPS[0]

        # Player 0 is president with 2 shares
        # Buy increases shares to 3, should maintain presidency
        # Adjust unissued to allow more bank shares (7 total = 3 unissued + 4 bank + 0 player)
        corp.set_unissued_shares(trade_state, 3)
        corp.set_bank_shares(trade_state, 4)
        corp.set_issued_shares(trade_state, 0)  # No shares issued yet
        PLAYERS[0].set_shares(trade_state, 0, 0)
        # After this setup, we need to actually give player 0 shares
        corp.set_issued_shares(trade_state, 2)
        corp.set_bank_shares(trade_state, 2)
        PLAYERS[0].set_shares(trade_state, 0, 2)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Player 0 should still be president (now with 3 shares)
        assert PLAYERS[0].is_president_of(trade_state, 0)
        assert PLAYERS[0].get_shares(trade_state, 0) == 3

    def test_presidency_transfer_on_buy(self, trade_state):
        """INV-18: Buying shares can trigger presidency transfer."""
        corp = CORPS[0]

        # Player 0 starts as president with 2 shares (from trade_state fixture)
        # Give player 1 more shares - this triggers automatic presidency recalculation
        # P1 with 3 shares becomes president (3 > 2)
        PLAYERS[1].set_shares(trade_state, 0, 3)
        # Update issued shares to match total
        corp.set_issued_shares(trade_state, 5)  # bank(2) + P0(2) + P1(3) - but only 7 total shares
        # So we need: unissued(0) + bank(2) + P0(2) + P1(3) = 7
        corp.set_unissued_shares(trade_state, 0)

        # At this point P1 is president (automatic recalculation when set_shares was called)
        assert PLAYERS[1].is_president_of(trade_state, 0), "P1 should be president after getting 3 shares"

        # Player 0 buys, now has 3 shares - tie with player 1
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # After buy: P0 has 3, P1 has 3 - tie, incumbent (P1) keeps
        assert PLAYERS[1].is_president_of(trade_state, 0), "On tie, incumbent P1 keeps presidency"
        assert_invariants(trade_state, "After buy with tie")

    def test_presidency_three_way_competition(self, trade_state):
        """Presidency goes to player with most shares among three shareholders."""
        # Set up: P0=2 (president), P1=1, P2=1
        PLAYERS[1].set_shares(trade_state, 0, 1)
        PLAYERS[2].set_shares(trade_state, 0, 1)
        # Update issued shares: bank(2) + P0(2) + P1(1) + P2(1) = 6, but corp has 7 total
        # So: unissued(1) + bank(2) + P0(2) + P1(1) + P2(1) = 7
        corp = CORPS[0]
        corp.set_unissued_shares(trade_state, 1)
        corp.set_issued_shares(trade_state, 6)

        # P0 sells, now has 1 share - three-way tie
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # All have 1 share - incumbent keeps presidency
        assert PLAYERS[0].is_president_of(trade_state, 0)
        assert_invariants(trade_state, "After three-way tie")

    def test_presidency_turn_order_tiebreaker(self, trade_state):
        """INV-20: When multiple players have more shares, use turn order from incumbent."""
        # Turn order with seed=42: P0=pos0, P1=pos1, P2=pos2
        # Set up: P0=2 (president), P1=3, P2=3
        # Both P1 and P2 have more than incumbent P0
        # P1 should win (next in turn order after P0)
        corp = CORPS[0]
        PLAYERS[1].set_shares(trade_state, 0, 3)
        PLAYERS[2].set_shares(trade_state, 0, 3)
        # Total: P0(2) + P1(3) + P2(3) + bank(2) = 10, but corp has 7 total
        # Adjust: unissued(0) + bank(0) + P0(2) + P1(3) + P2(2) = 7
        # Actually let's use: unissued(0) + bank(1) + P0(1) + P1(3) + P2(2) = 7
        PLAYERS[0].set_shares(trade_state, 0, 1)
        PLAYERS[2].set_shares(trade_state, 0, 2)
        corp.set_unissued_shares(trade_state, 0)
        corp.set_bank_shares(trade_state, 1)
        corp.set_issued_shares(trade_state, 6)

        # P0 sells their only share - triggers presidency check
        # After: P0=0, P1=3, P2=2 -> P1 wins (most shares)
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        assert PLAYERS[1].is_president_of(trade_state, 0)
        assert not PLAYERS[0].is_president_of(trade_state, 0)
        assert not PLAYERS[2].is_president_of(trade_state, 0)
        assert_invariants(trade_state, "After turn order tiebreak")

    def test_presidency_turn_order_tiebreaker_tie_at_max(self, trade_state):
        """When multiple players tie for max shares (more than incumbent), use turn order."""
        # Turn order with seed=42: P0=pos0, P1=pos1, P2=pos2
        # Set up: P0=1 (president), P1=3, P2=3
        # Both P1 and P2 tie at 3 shares, more than P0
        # P1 should win (next in turn order after P0 is position 1 = P1)
        corp = CORPS[0]
        PLAYERS[0].set_shares(trade_state, 0, 1)
        PLAYERS[1].set_shares(trade_state, 0, 3)
        PLAYERS[2].set_shares(trade_state, 0, 3)
        # Total: P0(1) + P1(3) + P2(3) = 7, no bank shares needed
        corp.set_unissued_shares(trade_state, 0)
        corp.set_bank_shares(trade_state, 0)
        corp.set_issued_shares(trade_state, 7)

        # Sell P0's share to trigger presidency check
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # P1 should be president (next in turn order after P0 with max shares)
        assert PLAYERS[1].is_president_of(trade_state, 0)
        assert not PLAYERS[0].is_president_of(trade_state, 0)
        assert not PLAYERS[2].is_president_of(trade_state, 0)
        assert_invariants(trade_state, "After turn order tiebreak with tie at max")

    def test_presidency_turn_order_wraps_around(self):
        """Turn order wraps around when checking for new president."""
        # Create fresh state
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 with P0 as initial president
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 15, 1)

        # Modify turn order: P0->pos2, P1->pos0, P2->pos1
        # This way when P0 (the president and active player) sells,
        # we check from position 2+1=0 (P1) for the tie-breaker
        PLAYERS[0].set_turn_order(state, 2)  # P0 at position 2
        PLAYERS[1].set_turn_order(state, 0)  # P1 at position 0
        PLAYERS[2].set_turn_order(state, 1)  # P2 at position 1

        # Set up shares: P0=1 (president), P1=3, P2=3, all 7 issued
        PLAYERS[0].set_shares(state, 0, 1)
        PLAYERS[1].set_shares(state, 0, 3)
        PLAYERS[2].set_shares(state, 0, 3)
        corp.set_unissued_shares(state, 0)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 7)

        PLAYERS[0].set_cash(state, 100)

        # P0 sells their share - triggers presidency check
        # After: P0=0, P1=3, P2=3
        # P0 was at position 2, so we start checking from position (2+1)%3 = 0
        # Position 0 is P1 with 3 shares -> P1 becomes president
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(state, sell_idx)

        # P1 should be president (at position 0, first after position 2 wrapping)
        assert PLAYERS[1].is_president_of(state, 0)
        assert not PLAYERS[0].is_president_of(state, 0)
        assert not PLAYERS[2].is_president_of(state, 0)
        assert_invariants(state, "After turn order wrap-around")

    def test_ipo_sets_player_as_president(self, ipo_state_with_company):
        """IPO sets company owner as corporation president."""
        state = ipo_state_with_company

        assert not PLAYERS[0].is_president_of(state, 0)

        apply_ipo_action_py(state, 0, 0)

        assert PLAYERS[0].is_president_of(state, 0)


# =============================================================================
# RECEIVERSHIP TESTS
# =============================================================================

class TestReceivership:
    """Test receivership mechanics (INV-20, INV-21)."""

    def test_receivership_when_all_shares_sold(self, trade_state):
        """INV-20: Corp enters receivership when all player shares = 0."""
        corp = CORPS[0]

        # Set up: only player 0 has 1 share
        PLAYERS[0].set_shares(trade_state, 0, 1)

        # Sell the last share
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # Corp should be in receivership (all player shares = 0)
        assert corp.is_in_receivership(trade_state)
        # No president in receivership
        for player_id in range(3):
            assert not PLAYERS[player_id].is_president_of(trade_state, 0)

    def test_receivership_exit_on_buy(self, trade_state):
        """INV-21: Buying share from receivership corp exits receivership and makes buyer president."""
        corp = CORPS[0]

        # Put corp in receivership
        corp.set_in_receivership(trade_state, True)
        PLAYERS[0].set_shares(trade_state, 0, 0)
        corp.set_bank_shares(trade_state, 5)

        # Buy a share
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Corp exits receivership
        assert not corp.is_in_receivership(trade_state)
        # Buyer becomes president (they have the most shares - the only holder)
        # Per CONTEXT.md: shares are fungible, no special "president share" handling
        assert PLAYERS[0].is_president_of(trade_state, 0)

    def test_receivership_corp_still_tradeable(self, trade_state):
        """Corp in receivership can still have shares bought (exits receivership)."""
        corp = CORPS[0]

        # Put corp in receivership - adjust shares to maintain invariant
        # Corp 0 has 7 total: unissued(3) + bank(4) + all_players(0) = 7
        corp.set_in_receivership(trade_state, True)
        corp.set_bank_shares(trade_state, 4)
        corp.set_issued_shares(trade_state, 0)
        PLAYERS[0].set_shares(trade_state, 0, 0)

        # Verify corp is in receivership
        assert corp.is_in_receivership(trade_state)

        # Buy should be valid
        mask = get_valid_action_mask(trade_state)
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        assert mask[buy_idx] == 1.0, "Buy should be valid for receivership corp"

        DRIVER.apply_action(trade_state, buy_idx)

        # No longer in receivership
        assert not corp.is_in_receivership(trade_state)
        assert PLAYERS[0].is_president_of(trade_state, 0)
        assert_invariants(trade_state, "After exit receivership")

    def test_receivership_sell_all_shares_from_multiple_players(self):
        """Multiple players selling down eventually leads to receivership."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp with P0 as president (1 share each to player and bank)
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 15, 1)

        # Adjust to have only P0 with 1 share, bank has 0
        corp.set_unissued_shares(state, 6)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 1)
        PLAYERS[0].set_shares(state, 0, 1)
        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0

        # P0 sells their only share - corp should enter receivership
        DRIVER.apply_action(state, sell_idx)

        # Corp should be in receivership (all player shares = 0)
        assert corp.is_in_receivership(state)
        # No one should be president
        for player_id in range(3):
            assert not PLAYERS[player_id].is_president_of(state, 0)

        assert_invariants(state, "After entering receivership")


# =============================================================================
# PRESIDENCY ON BANKRUPTCY
# =============================================================================

class TestPresidencyOnBankruptcy:
    """Test that bankruptcy clears president flags."""

    def test_bankruptcy_clears_president_flags(self, bankruptcy_state):
        """Bankruptcy clears all president flags for that corp."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        for player_id in range(3):
            assert not PLAYERS[player_id].is_president_of(bankruptcy_state, 0)
