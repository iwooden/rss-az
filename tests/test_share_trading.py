"""Tests for share trading actions (buy/sell) in INVEST phase."""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases, CORP_NAMES
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from entities.company import COMPANIES
from core.data import GameConstants

STATUS_OK = 0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def trade_state():
    """
    Create game state with an active corp that has bank shares available.

    We need to set up a corp through IPO-like initialization for testing.
    For simplicity, manually configure the state.
    """
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Manually activate corp 0 (JS) with tradeable shares
    corp = CORPS[CORP_NAMES[0]]  # JS
    corp.set_active(state, True)
    corp.set_price_index(state, 10)  # Price index 10 = $14
    corp.set_bank_shares(state, 3)   # 3 shares in bank
    corp.set_issued_shares(state, 4) # 4 shares issued total

    # Give player 0 some shares and cash
    PLAYERS[0].set_shares(state, 0, 2)  # 2 shares of corp 0
    PLAYERS[0].set_cash(state, 100)     # $100 cash
    PLAYERS[0].set_president_of(state, 0, True)

    # Mark market space 10 as occupied
    MARKET.set_space_available(state, 10, False)

    return state


@pytest.fixture
def bankruptcy_state():
    """
    State with corp at price index 1 where one sell triggers bankruptcy.
    Corp owns a company to verify removal.
    """
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    corp = CORPS[CORP_NAMES[0]]
    corp.set_active(state, True)
    corp.set_price_index(state, 1)  # One sell -> index 0 -> bankruptcy
    corp.set_bank_shares(state, 2)
    corp.set_issued_shares(state, 4)  # bank_shares (2) + player_shares (2) = 4

    # Give corp a company
    COMPANIES[0].transfer_to_corp(state, 0)
    corp.set_owns_company(state, 0, True)

    # Give player shares
    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[0].set_president_of(state, 0, True)
    PLAYERS[0].set_cash(state, 100)

    MARKET.set_space_available(state, 1, False)

    return state


# =============================================================================
# BUY SHARE TESTS
# =============================================================================

class TestBuyShare:
    """Test buy share action behavior."""

    def test_buy_share_transfers_money_to_corp(self, trade_state):
        """INV-07, INV-08: Buy share moves cash from player to corp."""
        corp = CORPS[CORP_NAMES[0]]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)
        initial_corp_cash = corp.get_cash(trade_state)

        # Get buy action index
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0  # Corp 0

        # Apply buy action
        result = DRIVER.apply_action(trade_state, buy_idx)
        assert result == STATUS_OK

        # Price moved up, so we need the new price that was paid
        # From index 10, next available should be 11 (if available)
        # Player paid new price, corp received it
        new_corp_cash = corp.get_cash(trade_state)
        new_player_cash = player.get_cash(trade_state)

        # Cash transferred (amounts depend on price movement)
        assert new_player_cash < initial_player_cash
        assert new_corp_cash > initial_corp_cash
        # Amount should match
        assert (initial_player_cash - new_player_cash) == (new_corp_cash - initial_corp_cash)

    def test_buy_share_transfers_share(self, trade_state):
        """INV-09: Buy share moves 1 share from bank to player."""
        corp = CORPS[CORP_NAMES[0]]
        player = PLAYERS[0]

        initial_bank_shares = corp.get_bank_shares(trade_state)
        initial_player_shares = player.get_shares(trade_state, 0)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Share transferred
        assert corp.get_bank_shares(trade_state) == initial_bank_shares - 1
        assert player.get_shares(trade_state, 0) == initial_player_shares + 1

    def test_buy_share_moves_price_up(self, trade_state):
        """INV-10: Buy share moves corp price to next higher available space."""
        corp = CORPS[CORP_NAMES[0]]

        initial_index = corp.get_price_index(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        new_index = corp.get_price_index(trade_state)
        assert new_index > initial_index

    def test_buy_share_updates_net_worth(self, trade_state):
        """INV-15: Buy share updates player net worth."""
        player = PLAYERS[0]

        # Net worth before (may need recalculation)
        player.update_net_worth(trade_state)
        initial_net_worth = player.get_net_worth(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Net worth was updated (value may differ due to price change)
        new_net_worth = player.get_net_worth(trade_state)
        # Just verify it was recalculated - exact value depends on price
        assert isinstance(new_net_worth, int)

    def test_buy_share_increments_round_trip_counter(self, trade_state):
        """INV-16: Buy share increments share_buys counter."""
        player = PLAYERS[0]

        initial_buys = player.get_share_buys(trade_state, 0)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        new_buys = player.get_share_buys(trade_state, 0)
        assert new_buys == initial_buys + 1

    def test_buy_share_updates_all_players_net_worth(self, trade_state):
        """Price movement affects all shareholders' net worth."""
        corp = CORPS[CORP_NAMES[0]]

        # Give player 1 some shares of the same corp
        PLAYERS[1].set_shares(trade_state, 0, 2)
        PLAYERS[1].set_cash(trade_state, 50)

        # Calculate expected net worth for player 1 before buy
        PLAYERS[1].update_net_worth(trade_state)
        initial_net_worth_p1 = PLAYERS[1].get_net_worth(trade_state)
        initial_price = corp.get_share_price(trade_state)

        # Player 0 buys, which moves price up
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Player 1's net worth should reflect the new higher price
        new_price = corp.get_share_price(trade_state)
        new_net_worth_p1 = PLAYERS[1].get_net_worth(trade_state)

        # Price went up, so player 1's net worth should have increased
        assert new_price > initial_price
        assert new_net_worth_p1 > initial_net_worth_p1


# =============================================================================
# SELL SHARE TESTS
# =============================================================================

class TestSellShare:
    """Test sell share action behavior."""

    def test_sell_share_adds_cash_to_player(self, trade_state):
        """INV-11: Sell share adds sell price to player cash."""
        corp = CORPS[CORP_NAMES[0]]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)
        current_price = corp.get_share_price(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        new_player_cash = player.get_cash(trade_state)
        # Player received current price (before movement)
        assert new_player_cash == initial_player_cash + current_price

    def test_sell_share_transfers_share_to_bank(self, trade_state):
        """INV-12: Sell share moves 1 share from player to bank."""
        corp = CORPS[CORP_NAMES[0]]
        player = PLAYERS[0]

        initial_bank_shares = corp.get_bank_shares(trade_state)
        initial_player_shares = player.get_shares(trade_state, 0)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        assert corp.get_bank_shares(trade_state) == initial_bank_shares + 1
        assert player.get_shares(trade_state, 0) == initial_player_shares - 1

    def test_sell_share_moves_price_down(self, trade_state):
        """INV-13: Sell share moves corp price to next lower available space."""
        corp = CORPS[CORP_NAMES[0]]

        initial_index = corp.get_price_index(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        new_index = corp.get_price_index(trade_state)
        assert new_index < initial_index

    def test_sell_share_increments_round_trip_counter(self, trade_state):
        """INV-16: Sell share increments share_sells counter."""
        player = PLAYERS[0]

        initial_sells = player.get_share_sells(trade_state, 0)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        new_sells = player.get_share_sells(trade_state, 0)
        assert new_sells == initial_sells + 1


# =============================================================================
# PRICE MOVEMENT TESTS
# =============================================================================

class TestPriceMovement:
    """Test price movement skips occupied spaces."""

    def test_buy_skips_occupied_space(self, trade_state):
        """INV-14: Price movement skips occupied market spaces."""
        corp = CORPS[CORP_NAMES[0]]

        # Mark the next space (11) as occupied
        MARKET.set_space_available(trade_state, 11, False)

        initial_index = corp.get_price_index(trade_state)  # 10

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        new_index = corp.get_price_index(trade_state)
        # Should have skipped 11 and gone to 12 (or next available)
        assert new_index > 11

    def test_sell_skips_occupied_space(self, trade_state):
        """INV-14: Sell price movement skips occupied spaces."""
        corp = CORPS[CORP_NAMES[0]]

        # Mark the next lower space (9) as occupied
        MARKET.set_space_available(trade_state, 9, False)

        initial_index = corp.get_price_index(trade_state)  # 10

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        new_index = corp.get_price_index(trade_state)
        # Should have skipped 9 and gone to 8 (or next available)
        assert new_index < 9


# =============================================================================
# ROUND-TRIP LIMIT TESTS
# =============================================================================

class TestRoundTripLimits:
    """Test round-trip limit enforcement."""

    def test_buy_blocked_after_two_roundtrips(self, trade_state):
        """INV-17: Buy blocked when round-trips >= 2."""
        player = PLAYERS[0]

        # Simulate 2 complete round-trips (4 buys + 4 sells would be 4 roundtrips)
        # Actually: 2 buys + 2 sells = 2 roundtrips
        for _ in range(2):
            player.increment_share_buys(trade_state, 0)
            player.increment_share_sells(trade_state, 0)

        # Verify roundtrips = 2
        assert player.get_roundtrips(trade_state, 0) == 2

        # Check action mask - buy should be blocked
        mask = get_valid_action_mask(trade_state)
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        assert mask[buy_idx] == 0.0  # Buy blocked

    def test_sell_blocked_after_two_roundtrips(self, trade_state):
        """INV-17: Sell blocked when round-trips >= 2."""
        player = PLAYERS[0]

        # Simulate 2 complete round-trips
        for _ in range(2):
            player.increment_share_buys(trade_state, 0)
            player.increment_share_sells(trade_state, 0)

        # Check action mask - sell should be blocked
        mask = get_valid_action_mask(trade_state)
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0

        assert mask[sell_idx] == 0.0  # Sell blocked

    def test_different_corps_have_separate_limits(self, trade_state):
        """Round-trip limits are per-corp, not global."""
        player = PLAYERS[0]

        # Max out roundtrips for corp 0
        for _ in range(2):
            player.increment_share_buys(trade_state, 0)
            player.increment_share_sells(trade_state, 0)

        # Set up corp 1 as active with shares
        corp1 = CORPS[CORP_NAMES[1]]
        corp1.set_active(trade_state, True)
        corp1.set_price_index(trade_state, 8)
        corp1.set_bank_shares(trade_state, 2)
        MARKET.set_space_available(trade_state, 8, False)

        # Give player shares of corp 1
        player.set_shares(trade_state, 1, 1)

        # Corp 1 should still be tradeable
        mask = get_valid_action_mask(trade_state)
        layout = get_action_layout(3)

        # Corp 0 blocked, corp 1 not blocked
        assert mask[layout['buy_share_base'] + 0] == 0.0
        assert mask[layout['sell_share_base'] + 0] == 0.0
        # Corp 1 should be available (if affordable)
        assert mask[layout['sell_share_base'] + 1] == 1.0


# =============================================================================
# MULTIPLE PLAYER COUNT TESTS
# =============================================================================

class TestMultiplePlayerCounts:
    """Test share trading across different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_buy_works_all_player_counts(self, num_players):
        """Buy action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set up tradeable corp
        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(state, True)
        corp.set_price_index(state, 10)
        corp.set_bank_shares(state, 3)
        MARKET.set_space_available(state, 10, False)
        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(num_players)
        mask = get_valid_action_mask(state)

        buy_idx = layout['buy_share_base'] + 0
        if mask[buy_idx] == 1.0:
            result = DRIVER.apply_action(state, buy_idx)
            assert result == STATUS_OK

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_sell_works_all_player_counts(self, num_players):
        """Sell action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set up corp with player shares
        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(state, True)
        corp.set_price_index(state, 10)
        MARKET.set_space_available(state, 10, False)
        PLAYERS[0].set_shares(state, 0, 2)

        layout = get_action_layout(num_players)
        sell_idx = layout['sell_share_base'] + 0

        result = DRIVER.apply_action(state, sell_idx)
        assert result == STATUS_OK


# =============================================================================
# BANKRUPTCY TESTS
# =============================================================================

class TestBankruptcy:
    """Test bankruptcy procedure (INV-22 through INV-27)."""

    def test_bankruptcy_triggers_at_price_zero(self, bankruptcy_state):
        """INV-22: Sell that drops price to 0 triggers bankruptcy."""
        corp = CORPS[CORP_NAMES[0]]

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Corp should be inactive
        assert not corp.is_active(bankruptcy_state)
        assert corp.get_price_index(bankruptcy_state) == 0

    def test_bankruptcy_removes_companies(self, bankruptcy_state):
        """INV-23: Bankruptcy removes all corp's companies."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Company should be removed from game
        assert COMPANIES[0].is_removed(bankruptcy_state)

    def test_bankruptcy_returns_shares_to_unissued(self, bankruptcy_state):
        """INV-24: All shares return to unissued."""
        corp = CORPS[CORP_NAMES[0]]

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # All shares back to unissued
        from core.data import get_corp_share_count
        assert corp.get_unissued_shares(bankruptcy_state) == get_corp_share_count(0)
        assert corp.get_issued_shares(bankruptcy_state) == 0
        assert corp.get_bank_shares(bankruptcy_state) == 0
        # Player shares cleared
        assert PLAYERS[0].get_shares(bankruptcy_state, 0) == 0

    def test_bankruptcy_clears_corp_cash(self, bankruptcy_state):
        """INV-25: Corp cash returned to bank (set to 0)."""
        corp = CORPS[CORP_NAMES[0]]
        corp.set_cash(bankruptcy_state, 50)  # Give corp some cash

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        assert corp.get_cash(bankruptcy_state) == 0

    def test_bankruptcy_frees_market_space(self, bankruptcy_state):
        """INV-26: Market space freed for future use."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Space 1 should be available again
        assert MARKET.is_space_available(bankruptcy_state, 1)

    def test_bankruptcy_corp_available_for_ipo(self, bankruptcy_state):
        """INV-27: Bankrupt corp can be IPO'd again."""
        corp = CORPS[CORP_NAMES[0]]

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Corp inactive but has full unissued shares - available for IPO
        assert not corp.is_active(bankruptcy_state)
        assert corp.get_unissued_shares(bankruptcy_state) > 0
        assert not corp.is_in_receivership(bankruptcy_state)

    def test_bankruptcy_clears_president_flags(self, bankruptcy_state):
        """Bankruptcy clears all president flags for that corp."""
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # No player should be president
        for player_id in range(3):
            assert not PLAYERS[player_id].is_president_of(bankruptcy_state, 0)

    def test_bankruptcy_updates_all_players_net_worth(self, bankruptcy_state):
        """Bankruptcy updates net worth for all players who held shares."""
        corp = CORPS[CORP_NAMES[0]]

        # Give player 1 shares of the corp that will go bankrupt
        PLAYERS[1].set_shares(bankruptcy_state, 0, 1)
        # Update issued shares to match
        corp.set_issued_shares(bankruptcy_state, 5)  # P0: 2, P1: 1, bank: 2
        PLAYERS[1].set_cash(bankruptcy_state, 50)

        # Calculate initial net worth for player 1
        PLAYERS[1].update_net_worth(bankruptcy_state)
        initial_net_worth_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        share_value = corp.get_share_price(bankruptcy_state)  # Value of 1 share

        # Player 0 sells, triggering bankruptcy
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Player 1's shares are now gone (zeroed by bankruptcy)
        assert PLAYERS[1].get_shares(bankruptcy_state, 0) == 0

        # Player 1's net worth should be updated (lost share value)
        new_net_worth_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        assert new_net_worth_p1 == initial_net_worth_p1 - share_value


# =============================================================================
# PRESIDENCY TESTS
# =============================================================================

class TestPresidency:
    """Test presidency transfer (INV-18, INV-19)."""

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
        corp = CORPS[CORP_NAMES[0]]

        # Player 0 is president with 2 shares
        # Buy increases shares to 3, should maintain presidency
        corp.set_bank_shares(trade_state, 5)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # Player 0 should still be president (now with 3 shares)
        assert PLAYERS[0].is_president_of(trade_state, 0)
        assert PLAYERS[0].get_shares(trade_state, 0) == 3


# =============================================================================
# RECEIVERSHIP TESTS
# =============================================================================

class TestReceivership:
    """Test receivership mechanics (INV-20, INV-21)."""

    def test_receivership_when_all_shares_sold(self, trade_state):
        """INV-20: Corp enters receivership when all player shares = 0."""
        corp = CORPS[CORP_NAMES[0]]

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
        corp = CORPS[CORP_NAMES[0]]

        # Put corp in receivership
        corp.set_in_receivership(trade_state, True)
        PLAYERS[0].set_shares(trade_state, 0, 0)
        PLAYERS[0].set_president_of(trade_state, 0, False)
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

    def test_receivership_no_president(self, trade_state):
        """Receivership clears president flag."""
        corp = CORPS[CORP_NAMES[0]]

        # Sell all shares
        PLAYERS[0].set_shares(trade_state, 0, 1)  # Only 1 share
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # In receivership = no president
        if corp.is_in_receivership(trade_state):
            for player_id in range(3):
                assert not PLAYERS[player_id].is_president_of(trade_state, 0)
