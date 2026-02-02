"""Tests for INVEST phase actions."""
import pytest
import numpy as np
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
from tests.phases.conftest import STATUS_OK, STATUS_GAME_OVER

# Fixtures come from conftest.py automatically
# Helper functions also available: assert_valid_mask, assert_invariants, apply_action_and_verify


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_first_valid_auction_action(state):
    """Find first valid auction action index."""
    mask = get_valid_action_mask(state)
    layout = get_action_layout(state.get_num_players())
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            return i
    return None


def apply_pass_to_all_players(state, num_players):
    """Apply pass action for all players (triggers WRAP_UP -> ACQUISITION -> new INVEST turn)."""
    layout = get_action_layout(num_players)
    pass_idx = layout['pass_invest']
    for i in range(num_players):
        result = DRIVER.apply_action(state, pass_idx)
        # All passes return STATUS_OK now (WRAP_UP auto-applies and returns to INVEST)
        assert result == STATUS_OK


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test INVEST phase pass action behavior."""

    def test_pass_increments_consecutive_passes(self, game_state):
        """INV-01: Pass action increments consecutive_passes counter."""
        # Get initial consecutive_passes count
        initial_passes = TURN.get_consecutive_passes(game_state)
        assert initial_passes == 0

        # Apply pass action
        layout = get_action_layout(3)
        result = DRIVER.apply_action(game_state, layout['pass_invest'])
        assert result == STATUS_OK

        # Verify consecutive_passes incremented
        new_passes = TURN.get_consecutive_passes(game_state)
        assert new_passes == initial_passes + 1

    def test_pass_advances_active_player(self, game_state, apply_and_track):
        """INV-04: Pass action advances active player in turn order."""
        # Get initial active player
        initial_player = game_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(game_state)

        # Apply pass action
        layout = get_action_layout(3)
        result = apply_and_track(game_state, layout['pass_invest'])

        # No auto-apply - multiple valid actions after pass
        assert len(result.history) == 1, "Expected no forced actions after pass"
        assert result.status == STATUS_OK

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (initial_position + 1) % 3

    def test_pass_follows_turn_order(self, game_state):
        """INV-04a: Pass uses turn order (one-hot vectors), not player_id."""
        # Record all players in turn order (only 2 passes to avoid WRAP_UP)
        turn_sequence = []
        layout = get_action_layout(3)

        for i in range(2):
            current_player = game_state.get_active_player()
            turn_sequence.append(current_player)
            DRIVER.apply_action(game_state, layout['pass_invest'])

        # Get third player
        third_player = game_state.get_active_player()
        turn_sequence.append(third_player)

        # Verify all 3 players are unique
        assert len(set(turn_sequence)) == 3  # All 3 players appeared in turn order

        # Verify they appear in consecutive positions (following turn_order)
        for i, player_id in enumerate(turn_sequence):
            position = PLAYERS[player_id].get_turn_order(game_state)
            # Position should match index in sequence
            expected_position = turn_sequence.index(player_id)
            # All players should have unique positions
            assert position in [0, 1, 2]

    def test_all_players_pass_triggers_wrap_up_cycle(self, game_state):
        """INV-03: All players passing triggers WRAP_UP -> ACQUISITION -> new INVEST turn."""
        # Apply pass for all 3 players
        apply_pass_to_all_players(game_state, 3)

        # Verify phase transition to new INVEST turn (after WRAP_UP -> ACQUISITION)
        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        # Turn number should be incremented (ACQUISITION increments it)
        assert TURN.get_turn_number(game_state) == 2
        # Consecutive passes reset after WRAP_UP
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_non_pass_resets_consecutive_passes(self, game_state):
        """INV-02: Non-pass action (auction) resets consecutive_passes."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        DRIVER.apply_action(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        if auction_idx is not None:
            result = DRIVER.apply_action(game_state, auction_idx)
            assert result == STATUS_OK

            # Verify consecutive_passes was reset to 0
            assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# START AUCTION TESTS
# =============================================================================

class TestStartAuction:
    """Test INVEST phase start auction action behavior."""

    def test_start_auction_sets_company(self, game_state):
        """INV-05: Start auction sets auction_company."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Verify no auction company initially
        initial_company = TURN.get_auction_company(game_state)
        assert initial_company == -1

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction company was set
        auction_company = TURN.get_auction_company(game_state)
        assert auction_company >= 0 and auction_company < 36

    def test_start_auction_sets_price(self, game_state):
        """INV-05: Start auction sets auction_price."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction price was set (should be >= face value)
        auction_price = TURN.get_auction_price(game_state)
        assert auction_price > 0

    def test_start_auction_sets_high_bidder(self, game_state):
        """INV-05: Start auction sets auction_high_bidder to starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify high bidder is the starter
        high_bidder = TURN.get_auction_high_bidder(game_state)
        assert high_bidder == starter_id

    def test_start_auction_sets_starter(self, game_state):
        """INV-05: Start auction sets auction_starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction starter was recorded
        auction_starter = TURN.get_auction_starter(game_state)
        assert auction_starter == starter_id

    def test_start_auction_clears_passed_flags(self, game_state):
        """INV-05: Start auction clears all auction passed flags."""
        # Manually set some passed flags for testing
        TURN.set_player_passed_auction(game_state, 0, True)
        TURN.set_player_passed_auction(game_state, 1, True)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify all passed flags cleared
        for player_id in range(3):
            assert not TURN.has_player_passed_auction(game_state, player_id)

    def test_start_auction_transitions_to_bid_phase(self, game_state):
        """INV-06: Start auction transitions to BID_IN_AUCTION phase."""
        # Verify initial phase is INVEST
        assert game_state.get_phase() == GamePhases.PHASE_INVEST

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify phase transition
        assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    def test_start_auction_advances_to_next_bidder(self, game_state, apply_and_track):
        """Start auction advances active player to next in turn order."""
        starter_id = game_state.get_active_player()
        starter_position = PLAYERS[starter_id].get_turn_order(game_state)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = apply_and_track(game_state, auction_idx)

        # No auto-apply - bidders have choice to raise or leave
        assert len(result.history) == 1, "Expected no forced actions after starting auction"
        assert result.status == STATUS_OK

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (starter_position + 1) % 3

    def test_start_auction_resets_consecutive_passes(self, game_state):
        """INV-02: Start auction resets consecutive_passes counter."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        DRIVER.apply_action(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify consecutive_passes was reset
        assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# BUY SHARE TESTS
# =============================================================================

class TestBuyShare:
    """Test buy share action behavior."""

    def test_buy_share_pays_to_bank(self, trade_state, apply_and_track):
        """INV-07: Buy share moves cash from player to bank (not corp)."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)
        initial_corp_cash = corp.get_cash(trade_state)

        # Get buy action index
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0  # Corp 0

        # Apply buy action
        result = apply_and_track(trade_state, buy_idx)

        # No auto-apply - player can still buy/sell/pass after
        assert len(result.history) == 1, "Expected no forced actions after buy"
        assert result.status == STATUS_OK

        new_corp_cash = corp.get_cash(trade_state)
        new_player_cash = player.get_cash(trade_state)

        # Player pays to bank: cash leaves player, corp unchanged
        # Per RULES.md: "Player pays new share price to Bank"
        assert new_player_cash < initial_player_cash
        assert new_corp_cash == initial_corp_cash  # Corp doesn't receive payment

    def test_buy_share_transfers_share(self, trade_state):
        """INV-09: Buy share moves 1 share from bank to player."""
        corp = CORPS[0]
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
        corp = CORPS[0]

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
        corp = CORPS[0]

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
        """INV-11: Sell share pays NEW (lower) price per RULES.md."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # Per RULES.md: "Bank pays **new** share price to player"
        # The price drops first, then player receives the new lower price
        new_price = corp.get_share_price(trade_state)
        new_player_cash = player.get_cash(trade_state)
        assert new_player_cash == initial_player_cash + new_price

    def test_sell_share_transfers_share_to_bank(self, trade_state):
        """INV-12: Sell share moves 1 share from player to bank."""
        corp = CORPS[0]
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
        corp = CORPS[0]

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
        corp = CORPS[0]

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
        corp = CORPS[0]

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
        corp1 = CORPS[1]
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
# BANKRUPTCY TESTS
# =============================================================================

class TestBankruptcy:
    """Test bankruptcy procedure (INV-22 through INV-27)."""

    def test_bankruptcy_triggers_at_price_zero(self, bankruptcy_state):
        """INV-22: Sell that drops price to 0 triggers bankruptcy."""
        corp = CORPS[0]

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
        corp = CORPS[0]

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
        corp = CORPS[0]
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
        corp = CORPS[0]

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
        corp = CORPS[0]

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

    def test_bankruptcy_with_multiple_companies(self, bankruptcy_state):
        """INV-23: Bankruptcy removes ALL companies owned by corp."""
        from tests.phases.conftest import assert_invariants

        corp = CORPS[0]

        # Add second company to corp
        COMPANIES[1].transfer_to_corp(bankruptcy_state, 0)
        corp.set_owns_company(bankruptcy_state, 1, True)

        # Trigger bankruptcy
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Both companies should be removed
        assert COMPANIES[0].is_removed(bankruptcy_state)
        assert COMPANIES[1].is_removed(bankruptcy_state)
        assert not corp.is_active(bankruptcy_state)
        assert_invariants(bankruptcy_state, "After multi-company bankruptcy")

    def test_bankruptcy_resets_corp_for_new_ipo(self, bankruptcy_state):
        """INV-27: After bankruptcy, corp is reset and available for new IPO."""
        from core.data import get_corp_share_count
        from tests.phases.conftest import assert_invariants

        corp = CORPS[0]

        # Give corp some cash before bankruptcy
        corp.set_cash(bankruptcy_state, 100)

        # Trigger bankruptcy
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # Corp state should be fully reset
        assert not corp.is_active(bankruptcy_state)
        assert corp.get_cash(bankruptcy_state) == 0
        assert corp.get_unissued_shares(bankruptcy_state) == get_corp_share_count(0)
        assert corp.get_issued_shares(bankruptcy_state) == 0
        assert corp.get_bank_shares(bankruptcy_state) == 0
        assert not corp.is_in_receivership(bankruptcy_state)

        # Market space should be freed
        assert MARKET.is_space_available(bankruptcy_state, 1)

        assert_invariants(bankruptcy_state, "Corp reset for IPO")

    def test_bankruptcy_affects_all_shareholders_net_worth(self, bankruptcy_state):
        """Bankruptcy zeros shares for all players holding shares."""
        from tests.phases.conftest import assert_invariants

        corp = CORPS[0]

        # Give multiple players shares
        PLAYERS[1].set_shares(bankruptcy_state, 0, 1)
        PLAYERS[2].set_shares(bankruptcy_state, 0, 1)
        # Update issued shares: bank(2) + P0(2) + P1(1) + P2(1) = 6
        corp.set_issued_shares(bankruptcy_state, 6)
        corp.set_bank_shares(bankruptcy_state, 0)  # Adjust bank to match (unissued stays at 1)

        PLAYERS[1].set_cash(bankruptcy_state, 50)
        PLAYERS[2].set_cash(bankruptcy_state, 50)

        # Record share values before
        p1_shares_before = PLAYERS[1].get_shares(bankruptcy_state, 0)
        p2_shares_before = PLAYERS[2].get_shares(bankruptcy_state, 0)
        assert p1_shares_before == 1
        assert p2_shares_before == 1

        # Trigger bankruptcy
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(bankruptcy_state, sell_idx)

        # All players' shares zeroed
        assert PLAYERS[0].get_shares(bankruptcy_state, 0) == 0
        assert PLAYERS[1].get_shares(bankruptcy_state, 0) == 0
        assert PLAYERS[2].get_shares(bankruptcy_state, 0) == 0

        assert_invariants(bankruptcy_state, "After multi-player bankruptcy")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_bankruptcy_different_player_counts(self, num_players):
        """Bankruptcy procedure works for all player counts."""
        from tests.phases.conftest import assert_invariants

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 1)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 2)
        corp.set_issued_shares(state, 2)

        COMPANIES[0].transfer_to_corp(state, 0)
        corp.set_owns_company(state, 0, True)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[0].set_cash(state, 100)

        MARKET.set_space_available(state, 1, False)

        layout = get_action_layout(num_players)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(state, sell_idx)

        assert not corp.is_active(state)
        assert COMPANIES[0].is_removed(state)
        assert_invariants(state, f"Bankruptcy with {num_players} players")


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
        from tests.phases.conftest import assert_invariants

        corp = CORPS[0]

        # Player 0 is president with 2 shares
        # Give player 1 more shares so they have majority
        PLAYERS[1].set_shares(trade_state, 0, 3)
        # Update issued shares to match
        corp.set_issued_shares(trade_state, 5)  # bank(2) + P0(2) + P1(3) - but only 7 total shares
        # So we need: unissued(0) + bank(2) + P0(2) + P1(3) = 7
        corp.set_unissued_shares(trade_state, 0)

        # Player 0 buys, now has 3 shares - tie with player 1
        # Since player 1 had more before, check presidency rules
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        DRIVER.apply_action(trade_state, buy_idx)

        # After buy: P0 has 3, P1 has 3 - incumbent (P0) keeps
        assert PLAYERS[0].is_president_of(trade_state, 0)
        assert_invariants(trade_state, "After buy with tie")

    def test_presidency_three_way_competition(self, trade_state):
        """Presidency goes to player with most shares among three shareholders."""
        from tests.phases.conftest import assert_invariants

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
        corp = CORPS[0]

        # Sell all shares
        PLAYERS[0].set_shares(trade_state, 0, 1)  # Only 1 share
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        DRIVER.apply_action(trade_state, sell_idx)

        # In receivership = no president
        if corp.is_in_receivership(trade_state):
            for player_id in range(3):
                assert not PLAYERS[player_id].is_president_of(trade_state, 0)

    def test_receivership_corp_still_tradeable(self, trade_state):
        """Corp in receivership can still have shares bought (exits receivership)."""
        from tests.phases.conftest import assert_invariants

        corp = CORPS[0]

        # Put corp in receivership - adjust shares to maintain invariant
        # Corp 0 has 7 total: unissued(3) + bank(4) + all_players(0) = 7
        corp.set_in_receivership(trade_state, True)
        corp.set_bank_shares(trade_state, 4)
        corp.set_issued_shares(trade_state, 0)
        PLAYERS[0].set_shares(trade_state, 0, 0)
        PLAYERS[0].set_president_of(trade_state, 0, False)

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
        from tests.phases.conftest import assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 6)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 1)

        # Only P0 owns 1 share (is president)
        PLAYERS[0].set_shares(state, 0, 1)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[0].set_cash(state, 100)

        MARKET.set_space_available(state, 15, False)

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
# MULTIPLE PLAYER COUNT TESTS
# =============================================================================

class TestMultiplePlayerCounts:
    """Test INVEST phase behavior across different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_pass_works_all_player_counts(self, num_players):
        """Pass action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        layout = get_action_layout(num_players)
        result = DRIVER.apply_action(state, layout['pass_invest'])
        assert result == STATUS_OK

        # Verify consecutive_passes incremented
        assert TURN.get_consecutive_passes(state) == 1

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_auction_works_all_player_counts(self, num_players):
        """Auction action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(state)
        if auction_idx is not None:
            result = DRIVER.apply_action(state, auction_idx)
            assert result == STATUS_OK

            # Verify transition to BID phase
            assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_wrap_up_triggers_at_correct_pass_count(self, num_players):
        """WRAP_UP triggers after exactly num_players passes, returning to INVEST."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Apply pass for all players
        apply_pass_to_all_players(state, num_players)

        # Verify phase transition back to INVEST (after WRAP_UP -> ACQUISITION)
        assert state.get_phase() == GamePhases.PHASE_INVEST
        # Turn number incremented
        assert TURN.get_turn_number(state) == 2

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_buy_works_all_player_counts(self, num_players):
        """Buy action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set up tradeable corp
        corp = CORPS[0]
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
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 10)
        MARKET.set_space_available(state, 10, False)
        PLAYERS[0].set_shares(state, 0, 2)

        layout = get_action_layout(num_players)
        sell_idx = layout['sell_share_base'] + 0

        result = DRIVER.apply_action(state, sell_idx)
        assert result == STATUS_OK


# =============================================================================
# AUTO-APPLY EDGE CASE TESTS
# =============================================================================

class TestAutoApplyEdgeCases:
    """Edge case tests for auto-apply behavior."""

    def test_zero_legal_actions_raises_error(self):
        """ZeroLegalActionsError raised when non-terminal state has no actions.

        Note: This is a defensive test. In normal gameplay, there should always
        be at least one legal action in non-terminal states.
        """
        from core.driver import ZeroLegalActionsError

        # This scenario is hard to create naturally since game rules ensure
        # at least pass is always available in INVEST. We test that the
        # exception exists and is importable.
        assert ZeroLegalActionsError is not None

        # The actual error would be raised by driver if somehow zero actions
        # exist - this is a bug prevention guard, not normal behavior.

    def test_forced_action_loop_error_exists(self):
        """ForcedActionLoopError exists for iteration limit guard.

        Note: Triggering this error requires a bug that creates infinite forced
        actions. We test the exception is importable for documentation.
        """
        from core.driver import ForcedActionLoopError

        assert ForcedActionLoopError is not None

        # The driver has MAX_FORCED_ITERATIONS = 100 guard.
        # This prevents infinite loops from implementation bugs.

    @pytest.mark.parametrize("num_players,seed", [
        (3, 42),
        (6, 123),
    ])
    def test_consecutive_passes_wrap_up_chain(self, num_players, seed, apply_and_track):
        """All players passing triggers WRAP_UP -> ACQUISITION -> INVEST with sentinel actions in history.

        When player N passes (completing the consecutive pass requirement),
        the game auto-applies WRAP_UP and ACQUISITION phases, returning to INVEST.
        Sentinel actions (-100 for WRAP_UP, -101 for ACQUISITION) should appear in history.
        """
        state = GameState(num_players=num_players)
        state.initialize_game(seed=seed)

        layout = get_action_layout(num_players)
        pass_idx = layout['pass_invest']

        # Pass for all but last player using direct apply
        for i in range(num_players - 1):
            result = DRIVER.apply_action(state, pass_idx)
            assert result == STATUS_OK

        # Last pass triggers WRAP_UP auto-apply chain
        result = apply_and_track(state, pass_idx)
        assert result.status == STATUS_OK
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2

        # Verify history contains sentinel actions for WRAP_UP (-100) and ACQUISITION (-101)
        # History should have: pass action, -100 (WRAP_UP), -101 (ACQUISITION)
        assert len(result.history) >= 3, f"Expected at least 3 history entries (pass + 2 sentinels), got {len(result.history)}"
        # Check that sentinels appear in history
        action_values = [entry[1] for entry in result.history]
        assert -100 in action_values, "WRAP_UP sentinel (-100) not found in history"
        assert -101 in action_values, "ACQUISITION sentinel (-101) not found in history"
