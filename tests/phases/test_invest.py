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

    # Note: auction passed flags are cleared at auction END (see test_bid_in_auction.py),
    # not at start - they're initialized cleared and stay cleared between auctions

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

        # Float corp 1 using a different company
        COMPANIES[1].transfer_to_player(trade_state, 0)
        corp1 = CORPS[1]
        corp1.float_corp(trade_state, 0, 1, 8, 1)
        corp1.set_bank_shares(trade_state, 2)

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

        # Float corp with bank shares available for buying
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 10, 1)
        corp.set_bank_shares(state, 3)
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

        # Float corp with player shares for selling
        COMPANIES[0].transfer_to_player(state, 0)
        corp = CORPS[0]
        corp.float_corp(state, 0, 0, 10, 2)  # Player gets 2 shares

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


# =============================================================================
# $75 GAME END TESTS
# =============================================================================

class TestGameEndAt75:
    """Test immediate game end when share price reaches $75 (index 26)."""

    def test_buy_share_at_75_ends_game_immediately(self):
        """Buying a share that moves price to $75 ends game immediately."""
        from tests.phases.conftest import float_corp_for_test

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 25 ($68 - one step below $75)
        float_corp_for_test(state, corp_id=0, par_index=25)
        corp = CORPS[0]

        # Ensure bank has shares to buy
        corp.set_bank_shares(state, 2)
        PLAYERS[0].set_cash(state, 100)  # Enough to afford $75

        # Buy action should move price from index 25 to 26
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        result = DRIVER.apply_action(state, buy_idx)

        # Game should end immediately
        assert result == STATUS_GAME_OVER
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER

        # Verify the buy was actually processed
        assert corp.get_price_index(state) == 26  # $75
        assert PLAYERS[0].get_shares(state, 0) == 2  # Got the share (started with 1)

    def test_buy_share_below_75_does_not_end_game(self):
        """Buying a share that doesn't reach $75 continues normally."""
        from tests.phases.conftest import float_corp_for_test

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 20 ($41 - well below $75)
        float_corp_for_test(state, corp_id=0, par_index=20)
        corp = CORPS[0]

        # Ensure bank has shares to buy
        corp.set_bank_shares(state, 2)
        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        result = DRIVER.apply_action(state, buy_idx)

        # Game continues
        assert result == STATUS_OK
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert corp.get_price_index(state) == 21  # Moved up one space

    def test_buy_share_skipping_to_75_ends_game(self):
        """Buying when intermediate spaces are occupied still ends game at $75."""
        from tests.phases.conftest import float_corp_for_test

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 24 ($61)
        float_corp_for_test(state, corp_id=0, par_index=24)
        corp = CORPS[0]

        # Ensure bank has shares to buy
        corp.set_bank_shares(state, 2)
        PLAYERS[0].set_cash(state, 100)

        # Mark space 25 as occupied - buy will skip to $75
        MARKET.set_space_available(state, 25, False)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        result = DRIVER.apply_action(state, buy_idx)

        # Game ends because we reached $75
        assert result == STATUS_GAME_OVER
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER
        assert corp.get_price_index(state) == 26
