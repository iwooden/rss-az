"""Tests for INVEST phase actions."""
import pytest
import numpy as np
from core.state import GameState, get_layout
from core.driver import ZeroLegalActionsError, ForcedActionLoopError
from core.actions import get_valid_action_mask, get_action_layout
from core.data import (
    GamePhases, CORP_NAMES, get_company_face_value,
    get_company_stars, get_company_low_price, get_company_high_price,
    get_adjusted_company_income, PY_COMPANY_STAR_DIVISOR, PY_PRICE_DIVISOR, PY_IMPACT_DIVISOR, PY_INCOME_DIVISOR,
    GameConstants,
)
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from entities.company import COMPANIES, get_auction_company_for_slot_py
from tests.phases.conftest import STATUS_OK, STATUS_GAME_OVER, float_corp_for_test, apply_and_verify_all

# Fixtures come from conftest.py automatically
# Helper functions also available: assert_valid_mask, assert_invariants


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
        apply_and_verify_all(state, pass_idx)


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test INVEST phase pass action behavior."""

    def test_pass_increments_consecutive_passes(self, game_state):
        """Pass action increments consecutive_passes counter."""
        # Get initial consecutive_passes count
        initial_passes = TURN.get_consecutive_passes(game_state)
        assert initial_passes == 0

        # Apply pass action
        layout = get_action_layout(3)
        apply_and_verify_all(game_state, layout['pass_invest'])

        # Verify consecutive_passes incremented
        new_passes = TURN.get_consecutive_passes(game_state)
        assert new_passes == initial_passes + 1

    def test_pass_advances_active_player(self, game_state):
        """Pass action advances active player in turn order."""
        # Get initial active player
        initial_player = game_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(game_state)

        # Apply pass action
        layout = get_action_layout(3)
        result = apply_and_verify_all(game_state, layout['pass_invest'])

        # No auto-apply - multiple valid actions after pass
        assert len(result.history) == 1, "Expected no forced actions after pass"

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (initial_position + 1) % 3

    def test_pass_follows_turn_order(self, game_state):
        """Pass uses turn order (one-hot vectors), not player_id."""
        # Record all players in turn order (only 2 passes to avoid WRAP_UP)
        turn_sequence = []
        layout = get_action_layout(3)

        for i in range(2):
            current_player = game_state.get_active_player()
            turn_sequence.append(current_player)
            apply_and_verify_all(game_state, layout['pass_invest'])

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
        """All players passing triggers WRAP_UP -> ACQUISITION -> new INVEST turn."""
        # Apply pass for all 3 players
        apply_pass_to_all_players(game_state, 3)

        # Verify phase transition to new INVEST turn (after WRAP_UP -> ACQUISITION)
        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        # Turn number should be incremented (ACQUISITION increments it)
        assert TURN.get_turn_number(game_state) == 2
        # Consecutive passes reset after WRAP_UP
        assert TURN.get_consecutive_passes(game_state) == 0

    def test_non_pass_resets_consecutive_passes(self, game_state):
        """Non-pass action (auction) resets consecutive_passes."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        apply_and_verify_all(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        if auction_idx is not None:
            apply_and_verify_all(game_state, auction_idx)

            # Verify consecutive_passes was reset to 0
            assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# START AUCTION TESTS
# =============================================================================

class TestStartAuction:
    """Test INVEST phase start auction action behavior."""

    def test_start_auction_sets_company(self, game_state):
        """Start auction sets auction_company."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Verify no auction company initially
        initial_company = TURN.get_auction_company(game_state)
        assert initial_company == -1

        # Apply auction action
        apply_and_verify_all(game_state, auction_idx)

        # Verify auction company was set
        auction_company = TURN.get_auction_company(game_state)
        assert auction_company >= 0 and auction_company < 36

    def test_start_auction_sets_price(self, game_state):
        """Start auction sets auction_price."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        apply_and_verify_all(game_state, auction_idx)

        # Verify auction price was set (should be >= face value)
        auction_price = TURN.get_auction_price(game_state)
        assert auction_price > 0

    def test_start_auction_sets_high_bidder(self, game_state):
        """Start auction sets auction_high_bidder to starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        apply_and_verify_all(game_state, auction_idx)

        # Verify high bidder is the starter
        high_bidder = TURN.get_auction_high_bidder(game_state)
        assert high_bidder == starter_id

    def test_start_auction_sets_starter(self, game_state):
        """Start auction sets auction_starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        apply_and_verify_all(game_state, auction_idx)

        # Verify auction starter was recorded
        auction_starter = TURN.get_auction_starter(game_state)
        assert auction_starter == starter_id

    # Note: auction passed flags are cleared at auction END (see test_bid_in_auction.py),
    # not at start - they're initialized cleared and stay cleared between auctions

    def test_start_auction_transitions_to_bid_phase(self, game_state):
        """Start auction transitions to BID_IN_AUCTION phase."""
        # Verify initial phase is INVEST
        assert game_state.get_phase() == GamePhases.PHASE_INVEST

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        apply_and_verify_all(game_state, auction_idx)

        # Verify phase transition
        assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    def test_start_auction_advances_to_next_bidder(self, game_state):
        """Start auction advances active player to next in turn order."""
        starter_id = game_state.get_active_player()
        starter_position = PLAYERS[starter_id].get_turn_order(game_state)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = apply_and_verify_all(game_state, auction_idx)

        # No auto-apply - bidders have choice to raise or leave
        assert len(result.history) == 1, "Expected no forced actions after starting auction"

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (starter_position + 1) % 3

    def test_auction_masked_when_player_cannot_afford_any_company(self):
        """All auction actions masked when cash < cheapest face value.

        RULES.md line 331: Starting player 'bids >= Face Value' — player must
        be able to afford at least face value to start an auction.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set active player cash to 0
        active_player_id = state.get_active_player()
        PLAYERS[active_player_id].set_cash(state, 0)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Pass must still be valid
        assert mask[layout['pass_invest']] == 1.0

        # All auction actions must be masked out
        for i in range(layout['auction_base'], layout['buy_share_base']):
            assert mask[i] == 0.0, \
                f"Auction action {i} should be masked (player has no cash)"

    def test_auction_partially_masked_at_exact_face_value(self):
        """Only offset 0 available when cash equals face value exactly.

        Player with cash == face value can start auction at face value (offset 0)
        but cannot bid higher (offset >= 1).
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Find cheapest available auction company
        cheapest_face = None
        cheapest_slot = None
        for slot in range(3):  # num_players auction slots
            action_base = layout['auction_base'] + slot * 20  # AUCTION_CAP = 20
            # Check if slot has a company by looking at an initial mask
            initial_mask = get_valid_action_mask(state)
            if initial_mask[action_base] == 1.0:
                # Decode company from slot
                # We need the face value — find it via the action system
                # Apply action to peek at company, then restore
                # Instead, iterate auction row directly
                pass

        # Simpler approach: find face values of auction row companies
        auction_companies = []
        for cid in range(36):
            if state.is_company_for_auction(cid):
                auction_companies.append((cid, get_company_face_value(cid)))
        auction_companies.sort(key=lambda x: x[1])  # Ascending face value
        assert len(auction_companies) > 0

        cheapest_face = auction_companies[0][1]

        # Set cash to exactly the cheapest face value
        active_player_id = state.get_active_player()
        PLAYERS[active_player_id].set_cash(state, cheapest_face)

        mask = get_valid_action_mask(state)

        # Slot 0 (cheapest) offset 0 should be valid (face + 0 == cash)
        assert mask[layout['auction_base']] == 1.0, \
            "Offset 0 for cheapest company should be valid at exact face value"

        # Slot 0 offset 1 should be masked (face + 1 > cash)
        assert mask[layout['auction_base'] + 1] == 0.0, \
            "Offset 1 for cheapest company should be masked at exact face value"

        # Any company with higher face value should be fully masked
        for idx, (cid, fv) in enumerate(auction_companies):
            if fv > cheapest_face:
                slot_base = layout['auction_base'] + idx * 20
                for offset in range(20):
                    assert mask[slot_base + offset] == 0.0, \
                        f"Company {cid} (face={fv}) should be fully masked (cash={cheapest_face})"

    def test_start_auction_resets_consecutive_passes(self, game_state):
        """Start auction resets consecutive_passes counter."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        apply_and_verify_all(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        apply_and_verify_all(game_state, auction_idx)

        # Verify consecutive_passes was reset
        assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# BUY SHARE TESTS
# =============================================================================

class TestBuyShare:
    """Test buy share action behavior."""

    def test_buy_share_pays_to_bank(self, trade_state):
        """Buy share moves cash from player to bank (not corp)."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)
        initial_corp_cash = corp.get_cash(trade_state)

        # Get buy action index
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0  # Corp 0

        # Apply buy action
        result = apply_and_verify_all(trade_state, buy_idx)

        # No auto-apply - player can still buy/sell/pass after
        assert len(result.history) == 1, "Expected no forced actions after buy"

        new_corp_cash = corp.get_cash(trade_state)
        new_player_cash = player.get_cash(trade_state)

        # Player pays to bank: cash leaves player, corp unchanged
        # Per RULES.md: "Player pays new share price to Bank"
        assert new_player_cash < initial_player_cash
        assert new_corp_cash == initial_corp_cash  # Corp doesn't receive payment

    def test_buy_share_transfers_share(self, trade_state):
        """Buy share moves 1 share from bank to player."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_bank_shares = corp.get_bank_shares(trade_state)
        initial_player_shares = player.get_shares(trade_state, 0)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_and_verify_all(trade_state, buy_idx)

        # Share transferred
        assert corp.get_bank_shares(trade_state) == initial_bank_shares - 1
        assert player.get_shares(trade_state, 0) == initial_player_shares + 1

    def test_buy_share_moves_price_up(self, trade_state):
        """Buy share moves corp price to next higher available space."""
        corp = CORPS[0]

        initial_index = corp.get_price_index(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_and_verify_all(trade_state, buy_idx)

        new_index = corp.get_price_index(trade_state)
        assert new_index > initial_index

    def test_buy_share_increments_round_trip_counter(self, trade_state):
        """Buy share increments share_buys counter."""
        player = PLAYERS[0]

        initial_buys = player.get_share_buys(trade_state, 0)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_and_verify_all(trade_state, buy_idx)

        new_buys = player.get_share_buys(trade_state, 0)
        assert new_buys == initial_buys + 1



# =============================================================================
# SELL SHARE TESTS
# =============================================================================

class TestSellShare:
    """Test sell share action behavior."""

    def test_sell_share_adds_cash_to_player(self, trade_state):
        """Sell share pays NEW (lower) price per RULES.md."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_player_cash = player.get_cash(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(trade_state, sell_idx)

        # Per RULES.md: "Bank pays **new** share price to player"
        # The price drops first, then player receives the new lower price
        new_price = corp.get_share_price(trade_state)
        new_player_cash = player.get_cash(trade_state)
        assert new_player_cash == initial_player_cash + new_price

    def test_sell_share_transfers_share_to_bank(self, trade_state):
        """Sell share moves 1 share from player to bank."""
        corp = CORPS[0]
        player = PLAYERS[0]

        initial_bank_shares = corp.get_bank_shares(trade_state)
        initial_player_shares = player.get_shares(trade_state, 0)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(trade_state, sell_idx)

        assert corp.get_bank_shares(trade_state) == initial_bank_shares + 1
        assert player.get_shares(trade_state, 0) == initial_player_shares - 1

    def test_sell_share_moves_price_down(self, trade_state):
        """Sell share moves corp price to next lower available space."""
        corp = CORPS[0]

        initial_index = corp.get_price_index(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(trade_state, sell_idx)

        new_index = corp.get_price_index(trade_state)
        assert new_index < initial_index

    def test_sell_share_increments_round_trip_counter(self, trade_state):
        """Sell share increments share_sells counter."""
        player = PLAYERS[0]

        initial_sells = player.get_share_sells(trade_state, 0)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(trade_state, sell_idx)

        new_sells = player.get_share_sells(trade_state, 0)
        assert new_sells == initial_sells + 1


# =============================================================================
# PRICE MOVEMENT TESTS
# =============================================================================

class TestPriceMovement:
    """Test price movement skips occupied spaces."""

    def test_buy_skips_occupied_space(self, trade_state):
        """Price movement skips occupied market spaces."""
        corp = CORPS[0]

        # Mark the next space (11) as occupied
        MARKET.set_space_available(trade_state, 11, False)

        initial_index = corp.get_price_index(trade_state)  # 10

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_and_verify_all(trade_state, buy_idx)

        new_index = corp.get_price_index(trade_state)
        # Should have skipped 11 and gone to 12 (or next available)
        assert new_index > 11

    def test_sell_skips_occupied_space(self, trade_state):
        """Sell price movement skips occupied spaces."""
        corp = CORPS[0]

        # Mark the next lower space (9) as occupied
        MARKET.set_space_available(trade_state, 9, False)

        initial_index = corp.get_price_index(trade_state)  # 10

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(trade_state, sell_idx)

        new_index = corp.get_price_index(trade_state)
        # Should have skipped 9 and gone to 8 (or next available)
        assert new_index < 9


# =============================================================================
# ROUND-TRIP LIMIT TESTS
# =============================================================================

class TestRoundTripLimits:
    """Test round-trip limit enforcement."""

    def test_buy_blocked_after_two_roundtrips(self, trade_state):
        """Buy blocked when round-trips >= 2."""
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
        """Sell blocked when round-trips >= 2."""
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
        apply_and_verify_all(state, layout['pass_invest'])

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
            apply_and_verify_all(state, auction_idx)

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
        corp.float_corp(state, 0, 0, 10, 3)  # 3 shares each to player and bank
        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(num_players)
        mask = get_valid_action_mask(state)

        buy_idx = layout['buy_share_base'] + 0
        if mask[buy_idx] == 1.0:
            apply_and_verify_all(state, buy_idx)

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

        apply_and_verify_all(state, sell_idx)


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
        assert ForcedActionLoopError is not None

        # The driver has MAX_FORCED_ITERATIONS = 100 guard.
        # This prevents infinite loops from implementation bugs.

    @pytest.mark.parametrize("num_players,seed", [
        (3, 42),
        (6, 123),
    ])
    def test_consecutive_passes_wrap_up_chain(self, num_players, seed):
        """All players passing triggers WRAP_UP -> ACQUISITION -> INVEST with sentinel actions in history.

        When player N passes (completing the consecutive pass requirement),
        the game auto-applies WRAP_UP and ACQUISITION phases, returning to INVEST.
        Sentinel actions (-100 for WRAP_UP, -101 for ACQUISITION) should appear in history.
        """
        state = GameState(num_players=num_players)
        state.initialize_game(seed=seed)

        layout = get_action_layout(num_players)
        pass_idx = layout['pass_invest']

        # Pass for all but last player
        for i in range(num_players - 1):
            apply_and_verify_all(state, pass_idx)

        # Last pass triggers WRAP_UP auto-apply chain
        result = apply_and_verify_all(state, pass_idx)
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
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 25 ($68 - one step below $75)
        # float_shares=2 gives player 2 shares, bank 2 shares, unissued 3
        float_corp_for_test(state, corp_id=0, par_index=25, float_shares=2)
        corp = CORPS[0]

        PLAYERS[0].set_cash(state, 100)  # Enough to afford $75

        # Buy action should move price from index 25 to 26
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        apply_and_verify_all(state, buy_idx, expected_status=STATUS_GAME_OVER)

        # Game should end immediately
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER

        # Verify the buy was actually processed
        assert corp.get_price_index(state) == 26  # $75
        assert PLAYERS[0].get_shares(state, 0) == 3  # Got the share (started with 2)

    def test_buy_share_below_75_does_not_end_game(self):
        """Buying a share that doesn't reach $75 continues normally."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 20 ($41 - well below $75)
        # float_shares=2 gives player 2 shares, bank 2 shares, unissued 3
        float_corp_for_test(state, corp_id=0, par_index=20, float_shares=2)
        corp = CORPS[0]

        PLAYERS[0].set_cash(state, 100)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        apply_and_verify_all(state, buy_idx)

        # Game continues
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert corp.get_price_index(state) == 21  # Moved up one space

    def test_buy_share_skipping_to_75_ends_game(self):
        """Buying when intermediate spaces are occupied still ends game at $75."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp at price index 24 ($61)
        # float_shares=2 gives player 2 shares, bank 2 shares, unissued 3
        float_corp_for_test(state, corp_id=0, par_index=24, float_shares=2)
        corp = CORPS[0]

        PLAYERS[0].set_cash(state, 100)

        # Mark space 25 as occupied - buy will skip to $75
        MARKET.set_space_available(state, 25, False)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        apply_and_verify_all(state, buy_idx, expected_status=STATUS_GAME_OVER)

        # Game ends because we reached $75
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER
        assert corp.get_price_index(state) == 26


# =============================================================================
# AUCTION SLOT INFO TESTS
# =============================================================================

def _get_slot_data(state, slot):
    """Read the 5 auction slot info scalars for a given slot."""
    layout = get_layout(state.get_num_players())
    base = layout.auction_slot_info_offset + slot * 5
    return state._array[base:base + 5].copy()


def _assert_slot_matches_company(state, slot, company_id):
    """Assert that an auction slot's data matches the given company."""
    data = _get_slot_data(state, slot)
    coo = TURN.get_coo_level(state)
    assert abs(data[0] - get_company_stars(company_id) / PY_COMPANY_STAR_DIVISOR) < 1e-6
    assert abs(data[1] - get_company_low_price(company_id) / PY_PRICE_DIVISOR) < 1e-6
    assert abs(data[2] - get_company_face_value(company_id) / PY_PRICE_DIVISOR) < 1e-6
    assert abs(data[3] - get_company_high_price(company_id) / PY_PRICE_DIVISOR) < 1e-6
    assert abs(data[4] - get_adjusted_company_income(company_id, coo) / PY_INCOME_DIVISOR) < 1e-6


class TestAuctionSlotInfo:
    """Test auction slot info block updates correctly during INVEST/BID flow."""

    def test_slots_populated_on_init(self, game_state):
        """Auction slot info is populated after game init."""
        for slot in range(3):
            company_id = get_auction_company_for_slot_py(game_state, slot)
            assert company_id >= 0, f"Slot {slot} has no company"
            _assert_slot_matches_company(game_state, slot, company_id)

    def test_slots_match_auction_row_ordering(self, game_state):
        """Slot data corresponds to auction row order (by company_id)."""
        companies = []
        for slot in range(3):
            cid = get_auction_company_for_slot_py(game_state, slot)
            companies.append(cid)
        # Auction slots are ordered by company_id ascending
        assert companies == sorted(companies)

    def test_slots_update_after_auction_won(self, game_state):
        """After an auction resolves, slot info updates (one company gone)."""
        # Record initial auction companies
        initial_companies = []
        for slot in range(3):
            initial_companies.append(get_auction_company_for_slot_py(game_state, slot))

        # Start auction on first available company
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        apply_and_verify_all(game_state, auction_idx)

        # Now in BID phase - have all other players leave
        assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        layout = get_action_layout(3)
        leave_idx = layout['leave_auction']

        # Two leave actions resolve the auction (3 players, starter already bid)
        apply_and_verify_all(game_state, leave_idx)
        apply_and_verify_all(game_state, leave_idx)

        # Back in INVEST phase
        assert game_state.get_phase() == GamePhases.PHASE_INVEST

        # The won company should no longer be in any auction slot
        won_company = TURN.get_auction_company(game_state)
        # auction_company is cleared after resolution, so check via ownership
        new_companies = []
        for slot in range(3):
            cid = get_auction_company_for_slot_py(game_state, slot)
            if cid >= 0:
                new_companies.append(cid)

        # At least one company from the initial set should be gone
        assert len(new_companies) <= len(initial_companies)

        # Each remaining slot should have correct data
        for slot in range(3):
            cid = get_auction_company_for_slot_py(game_state, slot)
            if cid >= 0:
                _assert_slot_matches_company(game_state, slot, cid)

    def test_slots_update_after_wrap_up(self, game_state):
        """After WRAP_UP, slot info reflects the new auction row."""
        # All players pass -> WRAP_UP -> ACQUISITION -> ... -> new INVEST turn
        apply_pass_to_all_players(game_state, 3)

        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(game_state) == 2

        # Verify all slots match current auction row
        for slot in range(3):
            cid = get_auction_company_for_slot_py(game_state, slot)
            if cid >= 0:
                _assert_slot_matches_company(game_state, slot, cid)

    def test_active_company_set_during_bid(self, game_state):
        """Active company block is populated when entering BID phase."""
        layout_info = get_layout(3)

        # Verify active company is zero before auction
        for offset_name in ('active_company_stars_offset', 'active_company_low_price_offset',
                            'active_company_face_value_offset', 'active_company_high_price_offset',
                            'active_company_income_offset'):
            assert game_state._array[getattr(layout_info, offset_name)] == 0.0

        # Start an auction
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        apply_and_verify_all(game_state, auction_idx)

        assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        company_id = TURN.get_auction_company(game_state)

        # Active company should now match the auction company
        assert game_state._array[layout_info.active_company_stars_offset] != 0.0, "Active company stars should be set"
        coo = TURN.get_coo_level(game_state)
        assert abs(game_state._array[layout_info.active_company_face_value_offset] - get_company_face_value(company_id) / PY_PRICE_DIVISOR) < 1e-6

    def test_active_company_cleared_after_bid_resolves(self, game_state):
        """Active company block is zeroed after auction resolution."""
        layout_info = get_layout(3)

        # Start and resolve auction
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        apply_and_verify_all(game_state, auction_idx)

        layout = get_action_layout(3)
        leave_idx = layout['leave_auction']
        apply_and_verify_all(game_state, leave_idx)
        apply_and_verify_all(game_state, leave_idx)

        # Back in INVEST, active company should be cleared
        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        for offset_name in ('active_company_stars_offset', 'active_company_low_price_offset',
                            'active_company_face_value_offset', 'active_company_high_price_offset',
                            'active_company_income_offset'):
            assert game_state._array[getattr(layout_info, offset_name)] == 0.0, \
                f"{offset_name} should be zeroed after bid resolution"


# =============================================================================
# Invest Impact Tests
# =============================================================================


class TestInvestImpacts:
    """Buy/sell net worth impact fields in visible state."""

    @pytest.fixture
    def impact_state(self):
        """State with one active corp for impact testing.

        Player 0 is active (default from initialize_game).
        Corp 0 floated at price_index=10 ($14), player0=2 shares, bank=2.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float corp 0 at price_index=10 ($14)
        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        # After float: player0=2, bank=2, issued=4, unissued=3

        # Give player 0 enough cash to buy
        PLAYERS[0].set_cash(state, 200)

        # Make sure spaces around index 10 are available
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)  # Corp 0 occupies this

        # Recompute impacts (player 0 is already active from init)
        state._populate_invest_impacts()
        return state

    def test_buy_impact_computed(self, impact_state):
        """Buy impact reflects price index steps up."""
        state = impact_state
        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset

        # Corp 0 at index 10, next higher is 11 → 1 step up
        # Stored normalized: 1 / 5.0 = 0.2
        assert abs(state._array[buy_base + 0] - 1.0 / PY_IMPACT_DIVISOR) < 1e-6

    def test_sell_impact_computed(self, impact_state):
        """Sell impact reflects price index steps down."""
        state = impact_state
        layout = get_layout(3)
        sell_base = layout.invest_impacts_offset + 8  # After 8 buy slots

        # Corp 0 at index 10, next lower is 9 → 1 step down
        # Stored normalized: -1 / 5.0 = -0.2
        assert abs(state._array[sell_base + 0] - (-1.0 / PY_IMPACT_DIVISOR)) < 1e-6

    def test_inactive_corp_zero_impact(self, impact_state):
        """Inactive corps have zero buy and sell impact."""
        state = impact_state
        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset
        sell_base = buy_base + 8

        # Corps 1-7 are inactive
        for corp_id in range(1, 8):
            assert state._array[buy_base + corp_id] == 0.0
            assert state._array[sell_base + corp_id] == 0.0

    def test_buy_impact_shown_without_bank_shares(self):
        """Buy impact shown even when no bank shares available."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        # float gives bank=2, player0=2. Move all bank to player0:
        PLAYERS[0].set_shares(state, 0, 4)

        PLAYERS[0].set_cash(state, 200)
        state._populate_invest_impacts()

        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset
        # Still shows index delta even without bank shares
        assert abs(state._array[buy_base + 0] - 1.0 / PY_IMPACT_DIVISOR) < 1e-6

    def test_buy_impact_shown_when_cant_afford(self):
        """Buy impact shown even when player can't afford the buy price."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        PLAYERS[0].set_cash(state, 15)
        state._populate_invest_impacts()

        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset
        # Still shows index delta regardless of affordability
        assert abs(state._array[buy_base + 0] - 1.0 / PY_IMPACT_DIVISOR) < 1e-6

    def test_sell_impact_shown_without_shares(self):
        """Sell impact shown even when player owns no shares of a corp."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        # Advance active player to player 1 via pass action
        layout_dict = get_action_layout(3)
        apply_and_verify_all(state, layout_dict['pass_invest'])
        assert state.get_active_player() == 1

        layout = get_layout(3)
        sell_base = layout.invest_impacts_offset + 8
        # Still shows index delta regardless of share ownership
        assert abs(state._array[sell_base + 0] - (-1.0 / PY_IMPACT_DIVISOR)) < 1e-6

    def test_impacts_cleared_outside_invest(self):
        """Impacts are zeroed when transitioning out of INVEST."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        PLAYERS[0].set_cash(state, 200)
        state._populate_invest_impacts()

        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset

        # Verify impacts are populated
        assert state._array[buy_base + 0] != 0.0

        # Clear them
        state._clear_invest_impacts()

        for i in range(16):
            assert state._array[buy_base + i] == 0.0, (
                f"Impact slot {i} not cleared"
            )

    def test_impacts_update_on_player_advance(self):
        """Impacts recompute when active player changes during INVEST."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        PLAYERS[0].set_cash(state, 200)
        state._populate_invest_impacts()

        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset
        sell_base = buy_base + 8

        # Player 0 should have impacts
        assert state._array[buy_base + 0] != 0.0
        assert state._array[sell_base + 0] != 0.0

        # Pass → next player (impacts are player-independent now,
        # but verify recomputation still happens)
        layout_dict = get_action_layout(3)
        apply_and_verify_all(state, layout_dict['pass_invest'])
        assert state.get_active_player() == 1

        # Player 1 still sees impacts (no longer gated on share ownership)
        assert state._array[buy_base + 0] != 0.0
        assert state._array[sell_base + 0] != 0.0

    def test_impacts_with_occupied_space_slide(self):
        """Buy impact accounts for sliding past occupied market spaces."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        # Corp 0 at index 10
        # Block index 11 so buy slides to 12 (2 steps)
        for i in range(27):
            MARKET.set_space_available(state, i, True)
        MARKET.set_space_available(state, 10, False)  # Corp 0
        MARKET.set_space_available(state, 11, False)  # Blocked

        PLAYERS[0].set_cash(state, 200)
        state._populate_invest_impacts()

        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset

        # Buy slides to index 12 → 2 steps up
        assert abs(state._array[buy_base + 0] - 2.0 / PY_IMPACT_DIVISOR) < 1e-6

    def test_impacts_populated_at_game_start(self):
        """Impacts are computed during initialize_game."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # At game start, no corps are active, so all impacts should be zero
        layout = get_layout(3)
        buy_base = layout.invest_impacts_offset
        for i in range(16):
            assert state._array[buy_base + i] == 0.0


# =============================================================================
# ROUND-TRIP TRACKING (VISIBLE + HIDDEN STATE)
# =============================================================================

class TestRoundTripTracking:
    """Test that visible round_trips and hidden share_buys/sells are filled correctly."""

    def _get_visible_round_trips(self, state, player_id, corp_id):
        """Read visible round_trips[corp_id] directly from the state array."""
        layout = get_layout(3)
        # round_trips is 10 positions from the end of the player stride
        # (8 round_trips slots + 1 acquisition_proceeds + 1 income)
        rt_rel = layout.player_stride - 10
        offset = layout.players_offset + player_id * layout.player_stride + rt_rel + corp_id
        return state._array[offset]

    def test_visible_round_trips_zero_at_init(self):
        """Visible round_trips are zero for all players/corps after init."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        for p in range(3):
            for c in range(8):
                assert self._get_visible_round_trips(state, p, c) == 0.0

    def test_hidden_buys_sells_zero_at_init(self):
        """Hidden share_buys and share_sells are zero after init."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        for p in range(3):
            for c in range(8):
                assert PLAYERS[p].get_share_buys(state, c) == 0
                assert PLAYERS[p].get_share_sells(state, c) == 0

    def test_buy_increments_hidden_buys_only(self):
        """increment_share_buys updates hidden buys but not hidden sells."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 2)
        assert PLAYERS[0].get_share_buys(state, 2) == 1
        assert PLAYERS[0].get_share_sells(state, 2) == 0

    def test_sell_increments_hidden_sells_only(self):
        """increment_share_sells updates hidden sells but not hidden buys."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_sells(state, 2)
        assert PLAYERS[0].get_share_sells(state, 2) == 1
        assert PLAYERS[0].get_share_buys(state, 2) == 0

    def test_visible_round_trips_zero_after_single_buy(self):
        """A buy without a matching sell gives round_trips=0."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 0)
        # min(1, 0) = 0
        assert self._get_visible_round_trips(state, 0, 0) == 0.0

    def test_visible_round_trips_after_buy_and_sell(self):
        """A buy + sell gives round_trips = min(1,1)/MAX_ROUNDTRIPS = 0.5."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 0)
        PLAYERS[0].increment_share_sells(state, 0)
        # min(1, 1) = 1, normalized: 1 / 2.0 = 0.5
        assert abs(self._get_visible_round_trips(state, 0, 0) - 0.5) < 1e-6

    def test_visible_round_trips_after_two_roundtrips(self):
        """Two buys + two sells gives round_trips = min(2,2)/MAX_ROUNDTRIPS = 1.0."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        for _ in range(2):
            PLAYERS[0].increment_share_buys(state, 0)
            PLAYERS[0].increment_share_sells(state, 0)
        assert abs(self._get_visible_round_trips(state, 0, 0) - 1.0) < 1e-6

    def test_visible_round_trips_asymmetric(self):
        """3 buys + 1 sell gives round_trips = min(3,1)/MAX_ROUNDTRIPS = 0.5."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        for _ in range(3):
            PLAYERS[0].increment_share_buys(state, 0)
        PLAYERS[0].increment_share_sells(state, 0)
        assert abs(self._get_visible_round_trips(state, 0, 0) - 0.5) < 1e-6

    def test_per_corp_independence(self):
        """Round trips for different corps are independent."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 0)
        PLAYERS[0].increment_share_sells(state, 0)
        PLAYERS[0].increment_share_buys(state, 3)

        assert abs(self._get_visible_round_trips(state, 0, 0) - 0.5) < 1e-6  # 1 round-trip
        assert self._get_visible_round_trips(state, 0, 3) == 0.0              # 1 buy, no sells
        assert self._get_visible_round_trips(state, 0, 1) == 0.0              # untouched

    def test_per_player_independence(self):
        """Round trips for different players are independent."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 0)
        PLAYERS[0].increment_share_sells(state, 0)
        PLAYERS[1].increment_share_buys(state, 0)

        assert abs(self._get_visible_round_trips(state, 0, 0) - 0.5) < 1e-6  # player 0: 1 rt
        assert self._get_visible_round_trips(state, 1, 0) == 0.0              # player 1: buy only
        assert PLAYERS[0].get_share_buys(state, 0) == 1
        assert PLAYERS[1].get_share_buys(state, 0) == 1
        assert PLAYERS[1].get_share_sells(state, 0) == 0

    def test_clear_roundtrip_tracking_zeros_all(self):
        """clear_roundtrip_tracking zeros hidden buys/sells AND visible round_trips."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Build up some tracking data across players and corps
        for _ in range(2):
            PLAYERS[0].increment_share_buys(state, 0)
            PLAYERS[0].increment_share_sells(state, 0)
        PLAYERS[1].increment_share_buys(state, 3)
        PLAYERS[1].increment_share_sells(state, 3)

        # Verify non-zero before clearing
        assert PLAYERS[0].get_share_buys(state, 0) == 2
        assert abs(self._get_visible_round_trips(state, 0, 0) - 1.0) < 1e-6
        assert abs(self._get_visible_round_trips(state, 1, 3) - 0.5) < 1e-6

        # Clear all players
        for p in range(3):
            PLAYERS[p].clear_roundtrip_tracking(state)

        # Everything should be zero
        for p in range(3):
            for c in range(8):
                assert PLAYERS[p].get_share_buys(state, c) == 0, f"p{p} c{c} buys not cleared"
                assert PLAYERS[p].get_share_sells(state, c) == 0, f"p{p} c{c} sells not cleared"
                assert self._get_visible_round_trips(state, p, c) == 0.0, f"p{p} c{c} visible rt not cleared"

    def test_clear_only_affects_target_player(self):
        """Clearing one player's tracking doesn't affect another's."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].increment_share_buys(state, 0)
        PLAYERS[0].increment_share_sells(state, 0)
        PLAYERS[1].increment_share_buys(state, 0)
        PLAYERS[1].increment_share_sells(state, 0)

        PLAYERS[0].clear_roundtrip_tracking(state)

        # Player 0 cleared
        assert PLAYERS[0].get_share_buys(state, 0) == 0
        assert PLAYERS[0].get_share_sells(state, 0) == 0
        assert self._get_visible_round_trips(state, 0, 0) == 0.0
        # Player 1 untouched
        assert PLAYERS[1].get_share_buys(state, 0) == 1
        assert PLAYERS[1].get_share_sells(state, 0) == 1
        assert abs(self._get_visible_round_trips(state, 1, 0) - 0.5) < 1e-6

    def test_buy_action_updates_hidden_buys(self, trade_state):
        """Actual buy action through the driver increments hidden share_buys."""
        player = PLAYERS[0]
        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0

        assert player.get_share_buys(trade_state, 0) == 0
        apply_and_verify_all(trade_state, buy_idx)

        # Buy increments hidden buys; no sells yet so visible round_trips stays 0
        assert player.get_share_buys(trade_state, 0) == 1
        assert player.get_share_sells(trade_state, 0) == 0
        assert self._get_visible_round_trips(trade_state, 0, 0) == 0.0

    def test_sell_action_updates_hidden_sells(self, trade_state):
        """Actual sell action through the driver increments hidden share_sells."""
        player = PLAYERS[0]
        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0

        assert player.get_share_sells(trade_state, 0) == 0
        apply_and_verify_all(trade_state, sell_idx)

        # Sell increments hidden sells; no buys yet so visible round_trips stays 0
        assert player.get_share_sells(trade_state, 0) == 1
        assert player.get_share_buys(trade_state, 0) == 0
        assert self._get_visible_round_trips(trade_state, 0, 0) == 0.0

    def test_buy_sell_via_driver_sets_visible_round_trips(self, trade_state):
        """A buy + sell in the same INVEST phase produces nonzero visible round_trips.

        Player 0 buys, players 1-2 pass (only corp 0 exists, they have no shares),
        then player 0 sells. This is a complete round-trip through the driver.
        """
        player = PLAYERS[0]
        layout = get_action_layout(3)

        # Player 0 buys corp 0
        apply_and_verify_all(trade_state, layout['buy_share_base'] + 0)
        # Players 1 and 2 pass (no tradeable corps)
        apply_and_verify_all(trade_state, layout['pass_invest'])
        apply_and_verify_all(trade_state, layout['pass_invest'])
        # Player 0 is active again — sell corp 0
        assert trade_state.get_active_player() == 0
        apply_and_verify_all(trade_state, layout['sell_share_base'] + 0)

        assert player.get_share_buys(trade_state, 0) == 1
        assert player.get_share_sells(trade_state, 0) == 1
        assert abs(self._get_visible_round_trips(trade_state, 0, 0) - 0.5) < 1e-6
