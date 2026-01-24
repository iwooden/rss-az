"""Integration tests verifying invariants throughout multi-action sequences.

This file consolidates integration-style tests from per-phase test files.
Tests here verify that invariants hold across phase transitions and
multi-action sequences, not just individual action correctness.

Add new integration tests here as phases are implemented.
"""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN


# =============================================================================
# INVEST PHASE INTEGRATION
# =============================================================================

class TestInvestIntegration:
    """Integration tests verifying invariants after every action."""

    def test_multiple_passes_maintains_invariants(self, game_state):
        """Multiple pass actions maintain game invariants throughout."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(game_state, "Initial state")

        layout = get_action_layout(3)
        # Pass twice (not enough for WRAP_UP)
        for i in range(2):
            apply_action_and_verify(game_state, layout['pass_invest'], f"Pass {i+1}")

        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(game_state, "After two passes")

    def test_auction_cycle_maintains_invariants(self, game_state):
        """Starting auction and completing it maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants, assert_valid_mask

        assert_invariants(game_state, "Initial state")
        assert_valid_mask(game_state, msg="Initial mask valid")

        # Find and start auction
        mask = get_valid_action_mask(game_state)
        layout = get_action_layout(3)
        auction_idx = None
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                auction_idx = i
                break

        if auction_idx is not None:
            apply_action_and_verify(game_state, auction_idx, "Start auction")
            assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
            assert_invariants(game_state, "After auction start")

    def test_buy_share_maintains_invariants(self, trade_state):
        """Buy share action maintains all game invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(trade_state, "Initial trade state")

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_action_and_verify(trade_state, buy_idx, "Buy share")

        assert_invariants(trade_state, "After buy share")

    def test_sell_share_maintains_invariants(self, trade_state):
        """Sell share action maintains all game invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(trade_state, "Initial trade state")

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_action_and_verify(trade_state, sell_idx, "Sell share")

        assert_invariants(trade_state, "After sell share")

    def test_multiple_trades_maintain_invariants(self, trade_state):
        """Multiple buy actions in sequence maintain invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        layout = get_action_layout(3)

        assert_invariants(trade_state, "Initial state")

        # Buy twice (trade_state has 2 bank shares available)
        buy_idx = layout['buy_share_base'] + 0

        apply_action_and_verify(trade_state, buy_idx, "First buy")
        apply_action_and_verify(trade_state, buy_idx, "Second buy")

        assert_invariants(trade_state, "After two buys")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_wrap_up_transition_maintains_invariants(self, num_players):
        """Phase transition through WRAP_UP maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        layout = get_action_layout(num_players)
        # Apply all passes with verify helper
        for i in range(num_players):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")

        # After all passes, should be back in INVEST (after WRAP_UP -> ACQUISITION)
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After WRAP_UP -> ACQUISITION -> INVEST transition")


# =============================================================================
# BID PHASE INTEGRATION
# =============================================================================

class TestBidIntegration:
    """Integration tests verifying invariants throughout auction cycles."""

    def test_full_auction_maintains_invariants(self):
        """Complete auction cycle (start -> bids -> resolution) maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants, assert_valid_mask

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")
        assert_valid_mask(state, msg="Initial INVEST mask")

        layout = get_action_layout(3)

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Start auction")
                break

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        assert_invariants(state, "After auction start")
        assert_valid_mask(state, msg="BID phase mask")

        # First player leaves
        apply_action_and_verify(state, layout['leave_auction'], "First leave")
        assert_invariants(state, "After first leave")

        # Second player leaves - triggers resolution
        if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action_and_verify(state, layout['leave_auction'], "Second leave")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(state, "After auction resolution")
        assert_valid_mask(state, msg="Post-auction INVEST mask")

    def test_auction_with_raises_maintains_invariants(self):
        """Auction with multiple raise bids maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Start auction")
                break

        assert_invariants(state, "After auction start")

        # First player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "First raise")
                break

        assert_invariants(state, "After first raise")

        # Second player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Second raise")
                break

        assert_invariants(state, "After second raise")

        # Resolve via leaves
        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action_and_verify(state, layout['leave_auction'], "Leave to resolve")

        assert_invariants(state, "After resolution with raises")

    def test_multiple_auctions_maintain_invariants(self):
        """Multiple auction cycles in sequence maintain invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        for auction_num in range(2):  # Two auction cycles
            assert_invariants(state, f"Before auction {auction_num + 1}")

            # Start auction
            mask = get_valid_action_mask(state)
            auction_started = False
            for i in range(layout['auction_base'], layout['buy_share_base']):
                if mask[i] == 1.0:
                    apply_action_and_verify(state, i, f"Start auction {auction_num + 1}")
                    auction_started = True
                    break

            if not auction_started:
                break  # No more auctions available

            # Complete auction via leaves
            while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action_and_verify(state, layout['leave_auction'], f"Leave auction {auction_num + 1}")

            assert_invariants(state, f"After auction {auction_num + 1}")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_auction_maintains_invariants_all_player_counts(self, num_players):
        """Auction cycle maintains invariants for all player counts."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        layout = get_action_layout(num_players)

        assert_invariants(state, f"Initial {num_players}p")

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, f"Start auction {num_players}p")
                break

        # All but one leave
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action_and_verify(state, layout['leave_auction'], f"Leave {num_players}p")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(state, f"After auction {num_players}p")
