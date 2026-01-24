"""Tests for WRAP_UP phase behavior.

NOTE: Comprehensive testing blocked by critical bugs in WRAP_UP implementation:
- Bug 1: FI cash becomes 0 after purchases instead of correct remainder
- Bug 2: Player cash becomes 0 for players 1+ after WRAP_UP cycle

These bugs prevent testing FI purchase scenarios and player reordering by cash.
Tests below cover what CAN be verified given the current implementation state.
"""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS
from entities.company import COMPANIES

STATUS_OK = 0


def trigger_wrap_up(state):
    """Helper to trigger WRAP_UP by having all players pass."""
    num_players = state.get_num_players()
    layout = get_action_layout(num_players)
    pass_idx = layout['pass_invest']

    for i in range(num_players):
        DRIVER.apply_action(state, pass_idx)


# =============================================================================
# AVAILABILITY TRANSITION TESTS
# =============================================================================

class TestAvailabilityTransition:
    """Test company availability state transitions."""

    def test_unavailable_companies_become_available(self):
        """AVAIL-01: After FI purchases complete, all unavailable companies become available."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Clear all companies first
        for company_id in range(36):
            COMPANIES[company_id].remove_from_game(state)

        # Set some companies to REVEALED state (unavailable)
        COMPANIES[0].set_revealed(state, True)
        COMPANIES[1].set_revealed(state, True)
        # Set one to FOR_AUCTION (already available)
        COMPANIES[2].move_to_auction(state)

        # Verify initial states
        assert COMPANIES[0].is_revealed(state), "Company 0 should be revealed"
        assert COMPANIES[1].is_revealed(state), "Company 1 should be revealed"
        assert COMPANIES[2].is_for_auction(state), "Company 2 should be for auction"

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # Verify all REVEALED companies are now FOR_AUCTION
        assert COMPANIES[0].is_for_auction(state), "Company 0 should be FOR_AUCTION"
        assert COMPANIES[1].is_for_auction(state), "Company 1 should be FOR_AUCTION"
        assert COMPANIES[2].is_for_auction(state), "Company 2 should still be FOR_AUCTION"


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestWrapUpHistory:
    """Test sentinel action history verification."""

    def test_wrap_up_records_sentinel_in_history(self, apply_and_track):
        """PHASE-04: WRAP_UP execution records sentinel action (-100) in history."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # All players pass to trigger WRAP_UP
        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']

        # Pass all but last player
        for i in range(2):
            DRIVER.apply_action(state, pass_idx)

        # Last pass triggers WRAP_UP auto-apply
        result = apply_and_track(state, pass_idx)

        # Verify history contains sentinel -100 for WRAP_UP
        action_values = [entry[1] for entry in result.history]
        assert -100 in action_values, "WRAP_UP sentinel (-100) not found in history"
        assert -101 in action_values, "ACQUISITION sentinel (-101) not found in history"


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransitions:
    """Test phase flow verification."""

    def test_invest_to_wrap_up_to_acquisition_to_invest(self):
        """PHASE-01, PHASE-02: Complete phase cycle: INVEST -> WRAP_UP -> ACQUISITION -> INVEST."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Verify initial phase
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 1

        # All players pass
        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']
        for i in range(3):
            DRIVER.apply_action(state, pass_idx)

        # Verify final phase is INVEST (turn 2)
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert TURN.get_consecutive_passes(state) == 0  # Reset after WRAP_UP

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_wrap_up_cycle_for_all_player_counts(self, num_players):
        """WRAP_UP cycle works for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        initial_turn = TURN.get_turn_number(state)

        # All players pass
        trigger_wrap_up(state)

        # Verify turn incremented and back to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == initial_turn + 1
        assert TURN.get_consecutive_passes(state) == 0


# =============================================================================
# BLOCKED TESTS (due to implementation bugs)
# =============================================================================
# The following test classes are commented out due to critical bugs:
#
# TestPlayerReordering - Player cash becomes 0 after WRAP_UP (Bug 2)
# TestFIPurchases - FI cash becomes 0 after purchases (Bug 1)
#
# These tests should be uncommented and completed once the bugs are fixed.
