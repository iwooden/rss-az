"""Tests for ISSUE_SHARES phase (Phase 8)."""
import pytest
from core.state import get_layout
from core.data import GamePhases, CorpIndices, get_market_price, PY_PRICE_DIVISOR, PY_IMPACT_DIVISOR
from core.actions import get_valid_action_mask, get_action_layout
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from phases.issue import (
    setup_issue_phase_py,
    apply_issue_action_py,
    find_next_issue_corp_py,
)
from tests.phases.conftest import float_corp_for_test, assert_invariants


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def issue_state(game_state):
    """
    3-player game state set up at PHASE_ISSUE_SHARES.

    By default:
    - No active corps (phase will auto-transition to IPO then INVEST)
    """
    TURN.set_phase(game_state, GamePhases.PHASE_ISSUE_SHARES)
    return game_state


@pytest.fixture
def issue_state_with_corp(game_state):
    """
    3-player game with one active corp ready for issue phase.

    Corp 0 (JS) is active with:
    - price_index = 15 (price $24)
    - unissued_shares = 3, issued = 4, player = 2, bank = 2
    - cash = 50
    - Player 0 is president
    """
    # Float corp 0 at price index 15 with 2 shares each to player/bank
    float_corp_for_test(game_state, corp_id=0, par_index=15, float_shares=2)
    CORPS[0].set_cash(game_state, 50)

    TURN.set_phase(game_state, GamePhases.PHASE_ISSUE_SHARES)

    return game_state


# =============================================================================
# Basic Issue Mechanics
# =============================================================================


class TestBasicIssueMechanics:
    """Basic issue mechanics (shares transfer, cash flow)."""

    def test_issue_transfers_one_share(self, issue_state_with_corp):
        """Issuing transfers one share from unissued to issued and bank."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_unissued = corp.get_unissued_shares(state)
        initial_issued = corp.get_issued_shares(state)
        initial_bank = corp.get_bank_shares(state)

        # Set up phase and apply issue
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert corp.get_unissued_shares(state) == initial_unissued - 1
        assert corp.get_issued_shares(state) == initial_issued + 1
        assert corp.get_bank_shares(state) == initial_bank + 1

    def test_issue_pays_corp_proceeds(self, issue_state_with_corp):
        """Corp receives cash proceeds from issuing."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_cash = corp.get_cash(state)

        # Set up phase and apply issue
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        # Get the expected new price after drop
        current_index = corp.get_price_index(state)
        new_index = MARKET.find_next_lower_space(state, current_index)
        expected_proceeds = get_market_price(new_index)

        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert corp.get_cash(state) == initial_cash + expected_proceeds

    def test_pass_does_not_transfer_shares(self, issue_state_with_corp):
        """Passing does not transfer shares or change cash."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_unissued = corp.get_unissued_shares(state)
        initial_issued = corp.get_issued_shares(state)
        initial_bank = corp.get_bank_shares(state)
        initial_cash = corp.get_cash(state)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=False)  # Pass
        assert_invariants(state, "After issue pass action")

        assert corp.get_unissued_shares(state) == initial_unissued
        assert corp.get_issued_shares(state) == initial_issued
        assert corp.get_bank_shares(state) == initial_bank
        assert corp.get_cash(state) == initial_cash


# =============================================================================
# Price Movement (Normal Corps)
# =============================================================================


class TestPriceMovement:
    """Price movement for normal corporations."""

    def test_price_drops_on_issue(self, issue_state_with_corp):
        """Normal corp price drops when issuing."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        final_index = corp.get_price_index(state)
        assert final_index < initial_index

    def test_price_drops_to_next_available_space(self, issue_state_with_corp):
        """Price drops to next available (lower) market space."""
        state = issue_state_with_corp
        corp = CORPS[0]

        # Block some spaces to create a gap
        MARKET.set_space_available(state, 14, False)  # Occupied
        MARKET.set_space_available(state, 13, True)   # Available

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        # Should drop to 13 (skipping occupied 14, from 15)
        assert corp.get_price_index(state) == 13

    def test_old_space_freed_after_issue(self, issue_state_with_corp):
        """Old market space becomes available after price drop."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)  # 15
        assert not MARKET.is_space_available(state, initial_index)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert MARKET.is_space_available(state, initial_index)

    def test_new_space_occupied_after_issue(self, issue_state_with_corp):
        """New market space becomes occupied after price drop."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)  # 15
        new_index = MARKET.find_next_lower_space(state, initial_index)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert not MARKET.is_space_available(state, new_index)

    def test_proceeds_equal_new_price(self, issue_state_with_corp):
        """Corp receives the NEW (lower) price as proceeds."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_cash = corp.get_cash(state)
        initial_index = corp.get_price_index(state)
        new_index = MARKET.find_next_lower_space(state, initial_index)
        new_price = get_market_price(new_index)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert corp.get_cash(state) == initial_cash + new_price


# =============================================================================
# Stock Masters Special
# =============================================================================


class TestStockMastersSpecial:
    """Stock Masters (CORP_SM) price does not change."""

    def test_sm_price_unchanged_on_issue(self, game_state):
        """Stock Masters price stays the same when issuing."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]  # Corp 3

        float_corp_for_test(state, corp_id=CorpIndices.CORP_SM, par_index=15)
        corp.set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_index = corp.get_price_index(state)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert corp.get_price_index(state) == initial_index

    def test_sm_receives_current_price(self, game_state):
        """Stock Masters receives current (unchanged) price as proceeds."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]

        float_corp_for_test(state, corp_id=CorpIndices.CORP_SM, par_index=15)
        corp.set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_cash = corp.get_cash(state)
        current_price = get_market_price(15)  # $24

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert corp.get_cash(state) == initial_cash + current_price

    def test_sm_market_space_unchanged(self, game_state):
        """Stock Masters keeps same market space occupied."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]

        float_corp_for_test(state, corp_id=CorpIndices.CORP_SM, par_index=15)
        corp.set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        # Space 15 should still be occupied
        assert not MARKET.is_space_available(state, 15)

    def test_sm_at_price_1_does_not_go_bankrupt(self, game_state):
        """Stock Masters at price index 1 survives issuing (no bankruptcy).

        A normal corp at price index 1 would drop to 0 and go bankrupt.
        SM's special ability prevents the price change, so it stays at
        index 1 ($5), remains active, and receives $5 proceeds.
        """
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]

        # Float SM then manually drop price to index 1
        float_corp_for_test(state, corp_id=CorpIndices.CORP_SM, par_index=10)
        MARKET.set_space_available(state, 10, True)   # Free old space
        MARKET.set_space_available(state, 1, False)    # Occupy new space
        corp.set_price_index(state, 1)
        corp.set_cash(state, 0)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        # SM should survive — still active, price unchanged, received $5
        assert corp.is_active(state), "SM should NOT go bankrupt at price index 1"
        assert corp.get_price_index(state) == 1, "SM price should stay at index 1"
        assert corp.get_cash(state) == get_market_price(1), \
            f"SM should receive ${get_market_price(1)} proceeds"


# =============================================================================
# Processing Order
# =============================================================================


class TestProcessingOrder:
    """Corps processed in descending share price order."""

    def test_higher_price_corp_processed_first(self, game_state):
        """Corp with higher price index is processed before lower."""
        state = game_state

        # Set up two corps at different prices
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)  # Lower price
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15)  # Higher price

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        # Initialize remaining flags
        TURN.set_issue_remaining(state, 0, True)
        TURN.set_issue_remaining(state, 1, True)

        # Find next should return corp 1 (higher price)
        next_corp = find_next_issue_corp_py(state)
        assert next_corp == 1

    def test_three_corps_descending_order(self, game_state):
        """Three corps processed in correct descending price order."""
        state = game_state

        # Set up three corps at different prices
        # Expected order: 1, 2, 0 (prices: 20, 12, 8)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=8)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=20)
        float_corp_for_test(state, corp_id=2, player_id=2, par_index=12)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        # Initialize all remaining
        for corp_id in range(3):
            TURN.set_issue_remaining(state, corp_id, True)

        # Check order
        expected_order = [1, 2, 0]  # Descending price: 20, 12, 8

        for expected_corp in expected_order:
            next_corp = find_next_issue_corp_py(state)
            assert next_corp == expected_corp, f"Expected {expected_corp}, got {next_corp}"
            TURN.set_issue_remaining(state, next_corp, False)


# =============================================================================
# Receivership Handling
# =============================================================================


class TestReceivershipHandling:
    """Receivership corps must issue if they have unissued shares."""

    def test_receivership_with_unissued_auto_issues(self, game_state):
        """Receivership corp with unissued shares auto-issues."""
        state = game_state
        corp = CORPS[0]

        float_corp_for_test(state, corp_id=0, par_index=15)

        # Put into receivership: set_shares auto-adjusts bank shares and triggers receivership
        PLAYERS[0].set_shares(state, 0, 0)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_unissued = corp.get_unissued_shares(state)

        # Setup should auto-process receivership corps
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (receivership auto-issue)")

        # Should have auto-issued (one less unissued)
        assert corp.get_unissued_shares(state) == initial_unissued - 1

    def test_receivership_without_unissued_auto_passes(self, game_state):
        """Receivership corp without unissued shares auto-passes."""
        state = game_state
        corp = CORPS[0]

        float_corp_for_test(state, corp_id=0, par_index=15)

        # All shares issued (no unissued): total=7, all in bank for receivership
        # Set issued/bank first, then set_shares auto-adjusts bank by +1 (player 1->0)
        corp.set_unissued_shares(state, 0)
        corp.set_issued_shares(state, 7)
        corp.set_bank_shares(state, 6)
        corp.set_cash(state, 50)

        # Put into receivership: set_shares auto-adjusts bank shares and triggers receivership
        PLAYERS[0].set_shares(state, 0, 0)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_cash = corp.get_cash(state)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (receivership auto-pass)")

        # Should have auto-passed (cash unchanged)
        assert corp.get_cash(state) == initial_cash

    def test_receivership_skipped_for_player_decision(self, game_state):
        """Player-controlled corp waits for player decision."""
        state = game_state
        corp = CORPS[0]

        float_corp_for_test(state, corp_id=0, par_index=15)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_unissued = corp.get_unissued_shares(state)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (player decision)")

        # Should NOT auto-issue (waiting for player decision)
        assert corp.get_unissued_shares(state) == initial_unissued
        # Issue corp should be set
        assert TURN.get_issue_corp(state) == 0


# =============================================================================
# Action Mask Validation
# =============================================================================


class TestActionMaskValidation:
    """Action mask validation for ISSUE_SHARES phase."""

    def test_mask_has_issue_and_pass_with_unissued(self, issue_state_with_corp):
        """Mask includes both ISSUE and PASS when corp has unissued shares."""
        state = issue_state_with_corp

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['issue_action']] == 1.0
        assert mask[layout['issue_pass']] == 1.0

    def test_mask_only_pass_without_unissued(self, game_state):
        """Mask only has PASS when corp has no unissued shares."""
        state = game_state
        corp = CORPS[0]

        float_corp_for_test(state, corp_id=0, par_index=15)

        # All shares issued (no unissued): total=7, player has 1, bank has 6
        corp.set_unissued_shares(state, 0)
        corp.set_issued_shares(state, 7)
        corp.set_bank_shares(state, 6)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['issue_action']] == 0.0  # Can't issue
        assert mask[layout['issue_pass']] == 1.0    # Can pass

    def test_no_actions_when_phase_complete(self, issue_state):
        """No valid actions when no corps remain to process."""
        state = issue_state

        # No active corps -> setup transitions to IPO then INVEST
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (no active corps)")

        # Phase should have transitioned
        assert TURN.get_phase(state) != GamePhases.PHASE_ISSUE_SHARES


# =============================================================================
# Phase Transitions
# =============================================================================


class TestPhaseTransitions:
    """Phase transitions for ISSUE_SHARES."""

    def test_transitions_to_invest_when_done(self, issue_state):
        """Transitions to INVEST (via IPO) when all corps processed."""
        state = issue_state

        # No active corps
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (transitions to INVEST)")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_single_corp_transitions_after_action(self, issue_state_with_corp):
        """Single corp transitions to INVEST (via IPO) after action."""
        state = issue_state_with_corp

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES

        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After issue action")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_multiple_corps_process_all_before_transition(self, game_state):
        """Multiple corps all processed before transitioning."""
        state = game_state

        # Set up two corps at different prices
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        # First corp (higher price index = 15)
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        apply_issue_action_py(state, issue=False)  # Pass
        assert_invariants(state, "After first corp pass")

        # Second corp
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        apply_issue_action_py(state, issue=False)  # Pass
        assert_invariants(state, "After second corp pass")

        # Now should transition to INVEST (via IPO)
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_setup_from_end_card(self, game_state):
        """Issue phase can be set up from END_CARD transition."""
        state = game_state

        float_corp_for_test(state, corp_id=0, par_index=15)

        # Simulate END_CARD -> ISSUE_SHARES transition
        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (from END_CARD)")

        # Should be ready for player action
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        assert TURN.get_issue_corp(state) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for ISSUE_SHARES phase."""

    def test_mixed_receivership_and_player_corps(self, game_state):
        """Mix of receivership and player-controlled corps processed correctly."""
        state = game_state

        # Corp 0: Player-controlled, highest price
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=20)

        # Corp 1: Receivership, middle price (set_shares auto-adjusts bank and triggers receivership)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15)
        PLAYERS[1].set_shares(state, 1, 0)

        # Corp 2: Player-controlled, lowest price
        float_corp_for_test(state, corp_id=2, player_id=2, par_index=10)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        corp1_initial_unissued = CORPS[1].get_unissued_shares(state)

        # Setup finds highest-price corp first
        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase (mixed corps)")

        # Corp 1 (receivership) NOT yet processed - Corp 0 (highest price) is first
        assert CORPS[1].get_unissued_shares(state) == corp1_initial_unissued

        # Should be waiting on corp 0 (highest price player-controlled)
        assert TURN.get_issue_corp(state) == 0

        apply_issue_action_py(state, issue=False)  # Pass on corp 0
        assert_invariants(state, "After pass on corp 0")

        # Now corp 1 (receivership) should have auto-issued during advance
        # And we should be on corp 2
        assert CORPS[1].get_unissued_shares(state) == corp1_initial_unissued - 1
        assert TURN.get_issue_corp(state) == 2

        apply_issue_action_py(state, issue=False)  # Pass on corp 2
        assert_invariants(state, "After pass on corp 2")

        # Should transition to INVEST (via IPO)
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_all_corps_pass_no_share_changes(self, game_state):
        """All corps passing results in no share changes."""
        state = game_state

        # Set up two corps at different prices
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15)

        initial_shares = {}
        for corp_id in range(2):
            corp = CORPS[corp_id]
            initial_shares[corp_id] = {
                'unissued': corp.get_unissued_shares(state),
                'issued': corp.get_issued_shares(state),
                'bank': corp.get_bank_shares(state),
            }

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        # Pass on both
        apply_issue_action_py(state, issue=False)
        assert_invariants(state, "After first corp pass")
        apply_issue_action_py(state, issue=False)
        assert_invariants(state, "After second corp pass")

        # Verify no share changes
        for corp_id in range(2):
            corp = CORPS[corp_id]
            assert corp.get_unissued_shares(state) == initial_shares[corp_id]['unissued']
            assert corp.get_issued_shares(state) == initial_shares[corp_id]['issued']
            assert corp.get_bank_shares(state) == initial_shares[corp_id]['bank']

    def test_all_corps_issue_share_counts_correct(self, game_state):
        """All corps issuing results in correct share counts."""
        state = game_state

        # Set up two corps at different prices with specific share counts
        # float_shares=2 gives: player=2, bank=2, issued=4, unissued=3 (total=7)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10, float_shares=2)
        float_corp_for_test(state, corp_id=1, player_id=1, par_index=15, float_shares=2)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        assert_invariants(state, "After setup_issue_phase")

        # Issue on both
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After first corp issue")
        apply_issue_action_py(state, issue=True)
        assert_invariants(state, "After second corp issue")

        # Verify share changes (unissued -1, issued +1, bank +1)
        for corp_id in range(2):
            corp = CORPS[corp_id]
            # Both corps have 7 shares: float_shares=2 gives issued=4, unissued=3, bank=2
            assert corp.get_unissued_shares(state) == 2  # 3 - 1
            assert corp.get_issued_shares(state) == 5    # 4 + 1
            assert corp.get_bank_shares(state) == 3      # 2 + 1


class TestActiveCorpIssue:
    """Test active corp block during ISSUE phase."""

    @pytest.fixture
    def issue_state(self, game_state):
        """State with a floated corp ready for issue."""
        float_corp_for_test(game_state, corp_id=0, par_index=10, player_id=0)
        TURN.set_phase(game_state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(game_state)
        return game_state

    def test_active_corp_set_during_issue(self, issue_state):
        """Active corp info should be populated when issue corp is set."""
        layout = get_layout(3)
        corp_id = TURN.get_issue_corp(issue_state)
        assert corp_id >= 0, "Expected an issue corp to be set"

        oh_base = layout.active_corp_offset
        assert issue_state._array[oh_base + corp_id] == 1.0

        corp_offset = layout.corps_offset + corp_id * layout.corp_stride
        assert abs(issue_state._array[layout.active_corp_income_offset] - issue_state._array[corp_offset + 5]) < 1e-6  # income
        assert abs(issue_state._array[layout.active_corp_stars_offset] - issue_state._array[corp_offset + 6]) < 1e-6  # stars
        assert abs(issue_state._array[layout.active_corp_share_price_offset] - issue_state._array[corp_offset + 7]) < 1e-6  # share_price

    def test_active_corp_cleared_after_issue(self, issue_state):
        """Active corp should be cleared when transitioning out of ISSUE."""
        layout = get_layout(3)
        # Pass on issue to advance
        apply_issue_action_py(issue_state, issue=False)

        phase = TURN.get_phase(issue_state)
        if phase != GamePhases.PHASE_ISSUE_SHARES:
            for offset_name in ('active_corp_income_offset', 'active_corp_stars_offset',
                                'active_corp_share_price_offset'):
                assert issue_state._array[getattr(layout, offset_name)] == 0.0, (
                    f"{offset_name} not cleared after issue transition"
                )


# =============================================================================
# Issue Impact Scalars
# =============================================================================


class TestIssueImpactScalars:
    """Test issue_price_impact and issue_cash_gain context-dependent scalars."""

    def test_zero_on_game_init(self):
        """Impact scalars should be zero after game initialization."""
        from core.state import GameState
        state = GameState(3)
        state.initialize_game(seed=42)

        assert TURN.get_issue_price_impact(state) == 0.0
        assert TURN.get_issue_cash_gain(state) == 0.0

    def test_populated_on_issue_setup(self, issue_state_with_corp):
        """Impact scalars set when corp is presented for issue decision."""
        state = issue_state_with_corp
        setup_issue_phase_py(state)

        # Corp 0 at index 15 ($24), next lower should be index 14 ($22)
        # Impact = 14 - 15 = -1, cash gain = $22
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        impact = TURN.get_issue_price_impact(state)
        cash_gain = TURN.get_issue_cash_gain(state)

        assert abs(impact - (-1.0 / PY_IMPACT_DIVISOR)) < 1e-6, (
            f"Expected impact -1/5 = -0.2, got {impact}"
        )
        assert abs(cash_gain - (22.0 / PY_PRICE_DIVISOR)) < 1e-6, (
            f"Expected cash_gain 22/40 = 0.55, got {cash_gain}"
        )

    def test_cleared_after_transition_out(self, issue_state_with_corp):
        """Impact scalars zeroed when transitioning out of ISSUE phase."""
        state = issue_state_with_corp
        setup_issue_phase_py(state)

        # Pass on issue — should transition out since only one corp
        apply_issue_action_py(state, issue=False)

        assert TURN.get_issue_price_impact(state) == 0.0
        assert TURN.get_issue_cash_gain(state) == 0.0

    def test_stock_masters_zero_impact(self, game_state):
        """Stock Masters (CORP_SM) has zero price impact."""
        state = game_state
        # Float SM (corp 3) at price index 15 ($24)
        float_corp_for_test(state, corp_id=3, par_index=15, float_shares=2)
        CORPS[3].set_cash(state, 50)
        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)

        # SM should be presented (highest price)
        assert TURN.get_issue_corp(state) == 3

        impact = TURN.get_issue_price_impact(state)
        cash_gain = TURN.get_issue_cash_gain(state)

        # SM: no price change, receives current price ($24)
        assert impact == 0.0, f"SM impact should be 0, got {impact}"
        assert abs(cash_gain - (24.0 / PY_PRICE_DIVISOR)) < 1e-6, (
            f"SM cash_gain should be 24/40 = 0.60, got {cash_gain}"
        )

    def test_impact_with_occupied_space_slide(self, game_state):
        """Impact reflects sliding past occupied market spaces."""
        state = game_state
        # Float two corps: corp 0 at index 14, corp 1 at index 15
        float_corp_for_test(state, corp_id=0, par_index=14, float_shares=2)
        float_corp_for_test(state, corp_id=1, par_index=15, float_shares=2)
        CORPS[1].set_cash(state, 50)

        # Corp 1 at index 15 would normally go to 14, but 14 is occupied
        # So it slides to 13
        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)

        # Corp 1 has higher price, should be presented first
        assert TURN.get_issue_corp(state) == 1

        impact = TURN.get_issue_price_impact(state)
        cash_gain = TURN.get_issue_cash_gain(state)

        # Slides from 15 to 13 (skipping occupied 14): impact = -2
        assert abs(impact - (-2.0 / PY_IMPACT_DIVISOR)) < 1e-6, (
            f"Expected impact -2/5 = -0.4, got {impact}"
        )
        expected_price = get_market_price(13)  # $20
        assert abs(cash_gain - (expected_price / PY_PRICE_DIVISOR)) < 1e-6
