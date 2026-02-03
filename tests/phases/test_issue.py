"""Tests for ISSUE_SHARES phase (Phase 8).

Requirements covered:
- ISS-01: Basic issue mechanics (shares transfer, cash flow)
- ISS-02: Price movement (normal corps)
- ISS-03: Stock Masters special (no price change)
- ISS-04: Processing order (descending price)
- ISS-05: Receivership handling (must issue vs auto-pass)
- ISS-06: Bankruptcy (price drops to 0)
- ISS-07: Action mask validation
- ISS-08: Phase transitions (END_CARD -> ISSUE_SHARES -> IPO -> INVEST)
- ISS-09: Integration tests
"""
import pytest
from core.state import GameState
from core.data import GamePhases, GameConstants, CorpIndices, get_market_price
from core.actions import get_valid_action_mask, get_action_layout
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from phases.issue import (
    setup_issue_phase_py,
    apply_issue_action_py,
    find_next_issue_corp_py,
    process_issue_share_py,
)


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
    - unissued_shares = 3
    - issued_shares = 4
    - bank_shares = 0
    - cash = 50
    - Player 0 is president with 2 shares
    """
    # Set up corp 0
    corp = CORPS[0]
    corp.initialize(game_state)
    corp.set_active(game_state, True)
    corp.set_price_index(game_state, 15)  # $24
    corp.set_unissued_shares(game_state, 3)
    corp.set_issued_shares(game_state, 4)
    corp.set_bank_shares(game_state, 0)
    corp.set_cash(game_state, 50)
    corp.set_in_receivership(game_state, False)

    # Set up market - occupy space 15
    MARKET.initialize(game_state)
    MARKET.set_space_available(game_state, 15, False)

    # Set up player 0 as president
    PLAYERS[0].set_shares(game_state, 0, 2)
    PLAYERS[0].set_president_of(game_state, 0, True)

    # Set phase
    TURN.set_phase(game_state, GamePhases.PHASE_ISSUE_SHARES)

    return game_state


# =============================================================================
# ISS-01: Basic Issue Mechanics
# =============================================================================


class TestBasicIssueMechanics:
    """ISS-01: Basic issue mechanics (shares transfer, cash flow)."""

    def test_issue_transfers_one_share(self, issue_state_with_corp):
        """Issuing transfers one share from unissued to issued and bank."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_unissued = corp.get_unissued_shares(state)
        initial_issued = corp.get_issued_shares(state)
        initial_bank = corp.get_bank_shares(state)

        # Set up phase and apply issue
        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

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

        # Get the expected new price after drop
        current_index = corp.get_price_index(state)
        new_index = MARKET.find_next_lower_space(state, current_index)
        expected_proceeds = get_market_price(new_index)

        apply_issue_action_py(state, issue=True)

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
        apply_issue_action_py(state, issue=False)  # Pass

        assert corp.get_unissued_shares(state) == initial_unissued
        assert corp.get_issued_shares(state) == initial_issued
        assert corp.get_bank_shares(state) == initial_bank
        assert corp.get_cash(state) == initial_cash


# =============================================================================
# ISS-02: Price Movement (Normal Corps)
# =============================================================================


class TestPriceMovement:
    """ISS-02: Price movement for normal corporations."""

    def test_price_drops_on_issue(self, issue_state_with_corp):
        """Normal corp price drops when issuing."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        final_index = corp.get_price_index(state)
        assert final_index < initial_index

    def test_price_drops_to_next_available_space(self, issue_state_with_corp):
        """Price drops to next available (lower) market space."""
        state = issue_state_with_corp
        corp = CORPS[0]

        # Block some spaces to create a gap
        MARKET.set_space_available(state, 14, False)  # Occupied
        MARKET.set_space_available(state, 13, True)   # Available

        initial_index = corp.get_price_index(state)  # 15

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        # Should drop to 13 (skipping occupied 14)
        assert corp.get_price_index(state) == 13

    def test_old_space_freed_after_issue(self, issue_state_with_corp):
        """Old market space becomes available after price drop."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)  # 15
        assert not MARKET.is_space_available(state, initial_index)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        assert MARKET.is_space_available(state, initial_index)

    def test_new_space_occupied_after_issue(self, issue_state_with_corp):
        """New market space becomes occupied after price drop."""
        state = issue_state_with_corp
        corp = CORPS[0]

        initial_index = corp.get_price_index(state)  # 15
        new_index = MARKET.find_next_lower_space(state, initial_index)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

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
        apply_issue_action_py(state, issue=True)

        assert corp.get_cash(state) == initial_cash + new_price


# =============================================================================
# ISS-03: Stock Masters Special
# =============================================================================


class TestStockMastersSpecial:
    """ISS-03: Stock Masters (CORP_SM) price does not change."""

    def test_sm_price_unchanged_on_issue(self, game_state):
        """Stock Masters price stays the same when issuing."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]  # Corp 3

        # Set up Stock Masters
        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)  # $24
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, CorpIndices.CORP_SM, 2)
        PLAYERS[0].set_president_of(state, CorpIndices.CORP_SM, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_index = corp.get_price_index(state)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        assert corp.get_price_index(state) == initial_index

    def test_sm_receives_current_price(self, game_state):
        """Stock Masters receives current (unchanged) price as proceeds."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)  # $24
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, CorpIndices.CORP_SM, 2)
        PLAYERS[0].set_president_of(state, CorpIndices.CORP_SM, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_cash = corp.get_cash(state)
        current_price = get_market_price(15)  # $24

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        assert corp.get_cash(state) == initial_cash + current_price

    def test_sm_market_space_unchanged(self, game_state):
        """Stock Masters keeps same market space occupied."""
        state = game_state
        corp = CORPS[CorpIndices.CORP_SM]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, CorpIndices.CORP_SM, 2)
        PLAYERS[0].set_president_of(state, CorpIndices.CORP_SM, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        # Space 15 should still be occupied
        assert not MARKET.is_space_available(state, 15)


# =============================================================================
# ISS-04: Processing Order
# =============================================================================


class TestProcessingOrder:
    """ISS-04: Corps processed in descending share price order."""

    def test_higher_price_corp_processed_first(self, game_state):
        """Corp with higher price index is processed before lower."""
        state = game_state

        # Set up two corps at different prices
        corp0 = CORPS[0]
        corp0.initialize(state)
        corp0.set_active(state, True)
        corp0.set_price_index(state, 10)  # Lower price
        corp0.set_unissued_shares(state, 3)
        corp0.set_issued_shares(state, 4)
        corp0.set_cash(state, 50)
        corp0.set_in_receivership(state, False)

        corp1 = CORPS[1]
        corp1.initialize(state)
        corp1.set_active(state, True)
        corp1.set_price_index(state, 15)  # Higher price
        corp1.set_unissued_shares(state, 3)
        corp1.set_issued_shares(state, 4)
        corp1.set_cash(state, 50)
        corp1.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 10, False)
        MARKET.set_space_available(state, 15, False)

        # Both have presidents
        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 1, 2)
        PLAYERS[1].set_president_of(state, 1, True)

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
        prices = {0: 8, 1: 20, 2: 12}  # Expected order: 1, 2, 0

        for corp_id, price_idx in prices.items():
            corp = CORPS[corp_id]
            corp.initialize(state)
            corp.set_active(state, True)
            corp.set_price_index(state, price_idx)
            corp.set_unissued_shares(state, 3)
            corp.set_issued_shares(state, 4)
            corp.set_cash(state, 50)
            corp.set_in_receivership(state, False)
            MARKET.set_space_available(state, price_idx, False)

        MARKET.initialize(state)
        for price_idx in prices.values():
            MARKET.set_space_available(state, price_idx, False)

        # All have presidents
        for corp_id in range(3):
            PLAYERS[corp_id % 3].set_shares(state, corp_id, 2)
            PLAYERS[corp_id % 3].set_president_of(state, corp_id, True)

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
# ISS-05: Receivership Handling
# =============================================================================


class TestReceivershipHandling:
    """ISS-05: Receivership corps must issue if they have unissued shares."""

    def test_receivership_with_unissued_auto_issues(self, game_state):
        """Receivership corp with unissued shares auto-issues."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, True)  # In receivership

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_unissued = corp.get_unissued_shares(state)

        # Setup should auto-process receivership corps
        setup_issue_phase_py(state)

        # Should have auto-issued (one less unissued)
        assert corp.get_unissued_shares(state) == initial_unissued - 1

    def test_receivership_without_unissued_auto_passes(self, game_state):
        """Receivership corp without unissued shares auto-passes."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 0)  # No unissued shares
        corp.set_issued_shares(state, 7)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, True)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_cash = corp.get_cash(state)

        setup_issue_phase_py(state)

        # Should have auto-passed (cash unchanged)
        assert corp.get_cash(state) == initial_cash

    def test_receivership_skipped_for_player_decision(self, game_state):
        """Player-controlled corp waits for player decision."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)  # Has president

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        initial_unissued = corp.get_unissued_shares(state)

        setup_issue_phase_py(state)

        # Should NOT auto-issue (waiting for player decision)
        assert corp.get_unissued_shares(state) == initial_unissued
        # Issue corp should be set
        assert TURN.get_issue_corp(state) == 0


# =============================================================================
# ISS-06: Bankruptcy
# =============================================================================


class TestBankruptcy:
    """ISS-06: Bankruptcy when price drops to 0."""

    def test_issue_at_low_price_causes_bankruptcy(self, game_state):
        """Corp at price index 1 goes bankrupt when issuing."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 1)  # One above bankruptcy
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 1, False)
        # Space 0 is always available (bankruptcy)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        # Corp should be deactivated (bankrupt)
        assert not corp.is_active(state)

    def test_bankruptcy_deactivates_corp(self, game_state):
        """Bankruptcy deactivates the corporation."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 1)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 1, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        assert corp.is_active(state)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        assert not corp.is_active(state)

    def test_bankruptcy_clears_player_shares(self, game_state):
        """Bankruptcy clears all player shares."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 1)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 1, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 0, 1)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)
        apply_issue_action_py(state, issue=True)

        # All player shares should be cleared
        assert PLAYERS[0].get_shares(state, 0) == 0
        assert PLAYERS[1].get_shares(state, 0) == 0


# =============================================================================
# ISS-07: Action Mask Validation
# =============================================================================


class TestActionMaskValidation:
    """ISS-07: Action mask validation for ISSUE_SHARES phase."""

    def test_mask_has_issue_and_pass_with_unissued(self, issue_state_with_corp):
        """Mask includes both ISSUE and PASS when corp has unissued shares."""
        state = issue_state_with_corp

        setup_issue_phase_py(state)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['issue_action']] == 1.0
        assert mask[layout['issue_pass']] == 1.0

    def test_mask_only_pass_without_unissued(self, game_state):
        """Mask only has PASS when corp has no unissued shares."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 0)  # No unissued
        corp.set_issued_shares(state, 7)
        corp.set_bank_shares(state, 0)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['issue_action']] == 0.0  # Can't issue
        assert mask[layout['issue_pass']] == 1.0    # Can pass

    def test_no_actions_when_phase_complete(self, issue_state):
        """No valid actions when no corps remain to process."""
        state = issue_state

        # No active corps -> setup transitions to IPO then INVEST
        setup_issue_phase_py(state)

        # Phase should have transitioned
        assert TURN.get_phase(state) != GamePhases.PHASE_ISSUE_SHARES


# =============================================================================
# ISS-08: Phase Transitions
# =============================================================================


class TestPhaseTransitions:
    """ISS-08: Phase transitions for ISSUE_SHARES."""

    def test_transitions_to_invest_when_done(self, issue_state):
        """Transitions to INVEST (via IPO) when all corps processed."""
        state = issue_state

        # No active corps
        setup_issue_phase_py(state)

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_single_corp_transitions_after_action(self, issue_state_with_corp):
        """Single corp transitions to INVEST (via IPO) after action."""
        state = issue_state_with_corp

        setup_issue_phase_py(state)
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES

        apply_issue_action_py(state, issue=True)

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_multiple_corps_process_all_before_transition(self, game_state):
        """Multiple corps all processed before transitioning."""
        state = game_state

        # Set up two corps
        for corp_id in range(2):
            corp = CORPS[corp_id]
            corp.initialize(state)
            corp.set_active(state, True)
            corp.set_price_index(state, 10 + corp_id * 5)
            corp.set_unissued_shares(state, 3)
            corp.set_issued_shares(state, 4)
            corp.set_cash(state, 50)
            corp.set_in_receivership(state, False)
            MARKET.set_space_available(state, 10 + corp_id * 5, False)

        MARKET.initialize(state)
        for corp_id in range(2):
            MARKET.set_space_available(state, 10 + corp_id * 5, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 1, 2)
        PLAYERS[1].set_president_of(state, 1, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)

        # First corp (higher price index = 15)
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        apply_issue_action_py(state, issue=False)  # Pass

        # Second corp
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        apply_issue_action_py(state, issue=False)  # Pass

        # Now should transition to INVEST (via IPO)
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_setup_from_end_card(self, game_state):
        """Issue phase can be set up from END_CARD transition."""
        state = game_state
        corp = CORPS[0]

        corp.initialize(state)
        corp.set_active(state, True)
        corp.set_price_index(state, 15)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        corp.set_cash(state, 50)
        corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 15, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        # Simulate END_CARD -> ISSUE_SHARES transition
        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
        setup_issue_phase_py(state)

        # Should be ready for player action
        assert TURN.get_phase(state) == GamePhases.PHASE_ISSUE_SHARES
        assert TURN.get_issue_corp(state) == 0


# =============================================================================
# ISS-09: Integration Tests
# =============================================================================


class TestIntegration:
    """ISS-09: Integration tests for ISSUE_SHARES phase."""

    def test_mixed_receivership_and_player_corps(self, game_state):
        """Mix of receivership and player-controlled corps processed correctly."""
        state = game_state

        # Corp 0: Player-controlled, highest price
        corp0 = CORPS[0]
        corp0.initialize(state)
        corp0.set_active(state, True)
        corp0.set_price_index(state, 20)
        corp0.set_unissued_shares(state, 3)
        corp0.set_issued_shares(state, 4)
        corp0.set_cash(state, 50)
        corp0.set_in_receivership(state, False)

        # Corp 1: Receivership, middle price
        corp1 = CORPS[1]
        corp1.initialize(state)
        corp1.set_active(state, True)
        corp1.set_price_index(state, 15)
        corp1.set_unissued_shares(state, 3)
        corp1.set_issued_shares(state, 4)
        corp1.set_cash(state, 50)
        corp1.set_in_receivership(state, True)

        # Corp 2: Player-controlled, lowest price
        corp2 = CORPS[2]
        corp2.initialize(state)
        corp2.set_active(state, True)
        corp2.set_price_index(state, 10)
        corp2.set_unissued_shares(state, 3)
        corp2.set_issued_shares(state, 3)
        corp2.set_cash(state, 50)
        corp2.set_in_receivership(state, False)

        MARKET.initialize(state)
        MARKET.set_space_available(state, 20, False)
        MARKET.set_space_available(state, 15, False)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 2, 2)
        PLAYERS[1].set_president_of(state, 2, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        corp1_initial_unissued = corp1.get_unissued_shares(state)

        # Setup finds highest-price corp first
        setup_issue_phase_py(state)

        # Corp 1 (receivership) NOT yet processed - Corp 0 (highest price) is first
        assert corp1.get_unissued_shares(state) == corp1_initial_unissued

        # Should be waiting on corp 0 (highest price player-controlled)
        assert TURN.get_issue_corp(state) == 0

        apply_issue_action_py(state, issue=False)  # Pass on corp 0

        # Now corp 1 (receivership) should have auto-issued during advance
        # And we should be on corp 2
        assert corp1.get_unissued_shares(state) == corp1_initial_unissued - 1
        assert TURN.get_issue_corp(state) == 2

        apply_issue_action_py(state, issue=False)  # Pass on corp 2

        # Should transition to INVEST (via IPO)
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_all_corps_pass_no_share_changes(self, game_state):
        """All corps passing results in no share changes."""
        state = game_state

        initial_shares = {}

        for corp_id in range(2):
            corp = CORPS[corp_id]
            corp.initialize(state)
            corp.set_active(state, True)
            corp.set_price_index(state, 10 + corp_id * 5)
            corp.set_unissued_shares(state, 3)
            corp.set_issued_shares(state, 4)
            corp.set_cash(state, 50)
            corp.set_in_receivership(state, False)

            initial_shares[corp_id] = {
                'unissued': corp.get_unissued_shares(state),
                'issued': corp.get_issued_shares(state),
                'bank': corp.get_bank_shares(state),
            }

        MARKET.initialize(state)
        for corp_id in range(2):
            MARKET.set_space_available(state, 10 + corp_id * 5, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 1, 2)
        PLAYERS[1].set_president_of(state, 1, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)

        # Pass on both
        apply_issue_action_py(state, issue=False)
        apply_issue_action_py(state, issue=False)

        # Verify no share changes
        for corp_id in range(2):
            corp = CORPS[corp_id]
            assert corp.get_unissued_shares(state) == initial_shares[corp_id]['unissued']
            assert corp.get_issued_shares(state) == initial_shares[corp_id]['issued']
            assert corp.get_bank_shares(state) == initial_shares[corp_id]['bank']

    def test_all_corps_issue_share_counts_correct(self, game_state):
        """All corps issuing results in correct share counts."""
        state = game_state

        for corp_id in range(2):
            corp = CORPS[corp_id]
            corp.initialize(state)
            corp.set_active(state, True)
            corp.set_price_index(state, 10 + corp_id * 5)
            corp.set_unissued_shares(state, 3)
            corp.set_issued_shares(state, 4)
            corp.set_bank_shares(state, 0)
            corp.set_cash(state, 50)
            corp.set_in_receivership(state, False)

        MARKET.initialize(state)
        for corp_id in range(2):
            MARKET.set_space_available(state, 10 + corp_id * 5, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)
        PLAYERS[1].set_shares(state, 1, 2)
        PLAYERS[1].set_president_of(state, 1, True)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        setup_issue_phase_py(state)

        # Issue on both
        apply_issue_action_py(state, issue=True)
        apply_issue_action_py(state, issue=True)

        # Verify share changes
        for corp_id in range(2):
            corp = CORPS[corp_id]
            assert corp.get_unissued_shares(state) == 2  # 3 - 1
            assert corp.get_issued_shares(state) == 5    # 4 + 1
            assert corp.get_bank_shares(state) == 1      # 0 + 1
