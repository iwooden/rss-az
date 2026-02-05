"""Tests for WRAP_UP phase behavior."""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS
from entities.fi import FI
from entities.company import COMPANIES
from entities.deck import DECK
from phases.wrap_up import apply_wrap_up_py


def trigger_wrap_up(state):
    """Helper to trigger WRAP_UP by having all players pass."""
    num_players = state.get_num_players()
    layout = get_action_layout(num_players)
    pass_idx = layout['pass_invest']

    for _ in range(num_players):
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
        COMPANIES[0].mark_revealed(state)
        COMPANIES[1].mark_revealed(state)
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
        for _ in range(2):
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
        for _ in range(3):
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
# PLAYER CASH PRESERVATION TESTS - These expose Bug 2
# =============================================================================

class TestPlayerCashPreservation:
    """Test that player cash is preserved through WRAP_UP cycle.

    BUG: Player cash for players 1+ becomes 0 after WRAP_UP.
    These tests WILL FAIL until the bug is fixed.
    """

    def test_player_cash_preserved_through_wrap_up(self):
        """All players should retain their cash after WRAP_UP cycle."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set specific cash values
        PLAYERS[0].set_cash(state, 20)
        PLAYERS[1].set_cash(state, 30)
        PLAYERS[2].set_cash(state, 25)

        # Record cash before
        cash_before = [PLAYERS[i].get_cash(state) for i in range(3)]

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # All players should retain their cash
        for i in range(3):
            actual_cash = PLAYERS[i].get_cash(state)
            assert actual_cash == cash_before[i], \
                f"Player {i} cash should be {cash_before[i]}, got {actual_cash}"

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_player_cash_preserved_all_player_counts(self, num_players):
        """Player cash preservation works for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set cash: player i gets (i+1)*10
        for i in range(num_players):
            PLAYERS[i].set_cash(state, (i + 1) * 10)

        cash_before = [PLAYERS[i].get_cash(state) for i in range(num_players)]

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # All players should retain their cash
        for i in range(num_players):
            actual_cash = PLAYERS[i].get_cash(state)
            assert actual_cash == cash_before[i], \
                f"Player {i} cash should be {cash_before[i]}, got {actual_cash}"


# =============================================================================
# FI CASH PRESERVATION TESTS - These expose Bug 1
# =============================================================================

class TestFICashPreservation:
    """Test that FI cash is correctly calculated after purchases.

    BUG: FI cash becomes 0 after WRAP_UP regardless of purchases made.
    These tests WILL FAIL until the bug is fixed.
    """

    def test_fi_cash_preserved_when_no_purchases(self):
        """FI should retain cash when no companies are available to purchase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Clear all companies from auction (nothing to buy)
        for company_id in range(36):
            COMPANIES[company_id].remove_from_game(state)

        # Set FI cash
        FI.set_cash(state, 50)
        fi_cash_before = FI.get_cash(state)

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # FI should retain all cash (nothing to buy)
        assert FI.get_cash(state) == fi_cash_before, \
            f"FI cash should be {fi_cash_before}, got {FI.get_cash(state)}"

    def test_fi_cash_zero_means_no_purchases_in_wrap_up(self):
        """FI with 0 cash cannot purchase companies during WRAP_UP.

        Note: trigger_wrap_up goes through the full cycle (WRAP_UP -> ACQUISITION ->
        CLOSING -> INCOME -> IPO -> INVEST). FI receives income during
        INCOME phase, so final cash will include income from owned companies.
        This test verifies FI had 0 cash going into WRAP_UP (unable to buy).
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set FI cash to 0
        FI.set_cash(state, 0)

        # Calculate expected income FI will receive during INCOME phase
        fi_income = FI.calculate_income(state)

        # Trigger full cycle (WRAP_UP through INVEST)
        trigger_wrap_up(state)

        # FI should have only its income (could not purchase during WRAP_UP)
        assert FI.get_cash(state) == fi_income


# =============================================================================
# PLAYER REORDERING TESTS - These expose Bug 2 indirectly
# =============================================================================

class TestPlayerReordering:
    """Test player reordering by cash with tie-breaking.

    These tests verify REORDER-01, REORDER-02, REORDER-03.
    They WILL FAIL until Bug 2 is fixed (player cash becomes 0).
    """

    @pytest.mark.parametrize("cash_values,expected_order", [
        # No ties - descending cash order
        ([20, 30, 25], [1, 2, 0]),  # Player 1 (30) > Player 2 (25) > Player 0 (20)
        ([30, 25, 20], [0, 1, 2]),  # Already sorted
        ([25, 20, 30], [2, 0, 1]),  # Player 2 (30) > Player 0 (25) > Player 1 (20)
    ])
    def test_reorder_by_descending_cash(self, cash_values, expected_order):
        """Players should be reordered by descending cash."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set specific cash values
        for i in range(3):
            PLAYERS[i].set_cash(state, cash_values[i])

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # Check turn order matches expected
        for expected_position, player_id in enumerate(expected_order):
            actual_position = PLAYERS[player_id].get_turn_order(state)
            assert actual_position == expected_position, \
                f"Player {player_id} should be at position {expected_position}, got {actual_position}"

    @pytest.mark.parametrize("cash_values,expected_order", [
        # Two-way tie - old position preserved
        ([30, 30, 20], [0, 1, 2]),  # Players 0,1 tied at 30, old order preserved
        ([20, 30, 30], [1, 2, 0]),  # Players 1,2 tied at 30, old order preserved
        # Three-way tie
        ([30, 30, 30], [0, 1, 2]),  # All tied, old order preserved
    ])
    def test_tie_breaking_preserves_old_order(self, cash_values, expected_order):
        """When players have equal cash, old turn order is preserved."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set specific cash values
        for i in range(3):
            PLAYERS[i].set_cash(state, cash_values[i])

        # Trigger WRAP_UP
        trigger_wrap_up(state)

        # Check turn order matches expected
        for expected_position, player_id in enumerate(expected_order):
            actual_position = PLAYERS[player_id].get_turn_order(state)
            assert actual_position == expected_position, \
                f"Player {player_id} should be at position {expected_position}, got {actual_position}"


# =============================================================================
# FI PURCHASE BEHAVIOR TESTS
# =============================================================================

class TestFIPurchaseBehavior:
    """Test FI company purchase rules during WRAP_UP.

    Per RULES.md line 611: 'Foreign Investor cannot buy them in Phase 2 of same turn'
    - newly drawn companies should be unavailable for purchase in the same WRAP_UP cycle.
    """

    def test_fi_cannot_buy_revealed_company_same_wrap_up(self):
        """WRAP-FI-01: FI cannot purchase newly-drawn company in same WRAP_UP cycle.

        When FI buys a company and a replacement is drawn from the deck,
        that replacement is marked as revealed and cannot be purchased
        until the next turn's WRAP_UP phase.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Clear all companies first
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        # Set up controlled scenario:
        # - Company 0 (face value $1) is for auction
        # - Company 1 (face value $1) is in deck (will be drawn as replacement)
        # - FI has cash = $2 (can afford two $1 companies)
        COMPANIES[0].move_to_auction(state)
        DECK.set_order(state, [1])  # Only company 1 in deck
        FI.set_cash(state, 2)

        # Verify initial state
        assert COMPANIES[0].is_for_auction(state)
        assert FI.get_cash(state) == 2

        # Set phase to WRAP_UP and apply directly
        TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
        apply_wrap_up_py(state)

        # Verify:
        # - Company 0 should be owned by FI (purchased for $1)
        # - Company 1 should NOT be owned by FI (was revealed, not purchased)
        # - FI should have $1 remaining (only spent $1)
        assert COMPANIES[0].is_owned_by_fi(state), \
            "FI should have purchased company 0"
        assert not COMPANIES[1].is_owned_by_fi(state), \
            "FI should NOT have purchased newly-drawn company 1"
        assert FI.get_cash(state) == 1, \
            "FI should have $1 remaining (started with $2, spent $1)"

        # Company 1 should now be for_auction (revealed -> available at end of WRAP_UP)
        assert COMPANIES[1].is_for_auction(state), \
            "Company 1 should be available for auction after WRAP_UP"

    def test_fi_multiple_purchases_revealed_excluded(self):
        """WRAP-FI-02: Multiple FI purchases still exclude revealed companies.

        When FI makes multiple purchases in one WRAP_UP, each replacement
        drawn is marked revealed and excluded from subsequent purchases.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Clear all companies
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        # Face values: company 0=$1, company 1=$2, company 2=$5, company 3=$6
        # Set up:
        # - Companies 0 ($1) and 1 ($2) are for auction
        # - Companies 2 ($5) and 3 ($6) are in deck (replacements)
        # - FI has $15 (can buy 0+1 for $3, would have $12 left which could buy 2 if not revealed)
        COMPANIES[0].move_to_auction(state)
        COMPANIES[1].move_to_auction(state)
        DECK.set_order(state, [3, 2])  # Company 2 drawn first, then 3
        FI.set_cash(state, 15)

        TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
        apply_wrap_up_py(state)

        # FI should only own the two originally-available companies
        assert COMPANIES[0].is_owned_by_fi(state), "FI should own company 0 ($1)"
        assert COMPANIES[1].is_owned_by_fi(state), "FI should own company 1 ($2)"
        assert not COMPANIES[2].is_owned_by_fi(state), "FI should NOT own revealed company 2"
        assert not COMPANIES[3].is_owned_by_fi(state), "FI should NOT own revealed company 3"

        # FI spent $3 (company 0 for $1 + company 1 for $2), should have $12 remaining
        assert FI.get_cash(state) == 12, "FI should have $12 remaining"

        # Both revealed companies should now be for auction
        assert COMPANIES[2].is_for_auction(state)
        assert COMPANIES[3].is_for_auction(state)

    def test_fi_buys_in_ascending_order_until_broke(self):
        """WRAP-FI-03: FI buys cheapest first and stops when cash runs out.

        Tests both ascending face value ordering AND cash-exhaustion loop
        termination. Unlike WRAP-FI-01/02 which test revealed-company
        exclusion, this uses an empty deck so only cash matters.

        Per RULES.md: 'In ascending Face Value order, Foreign Investor buys
        as many available companies as possible at Face Value.'

        Companies at $1/$2/$5/$6, FI cash=$8. Ascending order buys $1+$2+$5
        (3 companies for $8). Descending order would buy $6 then $2 then
        can't afford $5 (only 2 companies). Owning all three cheapest is
        only possible with ascending ordering.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Clear all companies
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        # Place 4 companies for auction with no deck replacements
        COMPANIES[0].move_to_auction(state)  # $1
        COMPANIES[1].move_to_auction(state)  # $2
        COMPANIES[2].move_to_auction(state)  # $5
        COMPANIES[3].move_to_auction(state)  # $6
        DECK.set_order(state, [])  # Empty deck - no replacements

        FI.set_cash(state, 8)

        TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
        apply_wrap_up_py(state)

        # Ascending: buy 0($1,cash=7), 1($2,cash=5), 2($5,cash=0), can't afford 3($6)
        # This 3-company result is ONLY possible with ascending ordering
        assert COMPANIES[0].is_owned_by_fi(state), "FI should own company 0 ($1)"
        assert COMPANIES[1].is_owned_by_fi(state), "FI should own company 1 ($2)"
        assert COMPANIES[2].is_owned_by_fi(state), "FI should own company 2 ($5)"
        assert not COMPANIES[3].is_owned_by_fi(state), "FI should NOT own company 3 ($6, unaffordable)"
        assert COMPANIES[3].is_for_auction(state), "Company 3 should remain for auction"
        assert FI.get_cash(state) == 0, "FI should have $0 remaining (1+2+5=8)"
