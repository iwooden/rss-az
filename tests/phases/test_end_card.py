"""Tests for END_CARD phase (Phase 7)."""
import pytest
from core.state import GameState
from core.data import GamePhases, GameConstants
from core.actions import get_valid_action_mask
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.fi import FI
from entities.market import MARKET
from phases.end_card import apply_end_card_py
from tests.phases.conftest import float_corp_for_test, assert_invariants
from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def end_card_state(game_state):
    """
    3-player game state set up at PHASE_END_CARD.

    By default:
    - No corps at 75 price
    - Companies still in deck (unowned companies exist)
    - End card not flipped

    Should transition to ISSUE_SHARES.
    """
    TURN.set_phase(game_state, GamePhases.PHASE_END_CARD)
    return game_state


# =============================================================================
# 75 Price Check
# =============================================================================


class Test75PriceCheck:
    """Corp at price_index 26 triggers GAME_OVER."""

    def test_corp_at_75_price_triggers_game_over(self, end_card_state):
        """Corp with price_index 26 ($75) ends the game immediately."""
        # Float corp 0 at max price ($75)
        float_corp_for_test(end_card_state, corp_id=0, par_index=26)

        apply_end_card_py(end_card_state)
        # Skip assert_invariants: float_corp_for_test at par_index=26 marks
        # market space 26 as occupied, violating the "always available" invariant

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_corp_at_74_price_does_not_trigger(self, end_card_state):
        """Corp at price_index 25 ($74) does NOT trigger game over."""
        # Float corp 0 one below max price
        float_corp_for_test(end_card_state, corp_id=0, par_index=25)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # Should continue to ISSUE_SHARES, not GAME_OVER
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_ISSUE_SHARES

    def test_multiple_corps_one_at_75(self, end_card_state):
        """If any corp is at 75, game ends regardless of others."""
        # Corp 0 at low price
        float_corp_for_test(end_card_state, corp_id=0, par_index=10)

        # Corp 1 at 75 price
        float_corp_for_test(end_card_state, corp_id=1, player_id=1, par_index=26)

        apply_end_card_py(end_card_state)
        # Skip assert_invariants: float_corp_for_test at par_index=26 marks
        # market space 26 as occupied, violating the "always available" invariant

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_inactive_corp_at_75_ignored(self, end_card_state):
        """Inactive corps at 75 price do not trigger game over."""
        # Corp 0 starts inactive after initialize_game() - just set its price
        corp = CORPS[0]
        corp.set_price_index(end_card_state, 26)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # Should not trigger game over - transitions to INVEST for new turn
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_INVEST


# =============================================================================
# No Unowned Companies Check
# =============================================================================


class TestNoUnownedCompanies:
    """Empty deck + no auction/revealed companies flips end card."""

    def test_all_companies_owned_flips_end_card(self, end_card_state):
        """When no companies in deck/auction/revealed, end card flips but game continues."""
        # Move all companies out of deck/auction/revealed
        # Transfer all to players/corps/FI or remove from game
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            # Transfer to player 0 (simplest way to get them out of deck)
            COMPANIES[company_id].transfer_to_player(end_card_state, 0)

        # Verify end card was not flipped before
        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # End card should now be flipped
        assert TURN.is_end_card_flipped(end_card_state)
        # Game continues this turn — ends at NEXT END_CARD phase
        assert TURN.get_phase(end_card_state) != GamePhases.PHASE_GAME_OVER

    def test_company_in_deck_prevents_flip(self, end_card_state):
        """Company remaining in deck prevents end card flip."""
        # Default state has companies in deck
        # Verify at least one company is in deck
        found_in_deck = any(
            COMPANIES[cid].is_in_deck(end_card_state)
            for cid in range(int(GameConstants.NUM_COMPANIES))
        )

        assert found_in_deck, "Test setup: expected companies in deck"
        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # End card should NOT flip - continues to next turn
        assert not TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_INVEST

    def test_company_in_auction_prevents_flip(self, end_card_state):
        """Company in auction slot prevents end card flip."""
        # Move all but one company to players
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            if company_id == 0:
                # Keep company 0 in auction
                COMPANIES[0].move_to_auction(end_card_state)
            else:
                COMPANIES[company_id].transfer_to_player(end_card_state, 0)

        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # End card should NOT flip (company 0 is in auction)
        assert not TURN.is_end_card_flipped(end_card_state)
        # Flow is now END_CARD -> ISSUE_SHARES -> IPO (since players own companies)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_IPO

    def test_company_revealed_prevents_flip(self, end_card_state):
        """Revealed company prevents end card flip."""
        # Move all but one company to players
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            if company_id == 0:
                # Keep company 0 as revealed
                COMPANIES[0].mark_revealed(end_card_state)
            else:
                COMPANIES[company_id].transfer_to_player(end_card_state, 0)

        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # End card should NOT flip
        assert not TURN.is_end_card_flipped(end_card_state)
        # Flow is now END_CARD -> ISSUE_SHARES -> IPO (since players own companies)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_IPO


# =============================================================================
# End Card Already Flipped
# =============================================================================


class TestEndCardFlipped:
    """Pre-flipped card triggers GAME_OVER."""

    def test_preflipped_end_card_triggers_game_over(self, end_card_state):
        """If end card already flipped, game ends."""
        TURN.set_end_card_flipped(end_card_state, True)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_preflipped_without_75_price_still_ends(self, end_card_state):
        """Game ends on flipped card even without 75 price corp."""
        TURN.set_end_card_flipped(end_card_state, True)
        # All corps start inactive after initialize_game() - no 75-price check can trigger

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER


# =============================================================================
# Normal Transition
# =============================================================================


class TestNormalTransition:
    """Normal case transitions to ISSUE_SHARES."""

    def test_normal_state_transitions_to_issue_shares(self, end_card_state):
        """Default state (no end conditions) goes to ISSUE_SHARES."""
        # Default state has:
        # - No corp at 75 price
        # - Companies still in deck
        # - End card not flipped
        # Note: With no active corps, issue phase auto-transitions to INVEST

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # No active corps -> issue phase transitions directly to INVEST
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_INVEST

    def test_active_corps_below_75_continue(self, end_card_state):
        """Active corps below 75 price continue to issue phase."""
        # Set up some active corps at various prices
        for corp_id in range(3):
            float_corp_for_test(end_card_state, corp_id=corp_id, player_id=corp_id, par_index=10 + corp_id)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # Active corps exist -> stays in ISSUE_SHARES for player decisions
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_ISSUE_SHARES


# =============================================================================
# Non-Player Phase
# =============================================================================


class TestNonPlayerPhase:
    """END_CARD is non-player phase (0 valid actions, auto-executes)."""

    def test_end_card_has_no_valid_actions(self, end_card_state):
        """In END_CARD phase, action mask should be all zeros."""
        mask = get_valid_action_mask(end_card_state)

        valid_count = sum(1 for v in mask if v == 1.0)
        assert valid_count == 0, f"Expected 0 valid actions, got {valid_count}"

    def test_end_card_auto_executes_in_driver(self, end_card_state):
        """END_CARD phase should auto-execute when reached."""
        # The driver should detect this is a non-player phase
        # Verify by applying and checking transition
        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # Should have transitioned out of END_CARD
        phase = TURN.get_phase(end_card_state)
        assert phase != GamePhases.PHASE_END_CARD, "Should have auto-transitioned"


# =============================================================================
# CoO Level Update
# =============================================================================


class TestCoOLevelUpdate:
    """Flip sets CoO level to 7."""

    def test_flip_sets_coo_to_seven(self, end_card_state):
        """When end card flips, CoO level becomes 7."""
        # Move all companies out of deck/auction/revealed to trigger flip
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            COMPANIES[company_id].transfer_to_player(end_card_state, 0)

        # Verify starting CoO level
        starting_coo = TURN.get_coo_level(end_card_state)
        assert starting_coo < 7, f"Test setup: expected CoO < 7, got {starting_coo}"

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # CoO should now be 7
        assert TURN.get_coo_level(end_card_state) == 7

    def test_preflipped_does_not_change_coo(self, end_card_state):
        """Pre-flipped end card does not modify CoO level."""
        TURN.set_end_card_flipped(end_card_state, True)
        TURN.set_coo_level(end_card_state, 3)

        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After end card")

        # CoO should remain at 3 (flip already happened, we just end game)
        assert TURN.get_coo_level(end_card_state) == 3

    def test_75_price_does_not_change_coo(self, end_card_state):
        """Game ending via 75 price does not change CoO level."""
        float_corp_for_test(end_card_state, corp_id=0, par_index=26)

        # Set CoO after float_corp_for_test (which draws from deck and may
        # cross a color boundary, bumping CoO as a side effect)
        TURN.set_coo_level(end_card_state, 4)

        apply_end_card_py(end_card_state)
        # Skip assert_invariants: float_corp_for_test at par_index=26 marks
        # market space 26 as occupied, violating the "always available" invariant

        # CoO should remain at 4 (game ends without flip)
        assert TURN.get_coo_level(end_card_state) == 4
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER


# =============================================================================
# Integration: Priority Order of Checks
# =============================================================================


class TestCheckPriority:
    """Verify checks happen in correct order per RULES.md."""

    def test_75_price_takes_priority_over_flip(self, end_card_state):
        """75 price ends game before checking end card flip."""
        # Set both conditions
        TURN.set_end_card_flipped(end_card_state, True)

        float_corp_for_test(end_card_state, corp_id=0, par_index=26)

        apply_end_card_py(end_card_state)
        # Skip assert_invariants: float_corp_for_test at par_index=26 marks
        # market space 26 as occupied, violating the "always available" invariant

        # Game ends (both conditions would end game, but 75 check is first)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_flip_then_next_end_card_ends_game(self, end_card_state):
        """First END_CARD flips the card, second END_CARD ends the game."""
        # Move all companies out to trigger flip
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            COMPANIES[company_id].transfer_to_player(end_card_state, 0)

        assert not TURN.is_end_card_flipped(end_card_state)

        # First END_CARD: flips the card, game continues
        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After first end card")

        assert TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_coo_level(end_card_state) == 7
        assert TURN.get_phase(end_card_state) != GamePhases.PHASE_GAME_OVER

        # Second END_CARD: card already flipped → GAME_OVER
        TURN.set_phase(end_card_state, GamePhases.PHASE_END_CARD)
        apply_end_card_py(end_card_state)
        assert_invariants(end_card_state, "After second end card")

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER
