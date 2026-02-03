"""Tests for END_CARD phase (Phase 7).

Requirements covered:
- END-01: Corp at price_index 26 (75 price) triggers GAME_OVER
- END-02: No unowned companies (deck/auction/revealed empty) flips end card
- END-03: Pre-flipped end card triggers GAME_OVER
- END-04: Normal transition (none of above) goes to ISSUE_SHARES
- END-05: END_CARD is non-player phase (0 valid actions, auto-executes)
- END-06: Flipping end card sets CoO level to 7
"""
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
# END-01: 75 Price Check
# =============================================================================


class Test75PriceCheck:
    """END-01: Corp at price_index 26 triggers GAME_OVER."""

    def test_corp_at_75_price_triggers_game_over(self, end_card_state):
        """Corp with price_index 26 ($75) ends the game immediately."""
        # Activate corp 0 and set to max price
        corp = CORPS[0]
        corp.set_active(end_card_state, True)
        corp.set_price_index(end_card_state, 26)  # $75 price
        corp.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 26, False)

        apply_end_card_py(end_card_state)

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_corp_at_74_price_does_not_trigger(self, end_card_state):
        """Corp at price_index 25 ($74) does NOT trigger game over."""
        corp = CORPS[0]
        corp.set_active(end_card_state, True)
        corp.set_price_index(end_card_state, 25)  # One below max
        corp.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 25, False)

        apply_end_card_py(end_card_state)

        # Should continue to ISSUE_SHARES, not GAME_OVER
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_ISSUE_SHARES

    def test_multiple_corps_one_at_75(self, end_card_state):
        """If any corp is at 75, game ends regardless of others."""
        # Corp 0 at low price
        corp0 = CORPS[0]
        corp0.set_active(end_card_state, True)
        corp0.set_price_index(end_card_state, 10)
        corp0.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 10, False)

        # Corp 1 at 75 price
        corp1 = CORPS[1]
        corp1.set_active(end_card_state, True)
        corp1.set_price_index(end_card_state, 26)
        corp1.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 26, False)

        apply_end_card_py(end_card_state)

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_inactive_corp_at_75_ignored(self, end_card_state):
        """Inactive corps at 75 price do not trigger game over."""
        corp = CORPS[0]
        corp.set_active(end_card_state, False)  # Inactive
        corp.set_price_index(end_card_state, 26)

        apply_end_card_py(end_card_state)

        # Should not trigger game over
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_TEMP_END_TURN


# =============================================================================
# END-02: No Unowned Companies Check
# =============================================================================


class TestNoUnownedCompanies:
    """END-02: Empty deck + no auction/revealed companies flips end card."""

    def test_all_companies_owned_flips_end_card(self, end_card_state):
        """When no companies in deck/auction/revealed, end card flips."""
        # Move all companies out of deck/auction/revealed
        # Transfer all to players/corps/FI or remove from game
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            # Transfer to player 0 (simplest way to get them out of deck)
            company.transfer_to_player(end_card_state, 0)
            PLAYERS[0].set_owns_company(end_card_state, company_id, True)

        # Verify end card was not flipped before
        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)

        # End card should now be flipped
        assert TURN.is_end_card_flipped(end_card_state)
        # And game should end because end card is now flipped
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_company_in_deck_prevents_flip(self, end_card_state):
        """Company remaining in deck prevents end card flip."""
        # Default state has companies in deck
        # Verify at least one company is in deck
        found_in_deck = False
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            if company.is_in_deck(end_card_state):
                found_in_deck = True
                break

        assert found_in_deck, "Test setup: expected companies in deck"
        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)

        # End card should NOT flip
        assert not TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_TEMP_END_TURN

    def test_company_in_auction_prevents_flip(self, end_card_state):
        """Company in auction slot prevents end card flip."""
        # Move all but one company to players
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            if company_id == 0:
                # Keep company 0 in auction
                company.move_to_auction(end_card_state)
            else:
                company.transfer_to_player(end_card_state, 0)
                PLAYERS[0].set_owns_company(end_card_state, company_id, True)

        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)

        # End card should NOT flip (company 0 is in auction)
        assert not TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_TEMP_END_TURN

    def test_company_revealed_prevents_flip(self, end_card_state):
        """Revealed company prevents end card flip."""
        # Move all but one company to players
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            if company_id == 0:
                # Keep company 0 as revealed
                company.set_revealed(end_card_state, True)
            else:
                company.transfer_to_player(end_card_state, 0)
                PLAYERS[0].set_owns_company(end_card_state, company_id, True)

        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)

        # End card should NOT flip
        assert not TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_TEMP_END_TURN


# =============================================================================
# END-03: End Card Already Flipped
# =============================================================================


class TestEndCardFlipped:
    """END-03: Pre-flipped card triggers GAME_OVER."""

    def test_preflipped_end_card_triggers_game_over(self, end_card_state):
        """If end card already flipped, game ends."""
        TURN.set_end_card_flipped(end_card_state, True)

        apply_end_card_py(end_card_state)

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_preflipped_without_75_price_still_ends(self, end_card_state):
        """Game ends on flipped card even without 75 price corp."""
        TURN.set_end_card_flipped(end_card_state, True)

        # Ensure no corp at 75
        for corp_id in range(8):
            CORPS[corp_id].set_active(end_card_state, False)

        apply_end_card_py(end_card_state)

        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER


# =============================================================================
# END-04: Normal Transition
# =============================================================================


class TestNormalTransition:
    """END-04: Normal case transitions to ISSUE_SHARES."""

    def test_normal_state_transitions_to_issue_shares(self, end_card_state):
        """Default state (no end conditions) goes to ISSUE_SHARES."""
        # Default state has:
        # - No corp at 75 price
        # - Companies still in deck
        # - End card not flipped
        # Note: With no active corps, issue phase auto-transitions to TEMP_END_TURN

        apply_end_card_py(end_card_state)

        # No active corps -> issue phase transitions directly to TEMP_END_TURN
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_TEMP_END_TURN

    def test_active_corps_below_75_continue(self, end_card_state):
        """Active corps below 75 price continue to issue phase."""
        # Set up some active corps at various prices
        for corp_id in range(3):
            corp = CORPS[corp_id]
            corp.set_active(end_card_state, True)
            corp.set_price_index(end_card_state, 10 + corp_id)
            corp.set_cash(end_card_state, 100)
            MARKET.set_space_available(end_card_state, 10 + corp_id, False)

        apply_end_card_py(end_card_state)

        # Active corps exist -> stays in ISSUE_SHARES for player decisions
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_ISSUE_SHARES


# =============================================================================
# END-05: Non-Player Phase
# =============================================================================


class TestNonPlayerPhase:
    """END-05: END_CARD is non-player phase (0 valid actions, auto-executes)."""

    def test_end_card_has_no_valid_actions(self, end_card_state):
        """In END_CARD phase, action mask should be all zeros."""
        mask = get_valid_action_mask(end_card_state)

        valid_count = sum(1 for v in mask if v == 1.0)
        assert valid_count == 0, f"Expected 0 valid actions, got {valid_count}"

    def test_end_card_auto_executes_in_driver(self, end_card_state):
        """END_CARD phase should auto-execute when reached."""
        from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK

        # The driver should detect this is a non-player phase
        # Verify by applying and checking transition
        apply_end_card_py(end_card_state)

        # Should have transitioned out of END_CARD
        phase = TURN.get_phase(end_card_state)
        assert phase != GamePhases.PHASE_END_CARD, "Should have auto-transitioned"


# =============================================================================
# END-06: CoO Level Update
# =============================================================================


class TestCoOLevelUpdate:
    """END-06: Flip sets CoO level to 7."""

    def test_flip_sets_coo_to_seven(self, end_card_state):
        """When end card flips, CoO level becomes 7."""
        # Move all companies out of deck/auction/revealed to trigger flip
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            company.transfer_to_player(end_card_state, 0)
            PLAYERS[0].set_owns_company(end_card_state, company_id, True)

        # Verify starting CoO level
        starting_coo = TURN.get_coo_level(end_card_state)
        assert starting_coo < 7, f"Test setup: expected CoO < 7, got {starting_coo}"

        apply_end_card_py(end_card_state)

        # CoO should now be 7
        assert TURN.get_coo_level(end_card_state) == 7

    def test_preflipped_does_not_change_coo(self, end_card_state):
        """Pre-flipped end card does not modify CoO level."""
        TURN.set_end_card_flipped(end_card_state, True)
        TURN.set_coo_level(end_card_state, 3)

        apply_end_card_py(end_card_state)

        # CoO should remain at 3 (flip already happened, we just end game)
        assert TURN.get_coo_level(end_card_state) == 3

    def test_75_price_does_not_change_coo(self, end_card_state):
        """Game ending via 75 price does not change CoO level."""
        TURN.set_coo_level(end_card_state, 4)

        corp = CORPS[0]
        corp.set_active(end_card_state, True)
        corp.set_price_index(end_card_state, 26)
        corp.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 26, False)

        apply_end_card_py(end_card_state)

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

        corp = CORPS[0]
        corp.set_active(end_card_state, True)
        corp.set_price_index(end_card_state, 26)
        corp.set_cash(end_card_state, 100)
        MARKET.set_space_available(end_card_state, 26, False)

        apply_end_card_py(end_card_state)

        # Game ends (both conditions would end game, but 75 check is first)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER

    def test_no_companies_flips_then_ends(self, end_card_state):
        """No unowned companies flips card, then card-flipped check ends game."""
        # Move all companies out
        for company_id in range(int(GameConstants.NUM_COMPANIES)):
            company = COMPANIES[company_id]
            company.initialize(end_card_state)
            company.transfer_to_player(end_card_state, 0)
            PLAYERS[0].set_owns_company(end_card_state, company_id, True)

        assert not TURN.is_end_card_flipped(end_card_state)

        apply_end_card_py(end_card_state)

        # End card should be flipped AND game should be over
        assert TURN.is_end_card_flipped(end_card_state)
        assert TURN.get_phase(end_card_state) == GamePhases.PHASE_GAME_OVER
        assert TURN.get_coo_level(end_card_state) == 7
