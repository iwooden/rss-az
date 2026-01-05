"""Tests for End Card phase."""

import pytest
from state import GameState
from phases.endcard import EndCardPhase, get_constants
from data import py_get_corp_share_count

from tests.test_common import (
    StateBuilder, PHASE_DIVIDENDS, PHASE_END_CARD, PHASE_ISSUE_SHARES, PHASE_GAME_OVER,
    NUM_CORPS
)

# Market price index for 75
MARKET_INDEX_75 = 26


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in END_CARD phase."""
    s = GameState(3)
    s.phase = PHASE_END_CARD
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get End Card phase handler for 3 players."""
    return EndCardPhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# STATE ACCESSOR TESTS
# =============================================================================

class TestEndCardFlipped:
    """Test end_card_flipped state accessors."""

    def test_initial_state_not_flipped(self, state):
        """End card starts not flipped."""
        assert not state.get_end_card_flipped_py()

    def test_set_flipped(self, state):
        """Can set end card to flipped."""
        state.set_end_card_flipped_py(True)
        assert state.get_end_card_flipped_py()

    def test_set_not_flipped(self, state):
        """Can set end card to not flipped."""
        state.set_end_card_flipped_py(True)
        state.set_end_card_flipped_py(False)
        assert not state.get_end_card_flipped_py()


class TestPlayerNetWorth:
    """Test player net worth state accessors."""

    def test_initial_net_worth_zero(self, state):
        """Player net worth starts at 0."""
        assert state.get_player_net_worth_py(0) == 0

    def test_set_net_worth(self, state):
        """Can set player net worth."""
        state.set_player_net_worth_py(0, 150)
        assert state.get_player_net_worth_py(0) == 150

    def test_get_final_scores(self, state, builder):
        """get_final_scores returns sorted scores with tie-breaker."""
        builder.set_player_net_worth(0, 100)
        builder.set_player_net_worth(1, 150)
        builder.set_player_net_worth(2, 150)  # Tie with player 1
        builder.init_default_turn_order()

        scores = state.get_final_scores()

        # Player 1 and 2 tied at 150, player 1 wins tie by turn order
        assert scores[0] == (1, 150)
        assert scores[1] == (2, 150)
        assert scores[2] == (0, 100)


# =============================================================================
# GAME END CONDITION TESTS
# =============================================================================

class TestGameEndConditions:
    """Test game end condition checks."""

    def test_game_ends_if_corp_at_75(self, state, handler, builder):
        """Game ends if any corp has share price 75."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, MARKET_INDEX_75)

        handler.handle_end_card_phase(state)

        assert state.phase == PHASE_GAME_OVER

    def test_game_ends_if_end_card_already_flipped(self, state, handler, builder):
        """Game ends if end card was already flipped."""
        state.set_end_card_flipped_py(True)

        handler.handle_end_card_phase(state)

        assert state.phase == PHASE_GAME_OVER

    def test_no_game_end_if_no_conditions_met(self, state, handler, builder):
        """Game doesn't end if no conditions are met."""
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 10)  # Price index 10 = $14
        # Add a company for auction so end card doesn't flip
        builder.set_company_for_auction(0, True)

        handler.handle_end_card_phase(state)

        assert state.phase == PHASE_ISSUE_SHARES


# =============================================================================
# END CARD FLIP TESTS
# =============================================================================

class TestEndCardFlip:
    """Test end card flip logic."""

    def test_flips_when_no_auction_companies_and_empty_deck(self, state, handler, builder):
        """End card flips when no auction companies and deck is empty."""
        # Deck is empty by default (deck_top < 0)
        # No auction companies by default

        handler.handle_end_card_phase(state)

        assert state.get_end_card_flipped_py()
        assert state.coo_level == 7  # Level 7 = 10 cost

    def test_no_flip_if_auction_companies_exist(self, state, handler, builder):
        """End card doesn't flip if there are auction companies."""
        builder.set_company_for_auction(0, True)

        handler.handle_end_card_phase(state)

        assert not state.get_end_card_flipped_py()
        assert state.phase == PHASE_ISSUE_SHARES


# =============================================================================
# ISSUE PHASE SETUP TESTS
# =============================================================================

class TestIssuePhaseSetup:
    """Test setup for Issue Shares phase."""

    def test_transitions_to_issue_shares(self, state, handler, builder):
        """End Card phase transitions to Issue Shares."""
        builder.set_company_for_auction(0, True)  # Prevent game end

        handler.handle_end_card_phase(state)

        assert state.phase == PHASE_ISSUE_SHARES

    def test_marks_corps_that_can_issue(self, state, handler, builder):
        """Corps that can issue are marked in issue_remaining."""
        builder.set_company_for_auction(0, True)  # Prevent game end
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 10)  # Set a valid price (not 75)
        total_shares = py_get_corp_share_count(0)  # JS has 7 shares
        # Use state accessors for issued/unissued shares
        state.set_corp_issued_shares_py(0, 2)  # Only 2 issued, can issue more
        state.set_corp_unissued_shares_py(0, total_shares - 2)

        handler.handle_end_card_phase(state)

        # Corp 0 should be marked as able to issue
        assert builder.get_turn_issue_remaining(0) == 1.0

    def test_corps_fully_issued_not_marked(self, state, handler, builder):
        """Corps that are fully issued are not marked."""
        builder.set_company_for_auction(0, True)  # Prevent game end
        builder.set_corp_active(0, True)
        builder.set_corp_price_index(0, 10)  # Set a valid price (not 75)
        total_shares = py_get_corp_share_count(0)  # JS has 7 shares
        # Use state accessors for issued/unissued shares
        state.set_corp_issued_shares_py(0, total_shares)  # All issued
        state.set_corp_unissued_shares_py(0, 0)

        handler.handle_end_card_phase(state)

        # Corp 0 should not be marked as able to issue
        assert builder.get_turn_issue_remaining(0) == 0.0

    def test_inactive_corps_not_marked(self, state, handler, builder):
        """Inactive corps are not marked for issuing."""
        builder.set_company_for_auction(0, True)  # Prevent game end
        builder.set_corp_active(0, False)

        handler.handle_end_card_phase(state)

        # Corp 0 should be marked as inactive (-1)
        assert builder.get_turn_issue_remaining(0) == -1.0


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestConstants:
    """Test that constants are accessible."""

    def test_get_constants(self):
        """Can access phase constants."""
        constants = get_constants()
        assert constants['MAX_SHARE_PRICE'] == 75
        assert constants['COO_LEVEL_END_CARD_FLIPPED'] == 7
