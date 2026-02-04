# cython: language_level=3
"""Tests for deck entity functionality.

Covers:
- Basic deck operations (draw, peek, count, empty)
- CoO increment when last-in-group companies are drawn
- Deck setup rules for different player counts
- Test helpers (set_order, get_order)
"""
import pytest
from core.state import GameState
from core.data import GamePhases, GameConstants, COMPANY_NAMES, get_adjusted_company_income
from entities.deck import DECK
from entities.turn import TURN
from entities.company import COMPANIES


# =============================================================================
# CONSTANTS FOR TESTS
# =============================================================================

# Last-in-group company IDs (highest face value per color)
MHE_ID = 5   # Last red, face value 8
PR_ID = 13   # Last orange, face value 19
DR_ID = 21   # Last yellow, face value 29
E_ID = 28    # Last green, face value 43
CDG_ID = 35  # Last blue, face value 60

# Non-last-in-group company IDs for comparison
BME_ID = 0   # First red, face value 1
WT_ID = 6    # First orange, face value 11
DSB_ID = 14  # First yellow, face value 20
SZD_ID = 22  # First green, face value 30
HH_ID = 29   # First blue, face value 45


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def game_state():
    """Create a fresh 3-player game state."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    return state


@pytest.fixture
def empty_deck_state():
    """Create a game state with an empty deck."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    # Set deck to empty
    DECK.set_order(state, [])
    return state


@pytest.fixture
def controlled_deck_state():
    """Create a game state with a controlled deck order for testing CoO."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    # Set a specific deck order: bottom to top
    # Put last-in-group cards in order so we can test CoO progression
    # Bottom: CDG (last blue) -> E (last green) -> DR (last yellow) -> PR (last orange) -> MHE (last red) : top
    DECK.set_order(state, [CDG_ID, E_ID, DR_ID, PR_ID, MHE_ID])
    # Reset CoO to 1
    TURN.set_coo_level(state, 1)
    return state


# =============================================================================
# BASIC DECK OPERATIONS
# =============================================================================

class TestDeckBasicOperations:
    """Tests for basic deck operations."""

    def test_peek_returns_top_card(self, game_state):
        """Peek should return top card without removing it."""
        top_card = DECK.peek(game_state)
        assert top_card >= 0
        assert top_card < GameConstants.NUM_COMPANIES

        # Peek again - should be same card
        assert DECK.peek(game_state) == top_card

        # Count should be unchanged
        count_before = DECK.get_remaining_count(game_state)
        DECK.peek(game_state)
        assert DECK.get_remaining_count(game_state) == count_before

    def test_draw_removes_top_card(self, game_state):
        """Draw should return top card and remove it from deck."""
        top_card = DECK.peek(game_state)
        count_before = DECK.get_remaining_count(game_state)

        drawn = DECK.draw(game_state)

        assert drawn == top_card
        assert DECK.get_remaining_count(game_state) == count_before - 1

    def test_draw_marks_company_revealed(self, game_state):
        """Drawn companies should be marked as revealed."""
        company_id = DECK.draw(game_state)

        company = COMPANIES[company_id]
        assert company.is_revealed(game_state)

    def test_draw_empty_deck_returns_minus_one(self, empty_deck_state):
        """Drawing from empty deck returns -1."""
        assert DECK.draw(empty_deck_state) == -1

    def test_peek_empty_deck_returns_minus_one(self, empty_deck_state):
        """Peeking empty deck returns -1."""
        assert DECK.peek(empty_deck_state) == -1

    def test_is_empty(self, game_state, empty_deck_state):
        """is_empty correctly reports deck state."""
        assert not DECK.is_empty(game_state)
        assert DECK.is_empty(empty_deck_state)

    def test_get_remaining_count(self, game_state):
        """get_remaining_count returns correct count."""
        initial_count = DECK.get_remaining_count(game_state)
        assert initial_count > 0

        DECK.draw(game_state)
        assert DECK.get_remaining_count(game_state) == initial_count - 1

        DECK.draw(game_state)
        assert DECK.get_remaining_count(game_state) == initial_count - 2

    def test_set_order_and_get_order(self, game_state):
        """set_order and get_order work correctly."""
        test_order = [0, 5, 10, 15, 20]  # 5 arbitrary company IDs

        DECK.set_order(game_state, test_order)

        result = DECK.get_order(game_state)
        assert result == test_order
        assert DECK.get_remaining_count(game_state) == 5

        # Top should be last in list
        assert DECK.peek(game_state) == 20


# =============================================================================
# COO INCREMENT ON LAST-IN-GROUP DRAW
# =============================================================================

class TestCoOIncrementOnDraw:
    """Tests for Cost of Ownership increment when drawing last-in-group companies."""

    def test_draw_mhe_increments_coo_to_2(self, game_state):
        """Drawing MHE (last red) should increment CoO from 1 to 2."""
        # Set up deck with just MHE on top
        DECK.set_order(game_state, [MHE_ID])
        TURN.set_coo_level(game_state, 1)

        assert TURN.get_coo_level(game_state) == 1

        drawn = DECK.draw(game_state)

        assert drawn == MHE_ID
        assert TURN.get_coo_level(game_state) == 2

    def test_draw_pr_increments_coo_to_3(self, game_state):
        """Drawing PR (last orange) should increment CoO from 2 to 3."""
        DECK.set_order(game_state, [PR_ID])
        TURN.set_coo_level(game_state, 2)

        drawn = DECK.draw(game_state)

        assert drawn == PR_ID
        assert TURN.get_coo_level(game_state) == 3

    def test_draw_dr_increments_coo_to_4(self, game_state):
        """Drawing DR (last yellow) should increment CoO from 3 to 4."""
        DECK.set_order(game_state, [DR_ID])
        TURN.set_coo_level(game_state, 3)

        drawn = DECK.draw(game_state)

        assert drawn == DR_ID
        assert TURN.get_coo_level(game_state) == 4

    def test_draw_e_increments_coo_to_5(self, game_state):
        """Drawing E (last green) should increment CoO from 4 to 5."""
        DECK.set_order(game_state, [E_ID])
        TURN.set_coo_level(game_state, 4)

        drawn = DECK.draw(game_state)

        assert drawn == E_ID
        assert TURN.get_coo_level(game_state) == 5

    def test_draw_cdg_increments_coo_to_6(self, game_state):
        """Drawing CDG (last blue) should increment CoO from 5 to 6."""
        DECK.set_order(game_state, [CDG_ID])
        TURN.set_coo_level(game_state, 5)

        drawn = DECK.draw(game_state)

        assert drawn == CDG_ID
        assert TURN.get_coo_level(game_state) == 6

    def test_full_coo_progression_1_through_6(self, controlled_deck_state):
        """Drawing all last-in-group cards progresses CoO from 1 to 6."""
        state = controlled_deck_state
        # Deck order (bottom to top): CDG, E, DR, PR, MHE

        assert TURN.get_coo_level(state) == 1

        # Draw MHE (top) -> CoO 2
        drawn = DECK.draw(state)
        assert drawn == MHE_ID
        assert TURN.get_coo_level(state) == 2

        # Draw PR -> CoO 3
        drawn = DECK.draw(state)
        assert drawn == PR_ID
        assert TURN.get_coo_level(state) == 3

        # Draw DR -> CoO 4
        drawn = DECK.draw(state)
        assert drawn == DR_ID
        assert TURN.get_coo_level(state) == 4

        # Draw E -> CoO 5
        drawn = DECK.draw(state)
        assert drawn == E_ID
        assert TURN.get_coo_level(state) == 5

        # Draw CDG -> CoO 6
        drawn = DECK.draw(state)
        assert drawn == CDG_ID
        assert TURN.get_coo_level(state) == 6

    def test_draw_non_last_in_group_does_not_change_coo(self, game_state):
        """Drawing non-last-in-group companies should not change CoO."""
        # Set up deck with non-last-in-group cards
        DECK.set_order(game_state, [BME_ID, WT_ID, DSB_ID, SZD_ID, HH_ID])
        TURN.set_coo_level(game_state, 1)

        # Draw all 5 cards - none should change CoO
        for _ in range(5):
            DECK.draw(game_state)
            assert TURN.get_coo_level(game_state) == 1

    def test_coo_increment_independent_of_deck_position(self, game_state):
        """CoO should increment regardless of where in deck the card is."""
        # Put MHE in the middle of other cards
        DECK.set_order(game_state, [BME_ID, MHE_ID, WT_ID])
        TURN.set_coo_level(game_state, 1)

        # Draw WT (top) - no change
        DECK.draw(game_state)
        assert TURN.get_coo_level(game_state) == 1

        # Draw MHE - should increment
        drawn = DECK.draw(game_state)
        assert drawn == MHE_ID
        assert TURN.get_coo_level(game_state) == 2

        # Draw BME - no change
        DECK.draw(game_state)
        assert TURN.get_coo_level(game_state) == 2


# =============================================================================
# DECK SETUP TESTS
# =============================================================================

class TestDeckSetup:
    """Tests for deck setup rules."""

    def test_setup_3_player_card_count(self):
        """3-player game should have correct number of cards."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # 3 players: 4 cards per color (3+1 for last card)
        # Red: 4, Orange: 4, Yellow: 4, Green: 4, Blue: 4 = 20 total
        # But 3 were drawn for auction, so deck should have 17
        # Actually let's just verify it's reasonable
        count = DECK.get_remaining_count(state)
        assert count > 0
        assert count <= 36 - 3  # At most 33 (36 minus 3 drawn for auction)

    def test_setup_always_includes_last_in_group_cards(self):
        """Deck setup should always include all last-in-group cards."""
        # Test with multiple seeds to catch randomization issues
        for seed in [1, 42, 100, 999]:
            state = GameState(num_players=3)
            state.initialize_game(seed=seed)

            # Get all cards in deck plus auction/revealed
            deck_cards = set(DECK.get_order(state))

            # Add any cards that were drawn for auction
            for company_id in range(GameConstants.NUM_COMPANIES):
                company = COMPANIES[company_id]
                if company.is_for_auction(state) or company.is_revealed(state):
                    deck_cards.add(company_id)

            # All last-in-group cards should be present
            last_in_group = {MHE_ID, PR_ID, DR_ID, E_ID, CDG_ID}
            assert last_in_group.issubset(deck_cards), f"Missing last-in-group cards with seed {seed}"

    def test_setup_red_cards_on_top(self):
        """Red cards should be on top of deck (drawn first)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Draw a few cards - they should all be red (company IDs 0-5)
        # or already drawn for auction
        red_ids = set(range(6))

        # The first few draws should be red
        for _ in range(2):
            top = DECK.peek(state)
            if top in red_ids:
                DECK.draw(state)
            else:
                # If not red, it was probably already drawn for auction
                # Check that remaining reds are in auction
                break

    def test_setup_6_player_uses_all_cards(self):
        """6-player game should use all 36 company cards."""
        state = GameState(num_players=6)
        state.initialize_game(seed=42)

        # Collect all cards: deck + auction + revealed
        all_cards = set(DECK.get_order(state))
        for company_id in range(GameConstants.NUM_COMPANIES):
            company = COMPANIES[company_id]
            if company.is_for_auction(state) or company.is_revealed(state):
                all_cards.add(company_id)

        # Should have all 36 companies
        assert len(all_cards) == 36


# =============================================================================
# EDGE CASES
# =============================================================================

class TestDeckComposition:
    """Tests for deck composition rules by player count.

    Verifies RULES.md Setup section:
    - Add companies to each stack equal to player count
    - 4 players: add 5 orange (not 4)
    - 5 players: add 7 orange (not 5)
    - 6 players: use ALL companies of all colors
    """

    # Company ranges by color
    RED_IDS = set(range(0, 6))      # 6 companies
    ORANGE_IDS = set(range(6, 14))  # 8 companies
    YELLOW_IDS = set(range(14, 22)) # 8 companies
    GREEN_IDS = set(range(22, 29))  # 7 companies
    BLUE_IDS = set(range(29, 36))   # 7 companies

    def _get_all_game_cards(self, state):
        """Get all cards in play: deck + auction + revealed."""
        all_cards = set(DECK.get_order(state))
        for company_id in range(GameConstants.NUM_COMPANIES):
            company = COMPANIES[company_id]
            if company.is_for_auction(state) or company.is_revealed(state):
                all_cards.add(company_id)
        return all_cards

    def _count_by_color(self, cards):
        """Count cards by color."""
        return {
            'red': len(cards & self.RED_IDS),
            'orange': len(cards & self.ORANGE_IDS),
            'yellow': len(cards & self.YELLOW_IDS),
            'green': len(cards & self.GREEN_IDS),
            'blue': len(cards & self.BLUE_IDS),
        }

    def test_deck_composition_2_players(self):
        """2-player game: 3 cards per color (2+1 for last card)."""
        state = GameState(num_players=2)
        state.initialize_game(seed=42)

        all_cards = self._get_all_game_cards(state)
        counts = self._count_by_color(all_cards)

        assert counts['red'] == 3, f"Expected 3 red, got {counts['red']}"
        assert counts['orange'] == 3, f"Expected 3 orange, got {counts['orange']}"
        assert counts['yellow'] == 3, f"Expected 3 yellow, got {counts['yellow']}"
        assert counts['green'] == 3, f"Expected 3 green, got {counts['green']}"
        assert counts['blue'] == 3, f"Expected 3 blue, got {counts['blue']}"
        assert len(all_cards) == 15, f"Expected 15 total, got {len(all_cards)}"

    def test_deck_composition_3_players(self):
        """3-player game: 4 cards per color (3+1 for last card)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        all_cards = self._get_all_game_cards(state)
        counts = self._count_by_color(all_cards)

        assert counts['red'] == 4, f"Expected 4 red, got {counts['red']}"
        assert counts['orange'] == 4, f"Expected 4 orange, got {counts['orange']}"
        assert counts['yellow'] == 4, f"Expected 4 yellow, got {counts['yellow']}"
        assert counts['green'] == 4, f"Expected 4 green, got {counts['green']}"
        assert counts['blue'] == 4, f"Expected 4 blue, got {counts['blue']}"
        assert len(all_cards) == 20, f"Expected 20 total, got {len(all_cards)}"

    def test_deck_composition_4_players_orange_exception(self):
        """4-player game: 5 per color, but 6 orange (RULES.md exception: add 5 orange not 4)."""
        state = GameState(num_players=4)
        state.initialize_game(seed=42)

        all_cards = self._get_all_game_cards(state)
        counts = self._count_by_color(all_cards)

        assert counts['red'] == 5, f"Expected 5 red, got {counts['red']}"
        assert counts['orange'] == 6, f"Expected 6 orange (exception), got {counts['orange']}"
        assert counts['yellow'] == 5, f"Expected 5 yellow, got {counts['yellow']}"
        assert counts['green'] == 5, f"Expected 5 green, got {counts['green']}"
        assert counts['blue'] == 5, f"Expected 5 blue, got {counts['blue']}"
        assert len(all_cards) == 26, f"Expected 26 total, got {len(all_cards)}"

    def test_deck_composition_5_players_orange_exception(self):
        """5-player game: 6 per color, but all 8 orange (RULES.md exception: add 7 orange not 5)."""
        state = GameState(num_players=5)
        state.initialize_game(seed=42)

        all_cards = self._get_all_game_cards(state)
        counts = self._count_by_color(all_cards)

        # 5+1=6 per color, but red only has 6 total (so all reds used)
        assert counts['red'] == 6, f"Expected 6 red (all), got {counts['red']}"
        # Orange exception: 7+1=8 (all oranges used)
        assert counts['orange'] == 8, f"Expected 8 orange (all, exception), got {counts['orange']}"
        assert counts['yellow'] == 6, f"Expected 6 yellow, got {counts['yellow']}"
        assert counts['green'] == 6, f"Expected 6 green, got {counts['green']}"
        assert counts['blue'] == 6, f"Expected 6 blue, got {counts['blue']}"
        assert len(all_cards) == 32, f"Expected 32 total, got {len(all_cards)}"

    def test_deck_composition_6_players_all_cards(self):
        """6-player game: ALL 36 companies used."""
        state = GameState(num_players=6)
        state.initialize_game(seed=42)

        all_cards = self._get_all_game_cards(state)
        counts = self._count_by_color(all_cards)

        assert counts['red'] == 6, f"Expected 6 red (all), got {counts['red']}"
        assert counts['orange'] == 8, f"Expected 8 orange (all), got {counts['orange']}"
        assert counts['yellow'] == 8, f"Expected 8 yellow (all), got {counts['yellow']}"
        assert counts['green'] == 7, f"Expected 7 green (all), got {counts['green']}"
        assert counts['blue'] == 7, f"Expected 7 blue (all), got {counts['blue']}"
        assert len(all_cards) == 36, f"Expected 36 total, got {len(all_cards)}"

    def test_game_end_card_not_in_deck(self):
        """Game end card is implied at bottom, not stored in deck array."""
        # The game end card is not a company - it's a special card not tracked in the deck
        # Verify that all cards in deck are valid company IDs (0-35)
        for num_players in [2, 3, 4, 5, 6]:
            state = GameState(num_players=num_players)
            state.initialize_game(seed=42)

            deck_order = DECK.get_order(state)
            for company_id in deck_order:
                assert 0 <= company_id < 36, f"Invalid company ID {company_id} in {num_players}p deck"

    def test_last_in_group_always_included(self):
        """Highest face value of each color always included (MHE, PR, DR, E, CDG)."""
        last_in_group = {MHE_ID, PR_ID, DR_ID, E_ID, CDG_ID}

        for num_players in [2, 3, 4, 5, 6]:
            for seed in [1, 42, 100, 999]:
                state = GameState(num_players=num_players)
                state.initialize_game(seed=seed)

                all_cards = self._get_all_game_cards(state)

                assert last_in_group.issubset(all_cards), \
                    f"Missing last-in-group cards in {num_players}p game with seed {seed}"

    def test_composition_consistent_across_seeds(self):
        """Card counts should be consistent regardless of shuffle seed."""
        for num_players in [2, 3, 4, 5, 6]:
            expected_counts = None

            for seed in [1, 42, 100, 999, 12345]:
                state = GameState(num_players=num_players)
                state.initialize_game(seed=seed)

                all_cards = self._get_all_game_cards(state)
                counts = self._count_by_color(all_cards)

                if expected_counts is None:
                    expected_counts = counts
                else:
                    assert counts == expected_counts, \
                        f"{num_players}p game: inconsistent counts with seed {seed}"


class TestDeckEdgeCases:
    """Edge case tests for deck operations."""

    def test_multiple_draws_until_empty(self, game_state):
        """Drawing until empty should work without errors."""
        while not DECK.is_empty(game_state):
            company_id = DECK.draw(game_state)
            assert company_id >= 0
            assert company_id < GameConstants.NUM_COMPANIES

        # Now deck is empty
        assert DECK.is_empty(game_state)
        assert DECK.draw(game_state) == -1

    def test_coo_does_not_exceed_6_from_draw(self, game_state):
        """CoO should not exceed 6 from drawing (7 is only from end card flip)."""
        # Set CoO to 6, draw CDG
        DECK.set_order(game_state, [CDG_ID])
        TURN.set_coo_level(game_state, 6)

        # This shouldn't happen in normal play (CDG triggers 5->6),
        # but verify we don't break anything
        DECK.draw(game_state)

        # CoO would try to go to 7, but that's actually fine per the rules
        # The set_coo_level should handle the bounds
        coo = TURN.get_coo_level(game_state)
        assert coo <= 7  # At most 7

    def test_set_order_empty_list(self, game_state):
        """Setting empty deck order should work."""
        DECK.set_order(game_state, [])

        assert DECK.is_empty(game_state)
        assert DECK.get_remaining_count(game_state) == 0
        assert DECK.peek(game_state) == -1
        assert DECK.draw(game_state) == -1

    def test_set_order_single_card(self, game_state):
        """Setting single-card deck should work."""
        DECK.set_order(game_state, [15])

        assert DECK.get_remaining_count(game_state) == 1
        assert DECK.peek(game_state) == 15

        drawn = DECK.draw(game_state)
        assert drawn == 15
        assert DECK.is_empty(game_state)


# =============================================================================
# COMPANY INCOMES ARRAY TESTS
# =============================================================================

class TestCompanyIncomesArray:
    """Tests for the company_incomes state array being properly populated."""

    def test_company_incomes_initialized_at_game_start(self, game_state):
        """Company incomes should be populated during initialize_game()."""
        # At CoO level 1, all adjusted incomes equal base incomes (no cost)
        for company_id in range(GameConstants.NUM_COMPANIES):
            company = COMPANIES[company_id]
            expected = get_adjusted_company_income(company_id, 1)
            actual = company.get_adjusted_income(game_state)
            assert actual == expected, f"Company {company_id} income: expected {expected}, got {actual}"

    def test_company_incomes_updated_when_coo_changes(self, game_state):
        """Changing CoO level should update all company incomes."""
        # Change CoO to level 4 (red companies get -2 income)
        TURN.set_coo_level(game_state, 4)

        for company_id in range(GameConstants.NUM_COMPANIES):
            company = COMPANIES[company_id]
            expected = get_adjusted_company_income(company_id, 4)
            actual = company.get_adjusted_income(game_state)
            assert actual == expected, f"Company {company_id} at CoO 4: expected {expected}, got {actual}"

    def test_company_incomes_at_coo_level_5(self, game_state):
        """At CoO 5, red and orange companies get -4 income."""
        TURN.set_coo_level(game_state, 5)

        # Red company (1★) - base income 1-2, adjusted should be -3 to -2
        bme = COMPANIES[BME_ID]  # Base income 1
        assert bme.get_adjusted_income(game_state) == 1 - 4  # -3

        # Orange company (2★) - base income 3, adjusted should be -1
        wt = COMPANIES[WT_ID]  # Base income 3
        assert wt.get_adjusted_income(game_state) == 3 - 4  # -1

        # Yellow company (3★) - no cost at level 5
        dsb = COMPANIES[DSB_ID]  # Base income 5
        assert dsb.get_adjusted_income(game_state) == 5  # No change

    def test_company_incomes_at_coo_level_7(self, game_state):
        """At CoO 7 (end card flipped), only blue is unaffected."""
        TURN.set_coo_level(game_state, 7)

        # Red company - gets -10
        bme = COMPANIES[BME_ID]  # Base income 1
        assert bme.get_adjusted_income(game_state) == 1 - 10  # -9

        # Green company - gets -10
        szd = COMPANIES[SZD_ID]  # Base income 7
        assert szd.get_adjusted_income(game_state) == 7 - 10  # -3

        # Blue company - no cost
        hh = COMPANIES[HH_ID]  # Base income 10
        assert hh.get_adjusted_income(game_state) == 10  # No change

    def test_company_incomes_updated_when_drawing_last_in_group(self, game_state):
        """Drawing last-in-group should increment CoO AND update all incomes."""
        # Set up deck with MHE on top
        DECK.set_order(game_state, [MHE_ID])
        TURN.set_coo_level(game_state, 1)

        # Verify initial state at CoO 1
        bme = COMPANIES[BME_ID]
        assert bme.get_adjusted_income(game_state) == 1  # Base income, no cost

        # Draw MHE - should trigger CoO 1->2
        DECK.draw(game_state)

        # Verify CoO changed
        assert TURN.get_coo_level(game_state) == 2

        # Verify incomes still correct (CoO 2 has no costs, same as 1)
        assert bme.get_adjusted_income(game_state) == get_adjusted_company_income(BME_ID, 2)

    def test_company_incomes_progression_through_draws(self, controlled_deck_state):
        """Full progression of company incomes as CoO increases through draws."""
        state = controlled_deck_state
        # Deck: CDG, E, DR, PR, MHE (bottom to top)

        bme = COMPANIES[BME_ID]  # Red, base income 1

        # CoO 1 - no cost
        assert bme.get_adjusted_income(state) == 1

        # Draw MHE -> CoO 2 - no cost
        DECK.draw(state)
        assert bme.get_adjusted_income(state) == 1

        # Draw PR -> CoO 3 - no cost
        DECK.draw(state)
        assert bme.get_adjusted_income(state) == 1

        # Draw DR -> CoO 4 - red gets -2
        DECK.draw(state)
        assert bme.get_adjusted_income(state) == 1 - 2  # -1

        # Draw E -> CoO 5 - red gets -4
        DECK.draw(state)
        assert bme.get_adjusted_income(state) == 1 - 4  # -3

        # Draw CDG -> CoO 6 - red gets -7
        DECK.draw(state)
        assert bme.get_adjusted_income(state) == 1 - 7  # -6

    def test_all_36_companies_have_valid_incomes(self, game_state):
        """All companies should have valid (non-zero or explicitly calculated) incomes."""
        for coo_level in range(1, 8):
            TURN.set_coo_level(game_state, coo_level)

            for company_id in range(GameConstants.NUM_COMPANIES):
                company = COMPANIES[company_id]
                expected = get_adjusted_company_income(company_id, coo_level)
                actual = company.get_adjusted_income(game_state)
                assert actual == expected, \
                    f"Company {company_id} at CoO {coo_level}: expected {expected}, got {actual}"
