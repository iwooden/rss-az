# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Deck entity implementation.

The Deck manages the company draw pile stored in hidden state. It handles
drawing cards, tracking remaining count, and the complex setup rules that
vary by player count.

Deck setup rules (from RULES.md):
1. Game end card at bottom (not stored in deck - implied)
2. Highest face value company of each color is always included
3. Shuffle remaining companies by color, add N per color (N = num_players, with exceptions)
4. Stack: blue on bottom, then green, yellow, orange, red on top
"""

from libc.stdlib cimport rand, srand
from core.state cimport GameState, StateLayout
from core.data cimport GameConstants, get_company_stars


# Company index ranges by color (star tier)
# Red (1★): 0-5, Orange (2★): 6-13, Yellow (3★): 14-21, Green (4★): 22-28, Blue (5★): 29-35
DEF RED_START = 0
DEF RED_END = 6      # exclusive
DEF RED_LAST = 5     # MHE - highest face value red

DEF ORANGE_START = 6
DEF ORANGE_END = 14
DEF ORANGE_LAST = 13  # PR - highest face value orange

DEF YELLOW_START = 14
DEF YELLOW_END = 22
DEF YELLOW_LAST = 21  # DR - highest face value yellow

DEF GREEN_START = 22
DEF GREEN_END = 29
DEF GREEN_LAST = 28   # E - highest face value green

DEF BLUE_START = 29
DEF BLUE_END = 36
DEF BLUE_LAST = 35    # CDG - highest face value blue


cdef class Deck:
    """
    Entity handle for the company draw deck.

    The deck is stored in hidden state as:
    - deck_top: index of top card in deck_order (-1 if empty)
    - deck_order[36]: company IDs in draw order (top of deck at deck_order[deck_top])

    There is only one Deck instance, created at module load.
    """

    def __cinit__(self):
        self._deck_top_offset = 0
        self._deck_order_offset = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.
        """
        cdef StateLayout layout = state._layout

        # Hidden state offsets are already absolute (computed continuing from visible_size)
        self._deck_top_offset = layout.hidden_deck_top_offset
        self._deck_order_offset = layout.hidden_deck_order_offset

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    cpdef int draw(self, GameState state):
        """
        Draw the top card from the deck.

        Returns the company_id of the drawn card, or -1 if deck is empty.
        """
        cdef int top = <int>state._data[self._deck_top_offset]
        cdef int company_id

        if top < 0:
            return -1  # Deck is empty

        company_id = <int>state._data[self._deck_order_offset + top]

        # Move top pointer down
        state._data[self._deck_top_offset] = <float>(top - 1)

        return company_id

    cpdef int peek(self, GameState state):
        """
        Look at the top card without drawing it.

        Returns the company_id of the top card, or -1 if deck is empty.
        """
        cdef int top = <int>state._data[self._deck_top_offset]

        if top < 0:
            return -1

        return <int>state._data[self._deck_order_offset + top]

    cpdef int get_remaining_count(self, GameState state):
        """Get the number of cards remaining in the deck."""
        cdef int top = <int>state._data[self._deck_top_offset]
        return top + 1 if top >= 0 else 0

    cpdef bint is_empty(self, GameState state):
        """Check if the deck is empty."""
        return <int>state._data[self._deck_top_offset] < 0

    # =========================================================================
    # SETUP
    # =========================================================================

    cpdef void setup(self, GameState state, int num_players, int seed):
        """
        Build the deck according to game rules based on player count.

        Rules:
        1. Highest face value of each color always included (MHE, PR, DR, E, CDG)
        2. Add N companies per color (N = num_players), with exceptions:
           - 4 players: add 5 orange (not 4)
           - 5 players: add 7 orange (not 5)
           - 6 players: use ALL companies
        3. Shuffle each color group
        4. Stack: red on top, then orange, yellow, green, blue at bottom
        """
        cdef int[36] deck_cards
        cdef int deck_size = 0
        cdef int i, j, temp
        cdef int red_count, orange_count, yellow_count, green_count, blue_count

        # Determine counts per color
        if num_players == 6:
            # Use all cards
            red_count = RED_END - RED_START
            orange_count = ORANGE_END - ORANGE_START
            yellow_count = YELLOW_END - YELLOW_START
            green_count = GREEN_END - GREEN_START
            blue_count = BLUE_END - BLUE_START
        else:
            red_count = num_players + 1      # +1 because we include the "last" card
            yellow_count = num_players + 1
            green_count = num_players + 1
            blue_count = num_players + 1

            # Orange has special rules
            if num_players == 4:
                orange_count = 6   # 5 + 1 for last card
            elif num_players == 5:
                orange_count = 8   # 7 + 1 for last card (which is all oranges)
            else:
                orange_count = num_players + 1

        # Seed RNG
        srand(seed)

        # Build each color group: always include last card, shuffle rest, take first N-1
        # Then shuffle the whole group

        # === BLUE (bottom of deck) ===
        deck_size = self._add_color_group(deck_cards, deck_size, BLUE_START, BLUE_END, BLUE_LAST, blue_count)

        # === GREEN ===
        deck_size = self._add_color_group(deck_cards, deck_size, GREEN_START, GREEN_END, GREEN_LAST, green_count)

        # === YELLOW ===
        deck_size = self._add_color_group(deck_cards, deck_size, YELLOW_START, YELLOW_END, YELLOW_LAST, yellow_count)

        # === ORANGE ===
        deck_size = self._add_color_group(deck_cards, deck_size, ORANGE_START, ORANGE_END, ORANGE_LAST, orange_count)

        # === RED (top of deck) ===
        deck_size = self._add_color_group(deck_cards, deck_size, RED_START, RED_END, RED_LAST, red_count)

        # Write to state
        state._data[self._deck_top_offset] = <float>(deck_size - 1)
        for i in range(deck_size):
            state._data[self._deck_order_offset + i] = <float>deck_cards[i]

        # Clear remaining slots (not strictly necessary but clean)
        for i in range(deck_size, 36):
            state._data[self._deck_order_offset + i] = -1.0

    cdef int _add_color_group(self, int* deck_cards, int deck_size, int start, int end, int last_idx, int count):
        """
        Add a color group to the deck.

        Args:
            deck_cards: output array
            deck_size: current size of deck_cards
            start: start index of this color in company array
            end: end index (exclusive) of this color
            last_idx: index of the "last" company (highest face value, always included)
            count: total cards to include from this color

        Returns:
            New deck_size after adding cards
        """
        cdef int[8] pool  # Max 8 companies per color
        cdef int pool_size = 0
        cdef int i, j, temp
        cdef int group_start = deck_size

        # If count >= all available, use all
        if count >= end - start:
            for i in range(start, end):
                deck_cards[deck_size] = i
                deck_size += 1
        else:
            # Build pool of non-last cards
            for i in range(start, end):
                if i != last_idx:
                    pool[pool_size] = i
                    pool_size += 1

            # Shuffle pool (Fisher-Yates)
            for i in range(pool_size - 1, 0, -1):
                j = rand() % (i + 1)
                temp = pool[i]
                pool[i] = pool[j]
                pool[j] = temp

            # Take first (count - 1) from shuffled pool
            for i in range(count - 1):
                deck_cards[deck_size] = pool[i]
                deck_size += 1

            # Always include the last card
            deck_cards[deck_size] = last_idx
            deck_size += 1

        # Shuffle this color group within the deck
        cdef int group_size = deck_size - group_start
        for i in range(group_size - 1, 0, -1):
            j = rand() % (i + 1)
            temp = deck_cards[group_start + i]
            deck_cards[group_start + i] = deck_cards[group_start + j]
            deck_cards[group_start + j] = temp

        return deck_size

    # =========================================================================
    # DEBUG/TESTING HELPERS
    # =========================================================================

    cpdef list get_order(self, GameState state):
        """
        Get the current deck order as a Python list (for debugging/testing).

        Returns list of company IDs from bottom to top of deck.
        """
        cdef int top = <int>state._data[self._deck_top_offset]
        cdef list result = []
        cdef int i

        for i in range(top + 1):
            result.append(<int>state._data[self._deck_order_offset + i])

        return result

    cpdef void set_order(self, GameState state, list order):
        """
        Set the deck order from a Python list (for testing).

        Order should be from bottom to top (index 0 = bottom, last = top).
        """
        cdef int i
        cdef int size = len(order)

        state._data[self._deck_top_offset] = <float>(size - 1)

        for i in range(size):
            state._data[self._deck_order_offset + i] = <float>order[i]

        # Clear remaining
        for i in range(size, 36):
            state._data[self._deck_order_offset + i] = -1.0


# =============================================================================
# GLOBAL DECK INSTANCE
# =============================================================================

DECK = Deck()
