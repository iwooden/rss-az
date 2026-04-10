"""
Deck entity implementation.

Manages the company draw pile inside the compact GameState. The deck
entity owns all mutations of the deck section itself (`top` + `order[]`):
draws pop here, arbitrary removals splice here, and test-helper deck
rewrites normalize live-deck company locations here. Crossing a colour
boundary on draw increments the cost-of-ownership level on TurnState.

Deck setup rules (from RULES.md):
1. Game end card at bottom (not stored in deck — implied)
2. Highest face value company of each colour is always included
3. Shuffle the rest of each colour and add N per colour (N = num_players,
   with exceptions for orange and the 6-player "use everything" case)
4. Stack: blue on bottom, then green, yellow, orange, red on top

Companies that did NOT make it into the live deck for the active player
count are marked LOC_EXCLUDED so the engine can distinguish "in the live
deck" from "filtered out at setup for this player count".
"""

from libc.stdint cimport int16_t
from libc.stdlib cimport rand, srand

from core.state cimport GameState, LAYOUT, DECK_OFFSETS
from core.data cimport (
    GameConstants,
    COMPANY_STARS,
)
from entities.company cimport Company, LOC_DECK, LOC_REVEALED, LOC_EXCLUDED
from entities import company as company_module
from entities import turn as turn_module


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

    The deck is stored in the compact state as:
    - top: index of the top card in order (-1 if empty)
    - order[36]: company IDs in draw order; top of deck at order[top]

    Both live inside the deck section, reached via
    ``LAYOUT.deck_offset + DECK_OFFSETS.<field>``.

    There is only one Deck instance, created at module load. It is
    stateless — every read derives its slot inline from the module-level
    ``LAYOUT`` and ``DECK_OFFSETS`` constants on ``core.state``.
    """

    # =========================================================================
    # BASIC OPERATIONS
    # =========================================================================

    cdef void _sync_cards_remaining(self, GameState state):
        """Mirror deck top into TurnState.cards_remaining."""
        cdef int top = <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top]
        cdef int remaining = top + 1 if top >= 0 else 0
        turn_module.TURN.set_cards_remaining(state, remaining)

    cpdef int draw(self, GameState state):
        """
        Draw the top card from the deck and mark it revealed.

        Drawn companies are always marked LOC_REVEALED (unavailable for
        auction) until they are explicitly made available at the end of
        the WRAP_UP phase. The deck pops the top card itself before
        updating the company's semantic location, so draw no longer
        depends on company-side location state to keep the deck coherent.

        Cost of Ownership is determined by the back of the top deck card
        (RULES.md §10). When drawing causes the new top card to be a
        different colour tier than the drawn card, the CoO level
        increments — the deck has transitioned to the next colour group.
        Drawing the last card also increments CoO (game-end card exposed).

        Returns the company_id of the drawn card, or -1 if the deck is
        empty.
        """
        cdef int top_slot = LAYOUT.deck_offset + DECK_OFFSETS.top
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order
        cdef int top = <int>state._data[top_slot]
        cdef int company_id
        cdef int current_coo
        cdef int new_top
        cdef int next_company_id

        if top < 0:
            return -1  # Deck is empty

        company_id = <int>state._data[order_base + top]
        assert (<Company>company_module.COMPANIES[company_id])._get_location(state) == LOC_DECK, \
            f"deck.draw(): top company {company_id} not marked LOC_DECK"

        # Pop the top card directly. The deck entity owns the deck array,
        # so draw updates top/order here instead of relying on a company
        # transition to call back into Deck.remove().
        state._data[order_base + top] = <int16_t>-1
        state._data[top_slot] = <int16_t>(top - 1)
        self._sync_cards_remaining(state)

        # Now that the card is detached from the live deck, flip its
        # semantic state to REVEALED via the owner entity for company
        # locations.
        (<Company>company_module.COMPANIES[company_id])._set_location(
            state, LOC_REVEALED, -1)

        # Re-read the (post-splice) top to decide whether the colour
        # tier changed.
        new_top = <int>state._data[top_slot]
        if new_top < 0:
            # Deck exhausted — game-end card exposed
            current_coo = turn_module.TURN.get_coo_level(state)
            turn_module.TURN.set_coo_level(state, current_coo + 1)
        else:
            next_company_id = <int>state._data[order_base + new_top]
            if COMPANY_STARS[company_id] != COMPANY_STARS[next_company_id]:
                current_coo = turn_module.TURN.get_coo_level(state)
                turn_module.TURN.set_coo_level(state, current_coo + 1)

        return company_id

    cpdef int peek(self, GameState state):
        """
        Look at the top card without drawing it.

        Returns the company_id of the top card, or -1 if deck is empty.
        """
        cdef int top = <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top]
        if top < 0:
            return -1
        return <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.order + top]

    cpdef int get_remaining_count(self, GameState state):
        """Return the number of cards remaining in the deck."""
        cdef int top = <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top]
        return top + 1 if top >= 0 else 0

    cpdef bint is_empty(self, GameState state):
        """Return True if the deck has no cards left."""
        return <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top] < 0

    cpdef void remove(self, GameState state, int company_id):
        """Splice a specific company out of the live deck order array.

        Finds the company in deck_order[0..deck_top], shifts the cards
        above it down to fill the gap, clears the vacated top slot, and
        decrements deck_top. This mutates only the deck section; callers
        are responsible for any semantic company-location change.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"

        cdef int top_slot = LAYOUT.deck_offset + DECK_OFFSETS.top
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order
        cdef int top = <int>state._data[top_slot]
        cdef int i, found = -1

        for i in range(top + 1):
            if <int>state._data[order_base + i] == company_id:
                found = i
                break

        assert found >= 0, \
            f"deck.remove({company_id}): company not present in live deck (top={top})"

        # Shift remaining cards left to fill the hole
        for i in range(found, top):
            state._data[order_base + i] = state._data[order_base + i + 1]

        # Clear the vacated top slot and decrement deck top
        state._data[order_base + top] = <int16_t>-1
        state._data[top_slot] = <int16_t>(top - 1)
        self._sync_cards_remaining(state)

    # =========================================================================
    # SETUP
    # =========================================================================

    cpdef void setup(self, GameState state, int num_players, int seed):
        """
        Build the deck according to game rules based on player count.

        Rules:
        1. Highest face value of each colour always included (MHE, PR, DR,
           E, CDG)
        2. Add N companies per colour (N = num_players), with exceptions:
           - 4 players: add 5 orange (not 4)
           - 5 players: add 7 orange (not 5)
           - 6 players: use ALL companies
        3. Shuffle each colour group
        4. Stack: red on top, then orange, yellow, green, blue at bottom

        After the deck is written into the state array, every company
        that did not make it into the live deck is marked LOC_EXCLUDED so
        the engine can distinguish "in the live deck" from "filtered out
        at setup for this player count".
        """
        cdef int[36] deck_cards
        cdef bint included[36]
        cdef int deck_size = 0
        cdef int i, company_id
        cdef int red_count, orange_count, yellow_count, green_count, blue_count
        cdef int top_slot = LAYOUT.deck_offset + DECK_OFFSETS.top
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order

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

        # Build each colour group: always include the last card, shuffle the
        # rest, take the first count-1, then shuffle the whole group.

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

        # Write the deck into state
        state._data[top_slot] = <int16_t>(deck_size - 1)
        for i in range(deck_size):
            state._data[order_base + i] = <int16_t>deck_cards[i]

        # Clear remaining slots (not strictly necessary but clean)
        for i in range(deck_size, 36):
            state._data[order_base + i] = <int16_t>-1

        self._sync_cards_remaining(state)

        # Mark companies that did NOT make it into the live deck as
        # LOC_EXCLUDED. Companies start in LOC_DECK by zero-init, so the
        # ones we *do* include need no per-company write here.
        for i in range(<int>GameConstants.NUM_COMPANIES):
            included[i] = False
        for i in range(deck_size):
            included[deck_cards[i]] = True
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if not included[company_id]:
                (<Company>company_module.COMPANIES[company_id])._set_location(
                    state, LOC_EXCLUDED, -1)

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
        cdef int top = <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top]
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order
        cdef list result = []
        cdef int i

        for i in range(top + 1):
            result.append(<int>state._data[order_base + i])

        return result

    cpdef void set_order(self, GameState state, list order):
        """
        Set the deck order from a Python list (for testing).

        Order should be from bottom to top (index 0 = bottom, last = top).
        Semantic deck membership is normalized to match the rewritten live
        deck: included companies become LOC_DECK, and companies removed
        from the live deck are flipped to LOC_EXCLUDED only if they were
        previously LOC_DECK.
        """
        cdef int i, cid
        cdef int size = len(order)
        cdef int location
        cdef Company company
        cdef int top_slot = LAYOUT.deck_offset + DECK_OFFSETS.top
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order

        new_order = set(order)

        state._data[top_slot] = <int16_t>(size - 1)

        for i in range(size):
            state._data[order_base + i] = <int16_t>order[i]

        # Clear remaining slots
        for i in range(size, 36):
            state._data[order_base + i] = <int16_t>-1

        self._sync_cards_remaining(state)

        # Normalize semantic deck membership to match the rewritten live
        # deck. Included cards become LOC_DECK even if the fresh
        # initialized state had already drawn or excluded them. Cards
        # removed from the live deck are only flipped to LOC_EXCLUDED if
        # they were previously LOC_DECK; other semantic locations are
        # preserved.
        for cid in range(<int>GameConstants.NUM_COMPANIES):
            company = <Company>company_module.COMPANIES[cid]
            if cid in new_order:
                company._set_location(state, LOC_DECK, -1)
            else:
                location = company._get_location(state)
                if location == LOC_DECK:
                    company._set_location(state, LOC_EXCLUDED, -1)

    cpdef list get_ghost_entries(self, GameState state):
        """
        Get company IDs in deck slots past the top index (for invariant
        testing).

        These are slots that held companies before they were drawn. They
        should never contain a company that still has LOC_DECK as its
        location.

        Returns list of (slot_index, company_id) for valid entries past
        the current top index.
        """
        cdef int top = <int>state._data[LAYOUT.deck_offset + DECK_OFFSETS.top]
        cdef int order_base = LAYOUT.deck_offset + DECK_OFFSETS.order
        cdef list result = []
        cdef int i, val

        for i in range(top + 1, 36):
            val = <int>state._data[order_base + i]
            if val >= 0:
                result.append((i, val))

        return result


# =============================================================================
# GLOBAL DECK INSTANCE
# =============================================================================

DECK = Deck()
