# cython: language_level=3
"""
Deck entity declarations.

The Deck manages the company draw pile inside the compact GameState. Two
int16 slots back it: a top-of-deck index and a 36-slot order array. Cards
that did not make it into the live deck for the active player count are
marked LOC_EXCLUDED during setup.
"""

from core.state cimport GameState


cdef class Deck:
    # Cached absolute offsets into the compact state array.
    cdef int _deck_top_offset      # Index of top card (-1 = empty)
    cdef int _deck_order_offset    # Array of 36 company IDs in draw order

    # Initialization
    cpdef void initialize(self, GameState state)

    # Basic operations
    cpdef int draw(self, GameState state)
    cpdef int peek(self, GameState state)
    cpdef int get_remaining_count(self, GameState state)
    cpdef bint is_empty(self, GameState state)
    cpdef void remove(self, GameState state, int company_id)

    # Push the current deck-top count out to TurnState.cards_remaining
    # so phases / NN tokens can read deck size without touching the deck
    # array directly.
    cdef void _sync_cards_remaining(self, GameState state)

    # Setup - builds deck according to rules based on player count
    cpdef void setup(self, GameState state, int num_players, int seed)

    # Internal helper for building color groups
    cdef int _add_color_group(self, int* deck_cards, int deck_size, int start, int end, int last_idx, int count)

    # Debug/testing helpers
    cpdef list get_order(self, GameState state)
    cpdef void set_order(self, GameState state, list order)
    cpdef list get_ghost_entries(self, GameState state)
