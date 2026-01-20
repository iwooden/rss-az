# cython: language_level=3
"""
Deck entity declarations.

The Deck manages the company draw pile stored in hidden state. It handles
drawing cards, tracking remaining count, and the complex setup rules that
vary by player count.
"""

from core.state cimport GameState


cdef class Deck:
    # Cached offsets into hidden state
    cdef int _deck_top_offset      # Index of top card (-1 = empty)
    cdef int _deck_order_offset    # Array of 36 company IDs in draw order

    # Initialization
    cpdef void initialize(self, GameState state)

    # Basic operations
    cpdef int draw(self, GameState state)
    cpdef int peek(self, GameState state)
    cpdef int get_remaining_count(self, GameState state)
    cpdef bint is_empty(self, GameState state)

    # Setup - builds deck according to rules based on player count
    cpdef void setup(self, GameState state, int num_players, int seed)

    # Internal helper for building color groups
    cdef int _add_color_group(self, int* deck_cards, int deck_size, int start, int end, int last_idx, int count)

    # Debug/testing helpers
    cpdef list get_order(self, GameState state)
    cpdef void set_order(self, GameState state, list order)
