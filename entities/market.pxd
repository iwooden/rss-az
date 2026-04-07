# cython: language_level=3
"""
Market entity declarations.

The market is a flat array of 27 availability flags inside the compact
GameState — one per share-price space ($0 / $5 / ... / $75). Static price
data lives in core.data and is read directly via cimported arrays.
"""

from core.state cimport GameState


cdef class Market:
    # Cached absolute offset to the market availability flags inside the
    # compact state array.
    cdef int _market_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef bint _is_space_available(self, GameState state, int index) noexcept nogil
    cdef void _set_space_available(self, GameState state, int index, bint available) noexcept nogil
    cdef int _find_next_higher_space(self, GameState state, int current_index) noexcept nogil
    cdef int _find_next_lower_space(self, GameState state, int current_index) noexcept nogil

    # Space availability (Python-accessible wrappers)
    cpdef bint is_space_available(self, GameState state, int index)
    cpdef void set_space_available(self, GameState state, int index, bint available)

    # Price lookups (read directly from core.data static arrays)
    cpdef int get_price_at_index(self, int index)
    cpdef int get_index_for_price(self, int price)

    # Price movement helpers (Python-accessible wrappers)
    cpdef int find_next_higher_space(self, GameState state, int current_index)
    cpdef int find_next_lower_space(self, GameState state, int current_index)
