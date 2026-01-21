# cython: language_level=3
"""
Market entity declarations.
"""

from core.state cimport GameState


cdef class Market:
    cdef int _market_offset    # Offset to market availability array in state

    # Initialization
    cpdef void initialize(self, GameState state)

    # Space availability
    cpdef bint is_space_available(self, GameState state, int index)
    cpdef void set_space_available(self, GameState state, int index, bint available)

    # Price lookups (convenience wrappers around data.pyx functions)
    cpdef int get_price_at_index(self, int index)
    cpdef int get_index_for_price(self, int price)

    # Price movement helpers
    cpdef int find_next_higher_space(self, GameState state, int current_index)
    cpdef int find_next_lower_space(self, GameState state, int current_index)
