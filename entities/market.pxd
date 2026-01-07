# cython: language_level=3
"""
Market entity declarations.
"""

from state cimport GameState


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
