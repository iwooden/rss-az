"""MARKET: each corp's market cell and stack stamp, and the global stamp counter.

A cell is (row, column) in ``MARKET_PRICE``; row -1 is unparred. Every
placement takes the next stamp, and a lower stamp sits higher in a stack.
Moves follow Ruby's ``TwoDimensionalMovement`` (rules §6.7): a move that
leaves the cell goes to the bottom of the new cell's stack, and one blocked
at an edge leaves the stamp alone.
"""

from g1889.state cimport GameState

cdef int market_row(GameState state, int corp) noexcept nogil
cdef int market_col(GameState state, int corp) noexcept nogil
cdef int market_stamp(GameState state, int corp) noexcept nogil
cdef int market_price(GameState state, int corp) noexcept nogil

# Zones (rules §2.8): yellow shares skip the certificate limit; orange ones
# also skip the 60% holding limit.
cdef int market_type(GameState state, int corp) noexcept nogil
cdef bint market_counts_for_limit(GameState state, int corp) noexcept nogil
cdef bint market_unlimited(GameState state, int corp) noexcept nogil

# Stock price order (rules §6.7, Corporation#sort_order_key).
cdef bint market_precedes(GameState state, int a, int b) noexcept nogil
cdef int market_sort_by_price(GameState state, int mask, int* out) noexcept nogil


cdef class Market:
    cpdef int row(self, GameState state, int corp)
    cpdef int col(self, GameState state, int corp)
    cpdef int stamp(self, GameState state, int corp)
    cpdef int price(self, GameState state, int corp)
    cpdef void set_position(self, GameState state, int corp, int row, int col)
    cpdef int next_stamp(self, GameState state)
    cpdef void move_down(self, GameState state, int corp)
    cpdef void move_up(self, GameState state, int corp)
    cpdef int type(self, GameState state, int corp)
    cpdef bint counts_for_limit(self, GameState state, int corp)
    cpdef bint unlimited(self, GameState state, int corp)
    cpdef tuple order_key(self, GameState state, int corp)
    cpdef list sort_by_price(self, GameState state, int mask)
