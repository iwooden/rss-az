"""MARKET: each corp's market cell and stack stamp, and the global stamp counter.

A cell is (row, column) in ``MARKET_PRICE``; row -1 is unparred. Every
placement takes the next stamp, and a lower stamp sits higher in a stack.
"""

from g1889.state cimport GameState

cdef int market_row(GameState state, int corp) noexcept nogil
cdef int market_col(GameState state, int corp) noexcept nogil
cdef int market_price(GameState state, int corp) noexcept nogil


cdef class Market:
    cpdef int row(self, GameState state, int corp)
    cpdef int col(self, GameState state, int corp)
    cpdef int stamp(self, GameState state, int corp)
    cpdef int price(self, GameState state, int corp)
    cpdef void set_position(self, GameState state, int corp, int row, int col)
    cpdef int next_stamp(self, GameState state)
