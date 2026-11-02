"""MARKET entity; see market.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport MARKET_COLS, MARKET_PRICE, MARKET_ROWS, NUM_CORPS
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, GameState


cdef inline int _slot(int corp, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + field


cdef int market_row(GameState state, int corp) noexcept nogil:
    return <int>state._data[_slot(corp, CORP_FIELDS.market_row)]


cdef int market_col(GameState state, int corp) noexcept nogil:
    return <int>state._data[_slot(corp, CORP_FIELDS.market_col)]


cdef int market_price(GameState state, int corp) noexcept nogil:
    """The corp's share price, or 0 while it is unparred."""
    cdef int row = market_row(state, corp)
    return MARKET_PRICE[row][market_col(state, corp)] if row >= 0 else 0


cdef class Market:
    """Stateless handle over the market fields."""

    cpdef int row(self, GameState state, int corp):
        return market_row(state, corp)

    cpdef int col(self, GameState state, int corp):
        return market_col(state, corp)

    cpdef int stamp(self, GameState state, int corp):
        return <int>state._data[_slot(corp, CORP_FIELDS.market_stamp)]

    cpdef int price(self, GameState state, int corp):
        return market_price(state, corp)

    cpdef void set_position(self, GameState state, int corp, int row, int col):
        """Place the corp on a cell, at the bottom of that cell's stack."""
        cdef int counter = LAYOUT.global_offset + GLOBALS.market_stamp
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        assert 0 <= row < <int>MARKET_ROWS and 0 <= col < <int>MARKET_COLS and MARKET_PRICE[row][col] > 0, \
            f"no market cell ({row}, {col})"
        state._data[_slot(corp, CORP_FIELDS.market_row)] = <int16_t>row
        state._data[_slot(corp, CORP_FIELDS.market_col)] = <int16_t>col
        state._data[_slot(corp, CORP_FIELDS.market_stamp)] = state._data[counter]
        state._data[counter] += 1

    cpdef int next_stamp(self, GameState state):
        return <int>state._data[LAYOUT.global_offset + GLOBALS.market_stamp]


MARKET = Market()
