"""MARKET entity; see market.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport (
    MARKET_COLS, MARKET_NO_CERT_LIMIT, MARKET_NORMAL, MARKET_PRICE, MARKET_ROW_LEN, MARKET_ROWS, MARKET_TYPE,
    MARKET_UNLIMITED, NUM_CORPS,
)
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, GameState


cdef inline int _slot(int corp, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + field


cdef int market_row(GameState state, int corp) noexcept nogil:
    return <int>state._data[_slot(corp, CORP_FIELDS.market_row)]


cdef int market_col(GameState state, int corp) noexcept nogil:
    return <int>state._data[_slot(corp, CORP_FIELDS.market_col)]


cdef int market_stamp(GameState state, int corp) noexcept nogil:
    return <int>state._data[_slot(corp, CORP_FIELDS.market_stamp)]


cdef int market_price(GameState state, int corp) noexcept nogil:
    """The corp's share price, or 0 while it is unparred."""
    cdef int row = market_row(state, corp)
    return MARKET_PRICE[row][market_col(state, corp)] if row >= 0 else 0


cdef int market_type(GameState state, int corp) noexcept nogil:
    """The corp's cell type (MarketType); MARKET_NORMAL while unparred."""
    cdef int row = market_row(state, corp)
    return MARKET_TYPE[row][market_col(state, corp)] if row >= 0 else <int>MARKET_NORMAL


cdef bint market_counts_for_limit(GameState state, int corp) noexcept nogil:
    """Whether the corp's shares count toward the certificate limit: outside the
    yellow and orange zones (CERT_LIMIT_TYPES), and always while unparred."""
    cdef int t = market_type(state, corp)
    return t != <int>MARKET_NO_CERT_LIMIT and t != <int>MARKET_UNLIMITED


cdef bint market_unlimited(GameState state, int corp) noexcept nogil:
    """The orange zone, whose shares skip the 60% holding limit (``holding_ok?``)."""
    return market_type(state, corp) == <int>MARKET_UNLIMITED


cdef bint market_precedes(GameState state, int a, int b) noexcept nogil:
    """Whether parred corp ``a`` comes before ``b`` in stock price order: the
    higher price, then the column farther right, then the higher row (Ruby's
    key; equal price and column already mean one cell), then the higher place
    in the cell's stack."""
    cdef int price_a = market_price(state, a), price_b = market_price(state, b)
    if price_a != price_b:
        return price_a > price_b
    if market_col(state, a) != market_col(state, b):
        return market_col(state, a) > market_col(state, b)
    if market_row(state, a) != market_row(state, b):
        return market_row(state, a) < market_row(state, b)
    return market_stamp(state, a) < market_stamp(state, b)


cdef int market_sort_by_price(GameState state, int mask, int* out) noexcept nogil:
    """Write the parred corps in the corp bitmask ``mask`` to ``out`` (room for
    NUM_CORPS) in stock price order; return how many."""
    cdef int count = 0
    cdef int c, i
    for c in range(<int>NUM_CORPS):
        if not (mask >> c & 1) or market_row(state, c) < 0:
            continue
        i = count
        while i > 0 and market_precedes(state, c, out[i - 1]):
            out[i] = out[i - 1]
            i -= 1
        out[i] = c
        count += 1
    return count


cdef class Market:
    """Stateless handle over the market fields."""

    cpdef int row(self, GameState state, int corp):
        return market_row(state, corp)

    cpdef int col(self, GameState state, int corp):
        return market_col(state, corp)

    cpdef int stamp(self, GameState state, int corp):
        return market_stamp(state, corp)

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

    cpdef void move_down(self, GameState state, int corp):
        """One row down, as for each share sold; at the bottom of its column the
        token stays put (``TwoDimensionalMovement#down``)."""
        cdef int row = market_row(state, corp), col = market_col(state, corp)
        assert row >= 0, f"corp {corp} is not on the market"
        if row + 1 < <int>MARKET_ROWS and MARKET_PRICE[row + 1][col] > 0:
            self.set_position(state, corp, row + 1, col)

    cpdef void move_up(self, GameState state, int corp):
        """One row up, as when sold out at the end of an SR; the top row stays put."""
        cdef int row = market_row(state, corp)
        assert row >= 0, f"corp {corp} is not on the market"
        if row > 0:
            self.set_position(state, corp, row - 1, market_col(state, corp))

    cpdef void move_left(self, GameState state, int corp):
        """One column left, as for a withhold; at column 0 one row down instead
        (``TwoDimensionalMovement#left``), so ¥10 stays put."""
        cdef int row = market_row(state, corp), col = market_col(state, corp)
        assert row >= 0, f"corp {corp} is not on the market"
        if col > 0:
            self.set_position(state, corp, row, col - 1)
        else:
            self.move_down(state, corp)

    cpdef void move_right(self, GameState state, int corp):
        """One column right, as for a payout; at a row's end one row up instead
        (``TwoDimensionalMovement#right``), so ¥350 stays put."""
        cdef int row = market_row(state, corp), col = market_col(state, corp)
        assert row >= 0, f"corp {corp} is not on the market"
        if col + 1 < MARKET_ROW_LEN[row]:
            self.set_position(state, corp, row, col + 1)
        else:
            self.move_up(state, corp)

    cpdef int type(self, GameState state, int corp):
        return market_type(state, corp)

    cpdef bint counts_for_limit(self, GameState state, int corp):
        return market_counts_for_limit(state, corp)

    cpdef bint unlimited(self, GameState state, int corp):
        return market_unlimited(state, corp)

    cpdef tuple order_key(self, GameState state, int corp):
        """Ruby's ``sort_order_key`` with the stamp for the stack index: sorting
        parred corps by it gives stock price order."""
        assert market_row(state, corp) >= 0, f"corp {corp} is not on the market"
        return (-market_price(state, corp), -market_col(state, corp), market_row(state, corp),
                market_stamp(state, corp))

    cpdef list sort_by_price(self, GameState state, int mask):
        """The parred corps in the corp bitmask ``mask``, in stock price order."""
        cdef int[7] order  # NUM_CORPS
        cdef int count = market_sort_by_price(state, mask, order)
        return [order[i] for i in range(count)]


MARKET = Market()

# market_sort_by_price's callers size their buffers with the literal 7.
assert <int>NUM_CORPS == 7
