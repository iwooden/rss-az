# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Market entity implementation.

Provides clean access to market space availability in the compact game
state array. The market has 27 spaces (indices 0-26) with corresponding
share prices: index 0 is the bankruptcy space ($0), index 26 is the
maximum ($75).

Availability flags are stored as raw 0/1 int16 values inside the state
buffer; static price data (prices, reverse lookup) lives in core.data and
is read directly via cimported arrays — there are no helper functions to
wrap.
"""

from libc.stdint cimport int16_t

from core.state cimport GameState, LAYOUT
from core.data cimport (
    GameConstants,
    MARKET_PRICES,
    PRICE_TO_MARKET_INDEX,
)
from entities.player cimport invalidate_all_player_caches


cdef void copy_market_availability(GameState state, int16_t* out_flags) noexcept nogil:
    cdef int index
    for index in range(<int>GameConstants.NUM_MARKET_SPACES):
        out_flags[index] = state._data[LAYOUT.market_offset + index]


cdef class Market:
    """
    Entity handle for accessing market state.

    There is only one Market instance, created at module load. It is
    stateless: every read derives its slot inline from
    ``LAYOUT.market_offset`` on ``core.state``, so the same handle works
    against any GameState. All methods take a GameState as the first
    argument.
    """

    # =========================================================================
    # SPACE AVAILABILITY (low-level, nogil)
    # =========================================================================

    cdef inline bint _is_space_available(self, GameState state, int index) noexcept nogil:
        return state._data[LAYOUT.market_offset + index] == 1

    cdef inline void _set_space_available(self, GameState state, int index, bint available) noexcept nogil:
        state._data[LAYOUT.market_offset + index] = <int16_t>(1 if available else 0)

    # =========================================================================
    # SPACE AVAILABILITY (Python-accessible wrappers)
    # =========================================================================

    cpdef bint is_space_available(self, GameState state, int index):
        """Return True if the given market space is unoccupied."""
        assert 0 <= index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"market index {index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        return self._is_space_available(state, index)

    cpdef void set_space_available(self, GameState state, int index, bint available):
        """Set whether the given market space is available."""
        cdef bint old_available
        cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1
        assert 0 <= index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"market index {index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        assert available or (index != 0 and index != max_index), \
            "Market boundary spaces 0 ($0) and 26 ($75) must remain available"
        old_available = self._is_space_available(state, index)
        self._set_space_available(state, index, available)
        if old_available != available:
            invalidate_all_player_caches(state)

    # =========================================================================
    # PRICE LOOKUPS (static data from core.data)
    # =========================================================================

    cpdef int get_price_at_index(self, int index):
        """Return the share price ($) at the given market index.

        Index 0 = $0 (bankruptcy), Index 26 = $75 (maximum).
        """
        assert 0 <= index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"market index {index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        return MARKET_PRICES[index]

    cpdef int get_index_for_price(self, int price):
        """Return the market index for the given price.

        Returns -1 if `price` is not a valid market price (e.g. $4 or $99
        within the lookup range). The -1 sentinel is genuine business logic
        — callers use it to detect "not a market price" — so it is kept
        even with the new assert-on-bad-input convention; the assert only
        guards against indexing outside the static lookup table.
        """
        assert 0 <= price < 76, f"price {price} outside lookup range [0, 76)"
        return PRICE_TO_MARKET_INDEX[price]

    # =========================================================================
    # PRICE MOVEMENT HELPERS
    # =========================================================================

    cdef int _find_next_higher_space(self, GameState state, int current_index) noexcept nogil:
        """Find next available higher market space.

        INVARIANT: Index 26 ($75) is always available. Per RULES.md, when
        no higher card is available, the corp takes no card and has price
        $75. Multiple corps can share this "no card" state, so it is never
        marked occupied.
        """
        cdef int index
        cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1  # 26
        for index in range(current_index + 1, max_index):
            if state._data[LAYOUT.market_offset + index] == 1:
                return index
        # Index 26 ($75) is always available per game rules
        return max_index

    cdef int _find_next_lower_space(self, GameState state, int current_index) noexcept nogil:
        """Find next available lower market space.

        INVARIANT: Index 0 ($0, bankruptcy) is always available. When a
        corp lands on index 0 it goes bankrupt and is removed from the
        market, immediately freeing the space.
        """
        cdef int index
        for index in range(current_index - 1, -1, -1):
            if state._data[LAYOUT.market_offset + index] == 1:
                return index
        # Index 0 ($0) is always available per game rules
        return 0

    cpdef int find_next_higher_space(self, GameState state, int current_index):
        """Python wrapper around `_find_next_higher_space`."""
        assert 0 <= current_index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"current_index {current_index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        return self._find_next_higher_space(state, current_index)

    cpdef int find_next_lower_space(self, GameState state, int current_index):
        """Python wrapper around `_find_next_lower_space`."""
        assert 0 <= current_index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"current_index {current_index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        return self._find_next_lower_space(state, current_index)


# =============================================================================
# GLOBAL MARKET INSTANCE
# =============================================================================

# Single Market instance
MARKET = Market()
