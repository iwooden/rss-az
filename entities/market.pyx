# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Market entity implementation.

Provides clean getter/setter access to market space availability in the game
state vector. The market has 27 spaces (indices 0-26) with corresponding prices.
Index 0 is the bankruptcy space (price 0).
"""

from state cimport GameState, StateLayout
from data cimport get_market_price, get_market_index, GameConstants


cdef class Market:
    """
    Entity handle for accessing market state.

    There is only one Market instance, created at module load.
    Offsets are computed on first access to a GameState via initialize().
    All methods take GameState as first argument for stateless operation.
    """

    def __cinit__(self):
        self._market_offset = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this Market instance.
        """
        cdef StateLayout layout = state._layout
        self._market_offset = layout.market_offset

    # =========================================================================
    # SPACE AVAILABILITY
    # =========================================================================

    cpdef bint is_space_available(self, GameState state, int index):
        """Check if a market space is available (not occupied by a corp)."""
        if index < 0 or index >= GameConstants.NUM_MARKET_SPACES:
            return False
        return state._data[self._market_offset + index] == 1.0

    cpdef void set_space_available(self, GameState state, int index, bint available):
        """Set whether a market space is available."""
        if index >= 0 and index < GameConstants.NUM_MARKET_SPACES:
            state._data[self._market_offset + index] = 1.0 if available else 0.0

    # =========================================================================
    # PRICE LOOKUPS
    # =========================================================================

    cpdef int get_price_at_index(self, int index):
        """
        Get the price at a market index.

        Index 0 = $0 (bankruptcy), Index 26 = $75 (maximum).
        """
        return get_market_price(index)

    cpdef int get_index_for_price(self, int price):
        """
        Get the market index for a given price.

        Returns -1 if the price is not a valid market price.
        """
        return get_market_index(price)


# =============================================================================
# GLOBAL MARKET INSTANCE
# =============================================================================

# Single Market instance
MARKET = Market()
