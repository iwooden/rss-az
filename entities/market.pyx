# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Market entity implementation.

Provides clean getter/setter access to market space availability in the game
state vector. The market has 27 spaces (indices 0-26) with corresponding prices.
Index 0 is the bankruptcy space (price 0).
"""

from core.state cimport GameState, StateLayout
from core.data cimport get_market_price, get_market_index, GameConstants


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

    # =========================================================================
    # PRICE MOVEMENT HELPERS
    # =========================================================================

    cpdef int find_next_higher_space(self, GameState state, int current_index):
        """
        Find next available higher market space for price movement.

        Starting from current_index + 1, finds first space where
        state._data[market_offset + index] == 1.0 (available).

        Index 26 (price $75) is always available as fallback since
        multiple corps can share it.

        Args:
            state: Game state
            current_index: Current price index (0-26)

        Returns:
            Index of next available higher space (always returns valid index)
        """
        cdef int index
        for index in range(current_index + 1, GameConstants.NUM_MARKET_SPACES - 1):
            if state._data[self._market_offset + index] == 1.0:
                return index
        # No available space found before 26, return 26 (price $75 always valid)
        return 26

    cpdef int find_next_lower_space(self, GameState state, int current_index):
        """
        Find next available lower market space for price movement.

        Starting from current_index - 1, finds first space where
        state._data[market_offset + index] == 1.0 (available).

        If no available space found (all occupied), returns 0 (bankruptcy).

        Args:
            state: Game state
            current_index: Current price index (0-26)

        Returns:
            Index of next available lower space, or 0 if none found
        """
        cdef int index
        for index in range(current_index - 1, -1, -1):
            if state._data[self._market_offset + index] == 1.0:
                return index
        # No available space found, return 0 (bankruptcy)
        return 0


# =============================================================================
# GLOBAL MARKET INSTANCE
# =============================================================================

# Single Market instance
MARKET = Market()
