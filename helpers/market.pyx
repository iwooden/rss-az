# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Market state helper functions.

Provides functions for market space availability and share price navigation.
The market consists of NUM_MARKET_SPACES (27) price cards, indexed 0-26.
Index 0 = inactive/bankruptcy (multiple corps allowed)
Index 1-25 = prices $5-$68 (unique per active corp)
Index 26 = price $75 (multiple corps allowed, on or off market)
"""

from core.data cimport NUM_MARKET_SPACES


# =============================================================================
# MARKET SPACE AVAILABILITY
# =============================================================================

cdef inline bint is_market_space_available(float* market, int index) noexcept nogil:
    """
    Check if a market space is available (no corp occupying it).

    The market array stores 1.0 for available spaces, 0.0 for taken spaces.
    """
    if index < 0 or index >= NUM_MARKET_SPACES:
        return False
    return market[index] == 1.0


cdef inline void set_market_space_available(float* market, int index, bint available) noexcept nogil:
    """Set market space availability."""
    if index >= 0 and index < NUM_MARKET_SPACES:
        market[index] = 1.0 if available else 0.0


# =============================================================================
# PRICE NAVIGATION
# =============================================================================

cdef int find_next_higher_price_index(float* market, int current_index) noexcept nogil:
    """
    Find the next available higher price index.

    Scans from current_index+1 upward until an available space is found.
    Returns NUM_MARKET_SPACES - 1 if no higher space is available (will go off-market to 75).

    Used when buying shares (price moves up).
    """
    cdef int i

    for i in range(current_index + 1, NUM_MARKET_SPACES):
        if is_market_space_available(market, i):
            return i

    # No higher available - return max index (will go off market to price 75)
    return NUM_MARKET_SPACES - 1


cdef int find_next_lower_price_index(float* market, int current_index) noexcept nogil:
    """
    Find the next available lower price index.

    Scans from current_index-1 downward until an available space is found.
    Returns 0 if hitting bankruptcy threshold.

    Used when selling shares or issuing shares (price moves down).
    """
    cdef int i

    for i in range(current_index - 1, -1, -1):
        if is_market_space_available(market, i):
            return i

    # No lower available - return 0 (bankruptcy)
    return 0


cdef int find_adjusted_price_index(float* market, int current_index, int movement) noexcept nogil:
    """
    Find the target price index after movement, skipping taken spaces.

    Args:
        market: Pointer to market availability array
        current_index: Current price index (1-26 for active corps)
        movement: Number of steps to move (negative = down, positive = up)

    Returns:
        0 if hitting bankruptcy
        26 (NUM_MARKET_SPACES - 1) if going off the top (price = 75, off-market)
        Otherwise the new index (1-26)

    Used during dividend phase for share price adjustment.
    """
    cdef int target_index = current_index + movement
    cdef int direction = 1 if movement > 0 else -1

    # Slide until we find an available space
    while target_index > 0 and target_index < NUM_MARKET_SPACES:
        if is_market_space_available(market, target_index):
            return target_index
        target_index += direction

    # Handle edge cases
    if target_index <= 0:
        return 0  # Bankruptcy
    else:
        return NUM_MARKET_SPACES - 1  # Off the top (price = 75, index 26)
