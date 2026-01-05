# cython: language_level=3
"""
Market state helper declarations.

Provides functions for market space availability and price index navigation.
"""

from cython_core.data cimport NUM_MARKET_SPACES

# =============================================================================
# MARKET SPACE AVAILABILITY
# =============================================================================

cdef bint is_market_space_available(float* market, int index) noexcept nogil
cdef void set_market_space_available(float* market, int index, bint available) noexcept nogil


# =============================================================================
# PRICE NAVIGATION
# =============================================================================

cdef int find_next_higher_price_index(float* market, int current_index) noexcept nogil
cdef int find_next_lower_price_index(float* market, int current_index) noexcept nogil
cdef int find_adjusted_price_index(float* market, int current_index, int movement) noexcept nogil
