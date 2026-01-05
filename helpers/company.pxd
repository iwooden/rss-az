# cython: language_level=3
"""
Company state helper declarations.

Provides functions for company state access including auction slot mapping.
"""

from cython_core.state cimport GameState
from cython_core.data cimport NUM_COMPANIES


# =============================================================================
# AUCTION SLOT MAPPING
# =============================================================================

cdef int get_auction_company_for_slot(GameState state, int slot) noexcept nogil
