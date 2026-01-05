# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Company state helper functions.

Provides functions for company state access including auction slot mapping.
"""

from cython_core.state cimport GameState, NUM_COMPANIES


# =============================================================================
# AUCTION SLOT MAPPING
# =============================================================================

cdef inline int get_auction_company_for_slot(GameState state, int slot) noexcept nogil:
    """
    Return company_id for the Nth auction slot (by company_id order), or -1.

    Auction slots are ordered by company_id. Slot 0 maps to the lowest
    company_id that is available for auction, slot 1 to the next lowest, etc.

    Args:
        state: Game state to query
        slot: Slot index (0 to MAX_AUCTION_SLOTS-1)

    Returns:
        Company ID for the slot, or -1 if slot index is out of range
    """
    cdef int count = 0
    cdef int company_id
    for company_id in range(NUM_COMPANIES):
        if state.is_company_for_auction(company_id):
            if count == slot:
                return company_id
            count += 1
    return -1
