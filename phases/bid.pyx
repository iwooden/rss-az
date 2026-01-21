# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""BID_IN_AUCTION phase handler implementation."""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo, ActionType,
    ACTION_LEAVE_AUCTION, ACTION_RAISE_BID
)

cdef int apply_bid_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply BID_IN_AUCTION phase action to state.

    Returns: 0=success, 1=invalid

    STUB: Returns 0 for valid action types, 1 for invalid.
    Full implementation in Phase 3.
    """
    if info.action_type == ACTION_LEAVE_AUCTION:
        # TODO Phase 3: Set auction_passed flag, check for auction resolution
        return 0
    elif info.action_type == ACTION_RAISE_BID:
        # TODO Phase 3: Update auction price and high bidder
        return 0

    return 1  # Invalid action type for BID phase
