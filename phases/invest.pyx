# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""INVEST phase handler implementation."""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo, ActionType,
    ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE
)

cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply INVEST phase action to state.

    Returns: 0=success, 1=invalid

    STUB: Returns 0 for valid action types, 1 for invalid.
    Full implementation in Phase 3.
    """
    if info.action_type == ACTION_PASS:
        # TODO Phase 3: Increment consecutive_passes, check for phase transition
        return 0
    elif info.action_type == ACTION_AUCTION:
        # TODO Phase 3: Initialize auction state, transition to BID_IN_AUCTION
        return 0
    elif info.action_type == ACTION_BUY_SHARE:
        # TODO Phase 4: Buy share logic
        return 0
    elif info.action_type == ACTION_SELL_SHARE:
        # TODO Phase 4: Sell share logic
        return 0

    return 1  # Invalid action type for INVEST phase
