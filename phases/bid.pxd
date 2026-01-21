# cython: language_level=3
"""BID_IN_AUCTION phase handler declarations."""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_bid_action(GameState state, ActionInfo* info) noexcept
