# cython: language_level=3
"""BID phase handler declarations.

Handles the two BID actions: LEAVE (pass-class) and RAISE. The handler
assumes ``info`` is a legal BID action produced by ``decode_action(
DPHASE_BID, action_id)`` after the id was yielded by ``_enumerate_bid``.
All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo


cdef void apply_bid_action(GameState state, ActionInfo* info) noexcept
