# cython: language_level=3
"""
Declaration file for issue shares phase handler.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_issue_action(GameState state, ActionInfo* info) noexcept
cpdef void setup_issue_phase(GameState state)
