# cython: language_level=3
"""
Declaration file for IPO phase handler.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_ipo_action(GameState state, ActionInfo* info) noexcept
cpdef void setup_ipo_phase(GameState state)
