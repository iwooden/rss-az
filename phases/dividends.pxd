# cython: language_level=3
"""
Declaration file for dividends phase handler.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_dividend_action(GameState state, ActionInfo* info) noexcept
cpdef void setup_dividends_phase(GameState state)
