# cython: language_level=3
"""INVEST phase handler declarations."""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept
