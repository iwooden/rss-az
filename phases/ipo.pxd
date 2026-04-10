# cython: language_level=3
"""IPO phase handler declarations.

Two entry points: ``setup_ipo_phase`` initializes the per-company remaining
flags and finds the first company, ``apply_ipo_action`` dispatches IPO
decisions. All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_ipo_phase(GameState state) noexcept
cdef void apply_ipo_action(GameState state, ActionInfo* info) noexcept
