# cython: language_level=3
"""CLOSING phase handler declarations.

Two entry points: ``setup_closing_phase`` sets the phase, runs the
deterministic auto-close stages, and finds the first player for decisions;
``apply_closing_action`` dispatches player CLOSE/PASS decisions. All state
access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef bint _corp_closable_by_player(GameState state, int corp_id, int player_id) noexcept nogil
cdef void setup_closing_phase(GameState state) noexcept
cdef void apply_closing_action(GameState state, ActionInfo* info) noexcept
