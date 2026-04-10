# cython: language_level=3
"""CLOSING phase handler declarations.

Two entry points: ``apply_closing_auto`` runs the deterministic auto-close
stages on phase entry, ``apply_closing_action`` dispatches player CLOSE/PASS
decisions. All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void apply_closing_auto(GameState state) noexcept
cdef void apply_closing_action(GameState state, ActionInfo* info) noexcept
