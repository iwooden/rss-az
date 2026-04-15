"""ISSUE phase handler declarations.

Two entry points: ``setup_issue_phase`` initializes the per-corp remaining
flags and finds the first corp, ``apply_issue_action`` dispatches issue
decisions. All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_issue_phase(GameState state) noexcept
cdef void apply_issue_action(GameState state, ActionInfo* info) noexcept
