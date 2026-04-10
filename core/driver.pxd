# cython: language_level=3
"""
Game driver — declarations.

The driver is the single entry point between an external decision source
(MCTS, eval server, replay tests) and the engine. It owns:

  1. Validation of caller-supplied phase-local action ids against the
     legal set produced by ``core/actions.enumerate_legal_actions``.
  2. Dispatch of validated actions to the appropriate ``phases/<phase>``
     handler.
  3. Auto-fast-forward through forced decision points and automated
     engine phases (WRAP_UP, INCOME, END_CARD) until the next real
     decision or game end.

Action ids are *phase-local*: the caller is responsible for sampling
against the same decision phase the engine is currently in. The driver
re-derives ``decision_phase`` from ``state`` at dispatch time, so the
public ``apply_action(state, action_id)`` signature carries no explicit
phase argument — that would just be a way for caller and engine to
disagree.

Status codes are non-negative; ``apply_action`` uses ``except -1`` to
let exceptions raised inside the dispatch loop propagate out cleanly.

See ``driver-impl.md`` for the design rationale.
"""

from core.state cimport GameState


# =============================================================================
# STATUS CODES
# =============================================================================

cdef enum ActionStatus:
    STATUS_OK = 0          # Action applied; another decision is pending.
    STATUS_GAME_OVER = 1   # Engine reached PHASE_GAME_OVER.
    STATUS_INVALID = 2     # Caller-supplied action_id is not legal in
                           # the current decision phase, or the engine is
                           # currently sitting in a non-decision phase.


# =============================================================================
# DRIVER CLASS
# =============================================================================

cdef class GameDriver:
    # Internal hot-path helpers — keep cdef for cheap calls from
    # apply_action / _auto_chain. ``except -1`` so exceptions raised
    # inside (NotImplementedError from unported handlers, AssertionError
    # from broken state) propagate out.
    cdef int _dispatch(self, GameState state, int phase_id, int action_id, object history) except -1
    cdef int _run_automated_phase(self, GameState state, int engine_phase, object history) except -1
    cdef int _auto_chain(self, GameState state, object history) except -1

    # Public entry points.
    cpdef int apply_action(self, GameState state, int action_id, object history=*) except -1
