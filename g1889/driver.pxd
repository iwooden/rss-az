"""GameDriver: validate, dispatch, and auto-chain (1889/engine-design.md §6).

``apply_action`` checks engine legality, applies the action, then runs
automated steps and forced decisions until a decision with two or more policy
actions, or the end of the game. A decision is forced when its *policy* set
has one action.

History, when a list is passed, gets ``(row copy, decision phase, id)`` before
each decision and ``(row copy, -1, step)`` before each automated step.
"""

from g1889.state cimport GameState

cpdef enum ActionStatus:
    STATUS_OK = 0                   # applied; a decision is pending
    STATUS_GAME_OVER = 1            # the game is over
    STATUS_INVALID = 2              # not legal, or the state sits at an automated step


cdef class GameDriver:
    cdef int _forced_action(self, GameState state) except -2
    cdef int _dispatch(self, GameState state, int phase, int action_id, object history) except -1
    cdef int _run_automated_step(self, GameState state, int step, object history) except -1
    cdef int _auto_chain(self, GameState state, object history) except -1

    cpdef int apply_action(self, GameState state, int action_id, object history=*) except -1
