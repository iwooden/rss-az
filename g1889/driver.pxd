"""GameDriver: validate, dispatch, and auto-chain (1889/engine-design.md §6).

``apply_action`` checks engine legality, applies the action, then runs
automated steps and forced decisions until a decision with two or more policy
actions, or the end of the game. A decision is forced when its *policy* set
has one action, unless that action is the only legal grid point of an amount
decision with exact amounts above it (§6).

History, when a list is passed, gets ``(row copy, decision phase, id)`` before
each decision and ``(row copy, -1, step)`` before each automated step.
``advance`` runs the auto-chain on a state set up by hand.
``override_route_revenue`` lets replay pay a human's routes instead of the
solver's maximum, ``override_private_slot`` take a human's private move at a
decision that offers no slot for it, ``apply_out_of_turn`` DR's exchange at
another player's decision, and ``override_topmost_bid`` a human's bid on
private 0, which has no slot.
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
    cpdef int advance(self, GameState state, object history=*) except -1
    cpdef int override_route_revenue(self, GameState state, int revenue, object history=*) except -1
    cpdef int override_private_slot(self, GameState state, int action_id, object history=*) except -1
    cpdef int apply_out_of_turn(self, GameState state, int private, object history=*) except -1
    cpdef int override_topmost_bid(self, GameState state, object history=*) except -1
