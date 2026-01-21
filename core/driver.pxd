# cython: language_level=3
"""Declaration file for game driver."""

from core.state cimport GameState

# Status codes for apply_action return
cdef enum ActionStatus:
    STATUS_OK = 0           # Action applied successfully
    STATUS_INVALID = 1      # Invalid action for current state
    STATUS_GAME_OVER = 2    # Game ended after this action

cdef class GameDriver:
    cpdef int apply_action(self, GameState state, int action_idx)
    cpdef object get_legal_moves(self, GameState state)
