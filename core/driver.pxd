# cython: language_level=3
"""Declaration file for game driver."""

from core.state cimport GameState

# Status codes for apply_action return
cdef enum ActionStatus:
    STATUS_OK = 0           # Action applied successfully
    STATUS_INVALID = 1      # Invalid action for current state
    STATUS_GAME_OVER = 2    # Game ended after this action

# Result struct for forced action checking
cdef struct ForcedActionResult:
    int count       # 0, 1, or 2 (stop counting at 2 for early exit)
    int action_idx  # -1 if count != 1, otherwise the single valid action index

# Helper function for forced action detection
cdef ForcedActionResult _check_forced_action(GameState state) noexcept

# Terminal state detection (shared by acquisition and closing phases)
cdef bint _is_game_terminal(GameState state) noexcept

cdef class GameDriver:
    cdef int _dispatch_action(self, GameState state, int action_idx, object history)
    cdef int _apply_single_action(self, GameState state, int action_idx, object history)
    cpdef int apply_action(self, GameState state, int action_idx, object history=*)
    cpdef object get_legal_moves(self, GameState state)
    cpdef bint is_non_player_phase(self, GameState state)
    cpdef int advance_phase(self, GameState state, object history=*)
