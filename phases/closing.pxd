# cython: language_level=3
"""
Declaration file for closing phase (offer-based).
"""

from cython_core.state cimport GameState

# Constants
cdef enum:
    NUM_COMPANIES = 36
    NUM_CORPS = 8
    CLOSING_ACTION_CLOSE = 0  # Close the current company
    CLOSING_ACTION_PASS = 1   # Pass on current company
    CLOSING_ACTION_MAX = 2    # Total actions


cdef class ClosingPhase:
    cdef int _num_players

    # Phase setup and flow
    cpdef void setup_closing(self, GameState state)
    cpdef bint setup_next_closeable(self, GameState state)
    cpdef bint is_waiting_for_action(self, GameState state)

    # Action validation and execution
    cpdef bint can_do_action(self, GameState state, int action)
    cpdef void do_action(self, GameState state, int action)

    # Convenience methods
    cpdef list get_valid_actions(self, GameState state)
    cpdef void auto_close_and_transition(self, GameState state)

    # Internal helpers
    cdef bint _find_next_closeable(self, GameState state, int start_company)
    cdef void _auto_close_receivership_corp(self, GameState state, int corp_id, int coo_level) noexcept
    cdef void _force_close_player_companies(self, GameState state, int player_id, int coo_level) noexcept
