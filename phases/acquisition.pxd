# cython: language_level=3
"""
Declaration file for acquisition phase.
"""

from cython_core.state cimport GameState

# Constants
cdef enum:
    NUM_CORPS = 8
    NUM_COMPANIES = 36
    CORP_OS = 2  # Overseas Trading - buys from FI at face value

# Action constants
cdef enum:
    ACQ_ACTION_PASS = 50  # Pass on this acquisition offer
    ACQ_ACTION_MAX = 51   # Total actions for price selection (0-49 + pass)
    ACQ_FI_ACTION_BUY = 0     # Buy from FI at required price
    ACQ_FI_ACTION_PASS = 1    # Pass on FI offer
    ACQ_FI_ACTION_MAX = 2     # Total FI actions


cdef class AcquisitionPhase:
    cdef int _num_players

    # Offer generation
    cpdef bint setup_next_offer(self, GameState state)
    cpdef bint is_waiting_for_action(self, GameState state)

    # Action validation and execution
    cpdef bint can_do_action(self, GameState state, int action)
    cpdef void do_action(self, GameState state, int action)

    # Convenience methods
    cpdef list get_valid_actions(self, GameState state)
    cpdef void transition_to_closing(self, GameState state)
