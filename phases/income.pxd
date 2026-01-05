# cython: language_level=3
"""
Declaration file for income phase.

Income phase is fully automatic - no player actions.
"""

from cython_core.state cimport GameState

# Constants
cdef enum:
    NUM_COMPANIES = 36
    NUM_CORPS = 8


cdef class IncomePhase:
    cdef int _num_players

    # Main entry point
    cpdef void handle_income_phase(self, GameState state)

    # Income calculations
    cpdef int calculate_fi_income(self, GameState state)
    cpdef int calculate_player_income(self, GameState state, int player_id)
    cpdef int calculate_corp_income(self, GameState state, int corp_id)

    # Helper for corp synergies
    cpdef int calculate_corp_synergies(self, GameState state, int corp_id)
