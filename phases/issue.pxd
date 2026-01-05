# cython: language_level=3
"""
Declaration file for Issue Shares phase.

Corps issue shares in descending share price order.
Corps in receivership must issue if they have unissued shares.
Stock Masters (SM) special: share price does not decrease when issuing.
"""

from state cimport GameState
from helpers.player cimport PlayerOffsets
from helpers.corp cimport CorpOffsets
from helpers.turn cimport IssueTurnOffsets


cdef class IssuePhase:
    # Attributes
    cdef int _num_players
    cdef CorpOffsets _co
    cdef PlayerOffsets _po
    cdef IssueTurnOffsets _ito

    # Pointer helpers
    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil
    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil
    cdef float* _get_turn(self, GameState state) noexcept nogil
    cdef float* _get_market(self, GameState state) noexcept nogil

    # Main entry points
    cpdef void setup_issue_phase(self, GameState state)
    cpdef void advance_to_next_corp(self, GameState state)

    # Actions
    cpdef bint can_issue(self, GameState state)
    cpdef bint can_pass(self, GameState state)
    cpdef void do_issue(self, GameState state)
    cpdef void do_pass(self, GameState state)

    # Convenience methods
    cpdef int get_current_corp(self, GameState state)
    cpdef list get_valid_actions(self, GameState state)

    # Internal helpers
    cdef void _transition_to_ipo(self, GameState state) noexcept
