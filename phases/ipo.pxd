# cython: language_level=3
"""
Declaration file for IPO phase.

In descending Face Value order, private companies may form corporations.
Players choose a corp and par price; if par < face_value, 4 shares issued (2 each),
otherwise 2 shares issued (1 each).
"""

from cython_core.state cimport GameState
from cython_core.helpers.player cimport PlayerOffsets
from cython_core.helpers.corp cimport CorpOffsets
from cython_core.helpers.turn cimport IssueTurnOffsets


cdef class IPOPhase:
    # Attributes
    cdef int _num_players
    cdef CorpOffsets _co
    cdef PlayerOffsets _po
    cdef IssueTurnOffsets _ito  # Contains ipo_company and ipo_remaining

    # Pointer helpers
    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil
    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil
    cdef float* _get_turn(self, GameState state) noexcept nogil
    cdef float* _get_market(self, GameState state) noexcept nogil

    # Main entry points
    cpdef void setup_ipo_phase(self, GameState state)
    cpdef void advance_to_next_company(self, GameState state)

    # Actions
    cpdef bint can_pass(self, GameState state)
    cpdef bint can_ipo(self, GameState state, int corp_id, int par_index)
    cpdef void do_pass(self, GameState state)
    cpdef void do_ipo(self, GameState state, int corp_id, int par_index)

    # Convenience methods
    cpdef int get_current_company(self, GameState state)
    cpdef int get_current_company_owner(self, GameState state)
    cpdef list get_valid_ipo_options(self, GameState state)

    # Internal helpers
    cdef int _find_company_owner(self, GameState state, int company_id) noexcept nogil
    cdef bint _has_any_inactive_corp(self, GameState state) noexcept nogil
    cdef bint _can_player_afford_any_ipo(self, GameState state, int player_id, int company_id) noexcept nogil
    cdef int _calculate_ipo_cost(self, int face_value, int par_price) noexcept nogil
    cdef void _transition_to_invest(self, GameState state) noexcept
