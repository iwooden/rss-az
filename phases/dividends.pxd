# cython: language_level=3
"""
Declaration file for dividends phase.

In the dividends phase, corps (in descending share price order) choose how much
to pay out as dividends, then adjust their share price based on stars vs target.
"""

from state cimport GameState
from helpers.player cimport PlayerOffsets
from helpers.corp cimport CorpOffsets
from helpers.turn cimport DividendTurnOffsets


cdef class DividendsPhase:
    # Attributes
    cdef int _num_players
    cdef PlayerOffsets _po
    cdef CorpOffsets _co
    cdef DividendTurnOffsets _dto

    # Pointer helpers
    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil
    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil
    cdef float* _get_turn(self, GameState state) noexcept nogil
    cdef float* _get_market(self, GameState state) noexcept nogil

    # Phase setup and flow
    cpdef void setup_dividends(self, GameState state)
    cpdef void advance_to_next_corp(self, GameState state)

    # Action validation and execution
    cpdef bint can_do_dividend(self, GameState state, int amount)
    cpdef void do_dividend(self, GameState state, int amount)

    # Convenience methods
    cpdef list get_valid_actions(self, GameState state)
    cpdef int get_max_dividend(self, GameState state)
    cpdef int get_current_corp(self, GameState state)

    # Calculations
    cpdef int calculate_corp_stars(self, GameState state, int corp_id)
    cpdef int calculate_target_stars(self, GameState state, int corp_id)

    # Internal helpers
    cdef void _setup_dividend_state(self, GameState state, int corp_id) noexcept
    cdef void _pay_dividends(self, GameState state, int corp_id, int amount) noexcept
    cdef void _adjust_share_price(self, GameState state, int corp_id) noexcept
    cdef void _transition_to_end_card(self, GameState state) noexcept
