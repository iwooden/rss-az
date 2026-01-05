# cython: language_level=3
"""
Declaration file for End Card phase.

The End Card phase is automatic - no player actions.
It checks for game end conditions and transitions to Issue Shares phase.
"""

from state cimport GameState

# Constants
cdef enum:
    NUM_COMPANIES = 36
    NUM_CORPS = 8


cdef class EndCardPhase:
    # Attributes
    cdef int _num_players

    # Main entry point
    cpdef void handle_end_card_phase(self, GameState state)

    # Game end checks
    cdef bint _check_game_end(self, GameState state) noexcept
    cdef bint _any_corp_at_max_price(self, GameState state) noexcept nogil
    cdef bint _is_end_card_flipped(self, GameState state) noexcept nogil
    cdef void _end_game(self, GameState state) noexcept

    # End card flip
    cdef bint _should_flip_end_card(self, GameState state) noexcept nogil
    cdef void _flip_end_card(self, GameState state) noexcept

    # Issue phase setup
    cdef void _setup_issue_phase(self, GameState state) noexcept

    # Helpers
    cdef bint _has_any_auction_companies(self, GameState state) noexcept nogil
    cdef bint _is_deck_empty(self, GameState state) noexcept nogil
