# cython: language_level=3
"""
Turn state entity declarations.

Turn state in the compact GameState is split between metadata
(phase, coo_level, turn_number — single integer slots stored at the top
of the state array) and the turn block (per-turn tracking such as
end_card_flipped, consecutive_passes, cards_remaining, auction state,
and the dividend/issue/IPO "remaining" flag arrays).

All values are raw int16 — no one-hot encoding, no NN-side duplication of
active corp/company features, no normalization. Phase-specific context
the transformer needs (dividend impacts, par price tables, synergy
previews, etc.) is reconstructed by the token-extraction layer instead of
being maintained as engine state.
"""

from core.state cimport GameState


cdef class TurnState:
    cdef int _num_players

    # Metadata (single-slot integer fields outside the turn block)
    cdef int _phase_offset
    cdef int _coo_level_offset
    cdef int _turn_number_offset

    # Turn state base offset
    cdef int _turn_offset

    # Cached absolute offsets for turn block fields
    cdef int _end_card_flipped_offset
    cdef int _consecutive_passes_offset
    cdef int _cards_remaining_offset

    # Auction
    cdef int _auction_price_offset
    cdef int _auction_company_offset
    cdef int _auction_high_bidder_offset
    cdef int _auction_starter_offset

    # Phase remaining tracking
    cdef int _dividend_remaining_offset
    cdef int _issue_remaining_offset
    cdef int _ipo_remaining_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef int _get_phase(self, GameState state) noexcept nogil
    cdef int _get_coo_level(self, GameState state) noexcept nogil
    cdef int _get_auction_price(self, GameState state) noexcept nogil

    # Phase
    cpdef int get_phase(self, GameState state)
    cpdef void set_phase(self, GameState state, int phase)

    # Cost of ownership level (1-7 in game terms)
    cpdef int get_coo_level(self, GameState state)
    cpdef void set_coo_level(self, GameState state, int level)
    cdef void _update_all_company_incomes(self, GameState state, int coo_level)
    cdef void _update_all_corp_incomes(self, GameState state)
    cdef void _update_all_player_incomes(self, GameState state)

    # Turn number
    cpdef int get_turn_number(self, GameState state)
    cpdef void set_turn_number(self, GameState state, int turn)

    # End card flipped
    cpdef bint is_end_card_flipped(self, GameState state)
    cpdef void set_end_card_flipped(self, GameState state, bint flipped)

    # Consecutive passes (INVEST phase)
    cpdef int get_consecutive_passes(self, GameState state)
    cpdef void set_consecutive_passes(self, GameState state, int passes)
    cpdef void increment_consecutive_passes(self, GameState state)
    cpdef void clear_consecutive_passes(self, GameState state)

    # Cards remaining (deck mirror)
    cpdef int get_cards_remaining(self, GameState state)
    cpdef void set_cards_remaining(self, GameState state, int count)

    # Auction state
    cpdef int get_auction_company(self, GameState state)
    cpdef void set_auction_company(self, GameState state, int company_id)
    cpdef void clear_auction_company(self, GameState state)

    cpdef int get_auction_price(self, GameState state)
    cpdef void set_auction_price(self, GameState state, int price)

    cpdef int get_auction_high_bidder(self, GameState state)
    cpdef void set_auction_high_bidder(self, GameState state, int player_id)
    cpdef void clear_auction_high_bidder(self, GameState state)

    cpdef int get_auction_starter(self, GameState state)
    cpdef void set_auction_starter(self, GameState state, int player_id)
    cpdef void clear_auction_starter(self, GameState state)

    # Phase-remaining flag arrays
    cpdef bint is_dividend_remaining(self, GameState state, int corp_id)
    cpdef void set_dividend_remaining(self, GameState state, int corp_id, bint remaining)

    cpdef bint is_issue_remaining(self, GameState state, int corp_id)
    cpdef void set_issue_remaining(self, GameState state, int corp_id, bint remaining)

    cpdef bint is_ipo_remaining(self, GameState state, int company_id)
    cpdef void set_ipo_remaining(self, GameState state, int company_id, bint remaining)

    # Turn order navigation
    cpdef int find_player_at_position(self, GameState state, int position)
    cpdef void advance_to_next_bidder(self, GameState state)
    cpdef void set_active_player_after(self, GameState state, int player_id)
