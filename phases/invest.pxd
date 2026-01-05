# cython: language_level=3
"""
Declaration file for invest phase.
Allows other Cython modules to cimport the invest functions.
"""

from state cimport GameState
from helpers.player cimport PlayerOffsets
from helpers.corp cimport CorpOffsets


# =============================================================================
# COMPANY LOCATION HELPERS (invest-specific)
# =============================================================================

cdef bint is_company_for_auction(float* auction_companies, int company_id) noexcept nogil
cdef void remove_company_from_auction(float* auction_companies, int company_id) noexcept nogil


# =============================================================================
# INVEST PHASE ACTIONS
# =============================================================================

cdef void invest_buy_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    float* hidden_price_indices,
    int corp_id
) noexcept nogil

cdef void invest_sell_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    float* hidden_price_indices,
    int corp_id
) noexcept nogil


# =============================================================================
# VALID ACTION CHECKING
# =============================================================================

cdef bint can_buy_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    float* market,
    int corp_id
) noexcept nogil

cdef bint can_sell_share(
    float* player, PlayerOffsets* po,
    float* corp, CorpOffsets* co,
    int corp_id
) noexcept nogil

cdef bint can_start_auction(
    float* player, PlayerOffsets* po,
    float* auction_companies,
    int company_id,
    int bid_offset
) noexcept nogil

cdef bint can_raise_bid(
    float* player, PlayerOffsets* po,
    int company_id,
    int current_min_bid,
    int bid_offset
) noexcept nogil


# =============================================================================
# INVEST PHASE CLASS
# =============================================================================

cdef class InvestPhase:
    cdef PlayerOffsets _po
    cdef CorpOffsets _co
    cdef int _num_players

    # Pointer extraction helpers
    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil
    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil
    cdef float* _get_market(self, GameState state) noexcept nogil
    cdef float* _get_auction_companies(self, GameState state) noexcept nogil
    cdef int _get_active_player(self, GameState state) noexcept nogil

    # Buy share
    cpdef bint can_do_buy_share(self, GameState state, int corp_id)
    cpdef void do_buy_share(self, GameState state, int corp_id)

    # Sell share
    cpdef bint can_do_sell_share(self, GameState state, int corp_id)
    cpdef void do_sell_share(self, GameState state, int corp_id)

    # Start auction
    cpdef bint can_do_start_auction(self, GameState state, int company_id, int bid_offset)
    cpdef void do_start_auction(self, GameState state, int company_id, int bid_offset)

    # Raise bid (BID_IN_AUCTION phase)
    cpdef bint can_do_raise_bid(self, GameState state, int bid_offset)
    cpdef void do_raise_bid(self, GameState state, int bid_offset)

    # Leave auction (BID_IN_AUCTION phase)
    cpdef void do_leave_auction(self, GameState state)

    # Pass
    cpdef void do_pass(self, GameState state)

    # Valid actions
    cpdef dict get_valid_actions(self, GameState state)
    cdef dict _get_invest_valid_actions(self, GameState state)
    cdef dict _get_auction_valid_actions(self, GameState state)

    # Internal helpers
    cdef void _update_presidency(self, GameState state, int corp_id) noexcept
    cdef void _advance_auction_player(self, GameState state) noexcept
    cdef bint _check_auction_resolved(self, GameState state) noexcept nogil
    cdef void _resolve_auction(self, GameState state) noexcept
    cdef void _draw_company_to_auction(self, GameState state) noexcept
    cdef void _end_invest_phase(self, GameState state) noexcept
