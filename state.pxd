# cython: language_level=3
"""
Declaration file for game state.
Allows other Cython modules to cimport GameState and its methods.
"""

cimport numpy as cnp

# =============================================================================
# LAYOUT STRUCTURES
# =============================================================================

cdef struct StateLayout:
    int phase_size
    int coo_size
    int player_stride
    int players_size
    int fi_size
    int companies_size
    int company_incomes_size
    int market_size
    int corp_stride
    int corps_size
    int turn_size
    int static_size
    int visible_size
    int hidden_size
    int total_size
    int phase_offset
    int coo_offset
    int players_offset
    int fi_offset
    int auction_companies_offset
    int revealed_companies_offset
    int removed_companies_offset
    int company_incomes_offset
    int market_offset
    int corps_offset
    int turn_offset
    int static_offset
    int hidden_active_player_offset
    int hidden_num_players_offset
    int hidden_deck_top_offset
    int hidden_deck_order_offset
    # Compact storage for frequently-accessed one-hot fields (performance optimization)
    int hidden_phase_offset
    int hidden_coo_level_offset
    int hidden_auction_company_offset
    int hidden_auction_high_bidder_offset
    int hidden_auction_starter_offset
    int hidden_corp_price_indices_offset
    int hidden_acq_corps_done_offset

cdef struct TurnStateOffsets:
    int turn_number
    int end_card_flipped
    int consecutive_passes
    int auction_company
    int auction_price
    int auction_high_bidder
    int auction_starter
    int auction_passed
    int dividend_corp
    int dividend_impact
    int dividend_remaining
    int issue_corp
    int issue_remaining
    int ipo_company
    int ipo_remaining
    int acq_active_corp
    int acq_target_company
    int acq_is_fi_offer
    # Closing phase
    int closing_company

cdef struct PlayerFieldOffsets:
    int cash
    int net_worth
    int turn_order
    int is_auction_high_bidder
    int owned_companies
    int owned_shares
    int is_president
    int share_buys
    int share_sells

cdef struct CorpFieldOffsets:
    int active
    int cash
    int unissued_shares
    int issued_shares
    int bank_shares
    int income
    int stars
    int share_price
    int acquisition_proceeds
    int in_receivership
    int price_index
    int owned_companies
    int acquisition_companies

# =============================================================================
# LAYOUT COMPUTATION FUNCTIONS
# =============================================================================

cdef StateLayout compute_layout(int num_players) noexcept nogil
cdef TurnStateOffsets compute_turn_offsets(int num_players) noexcept nogil
cdef PlayerFieldOffsets compute_player_field_offsets(int num_players) noexcept nogil
cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil