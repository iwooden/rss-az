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
    int auction_slot_info_size
    int invest_impacts_size
    int invest_impacts_offset
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
    int auction_slot_info_offset
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
    int hidden_offer_count_offset
    int hidden_offer_index_offset
    int hidden_offer_buffer_offset
    int hidden_close_offer_count_offset
    int hidden_close_offer_index_offset
    int hidden_close_offer_buffer_offset
    # O(1) access for one-hot fields (performance optimization)
    int hidden_acq_active_corp_offset
    int hidden_acq_target_company_offset
    int hidden_closing_company_offset
    int hidden_dividend_corp_offset
    int hidden_issue_corp_offset
    int hidden_ipo_company_offset
    # Turn number (moved from visible to hidden)
    int hidden_turn_number_offset
    # Per-player share buy/sell tracking (moved from visible to hidden)
    int hidden_share_buys_offset   # [p0_buys(8), p1_buys(8), ...]
    int hidden_share_sells_offset  # [p0_sells(8), p1_sells(8), ...]
    # Company location tracking (O(1) clearing without scanning visible state)
    int hidden_company_locations_offset
    int hidden_company_owner_ids_offset

cdef struct TurnStateOffsets:
    int end_card_flipped
    int consecutive_passes
    int auction_price
    int auction_high_bidder
    int auction_starter
    int auction_passed
    int dividend_impact
    int dividend_remaining
    int issue_remaining
    int issue_price_impact
    int issue_cash_gain
    int ipo_remaining
    int acq_is_fi_offer
    int acq_synergy_values
    # Active company: one-hot (36) + contextual info (5 scalars)
    int active_company
    int active_company_info
    # Active corp: one-hot (8) + contextual info (3 scalars) + owned companies (36 flags)
    int active_corp
    int active_corp_info
    int active_corp_companies
    # Cards remaining in deck
    int cards_remaining

cdef struct PlayerFieldOffsets:
    int cash
    int net_worth
    int turn_order
    int owned_companies
    int owned_shares
    int is_president
    int round_trips
    int acquisition_proceeds
    int income

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

cdef class GameState:
    cdef float* _data
    cdef public object _array
    cdef StateLayout _layout
    cdef TurnStateOffsets _turn_offsets
    cdef TurnStateOffsets _turn  # Alias for convenience
    cdef PlayerFieldOffsets _player_fields
    cdef CorpFieldOffsets _corp_fields
    cdef int _num_players

    # Internal pointer access
    cdef float* _player_ptr(self, int player_id) noexcept nogil
    cdef float* _corp_ptr(self, int corp_id) noexcept nogil
    cdef float* _turn_ptr(self) noexcept nogil
    cdef int _get_active_player(self) noexcept nogil
    cdef void _set_active_player(self, int player_id) noexcept nogil

    # Active player and num_players access (Python-accessible)
    cpdef int get_active_player(self)
    cpdef int get_num_players(self)

    # Phase access (setter via TurnState entity to avoid duplication)
    cpdef int get_phase(self)

    # Player access
    cpdef int get_player_cash(self, int player_id)
    cpdef void set_player_cash(self, int player_id, int cash)
    cpdef int get_player_net_worth(self, int player_id)
    cpdef void set_player_net_worth(self, int player_id, int net_worth)

    # Corporation access
    cdef bint _is_corp_active(self, int corp_id) noexcept nogil
    cpdef bint is_corp_active(self, int corp_id)
    cpdef void set_corp_active(self, int corp_id, bint active)
    cpdef int get_corp_cash(self, int corp_id)
    cpdef void set_corp_cash(self, int corp_id, int cash)
    cpdef int get_corp_bank_shares(self, int corp_id)
    cpdef void set_corp_bank_shares(self, int corp_id, int shares)
    cpdef int get_corp_unissued_shares(self, int corp_id)
    cpdef void set_corp_unissued_shares(self, int corp_id, int shares)
    cpdef int get_corp_issued_shares(self, int corp_id)
    cpdef void set_corp_issued_shares(self, int corp_id, int shares)
    cdef int _get_corp_share_price(self, int corp_id) noexcept nogil
    cpdef int get_corp_share_price(self, int corp_id)
    cpdef void set_corp_share_price(self, int corp_id, int price)
    cpdef int get_corp_price_index(self, int corp_id)
    cpdef void set_corp_price_index(self, int corp_id, int index)
    cpdef bint is_corp_in_receivership(self, int corp_id)
    cpdef void set_corp_in_receivership(self, int corp_id, bint in_recv)
    cdef bint _corp_owns_company(self, int corp_id, int company_id) noexcept nogil
    cpdef bint corp_owns_company(self, int corp_id, int company_id)
    cpdef void set_corp_owns_company(self, int corp_id, int company_id, bint owns)

    # Market access
    cpdef bint is_market_space_available(self, int index)
    cpdef void set_market_space_available(self, int index, bint available)

    # Company access
    cdef bint _is_company_for_auction(self, int company_id) noexcept nogil
    cpdef bint is_company_for_auction(self, int company_id)
    cpdef void set_company_for_auction(self, int company_id, bint for_auction)

    # Auction state access (setters via TurnState entity to avoid duplication)
    cpdef int get_auction_company(self)
    cpdef int get_auction_price(self)

    # Acquisition state access (setters via TurnState entity to avoid duplication)
    cdef int _get_acq_active_corp(self) noexcept nogil
    cpdef int get_acq_active_corp(self)
    cdef int _get_acq_target_company(self) noexcept nogil
    cpdef int get_acq_target_company(self)
    cpdef bint is_acq_fi_offer(self)

    # Dividend state access (setter via TurnState entity to avoid duplication)
    cdef int _get_dividend_corp(self) noexcept nogil
    cpdef int get_dividend_corp(self)

    # Issue state access (setter via TurnState entity to avoid duplication)
    cdef int _get_issue_corp(self) noexcept nogil
    cpdef int get_issue_corp(self)

    # IPO state access (setter via TurnState entity to avoid duplication)
    cdef int _get_ipo_company(self) noexcept nogil
    cpdef int get_ipo_company(self)

    # Closing state access (setter via TurnState entity to avoid duplication)
    cdef int _get_current_closing_company(self) noexcept nogil
    cpdef int get_current_closing_company(self)

    # Auction slot info
    cpdef void _populate_auction_slot_info(self)

    # Invest phase impacts
    cpdef void _populate_invest_impacts(self)
    cpdef void _clear_invest_impacts(self)

    # Active company contextual info
    cpdef void set_active_company(self, int company_id)
    cpdef void clear_active_company(self)

    # Active corp contextual info
    cpdef void set_active_corp(self, int corp_id)
    cpdef void clear_active_corp(self)

    # Game initialization
    cpdef void initialize_game(self, int seed=*)