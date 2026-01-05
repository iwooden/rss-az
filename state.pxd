# cython: language_level=3
"""
Declaration file for game state.
Allows other Cython modules to cimport GameState and its methods.
"""

cimport numpy as cnp

# =============================================================================
# CONSTANTS
# =============================================================================

cdef enum:
    NUM_COMPANIES = 36
    NUM_CORPS = 8
    NUM_MARKET_SPACES = 27
    NUM_PHASES = 11
    NUM_COO_LEVELS = 7
    MAX_PLAYERS = 6
    MAX_DECK_SIZE = 36

cdef enum:
    PHASE_INVEST = 0
    PHASE_BID_IN_AUCTION = 1
    PHASE_WRAP_UP = 2
    PHASE_ACQUISITION = 3
    PHASE_CLOSING = 4
    PHASE_INCOME = 5
    PHASE_DIVIDENDS = 6
    PHASE_END_CARD = 7
    PHASE_ISSUE_SHARES = 8
    PHASE_IPO = 9
    PHASE_GAME_OVER = 10


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


# =============================================================================
# GAMESTATE CLASS
# =============================================================================

cdef class GameState:
    # Data
    cdef float* _data
    cdef cnp.ndarray _array
    cdef StateLayout _layout
    cdef TurnStateOffsets _turn
    cdef PlayerFieldOffsets _player_fields
    cdef CorpFieldOffsets _corp_fields
    cdef int _num_players

    # Hidden state accessors
    cdef int _get_active_player(self) noexcept nogil
    cdef void _set_active_player(self, int player_id) noexcept nogil
    cdef int _get_deck_top(self) noexcept nogil
    cdef void _set_deck_top(self, int top) noexcept nogil
    cdef int _get_deck_company(self, int index) noexcept nogil
    cdef void _set_deck_company(self, int index, int company_id) noexcept nogil

    # Phase accessors
    cdef int get_phase(self) noexcept nogil
    cdef void set_phase(self, int phase) noexcept nogil

    # Cost of ownership
    cdef int get_coo_level(self) noexcept nogil
    cdef void set_coo_level(self, int level) noexcept nogil

    # Pointer accessors
    cdef float* _player_ptr(self, int player_id) noexcept nogil
    cdef float* _corp_ptr(self, int corp_id) noexcept nogil
    cdef float* _turn_ptr(self) noexcept nogil
    cdef float* _market_ptr(self) noexcept nogil
    cdef float* _hidden_price_indices_ptr(self) noexcept nogil

    # Turn state
    cdef int get_turn_number(self) noexcept nogil
    cdef void set_turn_number(self, int turn_num) noexcept nogil
    cdef int get_consecutive_passes(self) noexcept nogil
    cdef void set_consecutive_passes(self, int count) noexcept nogil
    cdef void increment_consecutive_passes(self) noexcept nogil
    cdef void clear_consecutive_passes(self) noexcept nogil

    # Auction state
    cdef int get_auction_company(self) noexcept nogil
    cdef void set_auction_company(self, int company_id) noexcept nogil
    cdef int get_auction_price(self) noexcept nogil
    cdef void set_auction_price(self, int price) noexcept nogil
    cdef int get_auction_high_bidder(self) noexcept nogil
    cdef void set_auction_high_bidder(self, int player_id) noexcept nogil
    cdef int get_auction_starter(self) noexcept nogil
    cdef void set_auction_starter(self, int player_id) noexcept nogil
    cdef bint get_auction_passed(self, int player_id) noexcept nogil
    cdef void set_auction_passed(self, int player_id, bint passed) noexcept nogil
    cdef void clear_auction_state(self) noexcept nogil
    cdef void init_auction_passed(self) noexcept nogil

    # Core operations
    cdef void copy_from(self, GameState other) noexcept nogil
    cdef bint is_game_over(self) noexcept nogil
    cdef void advance_active_player(self) noexcept nogil

    # Player turn order
    cdef int get_player_turn_order(self, int player_id) noexcept nogil
    cdef void set_player_turn_order(self, int player_id, int position) noexcept nogil
    cdef int get_player_at_turn_order(self, int position) noexcept nogil
    cdef int get_player_cash(self, int player_id) noexcept nogil
    cdef void set_player_cash(self, int player_id, int cash) noexcept nogil

    # Foreign investor
    cdef int get_fi_cash(self) noexcept nogil
    cdef void set_fi_cash(self, int cash) noexcept nogil
    cdef void add_fi_cash(self, int amount) noexcept nogil
    cdef bint fi_owns_company(self, int company_id) noexcept nogil
    cdef void set_fi_owns_company(self, int company_id, bint owns) noexcept nogil

    # Company locations
    cdef bint is_company_for_auction(self, int company_id) noexcept nogil
    cdef void set_company_for_auction(self, int company_id, bint available) noexcept nogil
    cdef bint is_company_revealed(self, int company_id) noexcept nogil
    cdef void set_company_revealed(self, int company_id, bint revealed) noexcept nogil
    cdef void draw_company_to_revealed(self) noexcept nogil
    cdef void move_revealed_to_auction(self) noexcept nogil

    # Player company ownership
    cdef bint player_owns_company(self, int player_id, int company_id) noexcept nogil
    cdef void set_player_owns_company(self, int player_id, int company_id, bint owns) noexcept nogil
    cdef bint is_player_president(self, int player_id, int corp_id) noexcept nogil
    cdef void set_player_president(self, int player_id, int corp_id, bint is_pres) noexcept nogil
    cdef int get_player_shares(self, int player_id, int corp_id) noexcept nogil
    cdef void add_player_cash(self, int player_id, int amount) noexcept nogil

    # Corporation accessors
    cdef bint is_corp_active(self, int corp_id) noexcept nogil
    cdef void set_corp_active(self, int corp_id, bint active) noexcept nogil
    cdef int get_corp_cash(self, int corp_id) noexcept nogil
    cdef void set_corp_cash(self, int corp_id, int cash) noexcept nogil
    cdef void add_corp_cash(self, int corp_id, int amount) noexcept nogil
    cdef bint is_corp_in_receivership(self, int corp_id) noexcept nogil
    cdef void set_corp_in_receivership(self, int corp_id, bint in_recv) noexcept nogil
    cdef int get_corp_price_index(self, int corp_id) noexcept nogil
    cdef void set_corp_price_index(self, int corp_id, int index) noexcept nogil
    cdef int get_corp_share_price(self, int corp_id) noexcept nogil
    cdef bint corp_owns_company(self, int corp_id, int company_id) noexcept nogil
    cdef void set_corp_owns_company(self, int corp_id, int company_id, bint owns) noexcept nogil
    cdef int get_corp_company_count(self, int corp_id) noexcept nogil
    cdef bint corp_has_acquisition_company(self, int corp_id, int company_id) noexcept nogil
    cdef void set_corp_acquisition_company(self, int corp_id, int company_id, bint has) noexcept nogil
    cdef int get_corp_acquisition_proceeds(self, int corp_id) noexcept nogil
    cdef void set_corp_acquisition_proceeds(self, int corp_id, int amount) noexcept nogil
    cdef void add_corp_acquisition_proceeds(self, int corp_id, int amount) noexcept nogil

    # Acquisition offer state
    cdef int get_acq_active_corp(self) noexcept nogil
    cdef void set_acq_active_corp(self, int corp_id) noexcept nogil
    cdef int get_acq_target_company(self) noexcept nogil
    cdef void set_acq_target_company(self, int company_id) noexcept nogil
    cdef bint is_acq_fi_offer(self) noexcept nogil
    cdef void set_acq_is_fi_offer(self, bint is_fi) noexcept nogil
    cdef void clear_acq_offer(self) noexcept nogil
    cdef bint has_corp_done_acquisition(self, int corp_id) noexcept nogil
    cdef void set_corp_done_acquisition(self, int corp_id) noexcept nogil
    cdef void clear_acq_corps_done(self) noexcept nogil
    cdef void finalize_acquisitions(self) noexcept nogil

    # Closing phase offer state
    cdef int get_current_closing_company(self) noexcept nogil
    cdef void set_current_closing_company(self, int company_id) noexcept nogil
    cdef void clear_closing_company(self) noexcept nogil

    # Corporation share accessors
    cdef int get_corp_issued_shares(self, int corp_id) noexcept nogil
    cdef void set_corp_issued_shares(self, int corp_id, int shares) noexcept nogil
    cdef int get_corp_bank_shares(self, int corp_id) noexcept nogil
    cdef void set_corp_bank_shares(self, int corp_id, int shares) noexcept nogil
    cdef int get_corp_unissued_shares(self, int corp_id) noexcept nogil
    cdef void set_corp_unissued_shares(self, int corp_id, int shares) noexcept nogil

    # Removed companies (closed/out of game)
    cdef bint is_company_removed(self, int company_id) noexcept nogil
    cdef void set_company_removed(self, int company_id, bint removed) noexcept nogil

    # Company adjusted incomes (updated when CoO changes)
    cdef int get_company_adjusted_income(self, int company_id) noexcept nogil
    cdef void update_all_company_incomes(self) noexcept nogil

    # Market space availability
    cdef bint is_market_space_available(self, int index) noexcept nogil
    cdef void set_market_space_available(self, int index, bint available) noexcept nogil

    # Corporation bankruptcy
    cdef void bankrupt_corp(self, int corp_id) noexcept nogil

    # End card state
    cdef bint get_end_card_flipped(self) noexcept nogil
    cdef void set_end_card_flipped(self, bint flipped) noexcept nogil

    # Player net worth
    cdef int get_player_net_worth(self, int player_id) noexcept nogil
    cdef void set_player_net_worth(self, int player_id, int net_worth) noexcept nogil

    # NN rotation
    cdef void _rotate_for_nn(self, float* out, int active_player) noexcept nogil
    cdef void _rotate_turn_state(self, float* out, int active_player) noexcept nogil
