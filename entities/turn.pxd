# cython: language_level=3
"""
Turn state entity declarations.

Includes phase, cost-of-ownership, and all turn-specific tracking state.
"""

from core.state cimport GameState


# =============================================================================
# LOW-LEVEL NOGIL ACCESSORS
# =============================================================================

cdef struct TurnOffsets:
    # Offsets within turn state data block in the state vector
    int acq_active_corp
    int acq_target_company
    int acq_is_fi_offer
    int dividend_corp
    int issue_corp
    int ipo_company
    int closing_company

# Offset computation
cdef TurnOffsets get_turn_offsets(int num_players) noexcept nogil

# Turn state accessors (raw pointer, nogil)
cdef int get_acq_active_corp_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef int get_acq_target_company_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef bint is_acq_fi_offer_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef int get_dividend_corp_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef int get_issue_corp_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef int get_ipo_company_nogil(float* turn, TurnOffsets* t) noexcept nogil
cdef int get_closing_company_nogil(float* turn, TurnOffsets* t) noexcept nogil


# =============================================================================
# HIGH-LEVEL ENTITY CLASS
# =============================================================================

cdef class TurnState:
    cdef int _num_players

    # Phase & CoO offsets (at start of state vector, one-hot encoded)
    cdef int _phase_offset
    cdef int _coo_offset

    # Hidden state offsets (compact integer storage for fast O(1) access)
    cdef int _hidden_phase_offset
    cdef int _hidden_coo_level_offset
    cdef int _hidden_auction_company_offset
    cdef int _hidden_auction_high_bidder_offset
    cdef int _hidden_auction_starter_offset
    cdef int _hidden_acq_active_corp_offset
    cdef int _hidden_acq_target_company_offset
    cdef int _hidden_acq_is_fi_offer_offset  # One-hot offset (single value, already O(1))
    cdef int _hidden_dividend_corp_offset
    cdef int _hidden_issue_corp_offset
    cdef int _hidden_ipo_company_offset
    cdef int _hidden_closing_company_offset

    # Turn state base offset
    cdef int _turn_offset

    # Turn state field offsets (relative to turn_offset)
    cdef int _turn_number_offset
    cdef int _end_card_flipped_offset
    cdef int _consecutive_passes_offset

    # Auction offsets
    cdef int _auction_company_offset
    cdef int _auction_price_offset
    cdef int _auction_high_bidder_offset
    cdef int _auction_starter_offset
    cdef int _auction_passed_offset

    # Dividends offsets
    cdef int _dividend_corp_offset
    cdef int _dividend_impact_offset
    cdef int _dividend_remaining_offset

    # Issue offsets
    cdef int _issue_corp_offset
    cdef int _issue_remaining_offset

    # IPO offsets
    cdef int _ipo_company_offset
    cdef int _ipo_remaining_offset

    # Acquisition offsets
    cdef int _acq_active_corp_offset
    cdef int _acq_target_company_offset
    cdef int _acq_is_fi_offer_offset

    # Closing offset
    cdef int _closing_company_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Phase (one-hot, 11 values)
    cpdef int get_phase(self, GameState state)
    cpdef void set_phase(self, GameState state, int phase)

    # Cost of ownership level (one-hot, 7 values, 1-indexed in game terms)
    cpdef int get_coo_level(self, GameState state)
    cpdef void set_coo_level(self, GameState state, int level)

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

    cpdef bint has_player_passed_auction(self, GameState state, int player_id)
    cpdef void set_player_passed_auction(self, GameState state, int player_id, bint passed)
    cpdef void clear_auction_passed(self, GameState state)

    # Dividends state
    cpdef int get_dividend_corp(self, GameState state)
    cpdef void set_dividend_corp(self, GameState state, int corp_id)
    cpdef void clear_dividend_corp(self, GameState state)

    cpdef int get_dividend_impact(self, GameState state, int level)
    cpdef void set_dividend_impact(self, GameState state, int level, int impact)

    cpdef bint is_dividend_remaining(self, GameState state, int corp_id)
    cpdef void set_dividend_remaining(self, GameState state, int corp_id, bint remaining)

    # Issue state
    cpdef int get_issue_corp(self, GameState state)
    cpdef void set_issue_corp(self, GameState state, int corp_id)
    cpdef void clear_issue_corp(self, GameState state)

    cpdef bint is_issue_remaining(self, GameState state, int corp_id)
    cpdef void set_issue_remaining(self, GameState state, int corp_id, bint remaining)

    # IPO state
    cpdef int get_ipo_company(self, GameState state)
    cpdef void set_ipo_company(self, GameState state, int company_id)
    cpdef void clear_ipo_company(self, GameState state)

    cpdef bint is_ipo_remaining(self, GameState state, int company_id)
    cpdef void set_ipo_remaining(self, GameState state, int company_id, bint remaining)

    # Acquisition state
    cpdef int get_acq_active_corp(self, GameState state)
    cpdef void set_acq_active_corp(self, GameState state, int corp_id)
    cpdef void clear_acq_active_corp(self, GameState state)

    cpdef int get_acq_target_company(self, GameState state)
    cpdef void set_acq_target_company(self, GameState state, int company_id)
    cpdef void clear_acq_target_company(self, GameState state)

    cpdef bint is_acq_fi_offer(self, GameState state)
    cpdef void set_acq_fi_offer(self, GameState state, bint is_fi)

    # Closing state
    cpdef int get_closing_company(self, GameState state)
    cpdef void set_closing_company(self, GameState state, int company_id)
    cpdef void clear_closing_company(self, GameState state)

    # Turn order navigation
    cpdef int find_player_at_position(self, GameState state, int position)
    cpdef void advance_to_next_bidder(self, GameState state)
    cpdef void set_active_player_after(self, GameState state, int player_id)

    # Nogil accessors (for mask generation in actions.pyx)
    cdef inline int _get_acq_active_corp_nogil(self, float* data) noexcept nogil
    cdef inline int _get_acq_target_company_nogil(self, float* data) noexcept nogil
    cdef inline bint _is_acq_fi_offer_nogil(self, float* data) noexcept nogil
    cdef inline int _get_dividend_corp_nogil(self, float* data) noexcept nogil
    cdef inline int _get_issue_corp_nogil(self, float* data) noexcept nogil
    cdef inline int _get_ipo_company_nogil(self, float* data) noexcept nogil
    cdef inline int _get_closing_company_nogil(self, float* data) noexcept nogil
