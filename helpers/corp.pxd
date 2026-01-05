# cython: language_level=3
"""
Corporation state helper declarations.

Provides CorpOffsets struct and accessor functions for corporation state
stored in the float tensor representation.
"""

from state cimport GameState
from data cimport NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES
from helpers.player cimport PlayerOffsets

# =============================================================================
# CORP OFFSETS STRUCT
# =============================================================================

cdef struct CorpOffsets:
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
    int acquisition_companies  # Only used in acquisition phase


# =============================================================================
# OFFSET COMPUTATION
# =============================================================================

cdef CorpOffsets get_corp_offsets() noexcept nogil


# =============================================================================
# ACTIVE STATUS
# =============================================================================

cdef bint is_corp_active(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_active(float* corp, CorpOffsets* c, bint active) noexcept nogil


# =============================================================================
# CASH OPERATIONS
# =============================================================================

cdef int get_corp_cash(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_cash(float* corp, CorpOffsets* c, int cash) noexcept nogil
cdef void add_corp_cash(float* corp, CorpOffsets* c, int amount) noexcept nogil


# =============================================================================
# SHARE TRACKING
# =============================================================================

cdef int get_corp_issued_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_issued_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil

cdef int get_corp_bank_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_bank_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil

cdef int get_corp_unissued_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_unissued_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil


# =============================================================================
# SHARE PRICE
# =============================================================================

cdef int get_corp_share_price(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_share_price(float* corp, CorpOffsets* c, int price) noexcept nogil

cdef int get_corp_price_index(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_price_index(float* corp, CorpOffsets* c, int index, float* hidden_price_indices, int corp_id) noexcept nogil


# =============================================================================
# RECEIVERSHIP
# =============================================================================

cdef bint is_corp_in_receivership(float* corp, CorpOffsets* c) noexcept nogil
cdef void set_corp_in_receivership(float* corp, CorpOffsets* c, bint in_recv) noexcept nogil


# =============================================================================
# COMPANY OWNERSHIP
# =============================================================================

cdef bint corp_owns_company(float* corp, CorpOffsets* c, int company_id) noexcept nogil
cdef void set_corp_owns_company(float* corp, CorpOffsets* c, int company_id, bint owns) noexcept nogil

cdef int get_corp_company_count(float* corp, CorpOffsets* c) noexcept nogil


# =============================================================================
# PRESIDENT LOOKUP
# =============================================================================

cdef int get_president_of_corp(GameState state, int corp_id, int num_players) noexcept nogil
cdef void set_active_player_to_president(GameState state, int corp_id, int num_players) noexcept
cdef int find_corp_owning_company(GameState state, int player_id, int company_id) noexcept nogil


# =============================================================================
# STARS CALCULATION
# =============================================================================

cdef int calculate_corp_company_stars(float* corp, CorpOffsets* c) noexcept nogil
cdef int calculate_target_stars(float* corp, CorpOffsets* c) noexcept nogil


# =============================================================================
# BANKRUPTCY HANDLING
# =============================================================================

cdef void handle_corp_bankruptcy(
    GameState state,
    int corp_id,
    int old_price_index,
    int num_players
) noexcept
