# cython: language_level=3
"""
Corporation entity declarations.
"""

from core.state cimport GameState


# =============================================================================
# LOW-LEVEL NOGIL ACCESSORS
# =============================================================================

cdef struct IncomeBreakdown:
    int total
    int raw_revenue
    int synergy_income
    int coo_cost          # negative value
    int ability_income

cdef struct CorpOffsets:
    # Offsets within a corporation's data block in the state vector
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

# Offset computation
cdef CorpOffsets get_corp_offsets() noexcept nogil
cpdef int calculate_price_move(int owned_stars, int required_stars) noexcept nogil

# Corp state accessors (raw pointer, nogil)
cdef bint is_corp_active(float* corp, CorpOffsets* c) noexcept nogil
cdef int get_corp_cash(float* corp, CorpOffsets* c) noexcept nogil
cdef int get_corp_bank_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef int get_corp_unissued_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef int get_corp_issued_shares(float* corp, CorpOffsets* c) noexcept nogil
cdef bint is_corp_in_receivership(float* corp, CorpOffsets* c) noexcept nogil


# =============================================================================
# HIGH-LEVEL ENTITY CLASS
# =============================================================================

cdef class Corporation:
    cdef readonly int corp_id
    cdef readonly str name
    cdef int _base_offset      # Cached offset to this corp's data in state array
    cdef int _num_players      # Cached player count

    # Hidden state offset for fast price index access
    cdef int _hidden_price_index_offset

    # Field offsets within corp stride (cached on first use)
    cdef int _active_offset
    cdef int _cash_offset
    cdef int _unissued_shares_offset
    cdef int _issued_shares_offset
    cdef int _bank_shares_offset
    cdef int _income_offset
    cdef int _stars_offset
    cdef int _share_price_offset
    cdef int _acquisition_proceeds_offset
    cdef int _in_receivership_offset
    cdef int _price_index_norm_offset
    cdef int _pending_price_move_offset
    cdef int _raw_revenue_offset
    cdef int _synergy_income_offset
    cdef int _coo_cost_offset
    cdef int _ability_income_offset
    cdef int _owned_companies_offset
    cdef int _company_incomes_offset  # Global company_incomes array offset

    # Hidden state offsets for acquisition company lookups
    cdef int _hidden_company_locations_offset
    cdef int _hidden_company_owner_ids_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Active status
    cpdef bint is_active(self, GameState state)
    cpdef void set_active(self, GameState state, bint active)
    cpdef void float_corp(self, GameState state, int player_id, int company_id,
                          int par_index, int float_shares=*)

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Share tracking
    cpdef int get_unissued_shares(self, GameState state)
    cpdef void set_unissued_shares(self, GameState state, int shares)
    cpdef int get_issued_shares(self, GameState state)
    cpdef void set_issued_shares(self, GameState state, int shares)
    cpdef int get_bank_shares(self, GameState state)
    cpdef void set_bank_shares(self, GameState state, int shares)

    # Income and stars
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef int get_stars(self, GameState state)
    cpdef void set_stars(self, GameState state, int stars)

    # Share price
    cpdef int get_share_price(self, GameState state)
    cpdef void set_share_price(self, GameState state, int price)
    cpdef int get_price_index(self, GameState state)
    cpdef void set_price_index(self, GameState state, int index)

    # Acquisition proceeds
    cpdef int get_acquisition_proceeds(self, GameState state)
    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds)

    # Receivership
    cpdef bint is_in_receivership(self, GameState state)
    cpdef void set_in_receivership(self, GameState state, bint in_recv)

    # Company ownership (use Company.transfer_to_corp() to set)
    cpdef bint owns_company(self, GameState state, int company_id)
    cdef inline bint _owns_company_nogil(self, float* data, int company_id) noexcept nogil
    cpdef int count_companies(self, GameState state, bint include_acquisition=*)

    # Star recalculation
    cpdef void recalculate_stars(self, GameState state)
    cpdef void update_pending_price_move(self, GameState state)

    # Income calculation
    cdef IncomeBreakdown _calculate_income_nogil(self, float* data, int coo_level) noexcept nogil
    cpdef int calculate_income(self, GameState state)
    cpdef void apply_income(self, GameState state, int income)

    # Bankruptcy
    cpdef void go_bankrupt(self, GameState state)

    # President
    cpdef int get_president_id(self, GameState state)

    # Acquisition pile (use Company.transfer_to_corp_acquisition() to set)
    cpdef bint has_acquisition_company(self, GameState state, int company_id)
