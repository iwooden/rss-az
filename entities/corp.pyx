# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Corporation entity implementation.

Provides clean getter/setter access to corporation state in the game state vector.
Each Corporation instance is bound to a specific corp_id and caches offsets
for fast repeated access.
"""

from core.state cimport GameState, StateLayout, CorpFieldOffsets
from core.data cimport (
    GameConstants, CASH_DIVISOR, SHARE_DIVISOR, STAR_DIVISOR, MARKET_PRICES,
    get_company_income, get_company_stars, get_cost_of_ownership,
    compute_synergy_bonuses
)
from core.data import CORP_NAMES
from entities.encoding cimport set_one_hot
from entities import turn as turn_module


# =============================================================================
# LOW-LEVEL NOGIL ACCESSORS
# =============================================================================

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


cdef CorpOffsets get_corp_offsets() noexcept nogil:
    """
    Compute field offsets within corp data block.

    The corporation state is stored as a contiguous float array with the following layout:
    - active (1)
    - cash (1)
    - unissued_shares (1)
    - issued_shares (1)
    - bank_shares (1)
    - income (1)
    - stars (1)
    - share_price (1)
    - acquisition_proceeds (1)
    - in_receivership (1)
    - price_index (26) - omitted from struct (uses hidden compact storage)
    - owned_companies (36) - omitted from struct (not used in masks)
    - acquisition_companies (36) - omitted from struct (not used in masks)
    """
    cdef CorpOffsets c
    cdef int offset = 0

    c.active = offset
    offset += 1
    c.cash = offset
    offset += 1
    c.unissued_shares = offset
    offset += 1
    c.issued_shares = offset
    offset += 1
    c.bank_shares = offset
    offset += 1
    c.income = offset
    offset += 1
    c.stars = offset
    offset += 1
    c.share_price = offset
    offset += 1
    c.acquisition_proceeds = offset
    offset += 1
    c.in_receivership = offset

    return c


cdef inline bint is_corp_active(float* corp, CorpOffsets* c) noexcept nogil:
    """Check if corporation is active (has been IPO'd)."""
    return corp[c.active] == 1.0


cdef inline int get_corp_cash(float* corp, CorpOffsets* c) noexcept nogil:
    """Get corporation's cash (integer dollars)."""
    return <int>(corp[c.cash] * CASH_DIVISOR + 0.5)


cdef inline int get_corp_bank_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of shares held by the bank (sold by players)."""
    return <int>(corp[c.bank_shares] * SHARE_DIVISOR + 0.5)


cdef inline int get_corp_unissued_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of unissued shares remaining in treasury."""
    return <int>(corp[c.unissued_shares] * SHARE_DIVISOR + 0.5)


cdef inline int get_corp_issued_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of issued shares (held by players)."""
    return <int>(corp[c.issued_shares] * SHARE_DIVISOR + 0.5)


cdef inline bint is_corp_in_receivership(float* corp, CorpOffsets* c) noexcept nogil:
    """Check if corporation is in receivership (no president)."""
    return corp[c.in_receivership] == 1.0


# =============================================================================
# HIGH-LEVEL ENTITY CLASS
# =============================================================================

cdef class Corporation:
    """
    Entity handle for accessing corporation state.

    Corporations are instantiated once at module load with their corp_id and name.
    Offsets are computed on first access to a GameState via initialize().
    All methods take GameState as first argument for stateless operation.
    """

    def __cinit__(self, int corp_id, str name):
        self.corp_id = corp_id
        self.name = name
        self._base_offset = 0
        self._num_players = 0
        self._hidden_price_index_offset = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this Corporation instance.
        """
        cdef StateLayout layout = state._layout
        cdef CorpFieldOffsets fields = state._corp_fields

        self._num_players = state._num_players
        self._base_offset = layout.corps_offset + (self.corp_id * layout.corp_stride)

        # Hidden state offset for fast price index access (one slot per corp)
        # hidden_corp_price_indices_offset is already absolute
        self._hidden_price_index_offset = layout.hidden_corp_price_indices_offset + self.corp_id

        # Cache absolute offsets for each field
        self._active_offset = self._base_offset + fields.active
        self._cash_offset = self._base_offset + fields.cash
        self._unissued_shares_offset = self._base_offset + fields.unissued_shares
        self._issued_shares_offset = self._base_offset + fields.issued_shares
        self._bank_shares_offset = self._base_offset + fields.bank_shares
        self._income_offset = self._base_offset + fields.income
        self._stars_offset = self._base_offset + fields.stars
        self._share_price_offset = self._base_offset + fields.share_price
        self._acquisition_proceeds_offset = self._base_offset + fields.acquisition_proceeds
        self._in_receivership_offset = self._base_offset + fields.in_receivership
        self._price_index_offset = self._base_offset + fields.price_index
        self._owned_companies_offset = self._base_offset + fields.owned_companies
        self._acquisition_companies_offset = self._base_offset + fields.acquisition_companies

    # =========================================================================
    # ACTIVE STATUS
    # =========================================================================

    cpdef bint is_active(self, GameState state):
        """Check if corporation is active (has been IPO'd)."""
        return state._data[self._active_offset] == 1.0

    cpdef void set_active(self, GameState state, bint active):
        """Set whether corporation is active."""
        state._data[self._active_offset] = 1.0 if active else 0.0

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Get corporation's cash (integer dollars)."""
        return <int>(state._data[self._cash_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_cash(self, GameState state, int cash):
        """Set corporation's cash (integer dollars)."""
        state._data[self._cash_offset] = <float>cash / CASH_DIVISOR

    cpdef void add_cash(self, GameState state, int amount):
        """Add to corporation's cash (can be negative to subtract)."""
        cdef int current = self.get_cash(state)
        self.set_cash(state, current + amount)

    # =========================================================================
    # SHARE TRACKING
    # =========================================================================

    cpdef int get_unissued_shares(self, GameState state):
        """Get number of unissued shares remaining in treasury."""
        return <int>(state._data[self._unissued_shares_offset] * SHARE_DIVISOR + 0.5)

    cpdef void set_unissued_shares(self, GameState state, int shares):
        """Set number of unissued shares."""
        state._data[self._unissued_shares_offset] = <float>shares / SHARE_DIVISOR

    cpdef int get_issued_shares(self, GameState state):
        """Get number of issued shares (held by players)."""
        return <int>(state._data[self._issued_shares_offset] * SHARE_DIVISOR + 0.5)

    cpdef void set_issued_shares(self, GameState state, int shares):
        """Set number of issued shares."""
        state._data[self._issued_shares_offset] = <float>shares / SHARE_DIVISOR

    cpdef int get_bank_shares(self, GameState state):
        """Get number of shares held by the bank (sold by players)."""
        return <int>(state._data[self._bank_shares_offset] * SHARE_DIVISOR + 0.5)

    cpdef void set_bank_shares(self, GameState state, int shares):
        """Set number of bank shares."""
        state._data[self._bank_shares_offset] = <float>shares / SHARE_DIVISOR

    # =========================================================================
    # INCOME AND STARS
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Get corporation's total income (from owned companies)."""
        return <int>(state._data[self._income_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_income(self, GameState state, int income):
        """Set corporation's income."""
        state._data[self._income_offset] = <float>income / CASH_DIVISOR

    cpdef int get_stars(self, GameState state):
        """Get corporation's total stars (from owned companies)."""
        return <int>(state._data[self._stars_offset] * STAR_DIVISOR + 0.5)

    cpdef void set_stars(self, GameState state, int stars):
        """Set corporation's stars."""
        state._data[self._stars_offset] = <float>stars / STAR_DIVISOR

    # =========================================================================
    # SHARE PRICE
    # =========================================================================

    cpdef int get_share_price(self, GameState state):
        """Get current share price (integer dollars)."""
        return <int>(state._data[self._share_price_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_share_price(self, GameState state, int price):
        """Set share price (integer dollars)."""
        state._data[self._share_price_offset] = <float>price / CASH_DIVISOR

    cpdef int get_price_index(self, GameState state):
        """Get market price index (0-26, where 0 is bankruptcy). Uses hidden compact storage for O(1) access."""
        return <int>state._data[self._hidden_price_index_offset]

    cpdef void set_price_index(self, GameState state, int index):
        """Set market price index. Updates both one-hot and hidden compact storage."""
        set_one_hot(state._data, self._price_index_offset, GameConstants.NUM_MARKET_SPACES, index)
        if 0 <= index < GameConstants.NUM_MARKET_SPACES:
            state._data[self._hidden_price_index_offset] = <float>index
            # Also update the denormalized share_price field
            self.set_share_price(state, MARKET_PRICES[index])

    # =========================================================================
    # ACQUISITION PROCEEDS
    # =========================================================================

    cpdef int get_acquisition_proceeds(self, GameState state):
        """Get accumulated acquisition proceeds (for dividend calculation)."""
        return <int>(state._data[self._acquisition_proceeds_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds):
        """Set acquisition proceeds."""
        state._data[self._acquisition_proceeds_offset] = <float>proceeds / CASH_DIVISOR

    # =========================================================================
    # RECEIVERSHIP
    # =========================================================================

    cpdef bint is_in_receivership(self, GameState state):
        """Check if corporation is in receivership (no president)."""
        return state._data[self._in_receivership_offset] == 1.0

    cpdef void set_in_receivership(self, GameState state, bint in_recv):
        """Set whether corporation is in receivership."""
        state._data[self._in_receivership_offset] = 1.0 if in_recv else 0.0

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Check if corporation owns a company."""
        return state._data[self._owned_companies_offset + company_id] == 1.0

    cpdef void set_owns_company(self, GameState state, int company_id, bint owns):
        """Set whether corporation owns a company."""
        state._data[self._owned_companies_offset + company_id] = 1.0 if owns else 0.0

    # =========================================================================
    # INCOME CALCULATION
    # =========================================================================

    cpdef int calculate_income(self, GameState state):
        """
        Calculate total income for corporation with synergy bonuses.

        Formula: (sum_printed_income - total_coo) + synergy

        Note: Special abilities (PR, DA, S, VM) are NOT implemented here.
        That is deferred to Phase 22-02.

        Returns:
            Total income (can be negative)
        """
        cdef int company_id, base_income, stars, coo_value
        cdef int coo_level = turn_module.TURN.get_coo_level(state)

        # Accumulators
        cdef int gross_printed_income = 0
        cdef int total_coo = 0
        cdef int company_count = 0

        # Company ID collection for synergy calculation
        cdef int company_ids[36]

        # First pass: collect companies, sum printed income, sum CoO
        for company_id in range(GameConstants.NUM_COMPANIES):
            if self.owns_company(state, company_id):
                company_ids[company_count] = company_id
                company_count += 1

                base_income = get_company_income(company_id)
                gross_printed_income += base_income

                stars = get_company_stars(company_id)
                coo_value = get_cost_of_ownership(coo_level, stars)
                total_coo += coo_value

        # Compute synergy bonuses
        cdef int synergy_income = 0
        cdef int synergy_markers = 0
        if company_count > 1:
            (synergy_income, synergy_markers) = compute_synergy_bonuses(company_ids, company_count)

        # Final formula: printed - CoO + synergy
        # Note: Special abilities will be added in Phase 22-02
        return gross_printed_income - total_coo + synergy_income

    # =========================================================================
    # ACQUISITION PILE
    # =========================================================================

    cpdef bint has_acquisition_company(self, GameState state, int company_id):
        """Check if company is in corporation's acquisition pile (pending integration)."""
        return state._data[self._acquisition_companies_offset + company_id] == 1.0

    cpdef void set_acquisition_company(self, GameState state, int company_id, bint has):
        """Set whether company is in acquisition pile."""
        state._data[self._acquisition_companies_offset + company_id] = 1.0 if has else 0.0


# =============================================================================
# GLOBAL CORPORATION INSTANCES
# =============================================================================

# List indexed by corp_id (consistent with PLAYERS, COMPANIES)
CORPS = [Corporation(i, CORP_NAMES[i]) for i in range(GameConstants.NUM_CORPS)]
# Dict by name for convenience
CORPS_BY_NAME = {c.name: c for c in CORPS}
