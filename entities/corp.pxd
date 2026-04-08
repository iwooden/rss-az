# cython: language_level=3
"""
Corporation entity declarations.

Stateless handle around the per-corp block in the compact GameState.
Every read/write derives its slot inline from the module-level
``LAYOUT`` and ``CORP_FIELDS`` constants on ``core.state``; there is
no per-instance offset cache and no initialize() step. The handle's
only state is its ``corp_id`` and display ``name``, so a single
``CORPS`` list can be reused with any GameState at any player count.
Player-id bounds checks call ``TURN._get_num_players(state)`` (the
canonical slot inside the turn block).

All values are raw int16 — no normalization, no one-hot encoding.
Company ownership is read from the shared ``company_locations`` /
``company_owner_ids`` arrays via ``LOC_CORP`` / ``LOC_CORP_ACQ``;
there is no per-corp owned_companies bitmap any more.

The synergy bonus aggregation, the required-stars table, and the
ability-income formulas live as private cdef helpers inside
``corp.pyx`` — they are pure functions of corp + company state and
do not belong on ``core.data``, which is data-only.
"""

from core.state cimport GameState


# =============================================================================
# INCOME BREAKDOWN
# =============================================================================

cdef struct IncomeBreakdown:
    int total
    int raw_revenue
    int synergy_income
    int coo_cost          # negative value
    int ability_income


# =============================================================================
# PRICE-MOVEMENT HELPER (pure function, no GameState required)
# =============================================================================

cpdef int calculate_price_move(int owned_stars, int required_stars) noexcept nogil


# =============================================================================
# CORPORATION CLASS
# =============================================================================

cdef class Corporation:
    cdef readonly int corp_id
    cdef readonly str name

    # Slot helper (constant-offset arithmetic)
    cdef inline int _slot(self, int field) noexcept nogil

    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef int _get_cash(self, GameState state) noexcept nogil
    cdef int _get_bank_shares(self, GameState state) noexcept nogil
    cdef int _get_unissued_shares(self, GameState state) noexcept nogil
    cdef int _get_issued_shares(self, GameState state) noexcept nogil
    cdef int _get_price_index(self, GameState state) noexcept nogil
    cdef bint _is_active(self, GameState state) noexcept nogil
    cdef bint _is_in_receivership(self, GameState state) noexcept nogil
    cdef bint _owns_company(self, GameState state, int company_id) noexcept nogil
    cdef bint _has_acquisition_company(self, GameState state, int company_id) noexcept nogil

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

    # Share price / market index
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

    # Company ownership
    cpdef bint owns_company(self, GameState state, int company_id)
    cpdef int count_companies(self, GameState state, bint include_acquisition=*)

    # Star / pending-move recalculation
    cpdef void recalculate_stars(self, GameState state)
    cpdef void update_pending_price_move(self, GameState state)

    # Income calculation
    cpdef int calculate_income(self, GameState state)
    cpdef void apply_income(self, GameState state, int income)

    # Bankruptcy
    cpdef void go_bankrupt(self, GameState state)

    # President lookup
    cpdef int get_president_id(self, GameState state)

    # Acquisition pile
    cpdef bint has_acquisition_company(self, GameState state, int company_id)
