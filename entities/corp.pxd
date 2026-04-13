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
# CORPORATION CACHE
# =============================================================================

cdef void invalidate_corp_cache(GameState state, int corp_id) noexcept nogil
cdef void invalidate_all_corp_caches(GameState state) noexcept nogil
cdef int count_corp_companies(GameState state, int corp_id, bint include_acquisition) noexcept nogil


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
    cdef int _count_companies(self, GameState state, bint include_acquisition) noexcept nogil
    cdef void _refresh_cache(self, GameState state)
    cdef void _clear_cache(self, GameState state) noexcept

    # Active status
    cpdef bint is_active(self, GameState state)
    cpdef void set_active(self, GameState state, bint active)
    cpdef void float_corp(self, GameState state, int player_id, int company_id,
                          int market_index, int float_shares=*)

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

    # Income
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef int get_raw_revenue(self, GameState state)
    cpdef int get_synergy_income(self, GameState state)
    cpdef int get_coo_cost(self, GameState state)
    cpdef int get_ability_income(self, GameState state)

    # Stars. company_stars is cached behind the corp dirty bit; cash stars,
    # total stars, and pending price move are derived on demand from the
    # cached company value plus current cash / price state.
    cpdef int get_total_stars(self, GameState state)
    cpdef int get_cash_stars(self, GameState state)
    cpdef int get_company_stars(self, GameState state)

    # Share price / market index
    cpdef int get_share_price(self, GameState state)
    cpdef int get_price_index(self, GameState state)
    cpdef void set_price_index(self, GameState state, int index)
    cpdef int get_pending_price_move(self, GameState state)

    # Acquisition proceeds
    cpdef int get_acquisition_proceeds(self, GameState state)
    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds)

    # Receivership
    cpdef bint is_in_receivership(self, GameState state)
    cpdef void set_in_receivership(self, GameState state, bint in_recv)

    # Company ownership
    cpdef bint owns_company(self, GameState state, int company_id)
    cpdef int count_companies(self, GameState state, bint include_acquisition=*)

    # Bankruptcy
    cpdef void go_bankrupt(self, GameState state)

    # President
    cpdef int get_president_id(self, GameState state)
    cdef void _recalculate_presidency(self, GameState state)

    # ACQ_OFFER passed flag
    cpdef bint has_passed_acq_offer(self, GameState state)
    cpdef void set_passed_acq_offer(self, GameState state, bint passed)

    # Acquisition pile
    cpdef bint has_acquisition_company(self, GameState state, int company_id)
