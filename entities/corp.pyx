# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Corporation entity implementation.

The Corporation handle owns every per-corp slot inside the compact
GameState plus the income / stars / price-movement bookkeeping. It is
fully stateless: each access derives its slot inline from the module-
level ``LAYOUT`` and ``CORP_FIELDS`` constants on ``core.state``, so
the singletons in ``CORPS`` can be reused with any GameState at any
player count.

Layout summary (per-corp block, all raw int16):
  active, cash, unissued_shares, issued_shares, bank_shares, income,
  stars, share_price, acquisition_proceeds, in_receivership,
  price_index, pending_price_move, raw_revenue, synergy_income,
  coo_cost, ability_income.

Company ownership and acquisition-pile membership live in the companies
section — there is no per-corp owned_companies bitmap. Corp logic goes
through company-module query helpers for bulk ownership scans rather
than reaching into the companies layout directly.

The synergy aggregation, the required-stars table, and the per-corp
ability-income formulas live as private cdef helpers below; they are
pure functions of corp + company state and stay narrowly scoped to
the one entity that needs them. Other modules (dividends, closing)
will get their own copies when their slices land.
"""

from libc.stdint cimport int16_t

from core.state cimport GameState, LAYOUT, CORP_FIELDS, TURN_OFFSETS
from core.data cimport (
    GameConstants,
    CorpIndices,
    COMPANY_FACE_VALUE,
    COMPANY_INCOME,
    COMPANY_STARS,
    COMPANY_SYNERGY,
    CORP_SHARE_COUNT,
    MARKET_PRICES,
)
from core.data import CORP_NAMES
from entities.company cimport (
    company_adjusted_income,
    company_fill_corp_company_ids,
    company_in_corp_acquisition,
    company_owned_by_corp,
)
from entities.player cimport (
    invalidate_all_player_caches,
)
from entities.turn cimport TurnState

# Late imports to avoid circular dependencies (resolved at runtime)
from entities import turn as turn_module
from entities import company as company_module
from entities import market as market_module
from entities import player as player_module

# Lazy accessor for the TURN singleton. See the corresponding comment in
# player.pyx — caching at module init would force entities.turn to be fully
# loaded before this module finished its own init, so we defer the lookup
# to first call. After that it's a cached pointer return under the GIL
# (every callsite is in a cpdef/def context, never inside nogil).
cdef TurnState _TURN_CACHED = None

cdef TurnState _TURN():
    global _TURN_CACHED
    if _TURN_CACHED is None:
        _TURN_CACHED = <TurnState>turn_module.TURN
    return _TURN_CACHED


# =============================================================================
# CORPORATION CACHE DIRTY MASK
# =============================================================================

cdef inline int _corp_cache_dirty_slot() noexcept nogil:
    return LAYOUT.turn_offset + TURN_OFFSETS.corp_cache_dirty


cdef inline int _all_corps_mask() noexcept nogil:
    return (1 << <int>GameConstants.NUM_CORPS) - 1


cdef inline bint _cache_dirty(GameState state, int corp_id) noexcept nogil:
    return (<int>state._data[_corp_cache_dirty_slot()] & (1 << corp_id)) != 0


cdef inline void _clear_cache_dirty(GameState state, int corp_id) noexcept nogil:
    cdef int slot = _corp_cache_dirty_slot()
    state._data[slot] = <int16_t>(<int>state._data[slot] & ~(1 << corp_id))


cdef void invalidate_corp_cache(GameState state, int corp_id) noexcept nogil:
    cdef int slot = _corp_cache_dirty_slot()
    state._data[slot] = <int16_t>(<int>state._data[slot] | (1 << corp_id))


cdef void invalidate_all_corp_caches(GameState state) noexcept nogil:
    state._data[_corp_cache_dirty_slot()] = <int16_t>_all_corps_mask()


# =============================================================================
# PURE-FUNCTION HELPERS
# =============================================================================

cpdef int calculate_price_move(int owned_stars, int required_stars) noexcept nogil:
    """
    Raw price movement based on star comparison.

    Clamped to [-2, +2] per RULES.md lines 318-323.
    """
    cdef int diff = owned_stars - required_stars
    if diff < -2:
        return -2
    if diff > 2:
        return 2
    return diff


cdef inline int _required_stars(int price_index, int issued_shares) noexcept nogil:
    """
    Required star count for a corporation to maintain its share price.

    Formula: round(issued_shares * price / 10), per the 18xx.games RSS
    implementation (target_stars). Returns 0 for out-of-range inputs.
    """
    cdef int price
    if price_index < 1 or price_index > 26:
        return 0
    if issued_shares < 2 or issued_shares > 7:
        return 0
    price = MARKET_PRICES[price_index]
    return <int>(issued_shares * price / 10.0 + 0.5)


cdef inline (int, int) _aggregate_synergies(int* company_ids, int num_companies) noexcept nogil:
    """
    Aggregate synergy bonuses for a list of company IDs.

    Each ordered pair contributes its directional bonus (so the
    bidirectional synergy matrix is read in both directions). A pair is
    "marked" if either direction has a non-zero bonus — the marker count
    feeds the Synergistic (S) corp's ability income.

    Returns (total_income, marker_count).
    """
    cdef int i, j
    cdef int total_income = 0
    cdef int marker_count = 0
    cdef int bonus_a_to_b, bonus_b_to_a
    cdef int has_synergy

    for i in range(num_companies):
        for j in range(i + 1, num_companies):
            has_synergy = 0

            bonus_a_to_b = COMPANY_SYNERGY[company_ids[i]][company_ids[j]]
            if bonus_a_to_b > 0:
                total_income += bonus_a_to_b
                has_synergy = 1

            bonus_b_to_a = COMPANY_SYNERGY[company_ids[j]][company_ids[i]]
            if bonus_b_to_a > 0:
                total_income += bonus_b_to_a
                has_synergy = 1

            if has_synergy:
                marker_count += 1

    return (total_income, marker_count)


# =============================================================================
# CACHE FLUSH COMPATIBILITY HELPERS
# =============================================================================

cdef void update_all_corp_caches(GameState state) noexcept:
    cdef int corp_id
    cdef Corporation corp
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        corp = <Corporation>CORPS[corp_id]
        corp._refresh_cache(state)


# =============================================================================
# CORPORATION CLASS
# =============================================================================

cdef class Corporation:
    """
    Entity handle for a single corporation.

    Instances are created once at module load with their corp_id and
    name. Every accessor reads its slot inline from the module-level
    ``LAYOUT`` / ``CORP_FIELDS`` constants on ``core.state``, so a
    single CORPS list works for any GameState at any player count.
    """

    def __cinit__(self, int corp_id, str name):
        self.corp_id = corp_id
        self.name = name

    # =========================================================================
    # SLOT HELPER (constant-offset arithmetic)
    # =========================================================================

    cdef inline int _slot(self, int field) noexcept nogil:
        """Absolute index of a per-corp field for this corp."""
        return LAYOUT.corps_offset + self.corp_id * LAYOUT.corp_stride + field

    # =========================================================================
    # NOGIL ACCESSORS (used by hot paths inside the engine)
    # =========================================================================

    cdef inline int _get_cash(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(CORP_FIELDS.cash)]

    cdef inline int _get_bank_shares(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(CORP_FIELDS.bank_shares)]

    cdef inline int _get_unissued_shares(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(CORP_FIELDS.unissued_shares)]

    cdef inline int _get_issued_shares(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(CORP_FIELDS.issued_shares)]

    cdef inline int _get_price_index(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(CORP_FIELDS.price_index)]

    cdef inline bint _is_active(self, GameState state) noexcept nogil:
        return state._data[self._slot(CORP_FIELDS.active)] == 1

    cdef inline bint _is_in_receivership(self, GameState state) noexcept nogil:
        return state._data[self._slot(CORP_FIELDS.in_receivership)] == 1

    cdef inline bint _owns_company(self, GameState state, int company_id) noexcept nogil:
        """Check if this corp currently owns the given company.

        Delegates to the company module, which owns the companies
        section storage and its layout details.
        """
        return company_owned_by_corp(state, company_id, self.corp_id)

    cdef inline bint _has_acquisition_company(self, GameState state, int company_id) noexcept nogil:
        """Check if this corp has the given company in its acquisition pile."""
        return company_in_corp_acquisition(state, company_id, self.corp_id)

    # =========================================================================
    # ACTIVE STATUS
    # =========================================================================

    cpdef bint is_active(self, GameState state):
        """Return True if the corp has been IPO'd."""
        return self._is_active(state)

    cpdef void set_active(self, GameState state, bint active):
        """Set whether the corp is active."""
        cdef bint old_active = self._is_active(state)
        state._data[self._slot(CORP_FIELDS.active)] = <int16_t>(1 if active else 0)
        if old_active != active:
            if active:
                invalidate_corp_cache(state, self.corp_id)
                invalidate_all_player_caches(state)
            else:
                self._clear_cache(state)
                invalidate_all_player_caches(state)

    cpdef void float_corp(self, GameState state, int player_id, int company_id,
                          int par_index, int float_shares=1):
        """
        Float corporation via IPO.

        This encapsulates the full IPO procedure for the corporation:
        1. Set corp active
        2. Transfer company to corp
        3. Claim market space and set price
        4. Set share distribution (unissued, issued, bank)
        5. Give player their shares (triggers automatic presidency)

        Player payment is NOT handled here — that's phase-specific.
        """
        cdef int total_shares = CORP_SHARE_COUNT[self.corp_id]
        cdef int issued = float_shares * 2  # Player + bank each get float_shares
        cdef int unissued_shares = total_shares - issued

        # 1. Activate the corp first so the subsequent ownership and price
        #    writes can mark this corp's derived cache dirty.
        self.set_active(state, True)

        # 2. Transfer company to corporation (marks the corp cache dirty)
        company_module.COMPANIES[company_id].transfer_to_corp(state, self.corp_id)

        # 3. Claim market space and set price
        market_module.MARKET.set_space_available(state, par_index, False)
        self.set_price_index(state, par_index)

        # 4. Set share distribution
        self.set_unissued_shares(state, unissued_shares)
        self.set_issued_shares(state, issued)
        # Bank starts with all issued shares; set_shares() moves float_shares to player
        self.set_bank_shares(state, issued)

        # 5. Give player their shares (auto-adjusts bank shares and presidency)
        player_module.PLAYERS[player_id].set_shares(state, self.corp_id, float_shares)

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Return corp cash (raw integer dollars)."""
        return self._get_cash(state)

    cpdef void set_cash(self, GameState state, int cash):
        """Set corp cash (raw integer dollars)."""
        state._data[self._slot(CORP_FIELDS.cash)] = <int16_t>cash
        invalidate_corp_cache(state, self.corp_id)

    cpdef void add_cash(self, GameState state, int amount):
        """Add to corp cash (negative `amount` subtracts)."""
        self.set_cash(state, self._get_cash(state) + amount)

    # =========================================================================
    # SHARE TRACKING
    # =========================================================================

    cpdef int get_unissued_shares(self, GameState state):
        """Return number of unissued shares remaining in the treasury."""
        return self._get_unissued_shares(state)

    cpdef void set_unissued_shares(self, GameState state, int shares):
        """Set number of unissued shares."""
        state._data[self._slot(CORP_FIELDS.unissued_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    cpdef int get_issued_shares(self, GameState state):
        """Return number of issued shares (held by players + bank)."""
        return self._get_issued_shares(state)

    cpdef void set_issued_shares(self, GameState state, int shares):
        """Set number of issued shares."""
        state._data[self._slot(CORP_FIELDS.issued_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    cpdef int get_bank_shares(self, GameState state):
        """Return number of shares held by the bank (sold by players)."""
        return self._get_bank_shares(state)

    cpdef void set_bank_shares(self, GameState state, int shares):
        """Set number of bank shares."""
        state._data[self._slot(CORP_FIELDS.bank_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    # =========================================================================
    # INCOME
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Return cached corp income (raw integer dollars)."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.income)]

    cpdef void set_income(self, GameState state, int income):
        """Set cached corp income (raw integer dollars)."""
        state._data[self._slot(CORP_FIELDS.income)] = <int16_t>income

    cpdef int get_raw_revenue(self, GameState state):
        """Return cached raw revenue before CoO and bonuses."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.raw_revenue)]

    cpdef int get_synergy_income(self, GameState state):
        """Return cached synergy-income component."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.synergy_income)]

    cpdef int get_coo_cost(self, GameState state):
        """Return cached CoO component (stored as a non-positive value)."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.coo_cost)]

    cpdef int get_ability_income(self, GameState state):
        """Return cached ability-income component."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.ability_income)]

    # =========================================================================
    # STARS
    # =========================================================================
    #
    cpdef int get_stars(self, GameState state):
        """Compatibility alias for total cached stars."""
        return self.get_total_stars(state)

    cpdef int get_total_stars(self, GameState state):
        """Return total cached stars (company + cash + SI ability bonus)."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.total_stars)]

    cpdef int get_cash_stars(self, GameState state):
        """Return cached cash-stars component."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.cash_stars)]

    cpdef int get_company_stars(self, GameState state):
        """Return cached company-stars component."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.company_stars)]

    cpdef void recalculate_cash_stars(self, GameState state):
        """Compatibility wrapper that refreshes the full corp cache."""
        self._refresh_cache(state)

    cpdef void recalculate_company_stars(self, GameState state):
        """Compatibility wrapper that refreshes the full corp cache."""
        self._refresh_cache(state)

    cpdef void update_pending_price_move(self, GameState state):
        """Compatibility wrapper that refreshes the full corp cache."""
        self._refresh_cache(state)

    cpdef int get_pending_price_move(self, GameState state):
        """Return cached pending price-movement scalar."""
        if _cache_dirty(state, self.corp_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(CORP_FIELDS.pending_price_move)]

    # =========================================================================
    # SHARE PRICE / MARKET INDEX
    # =========================================================================

    cpdef int get_share_price(self, GameState state):
        """Return current share price (raw integer dollars)."""
        return <int>state._data[self._slot(CORP_FIELDS.share_price)]

    cpdef void set_share_price(self, GameState state, int price):
        """Set share price (raw integer dollars)."""
        cdef int slot = self._slot(CORP_FIELDS.share_price)
        cdef int old_price = <int>state._data[slot]
        state._data[slot] = <int16_t>price
        if old_price != price:
            invalidate_corp_cache(state, self.corp_id)
            invalidate_all_player_caches(state)

    cpdef int get_price_index(self, GameState state):
        """Return market price index (0-26, where 0 is bankruptcy)."""
        return self._get_price_index(state)

    cpdef void set_price_index(self, GameState state, int index):
        """Set market price index and share price."""
        assert 0 <= index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"price index {index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        state._data[self._slot(CORP_FIELDS.price_index)] = <int16_t>index
        self.set_share_price(state, MARKET_PRICES[index])

    # =========================================================================
    # ACQUISITION PROCEEDS
    # =========================================================================

    cpdef int get_acquisition_proceeds(self, GameState state):
        """Return accumulated acquisition proceeds (for dividend calculation)."""
        return <int>state._data[self._slot(CORP_FIELDS.acquisition_proceeds)]

    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds):
        """Set accumulated acquisition proceeds."""
        state._data[self._slot(CORP_FIELDS.acquisition_proceeds)] = <int16_t>proceeds

    # =========================================================================
    # RECEIVERSHIP
    # =========================================================================

    cpdef bint is_in_receivership(self, GameState state):
        """Return True if the corp is in receivership (no president)."""
        return self._is_in_receivership(state)

    cpdef void set_in_receivership(self, GameState state, bint in_recv):
        """Set whether the corp is in receivership."""
        state._data[self._slot(CORP_FIELDS.in_receivership)] = <int16_t>(1 if in_recv else 0)

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Return True if the corp owns the given company.

        Backed by the shared company_locations / company_owner_ids arrays.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return self._owns_company(state, company_id)

    cpdef int count_companies(self, GameState state, bint include_acquisition=False):
        """
        Count companies owned by this corp.

        Used for last-company rule validation. During the ACQUISITION
        phase, ``include_acquisition=True`` also counts companies sitting
        in this corp's acquisition pile (which will become owned once
        the phase ends).
        """
        return company_fill_corp_company_ids(
            state, self.corp_id, include_acquisition, <int*>NULL)

    # =========================================================================
    # DERIVED CACHE REFRESH
    # =========================================================================

    cdef void _clear_cache(self, GameState state) noexcept:
        state._data[self._slot(CORP_FIELDS.income)] = 0
        state._data[self._slot(CORP_FIELDS.total_stars)] = 0
        state._data[self._slot(CORP_FIELDS.cash_stars)] = 0
        state._data[self._slot(CORP_FIELDS.company_stars)] = 0
        state._data[self._slot(CORP_FIELDS.pending_price_move)] = 0
        state._data[self._slot(CORP_FIELDS.raw_revenue)] = 0
        state._data[self._slot(CORP_FIELDS.synergy_income)] = 0
        state._data[self._slot(CORP_FIELDS.coo_cost)] = 0
        state._data[self._slot(CORP_FIELDS.ability_income)] = 0
        _clear_cache_dirty(state, self.corp_id)

    cdef void _refresh_cache(self, GameState state):
        """
        Recompute all derived corporation fields from authoritative state.

        This folds stars, pending price move, and the full income
        breakdown behind a single dirty bit. When the cache is clean,
        getters can read the stored slots directly.
        """
        cdef int company_id, base_income, fv, i
        cdef int adjusted_income_sum = 0
        cdef int raw_revenue_sum = 0
        cdef int highest_fv = 0
        cdef int highest_fv_income = 0
        cdef int company_stars = 0
        cdef int cash_stars = 0
        cdef int total_stars = 0
        cdef int price_index
        cdef int issued_shares
        cdef int required
        cdef int pending_move = 0
        cdef int company_ids[36]
        cdef int synergy_income = 0
        cdef int synergy_markers = 0
        cdef int total_coo, ability, total
        cdef int company_count

        if not self._is_active(state):
            self._clear_cache(state)
            return

        company_count = company_fill_corp_company_ids(
            state, self.corp_id, True, company_ids)

        # Single ownership scan drives both star totals and the income
        # breakdown, so all corp-derived fields stay coherent together.
        for i in range(company_count):
            company_id = company_ids[i]
            company_stars += COMPANY_STARS[company_id]
            adjusted_income_sum += company_adjusted_income(state, company_id)

            base_income = COMPANY_INCOME[company_id]
            raw_revenue_sum += base_income
            fv = COMPANY_FACE_VALUE[company_id]
            if fv > highest_fv:
                highest_fv = fv
                highest_fv_income = base_income
            elif fv == highest_fv and base_income > highest_fv_income:
                highest_fv_income = base_income

        cash_stars = self._get_cash(state) // 10 if self._get_cash(state) > 0 else 0
        total_stars = company_stars + cash_stars
        if self.corp_id == <int>CorpIndices.CORP_SI:
            total_stars += 2

        price_index = self._get_price_index(state)
        issued_shares = self._get_issued_shares(state)
        required = _required_stars(price_index, issued_shares)
        pending_move = calculate_price_move(total_stars, required)

        total_coo = raw_revenue_sum - adjusted_income_sum

        if company_count > 1:
            (synergy_income, synergy_markers) = _aggregate_synergies(company_ids, company_count)

        ability = 0
        if self.corp_id == <int>CorpIndices.CORP_VM:
            ability = total_coo if total_coo < 10 else 10
        elif self.corp_id == <int>CorpIndices.CORP_PR:
            ability = company_count
        elif self.corp_id == <int>CorpIndices.CORP_DA:
            ability = highest_fv_income
        elif self.corp_id == <int>CorpIndices.CORP_S:
            ability = synergy_markers // 2

        total = raw_revenue_sum - total_coo + synergy_income + ability

        state._data[self._slot(CORP_FIELDS.income)] = <int16_t>total
        state._data[self._slot(CORP_FIELDS.total_stars)] = <int16_t>total_stars
        state._data[self._slot(CORP_FIELDS.cash_stars)] = <int16_t>cash_stars
        state._data[self._slot(CORP_FIELDS.company_stars)] = <int16_t>company_stars
        state._data[self._slot(CORP_FIELDS.pending_price_move)] = <int16_t>pending_move
        state._data[self._slot(CORP_FIELDS.raw_revenue)] = <int16_t>raw_revenue_sum
        state._data[self._slot(CORP_FIELDS.synergy_income)] = <int16_t>synergy_income
        state._data[self._slot(CORP_FIELDS.coo_cost)] = <int16_t>(-total_coo)
        state._data[self._slot(CORP_FIELDS.ability_income)] = <int16_t>ability
        _clear_cache_dirty(state, self.corp_id)

    # =========================================================================
    # BANKRUPTCY
    # =========================================================================

    cpdef void go_bankrupt(self, GameState state):
        """
        Execute bankruptcy procedure for this corporation.

        Triggered when share price drops to index 0. This is a complete
        reset:
        - All owned companies removed from game
        - All shares returned to unissued (cleared from players)
        - Corp cash returned to bank (set to 0)
        - Market space freed
        - Corp deactivated and available for future IPO
        """
        cdef int company_id, player_id, current_index

        # Step 1: Remove all companies attached to the corp from game.
        # In normal rules flow a bankrupt corp should not still have an
        # acquisition pile, but we clean up LOC_CORP_ACQ defensively so the
        # foundation layer never leaves companies attached to an inactive corp.
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if (self._owns_company(state, company_id)
                    or self._has_acquisition_company(state, company_id)):
                company_module.COMPANIES[company_id].remove_from_game(state)

        # Step 2: Return all shares to unissued — clear player shares.
        # set_shares(0) auto-moves each player's shares to bank.
        for player_id in range(_TURN()._get_num_players(state)):
            player_module.PLAYERS[player_id].set_shares(state, self.corp_id, 0)

        # Step 3: Reset corp share counts (bank accumulated player shares above).
        self.set_unissued_shares(state, CORP_SHARE_COUNT[self.corp_id])
        self.set_issued_shares(state, 0)
        self.set_bank_shares(state, 0)

        # Step 4: Return money to bank — clear corp cash.
        self.set_cash(state, 0)

        # Step 5: Free market space if needed.
        current_index = self._get_price_index(state)
        if current_index > 0:
            market_module.MARKET.set_space_available(state, current_index, True)

        # Step 6: Clear remaining state and deactivate. Reset the market
        # price before deactivation so the final set_active(False) leaves
        # the derived cache clean and zeroed.
        self.set_price_index(state, 0)
        self.set_in_receivership(state, False)
        self.set_acquisition_proceeds(state, 0)
        self.set_active(state, False)

    # =========================================================================
    # PRESIDENT
    # =========================================================================

    cpdef int get_president_id(self, GameState state):
        """Return player_id of the corp's president, or -1 if in receivership."""
        cdef int player_id
        for player_id in range(_TURN()._get_num_players(state)):
            if player_module.PLAYERS[player_id].is_president_of(state, self.corp_id):
                return player_id
        return -1

    # =========================================================================
    # ACQUISITION PILE
    # =========================================================================

    cpdef bint has_acquisition_company(self, GameState state, int company_id):
        """Return True if this corp has the given company in its acquisition pile."""
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return self._has_acquisition_company(state, company_id)


# =============================================================================
# GLOBAL CORPORATION INSTANCES
# =============================================================================

# List indexed by corp_id (consistent with PLAYERS, COMPANIES)
CORPS = [Corporation(i, CORP_NAMES[i]) for i in range(<int>GameConstants.NUM_CORPS)]
# Dict by name for convenience
CORPS_BY_NAME = {c.name: c for c in CORPS}
