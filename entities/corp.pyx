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
  company_stars, acquisition_proceeds, in_receivership, price_index,
  raw_revenue, synergy_income, coo_cost, ability_income.

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
    ALL_PAR_PRICES,
    COMPANY_FACE_VALUE,
    COMPANY_INCOME,
    COMPANY_STARS,
    COMPANY_SYNERGY,
    CORP_SHARE_COUNT,
    MARKET_PRICES,
    PRICE_TO_MARKET_INDEX,
)
from core.data import CORP_NAMES
from entities.company cimport (
    company_adjusted_income,
    company_fill_corp_company_ids,
    company_in_corp_acquisition,
    company_owned_by_corp,
)
from entities.player cimport (
    Player,
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


cdef inline int _corp_slot(int corp_id, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp_id * CORP_FIELDS.size + field


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


cdef int count_corp_companies(
    GameState state, int corp_id, bint include_acquisition,
) noexcept nogil:
    return company_fill_corp_company_ids(
        state, corp_id, include_acquisition, <int*>NULL)


cdef bint corp_is_active(GameState state, int corp_id) noexcept nogil:
    return state._data[_corp_slot(corp_id, CORP_FIELDS.active)] == 1


cdef int corp_cash(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.cash)]


cdef int corp_unissued_shares(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.unissued_shares)]


cdef int corp_issued_shares(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.issued_shares)]


cdef int corp_bank_shares(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.bank_shares)]


cdef int corp_price_index(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.price_index)]


cdef int corp_share_price(GameState state, int corp_id) noexcept nogil:
    return MARKET_PRICES[corp_price_index(state, corp_id)]


cdef int corp_acquisition_proceeds(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.acquisition_proceeds)]


cdef bint corp_is_in_receivership(GameState state, int corp_id) noexcept nogil:
    return state._data[_corp_slot(corp_id, CORP_FIELDS.in_receivership)] == 1


cdef int corp_president_id(GameState state, int corp_id) noexcept nogil:
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.president_id)]


cdef bint corp_has_passed_acq_offer(GameState state, int corp_id) noexcept nogil:
    return state._data[_corp_slot(corp_id, CORP_FIELDS.passed_acq_offer)] == 1


cdef bint corp_owns_company(GameState state, int corp_id, int company_id) noexcept nogil:
    return company_owned_by_corp(state, company_id, corp_id)


cdef bint corp_has_acquisition_company(
    GameState state, int corp_id, int company_id,
) noexcept nogil:
    return company_in_corp_acquisition(state, company_id, corp_id)


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


# =============================================================================
# SIMULATION HELPERS (dividend / float)
# =============================================================================

cdef int _simulate_dividend_price_move(
    GameState state, int corp_id, int amount_per_share,
) noexcept nogil:
    """Predicted market-index delta for ``corp_id`` paying ``amount_per_share``.

    The dividend amount reduces corp cash, which reduces cash_stars
    (``floor(cash/10)``), which shifts total_stars relative to required
    stars. ``calculate_price_move`` clamps the result to ``[-2, +2]``.

    Amount 0 recovers the "no dividend" pending-price-move path, which
    is what ``_refresh_corp_cache`` stores in the corp cache slot.
    """
    cdef int cash = corp_cash(state, corp_id)
    cdef int issued = corp_issued_shares(state, corp_id)
    cdef int price_idx = corp_price_index(state, corp_id)
    cdef int company_stars = corp_company_stars(state, corp_id)
    cdef int cash_after = cash - amount_per_share * issued
    cdef int cash_stars = cash_after // 10 if cash_after > 0 else 0
    cdef int si_bonus = 2 if corp_id == <int>CorpIndices.CORP_SI else 0
    cdef int total_stars = company_stars + cash_stars + si_bonus
    cdef int required = _required_stars(price_idx, issued)
    return calculate_price_move(total_stars, required)


cdef (int, int, int, int, int) _simulate_float(
    int face_value, int par_index,
) noexcept nogil:
    """IPO outcome for (face_value, par_index) — pure function of static data.

    Returns ``(float_shares, market_index, player_payment, corp_cash,
    issued_shares)``.

    The rule: if ``face_value > par_price`` the player only needs a
    single share to pay off the company (``float_shares == 1``);
    otherwise they need two to cover the company's face value
    (``float_shares == 2``). Bank matches the player's float_shares,
    issued = ``float_shares * 2``. Player pays ``float_shares*par -
    face`` (the shortfall vs. face), the bank pays ``float_shares*par``;
    corp starts with the sum.
    """
    cdef int par_price = ALL_PAR_PRICES[par_index]
    cdef int market_index = PRICE_TO_MARKET_INDEX[par_price]
    cdef int float_shares = 2 if face_value > par_price else 1
    cdef int player_payment = float_shares * par_price - face_value
    cdef int bank_payment = float_shares * par_price
    cdef int corp_cash_after = player_payment + bank_payment
    cdef int issued = float_shares * 2
    return (float_shares, market_index, player_payment, corp_cash_after, issued)


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


cdef void _clear_corp_cache(GameState state, int corp_id) noexcept nogil:
    state._data[_corp_slot(corp_id, CORP_FIELDS.income)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.company_stars)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.raw_revenue)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.synergy_income)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.coo_cost)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.ability_income)] = 0
    state._data[_corp_slot(corp_id, CORP_FIELDS.pending_price_move)] = 0
    _clear_cache_dirty(state, corp_id)


cdef void _refresh_corp_cache(GameState state, int corp_id) noexcept nogil:
    cdef int company_id, base_income, fv, i
    cdef int adjusted_income_sum = 0
    cdef int raw_revenue_sum = 0
    cdef int highest_fv = 0
    cdef int highest_fv_income = 0
    cdef int company_stars = 0
    cdef int company_ids[36]
    cdef int synergy_income = 0
    cdef int synergy_markers = 0
    cdef int total_coo, ability, total
    cdef int company_count
    cdef int pending

    if not corp_is_active(state, corp_id):
        _clear_corp_cache(state, corp_id)
        return

    company_count = company_fill_corp_company_ids(
        state, corp_id, True, company_ids)

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

    total_coo = raw_revenue_sum - adjusted_income_sum

    if company_count > 1:
        (synergy_income, synergy_markers) = _aggregate_synergies(company_ids, company_count)

    ability = 0
    if corp_id == <int>CorpIndices.CORP_VM:
        ability = total_coo if total_coo < 10 else 10
    elif corp_id == <int>CorpIndices.CORP_PR:
        ability = company_count
    elif corp_id == <int>CorpIndices.CORP_DA:
        ability = highest_fv_income
    elif corp_id == <int>CorpIndices.CORP_S:
        ability = synergy_markers // 2

    total = raw_revenue_sum - total_coo + synergy_income + ability

    # Persist the income breakdown + company_stars first, then clear the
    # cache-dirty bit, then call ``_simulate_dividend_price_move``. Order
    # matters: the simulate helper reads company_stars via
    # ``corp_company_stars`` which guards on the dirty bit — calling it
    # while the bit is still set would re-enter this function.
    state._data[_corp_slot(corp_id, CORP_FIELDS.income)] = <int16_t>total
    state._data[_corp_slot(corp_id, CORP_FIELDS.company_stars)] = <int16_t>company_stars
    state._data[_corp_slot(corp_id, CORP_FIELDS.raw_revenue)] = <int16_t>raw_revenue_sum
    state._data[_corp_slot(corp_id, CORP_FIELDS.synergy_income)] = <int16_t>synergy_income
    state._data[_corp_slot(corp_id, CORP_FIELDS.coo_cost)] = <int16_t>(-total_coo)
    state._data[_corp_slot(corp_id, CORP_FIELDS.ability_income)] = <int16_t>ability
    _clear_cache_dirty(state, corp_id)

    # Cached pending-price-move = amount=0 dividend simulation.
    pending = _simulate_dividend_price_move(state, corp_id, 0)
    state._data[_corp_slot(corp_id, CORP_FIELDS.pending_price_move)] = <int16_t>pending


cdef int corp_income(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.income)]


cdef int corp_raw_revenue(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.raw_revenue)]


cdef int corp_synergy_income(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.synergy_income)]


cdef int corp_coo_cost(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.coo_cost)]


cdef int corp_ability_income(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.ability_income)]


cdef int corp_company_stars(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.company_stars)]


cdef int corp_cash_stars(GameState state, int corp_id) noexcept nogil:
    cdef int cash
    if not corp_is_active(state, corp_id):
        return 0
    cash = corp_cash(state, corp_id)
    return cash // 10 if cash > 0 else 0


cdef int corp_total_stars(GameState state, int corp_id) noexcept nogil:
    if not corp_is_active(state, corp_id):
        return 0
    return (
        corp_company_stars(state, corp_id)
        + corp_cash_stars(state, corp_id)
        + (2 if corp_id == <int>CorpIndices.CORP_SI else 0)
    )


cdef int corp_pending_price_move(GameState state, int corp_id) noexcept nogil:
    if _cache_dirty(state, corp_id):
        _refresh_corp_cache(state, corp_id)
    return <int>state._data[_corp_slot(corp_id, CORP_FIELDS.pending_price_move)]


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
    # ACTIVE STATUS
    # =========================================================================

    cpdef bint is_active(self, GameState state):
        """Return True if the corp has been IPO'd."""
        return corp_is_active(state, self.corp_id)

    cpdef void set_active(self, GameState state, bint active):
        """Set whether the corp is active."""
        cdef bint old_active = corp_is_active(state, self.corp_id)
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.active)] = <int16_t>(1 if active else 0)
        if old_active != active:
            if active:
                invalidate_corp_cache(state, self.corp_id)
                invalidate_all_player_caches(state)
            else:
                _clear_corp_cache(state, self.corp_id)
                invalidate_all_player_caches(state)

    cpdef void float_corp(self, GameState state, int player_id, int company_id,
                          int market_index, int float_shares=1):
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
        market_module.MARKET.set_space_available(state, market_index, False)
        self.set_price_index(state, market_index)

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
        return corp_cash(state, self.corp_id)

    cpdef void set_cash(self, GameState state, int cash):
        """Set corp cash (raw integer dollars)."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.cash)] = <int16_t>cash
        invalidate_corp_cache(state, self.corp_id)

    cpdef void add_cash(self, GameState state, int amount):
        """Add to corp cash (negative `amount` subtracts)."""
        self.set_cash(state, corp_cash(state, self.corp_id) + amount)

    # =========================================================================
    # SHARE TRACKING
    # =========================================================================

    cpdef int get_total_shares(self):
        """Return total share count for this corp (static per corp_id)."""
        return CORP_SHARE_COUNT[self.corp_id]

    cpdef int get_unissued_shares(self, GameState state):
        """Return number of unissued shares remaining in the treasury."""
        return corp_unissued_shares(state, self.corp_id)

    cpdef void set_unissued_shares(self, GameState state, int shares):
        """Set number of unissued shares."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.unissued_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    cpdef int get_issued_shares(self, GameState state):
        """Return number of issued shares (held by players + bank)."""
        return corp_issued_shares(state, self.corp_id)

    cpdef void set_issued_shares(self, GameState state, int shares):
        """Set number of issued shares."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.issued_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    cpdef int get_bank_shares(self, GameState state):
        """Return number of shares held by the bank (sold by players)."""
        return corp_bank_shares(state, self.corp_id)

    cpdef void set_bank_shares(self, GameState state, int shares):
        """Set number of bank shares."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.bank_shares)] = <int16_t>shares
        invalidate_corp_cache(state, self.corp_id)

    # =========================================================================
    # INCOME
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Return cached corp income (raw integer dollars)."""
        return corp_income(state, self.corp_id)

    cpdef void set_income(self, GameState state, int income):
        """Set cached corp income (raw integer dollars)."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.income)] = <int16_t>income

    cpdef int get_raw_revenue(self, GameState state):
        """Return cached raw revenue before CoO and bonuses."""
        return corp_raw_revenue(state, self.corp_id)

    cpdef int get_synergy_income(self, GameState state):
        """Return cached synergy-income component."""
        return corp_synergy_income(state, self.corp_id)

    cpdef int get_coo_cost(self, GameState state):
        """Return cached CoO component (stored as a non-positive value)."""
        return corp_coo_cost(state, self.corp_id)

    cpdef int get_ability_income(self, GameState state):
        """Return cached ability-income component."""
        return corp_ability_income(state, self.corp_id)

    # =========================================================================
    # STARS
    # =========================================================================

    cpdef int get_total_stars(self, GameState state):
        """Return total stars as company_stars + cash stars + SI bonus."""
        return corp_total_stars(state, self.corp_id)

    cpdef int get_cash_stars(self, GameState state):
        """Return derived cash-stars component from current cash."""
        return corp_cash_stars(state, self.corp_id)

    cpdef int get_company_stars(self, GameState state):
        """Return cached company-stars component."""
        return corp_company_stars(state, self.corp_id)

    cpdef int get_pending_price_move(self, GameState state):
        """Return derived pending price-movement scalar.

        Equivalent to ``simulate_dividend_price_move(state, 0)`` but
        reads the cached slot populated by ``_refresh_corp_cache``.
        """
        return corp_pending_price_move(state, self.corp_id)

    cpdef int simulate_dividend_price_move(self, GameState state, int amount_per_share):
        """Predicted market-index delta if this corp paid ``amount_per_share``.

        Does NOT mutate state. Used by the dividend-token extractor to
        preview the price move for each possible dividend amount.
        """
        assert corp_is_active(state, self.corp_id), \
            f"simulate_dividend_price_move on inactive corp {self.corp_id}"
        assert amount_per_share >= 0, \
            f"amount_per_share {amount_per_share} must be non-negative"
        return _simulate_dividend_price_move(state, self.corp_id, amount_per_share)

    cpdef tuple simulate_float(self, int company_id, int par_index):
        """IPO outcome for floating this corp with ``company_id`` at ``par_index``.

        Returns ``(float_shares, market_index, player_payment, corp_cash,
        issued_shares)``. Pure function of static data — no GameState read.
        Callers still need to validate that ``par_index`` is valid for the
        company's star tier via ``PAR_PRICE_VALID``.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range"
        assert 0 <= par_index < 14, f"par_index {par_index} out of range [0, 14)"
        cdef int face_value = COMPANY_FACE_VALUE[company_id]
        cdef int float_shares, market_index, player_payment, corp_cash_after, issued
        (float_shares, market_index, player_payment, corp_cash_after, issued) = (
            _simulate_float(face_value, par_index)
        )
        return (float_shares, market_index, player_payment, corp_cash_after, issued)

    # =========================================================================
    # SHARE PRICE / MARKET INDEX
    # =========================================================================

    cpdef int get_share_price(self, GameState state):
        """Return current share price derived from the market index."""
        return corp_share_price(state, self.corp_id)

    cpdef int get_price_index(self, GameState state):
        """Return market price index (0-26, where 0 is bankruptcy)."""
        return corp_price_index(state, self.corp_id)

    cpdef void set_price_index(self, GameState state, int index):
        """Set the canonical market price index."""
        cdef int old_index
        assert 0 <= index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"price index {index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        old_index = corp_price_index(state, self.corp_id)
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.price_index)] = <int16_t>index
        if old_index != index:
            invalidate_corp_cache(state, self.corp_id)
            invalidate_all_player_caches(state)

    cpdef void move_to_price_index(self, GameState state, int new_index):
        """Atomically move the corp to a new (interior) market price index.

        Frees the previously-occupied interior space, claims the new space,
        and updates the price index. Boundary spaces 0 ($0 bankruptcy) and
        26 ($75 cap) are always available and are skipped on both sides.

        Does NOT trigger bankruptcy when ``new_index == 0`` — that's a
        policy decision left to callers (e.g. ``_issue_one_share``).
        """
        cdef int old_index
        cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1
        assert 0 <= new_index < <int>GameConstants.NUM_MARKET_SPACES, \
            f"price index {new_index} out of range [0, {<int>GameConstants.NUM_MARKET_SPACES})"
        old_index = corp_price_index(state, self.corp_id)
        if 0 < old_index < max_index:
            market_module.MARKET.set_space_available(state, old_index, True)
        if 0 < new_index < max_index:
            market_module.MARKET.set_space_available(state, new_index, False)
        self.set_price_index(state, new_index)

    # =========================================================================
    # ACQUISITION PROCEEDS
    # =========================================================================

    cpdef int get_acquisition_proceeds(self, GameState state):
        """Return accumulated acquisition proceeds (for dividend calculation)."""
        return corp_acquisition_proceeds(state, self.corp_id)

    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds):
        """Set accumulated acquisition proceeds."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.acquisition_proceeds)] = <int16_t>proceeds

    # =========================================================================
    # RECEIVERSHIP
    # =========================================================================

    cpdef bint is_in_receivership(self, GameState state):
        """Return True if the corp is in receivership (no president)."""
        return corp_is_in_receivership(state, self.corp_id)

    cpdef void set_in_receivership(self, GameState state, bint in_recv):
        """Set whether the corp is in receivership."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.in_receivership)] = <int16_t>(1 if in_recv else 0)

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Return True if the corp owns the given company.

        Backed by the shared company_locations / company_owner_ids arrays.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return corp_owns_company(state, self.corp_id, company_id)

    cpdef int count_companies(self, GameState state, bint include_acquisition=False):
        """
        Count companies owned by this corp.

        Used for last-company rule validation. During the ACQUISITION
        phase, ``include_acquisition=True`` also counts companies sitting
        in this corp's acquisition pile (which will become owned once
        the phase ends).
        """
        return count_corp_companies(state, self.corp_id, include_acquisition)

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
            if (corp_owns_company(state, self.corp_id, company_id)
                    or corp_has_acquisition_company(state, self.corp_id, company_id)):
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
        current_index = corp_price_index(state, self.corp_id)
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
        """Return player_id of the corp's president, or -1 if inactive/receivership."""
        return corp_president_id(state, self.corp_id)

    cdef void _recalculate_presidency(self, GameState state):
        """
        Recalculate presidency based on current share ownership.

        Called automatically whenever share counts change via Player.set_shares().
        Implements the presidency rules from RULES.md:

        1. If no player owns shares (max == 0): corporation is in receivership
        2. If current president is tied for max shares: they remain president (incumbency)
        3. Otherwise: first player in turn order AFTER current president with max shares
           becomes the new president
        """
        cdef int player_id, shares, max_shares, president_id, current_president
        cdef int incumbent_shares, incumbent_position, position, checked, candidate
        cdef TurnState turn = _TURN()
        cdef int num_players = turn._get_num_players(state)
        cdef int slot = _corp_slot(self.corp_id, CORP_FIELDS.president_id)

        # Skip inactive corporations
        if not corp_is_active(state, self.corp_id):
            return

        # Read current president from the corp's president_id field
        current_president = <int>state._data[slot]

        # Find maximum share count across all players
        max_shares = 0
        for player_id in range(num_players):
            shares = (<Player>player_module.PLAYERS[player_id]).get_shares(state, self.corp_id)
            if shares > max_shares:
                max_shares = shares

        # Determine new president
        president_id = -1

        if max_shares == 0:
            # No player owns shares - corporation enters receivership
            self.set_in_receivership(state, True)
            state._data[slot] = -1
            return

        # Someone owns shares - not in receivership
        self.set_in_receivership(state, False)

        if current_president >= 0:
            incumbent_shares = (<Player>player_module.PLAYERS[current_president]).get_shares(state, self.corp_id)

            if incumbent_shares >= max_shares:
                # Current president tied for max or has max - they keep it
                president_id = current_president
            else:
                # Someone has more shares than incumbent
                # Find first player in turn order (starting AFTER incumbent) with max shares
                incumbent_position = (<Player>player_module.PLAYERS[current_president]).get_turn_order(state)

                checked = 0
                position = incumbent_position
                while checked < num_players:
                    position = (position + 1) % num_players
                    candidate = turn.find_player_at_position(state, position)
                    if (<Player>player_module.PLAYERS[candidate]).get_shares(state, self.corp_id) == max_shares:
                        president_id = candidate
                        break
                    checked += 1
        else:
            # No current president - first player by turn order with max shares
            for position in range(num_players):
                candidate = turn.find_player_at_position(state, position)
                if (<Player>player_module.PLAYERS[candidate]).get_shares(state, self.corp_id) == max_shares:
                    president_id = candidate
                    break

        # Update president_id if changed
        if president_id >= 0 and president_id != current_president:
            state._data[slot] = <int16_t>president_id

    # =========================================================================
    # ACQ_OFFER PASSED FLAG
    # =========================================================================

    cpdef bint has_passed_acq_offer(self, GameState state):
        """Return True if this corp has passed on the current ACQ_OFFER."""
        return corp_has_passed_acq_offer(state, self.corp_id)

    cpdef void set_passed_acq_offer(self, GameState state, bint passed):
        """Set whether this corp has passed on the current ACQ_OFFER."""
        state._data[_corp_slot(self.corp_id, CORP_FIELDS.passed_acq_offer)] = <int16_t>(1 if passed else 0)

    # =========================================================================
    # ACQUISITION PILE
    # =========================================================================

    cpdef bint has_acquisition_company(self, GameState state, int company_id):
        """Return True if this corp has the given company in its acquisition pile."""
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return corp_has_acquisition_company(state, self.corp_id, company_id)


# =============================================================================
# GLOBAL CORPORATION INSTANCES
# =============================================================================

# List indexed by corp_id (consistent with PLAYERS, COMPANIES)
CORPS = [Corporation(i, CORP_NAMES[i]) for i in range(<int>GameConstants.NUM_CORPS)]
# Dict by name for convenience
CORPS_BY_NAME = {c.name: c for c in CORPS}
