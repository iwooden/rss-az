"""
Player entity implementation.

The Player handle owns every per-player slot inside the compact GameState
plus the cross-cutting helpers that the engine needs to keep player
finances coherent (net worth, liquidity, income, presidency
recalculation). It is fully stateless: each access derives its slot
inline from the module-level ``LAYOUT`` and ``PLAYER_FIELDS`` constants
on ``core.state``, so the singleton handles in ``PLAYERS`` can be reused
with any GameState at any player count.

Layout summary (per-player block, all raw int16):
  cash, net_worth, liquidity, turn_order (single int), owned_shares (8),
  income, share_buys (8), share_sells (8), has_passed (1).
Presidency is tracked by the corp entity.

Company ownership lives in the companies section, but player code reads
it through company-module query helpers rather than duplicating the
companies layout details locally. There is no per-player owned_companies
bitmap.

The handle reaches into the corp entity for share-bank / receivership
bookkeeping during share transfers and presidency recalculation. The
corp entity is itself a stateless constant-offset handle, so these
calls are thin cdef dispatches once typed.
"""

from libc.stdint cimport int16_t

from core.state cimport GameState, LAYOUT, PLAYER_FIELDS, TURN_OFFSETS
from core.data cimport (
    GameConstants,
    MARKET_PRICES,
)
from entities.company cimport (
    company_owned_by_player,
    company_sum_player_face_value,
    company_sum_player_adjusted_income,
)
from entities.turn cimport TurnState
from entities.corp cimport (
    Corporation,
    corp_is_active,
    corp_issued_shares,
    corp_bank_shares,
    corp_price_index,
    corp_share_price,
    corp_president_id,
)
from entities.market cimport copy_market_availability

# Late imports to avoid circular dependencies (resolved at runtime)
from entities import turn as turn_module
from entities import corp as corp_module

# Lazy accessor for the TURN singleton. We deliberately do NOT cache it at
# module init (``cdef TurnState TURN = turn_module.TURN``) because that
# pattern required entities.turn to be fully initialized *before* this
# module finishes its own init, which forced a fragile cross-module load
# order. Instead we look it up on first call and cache the typed reference.
# Every callsite runs under the GIL (cpdef methods or cdef-void methods,
# not inside nogil blocks), so the lookup is always safe. After the first
# call this costs a single None-check + pointer return.
cdef TurnState _TURN_CACHED = None

cdef TurnState _TURN():
    global _TURN_CACHED
    if _TURN_CACHED is None:
        _TURN_CACHED = <TurnState>turn_module.TURN
    return _TURN_CACHED


# =============================================================================
# PLAYER FINANCE DIRTY MASK
# =============================================================================

cdef inline int _player_cache_dirty_slot() noexcept nogil:
    return LAYOUT.turn_offset + TURN_OFFSETS.player_cache_dirty


cdef inline int _all_players_mask(GameState state) noexcept nogil:
    cdef int num_players = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
    return (1 << num_players) - 1


cdef inline bint _cache_dirty(GameState state, int player_id) noexcept nogil:
    return (<int>state._data[_player_cache_dirty_slot()] & (1 << player_id)) != 0


cdef inline void _clear_cache_dirty(GameState state, int player_id) noexcept nogil:
    cdef int slot = _player_cache_dirty_slot()
    state._data[slot] = <int16_t>(<int>state._data[slot] & ~(1 << player_id))


cdef void invalidate_player_cache(GameState state, int player_id) noexcept nogil:
    cdef int slot = _player_cache_dirty_slot()
    state._data[slot] = <int16_t>(<int>state._data[slot] | (1 << player_id))


cdef void invalidate_all_player_caches(GameState state) noexcept nogil:
    state._data[_player_cache_dirty_slot()] = <int16_t>_all_players_mask(state)


# =============================================================================
# NOGIL CACHE REFRESH (module-level, bypasses the Python PLAYERS[i] lookup)
# =============================================================================
#
# These mirror Player.calculate_net_worth / calculate_liquidity / _refresh_cache
# but operate on raw player_id, so they are callable from nogil contexts. The
# class-level helpers delegate here to avoid duplicating the math.

cdef inline int _player_base(int player_id) noexcept nogil:
    return LAYOUT.players_offset + player_id * PLAYER_FIELDS.size


cdef int _calc_net_worth(GameState state, int player_id) noexcept nogil:
    """Compute net worth = cash + face values + share value (active corps)."""
    cdef int base = _player_base(player_id)
    cdef int total = <int>state._data[base + PLAYER_FIELDS.cash]
    cdef int corp_id, shares

    total += company_sum_player_face_value(state, player_id)

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        shares = <int>state._data[base + PLAYER_FIELDS.owned_shares + corp_id]
        if shares <= 0:
            continue
        if corp_is_active(state, corp_id):
            total += shares * corp_share_price(state, corp_id)

    return total


cdef int _calc_liquidity(GameState state, int player_id) noexcept nogil:
    """Compute iterative-sale liquidation value across all owned shares."""
    cdef int base = _player_base(player_id)
    cdef int total = <int>state._data[base + PLAYER_FIELDS.cash]
    cdef int corp_id, shares, sim_index, new_index, _i
    cdef int16_t sim_market[27]

    copy_market_availability(state, sim_market)

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        shares = <int>state._data[base + PLAYER_FIELDS.owned_shares + corp_id]
        if shares <= 0:
            continue

        sim_index = corp_price_index(state, corp_id)
        for _i in range(shares):
            new_index = sim_index - 1
            while new_index > 0 and sim_market[new_index] != 1:
                new_index -= 1
            if new_index <= 0:
                break  # Bankruptcy — remaining shares worthless
            total += MARKET_PRICES[new_index]
            if sim_index < 26:
                sim_market[sim_index] = 1
            sim_market[new_index] = 0
            sim_index = new_index

    return total


cdef void refresh_player_cache_if_dirty(
    GameState state, int player_id,
) noexcept nogil:
    """Refresh income / net_worth / liquidity slots if the dirty bit is set.

    Module-level nogil equivalent of ``Player._refresh_cache``, callable
    without holding the GIL. ``core/token_data`` uses this to fold the
    cache-refresh prologue into the same nogil block as ``_fill_buffer``.
    """
    if not _cache_dirty(state, player_id):
        return
    cdef int base = _player_base(player_id)
    state._data[base + PLAYER_FIELDS.income] = (
        <int16_t>company_sum_player_adjusted_income(state, player_id)
    )
    state._data[base + PLAYER_FIELDS.net_worth] = (
        <int16_t>_calc_net_worth(state, player_id)
    )
    state._data[base + PLAYER_FIELDS.liquidity] = (
        <int16_t>_calc_liquidity(state, player_id)
    )
    _clear_cache_dirty(state, player_id)


# =============================================================================
# HIGH-LEVEL PLAYER CLASS
# =============================================================================

cdef class Player:
    """
    Entity handle for accessing player state.

    Players are instantiated once at module load with their player_id and
    have no other per-instance state. Every accessor reads its slot
    inline from the module-level ``LAYOUT`` / ``PLAYER_FIELDS`` constants
    on ``core.state``, so a single PLAYERS list works for any GameState
    at any player count.
    """

    def __cinit__(self, int player_id):
        self.player_id = player_id

    # =========================================================================
    # SLOT HELPER (constant-offset arithmetic)
    # =========================================================================

    cdef inline int _slot(self, int field) noexcept nogil:
        """Absolute index of a per-player field for this player."""
        return LAYOUT.players_offset + self.player_id * PLAYER_FIELDS.size + field

    # =========================================================================
    # NOGIL ACCESSORS (used by hot paths inside the engine)
    # =========================================================================

    cdef inline int _get_cash(self, GameState state) noexcept nogil:
        return <int>state._data[self._slot(PLAYER_FIELDS.cash)]

    cdef inline int _get_shares(self, GameState state, int corp_id) noexcept nogil:
        return <int>state._data[self._slot(PLAYER_FIELDS.owned_shares) + corp_id]

    cdef inline int _get_share_buys(self, GameState state, int corp_id) noexcept nogil:
        return <int>state._data[self._slot(PLAYER_FIELDS.share_buys) + corp_id]

    cdef inline int _get_share_sells(self, GameState state, int corp_id) noexcept nogil:
        return <int>state._data[self._slot(PLAYER_FIELDS.share_sells) + corp_id]

    cdef inline bint _owns_company(self, GameState state, int company_id) noexcept nogil:
        """Check if this player owns the given company.

        Delegates to the company module, which owns the companies
        section storage and its layout details.
        """
        return company_owned_by_player(state, company_id, self.player_id)

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Get player's cash (raw integer dollars)."""
        return self._get_cash(state)

    cpdef void set_cash(self, GameState state, int cash):
        """Set player's cash (raw integer dollars)."""
        state._data[self._slot(PLAYER_FIELDS.cash)] = <int16_t>cash
        invalidate_player_cache(state, self.player_id)

    cpdef void add_cash(self, GameState state, int amount):
        """Add to player's cash (negative `amount` subtracts)."""
        cdef int slot = self._slot(PLAYER_FIELDS.cash)
        state._data[slot] = <int16_t>(<int>state._data[slot] + amount)
        invalidate_player_cache(state, self.player_id)

    # =========================================================================
    # NET WORTH
    # =========================================================================

    cpdef int get_net_worth(self, GameState state):
        """Get player's cached net worth (raw integer dollars)."""
        if _cache_dirty(state, self.player_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(PLAYER_FIELDS.net_worth)]

    cpdef void set_net_worth(self, GameState state, int net_worth):
        """Set player's cached net worth (raw integer dollars)."""
        state._data[self._slot(PLAYER_FIELDS.net_worth)] = <int16_t>net_worth

    cpdef int calculate_net_worth(self, GameState state):
        """
        Calculate player's total net worth.

        Net worth = cash + sum(company face values) + sum(shares * share_price)
        """
        return _calc_net_worth(state, self.player_id)

    cdef void _refresh_cache(self, GameState state) noexcept nogil:
        """Recompute and store this player's derived finance cache.

        Caller is expected to have verified the cache is dirty (the cpdef
        getters do that). The body matches ``refresh_player_cache_if_dirty``
        without the dirty-check guard so we don't double-test the same bit.
        """
        cdef int base = _player_base(self.player_id)
        state._data[base + PLAYER_FIELDS.income] = (
            <int16_t>company_sum_player_adjusted_income(state, self.player_id)
        )
        state._data[base + PLAYER_FIELDS.net_worth] = (
            <int16_t>_calc_net_worth(state, self.player_id)
        )
        state._data[base + PLAYER_FIELDS.liquidity] = (
            <int16_t>_calc_liquidity(state, self.player_id)
        )
        _clear_cache_dirty(state, self.player_id)

    # =========================================================================
    # LIQUIDITY
    # =========================================================================

    cpdef int get_liquidity(self, GameState state):
        """Get player's cached liquidity (raw integer dollars)."""
        if _cache_dirty(state, self.player_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(PLAYER_FIELDS.liquidity)]

    cpdef void set_liquidity(self, GameState state, int liquidity):
        """Set player's cached liquidity (raw integer dollars)."""
        state._data[self._slot(PLAYER_FIELDS.liquidity)] = <int16_t>liquidity

    cpdef int calculate_liquidity(self, GameState state):
        """
        Calculate player's total liquidation value.

        Liquidity = cash + iterative proceeds from selling all held shares.
        Sells are simulated in corp index order (0-7). Each sell moves the
        corp's price to the next lower available market space. Cross-corp
        effects are captured: selling corp 0's shares frees/occupies market
        spaces that affect corp 1's simulation, etc.
        """
        return _calc_liquidity(state, self.player_id)

    # =========================================================================
    # TURN ORDER
    # =========================================================================

    cpdef int get_turn_order(self, GameState state):
        """Get player's position in turn order (0 = first).

        Turn order is now stored as a single integer slot per player; no
        more one-hot encoding or O(n) permutation scan.
        """
        return <int>state._data[self._slot(PLAYER_FIELDS.turn_order)]

    cpdef void set_turn_order(self, GameState state, int order):
        """Set player's position in turn order."""
        assert 0 <= order < _TURN()._get_num_players(state), \
            f"turn order {order} out of range [0, {_TURN()._get_num_players(state)})"
        state._data[self._slot(PLAYER_FIELDS.turn_order)] = <int16_t>order

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Check if player owns a private company.

        Backed by the shared company_locations / company_owner_ids arrays
        — there is no per-player owned_companies bitmap any more.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return self._owns_company(state, company_id)

    # =========================================================================
    # CORPORATION SHARES
    # =========================================================================

    cpdef int get_shares(self, GameState state, int corp_id):
        """Get player's shares of a corporation."""
        return self._get_shares(state, corp_id)

    cpdef void set_shares(self, GameState state, int corp_id, int shares):
        """
        Set player's shares of a corporation.

        Automatically adjusts bank shares by the inverse delta (shares moving
        to/from bank) and recalculates presidency. This mirrors the in-game
        mechanic where shares always transfer between player and bank.
        """
        cdef Corporation corp
        cdef int slot = self._slot(PLAYER_FIELDS.owned_shares) + corp_id
        cdef int old_shares = <int>state._data[slot]
        cdef int issued_shares
        cdef int bank_shares
        cdef int delta = shares - old_shares
        cdef int new_bank_shares

        assert 0 <= self.player_id < _TURN()._get_num_players(state), \
            f"player_id {self.player_id} out of range [0, {_TURN()._get_num_players(state)})"
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        assert shares >= 0, f"shares must be non-negative, got {shares}"

        corp = <Corporation>corp_module.CORPS[corp_id]
        assert corp_is_active(state, corp_id), \
            f"cannot assign shares of inactive corp {corp_id}"
        issued_shares = corp_issued_shares(state, corp_id)
        assert shares <= issued_shares, \
            f"player {self.player_id} shares {shares} exceed issued shares {issued_shares} for corp {corp_id}"

        bank_shares = corp_bank_shares(state, corp_id)
        new_bank_shares = bank_shares - delta
        assert 0 <= new_bank_shares <= issued_shares, \
            f"share transfer would leave bank_shares={new_bank_shares} outside [0, {issued_shares}] for corp {corp_id}"

        state._data[slot] = <int16_t>shares
        # Adjust bank shares by inverse delta
        if delta != 0:
            corp.set_bank_shares(state, new_bank_shares)
            invalidate_player_cache(state, self.player_id)
        corp._recalculate_presidency(state)

    # =========================================================================
    # PRESIDENT STATUS (read-only — derived from share ownership)
    # =========================================================================

    cpdef bint is_president_of(self, GameState state, int corp_id):
        """Check if player is president of a corporation."""
        return corp_president_id(state, corp_id) == self.player_id

    # =========================================================================
    # ROUND-TRIP TRACKING (INVEST PHASE)
    # =========================================================================

    cpdef int get_share_buys(self, GameState state, int corp_id):
        """Get share buy count for this corp this turn."""
        return self._get_share_buys(state, corp_id)

    cpdef void increment_share_buys(self, GameState state, int corp_id):
        """Increment share buy count for this corp this turn."""
        cdef int slot = self._slot(PLAYER_FIELDS.share_buys) + corp_id
        state._data[slot] = <int16_t>(<int>state._data[slot] + 1)

    cpdef int get_share_sells(self, GameState state, int corp_id):
        """Get share sell count for this corp this turn."""
        return self._get_share_sells(state, corp_id)

    cpdef void increment_share_sells(self, GameState state, int corp_id):
        """Increment share sell count for this corp this turn."""
        cdef int slot = self._slot(PLAYER_FIELDS.share_sells) + corp_id
        state._data[slot] = <int16_t>(<int>state._data[slot] + 1)

    cpdef int get_roundtrips(self, GameState state, int corp_id):
        """
        Get number of completed round-trips for a corp.

        A round-trip is a paired buy+sell (or sell+buy). Limiting round-trips prevents
        models from getting stuck in unprofitable buy/sell loops during training.
        Uses min(buys, sells) so multiple buys or multiple sells alone don't trigger
        the limit - only actual round-trips (paired operations) count.
        """
        cdef int buys = self._get_share_buys(state, corp_id)
        cdef int sells = self._get_share_sells(state, corp_id)
        return buys if buys < sells else sells

    cpdef void clear_roundtrip_tracking(self, GameState state):
        """Clear buy/sell tracking for all corps."""
        cdef int i
        cdef int buys_base = self._slot(PLAYER_FIELDS.share_buys)
        cdef int sells_base = self._slot(PLAYER_FIELDS.share_sells)
        for i in range(<int>GameConstants.NUM_CORPS):
            state._data[buys_base + i] = 0
            state._data[sells_base + i] = 0

    # =========================================================================
    # INCOME
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Get player's cached income (raw integer dollars)."""
        if _cache_dirty(state, self.player_id):
            self._refresh_cache(state)
        return <int>state._data[self._slot(PLAYER_FIELDS.income)]

    cpdef void set_income(self, GameState state, int income):
        """Set player's cached income (raw integer dollars)."""
        state._data[self._slot(PLAYER_FIELDS.income)] = <int16_t>income

    cpdef void calculate_income(self, GameState state):
        """Recalculate and store only the income component of the cache.

        Uses the cached companies-section incomes sub-array (updated when
        CoO changes). Note: Only player-owned privates, NOT corp
        subsidiaries.
        """
        self.set_income(state, company_sum_player_adjusted_income(state, self.player_id))

    # =========================================================================
    # GENERIC PASSED FLAG
    # =========================================================================

    cpdef bint has_passed(self, GameState state):
        """Return True if this player has passed in the current phase.

        Stored as a per-player int16 flag in the player block.
        """
        return state._data[self._slot(PLAYER_FIELDS.has_passed)] == 1

    cpdef void set_has_passed(self, GameState state, bint passed):
        """Mark whether this player has passed in the current phase."""
        state._data[self._slot(PLAYER_FIELDS.has_passed)] = <int16_t>(1 if passed else 0)

# =============================================================================
# GLOBAL PLAYER INSTANCES
# =============================================================================

PLAYERS = [Player(i) for i in range(<int>GameConstants.MAX_PLAYERS)]
