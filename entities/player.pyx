# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Player entity implementation.

Provides the Player class for accessing player state in the game vector.
The class uses cached offsets for efficient nogil access in performance-critical
code paths (e.g., action mask generation in actions.pyx).
"""

from libc.math cimport lround
from libc.stdint cimport int16_t

from core.state cimport GameState, StateLayout, PlayerFieldOffsets
from core.data cimport (
    GameConstants, CASH_DIVISOR, NET_WORTH_DIVISOR, COMPANY_INCOME_DIVISOR, ENTITY_INCOME_DIVISOR,
    SHARE_DIVISOR, MAX_ROUNDTRIPS, get_company_face_value, MARKET_PRICES,
)
from entities.encoding cimport set_one_hot, get_one_hot_index

# Late imports to avoid circular dependencies (resolved at runtime)
from entities import turn as turn_module
from entities import corp as corp_module


# =============================================================================
# PRESIDENCY RECALCULATION
# =============================================================================

cdef void _recalculate_presidency(GameState state, int corp_id):
    """
    Recalculate presidency for a corporation based on current share ownership.

    This is called automatically whenever share counts change via set_shares().
    Implements the presidency rules from RULES.md:

    1. If no player owns shares (max == 0): corporation is in receivership
    2. If current president is tied for max shares: they remain president (incumbency)
    3. Otherwise: first player in turn order AFTER current president with max shares
       becomes the new president

    Args:
        state: Game state
        corp_id: Corporation to recalculate presidency for
    """
    cdef int player_id, shares, max_shares, president_id, current_president
    cdef int incumbent_shares, incumbent_position, position, checked, candidate
    cdef object corp, player, turn

    corp = corp_module.CORPS[corp_id]
    turn = turn_module.TURN

    # Skip inactive corporations
    if not corp.is_active(state):
        return

    # Find current president (if any)
    current_president = -1
    for player_id in range(state._num_players):
        if (<Player>PLAYERS[player_id]).is_president_of(state, corp_id):
            current_president = player_id
            break

    # Find maximum share count across all players
    max_shares = 0
    for player_id in range(state._num_players):
        shares = (<Player>PLAYERS[player_id]).get_shares(state, corp_id)
        if shares > max_shares:
            max_shares = shares

    # Determine new president
    president_id = -1

    if max_shares == 0:
        # No player owns shares - corporation enters receivership
        corp.set_in_receivership(state, True)
        # Clear all president flags
        for player_id in range(state._num_players):
            state._data[(<Player>PLAYERS[player_id])._is_president_offset + corp_id] = 0.0
        return

    # Someone owns shares - not in receivership
    corp.set_in_receivership(state, False)

    if current_president >= 0:
        incumbent_shares = (<Player>PLAYERS[current_president]).get_shares(state, corp_id)

        if incumbent_shares >= max_shares:
            # Current president tied for max or has max - they keep it
            president_id = current_president
        else:
            # Someone has more shares than incumbent
            # Find first player in turn order (starting AFTER incumbent) with max shares
            incumbent_position = (<Player>PLAYERS[current_president]).get_turn_order(state)

            checked = 0
            position = incumbent_position
            while checked < state._num_players:
                position = (position + 1) % state._num_players
                candidate = turn.find_player_at_position(state, position)
                if (<Player>PLAYERS[candidate]).get_shares(state, corp_id) == max_shares:
                    president_id = candidate
                    break
                checked += 1
    else:
        # No current president - first player by turn order with max shares
        for position in range(state._num_players):
            candidate = turn.find_player_at_position(state, position)
            if (<Player>PLAYERS[candidate]).get_shares(state, corp_id) == max_shares:
                president_id = candidate
                break

    # Update president flags if changed
    if president_id >= 0 and president_id != current_president:
        if current_president >= 0:
            state._data[(<Player>PLAYERS[current_president])._is_president_offset + corp_id] = 0.0
        state._data[(<Player>PLAYERS[president_id])._is_president_offset + corp_id] = 1.0


# =============================================================================
# NET WORTH UPDATE (uses Player class methods)
# =============================================================================

cdef void update_all_player_net_worths(GameState state, int num_players) noexcept:
    """Update net worth for all players using Player class methods."""
    cdef int player_id
    cdef Player player
    for player_id in range(num_players):
        player = <Player>PLAYERS[player_id]
        player.update_net_worth(state)


def update_all_net_worths(GameState state):
    """Update net worth for all players. Python-visible wrapper."""
    update_all_player_net_worths(state, state._num_players)


# =============================================================================
# HIGH-LEVEL PLAYER CLASS
# =============================================================================

cdef class Player:
    """
    Entity handle for accessing player state.

    Players are instantiated once at module load with their player_id.
    Offsets are computed lazily on first access to a GameState.
    All methods take GameState as first argument for stateless operation.
    """

    def __cinit__(self, int player_id):
        self.player_id = player_id
        self._base_offset = 0
        self._num_players = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this Player instance.
        """
        cdef StateLayout layout = state._layout
        cdef PlayerFieldOffsets fields = state._player_fields

        self._num_players = state._num_players
        self._base_offset = layout.players_offset + (self.player_id * layout.player_stride)

        # Cache absolute offsets for visible fields
        self._cash_offset = self._base_offset + fields.cash
        self._net_worth_offset = self._base_offset + fields.net_worth
        self._liquidity_offset = self._base_offset + fields.liquidity
        self._market_offset = layout.market_offset
        self._turn_order_offset = self._base_offset + fields.turn_order
        self._owned_companies_offset = self._base_offset + fields.owned_companies
        self._owned_shares_offset = self._base_offset + fields.owned_shares
        self._is_president_offset = self._base_offset + fields.is_president
        self._round_trips_offset = self._base_offset + fields.round_trips
        self._income_offset = self._base_offset + fields.income
        self._auction_passed_offset = self._base_offset + fields.auction_passed

        # Cache absolute offsets for hidden share buy/sell tracking
        self._hidden_share_buys_offset = layout.hidden_share_buys_offset + self.player_id * <int>GameConstants.NUM_CORPS
        self._hidden_share_sells_offset = layout.hidden_share_sells_offset + self.player_id * <int>GameConstants.NUM_CORPS

    # =========================================================================
    # NOGIL METHODS (use cached offsets for performance)
    # =========================================================================

    cdef inline int _get_cash_nogil(self, float* data) noexcept nogil:
        """Get player's cash (nogil version using cached offset)."""
        return <int>(data[self._cash_offset] * CASH_DIVISOR + 0.5)

    cdef inline int _get_shares_nogil(self, float* data, int corp_id) noexcept nogil:
        """Get player's shares of a corporation (nogil version)."""
        return <int>(data[self._owned_shares_offset + corp_id] * SHARE_DIVISOR + 0.5)

    cdef inline int _get_share_buys_nogil(self, float* data, int corp_id) noexcept nogil:
        """Get share buy count from hidden state (nogil version)."""
        return <int>(data[self._hidden_share_buys_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cdef inline int _get_share_sells_nogil(self, float* data, int corp_id) noexcept nogil:
        """Get share sell count from hidden state (nogil version)."""
        return <int>(data[self._hidden_share_sells_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cdef inline bint _owns_company_nogil(self, float* data, int company_id) noexcept nogil:
        """Check if player owns a company (nogil version)."""
        return data[self._owned_companies_offset + company_id] == 1.0

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Get player's cash (integer dollars)."""
        return <int>(state._data[self._cash_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_cash(self, GameState state, int cash):
        """Set player's cash (integer dollars)."""
        state._data[self._cash_offset] = <float>cash / CASH_DIVISOR

    cpdef void add_cash(self, GameState state, int amount):
        """Add to player's cash (can be negative to subtract)."""
        cdef int current = self.get_cash(state)
        self.set_cash(state, current + amount)

    # =========================================================================
    # NET WORTH
    # =========================================================================

    cpdef int get_net_worth(self, GameState state):
        """Get player's stored net worth (integer dollars)."""
        return <int>(state._data[self._net_worth_offset] * NET_WORTH_DIVISOR + 0.5)

    cpdef void set_net_worth(self, GameState state, int net_worth):
        """Set player's net worth (integer dollars)."""
        state._data[self._net_worth_offset] = <float>net_worth / NET_WORTH_DIVISOR

    cpdef int calculate_net_worth(self, GameState state):
        """
        Calculate player's total net worth.

        Net worth = cash + sum(company face values) + sum(shares * share_price)
        """
        cdef int total = self.get_cash(state)
        cdef int company_id, corp_id
        cdef int shares

        # Add face value of owned private companies
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if self.owns_company(state, company_id):
                total += get_company_face_value(company_id)

        # Add value of corporation shares
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            shares = self.get_shares(state, corp_id)
            if shares > 0 and state.is_corp_active(corp_id):
                total += shares * state.get_corp_share_price(corp_id)

        return total

    cpdef void update_net_worth(self, GameState state):
        """Recalculate and store net worth and liquidity."""
        self.set_net_worth(state, self.calculate_net_worth(state))
        self.set_liquidity(state, self.calculate_liquidity(state))

    # =========================================================================
    # LIQUIDITY
    # =========================================================================

    cpdef int get_liquidity(self, GameState state):
        """Get player's stored liquidity (integer dollars)."""
        return <int>(state._data[self._liquidity_offset] * NET_WORTH_DIVISOR + 0.5)

    cpdef void set_liquidity(self, GameState state, int liquidity):
        """Set player's liquidity (integer dollars)."""
        state._data[self._liquidity_offset] = <float>liquidity / NET_WORTH_DIVISOR

    cpdef int calculate_liquidity(self, GameState state):
        """
        Calculate player's total liquidation value.

        Liquidity = cash + iterative proceeds from selling all held shares.
        Sells are simulated in corp index order (0-7). Each sell moves the
        corp's price to the next lower available market space. Cross-corp
        effects are captured: selling corp 0's shares frees/occupies market
        spaces that affect corp 1's simulation, etc.
        """
        cdef int total = self.get_cash(state)
        cdef int corp_id, shares, sim_index, new_index, i
        cdef float sim_market[27]

        # Copy market availability for simulation
        for i in range(27):
            sim_market[i] = state._data[self._market_offset + i]

        for corp_id in range(<int>GameConstants.NUM_CORPS):
            if not state._data[state._layout.corps_offset + corp_id * state._layout.corp_stride] == 1.0:
                continue  # Skip inactive corps
            shares = self.get_shares(state, corp_id)
            if shares <= 0:
                continue

            sim_index = <int>state._data[state._layout.hidden_corp_price_indices_offset + corp_id]
            for _ in range(shares):
                # Find next lower available space in simulated market
                new_index = sim_index - 1
                while new_index > 0 and sim_market[new_index] != 1.0:
                    new_index -= 1
                if new_index <= 0:
                    break  # Bankruptcy — remaining shares worthless
                total += MARKET_PRICES[new_index]
                # Update simulated market: free old space, occupy new
                if sim_index < 26:
                    sim_market[sim_index] = 1.0
                sim_market[new_index] = 0.0
                sim_index = new_index

        return total

    # =========================================================================
    # TURN ORDER
    # =========================================================================

    cpdef int get_turn_order(self, GameState state):
        """
        Get player's position in turn order (0 = first).

        Note: Turn order is stored as a permutation vector where each player has
        their own one-hot encoded position. This is O(n) lookup since there's no
        single compact mirror - use find_player_at_position() to go the other way.
        """
        return get_one_hot_index(state._data, self._turn_order_offset, self._num_players)

    cpdef void set_turn_order(self, GameState state, int order):
        """
        Set player's position in turn order.

        Note: Turn order is stored as a permutation vector where each player has
        their own one-hot encoded position. Unlike other one-hot fields, there is
        no hidden compact mirror since the full permutation requires N values.
        """
        set_one_hot(state._data, self._turn_order_offset, self._num_players, order)

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Check if player owns a private company."""
        return state._data[self._owned_companies_offset + company_id] == 1.0

    # =========================================================================
    # CORPORATION SHARES
    # =========================================================================

    cpdef int get_shares(self, GameState state, int corp_id):
        """Get player's shares of a corporation."""
        return <int>(state._data[self._owned_shares_offset + corp_id] * SHARE_DIVISOR + 0.5)

    cpdef void set_shares(self, GameState state, int corp_id, int shares):
        """
        Set player's shares of a corporation.

        Automatically adjusts bank shares by the inverse delta (shares moving
        to/from bank) and recalculates presidency. This mirrors the in-game
        mechanic where shares always transfer between player and bank.
        """
        cdef int old_shares = <int>(state._data[self._owned_shares_offset + corp_id] * SHARE_DIVISOR + 0.5)
        cdef int delta = shares - old_shares
        state._data[self._owned_shares_offset + corp_id] = <float>shares / SHARE_DIVISOR
        # Adjust bank shares by inverse delta
        if delta != 0:
            corp_module.CORPS[corp_id].set_bank_shares(
                state, corp_module.CORPS[corp_id].get_bank_shares(state) - delta)
        _recalculate_presidency(state, corp_id)

    # =========================================================================
    # PRESIDENT STATUS (read-only - presidency is derived from share ownership)
    # =========================================================================

    cpdef bint is_president_of(self, GameState state, int corp_id):
        """Check if player is president of a corporation."""
        return state._data[self._is_president_offset + corp_id] == 1.0

    # =========================================================================
    # ROUND-TRIP TRACKING (INVEST PHASE)
    # =========================================================================

    cpdef int get_share_buys(self, GameState state, int corp_id):
        """Get share buy count for this corp this turn (from hidden state)."""
        return <int>(state._data[self._hidden_share_buys_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cpdef void increment_share_buys(self, GameState state, int corp_id):
        """Increment share buy count and update visible round_trips."""
        cdef int current = self.get_share_buys(state, corp_id)
        state._data[self._hidden_share_buys_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)
        self._update_visible_roundtrips(state)

    cpdef int get_share_sells(self, GameState state, int corp_id):
        """Get share sell count for this corp this turn (from hidden state)."""
        return <int>(state._data[self._hidden_share_sells_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cpdef void increment_share_sells(self, GameState state, int corp_id):
        """Increment share sell count and update visible round_trips."""
        cdef int current = self.get_share_sells(state, corp_id)
        state._data[self._hidden_share_sells_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)
        self._update_visible_roundtrips(state)

    cdef void _update_visible_roundtrips(self, GameState state):
        """Recompute visible round_trips = max(min(buys, sells)) across all corps / MAX_ROUNDTRIPS."""
        cdef int i, buys, sells, rt, max_rt = 0
        for i in range(<int>GameConstants.NUM_CORPS):
            buys = self.get_share_buys(state, i)
            sells = self.get_share_sells(state, i)
            rt = buys if buys < sells else sells
            if rt > max_rt:
                max_rt = rt
        state._data[self._round_trips_offset] = <float>max_rt / MAX_ROUNDTRIPS

    cpdef int get_roundtrips(self, GameState state, int corp_id):
        """
        Get number of completed round-trips for a corp.

        A round-trip is a paired buy+sell (or sell+buy). Limiting round-trips prevents
        models from getting stuck in unprofitable buy/sell loops during training.
        Uses min(buys, sells) so multiple buys or multiple sells alone don't trigger
        the limit - only actual round-trips (paired operations) count.
        """
        cdef int buys = self.get_share_buys(state, corp_id)
        cdef int sells = self.get_share_sells(state, corp_id)
        return buys if buys < sells else sells

    cpdef void clear_roundtrip_tracking(self, GameState state):
        """Clear buy/sell tracking (hidden) and round_trips (visible) for all corps."""
        cdef int i
        for i in range(<int>GameConstants.NUM_CORPS):
            state._data[self._hidden_share_buys_offset + i] = 0.0
            state._data[self._hidden_share_sells_offset + i] = 0.0
        state._data[self._round_trips_offset] = 0.0

    # =========================================================================
    # INCOME
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Get player's stored income (integer dollars)."""
        return <int>lround(state._data[self._income_offset] * ENTITY_INCOME_DIVISOR)

    cpdef void set_income(self, GameState state, int income):
        """Set player's income (integer dollars)."""
        state._data[self._income_offset] = <float>income / ENTITY_INCOME_DIVISOR

    cpdef void calculate_income(self, GameState state):
        """Recalculate and store total income from player's private companies.

        Uses the cached company_incomes array (updated when CoO changes).
        Note: Only player-owned privates, NOT corp subsidiaries.
        """
        cdef int total = 0
        cdef int company_id
        cdef int company_incomes_offset = state._layout.company_incomes_offset
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if state._data[self._owned_companies_offset + company_id] == 1.0:
                total += <int>lround(state._data[company_incomes_offset + company_id] * COMPANY_INCOME_DIVISOR)
        self.set_income(state, total)

    # =========================================================================
    # AUCTION-PASSED FLAG
    # =========================================================================

    cpdef bint has_passed_auction(self, GameState state):
        """Return True if this player has left the current auction.

        Stored as a per-player int16 flag in the player block (previously
        lived as a num_players-wide array in the turn block).
        """
        return state._data[self._auction_passed_offset] == 1

    cpdef void set_passed_auction(self, GameState state, bint passed):
        """Mark whether this player has left the current auction."""
        state._data[self._auction_passed_offset] = <int16_t>(1 if passed else 0)


# =============================================================================
# GLOBAL PLAYER INSTANCES
# =============================================================================

PLAYERS = [Player(i) for i in range(6)]
