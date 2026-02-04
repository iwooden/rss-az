# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Player entity implementation.

Provides the Player class for accessing player state in the game vector.
The class uses cached offsets for efficient nogil access in performance-critical
code paths (e.g., action mask generation in actions.pyx).
"""

from libc.math cimport lround

from core.state cimport GameState, StateLayout, PlayerFieldOffsets
from core.data cimport (
    GameConstants, CASH_DIVISOR, SHARE_DIVISOR, MAX_ROUNDTRIPS,
    get_company_face_value
)
from entities.encoding cimport set_one_hot, get_one_hot_index


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

        # Cache absolute offsets for each field
        self._cash_offset = self._base_offset + fields.cash
        self._net_worth_offset = self._base_offset + fields.net_worth
        self._turn_order_offset = self._base_offset + fields.turn_order
        self._owned_companies_offset = self._base_offset + fields.owned_companies
        self._owned_shares_offset = self._base_offset + fields.owned_shares
        self._is_president_offset = self._base_offset + fields.is_president
        self._share_buys_offset = self._base_offset + fields.share_buys
        self._share_sells_offset = self._base_offset + fields.share_sells
        self._acquisition_proceeds_offset = self._base_offset + fields.acquisition_proceeds

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
        """Get share buy count (nogil version)."""
        return <int>(data[self._share_buys_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cdef inline int _get_share_sells_nogil(self, float* data, int corp_id) noexcept nogil:
        """Get share sell count (nogil version)."""
        return <int>(data[self._share_sells_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

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
        return <int>(state._data[self._net_worth_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_net_worth(self, GameState state, int net_worth):
        """Set player's net worth (integer dollars)."""
        state._data[self._net_worth_offset] = <float>net_worth / CASH_DIVISOR

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
        """Recalculate and store net worth."""
        self.set_net_worth(state, self.calculate_net_worth(state))

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
        """Set player's shares of a corporation."""
        state._data[self._owned_shares_offset + corp_id] = <float>shares / SHARE_DIVISOR

    # =========================================================================
    # PRESIDENT STATUS
    # =========================================================================

    cpdef bint is_president_of(self, GameState state, int corp_id):
        """Check if player is president of a corporation."""
        return state._data[self._is_president_offset + corp_id] == 1.0

    cpdef void set_president_of(self, GameState state, int corp_id, bint is_pres):
        """Set whether player is president of a corporation."""
        state._data[self._is_president_offset + corp_id] = 1.0 if is_pres else 0.0

    # =========================================================================
    # ROUND-TRIP TRACKING (INVEST PHASE)
    # =========================================================================

    cpdef int get_share_buys(self, GameState state, int corp_id):
        """Get share buy count for this corp this turn."""
        return <int>(state._data[self._share_buys_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cpdef void increment_share_buys(self, GameState state, int corp_id):
        """Increment share buy count."""
        cdef int current = self.get_share_buys(state, corp_id)
        state._data[self._share_buys_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)

    cpdef int get_share_sells(self, GameState state, int corp_id):
        """Get share sell count for this corp this turn."""
        return <int>(state._data[self._share_sells_offset + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)

    cpdef void increment_share_sells(self, GameState state, int corp_id):
        """Increment share sell count."""
        cdef int current = self.get_share_sells(state, corp_id)
        state._data[self._share_sells_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)

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
        """Clear buy/sell tracking for all corps. Called at end of INVEST phase."""
        cdef int i
        for i in range(<int>GameConstants.NUM_CORPS):
            state._data[self._share_buys_offset + i] = 0.0
            state._data[self._share_sells_offset + i] = 0.0

    # =========================================================================
    # ACQUISITION PROCEEDS
    # =========================================================================

    cpdef int get_acquisition_proceeds(self, GameState state):
        """Get player's acquisition proceeds (integer dollars)."""
        return <int>(state._data[self._acquisition_proceeds_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds):
        """Set player's acquisition proceeds (integer dollars)."""
        state._data[self._acquisition_proceeds_offset] = <float>proceeds / CASH_DIVISOR

    cpdef void add_acquisition_proceeds(self, GameState state, int amount):
        """Add to player's acquisition proceeds (can be negative to subtract)."""
        cdef int current = self.get_acquisition_proceeds(state)
        self.set_acquisition_proceeds(state, current + amount)

    cpdef void clear_acquisition_proceeds(self, GameState state):
        """Clear player's acquisition proceeds (set to 0)."""
        state._data[self._acquisition_proceeds_offset] = 0.0

    # =========================================================================
    # INCOME CALCULATION
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """
        Calculate total income from player's private companies.

        Uses the cached company_incomes array (updated when CoO changes).
        Note: Only player-owned privates, NOT corp subsidiaries.
        Used by mandatory close to check if player income + cash < 0.
        """
        cdef int total = 0
        cdef int company_id
        cdef int company_incomes_offset = state._layout.company_incomes_offset
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if state._data[self._owned_companies_offset + company_id] == 1.0:
                total += <int>lround(state._data[company_incomes_offset + company_id] * CASH_DIVISOR)
        return total


# =============================================================================
# GLOBAL PLAYER INSTANCES
# =============================================================================

PLAYERS = [Player(i) for i in range(6)]
