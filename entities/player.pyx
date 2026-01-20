# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Player entity implementation.

Provides clean getter/setter access to player state in the game state vector.
Each Player instance is bound to a specific player_id and caches offsets
for fast repeated access.
"""

from libc.math cimport round

from state cimport GameState, StateLayout, PlayerFieldOffsets
from data cimport GameConstants, CASH_DIVISOR, SHARE_DIVISOR, MAX_ROUNDTRIPS


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

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Get player's cash (integer dollars)."""
        return <int>round(state._data[self._cash_offset] * CASH_DIVISOR)

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
        return <int>round(state._data[self._net_worth_offset] * CASH_DIVISOR)

    cpdef void set_net_worth(self, GameState state, int net_worth):
        """Set player's net worth (integer dollars)."""
        state._data[self._net_worth_offset] = <float>net_worth / CASH_DIVISOR

    # TODO: calculate_net_worth() - requires Corp entity for share prices
    # Net worth = cash + sum(company face values) + sum(shares * share_price)

    # =========================================================================
    # TURN ORDER
    # =========================================================================

    cpdef int get_turn_order(self, GameState state):
        """Get player's position in turn order (0 = first)."""
        cdef int i
        for i in range(self._num_players):
            if state._data[self._turn_order_offset + i] == 1.0:
                return i
        return -1  # Not found (shouldn't happen in valid state)

    cpdef void set_turn_order(self, GameState state, int order):
        """Set player's position in turn order (one-hot encoded)."""
        cdef int i
        # Clear all positions
        for i in range(self._num_players):
            state._data[self._turn_order_offset + i] = 0.0
        # Set the specified position
        if order >= 0 and order < self._num_players:
            state._data[self._turn_order_offset + order] = 1.0

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Check if player owns a private company."""
        return state._data[self._owned_companies_offset + company_id] == 1.0

    cpdef void set_owns_company(self, GameState state, int company_id, bint owns):
        """Set whether player owns a private company."""
        state._data[self._owned_companies_offset + company_id] = 1.0 if owns else 0.0

    # =========================================================================
    # CORPORATION SHARES
    # =========================================================================

    cpdef int get_shares(self, GameState state, int corp_id):
        """Get player's shares of a corporation."""
        return <int>round(state._data[self._owned_shares_offset + corp_id] * SHARE_DIVISOR)

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
        """Get share buy count for round-trip tracking."""
        return <int>round(state._data[self._share_buys_offset + corp_id] * MAX_ROUNDTRIPS * 2)

    cpdef void increment_share_buys(self, GameState state, int corp_id):
        """Increment share buy count."""
        cdef int current = self.get_share_buys(state, corp_id)
        state._data[self._share_buys_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)

    cpdef int get_share_sells(self, GameState state, int corp_id):
        """Get share sell count for round-trip tracking."""
        return <int>round(state._data[self._share_sells_offset + corp_id] * MAX_ROUNDTRIPS * 2)

    cpdef void increment_share_sells(self, GameState state, int corp_id):
        """Increment share sell count."""
        cdef int current = self.get_share_sells(state, corp_id)
        state._data[self._share_sells_offset + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)

    cpdef int get_roundtrips(self, GameState state, int corp_id):
        """
        Get number of completed round-trips for a corp.

        A round-trip is a buy+sell or sell+buy pair. Players are limited to
        MAX_ROUNDTRIPS per corp per turn to prevent manipulation.
        """
        cdef int buys = self.get_share_buys(state, corp_id)
        cdef int sells = self.get_share_sells(state, corp_id)
        return (buys + sells) // 2

    cpdef void clear_roundtrip_tracking(self, GameState state):
        """Clear round-trip tracking for all corps (called at start of player's turn)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._share_buys_offset + i] = 0.0
            state._data[self._share_sells_offset + i] = 0.0


# =============================================================================
# GLOBAL PLAYER INSTANCES
# =============================================================================

# Initialize the global PLAYERS list at module load
# Using a Python list since Cython doesn't support C arrays of extension types
PLAYERS = [Player(i) for i in range(6)]
