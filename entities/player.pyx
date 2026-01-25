# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Player entity implementation.

Provides both:
1. Low-level cdef functions for nogil performance-critical code (used by actions.pyx)
2. High-level Player class for Python-accessible API

The class methods are thin wrappers around the cdef functions to avoid duplication.
"""

from libc.math cimport round

from core.state cimport GameState, StateLayout, PlayerFieldOffsets
from core.data cimport (
    GameConstants, CASH_DIVISOR, SHARE_DIVISOR, MAX_ROUNDTRIPS,
    get_company_face_value
)

# Local constants from enum for nogil usage
DEF NUM_COMPANIES = 36
DEF NUM_CORPS = 8


# =============================================================================
# LOW-LEVEL OFFSET COMPUTATION
# =============================================================================

cdef PlayerOffsets get_player_offsets(int num_players) noexcept nogil:
    """
    Compute field offsets within player data block.

    The player state is stored as a contiguous float array with the following layout:
    - cash (1)
    - net_worth (1)
    - turn_order (num_players) - one-hot encoding
    - is_auction_high_bidder (1)
    - owned_companies (NUM_COMPANIES) - binary flags
    - owned_shares (NUM_CORPS) - normalized share counts
    - is_president (NUM_CORPS) - binary flags
    - share_buys (NUM_CORPS) - round-trip tracking
    - share_sells (NUM_CORPS) - round-trip tracking
    """
    cdef PlayerOffsets p
    cdef int offset = 0

    p.cash = offset
    offset += 1

    p.net_worth = offset
    offset += 1

    p.turn_order = offset
    offset += num_players

    p.is_auction_high_bidder = offset
    offset += 1

    p.owned_companies = offset
    offset += NUM_COMPANIES

    p.owned_shares = offset
    offset += NUM_CORPS

    p.is_president = offset
    offset += NUM_CORPS

    p.share_buys = offset
    offset += NUM_CORPS

    p.share_sells = offset
    offset += NUM_CORPS

    p.acquisition_proceeds = offset

    return p


# =============================================================================
# LOW-LEVEL CASH OPERATIONS (nogil)
# =============================================================================

cdef inline int get_player_cash(float* player, PlayerOffsets* p) noexcept nogil:
    """Get player's cash (integer dollars)."""
    return <int>(player[p.cash] * CASH_DIVISOR + 0.5)


cdef inline void set_player_cash(float* player, PlayerOffsets* p, int cash) noexcept nogil:
    """Set player's cash (integer dollars)."""
    player[p.cash] = <float>cash / CASH_DIVISOR


cdef inline void add_player_cash(float* player, PlayerOffsets* p, int amount) noexcept nogil:
    """Add to player's cash (can be negative to subtract)."""
    cdef int current = get_player_cash(player, p)
    set_player_cash(player, p, current + amount)


# =============================================================================
# LOW-LEVEL SHARE OPERATIONS (nogil)
# =============================================================================

cdef inline int get_player_shares(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Get player's shares of a corporation."""
    return <int>(player[p.owned_shares + corp_id] * SHARE_DIVISOR + 0.5)


cdef inline void set_player_shares(float* player, PlayerOffsets* p, int corp_id, int shares) noexcept nogil:
    """Set player's shares of a corporation."""
    player[p.owned_shares + corp_id] = <float>shares / SHARE_DIVISOR


# =============================================================================
# LOW-LEVEL COMPANY OWNERSHIP (nogil)
# =============================================================================

cdef inline bint player_owns_company(float* player, PlayerOffsets* p, int company_id) noexcept nogil:
    """Check if player owns a private company."""
    return player[p.owned_companies + company_id] == 1.0


cdef inline void set_player_owns_company(float* player, PlayerOffsets* p, int company_id, bint owns) noexcept nogil:
    """Set whether player owns a private company."""
    player[p.owned_companies + company_id] = 1.0 if owns else 0.0


# =============================================================================
# LOW-LEVEL PRESIDENT STATUS (nogil)
# =============================================================================

cdef inline bint is_player_president(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Check if player is president of a corporation."""
    return player[p.is_president + corp_id] == 1.0


cdef inline void set_player_president(float* player, PlayerOffsets* p, int corp_id, bint is_pres) noexcept nogil:
    """Set whether player is president of a corporation."""
    player[p.is_president + corp_id] = 1.0 if is_pres else 0.0


# =============================================================================
# LOW-LEVEL ROUND-TRIP TRACKING (nogil)
# =============================================================================

cdef inline int get_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Get share buy count for round-trip tracking."""
    return <int>(player[p.share_buys + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)


cdef inline void increment_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Increment share buy count."""
    cdef int current = get_share_buys(player, p, corp_id)
    player[p.share_buys + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)


cdef inline int get_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Get share sell count for round-trip tracking."""
    return <int>(player[p.share_sells + corp_id] * MAX_ROUNDTRIPS * 2 + 0.5)


cdef inline void increment_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Increment share sell count."""
    cdef int current = get_share_sells(player, p, corp_id)
    player[p.share_sells + corp_id] = <float>(current + 1) / (MAX_ROUNDTRIPS * 2)


cdef inline int get_roundtrips(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """
    Get number of completed round-trips for a corp.

    A round-trip is a buy+sell or sell+buy pair. Players are limited to
    MAX_ROUNDTRIPS per corp per turn to prevent manipulation.
    """
    cdef int buys = get_share_buys(player, p, corp_id)
    cdef int sells = get_share_sells(player, p, corp_id)
    return (buys + sells) // 2


cdef inline void clear_roundtrip_tracking(float* player, PlayerOffsets* p) noexcept nogil:
    """Clear round-trip tracking for all corps (called at start of player's turn)."""
    cdef int i
    for i in range(NUM_CORPS):
        player[p.share_buys + i] = 0.0
        player[p.share_sells + i] = 0.0


# =============================================================================
# LOW-LEVEL NET WORTH CALCULATION
# =============================================================================

cdef int calculate_player_net_worth(GameState state, int player_id, int num_players) noexcept nogil:
    """
    Calculate player's total net worth.

    net_worth = cash + sum(company face values) + sum(shares * share_price)
    """
    cdef PlayerOffsets po = get_player_offsets(num_players)
    cdef float* player = state._player_ptr(player_id)
    cdef int total = get_player_cash(player, &po)
    cdef int company_id, corp_id
    cdef int shares, share_price

    # Add face value of owned private companies
    for company_id in range(NUM_COMPANIES):
        if player_owns_company(player, &po, company_id):
            total += get_company_face_value(company_id)

    # Add value of corporation shares
    for corp_id in range(NUM_CORPS):
        shares = get_player_shares(player, &po, corp_id)
        if shares > 0:
            if state._is_corp_active(corp_id):
                share_price = state._get_corp_share_price(corp_id)
                total += shares * share_price

    return total


cdef void update_all_player_net_worths(GameState state, int num_players) noexcept:
    """Update net worth for all players."""
    cdef int player_id, net_worth
    for player_id in range(num_players):
        net_worth = calculate_player_net_worth(state, player_id, num_players)
        state.set_player_net_worth(player_id, net_worth)


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

    cpdef int calculate_net_worth(self, GameState state):
        """
        Calculate player's total net worth.

        Net worth = cash + sum(company face values) + sum(shares * share_price)
        """
        cdef int total = self.get_cash(state)
        cdef int company_id, corp_id
        cdef int shares

        # Add face value of owned private companies
        for company_id in range(GameConstants.NUM_COMPANIES):
            if self.owns_company(state, company_id):
                total += get_company_face_value(company_id)

        # Add value of corporation shares
        for corp_id in range(GameConstants.NUM_CORPS):
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
        """Get player's position in turn order (0 = first)."""
        cdef int i
        for i in range(self._num_players):
            if state._data[self._turn_order_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_turn_order(self, GameState state, int order):
        """Set player's position in turn order (one-hot encoded)."""
        cdef int i
        for i in range(self._num_players):
            state._data[self._turn_order_offset + i] = 0.0
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

    # =========================================================================
    # ACQUISITION PROCEEDS
    # =========================================================================

    cpdef int get_acquisition_proceeds(self, GameState state):
        """Get player's acquisition proceeds (integer dollars)."""
        return <int>round(state._data[self._acquisition_proceeds_offset] * CASH_DIVISOR)

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


# =============================================================================
# GLOBAL PLAYER INSTANCES
# =============================================================================

PLAYERS = [Player(i) for i in range(6)]
