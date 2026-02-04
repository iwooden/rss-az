# cython: language_level=3
"""
Player entity declarations.

Includes both the high-level Player class and low-level cdef functions
for nogil performance-critical code paths.
"""

from core.state cimport GameState


# =============================================================================
# LOW-LEVEL STRUCTS AND FUNCTIONS (for nogil performance)
# =============================================================================

cdef struct PlayerOffsets:
    # Offsets within a player's data block in the state vector
    int cash
    int net_worth
    int turn_order
    int is_auction_high_bidder
    int owned_companies
    int owned_shares
    int is_president
    int share_buys
    int share_sells
    int acquisition_proceeds

# Offset computation
cdef PlayerOffsets get_player_offsets(int num_players) noexcept nogil

# Cash operations (raw pointer, nogil)
cdef int get_player_cash(float* player, PlayerOffsets* p) noexcept nogil
cdef void set_player_cash(float* player, PlayerOffsets* p, int cash) noexcept nogil
cdef void add_player_cash(float* player, PlayerOffsets* p, int amount) noexcept nogil

# Share operations (raw pointer, nogil)
cdef int get_player_shares(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void set_player_shares(float* player, PlayerOffsets* p, int corp_id, int shares) noexcept nogil

# Company ownership (raw pointer, nogil)
cdef bint player_owns_company(float* player, PlayerOffsets* p, int company_id) noexcept nogil
cdef void set_player_owns_company(float* player, PlayerOffsets* p, int company_id, bint owns) noexcept nogil

# President status (raw pointer, nogil)
cdef bint is_player_president(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void set_player_president(float* player, PlayerOffsets* p, int corp_id, bint is_pres) noexcept nogil

# Buy/sell tracking - prevents model training loops (raw pointer, nogil)
cdef int get_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void increment_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef int get_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void increment_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef int get_roundtrips(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void clear_roundtrip_tracking(float* player, PlayerOffsets* p) noexcept nogil

# Net worth calculation (requires GameState)
cdef int calculate_player_net_worth(GameState state, int player_id, int num_players) noexcept nogil
cdef void update_all_player_net_worths(GameState state, int num_players) noexcept


# =============================================================================
# HIGH-LEVEL PLAYER CLASS
# =============================================================================

cdef class Player:
    """
    Entity handle for accessing player state.
    Provides Python-accessible methods that wrap the low-level cdef functions.
    """
    cdef readonly int player_id
    cdef int _base_offset
    cdef int _num_players
    cdef bint _initialized

    # Cached field offsets
    cdef int _cash_offset
    cdef int _net_worth_offset
    cdef int _turn_order_offset
    cdef int _owned_companies_offset
    cdef int _owned_shares_offset
    cdef int _is_president_offset
    cdef int _share_buys_offset
    cdef int _share_sells_offset
    cdef int _acquisition_proceeds_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Nogil methods (use cached offsets for performance)
    cdef inline int _get_cash_nogil(self, float* data) noexcept nogil
    cdef inline int _get_shares_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline int _get_share_buys_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline int _get_share_sells_nogil(self, float* data, int corp_id) noexcept nogil
    cdef inline bint _owns_company_nogil(self, float* data, int company_id) noexcept nogil

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Net worth
    cpdef int get_net_worth(self, GameState state)
    cpdef void set_net_worth(self, GameState state, int net_worth)
    cpdef int calculate_net_worth(self, GameState state)
    cpdef void update_net_worth(self, GameState state)

    # Turn order
    cpdef int get_turn_order(self, GameState state)
    cpdef void set_turn_order(self, GameState state, int order)

    # Company ownership (use Company.transfer_to_player() to set)
    cpdef bint owns_company(self, GameState state, int company_id)

    # Corporation shares
    cpdef int get_shares(self, GameState state, int corp_id)
    cpdef void set_shares(self, GameState state, int corp_id, int shares)

    # President status
    cpdef bint is_president_of(self, GameState state, int corp_id)
    cpdef void set_president_of(self, GameState state, int corp_id, bint is_pres)

    # Round-trip tracking
    cpdef int get_share_buys(self, GameState state, int corp_id)
    cpdef void increment_share_buys(self, GameState state, int corp_id)
    cpdef int get_share_sells(self, GameState state, int corp_id)
    cpdef void increment_share_sells(self, GameState state, int corp_id)
    cpdef int get_roundtrips(self, GameState state, int corp_id)
    cpdef void clear_roundtrip_tracking(self, GameState state)

    # Acquisition proceeds
    cpdef int get_acquisition_proceeds(self, GameState state)
    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds)
    cpdef void add_acquisition_proceeds(self, GameState state, int amount)
    cpdef void clear_acquisition_proceeds(self, GameState state)

    # Income calculation
    cpdef int get_income(self, GameState state)
