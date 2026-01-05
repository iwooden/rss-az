# cython: language_level=3
"""
Player state helper declarations.

Provides PlayerOffsets struct and accessor functions for player state
stored in the float tensor representation.
"""

from cython_core.state cimport GameState
from cython_core.data cimport NUM_COMPANIES, NUM_CORPS

# =============================================================================
# PLAYER OFFSETS STRUCT
# =============================================================================

cdef struct PlayerOffsets:
    int cash
    int net_worth
    int turn_order
    int is_auction_high_bidder
    int owned_companies
    int owned_shares
    int is_president
    int share_buys      # Only used in invest phase
    int share_sells     # Only used in invest phase


# =============================================================================
# OFFSET COMPUTATION
# =============================================================================

cdef PlayerOffsets get_player_offsets(int num_players) noexcept nogil


# =============================================================================
# CASH OPERATIONS
# =============================================================================

cdef int get_player_cash(float* player, PlayerOffsets* p) noexcept nogil
cdef void set_player_cash(float* player, PlayerOffsets* p, int cash) noexcept nogil
cdef void add_player_cash(float* player, PlayerOffsets* p, int amount) noexcept nogil


# =============================================================================
# SHARE OPERATIONS
# =============================================================================

cdef int get_player_shares(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void set_player_shares(float* player, PlayerOffsets* p, int corp_id, int shares) noexcept nogil


# =============================================================================
# COMPANY OWNERSHIP
# =============================================================================

cdef bint player_owns_company(float* player, PlayerOffsets* p, int company_id) noexcept nogil
cdef void set_player_owns_company(float* player, PlayerOffsets* p, int company_id, bint owns) noexcept nogil


# =============================================================================
# PRESIDENT STATUS
# =============================================================================

cdef bint is_player_president(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void set_player_president(float* player, PlayerOffsets* p, int corp_id, bint is_pres) noexcept nogil


# =============================================================================
# ROUND-TRIP TRACKING (INVEST PHASE)
# =============================================================================

cdef int get_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void increment_share_buys(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef int get_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void increment_share_sells(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef int get_roundtrips(float* player, PlayerOffsets* p, int corp_id) noexcept nogil
cdef void clear_roundtrip_tracking(float* player, PlayerOffsets* p) noexcept nogil


# =============================================================================
# NET WORTH
# =============================================================================

# Forward declaration - requires CorpOffsets from corp.pxd
# Actual implementation in player.pyx imports corp helpers
cdef int calculate_player_net_worth(GameState state, int player_id, int num_players) noexcept nogil
cdef void update_all_player_net_worths(GameState state, int num_players) noexcept
