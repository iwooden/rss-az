# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Player state helper functions.

Provides accessor functions for player state stored in the float tensor representation.
All functions operate on raw float pointers for maximum performance in nogil contexts.
"""

from data cimport NUM_COMPANIES, NUM_CORPS, get_company_face_value

# Import constants - DEF statements need to be redeclared for use in this module
DEF CASH_DIVISOR = 200.0
DEF SHARE_DIVISOR = 7.0
DEF MAX_ROUNDTRIPS = 2


# =============================================================================
# OFFSET COMPUTATION
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
    # offset += NUM_CORPS  # Last field, no need to increment

    return p


# =============================================================================
# CASH OPERATIONS
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
# SHARE OPERATIONS
# =============================================================================

cdef inline int get_player_shares(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Get player's shares of a corporation."""
    return <int>(player[p.owned_shares + corp_id] * SHARE_DIVISOR + 0.5)


cdef inline void set_player_shares(float* player, PlayerOffsets* p, int corp_id, int shares) noexcept nogil:
    """Set player's shares of a corporation."""
    player[p.owned_shares + corp_id] = <float>shares / SHARE_DIVISOR


# =============================================================================
# COMPANY OWNERSHIP
# =============================================================================

cdef inline bint player_owns_company(float* player, PlayerOffsets* p, int company_id) noexcept nogil:
    """Check if player owns a private company."""
    return player[p.owned_companies + company_id] == 1.0


cdef inline void set_player_owns_company(float* player, PlayerOffsets* p, int company_id, bint owns) noexcept nogil:
    """Set whether player owns a private company."""
    player[p.owned_companies + company_id] = 1.0 if owns else 0.0


# =============================================================================
# PRESIDENT STATUS
# =============================================================================

cdef inline bint is_player_president(float* player, PlayerOffsets* p, int corp_id) noexcept nogil:
    """Check if player is president of a corporation."""
    return player[p.is_president + corp_id] == 1.0


cdef inline void set_player_president(float* player, PlayerOffsets* p, int corp_id, bint is_pres) noexcept nogil:
    """Set whether player is president of a corporation."""
    player[p.is_president + corp_id] = 1.0 if is_pres else 0.0


# =============================================================================
# ROUND-TRIP TRACKING (INVEST PHASE)
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
# NET WORTH CALCULATION
# =============================================================================
# Note: These functions require corp helpers for share price lookups.
# They import from corp.pyx at runtime to avoid circular import issues.

from state cimport GameState

# Forward declare corp helper imports - actual implementation uses cimport
cdef int calculate_player_net_worth(GameState state, int player_id, int num_players) noexcept nogil:
    """
    Calculate player's total net worth.

    net_worth = cash + sum(company face values) + sum(shares * share_price)

    Note: This function requires access to corp state for share prices.
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
            # Only count shares in active corps
            if state.is_corp_active(corp_id):
                share_price = state.get_corp_share_price(corp_id)
                total += shares * share_price

    return total


cdef void update_all_player_net_worths(GameState state, int num_players) noexcept:
    """Update net worth for all players."""
    cdef int player_id, net_worth
    for player_id in range(num_players):
        net_worth = calculate_player_net_worth(state, player_id, num_players)
        state.set_player_net_worth(player_id, net_worth)
