# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Corporation state helper functions.

Provides accessor functions for corporation state stored in the float tensor representation.
All functions operate on raw float pointers for maximum performance in nogil contexts.
"""

from cython_core.data cimport (
    NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES,
    get_market_price, get_corp_share_count, get_company_stars,
    CORP_SI
)
from cython_core.state cimport GameState
from cython_core.helpers.player cimport PlayerOffsets, get_player_offsets

# Import constants
DEF CASH_DIVISOR = 200.0
DEF SHARE_DIVISOR = 7.0


# =============================================================================
# OFFSET COMPUTATION
# =============================================================================

cdef CorpOffsets get_corp_offsets() noexcept nogil:
    """
    Compute field offsets within corporation data block.

    The corp state is stored as a contiguous float array with the following layout:
    - active (1) - binary flag
    - cash (1) - normalized
    - unissued_shares (1) - normalized
    - issued_shares (1) - normalized
    - bank_shares (1) - normalized
    - income (1) - normalized
    - stars (1) - normalized
    - share_price (1) - normalized
    - acquisition_proceeds (1) - normalized
    - in_receivership (1) - binary flag
    - price_index (NUM_MARKET_SPACES) - one-hot encoding
    - owned_companies (NUM_COMPANIES) - binary flags
    - acquisition_companies (NUM_COMPANIES) - binary flags (acquisition phase only)
    """
    cdef CorpOffsets c
    cdef int offset = 0

    c.active = offset
    offset += 1

    c.cash = offset
    offset += 1

    c.unissued_shares = offset
    offset += 1

    c.issued_shares = offset
    offset += 1

    c.bank_shares = offset
    offset += 1

    c.income = offset
    offset += 1

    c.stars = offset
    offset += 1

    c.share_price = offset
    offset += 1

    c.acquisition_proceeds = offset
    offset += 1

    c.in_receivership = offset
    offset += 1

    c.price_index = offset
    offset += NUM_MARKET_SPACES

    c.owned_companies = offset
    offset += NUM_COMPANIES

    c.acquisition_companies = offset
    # offset += NUM_COMPANIES  # Last field

    return c


# =============================================================================
# ACTIVE STATUS
# =============================================================================

cdef inline bint is_corp_active(float* corp, CorpOffsets* c) noexcept nogil:
    """Check if corporation is active (has been IPO'd)."""
    return corp[c.active] == 1.0


cdef inline void set_corp_active(float* corp, CorpOffsets* c, bint active) noexcept nogil:
    """Set corporation active status."""
    corp[c.active] = 1.0 if active else 0.0


# =============================================================================
# CASH OPERATIONS
# =============================================================================

cdef inline int get_corp_cash(float* corp, CorpOffsets* c) noexcept nogil:
    """Get corporation's cash (integer dollars)."""
    return <int>(corp[c.cash] * CASH_DIVISOR + 0.5)


cdef inline void set_corp_cash(float* corp, CorpOffsets* c, int cash) noexcept nogil:
    """Set corporation's cash (integer dollars)."""
    corp[c.cash] = <float>cash / CASH_DIVISOR


cdef inline void add_corp_cash(float* corp, CorpOffsets* c, int amount) noexcept nogil:
    """Add to corporation's cash (can be negative to subtract)."""
    cdef int current = get_corp_cash(corp, c)
    set_corp_cash(corp, c, current + amount)


# =============================================================================
# SHARE TRACKING
# =============================================================================

cdef inline int get_corp_issued_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of issued shares (player + bank shares)."""
    return <int>(corp[c.issued_shares] * SHARE_DIVISOR + 0.5)


cdef inline void set_corp_issued_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil:
    """Set number of issued shares."""
    corp[c.issued_shares] = <float>shares / SHARE_DIVISOR


cdef inline int get_corp_bank_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of bank shares (available for purchase)."""
    return <int>(corp[c.bank_shares] * SHARE_DIVISOR + 0.5)


cdef inline void set_corp_bank_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil:
    """Set number of bank shares."""
    corp[c.bank_shares] = <float>shares / SHARE_DIVISOR


cdef inline int get_corp_unissued_shares(float* corp, CorpOffsets* c) noexcept nogil:
    """Get number of unissued shares (can still be issued)."""
    return <int>(corp[c.unissued_shares] * SHARE_DIVISOR + 0.5)


cdef inline void set_corp_unissued_shares(float* corp, CorpOffsets* c, int shares) noexcept nogil:
    """Set number of unissued shares."""
    corp[c.unissued_shares] = <float>shares / SHARE_DIVISOR


# =============================================================================
# SHARE PRICE
# =============================================================================

cdef inline int get_corp_share_price(float* corp, CorpOffsets* c) noexcept nogil:
    """Get corporation's current share price (integer dollars)."""
    return <int>(corp[c.share_price] * CASH_DIVISOR + 0.5)


cdef inline void set_corp_share_price(float* corp, CorpOffsets* c, int price) noexcept nogil:
    """Set corporation's share price directly."""
    corp[c.share_price] = <float>price / CASH_DIVISOR


cdef int get_corp_price_index(float* corp, CorpOffsets* c) noexcept nogil:
    """
    Get corporation's market price index (0-26).

    Returns 0 if corp has no price card (inactive).
    Active corps should have index 1-26.
    Index 26 = price $75 (can have multiple corps, some off-market).
    """
    cdef int i
    for i in range(NUM_MARKET_SPACES):
        if corp[c.price_index + i] == 1.0:
            return i
    return 0  # No price card (inactive)


cdef void set_corp_price_index(float* corp, CorpOffsets* c, int index, float* hidden_price_indices, int corp_id) noexcept nogil:
    """
    Set corporation's market price index.

    Updates both the one-hot visible encoding and the compact hidden storage.
    Also updates the share_price field to match.

    Index 0 = inactive/no card (price shown as $0)
    Index 1-25 = prices $5-$68
    Index 26 = price $75 (on or off market)
    """
    cdef int i
    # Clear all price index bits (visible one-hot encoding)
    for i in range(NUM_MARKET_SPACES):
        corp[c.price_index + i] = 0.0

    if index > 0 and index < NUM_MARKET_SPACES:
        corp[c.price_index + index] = 1.0
        corp[c.share_price] = <float>get_market_price(index) / CASH_DIVISOR
    elif index == 0:
        # Inactive/no card - price 0
        corp[c.share_price] = 0.0
    else:
        # Invalid index, treat as inactive
        corp[c.share_price] = 0.0

    # Update compact hidden storage
    hidden_price_indices[corp_id] = <float>index


# =============================================================================
# RECEIVERSHIP
# =============================================================================

cdef inline bint is_corp_in_receivership(float* corp, CorpOffsets* c) noexcept nogil:
    """Check if corporation is in receivership (no president)."""
    return corp[c.in_receivership] == 1.0


cdef inline void set_corp_in_receivership(float* corp, CorpOffsets* c, bint in_recv) noexcept nogil:
    """Set corporation receivership status."""
    corp[c.in_receivership] = 1.0 if in_recv else 0.0


# =============================================================================
# COMPANY OWNERSHIP
# =============================================================================

cdef inline bint corp_owns_company(float* corp, CorpOffsets* c, int company_id) noexcept nogil:
    """Check if corporation owns a company (as subsidiary)."""
    return corp[c.owned_companies + company_id] == 1.0


cdef inline void set_corp_owns_company(float* corp, CorpOffsets* c, int company_id, bint owns) noexcept nogil:
    """Set whether corporation owns a company."""
    corp[c.owned_companies + company_id] = 1.0 if owns else 0.0


cdef int get_corp_company_count(float* corp, CorpOffsets* c) noexcept nogil:
    """Count number of companies owned by corporation."""
    cdef int count = 0
    cdef int i
    for i in range(NUM_COMPANIES):
        if corp[c.owned_companies + i] == 1.0:
            count += 1
    return count


# =============================================================================
# PRESIDENT LOOKUP
# =============================================================================

cdef int get_president_of_corp(GameState state, int corp_id, int num_players) noexcept nogil:
    """
    Get player ID who is president of corporation.

    Returns -1 if no president (corp in receivership or inactive).
    """
    cdef int player_id
    for player_id in range(num_players):
        if state.is_player_president(player_id, corp_id):
            return player_id
    return -1


cdef void set_active_player_to_president(GameState state, int corp_id, int num_players) noexcept:
    """Set the active player to the president of the given corporation."""
    cdef int player_id
    for player_id in range(num_players):
        if state.is_player_president(player_id, corp_id):
            state._set_active_player(player_id)
            return


# =============================================================================
# STARS CALCULATION
# =============================================================================

cdef int calculate_corp_company_stars(float* corp, CorpOffsets* c) noexcept nogil:
    """
    Calculate total stars from owned companies only (no cash contribution).
    """
    cdef int total = 0
    cdef int company_id

    for company_id in range(NUM_COMPANIES):
        if corp_owns_company(corp, c, company_id):
            total += get_company_stars(company_id)

    return total


cdef int calculate_corp_total_stars(GameState state, int corp_id) noexcept nogil:
    """
    Calculate corporation's total stars.

    stars = company_stars + cash // 10 + SI_bonus
    """
    cdef CorpOffsets co = get_corp_offsets()
    cdef float* corp = state._corp_ptr(corp_id)
    cdef int total = calculate_corp_company_stars(corp, &co)
    cdef int cash = get_corp_cash(corp, &co)

    # Add cash contribution
    total += cash // 10

    # SI (Stars, Inc.) gets +2 stars
    if corp_id == CORP_SI:
        total += 2

    return total


cdef int calculate_target_stars(float* corp, CorpOffsets* c) noexcept nogil:
    """
    Calculate target stars for share price maintenance.

    target = round(issued_shares * share_price / 10)
    """
    cdef int issued_shares = get_corp_issued_shares(corp, c)
    cdef int share_price = get_corp_share_price(corp, c)
    cdef float raw_val = <float>(issued_shares * share_price) / 10.0

    # Add tiny epsilon for floating point accuracy
    raw_val += 0.000000001

    return <int>(raw_val + 0.5)  # Round to nearest int


# =============================================================================
# BANKRUPTCY HANDLING
# =============================================================================

cdef void handle_corp_bankruptcy(
    GameState state,
    int corp_id,
    int old_price_index,
    int num_players
) noexcept:
    """
    Handle corporation bankruptcy.

    1. Release old market space
    2. Reset unissued shares to full count
    3. Clear player shares for this corp
    4. Clear president flags
    5. Call state.bankrupt_corp()
    """
    cdef int share_count
    cdef int player_id
    cdef PlayerOffsets po = get_player_offsets(num_players)
    cdef float* player

    # Release old market space
    if old_price_index >= 0:
        state.set_market_space_available(old_price_index, True)

    # Reset unissued shares before bankruptcy
    share_count = get_corp_share_count(corp_id)
    state.set_corp_unissued_shares(corp_id, share_count)

    # Clear player shares and president flags for this corp
    for player_id in range(num_players):
        player = state._player_ptr(player_id)
        player[po.owned_shares + corp_id] = 0.0
        player[po.is_president + corp_id] = 0.0

    # Bankrupt the corp (resets companies, etc.)
    state.bankrupt_corp(corp_id)
