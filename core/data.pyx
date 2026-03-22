# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Static game data for Rolling Stock Stars.

All company/corp/market data as C arrays for nogil access.
Companies are indexed 0-35, sorted by face_value (matching Python's all_companies_sorted).
Corps are indexed 0-7 in fixed order: JS, S, OS, SM, PR, DA, VM, SI.
"""

cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint64_t, int8_t

# =============================================================================
# NORMALIZATION CONSTANTS
# =============================================================================

cdef float CASH_DIVISOR = 150.0
cdef float NET_WORTH_DIVISOR = 200.0
cdef float COMPANY_INCOME_DIVISOR = 10.0
cdef float ENTITY_INCOME_DIVISOR = 80.0
cdef float SHARE_DIVISOR = 7.0
cdef float COMPANY_PRICE_DIVISOR = 80.0
cdef float SHARE_PRICE_DIVISOR = 75.0
cdef float COMPANY_STAR_DIVISOR = 5.0
cdef float CORP_STAR_DIVISOR = 40.0
cdef float MAX_ROUNDTRIPS = 2.0
cdef float IMPACT_DIVISOR = 5.0

# Python-accessible versions
PY_CASH_DIVISOR = CASH_DIVISOR
PY_NET_WORTH_DIVISOR = NET_WORTH_DIVISOR
PY_COMPANY_PRICE_DIVISOR = COMPANY_PRICE_DIVISOR
PY_SHARE_PRICE_DIVISOR = SHARE_PRICE_DIVISOR
PY_COMPANY_INCOME_DIVISOR = COMPANY_INCOME_DIVISOR
PY_ENTITY_INCOME_DIVISOR = ENTITY_INCOME_DIVISOR
PY_SHARE_DIVISOR = SHARE_DIVISOR
PY_COMPANY_STAR_DIVISOR = COMPANY_STAR_DIVISOR
PY_CORP_STAR_DIVISOR = CORP_STAR_DIVISOR
PY_MAX_ROUNDTRIPS = MAX_ROUNDTRIPS
PY_IMPACT_DIVISOR = IMPACT_DIVISOR

# =============================================================================
# COMPANY DATA
# =============================================================================

# Company IDs as strings (for debugging/display only)
COMPANY_NAMES = [
    "BME", "BSE", "KME", "AKE", "BPM", "MHE",  # Reds (stars=1)
    "WT", "BY", "BD", "HE", "OL", "SX", "MS", "PR",  # Oranges (stars=2)
    "DSB", "KK", "NS", "SBB", "B", "PKP", "SNCF", "DR",  # Yellows (stars=3)
    "SZD", "SJ", "FS", "RENFE", "BR", "BSR", "E",  # Greens (stars=4)
    "HH", "HA", "HR", "MAD", "FRA", "LHR", "CDG",  # Blues (stars=5)
]

# Map company name to index
COMPANY_NAME_TO_ID = {name: i for i, name in enumerate(COMPANY_NAMES)}

# Face values (sorted order, so this is ascending)
cdef int[36] COMPANY_FACE_VALUE = [
    1, 2, 5, 6, 7, 8,           # Reds
    11, 12, 13, 14, 15, 16, 17, 19,  # Oranges
    20, 21, 22, 23, 24, 25, 26, 29,  # Yellows
    30, 31, 32, 33, 34, 36, 43,      # Greens
    45, 46, 47, 50, 56, 58, 60,      # Blues
]

# Low prices (face_value / 2, rounded up for some)
cdef int[36] COMPANY_LOW_PRICE = [
    1, 1, 3, 3, 4, 4,           # Reds
    6, 6, 7, 7, 8, 8, 9, 10,    # Oranges
    10, 11, 11, 12, 12, 13, 13, 15,  # Yellows
    15, 16, 16, 17, 17, 18, 22,      # Greens
    23, 23, 24, 25, 28, 29, 30,      # Blues
]

# High prices
cdef int[36] COMPANY_HIGH_PRICE = [
    2, 3, 7, 8, 9, 10,          # Reds
    14, 16, 17, 18, 20, 21, 22, 25,  # Oranges
    26, 28, 29, 30, 32, 33, 34, 38,  # Yellows
    40, 41, 42, 44, 45, 48, 57,      # Greens
    60, 61, 62, 66, 74, 77, 80,      # Blues
]

# Star ratings (1-5)
cdef int[36] COMPANY_STARS = [
    1, 1, 1, 1, 1, 1,           # Reds
    2, 2, 2, 2, 2, 2, 2, 2,     # Oranges
    3, 3, 3, 3, 3, 3, 3, 3,     # Yellows
    4, 4, 4, 4, 4, 4, 4,        # Greens
    5, 5, 5, 5, 5, 5, 5,        # Blues
]

# Base income
cdef int[36] COMPANY_INCOME = [
    1, 1, 2, 2, 2, 2,           # Reds (1-2)
    3, 3, 3, 3, 3, 3, 3, 3,     # Oranges (3)
    5, 5, 5, 5, 5, 5, 5, 5,     # Yellows (5)
    7, 7, 7, 7, 7, 7, 7,        # Greens (7)
    10, 10, 10, 10, 10, 10, 10, # Blues (10)
]

# Last in group flag (triggers CoO increase when revealed)
cdef uint8_t[36] COMPANY_LAST_IN_GROUP = [
    0, 0, 0, 0, 0, 1,           # MHE is last red
    0, 0, 0, 0, 0, 0, 0, 1,     # PR is last orange
    0, 0, 0, 0, 0, 0, 0, 1,     # DR is last yellow
    0, 0, 0, 0, 0, 0, 1,        # E is last green
    0, 0, 0, 0, 0, 0, 1,        # CDG is last blue
]

# =============================================================================
# SYNERGY DATA
# =============================================================================

# Synergy matrix: SYNERGY[i][j] = bonus that company i gets when paired with company j
# Most entries are 0, so this is sparse in practice
# Using int8 since max synergy bonus is 16
cdef int8_t[36][36] COMPANY_SYNERGY

# Initialize synergy matrix (called at module load)
cdef void _init_synergies() noexcept nogil:
    cdef int i, j
    # Zero out the matrix
    for i in range(36):
        for j in range(36):
            COMPANY_SYNERGY[i][j] = 0

# We'll populate this from Python at module init since the data is complex
# Helper to set synergy from Python
def _set_synergy(int company_id, int target_id, int bonus):
    """Set synergy bonus that company_id gets when paired with target_id."""
    COMPANY_SYNERGY[company_id][target_id] = bonus

# =============================================================================
# CORPORATION DATA
# =============================================================================

# Corp IDs as strings
CORP_NAMES = ["JS", "S", "OS", "SM", "PR", "DA", "VM", "SI"]

# Map corp name to index
CORP_NAME_TO_ID = {name: i for i, name in enumerate(CORP_NAMES)}

# Share counts per corp
cdef int[8] CORP_SHARE_COUNT = [7, 7, 6, 6, 5, 5, 4, 4]

# =============================================================================
# MARKET DATA
# =============================================================================

# All 27 market spaces (index 0 = bankruptcy)
cdef int[27] MARKET_PRICES = [
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
    18, 20, 22, 24, 27, 30, 33, 37, 41, 45,
    50, 55, 61, 68, 75
]

# Reverse lookup: price -> index (for common prices)
# Returns -1 if not a valid market price
cdef int[76] PRICE_TO_MARKET_INDEX

cdef void _init_price_lookup() noexcept nogil:
    cdef int i
    for i in range(76):
        PRICE_TO_MARKET_INDEX[i] = -1
    for i in range(27):
        PRICE_TO_MARKET_INDEX[MARKET_PRICES[i]] = i

# All unique par prices (sorted)
cdef int[14] ALL_PAR_PRICES = [
    10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37
]

# Par price validity by star tier
# PAR_VALID[star-1][par_price_index] = 1 if valid for that tier
cdef bint[5][14] PAR_PRICE_VALID = [
    # Star 1 (reds): 10-14
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Star 2 (oranges): 10-20
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    # Star 3 (yellows): 16-27
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    # Star 4 (greens): 22-37
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    # Star 5 (blues): 30-37
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
]

# =============================================================================
# COST OF OWNERSHIP TABLE
# =============================================================================

# cost_of_ownership[coo_level-1][star_tier-1] = cost
# CoO levels are 1-7, star tiers are 1-5
cdef int[7][5] COST_OF_OWNERSHIP = [
    [0, 0, 0, 0, 0],  # Level 1 - Initial
    [0, 0, 0, 0, 0],  # Level 2 - Orange on top of deck
    [0, 0, 0, 0, 0],  # Level 3 - Yellow on top of deck
    [2, 0, 0, 0, 0],  # Level 4: reds cost 2 - Green on top of deck
    [4, 4, 0, 0, 0],  # Level 5: reds/oranges cost 4 - Blue on top of deck
    [7, 7, 7, 0, 0],  # Level 6: reds/oranges/yellows cost 7 - End card face-up
    [10, 10, 10, 10, 0],  # Level 7: all but blues cost 10 - End card flipped
]

# =============================================================================
# ACCESSOR FUNCTIONS (nogil)
# =============================================================================

cpdef inline int get_company_face_value(int company_id) noexcept nogil:
    return COMPANY_FACE_VALUE[company_id]

cpdef inline int get_company_low_price(int company_id) noexcept nogil:
    return COMPANY_LOW_PRICE[company_id]

cpdef inline int get_company_high_price(int company_id) noexcept nogil:
    return COMPANY_HIGH_PRICE[company_id]

cpdef inline int get_company_stars(int company_id) noexcept nogil:
    return COMPANY_STARS[company_id]

cpdef inline int get_company_income(int company_id) noexcept nogil:
    return COMPANY_INCOME[company_id]

cpdef inline int get_company_synergy(int company_id, int target_id) noexcept nogil:
    return COMPANY_SYNERGY[company_id][target_id]

cpdef inline bint is_last_in_group(int company_id) noexcept nogil:
    return COMPANY_LAST_IN_GROUP[company_id] != 0

cpdef inline int get_corp_share_count(int corp_id) noexcept nogil:
    return CORP_SHARE_COUNT[corp_id]

cpdef inline int get_market_price(int index) noexcept nogil:
    return MARKET_PRICES[index]

cpdef inline int get_market_index(int price) noexcept nogil:
    if price < 0 or price >= 76:
        return -1
    return PRICE_TO_MARKET_INDEX[price]

cpdef inline int get_cost_of_ownership(int coo_level, int star_tier) noexcept nogil:
    """Get cost of ownership for a company with given stars at given CoO level."""
    if coo_level < 1 or coo_level > 7 or star_tier < 1 or star_tier > 5:
        return 0
    return COST_OF_OWNERSHIP[coo_level - 1][star_tier - 1]

cpdef inline int get_adjusted_company_income(int company_id, int coo_level) noexcept nogil:
    """Get company income after cost of ownership."""
    cdef int base_income = COMPANY_INCOME[company_id]
    cdef int stars = COMPANY_STARS[company_id]
    cdef int cost = get_cost_of_ownership(coo_level, stars)
    return base_income - cost


cpdef inline bint is_valid_par_price(int star_tier, int par_index) noexcept nogil:
    """Check if par price at index is valid for given star tier."""
    if star_tier < 1 or star_tier > 5 or par_index < 0 or par_index >= 14:
        return False
    return PAR_PRICE_VALID[star_tier - 1][par_index] != 0

cpdef inline int get_par_price(int par_index) noexcept nogil:
    """Get par price at index."""
    if par_index < 0 or par_index >= 14:
        return -1
    return ALL_PAR_PRICES[par_index]

cpdef inline int get_par_index_for_slot(int star_tier, int par_slot) noexcept nogil:
    """
    Return par_index for Nth valid par price of star tier, or -1.

    Par slots map to valid par prices for a given star tier.
    For example, if star_tier=3 has valid par indices [2,3,4,5,6,7],
    then slot 0 maps to index 2, slot 1 to index 3, etc.
    """
    cdef int count = 0
    cdef int par_index
    for par_index in range(<int>GameConstants.NUM_PAR_PRICES):
        if is_valid_par_price(star_tier, <int>par_index):
            if count == par_slot:
                return par_index
            count += 1
    return -1

cpdef inline int get_required_stars(int price_index, int issued_shares) noexcept nogil:
    """
    Get required star count for a corporation to maintain its share price.

    Formula: round(issued_shares * price / 10)
    Source: 18xx.games RSS implementation (target_stars function)

    Args:
        price_index: Market price index (0-26)
        issued_shares: Number of issued shares (2-7)

    Returns:
        Required star count, or 0 for invalid inputs
    """
    cdef int price
    if price_index < 1 or price_index > 26:
        return 0
    if issued_shares < 2 or issued_shares > 7:
        return 0
    price = MARKET_PRICES[price_index]
    # Round to nearest integer: (x + 0.5) truncated
    return <int>(issued_shares * price / 10.0 + 0.5)

cpdef inline int get_max_dividend(int price_index) noexcept nogil:
    """
    Get maximum dividend per share for a given share price.

    Formula: price // 3
    Source: 18xx.games Rolling Stock implementation (max_dividend_per_share function)

    Args:
        price_index: Market price index (0-26)

    Returns:
        Maximum dividend per share, or 0 for invalid/bankrupt price
    """
    cdef int price
    if price_index < 1 or price_index > 26:
        return 0
    price = MARKET_PRICES[price_index]
    return price // 3

cdef inline (int, int) compute_synergy_bonuses(
    int* company_ids,
    int num_companies
) noexcept nogil:
    """
    Compute synergy bonuses for companies owned by a corporation.

    Counts each pair exactly once per RULES.md line 569.

    Args:
        company_ids: Array of company IDs (0-35) owned by corporation
        num_companies: Number of companies in array

    Returns:
        (total_income, marker_count): Total synergy income and number of pairs
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

def py_compute_synergy_bonuses(list company_ids):
    """Python wrapper for testing compute_synergy_bonuses."""
    cdef int[36] ids
    cdef int n = len(company_ids)
    cdef int i
    for i in range(n):
        ids[i] = company_ids[i]
    return compute_synergy_bonuses(ids, n)

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _init_module():
    """Initialize static data. Called once at module import."""
    _init_synergies()
    _init_price_lookup()
    _populate_synergies()

def _populate_synergies():
    """Populate the synergy matrix from the game data."""
    # This maps the complex synergy structure to our flat matrix
    # Format: (company_name, {bonus: [target_names]})
    synergy_data = {
        # Blues
        "CDG": {16: ["MAD", "FRA", "LHR"], 8: ["E"], 4: ["SBB", "SNCF"]},
        "MAD": {8: ["RENFE"]},
        "LHR": {8: ["BR", "E"], 16: ["FRA"]},
        "HH": {4: ["DSB", "PKP", "DR"], 8: ["BSR"]},
        "HR": {4: ["NS", "B", "DR"], 8: ["E"]},
        "HA": {4: ["NS", "B", "SNCF"], 8: ["E"]},
        "FRA": {4: ["KK", "SBB", "PKP", "DR"]},
        # Greens
        "E": {4: ["NS", "B", "SNCF"], 8: ["BR"]},
        "RENFE": {4: ["SNCF"]},
        "BSR": {4: ["DSB", "PKP", "DR"], 8: ["SJ"]},
        "FS": {4: ["KK", "SBB", "SNCF"]},
        "SZD": {4: ["PKP"]},
        # Yellows
        "DR": {2: ["WT", "BY", "BD", "HE", "OL", "SX", "MS", "PR"],
               4: ["DSB", "KK", "NS", "SBB", "B", "PKP", "SNCF"]},
        "DSB": {2: ["OL", "MS", "PR"]},
        "B": {2: ["PR"], 4: ["NS"]},
        "SNCF": {2: ["BD"], 4: ["SBB", "B"]},
        "PKP": {2: ["SX", "MS", "PR"], 4: ["KK"]},
        "KK": {2: ["BY", "SX"]},
        "SBB": {2: ["WT", "BD"], 4: ["KK"]},
        "NS": {2: ["OL", "PR"]},
        # Oranges
        "PR": {1: ["BME", "BSE", "KME", "AKE", "BPM", "MHE"], 2: ["HE", "OL", "SX", "MS"]},
        "SX": {1: ["BSE", "BPM", "MHE"], 2: ["BY"]},
        "BY": {2: ["WT"]},
        "HE": {1: ["BME", "KME"], 2: ["BY", "BD"]},
        "MS": {1: ["BSE", "AKE", "BPM", "MHE"], 2: ["OL", "SX"]},
        "BD": {1: ["BME"], 2: ["WT"]},
        "OL": {1: ["KME", "AKE", "MHE"]},
        # Reds
        "MHE": {1: ["KME", "AKE", "BPM"]},
        "BPM": {1: ["BSE", "AKE"]},
        "KME": {1: ["BME"]},
    }

    for company_name, syns in synergy_data.items():
        company_id = COMPANY_NAME_TO_ID[company_name]
        for bonus, targets in syns.items():
            for target_name in targets:
                target_id = COMPANY_NAME_TO_ID[target_name]
                _set_synergy(company_id, target_id, bonus)

# Auto-initialize on import
_init_module()