# cython: language_level=3
"""
Declaration file for static game data.
Allows other Cython modules to cimport the data arrays and accessor functions.
"""

from libc.stdint cimport uint8_t, uint16_t, uint64_t, int8_t

# Constants (these need to be accessible)
cdef enum:
    NUM_COMPANIES = 36
    NUM_CORPS = 8
    NUM_MARKET_SPACES = 27
    NUM_PHASES = 12
    MAX_PLAYERS = 6
    MAX_STAR_TIERS = 5
    NUM_PAR_PRICES = 14
    MAX_DIVIDEND = 26
    MAX_SHARE_PRICE = 75
    COO_LEVEL_END_CARD_FLIPPED = 7

# Encoding divisors (for float tensor storage)
# Note: DEF creates compile-time constants, usable as literals
DEF CASH_DIVISOR = 200.0
DEF SHARE_DIVISOR = 7.0

# Corp indices for special ability checks
cdef enum:
    CORP_JS = 0
    CORP_S = 1
    CORP_OS = 2
    CORP_SM = 3
    CORP_PR = 4
    CORP_DA = 5
    CORP_VM = 6
    CORP_SI = 7

# Company data arrays
cdef int[36] COMPANY_FACE_VALUE
cdef int[36] COMPANY_LOW_PRICE
cdef int[36] COMPANY_HIGH_PRICE
cdef int[36] COMPANY_STARS
cdef int[36] COMPANY_INCOME
cdef uint8_t[36] COMPANY_LAST_IN_GROUP
cdef int8_t[36][36] COMPANY_SYNERGY

# Corp data
cdef int[8] CORP_SHARE_COUNT

# Market data
cdef int[27] MARKET_PRICES
cdef int[76] PRICE_TO_MARKET_INDEX
cdef int[14] ALL_PAR_PRICES
cdef uint8_t[5][14] PAR_PRICE_VALID

# Cost of ownership
cdef int[7][5] COST_OF_OWNERSHIP

# Accessor functions
cdef int get_company_face_value(int company_id) noexcept nogil
cdef int get_company_low_price(int company_id) noexcept nogil
cdef int get_company_high_price(int company_id) noexcept nogil
cdef int get_company_stars(int company_id) noexcept nogil
cdef int get_company_income(int company_id) noexcept nogil
cdef int get_company_synergy(int company_id, int target_id) noexcept nogil
cdef bint is_last_in_group(int company_id) noexcept nogil

cdef int get_corp_share_count(int corp_id) noexcept nogil

cdef int get_market_price(int index) noexcept nogil
cdef int get_market_index(int price) noexcept nogil

cdef int get_cost_of_ownership(int coo_level, int star_tier) noexcept nogil
cdef int get_adjusted_company_income(int company_id, int coo_level) noexcept nogil

cdef bint is_valid_par_price(int star_tier, int par_index) noexcept nogil
cdef int get_par_price(int par_index) noexcept nogil
cdef int get_par_index_for_slot(int star_tier, int par_slot) noexcept nogil
