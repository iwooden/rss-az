# cython: language_level=3
"""
Declaration file for static game data.
Allows other Cython modules to cimport the data arrays and accessor functions.
"""

from libc.stdint cimport uint8_t, uint16_t, uint64_t, int8_t

# Constants enum - use GameConstants.NUM_COMPANIES etc.
cpdef enum GameConstants:
    NUM_COMPANIES = 36
    NUM_CORPS = 8
    NUM_MARKET_SPACES = 27
    NUM_PHASES = 11
    NUM_COO_LEVELS = 7
    MAX_PLAYERS = 6
    MAX_STAR_TIERS = 5
    MAX_DECK_SIZE = 36
    NUM_PAR_PRICES = 14
    MAX_DIVIDEND = 25
    MAX_SHARE_PRICE = 75
    COO_LEVEL_END_CARD_FLIPPED = 7

cpdef enum GamePhases:
    PHASE_INVEST = 0
    PHASE_BID_IN_AUCTION = 1
    PHASE_WRAP_UP = 2
    PHASE_ACQUISITION = 3
    PHASE_CLOSING = 4
    PHASE_INCOME = 5
    PHASE_DIVIDENDS = 6
    PHASE_END_CARD = 7
    PHASE_ISSUE_SHARES = 8
    PHASE_IPO = 9
    PHASE_GAME_OVER = 10

# Corp indices for special ability checks
cpdef enum CorpIndices:
    CORP_JS = 0 # Gets 2x income when closing companies
    CORP_S = 1  # Gets +1 income per 2 synergies
    CORP_OS = 2 # Buys from FI at face value (not high price)
    CORP_SM = 3 # Stock price does not decrease when issuing share
    CORP_PR = 4 # Gets +1 income per company owned
    CORP_DA = 5 # Gets +max(company incomes) bonus
    CORP_VM = 6 # Gets min(10, total_cost_of_ownership) bonus
    CORP_SI = 7 # Gets +2 share price movement after dividends

# Normalization constants
# Note: These are declared as extern here, defined in data.pyx
cdef float CASH_DIVISOR
cdef float SHARE_DIVISOR
cdef float STAR_DIVISOR
cdef float MAX_ROUNDTRIPS

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
cpdef int get_company_face_value(int company_id) noexcept nogil
cpdef int get_company_low_price(int company_id) noexcept nogil
cpdef int get_company_high_price(int company_id) noexcept nogil
cpdef int get_company_stars(int company_id) noexcept nogil
cpdef int get_company_income(int company_id) noexcept nogil
cpdef int get_company_synergy(int company_id, int target_id) noexcept nogil
cpdef bint is_last_in_group(int company_id) noexcept nogil

cpdef int get_corp_share_count(int corp_id) noexcept nogil

cpdef int get_market_price(int index) noexcept nogil
cpdef int get_market_index(int price) noexcept nogil

cpdef int get_cost_of_ownership(int coo_level, int star_tier) noexcept nogil
cpdef int get_adjusted_company_income(int company_id, int coo_level) noexcept nogil

cpdef bint is_valid_par_price(int star_tier, int par_index) noexcept nogil
cpdef int get_par_price(int par_index) noexcept nogil
cpdef int get_par_index_for_slot(int star_tier, int par_slot) noexcept nogil
