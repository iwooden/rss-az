# cython: language_level=3
"""
Declaration file for static game data.

Exposes the raw game-data arrays, enums, and normalization constants used by
the engine and the token-extraction layer. This file is data-only — there are
no field-level accessor functions; entity handles own all reads/writes against
GameState, and other modules cimport the underlying arrays directly when they
need static game data.
"""

from libc.stdint cimport uint8_t, int8_t

# Constants enum - use GameConstants.NUM_COMPANIES etc.
cpdef enum GameConstants:
    NUM_COMPANIES = 36
    NUM_CORPS = 8
    NUM_MARKET_SPACES = 27
    NUM_PHASES = 12
    NUM_DECISION_PHASES = 8
    NUM_COO_LEVELS = 7
    MAX_PLAYERS = 6
    MAX_STAR_TIERS = 5
    MAX_DECK_SIZE = 36
    NUM_PAR_PRICES = 14
    MAX_DIVIDEND = 26
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
    PHASE_PAR = 10
    PHASE_GAME_OVER = 11

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

# Normalization constants used during token extraction for NN input.
# Defined in data.pyx; cimported by the token-extraction layer.
cdef float CASH_DIVISOR
cdef float NET_WORTH_DIVISOR
cdef float COMPANY_INCOME_DIVISOR
cdef float ENTITY_INCOME_DIVISOR
cdef float SHARE_DIVISOR
cdef float COMPANY_PRICE_DIVISOR
cdef float SHARE_PRICE_DIVISOR
cdef float COMPANY_STAR_DIVISOR
cdef float CORP_STAR_DIVISOR
cdef float MAX_ROUNDTRIPS
cdef float IMPACT_DIVISOR

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
