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
    # 11 decision phases + 4 automated/terminal phases:
    # INV, BID, ACQ_SELECT_CORP, ACQ_SELECT_COMPANY, ACQ_SELECT_PRICE,
    # ACQ_OFFER, CLOSING, DIVIDENDS, ISSUE, IPO, PAR
    # plus WRAP_UP, INCOME, END_CARD, GAME_OVER.
    NUM_PHASES = 15
    NUM_DECISION_PHASES = 11
    NUM_COO_LEVELS = 7
    MAX_PLAYERS = 6
    MAX_STAR_TIERS = 5
    MAX_DECK_SIZE = 36
    NUM_PAR_PRICES = 14
    AUCTION_CAP = 15          # price-offset slots per company in INVEST auction (0..14)
    MAX_DIVIDEND = 26
    MAX_SHARE_PRICE = 75
    COO_LEVEL_END_CARD_FLIPPED = 7

cpdef enum GamePhases:
    PHASE_INVEST = 0
    PHASE_BID = 1
    PHASE_WRAP_UP = 2
    PHASE_ACQ_SELECT_CORP = 3
    PHASE_ACQ_OFFER = 4
    PHASE_CLOSING = 5
    PHASE_INCOME = 6
    PHASE_DIVIDENDS = 7
    PHASE_END_CARD = 8
    PHASE_ISSUE_SHARES = 9
    PHASE_IPO = 10
    PHASE_GAME_OVER = 11
    PHASE_PAR = 12
    PHASE_ACQ_SELECT_COMPANY = 13
    PHASE_ACQ_SELECT_PRICE = 14

# Decision phases — the 11-phase compressed space the transformer sees.
# The engine's 15 ``GamePhases`` fold down into these via the
# ``ENGINE_TO_DECISION_PHASE`` table declared below: WRAP_UP / INCOME /
# END_CARD / GAME_OVER are automated or terminal and map to -1 so the
# driver fast-forwards through them without consulting the action module.
# ``cpdef enum`` makes each ``DPHASE_*`` value both a cimport target for
# Cython code and an attribute on ``core.data.DecisionPhase`` for Python.
cpdef enum DecisionPhase:
    DPHASE_INVEST = 0
    DPHASE_BID = 1
    DPHASE_ACQ_SELECT_CORP = 2
    DPHASE_ACQ_OFFER = 3
    DPHASE_CLOSING = 4
    DPHASE_DIVIDENDS = 5
    DPHASE_ISSUE = 6
    DPHASE_IPO = 7
    DPHASE_PAR = 8
    DPHASE_ACQ_SELECT_COMPANY = 9
    DPHASE_ACQ_SELECT_PRICE = 10

# Engine-phase → decision-phase lookup table. Indexed by ``GamePhases``;
# slots corresponding to automated/terminal phases hold -1. Filled in at
# module import time in ``core/data.pyx``. Cython callers cimport this
# array directly (``get_decision_phase`` in ``core/actions.pyx`` reads it
# on the nogil hot path); Python callers get a plain-int list mirror
# named ``ENGINE_TO_DECISION_PHASE`` on ``core.data``.
cdef int ENGINE_TO_DECISION_PHASE[15]

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

# Per-decision-phase action-space sizes.
#
# Single source of truth for the policy head output widths. ``core/actions``
# cimports these to build its encode/decode formulae; ``nn/transformer.py``
# imports them to size its per-phase policy heads. Keeping them here (rather
# than in ``core/actions.pxd``) means the model module doesn't need to touch
# the actions Cython layer, and the import-time drift check between the two
# goes away — they share one definition.
#
# Any change here must stay consistent with the ``encode_*`` arithmetic in
# ``core/actions.pxd``; that file still holds the roundtrip asserts.
cpdef enum ActionSize:
    ACTION_SIZE_INVEST = 53             # 1 pass + 36 company-select + 8*2 trade
    ACTION_SIZE_BID = 16                # 1 pass (= leave auction) + AUCTION_CAP raise offsets
    ACTION_SIZE_ACQ_SELECT_CORP = 9     # 1 pass + 8 corp-select (company/price chosen later)
    ACTION_SIZE_ACQ_SELECT_COMPANY = 36 # 36 company-select (no pass; SELECT_CORP commits)
    ACTION_SIZE_ACQ_SELECT_PRICE = 51   # 51 price offsets (no pass; FI targets handled in SELECT_COMPANY)
    ACTION_SIZE_ACQ_OFFER = 2           # pass + buy
    ACTION_SIZE_CLOSING = 37            # 1 pass + 36 company closes
    ACTION_SIZE_DIVIDENDS = 26          # dividend amounts 0..25
    ACTION_SIZE_ISSUE = 2               # pass + issue
    ACTION_SIZE_IPO = 9                 # 1 pass + 8 corp-select (par index chosen in PAR)
    ACTION_SIZE_PAR = 14                # 14 par indices (no pass)
    MAX_ACTION_SIZE = 53                # max over all phases (INVEST)

# Normalization constants used during token extraction for NN input.
# Exposed as C #defines so the values are compile-time constants in every
# module that cimports them — the C compiler can fold them directly into
# multiplies/divisions in the token-extraction hot path instead of
# loading a module-level global at every use.
cdef extern from *:
    """
    #define CASH_DIVISOR 150.0f
    #define NET_WORTH_DIVISOR 200.0f
    #define COMPANY_INCOME_DIVISOR 10.0f
    #define ENTITY_INCOME_DIVISOR 80.0f
    #define SHARE_DIVISOR 7.0f
    #define COMPANY_PRICE_DIVISOR 80.0f
    #define SHARE_PRICE_DIVISOR 75.0f
    #define COMPANY_STAR_DIVISOR 5.0f
    #define CORP_STAR_DIVISOR 40.0f
    #define MAX_ROUNDTRIPS 2.0f
    #define IMPACT_DIVISOR 5.0f
    #define PRICE_RANGE_DIVISOR 51.0f
    """
    const float CASH_DIVISOR
    const float NET_WORTH_DIVISOR
    const float COMPANY_INCOME_DIVISOR
    const float ENTITY_INCOME_DIVISOR
    const float SHARE_DIVISOR
    const float COMPANY_PRICE_DIVISOR
    const float SHARE_PRICE_DIVISOR
    const float COMPANY_STAR_DIVISOR
    const float CORP_STAR_DIVISOR
    const float MAX_ROUNDTRIPS
    const float IMPACT_DIVISOR
    const float PRICE_RANGE_DIVISOR

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
