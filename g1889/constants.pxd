"""Rounds, engine steps, decision phases, action layouts, and owner codes.

Phase-local action ids follow 1889/engine-design.md §5.2. Each block of ids
is named by its first id; "+ x" marks a block indexed by x (corp c, private p,
hex h, train type t). A sale of n shares is encoded as n - 1. Exact amounts in
BID, TRAIN_PRICE, and PRIVATE_PRICE are ``*_EXACT + amount``: at or above the
phase size, legal by range check, and never enumerated.
"""

from libc.stdint cimport int8_t, int16_t

from g1889.data.tables cimport (
    MAX_LAY_CANDIDATES, NUM_CORPS, NUM_HEXES, NUM_PAR_PRICES, NUM_PORTS, NUM_PRIVATES, NUM_SIDES,
    NUM_TRADE_IN_TYPES, NUM_TRAIN_TYPES,
)

cpdef enum Round:
    ROUND_AUCTION = 0
    ROUND_STOCK = 1
    ROUND_OPERATING = 2
    ROUND_GAME_OVER = 3
    NUM_ROUNDS = 4

# Engine steps (§6.1). STEP_TO_DECISION_PHASE maps automated and terminal
# steps (SR_END, OR_START, TURN_START, ROUTE, TURN_END, OR_END, GAME_OVER) to -1.
cpdef enum Step:
    STEP_AUCTION = 0
    STEP_BID = 1
    STEP_STOCK = 2
    STEP_PAR = 3
    STEP_SELL_AMOUNT = 4
    STEP_PORT = 5
    STEP_SR_END = 6
    STEP_OR_START = 7
    STEP_TURN_START = 8
    STEP_TRACK = 9
    STEP_TILE = 10
    STEP_TOKEN = 11
    STEP_ROUTE = 12
    STEP_DIVIDEND = 13
    STEP_BUY_TRAIN = 14
    STEP_SELECT_TRAIN = 15
    STEP_TRAIN_PRICE = 16
    STEP_EMERGENCY_SELL_AMOUNT = 17
    STEP_DISCARD = 18
    STEP_BUY_PRIVATE = 19
    STEP_PRIVATE_PRICE = 20
    STEP_ER_BONUS = 21
    STEP_TURN_END = 22
    STEP_OR_END = 23
    STEP_GAME_OVER = 24
    NUM_STEPS = 25

cpdef enum DecisionPhase:
    DPHASE_AUCTION = 0
    DPHASE_BID = 1
    DPHASE_STOCK = 2
    DPHASE_PAR = 3
    DPHASE_SELL_AMOUNT = 4
    DPHASE_TRACK = 5
    DPHASE_TILE = 6
    DPHASE_TOKEN = 7
    DPHASE_DIVIDEND = 8
    DPHASE_DISCARD = 9
    DPHASE_BUY_TRAIN = 10
    DPHASE_SELECT_TRAIN = 11
    DPHASE_TRAIN_PRICE = 12
    DPHASE_EMERGENCY_SELL_AMOUNT = 13
    DPHASE_BUY_PRIVATE = 14
    DPHASE_PRIVATE_PRICE = 15
    DPHASE_ER_BONUS = 16
    DPHASE_PORT = 17
    NUM_DECISION_PHASES = 18

# Private owner codes (the privates block of the state).
cpdef enum Owner:
    OWNER_NOT_IN_PLAY = -3
    OWNER_CLOSED = -2
    OWNER_UNSOLD = -1
    OWNER_PLAYER = 0                # + player
    OWNER_CORP = 8                  # + corp

# Why the game ended (Ruby's game_end_reason); NONE until it is over.
cpdef enum EndReason:
    END_REASON_NONE = 0
    END_REASON_BANKRUPT = 1
    END_REASON_BANK = 2
    NUM_END_REASONS = 3

# A round of passes with A unsold takes this much off A's price (``all_passed!``).
cpdef enum AuctionRules:
    AUCTION_DISCOUNT_STEP = 5

# AUCTION: the turn player passes, buys the topmost private at its price, or bids on private p.
cpdef enum AuctionAction:
    AUCTION_PASS = 0
    AUCTION_BUY = 1
    AUCTION_BID = 2                 # + p
    ACTION_SIZE_AUCTION = AUCTION_BID + NUM_PRIVATES

# BID: point k bids the minimum + ¥5k. Slot 0 drops out, legal only in a closed auction
# (``auction_private``), where the lowest bidder acts and the minimum is the high bid + ¥5.
cpdef enum BidAction:
    BID_DROP = 0
    BID_POINT = 1                   # + k
    BID_POINTS = 16
    ACTION_SIZE_BID = BID_POINT + BID_POINTS
    BID_EXACT = ACTION_SIZE_BID     # + amount

# STOCK: pass or done, buy from the IPO or the pool, par (PAR follows), sell corp c's
# shares (SELL_AMOUNT follows), DR, MF.
cpdef enum StockAction:
    STOCK_PASS = 0
    STOCK_IPO = 1                   # + c
    STOCK_POOL = STOCK_IPO + NUM_CORPS  # + c
    STOCK_PAR = STOCK_POOL + NUM_CORPS  # + c
    STOCK_SELL = STOCK_PAR + NUM_CORPS  # + c
    STOCK_DR = STOCK_SELL + NUM_CORPS
    STOCK_MF = STOCK_DR + 1
    ACTION_SIZE_STOCK = STOCK_MF + 1

# SELL_AMOUNT (an SR sale) and EMERGENCY_SELL_AMOUNT (the president's, in an emergency):
# n shares of the pending corp, as n - 1.
cpdef enum SellAmountAction:
    MAX_SALE_SHARES = 5             # the pool holds at most 50%
    ACTION_SIZE_SELL_AMOUNT = MAX_SALE_SHARES
    ACTION_SIZE_EMERGENCY_SELL_AMOUNT = MAX_SALE_SHARES

# PAR: the par index, in ascending price order (PAR_PRICE).
cpdef enum ParAction:
    ACTION_SIZE_PAR = NUM_PAR_PRICES

# TRACK: pass, choose a hex (TILE follows), buy a private, DR, MF.
cpdef enum TrackAction:
    TRACK_PASS = 0
    TRACK_HEX = 1                   # + h
    TRACK_BUY_PRIVATE = TRACK_HEX + NUM_HEXES  # + p
    TRACK_DR = TRACK_BUY_PRIVATE + NUM_PRIVATES
    TRACK_MF = TRACK_DR + 1
    ACTION_SIZE_TRACK = TRACK_MF + 1

# TILE: an index into the chosen hex's lay candidates.
cpdef enum TileAction:
    ACTION_SIZE_TILE = MAX_LAY_CANDIDATES

# TOKEN: pass, token a hex, buy a private, DR.
cpdef enum TokenAction:
    TOKEN_PASS = 0
    TOKEN_HEX = 1                   # + h
    TOKEN_BUY_PRIVATE = TOKEN_HEX + NUM_HEXES  # + p
    TOKEN_DR = TOKEN_BUY_PRIVATE + NUM_PRIVATES
    ACTION_SIZE_TOKEN = TOKEN_DR + 1

cpdef enum DividendAction:
    DIVIDEND_WITHHOLD = 0
    DIVIDEND_PAYOUT = 1
    DIVIDEND_BUY_PRIVATE = 2        # + p
    DIVIDEND_DR = DIVIDEND_BUY_PRIVATE + NUM_PRIVATES
    ACTION_SIZE_DIVIDEND = DIVIDEND_DR + 1

# DISCARD: the train type to discard.
cpdef enum DiscardAction:
    ACTION_SIZE_DISCARD = NUM_TRAIN_TYPES

# A train purchase's source (``pending_train_source``): the depot, the pool, or corp c.
cpdef enum TrainSource:
    TRAIN_SOURCE_DEPOT = 0
    TRAIN_SOURCE_POOL = 1
    TRAIN_SOURCE_CORP = 2           # + c
    NUM_TRAIN_SOURCES = TRAIN_SOURCE_CORP + NUM_CORPS

# BUY_TRAIN: pass, buy from source s (BUY_TRAIN_DEPOT + s; SELECT_TRAIN follows), an
# emergency sale of corp c's shares (EMERGENCY_SELL_AMOUNT follows), bankruptcy, buy a
# private, DR.
cpdef enum BuyTrainAction:
    BUY_TRAIN_PASS = 0
    BUY_TRAIN_DEPOT = 1
    BUY_TRAIN_POOL = BUY_TRAIN_DEPOT + TRAIN_SOURCE_POOL
    BUY_TRAIN_CORP = BUY_TRAIN_DEPOT + TRAIN_SOURCE_CORP  # + c
    BUY_TRAIN_SELL = BUY_TRAIN_DEPOT + NUM_TRAIN_SOURCES  # + c
    BUY_TRAIN_BANKRUPT = BUY_TRAIN_SELL + NUM_CORPS
    BUY_TRAIN_BUY_PRIVATE = BUY_TRAIN_BANKRUPT + 1  # + p
    BUY_TRAIN_DR = BUY_TRAIN_BUY_PRIVATE + NUM_PRIVATES
    ACTION_SIZE_BUY_TRAIN = BUY_TRAIN_DR + 1

# SELECT_TRAIN: a train of type t from the pending source (TRAIN_PRICE follows a corp's),
# or from the depot a D for a traded TRADE_IN_TYPE[i].
cpdef enum SelectTrainAction:
    SELECT_TRAIN_TYPE = 0           # + t
    SELECT_TRAIN_TRADE_IN = SELECT_TRAIN_TYPE + NUM_TRAIN_TYPES  # + i
    ACTION_SIZE_SELECT_TRAIN = SELECT_TRAIN_TRADE_IN + NUM_TRADE_IN_TYPES

# TRAIN_PRICE points: ¥1, list price, ⌈cash/3⌉, ⌈2·cash/3⌉, cash.
cpdef enum TrainPriceAction:
    TRAIN_PRICE_POINTS = 5
    ACTION_SIZE_TRAIN_PRICE = TRAIN_PRICE_POINTS
    TRAIN_PRICE_EXACT = ACTION_SIZE_TRAIN_PRICE  # + price

cpdef enum BuyPrivateAction:
    BUY_PRIVATE_DONE = 0
    BUY_PRIVATE_PRIVATE = 1         # + p
    BUY_PRIVATE_DR = BUY_PRIVATE_PRIVATE + NUM_PRIVATES
    ACTION_SIZE_BUY_PRIVATE = BUY_PRIVATE_DR + 1

# PRIVATE_PRICE points: ⌈face/2⌉, face, 1½·face, min(2·face, cash).
cpdef enum PrivatePriceAction:
    PRIVATE_PRICE_POINTS = 4
    ACTION_SIZE_PRIVATE_PRICE = PRIVATE_PRICE_POINTS
    PRIVATE_PRICE_EXACT = ACTION_SIZE_PRIVATE_PRICE  # + price

# ER_BONUS: ER's seller passes or lays candidate k on C4.
cpdef enum ErBonusAction:
    ER_BONUS_PASS = 0
    ER_BONUS_CANDIDATE = 1          # + k
    ACTION_SIZE_ER_BONUS = ER_BONUS_CANDIDATE + MAX_LAY_CANDIDATES

# PORT: MF's owner lays the port tile on MF_PORT_HEX[i] at rotation r.
cpdef enum PortAction:
    PORT_LAY = 0                    # + NUM_SIDES·i + r
    ACTION_SIZE_PORT = NUM_PORTS * NUM_SIDES

cpdef enum PolicyDims:
    MAX_ACTION_SIZE = ACTION_SIZE_TRACK
    UNIFIED_ACTION_DIM = (
        ACTION_SIZE_AUCTION + ACTION_SIZE_BID + ACTION_SIZE_STOCK + ACTION_SIZE_PAR + ACTION_SIZE_SELL_AMOUNT
        + ACTION_SIZE_TRACK + ACTION_SIZE_TILE + ACTION_SIZE_TOKEN + ACTION_SIZE_DIVIDEND + ACTION_SIZE_DISCARD
        + ACTION_SIZE_BUY_TRAIN + ACTION_SIZE_SELECT_TRAIN + ACTION_SIZE_TRAIN_PRICE
        + ACTION_SIZE_EMERGENCY_SELL_AMOUNT + ACTION_SIZE_BUY_PRIVATE + ACTION_SIZE_PRIVATE_PRICE
        + ACTION_SIZE_ER_BONUS + ACTION_SIZE_PORT)

# Indexed by Step and DecisionPhase; filled at import in constants.pyx.
cdef int8_t[25] STEP_TO_DECISION_PHASE
cdef int16_t[18] PHASE_ACTION_SIZE
