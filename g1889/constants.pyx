"""Engine constants for the 1889 engine; constants.pxd defines them.

Python callers get every enum member as an int, ``STEP_TO_DECISION_PHASE``
and ``PHASE_ACTION_SIZES`` as lists, and ``PHASES_WITH_PASS_SLOT``.
"""

cdef void _init_tables() noexcept nogil:
    cdef int i
    for i in range(<int>NUM_STEPS):
        STEP_TO_DECISION_PHASE[i] = -1
    STEP_TO_DECISION_PHASE[<int>STEP_AUCTION] = <int>DPHASE_AUCTION
    STEP_TO_DECISION_PHASE[<int>STEP_BID] = <int>DPHASE_BID
    STEP_TO_DECISION_PHASE[<int>STEP_STOCK] = <int>DPHASE_STOCK
    STEP_TO_DECISION_PHASE[<int>STEP_PAR] = <int>DPHASE_PAR
    STEP_TO_DECISION_PHASE[<int>STEP_SELL_AMOUNT] = <int>DPHASE_SELL_AMOUNT
    STEP_TO_DECISION_PHASE[<int>STEP_PORT] = <int>DPHASE_PORT
    STEP_TO_DECISION_PHASE[<int>STEP_BUY_PRIVATE_PRE] = <int>DPHASE_BUY_PRIVATE_PRE
    STEP_TO_DECISION_PHASE[<int>STEP_TRACK] = <int>DPHASE_TRACK
    STEP_TO_DECISION_PHASE[<int>STEP_TILE] = <int>DPHASE_TILE
    STEP_TO_DECISION_PHASE[<int>STEP_TOKEN] = <int>DPHASE_TOKEN
    STEP_TO_DECISION_PHASE[<int>STEP_DIVIDEND] = <int>DPHASE_DIVIDEND
    STEP_TO_DECISION_PHASE[<int>STEP_BUY_TRAIN] = <int>DPHASE_BUY_TRAIN
    STEP_TO_DECISION_PHASE[<int>STEP_SELECT_TRAIN] = <int>DPHASE_SELECT_TRAIN
    STEP_TO_DECISION_PHASE[<int>STEP_TRAIN_PRICE] = <int>DPHASE_TRAIN_PRICE
    STEP_TO_DECISION_PHASE[<int>STEP_EMERGENCY_SELL_AMOUNT] = <int>DPHASE_EMERGENCY_SELL_AMOUNT
    STEP_TO_DECISION_PHASE[<int>STEP_DISCARD] = <int>DPHASE_DISCARD
    STEP_TO_DECISION_PHASE[<int>STEP_BUY_PRIVATE_POST] = <int>DPHASE_BUY_PRIVATE_POST
    STEP_TO_DECISION_PHASE[<int>STEP_PRIVATE_PRICE] = <int>DPHASE_PRIVATE_PRICE
    STEP_TO_DECISION_PHASE[<int>STEP_ER_BONUS] = <int>DPHASE_ER_BONUS

    PHASE_ACTION_SIZE[<int>DPHASE_AUCTION] = <int>ACTION_SIZE_AUCTION
    PHASE_ACTION_SIZE[<int>DPHASE_BID] = <int>ACTION_SIZE_BID
    PHASE_ACTION_SIZE[<int>DPHASE_STOCK] = <int>ACTION_SIZE_STOCK
    PHASE_ACTION_SIZE[<int>DPHASE_PAR] = <int>ACTION_SIZE_PAR
    PHASE_ACTION_SIZE[<int>DPHASE_SELL_AMOUNT] = <int>ACTION_SIZE_SELL_AMOUNT
    PHASE_ACTION_SIZE[<int>DPHASE_BUY_PRIVATE_PRE] = <int>ACTION_SIZE_BUY_PRIVATE_PRE
    PHASE_ACTION_SIZE[<int>DPHASE_TRACK] = <int>ACTION_SIZE_TRACK
    PHASE_ACTION_SIZE[<int>DPHASE_TILE] = <int>ACTION_SIZE_TILE
    PHASE_ACTION_SIZE[<int>DPHASE_TOKEN] = <int>ACTION_SIZE_TOKEN
    PHASE_ACTION_SIZE[<int>DPHASE_DIVIDEND] = <int>ACTION_SIZE_DIVIDEND
    PHASE_ACTION_SIZE[<int>DPHASE_DISCARD] = <int>ACTION_SIZE_DISCARD
    PHASE_ACTION_SIZE[<int>DPHASE_BUY_TRAIN] = <int>ACTION_SIZE_BUY_TRAIN
    PHASE_ACTION_SIZE[<int>DPHASE_SELECT_TRAIN] = <int>ACTION_SIZE_SELECT_TRAIN
    PHASE_ACTION_SIZE[<int>DPHASE_TRAIN_PRICE] = <int>ACTION_SIZE_TRAIN_PRICE
    PHASE_ACTION_SIZE[<int>DPHASE_EMERGENCY_SELL_AMOUNT] = <int>ACTION_SIZE_EMERGENCY_SELL_AMOUNT
    PHASE_ACTION_SIZE[<int>DPHASE_BUY_PRIVATE_POST] = <int>ACTION_SIZE_BUY_PRIVATE_POST
    PHASE_ACTION_SIZE[<int>DPHASE_PRIVATE_PRICE] = <int>ACTION_SIZE_PRIVATE_PRICE
    PHASE_ACTION_SIZE[<int>DPHASE_ER_BONUS] = <int>ACTION_SIZE_ER_BONUS
    PHASE_ACTION_SIZE[<int>DPHASE_PORT] = <int>ACTION_SIZE_PORT


# constants.pxd sizes these arrays with literals; Cython rejects enum dimensions.
assert <int>NUM_STEPS == 26 and <int>NUM_DECISION_PHASES == 19
_init_tables()

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _enum in (
    Round, Step, DecisionPhase, Owner, EndReason, Shares, AuctionRules, AuctionAction, BidAction, StockAction,
    SellAmountAction, ParAction, BuyPrivatePreAction, TrackAction, TileAction, TokenAction, DividendAction,
    DiscardAction, TrainSource, BuyTrainAction, SelectTrainAction, TrainPriceAction, BuyPrivatePostAction,
    PrivatePriceAction, ErBonusAction, PortAction, PolicyDims,
):
    for _name, _member in _enum.__members__.items():
        globals()[_name] = int(_member)
del _enum, _name, _member
globals()['STEP_TO_DECISION_PHASE'] = [<int>STEP_TO_DECISION_PHASE[i] for i in range(<int>NUM_STEPS)]
globals()['PHASE_ACTION_SIZES'] = [<int>PHASE_ACTION_SIZE[i] for i in range(<int>NUM_DECISION_PHASES)]

# Phases whose slot 0 is a pass (§5.2); the trainer's logit-scale diagnostics use them.
PHASES_WITH_PASS_SLOT = tuple(<int>p for p in (
    DPHASE_AUCTION, DPHASE_BID, DPHASE_STOCK, DPHASE_BUY_PRIVATE_PRE, DPHASE_TRACK, DPHASE_TOKEN, DPHASE_BUY_TRAIN,
    DPHASE_BUY_PRIVATE_POST, DPHASE_ER_BONUS,
))
