"""BUY_TRAIN, TRAIN_PRICE, and DISCARD handlers; see buy_train.pxd.

Ruby's BuyTrain and DiscardTrain steps (``step/buy_train.rb``,
``step/train.rb``, ``step/discard_train.rb``). The corp buys one train at a
time and stays in BUY_TRAIN until it passes; with nothing left to buy, the
pass is its one choice, which the driver applies as Ruby passes the step
(``pass_if_cannot_buy_train?``). A buy from another corp names its price in
TRAIN_PRICE (``pending_train_source``) and returns to BUY_TRAIN. Every
purchase runs ``PHASE.on_train_bought`` (``Phase#buying_train!``); when the
phase advances, each corp over the new limit discards in DISCARD, the first
in corp order first (``crowded_corps``), before BUY_TRAIN goes on
(``return_step``). Ruby accepts a discard from any crowded corp; the order
cannot change the result, so replay reorders such discards (§10.2).
"""

from g1889.actions cimport (
    ACTION_BUY_CORP_TRAIN, ACTION_BUY_DEPOT_TRAIN, ACTION_BUY_POOL_TRAIN, ACTION_DISCARD, ACTION_PASS,
    ACTION_TRADE_IN, ActionInfo, price_amount,
)
from g1889.constants cimport STEP_BUY_PRIVATE, STEP_BUY_TRAIN, STEP_DISCARD, STEP_TRAIN_PRICE, STEP_TURN_END
from g1889.data.tables cimport D_TRAIN, NUM_TRAIN_TYPES, PHASE_CAN_BUY_PRIVATES
from g1889.entities.corp cimport corp_president
from g1889.entities.phase cimport Phase, game_phase
from g1889.entities.trains cimport Trains, trains_first_crowded
from g1889.entities.turn cimport TurnState, turn_operating_corp, turn_pending_train_source, turn_return_step
from g1889.state cimport GameState

from g1889.entities import phase as _phase, trains as _trains, turn as _turn

cdef Phase PHASE = _phase.PHASE
cdef Trains TRAINS = _trains.TRAINS
cdef TurnState TURN = _turn.TURN


cdef void _bought(GameState state, int train_type) except *:
    """After a purchase: the phase may advance (``Phase#buying_train!``), and
    then the first corp over its new limit discards; otherwise BUY_TRAIN
    goes on."""
    cdef int crowded
    if not PHASE.on_train_bought(state, train_type):
        return
    crowded = trains_first_crowded(state)
    if crowded >= 0:
        TURN.set_return_step(state, STEP_BUY_TRAIN)
        TURN.set_step(state, STEP_DISCARD)
        TURN.set_active_player(state, corp_president(state, crowded))


cdef void apply_buy_train(GameState state, const ActionInfo* info) except *:
    """Pass: the turn's train buying is done; in phases 3-4 the corp may then
    buy privates (Ruby's blocking BuyCompany), otherwise the turn ends. Buy a
    depot or pool train at its list price, or trade a 4, 5, or 6 and ¥800
    for a D; or choose another corp's train (TRAIN_PRICE follows)."""
    cdef int corp = turn_operating_corp(state)
    cdef int t = info.train_type
    if info.type == ACTION_PASS:
        TURN.set_step(state, STEP_BUY_PRIVATE if PHASE_CAN_BUY_PRIVATES[game_phase(state)] else STEP_TURN_END)
        return
    if info.type == ACTION_BUY_CORP_TRAIN:
        TURN.set_pending_train_source(state, info.corp * <int>NUM_TRAIN_TYPES + t)
        TURN.set_step(state, STEP_TRAIN_PRICE)
        return
    if info.type == ACTION_BUY_DEPOT_TRAIN:
        TRAINS.buy_from_depot(state, corp, t)
    elif info.type == ACTION_TRADE_IN:
        TRAINS.trade_in_for_d(state, corp, t)
        t = <int>D_TRAIN
    else:
        assert info.type == ACTION_BUY_POOL_TRAIN, f"apply_buy_train: action type {info.type}"
        TRAINS.buy_from_pool(state, corp, t)
    _bought(state, t)


cdef void apply_train_price(GameState state, const ActionInfo* info) except *:
    """Buy the pending train from its corp for the chosen price, paid to the
    seller (``Game#buy_train``), and go back to BUY_TRAIN."""
    cdef int source = turn_pending_train_source(state)
    cdef int t = source % <int>NUM_TRAIN_TYPES
    TRAINS.buy_from_corp(state, turn_operating_corp(state), source // <int>NUM_TRAIN_TYPES, t,
                         price_amount(state, info))
    TURN.set_pending_train_source(state, -1)
    TURN.set_step(state, STEP_BUY_TRAIN)
    _bought(state, t)


cdef void apply_discard(GameState state, const ActionInfo* info) except *:
    """The first crowded corp discards a train of the chosen type to the
    pool (``Depot#reclaim_train``). The next crowded corp's president
    discards next, or the purchase's step goes on with the operating corp's
    president."""
    cdef int corp
    assert info.type == ACTION_DISCARD, f"apply_discard: action type {info.type}"
    TRAINS.discard(state, trains_first_crowded(state), info.train_type)
    corp = trains_first_crowded(state)
    if corp < 0:
        TURN.set_step(state, turn_return_step(state))
        TURN.set_return_step(state, -1)
        corp = turn_operating_corp(state)
    TURN.set_active_player(state, corp_president(state, corp))
