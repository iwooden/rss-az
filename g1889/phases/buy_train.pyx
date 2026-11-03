"""BUY_TRAIN, SELECT_TRAIN, TRAIN_PRICE, EMERGENCY_SELL_AMOUNT, and DISCARD
handlers; see buy_train.pxd.

Ruby's BuyTrain and DiscardTrain steps (``step/buy_train.rb``,
``step/train.rb``, ``step/emergency_money.rb``, ``step/discard_train.rb``)
and the OR's Bankrupt step (``step/bankrupt.rb``). The corp buys one train at
a time and stays in BUY_TRAIN until it passes; with nothing left to buy, the
pass is its one choice, which the driver applies as Ruby passes the step
(``pass_if_cannot_buy_train?``). A purchase is factored: BUY_TRAIN picks the
source, the depot, the pool, or another corp (``pending_train_source``), and
SELECT_TRAIN the train; a buy from another corp then names its price in
TRAIN_PRICE (``pending_train_type``). Each returns to BUY_TRAIN. Every
purchase runs ``PHASE.on_train_bought`` (``Phase#buying_train!``); when the
phase advances, each corp over the new limit discards in DISCARD, the first
in corp order first (``crowded_corps``), before BUY_TRAIN goes on
(``return_step``). Ruby accepts a discard from any crowded corp; the order
cannot change the result, so replay reorders such discards (§10.2).

Emergency money (§6.2), when the corp must buy and its treasury is short of
the cheapest depot or pool train (``actions.pyx`` has the legality): the
president pays what the treasury lacks toward that train, sells shares into
the pool first if need be (BUY_TRAIN picks the corp, ``pending_corp``, and
EMERGENCY_SELL_AMOUNT the block), and goes bankrupt when even that falls
short. Fields:

- ``ebuy_sold_mask`` holds the corps the president has sold this turn,
  each once (Ruby's ``corporations_sold``, ``MUST_SELL_IN_BLOCKS``);
- ``ebuy_sold_any`` is set from a sale to the next purchase, and bars buys
  from other corps (Ruby's ``last_share_sold_price``,
  ``EBUY_FROM_OTHERS = :never``).

TURN_END clears both. A bankruptcy ends the game at once (``GAME_END_CHECK``'s
``bankrupt: :immediate``).
"""

from g1889.actions cimport (
    ACTION_BANKRUPT, ACTION_BUY_CORP_TRAIN, ACTION_BUY_DEPOT_TRAIN, ACTION_BUY_POOL_TRAIN, ACTION_DISCARD,
    ACTION_PASS, ACTION_SELECT_SALE, ACTION_TRADE_IN, ActionInfo, price_amount,
)
from g1889.constants cimport (
    END_REASON_BANKRUPT, OWNER_CORP, OWNER_PLAYER, STEP_BUY_PRIVATE, STEP_BUY_TRAIN, STEP_DISCARD,
    STEP_EMERGENCY_SELL_AMOUNT, STEP_SELECT_TRAIN, STEP_TRAIN_PRICE, STEP_TURN_END, TRAIN_SOURCE_CORP,
    TRAIN_SOURCE_DEPOT, TRAIN_SOURCE_POOL,
)
from g1889.data.tables cimport D_TRAIN, NUM_CORPS, PHASE_CAN_BUY_PRIVATES, TRAIN_PRICE
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport Corporation, corp_cash, corp_emergency_sale, corp_president
from g1889.entities.phase cimport Phase, game_phase
from g1889.entities.player cimport Player, player_cash
from g1889.entities.trains cimport Trains, trains_first_crowded
from g1889.entities.turn cimport (
    TurnState, turn_ebuy_sold_mask, turn_operating_corp, turn_pending_corp, turn_pending_train_source,
    turn_pending_train_type,
)
from g1889.phases.operating cimport end_game, resort_order
from g1889.state cimport GameState

from g1889.entities import bank as _bank, corp as _corp, phase as _phase, player as _player, trains as _trains
from g1889.entities import turn as _turn

cdef Bank BANK = _bank.BANK
cdef list CORPS = _corp.CORPS
cdef Phase PHASE = _phase.PHASE
cdef list PLAYERS = _player.PLAYERS
cdef Trains TRAINS = _trains.TRAINS
cdef TurnState TURN = _turn.TURN


cdef void _bought(GameState state, int train_type) except *:
    """After a purchase: the pending source and type clear, and BUY_TRAIN
    goes on, unless the phase advances (``Phase#buying_train!``) and a corp
    is over its new limit: then the first such corp discards. A purchase
    after emergency sales spends the treasury, which passes Ruby's step and
    clears ``last_share_sold_price``."""
    cdef int crowded
    TURN.set_pending_train_source(state, -1)
    TURN.set_pending_train_type(state, -1)
    TURN.set_step(state, STEP_BUY_TRAIN)
    TURN.set_ebuy_sold_any(state, 0)
    if not PHASE.on_train_bought(state, train_type):
        return
    crowded = trains_first_crowded(state)
    if crowded >= 0:
        TURN.set_return_step(state, STEP_BUY_TRAIN)
        TURN.set_step(state, STEP_DISCARD)
        TURN.set_active_player(state, corp_president(state, crowded))


cdef void _contribute(GameState state, int corp, int price) except *:
    """The president pays the corp what its treasury lacks of ``price``
    (``buy_train_action``: legal only toward the cheapest depot or pool
    train in a must-buy, ``check_for_cheapest_train``)."""
    cdef int shortfall = price - corp_cash(state, corp)
    if shortfall > 0:
        BANK.transfer(state, <int>OWNER_PLAYER + corp_president(state, corp), <int>OWNER_CORP + corp, shortfall)


cdef void _sell(GameState state, int corp, int seller, int count) except *:
    """The president sells ``count`` of ``seller``'s shares to the pool
    (``sell_shares_and_change_price``: the price drops a row per share), and
    the corps yet to run re-sort (``recalculate_order``)."""
    (<Corporation>CORPS[seller]).sell_shares(state, corp_president(state, corp), count)
    TURN.set_ebuy_sold_mask(state, turn_ebuy_sold_mask(state) | 1 << seller)
    TURN.set_ebuy_sold_any(state, 1)
    resort_order(state)


cdef void _bankrupt(GameState state, int corp) except *:
    """``Bankrupt#process_bankrupt``: the president sells the largest block
    they may of each corp not sold this turn (``sell_bankrupt_shares``, with
    BuyTrain's ``can_sell?``), the corps yet to run re-sort, their cash goes
    to the bank (the rulebook gives it to the corp; nobody scores either),
    and they are bankrupt, which ends the game at once. Their remaining
    shares and privates still count toward their score. Ruby's priority deal
    skips a bankrupt player (``priority_deal_player``), so it passes on."""
    cdef int player = corp_president(state, corp)
    cdef int sold = turn_ebuy_sold_mask(state)
    cdef int c, count
    for c in range(<int>NUM_CORPS):
        count = corp_emergency_sale(state, player, c)
        if count and not (sold >> c & 1):
            (<Corporation>CORPS[c]).sell_shares(state, player, count)
    resort_order(state)
    BANK.transfer(state, <int>OWNER_PLAYER + player, <int>BANK_ACCOUNT, player_cash(state, player))
    (<Player>PLAYERS[player]).set_bankrupt(state, True)
    if TURN.priority_player(state) == player:
        TURN.set_priority_player(state, TURN.next_player(state, player))
    end_game(state, <int>END_REASON_BANKRUPT)


cdef void apply_buy_train(GameState state, const ActionInfo* info) except *:
    """Pass: the turn's train buying is done; in phases 3-4 the corp may then
    buy privates (Ruby's blocking BuyCompany), otherwise the turn ends. Or
    choose a source of trains (SELECT_TRAIN follows). In an emergency the
    president may also choose a corp to sell (EMERGENCY_SELL_AMOUNT follows),
    or go bankrupt."""
    cdef int source = TRAIN_SOURCE_DEPOT
    if info.type == ACTION_PASS:
        TURN.set_step(state, STEP_BUY_PRIVATE if PHASE_CAN_BUY_PRIVATES[game_phase(state)] else STEP_TURN_END)
    elif info.type == ACTION_SELECT_SALE:
        TURN.set_pending_corp(state, info.corp)
        TURN.set_step(state, STEP_EMERGENCY_SELL_AMOUNT)
    elif info.type == ACTION_BANKRUPT:
        _bankrupt(state, turn_operating_corp(state))
    else:
        if info.type == ACTION_BUY_POOL_TRAIN:
            source = TRAIN_SOURCE_POOL
        elif info.type == ACTION_BUY_CORP_TRAIN:
            source = TRAIN_SOURCE_CORP + info.corp
        else:
            assert info.type == ACTION_BUY_DEPOT_TRAIN, f"apply_buy_train: action type {info.type}"
        TURN.set_pending_train_source(state, source)
        TURN.set_step(state, STEP_SELECT_TRAIN)


cdef void apply_select_train(GameState state, const ActionInfo* info) except *:
    """Buy a train of the chosen type from the pending source: from the depot
    or the pool at its list price, the president paying what the treasury
    lacks in an emergency, or a D for a traded 4, 5, or 6 and ¥800; or choose
    another corp's train (TRAIN_PRICE follows). After a purchase, ``_bought``
    picks the next step."""
    cdef int corp = turn_operating_corp(state)
    cdef int source = turn_pending_train_source(state)
    cdef int t = info.train_type
    if source >= TRAIN_SOURCE_CORP:
        TURN.set_pending_train_type(state, t)
        TURN.set_step(state, STEP_TRAIN_PRICE)
        return
    if info.type == ACTION_TRADE_IN:
        TRAINS.trade_in_for_d(state, corp, t)
        t = <int>D_TRAIN
    else:
        _contribute(state, corp, TRAIN_PRICE[t])
        if source == TRAIN_SOURCE_POOL:
            TRAINS.buy_from_pool(state, corp, t)
        else:
            TRAINS.buy_from_depot(state, corp, t)
    _bought(state, t)


cdef void apply_train_price(GameState state, const ActionInfo* info) except *:
    """Buy the pending train from its corp for the chosen price, paid to the
    seller (``Game#buy_train``); ``_bought`` picks the next step."""
    cdef int t = turn_pending_train_type(state)
    TRAINS.buy_from_corp(state, turn_operating_corp(state), turn_pending_train_source(state) - TRAIN_SOURCE_CORP,
                         t, price_amount(state, info))
    _bought(state, t)


cdef void apply_emergency_sell_amount(GameState state, const ActionInfo* info) except *:
    """The president sells ``count`` of the chosen corp's shares; BUY_TRAIN
    goes on."""
    cdef int seller = turn_pending_corp(state)
    TURN.set_pending_corp(state, -1)
    _sell(state, turn_operating_corp(state), seller, info.count)
    TURN.set_step(state, STEP_BUY_TRAIN)


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
        TURN.resume(state)
        corp = turn_operating_corp(state)
    TURN.set_active_player(state, corp_president(state, corp))
