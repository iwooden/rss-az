"""BUY_TRAIN, TRAIN_PRICE, and DISCARD handlers; see buy_train.pxd.

Ruby's BuyTrain and DiscardTrain steps (``step/buy_train.rb``,
``step/train.rb``, ``step/emergency_money.rb``, ``step/discard_train.rb``)
and the OR's Bankrupt step (``step/bankrupt.rb``). The corp buys one train at
a time and stays in BUY_TRAIN until it passes; with nothing left to buy, the
pass is its one choice, which the driver applies as Ruby passes the step
(``pass_if_cannot_buy_train?``). A buy from another corp names its price in
TRAIN_PRICE (``pending_train_source``) and returns to BUY_TRAIN. Every
purchase runs ``PHASE.on_train_bought`` (``Phase#buying_train!``); when the
phase advances, each corp over the new limit discards in DISCARD, the first
in corp order first (``crowded_corps``), before BUY_TRAIN goes on
(``return_step``). Ruby accepts a discard from any crowded corp; the order
cannot change the result, so replay reorders such discards (§10.2).

Emergency money (§6.2), when the corp must buy and its treasury is short of
the cheapest depot or pool train (``actions.pyx`` has the legality): the
president pays what the treasury lacks toward that train, sells shares into
the pool first if need be, and goes bankrupt when even that falls short.
Fields:

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
    ACTION_PASS, ACTION_SELL, ACTION_TRADE_IN, ActionInfo, price_amount,
)
from g1889.constants cimport (
    END_REASON_BANKRUPT, OWNER_CORP, OWNER_PLAYER, STEP_BUY_PRIVATE, STEP_BUY_TRAIN, STEP_DISCARD,
    STEP_TRAIN_PRICE, STEP_TURN_END,
)
from g1889.data.tables cimport D_TRAIN, NUM_CORPS, NUM_TRAIN_TYPES, PHASE_CAN_BUY_PRIVATES, TRAIN_PRICE
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport Corporation, corp_cash, corp_emergency_sale, corp_president
from g1889.entities.phase cimport Phase, game_phase
from g1889.entities.player cimport Player, player_cash
from g1889.entities.trains cimport Trains, trains_first_crowded
from g1889.entities.turn cimport (
    TurnState, turn_ebuy_sold_mask, turn_operating_corp, turn_pending_train_source, turn_return_step,
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
    """After a purchase: the phase may advance (``Phase#buying_train!``), and
    then the first corp over its new limit discards; otherwise BUY_TRAIN
    goes on. A purchase after emergency sales spends the treasury, which
    passes Ruby's step and clears ``last_share_sold_price``."""
    cdef int crowded
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
    buy privates (Ruby's blocking BuyCompany), otherwise the turn ends. Buy a
    depot or pool train at its list price, the president paying what the
    treasury lacks in an emergency; trade a 4, 5, or 6 and ¥800 for a D; or
    choose another corp's train (TRAIN_PRICE follows). In an emergency the
    president may also sell shares, or go bankrupt."""
    cdef int corp = turn_operating_corp(state)
    cdef int t = info.train_type
    if info.type == ACTION_PASS:
        TURN.set_step(state, STEP_BUY_PRIVATE if PHASE_CAN_BUY_PRIVATES[game_phase(state)] else STEP_TURN_END)
        return
    if info.type == ACTION_BUY_CORP_TRAIN:
        TURN.set_pending_train_source(state, info.corp * <int>NUM_TRAIN_TYPES + t)
        TURN.set_step(state, STEP_TRAIN_PRICE)
        return
    if info.type == ACTION_SELL:
        _sell(state, corp, info.corp, info.count)
        return
    if info.type == ACTION_BANKRUPT:
        _bankrupt(state, corp)
        return
    if info.type == ACTION_BUY_DEPOT_TRAIN:
        _contribute(state, corp, TRAIN_PRICE[t])
        TRAINS.buy_from_depot(state, corp, t)
    elif info.type == ACTION_TRADE_IN:
        TRAINS.trade_in_for_d(state, corp, t)
        t = <int>D_TRAIN
    else:
        assert info.type == ACTION_BUY_POOL_TRAIN, f"apply_buy_train: action type {info.type}"
        _contribute(state, corp, TRAIN_PRICE[t])
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
