"""Shared fixtures and helpers for the 1889 phase tests (as tests/phases/conftest.py).

- ``game_state``: a fresh game at the auction, 2-6 players, policy flags off,
  the first seat first.
- ``new_game(num_players, ...)``: the same for one player count, with flags.
- ``get_legal_actions(state)``, ``get_policy_actions(state)``: decoded
  ``(id, info)`` lists of the engine-legal and policy ids.
- ``find_legal_action(state, **fields)``: the first legal id whose decoded
  fields (``type``, ``private``, ``count``, ...) match.
- ``opened(state, action_id)``: a copy at the sub-phase a legal id opens
  (SELL_AMOUNT, SELECT_TRAIN, ...), before any choice the driver forces;
  ``apply_factored`` applies a choice and its sub-phase's.
- Composed BUY_TRAIN choices (engine-design §5.3): ``depot(t)``,
  ``pool(t)``, ``trade(t)`` (a D for a traded t), ``from_corp(c, t)``, and
  ``sale(c, n)``; ``train_choices(state)`` lists BUY_TRAIN's legal ids with
  each source and sale composed with its sub-phase's choices, and
  ``choose_train`` and ``train_decisions`` make and spell one.
- ``apply_and_verify(state, action_id)``: apply through the driver with
  history, checking ``assert_invariants`` on every intermediate state and the
  result, and that a returned decision is a choice: two or more policy
  actions, or an amount decision with exact amounts beyond its one point.
- ``assert_invariants`` also checks the trains: every train is in the depot,
  the pool, or a floated corp until its rank rusts, and then none is left;
  no corp is over the limit outside DISCARD, which is for the first one that is.
  And the privates: TR and ER block their hexes exactly while a player owns
  them, none is open after phase 4 but a player's UTF, and MF's port tile is
  on a port hex once used. ``apply_and_verify`` checks that no corp's private
  ever goes back to a player. And the actor, the player ``active_player``
  names, at every decision: the turn player in the auction (the lowest
  bidder in a closed auction's BID); in an SR, the priority holder first, and
  the turn passes to the next seat only at STOCK's slot 0; MF's owner at
  PORT; the operating corp's president in an OR, the crowded corp's at
  DISCARD, and ER's seller at ER_BONUS. And the pending fields: a corp at
  PAR, SELL_AMOUNT, and EMERGENCY_SELL_AMOUNT, a source at SELECT_TRAIN and
  TRAIN_PRICE, and a train at TRAIN_PRICE. And the game's end and the
  emergencies: the game is over exactly at GAME_OVER with an
  ``end_reason``, which a bankruptcy's matches its one bankrupt player at ¥0
  and a broken bank's the bank; the turn's emergency sales stand from
  BUY_TRAIN to the turn's end, and ``ebuy_sold_any`` from a sale to the
  purchase; and
  ``final_scores_py`` equals Ruby's ``Player#value`` recomputed, and its
  sum the players' cash, the corps' shares outside the IPO and the pool at
  market, and the players' privates at face.
- ``enter_auction(state, turn_player, sold, bids)``: hand-built auction
  positions through ``setup_auction_py``.
- ``hold(state, corp, ...)`` and ``enter_stock_round(state, first, turn)``:
  hand-built SR positions through ``setup_stock_round_py``.
- ``enter_operating_round(state, or_num)``: an OR on hand-built holdings and
  map, through ``setup_operating_round_py`` and ``DRIVER.advance``, verified
  as ``apply_and_verify`` does; ``at_or``, ``lay``, ``set_cash``,
  ``enter_phase``, and ``give_train`` build its state.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER, STATUS_INVALID, STATUS_OK
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS, PRESIDENT_SHARES, SHARES_PER_CORP
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.phases.auction import setup_auction_py
from g1889.phases.operating import setup_operating_round_py
from g1889.phases.stock import setup_stock_round_py
from g1889.entities.bank import BANK_ACCOUNT
from g1889.state import GameState

POOL_LIMIT = D.MARKET_SHARE_LIMIT // 10
HEX = {hex_id: h for h, hex_id in enumerate(D.HEX_IDS)}
TILE = {name: t for t, name in enumerate(D.TILE_NAMES)}
# The steps of a corp's turn, from TURN_START to TURN_END.
BUY_TRAIN_STEPS = (K.STEP_BUY_TRAIN, K.STEP_SELECT_TRAIN, K.STEP_TRAIN_PRICE, K.STEP_EMERGENCY_SELL_AMOUNT)
TURN_STEPS = (K.STEP_TRACK, K.STEP_TILE, K.STEP_TOKEN, K.STEP_ROUTE, K.STEP_DIVIDEND, *BUY_TRAIN_STEPS,
              K.STEP_DISCARD, K.STEP_BUY_PRIVATE, K.STEP_PRIVATE_PRICE, K.STEP_ER_BONUS, K.STEP_PORT)
# The sub-phases that return to the step they interject in (``return_step``).
RETURNING_STEPS = (K.STEP_DISCARD, K.STEP_PRIVATE_PRICE, K.STEP_ER_BONUS, K.STEP_PORT)
OR_STEPS = (K.STEP_OR_START, K.STEP_TURN_START, *TURN_STEPS, K.STEP_TURN_END, K.STEP_OR_END)
# Where a turn's emergency sales may stand: from BUY_TRAIN to the turn's end, or a bankruptcy's game end.
EBUY_STEPS = (*BUY_TRAIN_STEPS, K.STEP_DISCARD, K.STEP_BUY_PRIVATE, K.STEP_PRIVATE_PRICE, K.STEP_ER_BONUS,
              K.STEP_TURN_END, K.STEP_GAME_OVER)


# =============================================================================
# STATES
# =============================================================================

def new_game(num_players, max_players=0, seed=42, priority_player=0, auction_pass_payout_cap=0,
             same_president=False):
    """A fresh game at the auction, by default with the first seat first;
    ``priority_player`` -1 lets the seed pick."""
    state = GameState(num_players, max_players=max_players, auction_pass_payout_cap=auction_pass_payout_cap,
                      private_sale_same_president=same_president, train_sale_same_president=same_president)
    state.initialize_game(num_players, seed=seed, priority_player=priority_player)
    return state


@pytest.fixture(params=[2, 3, 4, 5, 6])
def game_state(request):
    return new_game(request.param)


def enter_auction(state, turn_player, sold=(), bids=()):
    """Sell ``sold`` ``(private, player, price)`` and place ``bids``
    ``(player, private, amount)`` through the entity APIs, then enter the
    auction at ``turn_player``'s normal turn, resolving the topmost as after a
    sale (so bids on it sell it or open its closed auction)."""
    for private, player, price in sold:
        PRIVATES[private].sell_to_player(state, player, price)
    for player, private, amount in bids:
        PLAYERS[player].set_bid(state, private, amount)
    setup_auction_py(state, turn_player)
    assert_invariants(state, 'enter_auction')


def end_auction(state, owners=()):
    """Give the privates in play to ``owners`` (a player per private, cycling;
    the last seat by default, so DR and MF are not the usual president's) for
    free, leaving the auction behind."""
    owners = list(owners) or [TURN.num_players(state) - 1]
    in_play = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_UNSOLD]
    for i, p in enumerate(in_play):
        PRIVATES[p].sell_to_player(state, owners[i % len(owners)], 0)


def hold(state, corp, holdings, par=0, cell=None, pool=0, president=None):
    """Hand-built holdings of a parred ``corp``: ``holdings`` maps players to
    shares (the president's include the certificate's two), ``par`` is the par
    index, ``cell`` the market cell (the par cell by default), and the IPO keeps
    the rest. The president defaults to the first largest holder; the corp
    floats, with its capital from the bank, once half its shares are out."""
    c = CORPS[corp]
    c.set_par_index(state, par)
    MARKET.set_position(state, corp, *(cell or (D.PAR_ROW[par], D.PAR_COL[par])))
    for player, shares in holdings.items():
        PLAYERS[player].set_shares(state, corp, shares)
    c.set_pool_shares(state, pool)
    c.set_ipo_shares(state, SHARES_PER_CORP - pool - sum(holdings.values()))
    if president is None:
        president = max(holdings, key=lambda p: (holdings[p], -p))
    c.set_president(state, president)
    assert holdings[president] >= PRESIDENT_SHARES
    c.float_if_ready(state)


def enter_stock_round(state, first, turn=2):
    """Open SR ``turn`` at ``first``'s turn on the holdings set up by hand."""
    TURN.set_turn(state, turn)
    setup_stock_round_py(state, first)
    assert_invariants(state, 'enter_stock_round')


def at_or(num_players=3, owners=()):
    """A game past the auction, the privates in play free to ``owners`` (the
    last seat by default); holdings, the map, and ``enter_operating_round``
    follow."""
    state = new_game(num_players)
    end_auction(state, owners)
    return state


def lay(state, hex_id, tile, rotation):
    MAP.lay(state, HEX[hex_id], TILE[tile], rotation)


def set_cash(state, corp, cash):
    delta = CORPS[corp].cash(state) - cash
    BANK.transfer(state, *((K.OWNER_CORP + corp, BANK_ACCOUNT) if delta > 0 else (BANK_ACCOUNT, K.OWNER_CORP + corp)),
                  abs(delta))


def enter_phase(state, phase):
    """Hand-built game phase ``phase``: the depot has sold every rank below the
    one that opened it and one train of that rank, and those trains wait in
    the pool unless their rank has rusted (``give_train`` hands them out).
    From phase 5 the privates have closed, but a player-owned UTF."""
    PHASE.set_current(state, phase)
    for p in range(D.NUM_PRIVATES):
        owner = PRIVATES[p].owner(state)
        if (phase >= D.CLOSE_PRIVATES_PHASE and owner >= K.OWNER_UNSOLD
                and not (p == D.PRIVATE_UTF and K.OWNER_PLAYER <= owner < K.OWNER_CORP)):
            PRIVATES[p].close(state)
    for t in range(D.TRAIN_D):
        sold = TRAINS.depot(state, t) if t < phase else min(1, TRAINS.depot(state, t)) if t == phase else 0
        TRAINS.set_depot(state, t, TRAINS.depot(state, t) - sold)
        if not (D.TRAIN_RUSTS_ON[t] >= 0 and phase >= D.TRAIN_RUSTS_ON[t]):
            TRAINS.set_pool(state, t, TRAINS.pool(state, t) + sold)


def give_train(state, corp, train, count=1):
    """Move ``count`` trains of type ``train`` to ``corp`` from the pool, or from
    the depot when the pool has none (a D always comes from the depot)."""
    for _ in range(count):
        if TRAINS.pool(state, train):
            TRAINS.set_pool(state, train, TRAINS.pool(state, train) - 1)
        elif TRAINS.depot(state, train) > 0:
            TRAINS.set_depot(state, train, TRAINS.depot(state, train) - 1)
        else:
            assert TRAINS.depot(state, train) == -1, f'no type {train} train to give'
        TRAINS.set_corp_trains(state, corp, train, TRAINS.corp_trains(state, corp, train) + 1)


def enter_operating_round(state, or_num=1, or_count=None, msg='enter_operating_round'):
    """Open OR ``or_num`` of a set of ``or_count`` (the phase's by default) on
    the holdings and map set up by hand, and run it to the first choice."""
    TURN.set_or_count(state, or_count or D.PHASE_ORS[PHASE.current(state)])
    setup_operating_round_py(state, or_num)
    history = []
    status = DRIVER.advance(state, history)
    return _verify(ApplyResult(state, history, status), msg)


# =============================================================================
# ACTIONS
# =============================================================================

def _decoded(state, enumerate_ids):
    phase = A.get_decision_phase_py(state)
    assert phase >= 0, f'not at a decision (step {K.Step(TURN.step(state)).name})'
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    count = enumerate_ids(state, ids)
    return [(int(i), A.decode_action_py(phase, int(i))) for i in ids[:count]]


def get_legal_actions(state):
    return _decoded(state, A.enumerate_legal_actions_py)


def get_policy_actions(state):
    return _decoded(state, A.enumerate_policy_actions_py)


def legal_ids(state):
    return [action_id for action_id, _ in get_legal_actions(state)]


def policy_ids(state):
    return [action_id for action_id, _ in get_policy_actions(state)]


def find_legal_action(state, **fields):
    """The first legal id whose decoded info matches every given field."""
    actions = get_legal_actions(state)
    for action_id, info in actions:
        if all(getattr(info, name) == value for name, value in fields.items()):
            return action_id
    raise AssertionError(f'no legal action with {fields}; legal: {[info for _, info in actions]}')


def opened(state, action_id):
    """A copy of ``state`` at the sub-phase the legal ``action_id`` opens, as
    the driver opens it: before any choice it then forces there."""
    n = TURN.num_players(state)
    after = GameState.from_array(state._array, n, max_players=state.max_players)
    history = []
    assert DRIVER.apply_action(after, action_id, history) != STATUS_INVALID, action_id
    return after if len(history) == 1 else GameState.from_array(history[1][0], n, max_players=state.max_players)


def apply_factored(state, first, phase, second, msg='', expected_status=STATUS_OK):
    """Apply ``first``, then ``second`` at the sub-phase ``phase`` it opens,
    which the driver applies itself when it is the only choice there. The
    result's ``forced`` are the decisions the driver forced after ``second``."""
    result = apply_and_verify(state, first, msg, expected_status=None)
    if not result.forced:
        assert A.get_decision_phase_py(state) == phase, f'{msg}: {first} opened {A.get_decision_phase_py(state)}'
        return apply_and_verify(state, second, msg, expected_status)
    assert result.forced[0] == (phase, second), f'{msg}: forced {result.forced}'
    assert expected_status is None or result.status == expected_status, f'{msg}: status {result.status}'
    return ApplyResult(state, result.history[1:], result.status)


def depot(t):
    return 'depot', t


def pool(t):
    return 'pool', t


def trade(t):
    return 'trade', t


def from_corp(c, t):
    return c, t


def sale(c, n):
    return 'sell', c, n


TRAIN_SOURCES = {A.ACTION_BUY_DEPOT_TRAIN: 'depot', A.ACTION_BUY_POOL_TRAIN: 'pool', A.ACTION_BUY_CORP_TRAIN: None}


def train_choices(state):
    """BUY_TRAIN's legal ids, each source composed with the trains SELECT_TRAIN
    then offers and each emergency sale with EMERGENCY_SELL_AMOUNT's blocks."""
    choices = []
    for action_id, info in get_legal_actions(state):
        if info.type == A.ACTION_SELECT_SALE:
            choices += [sale(info.corp, sub.count) for _, sub in get_legal_actions(opened(state, action_id))]
        elif info.type in TRAIN_SOURCES:
            source = TRAIN_SOURCES[info.type] or info.corp
            choices += [('trade' if sub.type == A.ACTION_TRADE_IN else source, sub.train_type)
                        for _, sub in get_legal_actions(opened(state, action_id))]
        else:
            choices.append(action_id)
    return choices


def train_decisions(choice):
    """The two decisions of a composed BUY_TRAIN choice."""
    if choice[0] == 'sell':
        _, c, n = choice
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_SELL + c), (K.DPHASE_EMERGENCY_SELL_AMOUNT, n - 1)]
    source, t = choice
    if source == 'trade':
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_DEPOT),
                (K.DPHASE_SELECT_TRAIN, K.SELECT_TRAIN_TRADE_IN + D.TRADE_IN_TYPE.index(t))]
    slot = {'depot': K.BUY_TRAIN_DEPOT, 'pool': K.BUY_TRAIN_POOL}.get(source) or K.BUY_TRAIN_CORP + source
    return [(K.DPHASE_BUY_TRAIN, slot), (K.DPHASE_SELECT_TRAIN, K.SELECT_TRAIN_TYPE + t)]


def choose_train(state, choice, msg='', expected_status=STATUS_OK):
    """Make a composed BUY_TRAIN choice through ``apply_factored``."""
    (_, first), (phase, second) = train_decisions(choice)
    return apply_factored(state, first, phase, second, msg or str(choice), expected_status)


def exact_id(state, amount):
    """The exact-amount id of ``amount`` in the current BID, TRAIN_PRICE, or PRIVATE_PRICE."""
    return K.PHASE_ACTION_SIZES[A.get_decision_phase_py(state)] + amount


# =============================================================================
# INVARIANTS
# =============================================================================

def assert_invariants(state, msg=''):
    """Money is conserved, bids sit on unsold privates within each player's
    cash, privates sell top-down, and the auction fields agree with the step.
    Every corp's ten shares are somewhere, at most five in the pool; a parred
    corp sits on the market, its president holds the certificate's two shares
    and no fewer than anyone, and it has floated once half its shares are out."""
    n = TURN.num_players(state)
    total = BANK.cash(state) + sum(PLAYERS[i].cash(state) for i in range(n)) + sum(c.cash(state) for c in CORPS)
    assert total == D.BANK_CASH, f'{msg}: cash totals {total}'
    for c in CORPS:
        held = [PLAYERS[i].shares(state, c.corp_id) for i in range(n)]
        assert c.ipo_shares(state) + c.pool_shares(state) + sum(held) == SHARES_PER_CORP, f'{msg}: {c.sym} shares'
        assert c.pool_shares(state) <= POOL_LIMIT, f'{msg}: {c.sym} pool over the limit'
        if c.par_index(state) < 0:
            assert c.president(state) == -1 and MARKET.row(state, c.corp_id) == -1 and not c.floated(state), msg
            continue
        president = c.president(state)
        assert MARKET.row(state, c.corp_id) >= 0, f'{msg}: {c.sym} parred off the market'
        assert held[president] >= PRESIDENT_SHARES and held[president] == max(held), f'{msg}: {c.sym} president'
        assert c.floated(state) == (c.ipo_shares(state) <= SHARES_PER_CORP // 2), f'{msg}: {c.sym} float'
    owners = [p.owner(state) for p in PRIVATES]
    in_play = [o for o in owners if o != K.OWNER_NOT_IN_PLAY]
    unsold_last = sorted(in_play, key=lambda o: o == K.OWNER_UNSOLD)
    assert in_play == unsold_last, f'{msg}: privates sold out of order {owners}'
    assert all(K.OWNER_NOT_IN_PLAY <= o < K.OWNER_PLAYER + n or K.OWNER_CORP <= o < K.OWNER_CORP + D.NUM_CORPS
               for o in owners), f'{msg}: private owners {owners}'
    for i in range(n):
        player = PLAYERS[i]
        assert player.committed(state) <= player.cash(state), f'{msg}: player {i} bids beyond cash'
        for p in range(D.NUM_PRIVATES):
            if player.bid(state, p):
                assert owners[p] == K.OWNER_UNSOLD, f'{msg}: player {i} bids on sold private {p}'
                assert player.bid(state, p) >= D.PRIVATE_VALUE[p] + D.MIN_BID_INCREMENT, f'{msg}: low bid'
    _assert_map(state, msg)
    _assert_trains(state, msg)
    _assert_privates(state, msg)
    _assert_game_end(state, msg)

    step, top, active = TURN.step(state), topmost(state), TURN.active_player(state)
    if TURN.round(state) != K.ROUND_AUCTION:
        assert top == -1 and all(PLAYERS[i].committed(state) == 0 for i in range(n)), f'{msg}: auction left open'
        assert TURN.auction_private(state) == -1, msg
        assert (TURN.pending_private(state) >= 0) == (step == K.STEP_PRIVATE_PRICE), f'{msg}: pending private'
        assert (TURN.pending_seller(state) >= 0) == (step == K.STEP_ER_BONUS), f'{msg}: pending seller'
        assert (TURN.return_step(state) >= 0) == (step in RETURNING_STEPS), f'{msg}: return step'
        if step == K.STEP_PORT:  # in an SR or at TRACK, MF's owner lays its port tile
            mf = PRIVATES[D.PRIVATE_MF]
            assert mf.owner(state) == K.OWNER_PLAYER + active and not mf.flags(state) & 1, f'{msg}: PORT for MF'
        if TURN.round(state) == K.ROUND_STOCK:
            assert step in (K.STEP_STOCK, K.STEP_PAR, K.STEP_SELL_AMOUNT, K.STEP_PORT, K.STEP_SR_END), \
                f'{msg}: step {step} in the SR'
            assert step != K.STEP_PORT or TURN.return_step(state) == K.STEP_STOCK, f'{msg}: PORT returns to STOCK'
            corp = TURN.pending_corp(state)
            assert (corp >= 0) == (step in (K.STEP_PAR, K.STEP_SELL_AMOUNT)), f'{msg}: pending corp'
            assert step != K.STEP_SELL_AMOUNT or not TURN.sr_sold_mask(state) >> corp & 1, f'{msg}: sold twice'
            sold = TURN.sr_sold_mask(state)
            assert sold & ~PLAYERS[active].sold_mask(state) == 0, f'{msg}: turn sales outside the SR sales'
            assert not TURN.sr_sold_then_bought(state) or (sold and TURN.sr_bought(state)), msg
            if TURN.turn(state) == 1:
                assert all(PLAYERS[i].sold_mask(state) == 0 for i in range(n)), f'{msg}: sales in the first SR'
        else:  # the SR's fields clear when it ends
            assert TURN.sr_bought(state) == TURN.sr_sold_mask(state) == TURN.sr_sold_then_bought(state) == 0, msg
            assert all(PLAYERS[i].sold_mask(state) == 0 for i in range(n)), f'{msg}: SR sales left over'
        _assert_operating(state, msg)
        return
    assert step in (K.STEP_AUCTION, K.STEP_BID), f'{msg}: step {step} in the auction'
    if TURN.auction_private(state) >= 0:  # a closed auction, at BID
        auctioned = PRIVATES[TURN.auction_private(state)]
        assert step == K.STEP_BID and TURN.pending_private(state) == -1, f'{msg}: closed auction at {step}'
        assert TURN.auction_private(state) == top and auctioned.bid_count(state) >= 2, f'{msg}: closed auction'
        assert active == auctioned.lowest_bidder(state), f'{msg}: the lowest bidder acts in a closed auction'
        return
    assert TURN.auction_private(state) == -1 and PRIVATES[top].bid_count(state) == 0, f'{msg}: topmost has bids'
    assert active == TURN.auction_turn(state), f'{msg}: the turn player acts'
    assert (TURN.pending_private(state) >= 0) == (step == K.STEP_BID), f'{msg}: pending private'


def _assert_game_end(state, msg):
    """The game is over exactly at GAME_OVER, with its reason: a bankruptcy
    (one bankrupt player, at ¥0) or a broken bank; nobody is bankrupt
    before. The scores are Ruby's ``Player#value`` at any state, and they
    add up to the players' cash, the shares outside the IPO and the pool at
    market, and the players' privates at face."""
    n = TURN.num_players(state)
    over, reason = TURN.round(state) == K.ROUND_GAME_OVER, TURN.end_reason(state)
    assert over == (TURN.step(state) == K.STEP_GAME_OVER) == (reason != K.END_REASON_NONE), \
        f'{msg}: round {TURN.round(state)}, step {TURN.step(state)}, end reason {reason}'
    bankrupt = [i for i in range(n) if PLAYERS[i].bankrupt(state)]
    if reason == K.END_REASON_BANKRUPT:
        assert len(bankrupt) == 1 and PLAYERS[bankrupt[0]].cash(state) == 0, f'{msg}: bankrupt {bankrupt}'
    else:
        assert bankrupt == [] and (reason != K.END_REASON_BANK or BANK.broken(state)), f'{msg}: {reason} {bankrupt}'
    prices = [MARKET.price(state, c) for c in range(D.NUM_CORPS)]
    privates = [sum(D.PRIVATE_VALUE[p] for p in range(D.NUM_PRIVATES)
                    if PRIVATES[p].owner(state) == K.OWNER_PLAYER + i) for i in range(n)]
    values = [PLAYERS[i].cash(state) + privates[i] + sum(PLAYERS[i].shares(state, c) * prices[c]
                                                         for c in range(D.NUM_CORPS)) for i in range(n)]
    scores = A.final_scores_py(state)
    assert scores == values, f'{msg}: final scores {scores}, values {values}'
    held = sum((SHARES_PER_CORP - c.ipo_shares(state) - c.pool_shares(state)) * prices[c.corp_id] for c in CORPS)
    assert sum(scores) == sum(PLAYERS[i].cash(state) for i in range(n)) + held + sum(privates), f'{msg}: score sum'


def _assert_map(state, msg):
    """No supply below zero, and every charter token is left or placed, at most one per hex."""
    assert all(MAP.supply(state, t) >= 0 for t in range(D.NUM_SUPPLY_TILES)), f'{msg}: negative supply'
    for c in CORPS:
        placed = sum(MAP.tokens(state, h) >> c.corp_id & 1 for h in range(D.NUM_HEXES))
        assert c.tokens_left(state) + placed == D.CORP_TOKENS[c.corp_id], f'{msg}: {c.sym} tokens'


def _assert_privates(state, msg):
    """TR blocks K4 and ER blocks C4 exactly while a player owns them; after
    phase 4 every private has closed but a player's UTF; and a used MF has
    laid the port tile on a port hex."""
    for p, hex_id in ((D.PRIVATE_TR, 'K4'), (D.PRIVATE_ER, 'C4')):
        owner = PRIVATES[p].owner(state)
        assert PRIVATES[p].blocks_hex(state, HEX[hex_id]) == (K.OWNER_PLAYER <= owner < K.OWNER_CORP), \
            f'{msg}: {PRIVATES[p].sym} block, owner {owner}'
    if PHASE.current(state) >= D.CLOSE_PRIVATES_PHASE:
        open_ = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) >= K.OWNER_UNSOLD]
        utf = PRIVATES[D.PRIVATE_UTF].owner(state)
        assert open_ in ([], [D.PRIVATE_UTF]) and (not open_ or K.OWNER_PLAYER <= utf < K.OWNER_CORP), \
            f'{msg}: privates {open_} open in phase {PHASE.current(state)}'
    if PRIVATES[D.PRIVATE_MF].flags(state) & 1:
        assert any(MAP.tile(state, h) == D.PORT_TILE for h in D.MF_PORT_HEX), f'{msg}: MF used, no port tile'


def _assert_private_moves(before, after, msg):
    """A corp's private stays the corp's until it closes (``company_sellable``),
    and ER_BONUS opens for the player who sold ER."""
    for p in range(D.NUM_PRIVATES):
        owner = PRIVATES[p].owner(before)
        if owner >= K.OWNER_CORP:
            assert PRIVATES[p].owner(after) in (owner, K.OWNER_CLOSED), f'{msg}: corp private {p} moved'
    if TURN.step(after) == K.STEP_ER_BONUS and TURN.step(before) != K.STEP_ER_BONUS:
        seller = PRIVATES[D.PRIVATE_ER].owner(before) - K.OWNER_PLAYER
        assert TURN.pending_seller(after) == seller, f'{msg}: ER_BONUS for {TURN.pending_seller(after)}, not {seller}'


def _assert_sr_turns(before, after, entry, msg):
    """In an SR the turn player acts: the priority holder opens it, and only
    STOCK's slot 0 (a pass, or done) hands the turn to the next seat."""
    if TURN.round(after) != K.ROUND_STOCK or TURN.step(after) == K.STEP_SR_END:
        return
    if TURN.round(before) != K.ROUND_STOCK:
        expected = TURN.priority_player(after)
    else:
        expected = TURN.active_player(before)
        if entry == (K.DPHASE_STOCK, K.STOCK_PASS):
            expected = TURN.next_player(before, expected)
    assert TURN.active_player(after) == expected, f'{msg}: player {TURN.active_player(after)} acts, not {expected}'


def _assert_trains(state, msg):
    """Each finite rank's trains are in the depot, the pool, or floated corps
    until the rank rusts, and then none is left; the D is unlimited. No corp
    is over the limit but while DISCARD, for the first one that is, waits."""
    phase = PHASE.current(state)
    for t in range(D.NUM_TRAIN_TYPES):
        held = [TRAINS.corp_trains(state, c, t) for c in range(D.NUM_CORPS)]
        assert all(not n or CORPS[c].floated(state) for c, n in enumerate(held)), f'{msg}: an unfloated corp\'s train'
        if D.TRAIN_COUNT[t] < 0:
            assert TRAINS.depot(state, t) == -1, f'{msg}: the D is not unlimited'
            continue
        rusted = D.TRAIN_RUSTS_ON[t] >= 0 and phase >= D.PHASE_ON_TRAIN.index(D.TRAIN_RUSTS_ON[t])
        total = TRAINS.depot(state, t) + TRAINS.pool(state, t) + sum(held)
        assert total == (0 if rusted else D.TRAIN_COUNT[t]), f'{msg}: {total} trains of type {t} in phase {phase}'
    crowded = [c for c in range(D.NUM_CORPS) if TRAINS.crowded(state, c)]
    if TURN.step(state) == K.STEP_DISCARD:
        assert crowded and TRAINS.first_crowded(state) == crowded[0], f'{msg}: DISCARD without a crowded corp'
    else:
        assert not crowded, f'{msg}: corps {crowded} over the limit'


def _assert_operating(state, msg):
    """The OR order lists distinct floated corps, the operating corp is the one
    at ``or_index`` and its president acts (in DISCARD, the crowded corp's),
    TILE alone has a pending hex, SELECT_TRAIN and TRAIN_PRICE a pending
    source (another corp's at TRAIN_PRICE, which holds the pending train),
    and EMERGENCY_SELL_AMOUNT a pending corp in an emergency, the turn's
    revenue waits only at DIVIDEND (or a private's sub-phase that returns
    there), and every OR field is clear outside the OR. At ER_BONUS, ER's
    seller acts."""
    step, corp = TURN.step(state), TURN.operating_corp(state)
    order = [TURN.or_order(state, i) for i in range(D.NUM_CORPS)]
    corps = [c for c in order if c >= 0]
    assert order == corps + [-1] * (D.NUM_CORPS - len(corps)) and len(set(corps)) == len(corps), f'{msg}: {order}'
    assert (TURN.pending_hex(state) >= 0) == (step == K.STEP_TILE), f'{msg}: pending hex'
    source, train = TURN.pending_train_source(state), TURN.pending_train_type(state)
    assert (source >= 0) == (step in (K.STEP_SELECT_TRAIN, K.STEP_TRAIN_PRICE)), f'{msg}: pending source'
    assert (train >= 0) == (step == K.STEP_TRAIN_PRICE), f'{msg}: pending train'
    seller = source - K.TRAIN_SOURCE_CORP
    assert train < 0 or (seller >= 0 and TRAINS.corp_trains(state, seller, train)), f'{msg}: unheld train'
    if TURN.round(state) == K.ROUND_OPERATING:
        assert (TURN.pending_corp(state) >= 0) == (step == K.STEP_EMERGENCY_SELL_AMOUNT), f'{msg}: pending corp'
        assert step != K.STEP_EMERGENCY_SELL_AMOUNT or A.in_emergency_py(state), f'{msg}: a sale outside an emergency'
    assert TURN.route_revenue(state) == 0 or K.STEP_DIVIDEND in (step, TURN.return_step(state)), \
        f'{msg}: route revenue outside DIVIDEND'
    sold, sold_any = TURN.ebuy_sold_mask(state), TURN.ebuy_sold_any(state)
    assert not sold or (corp >= 0 and step in EBUY_STEPS), f'{msg}: emergency sales {sold:#x} at step {step}'
    if corp >= 0:  # a sale's flag stands until the turn's purchase
        assert sold_any == (sold != 0 and TRAINS.corp_total(state, corp) == 0), f'{msg}: ebuy_sold_any {sold_any}'
    else:
        assert not sold_any, f'{msg}: ebuy_sold_any between turns'
    if TURN.round(state) == K.ROUND_GAME_OVER:  # Ruby keeps the last round's fields
        assert corp < 0 or corp == order[TURN.or_index(state)], f'{msg}: operating corp {corp}, order {order}'
        return
    if TURN.round(state) != K.ROUND_OPERATING:
        assert corps == [] and corp == -1 and TURN.or_index(state) == 0, f'{msg}: OR fields outside the OR'
        return
    assert step in OR_STEPS, f'{msg}: step {step} in the OR'
    assert all(CORPS[c].floated(state) for c in corps), f'{msg}: an unfloated corp in {corps}'
    if step in TURN_STEPS or step == K.STEP_TURN_END:
        assert corp == order[TURN.or_index(state)] >= 0, f'{msg}: operating corp {corp}, order {order}'
        if step == K.STEP_ER_BONUS:  # ER's seller lays
            assert TURN.active_player(state) == TURN.pending_seller(state), f'{msg}: ER\'s seller acts'
            assert PRIVATES[D.PRIVATE_ER].owner(state) == K.OWNER_CORP + corp, f'{msg}: ER_BONUS without a sale'
        else:
            actor = TRAINS.first_crowded(state) if step == K.STEP_DISCARD else corp
            assert TURN.active_player(state) == CORPS[actor].president(state), f'{msg}: the president acts'
        assert MAP.tokens(state, D.CORP_HOME[corp]) >> corp & 1, f'{msg}: no home token'
    else:
        assert corp == -1, f'{msg}: operating corp {corp} between turns'


def assert_action_sets(state, msg=''):
    """Policy ids are legal, and present whenever legal ids are."""
    legal, policy = legal_ids(state), policy_ids(state)
    assert set(policy) <= set(legal) and (policy or not legal), f'{msg}: legal {legal}, policy {policy}'


def is_choice(state):
    """Whether the current decision is one the driver returns rather than
    applies: two or more policy actions, or an amount decision whose one legal
    grid point has legal exact amounts above it (engine-design §6)."""
    policy = get_policy_actions(state)
    if len(policy) >= 2:
        return True
    phase = A.get_decision_phase_py(state)
    if not A.has_exact_amounts(phase) or len(legal_ids(state)) != 1 or policy[0][1].type != A.ACTION_PRICE_POINT:
        return False
    return A.is_legal_action_py(state, exact_id(state, price_of(state, policy[0][1].count) + 1))


def price_of(state, point):
    """The amount of BID, TRAIN_PRICE, or PRIVATE_PRICE grid point ``point``."""
    first = K.BID_POINT if A.get_decision_phase_py(state) == K.DPHASE_BID else 0
    return A.price_amount_py(state, first + point)


# =============================================================================
# APPLY + VERIFY
# =============================================================================

class ApplyResult:
    """``history`` holds ``(row, decision phase, id)`` before each applied
    decision and ``(row, -1, step)`` before each automated step."""

    def __init__(self, state, history, status):
        self.state, self.history, self.status = state, history, status

    @property
    def decisions(self):
        return [(phase, action_id) for _, phase, action_id in self.history if phase >= 0]

    @property
    def steps(self):
        """The automated steps the driver ran."""
        return [step for _, phase, step in self.history if phase < 0]

    @property
    def forced(self):
        """The decisions the driver applied after the given one."""
        return self.decisions[1:]

    def state_at(self, index):
        return GameState.from_array(self.history[index][0], TURN.num_players(self.state),
                                    max_players=self.state.max_players)

    def state_before(self, step):
        """The state as automated step ``step`` first began, so the one before
        OR_START is the SR's end."""
        return self.state_at(next(i for i, (_, phase, s) in enumerate(self.history) if phase < 0 and s == step))

    def state_before_decision(self, phase, action_id):
        """The state as decision ``(phase, action_id)`` was first applied."""
        return self.state_at(next(i for i, (_, p, a) in enumerate(self.history) if (p, a) == (phase, action_id)))


def _verify(result, msg):
    states = [result.state_at(i) for i in range(len(result.history))] + [result.state]
    for i in range(len(result.history)):
        before = states[i]
        assert_invariants(before, f'{msg}: before entry {i} {result.history[i][1:]}')
        _assert_private_moves(before, states[i + 1], f'{msg}: entry {i}')
        _assert_sr_turns(before, states[i + 1], result.history[i][1:], f'{msg}: entry {i}')
        if result.history[i][1] >= 0:
            assert_action_sets(before, f'{msg}: before decision {i}')
    assert_invariants(result.state, f'{msg}: after')
    if TURN.round(result.state) != K.ROUND_GAME_OVER:
        assert_action_sets(result.state, f'{msg}: after')
        assert is_choice(result.state), f'{msg}: returned a forced decision {get_policy_actions(result.state)}'
    return result


def apply_and_verify(state, action_id, msg='', expected_status=STATUS_OK):
    """Apply a legal id (enumerated, or an exact amount in range) through the
    driver and check every state it passed through. ``expected_status`` None
    accepts the game's end as well as a decision."""
    assert A.is_legal_action_py(state, action_id), f'{msg}: id {action_id} is not legal; legal {legal_ids(state)}'
    history = []
    status = DRIVER.apply_action(state, action_id, history)
    expected = (STATUS_OK, STATUS_GAME_OVER) if expected_status is None else (expected_status,)
    assert status in expected, f'{msg}: status {status}, expected {expected_status}'
    return _verify(ApplyResult(state, history, status), msg)
