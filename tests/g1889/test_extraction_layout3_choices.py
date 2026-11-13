"""1889 input layout 3's choice table (1889/v2-design.md §5; token-data-1889.md,
"The choice table").

On the decisions of ``tests/g1889/test_extraction_layout3``'s games (every
``STRIDE``-th, and every one at a rare decision), for every row of every
decision phase, 3-5 players at every capacity:

- **applied rows:** the shared block and the phase block equal what the
  state ``apply_alone`` leaves holds, read through the entity, market, map,
  graph, and solver APIs, with the route maxes solved afresh on copies;
- **previews:** a row that opens a sub-decision equals the sub-decision's
  row it names, reached by playing the action; TRACK's best is the best of
  the TILE rows its hex opens, and a lay the driver plays by itself (one
  candidate) gives TRACK's values;
- **zeros:** illegal rows, rows past a phase's size, and the table without a
  decision; and every slot is exercised but build distance's.

Purity, batches, and capacities over the whole vector are
``tests/g1889/test_extraction_layout3.py``'s.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID, apply_alone
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue, route_hypotheticals_py
from g1889.token_data import ChoiceTable3

from tests.g1889.test_extraction_layout3 import Sample, _states, copy_of, hex_slots, split, inputs_of

ROWS, WIDTH = int(ChoiceTable3.CHOICE3_ROWS), int(ChoiceTable3.CHOICE3_WIDTH)
SHARED, PHASE_BLOCK = int(ChoiceTable3.CHOICE3_SHARED), int(ChoiceTable3.CHOICE3_PHASE)
STRIDE = 3
RARE = {K.DPHASE_PAR, K.DPHASE_SELL_AMOUNT, K.DPHASE_DISCARD, K.DPHASE_SELECT_TRAIN, K.DPHASE_TRAIN_PRICE,
        K.DPHASE_EMERGENCY_SELL_AMOUNT, K.DPHASE_BUY_PRIVATE_POST, K.DPHASE_PRIVATE_PRICE, K.DPHASE_ER_BONUS,
        K.DPHASE_PORT, K.DPHASE_TOKEN, K.DPHASE_TILE}
# Where S4 and S5 are solved; 0 by construction elsewhere.
ROUTE_PHASES = {K.DPHASE_TILE, K.DPHASE_ER_BONUS, K.DPHASE_TOKEN, K.DPHASE_DISCARD, K.DPHASE_SELECT_TRAIN,
                K.DPHASE_TRAIN_PRICE, K.DPHASE_PORT}
# Each phase block and its decisions; build distance's slots stay zero.
BLOCKS = {
    'AUCTION and BID': ((K.DPHASE_AUCTION, K.DPHASE_BID), 4, ()),
    'STOCK': ((K.DPHASE_STOCK,), 9, ()),
    'PAR': ((K.DPHASE_PAR,), 3, ()),
    'SELL_AMOUNT and EMERGENCY_SELL_AMOUNT': ((K.DPHASE_SELL_AMOUNT, K.DPHASE_EMERGENCY_SELL_AMOUNT), 4, ()),
    'BUY_PRIVATE': ((K.DPHASE_BUY_PRIVATE_PRE, K.DPHASE_BUY_PRIVATE_POST, K.DPHASE_PRIVATE_PRICE), 5, ()),
    'TRACK': ((K.DPHASE_TRACK,), 30, range(26, 30)),
    'TILE and ER_BONUS': ((K.DPHASE_TILE, K.DPHASE_ER_BONUS), 37, range(28, 37)),
    'TOKEN': ((K.DPHASE_TOKEN,), 14, range(5, 14)),
    'DIVIDEND': ((K.DPHASE_DIVIDEND,), 7, ()),
    'TRAIN': ((K.DPHASE_BUY_TRAIN, K.DPHASE_SELECT_TRAIN, K.DPHASE_TRAIN_PRICE), 16, ()),
    'DISCARD and PORT': ((K.DPHASE_DISCARD, K.DPHASE_PORT), 0, ()),
}
SUB_DECISIONS = {  # rows that open one: their shared block is zero
    K.DPHASE_AUCTION: {A.ACTION_SELECT_BID}, K.DPHASE_STOCK: {A.ACTION_SELECT_PAR, A.ACTION_SELECT_SALE,
                                                               A.ACTION_MF_PORT},
    K.DPHASE_BUY_PRIVATE_PRE: {A.ACTION_BUY_PRIVATE, A.ACTION_MF_PORT}, K.DPHASE_BUY_PRIVATE_POST: {A.ACTION_BUY_PRIVATE},
    K.DPHASE_TRACK: {A.ACTION_SELECT_HEX},
    K.DPHASE_BUY_TRAIN: {A.ACTION_BUY_DEPOT_TRAIN, A.ACTION_BUY_POOL_TRAIN, A.ACTION_BUY_CORP_TRAIN,
                         A.ACTION_SELECT_SALE},
}


def close(a, b):
    return bool(np.isclose(a, b, rtol=2e-6, atol=1e-6))


def legal(state):
    ids = np.empty(K.MAX_ACTION_SIZE, np.uint16)
    return ids[:A.enumerate_legal_actions_py(state, ids)].tolist()


def table_of(state):
    return split(inputs_of(state), state.max_players)['CHOICES']


def alone(state, action):
    """A copy of ``state`` with ``action`` applied alone."""
    after = copy_of(state)
    apply_alone(after, action)
    return after


def route(state, corp):
    return max_route_revenue(copy_of(state, stale=True), corp)


def with_next(state, corp):
    return route_hypotheticals_py(copy_of(state, stale=True), corp)[1]


def reach(state, corp):
    """The corp's hexes reached, centers reached, and their revenue now."""
    fresh = copy_of(state, stale=True)
    hexes = centers = revenue = 0
    for h in range(D.NUM_HEXES):
        hexes += GRAPH.sides_reached(fresh, corp, h) != 0
        if GRAPH.node_reached(fresh, corp, h):
            centers += 1
            revenue += MAP.revenue(fresh, h, PHASE.current(fresh))
    return hexes, centers, revenue


def to_float(state, corp):
    floated = CORPS[corp].floated(state)
    return 0.0 if floated else max(CORPS[corp].ipo_shares(state) - 5, 0) / 5


def presidency(state, after, corp, player):
    return float(CORPS[corp].president(after) == player) - float(CORPS[corp].president(state) == player)


def row_corp(state, phase, info):
    """token-data-1889.md's "row's corp"."""
    if phase in (K.DPHASE_AUCTION, K.DPHASE_BID):
        return -1
    if info.type == A.ACTION_DR_EXCHANGE:
        return D.DR_EXCHANGE_CORP
    if phase == K.DPHASE_STOCK:
        return info.corp
    if phase in (K.DPHASE_PAR, K.DPHASE_SELL_AMOUNT, K.DPHASE_EMERGENCY_SELL_AMOUNT):
        return TURN.pending_corp(state)
    if phase == K.DPHASE_DISCARD:
        return TRAINS.first_crowded(state)
    return TURN.operating_corp(state) if TURN.round(state) == K.ROUND_OPERATING else -1


def shared_block(state, after, actor, corp, routes):
    """S0-S5 of an applied row: ``after`` less ``state``."""
    s = np.zeros(7)
    if actor >= 0:
        s[0] = (PLAYERS[actor].cash(after) - PLAYERS[actor].cash(state)) / 1000
        s[3] = (PLAYERS[actor].cert_count(after) - PLAYERS[actor].cert_count(state)) / D.CERT_LIMIT[TURN.num_players(state)]
    if corp >= 0:
        s[1] = (CORPS[corp].cash(after) - CORPS[corp].cash(state)) / 1000
        if CORPS[corp].par_index(state) >= 0 and CORPS[corp].par_index(after) >= 0:
            s[2] = (MARKET.price(after, corp) - MARKET.price(state, corp)) / 100
        if routes:
            s[4] = (route(after, corp) - route(state, corp)) / 100
            s[5] = (with_next(after, corp) - with_next(state, corp)) / 100
    return s


def private_block(state, corp, p, price):
    face = D.PRIVATE_VALUE[p]
    return [price / 100, (face + 1) // 2 / 100, face / 100, PRIVATES[p].revenue(state) / 100,
            (CORPS[corp].cash(state) - price) / 1000]


class Expected:
    """Row k of ``state``'s choice table, from the state and the action played."""

    def __init__(self, state, k, seen):
        self.state, self.k, self.seen = state, k, seen
        self.phase = A.get_decision_phase_py(state)
        self.info = A.decode_action_py(self.phase, k)
        self.actor = TURN.active_player(state)
        self.op = TURN.operating_corp(state)
        self.row = np.zeros(WIDTH)
        self.shared = self.row[SHARED:PHASE_BLOCK]
        self.block = self.row[PHASE_BLOCK:]

    def applied(self, corp=None):
        corp = row_corp(self.state, self.phase, self.info) if corp is None else corp
        after = alone(self.state, self.k)
        routes = self.phase in ROUTE_PHASES and corp >= 0
        self.shared[:] = shared_block(self.state, after, self.actor, corp, routes)
        return after

    def compute(self):
        getattr(self, f'_{K.DecisionPhase(self.phase).name.removeprefix("DPHASE_").lower()}')()
        return self.row

    def _auction(self):
        state, info = self.state, self.info
        if info.type == A.ACTION_SELECT_BID:  # BID's minimum point, played
            p = info.private
            opened = alone(state, self.k)
            minimum = table_of(opened)[K.BID_POINT]
            assert A.decode_action_py(K.DPHASE_BID, K.BID_POINT).count == 0
            self.block[:4] = [PRIVATES[p].price(state) / 100, *minimum[PHASE_BLOCK + 1:PHASE_BLOCK + 4]]
            assert close(minimum[PHASE_BLOCK + 1], PRIVATES[p].min_bid(state) / 100)
            self.seen.add('auction preview')
            return
        self.applied()
        if info.type == A.ACTION_BUY_TOPMOST:
            top = topmost(state)
            price = PRIVATES[top].price(state)
            self.shared[6] = price / 1000
            self.block[0] = price / 100
            self.block[3] = (PLAYERS[self.actor].available(state, top) - price) / 1000

    def _bid(self):
        state = self.state
        self.applied()
        if self.info.type != A.ACTION_PASS:
            p = TURN.auction_private(state) if TURN.auction_private(state) >= 0 else TURN.pending_private(state)
            amount = A.price_amount_py(state, self.k)
            self.shared[6] = amount / 1000
            self.block[:4] = [amount / 100, PRIVATES[p].min_bid(state) / 100, PRIVATES[p].high_bid(state) / 100,
                              (PLAYERS[self.actor].available(state, p) - amount) / 1000]

    def _stock(self):
        state, info, c = self.state, self.info, self.info.corp
        if info.type == A.ACTION_MF_PORT:
            return
        if info.type == A.ACTION_SELECT_PAR:  # PAR, played at its first par
            opened = alone(state, self.k)
            parred = alone(opened, legal(opened)[0])
            self.block[2] = 1
            self.block[4] = presidency(state, parred, c, self.actor)
            self.block[6] = to_float(parred, c)
            return
        if info.type == A.ACTION_SELECT_SALE:  # SELL_AMOUNT's largest block, played
            opened = alone(state, self.k)
            blocks = legal(opened)
            sold = alone(opened, blocks[-1])
            largest = table_of(opened)[blocks[-1]]
            self.block[3] = 1
            self.block[4] = presidency(state, sold, c, self.actor)
            self.block[6] = to_float(state, c)
            self.block[7] = len(blocks) / 5
            self.block[8] = MARKET.price(sold, c) / 100
            assert close(self.block[8], largest[PHASE_BLOCK + 1]) and close(self.block[4], largest[PHASE_BLOCK + 2])
            self.seen.add('sale preview')
            return
        after = self.applied()
        if info.type == A.ACTION_PASS:
            return
        if info.type == A.ACTION_DR_EXCHANGE:
            c = D.DR_EXCHANGE_CORP
        else:
            kind = 0 if info.type == A.ACTION_BUY_IPO else 1
            self.block[kind] = 1
            self.shared[6] = (CORPS[c].par_price(state) if kind == 0 else MARKET.price(state, c)) / 1000
        self.block[4] = presidency(state, after, c, self.actor)
        self.block[5] = CORPS[c].floated(after) and not CORPS[c].floated(state)
        self.block[6] = to_float(after, c)

    def _par(self):
        state, c = self.state, TURN.pending_corp(self.state)
        par = D.PAR_PRICE[self.k]
        self.applied()
        self.shared[6] = 2 * par / 1000
        out = 10 - CORPS[c].ipo_shares(state)
        self.block[:3] = [par / 100, min(out + PLAYERS[self.actor].cash(state) // par, 5) / 5, 10 * par / 1000]

    def _sell_amount(self):
        state, c = self.state, TURN.pending_corp(self.state)
        after = self.applied()
        shares = self.k + 1
        self.block[0], self.block[1] = shares / 5, MARKET.price(after, c) / 100
        if self.phase == K.DPHASE_SELL_AMOUNT:
            self.block[2] = presidency(state, after, c, self.actor)
        else:
            need = (TRAINS.cheapest_price(state) - CORPS[self.op].cash(state)
                    - PLAYERS[CORPS[self.op].president(state)].cash(state))
            self.block[3] = max(need - shares * MARKET.price(state, c), 0) / 1000

    _emergency_sell_amount = _sell_amount

    def _buy_private_pre(self):
        state, info = self.state, self.info
        if info.type == A.ACTION_BUY_PRIVATE:  # PRIVATE_PRICE's last point, played
            opened = alone(state, self.k)
            last = legal(opened)[-1]
            self.block[:5] = private_block(state, self.op, info.private, A.price_amount_py(opened, last))
            np.testing.assert_allclose(table_of(opened)[last, PHASE_BLOCK:PHASE_BLOCK + 5], self.block[:5], rtol=2e-6)
            self.seen.add('private preview')
        elif info.type != A.ACTION_MF_PORT:
            self.applied()

    _buy_private_post = _buy_private_pre

    def _private_price(self):
        amount = A.price_amount_py(self.state, self.k)
        self.applied()
        self.shared[6] = amount / 1000
        self.block[:5] = private_block(self.state, self.op, TURN.pending_private(self.state), amount)

    def _track(self):
        state, op = self.state, self.op
        if self.info.type == A.ACTION_PASS:
            self.applied()
            return
        h = self.info.hex
        self.row[:3] = [h + 1, MAP.tile(state, h) + 1, MAP.rotation(state, h)]
        fresh = copy_of(state, stale=True)
        sides = GRAPH.sides_reached(fresh, op, h)
        self.block[:8] = hex_slots(state, h, PHASE.current(state))
        self.block[8:14] = [sides >> s & 1 for s in range(D.NUM_SIDES)]
        self.block[14] = GRAPH.node_reached(fresh, op, h)
        self.block[15] = MAP.blocked_for(state, h, op)
        candidates = A.lay_candidates_py(state, h)
        self.block[16] = len(candidates) / 15
        self.block[17] = MAP.lay_cost(state, h, op) / 100
        # The best of the TILE rows the hex opens, played.
        opened = alone(state, self.k)
        tile_rows = table_of(opened)[legal(opened)]
        assert len(tile_rows) == len(candidates)
        self.block[18:20] = tile_rows[:, SHARED + 4:SHARED + 6].max(axis=0)
        self.block[20:24] = tile_rows[:, PHASE_BLOCK + 22:PHASE_BLOCK + 26].max(axis=0)
        self.block[24:26] = tile_rows[:, PHASE_BLOCK + 26:PHASE_BLOCK + 28].min(axis=0)
        if len(candidates) == 1:  # the driver lays it with no TILE decision
            laid, history = copy_of(state), []
            assert DRIVER.apply_action(laid, self.k, history) != STATUS_INVALID
            assert history[1][1:] == (K.DPHASE_TILE, 0)
            if len(history) > 2:
                laid = copy_of(type(state).from_array(history[2][0], TURN.num_players(state), state.max_players))
            assert close(self.block[18], (route(laid, op) - route(state, op)) / 100)
            assert close(self.block[19], (with_next(laid, op) - with_next(state, op)) / 100)
            assert close(self.block[21], (reach(laid, op)[1] - reach(state, op)[1]) / 25)
            self.seen.add('forced lay')
        self.seen.add('track best')

    def _lays(self, hex_, candidate, cost):
        state, op = self.state, self.op
        tile, rotation = candidate
        after = self.applied()
        self.row[:3] = [hex_ + 1, tile + 1, rotation]
        self.shared[6] = cost / 1000
        added = D.TILE_MASK[tile][rotation] & ~D.TILE_MASK[MAP.tile(state, hex_)][MAP.rotation(state, hex_)]
        self.block[:21] = [added >> i & 1 for i in range(D.NUM_PAIRS)]
        assert MAP.tile(after, hex_) == tile
        self.block[21] = MAP.supply(after, tile) / D.TILE_COUNT[tile]
        before, now = reach(state, op), reach(after, op)
        self.block[22:25] = [(now[0] - before[0]) / 52, (now[1] - before[1]) / 25, (now[2] - before[2]) / 100]
        self.block[25] = GRAPH.route_available(copy_of(after, stale=True), op)
        gains = [reach(after, c)[1] - reach(state, c)[1] for c in range(D.NUM_CORPS) if c != op]
        self.block[26] = sum(g > 0 for g in gains) / 6
        self.block[27] = sum(g for g in gains if g > 0) / 50

    def _tile(self):
        hex_ = TURN.pending_hex(self.state)
        self._lays(hex_, A.lay_candidates_py(self.state, hex_)[self.k], MAP.lay_cost(self.state, hex_, self.op))

    def _er_bonus(self):
        if self.info.type == A.ACTION_PASS:
            self.applied()
        else:
            self._lays(D.ER_HEX, A.er_candidates_py(self.state)[self.k - K.ER_BONUS_CANDIDATE], 0)

    def _token(self):
        state, op = self.state, self.op
        if self.info.type == A.ACTION_PASS:
            self.applied()
            return
        h = self.info.hex
        after = self.applied()
        self.row[0] = h + 1
        self.shared[6] = CORPS[op].token_price(state) / 1000
        lost = [route(state, c) - route(after, c) for c in range(D.NUM_CORPS) if c != op]
        next_lost = [with_next(state, c) - with_next(after, c) for c in range(D.NUM_CORPS) if c != op]
        assert min(lost) >= 0 and min(next_lost) >= 0
        self.block[:4] = [sum(lost) / 100, max(lost) / 100, sum(next_lost) / 100, max(next_lost) / 100]
        self.block[4] = MAP.free_slots(after, h) / 3
        if max(lost) or max(next_lost):
            self.seen.add('token blocks')

    def _dividend(self):
        state, c = self.state, self.op
        self.applied()
        revenue, n = TURN.route_revenue(state), TURN.num_players(state)
        payout = self.info.type == A.ACTION_PAYOUT

        def paid(shares):
            return (shares * revenue + 9) // 10

        pool = paid(CORPS[c].pool_shares(state))
        if payout:
            self.block[0] = paid(1) / 100
            self.block[1] = (pool + sum(paid(PLAYERS[i].shares(state, c)) for i in range(n))) / 1000
            for j in range(1, n):
                self.block[2 + j] = paid(PLAYERS[(self.actor + j) % n].shares(state, c)) / 100
        next_type = TRAINS.cheapest_type(state)
        if TRAINS.corp_total(state, c) < TRAINS.limit(state) and next_type >= 0:
            self.block[2] = max(D.TRAIN_PRICE[next_type] - CORPS[c].cash(state) - (pool if payout else revenue), 0) / 1000

    def _discard(self):
        self.applied()

    def _buy_train(self):
        state, info, k = self.state, self.info, self.k
        if info.type in (A.ACTION_PASS, A.ACTION_BANKRUPT):
            self.applied()
        elif info.type == A.ACTION_SELECT_SALE:  # EMERGENCY_SELL_AMOUNT's largest block, played
            opened = alone(state, k)
            blocks = legal(opened)
            largest = table_of(opened)[blocks[-1]]
            self.block[13:16] = [len(blocks) / 5, largest[PHASE_BLOCK + 1], largest[PHASE_BLOCK + 3]]
            self.seen.add('emergency sale preview')
        else:  # SELECT_TRAIN's row with the largest gain, played
            opened = alone(state, k)
            choices = table_of(opened)
            ids = legal(opened)
            best = max(ids, key=lambda i: (choices[i, PHASE_BLOCK + 7], -i))
            self.block[:13] = choices[best, PHASE_BLOCK:PHASE_BLOCK + 13]
            if self.block[12]:  # an emergency's purchase: the president pays the rest, played
                bought = alone(opened, best)
                assert close(self.block[12], (PLAYERS[self.actor].cash(state) - PLAYERS[self.actor].cash(bought)) / 1000)
                self.seen.add('contribution')
            self.seen.add('source preview')

    def _select_train(self):
        state, info, op = self.state, self.info, self.op
        source = TURN.pending_train_source(state)
        traded = info.train_type if info.type == A.ACTION_TRADE_IN else -1
        bought = D.D_TRAIN if traded >= 0 else info.train_type
        self.block[bought] = 1
        if source >= K.TRAIN_SOURCE_CORP:  # TRAIN_PRICE, played
            opened = alone(state, self.k)
            priced = table_of(opened)[legal(opened)]
            assert (priced[:, PHASE_BLOCK + 7] == priced[0, PHASE_BLOCK + 7]).all()
            self.block[7], self.block[11] = priced[0, PHASE_BLOCK + 7], priced[0, PHASE_BLOCK + 11]
            self.seen.add('corp train preview')
            return
        after = self.applied()
        price = D.TRAIN_PRICE[bought] if traded < 0 else D.TRAIN_PRICE[D.D_TRAIN] - D.TRAIN_TRADE_IN[D.D_TRAIN][traded]
        self.shared[6] = self.block[6] = price / 1000
        self.block[7] = self.shared[4]
        starts = PHASE.current(after) > PHASE.current(state)
        self.block[8] = starts
        if starts:
            before = [[TRAINS.corp_trains(state, c, t) for t in range(6)] for c in range(D.NUM_CORPS)]
            now = [[TRAINS.corp_trains(after, c, t) for t in range(6)] for c in range(D.NUM_CORPS)]
            self.block[9] = (sum(before[op][t] - now[op][t] for t in range(6) if t != bought) - (traded >= 0)) / 4
            self.block[10] = sum(before[c][t] - now[c][t] for c in range(D.NUM_CORPS) if c != op for t in range(6)) / 6
            self.seen.add('phase start')
        if traded < 0:  # in an emergency, where SELECT_TRAIN is forced; BUY_TRAIN's previews show it
            self.block[12] = max(price - CORPS[op].cash(state), 0) / 1000

    def _train_price(self):
        state, op = self.state, self.op
        seller = TURN.pending_train_source(state) - K.TRAIN_SOURCE_CORP
        amount = A.price_amount_py(state, self.k)
        after = self.applied()
        assert PHASE.current(after) == PHASE.current(state)
        self.shared[6] = self.block[6] = amount / 1000
        self.block[TURN.pending_train_type(state)] = 1
        self.block[7] = self.shared[4]
        self.block[11] = (route(after, seller) - route(state, seller)) / 100

    def _port(self):
        self.applied()
        i = self.k - K.PORT_LAY
        self.row[:3] = [D.PORT_LAY_HEX[i] + 1, D.PORT_TILE + 1, D.PORT_LAY_ROT[i]]


@pytest.fixture(scope='module')
def decisions():
    """The rows test's states at a decision: every ``STRIDE``-th, and every
    one at a rare decision, with their tables."""
    out, i = [], 0
    for kind, state in _states():
        phase = A.get_decision_phase_py(state)
        if phase < 0:
            assert not Sample(kind, state).parts['CHOICES'].any()  # no decision, no table
            continue
        i += 1
        if i % STRIDE == 0 or phase in RARE:
            out.append(Sample(kind, state))
    return out


def test_every_row_is_its_actions_outcome(decisions):
    seen, phases = set(), set()
    for s in decisions:
        table, ids = s.parts['CHOICES'], legal(s.state)
        size = K.PHASE_ACTION_SIZES[s.phase]
        assert not table[[k for k in range(ROWS) if k not in ids]].any(), 'an illegal row or one past the size'
        assert all(k < size for k in ids)
        for k in ids:
            want = Expected(s.state, k, seen).compute()
            np.testing.assert_allclose(table[k], want, rtol=2e-6, atol=1e-6,
                                       err_msg=f'{s.kind} {K.DecisionPhase(s.phase).name} row {k}')
            info = A.decode_action_py(s.phase, k)
            if info.type in SUB_DECISIONS.get(s.phase, ()) or (
                    s.phase == K.DPHASE_SELECT_TRAIN and TURN.pending_train_source(s.state) >= K.TRAIN_SOURCE_CORP):
                assert not table[k, SHARED:PHASE_BLOCK].any(), 'a sub-decision has no shared block'
            if s.phase not in ROUTE_PHASES:
                assert not table[k, SHARED + 4:SHARED + 6].any()
        phases.add((s.phase, s.num_players, s.max_players))
    assert {phase for phase, _, _ in phases} == set(range(K.NUM_DECISION_PHASES))
    assert {(n, p) for _, n, p in phases} == {(3, 3), (3, 4), (3, 5), (4, 4), (4, 5), (5, 5)}
    assert seen >= {'auction preview', 'sale preview', 'private preview', 'track best', 'forced lay',
                    'emergency sale preview', 'source preview', 'corp train preview', 'phase start', 'contribution'}, seen


def test_every_slot_is_exercised(decisions):
    """Every column is nonzero somewhere in each phase block's decisions, but
    build distance's; the shared block's and the integers in some phase."""
    nonzero = {name: np.zeros(width, bool) for name, (_, width, _) in BLOCKS.items()}
    shared = np.zeros(PHASE_BLOCK, bool)
    for s in decisions:
        table = s.parts['CHOICES']
        shared |= (table[:, :PHASE_BLOCK] != 0).any(axis=0)
        for name, (phases, width, _) in BLOCKS.items():
            if s.phase in phases:
                nonzero[name] |= (table[:, PHASE_BLOCK:PHASE_BLOCK + width] != 0).any(axis=0)
                assert not table[:, PHASE_BLOCK + width:].any(), f'{name}: past its width'
    assert shared.all(), np.flatnonzero(~shared)
    for name, (_, width, zero) in BLOCKS.items():
        never = set(np.flatnonzero(~nonzero[name]).tolist())
        assert never == set(zero), (name, sorted(never))
