"""1889 input layout 3's rows, pair blocks, and packed writer (1889/v2-design.md
§4, §7.1; token-data-1889.md, "Layout 3").

On the states of whole games (the layout-1 fixture's self-play, human, and
random games, the random playouts of ``tests/g1889/test_cache_purity``, and
playouts at every player count and capacity from 3 to 5):

- **the entity facts:** every slot of GLOBAL, CORP, PRIVATE, CENTER, and
  PLAYER and every pair-block channel equals ``expected``, a reading of the
  state through the entity, map, market, and solver APIs written from
  token-data-1889.md's tables, not from the fill's helpers;
- **hidden and zero:** privates out of play or closed, padded seats, and
  build distance's slots and channels are zero;
- **capacity:** the same game at a larger capacity gives the same values,
  its padded seats zero;
- **purity:** the vector is the same whatever the row's caches hold, and the
  fill changes nothing in the row but its caches, which it leaves fresh;
- **exports:** the parts chain to ``get_input_size``, and the model's static
  tables (the tile table, choice slots, named rows) follow the engine's.

``tests/g1889/test_extraction_layout3_choices.py`` checks the choice table,
and ``tests/g1889/test_token_data_doc_layout3.py`` pins the slot tables.
"""

import math

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, PrivateFlag, topmost
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue, route_hexes_py, route_hypotheticals_py
from g1889.state import GameState, get_global_fields, get_layout
from g1889.token_data import (
    InputOffset3,
    NamedRow3,
    PairChannels3,
    TileBlock,
    TokenWidth3,
    TrunkRow3,
    fill_inputs,
    fill_inputs_batch,
    get_choice_slots,
    get_input_parts,
    get_input_size,
    get_named_rows,
    get_num_trunk_rows,
    get_tile_table,
    get_token_data,
    get_token_dim,
    get_num_tokens,
    TokenRow,
)
from games.g1889 import POLICY_LAYOUT

from tests.g1889.test_cache_purity import _random_calls, games
from tests.g1889.test_layout1_parity import apply_op, new_state

LAYOUT = 3
GLOBALS = get_global_fields()
GLOBAL_OFFSET = get_layout(0).global_offset
DIRTY = [GLOBAL_OFFSET + getattr(GLOBALS, name) for name in GLOBALS._fields if name.endswith('_dirty')]
STRIDE = 4  # the states of the fixture's games and the purity playouts that the tests read
WIDTHS = {'GLOBAL': 135, 'CORP': 76, 'PRIVATE': 15, 'CENTER': 21, 'PLAYER': 20}
# (num_players, max_players, seed): one playout for each count at each capacity.
CAPACITY_PLAYOUTS = ((3, 3, 301), (3, 4, 302), (3, 5, 303), (4, 4, 304), (4, 5, 305), (5, 5, 306))
SUPPLY_DIV = max(D.TILE_COUNT[t] for t in range(D.NUM_SUPPLY_TILES))
LOWEST_PAR = min(D.PAR_PRICE)
LAST_PHASE = D.NUM_GAME_PHASES - 1


def state_of(row):
    return GameState.from_buffer(row, int(row[GLOBAL_OFFSET + GLOBALS.num_players]),
                                 max_players=(len(row) - get_layout(0).players_offset) // get_layout(0).player_size)


def copy_of(state, stale=False):
    copy = state_of(state._array.copy())
    if stale:
        copy._array[DIRTY] = (1 << D.NUM_CORPS) - 1  # test setup: drop every cache
    return copy


def inputs_of(state, stale=False):
    """The packed vector of a copy of ``state``, at the state's capacity."""
    out = np.full(get_input_size(state.max_players, layout_version=LAYOUT), np.nan, np.float32)
    fill_inputs(copy_of(state, stale), out, state.max_players, layout_version=LAYOUT)
    return out


def split(vector, max_players):
    """The vector cut into its parts, by name."""
    return {part.name: vector[part.offset:part.offset + math.prod(part.shape)].reshape(part.shape)
            for part in get_input_parts(max_players, layout_version=LAYOUT)}


class Sample:
    def __init__(self, kind, state):
        self.kind = kind
        self.state = copy_of(state)  # caches as the driver left them
        self.max_players = state.max_players
        self.num_players = TURN.num_players(state)
        self.phase = A.get_decision_phase_py(state)
        self.vector = inputs_of(state)
        self.parts = split(self.vector, self.max_players)


def _capacity_games():
    for num_players, max_players, seed in CAPACITY_PLAYOUTS:
        spec = {'kind': 'capacity', 'label': f'{num_players}p@{max_players}', 'num_players': num_players,
                'max_players': max_players, 'seed': seed, 'priority_player': -1, 'flags': [False, False, 1]}
        yield spec, _random_calls(new_state(spec), np.random.default_rng(seed), 0.8)


def _states():
    """(kind, state) for every ``STRIDE``-th rest of the purity games, and
    every rest of the capacity playouts; the state is the live one."""
    sources = [(spec, calls, STRIDE) for _, spec, calls in games()]
    sources += [(spec, calls, 1) for spec, calls in _capacity_games()]
    for spec, calls, stride in sources:
        state = new_state(spec)
        for done in range(len(calls) + 1):
            if done % stride == 0 or done == len(calls):
                yield spec['kind'], state
            if done < len(calls):
                apply_op(state, *calls[done])


@pytest.fixture(scope='module')
def samples():
    return [Sample(kind, state) for kind, state in _states()]


# =============================================================================
# THE ENTITY FACTS
# =============================================================================

def _log(yen):
    return math.log1p(max(yen, 0) / 100) / 5


def _rust_phase(train_type):
    """The game phase whose start rusts ``train_type``, or -1."""
    rusts_on = D.TRAIN_RUSTS_ON[train_type]
    return next((p for p in range(D.NUM_GAME_PHASES) if rusts_on >= 0 and D.PHASE_ON_TRAIN[p] == rusts_on), -1)


def _next_rust_phase(game_phase):
    """The first later phase that rusts a rank, or -1."""
    return next((p for p in range(game_phase + 1, D.NUM_GAME_PHASES)
                 if D.PHASE_ON_TRAIN[p] in [t for t in D.TRAIN_RUSTS_ON if t >= 0]), -1)


def _payout(shares, revenue):
    return (shares * revenue + 9) // 10


def _mask(corps):
    return sum(1 << c for c in corps)


def hex_slots(state, h, game_phase):
    """HEX 86-93 of hex h: its dangling exits by side (track at an open side
    whose hex across has none facing back), and its upgrade targets now and
    in supply whatever the phase, /15."""
    tile, rotation = MAP.tile(state, h), MAP.rotation(state, h)
    slots = np.zeros(8)
    paths = D.TILE_MASK[tile][rotation]
    for s in range(D.NUM_SIDES):
        across = D.HEX_NEIGHBOR[h][s]
        slots[s] = bool(paths & D.ENDPOINT_PAIRS[s] and D.HEX_EXIT_MASK[h] >> s & 1
                        and not (D.TILE_MASK[MAP.tile(state, across)][MAP.rotation(state, across)]
                                 & D.ENDPOINT_PAIRS[(s + 3) % D.NUM_SIDES]))
    now = any_phase = 0
    k = tile * D.NUM_SIDES + rotation
    for i in range(D.UPG_START[k], D.UPG_START[k + 1]):
        target = D.UPG_TILE[i]
        if (D.TILE_BLOCKS_LAY[target] or MAP.supply(state, target) <= 0
                or D.TILE_EXITS[target][D.UPG_ROT[i]] & ~D.HEX_EXIT_MASK[h]):
            continue
        any_phase += 1
        now += D.PHASE_TILE_COLORS[game_phase] >> D.TILE_COLOR[target] & 1
    slots[6], slots[7] = now / 15, any_phase / 15
    return slots


class Facts:
    """What a state holds, read through the entity APIs; caches are read on a
    refreshed copy, so ``state`` is left as it is."""

    def __init__(self, state):
        self.state = state
        self.fresh = copy_of(state)
        self.n = TURN.num_players(state)
        self.phase = A.get_decision_phase_py(state)
        self.round = TURN.round(state)
        self.game_phase = PHASE.current(state)
        self.actor = TURN.active_player(state) if self.phase >= 0 else -1
        if self.phase == K.DPHASE_PAR:
            self.corp = TURN.pending_corp(state)
        elif self.phase == K.DPHASE_DISCARD:
            self.corp = TRAINS.first_crowded(state)
        elif self.phase >= 0 and self.round == K.ROUND_OPERATING:
            self.corp = TURN.operating_corp(state)
        else:
            self.corp = -1
        self.operating = TURN.operating_corp(state)
        self.routes = [max_route_revenue(self.fresh, c) for c in range(D.NUM_CORPS)]
        self.hypotheticals = [route_hypotheticals_py(self.fresh, c) for c in range(D.NUM_CORPS)]
        self.route_hexes = [set(route_hexes_py(self.fresh, c)) for c in range(D.NUM_CORPS)]
        self.emergency = self.corp >= 0 and self.round == K.ROUND_OPERATING and A.in_emergency_py(self.fresh, self.corp)
        self.limit = TRAINS.limit(state)
        self.next_limit = D.PHASE_TRAIN_LIMIT[min(self.game_phase + 1, LAST_PHASE)]
        self.ranks = self._ranks()

    def _ranks(self):
        state = self.state
        ranks = [-1] * D.NUM_CORPS
        if self.round == K.ROUND_OPERATING:
            for i in range(D.NUM_CORPS):
                if TURN.or_order(state, i) >= 0:
                    ranks[TURN.or_order(state, i)] = i
        elif self.round != K.ROUND_GAME_OVER:
            floated = [c for c in range(D.NUM_CORPS) if CORPS[c].floated(state)]
            for i, c in enumerate(MARKET.sort_by_price(state, _mask(floated))):
                ranks[c] = i
            for c in range(D.NUM_CORPS):
                if CORPS[c].par_index(state) >= 0 and c not in floated:
                    would = copy_of(state)
                    CORPS[c].set_floated(would, True)  # test setup: as if it floated now
                    ranks[c] = MARKET.sort_by_price(would, _mask([*floated, c])).index(c)
        return ranks

    def bid_private(self):
        p = TURN.auction_private(self.state)
        return p if p >= 0 else TURN.pending_private(self.state)

    # --- GLOBAL -------------------------------------------------------------

    def global_row(self):
        state, phase, game_phase = self.state, self.phase, self.game_phase
        g = np.zeros(WIDTHS['GLOBAL'])
        g[0] = 1
        if phase >= 0:
            g[1 + phase] = 1
        if self.round != K.ROUND_GAME_OVER:
            g[20 + self.round] = 1
        g[23 + game_phase] = 1
        g[29] = TURN.turn(state) <= 1
        if self.round == K.ROUND_OPERATING:
            g[30 + TURN.or_num(state) - 1] = 1
            g[33] = TURN.or_count(state) / 3
            g[46] = (TURN.or_count(state) - TURN.or_num(state)) / 2
        else:
            g[33] = D.PHASE_ORS[game_phase] / 3
        cash = BANK.cash(state)
        g[34], g[35] = cash / 7000, BANK.broken(state)
        g[36 + self.n - 3] = 1
        if self.round == K.ROUND_AUCTION:
            g[39], g[40] = TURN.auction_pass_payouts(state) / 3, TURN.auction_discount(state) / 20
        if phase in (K.DPHASE_SELECT_TRAIN, K.DPHASE_TRAIN_PRICE):
            g[41 + min(TURN.pending_train_source(state), K.TRAIN_SOURCE_CORP)] = 1
        if phase in (K.DPHASE_PRIVATE_PRICE, K.DPHASE_ER_BONUS, K.DPHASE_PORT):
            g[44] = TURN.return_step(state) == K.STEP_BUY_PRIVATE_PRE
            g[45] = TURN.return_step(state) == K.STEP_BUY_PRIVATE_POST
        drain = sum(self.routes[c] * (10 - CORPS[c].ipo_shares(state)) for c in range(D.NUM_CORPS)
                    if CORPS[c].floated(state))
        if cash > 0:
            g[47] = min(cash / (drain / 10) / 10, 1.0) if drain else 1.0
        g[48], g[49] = self.limit / 4, self.next_limit / 4
        for t in range(D.NUM_SUPPLY_TILES):
            g[50 + t] = MAP.supply(state, t) / SUPPLY_DIV
        types, cheapest = TRAINS.depot_types(state), TRAINS.cheapest_type(state)
        for t in range(D.NUM_TRAIN_TYPES):
            block = 90 + 7 * t
            g[block] = max(TRAINS.depot(state, t), 0) / 6
            g[block + 1] = TRAINS.pool(state, t) / 4
            g[block + 2] = types >> t & 1
            g[block + 3] = cheapest == t
            g[block + 4] = 0 <= _rust_phase(t) <= game_phase
            g[block + 5] = _rust_phase(t) == game_phase + 1
            g[block + 6] = game_phase < LAST_PHASE and D.PHASE_ON_TRAIN[game_phase + 1] == t
        if phase == K.DPHASE_BID:
            g[132] = D.PRIVATE_VALUE[self.bid_private()] / 1000
            g[133] = PLAYERS[self.actor].available(state, self.bid_private()) / 1000
        elif phase == K.DPHASE_PAR:
            g[133] = PLAYERS[self.actor].cash(state) / 1000
        elif phase == K.DPHASE_TRAIN_PRICE:
            g[132] = D.TRAIN_PRICE[TURN.pending_train_type(state)] / 1000
            g[133] = CORPS[self.corp].cash(state) / 1000
        elif phase == K.DPHASE_PRIVATE_PRICE:
            g[132] = D.PRIVATE_VALUE[TURN.pending_private(state)] / 1000
            g[133] = CORPS[self.corp].cash(state) / 1000
        elif phase == K.DPHASE_DIVIDEND:
            g[134] = TURN.route_revenue(state) / 100
        return g[None]

    # --- CORP ---------------------------------------------------------------

    def corp_row(self, c):
        state, fresh, phase, corp = self.state, self.fresh, self.phase, CORPS[c]
        row = np.zeros(WIDTHS['CORP'])
        row[0] = 1
        row[1] = c == self.corp
        if phase in (K.DPHASE_SELECT_TRAIN, K.DPHASE_TRAIN_PRICE):
            row[2] = TURN.pending_train_source(state) == K.TRAIN_SOURCE_CORP + c
        if phase in (K.DPHASE_SELL_AMOUNT, K.DPHASE_EMERGENCY_SELL_AMOUNT):
            row[3] = TURN.pending_corp(state) == c
        parred, floated = corp.par_index(state) >= 0, corp.floated(state)
        row[4], row[5], row[6] = parred, floated, MAP.tokens(state, D.CORP_HOME[c]) >> c & 1
        if parred:
            price = MARKET.price(state, c)
            row[7], row[8] = corp.par_price(state) / 100, price / 100
            row[9], row[10] = MARKET.row(state, c) / 10, MARKET.col(state, c) / 14
            row[11 + MARKET.type(state, c)] = 1
            for slot, move in ((51, MARKET.move_right), (52, MARKET.move_left), (53, MARKET.move_up)):
                moved = copy_of(state)
                move(moved, c)
                row[slot] = MARKET.price(moved, c) / 100
            row[67] = _log(price)
        ipo, pool, cash = corp.ipo_shares(state), corp.pool_shares(state), corp.cash(state)
        row[15] = ipo == 0 and pool == 0
        row[16], row[17], row[18] = cash / 1000, ipo / 10, pool / 10
        row[19], row[20] = corp.tokens_left(state) / 3, corp.token_price(state) / 100
        held = TRAINS.corp_total(state, c)
        for t in range(D.NUM_TRAIN_TYPES):
            row[21 + t] = TRAINS.corp_trains(state, c, t) / 4
        row[27] = held > self.limit
        row[28], row[29] = corp.last_revenue(state) / 100, self.routes[c] / 100
        row[30] = GRAPH.route_available(fresh, c)
        row[31] = A.must_buy_train_py(fresh, c)
        row[32] = c == self.corp and self.emergency
        if self.ranks[c] >= 0:
            row[33 + self.ranks[c]] = 1
        if self.round == K.ROUND_OPERATING:
            row[40] = 0 <= self.ranks[c] < TURN.or_index(state)
        hexes = centers = revenue = slots = 0
        for h in range(D.NUM_HEXES):
            hexes += GRAPH.sides_reached(fresh, c, h) != 0
            if GRAPH.node_reached(fresh, c, h):
                centers += 1
                revenue += MAP.revenue(state, h, self.game_phase)
                slots += MAP.has_slot_for(state, h, c)
        row[41], row[42], row[43], row[44] = hexes / 52, centers / 25, revenue / 100, slots / 15
        if self.actor >= 0:
            row[45] = PLAYERS[self.actor].sold_mask(state) >> c & 1
        row[46] = TURN.sr_sold_mask(state) >> c & 1
        row[47] = TURN.ebuy_sold_mask(state) >> c & 1
        row[48] = TURN.sr_bought(state) == c + 1
        row[49] = sum(PRIVATES[p].owner(state) == K.OWNER_CORP + c for p in range(D.NUM_PRIVATES)) / 7
        row[50] = c == self.operating
        row[54], row[55] = held / self.limit, (self.limit - held) / 4
        rust = _next_rust_phase(self.game_phase)
        rusting = sum(TRAINS.corp_trains(state, c, t) for t in range(D.NUM_TRAIN_TYPES)
                      if rust >= 0 and _rust_phase(t) == rust)
        rusting_next = sum(TRAINS.corp_trains(state, c, t) for t in range(D.NUM_TRAIN_TYPES)
                           if _rust_phase(t) == self.game_phase + 1)
        row[56] = (held - rusting) / 4
        row[57] = (self.next_limit - (held - rusting_next)) / 4
        row[58], row[59] = self.hypotheticals[c][0] / 100, self.hypotheticals[c][1] / 100
        if TRAINS.cheapest_type(state) >= 0:
            row[60] = max(D.TRAIN_PRICE[TRAINS.cheapest_type(state)] - cash, 0) / 1000
        president = corp.president(state)
        to_float = max(ipo - 5, 0)
        if to_float and not floated:
            cost = to_float * (corp.par_price(state) if parred else LOWEST_PAR)
            row[61], row[62] = to_float / 5, cost / 1000
            row[63] = president >= 0 and PLAYERS[president].cash(state) >= cost
            row[64] = self.actor >= 0 and PLAYERS[self.actor].cash(state) >= cost
        row[65] = president >= 0 and PLAYERS[president].sold_mask(state) >> c & 1
        if self.emergency and c == self.corp:
            need = TRAINS.cheapest_price(state) - cash - PLAYERS[president].cash(state)
            row[66] = max(need, 0) / 1000
        row[68], row[69] = _log(cash), _log(self.routes[c])
        return row  # 70-75: build distance, zero until it lands

    # --- PRIVATE ------------------------------------------------------------

    def private_row(self, p):
        state, private = self.state, PRIVATES[p]
        row = np.zeros(WIDTHS['PRIVATE'])
        owner = private.owner(state)
        if owner in (K.OWNER_NOT_IN_PLAY, K.OWNER_CLOSED):
            return row
        row[0] = 1
        row[1] = (self.phase == K.DPHASE_BID and self.bid_private() == p
                  or self.phase == K.DPHASE_PRIVATE_PRICE and TURN.pending_private(state) == p)
        row[2] = topmost(state) == p
        row[3 if owner == K.OWNER_UNSOLD else 4 if owner < K.OWNER_CORP else 5] = 1
        row[6], row[7] = D.PRIVATE_VALUE[p] / 100, private.revenue(state) / 100
        if owner == K.OWNER_UNSOLD:
            row[8], row[9] = private.price(state) / 100, private.high_bid(state) / 100
            row[10], row[11] = private.bid_count(state) / 5, private.min_bid(state) / 100
            if self.actor >= 0:
                row[12] = PLAYERS[self.actor].bid(state, p) / 100
        row[13] = any(private.blocks_hex(state, h) for h in range(D.NUM_HEXES) if D.HEX_BLOCKER[h] == p)
        row[14] = p == D.PRIVATE_MF and bool(private.flags(state) & PrivateFlag.MF_PORT_USED)
        return row

    # --- CENTER -------------------------------------------------------------

    def center_row(self, r):
        state, h = self.state, D.RC_HEX[r]
        tile, rotation = MAP.tile(state, h), MAP.rotation(state, h)
        row = np.zeros(WIDTHS['CENTER'])
        row[0] = 1
        row[1] = (self.phase == K.DPHASE_TILE and TURN.pending_hex(state) == h
                  or self.phase == K.DPHASE_ER_BONUS and h == D.ER_HEX)
        row[2], row[3] = tile + 1, rotation
        row[4] = MAP.revenue(state, h, self.game_phase) / 100
        row[5] = MAP.revenue(state, h, self.game_phase, True) / 100
        terrain = D.TILE_TERRAIN[D.NUM_SUPPLY_TILES + h]
        row[6], row[7] = bool(terrain & D.TERRAIN_MOUNTAIN), bool(terrain & D.TERRAIN_WATER)
        tokens, home = MAP.tokens(state, h), D.HEX_HOME_CORP[h]
        row[8], row[9] = bin(tokens).count('1') / 3, MAP.free_slots(state, h) / 3
        row[10] = home >= 0 and not tokens >> home & 1
        row[11] = D.TILE_COST[tile] / 100
        row[12] = D.HEX_BLOCKER[h] >= 0 and PRIVATES[D.HEX_BLOCKER[h]].blocks_hex(state, h)
        row[13:21] = hex_slots(state, h, self.game_phase)
        return row

    # --- PLAYER -------------------------------------------------------------

    def player_row(self, i):
        state, player = self.state, PLAYERS[i]
        row = np.zeros(WIDTHS['PLAYER'])
        cash = player.cash(state)
        row[0], row[1] = 1, i == self.actor
        if self.actor >= 0:
            row[2 + (i - self.actor) % self.n] = 1
        row[7] = i == TURN.priority_player(state)
        row[8], row[9] = cash / 1000, player.value(state) / 1000
        row[10] = player.cert_count(state) / D.CERT_LIMIT[self.n]
        row[11] = player.passed(state)
        row[12] = sum(CORPS[c].president(state) == i for c in range(D.NUM_CORPS)) / 7
        owned = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_PLAYER + i]
        row[13] = len(owned) / 7
        row[14] = (self.operating >= 0 and CORPS[self.operating].president(state) == i
                   and TURN.ebuy_sold_any(state))
        row[15] = cash >= 5 * LOWEST_PAR
        row[16] = (self.phase == K.DPHASE_BID and TURN.auction_private(state) >= 0
                   and i == TURN.auction_turn(state))
        income = sum(_payout(player.shares(state, c), self.routes[c]) for c in range(D.NUM_CORPS)
                     if CORPS[c].floated(state))
        row[17] = (income + sum(PRIVATES[p].revenue(state) for p in owned)) / 1000
        row[18], row[19] = _log(cash), _log(player.value(state))
        return row

    # --- PAIR BLOCKS --------------------------------------------------------

    def pair_blocks(self, max_players):
        state = self.state
        player_corp = np.zeros((max_players, D.NUM_CORPS, 3))
        player_private = np.zeros((max_players, D.NUM_PRIVATES, 2))
        corp_private = np.zeros((D.NUM_CORPS, D.NUM_PRIVATES, 1))
        corp_center = np.zeros((D.NUM_CORPS, D.NUM_RCS, 4))
        for i in range(self.n):
            for c in range(D.NUM_CORPS):
                player_corp[i, c] = (PLAYERS[i].shares(state, c) / 10, CORPS[c].president(state) == i,
                                     PLAYERS[i].sold_mask(state) >> c & 1)
        for p in range(D.NUM_PRIVATES):
            owner = PRIVATES[p].owner(state)
            if owner in (K.OWNER_NOT_IN_PLAY, K.OWNER_CLOSED):
                continue
            if owner >= K.OWNER_CORP:
                corp_private[owner - K.OWNER_CORP, p, 0] = 1
            for i in range(self.n):
                player_private[i, p] = (owner == K.OWNER_PLAYER + i, PLAYERS[i].bid(state, p) / 100)
        for c in range(D.NUM_CORPS):
            for r in range(D.NUM_RCS):
                h = D.RC_HEX[r]
                corp_center[c, r, :3] = (MAP.tokens(state, h) >> c & 1, GRAPH.node_reached(self.fresh, c, h),
                                         h in self.route_hexes[c])
        return {'PLAYER_CORP': player_corp, 'PLAYER_PRIVATE': player_private, 'CORP_PRIVATE': corp_private,
                'CORP_CENTER': corp_center, 'CENTER_CENTER': np.zeros((D.NUM_RCS, D.NUM_RCS, 1))}

    def parts(self, max_players):
        players = np.zeros((max_players, WIDTHS['PLAYER']))
        players[:self.n] = [self.player_row(i) for i in range(self.n)]
        return {
            'GLOBAL': self.global_row(),
            'CORP': np.stack([self.corp_row(c) for c in range(D.NUM_CORPS)]),
            'PRIVATE': np.stack([self.private_row(p) for p in range(D.NUM_PRIVATES)]),
            'CENTER': np.stack([self.center_row(r) for r in range(D.NUM_RCS)]),
            'PLAYER': players,
            **self.pair_blocks(max_players),
        }


def expected(state):
    """The packed vector's parts but the choice table, as token-data-1889.md
    defines them (``tests/g1889/test_extraction_layout3_choices.py`` checks the
    table)."""
    return Facts(state).parts(state.max_players)


def test_rows_and_pair_blocks_are_the_entity_facts(samples):
    seen = set()
    for s in samples:
        want = expected(s.state)
        assert set(want) == set(s.parts) - {'CHOICES'}
        for name, values in want.items():
            np.testing.assert_allclose(s.parts[name], values, rtol=2e-6, atol=1e-7,
                                       err_msg=f'{s.kind} phase {s.phase}: {name}')
        seen.add((s.kind, s.phase, s.num_players, s.max_players))
    assert {phase for _, phase, _, _ in seen} == {-1, *range(K.NUM_DECISION_PHASES)}
    assert {(n, capacity) for _, _, n, capacity in seen} >= {(n, p) for n, p, _ in CAPACITY_PLAYOUTS}
    assert len(samples) > 2_000


def _constant_by_the_tables():
    """The slots the tables keep at 0: GLOBAL's depot copies of the unlimited
    D, the pool copies of a type no discard or trade-in puts there (2s and
    Ds), the rust slots of types that never rust, and the phase-start slot of
    a type that starts no phase; CENTER's terrain, which no revenue center
    has; and CORP's build distance, until it lands."""
    constant = {name: set() for name in WIDTHS}
    for t in range(D.NUM_TRAIN_TYPES):
        block = 90 + 7 * t
        if D.TRAIN_COUNT[t] < 0:
            constant['GLOBAL'].add(block)
        if t not in (*D.DISCARD_TYPES, *D.TRADE_IN_TYPE):
            constant['GLOBAL'].add(block + 1)
        if _rust_phase(t) < 0:
            constant['GLOBAL'] |= {block + 4, block + 5}
        if t not in D.PHASE_ON_TRAIN[1:]:
            constant['GLOBAL'].add(block + 6)
    for slot, bit in ((6, D.TERRAIN_MOUNTAIN), (7, D.TERRAIN_WATER)):
        if not any(D.TILE_TERRAIN[D.NUM_SUPPLY_TILES + D.RC_HEX[r]] & bit for r in range(D.NUM_RCS)):
            constant['CENTER'].add(slot)
    constant['CORP'] = set(range(70, 76))
    return constant


def test_every_value_is_finite_and_every_slot_is_exercised(samples):
    """Every slot of every row type and every pair-block channel but build
    distance's is nonzero somewhere, but those the tables keep at 0, and
    every value is finite."""
    nonzero = {}
    for s in samples:
        assert np.isfinite(s.vector).all()
        for name, values in s.parts.items():
            flat = values.reshape(-1, values.shape[-1]) != 0
            nonzero[name] = nonzero.get(name, np.zeros(values.shape[-1], bool)) | flat.any(axis=0)
    constant = _constant_by_the_tables()
    assert len(constant['GLOBAL']) == 10 and constant['CENTER'] == {6, 7}
    for name, width in WIDTHS.items():
        never = set(np.flatnonzero(~nonzero[name]).tolist())
        assert never == constant[name], (name, sorted(never))
    assert nonzero['PLAYER_CORP'].all() and nonzero['PLAYER_PRIVATE'].all() and nonzero['CORP_PRIVATE'].all()
    assert nonzero['CORP_CENTER'].tolist() == [True, True, True, False]
    assert not nonzero['CENTER_CENTER'].any()


# =============================================================================
# HIDDEN ROWS, PADDING, AND CAPACITY
# =============================================================================

def test_hidden_privates_padded_seats_and_unfilled_parts_are_zero(samples):
    """A private out of play or closed is a zero row and zero pair columns; a
    padded seat is a zero PLAYER row and zero pair rows; build distance's
    slots and channels are zero, and so is the choice table without a
    decision."""
    closed = out_of_play = padded = 0
    for s in samples:
        parts = s.parts
        for p in range(D.NUM_PRIVATES):
            owner = PRIVATES[p].owner(s.state)
            if owner in (K.OWNER_NOT_IN_PLAY, K.OWNER_CLOSED):
                assert not parts['PRIVATE'][p].any()
                assert not parts['PLAYER_PRIVATE'][:, p].any() and not parts['CORP_PRIVATE'][:, p].any()
                closed += owner == K.OWNER_CLOSED
                out_of_play += owner == K.OWNER_NOT_IN_PLAY
            else:
                assert parts['PRIVATE'][p, 0] == 1
        assert (PRIVATES[D.PRIVATE_UTF].owner(s.state) == K.OWNER_NOT_IN_PLAY) == (s.num_players < 4)
        assert parts['PLAYER'][:s.num_players, 0].all()
        for name in ('PLAYER', 'PLAYER_CORP', 'PLAYER_PRIVATE'):
            assert not parts[name][s.num_players:].any(), name
        padded += s.max_players - s.num_players
        assert parts['GLOBAL'][0, 0] == 1 and parts['CORP'][:, 0].all() and parts['CENTER'][:, 0].all()
        assert not parts['CENTER_CENTER'].any() and (s.phase >= 0 or not parts['CHOICES'].any())
        assert not parts['CORP_CENTER'][:, :, 3].any() and not parts['CORP'][:, 70:].any()
    assert closed > 100 and out_of_play > 100 and padded > 100


def test_selections_mark_at_most_one_row(samples):
    for s in samples:
        for name, slot in (('CORP', 1), ('CORP', 2), ('CORP', 3), ('PRIVATE', 1), ('PRIVATE', 2), ('CENTER', 1),
                           ('PLAYER', 1)):
            assert s.parts[name][:, slot].sum() <= 1, (name, slot)
        # The subject: at most one of PRIVATE 1, CORP 2, CORP 3, and CENTER 1.
        subject = (s.parts['PRIVATE'][:, 1].sum() + s.parts['CORP'][:, 2].sum() + s.parts['CORP'][:, 3].sum()
                   + s.parts['CENTER'][:, 1].sum())
        assert subject <= 1


def test_a_larger_capacity_pads_the_same_game(samples):
    """Each capacity playout of fewer than 5 players replayed at capacity 5:
    every part is the same, the players' parts padded with zero seats."""
    compared = 0
    for spec, calls in _capacity_games():
        if spec['max_players'] == 5:
            continue
        small, large = new_state(spec), new_state({**spec, 'max_players': 5})
        for done in range(len(calls) + 1):
            if done % 3 == 0:
                a, b = split(inputs_of(small), spec['max_players']), split(inputs_of(large), 5)
                for name in a:
                    if name.startswith('PLAYER'):
                        np.testing.assert_array_equal(b[name][:spec['max_players']], a[name], err_msg=name)
                        assert not b[name][spec['max_players']:].any(), name
                    else:
                        np.testing.assert_array_equal(b[name], a[name], err_msg=name)
                compared += 1
            if done < len(calls):
                apply_op(small, *calls[done])
                apply_op(large, *calls[done])
    assert compared > 300


def test_batches_match_single_states(samples):
    """A batch at one capacity takes rows of every player count it holds,
    and ``max_players`` 0 reads the capacity from the buffer's width."""
    for capacity in (3, 4, 5):
        picked = [s for s in samples[::5] if s.max_players == capacity]
        assert {s.num_players for s in picked} == set(range(3, capacity + 1))
        out = np.full((len(picked), get_input_size(capacity, layout_version=LAYOUT)), np.nan, np.float32)
        fill_inputs_batch([s.state._array.copy() for s in picked], out, layout_version=LAYOUT)
        np.testing.assert_array_equal(out, np.stack([s.vector for s in picked]))


# =============================================================================
# PURITY
# =============================================================================

def test_the_vector_does_not_depend_on_the_caches_and_the_fill_writes_only_them(samples):
    layout = get_layout(0)
    caches = (set(range(layout.graph_offset, layout.players_offset))
              | set(range(layout.corps_offset, layout.privates_offset)) | set(DIRTY))
    for s in samples[::3]:
        np.testing.assert_array_equal(inputs_of(s.state, stale=True), s.vector)
        state, stale = copy_of(s.state), copy_of(s.state, stale=True)
        out = np.empty_like(s.vector)
        fill_inputs(state, out, s.max_players, layout_version=LAYOUT)
        fill_inputs(stale, out, s.max_players, layout_version=LAYOUT)
        np.testing.assert_array_equal(state._array, stale._array)  # both refreshed alike
        assert not state._array[DIRTY].any()
        assert set(np.flatnonzero(state._array != s.state._array).tolist()) <= caches


# =============================================================================
# EXPORTS
# =============================================================================

def test_the_parts_chain_to_the_size():
    for capacity in (3, 4, 5):
        parts = get_input_parts(capacity, layout_version=LAYOUT)
        assert [p.name for p in parts] == ['GLOBAL', 'CORP', 'PRIVATE', 'CENTER', 'CORP_PRIVATE', 'CORP_CENTER',
                                           'CENTER_CENTER', 'CHOICES', 'PLAYER', 'PLAYER_CORP', 'PLAYER_PRIVATE']
        offset = 0
        for part in parts:
            assert part.offset == offset, part.name
            offset += math.prod(part.shape)
            assert [count for _, count in part.rows] == list(part.shape[:len(part.rows)]), part.name
        assert offset == get_input_size(capacity, layout_version=LAYOUT) == 4833 + 55 * capacity
        assert get_num_trunk_rows(capacity, layout_version=LAYOUT) == 40 + capacity
        by_name = {p.name: p for p in parts}
        rows = {name: by_name[name].rows[0] for name in WIDTHS}
        assert rows == {'GLOBAL': (0, 1), 'CORP': (1, 7), 'PRIVATE': (8, 7), 'CENTER': (15, 25),
                        'PLAYER': (40, capacity)}
        assert {name: by_name[name].shape[1] for name in WIDTHS} == WIDTHS
        assert by_name['CORP_CENTER'].rows == (rows['CORP'], rows['CENTER'])
        assert by_name['PLAYER_PRIVATE'].rows == (rows['PLAYER'], rows['PRIVATE'])
    choices = next(p for p in get_input_parts(5, layout_version=LAYOUT) if p.name == 'CHOICES')
    assert {w.name: int(w) for w in TokenWidth3} == {f'TW3_{name}': width for name, width in WIDTHS.items()}
    assert [int(TrunkRow3[f'TRUNK3_{name}']) for name in WIDTHS] == [0, 1, 8, 15, 40]
    assert int(InputOffset3.IN3_CHOICES) == choices.offset and int(InputOffset3.IN3_PLAYERS) == 4833
    assert (int(PairChannels3.CH3_PLAYER_CORP), int(PairChannels3.CH3_CORP_CENTER)) == (3, 4)
    assert choices.shape == (K.MAX_ACTION_SIZE, 47)


@pytest.mark.parametrize('call', [
    lambda: get_input_size(5, layout_version=2),
    lambda: get_input_parts(5, layout_version=1),
    lambda: fill_inputs(new_state({'num_players': 3, 'max_players': 3, 'seed': 1, 'priority_player': -1,
                                   'flags': [False, False, 0]}), np.zeros(4998, np.float32), layout_version=2),
    lambda: get_token_dim(3),
])
def test_other_layout_ids_raise(call):
    with pytest.raises(ValueError, match='layout'):
        call()


def test_a_buffer_of_another_capacity_is_refused():
    state = new_state({'num_players': 3, 'max_players': 3, 'seed': 1, 'priority_player': -1, 'flags': [False, False, 0]})
    with pytest.raises(AssertionError, match='buffer'):
        fill_inputs(state, np.zeros(get_input_size(4, layout_version=LAYOUT), np.float32), layout_version=LAYOUT)
    with pytest.raises(AssertionError, match='buffer'):
        fill_inputs_batch([state._array.copy()], np.zeros((1, 5000), np.float32), layout_version=LAYOUT)


def test_choice_slots_are_the_policy_layouts():
    slots = get_choice_slots()
    assert slots.shape == (K.NUM_DECISION_PHASES, K.MAX_ACTION_SIZE) and slots.dtype == np.int16
    for phase, size in enumerate(POLICY_LAYOUT.phase_sizes):
        first = POLICY_LAYOUT.offsets[phase]
        assert slots[phase, :size].tolist() == list(range(first, first + size))
        assert (slots[phase, size:] == -1).all()


def test_named_rows_are_the_rows_each_slot_names():
    named, slots = get_named_rows(), get_choice_slots()
    assert named.shape == (POLICY_LAYOUT.unified_dim,) and named.dtype == np.int16

    def at(phase, action):
        return int(named[slots[phase, action]])

    corp, private, center = int(TrunkRow3.TRUNK3_CORP), int(TrunkRow3.TRUNK3_PRIVATE), int(TrunkRow3.TRUNK3_CENTER)
    want = {}
    want[(K.DPHASE_AUCTION, K.AUCTION_BUY)] = int(NamedRow3.NAMED3_TOPMOST)
    for p in range(1, D.NUM_PRIVATES):
        want[(K.DPHASE_AUCTION, K.AUCTION_BID + p - 1)] = private + p
    for c in range(D.NUM_CORPS):
        for first in (K.STOCK_IPO, K.STOCK_POOL, K.STOCK_PAR, K.STOCK_SELL):
            want[(K.DPHASE_STOCK, first + c)] = corp + c
        want[(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_CORP + c)] = want[(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_SELL + c)] = corp + c
    for phase, dr, mf in ((K.DPHASE_STOCK, K.STOCK_DR, K.STOCK_MF),
                          (K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DR, K.BUY_PRIVATE_PRE_MF),
                          (K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_DR, None)):
        want[(phase, dr)] = private + D.PRIVATE_DR
        if mf is not None:
            want[(phase, mf)] = private + D.PRIVATE_MF
    for p in range(D.NUM_PRIVATES):
        want[(K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_PRIVATE + p)] = private + p
        want[(K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_PRIVATE + p)] = private + p
    for i, h in enumerate(D.TRACK_HEXES):
        if D.HEX_RC[h] >= 0:
            want[(K.DPHASE_TRACK, K.TRACK_HEX + i)] = center + D.HEX_RC[h]
    for i, h in enumerate(D.TOKEN_HEXES):
        want[(K.DPHASE_TOKEN, K.TOKEN_HEX + i)] = center + D.HEX_RC[h]
    for i, h in enumerate(D.PORT_LAY_HEX):
        want[(K.DPHASE_PORT, K.PORT_LAY + i)] = center + D.HEX_RC[h]
    got = {(phase, k): at(phase, k) for phase, size in enumerate(POLICY_LAYOUT.phase_sizes) for k in range(size)
           if at(phase, k) != int(NamedRow3.NAMED3_NONE)}
    assert got == want
    assert sum(1 for (phase, _) in want if phase == K.DPHASE_TRACK) == 19
    assert all(D.RC_TYPE[D.HEX_RC[h]] == D.RC_CITY for h in D.TOKEN_HEXES)


def test_the_tile_table_is_the_tile_block_of_each_tile(samples):
    """Entry [code, rotation] is layout 2's tile block of that tile as laid,
    with the tile's own revenue: each HEX row's tile block on the sample
    states, its revenue aside for an offboard, whose CENTER row has it."""
    table = get_tile_table()
    assert table.shape == (D.NUM_TILES + 1, D.NUM_SIDES, TileBlock.TW_TILE_BLOCK) and table.dtype == np.float32
    assert not table[0].any()
    block = slice(TileBlock.TILE_BLOCK, TileBlock.TILE_BLOCK + TileBlock.TW_TILE_BLOCK)
    revenue = 71
    checked = set()
    for s in samples[::25]:
        tokens = np.zeros((get_num_tokens(s.max_players), get_token_dim(2)), np.float32)
        get_token_data(copy_of(s.state), tokens, s.max_players, layout_version=2)
        for h in range(D.NUM_HEXES):
            tile, rotation = MAP.tile(s.state, h), MAP.rotation(s.state, h)
            got, want = table[tile + 1, rotation].copy(), tokens[TokenRow.ROW_HEX + h, block].copy()
            if D.TILE_RC[tile] == D.RC_OFFBOARD:
                assert got[revenue] == 0
                got[revenue] = want[revenue] = 0
            np.testing.assert_array_equal(got, want)
            checked.add((tile, rotation))
    for tile in range(D.NUM_TILES):
        for rotation in range(D.NUM_SIDES):
            paths = table[tile + 1, rotation, 41:62]
            assert paths.tolist() == [float(D.TILE_MASK[tile][rotation] >> i & 1) for i in range(D.NUM_PAIRS)]
            assert table[tile + 1, rotation, 71] == np.float32(D.TILE_REVENUE[tile]) / np.float32(100)
    assert len(checked) > 150
