"""Compare engine states with Ruby snapshots, and build states from them.

``compare(state, snap)`` lists every difference between a GameState and a
snapshot from ``extracts.py`` (schema in its docstring) as
``'<field path>: engine <value> != ruby <value>'``. ``load_state(snap, ...)``
builds a GameState from a snapshot through the entity APIs, so oracle tests can
start anywhere without replaying up to it (1889/engine-design.md §10.2).

Ids map through the tables mirrors. The engine stores some fields in another
form than Ruby; ``compare`` treats these as equal (``EQUIVALENT:`` below):

- rotations: Ruby keeps the logged rotation, the engine its canonical form
  (``TILE_CANON_ROT``), the lowest rotation with the same paths;
- the depot's D: Ruby keeps one D and clones it per sale, the engine -1;
- tokens: Ruby lists corporations by city slot, the engine keeps a corp mask
  per hex, so tokens compare as sets (and a corp's token hexes as sets);
- market stacks: Ruby's index in a cell's corporation list is arrival order,
  which is the order of the engine's stamps;
- home reservations: Ruby stores them, the engine derives them from
  ``HEX_HOME_CORP`` and the home corp's token bit;
- player order: Ruby rotates ``@players`` to put the priority player first,
  the engine keeps seat order and stores ``priority_player``;
- ``last_revenue``: Ruby's None before a corp operates is the engine's 0;
- the SR turn: Ruby lists this turn's actions (``current_actions``) and marks
  this turn's sales ``'now'``; the engine keeps ``sr_bought``,
  ``sr_sold_then_bought``, and the turn player's ``sr_sold_mask``.

At a STOCK decision, ``compare`` also checks the engine's legal ids against
Ruby's ``stock.legal``: the pass (absent on a must-sell turn), each IPO and
pool buy, each par, and each sellable block. PAR is not checked there; it
lies inside one Ruby ``par``.

Not compared, because the engine does not model them (or not yet): player
``value`` and the game ``result`` (``final_scores``, 6.8); a corporation's
``train_ids``, ``operated``, ``last_dividend``, and ``closed`` (1889
corporations never close); private abilities other than MF's port lay (6.8);
the SR's ``bought_from_ipo`` (Ruby uses it only for multiple buys, which 1889
lacks), ``last_to_act`` (the engine's ``priority_player`` follows it, and
``game.priority`` is compared), and ``pass_order`` (only the ``passed`` flags
matter: 1889 orders the next SR after the last to act);
``ebuy_last_share_price`` (6.8); and ``round_state``, which is diagnostics
only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.data import COLORS
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState

Snapshot = dict[str, Any]

HEX = {hex_id: h for h, hex_id in enumerate(D.HEX_IDS)}
TILE = {name: t for t, name in enumerate(D.TILE_NAMES)}  # a printed tile is named by its hex
CORP = {sym: c for c, sym in enumerate(D.CORP_SYMS)}
PRIVATE = {sym: p for p, sym in enumerate(D.PRIVATE_SYMS)}
TRAIN = {name: t for t, name in enumerate(D.TRAIN_NAMES)}
GAME_PHASE = {name: p for p, name in enumerate(D.PHASE_NAMES)}

RUBY_ROUNDS = {K.ROUND_AUCTION: 'ISR', K.ROUND_STOCK: 'SR', K.ROUND_OPERATING: 'OR'}
END_REASONS = {K.END_REASON_NONE: None, K.END_REASON_BANKRUPT: 'bankrupt', K.END_REASON_BANK: 'bank'}
# Ruby's active step at each engine step the engine can rest at between Ruby
# actions. Sub-phases inside one Ruby action (BID, PORT, PRIVATE_PRICE) and
# automated steps are absent.
RUBY_STEPS = {
    K.STEP_AUCTION: 'WaterfallAuction', K.STEP_RAISE: 'WaterfallAuction', K.STEP_STOCK: 'BuySellParShares',
    K.STEP_PAR: 'BuySellParShares', K.STEP_TRACK: 'Track', K.STEP_TILE: 'Track', K.STEP_TOKEN: 'Token',
    K.STEP_DIVIDEND: 'Dividend', K.STEP_BUY_TRAIN: 'BuyTrain', K.STEP_TRAIN_PRICE: 'BuyTrain',
    K.STEP_DISCARD: 'DiscardTrain', K.STEP_BUY_PRIVATE: 'BuyCompany', K.STEP_ER_BONUS: 'SpecialTrack',
}
# Ruby applies the auction discount only to the cheapest private, A.
DISCOUNTED_PRIVATE = 0


def _bits(mask: int, names) -> list[str]:
    return [name for i, name in enumerate(names) if mask >> i & 1]


def _owner_code(ref: dict | None, closed: bool, ids: list) -> int:
    if closed:
        return K.OWNER_CLOSED
    if ref is None:
        return K.OWNER_UNSOLD
    if ref['type'] == 'player':
        return K.OWNER_PLAYER + ids.index(ref['id'])
    if ref['type'] == 'corporation':
        return K.OWNER_CORP + CORP[ref['id']]
    raise ValueError(f'private owner {ref}')


def _owner_name(code: int, ids: list) -> str:
    if code >= K.OWNER_CORP:
        return f'corporation {D.CORP_SYMS[code - K.OWNER_CORP]}'
    if code >= K.OWNER_PLAYER:
        return f'player {ids[code - K.OWNER_PLAYER]}'
    return {K.OWNER_UNSOLD: 'unsold', K.OWNER_CLOSED: 'closed', K.OWNER_NOT_IN_PLAY: 'not in play'}[code]


def _market(state: GameState, corp: int) -> list[int] | None:
    """[row, column, stack]; EQUIVALENT: the stack index is the stamp's rank in its cell."""
    row, col = MARKET.row(state, corp), MARKET.col(state, corp)
    if row < 0:
        return None
    stamps = sorted(MARKET.stamp(state, c) for c in range(D.NUM_CORPS)
                    if (MARKET.row(state, c), MARKET.col(state, c)) == (row, col))
    return [row, col, stamps.index(MARKET.stamp(state, corp))]


def _min_bids(state: GameState) -> dict[str, int]:
    """Ruby's WaterfallAuction#min_bid for each unsold private: the topmost's
    price outside a closed auction, else the high bid (or face value) + ¥5."""
    n = TURN.num_players(state)
    unsold = [p for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_UNSOLD]
    bids = {}
    for p in unsold:
        face = D.PRIVATE_VALUE[p] - (TURN.auction_discount(state) if p == DISCOUNTED_PRIVATE else 0)
        if p == unsold[0] and TURN.auction_private(state) < 0:
            bids[D.PRIVATE_SYMS[p]] = face
        else:
            high = max(PLAYERS[i].bid(state, p) for i in range(n))
            bids[D.PRIVATE_SYMS[p]] = (high or face) + D.MIN_BID_INCREMENT
    return bids


# =============================================================================
# COMPARE
# =============================================================================

def compare(state: GameState, snap: Snapshot) -> list[str]:
    """Every difference between ``state`` and ``snap``; empty when they agree."""
    out: list[str] = []

    def check(path: str, engine, ruby) -> None:
        if engine != ruby:
            out.append(f'{path}: engine {engine!r} != ruby {ruby!r}')

    ids = [p['id'] for p in snap['players']]
    check('players', TURN.num_players(state), len(ids))
    if out:
        return out
    _compare_game(state, snap, ids, check)
    _compare_players(state, snap, ids, check)
    _compare_corporations(state, snap, ids, check)
    _compare_privates(state, snap, ids, check)
    _compare_map(state, snap, check)
    _compare_rounds(state, snap, ids, check)
    return out


def _compare_game(state, snap, ids, check) -> None:
    g = snap['game']
    phase = PHASE.current(state)
    check('game.phase', D.PHASE_NAMES[phase], g['phase'])
    check('game.train_limit', D.PHASE_TRAIN_LIMIT[phase], g['train_limit'])
    check('game.tile_colors', _bits(D.PHASE_TILE_COLORS[phase], COLORS), g['tile_colors'])
    finished = TURN.round(state) == K.ROUND_GAME_OVER
    check('game.finished', finished, g['finished'])
    check('game.end_reason', END_REASONS.get(TURN.end_reason(state)), g['end_reason'])
    check('game.round_num', TURN.or_num(state), g['round_num'])
    check('game.turn', TURN.turn(state), g['turn'])
    check('game.or_count', TURN.or_count(state), g['or_count'])
    check('game.priority', ids[TURN.priority_player(state)], g['priority'])
    # EQUIVALENT: only the cyclic order is stored (seat order).
    start = ids.index(g['player_order'][0])
    check('game.player_order', ids[start:] + ids[:start], g['player_order'])
    check('game.bank_cash', BANK.cash(state), g['bank_cash'])
    check('game.bank_broken', BANK.broken(state), g['bank_broken'])
    if finished:  # Ruby keeps the last round's fields once the game is over
        return
    check('game.round', RUBY_ROUNDS.get(TURN.round(state)), g['round'])
    step = TURN.step(state)
    check('game.step', RUBY_STEPS.get(step, K.Step(step).name), g['step'])
    check('game.active_players', [ids[TURN.active_player(state)]], g['active_players'])
    entity = g['current_entity'] or {}
    if entity.get('type') == 'player':
        check('game.current_entity', ids[TURN.active_player(state)], entity['id'])
    elif entity.get('type') == 'corporation':
        corp = TURN.operating_corp(state)
        check('game.current_entity', D.CORP_SYMS[corp] if corp >= 0 else None, entity['id'])
    if g['round'] == 'ISR':
        check('game.round_entities[entity_index]', ids[TURN.auction_turn(state)],
              g['round_entities'][g['entity_index']])
    elif g['round'] == 'OR':
        order = [D.CORP_SYMS[c] for c in (TURN.or_order(state, i) for i in range(D.NUM_CORPS)) if c >= 0]
        check('game.round_entities', order, g['round_entities'])
        check('game.entity_index', TURN.or_index(state), g['entity_index'])


def _compare_players(state, snap, ids, check) -> None:
    for i, ruby in enumerate(snap['players']):
        player, path = PLAYERS[i], f'players[{i}]'
        check(f'{path}.cash', player.cash(state), ruby['cash'])
        shares = {sym: {'percent': 10 * player.shares(state, c), 'president': CORPS[c].president(state) == i}
                  for c, sym in enumerate(D.CORP_SYMS) if player.shares(state, c)}
        check(f'{path}.shares', shares, ruby['shares'])
        check(f'{path}.privates', [sym for p, sym in enumerate(D.PRIVATE_SYMS)
                                   if PRIVATES[p].owner(state) == K.OWNER_PLAYER + i], ruby['privates'])
        check(f'{path}.bids', {sym: player.bid(state, p) for p, sym in enumerate(D.PRIVATE_SYMS)
                               if player.bid(state, p)}, ruby['bids'])
        check(f'{path}.passed', player.passed(state), ruby['passed'])
        check(f'{path}.bankrupt', player.bankrupt(state), ruby['bankrupt'])


def _compare_corporations(state, snap, ids, check) -> None:
    for c, sym in enumerate(D.CORP_SYMS):
        corp, ruby, path = CORPS[c], snap['corporations'][sym], f'corporations.{sym}'
        par = corp.par_index(state)
        check(f'{path}.ipoed', par >= 0, ruby['ipoed'])
        check(f'{path}.par', D.PAR_PRICE[par] if par >= 0 else None, ruby['par'])
        market = _market(state, c)
        check(f'{path}.price', MARKET.price(state, c) if market else None, ruby['price'])
        check(f'{path}.market', market, ruby['market'])
        check(f'{path}.floated', corp.floated(state), ruby['floated'])
        check(f'{path}.cash', corp.cash(state), ruby['cash'])
        check(f'{path}.ipo_percent', 10 * corp.ipo_shares(state), ruby['ipo_percent'])
        check(f'{path}.pool_percent', 10 * corp.pool_shares(state), ruby['pool_percent'])
        president = corp.president(state)
        check(f'{path}.president', ids[president] if president >= 0 else None, ruby['president'])
        check(f'{path}.tokens_left', corp.tokens_left(state), ruby['tokens_left'])
        check(f'{path}.token_hexes', [hex_id for h, hex_id in enumerate(D.HEX_IDS) if MAP.tokens(state, h) >> c & 1],
              sorted(ruby['token_hexes'], key=HEX.__getitem__))
        check(f'{path}.trains', {name: TRAINS.corp_trains(state, c, t) for t, name in enumerate(D.TRAIN_NAMES)
                                 if TRAINS.corp_trains(state, c, t)}, ruby['trains'])
        check(f'{path}.privates', [p_sym for p, p_sym in enumerate(D.PRIVATE_SYMS)
                                   if PRIVATES[p].owner(state) == K.OWNER_CORP + c], ruby['privates'])
        check(f'{path}.last_revenue', corp.last_revenue(state), ruby['last_revenue'] or 0)


def _compare_privates(state, snap, ids, check) -> None:
    for p, sym in enumerate(D.PRIVATE_SYMS):
        private, ruby, path = PRIVATES[p], snap['privates'].get(sym), f'privates.{sym}'
        owner = private.owner(state)
        if ruby is None:  # Ruby drops privates that are not in play
            check(f'{path}.owner', _owner_name(owner, ids), 'not in play')
            continue
        ruby_owner = _owner_code(ruby['owner'], ruby['closed'], ids)
        check(f'{path}.owner', _owner_name(owner, ids), _owner_name(ruby_owner, ids))
        check(f'{path}.value', D.PRIVATE_VALUE[p], ruby['value'])
        check(f'{path}.revenue', private.revenue(state), ruby['revenue'])
        check(f'{path}.discount', TURN.auction_discount(state) if p == DISCOUNTED_PRIVATE else 0, ruby['discount'])
        if p == D.PRIVATE_MF and not ruby['closed']:
            check(f'{path}.port_used', bool(private.flags(state) & 1), 'tile_lay' not in ruby['abilities'])


def _compare_map(state, snap, check) -> None:
    for h, hex_id in enumerate(D.HEX_IDS):
        ruby, path = snap['hexes'][hex_id], f'hexes.{hex_id}'
        tile, mask = MAP.tile(state, h), MAP.tokens(state, h)
        check(f'{path}.tile', D.TILE_NAMES[tile], ruby['tile'])
        if D.TILE_NAMES[tile] == ruby['tile']:  # EQUIVALENT: rotations with the same paths
            check(f'{path}.rotation', MAP.rotation(state, h), D.TILE_CANON_ROT[tile][ruby['rotation']])
        check(f'{path}.tokens', _bits(mask, D.CORP_SYMS), sorted(ruby['tokens'], key=CORP.__getitem__))
        home = D.HEX_HOME_CORP[h]
        check(f'{path}.reserved', [D.CORP_SYMS[home]] if home >= 0 and not mask >> home & 1 else [], ruby['reserved'])
    check('tiles', {D.TILE_NAMES[t]: MAP.supply(state, t) for t in range(D.NUM_SUPPLY_TILES)}, snap['tiles'])
    # EQUIVALENT: the unlimited (-1) depot D is Ruby's single cloneable D.
    depot = {name: TRAINS.depot(state, t) for t, name in enumerate(D.TRAIN_NAMES) if TRAINS.depot(state, t)}
    if 'D' in depot:
        depot['D'] = 1 if depot['D'] == -1 else f"{depot['D']} (not unlimited)"
    check('depot', depot, snap['depot'])
    check('pool', {name: TRAINS.pool(state, t) for t, name in enumerate(D.TRAIN_NAMES) if TRAINS.pool(state, t)},
          snap['pool'])


def _compare_rounds(state, snap, ids, check) -> None:
    if (auction := snap['auction']) is not None:
        unsold = [sym for p, sym in enumerate(D.PRIVATE_SYMS) if PRIVATES[p].owner(state) == K.OWNER_UNSOLD]
        check('auction.companies', unsold, auction['companies'])
        private = TURN.auction_private(state)
        check('auction.auctioning', D.PRIVATE_SYMS[private] if private >= 0 else None, auction['auctioning'])
        check('auction.min_bids', _min_bids(state), auction['min_bids'])
    if (stock := snap['stock']) is not None:
        active = TURN.active_player(state)
        for i, player_id in enumerate(ids):
            sold = stock['sold'][str(player_id)]
            check(f'stock.sold.{player_id}', _bits(PLAYERS[i].sold_mask(state), D.CORP_SYMS), list(sold))
            # EQUIVALENT: 'now' marks the turn player's sales this turn.
            check(f'stock.sold_now.{player_id}', _bits(TURN.sr_sold_mask(state) if i == active else 0, D.CORP_SYMS),
                  [sym for sym, when in sold.items() if when == 'now'])
        bought, sold_then_bought = _turn_purchases(stock['current_actions'])
        check('stock.current_actions (bought)', bool(TURN.sr_bought(state)), bought)
        check('stock.current_actions (sold, then bought)', bool(TURN.sr_sold_then_bought(state)), sold_then_bought)
        if stock['legal'] is not None and TURN.step(state) == K.STEP_STOCK:
            _compare_stock_legal(state, stock['legal'], check)
    if (operating := snap['operating']) is not None:
        corp = TURN.operating_corp(state)
        check('operating.operator', D.CORP_SYMS[corp] if corp >= 0 else None, operating['operator'])
        if operating['routes_revenue'] is not None:
            check('operating.routes_revenue', TURN.route_revenue(state), operating['routes_revenue'])
        check('operating.ebuy_sold', _bits(TURN.ebuy_sold_mask(state), D.CORP_SYMS),
              sorted(operating['ebuy_sold'], key=CORP.__getitem__))


def _turn_purchases(current_actions: list[str]) -> tuple[bool, bool]:
    """Whether this turn bought (a share or a par), and bought after selling."""
    purchases = [i for i, kind in enumerate(current_actions) if kind in ('buy_shares', 'par')]
    sales = [i for i, kind in enumerate(current_actions) if kind == 'sell_shares']
    return bool(purchases), bool(purchases and sales and sales[0] < purchases[-1])


def _compare_stock_legal(state, legal, check) -> None:
    """The engine's legal STOCK ids against Ruby's options for the turn player."""
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    infos = [A.decode_action_py(K.DPHASE_STOCK, int(i)) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]

    def corps(kind):
        return [D.CORP_SYMS[info.corp] for info in infos if info.type == kind]

    check('stock.legal.pass', any(info.type == A.ACTION_PASS for info in infos), 'pass' in legal['actions'])
    check('stock.legal.buy_ipo', corps(A.ACTION_BUY_IPO), legal['buy_ipo'])
    check('stock.legal.buy_pool', corps(A.ACTION_BUY_POOL), legal['buy_pool'])
    check('stock.legal.par', corps(A.ACTION_SELECT_PAR), legal['par'])
    sales: dict[str, list[int]] = {}
    for info in infos:
        if info.type == A.ACTION_SELL:
            sales.setdefault(D.CORP_SYMS[info.corp], []).append(10 * info.count)
    check('stock.legal.sell', sales, legal['sell'])


# =============================================================================
# LOAD
# =============================================================================

def load_state(snap: Snapshot, num_players: int, max_players: int = 0) -> GameState:
    """A GameState equal to ``snap`` under ``compare``, built through the entity
    APIs. Cash starts in the bank and moves with ``BANK.transfer``; every cache
    is left dirty. ``auction_pass_payouts`` has no Ruby counterpart and loads
    as 0. Auction, SR, and finished snapshots load fully; OR turn fields raise
    NotImplementedError until 6.5 defines them.
    """
    ids = [p['id'] for p in snap['players']]
    assert len(ids) == num_players, f'snapshot has {len(ids)} players, not {num_players}'
    g = snap['game']
    state = GameState(num_players, max_players=max_players)

    for i, ruby in enumerate(snap['players']):
        BANK.transfer(state, BANK_ACCOUNT, K.OWNER_PLAYER + i, ruby['cash'])
    for c, sym in enumerate(D.CORP_SYMS):
        BANK.transfer(state, BANK_ACCOUNT, K.OWNER_CORP + c, snap['corporations'][sym]['cash'])
    if BANK.cash(state) != g['bank_cash']:
        raise ValueError(f'snapshot cash is not conserved: the bank should hold {BANK.cash(state)}')
    if g['bank_broken']:
        BANK.set_broken(state)

    for i, ruby in enumerate(snap['players']):
        player = PLAYERS[i]
        for sym, held in ruby['shares'].items():
            player.set_shares(state, CORP[sym], held['percent'] // 10)
        for sym, price in ruby['bids'].items():
            player.set_bid(state, PRIVATE[sym], price)
        player.set_passed(state, ruby['passed'])
        player.set_bankrupt(state, ruby['bankrupt'])

    stacked = []
    for c, sym in enumerate(D.CORP_SYMS):
        corp, ruby = CORPS[c], snap['corporations'][sym]
        corp.set_par_index(state, D.PAR_PRICE.index(ruby['par']) if ruby['par'] is not None else -1)
        corp.set_floated(state, ruby['floated'])
        corp.set_ipo_shares(state, ruby['ipo_percent'] // 10)
        corp.set_pool_shares(state, ruby['pool_percent'] // 10)
        corp.set_president(state, ids.index(ruby['president']) if ruby['president'] is not None else -1)
        corp.set_tokens_left(state, ruby['tokens_left'])
        corp.set_last_revenue(state, ruby['last_revenue'] or 0)
        for name, count in ruby['trains'].items():
            TRAINS.set_corp_trains(state, c, TRAIN[name], count)
        if ruby['market'] is not None:
            stacked.append((ruby['market'][2], c, ruby['market'][:2]))
    for _, c, (row, col) in sorted(stacked):  # arrival order within each cell
        MARKET.set_position(state, c, row, col)

    for sym, ruby in snap['privates'].items():
        private = PRIVATES[PRIVATE[sym]]
        private.set_owner(state, _owner_code(ruby['owner'], ruby['closed'], ids))
        if sym == 'MF' and not ruby['closed'] and 'tile_lay' not in ruby['abilities']:
            private.set_flags(state, 1)

    for hex_id, ruby in snap['hexes'].items():
        h = HEX[hex_id]
        MAP.set_tile(state, h, TILE[ruby['tile']], ruby['rotation'])
        MAP.set_tokens(state, h, sum(1 << CORP[sym] for sym in set(ruby['tokens'])))
    for name, count in snap['tiles'].items():
        MAP.set_supply(state, TILE[name], count)
    for name, count in snap['depot'].items():
        TRAINS.set_depot(state, TRAIN[name], -1 if name == 'D' else count)
    for name, count in snap['pool'].items():
        TRAINS.set_pool(state, TRAIN[name], count)

    PHASE.set_current(state, GAME_PHASE[g['phase']])
    TURN.set_turn(state, g['turn'])
    TURN.set_or_num(state, g['round_num'])
    TURN.set_or_count(state, g['or_count'])
    TURN.set_priority_player(state, ids.index(g['priority']))
    TURN.set_end_reason(state, {name: code for code, name in END_REASONS.items()}[g['end_reason']])
    if g['finished']:
        TURN.set_round(state, K.ROUND_GAME_OVER)
        TURN.set_step(state, K.STEP_GAME_OVER)
    elif g['round'] == 'ISR':
        auction = snap['auction']
        TURN.set_round(state, K.ROUND_AUCTION)
        TURN.set_step(state, K.STEP_RAISE if auction['auctioning'] else K.STEP_AUCTION)
        TURN.set_auction_private(state, PRIVATE[auction['auctioning']] if auction['auctioning'] else -1)
        TURN.set_auction_discount(state, snap['privates'][D.PRIVATE_SYMS[DISCOUNTED_PRIVATE]]['discount'])
        TURN.set_auction_turn(state, ids.index(g['round_entities'][g['entity_index']]))
        TURN.set_active_player(state, ids.index(g['active_players'][0]))
    elif g['round'] == 'SR' and g['step'] == 'BuySellParShares':
        _load_stock(state, snap['stock'], ids, ids.index(g['active_players'][0]))
    else:
        raise NotImplementedError(f"load_state: {g['round']} turn fields at step {g['step']} land with "
                                  f"{'6.5' if g['round'] == 'OR' else 'a later sub-step'}")
    return state


def _load_stock(state: GameState, stock: dict, ids: list, active: int) -> None:
    """An SR turn: the turn player, the SR's sales, and this turn's purchase and sales."""
    TURN.set_round(state, K.ROUND_STOCK)
    TURN.set_step(state, K.STEP_STOCK)
    TURN.set_active_player(state, active)
    now = 0
    for i, player_id in enumerate(ids):
        sold = stock['sold'][str(player_id)]
        PLAYERS[i].set_sold_mask(state, sum(1 << CORP[sym] for sym in sold))
        mask = sum(1 << CORP[sym] for sym, when in sold.items() if when == 'now')
        if mask:
            assert i == active, f"player {player_id} has this turn's sales on another player's turn"
            now = mask
    bought, sold_then_bought = _turn_purchases(stock['current_actions'])
    TURN.set_sr_sold_mask(state, now)
    TURN.set_sr_bought(state, bought)
    TURN.set_sr_sold_then_bought(state, sold_then_bought)
