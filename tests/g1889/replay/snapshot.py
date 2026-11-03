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
  ``sr_sold_then_bought``, and the turn player's ``sr_sold_mask``;
- lays: Ruby lists every legal rotation with its cost and its
  semi-restrictive verdict, the engine one canonical rotation per candidate,
  legal only if affordable and passing the rule;
- Track after the lay: Ruby keeps the Track step open after the corp's lay
  while its president owns MF (``tile_lay_abilities_should_block?``, for the
  port tile; the extractor's ``lays`` is None there), though the corp cannot
  lay again; the engine has moved on to TOKEN (engine-design §10.2);
- the turn's emergency sales: Ruby clears ``corporations_sold`` as the next
  corp's turn starts, the engine at the turn's end, so they compare while a
  corp operates; ``last_share_sold_price`` is the engine's ``ebuy_sold_any``
  flag.

At a STOCK decision, ``compare`` also checks the engine's legal ids against
Ruby's ``stock.legal``: the pass (absent on a must-sell turn), each IPO and
pool buy, each par, and each sellable block, which composes STOCK's "sell
corp c" with the blocks SELL_AMOUNT then offers (``sub_decisions``). PAR is
not checked there; it lies inside one Ruby ``par``. In an OR it checks
Ruby's network oracles for every corp with a token (``operating.graph``,
``tokens``, and ``lays``): GRAPH's sides and nodes reached and
``route_available``, the cities the corp may token, and its lay candidates
with their costs. They do not depend on
whose turn it is; the oracle tests also check the operating corp's TRACK,
TILE, and TOKEN ids. At a BuyTrain snapshot it checks ``must_buy_train``
against Ruby's ``must_buy_train?`` on a fresh graph (``operating.must_buy``)
and BUY_TRAIN's legal choices against Ruby's BuyTrain step
(``operating.trains``, ``ruby_train_buys``), with the president's emergency
sales and bankruptcy in a must-buy (``operating.emergency``), each source
composed with the trains SELECT_TRAIN then offers and each sale with
EMERGENCY_SELL_AMOUNT's blocks (``engine_train_buys``); at a DiscardTrain
snapshot the crowded corps in Ruby's order (``operating.crowded``), the
first of which is Ruby's current entity there; and at a Route snapshot the route solver's
maximum against AutoRouter's (``operating.auto_router``;
``test_route_oracles.py`` checks the routes). With an operator it checks the
privates the corp may buy (``operating.companies``, ``ruby_private_buys``),
and at ER's SpecialTrack step ER_BONUS's lays (``operating.er_bonus``). In
an SR or OR it checks MF's port lays while Ruby's ability is usable
(``abilities.port``) and whether DR's owner may exchange it
(``abilities.exchange``), whoever's turn it is; ``test_private_oracles.py``
checks which of them the turn's decisions offer.

Each player's ``value`` compares with ``Player.value``, and once the game
is over its ``result`` with ``final_scores_py``.

Not compared, because the engine does not model them: a corporation's
``train_ids``, ``operated``, ``last_dividend``, and ``closed`` (1889
corporations never close); private abilities other than MF's port lay (the
others follow the owner and the phase); the SR's ``bought_from_ipo`` (Ruby
uses it only for multiple buys, which 1889 lacks), ``last_to_act`` (the
engine's ``priority_player`` follows it, and ``game.priority`` is compared),
and ``pass_order`` (only the ``passed`` flags matter: 1889 orders the next SR
after the last to act); the emergency oracle's sales outside a must-buy, and
its bankruptcy outside a must-buy's BuyTrain, which Ruby's steps accept but
its UI does not offer (``test_train_oracles.py`` counts them); and
``round_state``, which is diagnostics only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.data import COLORS
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
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
# actions; BID rests there in a closed auction. Sub-phases inside one Ruby
# action (SELL_AMOUNT, PORT, SELECT_TRAIN, EMERGENCY_SELL_AMOUNT,
# PRIVATE_PRICE) and automated steps are absent.
RUBY_STEPS = {
    K.STEP_AUCTION: 'WaterfallAuction', K.STEP_BID: 'WaterfallAuction', K.STEP_STOCK: 'BuySellParShares',
    K.STEP_PAR: 'BuySellParShares', K.STEP_TRACK: 'Track', K.STEP_TILE: 'Track', K.STEP_TOKEN: 'Token',
    K.STEP_DIVIDEND: 'Dividend', K.STEP_BUY_TRAIN: 'BuyTrain', K.STEP_TRAIN_PRICE: 'BuyTrain',
    K.STEP_DISCARD: 'DiscardTrain', K.STEP_BUY_PRIVATE: 'BuyCompany', K.STEP_ER_BONUS: 'SpecialTrack',
    K.STEP_ROUTE: 'Route',  # automated: the engine is there from load_state, or replay's row before it
}
# The engine step each Ruby OR step loads as.
OR_STEPS = {
    'Track': K.STEP_TRACK, 'Token': K.STEP_TOKEN, 'Route': K.STEP_ROUTE, 'Dividend': K.STEP_DIVIDEND,
    'BuyTrain': K.STEP_BUY_TRAIN, 'DiscardTrain': K.STEP_DISCARD, 'BuyCompany': K.STEP_BUY_PRIVATE,
    'SpecialTrack': K.STEP_ER_BONUS,
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
    _compare_abilities(state, snap['abilities'], check)
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
    if g['end_reason'] != 'manually_ended':  # Ruby's end_game, which the engine does not model
        check('game.result', engine_result(state, ids) if finished else None, g['result'])
    if finished:  # Ruby keeps the last round's fields once the game is over
        return
    check('game.round', RUBY_ROUNDS.get(TURN.round(state)), g['round'])
    step = TURN.step(state)
    ruby_step = 'Token' if track_after_lay(snap) and step == K.STEP_TOKEN else g['step']  # EQUIVALENT
    check('game.step', RUBY_STEPS.get(step, K.Step(step).name), ruby_step)
    check('game.active_players', [ids[TURN.active_player(state)]], g['active_players'])
    entity = g['current_entity'] or {}
    if entity.get('type') == 'player':
        check('game.current_entity', ids[TURN.active_player(state)], entity['id'])
    elif entity.get('type') == 'corporation':  # at DiscardTrain, the crowded corp that discards
        corp = TRAINS.first_crowded(state) if step == K.STEP_DISCARD else TURN.operating_corp(state)
        check('game.current_entity', D.CORP_SYMS[corp] if corp >= 0 else None, entity['id'])
    if g['round'] == 'ISR':
        check('game.round_entities[entity_index]', ids[TURN.auction_turn(state)],
              g['round_entities'][g['entity_index']])
    elif g['round'] == 'OR':
        order = [D.CORP_SYMS[c] for c in (TURN.or_order(state, i) for i in range(D.NUM_CORPS)) if c >= 0]
        check('game.round_entities', order, g['round_entities'])
        check('game.entity_index', TURN.or_index(state), g['entity_index'])


def engine_result(state: GameState, ids: list) -> dict[str, int]:
    """``final_scores_py`` as Ruby's ``result``: ``{str(player id): value}``."""
    return {str(player_id): score for player_id, score in zip(ids, A.final_scores_py(state))}


def track_after_lay(snap: Snapshot) -> bool:
    """Ruby's Track step, kept open after the turn's lay for MF's port tile."""
    return snap['game']['step'] == 'Track' and snap['operating'] is not None and snap['operating']['lays'] is None


def _compare_players(state, snap, ids, check) -> None:
    for i, ruby in enumerate(snap['players']):
        player, path = PLAYERS[i], f'players[{i}]'
        check(f'{path}.cash', player.cash(state), ruby['cash'])
        check(f'{path}.value', player.value(state), ruby['value'])
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
        if operating['operator'] is not None:  # EQUIVALENT: Ruby clears them as the next turn starts
            check('operating.ebuy_sold', _bits(TURN.ebuy_sold_mask(state), D.CORP_SYMS),
                  sorted(operating['ebuy_sold'], key=CORP.__getitem__))
        check('operating.ebuy_sold_any', bool(TURN.ebuy_sold_any(state)),
              operating['ebuy_last_share_price'] is not None)
        if operating['must_buy'] is not None and corp >= 0:
            check('operating.must_buy', A.must_buy_train_py(state, corp), operating['must_buy']['fresh'])
        if operating['trains'] is not None and TURN.step(state) == K.STEP_BUY_TRAIN:
            _compare_train_buys(state, operating, check)
        if operating['crowded'] is not None:
            check('operating.crowded', [D.CORP_SYMS[c] for c in range(D.NUM_CORPS) if TRAINS.crowded(state, c)],
                  operating['crowded'])
        if operating['auto_router'] is not None and corp >= 0:
            check('operating.auto_router.revenue', max_route_revenue(state, corp), operating['auto_router']['revenue'])
        if operating['companies'] is not None and corp >= 0:
            check('operating.companies', engine_private_buys(state, corp),
                  ruby_private_buys(operating['companies'], CORPS[corp].cash(state)))
        if operating['er_bonus'] is not None and TURN.step(state) == K.STEP_ER_BONUS:
            lays = operating['er_bonus']['lays']
            check('operating.er_bonus.lays', engine_er_lays(state), ruby_special_lays(lays))
            check('operating.er_bonus.cost', sorted({cost for *_, cost in lays}), [0] if lays else [])
        _compare_network(state, operating, check)


def ruby_private_buys(companies: dict, cash: int) -> list[str]:
    """The privates Ruby's BuyCompany lets the operator buy from its treasury:
    none unless the step offers ``buy_company``, else each whose price range
    holds a price within the cash."""
    if 'buy_company' not in companies['actions']:
        return []
    return sorted((sym for sym, low, high in companies['companies'] if low <= min(high, cash)), key=PRIVATE.__getitem__)


def engine_private_buys(state: GameState, corp: int) -> list[str]:
    return [D.PRIVATE_SYMS[p] for p in A.purchasable_privates_py(state, corp)]


def ruby_special_lays(lays: list) -> list[tuple[str, str, int]]:
    """A private's lays as sorted unique (hex, tile, canonical rotation)."""
    return sorted({(hex_id, tile, D.TILE_CANON_ROT[TILE[tile]][rotation]) for hex_id, tile, rotation, _ in lays})


def engine_er_lays(state: GameState) -> list[tuple[str, str, int]]:
    return sorted((D.HEX_IDS[D.ER_HEX], D.TILE_NAMES[tile], rotation) for tile, rotation in A.er_candidates_py(state))


def engine_port_lays(state: GameState) -> list[tuple[str, str, int]]:
    return sorted((D.HEX_IDS[h], D.TILE_NAMES[D.PORT_TILE], rotation) for h, rotation in A.port_lays_py(state))


def _compare_abilities(state, abilities, check) -> None:
    """MF's port lays and DR's exchange, whoever's turn it is, where Ruby's
    ability is usable. The engine charges nothing for the port tile, as Ruby
    charges the port hexes' terrain, which is nothing."""
    if (port := abilities['port']) is not None:
        check('abilities.port', engine_port_lays(state), ruby_special_lays(port))
        check('abilities.port.cost', sorted({cost for *_, cost in port}), [0] if port else [])
    if abilities['exchange'] is not None:
        owner = PRIVATES[D.PRIVATE_DR].owner(state) - K.OWNER_PLAYER
        if 0 <= owner < TURN.num_players(state):
            check('abilities.exchange', A.can_exchange_dr_py(state, owner), abilities['exchange'])


def _turn_purchases(current_actions: list[str]) -> tuple[bool, bool]:
    """Whether this turn bought (a share or a par), and bought after selling."""
    purchases = [i for i, kind in enumerate(current_actions) if kind in ('buy_shares', 'par')]
    sales = [i for i, kind in enumerate(current_actions) if kind == 'sell_shares']
    return bool(purchases), bool(purchases and sales and sales[0] < purchases[-1])


def ruby_lays(lays: list, cash: int) -> set[tuple[str, str, int]]:
    """Ruby's legal lays for one corp as (hex, tile, canonical rotation): the
    affordable ones that pass the semi-restrictive rule."""
    return {(hex_id, tile, D.TILE_CANON_ROT[TILE[tile]][rotation])
            for hex_id, tile, rotation, cost, ok in lays if ok and cost <= cash}


def engine_lays(state: GameState, corp: int) -> set[tuple[str, str, int]]:
    return {(D.HEX_IDS[h], D.TILE_NAMES[tile], rotation)
            for h in range(D.NUM_HEXES) for tile, rotation in A.lay_candidates_py(state, h, corp)}


def _compare_network(state, operating, check) -> None:
    """GRAPH, token, and lay legality against Ruby's oracles (``extracts.py``)."""
    for sym, ruby in operating['graph'].items():
        c, path = CORP[sym], f'operating.graph.{sym}'
        check(f'{path}.hexes', {D.HEX_IDS[h]: _bits(GRAPH.sides_reached(state, c, h), range(D.NUM_SIDES))
                                for h in range(D.NUM_HEXES) if GRAPH.sides_reached(state, c, h)}, ruby['hexes'])
        check(f'{path}.nodes', sorted(D.HEX_IDS[h] for h in range(D.NUM_HEXES) if GRAPH.node_reached(state, c, h)),
              ruby['nodes'])
        check(f'{path}.route', GRAPH.route_available(state, c), ruby['route'])
    for sym, ruby in operating['tokens'].items():
        check(f'operating.tokens.{sym}', sorted(D.HEX_IDS[h] for h in A.token_hexes_py(state, CORP[sym])), ruby)
    for sym, lays in (operating['lays'] or {}).items():
        c, path = CORP[sym], f'operating.lays.{sym}'
        ours, theirs = engine_lays(state, c), ruby_lays(lays, CORPS[c].cash(state))
        check(f'{path} (engine only)', sorted(ours - theirs), [])
        check(f'{path} (ruby only)', sorted(theirs - ours), [])
        for hex_id, cost in sorted({(hex_id, cost) for hex_id, _, _, cost, _ in lays}):
            check(f'{path}.cost.{hex_id}', MAP.lay_cost(state, HEX[hex_id], c), cost)


def ruby_train_buys(trains: dict, cash: int, president_cash: int = 0,
                    emergency: dict | None = None) -> set[tuple[str, ...]]:
    """Ruby's BuyTrain choices for a corp with ``cash``, as ``('pass',)``,
    ``('depot' | 'pool' | seller, type)``, ``('trade', type)``, ``('sell',
    corporation, percent)``, and ``('bankrupt',)``: the pass when offered;
    with ``room``, each depot or pool train its funds cover, and another
    corp's train when its range holds a price (``spend_minmax``: ¥1 to the
    cash); and each trade-in at most the cash, room or not. In a must-buy
    (``sell_shares`` is offered, as the president may contribute), the
    president's cash counts toward the cheapest depot or pool train
    (``check_for_cheapest_train``; in an emergency ``buyable_trains`` lists
    only that one), and ``emergency`` (``operating.emergency``) adds the
    president's sales and bankruptcy; outside a must-buy the UI offers
    neither, though Ruby's steps would take them."""
    must_buy = 'sell_shares' in trains['actions']
    buys: set[tuple[str, ...]] = {('pass',)} if 'pass' in trains['actions'] else set()
    cheapest = min((low for source, _, low, _ in trains['buyable'] if source in ('depot', 'pool')), default=None)
    if trains['room']:
        for source, train, low, high in trains['buyable']:
            if source not in ('depot', 'pool'):
                funds = min(high, cash)
            else:
                funds = cash + president_cash if must_buy and low == cheapest else cash
            if low <= funds:
                buys.add((source, train))
    buys |= {('trade', train) for train, price in trains['exchanges'] if price <= cash}
    if must_buy and emergency is not None:
        buys |= {('sell', corp, percent) for corp, percents in (emergency['sell'] or {}).items()
                 for percent in percents}
        if emergency['bankrupt']:
            buys.add(('bankrupt',))
    return buys


def legal_actions(state: GameState) -> list[tuple[int, A.ActionInfoTuple]]:
    """The current decision's engine-legal ids, decoded."""
    phase = A.get_decision_phase_py(state)
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    return [(int(i), A.decode_action_py(phase, int(i))) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]


def sub_decisions(state: GameState, action_id: int) -> list[tuple[int, A.ActionInfoTuple]]:
    """The legal choices of the sub-phase ``action_id`` opens (SELL_AMOUNT,
    SELECT_TRAIN, EMERGENCY_SELL_AMOUNT), through the driver on a copy: the
    sub-phase as it opens, before any choice the driver then forces."""
    n = TURN.num_players(state)
    after = GameState.from_array(state._array, n, state.max_players)
    history: list = []
    assert DRIVER.apply_action(after, action_id, history) != STATUS_INVALID, action_id
    if len(history) > 1:
        after = GameState.from_array(history[1][0], n, state.max_players)
    return legal_actions(after)


def engine_train_buys(state: GameState) -> set[tuple[str, ...]]:
    """The operating corp's legal BUY_TRAIN choices in ``ruby_train_buys``'s
    form: each source with each train SELECT_TRAIN then offers (a trade-in
    from the depot), and each corp the president may sell with each block
    EMERGENCY_SELL_AMOUNT then offers."""
    sources = {A.ACTION_BUY_DEPOT_TRAIN: 'depot', A.ACTION_BUY_POOL_TRAIN: 'pool', A.ACTION_BUY_CORP_TRAIN: None}
    buys: set[tuple[str, ...]] = set()
    for action_id, info in legal_actions(state):
        if info.type in (A.ACTION_BUY_PRIVATE, A.ACTION_DR_EXCHANGE):  # the interjection slots
            continue
        if info.type == A.ACTION_SELECT_SALE:
            buys |= {('sell', D.CORP_SYMS[info.corp], 10 * sale.count) for _, sale in sub_decisions(state, action_id)}
        elif info.type in sources:
            source = sources[info.type] or D.CORP_SYMS[info.corp]
            buys |= {('trade' if train.type == A.ACTION_TRADE_IN else source, D.TRAIN_NAMES[train.train_type])
                     for _, train in sub_decisions(state, action_id)}
        else:
            buys.add({A.ACTION_PASS: ('pass',), A.ACTION_BANKRUPT: ('bankrupt',)}[info.type])
    return buys


def ruby_operator_buys(state: GameState, operating: dict) -> set[tuple[str, ...]]:
    """``ruby_train_buys`` for the operating corp, with its president's cash."""
    corp = TURN.operating_corp(state)
    president_cash = PLAYERS[CORPS[corp].president(state)].cash(state)
    return ruby_train_buys(operating['trains'], CORPS[corp].cash(state), president_cash, operating['emergency'])


def _compare_train_buys(state, operating, check) -> None:
    ours, theirs = engine_train_buys(state), ruby_operator_buys(state, operating)
    check('operating.trains (engine only)', sorted(ours - theirs), [])
    check('operating.trains (ruby only)', sorted(theirs - ours), [])


def _compare_stock_legal(state, legal, check) -> None:
    """The engine's legal STOCK ids against Ruby's options for the turn
    player, each corp to sell with the blocks SELL_AMOUNT then offers."""
    actions = legal_actions(state)
    infos = [info for _, info in actions]

    def corps(kind):
        return [D.CORP_SYMS[info.corp] for info in infos if info.type == kind]

    check('stock.legal.pass', any(info.type == A.ACTION_PASS for info in infos), 'pass' in legal['actions'])
    check('stock.legal.buy_ipo', corps(A.ACTION_BUY_IPO), legal['buy_ipo'])
    check('stock.legal.buy_pool', corps(A.ACTION_BUY_POOL), legal['buy_pool'])
    check('stock.legal.par', corps(A.ACTION_SELECT_PAR), legal['par'])
    sales = {D.CORP_SYMS[info.corp]: [10 * sale.count for _, sale in sub_decisions(state, action_id)]
             for action_id, info in actions if info.type == A.ACTION_SELECT_SALE}
    check('stock.legal.sell', sales, legal['sell'])


# =============================================================================
# LOAD
# =============================================================================

def load_state(snap: Snapshot, num_players: int, max_players: int = 0) -> GameState:
    """A GameState equal to ``snap`` under ``compare``, built through the entity
    APIs. Cash starts in the bank and moves with ``BANK.transfer``; every cache
    is left dirty. ``auction_pass_payouts`` has no Ruby counterpart and loads
    as 0. Auction, SR, OR, and finished snapshots load fully (a finished one
    with the OR's fields Ruby keeps); an SR step other than BuySellParShares,
    which 1889 never blocks at, raises ValueError.
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
        if g['round'] == 'OR':  # Ruby keeps the last OR's fields once the game is over
            _load_operating_fields(state, snap, ids)
        TURN.set_round(state, K.ROUND_GAME_OVER)
        TURN.set_step(state, K.STEP_GAME_OVER)
    elif g['round'] == 'ISR':
        auction = snap['auction']
        TURN.set_round(state, K.ROUND_AUCTION)
        TURN.set_step(state, K.STEP_BID if auction['auctioning'] else K.STEP_AUCTION)
        TURN.set_auction_private(state, PRIVATE[auction['auctioning']] if auction['auctioning'] else -1)
        TURN.set_auction_discount(state, snap['privates'][D.PRIVATE_SYMS[DISCOUNTED_PRIVATE]]['discount'])
        TURN.set_auction_turn(state, ids.index(g['round_entities'][g['entity_index']]))
        TURN.set_active_player(state, ids.index(g['active_players'][0]))
    elif g['round'] == 'SR' and g['step'] == 'BuySellParShares':
        _load_stock(state, snap['stock'], ids, ids.index(g['active_players'][0]))
    elif g['round'] == 'OR':
        _load_operating(state, snap, ids)
    else:
        raise ValueError(f"load_state: 1889 never blocks at {g['round']} step {g['step']}")
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
    # The extract lists the turn's action types only, so a purchase loads as
    # AR's (the corp + 1); play reads only whether there was one.
    TURN.set_sr_bought(state, bought)
    TURN.set_sr_sold_then_bought(state, sold_then_bought)


def _load_operating(state: GameState, snap: Snapshot, ids: list) -> None:
    """An OR turn: the operating order and index, the operating corp and the
    player Ruby waits on, Ruby's step, this turn's route revenue once run, and
    the emergency sales. A Track or Token snapshot opens its step, with the
    turn's lay or token still to come, except a Track snapshot after the lay,
    which Ruby keeps open for MF's port tile (engine-design §10.2): it loads
    at TOKEN, where the engine stands then. A DiscardTrain snapshot returns
    to BUY_TRAIN, the only step whose purchases change the phase, and Ruby's
    active player is the crowded corp's president. ER's SpecialTrack
    snapshot loads at ER_BONUS, for its seller, Ruby's active player, and
    returns to the step Ruby resumes (``operating.er_bonus.resume``)."""
    g, operating = snap['game'], snap['operating']
    TURN.set_round(state, K.ROUND_OPERATING)
    TURN.set_step(state, K.STEP_TOKEN if track_after_lay(snap) else OR_STEPS[g['step']])
    if g['step'] == 'DiscardTrain':
        TURN.set_return_step(state, K.STEP_BUY_TRAIN)
    if g['step'] == 'SpecialTrack':
        TURN.set_return_step(state, OR_STEPS[operating['er_bonus']['resume']])
        TURN.set_pending_seller(state, ids.index(g['active_players'][0]))
    _load_operating_fields(state, snap, ids)
    TURN.set_active_player(state, ids.index(g['active_players'][0]))
    if operating['routes_revenue'] is not None:
        TURN.set_route_revenue(state, operating['routes_revenue'])


def _load_operating_fields(state: GameState, snap: Snapshot, ids: list) -> None:
    """The OR's order and index, the operating corp, and the turn's emergency
    sales, which a finished game keeps too."""
    g, operating = snap['game'], snap['operating']
    for i, sym in enumerate(g['round_entities']):
        TURN.set_or_order(state, i, CORP[sym])
    TURN.set_or_index(state, g['entity_index'])
    if operating['operator'] is None:  # the OR's last turn is over
        return
    TURN.set_operating_corp(state, CORP[operating['operator']])
    TURN.set_ebuy_sold_mask(state, sum(1 << CORP[sym] for sym in operating['ebuy_sold']))
    TURN.set_ebuy_sold_any(state, operating['ebuy_last_share_price'] is not None)
