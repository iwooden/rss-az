"""Map 18xx.games actions to engine decisions (1889/engine-design.md §5.5).

``map_action(action, pre, phase, state)`` turns one Ruby action (a host action
or one of its ``auto_actions``) into the engine decisions that perform it: a
list of ``(decision phase, phase-local id)``, starting in ``phase``. It reads
the extractor's ``pre`` annotation for what the log entry leaves implicit (a
bid's purchase or closed auction, a share's source, a token's hex, a train's
owner). Prices are exact ids, so any price a human chose is expressible;
bids and corp train prices that land on the BID, RAISE, or TRAIN_PRICE grid
use the grid point instead.

Actions with no engine decision map to ``[]``: ``message``, ``program_*``,
and ``run_routes`` whose revenue the engine's ROUTE step already matches.
Any other ``run_routes`` maps to ``(ROUTE_OVERRIDE, revenue)``, which the
harness applies with ``DRIVER.override_route_revenue``: humans' routes often
earn less than the maximum, and Ruby pays logged revenue unchecked. A
``discard_train`` by a crowded corp other than the one the engine's DISCARD
is for maps to ``(HELD_DISCARD, NUM_TRAIN_TYPES * corp + type)``: Ruby takes
discards from any crowded corp, the engine in corp order, and the order
cannot change the result, so the harness holds it until the engine reaches
that corp (314 record 339, engine-design.md §10.2). ``end_game`` raises
``ReplayEnd``. An action the mapper cannot express yet raises
``UnsupportedAction`` naming its type.

An OR pass maps by Ruby's step (Track, Token, BuyTrain, BuyCompany), not by
the engine's phase, so the sync rule never takes a pass in one step for a
forced pass in another. Ruby's Track pass after a lay, where it keeps Track
open for MF's port tile, maps to no decision; the replay harness handles it
(``test_differential_replay.py``, engine-design.md §10.2).

The privates' actions map by Ruby's step too, as their non-blocking steps
join the step Ruby stops at: a ``buy_company`` to that step's buy-private
slot and then PRIVATE_PRICE (a grid point when the price lands on one, else
the exact id); DR's exchange (``buy_shares`` by company DR) to the DR slot;
MF's ``lay_tile`` to the MF slot and then PORT; and ER's ``lay_tile`` or
``pass`` to ER_BONUS. The engine offers DR and MF only on their owner's own
turns (engine-design §12 item 4), and the privates' slots not at Ruby's
Route step, which the engine runs by itself, so an exchange or port tile out
of turn, or anything at Route, raises ``UnsupportedAction``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.corp import CORPS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState

from tests.g1889.replay.snapshot import CORP, HEX, PRIVATE, TILE, TRAIN

Decision = tuple[int, int]
# Not decision phases: ``(ROUTE_OVERRIDE, revenue)`` is ``DRIVER.override_route_revenue``, and
# ``(HELD_DISCARD, NUM_TRAIN_TYPES * corp + type)`` a discard the harness holds for its corp's turn.
ROUTE_OVERRIDE = -2
HELD_DISCARD = -3


class UnsupportedAction(Exception):
    """An 18xx action the mapper cannot express yet, or one it rejects."""


class ReplayEnd(Exception):
    """``end_game``: Ruby ends the game by hand, which the engine does not model."""


# The slots that interject in several phases (§5.1).
BUY_PRIVATE_SLOT = {
    K.DPHASE_TRACK: K.TRACK_BUY_PRIVATE, K.DPHASE_TOKEN: K.TOKEN_BUY_PRIVATE,
    K.DPHASE_DIVIDEND: K.DIVIDEND_BUY_PRIVATE, K.DPHASE_BUY_TRAIN: K.BUY_TRAIN_BUY_PRIVATE,
    K.DPHASE_BUY_PRIVATE: K.BUY_PRIVATE_PRIVATE,
}
DR_SLOT = {
    K.DPHASE_STOCK: K.STOCK_DR, K.DPHASE_TRACK: K.TRACK_DR, K.DPHASE_TOKEN: K.TOKEN_DR,
    K.DPHASE_DIVIDEND: K.DIVIDEND_DR, K.DPHASE_BUY_TRAIN: K.BUY_TRAIN_DR, K.DPHASE_BUY_PRIVATE: K.BUY_PRIVATE_DR,
}
SELL_BLOCK = {K.DPHASE_STOCK: K.STOCK_SELL, K.DPHASE_BUY_TRAIN: K.BUY_TRAIN_SELL}
MF_SLOT = {K.DPHASE_STOCK: K.STOCK_MF, K.DPHASE_TRACK: K.TRACK_MF}
# The decision phase of a pass in each Ruby OR step.
OR_PASS_PHASE = {
    'Track': K.DPHASE_TRACK, 'Token': K.DPHASE_TOKEN, 'BuyTrain': K.DPHASE_BUY_TRAIN,
    'BuyCompany': K.DPHASE_BUY_PRIVATE,
}
# The decision phase each Ruby OR step's interjections join.
OR_STEP_PHASE = {**OR_PASS_PHASE, 'Dividend': K.DPHASE_DIVIDEND}


def _phase_name(phase: int) -> str:
    return K.DecisionPhase(phase).name.removeprefix('DPHASE_') if phase >= 0 else 'an automated step'


def _slot(table: dict[int, int], phase: int, what: str) -> int:
    if phase not in table:
        raise UnsupportedAction(f'{what} in {_phase_name(phase)}')
    return table[phase]


def _bid_id(first_point: int, points: int, exact: int, minimum: int, price: int) -> int:
    """A grid point when the price lands on one, else an exact id. Forced
    decisions are always grid points, so the sync rule can match them."""
    k, off = divmod(price - minimum, D.MIN_BID_INCREMENT)
    return first_point + k if off == 0 and 0 <= k < points else exact + price


def _bid(action, pre, phase, state) -> list[Decision]:
    price, minimum = pre['price'], pre['min_bid']
    if pre['auctioning'] is not None:
        if pre['company'] != pre['auctioning']:  # Ruby takes it as a bid outside the closed auction
            raise UnsupportedAction(f"bid on {pre['company']} during the closed auction for {pre['auctioning']}")
        return [(K.DPHASE_RAISE, _bid_id(K.RAISE_POINT, K.RAISE_POINTS, K.RAISE_EXACT, minimum, price))]
    if pre['purchase'] and price < minimum:  # Ruby does not check a purchase's price
        raise UnsupportedAction(f'bid: the topmost private bought for {price}, below its price {minimum}')
    if pre['purchase'] and price == minimum:
        return [(K.DPHASE_AUCTION, K.AUCTION_BUY)]
    # A bid on another private, or on the topmost above its price: a purchase at the bid.
    return [(K.DPHASE_AUCTION, K.AUCTION_BID + PRIVATE[pre['company']]),
            (K.DPHASE_BID, _bid_id(K.BID_POINT, K.BID_POINTS, K.BID_EXACT, minimum, price))]


def _pass(action, pre, phase, state) -> list[Decision]:
    if action['entity_type'] == 'company':  # only ER's seller passes for a private
        return [(K.DPHASE_ER_BONUS, K.ER_BONUS_PASS)]
    if pre.get('round') == 'OR':
        if pre['step'] not in OR_PASS_PHASE:
            raise UnsupportedAction(f"pass in Ruby's {pre['step']} step")
        return [(OR_PASS_PHASE[pre['step']], 0)]
    if phase not in K.PHASES_WITH_PASS_SLOT:
        raise UnsupportedAction(f'pass in {_phase_name(phase)}')
    return [(phase, 0)]


def _par(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_STOCK, K.STOCK_PAR + CORP[pre['corporation']]),
            (K.DPHASE_PAR, D.PAR_PRICE.index(pre['par_price']))]


def _interjection_phase(pre, what: str) -> int:
    """The decision a private's action joins: STOCK in an SR, else the one
    Ruby's OR step maps to."""
    if pre['round'] == 'SR':
        return K.DPHASE_STOCK
    if pre['round'] != 'OR' or pre['step'] not in OR_STEP_PHASE:
        raise UnsupportedAction(f"{what} in Ruby's {pre['round']} {pre['step']} step")
    return OR_STEP_PHASE[pre['step']]


def _own_turn(state, private: int, what: str) -> None:
    """DR and MF act only on their owner's own turn (engine-design §12 item
    4); replay fails loudly on Ruby's out-of-turn use."""
    if PRIVATES[private].owner(state) != K.OWNER_PLAYER + TURN.active_player(state):
        raise UnsupportedAction(f"{what} out of its owner's turn (engine-design §12 item 4)")


def _buy_shares(action, pre, phase, state) -> list[Decision]:
    if action['entity_type'] == 'company':  # DR's exchange
        phase = _interjection_phase(pre, "DR's exchange")
        _own_turn(state, D.PRIVATE_DR, "DR's exchange")
        return [(phase, _slot(DR_SLOT, phase, "DR's exchange"))]
    if pre['percent'] != 10 or pre['source'] not in ('ipo', 'pool'):
        raise UnsupportedAction(f"buy_shares of {pre['percent']}% from {pre['source']}")
    first = K.STOCK_IPO if pre['source'] == 'ipo' else K.STOCK_POOL
    return [(K.DPHASE_STOCK, first + CORP[pre['corporation']])]


def _sell_shares(action, pre, phase, state) -> list[Decision]:
    first = _slot(SELL_BLOCK, phase, 'sell_shares')
    return [(phase, first + K.MAX_SALE_SHARES * CORP[pre['corporation']] + pre['percent'] // 10 - 1)]


def _lay_tile(action, pre, phase, state) -> list[Decision]:
    """A corp's lay: TRACK at the hex, then TILE at the index of (tile,
    canonical rotation) among the corp's candidates there. ER's: ER_BONUS at
    the index among its candidates on C4. MF's: the MF slot, then PORT."""
    h, tile = HEX[pre['hex']], TILE[pre['tile']]
    lay = (tile, D.TILE_CANON_ROT[tile][pre['rotation']])
    if action['entity'] == 'ER':
        candidates = A.er_candidates_py(state)
        if h != D.ER_HEX or lay not in candidates:
            raise ValueError(f"ER's lay of {pre['tile']} on {pre['hex']} is not an engine candidate: {candidates}")
        return [(K.DPHASE_ER_BONUS, K.ER_BONUS_CANDIDATE + candidates.index(lay))]
    if action['entity'] == 'MF':
        phase = _interjection_phase(pre, "MF's port tile")
        _own_turn(state, D.PRIVATE_MF, "MF's port tile")
        port = (h, lay[1])
        if phase not in MF_SLOT or tile != D.PORT_TILE or port not in A.port_lays_py(state):
            raise UnsupportedAction(f"MF's lay of {pre['tile']} on {pre['hex']} in {_phase_name(phase)}")
        return [(phase, MF_SLOT[phase]), (K.DPHASE_PORT, K.PORT_LAY + D.NUM_SIDES * D.MF_PORT_HEX.index(h) + lay[1])]
    if action['entity_type'] != 'corporation':
        raise UnsupportedAction(f"lay_tile by private {action['entity']}")
    candidates = A.lay_candidates_py(state, h, CORP[action['entity']])
    if lay not in candidates:
        raise ValueError(f"Ruby's lay of {pre['tile']} on {pre['hex']} at rotation {pre['rotation']} "
                         f'is not an engine candidate: {candidates}')
    return [(K.DPHASE_TRACK, K.TRACK_HEX + h), (K.DPHASE_TILE, candidates.index(lay))]


def _place_token(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_TOKEN, K.TOKEN_HEX + HEX[pre['hex']])]


def _run_routes(action, pre, phase, state) -> list[Decision]:
    """Nothing when Ruby pays the engine's ROUTE revenue, else the override."""
    return [] if pre['revenue'] == TURN.route_revenue(state) else [(ROUTE_OVERRIDE, pre['revenue'])]


def _dividend(action, pre, phase, state) -> list[Decision]:
    kinds = {'payout': K.DIVIDEND_PAYOUT, 'withhold': K.DIVIDEND_WITHHOLD}
    if action['kind'] not in kinds:
        raise UnsupportedAction(f"dividend of kind {action['kind']}")
    return [(K.DPHASE_DIVIDEND, kinds[action['kind']])]


def _discard_train(action, pre, phase, state) -> list[Decision]:
    corp, train = CORP[action['entity']], TRAIN[pre['train_type']]
    crowded = TRAINS.first_crowded(state) if TURN.step(state) == K.STEP_DISCARD else -1
    if crowded >= 0 and crowded != corp:  # EQUIVALENT: Ruby takes discards in any order
        return [(HELD_DISCARD, D.NUM_TRAIN_TYPES * corp + train)]
    return [(K.DPHASE_DISCARD, train)]


def _train_price_id(state, corp: int, train: int, price: int) -> int:
    """TRAIN_PRICE's grid point for ``price`` (¥1, list, ⌈cash/3⌉, ⌈2·cash/3⌉,
    the buyer's cash) when it lands on one, else the exact id."""
    cash = CORPS[corp].cash(state)
    points = [1, D.TRAIN_PRICE[train], -(-cash // 3), -(-2 * cash // 3), cash]
    return points.index(price) if price in points else K.TRAIN_PRICE_EXACT + price


def _buy_train(action, pre, phase, state) -> list[Decision]:
    """By source: the depot or pool at the list price, a D for a traded 4, 5,
    or 6 at ¥800, or another corp's train at its price (TRAIN_PRICE). Ruby
    checks no depot or pool price (``check_spend`` admits ¥1 to the cash),
    so any other price is rejected."""
    train, price = TRAIN[pre['train_type']], pre['price']
    if pre['source'] == 'corporation':
        corp = CORP[action['entity']]
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_CORP + D.NUM_TRAIN_TYPES * CORP[pre['seller']] + train),
                (K.DPHASE_TRAIN_PRICE, _train_price_id(state, corp, train, price))]
    if 'exchange_type' in pre:
        traded = TRAIN[pre['exchange_type']]
        expected, decision = D.TRAIN_PRICE[train] - D.TRAIN_TRADE_IN[train][traded], \
            (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_TRADE_IN + D.TRADE_IN_TYPE.index(traded))
    else:
        expected = D.TRAIN_PRICE[train]
        first = K.BUY_TRAIN_POOL if pre['source'] == 'pool' else K.BUY_TRAIN_DEPOT
        decision = (K.DPHASE_BUY_TRAIN, first + train)
    if price != expected:
        raise UnsupportedAction(f"buy_train of a {pre['train_type']} from the {pre['source']} for {price}, "
                                f'not {expected}')
    return [decision]


def _private_price_id(state, corp: int, private: int, price: int) -> int:
    """PRIVATE_PRICE's grid point for ``price`` (⌈face/2⌉, face, 1½·face, the
    most the corp may pay, each kept only inside the legal range and before
    an equal one) when it lands on one, else the exact id."""
    face = D.PRIVATE_VALUE[private]
    low, high = -(-face // 2), min(2 * face, CORPS[corp].cash(state))
    points = [low, face, -(-3 * face // 2), high]
    for k, point in enumerate(points):
        if low <= point <= high and point not in points[:k] and point == price:
            return k
    return K.PRIVATE_PRICE_EXACT + price


def _buy_company(action, pre, phase, state) -> list[Decision]:
    """The buy-private slot of the decision Ruby's step maps to, then the price."""
    phase = _interjection_phase(pre, 'buy_company')
    p = PRIVATE[pre['company']]
    return [(phase, _slot(BUY_PRIVATE_SLOT, phase, 'buy_company') + p),
            (K.DPHASE_PRIVATE_PRICE, _private_price_id(state, CORP[action['entity']], p, pre['price']))]


def _bankrupt(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_BANKRUPT)]


def _end_game(action, pre, phase, state) -> list[Decision]:
    raise ReplayEnd('end_game')


MAPPERS: dict[str, Callable[..., list[Decision]]] = {
    'bid': _bid, 'pass': _pass, 'par': _par, 'buy_shares': _buy_shares, 'sell_shares': _sell_shares,
    'lay_tile': _lay_tile, 'place_token': _place_token, 'run_routes': _run_routes, 'dividend': _dividend,
    'discard_train': _discard_train, 'buy_train': _buy_train, 'buy_company': _buy_company,
    'bankrupt': _bankrupt, 'end_game': _end_game,
}


def map_action(action: dict[str, Any], pre: dict[str, Any], phase: int, state: GameState) -> list[Decision]:
    """The engine decisions for one Ruby action taken in decision ``phase``."""
    kind = action['type']
    if kind == 'message' or kind.startswith('program_'):
        return []
    if kind not in MAPPERS:
        raise UnsupportedAction(f'18xx action {kind!r} is not supported')
    return MAPPERS[kind](action, pre, phase, state)
