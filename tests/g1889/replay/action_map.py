"""Map 18xx.games actions to engine decisions (1889/engine-design.md §5.5).

``map_action(action, pre, phase, state)`` turns one Ruby action (a host action
or one of its ``auto_actions``) into the engine decisions that perform it: a
list of ``(decision phase, phase-local id)``, starting in ``phase``. It reads
the extractor's ``pre`` annotation for what the log entry leaves implicit (a
bid's purchase or closed auction, a share's source, a token's hex, a train's
owner). Prices are exact ids, so any price a human chose is expressible;
bids that land on the BID or RAISE grid use the grid point instead.

Actions with no engine decision map to ``[]``: ``message``, ``program_*``,
and ``run_routes`` whose revenue the engine's ROUTE step already matches.
``end_game`` raises ``ReplayEnd``. An action the mapper cannot express yet
raises ``UnsupportedAction`` naming its type.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.turn import TURN
from g1889.state import GameState

from tests.g1889.replay.snapshot import CORP, HEX, PRIVATE, TRAIN

Decision = tuple[int, int]


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
    if phase not in K.PHASES_WITH_PASS_SLOT:
        raise UnsupportedAction(f'pass in {_phase_name(phase)}')
    return [(phase, 0)]


def _par(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_STOCK, K.STOCK_PAR + CORP[pre['corporation']]),
            (K.DPHASE_PAR, D.PAR_PRICE.index(pre['par_price']))]


def _buy_shares(action, pre, phase, state) -> list[Decision]:
    if action['entity_type'] == 'company':  # DR's exchange
        return [(phase, _slot(DR_SLOT, phase, 'the DR exchange'))]
    if pre['percent'] != 10 or pre['source'] not in ('ipo', 'pool'):
        raise UnsupportedAction(f"buy_shares of {pre['percent']}% from {pre['source']}")
    first = K.STOCK_IPO if pre['source'] == 'ipo' else K.STOCK_POOL
    return [(K.DPHASE_STOCK, first + CORP[pre['corporation']])]


def _sell_shares(action, pre, phase, state) -> list[Decision]:
    first = _slot(SELL_BLOCK, phase, 'sell_shares')
    return [(phase, first + K.MAX_SALE_SHARES * CORP[pre['corporation']] + pre['percent'] // 10 - 1)]


def _lay_tile(action, pre, phase, state) -> list[Decision]:
    entity = action['entity']
    if action['entity_type'] == 'corporation':
        raise UnsupportedAction('lay_tile by a corporation needs enumerate_lay_candidates (6.5)')
    raise UnsupportedAction(f'lay_tile by private {entity} (6.8)')


def _place_token(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_TOKEN, K.TOKEN_HEX + HEX[pre['hex']])]


def _run_routes(action, pre, phase, state) -> list[Decision]:
    if pre['revenue'] != TURN.route_revenue(state):
        raise UnsupportedAction(f"run_routes for {pre['revenue']} against the engine's "
                                f'{TURN.route_revenue(state)} needs override_route_revenue (6.6)')
    return []


def _dividend(action, pre, phase, state) -> list[Decision]:
    kinds = {'payout': K.DIVIDEND_PAYOUT, 'withhold': K.DIVIDEND_WITHHOLD}
    if action['kind'] not in kinds:
        raise UnsupportedAction(f"dividend of kind {action['kind']}")
    return [(K.DPHASE_DIVIDEND, kinds[action['kind']])]


def _discard_train(action, pre, phase, state) -> list[Decision]:
    return [(K.DPHASE_DISCARD, TRAIN[pre['train_type']])]


def _buy_train(action, pre, phase, state) -> list[Decision]:
    train = TRAIN[pre['train_type']]
    if pre['source'] == 'corporation':
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_CORP + D.NUM_TRAIN_TYPES * CORP[pre['seller']] + train),
                (K.DPHASE_TRAIN_PRICE, K.TRAIN_PRICE_EXACT + pre['price'])]
    if pre['source'] == 'pool':
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_POOL + train)]
    if 'exchange_type' in pre:
        return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_TRADE_IN + D.TRADE_IN_TYPE.index(TRAIN[pre['exchange_type']]))]
    return [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_DEPOT + train)]


def _buy_company(action, pre, phase, state) -> list[Decision]:
    first = _slot(BUY_PRIVATE_SLOT, phase, 'buy_company')
    return [(phase, first + PRIVATE[pre['company']]), (K.DPHASE_PRIVATE_PRICE, K.PRIVATE_PRICE_EXACT + pre['price'])]


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
