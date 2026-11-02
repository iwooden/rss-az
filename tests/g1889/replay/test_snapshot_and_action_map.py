"""snapshot.compare/load_state and action_map.map_action against the Ruby extracts."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout

from tests.g1889.replay.action_map import ReplayEnd, UnsupportedAction, map_action
from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.snapshot import CORP, HEX, PRIVATE, TRAIN, compare, load_state
from tests.g1889.replay.test_differential_replay import new_game

FIELDS = get_global_fields()
CACHE_SLOTS = [get_layout(D.MAX_PLAYERS).global_offset + f for f in (FIELDS.conn_dirty, FIELDS.route_dirty)]


def extracts():
    return [load_fixture_extract(f) for f in FIXTURES] + [load_new_game_extract(n) for n in range(2, 7)]


def load(snap, max_players=0):
    return load_state(snap, len(snap['players']), max_players)


# load_state


def test_load_state_rebuilds_initial_snapshots():
    for extract in extracts():
        snap = extract['records'][0]['state']
        loaded, fresh = load(snap), new_game(extract)
        assert compare(loaded, snap) == []
        # Everything else equals initialize_game's row; load_state leaves the caches dirty.
        assert all(loaded._array[s] == (1 << D.NUM_CORPS) - 1 for s in CACHE_SLOTS)
        loaded._array[CACHE_SLOTS] = 0
        np.testing.assert_array_equal(loaded._array, fresh._array)


def test_load_state_rebuilds_every_auction_snapshot():
    count = 0
    for fixture in FIXTURES:
        for record in load_fixture_extract(fixture)['records']:
            snap = record['state']
            if snap['game']['round'] == 'ISR' and not snap['game']['finished']:
                assert compare(load(snap, max_players=D.MAX_PLAYERS), snap) == [], (fixture, record['index'])
                count += 1
    assert count == 34


@pytest.mark.parametrize('round_name', ['SR', 'OR'])
def test_load_state_defers_stock_and_operating_rounds(round_name):
    records = load_fixture_extract('314')['records']
    snap = next(r['state'] for r in records if r['state']['game']['round'] == round_name)
    with pytest.raises(NotImplementedError, match=round_name):
        load(snap)


def test_load_state_checks_cash():
    snap = copy.deepcopy(load_fixture_extract('962')['records'][0]['state'])
    snap['game']['bank_cash'] += 5
    with pytest.raises(ValueError, match='conserved'):
        load(snap)


# compare


def test_compare_reports_each_difference():
    extract = load_fixture_extract('314')
    snap = extract['records'][0]['state']
    state = new_game(extract)
    BANK.transfer(state, 1, -1, 25)
    PLAYERS[2].set_bid(state, D.PRIVATE_ER, 45)
    CORPS[D.CORP_KU].set_tokens_left(state, 0)
    PRIVATES[D.PRIVATE_SIR].set_owner(state, K.OWNER_PLAYER)
    MAP.set_supply(state, 0, 1)
    TRAINS.set_depot(state, D.TRAIN_D, 1)
    TURN.set_auction_private(state, D.PRIVATE_TR)
    paths = {diff.split(':')[0] for diff in compare(state, snap)}
    assert paths == {
        'players[1].cash', 'game.bank_cash', 'players[2].bids', 'corporations.KU.tokens_left',
        'privates.SIR.owner', 'players[0].privates', 'tiles', 'depot', 'auction.companies', 'auction.auctioning',
        'auction.min_bids',
    }


def _with_tokens_and_market(snap, tokens, stacks):
    """A copy of an initial snapshot with AR and IR tokened on K8 and parred at ¥70 in one cell."""
    snap = copy.deepcopy(snap)
    snap['hexes']['K8'].update(tokens=tokens, reserved=[])
    for sym, stack in zip(('AR', 'IR'), stacks):
        corp = snap['corporations'][sym]
        corp.update(ipoed=True, par=70, price=70, market=[4, 3, stack], token_hexes=['K8'],
                    tokens_left=corp['tokens_left'] - 1)
    return snap


def test_compare_equivalences():
    snap = load_fixture_extract('962')['records'][0]['state']
    # Rotations with the same paths: tile 9 is a straight, so 3 is 0.
    straight = copy.deepcopy(snap)
    straight['hexes']['J3'].update(tile='9', rotation=3)
    state = load(straight)
    assert MAP.rotation(state, HEX['J3']) == 0
    straight['hexes']['J3']['rotation'] = 0
    assert compare(state, straight) == []
    straight['hexes']['J3']['rotation'] = 1
    assert compare(state, straight) == ['hexes.J3.rotation: engine 0 != ruby 1']
    # Tokens by slot are a set; market stacks follow arrival.
    stacked = _with_tokens_and_market(snap, ['AR', 'IR'], [0, 1])
    state = load(stacked)
    assert compare(state, _with_tokens_and_market(snap, ['IR', 'AR'], [0, 1])) == []
    swapped = {d.split(':')[0] for d in compare(state, _with_tokens_and_market(snap, ['AR', 'IR'], [1, 0]))}
    assert swapped == {'corporations.AR.market', 'corporations.IR.market'}


# map_action

RUBY_STEP_PHASE = {
    'BuySellParShares': K.DPHASE_STOCK, 'Track': K.DPHASE_TRACK, 'Token': K.DPHASE_TOKEN,
    'Dividend': K.DPHASE_DIVIDEND, 'BuyTrain': K.DPHASE_BUY_TRAIN, 'DiscardTrain': K.DPHASE_DISCARD,
    'BuyCompany': K.DPHASE_BUY_PRIVATE, 'SpecialTrack': K.DPHASE_ER_BONUS, 'Route': -1,
}


def _ruby_phase(pre, before):
    if pre['step'] == 'WaterfallAuction':
        return K.DPHASE_RAISE if before['auction']['auctioning'] else K.DPHASE_AUCTION
    return RUBY_STEP_PHASE[pre['step']]


def _expected(kind, pre, decoded):
    """The decoded decisions agree with the annotation."""
    last = decoded[-1]
    if kind == 'bid':
        if pre['purchase'] and pre['price'] == pre['min_bid']:
            return last.type == A.ACTION_BUY_TOPMOST
        # Grid point k bids the minimum + ¥5k.
        point = pre['min_bid'] + D.MIN_BID_INCREMENT * last.count
        amount = last.amount if last.type == A.ACTION_EXACT_PRICE else point
        return amount == pre['price'] and (pre['auctioning'] or decoded[0].private == PRIVATE[pre['company']])
    if kind == 'par':
        return decoded[0].corp == CORP[pre['corporation']] and last.amount == pre['par_price']
    if kind == 'buy_shares':
        return last.corp == CORP[pre['corporation']]
    if kind == 'sell_shares':
        return last.corp == CORP[pre['corporation']] and 10 * last.count == pre['percent']
    if kind == 'place_token':
        return last.hex == HEX[pre['hex']]
    if kind in ('buy_train', 'discard_train'):
        return decoded[0].train_type in (TRAIN[pre['train_type']], TRAIN.get(pre.get('exchange_type')))
    if kind == 'buy_company':
        return decoded[0].private == PRIVATE[pre['company']] and last.amount == pre['price']
    return True


def test_map_action_covers_the_fixtures():
    scratch = GameState(3)
    mapped, unsupported, ended = 0, {}, 0
    for fixture in FIXTURES:
        records = load_fixture_extract(fixture)['records']
        for before, record in zip(records, records[1:]):
            host = record['action']
            autos = zip(host.get('auto_actions') or [], record.get('auto_pre') or [])
            for action, pre in [(host, record['pre']), *autos]:
                kind = action['type']
                if kind == 'run_routes':
                    TURN.set_route_revenue(scratch, pre['revenue'])
                try:
                    decisions = map_action(action, pre, _ruby_phase(pre, before['state']), scratch)
                except UnsupportedAction:
                    key = (kind, action['entity_type'])
                    unsupported[key] = unsupported.get(key, 0) + 1
                    continue
                except ReplayEnd:
                    assert kind == 'end_game'
                    ended += 1
                    continue
                decoded = [A.decode_action_py(phase, action_id) for phase, action_id in decisions]
                assert not decoded or _expected(kind, pre, decoded), (fixture, record['index'], kind, decoded)
                mapped += 1
    # Only tile lays are left: 91 by corps (6.5), MF's port lay and ER's bonus lay (6.8).
    assert unsupported == {('lay_tile', 'corporation'): 91, ('lay_tile', 'company'): 2}
    assert (mapped, ended) == (914, 2)


def test_map_action_skips_and_rejects():
    scratch = GameState(3)
    assert map_action({'type': 'message'}, {}, K.DPHASE_STOCK, scratch) == []
    assert map_action({'type': 'program_share_pass'}, {}, K.DPHASE_STOCK, scratch) == []
    with pytest.raises(UnsupportedAction, match="'undo'"):
        map_action({'type': 'undo'}, {}, K.DPHASE_STOCK, scratch)
    with pytest.raises(UnsupportedAction, match='pass in PAR'):
        map_action({'type': 'pass', 'entity_type': 'player'}, {}, K.DPHASE_PAR, scratch)
    TURN.set_route_revenue(scratch, 100)
    with pytest.raises(UnsupportedAction, match='6.6'):
        map_action({'type': 'run_routes'}, {'revenue': 120}, -1, scratch)
