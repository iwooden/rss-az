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

from tests.g1889.replay.action_map import HELD_DISCARD, ROUTE_OVERRIDE, ReplayEnd, UnsupportedAction, map_action
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


def test_load_state_rebuilds_every_stock_round_snapshot():
    count = turns = 0
    for fixture in FIXTURES:
        for record in load_fixture_extract(fixture)['records']:
            snap = record['state']
            if snap['game']['round'] == 'SR' and snap['game']['step'] == 'BuySellParShares':
                state = load(snap, max_players=D.MAX_PLAYERS)
                assert compare(state, snap) == [], (fixture, record['index'])
                count += 1
                turns += bool(snap['stock']['current_actions'])
    assert (count, turns) == (522, 179)  # 179 mid-turn: after a purchase or a sale


def test_load_state_rebuilds_every_operating_round_snapshot():
    """Including Ruby's oracles, which compare checks in ORs: the network,
    BUY_TRAIN's choices at BuyTrain snapshots, and at DiscardTrain snapshots
    the crowded corps, the first of which is Ruby's current entity and
    discards in DISCARD, returning to BUY_TRAIN."""
    count = discards = 0
    for fixture in FIXTURES:
        for record in load_fixture_extract(fixture)['records']:
            snap = record['state']
            if snap['game']['round'] != 'OR' or snap['game']['finished']:
                continue
            state = load(snap, max_players=D.MAX_PLAYERS)
            assert compare(state, snap) == [], (fixture, record['index'])
            if snap['game']['step'] == 'DiscardTrain':
                assert TURN.return_step(state) == K.STEP_BUY_TRAIN and TURN.operating_corp(state) == CORP['AR']
                discards += 1
            count += 1
    assert (count, discards) == (444, 2)


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
        'players[1].cash', 'players[1].value', 'game.bank_cash', 'players[2].bids', 'corporations.KU.tokens_left',
        'privates.SIR.owner', 'players[0].privates', 'players[0].value', 'tiles', 'depot', 'auction.companies',
        'auction.auctioning', 'auction.min_bids',
    }


def test_compare_checks_the_game_end_and_emergency_sales():
    """314's last record: TR's president is bankrupt, and the game over. It
    loads with the OR's fields Ruby keeps, and compares the result, each
    player's value, and the turn's emergency sales (KO's)."""
    snap = load_fixture_extract('314')['records'][-1]['state']
    assert snap['game']['finished'] and snap['operating']['ebuy_sold'] == ['KO']
    state = load(snap)
    assert compare(state, snap) == []
    assert TURN.operating_corp(state) == CORP['TR'] and TURN.ebuy_sold_mask(state) == 1 << CORP['KO']
    TURN.set_ebuy_sold_any(state, 0)
    PLAYERS[0].set_bankrupt(state, False)
    BANK.transfer(state, -1, K.OWNER_PLAYER + 2, 10)
    BANK.transfer(state, K.OWNER_CORP + CORP['AR'], -1, 10)
    paths = {diff.split(':')[0] for diff in compare(state, snap)}
    assert paths == {'operating.ebuy_sold_any', 'players[0].bankrupt', 'players[2].cash', 'players[2].value',
                     'game.result', 'corporations.AR.cash'}
    # Between turns Ruby keeps the last turn's sales until the next starts; the engine has cleared them.
    between = copy.deepcopy(snap)
    between['operating'].update(operator=None, ebuy_sold=['UR'])
    TURN.set_ebuy_sold_any(state, 1)
    assert 'operating.ebuy_sold' not in {diff.split(':')[0] for diff in compare(state, between)}


def _mid_turn_snapshot():
    """314 record 104: player 90 bought SR from the IPO, then sold one SR."""
    snap = load_fixture_extract('314')['records'][104]['state']
    assert snap['stock']['current_actions'] == ['buy_shares', 'sell_shares']
    return snap


def test_compare_checks_the_stock_turn():
    snap = _mid_turn_snapshot()
    state = load(snap)
    assert (TURN.sr_bought(state), TURN.sr_sold_mask(state), TURN.sr_sold_then_bought(state)) == \
        (1, 1 << D.CORP_SR, 0)
    TURN.set_sr_sold_then_bought(state, 1)
    TURN.set_sr_sold_mask(state, 1 << D.CORP_UR)
    paths = {diff.split(':')[0] for diff in compare(state, snap)}
    assert paths == {'stock.sold_now.90', 'stock.current_actions (sold, then bought)', 'stock.legal.sell'}


def test_compare_checks_legality_against_ruby():
    snap = _mid_turn_snapshot()
    state = load(snap)
    for key, value in (('actions', ['sell_shares']), ('buy_pool', ['UR']), ('sell', {'SR': [10]})):
        wrong = copy.deepcopy(snap)
        wrong['stock']['legal'][key] = value
        assert [d.split(':')[0] for d in compare(state, wrong)] == \
            [{'actions': 'stock.legal.pass', 'buy_pool': 'stock.legal.buy_pool', 'sell': 'stock.legal.sell'}[key]]


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
    if kind == 'lay_tile' and decoded[0].phase == K.DPHASE_ER_BONUS:  # ER's seller, on C4
        return pre['hex'] == 'C4' and len(decoded) == 1 and decoded[0].type == A.ACTION_LAY_CANDIDATE
    if kind == 'lay_tile' and decoded[0].type == A.ACTION_MF_PORT:  # MF's port tile
        return last.phase == K.DPHASE_PORT and last.hex == HEX[pre['hex']] and pre['tile'] == '437'
    if kind == 'lay_tile':
        return decoded[0].hex == HEX[pre['hex']] and last.phase == K.DPHASE_TILE
    if kind == 'discard_train':
        return decoded[0].train_type == TRAIN[pre['train_type']]
    if kind == 'buy_train':
        if decoded[0].type == A.ACTION_TRADE_IN:
            return decoded[0].train_type == TRAIN[pre['exchange_type']]
        if decoded[0].type == A.ACTION_BUY_CORP_TRAIN:  # the price, a grid point or exact, is checked below
            return decoded[0].corp == CORP[pre['seller']] and decoded[0].train_type == TRAIN[pre['train_type']]
        return decoded[0].train_type == TRAIN[pre['train_type']]
    if kind == 'buy_company':  # the price, a grid point or exact, is checked below
        return decoded[0].private == PRIVATE[pre['company']] and last.phase == K.DPHASE_PRIVATE_PRICE
    return True


def test_map_action_covers_the_fixtures():
    """A corp's lay maps against the state before it, where TILE's index
    names Ruby's tile and canonical rotation among the engine's candidates;
    so do ER's and MF's lays (ER_BONUS's candidates, PORT's hexes and
    rotations) and a private's price (PRIVATE_PRICE's grid or exact id)."""
    scratch = GameState(3)
    mapped, unsupported, ended, grid, held = 0, {}, 0, {True: 0, False: 0}, 0
    private_grid, privates = {True: 0, False: 0}, {'company': 0, 'corporation': 0}
    for fixture in FIXTURES:
        records = load_fixture_extract(fixture)['records']
        for before, record in zip(records, records[1:]):
            host = record['action']
            autos = zip(host.get('auto_actions') or [], record.get('auto_pre') or [])
            for action, pre in [(host, record['pre']), *autos]:
                kind = action['type']
                state = scratch
                if kind == 'run_routes':
                    TURN.set_route_revenue(scratch, pre['revenue'])
                elif kind in ('lay_tile', 'buy_train', 'discard_train', 'buy_company'):
                    state = load(before['state'])  # candidates, the buyer's cash, the discarding corp, owners
                try:
                    decisions = map_action(action, pre, _ruby_phase(pre, before['state']), state)
                except UnsupportedAction:
                    key = (kind, action['entity_type'])
                    unsupported[key] = unsupported.get(key, 0) + 1
                    continue
                except ReplayEnd:
                    assert kind == 'end_game'
                    ended += 1
                    continue
                if decisions and decisions[0][0] == HELD_DISCARD:  # 314 record 339: UR's waits for SR's
                    assert decisions == [(HELD_DISCARD, D.NUM_TRAIN_TYPES * CORP[action['entity']]
                                          + TRAIN[pre['train_type']])]
                    held += 1
                    mapped += 1
                    continue
                decoded = [A.decode_action_py(phase, action_id) for phase, action_id in decisions]
                assert not decoded or _expected(kind, pre, decoded), (fixture, record['index'], kind, decoded)
                if kind == 'lay_tile':
                    tile = D.TILE_NAMES.index(pre['tile'])
                    lay = (tile, D.TILE_CANON_ROT[tile][pre['rotation']])
                    if decoded[0].phase == K.DPHASE_ER_BONUS:
                        assert A.er_candidates_py(state)[decoded[0].count] == lay
                    elif decoded[0].type == A.ACTION_MF_PORT:
                        assert (HEX[pre['hex']], lay[1]) in A.port_lays_py(state) and lay[0] == D.PORT_TILE
                    else:
                        candidates = A.lay_candidates_py(state, HEX[pre['hex']], CORP[action['entity']])
                        assert candidates[decoded[1].count] == lay
                    privates[action['entity_type']] += action['entity_type'] == 'company'
                elif kind == 'buy_company':
                    TURN.set_step(state, K.STEP_PRIVATE_PRICE)
                    TURN.set_pending_private(state, decoded[0].private)
                    assert A.price_amount_py(state, decisions[1][1]) == pre['price']
                    private_grid[decoded[1].type == A.ACTION_PRICE_POINT] += 1
                elif kind == 'buy_train' and decoded[0].type == A.ACTION_BUY_CORP_TRAIN:
                    TURN.set_step(state, K.STEP_TRAIN_PRICE)
                    TURN.set_pending_train_source(state, decisions[0][1] - K.BUY_TRAIN_CORP)
                    assert A.price_amount_py(state, decisions[1][1]) == pre['price']
                    grid[decoded[1].type == A.ACTION_PRICE_POINT] += 1
                mapped += 1
    assert unsupported == {}
    assert (mapped, ended) == (1007, 2)
    # Corp train prices on the grid: ¥1 three times, the buyer's whole treasury
    # three times (¥360, ¥370, ¥450), and ⌈2·cash/3⌉ of ¥2 (314 record 350);
    # ¥100 twice, ¥150, and ¥165 are exact. 314 record 339's discard waits.
    assert (grid, held) == ({True: 7, False: 4}, 1)
    # Every private sale is at twice the face value, the top grid point; MF's
    # port tile and ER's bonus are the two lays by a private.
    assert (private_grid, privates['company']) == ({True: 13, False: 0}, 2)


def test_map_action_skips_and_rejects():
    scratch = GameState(3)
    assert map_action({'type': 'message'}, {}, K.DPHASE_STOCK, scratch) == []
    assert map_action({'type': 'program_share_pass'}, {}, K.DPHASE_STOCK, scratch) == []
    with pytest.raises(UnsupportedAction, match="'undo'"):
        map_action({'type': 'undo'}, {}, K.DPHASE_STOCK, scratch)
    with pytest.raises(UnsupportedAction, match='pass in PAR'):
        map_action({'type': 'pass', 'entity_type': 'player'}, {}, K.DPHASE_PAR, scratch)
    # An OR pass is the pass of Ruby's step, whatever phase the engine is in.
    corp_pass = {'type': 'pass', 'entity_type': 'corporation'}
    assert map_action(corp_pass, {'round': 'OR', 'step': 'BuyTrain'}, K.DPHASE_TOKEN, scratch) == \
        [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)]
    with pytest.raises(UnsupportedAction, match='Dividend'):
        map_action(corp_pass, {'round': 'OR', 'step': 'Dividend'}, K.DPHASE_DIVIDEND, scratch)
    # The privates' slots join Ruby's step; the engine runs Route by itself.
    with pytest.raises(UnsupportedAction, match='Route'):
        map_action({'type': 'buy_company', 'entity_type': 'corporation', 'entity': 'KO'},
                   {'round': 'OR', 'step': 'Route', 'company': 'SIR', 'price': 80}, K.DPHASE_DIVIDEND, scratch)
    # DR and MF act only on their owner's turn (engine-design §12 item 4).
    PRIVATES[D.PRIVATE_DR].set_owner(scratch, K.OWNER_PLAYER + 1)
    TURN.set_active_player(scratch, 0)
    with pytest.raises(UnsupportedAction, match='out of its owner'):
        map_action({'type': 'buy_shares', 'entity_type': 'company', 'entity': 'DR'},
                   {'round': 'SR', 'step': 'BuySellParShares'}, K.DPHASE_STOCK, scratch)
    # run_routes: nothing when Ruby pays the engine's revenue, else the override.
    TURN.set_route_revenue(scratch, 100)
    assert map_action({'type': 'run_routes'}, {'revenue': 100}, K.DPHASE_DIVIDEND, scratch) == []
    assert map_action({'type': 'run_routes'}, {'revenue': 120}, K.DPHASE_DIVIDEND, scratch) == [(ROUTE_OVERRIDE, 120)]
