"""The Ruby state extractor (extract_states.rb) and the schema extracts.py documents.

Runs the extractor on the four 1889 fixtures and on fresh 2-6 player games,
then checks its output against the fixture files, the tables, and itself. It
never imports the Cython engine.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from g1889.data import load_tables
from tests.g1889.replay.extracts import (
    AUCTION_KEYS, CORPORATION_KEYS, FIXTURES, GAME_KEYS, HEX_KEYS, OPERATING_KEYS, PLAYER_KEYS, PRE_EXTRA_KEYS,
    PRE_KEYS, PRE_OPTIONAL_KEYS, PRIVATE_KEYS, STATE_KEYS, STOCK_KEYS, Extract, load_fixture, load_fixture_extract,
    load_new_game_extract,
)

T = load_tables()
BANK_CASH = T['rules']['BANK_CASH']
HEX_IDS = [h['id'] for h in T['hexes']]
CORP_SYMS = [c['sym'] for c in T['corporations']]
PRIVATE_SYMS = [p['sym'] for p in T['privates']]
TILE_NAMES = {t['name'] for t in T['tiles']}
SUPPLY_TILES = {t['name']: t['count'] for t in T['tiles'] if t['hex'] < 0}
TRAIN_COUNTS = {t['name']: t['count'] for t in T['trains']}
ROUND_DETAIL = {'ISR': 'auction', 'SR': 'stock', 'OR': 'operating'}

State = dict[str, Any]


@pytest.fixture(scope='module', params=FIXTURES)
def case(request) -> tuple[str, dict[str, Any], Extract]:
    name = request.param
    return name, load_fixture(name), load_fixture_extract(name)


def states(extract: Extract) -> list[State]:
    found = [r['state'] for r in extract['records']]
    if 'full_load' in extract:
        found.append(extract['full_load']['state'])
    return found


def check_state(state: State, num_players: int) -> None:
    game = state['game']
    assert state.keys() == STATE_KEYS
    assert game.keys() == GAME_KEYS

    assert len(state['players']) == num_players
    for player in state['players']:
        assert player.keys() == PLAYER_KEYS
        assert list(player['shares']) == [c for c in CORP_SYMS if c in player['shares']]
        assert all(s.keys() == {'percent', 'president'} for s in player['shares'].values())
        assert set(player['privates']) | set(player['bids']) <= set(PRIVATE_SYMS)

    assert list(state['corporations']) == CORP_SYMS
    for corp in state['corporations'].values():
        assert corp.keys() == CORPORATION_KEYS
        assert set(corp['trains']) <= TRAIN_COUNTS.keys()
        assert sum(corp['trains'].values()) == len(corp['train_ids'])
        assert set(corp['token_hexes']) <= set(HEX_IDS)

    assert list(state['privates']) == [p for p in PRIVATE_SYMS if p in state['privates']]
    assert all(p.keys() == PRIVATE_KEYS for p in state['privates'].values())

    assert state['hexes'].keys() == set(HEX_IDS)
    for hex_ in state['hexes'].values():
        assert hex_.keys() == HEX_KEYS
        assert hex_['tile'] in TILE_NAMES
        assert set(hex_['tokens']) | set(hex_['reserved']) <= set(CORP_SYMS)

    assert state['tiles'].keys() == SUPPLY_TILES.keys()
    assert set(state['depot']) | set(state['pool']) <= TRAIN_COUNTS.keys()

    detail = ROUND_DETAIL[game['round']]
    for key, keys in (('auction', AUCTION_KEYS), ('stock', STOCK_KEYS), ('operating', OPERATING_KEYS)):
        if key == detail:
            assert state[key].keys() == keys
        else:
            assert state[key] is None

    assert (game['result'] is not None) == game['finished']


def check_cash(state: State) -> None:
    held = sum(p['cash'] for p in state['players']) + sum(c['cash'] for c in state['corporations'].values())
    assert held + state['game']['bank_cash'] == BANK_CASH


def test_one_record_per_surviving_action(case):
    _, game, extract = case
    records = extract['records']
    actions = [r['action'] for r in records[1:]]
    ids = [a['id'] for a in actions]

    assert extract['raw_action_count'] == len(game['actions'])
    assert len(records) == extract['surviving_action_count'] + 1
    assert records[0]['action'] is None
    assert [r['index'] for r in records] == list(range(len(records)))
    assert ids == sorted(set(ids))
    assert not {a['type'] for a in actions} & {'undo', 'redo'}
    # Engine::Game.load's own resolution.
    assert ids == extract['full_load']['processed_action_ids']
    by_id = {a['id']: a for a in game['actions']}
    assert all(a == by_id[a['id']] for a in actions)


def test_undo_and_redo_resolve_as_the_fixture_marks_them():
    # 962 marks each undo, and each action it undid, with "skip".
    game, extract = load_fixture('962'), load_fixture_extract('962')
    assert Counter(a['type'] for a in game['actions'])['undo'] == 38
    assert extract['surviving_action_count'] == 446
    assert [r['action']['id'] for r in extract['records'][1:]] == [a['id'] for a in game['actions'] if not a.get('skip')]


def test_snapshots_and_annotations_have_the_documented_fields(case):
    _, game, extract = case
    for state in states(extract):
        check_state(state, len(game['players']))

    for record in extract['records'][1:]:
        actions = [record['action'], *record['action'].get('auto_actions', [])]
        pres = [record['pre'], *record.get('auto_pre', [])]
        assert len(pres) == len(actions)
        for pre, action in zip(pres, actions):
            assert pre['type'] == action['type']
            required = PRE_KEYS | PRE_EXTRA_KEYS.get(pre['type'], frozenset())
            assert required <= pre.keys() <= required | PRE_OPTIONAL_KEYS.get(pre['type'], frozenset())


def test_cash_is_conserved(case):
    for state in states(case[2]):
        check_cash(state)


def test_final_result_matches_the_fixture(case):
    _, game, extract = case
    final = extract['records'][-1]['state']['game']
    assert final['finished']
    assert final['result'] == game['result']
    assert final['end_reason'] == game['game_end_reason']


def test_stepwise_replay_matches_a_full_load(case):
    extract = case[2]
    assert extract['records'][-1]['state'] == extract['full_load']['state']


def test_annotations_agree_with_the_states(case):
    extract = case[2]
    records = extract['records']
    for before, record in zip(records, records[1:]):
        action, pre = record['action'], record['pre']
        old, new = before['state'], record['state']
        if action.get('auto_actions'):
            continue
        entity = action['entity']
        match action['type']:
            case 'buy_shares':
                field = {'ipo': 'ipo_percent', 'pool': 'pool_percent'}[pre['source']]
                corp = pre['corporation']
                assert old['corporations'][corp][field] - new['corporations'][corp][field] == pre['percent']
            case 'place_token':
                assert entity not in old['hexes'][pre['hex']]['tokens']
                assert entity in new['hexes'][pre['hex']]['tokens']
            case 'lay_tile':
                assert (old['hexes'][pre['hex']]['tile'], old['hexes'][pre['hex']]['rotation']) == \
                    (pre['old_tile'], pre['old_rotation'])
                assert (new['hexes'][pre['hex']]['tile'], new['hexes'][pre['hex']]['rotation']) == \
                    (pre['tile'], pre['rotation'])
            case 'buy_train':
                kind = pre['train_type']
                assert action['train'].split('-')[0] == kind
                assert new['corporations'][entity]['trains'].get(kind, 0) == \
                    old['corporations'][entity]['trains'].get(kind, 0) + 1 - (pre.get('exchange_type') == kind)
                if pre['source'] == 'corporation':
                    seller = pre['seller']
                    assert new['corporations'][seller]['trains'].get(kind, 0) == \
                        old['corporations'][seller]['trains'][kind] - 1
                elif pre['source'] == 'pool':
                    assert new['pool'].get(kind, 0) == old['pool'][kind] - 1
                elif TRAIN_COUNTS[kind] > 0:
                    assert new['depot'].get(kind, 0) == old['depot'][kind] - 1
            case 'run_routes':
                if all('revenue' in r for r in action['routes']):
                    assert pre['revenue'] == sum(r['revenue'] for r in action['routes'])
                else:
                    assert pre['revenue'] == pre['recomputed_revenue'] == sum(pre['recomputed_routes'])
                assert pre['trains'] == [r['train'] for r in action['routes']]
                assert new['operating']['routes_revenue'] == pre['revenue']
                assert pre['max_revenue'] % 10 == 0 and not pre['max_timed_out']
            case 'dividend':
                assert pre['revenue'] == old['operating']['routes_revenue']


def test_stock_legality_allows_each_logged_move(case):
    """``stock.legal`` admits every SR action the next record logs, and a
    must-sell turn (sales only) never logs a pass."""
    records = case[2]['records']
    checked = 0
    for before, record in zip(records, records[1:]):
        legal, action, pre = (before['state']['stock'] or {}).get('legal'), record['action'], record['pre']
        if legal is None or action.get('auto_actions') or action['entity_type'] != 'player':
            continue
        kind = action['type']
        assert kind in legal['actions'] or kind in ('message', 'end_game') or kind.startswith('program_')
        if kind == 'buy_shares':
            assert pre['corporation'] in legal['buy_ipo' if pre['source'] == 'ipo' else 'buy_pool']
        elif kind == 'par':
            assert pre['corporation'] in legal['par']
        elif kind == 'sell_shares':
            assert pre['percent'] in legal['sell'][pre['corporation']]
        assert ('pass' in legal['actions']) == (legal['actions'] != ['sell_shares'])
        checked += 1
    assert checked > 20


def test_logged_revenue_exceeds_ruby_max_only_where_known():
    # Ruby pays logged route revenue unchecked. 314's 367 logs ¥400 for IR, whose
    # 3-train route I4-I2-J3-K4 passes through I2, which SR and AR fill; the
    # legal maximum is ¥370. 962 logs no revenues, so Ruby checks its routes.
    over = {(name, r['action']['id']): (r['pre']['revenue'], r['pre']['max_revenue'])
            for name in FIXTURES for r in load_fixture_extract(name)['records'][1:]
            if r['action']['type'] == 'run_routes' and r['pre']['revenue'] > r['pre']['max_revenue']}
    assert over == {('314', 367): (400, 370)}


@pytest.mark.parametrize('num_players', range(2, 7))
def test_new_game_snapshot(num_players):
    ids = [10 * (num_players - i) for i in range(num_players)]
    extract = load_new_game_extract(num_players, seed=17, player_ids=ids)
    assert extract['source']['seed'] == 17
    assert [p['id'] for p in extract['players']] == ids
    (record,) = extract['records']
    state = record['state']
    game = state['game']
    check_state(state, num_players)
    check_cash(state)

    row = T['players'][num_players - 2]
    assert row['players'] == num_players
    assert [p['cash'] for p in state['players']] == [row['starting_cash']] * num_players
    assert list(state['privates']) == [PRIVATE_SYMS[i] for i in row['privates']]
    assert all(p['owner'] is None and not p['closed'] for p in state['privates'].values())
    assert state['auction']['companies'] == list(state['privates'])

    assert (game['round'], game['turn'], game['phase']) == ('ISR', 1, '2')
    assert game['current_entity'] == {'type': 'player', 'id': ids[0]}
    assert game['priority'] == ids[0] and game['player_order'] == ids

    assert all(c['par'] is None and c['cash'] == 0 and c['ipo_percent'] == 100 for c in state['corporations'].values())
    homes = {HEX_IDS[c['home']]: c['sym'] for c in T['corporations']}
    for hex_id, hex_ in state['hexes'].items():
        assert hex_ == {'tile': hex_id, 'rotation': 0, 'tokens': [],
                        'reserved': [homes[hex_id]] if hex_id in homes else []}
    assert state['tiles'] == SUPPLY_TILES
    # Ruby holds one D at a time and clones it on each sale.
    assert state['depot'] == {name: max(count, 1) for name, count in TRAIN_COUNTS.items()}
    assert state['pool'] == {}
