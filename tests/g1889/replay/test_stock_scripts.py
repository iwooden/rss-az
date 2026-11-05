"""Scripted stock rounds replayed against Ruby: the SR paths the fixtures'
replay from the start never takes (it stops after their first SR).

The scripts run through Ruby's extractor as game JSONs and replay like the
fixtures, compared after every action, with Ruby's legal STOCK set checked at
every turn. A second SR needs an OR first, which the engine cannot play yet,
so its replay starts from ``load_state`` of the SR's first snapshot.

- ``sold_out_stacking``: at six players KO and IR are parred at ¥65 into one
  cell and sell out in the first SR, while AR sits at ¥70. At the SR's end both
  move up in price order, so they keep their stack order, beneath AR.
- ``presidency_changes``: IR's president changes on an IPO purchase in the
  first SR. In the second a purchase that only ties changes nothing; a buy is
  followed by a sale of the same corp; the president sells normal shares below
  another player; a sale is followed by a purchase, which ends the turn; a
  president sells into the President's Certificate, which goes to the nearest
  holder of two shares clockwise; and a pool purchase takes the presidency.
  Prices fall through the yellow zone into the orange one.
- ``back_to_the_buyer``: seat 1's purchase ends its turn, and Ruby, with nobody
  else able to act, returns to seat 1 at once; its pass there is its next
  turn's, not the "done" the driver forced after the purchase.
- ``forced_pass_before_the_sr``: KO's fourth train fills it, so its president's
  BuyTrain pass is forced, and Ruby skips it. The SR opens with the same
  player's pass, which is the SR's, not that one.
- ``auto_actions_after_a_forced_done``: a purchase whose forced "done" waits
  for the next record, which carries an auto action. The record before is
  compared once, before the host action applies.

Ruby skips a player with nothing to do, logging nothing; the engine's driver
applies their pass, and the sync rule drops it. In ``sold_out_stacking``
seat 1 owns MF, so the engine offers it the port tile on such a turn, and
the harness passes for it (``skipped_sr_turn``). A must-sell turn is not
scripted: reaching the certificate limit, or 60% outside the orange zone,
takes operating rounds that pay dividends (withholds move a corp back into
the zone it left), so unit tests cover it on hand-built states.
"""

from __future__ import annotations

import pytest

import g1889.data.tables as D

from tests.g1889.replay.extracts import load_game_extract
from tests.g1889.replay.harness import replay_extract

PASS = None
AUTO = 'auto'


def _action(ids: list[int], actor, kind, *args) -> dict:
    """One move of ``_game``, as a log entry."""
    if kind == 'exchange':
        return {'type': 'buy_shares', 'entity': actor, 'entity_type': 'company', 'shares': args, 'percent': 10}
    if kind == 'lay':
        hex_id, tile, rotation = args
        return {'type': 'lay_tile', 'entity': actor, 'entity_type': 'company', 'hex': hex_id, 'tile': tile,
                'rotation': rotation}
    if kind == 'company_pass':
        return {'type': 'pass', 'entity': actor, 'entity_type': 'company'}
    if kind == 'log':
        return {'type': 'log', 'entity': ids[actor], 'entity_type': 'player', 'message': args[0]}
    if isinstance(actor, str):  # a corp
        action: dict = {'entity': actor, 'entity_type': 'corporation'}
        if kind is PASS:
            action['type'] = 'pass'
        elif kind == 'corp_lay':
            hex_id, tile, rotation = args
            action.update(type='lay_tile', hex=hex_id, tile=tile, rotation=rotation)
        elif kind == 'buy_train':
            train, price, variant = args
            action.update(type='buy_train', train=train, price=price, variant=variant)
        elif kind == 'buy_company':
            action.update(type='buy_company', company=args[0], price=args[1])
        elif kind == 'run_routes':
            action.update(type='run_routes', routes=args[0], extra_revenue=0, subsidy=0)
        else:
            assert kind == 'dividend', kind
            action.update(type='dividend', kind=args[0])
        return action
    action = {'entity': ids[actor], 'entity_type': 'player'}
    if kind is PASS:
        action['type'] = 'pass'
    elif kind == 'par':
        corp, price = args
        k = D.PAR_PRICE.index(price)
        action.update(type='par', corporation=corp, share_price=f'{price},{D.PAR_ROW[k]},{D.PAR_COL[k]}')
    elif kind == 'buy':
        action.update(type='buy_shares', shares=args, percent=10)
    elif kind == 'sell':
        action.update(type='sell_shares', shares=args[0], percent=args[1])
    else:
        action.update(type='bid', company=kind, price=args[0])
    return action


def _game(label: str, num_players: int, moves: list[tuple]) -> dict:
    """A 1889 game JSON. ``moves`` are ``(actor, PASS)`` for a player seat or a
    corp symbol, ``(seat, private, price)`` bids, ``(seat, 'par', corp, price)``,
    ``(seat, 'buy', share id)``, ``(seat, 'sell', share ids, percent)``, and
    ``(seat, 'log', text)``, a note; for a private, ``('DR', 'exchange', share
    id)``, ``('MF' or 'ER', 'lay', hex, tile id, rotation)``, and ``('ER',
    'company_pass')``; for a corp, ``(corp, 'corp_lay', hex, tile id,
    rotation)``, ``(corp, 'buy_train', train id, price, variant)``, ``(corp,
    'buy_company', private, price)``, ``(corp, 'run_routes', routes)``, and
    ``(corp, 'dividend', kind)``; and ``(AUTO, host move, *moves)``, a move
    that carries others as its ``auto_actions``."""
    ids = [101 + i for i in range(num_players)]
    actions = []
    for i, move in enumerate(moves, 1):
        if move[0] is AUTO:
            host, *autos = move[1:]
            action = {**_action(ids, *host), 'auto_actions': [_action(ids, *auto) for auto in autos]}
        else:
            action = _action(ids, *move)
        actions.append({**action, 'id': i})
    return {'id': f'scripted-{label}', 'title': '1889', 'players': [{'id': p, 'name': f'Player {p}'} for p in ids],
            'settings': {'seed': 1, 'optional_rules': []}, 'actions': actions}


def route(train: str, *hexes: str, revenue: int) -> dict:
    """A logged route of one train, stop to stop along ``hexes``, for ``(corp, 'run_routes', routes)``."""
    return {'train': train, 'connections': [list(hexes)], 'hexes': [hexes[0], hexes[-1]], 'revenue': revenue,
            'revenue_str': f'{hexes[0]}-{hexes[-1]}', 'nodes': [f'{hexes[0]}-0', f'{hexes[-1]}-0']}


SCRIPTS = {
    'sold_out_stacking': (6, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (3, 'SMR', 50), (4, 'DR', 60), (5, 'SIR', 80), (0, 'UTF', 150),
        # SR 1: KO, then IR, into the ¥65 cell; AR at ¥70.
        (1, 'par', 'KO', 65), (2, 'par', 'IR', 65), (3, 'par', 'AR', 70),
        (4, 'buy', 'KO_1'), (5, 'buy', 'KO_2'), (0, 'buy', 'KO_3'), (1, 'buy', 'IR_1'), (2, 'buy', 'IR_2'),
        (3, 'buy', 'KO_4'), (4, 'buy', 'IR_3'), (5, 'buy', 'IR_4'), (0, 'buy', 'IR_5'), (1, 'buy', 'KO_5'),
        (2, 'buy', 'KO_6'), (3, 'buy', 'KO_7'), (4, 'buy', 'KO_8'), (5, 'buy', 'IR_6'), (0, 'buy', 'IR_7'),
        (1, 'buy', 'IR_8'),
        # Seats 0 and 1 cannot afford anything, so Ruby skips them.
        (2, PASS), (3, PASS), (4, PASS), (5, PASS),
    ]),
    'presidency_changes': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        # SR 1: seat 1's fourth IR share beats seat 0's three.
        (0, 'par', 'IR', 65), (1, 'buy', 'IR_1'), (2, 'par', 'KO', 65), (0, 'buy', 'IR_2'), (1, 'buy', 'IR_3'),
        (2, 'buy', 'IR_4'), (0, PASS), (1, 'buy', 'IR_5'), (2, PASS), (0, PASS), (1, 'buy', 'IR_6'),
        (2, PASS), (0, PASS), (1, PASS),
        # OR 1: IR has no route; it withholds and moves left.
        ('IR', PASS), ('IR', PASS),
        # SR 2.
        (2, PASS),
        (0, 'buy', 'IR_7'), (0, 'sell', ['IR_2'], 10),          # four ties seat 1; then three again
        (1, 'sell', ['IR_5', 'IR_6'], 20), (1, 'buy', 'KO_1'),  # seat 0 takes IR; selling ends
        (2, 'buy', 'IR_2'), (2, PASS),
        (0, 'sell', ['IR_7', 'IR_0'], 20), (0, PASS),           # into the certificate: seat 1 takes IR
        (1, PASS),
        (2, 'buy', 'IR_1'), (2, PASS),                          # three beat seat 1's two
        (0, PASS), (1, PASS), (2, PASS),
    ]),
    'back_to_the_buyer': (3, [
        (0, 'MF', 35), (1, 'ER', 45), (2, 'SMR', 55), (0, 'DR', 65), (1, 'TR', 20), (2, PASS), (0, PASS),
        (1, 'SIR', 80),
        (2, 'par', 'SR', 65), ('DR', 'exchange', 'IR_1'), (0, 'par', 'IR', 70), (1, PASS), (2, 'buy', 'SR_1'),
        (0, 'buy', 'IR_2'), (1, PASS), (2, 'buy', 'SR_2'), (0, 'buy', 'IR_3'), (1, 'buy', 'IR_4'), (2, 'buy', 'SR_3'),
        (1, 'buy', 'SR_4'),  # seats 2 and 0 cannot afford another share, so Ruby is back at seat 1
        (1, PASS),
        ('IR', 'corp_lay', 'E2', '5-0', 4),
    ]),
    'forced_pass_before_the_sr': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        (0, 'par', 'IR', 65), (1, 'par', 'KO', 65), (2, 'buy', 'IR_1'), (0, 'buy', 'IR_2'), (1, 'buy', 'KO_1'),
        (2, 'buy', 'KO_2'), (0, 'buy', 'KO_3'), (1, PASS), (2, PASS), (0, PASS),
        ('KO', PASS), ('KO', 'buy_train', '2-0', 80, '2'), ('KO', 'buy_train', '2-1', 80, '2'),
        ('KO', 'buy_train', '2-2', 80, '2'), ('KO', 'buy_train', '2-3', 80, '2'),
        (1, PASS), (2, PASS), (0, PASS),
    ]),
    'auto_actions_after_a_forced_done': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        (0, 'par', 'IR', 65), (AUTO, (1, 'buy', 'IR_1'), (2, 'buy', 'IR_2')), (0, 'buy', 'IR_3'), (1, PASS), (2, PASS),
        (0, PASS),
    ]),
}


def _extract(label):
    num_players, moves = SCRIPTS[label]
    return load_game_extract(label, _game(label, num_players, moves))


def _sr_starts(extract):
    """Indices of the records that open each SR after the first."""
    records = extract['records']
    return [r['index'] for prev, r in zip(records, records[1:])
            if r['state']['game']['round'] == 'SR' and prev['state']['game']['round'] == 'OR']


def _last_sr_record(extract, start):
    records = extract['records']
    return next(r['index'] for r in records[start + 1:] if r['state']['game']['round'] != 'SR')


@pytest.mark.parametrize('label', ('sold_out_stacking', 'presidency_changes'))
def test_scripted_first_sr_matches_ruby(label):
    extract = _extract(label)
    first_sr = next(r['index'] for r in extract['records'] if r['state']['game']['round'] == 'SR')
    last = _last_sr_record(extract, first_sr)
    progress = replay_extract(label, extract, end=last)
    assert progress.matched == last and progress.stop is None, str(progress)


@pytest.mark.parametrize('label', ('presidency_changes', 'back_to_the_buyer', 'forced_pass_before_the_sr',
                                   'auto_actions_after_a_forced_done'))
def test_scripted_game_replays_end_to_end(label):
    """``presidency_changes``: from a new game through SR 1, OR 1 (IR passes
    Track and BuyTrain; Ruby skips its token and withhold silently, the driver
    forces them), and SR 2. Each other from a new game to its last record."""
    extract = _extract(label)
    progress = replay_extract(label, extract)
    assert progress.stop is None and progress.matched == len(extract['records']) - 1, str(progress)


def test_scripted_second_sr_matches_ruby():
    extract = _extract('presidency_changes')
    (start,) = _sr_starts(extract)
    progress = replay_extract('presidency_changes', extract, start=start)
    assert progress.stop is None and progress.matched == len(extract['records']) - 1, str(progress)


def test_scripts_take_their_paths_in_ruby():
    """The scripts do what their comments say, in Ruby's own states."""
    corps = _extract('sold_out_stacking')['records'][-1]['state']['corporations']
    assert [corps[sym]['market'] for sym in ('AR', 'KO', 'IR')] == [[4, 3, 0], [4, 3, 1], [4, 3, 2]]

    extract = _extract('presidency_changes')
    seats = [p['id'] for p in extract['players']]
    presidents = [seats.index(r['state']['corporations']['IR']['president'])
                  for r in extract['records'] if r['state']['corporations']['IR']['president'] is not None]
    changes = [p for i, p in enumerate(presidents) if i == 0 or p != presidents[i - 1]]
    assert changes == [0, 1, 0, 1, 2]
    final = extract['records'][-1]['state']['corporations']['IR']
    assert (final['price'], final['market'][:2]) == (30, [10, 2])  # the orange zone

    records = _extract('back_to_the_buyer')['records']
    assert records[-3]['action']['shares'] == ['SR_4'] and records[-3]['pre']['active_players'] == [102]
    assert records[-3]['state']['game']['active_players'] == [102]
    assert not records[-3]['state']['stock']['current_actions']
    assert records[-1]['pre']['round'] == 'OR'

    records = _extract('forced_pass_before_the_sr')['records']
    full, first_pass = records[-4], records[-3]
    assert full['pre']['active_players'] == first_pass['pre']['active_players'] == [102]
    assert (full['pre']['round'], first_pass['pre']['round']) == ('OR', 'SR')

    records = _extract('auto_actions_after_a_forced_done')['records']
    assert records[8]['action']['auto_actions'][0]['entity'] == 103 and len(records[8]['auto_pre']) == 1
    assert records[8]['state']['players'][2]['shares'] == {'IR': {'percent': 10, 'president': False}}
