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
from tests.g1889.replay.test_differential_replay import replay_extract

PASS = None


def _game(label: str, num_players: int, moves: list[tuple]) -> dict:
    """A 1889 game JSON. ``moves`` are ``(actor, PASS)`` for a player seat or a
    corp symbol, ``(seat, private, price)`` bids, ``(seat, 'par', corp, price)``,
    ``(seat, 'buy', share id)``, ``(seat, 'sell', share ids, percent)``, and
    for a private, ``('DR', 'exchange', share id)`` and ``('MF', 'lay', hex,
    tile id, rotation)``."""
    ids = [101 + i for i in range(num_players)]
    actions = []
    for i, (actor, kind, *args) in enumerate(moves, 1):
        if kind == 'exchange':
            action = {'type': 'buy_shares', 'entity': actor, 'entity_type': 'company', 'shares': args,
                      'percent': 10}
        elif kind == 'lay':
            hex_id, tile, rotation = args
            action = {'type': 'lay_tile', 'entity': actor, 'entity_type': 'company', 'hex': hex_id, 'tile': tile,
                      'rotation': rotation}
        elif isinstance(actor, str):
            action = {'type': 'pass', 'entity': actor, 'entity_type': 'corporation'}
        else:
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
        actions.append({**action, 'id': i})
    return {'id': f'scripted-{label}', 'title': '1889', 'players': [{'id': p, 'name': f'Player {p}'} for p in ids],
            'settings': {'seed': 1, 'optional_rules': []}, 'actions': actions}


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


@pytest.mark.parametrize('label', SCRIPTS)
def test_scripted_first_sr_matches_ruby(label):
    extract = _extract(label)
    first_sr = next(r['index'] for r in extract['records'] if r['state']['game']['round'] == 'SR')
    last = _last_sr_record(extract, first_sr)
    progress = replay_extract(label, extract, end=last)
    assert progress.matched == last and progress.stop is None, str(progress)


def test_scripted_game_replays_end_to_end():
    """From a new game through SR 1, OR 1 (IR passes Track and BuyTrain;
    Ruby skips its token and withhold silently, the driver forces them), and
    SR 2."""
    extract = _extract('presidency_changes')
    progress = replay_extract('presidency_changes', extract)
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
