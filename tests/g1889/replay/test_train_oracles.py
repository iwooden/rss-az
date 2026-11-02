"""BUY_TRAIN and DISCARD legality against Ruby's oracles (6.7's acceptance).

At every BuyTrain snapshot of the four fixtures, the state comes from
``load_state`` and the operating corp's BUY_TRAIN ids are checked against
the extract's ``operating.trains`` (``extracts.py``): Ruby's own
``BuyTrain#buyable_trains``, ``spend_minmax``, ``room?``, and
``discountable_trains_for``, read through ``snapshot.ruby_train_buys``.
Where Ruby's corp must buy and its treasury is short of every depot and
pool train, the president's money and share sales take over (6.8), and the
engine's enumerator raises there and nowhere else. At every DiscardTrain
snapshot the crowded corps match ``operating.crowded``, and DISCARD offers
the first one's train types.
"""

from __future__ import annotations

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.corp import CORPS
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract
from tests.g1889.replay.snapshot import CORP, engine_train_buys, load_state, ruby_train_buys

# (BuyTrain snapshots, emergencies, legal choices, from other corps, from the
# pool, trade-ins, DiscardTrain snapshots) per fixture.
COVERAGE = {
    '314': (60, 9, 303, 219, 4, 1, 2),
    '962': (52, 0, 385, 309, 0, 0, 0),
    'hs_jmmljkbw_17502': (1, 0, 2, 0, 0, 0, 0),
    'hs_lhglbbiz_17502': (1, 0, 2, 0, 0, 0, 0),
}


def _ids(state) -> list[int]:
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    return [int(i) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]


@pytest.mark.parametrize('fixture', FIXTURES)
def test_buy_train_and_discard_match_ruby(fixture):
    extract = load_fixture_extract(fixture)
    snapshots = emergencies = choices = corps = pool = trades = discards = 0
    for record in extract['records']:
        snap = record['state']
        operating = snap['operating']
        if operating is None or snap['game']['finished']:
            continue
        if operating['trains'] is not None:
            state = load_state(snap, len(extract['players']))
            corp = CORP[operating['operator']]
            ours = engine_train_buys(state)
            assert ours == ruby_train_buys(operating['trains'], CORPS[corp].cash(state)), (fixture, record['index'])
            snapshots += 1
            if ours is None:
                emergencies += 1
                assert A.must_buy_train_py(state) and CORPS[corp].cash(state) < TRAINS.cheapest_price(state)
                continue
            choices += len(ours)
            corps += sum(source in D.CORP_SYMS for source, *_ in ours)
            pool += sum(source == 'pool' for source, *_ in ours)
            trades += sum(source == 'trade' for source, *_ in ours)
        if operating['crowded'] is not None:
            state = load_state(snap, len(extract['players']))
            first = CORP[operating['crowded'][0]]
            assert [D.CORP_SYMS[c] for c in range(D.NUM_CORPS) if TRAINS.crowded(state, c)] == operating['crowded']
            assert TRAINS.first_crowded(state) == first and TURN.active_player(state) == CORPS[first].president(state)
            assert _ids(state) == [t for t in range(D.NUM_TRAIN_TYPES) if TRAINS.corp_trains(state, first, t)]
            discards += 1
    assert (snapshots, emergencies, choices, corps, pool, trades, discards) == COVERAGE[fixture]
