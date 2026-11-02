"""BUY_TRAIN and DISCARD legality against Ruby's oracles (6.7's and 6.8b's
acceptance).

At every BuyTrain snapshot of the four fixtures, the state comes from
``load_state`` and the operating corp's BUY_TRAIN choices are checked
against the extract's ``operating.trains`` and ``operating.emergency``
(``extracts.py``). BUY_TRAIN offers sources and corps to sell, so each is
composed with the choices of the sub-phase it opens, through the driver
(``snapshot.engine_train_buys``): SELECT_TRAIN's trains and
EMERGENCY_SELL_AMOUNT's blocks. Ruby's side is its own
``BuyTrain#buyable_trains``, ``spend_minmax``, ``room?``, and
``discountable_trains_for``, and in a must-buy the president's contribution
toward the cheapest depot or pool train, the bundles BuyTrain's
``can_sell?`` lets them sell, and ``can_go_bankrupt?``, read through
``snapshot.ruby_train_buys``. At every DiscardTrain snapshot the crowded
corps match ``operating.crowded``, and DISCARD offers the first one's train
types.

Ruby's steps take more than its UI offers: BuyTrain takes the president's
share sales whenever their cash and the treasury fall short of the cheapest
train, must-buy or not, and the OR's Bankrupt step takes a bankruptcy at any
step of the turn on the same shortfall (at a step other than BuyTrain,
counting no shares). The UI offers both only in a must-buy's emergency, and
so does the engine; the test counts the snapshots where Ruby's steps would
take more.

In-place probes of Ruby (``extract_states.rb --probes``) cover what no
fixture reaches: a president beside a co-holder of three or four shares,
the need at ¥0 and ¥1, and a bankruptcy either side of its boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.corp import CORPS
from g1889.entities.player import PLAYERS
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_probe_extract
from tests.g1889.replay.snapshot import (
    CORP, compare, engine_train_buys, legal_actions, load_state, ruby_operator_buys, sub_decisions,
)

# (BuyTrain snapshots, emergencies, legal choices, from other corps, from the
# pool, trade-ins, DiscardTrain snapshots) per fixture.
COVERAGE = {
    '314': (60, 9, 366, 242, 5, 1, 2),
    '962': (52, 0, 385, 309, 0, 0, 0),
    'hs_jmmljkbw_17502': (1, 0, 2, 0, 0, 0, 0),
    'hs_lhglbbiz_17502': (1, 0, 2, 0, 0, 0, 0),
}
# Per fixture, the sub-phases composed at the BuyTrain snapshots: (SELECT_TRAIN
# for a source, of which with one train, which the driver would apply;
# EMERGENCY_SELL_AMOUNT for a corp to sell, of which with one block).
SUB_PHASES = {
    '314': (214, 144, 27, 21),
    '962': (225, 111, 0, 0),
    'hs_jmmljkbw_17502': (1, 1, 0, 0),
    'hs_lhglbbiz_17502': (1, 1, 0, 0),
}
SOURCES = (A.ACTION_BUY_DEPOT_TRAIN, A.ACTION_BUY_POOL_TRAIN, A.ACTION_BUY_CORP_TRAIN)
# Per fixture, in the emergencies: (president-funded buys offered, sales
# offered, bankruptcies offered); and the snapshots where Ruby's steps take
# more than the UI offers: (BuyTrain snapshots outside a must-buy where
# BuyTrain would take a sale, OR snapshots outside an emergency where
# Bankrupt would take a bankruptcy).
EMERGENCIES = {
    '314': ((3, 35, 2), (8, 22)),
    '962': ((0, 0, 0), (11, 81)),
    'hs_jmmljkbw_17502': ((0, 0, 0), (0, 0)),
    'hs_lhglbbiz_17502': ((0, 0, 0), (0, 0)),
}


def _ids(state) -> list[int]:
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    return [int(i) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]


@pytest.mark.parametrize('fixture', FIXTURES)
def test_buy_train_and_discard_match_ruby(fixture):
    extract = load_fixture_extract(fixture)
    snapshots = emergencies = choices = corps = pool = trades = discards = 0
    funded = sales = bankruptcies = ruby_sales = ruby_bankruptcies = 0
    sub_phases = [0] * 4
    for record in extract['records']:
        snap = record['state']
        operating = snap['operating']
        if operating is None or snap['game']['finished']:
            continue
        where = (fixture, record['index'])
        state = None
        if operating['trains'] is not None:
            state = load_state(snap, len(extract['players']))
            corp = CORP[operating['operator']]
            ours = engine_train_buys(state)
            assert ours == ruby_operator_buys(state, operating), where
            snapshots += 1
            choices += len(ours)
            for action_id, info in legal_actions(state):
                if info.type in SOURCES or info.type == A.ACTION_SELECT_SALE:
                    k = 0 if info.type in SOURCES else 2
                    sub_phases[k] += 1
                    sub_phases[k + 1] += len(sub_decisions(state, action_id)) == 1
            corps += sum(source in D.CORP_SYMS for source, *_ in ours)
            pool += sum(source == 'pool' for source, *_ in ours)
            trades += sum(source == 'trade' for source, *_ in ours)
            emergency = 'sell_shares' in operating['trains']['actions'] and \
                CORPS[corp].cash(state) < TRAINS.cheapest_price(state)
            assert A.in_emergency_py(state) == emergency, where
            if emergency:
                emergencies += 1
                cheapest = TRAINS.cheapest_price(state)
                funded += any(source in ('depot', 'pool') for source, *_ in ours)
                sales += sum(source == 'sell' for source, *_ in ours)
                bankruptcies += ('bankrupt',) in ours
                assert (cheapest <= CORPS[corp].cash(state) + PLAYERS[CORPS[corp].president(state)].cash(state)) \
                    == any(source in ('depot', 'pool') for source, *_ in ours), where
            else:
                ruby_sales += bool(operating['emergency']['sell'])
        if operating['emergency'] is not None and operating['emergency']['bankrupt']:
            state = state or load_state(snap, len(extract['players']))
            in_emergency = TURN.step(state) == K.STEP_BUY_TRAIN and A.in_emergency_py(state)
            assert A.can_go_bankrupt_py(state) or not in_emergency, where
            ruby_bankruptcies += not in_emergency
        if operating['crowded'] is not None:
            state = load_state(snap, len(extract['players']))
            first = CORP[operating['crowded'][0]]
            assert [D.CORP_SYMS[c] for c in range(D.NUM_CORPS) if TRAINS.crowded(state, c)] == operating['crowded']
            assert TRAINS.first_crowded(state) == first and TURN.active_player(state) == CORPS[first].president(state)
            assert _ids(state) == [t for t in range(D.NUM_TRAIN_TYPES) if TRAINS.corp_trains(state, first, t)]
            discards += 1
    assert (snapshots, emergencies, choices, corps, pool, trades, discards) == COVERAGE[fixture]
    assert tuple(sub_phases) == SUB_PHASES[fixture]
    assert ((funded, sales, bankruptcies), (ruby_sales, ruby_bankruptcies)) == EMERGENCIES[fixture]


# Ruby's state at 314's emergencies, changed in place: KO's (record 448: the
# need is ¥444, seat 639 president of KO with six shares, KO's pool three) and
# TR's (record 455: ¥1,034 short, ¥195 to sell). Per probe: the spec's action
# and ops, KO's sellable percents, and whether bankruptcy is legal.
PROBES = {
    # Seat 344 takes the pool's KO shares: the president keeps as many as they hold.
    'a co-holder of four': (448, [['pool_shares', 'KO', 344, 3]], [10, 20], False),
    'a co-holder of three': (448, [['pool_shares', 'KO', 344, 2]], [10, 20, 30], False),
    # The treasury and the president's cash cover the D exactly, or fall ¥1 short.
    'no need': (448, [['cash', 639, 518]], [], False),
    'a need of one': (448, [['cash', 639, 517]], [10], False),
    # The president's cash and their ¥195 of shares fall short of the D by ¥1, or cover it.
    'just bankrupt': (455, [['cash', 639, 849]], [10, 20], True),
    'just solvent': (455, [['cash', 639, 850]], [10, 20], False),
}


def test_emergency_probes_match_ruby():
    specs = [{'fixture': '314.json', 'action': action, 'ops': ops} for action, ops, *_ in PROBES.values()]
    extract = load_probe_extract('emergency', specs)
    for (name, (*_, ko_sales, bankrupt)), probe in zip(PROBES.items(), extract['probes']):
        snap = probe['state']
        state = load_state(snap, len(probe['players']))
        assert compare(state, snap) == [], name
        ours = engine_train_buys(state)
        assert A.in_emergency_py(state) and ours == ruby_operator_buys(state, snap['operating']), name
        assert sorted(buy[2] for buy in ours if buy[0] == 'sell' and buy[1] == 'KO') == ko_sales, name
        assert (('bankrupt',) in ours) == bankrupt == snap['operating']['emergency']['bankrupt'], name
