"""Private purchases and abilities against Ruby's oracles (6.8a's acceptance).

At every SR and OR snapshot of the four fixtures and of two scripted games,
the state comes from ``load_state`` and is checked against the extract's
schema-7 oracles (``extracts.py``), which Ruby's own steps answer:

- ``operating.companies``: what BuyCompany lets the operator buy
  (``purchasable_companies``, ``min_price``/``max_price``,
  ``can_buy_company?``) against ``purchasable_privates_py``; against the
  buy-private slots at BUY_PRIVATE_PRE (a Track snapshot before the lay) and
  BUY_PRIVATE_POST (BuyCompany); and against what the override accepts at
  TRACK, TOKEN, ROUTE, DIVIDEND, and BUY_TRAIN, which offer no slot;
- ``operating.er_bonus``: ER's lays on C4 for its seller (1889's
  SpecialTrack) against ER_BONUS, whose pass Ruby always offers;
- ``abilities.port``: MF's port lays while Ruby's ability is usable,
  whoever's turn it is, against ``port_lays_py``; and the MF slot at STOCK
  and BUY_PRIVATE_PRE, legal exactly where its owner acts with a lay to make;
- ``abilities.exchange``: ``Exchange#can_exchange?`` against
  ``can_exchange_dr_py``; and the DR slot at STOCK, BUY_PRIVATE_PRE, and
  BUY_PRIVATE_POST, and the override elsewhere, legal exactly where its owner
  acts and Ruby could exchange.

Ruby's non-blocking steps join only a step Ruby stops at, so every snapshot
at one of these decisions is one the engine asks too.

No fixture exchanges DR or lays MF's port tile in an SR, so two scripted
games do, through Ruby's extractor, replayed and compared after every action:

- ``dr_and_mf_in_the_sr``: in SR 1, seat 1 exchanges DR for an IR share
  before anyone has parred IR (the share is ¥0 and makes nobody president),
  lays MF's port tile on Nakamura (B11), and buys a share, all in one turn;
  seat 2 then pars IR, which floats once two more IPO shares sell.
- ``dr_floats_ir_mid_or``: IR has four shares out after SR 1; in OR 1 the
  president of KO, which alone operates, exchanges DR at KO's Track step,
  and IR floats then, but runs only from the next OR (rules §12.10).
- ``overrides_after_forced_decisions``: the same SR 1; KO's president owns MF
  and DR. In OR 1 KO lays its track, which leaves it a must-buy with one
  source, and the engine's forced rule runs on through the 2-train. Ruby waits:
  MF's port tile comes at the Track step it keeps open after the lay, and DR's
  exchange at BuyTrain, so the replay backs up each time, and both moves take
  the override; then KO buys, as the driver forced again.
- ``privates_at_route``: IR's lay leaves it at Route with ¥50 to earn. There
  it buys TR, a note is logged, seat 1 exchanges DR out of turn, and it buys
  ER, whose seller lays a green city on C4 that opens a second route, so Ruby's
  Route step, and the engine's ROUTE run again, then find ¥100.
- ``er_leaves_nothing_to_buy``: KO spends its last yen on ER at BuyCompany, so
  after the bonus no step of Ruby's has anything to ask (``er_bonus.resume``
  is None) and the turn ends.
- ``dr_makes_its_owner_president``: seat 1 exchanges DR at IR's Track, while
  seat 0 is its president and acts; the share makes seat 1 president, who acts
  from then on.
- ``dr_out_of_turn_in_the_sr``: seat 1 exchanges DR at seat 2's SR turn.
- ``dr_out_of_turn_past_forced_decisions``: seat 1 exchanges DR at KO's
  BuyTrain, where Ruby waits and the driver has forced KO's train.
"""

from __future__ import annotations

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.corp import CORPS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.state import GameState

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_game_extract
from tests.g1889.replay.snapshot import (
    CORP, engine_er_lays, engine_port_lays, engine_private_buys, load_state, ruby_private_buys, ruby_special_lays,
    track_after_lay,
)
from tests.g1889.replay.harness import replay_extract
from tests.g1889.replay.test_stock_scripts import PASS, _game, route

AUCTION = [(0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80)]
# SR 1 floats IR, whose president is seat 0, and KO, whose president is seat 1.
BOTH_FLOAT = [(0, 'par', 'IR', 65), (1, 'par', 'KO', 65), (2, 'buy', 'IR_1'), (0, 'buy', 'IR_2'), (1, 'buy', 'KO_1'),
              (2, 'buy', 'KO_2'), (0, 'buy', 'KO_3'), (1, 'buy', 'IR_3'), (2, PASS), (0, PASS), (1, PASS)]
# OR 1: IR's tile on E2 gives it a route to F1, and its 2s and KO's take the depot to the first 3, KO's.
TO_PHASE_3 = [('IR', 'corp_lay', 'E2', '57-0', 1), ('IR', 'buy_train', '2-0', 80, '2'),
              ('IR', 'buy_train', '2-1', 80, '2'), ('IR', 'buy_train', '2-2', 80, '2'), ('IR', PASS), ('KO', PASS),
              ('KO', 'buy_train', '2-3', 80, '2'), ('KO', 'buy_train', '2-4', 80, '2'),
              ('KO', 'buy_train', '2-5', 80, '2'), ('KO', 'buy_train', '3-0', 180, '3')]
SCRIPTS = {
    'dr_and_mf_in_the_sr': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        # SR 1: seat 1's turn exchanges DR for an unparred IR share, lays the port tile, and buys.
        (0, 'par', 'KO', 65), ('DR', 'exchange', 'IR_1'), ('MF', 'lay', 'B11', '437-0', 2), (1, 'buy', 'KO_1'),
        (2, 'par', 'IR', 65), (0, 'buy', 'IR_2'), (1, 'buy', 'IR_3'),  # five IR shares out: IR floats
        (2, PASS), (0, PASS), (1, PASS),
    ]),
    'dr_floats_ir_mid_or': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        # SR 1: IR has four shares out, KO five (it floats).
        (0, 'par', 'IR', 65), (1, 'par', 'KO', 65), (2, 'buy', 'IR_1'), (0, 'buy', 'IR_2'), (1, 'buy', 'KO_1'),
        (2, 'buy', 'KO_2'), (0, 'buy', 'KO_3'), (1, PASS), (2, PASS), (0, PASS),
        # OR 1: KO's president exchanges DR at its Track step; IR floats, and the OR ends without it.
        ('DR', 'exchange', 'IR_3'), ('KO', PASS), ('KO', PASS),
    ]),
    'overrides_after_forced_decisions': (3, [
        (0, 'TR', 20), (1, 'MF', 30), (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        (0, 'par', 'IR', 65), (1, 'par', 'KO', 65), (2, 'buy', 'IR_1'), (0, 'buy', 'IR_2'), (1, 'buy', 'KO_1'),
        (2, 'buy', 'KO_2'), (0, 'buy', 'KO_3'), (1, PASS), (2, PASS), (0, PASS),
        ('KO', 'corp_lay', 'J3', '8-0', 3), ('MF', 'lay', 'B11', '437-0', 2), ('DR', 'exchange', 'IR_3'),
        ('KO', 'buy_train', '2-0', 80, '2'), ('KO', PASS),
    ]),
    'privates_at_route': (3, [
        *AUCTION, *BOTH_FLOAT, *TO_PHASE_3, ('KO', PASS), (2, PASS), (0, PASS), (1, PASS),
        ('IR', 'corp_lay', 'D3', '9-0', 1), ('IR', 'buy_company', 'TR', 40), (0, 'log', 'a note at Route'),
        ('DR', 'exchange', 'IR_4'), ('IR', 'buy_company', 'ER', 80), ('ER', 'lay', 'C4', '13-0', 0),
        ('IR', 'run_routes', [route('2-0', 'E2', 'F1', revenue=50), route('2-1', 'E2', 'D3', 'C4', revenue=50)]),
        ('IR', 'dividend', 'payout'), ('IR', PASS), ('IR', PASS),
    ]),
    'er_leaves_nothing_to_buy': (3, [
        *AUCTION, *BOTH_FLOAT, *TO_PHASE_3,
        ('KO', 'buy_company', 'SMR', 100), ('KO', 'buy_company', 'MF', 60), ('KO', 'buy_company', 'ER', 70),
        ('ER', 'company_pass'), (2, PASS),
    ]),
    'dr_makes_its_owner_president': (3, [
        *AUCTION,
        (0, 'par', 'IR', 65), (1, 'buy', 'IR_1'), (2, 'buy', 'IR_2'), (0, PASS), (1, 'buy', 'IR_3'), (2, PASS),
        (0, PASS), (1, PASS),
        ('DR', 'exchange', 'IR_4'), ('IR', PASS), ('IR', PASS),
    ]),
    'dr_out_of_turn_in_the_sr': (3, [
        *AUCTION, (0, 'par', 'KO', 65), (1, 'buy', 'KO_1'), ('DR', 'exchange', 'IR_1'), (2, PASS), (0, 'buy', 'KO_2'),
    ]),
    'dr_out_of_turn_past_forced_decisions': (3, [
        *AUCTION,
        (0, 'par', 'KO', 65), (1, 'par', 'IR', 65), (2, 'buy', 'KO_1'), (0, 'buy', 'KO_2'), (1, 'buy', 'IR_1'),
        (2, 'buy', 'IR_2'), (0, 'buy', 'KO_3'), (1, PASS), (2, PASS), (0, PASS),
        ('KO', 'corp_lay', 'J3', '8-0', 3), ('DR', 'exchange', 'IR_3'), ('KO', 'buy_train', '2-0', 80, '2'),
        ('KO', PASS),
    ]),
}

# Ruby's step at a snapshot, and the engine decision there that takes the
# privates' slots. A Track snapshot before the lay loads at TRACK, and the
# oracle moves it to BUY_PRIVATE_PRE, where the turn would be.
DECISION_STEPS = {'BuySellParShares': K.STEP_STOCK, 'Track': K.STEP_BUY_PRIVATE_PRE,
                  'BuyCompany': K.STEP_BUY_PRIVATE_POST}
# The steps of a corp's turn that offer no slot, where replay takes the override.
OVERRIDE_STEPS = (K.STEP_TRACK, K.STEP_TOKEN, K.STEP_ROUTE, K.STEP_DIVIDEND, K.STEP_BUY_TRAIN)
INTERJECTIONS = (A.ACTION_BUY_PRIVATE, A.ACTION_DR_EXCHANGE, A.ACTION_MF_PORT)

# Per extract: (OR snapshots with an operator, of which the operator may buy a
# private; snapshots at a decision with the slots, of which some private is
# buyable there; ER bonus snapshots; snapshots where Ruby's port ability is
# usable, and the decisions MF's owner takes then; snapshots where Ruby
# answers DR's exchange, those it allows, and the decisions DR's owner takes
# with an exchange to make; snapshots at a step without the slots, of which
# the override may buy a private there, and may exchange DR).
COVERAGE = {
    '314': (197, 20, 305, 10, 1, 63, 19, 76, 76, 23, 186, 11, 5),
    '962': (243, 113, 274, 42, 1, 19, 5, 264, 221, 26, 220, 90, 0),
    'hs_jmmljkbw_17502': (2, 0, 39, 0, 0, 39, 5, 40, 40, 8, 2, 0, 2),
    'hs_lhglbbiz_17502': (2, 0, 36, 0, 0, 36, 5, 37, 37, 7, 2, 0, 2),
    'dr_and_mf_in_the_sr': (1, 0, 11, 0, 0, 3, 2, 2, 2, 1, 1, 0, 0),
    'dr_floats_ir_mid_or': (3, 0, 13, 0, 0, 13, 6, 11, 11, 4, 3, 0, 1),
    'overrides_after_forced_decisions': (5, 0, 12, 0, 0, 12, 4, 13, 13, 4, 5, 0, 3),
    'privates_at_route': (22, 12, 20, 4, 1, 18, 7, 29, 29, 7, 19, 9, 5),
    'er_leaves_nothing_to_buy': (14, 3, 18, 3, 1, 13, 5, 27, 27, 8, 10, 0, 5),
    'dr_makes_its_owner_president': (3, 0, 11, 0, 0, 11, 4, 9, 9, 3, 3, 0, 0),
    'dr_out_of_turn_in_the_sr': (0, 0, 6, 0, 0, 6, 2, 3, 3, 1, 0, 0, 0),
    'dr_out_of_turn_past_forced_decisions': (4, 0, 12, 0, 0, 12, 4, 12, 12, 3, 4, 0, 0),
}


def _extract(label):
    if label in SCRIPTS:
        num_players, moves = SCRIPTS[label]
        return load_game_extract(label, _game(label, num_players, moves))
    return load_fixture_extract(label)


def _legal(state) -> list[tuple[int, A.ActionInfoTuple]]:
    phase = A.get_decision_phase_py(state)
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)
    return [(int(i), A.decode_action_py(phase, int(i))) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]


def _owner(state, private) -> int:
    """The private's player owner, or -1."""
    owner = PRIVATES[private].owner(state)
    return owner - K.OWNER_PLAYER if K.OWNER_PLAYER <= owner < K.OWNER_CORP else -1


def _override_accepts(state, slot: int) -> bool:
    """Whether ``DRIVER.override_private_slot`` takes BUY_PRIVATE_PRE's ``slot`` here, on a copy."""
    copy = GameState.from_array(state._array, TURN.num_players(state), state.max_players)
    return DRIVER.override_private_slot(copy, slot) != STATUS_INVALID


def check_privates(label, extract) -> tuple[int, ...]:
    """The counts of ``COVERAGE``."""
    counts = [0] * 13
    for record in extract['records']:
        snap, index = record['state'], record['index']
        g, operating, abilities = snap['game'], snap['operating'], snap['abilities']
        if g['finished'] or g['round'] == 'ISR' or g['round'] == 'SR' and g['step'] != 'BuySellParShares':
            continue
        state = load_state(snap, len(extract['players']))
        where = (label, index)
        corp = TURN.operating_corp(state)
        if operating is not None and corp >= 0:
            ruby = ruby_private_buys(operating['companies'], CORPS[corp].cash(state))
            assert engine_private_buys(state, corp) == ruby, where
            counts[0] += 1
            counts[1] += bool(ruby)
        if operating is not None and operating['er_bonus'] is not None:
            assert TURN.step(state) == K.STEP_ER_BONUS and TURN.active_player(state) == TURN.pending_seller(state)
            assert engine_er_lays(state) == ruby_special_lays(operating['er_bonus']['lays']), where
            assert [i for i, _ in _legal(state)] == list(range(1 + len(A.er_candidates_py(state)))), where
            counts[4] += 1
        if abilities['port'] is not None:
            assert engine_port_lays(state) == ruby_special_lays(abilities['port']), where
            counts[5] += 1
        if abilities['exchange'] is not None:
            assert A.can_exchange_dr_py(state, _owner(state, D.PRIVATE_DR)) == abilities['exchange'], where
            counts[7] += 1
            counts[8] += abilities['exchange']
        if TURN.step(state) in OVERRIDE_STEPS and corp >= 0:
            if TURN.step(state) != K.STEP_ROUTE:  # automated
                assert not {info.type for _, info in _legal(state)} & set(INTERJECTIONS), where
            ruby = ruby_private_buys(operating['companies'], CORPS[corp].cash(state))
            accepted = [sym for p, sym in enumerate(D.PRIVATE_SYMS)
                        if _override_accepts(state, K.BUY_PRIVATE_PRE_PRIVATE + p)]
            assert accepted == ruby, where
            exchange = bool(abilities['exchange']) and _owner(state, D.PRIVATE_DR) == TURN.active_player(state)
            assert _override_accepts(state, K.BUY_PRIVATE_PRE_DR) == exchange, where
            counts[10] += 1
            counts[11] += bool(ruby)
            counts[12] += exchange
        step = DECISION_STEPS.get(g['step'])
        if step == K.STEP_BUY_PRIVATE_PRE and TURN.step(state) == K.STEP_TRACK:
            TURN.set_step(state, step)
        if step is None or TURN.step(state) != step or track_after_lay(snap):
            continue
        legal = _legal(state)
        # The decision's own actions (POST's are its purchases) hold one beyond slot 0: Ruby asks.
        own = [i for i, info in legal if info.type not in INTERJECTIONS
               or step == K.STEP_BUY_PRIVATE_POST and info.type == A.ACTION_BUY_PRIVATE]
        assert step == K.STEP_BUY_PRIVATE_PRE or len(own) > 1 or own and own[0] != 0, where
        counts[2] += 1
        actor = TURN.active_player(state)
        kinds = [info.type for _, info in legal]
        if step != K.STEP_STOCK:
            buys = [D.PRIVATE_SYMS[info.private] for _, info in legal if info.type == A.ACTION_BUY_PRIVATE]
            assert buys == ruby_private_buys(operating['companies'], CORPS[corp].cash(state)), where
            counts[3] += bool(buys)
        port = (bool(abilities['port']) and _owner(state, D.PRIVATE_MF) == actor
                and step in (K.STEP_STOCK, K.STEP_BUY_PRIVATE_PRE))
        assert (A.ACTION_MF_PORT in kinds) == port, where
        counts[6] += port
        exchange = bool(abilities['exchange']) and _owner(state, D.PRIVATE_DR) == actor
        assert (A.ACTION_DR_EXCHANGE in kinds) == exchange, where
        counts[9] += exchange
    return tuple(counts)


@pytest.mark.parametrize('label', (*FIXTURES, *SCRIPTS))
def test_privates_match_ruby(label):
    assert check_privates(label, _extract(label)) == COVERAGE[label]


# The privates' moves each script makes, by the path they replay through.
SCRIPT_PATHS = {
    'dr_and_mf_in_the_sr': {'STOCK': 2},
    'dr_floats_ir_mid_or': {'BUY_PRIVATE_PRE': 1},
    'overrides_after_forced_decisions': {'backed up': 2, 'override at Track': 1, 'override at BuyTrain': 1},
    'privates_at_route': {'backed up': 3, 'override at Route': 2, 'out of turn': 1, 'ER_BONUS': 1},
    'er_leaves_nothing_to_buy': {'BUY_PRIVATE_POST': 3},
    'dr_makes_its_owner_president': {'out of turn': 1},
    'dr_out_of_turn_in_the_sr': {'out of turn': 1},
    'dr_out_of_turn_past_forced_decisions': {'backed up': 1, 'out of turn': 1},
}


@pytest.mark.parametrize('label', SCRIPTS)
def test_scripted_game_replays_end_to_end(label):
    extract = _extract(label)
    progress = replay_extract(label, extract)
    assert progress.stop is None and progress.matched == len(extract['records']) - 1, str(progress)
    assert progress.private_moves == SCRIPT_PATHS[label]


def test_the_overrides_script_takes_its_paths_in_ruby():
    """Ruby waits at Track, kept open after the lay, for the port tile, and at
    BuyTrain for the exchange, each time past where the driver forces itself."""
    records = _extract('overrides_after_forced_decisions')['records']
    lay, port, exchange = records[-5], records[-4], records[-3]
    assert lay['action']['type'] == 'lay_tile' and track_after_lay(lay['state'])
    assert port['pre']['step'] == 'Track' and port['state']['hexes']['B11']['tile'] == '437'
    assert exchange['pre']['step'] == 'BuyTrain' and exchange['state']['privates']['DR']['closed']
    assert exchange['state']['operating']['must_buy']['fresh']


def test_scripts_take_their_paths_in_ruby():
    """The scripts do what their comments say, in Ruby's own states."""
    records = _extract('dr_and_mf_in_the_sr')['records']
    seat_1 = _extract('dr_and_mf_in_the_sr')['players'][1]['id']
    exchanged = records[8]['state']
    assert exchanged['privates']['DR']['closed'] and not exchanged['corporations']['IR']['ipoed']
    assert exchanged['players'][1]['shares'] == {'IR': {'percent': 10, 'president': False}}
    assert exchanged['corporations']['IR']['president'] is None
    assert records[9]['state']['hexes']['B11']['tile'] == '437' and records[9]['state']['game']['step'] == \
        'BuySellParShares' and records[9]['state']['game']['active_players'] == [seat_1]
    floated = records[13]['state']['corporations']['IR']
    assert floated['floated'] and floated['ipo_percent'] == 50 and floated['cash'] == 650

    records = _extract('dr_floats_ir_mid_or')['records']
    before, after = records[-4]['state'], records[-3]['state']
    assert before['game']['round'] == 'OR' and before['game']['round_entities'] == ['KO']
    assert not before['corporations']['IR']['floated'] and after['corporations']['IR']['floated']
    assert after['game']['round_entities'] == ['KO'] and records[-1]['state']['game']['round'] == 'SR'


def test_scripted_exchange_floats_ir_for_the_next_or():
    """Replayed in the engine: IR floats at KO's TRACK, keeps out of OR 1's
    order, and gets its capital; the SR follows OR 1."""
    extract = _extract('dr_floats_ir_mid_or')
    state = load_state(extract['records'][-4]['state'], len(extract['players']))
    assert TURN.step(state) == K.STEP_TRACK and A.can_exchange_dr_py(state, TURN.active_player(state))
    DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_DR)
    assert CORPS[CORP['IR']].floated(state) and CORPS[CORP['IR']].cash(state) == 650
    assert [TURN.or_order(state, i) for i in range(2)] == [CORP['KO'], -1] and TURN.step(state) == K.STEP_TRACK


def _index_of(records, kind: str, **fields) -> int:
    return next(r['index'] for r in records[1:] if r['action']['type'] == kind
                and all(r['action'].get(key) == value for key, value in fields.items()))


def test_privates_at_route_take_their_paths_in_ruby():
    """Ruby stays at Route through each move, but for ER's bonus, and the bonus
    raises what its routes earn; the engine's ROUTE, run again, finds the same."""
    records = _extract('privates_at_route')['records']
    first = _index_of(records, 'buy_company', company='TR')
    moves = records[first:first + 5]
    assert [r['pre']['step'] for r in moves] == ['Route', 'Route', 'Route', 'Route', 'SpecialTrack']
    assert [r['state']['game']['step'] for r in moves] == ['Route', 'Route', 'Route', 'SpecialTrack', 'Route']
    assert records[first - 1]['state']['operating']['auto_router']['revenue'] == 50
    assert moves[-1]['state']['operating']['auto_router']['revenue'] == 100
    assert moves[2]['pre']['active_players'] == [101]  # seat 0 acts, and seat 1 exchanges


def test_a_replay_takes_the_revenue_after_ers_bonus_without_the_override():
    """No private's move leaves a stale revenue: the engine's ROUTE runs after
    the bonus, so the ¥100 that Ruby pays is already the turn's."""
    extract = _extract('privates_at_route')
    records = extract['records']
    routes = _index_of(records, 'run_routes')
    assert records[routes]['pre']['revenue'] == 100
    progress = replay_extract('privates_at_route', extract, end=routes - 1)
    assert progress.stop is None and progress.matched == routes - 1, str(progress)
    assert TURN.step(progress.state) == K.STEP_DIVIDEND and TURN.route_revenue(progress.state) == 100


def test_a_replay_can_start_at_ruby_route_step():
    """``load_state`` stops at ROUTE, and the replay runs it, so a private's
    move and the routes both replay from there."""
    extract = _extract('privates_at_route')
    records = extract['records']
    first = _index_of(records, 'buy_company', company='TR')
    assert records[first - 1]['state']['game']['step'] == 'Route'
    progress = replay_extract('privates_at_route', extract, start=first - 1)
    assert progress.stop is None and progress.matched == len(records) - 1, str(progress)
    bonus = _index_of(records, 'buy_company', company='ER')
    progress = replay_extract('privates_at_route', extract, start=bonus + 1)
    assert progress.stop is None and progress.matched == len(records) - 1, str(progress)


def test_er_bonus_with_no_step_left_to_resume_loads_and_replays():
    """Ruby's ``resume`` is None once the corp can buy nothing more: the seller's
    bonus is the turn's last move, and the driver's forced "done" ends it."""
    extract = _extract('er_leaves_nothing_to_buy')
    records = extract['records']
    bonus = _index_of(records, 'buy_company', company='ER')
    assert records[bonus]['state']['operating']['er_bonus']['resume'] is None
    assert records[bonus + 1]['state']['game']['round'] == 'SR'
    state = load_state(records[bonus]['state'], len(extract['players']))
    assert TURN.step(state) == K.STEP_ER_BONUS and TURN.return_step(state) == K.STEP_BUY_PRIVATE_POST
    progress = replay_extract('er_leaves_nothing_to_buy', extract, start=bonus)
    assert progress.stop is None and progress.matched == len(records) - 1, str(progress)


def test_dr_out_of_turn_takes_their_paths_in_ruby():
    """Each exchange is by a player Ruby is not waiting on, and Ruby's presidency
    and active player move with the share."""
    records = _extract('dr_makes_its_owner_president')['records']
    exchange = _index_of(records, 'buy_shares', entity='DR')
    assert records[exchange]['pre']['active_players'] == [101] and records[exchange]['pre']['step'] == 'Track'
    assert records[exchange - 1]['state']['corporations']['IR']['president'] == 101
    assert records[exchange]['state']['corporations']['IR']['president'] == 102
    assert records[exchange]['state']['game']['active_players'] == [102]

    records = _extract('dr_out_of_turn_in_the_sr')['records']
    exchange = _index_of(records, 'buy_shares', entity='DR')
    assert records[exchange]['pre']['active_players'] == [103] and records[exchange]['pre']['round'] == 'SR'
    assert records[exchange]['state']['game']['active_players'] == [103]
    assert records[exchange]['state']['privates']['DR']['closed']

    records = _extract('dr_out_of_turn_past_forced_decisions')['records']
    exchange = _index_of(records, 'buy_shares', entity='DR')
    assert records[exchange]['pre']['active_players'] == [101] and records[exchange]['pre']['step'] == 'BuyTrain'
    assert records[exchange + 1]['action']['type'] == 'buy_train'
