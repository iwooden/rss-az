"""Private purchases and abilities against Ruby's oracles (6.8a's acceptance).

At every SR and OR snapshot of the four fixtures and of two scripted games,
the state comes from ``load_state`` and is checked against the extract's
schema-7 oracles (``extracts.py``), which Ruby's own steps answer:

- ``operating.companies``: what BuyCompany lets the operator buy
  (``purchasable_companies``, ``min_price``/``max_price``,
  ``can_buy_company?``) against ``purchasable_privates_py``; and at the
  engine's TRACK, TOKEN, DIVIDEND, BUY_TRAIN, and BUY_PRIVATE decisions, the
  buy-private slots, where Ruby's step is the one Ruby stops at;
- ``operating.er_bonus``: ER's lays on C4 for its seller (1889's
  SpecialTrack) against ER_BONUS, whose pass Ruby always offers;
- ``abilities.port``: MF's port lays while Ruby's ability is usable,
  whoever's turn it is, against ``port_lays_py``; and the MF slot at STOCK
  and TRACK, legal exactly where its owner acts with a lay to make;
- ``abilities.exchange``: ``Exchange#can_exchange?`` against
  ``can_exchange_dr_py``; and the DR slot at every top-level decision,
  legal exactly where its owner acts and Ruby could exchange.

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
"""

from __future__ import annotations

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER
from g1889.entities.corp import CORPS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN

from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_game_extract
from tests.g1889.replay.snapshot import (
    CORP, engine_er_lays, engine_port_lays, engine_private_buys, load_state, ruby_private_buys, ruby_special_lays,
    track_after_lay,
)
from tests.g1889.replay.test_differential_replay import replay_extract
from tests.g1889.replay.test_stock_scripts import PASS, _game

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
}

# Ruby's step at a snapshot, and the engine decision there that takes the
# privates' slots; Track after the lay is the engine's TOKEN, past TRACK.
DECISION_STEPS = {
    'BuySellParShares': K.STEP_STOCK, 'Track': K.STEP_TRACK, 'Token': K.STEP_TOKEN, 'Dividend': K.STEP_DIVIDEND,
    'BuyTrain': K.STEP_BUY_TRAIN, 'BuyCompany': K.STEP_BUY_PRIVATE,
}
INTERJECTIONS = (A.ACTION_BUY_PRIVATE, A.ACTION_DR_EXCHANGE, A.ACTION_MF_PORT)

# Per extract: (OR snapshots with an operator, of which the operator may buy a
# private; snapshots at a top-level decision, of which some private is
# buyable there; ER bonus snapshots; snapshots where Ruby's port ability is
# usable, and the decisions MF's owner takes then; snapshots where Ruby
# answers DR's exchange, those it allows, and the decisions DR's owner takes
# with an exchange to make).
COVERAGE = {
    '314': (197, 20, 404, 16, 1, 63, 19, 76, 76, 27),
    '962': (243, 113, 389, 96, 1, 19, 5, 264, 221, 26),
    'hs_jmmljkbw_17502': (2, 0, 40, 0, 0, 39, 5, 40, 40, 9),
    'hs_lhglbbiz_17502': (2, 0, 37, 0, 0, 36, 5, 37, 37, 8),
    'dr_and_mf_in_the_sr': (1, 0, 11, 0, 0, 3, 2, 2, 2, 1),
    'dr_floats_ir_mid_or': (3, 0, 14, 0, 0, 13, 6, 11, 11, 4),
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


@pytest.mark.parametrize('label', (*FIXTURES, *SCRIPTS))
def test_privates_match_ruby(label):
    extract = _extract(label)
    counts = [0] * 10
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
        step = DECISION_STEPS.get(g['step'])
        if step is None or TURN.step(state) != step or track_after_lay(snap):
            continue
        try:
            legal = _legal(state)
        except NotImplementedError:  # emergency money (6.8b)
            assert step == K.STEP_BUY_TRAIN and A.must_buy_train_py(state), where
            continue
        # The decision's own actions (BUY_PRIVATE's are its purchases) hold one beyond slot 0: Ruby asks.
        own = [i for i, info in legal if info.type not in INTERJECTIONS
               or step == K.STEP_BUY_PRIVATE and info.type == A.ACTION_BUY_PRIVATE]
        assert step == K.STEP_TRACK or len(own) > 1 or own and own[0] != 0, where
        counts[2] += 1
        actor = TURN.active_player(state)
        kinds = [info.type for _, info in legal]
        if step != K.STEP_STOCK:
            buys = [D.PRIVATE_SYMS[info.private] for _, info in legal if info.type == A.ACTION_BUY_PRIVATE]
            assert buys == ruby_private_buys(operating['companies'], CORPS[corp].cash(state)), where
            counts[3] += bool(buys)
        port = bool(abilities['port']) and _owner(state, D.PRIVATE_MF) == actor and step in (K.STEP_STOCK, K.STEP_TRACK)
        assert (A.ACTION_MF_PORT in kinds) == port, where
        counts[6] += port
        exchange = bool(abilities['exchange']) and _owner(state, D.PRIVATE_DR) == actor
        assert (A.ACTION_DR_EXCHANGE in kinds) == exchange, where
        counts[9] += exchange
    assert tuple(counts) == COVERAGE[label]


@pytest.mark.parametrize('label', SCRIPTS)
def test_scripted_game_replays_end_to_end(label):
    extract = _extract(label)
    progress = replay_extract(label, extract)
    assert progress.stop is None and progress.matched == len(extract['records']) - 1, str(progress)


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
    assert TURN.step(state) == K.STEP_TRACK and K.TRACK_DR in [i for i, _ in _legal(state)]
    DRIVER.apply_action(state, K.TRACK_DR)
    assert CORPS[CORP['IR']].floated(state) and CORPS[CORP['IR']].cash(state) == 650
    assert [TURN.or_order(state, i) for i in range(2)] == [CORP['KO'], -1] and TURN.step(state) == K.STEP_TRACK
