"""g1889.actions: encode/decode for every phase-local id (1889/engine-design.md
§5.2), decision phases, and the legality entry points ahead of the phases."""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.turn import TURN
from g1889.state import GameState

EXACT_PHASES = (K.DPHASE_BID, K.DPHASE_RAISE, K.DPHASE_TRAIN_PRICE, K.DPHASE_PRIVATE_PRICE)
# Phases whose enumerators have landed; tests/g1889/phases/ and test_track_and_tokens.py cover them.
IMPLEMENTED_PHASES = (K.DPHASE_AUCTION, K.DPHASE_BID, K.DPHASE_RAISE, K.DPHASE_STOCK, K.DPHASE_PAR,
                      K.DPHASE_TRACK, K.DPHASE_TILE, K.DPHASE_TOKEN, K.DPHASE_DIVIDEND, K.DPHASE_DISCARD,
                      K.DPHASE_BUY_TRAIN, K.DPHASE_TRAIN_PRICE, K.DPHASE_BUY_PRIVATE)
SAMPLED_AMOUNTS = (0, 1, 5, 65, 100, 150, 165, 360, 370, 450, 1100, 4000)


@pytest.fixture
def state():
    s = GameState(4)
    s.initialize_game(4, seed=1, priority_player=0)
    return s


def decode(phase, action_id):
    return A.decode_action_py(phase, action_id)


def info(phase, type, corp=-1, private=-1, hex=-1, train_type=-1, count=-1, amount=-1):
    return A.ActionInfoTuple(phase, type, corp, private, hex, train_type, count, amount)


def buffer():
    return np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)


@pytest.mark.parametrize('phase', range(K.NUM_DECISION_PHASES))
def test_every_id_round_trips(phase):
    ids = list(range(K.PHASE_ACTION_SIZES[phase]))
    if A.has_exact_amounts(phase):
        ids += [K.PHASE_ACTION_SIZES[phase] + amount for amount in SAMPLED_AMOUNTS]
    decoded = [decode(phase, a) for a in ids]
    assert [A.encode_action_py(d) for d in decoded] == ids
    assert all(d.phase == phase for d in decoded)
    assert len(set(decoded)) == len(ids)


def test_exact_phases():
    assert [p for p in range(K.NUM_DECISION_PHASES) if A.has_exact_amounts(p)] == list(EXACT_PHASES)
    exacts = (K.BID_EXACT, K.RAISE_EXACT, K.TRAIN_PRICE_EXACT, K.PRIVATE_PRICE_EXACT)
    for phase, exact in zip(EXACT_PHASES, exacts):
        assert decode(phase, exact + 370) == info(phase, A.ACTION_EXACT_PRICE, amount=370)
        assert decode(phase, exact - 1).type == A.ACTION_PRICE_POINT


@pytest.mark.parametrize('phase, action_id, expected', [
    (K.DPHASE_AUCTION, K.AUCTION_PASS, dict(type=A.ACTION_PASS)),
    (K.DPHASE_AUCTION, K.AUCTION_BUY, dict(type=A.ACTION_BUY_TOPMOST)),
    (K.DPHASE_AUCTION, K.AUCTION_BID + D.PRIVATE_UTF, dict(type=A.ACTION_SELECT_BID, private=D.PRIVATE_UTF)),
    (K.DPHASE_BID, 3, dict(type=A.ACTION_PRICE_POINT, count=3)),
    (K.DPHASE_RAISE, K.RAISE_DROP, dict(type=A.ACTION_PASS)),
    (K.DPHASE_RAISE, K.RAISE_POINT, dict(type=A.ACTION_PRICE_POINT, count=0)),
    (K.DPHASE_STOCK, K.STOCK_IPO + D.CORP_KU, dict(type=A.ACTION_BUY_IPO, corp=D.CORP_KU)),
    (K.DPHASE_STOCK, K.STOCK_POOL + D.CORP_AR, dict(type=A.ACTION_BUY_POOL, corp=D.CORP_AR)),
    (K.DPHASE_STOCK, K.STOCK_PAR + D.CORP_IR, dict(type=A.ACTION_SELECT_PAR, corp=D.CORP_IR)),
    (K.DPHASE_STOCK, K.STOCK_SELL + 5 * D.CORP_KO + 2, dict(type=A.ACTION_SELL, corp=D.CORP_KO, count=3)),
    (K.DPHASE_STOCK, K.STOCK_DR, dict(type=A.ACTION_DR_EXCHANGE)),
    (K.DPHASE_STOCK, K.STOCK_MF, dict(type=A.ACTION_MF_PORT)),
    (K.DPHASE_PAR, 5, dict(type=A.ACTION_PAR, count=5, amount=100)),
    (K.DPHASE_TRACK, K.TRACK_HEX + 51, dict(type=A.ACTION_SELECT_HEX, hex=51)),
    (K.DPHASE_TRACK, K.TRACK_BUY_PRIVATE + D.PRIVATE_SIR, dict(type=A.ACTION_BUY_PRIVATE, private=D.PRIVATE_SIR)),
    (K.DPHASE_TRACK, K.TRACK_MF, dict(type=A.ACTION_MF_PORT)),
    (K.DPHASE_TILE, 14, dict(type=A.ACTION_LAY_CANDIDATE, count=14)),
    (K.DPHASE_TOKEN, K.TOKEN_HEX, dict(type=A.ACTION_PLACE_TOKEN, hex=0)),
    (K.DPHASE_TOKEN, K.TOKEN_DR, dict(type=A.ACTION_DR_EXCHANGE)),
    (K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD, dict(type=A.ACTION_WITHHOLD)),
    (K.DPHASE_DIVIDEND, K.DIVIDEND_PAYOUT, dict(type=A.ACTION_PAYOUT)),
    (K.DPHASE_DISCARD, D.TRAIN_D, dict(type=A.ACTION_DISCARD, train_type=D.TRAIN_D)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_DEPOT + D.TRAIN_D, dict(type=A.ACTION_BUY_DEPOT_TRAIN, train_type=D.TRAIN_D)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_TRADE_IN, dict(type=A.ACTION_TRADE_IN, train_type=D.TRAIN_4)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_TRADE_IN + 2, dict(type=A.ACTION_TRADE_IN, train_type=D.TRAIN_6)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_POOL + D.TRAIN_3, dict(type=A.ACTION_BUY_POOL_TRAIN, train_type=D.TRAIN_3)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_CORP + 6 * D.CORP_SR + D.TRAIN_5,
     dict(type=A.ACTION_BUY_CORP_TRAIN, corp=D.CORP_SR, train_type=D.TRAIN_5)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_SELL + 5 * D.CORP_UR + 4, dict(type=A.ACTION_SELL, corp=D.CORP_UR, count=5)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_BANKRUPT, dict(type=A.ACTION_BANKRUPT)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_BUY_PRIVATE, dict(type=A.ACTION_BUY_PRIVATE, private=0)),
    (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_DR, dict(type=A.ACTION_DR_EXCHANGE)),
    (K.DPHASE_BUY_PRIVATE, K.BUY_PRIVATE_DONE, dict(type=A.ACTION_PASS)),
    (K.DPHASE_BUY_PRIVATE, K.BUY_PRIVATE_PRIVATE + D.PRIVATE_ER, dict(type=A.ACTION_BUY_PRIVATE, private=D.PRIVATE_ER)),
    (K.DPHASE_ER_BONUS, K.ER_BONUS_PASS, dict(type=A.ACTION_PASS)),
    (K.DPHASE_ER_BONUS, K.ER_BONUS_CANDIDATE + 4, dict(type=A.ACTION_LAY_CANDIDATE, count=4)),
    (K.DPHASE_PORT, 6 * 3 + 5, dict(type=A.ACTION_PORT_LAY, hex=D.MF_PORT_HEX[3], count=5)),
])
def test_decoded_fields(phase, action_id, expected):
    assert decode(phase, action_id) == info(phase, **expected)


@pytest.mark.parametrize('bad, error', [
    (info(K.DPHASE_PAR, A.ACTION_PASS), ValueError),                                  # PAR has no pass
    (info(K.DPHASE_DIVIDEND, A.ACTION_PASS), ValueError),                             # nor DIVIDEND
    (info(K.DPHASE_STOCK, A.ACTION_SELL, corp=0, count=6), AssertionError),
    (info(K.DPHASE_AUCTION, A.ACTION_SELECT_BID, private=D.NUM_PRIVATES), AssertionError),
    (info(K.DPHASE_BUY_TRAIN, A.ACTION_TRADE_IN, train_type=D.TRAIN_3), ValueError),  # no trade-in for a 3
    (info(K.DPHASE_PORT, A.ACTION_PORT_LAY, hex=0, count=0), ValueError),             # not a port hex
    (info(K.DPHASE_BID, A.ACTION_EXACT_PRICE, amount=-1), AssertionError),
    (info(K.NUM_DECISION_PHASES, A.ACTION_PASS), AssertionError),
])
def test_encode_rejects_invalid_info(bad, error):
    with pytest.raises(error):
        A.encode_action_py(bad)


def test_decode_rejects_ids_outside_the_phase():
    with pytest.raises(AssertionError):
        decode(K.DPHASE_STOCK, K.ACTION_SIZE_STOCK)
    with pytest.raises(AssertionError):
        decode(K.DPHASE_AUCTION, -1)
    with pytest.raises(AssertionError):
        decode(K.DPHASE_BID, 0x10000)


@pytest.mark.parametrize('step', range(K.NUM_STEPS))
def test_decision_phase_follows_the_step(state, step):
    TURN.set_step(state, step)
    phase = K.STEP_TO_DECISION_PHASE[step]
    assert A.get_decision_phase_py(state) == phase
    ids = buffer()
    if phase < 0:
        assert A.enumerate_legal_actions_py(state, ids) == 0
        assert A.enumerate_policy_actions_py(state, ids) == 0
        assert not A.is_legal_action_py(state, 0)
        return
    if phase in IMPLEMENTED_PHASES:
        return
    # Each phase's enumerator lands with its sub-step; until then the call fails loudly.
    name = K.DecisionPhase(phase).name.removeprefix('DPHASE_')
    for call in (A.enumerate_legal_actions_py, A.enumerate_policy_actions_py):
        with pytest.raises(NotImplementedError, match=name):
            call(state, ids)
    with pytest.raises(NotImplementedError):
        A.is_legal_action_py(state, 0)
    size = K.PHASE_ACTION_SIZES[phase]
    if A.has_exact_amounts(phase):
        with pytest.raises(NotImplementedError, match='exact amounts'):
            A.is_legal_action_py(state, size + 100)
    else:
        assert not A.is_legal_action_py(state, size)  # beyond the phase: illegal without enumerating
    assert not A.is_legal_action_py(state, -1)


def test_id_buffers_are_checked(state):
    TURN.set_step(state, K.STEP_SR_END)
    with pytest.raises(AssertionError):
        A.enumerate_legal_actions_py(state, np.zeros(K.MAX_ACTION_SIZE - 1, dtype=np.uint16))
    with pytest.raises(AssertionError):
        A.enumerate_legal_actions_py(state, np.zeros(K.MAX_ACTION_SIZE, dtype=np.int32))
