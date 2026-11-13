"""``apply_alone`` (g1889/driver.pxd): one decision by its handler, without
the auto-chain, as extraction's choice table will apply it (1889/v2-design.md
§5.3).

On the purity games' states (``tests/g1889/test_cache_purity.games``), every
engine-legal action applied alone leaves the row ``apply_action`` records
right after its dispatch: its history's next row, or its result when no
automated step or forced decision follows. The two are compared with their
caches refreshed, since the driver's next step may refresh some (its forced
action enumerates) before history copies the row. Where there is no
decision, ``apply_alone`` refuses.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
from g1889.driver import DRIVER, STATUS_INVALID, apply_alone
from g1889.entities.turn import TURN

from tests.g1889.test_cache_purity import _refresh, games
from tests.g1889.test_extraction_layout3 import copy_of, state_of
from tests.g1889.test_layout1_parity import apply_op, new_state

STRIDE = 11  # the states checked, of each game's rests, and every one at a rare decision
RARE = (K.DPHASE_DISCARD, K.DPHASE_EMERGENCY_SELL_AMOUNT, K.DPHASE_ER_BONUS, K.DPHASE_PORT)


def _legal(state):
    ids = np.empty(K.MAX_ACTION_SIZE, np.uint16)
    return ids[:A.enumerate_legal_actions_py(state, ids)].tolist()


def test_apply_alone_is_the_dispatch_without_the_chain():
    phases, chained = set(), 0
    for _, spec, calls in games():
        state = new_state(spec)
        for done in range(len(calls) + 1):
            phase = A.get_decision_phase_py(state)
            if phase >= 0 and (done % STRIDE == 0 or phase in RARE):
                for action in _legal(state):
                    alone = copy_of(state)
                    assert apply_alone(alone, action) == 0
                    after, history = copy_of(state), []
                    assert DRIVER.apply_action(after, action, history) != STATUS_INVALID
                    assert history[0][1:] == (phase, action)
                    want = state_of(history[1][0].copy()) if len(history) > 1 else after
                    _refresh(alone)
                    _refresh(want)
                    np.testing.assert_array_equal(alone._array, want._array, err_msg=f'phase {phase} action {action}')
                    chained += len(history) > 1
                phases.add(phase)
            if done < len(calls):
                apply_op(state, *calls[done])
    assert phases == set(range(K.NUM_DECISION_PHASES)) and chained > 1000


def test_apply_alone_refuses_where_there_is_no_decision():
    for _, spec, calls in games():
        state = new_state(spec)
        for call in calls:
            apply_op(state, *call)
        assert TURN.round(state) == K.ROUND_GAME_OVER
        with pytest.raises(AssertionError, match='no decision'):
            apply_alone(state, 0)
        return
