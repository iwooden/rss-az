from collections import deque

import numpy as np
import pytest

from core.data import GamePhases
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_OK_PY as STATUS_OK,
)
from core.state import GameState
from entities.turn import TURN
from tests.phases.conftest import (
    assert_invariants,
    assert_token_data_invariants,
    game_state,
    get_legal_actions,
)


PHASE_GAME_OVER = int(GamePhases.PHASE_GAME_OVER)
MAX_RANDOM_STEPS = 5000


def _apply_action_and_verify_driver_states(state, action_id, info, *, step, seed):
    num_players = TURN.get_num_players(state)
    history = []
    status = DRIVER.apply_action(state, action_id, history=history)
    assert status in (STATUS_OK, STATUS_GAME_OVER), (
        f"step={step} seed={seed} action_id={action_id} info={info} "
        f"returned unexpected status {status}"
    )

    base_ctx = (
        f"random game step={step} seed={seed} players={num_players} "
        f"action_id={action_id} info={info}"
    )
    for i, (state_array, phase_id, hist_action_id) in enumerate(history):
        intermediate = GameState.from_array(state_array, num_players)
        ctx = (
            f"{base_ctx}\n"
            f"intermediate state {i}/{len(history)} before phase={phase_id} "
            f"action={hist_action_id}"
        )
        assert_invariants(intermediate, ctx)
        assert_token_data_invariants(
            intermediate,
            ctx,
            expected_decision_phase=phase_id,
        )

    final_ctx = f"{base_ctx}\nfinal state after driver chain"
    assert_invariants(state, final_ctx)
    assert_token_data_invariants(state, final_ctx)
    if TURN.get_phase(state) != PHASE_GAME_OVER:
        assert get_legal_actions(state), f"{final_ctx}\nno legal actions in non-terminal state"
    return status


def _play_random_game_to_completion(state, seed=None, max_steps=MAX_RANDOM_STEPS):
    num_players = TURN.get_num_players(state)
    seed = num_players if seed is None else seed
    rng = np.random.default_rng(seed)
    trace = deque(maxlen=25)

    for step in range(max_steps):
        if TURN.get_phase(state) == PHASE_GAME_OVER:
            return step

        actions = get_legal_actions(state)
        assert actions, (
            f"step={step} seed={seed} players={num_players} "
            f"reached a non-terminal state with no legal actions"
        )
        action_index = int(rng.integers(len(actions)))
        action_id, info = actions[action_index]
        trace.append(
            f"step={step} phase={TURN.get_phase(state)} action_id={action_id} info={info}"
        )
        status = _apply_action_and_verify_driver_states(
            state,
            action_id,
            info,
            step=step,
            seed=seed,
        )
        if status == STATUS_GAME_OVER:
            return step + 1

    pytest.fail(
        f"random game did not finish within {max_steps} decisions "
        f"for players={num_players} seed={seed}. "
        f"Recent trace: {list(trace)}"
    )


def test_random_legal_actions_reach_game_over_with_invariants(game_state):
    steps = _play_random_game_to_completion(game_state)
    assert steps > 0
    assert TURN.get_phase(game_state) == PHASE_GAME_OVER
