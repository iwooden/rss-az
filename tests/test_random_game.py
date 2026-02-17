"""Play complete games with random valid actions, checking invariants at every step."""

import numpy as np
import pytest

from core.actions import get_valid_action_mask
from core.driver import DRIVER, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.state import GameState
from tests.phases.conftest import assert_invariants

MAX_ACTIONS = 10_000  # Safety cap — real games finish well under this


def _play_random_game(num_players: int, seed: int) -> int:
    """Play a full game with random valid actions. Returns action count."""
    rng = np.random.default_rng(seed)

    state = GameState(num_players=num_players)
    state.initialize_game(seed=seed)

    actions_taken = 0

    while actions_taken < MAX_ACTIONS:
        mask = get_valid_action_mask(state)
        valid = np.flatnonzero(mask)
        assert len(valid) > 0, f"No valid actions at step {actions_taken}"

        action = int(rng.choice(valid))

        history: list[tuple[np.ndarray, int]] = []
        status = DRIVER.apply_action(state, action, history=history)

        # Check invariants on every intermediate state
        for i, (state_array, action_id) in enumerate(history):
            intermediate = GameState.from_array(state_array, num_players)
            assert_invariants(
                intermediate,
                f"step {actions_taken}, intermediate {i}/{len(history)}, "
                f"before action {action_id}",
            )

        # Check invariants on final state
        assert_invariants(state, f"step {actions_taken}, after action {action}")

        actions_taken += 1

        if status == STATUS_GAME_OVER:
            return actions_taken

    pytest.fail(f"Game did not finish within {MAX_ACTIONS} actions")


# 5 seeds x 5 player counts = 25 test cases
SEEDS = [1, 42, 123, 9999, 31415]
PLAYER_COUNTS = [2, 3, 4, 5, 6]


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_random_game_completes(num_players: int, seed: int):
    """A game with random valid actions should reach GAME_OVER."""
    actions = _play_random_game(num_players, seed)
    assert actions > 0
