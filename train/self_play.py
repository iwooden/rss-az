"""Self-play game generation via MCTS."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState
from mcts.evaluator import NNEvaluator, rotate_visible_state
from mcts.search import get_action_probabilities, get_greedy_leaf_value, run_search
from train.config import TrainingConfig
from train.replay_buffer import TrainingExample


@dataclass
class GameRecord:
    """Results from a single self-play game."""

    examples: list[TrainingExample]
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # Final net worth per player (canonical order)
    duration_secs: float  # Wall-clock time


def play_game(
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
    on_move: Callable[[int], None] | None = None,
) -> GameRecord:
    """Play one self-play game, returning training examples.

    The model must be in eval() mode before calling this function.

    Args:
        on_move: Optional callback invoked after each decision point
            with the current move count. Used for live UI updates.
    """
    t0 = time.perf_counter()

    state = GameState(config.num_players)
    state.initialize_game(seed=game_seed)

    evaluator = NNEvaluator(model, device, num_players=config.num_players)
    mcts_config = config.to_mcts_config()

    examples: list[TrainingExample] = []
    move_count = 0

    while True:
        active_player = state.get_active_player()
        legal_mask = DRIVER.get_legal_moves(state)

        # MCTS search
        root = run_search(state, evaluator, mcts_config, rng)

        # Temperature schedule
        temp = config.temp_initial if move_count < config.temp_threshold else config.temp_final
        policy = get_action_probabilities(root, temp, config.action_dim)
        value_target = get_greedy_leaf_value(root, config.num_players)

        # Store training example with rotated state and values
        rotated_state = rotate_visible_state(
            state._array, active_player, config.num_players
        )
        rotated_value = np.roll(value_target, -active_player)
        examples.append(
            TrainingExample(
                state=rotated_state,
                legal_mask=legal_mask,
                policy_target=policy,
                value_target=rotated_value,
            )
        )

        # Sample and apply action
        action_idx = int(rng.choice(config.action_dim, p=policy))
        status = DRIVER.apply_action(state, action_idx)
        move_count += 1

        if on_move is not None:
            on_move(move_count)

        if status == STATUS_GAME_OVER_PY:
            break

    net_worths = [
        state.get_player_net_worth(i) for i in range(config.num_players)
    ]

    return GameRecord(
        examples=examples,
        total_moves=move_count,
        net_worths=net_worths,
        duration_secs=time.perf_counter() - t0,
    )
