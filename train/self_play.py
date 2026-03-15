"""Self-play game generation via MCTS."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

import numpy as np

from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState
from mcts.evaluator import rotate_visible_state
from mcts.search import StatePool, get_action_probabilities, get_greedy_leaf_value, run_search
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
    evaluator: Any,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool | None = None,
) -> GameRecord:
    """Play one self-play game, returning training examples.

    Args:
        evaluator: NNEvaluator or RemoteEvaluator for leaf evaluation.
        state_pool: Optional pre-allocated StatePool for MCTS node states.
            Reused across searches within the game and across games.
    """
    t0 = time.perf_counter()

    state = GameState(config.num_players)
    state.initialize_game(seed=game_seed)

    mcts_config = config.to_mcts_config()

    examples: list[TrainingExample] = []
    move_count = 0

    while True:
        active_player = state.get_active_player()
        legal_mask = DRIVER.get_legal_moves(state)

        # MCTS search
        root = run_search(state, evaluator, mcts_config, rng, state_pool=state_pool)

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


def self_play_worker(
    eval_conn: Connection,
    task_queue: Any,
    result_queue: Any,
    config: TrainingConfig,
    shared_bufs: Any = None,
    worker_idx: int = 0,
) -> None:
    """Worker process: play games using remote NN evaluation.

    Loops until a None sentinel is received on the task queue
    or the eval connection breaks (shutdown).
    """
    from train.eval_server import RemoteEvaluator

    evaluator = RemoteEvaluator(
        eval_conn, config.num_players, shared_bufs, worker_idx
    )

    from core.state import get_layout

    total_size = get_layout(config.num_players).total_size
    state_pool = StatePool(config.num_simulations + 1, total_size)

    try:
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            game_seed, rng_seed = task
            rng = np.random.default_rng(rng_seed)
            record = play_game(evaluator, config, game_seed, rng, state_pool=state_pool)
            result_queue.put(record)
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass
    finally:
        eval_conn.close()
