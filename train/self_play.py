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
from mcts.search import StatePool, get_action_probabilities, get_greedy_leaf_value, prepare_reuse_root, run_search
from train.config import TrainingConfig
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats
from train.replay_buffer import TrainingExample


@dataclass
class GameRecord:
    """Results from a single self-play game."""

    examples: list[TrainingExample]
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # Final net worth per player (canonical order)
    duration_secs: float  # Wall-clock time
    profile: GameProfileData | None = None


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

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        from core.state import get_layout
        total_size = get_layout(config.num_players).total_size
        state_pool = StatePool(config.num_simulations + 1, total_size)

    mcts_config = config.to_mcts_config()

    # Profile stats (None when --profile not set → zero overhead)
    search_stats: SearchStats | None = None
    if config.profile:
        search_stats = SearchStats()
        if hasattr(evaluator, "reset_profile_stats"):
            evaluator.reset_profile_stats()

    examples: list[TrainingExample] = []
    move_count = 0
    reuse_root: Any = None

    while True:
        active_player = state.get_active_player()
        legal_mask = DRIVER.get_legal_moves(state)

        # MCTS search (reuses subtree from previous move when available)
        root = run_search(
            state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats,
        )

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

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action_idx, state_pool)

    net_worths = [
        state.get_player_net_worth(i) for i in range(config.num_players)
    ]

    game_profile: GameProfileData | None = None
    if config.profile and search_stats is not None:
        eval_client: EvalClientStats | None = None
        if hasattr(evaluator, "get_profile_stats"):
            eval_client = evaluator.get_profile_stats()
        game_profile = GameProfileData(
            search=search_stats,
            eval_client=eval_client,
            game_duration=time.perf_counter() - t0,
        )

    return GameRecord(
        examples=examples,
        total_moves=move_count,
        net_worths=net_worths,
        duration_secs=time.perf_counter() - t0,
        profile=game_profile,
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
        eval_conn, config.num_players, shared_bufs, worker_idx,
        profile=config.profile,
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
