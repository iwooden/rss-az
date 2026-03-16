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
from mcts.eval_cache import EvalCache
from mcts.evaluator import compute_terminal_values, rotate_visible_state
from mcts.search import StatePool, get_action_probabilities, get_greedy_leaf_value, prepare_reuse_root, run_search
from train.config import EpochConfig, TrainingConfig
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats
from train.replay_buffer import TrainingExample


@dataclass
class GameRecord:
    """Results from a single self-play game."""

    examples: list[TrainingExample]
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # Final net worth per player (canonical order)
    duration_secs: float  # Wall-clock time
    policy_entropy_mean: float = 0.0  # Mean entropy of MCTS policy targets (nats)
    top1_visit_fraction: float = 0.0  # Mean fraction of visits on top action
    profile: GameProfileData | None = None


def _compute_temperature(move_count: int, config: TrainingConfig) -> float:
    """Compute temperature for the current move using the linear ramp schedule.

    Schedule: temp_initial from move 0 to temp_anneal_start, then linearly
    decreases to temp_final at temp_anneal_end. Stays at temp_final after.
    """
    if move_count <= config.temp_anneal_start:
        return config.temp_initial
    if move_count >= config.temp_anneal_end:
        return config.temp_final
    # Linear interpolation
    span = config.temp_anneal_end - config.temp_anneal_start
    t = (move_count - config.temp_anneal_start) / span
    return config.temp_initial + t * (config.temp_final - config.temp_initial)


def play_game(
    evaluator: Any,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool | None = None,
    epoch_config: EpochConfig | None = None,
) -> GameRecord:
    """Play one self-play game, returning training examples.

    Args:
        evaluator: NNEvaluator or RemoteEvaluator for leaf evaluation.
        state_pool: Optional pre-allocated StatePool for MCTS node states.
            Reused across searches within the game and across games.
        epoch_config: Per-epoch dynamic parameters (c_puct, value blend,
            subtree reuse). If None, uses config defaults (pure A0GB,
            c_puct_final, subtree reuse enabled).
    """
    t0 = time.perf_counter()

    state = GameState(config.num_players)
    state.initialize_game(seed=game_seed)

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        from core.state import get_layout
        total_size = get_layout(config.num_players).total_size
        state_pool = StatePool(config.num_simulations + 1, total_size)

    # Use epoch-specific c_puct if provided
    c_puct_override = epoch_config.c_puct if epoch_config is not None else None
    mcts_config = config.to_mcts_config(c_puct_override=c_puct_override)

    # Whether to reuse subtrees between moves
    enable_reuse = epoch_config is None or epoch_config.enable_subtree_reuse

    # Per-game eval cache (when subtree reuse is disabled)
    eval_cache: EvalCache | None = None
    if not enable_reuse:
        eval_cache = EvalCache(config.action_dim, config.num_players)

    # Profile stats (None when --profile not set → zero overhead)
    search_stats: SearchStats | None = None
    if config.profile:
        search_stats = SearchStats()
        if hasattr(evaluator, "reset_profile_stats"):
            evaluator.reset_profile_stats()

    examples: list[TrainingExample] = []
    active_player_ids: list[int] = []
    entropy_sum = 0.0
    top1_sum = 0.0
    move_count = 0
    reuse_root: Any = None

    while True:
        active_player = state.get_active_player()
        legal_mask = DRIVER.get_legal_moves(state)

        # MCTS search (reuses subtree from previous move when available)
        root = run_search(
            state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats, eval_cache=eval_cache,
        )

        # Temperature schedule (linear ramp)
        temp = _compute_temperature(move_count, config)
        policy = get_action_probabilities(root, temp, config.action_dim)
        value_target = get_greedy_leaf_value(root, config.num_players)

        # Track policy concentration stats
        nonzero = policy[policy > 0]
        entropy_sum += float(-np.sum(nonzero * np.log(nonzero)))
        top1_sum += float(np.max(policy))

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
        active_player_ids.append(active_player)

        # Sample and apply action
        action_idx = int(rng.choice(config.action_dim, p=policy))
        status = DRIVER.apply_action(state, action_idx)
        move_count += 1

        if status == STATUS_GAME_OVER_PY:
            break

        # Extract chosen child's subtree for reuse in next search
        if enable_reuse:
            reuse_root = prepare_reuse_root(root, action_idx, state_pool)
        else:
            reuse_root = None

    net_worths = [
        state.get_player_net_worth(i) for i in range(config.num_players)
    ]

    # Blend A0GB value targets with game outcome if configured
    blend_alpha = epoch_config.value_blend_alpha if epoch_config is not None else 1.0
    if blend_alpha < 1.0:
        terminal_values = compute_terminal_values(net_worths, config.num_players)
        for i, ex in enumerate(examples):
            rotated_terminal = np.roll(terminal_values, -active_player_ids[i])
            blended = blend_alpha * ex.value_target + (1.0 - blend_alpha) * rotated_terminal
            examples[i] = ex._replace(value_target=blended)

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
        policy_entropy_mean=entropy_sum / max(move_count, 1),
        top1_visit_fraction=top1_sum / max(move_count, 1),
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
            game_seed, rng_seed, epoch_config = task
            rng = np.random.default_rng(rng_seed)
            record = play_game(
                evaluator, config, game_seed, rng,
                state_pool=state_pool, epoch_config=epoch_config,
            )
            result_queue.put(record)
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass
    finally:
        eval_conn.close()
