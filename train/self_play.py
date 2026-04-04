"""Self-play game generation via MCTS."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState
from mcts.evaluator import compute_terminal_values, rotate_visible_state
from mcts.search import StatePool, get_action_probabilities, get_greedy_leaf_value, prepare_reuse_root, run_search
from train.config import EpochConfig, TrainingConfig
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats
from train.replay_buffer import TrainingExample


@dataclass
class GameRecord:
    """Results from a single self-play game.

    Training data is pre-stacked into contiguous arrays (4 arrays instead of
    N×4 small arrays) so that pickling through mp.Queue is a fast memcpy
    rather than per-object serialization.
    """

    states: np.ndarray  # (num_examples, visible_size), float32
    legal_masks: np.ndarray  # (num_examples, action_dim), float32
    policy_targets: np.ndarray  # (num_examples, action_dim), float32
    value_targets: np.ndarray  # (num_examples, num_players), float32
    num_examples: int  # Number of training examples
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # Final net worth per player (canonical order)
    shares_per_player: list[int]  # Total shares held per player (canonical order)
    companies_per_player: list[int]  # Companies owned per player (canonical order)
    pres_share_values: list[float]  # Value of shares in corps where player is president
    nw_cash_pct: list[float]  # % of net worth from cash per player
    nw_companies_pct: list[float]  # % of net worth from owned company face values
    nw_shares_pct: list[float]  # % of net worth from owned shares (count * price)
    avg_active_corp_price: float  # Average share price of active corps
    corps_in_receivership: int  # Number of corps in receivership
    has_max_price_corp: bool  # Whether any active corp finished at max share price (75)
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
        epoch_config: Per-epoch dynamic parameters (c_puct, value blend).
            If None, uses config defaults (pure A0GB, c_puct_final).
    """
    t0 = time.perf_counter()

    state = GameState(config.num_players)
    state.initialize_game(seed=game_seed)

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        from core.state import get_layout
        total_size = get_layout(config.num_players).total_size
        state_pool = StatePool(2 * (config.max_simulations + 1), total_size)

    # Use epoch-specific overrides if provided
    c_puct_override = epoch_config.c_puct if epoch_config is not None else None
    sims_override = (epoch_config.num_simulations if epoch_config is not None
                     and epoch_config.num_simulations > 0 else None)
    mcts_config = config.to_mcts_config(
        c_puct_override=c_puct_override,
        num_simulations_override=sims_override,
    )

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
            profile=search_stats,
        )

        # Policy target: raw visit count proportions (no temperature scaling)
        policy_target = get_action_probabilities(root, 1.0, config.action_dim)
        value_target = get_greedy_leaf_value(root, config.num_players)

        # Temperature-scaled distribution for action selection only
        temp = _compute_temperature(move_count, config)
        action_probs = get_action_probabilities(root, temp, config.action_dim)

        # Track policy concentration stats (action selection distribution)
        nonzero = action_probs[action_probs > 0]
        entropy_sum += float(-np.sum(nonzero * np.log(nonzero)))
        top1_sum += float(np.max(action_probs))

        # Store training example with rotated state and values
        rotated_state = rotate_visible_state(
            state._array, active_player, config.num_players
        )
        rotated_value = np.roll(value_target, -active_player)
        examples.append(
            TrainingExample(
                state=rotated_state,
                legal_mask=legal_mask,
                policy_target=policy_target,
                value_target=rotated_value,
            )
        )
        active_player_ids.append(active_player)

        # Sample and apply action
        action_idx = int(rng.choice(config.action_dim, p=action_probs))
        status = DRIVER.apply_action(state, action_idx)
        move_count += 1

        if status == STATUS_GAME_OVER_PY:
            break

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action_idx, state_pool)

    net_worths = [
        state.get_player_net_worth(i) for i in range(config.num_players)
    ]

    # Extract end-of-game ownership stats
    from entities.player import PLAYERS
    from entities.corp import CORPS
    from core.data import GameConstants
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)
    shares_per_player = [
        sum(PLAYERS[i].get_shares(state, c) for c in range(num_corps))
        for i in range(config.num_players)
    ]
    companies_per_player = [
        sum(1 for c in range(num_companies) if PLAYERS[i].owns_company(state, c))
        for i in range(config.num_players)
    ]

    # Per-player value of shares in corps where they are president
    pres_share_values: list[float] = []
    for i in range(config.num_players):
        val = 0.0
        for c in range(num_corps):
            if PLAYERS[i].is_president_of(state, c):
                val += PLAYERS[i].get_shares(state, c) * CORPS[c].get_share_price(state)
        pres_share_values.append(val)

    # Net worth component breakdown (% of total)
    from entities.company import COMPANIES
    nw_cash_pct: list[float] = []
    nw_companies_pct: list[float] = []
    nw_shares_pct: list[float] = []
    for i in range(config.num_players):
        nw = net_worths[i]
        cash = PLAYERS[i].get_cash(state)
        company_value = sum(
            COMPANIES[c].get_face_value()
            for c in range(num_companies)
            if PLAYERS[i].owns_company(state, c)
        )
        share_value = sum(
            PLAYERS[i].get_shares(state, c) * CORPS[c].get_share_price(state)
            for c in range(num_corps)
        )
        if nw > 0:
            nw_cash_pct.append(cash / nw)
            nw_companies_pct.append(company_value / nw)
            nw_shares_pct.append(share_value / nw)
        else:
            nw_cash_pct.append(0.0)
            nw_companies_pct.append(0.0)
            nw_shares_pct.append(0.0)

    # Corp-level stats
    active_prices: list[int] = []
    corps_in_receivership = 0
    for c in range(num_corps):
        if CORPS[c].is_active(state):
            active_prices.append(CORPS[c].get_share_price(state))
            if CORPS[c].is_in_receivership(state):
                corps_in_receivership += 1
    avg_active_corp_price = sum(active_prices) / len(active_prices) if active_prices else 0.0
    has_max_price_corp = any(p == 75 for p in active_prices)

    # Pre-stack training data into contiguous arrays (4 large arrays instead
    # of N×4 small ones) so pickle through mp.Queue is a fast memcpy.
    stacked_states = np.stack([e.state for e in examples])
    stacked_legal_masks = np.stack([e.legal_mask for e in examples])
    stacked_policy_targets = np.stack([e.policy_target for e in examples])
    stacked_value_targets = np.stack([e.value_target for e in examples])

    # Blend A0GB value targets with game outcome if configured
    blend_alpha = epoch_config.value_blend_alpha if epoch_config is not None else 1.0
    if blend_alpha < 1.0:
        rank_weight = getattr(evaluator, "terminal_rank_weight", 0.5)
        terminal_values = compute_terminal_values(
            net_worths, config.num_players, rank_weight
        )
        for i in range(len(examples)):
            rotated_terminal = np.roll(terminal_values, -active_player_ids[i])
            stacked_value_targets[i] = (
                blend_alpha * stacked_value_targets[i]
                + (1.0 - blend_alpha) * rotated_terminal
            )

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
        states=stacked_states,
        legal_masks=stacked_legal_masks,
        policy_targets=stacked_policy_targets,
        value_targets=stacked_value_targets,
        num_examples=len(examples),
        total_moves=move_count,
        net_worths=net_worths,
        shares_per_player=shares_per_player,
        companies_per_player=companies_per_player,
        pres_share_values=pres_share_values,
        nw_cash_pct=nw_cash_pct,
        nw_companies_pct=nw_companies_pct,
        nw_shares_pct=nw_shares_pct,
        avg_active_corp_price=avg_active_corp_price,
        corps_in_receivership=corps_in_receivership,
        has_max_price_corp=has_max_price_corp,
        duration_secs=time.perf_counter() - t0,
        policy_entropy_mean=entropy_sum / max(move_count, 1),
        top1_visit_fraction=top1_sum / max(move_count, 1),
        profile=game_profile,
    )


def self_play_worker(
    task_queue: Any,
    result_queue: Any,
    config: TrainingConfig,
    shared_bufs: Any,
    worker_idx: int,
) -> None:
    """Worker process: play games using remote NN evaluation.

    Loops until a None sentinel is received on the task queue
    or the connection breaks (shutdown).
    """
    import torch
    torch.set_num_threads(1)  # Prevent OpenMP oversubscription with many workers

    from train.eval_server import RemoteEvaluator

    evaluator = RemoteEvaluator(
        config.num_players, shared_bufs, worker_idx,
        profile=config.profile,
        terminal_rank_weight=config.terminal_blend,
    )

    from core.state import get_layout

    total_size = get_layout(config.num_players).total_size
    state_pool = StatePool(2 * (config.max_simulations + 1), total_size)

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
