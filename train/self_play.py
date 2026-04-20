"""Self-play game generation via MCTS.

Post-refactor contract: sparse policy targets over legal actions, no state
rotation, raw int16 game state stored on examples. The trainer consumes
(state, phase_id, action_ids, policy_target, value_target) — it runs
``get_token_data`` at training time to materialize the token buffer, and
computes policy cross-entropy over the legal list only (no dense −∞ mask).
"""

from __future__ import annotations

import queue
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.actions import (
    MAX_LEGAL_ACTIONS_PY,
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.data import GameConstants
from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState, get_layout
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.player import PLAYERS
from mcts.evaluator import compute_terminal_values
from mcts.search import StatePool, get_greedy_leaf_value, prepare_reuse_root, run_search
from train.config import EpochConfig, TrainingConfig
from train.eval_server import RemoteEvaluator
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats


K_MAX = int(MAX_LEGAL_ACTIONS_PY)


@dataclass
class SelfPlayExample:
    """Single training example from self-play.

    Raw compact int16 game state — the trainer runs ``get_token_data`` at
    training time to build the (num_tokens, token_dim) float32 eval buffer.
    Policy target and action_ids are sparse over the leaf's legal-action
    list (length ``n_legal``); value target is canonical per-player (no
    active-player rotation).
    """

    state: np.ndarray  # (total_int16_size,), int16 — raw compact state
    phase_id: int  # decision phase id 0-8
    n_legal: int  # number of legal actions at this state
    action_ids: np.ndarray  # (n_legal,), uint16 — phase-local legal ids
    policy_target: np.ndarray  # (n_legal,), float32 — MCTS visit probs
    value_target: np.ndarray  # (num_players,), float32 — canonical A0GB


@dataclass
class GameRecord:
    """Results from a single self-play game.

    Training data is pre-stacked into contiguous arrays (6 arrays instead
    of N×6 small arrays) so that pickling through mp.Queue is a fast
    memcpy rather than per-object serialization. Sparse fields
    (``action_ids``, ``policy_targets``) are zero-padded to ``K_MAX``;
    only ``[i, :n_legals[i]]`` is meaningful per row.
    """

    states: np.ndarray  # (num_examples, total_int16_size), int16
    phase_ids: np.ndarray  # (num_examples,), int8
    n_legals: np.ndarray  # (num_examples,), int16
    action_ids: np.ndarray  # (num_examples, K_MAX), uint16 — zero-padded
    policy_targets: np.ndarray  # (num_examples, K_MAX), float32 — zero-padded
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

    num_players = config.num_players
    state = GameState(num_players)
    state.initialize_game(num_players, seed=game_seed)

    total_int16_size = get_layout(num_players).total_size

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        state_pool = StatePool(2 * (config.max_simulations + 1), total_int16_size)

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

    examples: list[SelfPlayExample] = []
    entropy_sum = 0.0
    top1_sum = 0.0
    move_count = 0
    reuse_root: Any = None

    # Scratch buffer for enumerating legal actions at each decision point.
    # Copied-out per move so the buffer is free to be reused.
    legal_scratch = np.empty(K_MAX, dtype=np.uint16)

    while True:
        phase_id = get_decision_phase_py(state)
        n_legal = enumerate_legal_actions_py(state, legal_scratch)
        legal_actions = legal_scratch[:n_legal].copy()

        # MCTS search (reuses subtree from previous move when available).
        # The root's own enumerate inside run_search is deterministic against
        # the same state, so root.visit_counts aligns with legal_actions.
        root = run_search(
            state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats,
        )

        # Sparse policy target: raw visit-count proportions over legal actions.
        assert root.visit_counts is not None
        counts = root.visit_counts.astype(np.float32)
        counts_sum = float(counts.sum())
        assert counts_sum > 0.0, "run_search produced zero total visits"
        policy_target_sparse = counts / counts_sum

        # A0GB value target — already canonical (no np.roll).
        value_target = get_greedy_leaf_value(root, num_players)

        # Temperature-scaled sampling distribution over the same sparse list.
        temperature = _compute_temperature(move_count, config)
        if temperature < 1e-8:
            sample_probs = np.zeros(n_legal, dtype=np.float32)
            sample_probs[int(np.argmax(counts))] = 1.0
        else:
            temp_scaled = counts ** (1.0 / temperature)
            sample_probs = (temp_scaled / temp_scaled.sum()).astype(np.float32)

        # Stats: entropy of the raw visit distribution, top-1 of the
        # temperature-scaled sampling distribution.
        nonzero = policy_target_sparse[policy_target_sparse > 0]
        entropy_sum += float(-np.sum(nonzero * np.log(nonzero)))
        top1_sum += float(np.max(sample_probs))

        examples.append(
            SelfPlayExample(
                state=state._array.copy(),
                phase_id=phase_id,
                n_legal=n_legal,
                action_ids=legal_actions,
                policy_target=policy_target_sparse,
                value_target=value_target,
            )
        )

        # Sample and apply action.
        chosen_idx = int(rng.choice(n_legal, p=sample_probs))
        action_idx = int(legal_actions[chosen_idx])
        status = DRIVER.apply_action(state, action_idx)
        move_count += 1

        if status == STATUS_GAME_OVER_PY:
            break

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action_idx, state_pool)

    # End-of-game stats via entity handles (GameState no longer exposes them).
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)

    net_worths = [
        PLAYERS[i].get_net_worth(state) for i in range(num_players)
    ]
    shares_per_player = [
        sum(PLAYERS[i].get_shares(state, c) for c in range(num_corps))
        for i in range(num_players)
    ]
    companies_per_player = [
        sum(1 for c in range(num_companies) if PLAYERS[i].owns_company(state, c))
        for i in range(num_players)
    ]

    # Per-player value of shares in corps where they are president
    pres_share_values: list[float] = []
    for i in range(num_players):
        val = 0.0
        for c in range(num_corps):
            if PLAYERS[i].is_president_of(state, c):
                val += PLAYERS[i].get_shares(state, c) * CORPS[c].get_share_price(state)
        pres_share_values.append(val)

    # Net worth component breakdown (% of total)
    nw_cash_pct: list[float] = []
    nw_companies_pct: list[float] = []
    nw_shares_pct: list[float] = []
    for i in range(num_players):
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

    # Pre-stack training data into contiguous arrays (6 large arrays instead
    # of N×6 small ones) so pickle through mp.Queue is a fast memcpy. Sparse
    # fields are zero-padded to K_MAX; only [:n_legals[i]] per row is valid.
    n_examples = len(examples)
    stacked_states = np.empty((n_examples, total_int16_size), dtype=np.int16)
    stacked_phase_ids = np.empty(n_examples, dtype=np.int8)
    stacked_n_legals = np.empty(n_examples, dtype=np.int16)
    stacked_action_ids = np.zeros((n_examples, K_MAX), dtype=np.uint16)
    stacked_policy_targets = np.zeros((n_examples, K_MAX), dtype=np.float32)
    stacked_value_targets = np.empty((n_examples, num_players), dtype=np.float32)
    for i, ex in enumerate(examples):
        stacked_states[i] = ex.state
        stacked_phase_ids[i] = ex.phase_id
        stacked_n_legals[i] = ex.n_legal
        stacked_action_ids[i, :ex.n_legal] = ex.action_ids
        stacked_policy_targets[i, :ex.n_legal] = ex.policy_target
        stacked_value_targets[i] = ex.value_target

    # Blend A0GB value targets with canonical game outcome if configured.
    # No rotation — compute_terminal_values already returns canonical order.
    blend_alpha = epoch_config.value_blend_alpha if epoch_config is not None else 1.0
    if blend_alpha < 1.0:
        rank_weight = getattr(evaluator, "terminal_rank_weight", 0.5)
        terminal_values = compute_terminal_values(
            net_worths, num_players, rank_weight
        )
        stacked_value_targets = (
            blend_alpha * stacked_value_targets
            + (1.0 - blend_alpha) * terminal_values[None, :]
        ).astype(np.float32)

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
        phase_ids=stacked_phase_ids,
        n_legals=stacked_n_legals,
        action_ids=stacked_action_ids,
        policy_targets=stacked_policy_targets,
        value_targets=stacked_value_targets,
        num_examples=n_examples,
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
    # Main drives shutdown via None sentinel + eval-server stop_event; Ctrl-C
    # SIGINT delivered to the process group would otherwise interrupt the
    # RemoteEvaluator's Condition.wait() mid lock-reacquire and bubble up as
    # AssertionError from RLock.__exit__.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)  # Prevent OpenMP oversubscription with many workers

    evaluator = RemoteEvaluator(
        config.num_players, shared_bufs, worker_idx,
        profile=config.profile,
        terminal_rank_weight=config.terminal_blend,
    )

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
