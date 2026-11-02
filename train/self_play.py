"""Self-play game generation via MCTS.

Post-refactor contract: dense unified-slot policy targets + legal masks,
raw canonical int16 game state stored on examples, and canonical value
targets. The trainer consumes
``(state, phase_id, legal_mask, policy_target, value_target)`` and
materializes token and relation inputs from the raw state at training time.
Policy cross-entropy is computed over the full unified-logit slot space
(illegal slots are already zero in ``policy_target`` and masked to -1e9
inside the model).
"""

from __future__ import annotations

import queue
import signal
import time
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from games import STATUS_GAME_OVER, get_game
from mcts.evaluator import compute_terminal_values
from mcts.search import (
    StatePool,
    get_greedy_leaf_depth,
    get_greedy_leaf_value,
    policy_target_visits,
    prepare_reuse_root,
    run_search,
    scale_visit_counts_by_temperature,
)
from train.config import EpochConfig, TrainingConfig
from train.eval_server import RemoteEvaluator
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats
from train.policy_metrics import PolicyMetrics, kl_to_prior


@dataclass
class SelfPlayExample:
    """Single training example from self-play.

    Raw compact int16 game state — the trainer derives model-family-specific
    input buffers from this row at training time.
    ``legal_mask`` and ``policy_target`` are dense over the unified-logit
    slot space; slots outside the current phase's legal set are zero in
    both. ``phase_id`` is carried purely for per-phase TB bucketing.
    ``value_target`` is always canonical player order.
    """

    state: np.ndarray  # (total_int16_size,), int16 — raw compact state
    phase_id: int  # decision phase id (TB reporting only)
    legal_mask: np.ndarray  # (unified_dim,), uint8 — 1 = legal slot
    policy_target: np.ndarray  # (unified_dim,), float32 — MCTS visit probs
    value_target: np.ndarray  # (num_players,), float32 — canonical A0GB
    policy_priority: float  # KL(policy target || unnoised prior), replay priority


@dataclass
class GameRecord:
    """Results from a single self-play game.

    Training data is pre-stacked into one contiguous array per field
    (instead of per-example arrays) so that pickling through mp.Queue is a
    fast memcpy rather than per-object serialization. ``legal_masks`` and
    ``policy_targets`` are dense over the game's ``policy.unified_dim``
    unified slots; illegal slots carry 0 in both.
    """

    states: np.ndarray  # (num_examples, total_int16_size), int16
    phase_ids: np.ndarray  # (num_examples,), int8 — TB reporting only
    legal_masks: np.ndarray  # (num_examples, unified_dim), uint8
    policy_targets: np.ndarray  # (num_examples, unified_dim), float32
    value_targets: np.ndarray  # (num_examples, num_players), float32
    policy_priorities: np.ndarray  # (num_examples,), float32 — replay priority
    num_players: int  # Actual player count for this game
    num_examples: int  # Number of training examples
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # game.final_scores: final net worth per player (canonical order)
    duration_secs: float  # Wall-clock time
    policy_entropy_mean: float = 0.0  # Legacy alias: policy target entropy
    top1_visit_fraction: float = 0.0  # Legacy alias: sample top-1 fraction
    policy_target_entropy_mean: float = 0.0  # Mean target entropy (nats)
    policy_target_top1_fraction: float = 0.0  # Mean target top-1 mass
    sample_policy_entropy_mean: float = 0.0  # Mean action-sampling entropy (nats)
    sample_top1_action_fraction: float = 0.0  # Mean action-sampling top-1 mass
    profile: GameProfileData | None = None
    game_id: int = -1
    game_seed: int = -1
    rng_seed: int = -1
    final_state: np.ndarray | None = None
    strategy_trace: Any = None  # the game's trace (RSS only), with collect_strategy_trace
    # The game's self-play diagnostics, {name: (numerator, denominator)}
    # (games.SelfPlayStats.finish)
    game_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    policy_metrics: PolicyMetrics = field(default_factory=PolicyMetrics)


def _compute_linear_temperature(
    move_count: int,
    initial: float,
    anneal_start: int,
    anneal_end: int,
    final: float,
) -> float:
    """Compute a move-indexed linear temperature schedule."""
    if move_count <= anneal_start:
        return initial
    if move_count >= anneal_end:
        return final
    # Linear interpolation
    span = anneal_end - anneal_start
    t = (move_count - anneal_start) / span
    return initial + t * (final - initial)


def _compute_temperature(
    move_count: int,
    config: TrainingConfig,
    num_players: int,
) -> float:
    """Compute action-sampling temperature for the current move."""
    anneal_start, anneal_end = config.temp_anneal_window(num_players)
    return _compute_linear_temperature(
        move_count,
        config.temp_initial,
        anneal_start,
        anneal_end,
        config.temp_final,
    )


def _compute_policy_target_temperature(
    move_count: int,
    config: TrainingConfig,
    num_players: int,
) -> float:
    """Compute policy-target temperature for the current move."""
    anneal_start, anneal_end = config.policy_target_temp_anneal_window(
        num_players
    )
    return _compute_linear_temperature(
        move_count,
        config.policy_target_temp_initial,
        anneal_start,
        anneal_end,
        config.policy_target_temp_final,
    )


def build_epoch_player_count_schedule(
    config: TrainingConfig,
    games_per_epoch: int | None = None,
) -> list[int]:
    """Build the deterministic per-epoch player-count assignment schedule.

    Mixed-count epochs allocate an even quota to each player count, put the
    remainder on the lowest count, then emit assignments round-robin from
    non-depleted quotas.
    """
    total_games = (
        config.games_per_epoch
        if games_per_epoch is None
        else int(games_per_epoch)
    )
    if total_games < 0:
        raise ValueError(f"games_per_epoch must be >= 0, got {total_games}")

    player_counts = list(config.iter_player_counts())
    if len(player_counts) == 1:
        return [player_counts[0]] * total_games

    base_quota, remainder = divmod(total_games, len(player_counts))
    quotas = {num_players: base_quota for num_players in player_counts}
    quotas[player_counts[0]] += remainder

    schedule: list[int] = []
    while len(schedule) < total_games:
        for num_players in player_counts:
            remaining = quotas[num_players]
            if remaining <= 0:
                continue
            schedule.append(num_players)
            quotas[num_players] = remaining - 1
            if len(schedule) == total_games:
                break
    return schedule


def _validate_assigned_num_players(
    config: TrainingConfig,
    num_players: int,
) -> int:
    """Validate a main-process player-count assignment."""
    num_players = int(num_players)
    if not (
        config.effective_min_players
        <= num_players
        <= config.effective_max_players
    ):
        raise ValueError(
            f"assigned num_players must be in "
            f"[{config.effective_min_players}, {config.effective_max_players}], "
            f"got {num_players}"
        )
    return num_players


def _resolve_game_num_players(
    config: TrainingConfig,
    num_players: int | None,
) -> int:
    """Resolve the actual player count for a game task."""
    if num_players is None:
        if config.is_mixed_player_training:
            raise ValueError(
                "mixed player-count self-play requires an explicit "
                "num_players assignment"
            )
        return config.num_players
    return _validate_assigned_num_players(config, num_players)


def play_game(
    evaluator: Any,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool | None = None,
    epoch_config: EpochConfig | None = None,
    num_players: int | None = None,
    collect_strategy_trace: bool = False,
    game_id: int = -1,
    rng_seed: int = -1,
) -> GameRecord:
    """Play one self-play game, returning training examples.

    Args:
        evaluator: NNEvaluator or RemoteEvaluator for leaf evaluation.
        state_pool: Optional pre-allocated StatePool for MCTS node states.
            Reused across searches within the game and across games.
        epoch_config: Per-epoch dynamic parameters (c_puct, value blend).
            If None, uses config defaults (pure A0GB, c_puct_final).
        num_players: Main-process assigned actual player count. Required for
            mixed player-count training.
        collect_strategy_trace: Also record the game's strategy trace and
            final state. RSS only; other games raise ``ValueError``.
    """
    t0 = time.perf_counter()

    num_players = _resolve_game_num_players(config, num_players)
    max_players = config.effective_max_players
    game = get_game(config)
    trace_builder: Any = None
    if collect_strategy_trace:
        if game.new_strategy_trace is None:
            raise ValueError(
                f"collect_strategy_trace: game {game.name!r} has no strategy trace (RSS only)"
            )
        trace_builder = game.new_strategy_trace(num_players)
    policy = game.policy
    state = game.new_game(num_players, max_players, game_seed)

    total_int16_size = game.state_size(max_players)

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        state_pool = StatePool(game, 2 * (config.max_simulations + 1), total_int16_size)
    elif state_pool.states.shape[1] != total_int16_size:
        raise ValueError(
            f"state_pool row width {state_pool.states.shape[1]} does not match "
            f"configured max-player state width {total_int16_size}"
        )

    # Use epoch-specific overrides if provided
    c_puct_override = epoch_config.c_puct if epoch_config is not None else None
    sims_override = (epoch_config.num_simulations if epoch_config is not None
                     and epoch_config.num_simulations > 0 else None)
    mcts_config = config.to_mcts_config(
        c_puct_override=c_puct_override,
        num_simulations_override=sims_override,
        num_players=num_players,
    )
    # Forced playouts only make sense with pruned self-play targets, so
    # to_mcts_config (shared with tournament/live/analysis) leaves them off.
    mcts_config = replace(mcts_config, forced_playouts_k=config.forced_playouts_k)

    # Profile stats (None when --profile not set → zero overhead)
    search_stats: SearchStats | None = None
    if config.profile:
        search_stats = SearchStats()
        if hasattr(evaluator, "reset_profile_stats"):
            evaluator.reset_profile_stats()

    examples: list[SelfPlayExample] = []
    target_entropy_sum = 0.0
    target_top1_sum = 0.0
    sample_entropy_sum = 0.0
    sample_top1_sum = 0.0
    move_count = 0
    stats = game.new_selfplay_stats(num_players)
    policy_metrics = PolicyMetrics()
    root_priors: list[np.ndarray] = []
    target_anneal_window = config.policy_target_temp_anneal_window(num_players)
    reuse_root: Any = None

    # Scratch buffer for enumerating legal actions at each decision point.
    # Copied-out per move so the buffer is free to be reused.
    legal_scratch = np.empty(policy.max_action_size, dtype=np.uint16)
    # Static LUT mapping (phase_id, phase-local action id) → unified slot.
    # Used once per decision to scatter the sparse visit distribution and
    # legal set into dense (unified_dim,) rows for the trainer.
    action_lut_np = policy.build_action_lut().numpy()
    u_dim = policy.unified_dim
    apply_action = game.apply_action
    decision_phase = game.decision_phase
    enumerate_policy_actions = game.enumerate_policy_actions
    observe_stats = stats.observe

    while True:
        phase_id = decision_phase(state)
        n_legal = enumerate_policy_actions(state, legal_scratch)
        legal_actions = legal_scratch[:n_legal].copy()
        slots = action_lut_np[phase_id, legal_actions]

        if trace_builder is not None:
            nn_priors, nn_values, nn_actions, nn_n_legal, nn_phase_id = (
                evaluator.evaluate(state)
            )
            if (
                nn_phase_id != phase_id
                or nn_n_legal != n_legal
                or not np.array_equal(nn_actions, legal_actions)
            ):
                raise AssertionError(
                    "trace root eval legal actions diverged from self-play "
                    f"enumeration: phase {phase_id}/{nn_phase_id}, "
                    f"n {n_legal}/{nn_n_legal}"
                )
            trace_builder.append_pre_search(state, slots, nn_priors, nn_values)

        # MCTS search (reuses subtree from previous move when available).
        # The root's own enumerate inside run_search is deterministic against
        # the same state, so root.visit_counts aligns with legal_actions.
        root = run_search(
            game, state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats,
            root_priors_out=root_priors,
        )

        # Sparse policy target: temperature-shaped visit-count proportions
        # over legal actions, after any policy-target pruning. Setting
        # policy_target_temp_* to a constant 1.0 recovers raw visit-count
        # targets. Action sampling below uses the unpruned visits.
        assert root.visit_counts is not None
        counts = root.visit_counts.astype(np.float32)
        counts_sum = float(counts.sum())
        assert counts_sum > 0.0, "run_search produced zero total visits"
        target_counts = policy_target_visits(
            root, mcts_config.c_puct, config.policy_target_pruning,
            root_priors[0], mcts_config.forced_playouts_k,
        )
        target_temperature = _compute_policy_target_temperature(
            move_count, config, num_players,
        )
        policy_target_sparse = scale_visit_counts_by_temperature(
            target_counts, target_temperature,
        )
        policy_metrics.observe(
            root_priors[0], counts, policy_target_sparse,
            policy.phase_names[phase_id], move_count, target_anneal_window,
            target_counts=target_counts,
        )

        # A0GB value target — already canonical (no np.roll).
        value_target = get_greedy_leaf_value(root, num_players)

        # Temperature-scaled sampling distribution over the same sparse list.
        temperature = _compute_temperature(move_count, config, num_players)
        sample_probs = scale_visit_counts_by_temperature(counts, temperature)

        # Stats: keep target and action-sampling distributions separate.
        target_nonzero = policy_target_sparse[policy_target_sparse > 0]
        sample_nonzero = sample_probs[sample_probs > 0]
        target_entropy_sum += float(
            -np.sum(target_nonzero * np.log(target_nonzero))
        )
        target_top1_sum += float(np.max(policy_target_sparse))
        sample_entropy_sum += float(
            -np.sum(sample_nonzero * np.log(sample_nonzero))
        )
        sample_top1_sum += float(np.max(sample_probs))

        # Scatter sparse visit probs + legal set into dense unified-slot
        # rows. The same LUT the model uses to collapse its dense forward
        # back to phase-local ids gives us the inverse mapping here.
        dense_legal_mask = np.zeros(u_dim, dtype=np.uint8)
        dense_legal_mask[slots] = 1
        dense_policy_target = np.zeros(u_dim, dtype=np.float32)
        dense_policy_target[slots] = policy_target_sparse

        examples.append(
            SelfPlayExample(
                state=state._array.copy(),
                phase_id=phase_id,
                legal_mask=dense_legal_mask,
                policy_target=dense_policy_target,
                value_target=value_target,
                policy_priority=kl_to_prior(policy_target_sparse, root_priors[0]),
            )
        )

        # Sample and apply action.
        chosen_idx = int(rng.choice(n_legal, p=sample_probs))
        action_idx = int(legal_actions[chosen_idx])
        observe_stats(state, legal_actions, chosen_idx, root_priors[0], counts)
        if trace_builder is not None:
            trace_builder.append_search_result(
                root=root,
                slots=slots,
                a0gb_value=value_target,
                greedy_leaf_depth=get_greedy_leaf_depth(root),
                phase_id=phase_id,
                action_id=action_idx,
                selected_slot=int(action_lut_np[phase_id, action_idx]),
                target_temperature=target_temperature,
                sample_temperature=temperature,
            )
        status = apply_action(state, action_idx)
        if trace_builder is not None:
            trace_builder.append_action_events(state, move_number=move_count)
        move_count += 1

        if status == STATUS_GAME_OVER:
            break

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action_idx, state_pool)

    net_worths = game.final_scores(state)

    # Pre-stack training data into contiguous arrays (5 large arrays instead
    # of N×5 small ones) so pickle through mp.Queue is a fast memcpy. Rows
    # are dense over unified_dim; illegal slots are zero in both the
    # mask and the policy target.
    n_examples = len(examples)
    stacked_states = np.empty((n_examples, total_int16_size), dtype=np.int16)
    stacked_phase_ids = np.empty(n_examples, dtype=np.int8)
    stacked_legal_masks = np.empty((n_examples, u_dim), dtype=np.uint8)
    stacked_policy_targets = np.empty((n_examples, u_dim), dtype=np.float32)
    stacked_value_targets = np.empty((n_examples, num_players), dtype=np.float32)
    stacked_policy_priorities = np.empty(n_examples, dtype=np.float32)
    for i, ex in enumerate(examples):
        stacked_states[i] = ex.state
        stacked_phase_ids[i] = ex.phase_id
        stacked_legal_masks[i] = ex.legal_mask
        stacked_policy_targets[i] = ex.policy_target
        stacked_value_targets[i] = ex.value_target
        stacked_policy_priorities[i] = ex.policy_priority

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
        legal_masks=stacked_legal_masks,
        policy_targets=stacked_policy_targets,
        value_targets=stacked_value_targets,
        policy_priorities=stacked_policy_priorities,
        num_players=num_players,
        num_examples=n_examples,
        total_moves=move_count,
        game_stats=stats.finish(state),
        policy_metrics=policy_metrics,
        net_worths=net_worths,
        duration_secs=time.perf_counter() - t0,
        policy_entropy_mean=target_entropy_sum / max(move_count, 1),
        top1_visit_fraction=sample_top1_sum / max(move_count, 1),
        policy_target_entropy_mean=target_entropy_sum / max(move_count, 1),
        policy_target_top1_fraction=target_top1_sum / max(move_count, 1),
        sample_policy_entropy_mean=sample_entropy_sum / max(move_count, 1),
        sample_top1_action_fraction=sample_top1_sum / max(move_count, 1),
        profile=game_profile,
        game_id=game_id,
        game_seed=game_seed,
        rng_seed=rng_seed,
        final_state=(state._array.copy() if trace_builder is not None else None),
        strategy_trace=(
            trace_builder.finalize() if trace_builder is not None else None
        ),
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

    game = get_game(config)
    evaluator = RemoteEvaluator(
        game, config.effective_max_players, shared_bufs, worker_idx,
        profile=config.profile,
        terminal_rank_weight=config.terminal_blend,
    )

    total_size = game.state_size(config.effective_max_players)
    state_pool = StatePool(game, 2 * (config.max_simulations + 1), total_size)

    try:
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            task_len = len(task)
            if task_len == 3:
                game_seed, rng_seed, epoch_config = task
                num_players = None
            elif task_len == 4:
                game_seed, rng_seed, num_players, epoch_config = task
            else:
                raise ValueError(
                    f"self-play task must have 3 or 4 fields, got {task_len}"
                )
            rng = np.random.default_rng(rng_seed)
            record = play_game(
                evaluator, config, game_seed, rng,
                state_pool=state_pool, epoch_config=epoch_config,
                num_players=num_players, rng_seed=rng_seed,
            )
            result_queue.put(record)
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass
