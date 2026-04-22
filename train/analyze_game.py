"""Play and log a self-play game using a trained model checkpoint.

Shows NN evaluation (priors, values) and MCTS visit counts at each
decision point to visualize what the model has learned.

Usage:
    .venv/bin/python -m train.analyze_game CHECKPOINT [options]

    .venv/bin/python -m train.analyze_game checkpoints/checkpoint_epoch_0005.pt
    .venv/bin/python -m train.analyze_game latest --checkpoint-dir checkpoints
    .venv/bin/python -m train.analyze_game latest --seed 123 --simulations 200
    .venv/bin/python -m train.analyze_game latest --output game_log.md
    .venv/bin/python -m train.analyze_game new --num-players 4 --simulations 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core.actions import get_decision_phase_py
from core.data import GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.state import GameState, get_layout
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, compute_terminal_values
from mcts.node import MCTSNode
from mcts.search import StatePool, get_greedy_leaf_value, prepare_reuse_root, run_search
from nn import create_model
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import MCTSConfig, TrainingConfig
from train.debug_trace import (
    PHASE_NAMES,
    TokenNormalizationAccumulator,
    format_action,
    format_phase_context,
    format_state_full,
    format_token_dump,
    format_token_dump_from_buffer,
)
from train.profile_stats import SearchStats


def _format_nn_eval(
    priors: np.ndarray,
    values: np.ndarray,
    action_ids: np.ndarray,
    phase_id: int,
    num_players: int,
    state: GameState,
    top_n: int = 10,
    noised_priors: dict[int, float] | None = None,
) -> list[str]:
    """Format NN evaluation into readable lines.

    ``priors`` is the sparse softmax over the legal list (shape ``(n_legal,)``)
    and ``action_ids`` is the aligned phase-local id list. ``phase_id`` is
    needed because action ids are phase-local — the same integer means
    different things across phases.

    Args:
        noised_priors: If provided, maps phase-local action id to the
            noise-applied prior from the MCTS root. Shows the delta
            next to each raw prior.
    """
    lines = []

    # Values (canonical order)
    val_parts = [f"P{i}={values[i]:+.3f}" for i in range(num_players)]
    lines.append(f"  NN Values: {', '.join(val_parts)}")

    sorted_order = np.argsort(-priors)[:top_n]
    n_legal = len(action_ids)

    lines.append(
        f"  NN Priors (top {min(top_n, n_legal)} of {n_legal} legal):"
    )
    for rank, idx in enumerate(sorted_order):
        action_id = int(action_ids[idx])
        prior = float(priors[idx])
        action_str = format_action(phase_id, action_id, state)
        bar = "\u2588" * int(prior * 40)
        if noised_priors is not None and action_id in noised_priors:
            noised = noised_priors[action_id]
            delta_pp = (noised - prior) * 100
            lines.append(f"    {rank+1:2d}. {prior:6.1%} ({delta_pp:+5.1f}pp) {bar} {action_str}")
        else:
            lines.append(f"    {rank+1:2d}. {prior:6.1%} {bar} {action_str}")

    return lines


def _format_mcts_visits(
    root: MCTSNode,
    phase_id: int,
    state: GameState,
    top_n: int = 10,
) -> list[str]:
    """Format MCTS visit counts into readable lines."""
    lines = []

    if root.legal_actions is None or root.visit_counts is None:
        lines.append("  MCTS: (no search data)")
        return lines
    assert root.value_sums is not None

    counts = root.visit_counts
    total_visits = int(counts.sum())
    sorted_order = np.argsort(-counts)[:top_n]

    lines.append(f"  MCTS Visits (top {min(top_n, len(sorted_order))}, {total_visits} total):")
    for rank, idx in enumerate(sorted_order):
        action_id = int(root.legal_actions[idx])
        visits = int(counts[idx])
        if visits == 0:
            break
        pct = visits / max(total_visits, 1)
        action_str = format_action(phase_id, action_id, state)

        # Show Q value for this action (max(1,vc) matches select_child convention)
        q_val = float(root.value_sums[idx, root.active_player_id] / max(visits, 1))
        bar = "\u2588" * int(pct * 40)
        lines.append(f"    {rank+1:2d}. {visits:5d} ({pct:5.1%}) Q={q_val:+.3f} {bar} {action_str}")

    return lines


def _compute_tree_metrics(root: MCTSNode) -> dict[str, float | int]:
    """Compute compact search-tree metrics for one root."""
    assert root.legal_actions is not None and root.visit_counts is not None

    counts = root.visit_counts.astype(np.float64)
    total_visits = int(counts.sum())
    visited_actions = int(np.count_nonzero(counts))
    legal_actions = int(len(root.legal_actions))

    probs = counts / max(total_visits, 1)
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    effective_actions = float(np.exp(entropy))

    sorted_counts = np.sort(counts)[::-1]
    top_share = float(sorted_counts[0] / max(total_visits, 1)) if len(sorted_counts) else 0.0
    second_share = float(sorted_counts[1] / max(total_visits, 1)) if len(sorted_counts) > 1 else 0.0
    top_gap_pp = (top_share - second_share) * 100.0

    total_nodes = 0
    depth_sum = 0
    max_depth = 0
    stack: list[tuple[MCTSNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        total_nodes += 1
        depth_sum += depth
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node.children.values())
    mean_depth = depth_sum / max(total_nodes, 1)

    greedy_depth = 0
    node = root
    while node.expanded() and not node.is_terminal:
        assert node.visit_counts is not None and node.legal_actions is not None
        best_idx = int(np.argmax(node.visit_counts))
        if int(node.visit_counts[best_idx]) == 0:
            break
        best_action = int(node.legal_actions[best_idx])
        child = node.children.get(best_action)
        if child is None:
            break
        greedy_depth += 1
        node = child

    return {
        "legal_actions": legal_actions,
        "visited_actions": visited_actions,
        "top_share": top_share,
        "top_gap_pp": top_gap_pp,
        "entropy": entropy,
        "effective_actions": effective_actions,
        "mean_depth": mean_depth,
        "max_depth": max_depth,
        "greedy_depth": greedy_depth,
    }


def _format_mcts_stats_step(
    step: int,
    turn: int,
    active_player: int,
    phase_name: str,
    action_str: str,
    metrics: dict[str, float | int],
    search_stats: SearchStats,
    auto_count: int,
) -> str:
    """Format one compact per-move MCTS summary line."""
    avg_eval_batch = (
        search_stats.total_leaves / search_stats.num_eval_batches
        if search_stats.num_eval_batches > 0 else 0.0
    )
    return (
        f"{step:03d} T{turn:02d} P{active_player} {phase_name:<14} | "
        f"legal={metrics['legal_actions']:2d} vis={metrics['visited_actions']:2d} "
        f"top={metrics['top_share'] * 100:5.1f}% gap={metrics['top_gap_pp']:5.1f}pp "
        f"eff={metrics['effective_actions']:4.1f} "
        f"depth={metrics['mean_depth']:4.1f}/{metrics['max_depth']:2d}/{metrics['greedy_depth']:2d} "
        f"batches={search_stats.num_eval_batches:3d} avgB={avg_eval_batch:4.1f} "
        f"vb={search_stats.virtual_backups:2d} auto={auto_count} | {action_str}"
    )


def _append_mcts_stats_summary(
    lines: list[str],
    per_move_stats: list[dict[str, float | int]],
    num_players: int,
    state: GameState,
    total_vbackups: int,
    terminal_rank_weight: float,
) -> None:
    """Append compact end-of-game aggregates for stats-only mode."""
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    move_count = len(per_move_stats)
    lines.append(
        f"Completed in {move_count} decision points "
        f"({total_vbackups} virtual backups from subtree reuse)"
    )

    if per_move_stats:
        top_shares = np.array([row["top_share"] for row in per_move_stats], dtype=np.float64)
        top_gaps = np.array([row["top_gap_pp"] for row in per_move_stats], dtype=np.float64)
        visited = np.array([row["visited_actions"] for row in per_move_stats], dtype=np.float64)
        eff_actions = np.array([row["effective_actions"] for row in per_move_stats], dtype=np.float64)
        mean_depths = np.array([row["mean_depth"] for row in per_move_stats], dtype=np.float64)
        max_depths = np.array([row["max_depth"] for row in per_move_stats], dtype=np.float64)
        greedy_depths = np.array([row["greedy_depth"] for row in per_move_stats], dtype=np.float64)
        eval_batches = np.array([row["num_eval_batches"] for row in per_move_stats], dtype=np.float64)
        avg_eval_batch = np.array([row["avg_eval_batch"] for row in per_move_stats], dtype=np.float64)
        sel_ms = np.array([row["selection_ms"] for row in per_move_stats], dtype=np.float64)
        eval_ms = np.array([row["eval_ms"] for row in per_move_stats], dtype=np.float64)
        bak_ms = np.array([row["backup_ms"] for row in per_move_stats], dtype=np.float64)

        high_conf = int(np.count_nonzero(top_shares >= 0.9))
        lines.append(
            "Root concentration: "
            f"top avg={top_shares.mean() * 100:.1f}% med={np.median(top_shares) * 100:.1f}% "
            f"| gap avg={top_gaps.mean():.1f}pp | high-conf(>=90%)={high_conf}"
        )
        lines.append(
            "Breadth/depth: "
            f"visited avg={visited.mean():.1f} | eff avg={eff_actions.mean():.1f} "
            f"| mean depth avg={mean_depths.mean():.1f} "
            f"| max depth avg={max_depths.mean():.1f} "
            f"| greedy depth avg={greedy_depths.mean():.1f}"
        )
        lines.append(
            "Batching/timing avg/move: "
            f"eval batches={eval_batches.mean():.1f} | avg leaf batch={avg_eval_batch.mean():.2f} "
            f"| select={sel_ms.mean():.1f}ms eval={eval_ms.mean():.1f}ms backup={bak_ms.mean():.1f}ms"
        )

    net_worths = [PLAYERS[pid].get_net_worth(state) for pid in range(num_players)]
    winner = max(range(num_players), key=lambda i: net_worths[i])
    nw_str = ", ".join(f"P{pid}=${net_worths[pid]}" for pid in range(num_players))
    lines.append(f"Net worths: {nw_str}")
    lines.append(f"Winner: P{winner} (${net_worths[winner]})")

    terminal_values = compute_terminal_values(net_worths, num_players, terminal_rank_weight)
    tv_parts = [f"P{i}={terminal_values[i]:+.3f}" for i in range(num_players)]
    lines.append(f"Terminal values (blend={terminal_rank_weight}): {', '.join(tv_parts)}")


def analyze_game(
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    seed: int,
    num_simulations: int,
    search_batch_size: int = 8,
    top_n: int = 10,
    verbose: bool = False,
    *,
    dirichlet_epsilon: float | None = None,
    dirichlet_dynamic: bool | None = None,
    terminal_blend: float | None = None,
    c_puct: float | None = None,
    mcts_stats_only: bool = False,
    token_dump: bool = False,
    skip_static_tokens: bool = False,
) -> str:
    """Play a self-play game with full MCTS and return a detailed log."""
    num_players = config.num_players

    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)

    terminal_rank_weight = terminal_blend if terminal_blend is not None else config.terminal_blend
    evaluator = NNEvaluator(
        model, device, num_players=num_players,
        terminal_rank_weight=terminal_rank_weight,
        eval_dtype=config.eval_dtype,
    )
    mcts_config = config.to_mcts_config(c_puct_override=c_puct)
    mcts_config = MCTSConfig(
        num_simulations=num_simulations,
        c_puct=mcts_config.c_puct,
        dirichlet_alpha=mcts_config.dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon if dirichlet_epsilon is not None else mcts_config.dirichlet_epsilon,
        dirichlet_dynamic=dirichlet_dynamic if dirichlet_dynamic is not None else mcts_config.dirichlet_dynamic,
        dirichlet_alpha_numerator=mcts_config.dirichlet_alpha_numerator,
        num_players=num_players,
        search_batch_size=search_batch_size,
    )

    layout = get_layout(num_players)
    state_pool = StatePool(2 * (num_simulations + 1), layout.total_size)
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    token_normalization = TokenNormalizationAccumulator(num_players) if token_dump else None
    noise_desc = f"epsilon={mcts_config.dirichlet_epsilon}"
    if mcts_config.dirichlet_epsilon > 0:
        if mcts_config.dirichlet_dynamic:
            noise_desc += f", dynamic alpha={mcts_config.dirichlet_alpha_numerator}/K"
        else:
            noise_desc += f", alpha={mcts_config.dirichlet_alpha}"
    if mcts_stats_only:
        lines.append(
            f"# Self-Play MCTS Stats: seed={seed}, {num_simulations} simulations/move, "
            f"batch={search_batch_size}"
        )
    else:
        lines.append(f"# Self-Play Analysis: seed={seed}, {num_simulations} simulations/move")
    lines.append(f"# Noise: {noise_desc} | Terminal blend: {terminal_rank_weight}")
    if mcts_stats_only:
        lines.append(
            "# Columns: step turn player phase | legal visited top gap eff "
            "depth(mean/max/greedy) batches avgB vb auto | action"
        )
    else:
        lines.append("")
        lines.append(format_state_full(state))
        lines.append("")
        lines.append("---")
        lines.append("")

    step = 0
    prev_phase = TURN.get_phase(state)
    prev_turn = TURN.get_turn_number(state)
    reuse_root: MCTSNode | None = None
    total_vbackups = 0
    per_move_stats: list[dict[str, float | int]] = []

    while TURN.get_phase(state) != GamePhases.PHASE_GAME_OVER:
        active_player = TURN.get_active_player(state)
        engine_phase = TURN.get_phase(state)
        cur_phase = PHASE_NAMES.get(engine_phase, str(engine_phase))
        cur_turn = int(TURN.get_turn_number(state))
        # Phase id is needed for action rendering in both modes, so compute
        # it unconditionally — cheap lookup off the state.
        phase_id = get_decision_phase_py(state)

        token_dump_text: str | None = None
        if token_normalization is not None:
            token_buffer = token_normalization.add_state(state)
            token_dump_text = format_token_dump_from_buffer(
                token_buffer,
                token_normalization.widths,
                token_normalization.labels,
                skip_static_tokens=skip_static_tokens,
            )

        # Raw NN evaluation for the log. Only needed for the full log mode;
        # stats-only mode skips this extra forward pass. Keeping the raw
        # priors out of MCTSNode keeps per-node memory tight in self-play,
        # at the cost of one extra root-only forward pass here.
        priors: np.ndarray | None = None
        values: np.ndarray | None = None
        action_ids_arr: np.ndarray | None = None
        if not mcts_stats_only:
            priors, values, action_ids_arr, _, _ = evaluator.evaluate(state)

        # MCTS search (reuses subtree from previous move when available)
        search_stats = SearchStats()
        root = run_search(
            state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats,
        )

        # Choose action (argmax = best play)
        assert root.legal_actions is not None and root.visit_counts is not None
        action = int(root.legal_actions[np.argmax(root.visit_counts)])
        action_str = format_action(phase_id, action, state)

        # Build noised prior map if noise was applied. Root priors are
        # aligned to root.legal_actions (sparse, phase-local ids).
        noised_map: dict[int, float] | None = None
        if (not mcts_stats_only
                and mcts_config.dirichlet_epsilon > 0
                and root.priors is not None):
            noised_map = {
                int(root.legal_actions[i]): float(root.priors[i])
                for i in range(len(root.legal_actions))
            }

        metrics: dict[str, float | int] | None = None
        if mcts_stats_only:
            metrics = _compute_tree_metrics(root)
        else:
            assert priors is not None and values is not None and action_ids_arr is not None
            # Log this decision point
            lines.append(f"### Step {step}: P{active_player} [{cur_phase}]")
            lines.append("")
            phase_ctx = format_phase_context(state)
            if phase_ctx:
                lines.append(f"  {phase_ctx}")
                lines.append("")
            if token_dump_text is not None:
                lines.append("## Token Dump")
                lines.append("")
                lines.append(token_dump_text)
                lines.append("")
            lines.extend(_format_nn_eval(
                priors, values, action_ids_arr, phase_id, num_players, state, top_n,
                noised_priors=noised_map,
            ))
            lines.append("")
            lines.extend(_format_mcts_visits(root, phase_id, state, top_n))
            a0gb = get_greedy_leaf_value(root, num_players)
            a0gb_parts = [f"P{i}={a0gb[i]:+.3f}" for i in range(num_players)]
            vb = search_stats.virtual_backups
            lines.append(f"  A0GB Value: {', '.join(a0gb_parts)}{f' (vbackups: {vb})' if vb > 0 else ''}")
            lines.append("")
            lines.append(f"  **Action: {action_str}**")
            lines.append("")

        # Apply action. Driver records (pre_state_copy, phase_id, action_id)
        # per dispatch — phase_id == -1 flags an automated engine phase
        # transition (WRAP_UP / INCOME / END_CARD), in which case the third
        # slot holds the engine phase id rather than an action id.
        history: list[tuple[np.ndarray, int, int]] = []
        status = DRIVER.apply_action(state, action, history=history)
        auto_count = max(len(history) - 1, 0)
        total_vbackups += search_stats.virtual_backups

        if mcts_stats_only:
            assert metrics is not None
            lines.append(_format_mcts_stats_step(
                step, cur_turn, active_player, cur_phase, action_str,
                metrics, search_stats, auto_count,
            ))
            if token_dump_text is not None:
                lines.append("## Token Dump")
                lines.append("")
                lines.append(token_dump_text)
                lines.append("")
            per_move_stats.append({
                **metrics,
                "num_eval_batches": search_stats.num_eval_batches,
                "avg_eval_batch": (
                    search_stats.total_leaves / search_stats.num_eval_batches
                    if search_stats.num_eval_batches > 0 else 0.0
                ),
                "selection_ms": search_stats.selection_secs * 1000.0,
                "eval_ms": search_stats.eval_secs * 1000.0,
                "backup_ms": search_stats.backup_secs * 1000.0,
            })

        # Show auto-applied actions
        if not mcts_stats_only and len(history) > 1:
            for pre_state, h_phase_id, h_action_id in history[1:]:
                history_state = GameState.from_array(pre_state, num_players)
                auto_str = format_action(h_phase_id, h_action_id, history_state)
                lines.append(f"  \u21b3 auto: {auto_str}")
            lines.append("")

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action, state_pool)

        step += 1
        new_phase = TURN.get_phase(state)
        new_turn = TURN.get_turn_number(state)

        # State dump on phase/turn change (or every step if verbose)
        if (not mcts_stats_only
                and (verbose or new_phase != prev_phase or new_turn != prev_turn)):
            if new_turn != prev_turn:
                lines.append(f"--- Turn {new_turn} ---")
                lines.append("")
            lines.append(format_state_full(state))
            lines.append("")

        prev_phase = new_phase
        prev_turn = new_turn

        if status == STATUS_GAME_OVER:
            break

    if mcts_stats_only:
        _append_mcts_stats_summary(
            lines, per_move_stats, num_players, state,
            total_vbackups, terminal_rank_weight,
        )
    else:
        # Game over summary
        lines.append("---")
        lines.append("")
        lines.append("## Game Over")
        lines.append("")
        lines.append(f"Completed in {step} decision points ({total_vbackups} virtual backups from subtree reuse)")
        lines.append("")

        net_worths = [PLAYERS[pid].get_net_worth(state) for pid in range(num_players)]
        for pid in range(num_players):
            lines.append(f"  P{pid}: net worth ${net_worths[pid]}")

        winner = max(range(num_players), key=lambda i: net_worths[i])
        lines.append("")
        lines.append(f"**Winner: P{winner} (${net_worths[winner]})**")

        # Terminal reward values
        terminal_values = compute_terminal_values(net_worths, num_players, terminal_rank_weight)
        tv_parts = [f"P{i}={terminal_values[i]:+.3f}" for i in range(num_players)]
        lines.append(f"Terminal values (blend={terminal_rank_weight}): {', '.join(tv_parts)}")

    if token_normalization is not None:
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Token Normalization Report")
        lines.append("")
        lines.append(token_normalization.format_report(skip_static_tokens=skip_static_tokens))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a self-play game using a trained checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help='Path to checkpoint file, "latest" for the newest in --checkpoint-dir, '
             'or "new" for a freshly-initialized (untrained) model',
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--num-players", type=int, default=3,
        help='Player count when using "new" (3-5). Ignored for loaded checkpoints — '
             'num_players comes from the checkpoint config.',
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--search-batch-size", type=int, default=8)
    parser.add_argument(
        "--c-puct", type=float, default=None,
        help="Override c_puct for this analysis run (default from checkpoint)",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Top N actions to show")
    parser.add_argument("--verbose", action="store_true", help="Full state dump every step")
    parser.add_argument(
        "--token-dump",
        action="store_true",
        help="Dump denormalized token rows for every decision state",
    )
    parser.add_argument(
        "--skip-static-tokens",
        action="store_true",
        help="With --token-dump, omit the static prefix (market_prices + company[0..35])",
    )
    parser.add_argument(
        "--mcts-stats-only", action="store_true",
        help="Compact one-line-per-move MCTS summary plus end-of-game aggregates",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file (default: stdout)")
    parser.add_argument(
        "--terminal-blend", type=float, default=None,
        help="Rank vs margin weight for terminal rewards (0=margin, 1=rank, default from checkpoint)",
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--no-dirichlet-noise", dest="dirichlet_epsilon", action="store_const", const=0.0,
        help="Disable Dirichlet noise at root (pure NN priors)",
    )
    noise_group.add_argument(
        "--dirichlet-epsilon", type=float, default=None,
        help="Dirichlet noise epsilon (default from checkpoint)",
    )
    dyn_group = parser.add_mutually_exclusive_group()
    dyn_group.add_argument(
        "--dynamic-dirichlet", dest="dirichlet_dynamic", action="store_true", default=None,
        help="Use dynamic alpha = numerator / n_legal_actions",
    )
    dyn_group.add_argument(
        "--no-dynamic-dirichlet", dest="dirichlet_dynamic", action="store_false",
        help="Use static alpha",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (or build a fresh untrained model when "new")
    if args.checkpoint == "new":
        assert 3 <= args.num_players <= 5, \
            f"--num-players must be in [3, 5], got {args.num_players}"
        config = TrainingConfig(num_players=args.num_players)
        model = create_model(num_players=args.num_players).to(device)
        model.eval()
        print(f"Using freshly-initialized (untrained) model, num_players={args.num_players}, device={device}")
    else:
        if args.checkpoint == "latest":
            cp_path = find_latest_checkpoint(Path(args.checkpoint_dir))
            if cp_path is None:
                print(f"No checkpoint found in {args.checkpoint_dir}")
                return
        else:
            cp_path = Path(args.checkpoint)

        print(f"Loading checkpoint: {cp_path}")
        cp = load_checkpoint(cp_path, device)
        config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

        model = create_model(num_players=config.num_players).to(device)
        model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        model.eval()

        epoch = cp.get("epoch", "?")
        print(f"Model from epoch {epoch}, device={device}")
    print(f"Running {args.simulations} MCTS simulations per move...")
    print()

    result = analyze_game(
        model, device, config, args.seed, args.simulations,
        args.search_batch_size, args.top_n, args.verbose,
        dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_dynamic=args.dirichlet_dynamic,
        terminal_blend=args.terminal_blend,
        c_puct=args.c_puct,
        mcts_stats_only=args.mcts_stats_only,
        token_dump=args.token_dump,
        skip_static_tokens=args.skip_static_tokens,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
            f.write("\n")
        print(f"Game log written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
