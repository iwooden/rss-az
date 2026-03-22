"""Play and log a self-play game using a trained model checkpoint.

Shows NN evaluation (priors, values) and MCTS visit counts at each
decision point to visualize what the model has learned.

Usage:
    .venv/bin/python -m train.analyze_game CHECKPOINT [options]

    .venv/bin/python -m train.analyze_game checkpoints/checkpoint_epoch_0005.pt
    .venv/bin/python -m train.analyze_game latest --checkpoint-dir checkpoints
    .venv/bin/python -m train.analyze_game latest --seed 123 --simulations 200
    .venv/bin/python -m train.analyze_game latest --output game_log.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from core.data import GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.state import GameState, get_layout
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, compute_terminal_values
from mcts.search import StatePool, run_search, get_greedy_leaf_value
from nn import create_model
from tests.debug_trace import (
    format_action,
    format_state_full,
    PHASE_NAMES,
)
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import MCTSConfig, TrainingConfig


def _format_nn_eval(
    policy_probs: np.ndarray,
    values: np.ndarray,
    legal_mask: np.ndarray,
    num_players: int,
    state: GameState,
    top_n: int = 10,
) -> list[str]:
    """Format NN evaluation into readable lines."""
    lines = []

    # Values (canonical order)
    val_parts = [f"P{i}={values[i]:+.3f}" for i in range(num_players)]
    lines.append(f"  NN Values: {', '.join(val_parts)}")

    # Top priors for legal actions
    legal_indices = np.flatnonzero(legal_mask)
    legal_priors = policy_probs[legal_indices]
    sorted_order = np.argsort(-legal_priors)[:top_n]

    lines.append(f"  NN Priors (top {min(top_n, len(sorted_order))} of {len(legal_indices)} legal):")
    for rank, idx in enumerate(sorted_order):
        action_idx = int(legal_indices[idx])
        prior = legal_priors[idx]
        action_str = format_action(action_idx, num_players, state)
        bar = "\u2588" * int(prior * 40)
        lines.append(f"    {rank+1:2d}. {prior:6.1%} {bar} {action_str}")

    return lines


def _format_mcts_visits(
    root,
    num_players: int,
    state: GameState,
    top_n: int = 10,
) -> list[str]:
    """Format MCTS visit counts into readable lines."""
    lines = []

    if root.legal_actions is None or root.visit_counts is None:
        lines.append("  MCTS: (no search data)")
        return lines

    counts = root.visit_counts
    total_visits = int(counts.sum())
    sorted_order = np.argsort(-counts)[:top_n]

    lines.append(f"  MCTS Visits (top {min(top_n, len(sorted_order))}, {total_visits} total):")
    for rank, idx in enumerate(sorted_order):
        action_idx = int(root.legal_actions[idx])
        visits = int(counts[idx])
        if visits == 0:
            break
        pct = visits / max(total_visits, 1)
        action_str = format_action(action_idx, num_players, state)

        # Show Q value for this action (max(1,vc) matches select_child convention)
        q_val = float(root.value_sums[idx, root.active_player_id] / max(visits, 1))
        bar = "\u2588" * int(pct * 40)
        lines.append(f"    {rank+1:2d}. {visits:5d} ({pct:5.1%}) Q={q_val:+.3f} {bar} {action_str}")

    return lines


def analyze_game(
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    seed: int,
    num_simulations: int,
    search_batch_size: int = 1,
    top_n: int = 10,
    verbose: bool = False,
    *,
    dirichlet_epsilon: float | None = None,
    dirichlet_dynamic: bool | None = None,
    terminal_blend: float | None = None,
) -> str:
    """Play a self-play game with full MCTS and return a detailed log."""
    num_players = config.num_players

    state = GameState(num_players)
    state.initialize_game(seed=seed)

    terminal_rank_weight = terminal_blend if terminal_blend is not None else config.terminal_blend
    evaluator = NNEvaluator(model, device, num_players=num_players)
    mcts_config = config.to_mcts_config()
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
    state_pool = StatePool(num_simulations + 1, layout.total_size)
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    noise_desc = f"epsilon={mcts_config.dirichlet_epsilon}"
    if mcts_config.dirichlet_epsilon > 0:
        if mcts_config.dirichlet_dynamic:
            noise_desc += f", dynamic alpha={mcts_config.dirichlet_alpha_numerator}/K"
        else:
            noise_desc += f", alpha={mcts_config.dirichlet_alpha}"
    lines.append(f"# Self-Play Analysis: seed={seed}, {num_simulations} simulations/move")
    lines.append(f"# Noise: {noise_desc} | Terminal blend: {terminal_rank_weight}")
    lines.append("")
    lines.append(format_state_full(state))
    lines.append("")
    lines.append("---")
    lines.append("")

    step = 0
    prev_phase = state.get_phase()
    prev_turn = TURN.get_turn_number(state)

    while state.get_phase() != GamePhases.PHASE_GAME_OVER:
        active_player = state.get_active_player()
        cur_phase = PHASE_NAMES.get(state.get_phase(), str(state.get_phase()))

        # NN evaluation (raw, before MCTS)
        policy_probs, values, legal_mask = evaluator.evaluate(state)

        # MCTS search
        root = run_search(state, evaluator, mcts_config, rng, state_pool=state_pool)

        # Choose action (argmax = best play)
        assert root.legal_actions is not None and root.visit_counts is not None
        action = int(root.legal_actions[np.argmax(root.visit_counts)])
        action_str = format_action(action, num_players, state)

        # Log this decision point
        lines.append(f"### Step {step}: P{active_player} [{cur_phase}]")
        lines.append("")
        lines.extend(_format_nn_eval(
            policy_probs, values, legal_mask, num_players, state, top_n
        ))
        lines.append("")
        lines.extend(_format_mcts_visits(root, num_players, state, top_n))
        a0gb = get_greedy_leaf_value(root, num_players)
        a0gb_parts = [f"P{i}={a0gb[i]:+.3f}" for i in range(num_players)]
        lines.append(f"  A0GB Value: {', '.join(a0gb_parts)}")
        lines.append("")
        lines.append(f"  **Action: {action_str}**")
        lines.append("")

        # Apply action
        history: list[tuple[int, int]] = []
        status = DRIVER.apply_action(state, action, history=history)

        # Show auto-applied actions
        if len(history) > 1:
            for _, aid in history[1:]:
                auto_str = format_action(aid, num_players)
                lines.append(f"  \u21b3 auto: {auto_str}")
            lines.append("")

        step += 1
        new_phase = state.get_phase()
        new_turn = TURN.get_turn_number(state)

        # State dump on phase/turn change (or every step if verbose)
        if verbose or new_phase != prev_phase or new_turn != prev_turn:
            if new_turn != prev_turn:
                lines.append(f"--- Turn {new_turn} ---")
                lines.append("")
            lines.append(format_state_full(state))
            lines.append("")

        prev_phase = new_phase
        prev_turn = new_turn

        if status == STATUS_GAME_OVER:
            break

    # Game over summary
    lines.append("---")
    lines.append("")
    lines.append("## Game Over")
    lines.append("")
    lines.append(f"Completed in {step} decision points")
    lines.append("")

    from entities.player import PLAYERS
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

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a self-play game using a trained checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help='Path to checkpoint file, or "latest"',
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simulations", type=int, default=800)
    parser.add_argument("--search-batch-size", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=10, help="Top N actions to show")
    parser.add_argument("--verbose", action="store_true", help="Full state dump every step")
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

    # Load checkpoint
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

    # Build model
    model = create_model(
        config.model_arch,
        input_dim=config.visible_size,
        action_dim=config.action_dim,
        value_dim=config.num_players,
    ).to(device)
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
