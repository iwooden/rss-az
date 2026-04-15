"""Tournament between model checkpoints.

Play round-robin games between different training checkpoints
to evaluate relative model strength over time.

Usage:
    .venv/bin/python -m train.tournament cp1.pt,cp2.pt,cp3.pt [options]

    # Compare 3 checkpoints with 200 sims/move
    .venv/bin/python -m train.tournament \\
        checkpoints/checkpoint_epoch_0005.pt,\\
        checkpoints/checkpoint_epoch_0010.pt,\\
        checkpoints/checkpoint_epoch_0015.pt \\
        --simulations 200

    # Quick comparison of 2 checkpoints
    .venv/bin/python -m train.tournament cp_old.pt,cp_new.pt --simulations 100
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from core.data import GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.state import GameState, get_layout
from entities.player import PLAYERS
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, run_search
from nn import create_model
from train.checkpoint import load_checkpoint
from train.config import MCTSConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """A loaded checkpoint ready for tournament play."""
    path: Path
    epoch: int
    model: torch.nn.Module
    label: str  # short display name


def _load_model(cp_path: Path, device: torch.device) -> tuple[torch.nn.Module, TrainingConfig, int]:
    """Load model from checkpoint. Returns (model, config, epoch)."""
    cp = load_checkpoint(cp_path, device)
    config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]
    model = create_model(num_players=config.num_players).to(device)
    model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    model.eval()
    epoch = int(cp.get("epoch", -1))  # type: ignore[arg-type]
    return model, config, epoch


# ---------------------------------------------------------------------------
# Game play
# ---------------------------------------------------------------------------

def _play_game(
    evaluators: list[NNEvaluator],
    seat_to_model: list[int],
    num_players: int,
    mcts_config: MCTSConfig,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool,
) -> list[int]:
    """Play one tournament game. Returns net worths per seat."""
    state = GameState(num_players)
    state.initialize_game(seed=game_seed)

    while state.get_phase() != GamePhases.PHASE_GAME_OVER:
        active_player = state.get_active_player()
        evaluator = evaluators[seat_to_model[active_player]]

        # Fresh search each move (no subtree reuse — different models)
        root = run_search(state, evaluator, mcts_config, rng, state_pool=state_pool)

        assert root.legal_actions is not None and root.visit_counts is not None
        action = int(root.legal_actions[np.argmax(root.visit_counts)])

        history: list[tuple[int, int]] = []
        status = DRIVER.apply_action(state, action, history=history)
        if status == STATUS_GAME_OVER:
            break

    return [PLAYERS[pid].get_net_worth(state) for pid in range(num_players)]


# ---------------------------------------------------------------------------
# Matchup scheduling
# ---------------------------------------------------------------------------

def _generate_schedule(
    num_models: int, min_games_per_pair: int, num_players: int,
) -> list[tuple[tuple[int, ...], int]]:
    """Generate tournament schedule as (model_group, num_games) pairs.

    For num_models < num_players, duplicates models to fill seats.
    For num_models >= num_players, uses all C(N, num_players) combinations
    with enough games so every pair plays together at least
    ``min_games_per_pair`` times.
    """
    if num_models < num_players:
        # Fill seats by cycling models (e.g. 2 models in a 4-player game)
        groups: list[tuple[tuple[int, ...], int]] = []
        base = list(range(num_models))
        while len(base) < num_players:
            base.append(base[len(base) % num_models])
        perms = list(set(itertools.permutations(base)))
        games_each = max(1, math.ceil(min_games_per_pair / len(perms)))
        for p in perms:
            groups.append((p, games_each))
        return groups

    combos = list(itertools.combinations(range(num_models), num_players))
    # Each pair co-occurs in C(N-2, num_players-2) combos
    co_occurrence = math.comb(num_models - 2, num_players - 2) if num_models > 2 else 1
    games_per_combo = math.ceil(min_games_per_pair / max(co_occurrence, 1))
    return [(c, games_per_combo) for c in combos]


def _rank_players(net_worths: list[int]) -> list[int]:
    """Convert net worths to 1-indexed ranks (1=best). Ties share rank."""
    sorted_nw = sorted(enumerate(net_worths), key=lambda x: -x[1])
    ranks = [0] * len(net_worths)
    for i, (idx, nw) in enumerate(sorted_nw):
        if i > 0 and nw < sorted_nw[i - 1][1]:
            ranks[idx] = i + 1
        else:
            ranks[idx] = (ranks[sorted_nw[i - 1][0]] if i > 0 else 1)
    return ranks


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class PairStats:
    """Head-to-head results for one ordered model pair (a vs b)."""
    finishes: list[int] = field(default_factory=list)  # per-place counts (1st, 2nd, ...)
    wins: int = 0   # times a ranked strictly better than b
    games: int = 0  # games where both a and b played


@dataclass
class GameResult:
    """Record of one tournament game."""
    seat_to_model: list[int]
    net_worths: list[int]
    ranks: list[int]
    seed: int


def _collect_pair_stats(
    results: list[GameResult], num_models: int, num_players: int,
) -> dict[tuple[int, int], PairStats]:
    """Aggregate per-pair statistics from game results."""
    stats: dict[tuple[int, int], PairStats] = {}
    for a in range(num_models):
        for b in range(num_models):
            if a != b:
                stats[(a, b)] = PairStats(finishes=[0] * num_players)

    for gr in results:
        # Map model -> best rank achieved in this game
        model_best_rank: dict[int, int] = {}
        for seat, model_idx in enumerate(gr.seat_to_model):
            r = gr.ranks[seat]
            if model_idx not in model_best_rank or r < model_best_rank[model_idx]:
                model_best_rank[model_idx] = r

        models_in_game = list(model_best_rank.keys())
        for i, model_a in enumerate(models_in_game):
            for model_b in models_in_game[i + 1:]:
                rank_a = model_best_rank[model_a]
                rank_b = model_best_rank[model_b]

                # Update a vs b
                ps = stats[(model_a, model_b)]
                ps.games += 1
                if rank_a <= num_players:
                    ps.finishes[rank_a - 1] += 1
                if rank_a < rank_b:
                    ps.wins += 1

                # Update b vs a
                ps = stats[(model_b, model_a)]
                ps.games += 1
                if rank_b <= num_players:
                    ps.finishes[rank_b - 1] += 1
                if rank_b < rank_a:
                    ps.wins += 1

    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _format_report(
    entries: list[ModelEntry],
    results: list[GameResult],
    stats: dict[tuple[int, int], PairStats],
    elapsed: float,
    num_players: int,
) -> str:
    """Format tournament results as a readable report."""
    lines: list[str] = []
    total_games = len(results)
    rank_labels = ["1st", "2nd", "3rd", "4th", "5th"][:num_players]

    lines.append(f"# Tournament Report ({total_games} games, {elapsed:.1f}s)")
    lines.append("")
    lines.append("## Models")
    for i, e in enumerate(entries):
        lines.append(f"  [{i}] {e.label}  ({e.path.name})")
    lines.append("")

    for i, entry in enumerate(entries):
        lines.append(f"## [{i}] {entry.label}")
        lines.append("")
        rank_hdr = "  ".join(f"{lbl:>5s}" for lbl in rank_labels)
        lines.append(f"  {'Opponent':<30s}  {rank_hdr}  {'Better':>10s}  {'Games':>5s}")
        rank_sep = "  ".join("-" * 5 for _ in rank_labels)
        lines.append(f"  {'-' * 30}  {rank_sep}  {'-' * 10}  {'-' * 5}")
        for j, opp in enumerate(entries):
            if i == j:
                continue
            ps = stats[(i, j)]
            if ps.games == 0:
                continue
            better_str = f"{ps.wins}/{ps.games}"
            rank_vals = "  ".join(f"{ps.finishes[k]:>5d}" for k in range(num_players))
            lines.append(
                f"  {f'[{j}] {opp.label}':<30s}  {rank_vals}  "
                f"{better_str:>10s}  {ps.games:>5d}"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tournament(
    entries: list[ModelEntry],
    device: torch.device,
    num_players: int,
    mcts_config: MCTSConfig,
    min_games_per_pair: int,
    base_seed: int,
    terminal_rank_weight: float,
) -> tuple[list[GameResult], float]:
    """Run the full tournament. Returns (results, elapsed_seconds)."""
    evaluators = [
        NNEvaluator(e.model, device, num_players=num_players,
                     terminal_rank_weight=terminal_rank_weight)
        for e in entries
    ]

    layout = get_layout(num_players)
    state_pool = StatePool(2 * (mcts_config.num_simulations + 1), layout.total_size)
    rng = np.random.default_rng(base_seed)

    schedule = _generate_schedule(len(entries), min_games_per_pair, num_players)
    total_games = sum(g for _, g in schedule)

    print(f"Tournament: {len(entries)} models, {total_games} games scheduled")
    print(f"  Simulations/move: {mcts_config.num_simulations}, "
          f"batch size: {mcts_config.search_batch_size}")
    print()

    results: list[GameResult] = []
    game_num = 0
    t0 = time.perf_counter()

    for triple, num_games in schedule:
        # All seat permutations for this triple, cycled over num_games
        perms = list(itertools.permutations(triple))
        for g in range(num_games):
            seat_to_model = list(perms[g % len(perms)])
            game_seed = rng.integers(0, 2**31)

            t_game = time.perf_counter()
            net_worths = _play_game(
                evaluators, seat_to_model, num_players,
                mcts_config, int(game_seed), rng, state_pool,
            )
            ranks = _rank_players(net_worths)
            dt = time.perf_counter() - t_game

            results.append(GameResult(seat_to_model, net_worths, ranks, int(game_seed)))
            game_num += 1

            # Progress line
            seat_desc = ", ".join(
                f"P{s}=[{m}]" for s, m in enumerate(seat_to_model)
            )
            rank_desc = ", ".join(
                f"[{seat_to_model[s]}]=${net_worths[s]}(#{ranks[s]})"
                for s in range(num_players)
            )
            print(f"  Game {game_num}/{total_games} ({dt:.1f}s): "
                  f"{seat_desc} → {rank_desc}")

    elapsed = time.perf_counter() - t0
    print(f"\nAll {total_games} games completed in {elapsed:.1f}s")
    return results, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tournament between model checkpoints"
    )
    parser.add_argument(
        "checkpoints",
        type=str,
        help="Comma-separated list of checkpoint file paths",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-batch-size", type=int, default=1,
                        help="Batched leaf evaluation size (default: 1)")
    parser.add_argument("--games-per-pair", type=int, default=10,
                        help="Minimum games per model pair (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument(
        "--terminal-blend", type=float, default=None,
        help="Rank vs margin weight for terminal rewards "
             "(0=margin, 1=rank, default from first checkpoint)",
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--no-dirichlet-noise", dest="dirichlet_epsilon",
        action="store_const", const=0.0,
        help="Disable Dirichlet noise at root",
    )
    noise_group.add_argument(
        "--dirichlet-epsilon", type=float, default=None,
        help="Dirichlet noise epsilon (default from checkpoint)",
    )
    dyn_group = parser.add_mutually_exclusive_group()
    dyn_group.add_argument(
        "--dynamic-dirichlet", dest="dirichlet_dynamic",
        action="store_true", default=None,
        help="Use dynamic alpha = numerator / n_legal_actions",
    )
    dyn_group.add_argument(
        "--no-dynamic-dirichlet", dest="dirichlet_dynamic",
        action="store_false",
        help="Use static alpha",
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse checkpoint paths
    cp_paths = [Path(p.strip()) for p in args.checkpoints.split(",")]
    if len(cp_paths) < 2:
        print("Error: need at least 2 checkpoint paths (comma-separated)")
        sys.exit(1)
    for p in cp_paths:
        if not p.exists():
            print(f"Error: checkpoint not found: {p}")
            sys.exit(1)

    # Load all models
    print(f"Loading {len(cp_paths)} checkpoints on {device}...")
    entries: list[ModelEntry] = []
    ref_config: TrainingConfig | None = None

    for i, cp_path in enumerate(cp_paths):
        model, config, epoch = _load_model(cp_path, device)
        if ref_config is None:
            ref_config = config
        else:
            if config.num_players != ref_config.num_players:
                print(f"Error: checkpoint {cp_path} has num_players="
                      f"{config.num_players}, expected {ref_config.num_players}")
                sys.exit(1)

        label = f"epoch {epoch}" if epoch >= 0 else f"model {i}"
        entries.append(ModelEntry(cp_path, epoch, model, label))
        print(f"  [{i}] {label}: {cp_path.name}")

    assert ref_config is not None
    print()

    # Build MCTS config from first checkpoint + CLI overrides
    base_mcts = ref_config.to_mcts_config()
    terminal_blend = (args.terminal_blend if args.terminal_blend is not None
                      else ref_config.terminal_blend)
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        c_puct=base_mcts.c_puct,
        dirichlet_alpha=base_mcts.dirichlet_alpha,
        dirichlet_epsilon=(args.dirichlet_epsilon if args.dirichlet_epsilon is not None
                           else base_mcts.dirichlet_epsilon),
        dirichlet_dynamic=(args.dirichlet_dynamic if args.dirichlet_dynamic is not None
                           else base_mcts.dirichlet_dynamic),
        dirichlet_alpha_numerator=base_mcts.dirichlet_alpha_numerator,
        num_players=ref_config.num_players,
        search_batch_size=args.search_batch_size,
    )

    # Run tournament
    results, elapsed = run_tournament(
        entries, device, ref_config.num_players, mcts_config,
        args.games_per_pair, args.seed, terminal_blend,
    )

    # Build report
    stats = _collect_pair_stats(results, len(entries), ref_config.num_players)
    report = _format_report(entries, results, stats, elapsed, ref_config.num_players)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
            f.write("\n")
        print(f"\nReport written to {args.output}")
    else:
        print()
        print(report)


if __name__ == "__main__":
    main()
