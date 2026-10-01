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

    # One checkpoint at two c_puct values
    .venv/bin/python -m train.tournament cp.pt@c_puct=1.2,cp.pt@c_puct=1.7

Checkpoints trained under different engine rules can share a game. The game
uses the most permissive checkpoint rules (v3 behavior and cross-president
offers if any checkpoint was trained with them). Each checkpoint searches under
its own training rules within the game's: a model trained with same-president
acquisitions searches only those, and a legacy (v2) model sees legacy
trade-history lifetime. A model trained without cross-president offers answers
incoming ones with equal accept/reject priors, as in live play.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.actions import enumerate_legal_actions_py
from core.data import MAX_ACTION_SIZE, GamePhases
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
)
from core.state import GameState, get_layout
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import CrossPresidentOfferPriorEvaluator, NNEvaluator
from mcts.search import StatePool, run_search
from nn import get_model_input_spec
from train.checkpoint import load_model_from_checkpoint
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
    config: TrainingConfig
    label: str  # short display name
    c_puct: float | None = None  # per-entry override of the checkpoint's c_puct


def _parse_checkpoint_spec(spec: str) -> tuple[Path, float | None]:
    """Split ``PATH`` or ``PATH@c_puct=VALUE`` into path and c_puct override."""
    path, sep, override = spec.strip().partition("@")
    if not sep:
        return Path(path), None
    key, _, value = override.partition("=")
    if key.strip() != "c_puct":
        raise ValueError(f"unsupported checkpoint override {override!r} (expected c_puct=VALUE)")
    try:
        c_puct = float(value)
    except ValueError:
        raise ValueError(f"invalid c_puct value {value!r} in {spec!r}") from None
    if c_puct < 0:
        raise ValueError(f"c_puct must be >= 0, got {c_puct}")
    return Path(path), c_puct


def _entry_mcts_config(
    entry: ModelEntry, num_players: int, overrides: dict[str, Any],
) -> MCTSConfig:
    """Checkpoint search settings with tournament-wide and per-entry overrides."""
    if entry.c_puct is not None:
        overrides = {**overrides, "c_puct": entry.c_puct}
    return dataclasses.replace(entry.config.to_mcts_config(num_players=num_players), **overrides)


def _load_model(cp_path: Path, device: torch.device) -> tuple[torch.nn.Module, TrainingConfig, int]:
    """Load model from checkpoint. Returns (model, config, epoch)."""
    model, config, cp = load_model_from_checkpoint(cp_path, device)
    model.eval()
    epoch = int(cp.get("epoch", -1))  # type: ignore[arg-type]
    return model, config, epoch


def _model_name(config: TrainingConfig) -> str:
    """Short model family name, e.g. ``transformer-v3``."""
    return (config.model_path or "model").rsplit("/", 1)[-1].removesuffix(".py")


# ---------------------------------------------------------------------------
# Engine rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineRules:
    """Engine rule flags (runtime-only; not stored in the raw state array)."""
    v3_behavior: bool
    acq_same_president: bool

    @classmethod
    def from_config(cls, config: TrainingConfig) -> EngineRules:
        return cls(config.v3_behavior, config.acq_same_president)

    def describe(self) -> str:
        mode = "v3" if self.v3_behavior else "legacy"
        acq = ("same-president acquisitions" if self.acq_same_president
               else "cross-president offers")
        return f"{mode} rules, {acq}"


def _resolve_game_rules(
    entries: list[ModelEntry],
    v3_behavior: bool | None = None,
    acq_same_president: bool | None = None,
) -> EngineRules:
    """Game rules: explicit overrides, else the most permissive checkpoint's."""
    trained = [EngineRules.from_config(e.config) for e in entries]
    return EngineRules(
        v3_behavior=(any(r.v3_behavior for r in trained)
                     if v3_behavior is None else v3_behavior),
        acq_same_president=(all(r.acq_same_president for r in trained)
                            if acq_same_president is None else acq_same_president),
    )


def _search_rules(game: EngineRules, trained: EngineRules) -> EngineRules:
    """Rules a checkpoint searches under: its training rules, within the game's.

    Searching same-president only restricts the game's legal actions. Legacy
    behavior drops v3's cross-president price floors, so it is used only with
    same-president search; there it differs from v3 only in trade-history
    lifetime (the rejection floor never applies to a player that makes no
    cross-president offers).
    """
    same_president = game.acq_same_president or trained.acq_same_president
    return EngineRules(
        v3_behavior=game.v3_behavior and (trained.v3_behavior or not same_president),
        acq_same_president=same_president,
    )


# Phases where legacy rules keep trade history; they clear it when INVEST ends.
_INVEST_ROUND_PHASES = (int(GamePhases.PHASE_INVEST), int(GamePhases.PHASE_BID))


def _search_state(
    state: GameState, rules: EngineRules, num_players: int, max_players: int,
) -> GameState:
    """Return ``state`` as seen under ``rules``, copying when they differ."""
    if (rules.v3_behavior == state.v3_behavior
            and rules.acq_same_president == state.acq_same_president):
        return state
    view = GameState.from_array(
        state._array, num_players, max_players=max_players,
        v3_behavior=rules.v3_behavior,
    )
    view.acq_same_president = rules.acq_same_president
    if (state.v3_behavior and not rules.v3_behavior
            and TURN.get_phase(view) not in _INVEST_ROUND_PHASES):
        # V3 keeps trade history until the next INVEST.
        for player_id in range(num_players):
            PLAYERS[player_id].clear_roundtrip_tracking(view)
    return view


@dataclass
class ModelPlayer:
    """How one checkpoint searches in this tournament."""
    evaluator: Any
    mcts_config: MCTSConfig
    rules: EngineRules


# ---------------------------------------------------------------------------
# Game play
# ---------------------------------------------------------------------------

def _play_game(
    players: list[ModelPlayer],
    seat_to_model: list[int],
    num_players: int,
    max_players: int,
    game_rules: EngineRules,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool,
) -> list[int]:
    """Play one tournament game. Returns net worths per seat."""
    state = GameState(num_players, max_players=max_players,
                      v3_behavior=game_rules.v3_behavior,
                      acq_same_president=game_rules.acq_same_president)
    state.initialize_game(num_players, seed=game_seed, max_players=max_players)
    legal = np.zeros(MAX_ACTION_SIZE, dtype=np.uint16)

    while TURN.get_phase(state) != GamePhases.PHASE_GAME_OVER:
        active_player = TURN.get_active_player(state)
        player = players[seat_to_model[active_player]]
        search_state = _search_state(state, player.rules, num_players, max_players)

        if enumerate_legal_actions_py(search_state, legal) == 1:
            # Forced under this seat's rules; its own driver would auto-apply it.
            action = int(legal[0])
        else:
            # Fresh search each move (no subtree reuse — different models)
            root = run_search(search_state, player.evaluator, player.mcts_config,
                              rng, state_pool=state_pool)
            assert root.legal_actions is not None and root.visit_counts is not None
            action = int(root.legal_actions[np.argmax(root.visit_counts)])

        history: list[tuple[int, int]] = []
        status = DRIVER.apply_action(state, action, history=history)
        assert status != STATUS_INVALID, (
            f"seat {active_player} chose action {action}, illegal under game rules"
        )
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

    lines.append(
        f"# Tournament Report ({total_games} games, {elapsed:.1f}s, "
        f"{num_players} players)"
    )
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

def _build_players(
    entries: list[ModelEntry],
    device: torch.device,
    num_players: int,
    max_players: int,
    game_rules: EngineRules,
    mcts_configs: list[MCTSConfig],
    terminal_rank_weights: list[float],
) -> list[ModelPlayer]:
    """Build each checkpoint's evaluator and search rules for this game."""
    players: list[ModelPlayer] = []
    for e, mcts_config, terminal_rank_weight in zip(
        entries, mcts_configs, terminal_rank_weights, strict=True,
    ):
        input_spec = get_model_input_spec(e.config)
        evaluator: Any = NNEvaluator(
            e.model,
            device,
            num_players=input_spec.num_players,
            terminal_rank_weight=terminal_rank_weight,
            input_spec=input_spec,
        )
        trained = EngineRules.from_config(e.config)
        if trained.acq_same_president and not game_rules.acq_same_president:
            evaluator = CrossPresidentOfferPriorEvaluator(
                evaluator, num_players=num_players, max_players=max_players,
            )
        players.append(ModelPlayer(
            evaluator, mcts_config, _search_rules(game_rules, trained),
        ))
    return players


def run_tournament(
    entries: list[ModelEntry],
    device: torch.device,
    num_players: int,
    max_players: int,
    mcts_configs: list[MCTSConfig],
    min_games_per_pair: int,
    base_seed: int,
    terminal_rank_weights: list[float],
    game_rules: EngineRules,
) -> tuple[list[GameResult], float]:
    """Run the full tournament. Returns (results, elapsed_seconds)."""
    players = _build_players(
        entries, device, num_players, max_players, game_rules,
        mcts_configs, terminal_rank_weights,
    )

    layout = get_layout(max_players)
    max_sims = max(c.num_simulations for c in mcts_configs)
    state_pool = StatePool(2 * (max_sims + 1), layout.total_size)
    rng = np.random.default_rng(base_seed)

    schedule = _generate_schedule(len(entries), min_games_per_pair, num_players)
    total_games = sum(g for _, g in schedule)

    print(f"Tournament: {len(entries)} models, {total_games} games scheduled")
    print(f"  Players/game: {num_players}")
    print(f"  Game rules: {game_rules.describe()}")
    for i, (e, player) in enumerate(zip(entries, players, strict=True)):
        cfg = player.mcts_config
        price_cap = cfg.max_acq_price_actions or "all"
        offer_priors = (
            "; equal cross-president offer priors"
            if isinstance(player.evaluator, CrossPresidentOfferPriorEvaluator) else ""
        )
        print(f"  [{i}] {e.label}: searches {player.rules.describe()}{offer_priors}; "
              f"{cfg.num_simulations} sims, batch {cfg.search_batch_size}, "
              f"c_puct {cfg.c_puct}, dirichlet eps {cfg.dirichlet_epsilon}, "
              f"acq prices {price_cap}")
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
                players, seat_to_model, num_players, max_players,
                game_rules, int(game_seed), rng, state_pool,
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


def _resolve_tournament_num_players(
    config: TrainingConfig,
    requested_num_players: int | None,
) -> int:
    """Resolve the actual player count for every tournament game."""
    if requested_num_players is None:
        return config.effective_min_players

    if isinstance(requested_num_players, bool):
        raise ValueError("num_players must be an integer player count")
    num_players = int(requested_num_players)
    if not (
        config.effective_min_players
        <= num_players
        <= config.effective_max_players
    ):
        raise ValueError(
            "tournament num_players must be within the configured player range "
            f"{config.effective_min_players}-{config.effective_max_players}, "
            f"got {num_players}"
        )
    return num_players


def _validate_tournament_num_players(
    entries: list[ModelEntry],
    num_players: int,
) -> None:
    """Ensure every checkpoint can play the selected actual player count."""
    for entry in entries:
        config = entry.config
        if not (
            config.effective_min_players
            <= num_players
            <= config.effective_max_players
        ):
            raise ValueError(
                f"checkpoint {entry.path} supports player range "
                f"{config.effective_min_players}-{config.effective_max_players}, "
                f"but tournament num_players is {num_players}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tournament between model checkpoints"
    )
    parser.add_argument(
        "checkpoints",
        type=str,
        help="Comma-separated list of checkpoint file paths. Append "
             "@c_puct=VALUE to override an entry's c_puct; the same "
             "checkpoint may appear with different values.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--v3-behavior", action=argparse.BooleanOptionalAction, default=None,
        help="Game engine behavior (default: v3 if any checkpoint was trained "
             "with it). Each checkpoint still searches under its own rules.",
    )
    parser.add_argument(
        "--acq-same-president", action=argparse.BooleanOptionalAction, default=None,
        help="Game acquisition scope (default: cross-president offers if any "
             "checkpoint was trained with them)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--search-batch-size", type=int, default=1,
                        help="Batched leaf evaluation size (default: 1)")
    parser.add_argument("--games-per-pair", type=int, default=10,
                        help="Minimum games per model pair (default: 10)")
    parser.add_argument(
        "--num-players", type=int, default=None,
        help="Actual player count for every game (3-5). Defaults to the "
             "configured count, or the configured minimum for mixed-player "
             "checkpoints.",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: stdout)")
    parser.add_argument(
        "--terminal-blend", type=float, default=None,
        help="Rank vs margin weight for terminal rewards "
             "(0=margin, 1=rank, default from each checkpoint)",
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--no-dirichlet-noise", dest="dirichlet_epsilon",
        action="store_const", const=0.0,
        help="Disable Dirichlet noise at root",
    )
    noise_group.add_argument(
        "--dirichlet-epsilon", type=float, default=None,
        help="Dirichlet noise epsilon (default from each checkpoint)",
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

    # Parse checkpoint paths and per-entry overrides
    try:
        specs = [_parse_checkpoint_spec(spec) for spec in args.checkpoints.split(",")]
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    cp_paths = [path for path, _ in specs]
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

    for i, (cp_path, c_puct) in enumerate(specs):
        model, config, epoch = _load_model(cp_path, device)
        if ref_config is None:
            ref_config = config

        label = f"{_model_name(config)} " + (f"epoch {epoch}" if epoch >= 0 else f"model {i}")
        if c_puct is not None:
            label += f" c_puct {c_puct:g}"
        entries.append(ModelEntry(cp_path, epoch, model, config, label, c_puct))
        print(f"  [{i}] {label}: {cp_path.name}")

    assert ref_config is not None
    print()

    try:
        tournament_num_players = _resolve_tournament_num_players(
            ref_config, args.num_players,
        )
        _validate_tournament_num_players(entries, tournament_num_players)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    tournament_max_players = max(
        tournament_num_players,
        *(entry.config.effective_max_players for entry in entries),
    )

    # Each checkpoint's own search settings, with tournament-wide CLI overrides
    mcts_configs: list[MCTSConfig] = []
    terminal_blends: list[float] = []
    for entry in entries:
        overrides: dict[str, Any] = {
            "num_simulations": args.simulations,
            "search_batch_size": args.search_batch_size,
        }
        if args.dirichlet_epsilon is not None:
            overrides["dirichlet_epsilon"] = args.dirichlet_epsilon
        if args.dirichlet_dynamic is not None:
            overrides["dirichlet_dynamic"] = args.dirichlet_dynamic
        mcts_configs.append(
            _entry_mcts_config(entry, tournament_num_players, overrides),
        )
        terminal_blends.append(args.terminal_blend if args.terminal_blend is not None
                               else entry.config.terminal_blend)

    # Run tournament
    results, elapsed = run_tournament(
        entries, device, tournament_num_players, tournament_max_players, mcts_configs,
        args.games_per_pair, args.seed, terminal_blends,
        _resolve_game_rules(entries, args.v3_behavior, args.acq_same_president),
    )

    # Build report
    stats = _collect_pair_stats(results, len(entries), tournament_num_players)
    report = _format_report(
        entries, results, stats, elapsed, tournament_num_players,
    )

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
