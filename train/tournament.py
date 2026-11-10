"""Tournament between model checkpoints.

Play round-robin games between checkpoints of one game (RSS or 1889, through
the ``Game`` adapter) to compare their strength. Each checkpoint searches with
its own model, input layout, search settings, and terminal value settings.

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

Seats count from the player who acts first: P0 (1889's seed picks who that
is; in RSS it is always player 0). A 1889 schedule is seat-balanced: it plays
in rounds in which every model sits in every seat equally often. With fewer
models than seats, a round is every arrangement of the most even splits (2
models at 4 seats: the six 2-2 arrangements; at 3 seats: the 2-1 and 1-2
ones); otherwise it is every rotation, in both directions, of every group of
``num_players`` models. RSS keeps its original schedule, so its tournaments
replay as before. Moves are the most-visited action after a search with each
checkpoint's root noise; ``--sample-moves N`` samples a game's first N
decisions from the visits instead, so games of near-deterministic models
differ from their first moves (self-play's ``start_explore_moves`` does the
same for started games). Games run one at a time in this process, or with
``--workers K`` in K processes that each load every checkpoint; a seed then
gives the same games for any K > 1, though not the ones a single process plays.

The report adds a pairwise score to the per-place counts: the share of seat
pairs in which one model's seat finished ahead of the other's, in the games
both played (ties count half), with its standard error over those games.

Checkpoints of RSS trained under different engine rules can share a game
(``games/rss_tournament.py``). Every 1889 checkpoint must have been trained
under the same engine options, which the games are played under.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import math
import multiprocessing as mp
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from games import STATUS_INVALID, Game, get_game
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, run_search
from nn import get_model_input_spec
from train.checkpoint import load_model_from_checkpoint
from train.config import _ENGINE_OPTIONS, MCTSConfig, TrainingConfig


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

def _same_state(state: Any) -> Any:
    return state


class GameRules(Protocol):
    """The engine rules a tournament's games use, and how each checkpoint
    searches within them (``_resolve_game_rules``)."""

    def describe(self) -> str: ...

    def game_config(self, config: TrainingConfig) -> TrainingConfig:
        """``config`` with these rules: its adapter's ``new_game`` plays under them."""
        ...

    def seat(
        self, config: TrainingConfig, evaluator: Any, num_players: int, max_players: int,
    ) -> tuple[Any, Callable[[Any], Any], str]:
        """For a checkpoint trained under ``config``: its evaluator, the game's
        state as it searches it, and how that differs from the game's rules."""
        ...


@dataclass(frozen=True)
class SharedRules:
    """Engine options every checkpoint was trained with: the games use them,
    and each checkpoint searches the game's own state."""
    options: tuple[tuple[str, Any], ...]

    def describe(self) -> str:
        return ", ".join(f"{name}={value}" for name, value in self.options)

    def game_config(self, config: TrainingConfig) -> TrainingConfig:
        return config

    def seat(
        self, config: TrainingConfig, evaluator: Any, num_players: int, max_players: int,
    ) -> tuple[Any, Callable[[Any], Any], str]:
        del config, num_players, max_players
        return evaluator, _same_state, ""


def _resolve_game_rules(
    entries: list[ModelEntry],
    v3_behavior: bool | None = None,
    acq_same_president: bool | None = None,
) -> GameRules:
    """The rules for one game's checkpoints. RSS checkpoints may differ in
    rules (``games.rss_tournament``), and the overrides are RSS's; every other
    game's checkpoints must share their engine options (``SharedRules``)."""
    games = {e.config.game for e in entries}
    if len(games) > 1:
        played = ", ".join(f"{e.path.name} {e.config.game}" for e in entries)
        raise ValueError(f"checkpoints play different games: {played}")
    game = games.pop()
    if game == "rss":
        from games.rss_tournament import resolve_game_rules

        return resolve_game_rules(
            [e.config for e in entries], v3_behavior, acq_same_president,
        )
    if v3_behavior is not None or acq_same_president is not None:
        raise ValueError("--v3-behavior and --acq-same-president are RSS options")
    names = _ENGINE_OPTIONS[game]
    differences = [
        f"{name}: " + ", ".join(f"{e.path.name} {getattr(e.config, name)}" for e in entries)
        for name in names
        if len({getattr(e.config, name) for e in entries}) > 1
    ]
    if differences:
        raise ValueError(
            f"checkpoints were trained under different {game} engine options, "
            f"which a tournament does not mix: {'; '.join(differences)}"
        )
    return SharedRules(tuple((name, getattr(entries[0].config, name)) for name in names))


@dataclass
class ModelPlayer:
    """How one checkpoint searches in this tournament."""
    evaluator: Any
    mcts_config: MCTSConfig
    # The game's state as this checkpoint searches it
    view: Callable[[Any], Any] = _same_state
    rules: str = ""  # how its search differs from the game's rules, for the header


# ---------------------------------------------------------------------------
# Game play
# ---------------------------------------------------------------------------

def _play_game(
    game: Game,
    players: list[ModelPlayer],
    seat_to_model: list[int],
    num_players: int,
    max_players: int,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool,
    sample_moves: int = 0,
) -> list[int]:
    """Play one tournament game. Returns final scores per seat; seat 0 acts first."""
    state = game.new_game(num_players, max_players, game_seed)
    first = game.active_player(state)
    legal = np.empty(game.policy.max_action_size, dtype=np.uint16)

    for move in itertools.count():
        if game.is_terminal(state):
            break
        seat = (game.active_player(state) - first) % num_players
        player = players[seat_to_model[seat]]
        search_state = player.view(state)

        if game.enumerate_policy_actions(search_state, legal) == 1:
            # Forced under this seat's rules; its own driver would auto-apply it.
            action = int(legal[0])
        else:
            # Fresh search each move (no subtree reuse — different models)
            root = run_search(game, search_state, player.evaluator, player.mcts_config,
                              rng, state_pool=state_pool)
            assert root.legal_actions is not None and root.visit_counts is not None
            visits = root.visit_counts
            choice = (rng.choice(len(visits), p=visits / visits.sum())
                      if move < sample_moves else np.argmax(visits))
            action = int(root.legal_actions[choice])

        status = game.apply_action(state, action)
        assert status != STATUS_INVALID, (
            f"seat {seat} chose action {action}, illegal under game rules"
        )

    scores = game.final_scores(state)
    return [scores[(seat + first) % num_players] for seat in range(num_players)]


# ---------------------------------------------------------------------------
# Matchup scheduling
# ---------------------------------------------------------------------------

def _balanced_schedule(
    num_models: int, min_games_per_pair: int, num_players: int,
) -> list[tuple[int, ...]]:
    """Seat assignments, one per game, in rounds in which every model sits in
    every seat equally often (see the module docstring). Rounds repeat until
    every pair has played ``min_games_per_pair`` games together."""
    if num_models < num_players:
        base, extra = divmod(num_players, num_models)
        round_ = sorted({
            seating
            for heavy in itertools.combinations(range(num_models), extra)
            for seating in itertools.permutations([
                m for m in range(num_models) for _ in range(base + (m in heavy))
            ])
        })
    else:
        round_ = [
            group[k:] + group[:k]
            for combo in itertools.combinations(range(num_models), num_players)
            for group in (combo, combo[::-1])
            for k in range(num_players)
        ]
    # By symmetry every pair plays together as often as models 0 and 1.
    together = sum({0, 1} <= set(seating) for seating in round_)
    return round_ * max(1, math.ceil(min_games_per_pair / together))


def _rss_schedule(
    num_models: int, min_games_per_pair: int, num_players: int,
) -> Iterator[tuple[int, ...]]:
    """RSS's original seat assignments, one per game."""
    for triple, num_games in _generate_schedule(num_models, min_games_per_pair, num_players):
        # All seat permutations for this triple, cycled over num_games
        perms = list(itertools.permutations(triple))
        for g in range(num_games):
            yield perms[g % len(perms)]


def _generate_schedule(
    num_models: int, min_games_per_pair: int, num_players: int,
) -> list[tuple[tuple[int, ...], int]]:
    """Generate RSS's tournament schedule as (model_group, num_games) pairs.

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
    # Per such game: the share of (a seat, b seat) pairs where a's seat
    # finished ahead, ties counting half
    pair_scores: list[float] = field(default_factory=list)

    def pair_score(self) -> tuple[float, float]:
        """Mean pairwise score and its standard error over games."""
        scores = np.asarray(self.pair_scores, dtype=np.float64)
        se = scores.std(ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else math.nan
        return float(scores.mean()), float(se)


@dataclass
class GameResult:
    """Record of one tournament game, by seat (seat 0 acts first)."""
    seat_to_model: list[int]
    scores: list[int]
    ranks: list[int]
    seed: int


def _ahead(rank_a: int, rank_b: int) -> float:
    return 1.0 if rank_a < rank_b else 0.5 if rank_a == rank_b else 0.0


def _seat_scores(ranks: list[int]) -> list[float]:
    """Each seat's share of the other seats it finished ahead of, ties half."""
    # A seat ties itself, which counts half.
    return [(sum(_ahead(r, other) for other in ranks) - 0.5) / (len(ranks) - 1) for r in ranks]


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
        seats: dict[int, list[int]] = {}
        for seat, model_idx in enumerate(gr.seat_to_model):
            seats.setdefault(model_idx, []).append(seat)
        for model_a, model_b in itertools.permutations(seats, 2):
            stats[(model_a, model_b)].pair_scores.append(float(np.mean([
                _ahead(gr.ranks[i], gr.ranks[j])
                for i in seats[model_a] for j in seats[model_b]
            ])))

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

    lines.append("## Pairwise scores")
    lines.append("")
    lines.append("  The share of seat pairs in which the first model's seat finished ahead of")
    lines.append("  the second's, ties counting half (0.5 = even), mean ± standard error over")
    lines.append("  the games both played.")
    lines.append("")
    for i, j in itertools.combinations(range(len(entries)), 2):
        ps = stats[(i, j)]
        if ps.games == 0:
            continue
        mean, se = ps.pair_score()
        lines.append(
            f"  [{i}] vs [{j}]: {mean:.3f} ± {se:.3f}  ({ps.games} games)"
        )
    lines.append("")

    seat_hdr = "  ".join(f"{f'P{s}':>6s}" for s in range(num_players))
    lines.append("## Seats (P0 acts first)")
    lines.append("")
    lines.append(f"  {'Games per seat':<30s}  {seat_hdr}")
    for i, entry in enumerate(entries):
        counts = [sum(r.seat_to_model[s] == i for r in results) for s in range(num_players)]
        lines.append(f"  {f'[{i}] {entry.label}':<30s}  "
                     + "  ".join(f"{c:>6d}" for c in counts))
    if results:
        seat_scores = np.mean([_seat_scores(r.ranks) for r in results], axis=0)
        lines.append(f"  {'Seat score (0.5 = even)':<30s}  "
                     + "  ".join(f"{s:>6.3f}" for s in seat_scores))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_players(
    game: Game,
    entries: list[ModelEntry],
    device: torch.device,
    num_players: int,
    max_players: int,
    game_rules: GameRules,
    mcts_configs: list[MCTSConfig],
    terminal_rank_weights: list[float],
) -> list[ModelPlayer]:
    """Build each checkpoint's evaluator, on its own model's input spec, and
    its search rules within the game's."""
    players: list[ModelPlayer] = []
    for e, mcts_config, terminal_rank_weight in zip(
        entries, mcts_configs, terminal_rank_weights, strict=True,
    ):
        input_spec = get_model_input_spec(e.config)
        evaluator = NNEvaluator(
            game,
            e.model,
            device,
            num_players=input_spec.num_players,
            terminal_rank_weight=terminal_rank_weight,
            terminal_rank_tau=e.config.terminal_rank_tau,
            input_spec=input_spec,
        )
        seat_evaluator, view, rules = game_rules.seat(
            e.config, evaluator, num_players, max_players,
        )
        players.append(ModelPlayer(seat_evaluator, mcts_config, view, rules))
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
    game_rules: GameRules,
    sample_moves: int = 0,
    workers: int = 1,
) -> tuple[list[GameResult], float]:
    """Run the full tournament under ``game_rules`` (``_resolve_game_rules``).
    With ``workers`` > 1, games run in that many processes, each loading every
    checkpoint. Returns (results in schedule order, elapsed_seconds)."""
    # Each seat's rules travel on its search state.
    game = get_game(game_rules.game_config(entries[0].config))
    players = _build_players(
        game, entries, device, num_players, max_players, game_rules,
        mcts_configs, terminal_rank_weights,
    )

    max_sims = max(c.num_simulations for c in mcts_configs)
    state_pool = StatePool(game, 2 * (max_sims + 1), game.state_size(max_players))
    rng = np.random.default_rng(base_seed)

    # RSS keeps its original schedule, so its tournaments replay as before.
    schedule = list((_rss_schedule if game.name == "rss" else _balanced_schedule)(
        len(entries), min_games_per_pair, num_players,
    ))
    total_games = len(schedule)

    print(f"Tournament: {len(entries)} models, {total_games} games scheduled")
    print(f"  Players/game: {num_players}")
    print(f"  Game rules: {game.name}, {game_rules.describe()}")
    if sample_moves:
        print(f"  Moves sampled from search visits: each game's first {sample_moves}")
    if workers > 1:
        print(f"  Worker processes: {workers}")
    for i, (e, player) in enumerate(zip(entries, players, strict=True)):
        cfg = player.mcts_config
        print(f"  [{i}] {e.label}: "
              f"{cfg.num_simulations} sims, batch {cfg.search_batch_size}, "
              f"c_puct {cfg.c_puct}, dirichlet eps {cfg.dirichlet_epsilon}, "
              f"input layout {player.evaluator.layout_version}"
              + (f"; {player.rules}" if player.rules else ""))
    print()

    def play_here() -> Iterator[tuple[int, int, list[int], float]]:
        for game_num, seating in enumerate(schedule, 1):
            game_seed = int(rng.integers(0, 2**31))
            t_game = time.perf_counter()
            scores = _play_game(
                game, players, list(seating), num_players, max_players,
                game_seed, rng, state_pool, sample_moves,
            )
            yield game_num, game_seed, scores, time.perf_counter() - t_game

    outcomes = play_here()
    if workers > 1:
        # Each game gets its own search rng, so results follow the seed
        # whatever the number of workers, but differ from one process's.
        tasks = [
            (game_num, seating, int(rng.integers(0, 2**31)), base_seed)
            for game_num, seating in enumerate(schedule, 1)
        ]
        outcomes = _play_in_workers(workers, tasks, (
            [(e.path, e.c_puct) for e in entries], str(device), num_players, max_players,
            mcts_configs, terminal_rank_weights, game_rules, sample_moves,
        ))

    results: list[tuple[int, GameResult]] = []
    t0 = time.perf_counter()

    for game_num, game_seed, scores, dt in outcomes:
        seat_to_model = list(schedule[game_num - 1])
        ranks = _rank_players(scores)
        results.append((game_num, GameResult(seat_to_model, scores, ranks, game_seed)))

        # Progress line
        seat_desc = ", ".join(
            f"P{s}=[{m}]" for s, m in enumerate(seat_to_model)
        )
        rank_desc = ", ".join(
            f"[{seat_to_model[s]}]=${scores[s]}(#{ranks[s]})"
            for s in range(num_players)
        )
        print(f"  Game {game_num}/{total_games} ({dt:.1f}s): "
              f"{seat_desc} → {rank_desc}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nAll {total_games} games completed in {elapsed:.1f}s")
    return [result for _, result in sorted(results, key=lambda r: r[0])], elapsed


def _play_in_workers(
    workers: int, tasks: list[tuple[int, tuple[int, ...], int, int]], setup: tuple[Any, ...],
) -> Iterator[tuple[int, int, list[int], float]]:
    """Play ``tasks`` in worker processes, yielding results as games finish."""
    with mp.get_context("spawn").Pool(workers, _init_worker, setup) as pool:
        yield from pool.imap_unordered(_play_task, tasks)


# A worker process's players, from _init_worker
_worker: dict[str, Any] = {}


def _init_worker(
    specs: list[tuple[Path, float | None]],
    device: str,
    num_players: int,
    max_players: int,
    mcts_configs: list[MCTSConfig],
    terminal_rank_weights: list[float],
    game_rules: GameRules,
    sample_moves: int,
) -> None:
    """Load every checkpoint and build its player, as ``run_tournament`` does."""
    torch.set_num_threads(1)  # parallelism comes from the workers
    entries = []
    for path, c_puct in specs:
        model, config, epoch = _load_model(path, torch.device(device))
        entries.append(ModelEntry(path, epoch, model, config, path.name, c_puct))
    game = get_game(game_rules.game_config(entries[0].config))
    players = _build_players(
        game, entries, torch.device(device), num_players, max_players, game_rules,
        mcts_configs, terminal_rank_weights,
    )
    max_sims = max(c.num_simulations for c in mcts_configs)
    _worker.update(
        game=game, players=players, num_players=num_players, max_players=max_players,
        state_pool=StatePool(game, 2 * (max_sims + 1), game.state_size(max_players)),
        sample_moves=sample_moves,
    )


def _play_task(
    task: tuple[int, tuple[int, ...], int, int],
) -> tuple[int, int, list[int], float]:
    game_num, seating, game_seed, base_seed = task
    t_game = time.perf_counter()
    w = _worker
    scores = _play_game(
        w["game"], w["players"], list(seating), w["num_players"], w["max_players"],
        game_seed, np.random.default_rng([base_seed, game_num]), w["state_pool"],
        w["sample_moves"],
    )
    return game_num, game_seed, scores, time.perf_counter() - t_game


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
        help="RSS game engine behavior (default: v3 if any checkpoint was "
             "trained with it). Each checkpoint still searches under its own rules.",
    )
    parser.add_argument(
        "--acq-same-president", action=argparse.BooleanOptionalAction, default=None,
        help="RSS game acquisition scope (default: cross-president offers if "
             "any checkpoint was trained with them)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Play games in this many processes, each loading every checkpoint "
             "(default: 1, in this process). Results follow the seed for any "
             "count above 1, but differ from a single process's.",
    )
    parser.add_argument(
        "--sample-moves", type=int, default=0,
        help="Sample each game's first N decisions in proportion to search "
             "visits instead of playing the most visited (default: 0)",
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
        game_rules = _resolve_game_rules(entries, args.v3_behavior, args.acq_same_president)
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
        args.games_per_pair, args.seed, terminal_blends, game_rules, args.sample_moves,
        args.workers,
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
