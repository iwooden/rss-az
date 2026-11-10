"""Start positions for 1889 self-play from the scraped human games (1889/v1-design.md §2.1).

    python -m utils_18xx.g1889_start_positions --out g1889/data/human_starts.npz \\
        [--games DIR] [--config CFG] [--min-left 50] [--workers N] [--seed 0]

Each game in ``--games`` at the config's player counts replays through the
differential-replay harness (``tests/g1889/replay/harness.py``): its Ruby extract
(``load_scraped_extract``, from ``tests/g1889/replay/.extract_cache/``, which
needs ``submodules/18xx``, and Ruby only to remake a stale extract) drives the
engine's own driver, compared with Ruby's snapshot at every record. The rows the
harness hands ``DRIVER.apply_action`` are where the engine rested through the
game, in order: the humans' decisions, and the harness's passes for the stops
Ruby has no step for (BUY_PRIVATE_PRE, SR turns Ruby skips).

**Decisions** are counted as ``play_game`` counts ``move_count``: one for each
of those rests that is also one under the config's engine flags. The driver
applies automated steps and forced decisions by itself, so a decision the flags
leave one choice at is not one, nor one they leave no policy action at, nor a
human's move outside the action space (a route, private, out-of-turn, or
topmost-bid override). A position's ``decision_index`` is the decisions before
it in its game, the ``move_count`` a started game begins at, and
``decisions_left`` those from it to the game's end, itself included.

**A position is kept** if its game has 3-5 players, its bank is not broken,
more than ``--min-left`` decisions are left, and a uniformly random policy
playout from it, under the flags, reaches the game's end without an error. Rows
are normalised first: at DIVIDEND the revenue is the engine's best, as
``utils_18xx/g1889_deviation.py`` sets it for human positions, and the
extraction prologue refreshes every graph and route cache in the row.

The ``.npz`` holds the rows at the config's player capacity, what the builder
knows of each, and the engine flags; ``games/g1889_starts.py`` has its format
and its loader, which self-play reads it through.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import zlib
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER, STATUS_INVALID
from g1889.entities.bank import BANK
from g1889.entities.phase import PHASE
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
from g1889.state import GameState, get_global_fields, get_layout
from games import RELATION_RECORD_WIDTH, Game, get_game
from games.g1889_starts import StartPositions, engine_options, save_start_positions, state_layout_hash
from train.config import TrainingConfig

HUMAN_GAMES = Path("tests/g1889/games/data")
CONFIG = Path("train_configs/1889-v0.json")
MIN_LEFT = 50
_NEWEST_LAYOUT = 2  # the token layout whose fill leaves every cache in the row fresh
MAX_PLAYOUT = 20_000  # decisions; uniform playouts from the opening take about 800
_NUM_PLAYERS = get_layout(0).global_offset + get_global_fields().num_players


# =============================================================================
# BUILDING
# =============================================================================


class _RestRecorder:
    """``DRIVER`` for the replay harness, keeping the row of each decision it is
    asked to apply. A harness that backs up to a recorded row (MF's port tile on an
    SR turn it passed) undoes the rows after it."""

    def __init__(self) -> None:
        self.rows: list[np.ndarray] = []
        self._at: dict[bytes, int] = {}

    def apply_action(self, state: GameState, action_id: int, history: list | None = None) -> int:
        key = state._array.tobytes()
        if (k := self._at.get(key)) is not None:
            for row in self.rows[k:]:
                del self._at[row.tobytes()]
            del self.rows[k:]
        self._at[key] = len(self.rows)
        self.rows.append(state._array.copy())
        return DRIVER.apply_action(state, action_id, history)

    def __getattr__(self, name: str) -> Any:
        return getattr(DRIVER, name)


def human_rests(path: Path) -> list[np.ndarray]:
    """The rows of the human game's decisions in ``path``, in order, at its player
    count, from a replay that reaches Ruby's natural end."""
    # The harness and extracts live with the replay tests, which loading a file needs no part of.
    from tests.g1889.replay import harness
    from tests.g1889.replay.extracts import forget_extracts, load_scraped_extract

    recorder = _RestRecorder()
    try:
        with patch.object(harness, "DRIVER", recorder):
            progress = harness.replay_extract(path.stem, load_scraped_extract(path))
    finally:
        forget_extracts()  # each is used once
    if progress.stop or progress.state is None or TURN.round(progress.state) != K.ROUND_GAME_OVER:
        raise RuntimeError(f"replay did not reach the game's end: {progress}")
    return recorder.rows


def _normalize(game: Game, state: GameState, records: np.ndarray) -> str | None:
    """Make ``state``'s row what self-play would hold there; why it is not a
    decision the driver rests at under its flags, or None."""
    actions = np.empty(game.policy.max_action_size, dtype=np.uint16)
    if game.enumerate_policy_actions(state, actions) == 0:
        return "no policy action under the flags"
    if game.decision_phase(state) == K.DPHASE_DIVIDEND:  # humans may have run less
        TURN.set_route_revenue(state, max_route_revenue(state, TURN.operating_corp(state)))
    if DRIVER.forced_action_py(state) >= 0:
        return "forced under the flags"
    # Extraction's prologue refreshes the graph and route caches; a layout-2
    # token fill also refreshes the hypothetical route maxes it alone reads.
    game.fill_relations_sparse(state, records, state.max_players)
    tokens = np.empty((game.num_tokens(state.max_players), game.token_dim(_NEWEST_LAYOUT)), dtype=np.float32)
    game.fill_tokens(state, tokens, state.max_players, layout_version=_NEWEST_LAYOUT)
    return None


def playout(game: Game, root: GameState, row: np.ndarray, rng: np.random.Generator,
            max_decisions: int = MAX_PLAYOUT) -> tuple[str, str] | None:
    """Uniformly random policy decisions from a copy of ``row``, with ``root``'s
    flags, to the game's end: None, or why not and at which decision phase."""
    state = game.scratch_state(root, row.copy())
    actions = np.empty(game.policy.max_action_size, dtype=np.uint16)
    for _ in range(max_decisions):
        phase = game.decision_phase(state)
        try:
            count = game.enumerate_policy_actions(state, actions)
            if count == 0:
                return "no policy action", K.DecisionPhase(phase).name
            status = game.apply_action(state, int(actions[rng.integers(count)]))
        except Exception as e:  # noqa: BLE001 - every engine error drops the position
            return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:200]}", K.DecisionPhase(phase).name
        if status == STATUS_GAME_OVER:
            return None
        if status == STATUS_INVALID:
            return "a policy action is illegal", K.DecisionPhase(phase).name
    return f"no end within {max_decisions} decisions", K.DecisionPhase(game.decision_phase(state)).name


def playout_rng(seed: int, game_id: str, decision_index: int) -> np.random.Generator:
    """A position's playout generator, whatever other games the build reads."""
    return np.random.default_rng((seed, zlib.crc32(game_id.encode()), decision_index))


@dataclass
class GameResult:
    rows: np.ndarray
    meta: np.ndarray            # (n, 5): decision index, decisions left, game phase, decision phase, round
    candidates: int             # the human game's rests
    decisions: int              # those self-play rests at too: the game's length
    drops: Counter
    failures: list[tuple[str, str, str, int]]  # (reason, phase, game id, decision index)


_WORKER: tuple[Game, GameState, int] | None = None


def _init_worker(config: TrainingConfig) -> None:
    global _WORKER
    game = get_game(config)
    max_players = config.effective_max_players
    _WORKER = (game, game.new_game(max_players, max_players, 0), max_players)


def game_positions(path: Path, min_left: int, seed: int) -> GameResult:
    """The kept positions of one game, with the run's flags (``_init_worker``)."""
    assert _WORKER is not None
    game, root, max_players = _WORKER
    rests = human_rests(path)
    width = game.state_size(max_players)
    records = np.empty((game.max_relation_edges, RELATION_RECORD_WIDTH), dtype=np.uint8)
    drops: Counter = Counter()
    rested: list[GameState] = []
    for raw in rests:
        row = np.zeros(width, dtype=np.int16)
        row[:len(raw)] = raw  # an empty player slot is all zeros
        state = game.scratch_state(root, row)
        if reason := _normalize(game, state, records):
            drops[reason] += 1
        else:
            rested.append(state)
    total = len(rested)
    kept, meta, failures = [], [], []
    for index, state in enumerate(rested):
        if BANK.broken(state):
            drops["bank broken"] += 1
            continue
        if total - index <= min_left:
            drops[f"{min_left} or fewer decisions left"] += 1
            continue
        if failed := playout(game, root, state._array, playout_rng(seed, path.stem, index)):
            drops["playout failed"] += 1
            failures.append((*failed, path.stem, index))
            continue
        kept.append(state._array)
        meta.append((index, total - index, PHASE.current(state), game.decision_phase(state), TURN.round(state)))
    return GameResult(np.array(kept, dtype=np.int16).reshape(-1, width), np.array(meta, dtype=np.int32).reshape(-1, 5),
                      len(rests), total, drops, failures)


def _run(task: tuple[Path, int, int]) -> GameResult:
    return game_positions(*task)


def _results(tasks: list[tuple[Path, int, int]], config: TrainingConfig, workers: int) -> Iterator[GameResult]:
    if workers == 1:
        _init_worker(config)
        yield from map(_run, tasks)
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             initargs=(config,)) as pool:
        yield from pool.map(_run, tasks)


def build(paths: list[Path], config: TrainingConfig, min_left: int = MIN_LEFT, seed: int = 0,
          workers: int = 1) -> tuple[StartPositions, dict[str, Any]]:
    """The positions of the games in ``paths`` at the config's player counts, and a report."""
    counts = set(config.iter_player_counts())
    used = [p for p in sorted(paths) if len(json.loads(p.read_text())["players"]) in counts]
    if not used:
        raise ValueError(f"none of the {len(paths)} games has {sorted(counts)} players")
    tasks = [(p, min_left, seed) for p in used]
    results = list(_results(tasks, config, workers))
    rows = np.concatenate([r.rows for r in results])
    meta = np.concatenate([r.meta for r in results])
    source = np.repeat(np.arange(len(results), dtype=np.int16), [len(r.rows) for r in results])
    positions = StartPositions(
        rows=rows, num_players=rows[:, _NUM_PLAYERS].astype(np.int8), source_game=source,
        decision_index=meta[:, 0], decisions_left=meta[:, 1], game_phase=meta[:, 2].astype(np.int8), decision_phase=meta[:, 3].astype(np.int8),
        round_type=meta[:, 4].astype(np.int8), source_games=np.array([p.stem for p in used]),
        phase_names=np.array(D.PHASE_NAMES), max_players=config.effective_max_players,
        layout_hash=state_layout_hash(), engine_options=json.dumps(engine_options(config), sort_keys=True),
    )
    report = {
        "games": len(used), "games_skipped": len(paths) - len(used),
        "candidates": sum(r.candidates for r in results), "decisions": sum(r.decisions for r in results),
        "game_lengths": [r.decisions for r in results],
        "drops": sum((r.drops for r in results), Counter()),
        "failures": [f for r in results for f in r.failures],
    }
    return positions, report


# =============================================================================
# THE COMMAND LINE
# =============================================================================


def _quantiles(values: np.ndarray) -> str:
    if not len(values):
        return "none"
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100]).round().astype(int)
    return f"min {q[0]}, p10 {q[1]}, p25 {q[2]}, median {q[3]}, p75 {q[4]}, p90 {q[5]}, max {q[6]}"


def format_report(positions: StartPositions, report: dict[str, Any], out: Path) -> str:
    n = len(positions.rows)
    by_count = Counter(positions.num_players.tolist())
    by_phase = Counter(positions.phase_names[positions.game_phase].tolist())
    lines = [
        f"{report['games']} games used ({report['games_skipped']} at other player counts); "
        f"{report['candidates']} candidate positions (the human games' decisions), "
        f"{report['decisions']} of them decisions under the flags; {n} kept",
        f"game lengths in decisions: {_quantiles(np.array(report['game_lengths']))}",
        "kept by player count: " + ", ".join(f"{k}p {v}" for k, v in sorted(by_count.items())),
        "kept by game phase: " + ", ".join(f"{name} {by_phase[name]}" for name in D.PHASE_NAMES),
        f"decision index: {_quantiles(positions.decision_index)}",
        f"decisions left: {_quantiles(positions.decisions_left)}",
        "dropped: " + ", ".join(f"{reason} {count}" for reason, count in report["drops"].most_common()),
    ]
    failures = report["failures"]
    if failures:
        lines.append(f"playout failures by reason and the decision phase it arose at ({len(failures)}):")
        groups: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for reason, phase, game_id, index in failures:
            groups.setdefault((reason, phase), []).append((game_id, index))
        for (reason, phase), where in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"  {len(where):5d}  {phase}: {reason}  (first: game {where[0][0]}, decision {where[0][1]})")
    raw = sum(v.nbytes for v in (getattr(positions, f.name) for f in fields(positions)) if isinstance(v, np.ndarray))
    lines.append(f"{out}: {n} rows of {positions.rows.shape[1]} slots, {raw / 1e6:.1f} MB raw, "
                 f"{out.stat().st_size / 1e6:.2f} MB compressed; layout hash {positions.layout_hash}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the .npz to write")
    parser.add_argument("--games", type=Path, default=HUMAN_GAMES, help=f"scraped games (default {HUMAN_GAMES})")
    parser.add_argument("--config", type=Path, default=CONFIG,
                        help=f"the training config whose engine flags and player counts apply (default {CONFIG})")
    parser.add_argument("--min-left", type=int, default=MIN_LEFT,
                        help=f"keep positions with more decisions left than this (default {MIN_LEFT})")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0, help="the playouts' seed")
    args = parser.parse_args(argv)
    if args.out.suffix != ".npz":
        parser.error("--out must end in .npz")
    config = TrainingConfig.from_json(args.config.read_text())
    if config.game != "1889":
        parser.error(f"{args.config} is a {config.game} config")
    start = time.perf_counter()
    positions, report = build(sorted(args.games.glob("*.json")), config, args.min_left, args.seed, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_start_positions(args.out, positions)
    print(format_report(positions, report, args.out))
    print(f"{time.perf_counter() - start:.0f} s with {args.workers} workers; seed {args.seed}")


if __name__ == "__main__":
    main()
