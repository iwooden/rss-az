"""The 1889 deviation test (1889/model-design.md §7.3): does financing the next train pay?

Self-play stalls at about half the 5-trains and almost no 6s or Ds, and its corps almost
never withhold. This finds DIVIDEND decisions in phases 4 and 5 where withholding this
OR's revenue would pay for the next depot train, a 5 or 6, and paying out would not, and
plays two branches from each, every seat the checkpoint at its self-play settings:

- ``model``: the model's own choice there, then the model.
- ``finance``: withhold, buy that train from the depot, then the model.

Positions where the bank has broken, or where withholding would break it, are left out:
the game then ends with the OR set.

The president's terminal value (``compute_terminal_values``, what the value head learns),
paired by position, says whether financing pays against this population. Each position's
``readout`` says what the model expects of the same branches: its search at the training
simulations and at ``--deep`` times them, without root noise, and its raw value head.
Against the played-out difference, that tells a value blind spot (financing pays, the
model expects it not to) from an equilibrium (it does not pay, and the model knows). The
other seats' results and the ORs left show whom a purchase helps.

Stages, each from the repo root:

    python -m train.deviation_1889 positions --buffer DIR --out positions.json [--per-cell 5]
    python -m train.deviation_1889 play --checkpoint CKPT --positions positions.json \\
        --results results.jsonl [--continuations 2] [--deep 4] [--workers 1] [--device D]
    python -m train.deviation_1889 report --positions positions.json --results results.jsonl

``positions`` reads a saved replay buffer (``metadata.json``, ``states.npy``, and
``player_counts.npy``; copy them off a slow mount first) in write order, splits it into
games at their first rows (``Game.game_start_rows``), keeps at most one position a game,
and draws ``--per-cell`` games for each player count and phase. ``play`` runs the
readouts and continuations, a JSON line each, and skips those already in ``--results``,
so rerunning it resumes. A continuation is ``train.self_play.play_game`` from the
position, its start state patched in as ``utils_18xx/g1889_selfplay_log.py`` patches its
collector: the run's config at the checkpoint's epoch (simulations, c_puct, root noise,
forced playouts), at the late-game temperature, ``temp_final``, from the first move. Both
branches of a continuation share its seed. ``play`` evaluates in fp32, uncompiled, so it
runs on the local ROCm GPU.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.driver import STATUS_INVALID
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.phase import PHASE
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout
from games import Game, get_game
from games.g1889_selfplay import G1889SelfPlayStats, _trains_bought
from mcts.evaluator import NNEvaluator, compute_terminal_values
from mcts.search import StatePool, run_search
from nn import get_model_input_spec
from train import self_play
from train.checkpoint import load_model_from_checkpoint
from train.config import EpochConfig, TrainingConfig
from train.replay_buffer import _write_order

PHASES = (D.PHASE_NAMES.index("4"), D.PHASE_NAMES.index("5"))
TARGETS = (D.TRAIN_5, D.TRAIN_6)
BRANCHES = ("model", "finance")
FIELDS = get_global_fields()
GLOBALS = get_layout(D.MIN_PLAYERS).global_offset  # the same for every player count
ROUND, GAME_PHASE, TURN_NUM, OR_NUM = (GLOBALS + f for f in (FIELDS.round, FIELDS.game_phase, FIELDS.turn, FIELDS.or_num))
# The global block, less the cache flags a search's extraction clears: where a game is.
WHERE = np.setdiff1d(np.arange(GLOBALS, GLOBALS + FIELDS.size), [GLOBALS + FIELDS.conn_dirty, GLOBALS + FIELDS.route_dirty])


# =============================================================================
# POSITIONS
# =============================================================================


@dataclass
class Position:
    id: int
    num_players: int
    phase: str  # the game phase's name
    corp: int
    president: int
    cash: int
    revenue: int
    train: int  # the train type withholding pays for
    price: int
    decisions_left: int  # in its buffer game, this decision included
    buffer_row: int  # in write order
    row: list[int]


def finance_target(state: GameState) -> tuple[int, int, int] | None:
    """``(corp, train type, price)`` when ``state`` is a DIVIDEND decision in phase 4 or 5,
    before the bank breaks, whose corp has room for a train, the cheapest train on sale is a
    5 or 6 from the depot, and withholding the revenue would pay for it where paying out
    would not. A payout pays the corp too, for its shares in the Open Market
    (``dividend.pyx``'s ``_pay``). Once the bank breaks the game ends with the OR set, so
    nothing bought then has time to pay; a withhold the bank cannot cover breaks it."""
    if (get_decision_phase_py(state) != K.DPHASE_DIVIDEND or PHASE.current(state) not in PHASES
            or BANK.broken(state)):
        return None
    corp = TURN.operating_corp(state)
    train = TRAINS.cheapest_type(state)
    if (train not in TARGETS or TRAINS.cheapest_in_pool(state)
            or TRAINS.corp_total(state, corp) >= TRAINS.limit(state)):
        return None
    cash, price, revenue = CORPS[corp].cash(state), TRAINS.cheapest_price(state), TURN.route_revenue(state)
    if BANK.cash(state) < revenue:
        return None
    paid_out = cash + (CORPS[corp].pool_shares(state) * revenue + K.SHARES_PER_CORP - 1) // K.SHARES_PER_CORP
    return (corp, train, price) if paid_out < price <= cash + revenue else None


def load_buffer(directory: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """A saved buffer's states and player counts, oldest first, and its max players."""
    meta = json.loads((directory / "metadata.json").read_text())
    if meta["game"] != "1889":
        raise SystemExit(f"{directory} holds a {meta['game']} buffer")
    order = _write_order(meta["size"], meta["index"], meta["capacity"])
    states = np.load(directory / "states.npy", mmap_mode="r")[order]
    counts = np.load(directory / "player_counts.npy")[order]
    return states, counts, int(meta["max_players"])


def find_positions(game: Game, states: np.ndarray, player_counts: np.ndarray, max_players: int,
                   per_cell: int, rng: np.random.Generator) -> list[Position]:
    """``per_cell`` positions for each player count and phase, from as many games, drawn at
    random. ``states`` are a buffer's rows in write order; rows before its first game start
    are the end of a game the ring overwrote, and are skipped."""
    assert game.game_start_rows is not None
    starts = np.asarray(game.game_start_rows(states, max_players))
    ends = np.r_[starts[1:], len(states)]
    by_cell: dict[tuple[int, str], list[Position]] = defaultdict(list)
    for start, end in zip(starts.tolist(), ends.tolist()):
        rows = states[start:end]
        maybe = (rows[:, ROUND] == K.ROUND_OPERATING) & np.isin(rows[:, GAME_PHASE], PHASES)
        found = []
        for i in (np.flatnonzero(maybe) + start).tolist():
            row = np.array(states[i])
            state = GameState.from_array(row, int(player_counts[i]), max_players=max_players)
            target = finance_target(state)
            if target is not None:
                corp, train, price = target
                found.append(Position(
                    -1, int(player_counts[i]), D.PHASE_NAMES[PHASE.current(state)], corp,
                    CORPS[corp].president(state), CORPS[corp].cash(state), TURN.route_revenue(state),
                    train, price, end - i, i, [int(v) for v in row],
                ))
        if found:
            pick = found[int(rng.integers(len(found)))]
            by_cell[(pick.num_players, pick.phase)].append(pick)
    chosen = []
    for cell in sorted(by_cell):
        games = by_cell[cell]
        chosen += [games[i] for i in sorted(rng.choice(len(games), min(per_cell, len(games)), replace=False))]
    chosen.sort(key=lambda p: p.buffer_row)
    return [replace(p, id=i) for i, p in enumerate(chosen)]


def save_positions(path: Path, positions: list[Position], source: dict[str, Any]) -> None:
    path.write_text(json.dumps({"source": source, "positions": [asdict(p) for p in positions]}))


def load_positions(path: Path) -> list[Position]:
    return [Position(**p) for p in json.loads(path.read_text())["positions"]]


# =============================================================================
# BRANCHES
# =============================================================================


def position_state(game: Game, root: GameState, position: Position) -> GameState:
    """A fresh state at ``position``, with ``root``'s runtime flags."""
    return game.scratch_state(root, np.array(position.row, dtype=np.int16))


def _apply_choice(game: Game, state: GameState, phase: int, want: Callable[[Any], bool]) -> None:
    """Apply the one choice at ``state``'s ``phase`` decision whose decoded action ``want`` takes."""
    assert game.decision_phase(state) == phase, f"expected {K.DecisionPhase(phase).name}, at {game.decision_phase(state)}"
    actions = np.empty(game.policy.max_action_size, dtype=np.uint16)
    count = game.enumerate_policy_actions(state, actions)
    matches = [int(a) for a in actions[:count] if want(A.decode_action_py(phase, int(a)))]
    assert len(matches) == 1, f"{len(matches)} matching choices at {K.DecisionPhase(phase).name}"
    status = game.apply_action(state, matches[0])
    assert status != STATUS_INVALID


def finance(game: Game, state: GameState, position: Position) -> None:
    """The ``finance`` branch's own moves: withhold at the position's DIVIDEND, then buy its
    train from the depot at the corp's BUY_TRAIN, choosing the type if the depot asks."""
    held = TRAINS.corp_trains(state, position.corp, position.train)
    _apply_choice(game, state, K.DPHASE_DIVIDEND, lambda info: info.type == A.ACTION_WITHHOLD)
    assert TURN.operating_corp(state) == position.corp
    _apply_choice(game, state, K.DPHASE_BUY_TRAIN, lambda info: info.type == A.ACTION_BUY_DEPOT_TRAIN)
    if game.decision_phase(state) == K.DPHASE_SELECT_TRAIN and TURN.operating_corp(state) == position.corp:
        _apply_choice(game, state, K.DPHASE_SELECT_TRAIN,
                      lambda info: info.type == A.ACTION_TRAIN_TYPE and info.train_type == position.train)
    assert TRAINS.corp_trains(state, position.corp, position.train) == held + 1


def branch_state(game: Game, root: GameState, position: Position, branch: str) -> GameState:
    state = position_state(game, root, position)
    if branch == "finance":
        finance(game, state, position)
    else:
        assert branch == "model", branch
    return state


def continuation_config(config: TrainingConfig) -> TrainingConfig:
    """``config`` at its late-game temperature from the first move: ``play_game`` counts
    moves from its start, and every position is past the temperature windows."""
    return replace(config, temp_initial=config.temp_final)


def _or_key(rows: np.ndarray) -> set[tuple[int, int]]:
    ops = rows[:, ROUND] == K.ROUND_OPERATING
    return set(zip(rows[ops, TURN_NUM].tolist(), rows[ops, OR_NUM].tolist()))


def continue_game(game: Game, evaluator: Any, config: TrainingConfig, epoch_config: EpochConfig,
                  start: GameState, seed: int, pool: StatePool, position: GameState | None = None) -> dict[str, Any]:
    """One ``play_game`` from ``start`` to the end, and how it ended. ``config`` should be
    ``continuation_config``'s. The ORs left and the trains bought count from ``position``,
    the branch's position before any forced move of its own (default ``start``)."""
    baseline = start if position is None else position
    num_players = TURN.num_players(start)
    first: list[tuple[int, int, bool]] = []  # the first decision's choice, phase, and whether it is where start is

    class Recorder(G1889SelfPlayStats):
        def observe(self, state, legal_actions, chosen_idx, priors, visit_counts) -> None:
            super().observe(state, legal_actions, chosen_idx, priors, visit_counts)
            if not first:
                first.append((int(legal_actions[chosen_idx]), get_decision_phase_py(state),
                              bool(np.array_equal(state._array[WHERE], start._array[WHERE]))))

    def new_game(_num_players: int, _max_players: int, _seed: int) -> GameState:
        return game.scratch_state(start, start._array.copy())

    spied = replace(game, new_game=new_game, new_selfplay_stats=Recorder)
    with patch.object(self_play, "get_game", lambda _: spied):
        record = self_play.play_game(
            evaluator, config, game_seed=seed, rng=np.random.default_rng(seed), state_pool=pool,
            epoch_config=epoch_config, num_players=num_players,
        )
    stats = record.game_stats
    net_worths = [int(v) for v in record.net_worths]
    values = compute_terminal_values(net_worths, num_players, config.terminal_blend)
    return {
        "net_worths": net_worths,
        "values": [round(float(v), 4) for v in values],
        "decisions": int(record.total_moves),
        "ors_left": len(_or_key(baseline._array[None, :]) | _or_key(record.states)),
        "last_phase": next(name for name in D.PHASE_NAMES if stats[f"last_phase_{name}"][0]),
        "end": "bank" if stats["end_bank"][0] else "bankruptcy",
        "trains_after": {name: int(stats[f"trains_bought_{name}"][0]) - _trains_bought(baseline, t)
                         for t, name in enumerate(D.TRAIN_NAMES)},
        "first_action": first[0][0] if first else None,
        "first_phase": K.DecisionPhase(first[0][1]).name if first else None,
        "started_at_start": first[0][2] if first else None,
    }


def readout(game: Game, evaluator: Any, config: TrainingConfig, epoch_config: EpochConfig,
            root: GameState, position: Position, deep: int, pools: dict[int, StatePool]) -> dict[str, Any]:
    """What the model expects at ``position``, for its president: searches at the epoch's
    simulations and ``deep`` times them, without root noise, of the position (its value,
    and each choice's Q and visit share) and of the ``finance`` branch's state after its
    purchase; and the raw value head on both."""
    president = position.president
    out: dict[str, Any] = {}
    for label, sims in (("base", epoch_config.num_simulations), ("deep", deep * epoch_config.num_simulations)):
        mcts_config = replace(
            config.to_mcts_config(c_puct_override=epoch_config.c_puct, num_simulations_override=sims,
                                  num_players=position.num_players),
            dirichlet_epsilon=0.0,
        )
        if sims not in pools:
            pools[sims] = StatePool(game, 2 * (sims + 1), game.state_size(config.effective_max_players))
        rng = np.random.default_rng(0)
        search = run_search(game, branch_state(game, root, position, "model"), evaluator, mcts_config, rng,
                            state_pool=pools[sims])
        assert search.legal_actions is not None and search.visit_counts is not None and search.value_sums is not None
        entry: dict[str, Any] = {"root": float(search.value_sum[president] / search.visit_count)}
        visits = search.visit_counts
        for i, action in enumerate(search.legal_actions.tolist()):
            kind = A.decode_action_py(K.DPHASE_DIVIDEND, int(action)).type
            name = "withhold" if kind == A.ACTION_WITHHOLD else "payout"
            entry[name] = {"q": float(search.value_sums[i, president] / max(1, visits[i])),
                           "share": float(visits[i] / visits.sum())}
        financed = run_search(game, branch_state(game, root, position, "finance"), evaluator, mcts_config, rng,
                              state_pool=pools[sims])
        entry["finance"] = float(financed.value_sum[president] / financed.visit_count)
        out[label] = {k: v if isinstance(v, dict) else round(v, 4) for k, v in entry.items()}
    out["value_head"] = {branch_name: round(float(evaluator.evaluate(branch_state(game, root, position, branch))[1][president]), 4)
                         for branch_name, branch in (("root", "model"), ("finance", "finance"))}
    return out


# =============================================================================
# PLAYING
# =============================================================================

Task = tuple[int, str, int]  # (position id, "readout" or a branch, continuation)


def tasks_for(positions: list[Position], continuations: int) -> list[Task]:
    """Each position's readout, then its continuations, branch by branch."""
    return [task for p in positions for task in
            [(p.id, "readout", 0), *((p.id, b, k) for k in range(continuations) for b in BRANCHES)]]


def continuation_seed(position_id: int, k: int) -> int:
    return 1_000_003 * position_id + k


class Player:
    """What ``play`` needs loaded once: the checkpoint's model, settings, and pools."""

    def __init__(self, checkpoint: Path, positions: list[Position], device: torch.device, deep: int) -> None:
        model, config, cp = load_model_from_checkpoint(checkpoint, device)
        if config.game != "1889":
            raise SystemExit(f"{checkpoint} is a {config.game} checkpoint")
        self.config = continuation_config(config)
        self.epoch_config = self.config.compute_epoch_config(int(cp["epoch"]) + 1)  # type: ignore[arg-type]
        self.game = get_game(self.config)
        max_players = self.config.effective_max_players
        self.evaluator = NNEvaluator(
            self.game, model, device, num_players=max_players, terminal_rank_weight=self.config.terminal_blend,
            eval_dtype=None, input_spec=get_model_input_spec(self.config),
        )
        self.root = self.game.new_game(max_players, max_players, 0)  # carries the config's runtime flags
        width = self.game.state_size(max_players)
        if any(len(p.row) != width for p in positions):
            raise SystemExit(f"the positions' rows are not {checkpoint}'s width, {width}")
        self.positions = {p.id: p for p in positions}
        self.deep = deep
        sims = self.epoch_config.num_simulations
        self.pool = StatePool(self.game, 2 * (max(self.config.max_simulations, sims) + 1),
                              self.game.state_size(max_players))
        self.readout_pools: dict[int, StatePool] = {}

    def run(self, task: Task) -> dict[str, Any]:
        position_id, kind, k = task
        position = self.positions[position_id]
        start = time.perf_counter()
        if kind == "readout":
            result = readout(self.game, self.evaluator, self.config, self.epoch_config, self.root, position,
                             self.deep, self.readout_pools)
        else:
            state = branch_state(self.game, self.root, position, kind)
            result = continue_game(self.game, self.evaluator, self.config, self.epoch_config, state,
                                   continuation_seed(position_id, k), self.pool,
                                   position_state(self.game, self.root, position))
        return {"position": position_id, "buffer_row": position.buffer_row, "kind": kind, "k": k, **result,
                "simulations": self.epoch_config.num_simulations, "secs": round(time.perf_counter() - start, 1)}


_PLAYER: Player | None = None


def _init_worker(checkpoint: str, positions_path: str, device: str, deep: int) -> None:
    global _PLAYER
    _PLAYER = Player(Path(checkpoint), load_positions(Path(positions_path)), torch.device(device), deep)


def _init_pool_worker(*args: Any) -> None:
    torch.set_num_threads(1)  # the workers share the CPU
    _init_worker(*args)


def _run_task(task: Task) -> dict[str, Any]:
    assert _PLAYER is not None
    return _PLAYER.run(task)


def run_tasks(tasks: list[Task], *, checkpoint: Path, positions_path: Path, device: str, deep: int,
              workers: int) -> Iterator[dict[str, Any]]:
    """The tasks' results as they finish: in this process, or in ``workers`` processes
    that each load the model."""
    initargs = (str(checkpoint), str(positions_path), device, deep)
    if workers == 1:
        _init_worker(*initargs)
        yield from map(_run_task, tasks)
        return
    # A worker that dies, or fails to start, breaks the pool and raises, where mp.Pool would hang.
    executor = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init_pool_worker,
                                   initargs=initargs)
    try:
        for future in as_completed([executor.submit(_run_task, task) for task in tasks]):
            yield future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def read_results(path: Path) -> list[dict[str, Any]]:
    """The results in ``path``, less a last line cut off mid-write."""
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def _drop_partial_line(path: Path) -> None:
    """Cut a last line left unfinished by a kill, so the next result starts a line."""
    if path.exists() and (text := path.read_text()) and not text.endswith("\n"):
        path.write_text(text[:text.rfind("\n") + 1])


def play(args: argparse.Namespace) -> None:
    positions = load_positions(args.positions)
    done = {(r["position"], r["kind"], r["k"]) for r in read_results(args.results)}
    tasks = [t for t in tasks_for(positions, args.continuations) if t not in done]
    print(f"{len(positions)} positions, {args.continuations} continuations a branch: {len(tasks)} tasks to run, "
          f"{len(done)} already in {args.results}", file=sys.stderr, flush=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    _drop_partial_line(args.results)
    with args.results.open("a") as out:
        for n, result in enumerate(run_tasks(tasks, checkpoint=args.checkpoint, positions_path=args.positions,
                                             device=device, deep=args.deep, workers=args.workers), 1):
            out.write(json.dumps(result) + "\n")
            out.flush()
            elapsed = time.perf_counter() - start
            print(f"  {n}/{len(tasks)} position {result['position']} {result['kind']} {result['k']}: "
                  f"{result['secs']}s; {elapsed / 60:.0f} min, about {elapsed / n * (len(tasks) - n) / 60:.0f} left",
                  file=sys.stderr, flush=True)


# =============================================================================
# THE REPORT
# =============================================================================


def _mean_ci(values: np.ndarray, rng: np.random.Generator, resamples: int) -> tuple[float, float, float]:
    """The mean and a bootstrap 95% interval, resampling positions."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = rng.choice(values, (resamples, len(values))).mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(positions: list[Position], results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each position with a continuation in both branches: the branches' means over the
    seeds both have, the ``finance`` minus ``model`` differences, for the president and the
    others, and its readout's predicted differences."""
    by_position: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_position[r["position"]][r["kind"]].append(r)
    rows = []
    for p in positions:
        runs = by_position.get(p.id, {})
        if any(r["buffer_row"] != p.buffer_row for kind in runs.values() for r in kind):
            raise ValueError(f"results for position {p.id} come from another positions file")
        shared = {r["k"] for r in runs.get("model", [])} & {r["k"] for r in runs.get("finance", [])}
        if not shared:
            continue
        runs = {kind: [r for r in rs if kind == "readout" or r["k"] in shared] for kind, rs in runs.items()}
        others = [i for i in range(p.num_players) if i != p.president]

        def mean(branch: str, measure: Callable[[dict[str, Any]], float]) -> float:
            return float(np.mean([measure(r) for r in runs[branch]]))

        measures: dict[str, Callable[[dict[str, Any]], float]] = {
            "value": lambda r: r["values"][p.president],
            "worth": lambda r: r["net_worths"][p.president],
            "others_worth": lambda r: float(np.mean([r["net_worths"][i] for i in others])),
            "win": lambda r: float(r["net_worths"][p.president] == max(r["net_worths"])),
            "ors_left": lambda r: r["ors_left"],
            "late": lambda r: float(r["last_phase"] in ("6", "D")),
            "sixes": lambda r: r["trains_after"]["6"],
            "ds": lambda r: r["trains_after"]["D"],
        }
        row: dict[str, Any] = {"position": p.id, "num_players": p.num_players, "phase": p.phase,
                               "continuations": len(shared)}
        for name, measure in measures.items():
            row[f"model_{name}"], row[f"finance_{name}"] = mean("model", measure), mean("finance", measure)
            row[f"d_{name}"] = row[f"finance_{name}"] - row[f"model_{name}"]
        row["model_withheld"] = mean("model", lambda r: float(
            r["first_action"] is not None
            and A.decode_action_py(K.DPHASE_DIVIDEND, r["first_action"]).type == A.ACTION_WITHHOLD))
        values = [r["values"][p.president] for b in BRANCHES for r in runs[b]]
        row["within_sd"] = float(np.mean([np.std([r["values"][p.president] for r in runs[b]], ddof=1)
                                          for b in BRANCHES if len(runs[b]) > 1])) if len(values) > 2 else float("nan")
        if runs.get("readout"):
            read = runs["readout"][0]
            for label in ("base", "deep", "value_head"):
                row[f"predicted_{label}"] = read[label]["finance"] - read[label]["root"]
            for label in ("base", "deep"):
                row[f"withhold_share_{label}"] = read[label].get("withhold", {}).get("share", float("nan"))
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]], *, resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Means with bootstrap intervals over positions, overall, by player count, and by phase;
    the readout's predictions against the played-out differences; and the spread."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict[str, Any]]] = {"all": rows}
    for n in sorted({r["num_players"] for r in rows}):
        groups[f"{n}p"] = [r for r in rows if r["num_players"] == n]
    for phase in sorted({r["phase"] for r in rows}):
        groups[f"phase {phase}"] = [r for r in rows if r["phase"] == phase]
    names = ["value", "worth", "others_worth", "win", "ors_left", "late", "sixes", "ds"]
    summary: dict[str, Any] = {"groups": {}}
    for label, group in groups.items():
        entry: dict[str, Any] = {"positions": len(group)}
        for name in names:
            entry[f"d_{name}"] = _mean_ci(np.array([r[f"d_{name}"] for r in group]), rng, resamples)
            entry[f"model_{name}"] = float(np.mean([r[f"model_{name}"] for r in group])) if group else float("nan")
            entry[f"finance_{name}"] = float(np.mean([r[f"finance_{name}"] for r in group])) if group else float("nan")
        summary["groups"][label] = entry
    realized = np.array([r["d_value"] for r in rows])
    predictions: dict[str, Any] = {}
    for label in ("base", "deep", "value_head"):
        have = [r for r in rows if f"predicted_{label}" in r]
        if not have:
            continue
        predicted = np.array([r[f"predicted_{label}"] for r in have])
        actual = np.array([r["d_value"] for r in have])
        predictions[label] = {
            "mean": _mean_ci(predicted, rng, resamples),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "correlation": float(np.corrcoef(predicted, actual)[0, 1]) if len(have) > 2 and predicted.std() > 0 else float("nan"),
        }
    summary["predictions"] = predictions
    for label in ("base", "deep"):
        shares = [r[f"withhold_share_{label}"] for r in rows if f"withhold_share_{label}" in r]
        summary[f"withhold_visit_share_{label}"] = float(np.nanmean(shares)) if shares else float("nan")
    summary["model_withheld"] = float(np.mean([r["model_withheld"] for r in rows])) if rows else float("nan")
    between = float(realized.std(ddof=1)) if len(rows) > 1 else float("nan")
    summary["spread"] = {
        "between_positions_sd": between,
        "within_branch_sd": float(np.nanmean([r["within_sd"] for r in rows])) if rows else float("nan"),
        # Positions for a 95% interval of +/-0.03 on the mean value difference, at this spread.
        "positions_for_0.03": int(np.ceil((1.96 * between / 0.03) ** 2)) if between == between else None,
    }
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    def ci(t: tuple[float, float, float], fmt: str = "{:+.3f}") -> str:
        return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"

    lines = ["President, finance minus model (95% bootstrap interval over positions):", ""]
    lines.append(f"{'group':10s} {'n':>3s}  {'terminal value':26s} {'net worth':28s} {'others net worth':28s} "
                 f"{'win rate':26s} {'ORs left':26s}")
    for label, g in summary["groups"].items():
        lines.append(f"{label:10s} {g['positions']:3d}  {ci(g['d_value']):26s} {ci(g['d_worth'], '{:+,.0f}'):28s} "
                     f"{ci(g['d_others_worth'], '{:+,.0f}'):28s} {ci(g['d_win']):26s} {ci(g['d_ors_left'], '{:+.2f}'):26s}")
    lines += ["", "Late game, model -> finance: games ending in phase 6 or D; 6s and Ds bought after the position"]
    for label, g in summary["groups"].items():
        lines.append(f"{label:10s} {g['model_late']:.0%} -> {g['finance_late']:.0%};  6s {g['model_sixes']:.2f} -> "
                     f"{g['finance_sixes']:.2f};  Ds {g['model_ds']:.2f} -> {g['finance_ds']:.2f}")
    lines += ["", "The model's predicted difference (finance minus the position's value), against the played-out one:"]
    for label, p in summary["predictions"].items():
        lines.append(f"  {label:10s} mean {ci(p['mean'])}  sign agrees {p['sign_agreement']:.0%}  "
                     f"correlation {p['correlation']:+.2f}")
    lines.append(f"  withhold's visit share at the position: base {summary['withhold_visit_share_base']:.1%}, "
                 f"deep {summary['withhold_visit_share_deep']:.1%}; the model branch withheld "
                 f"{summary['model_withheld']:.1%} of its continuations")
    s = summary["spread"]
    lines += ["", f"Spread of the value difference: {s['between_positions_sd']:.3f} between positions, "
              f"{s['within_branch_sd']:.3f} between continuations of a branch; about {s['positions_for_0.03']} "
              "positions for an interval of +/-0.03"]
    return "\n".join(lines)


def report(args: argparse.Namespace) -> None:
    rows = paired(load_positions(args.positions), read_results(args.results))
    summary = summarize(rows)
    print(format_summary(summary))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "positions": rows}, indent=1))


# =============================================================================
# THE COMMAND LINE
# =============================================================================


def positions_command(args: argparse.Namespace) -> None:
    config = TrainingConfig(game="1889", model_path="nn/transformer-1889.py", min_players=3, max_players=5,
                            num_players=0)
    states, counts, max_players = load_buffer(args.buffer)
    positions = find_positions(get_game(config), states, counts, max_players, args.per_cell,
                               np.random.default_rng(args.seed))
    save_positions(args.out, positions, {"buffer": str(args.buffer), "per_cell": args.per_cell, "seed": args.seed})
    cells: dict[tuple[int, str], int] = defaultdict(int)
    for p in positions:
        cells[(p.num_players, p.phase)] += 1
    print(f"{len(positions)} positions in {args.out}: " + ", ".join(
        f"{n}p phase {phase}: {count}" for (n, phase), count in sorted(cells.items())))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("positions", help="draw positions from a saved replay buffer")
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-cell", type=int, default=5, help="positions per player count and phase")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("play", help="run the readouts and continuations")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--continuations", type=int, default=2, help="per branch and position")
    p.add_argument("--deep", type=int, default=4, help="the deep readout's multiple of the epoch's simulations")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p = sub.add_parser("report", help="summarize the results")
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--json", type=Path, default=None, help="also write the summary and per-position rows here")
    args = parser.parse_args(argv)
    {"positions": positions_command, "play": play, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
