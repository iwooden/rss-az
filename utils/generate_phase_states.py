#!/usr/bin/env python3
"""Generate one saved raw GameState per DecisionPhase.

Runs random legal games until every decision phase has at least one pre-action
state, then writes the phase-ordered fixture to ``tests/states.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.actions import enumerate_legal_actions_py, get_decision_phase_py
from core.data import DecisionPhase, GameConstants, GamePhases, MAX_ACTION_SIZE
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
)
from core.state import GameState
from entities.turn import TURN


DEFAULT_OUTPUT = REPO_ROOT / "tests" / "states.npz"
DEFAULT_NUM_PLAYERS = 3
DEFAULT_BASE_SEED = 0
DEFAULT_MAX_GAMES = 10_000
DEFAULT_MAX_STEPS_PER_GAME = 5_000


def _phase_names() -> list[str]:
    names_by_id = {
        int(member): name.removeprefix("DPHASE_")
        for name, member in DecisionPhase.__members__.items()
    }
    return [
        names_by_id[i]
        for i in range(int(GameConstants.NUM_DECISION_PHASES))
    ]


def _record_state(
    *,
    state_array: np.ndarray,
    phase_id: int,
    num_players: int,
    found: dict[int, dict[str, object]],
    game_index: int,
    game_seed: int,
    decision_step: int,
    history_index: int,
    action_id: int,
) -> None:
    if phase_id < 0 or phase_id in found:
        return

    state = GameState.from_array(state_array, num_players)
    actual_phase_id = int(get_decision_phase_py(state))
    if actual_phase_id != phase_id:
        raise RuntimeError(
            f"history phase {phase_id} disagrees with state phase "
            f"{actual_phase_id}"
        )

    legal_scratch = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    legal_count = int(enumerate_legal_actions_py(state, legal_scratch))
    if legal_count <= 0:
        raise RuntimeError(f"phase {phase_id} state has no legal actions")

    legal_actions = np.full(MAX_ACTION_SIZE, -1, dtype=np.int16)
    legal_actions[:legal_count] = legal_scratch[:legal_count].astype(np.int16)

    found[phase_id] = {
        "state": state_array.copy(),
        "legal_actions": legal_actions,
        "legal_count": legal_count,
        "game_index": game_index,
        "game_seed": game_seed,
        "decision_step": decision_step,
        "history_index": history_index,
        "action_id": action_id,
    }


def _play_random_game(
    *,
    num_players: int,
    game_index: int,
    game_seed: int,
    found: dict[int, dict[str, object]],
    max_steps: int,
) -> int:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=game_seed)
    rng = np.random.default_rng(game_seed)
    legal_scratch = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)

    for decision_step in range(max_steps):
        if TURN.get_phase(state) == int(GamePhases.PHASE_GAME_OVER):
            return decision_step

        phase_id = int(get_decision_phase_py(state))
        if phase_id < 0:
            raise RuntimeError(
                f"game {game_index} seed {game_seed} reached non-decision "
                f"engine phase {TURN.get_phase(state)} before apply_action"
            )

        legal_count = int(enumerate_legal_actions_py(state, legal_scratch))
        if legal_count <= 0:
            raise RuntimeError(
                f"game {game_index} seed {game_seed} step {decision_step} "
                f"phase {phase_id} has no legal actions"
            )

        chosen_idx = int(rng.integers(legal_count))
        action_id = int(legal_scratch[chosen_idx])
        history: list[tuple[np.ndarray, int, int]] = []
        status = int(DRIVER.apply_action(state, action_id, history=history))
        if status not in (STATUS_OK, STATUS_GAME_OVER):
            invalid_suffix = " (invalid)" if status == STATUS_INVALID else ""
            raise RuntimeError(
                f"game {game_index} seed {game_seed} step {decision_step} "
                f"action {action_id} returned status {status}{invalid_suffix}"
            )

        for history_index, (hist_state, hist_phase_id, hist_action_id) in enumerate(history):
            _record_state(
                state_array=hist_state,
                phase_id=int(hist_phase_id),
                num_players=num_players,
                found=found,
                game_index=game_index,
                game_seed=game_seed,
                decision_step=decision_step,
                history_index=history_index,
                action_id=int(hist_action_id),
            )

        if len(found) == int(GameConstants.NUM_DECISION_PHASES):
            return decision_step + 1
        if status == STATUS_GAME_OVER:
            return decision_step + 1

    raise RuntimeError(
        f"game {game_index} seed {game_seed} did not finish within "
        f"{max_steps} decisions"
    )


def generate_states(
    *,
    output: Path,
    num_players: int,
    base_seed: int,
    max_games: int,
    max_steps_per_game: int,
) -> dict[str, int]:
    found: dict[int, dict[str, object]] = {}
    num_phases = int(GameConstants.NUM_DECISION_PHASES)

    games_played = 0
    decisions_played = 0
    for game_index in range(max_games):
        game_seed = base_seed + game_index
        decisions_played += _play_random_game(
            num_players=num_players,
            game_index=game_index,
            game_seed=game_seed,
            found=found,
            max_steps=max_steps_per_game,
        )
        games_played = game_index + 1
        if len(found) == num_phases:
            break

    missing = sorted(set(range(num_phases)) - set(found))
    if missing:
        names = _phase_names()
        missing_names = ", ".join(names[i] for i in missing)
        raise RuntimeError(
            f"missing {len(missing)} decision phase(s) after {games_played} "
            f"games: {missing_names}"
        )

    phase_ids = np.arange(num_phases, dtype=np.int16)
    phase_names = np.asarray(_phase_names(), dtype="U32")
    states = np.stack(
        [found[i]["state"] for i in range(num_phases)]
    ).astype(np.int16, copy=False)
    legal_actions = np.stack(
        [found[i]["legal_actions"] for i in range(num_phases)]
    ).astype(np.int16, copy=False)
    legal_counts = np.asarray(
        [found[i]["legal_count"] for i in range(num_phases)],
        dtype=np.int16,
    )
    source_game_indices = np.asarray(
        [found[i]["game_index"] for i in range(num_phases)],
        dtype=np.int32,
    )
    source_game_seeds = np.asarray(
        [found[i]["game_seed"] for i in range(num_phases)],
        dtype=np.int64,
    )
    source_decision_steps = np.asarray(
        [found[i]["decision_step"] for i in range(num_phases)],
        dtype=np.int32,
    )
    source_history_indices = np.asarray(
        [found[i]["history_index"] for i in range(num_phases)],
        dtype=np.int16,
    )
    source_action_ids = np.asarray(
        [found[i]["action_id"] for i in range(num_phases)],
        dtype=np.int16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=states,
        phase_ids=phase_ids,
        phase_names=phase_names,
        legal_actions=legal_actions,
        legal_counts=legal_counts,
        num_players=np.asarray(num_players, dtype=np.int16),
        base_seed=np.asarray(base_seed, dtype=np.int64),
        games_played=np.asarray(games_played, dtype=np.int32),
        decisions_played=np.asarray(decisions_played, dtype=np.int32),
        source_game_indices=source_game_indices,
        source_game_seeds=source_game_seeds,
        source_decision_steps=source_decision_steps,
        source_history_indices=source_history_indices,
        source_action_ids=source_action_ids,
    )
    return {
        "games_played": games_played,
        "decisions_played": decisions_played,
        "num_phases": num_phases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play random legal games until one GameState is collected for "
            "each DecisionPhase, then write tests/states.npz."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output .npz path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=DEFAULT_NUM_PLAYERS,
        help=f"player count for generated games (default: {DEFAULT_NUM_PLAYERS})",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"seed for the first random game (default: {DEFAULT_BASE_SEED})",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=DEFAULT_MAX_GAMES,
        help=f"fail if coverage is not reached by this many games (default: {DEFAULT_MAX_GAMES})",
    )
    parser.add_argument(
        "--max-steps-per-game",
        type=int,
        default=DEFAULT_MAX_STEPS_PER_GAME,
        help=(
            "fail if one game does not finish by this many external decisions "
            f"(default: {DEFAULT_MAX_STEPS_PER_GAME})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = generate_states(
        output=args.output,
        num_players=args.num_players,
        base_seed=args.base_seed,
        max_games=args.max_games,
        max_steps_per_game=args.max_steps_per_game,
    )
    print(
        f"wrote {args.output} with {summary['num_phases']} phases from "
        f"{summary['games_played']} game(s), "
        f"{summary['decisions_played']} external decisions"
    )


if __name__ == "__main__":
    main()
