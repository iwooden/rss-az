"""Locked 1889 layout 1: v0's inputs must not drift (1889/v1-design.md §8).

``nn/transformer-1889.py`` (v0) reads token layout 1 and relations 0-24, and
stays playable beside later layouts as the tournament's opponent. On the
fixture's states the engine must give layout 1's tokens bit for bit, and the
relation records with an id of 24 or below, in order, whatever later ids the
emitter adds around them.

The fixture, ``tests/g1889/layout1_parity/expected.npz``, names each state by
how to reach it, not by its raw row, so it survives changes to the state
layout: a game is its player count, capacity, first player or seed, engine
flags, and the driver calls made from ``new_game``; a state is a game and how
many of its calls come before it. The games are self-play games (the saved
decisions of ``utils_18xx/g1889_selfplay_log.py``), human games (the calls the
differential-replay harness makes for them, its replay-only overrides
included), and random playouts under both flag settings; together their
states cover every decision phase and the game's end, 3-5 players at
capacities 3-5, and every game phase.

Each state is checked with its caches as the driver left them and with every
cache marked stale.

Regenerating the fixture redefines what v0 is given. Do it only for an
approved change to layout 1 (``--selfplay`` takes saved games, ``--human``
scraped ones, which need the replay tests' Ruby extracts):

    .venv/bin/python -m tests.g1889.test_layout1_parity --capture \\
        --selfplay 'DIR/*.played.json' [--human tests/g1889/games/data]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER, STATUS_INVALID
from g1889.entities.phase import PHASE
from g1889.relations import RelationSize, get_relation_coord_data
from g1889.state import GameState, get_corp_fields, get_global_fields, get_layout
from g1889.token_data import get_num_tokens, get_token_data

FIXTURE = Path(__file__).with_name("layout1_parity") / "expected.npz"
LAYOUT_VERSION = 1
TOKEN_DIM = 108          # layout 1's, pinned here and not read from the engine
LAST_RELATION = 24       # v0 consumes ids 0-24
RECORD_WIDTH = 4

# A driver call: (kind, argument). The harness's replay-only calls are kinds 1-5.
OPS = ("apply_action", "apply_out_of_turn", "override_route_revenue", "override_private_slot",
       "override_topmost_bid", "advance")
_NO_ARGUMENT = (OPS.index("override_topmost_bid"), OPS.index("advance"))

_GLOBALS = get_global_fields()
_GLOBAL_OFFSET = get_layout(0).global_offset  # the same at every capacity
# Capture-only settings; the fixture records the resulting games.
_RANDOM_GAMES = (  # (num_players, max_players, seed, (private_sale, train_sale, payout cap))
    (3, 3, 11, (True, True, 1)), (4, 5, 12, (True, True, 1)), (5, 5, 13, (True, True, 1)),
    (3, 5, 21, (False, False, 1)), (4, 4, 22, (False, False, 1)), (5, 5, 23, (False, False, 1)),
)
_STRIDE = 37             # every state at a multiple of it, and the first of each rarer kind
_PER_KIND = 2            # states kept per (decision phase, game phase, player count)


# =============================================================================
# GAMES AND STATES
# =============================================================================

def new_state(spec: dict[str, Any]) -> GameState:
    private_sale, train_sale, payout_cap = spec["flags"]
    state = GameState(spec["num_players"], max_players=spec["max_players"],
                      private_sale_same_president=private_sale, train_sale_same_president=train_sale,
                      auction_pass_payout_cap=payout_cap)
    state.initialize_game(spec["num_players"], seed=spec["seed"], max_players=spec["max_players"],
                          priority_player=spec["priority_player"])
    return state


def apply_op(state: GameState, kind: int, argument: int) -> int:
    args = () if kind in _NO_ARGUMENT else (argument,)
    status = getattr(DRIVER, OPS[kind])(state, *args)
    assert status != STATUS_INVALID, f"{OPS[kind]}({argument}) is illegal"
    return status


def _copy(state: GameState, stale: bool) -> GameState:
    """A copy of the state, flags aside (inputs never read them), with its
    caches as they are or every one marked stale: each ``*_dirty`` mask set."""
    copy = GameState.from_array(state._array, state._array[_GLOBAL_OFFSET + _GLOBALS.num_players],
                                max_players=state.max_players)
    if stale:
        for name in _GLOBALS._fields:
            if name.endswith("_dirty"):
                copy._array[_GLOBAL_OFFSET + getattr(_GLOBALS, name)] = (1 << D.NUM_CORPS) - 1
    return copy


def inputs(state: GameState, stale: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Layout 1's tokens and the records of relations 0-24, in order, for a
    copy of ``state``."""
    copy = _copy(state, stale)
    tokens = np.full((get_num_tokens(state.max_players), TOKEN_DIM), np.nan, np.float32)
    get_token_data(copy, tokens, state.max_players, layout_version=LAYOUT_VERSION)
    records = np.zeros((int(RelationSize.MAX_RELATION_EDGES), RECORD_WIDTH), np.uint8)
    count = get_relation_coord_data(copy, records, state.max_players)
    records = records[:count]
    return tokens, records[records[:, 0] <= LAST_RELATION]


def load_fixture() -> dict[str, Any]:
    with np.load(FIXTURE, allow_pickle=False) as data:
        fixture = {name: data[name] for name in data.files}
    fixture["games"] = json.loads(fixture["games"].item())
    return fixture


# =============================================================================
# THE TEST
# =============================================================================

@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    return load_fixture()


def test_the_fixture_covers_every_phase_player_count_and_source(fixture):
    games = fixture["games"]
    of_game = fixture["state_game"]
    assert set(fixture["state_decision_phase"].tolist()) == {-1, *range(K.NUM_DECISION_PHASES)}
    assert set(fixture["state_game_phase"].tolist()) == set(range(D.NUM_GAME_PHASES))
    assert {games[g]["num_players"] for g in of_game} == {3, 4, 5}
    assert {games[g]["max_players"] for g in of_game} == {3, 4, 5}
    assert {games[g]["kind"] for g in of_game} == {"selfplay", "human", "random"}
    assert len(of_game) >= 300


def test_layout_1_tokens_and_relations_0_to_24_are_unchanged(fixture):
    """Re-drive each game; at each of its fixture states the tokens and the
    records are the stored ones, with caches riding or stale."""
    token_start, record_start = fixture["token_start"], fixture["record_start"]
    checked = 0
    for g, spec in enumerate(fixture["games"]):
        ops = fixture[f"ops_{g}"]
        states = np.flatnonzero(fixture["state_game"] == g)
        at = {int(fixture["state_ops"][s]): int(s) for s in states}
        state = new_state(spec)
        for done in range(len(ops) + 1):
            if (s := at.get(done)) is not None:
                where = f"{spec['kind']} game {spec['label']} after {done} calls"
                assert A.get_decision_phase_py(state) == fixture["state_decision_phase"][s], where
                assert PHASE.current(state) == fixture["state_game_phase"][s], where
                tokens = fixture["tokens"][token_start[s]:token_start[s + 1]]
                records = fixture["records"][record_start[s]:record_start[s + 1]]
                for stale in (False, True):
                    got_tokens, got_records = inputs(state, stale)
                    np.testing.assert_array_equal(got_tokens, tokens, err_msg=f"{where}, stale {stale}")
                    np.testing.assert_array_equal(got_records, records, err_msg=f"{where}, stale {stale}")
                checked += 1
            if done < len(ops):
                apply_op(state, int(ops[done, 0]), int(ops[done, 1]))
    assert checked == len(fixture["state_game"])


# =============================================================================
# CAPTURE
# =============================================================================

def _cache_free(row: np.ndarray) -> bytes:
    """The row less its caches, which the harness's comparisons refresh between calls."""
    layout, corp = get_layout(0), get_corp_fields()
    row = row.copy()
    row[layout.graph_offset:layout.graph_offset + D.NUM_CORPS * D.NUM_HEXES] = 0
    caches = [getattr(corp, name) for name in corp._fields if name.startswith("route_")]
    for c in range(D.NUM_CORPS):
        row[[layout.corps_offset + c * corp.size + field for field in caches]] = 0
    row[[_GLOBAL_OFFSET + getattr(_GLOBALS, name) for name in _GLOBALS._fields if name.endswith("_dirty")]] = 0
    return row.tobytes()


class _CallRecorder:
    """``DRIVER`` for the replay harness, keeping each call it makes. The
    harness backs up to an earlier row in a few equivalences. Back at a row a
    call was made from, the calls after it are dropped. Back at a row inside a
    call's automated chain, which no call from ``new_game`` reaches, the
    record ends: ``calls`` is then a prefix of the game."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []
        self.first: np.ndarray | None = None
        self.last: np.ndarray | None = None
        self.ended = False
        self._before: list[bytes] = []
        self._at: dict[bytes, int] = {}
        self._after: bytes | None = None

    def _call(self, kind: int, state: GameState, args: tuple) -> int:
        key = _cache_free(state._array)
        if self.first is None:
            self.first = state._array.copy()
        if not self.ended and self._after is not None and key != self._after:
            k = self._at.get(key)
            if k is None:
                self.ended = True
            else:
                for row in self._before[k:]:
                    self._at.pop(row, None)
                del self.calls[k:], self._before[k:]
        if not self.ended:
            self._at[key] = len(self.calls)
            self.calls.append((kind, 0 if kind in _NO_ARGUMENT else int(args[0])))
            self._before.append(key)
        status = getattr(DRIVER, OPS[kind])(state, *args)
        if not self.ended:
            self._after = _cache_free(state._array)
            self.last = state._array.copy()
        return status

    def __getattr__(self, name: str) -> Any:
        if name in OPS:
            return lambda state, *args: self._call(OPS.index(name), state, args)
        return getattr(DRIVER, name)


def _human_game(path: Path, max_players: int) -> tuple[dict[str, Any], np.ndarray] | None:
    """The harness's calls for a scraped game, if a fresh game replays them."""
    from unittest.mock import patch

    from tests.g1889.replay import harness
    from tests.g1889.replay.extracts import forget_extracts, load_scraped_extract

    recorder = _CallRecorder()
    try:
        with patch.object(harness, "DRIVER", recorder):
            harness.replay_extract(path.stem, load_scraped_extract(path))
    finally:
        forget_extracts()
    if recorder.first is None:
        return None
    first = recorder.first
    spec = {"kind": "human", "label": path.stem, "num_players": int(first[_GLOBAL_OFFSET + _GLOBALS.num_players]),
            "max_players": max_players, "seed": -1,
            "priority_player": int(first[_GLOBAL_OFFSET + _GLOBALS.priority_player]), "flags": [False, False, 0]}
    calls = np.array(recorder.calls, np.int16).reshape(-1, 2)
    state = new_state({**spec, "max_players": spec["num_players"]})  # the harness's capacity
    if not np.array_equal(state._array, first):
        return None
    for kind, argument in calls.tolist():
        apply_op(state, kind, argument)
    assert recorder.last is not None and _cache_free(state._array) == _cache_free(recorder.last), \
        f"{path.stem}: a fresh game does not replay the harness's calls"
    return spec, calls


def _selfplay_game(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    played = json.loads(path.read_text())
    config = json.loads(played["config_json"])
    spec = {"kind": "selfplay", "label": path.name.removesuffix(".played.json"), "num_players": played["num_players"],
            "max_players": 5, "seed": played["seed"], "priority_player": -1,
            "flags": [config["private_sale_same_president"], config["train_sale_same_president"],
                      config["auction_pass_payout_cap"]]}
    return spec, np.array([(0, a) for a in played["actions"]], np.int16).reshape(-1, 2)


def _random_game(num_players: int, max_players: int, seed: int, flags: tuple) -> tuple[dict[str, Any], np.ndarray]:
    """Uniform over the policy set, leaning to depot purchases so the game reaches the late phases."""
    spec = {"kind": "random", "label": f"{num_players}p-{seed}", "num_players": num_players,
            "max_players": max_players, "seed": seed, "priority_player": -1, "flags": list(flags)}
    state = new_state(spec)
    rng = np.random.default_rng(seed)
    ids = np.empty(K.MAX_ACTION_SIZE, np.uint16)
    calls = []
    while not A.is_terminal_py(state):
        count = A.enumerate_policy_actions_py(state, ids)
        choices = ids[:count].tolist()
        if A.get_decision_phase_py(state) == K.DPHASE_BUY_TRAIN and K.BUY_TRAIN_DEPOT in choices \
                and rng.random() < 0.7:
            action = K.BUY_TRAIN_DEPOT
        else:
            action = choices[int(rng.integers(count))]
        calls.append((0, action))
        apply_op(state, 0, action)
    return spec, np.array(calls, np.int16).reshape(-1, 2)


def _kinds(spec: dict[str, Any], ops: np.ndarray) -> list[tuple[int, int]]:
    """(decision phase, game phase) after each prefix of the game's calls."""
    state = new_state(spec)
    kinds = [(A.get_decision_phase_py(state), PHASE.current(state))]
    for kind, argument in ops.tolist():
        apply_op(state, kind, argument)
        kinds.append((A.get_decision_phase_py(state), PHASE.current(state)))
    return kinds


def capture(selfplay: list[Path], human: list[Path]) -> None:
    games: list[tuple[dict[str, Any], np.ndarray]] = [_selfplay_game(path) for path in selfplay]
    capacities = {3: (3, 5), 4: (4, 5), 5: (5, 5)}
    taken: Counter = Counter()
    for path in human:
        players = len(json.loads(path.read_text())["players"])
        if players not in capacities or taken[players] == 2:
            continue
        game = _human_game(path, capacities[players][taken[players]])
        # A game that reaches phase D by calls a fresh game replays.
        if game is not None and _kinds(*game)[-1][1] == D.NUM_GAME_PHASES - 1:
            games.append(game)
            taken[players] += 1
            print(f"human game {path.stem}: {players} players, {len(game[1])} calls")
    games += [_random_game(*setup) for setup in _RANDOM_GAMES]

    out: dict[str, Any] = {"games": np.array(json.dumps([spec for spec, _ in games]))}
    state_game, state_ops, decision_phases, game_phases, tokens, records = [], [], [], [], [], []
    seen: Counter = Counter()
    for g, (spec, ops) in enumerate(games):
        out[f"ops_{g}"] = ops
        state = new_state(spec)
        status = None
        for done in range(len(ops) + 1):
            kind = (A.get_decision_phase_py(state), PHASE.current(state), spec["num_players"])
            if done % _STRIDE == 0 or seen[kind] < _PER_KIND or done == len(ops):
                seen[kind] += 1
                state_tokens, state_records = inputs(state)
                for stale_part, part in zip(inputs(state, stale=True), (state_tokens, state_records)):
                    assert np.array_equal(stale_part, part), "the inputs depend on the caches"
                state_game.append(g)
                state_ops.append(done)
                decision_phases.append(kind[0])
                game_phases.append(kind[1])
                tokens.append(state_tokens)
                records.append(state_records)
            if done < len(ops):
                status = apply_op(state, int(ops[done, 0]), int(ops[done, 1]))
        if spec["kind"] != "human":
            assert status == STATUS_GAME_OVER, f"{spec['label']} does not end"
    out.update(
        state_game=np.array(state_game, np.int16), state_ops=np.array(state_ops, np.int32),
        state_decision_phase=np.array(decision_phases, np.int8), state_game_phase=np.array(game_phases, np.int8),
        tokens=np.concatenate(tokens), token_start=np.cumsum([0, *map(len, tokens)]),
        records=np.concatenate(records), record_start=np.cumsum([0, *map(len, records)]),
    )
    FIXTURE.parent.mkdir(exist_ok=True)
    np.savez_compressed(FIXTURE, **out)
    by_kind = Counter(games[g][0]["kind"] for g in state_game)
    print(f"{FIXTURE}: {len(games)} games, {len(state_game)} states {dict(by_kind)}, "
          f"{FIXTURE.stat().st_size / 1e6:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", action="store_true", required=True)
    parser.add_argument("--selfplay", required=True, help="a glob of *.played.json games")
    parser.add_argument("--human", type=Path, default=Path("tests/g1889/games/data"))
    args = parser.parse_args()
    capture([Path(p) for p in sorted(glob.glob(args.selfplay))], sorted(args.human.glob("*.json")))


if __name__ == "__main__":
    main()
