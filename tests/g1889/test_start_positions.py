"""Start positions from the human games (``utils_18xx/g1889_start_positions.py``,
1889/v1-design.md §2.1): a build from a few scraped games, which skips without
them, ``submodules/18xx``, or the means to make their extracts; the loader's
checks; and the checked-in file, if there is one."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import g1889.constants as K
from g1889.actions import enumerate_legal_actions_py
from g1889.driver import DRIVER
from g1889.entities.bank import BANK
from g1889.entities.phase import PHASE
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue
from g1889.state import GameState
from games import RELATION_RECORD_WIDTH, get_game
from train.config import TrainingConfig
from utils_18xx import g1889_start_positions as S
from utils_18xx.g1889_start_positions import (
    MIN_LEFT, StartPositions, build, load_start_positions, playout, playout_rng, save_start_positions,
    state_layout_hash,
)

from tests.g1889.replay.extracts import SUBMODULE

REPO = Path(__file__).resolve().parents[2]
CONFIG = TrainingConfig.from_json((REPO / S.CONFIG).read_text())
GAME = get_game(CONFIG)
MAX_PLAYERS = CONFIG.effective_max_players
ROOT = GAME.new_game(MAX_PLAYERS, MAX_PLAYERS, 0)
CHECKED_IN = REPO / "g1889/data/human_starts.npz"
# 2 players (left out), then 5, 3, and 4: short games, so the playouts stay quick.
SAMPLE = ("264038", "268833", "270133", "270907")
TOKEN_LAYOUT = 1


@pytest.fixture(scope="module")
def built() -> tuple[StartPositions, dict]:
    paths = [REPO / S.HUMAN_GAMES / f"{stem}.json" for stem in SAMPLE]
    if not all(p.exists() for p in paths):
        pytest.skip(f"needs the scraped games {SAMPLE} in {S.HUMAN_GAMES}")
    if not (SUBMODULE / "lib" / "engine.rb").exists():
        pytest.skip("needs submodules/18xx")
    return build(paths, CONFIG)


def _state(row: np.ndarray) -> GameState:
    return GAME.scratch_state(ROOT, row.copy())


def test_the_filters_hold(built):
    positions, report = built
    assert (report["games"], report["games_skipped"], len(positions.source_games)) == (3, 1, 3)
    assert report["drops"]["bank broken"] and report["drops"][f"{MIN_LEFT} or fewer decisions left"]
    assert len(positions.rows) > 1000 and set(positions.num_players.tolist()) == {3, 4, 5}
    assert (positions.decisions_left > MIN_LEFT).all()
    for i, row in enumerate(positions.rows):
        state = _state(row)
        assert not BANK.broken(state), i
        assert TURN.num_players(state) == positions.num_players[i]
        assert (PHASE.current(state), GAME.decision_phase(state), TURN.round(state)) == (
            positions.game_phase[i], positions.decision_phase[i], positions.round_type[i]), i
        assert DRIVER.forced_action_py(state) == -1, f"row {i}: not a decision the driver rests at"
        if GAME.decision_phase(state) == K.DPHASE_DIVIDEND:
            assert TURN.route_revenue(state) == max_route_revenue(state, TURN.operating_corp(state)), i


def test_decisions_count_through_each_game(built):
    """Within a game the index rises and index + left is the game's length, which
    begins at its first decision, the auction's first."""
    positions, report = built
    for g in range(len(positions.source_games)):
        mine = positions.source_game == g
        index, left = positions.decision_index[mine], positions.decisions_left[mine]
        assert index[0] == 0 and (np.diff(index) > 0).all()
        assert len(set((index + left).tolist())) == 1 and index[0] + left[0] == report["game_lengths"][g]


def test_every_kept_position_plays_out(built):
    """With other seeds than the build's."""
    positions, _ = built
    failed = [(i, why) for i, row in enumerate(positions.rows)
              if (why := playout(GAME, ROOT, row, playout_rng(1, positions.source_games[positions.source_game[i]],
                                                               positions.decision_index[i])))]
    assert not failed, failed[:5]


def test_a_saved_row_restores_to_the_state_built(built, tmp_path):
    """Through the file and ``scratch_state(new_game(...), row)``, as self-play builds
    its start: the same row and flags, legal and policy actions, and token and relation
    output. Extraction leaves the row as it is: its caches were fresh."""
    positions, _ = built
    path = tmp_path / "starts.npz"
    save_start_positions(path, positions)
    loaded = load_start_positions(path, MAX_PLAYERS)
    for f in fields(StartPositions):
        assert np.array_equal(getattr(loaded, f.name), getattr(positions, f.name)), f.name
    width = GAME.policy.max_action_size

    def observe(state: GameState) -> list:
        row = state._array.copy()
        legal, policy = np.empty(width, np.uint16), np.empty(width, np.uint16)
        tokens = np.empty((GAME.num_tokens(MAX_PLAYERS), GAME.token_dim(TOKEN_LAYOUT)), np.float32)
        records = np.zeros((GAME.max_relation_edges, RELATION_RECORD_WIDTH), np.uint8)
        GAME.fill_tokens(state, tokens, MAX_PLAYERS, layout_version=TOKEN_LAYOUT)
        count = GAME.fill_relations_sparse(state, records, MAX_PLAYERS)
        assert np.array_equal(state._array, row), "extraction changed the row"
        return [row, GAME.decision_phase(state), legal[:enumerate_legal_actions_py(state, legal)].tolist(),
                policy[:GAME.enumerate_policy_actions(state, policy)].tolist(), tokens, records[:count]]

    for i in range(len(positions.rows)):
        n = int(positions.num_players[i])
        saved = GameState(n, max_players=MAX_PLAYERS, private_sale_same_president=CONFIG.private_sale_same_president,
                          train_sale_same_president=CONFIG.train_sale_same_president,
                          auction_pass_payout_cap=CONFIG.auction_pass_payout_cap)
        saved.rebind(positions.rows[i].copy(), n, MAX_PLAYERS)
        restored = GAME.scratch_state(GAME.new_game(n, MAX_PLAYERS, i), loaded.rows[i].copy())
        for a, b in zip(observe(saved), observe(restored), strict=True):
            assert np.array_equal(a, b), f"row {i}"


def _synthetic(tmp_path: Path, **changes) -> Path:
    """A file of two opening rows, with ``changes`` to its fields."""
    rows = np.stack([GAME.new_game(n, MAX_PLAYERS, n)._array for n in (3, 4)])
    zeros = np.zeros(2, np.int32)
    positions = StartPositions(
        rows=rows, num_players=np.array([3, 4], np.int8), source_game=zeros, decision_index=zeros,
        decisions_left=zeros + 500, game_phase=zeros.astype(np.int8), decision_phase=zeros.astype(np.int8),
        round_type=zeros.astype(np.int8), source_games=np.array(["opening"]), phase_names=np.array(["2"]),
        max_players=MAX_PLAYERS, layout_hash=state_layout_hash(), engine_options="{}",
    )
    path = tmp_path / f"{len(list(tmp_path.iterdir()))}.npz"
    save_start_positions(path, replace(positions, **changes))
    return path


def test_the_loader_refuses_another_layout_or_row_size(tmp_path):
    assert len(load_start_positions(_synthetic(tmp_path), MAX_PLAYERS).rows) == 2
    with pytest.raises(ValueError, match="layout hash"):
        load_start_positions(_synthetic(tmp_path, layout_hash="0" * 16))
    wide = GAME.state_size(MAX_PLAYERS) + 4
    rows = np.zeros((2, wide), np.int16)
    rows[:, :-4] = load_start_positions(_synthetic(tmp_path)).rows
    with pytest.raises(ValueError, match="slots"):
        load_start_positions(_synthetic(tmp_path, rows=rows))
    with pytest.raises(ValueError, match="capacity"):
        load_start_positions(_synthetic(tmp_path), MAX_PLAYERS - 1)
    np.savez_compressed(tmp_path / "other.npz", rows=rows)
    with pytest.raises(ValueError, match="missing"):
        load_start_positions(tmp_path / "other.npz")


def test_the_layout_hash_follows_the_fields(monkeypatch):
    before = state_layout_hash()
    fields_ = S.get_corp_fields()
    monkeypatch.setattr(S, "get_corp_fields", lambda: fields_._replace(route_cache=fields_.route_cache + 1))
    assert state_layout_hash() != before


@pytest.mark.skipif(not CHECKED_IN.exists(), reason=f"no {CHECKED_IN.name} checked in")
def test_the_checked_in_file_loads_and_plays_out():
    positions = load_start_positions(CHECKED_IN, MAX_PLAYERS)
    assert set(positions.num_players.tolist()) <= set(CONFIG.iter_player_counts())
    assert (positions.decisions_left > MIN_LEFT).all()
    rng = np.random.default_rng(0)
    for i in rng.choice(len(positions.rows), 20, replace=False).tolist():
        assert not BANK.broken(_state(positions.rows[i]))
        assert playout(GAME, ROOT, positions.rows[i], rng) is None, i
