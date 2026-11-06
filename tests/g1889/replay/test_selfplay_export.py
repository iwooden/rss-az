"""The self-play log (``utils_18xx/g1889_selfplay_log.py``) round trip through Ruby.

Nothing else proves that what the engine offers Ruby accepts: the replays prove the
reverse, that every human action is legal in the engine. So random games at the
self-play flags go through the converter, whose composer validates each action as
Ruby's UI would, and ``harness.check_export``, which replays the log against Ruby's
state at every record. These seeds are short games that reach the rare actions.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from typing import Any

import numpy as np
import pytest
import torch

import g1889.constants as K
from g1889.driver import DRIVER, STATUS_GAME_OVER
from g1889.entities.turn import TURN
from games import get_game
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train.config import TrainingConfig
from train.self_play import play_game
from utils_18xx import g1889_selfplay_log as selfplay_log
from utils_18xx.g1889_selfplay_log import PlayedGame, _view, compose, game_intents, play, redrive

from tests.g1889.phases.test_random_playouts import _choose
from tests.g1889.replay.extracts import load_game_extract
from tests.g1889.replay.harness import check_export, replay_extract

MAX_PLAYERS = 5
# The seeds' games reach every kind of action between them (test_the_games_reach_the_rare_actions).
GAMES = [(3, 7), (3, 146), (4, 163), (4, 190), (5, 184), (5, 219)]
CONFIG = TrainingConfig(
    game="1889", model_path="nn/transformer-1889.py", num_players=0, min_players=3, max_players=MAX_PLAYERS,
    d_model=32, num_heads=4, num_layers=1, num_simulations=2, search_batch_size=2, num_workers=0,
    zero_sum_values=True, private_sale_same_president=True, train_sale_same_president=True,
    auction_pass_payout_cap=1,
)
GAME = get_game(CONFIG)
ACTION_PATTERN = re.compile(r"^[A-Z_]+ v=[+-]\d\.\d\d( \| \*?\S+( \S+)* p(\.\d\d|1\.00) n(\.\d\d|1\.00))+$")


def random_game(num_players: int, seed: int) -> PlayedGame:
    """A random playout at the self-play flags, with made-up evaluations of each returned decision."""
    rng, eval_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    state = GAME.new_game(num_players, MAX_PLAYERS, seed)
    scratch = np.empty(GAME.policy.max_action_size, dtype=np.uint16)
    actions: list[int] = []
    evals: list[dict[str, Any]] = []
    while True:
        legal = [int(a) for a in scratch[:GAME.enumerate_policy_actions(state, scratch)]]
        action = _choose(state, rng, True)
        priors = eval_rng.dirichlet(np.ones(len(legal)))
        actions.append(int(action))
        evals.append({"value": float(eval_rng.uniform(-1, 1)), "legal": legal, "priors": [float(p) for p in priors],
                      "visits": [int(v) for v in eval_rng.multinomial(8, priors)]})
        if DRIVER.apply_action(state, action) == STATUS_GAME_OVER:
            return PlayedGame(CONFIG.to_json(), num_players, seed, actions, evals, [int(v) for v in state._array],
                              GAME.final_scores(state), {})


@pytest.fixture(scope="module")
def games() -> dict[tuple[int, int], tuple[PlayedGame, dict[str, Any], list[dict[str, Any]]]]:
    return {key: (played := random_game(*key), *compose(played, created_at=1)) for key in GAMES}


@pytest.mark.parametrize("key", GAMES, ids=[f"{n}p-{seed}" for n, seed in GAMES])
def test_a_random_game_round_trips_through_ruby(games, key):
    played, log, notes = games[key]
    assert [note for note in notes if note["kind"] != "moved_exchange"] == []
    assert check_export(f"export-{key[0]}p-{key[1]}", log, played.net_worths, redrive(played)[1]) == []


def test_the_games_reach_the_rare_actions(games):
    kinds = Counter((a["type"], a["entity_type"]) for _, log, _ in games.values() for a in log["actions"])
    for kind in [("bankrupt", "corporation"), ("discard_train", "corporation"), ("buy_shares", "company"),
                 ("lay_tile", "company"), ("pass", "company"), ("buy_company", "corporation"),
                 ("place_token", "corporation")]:
        assert kinds[kind], kind
    assert any(note["kind"] == "moved_exchange" for _, _, notes in games.values() for note in notes)
    assert any(a.get("exchange") for _, log, _ in games.values() for a in log["actions"])
    assert {log["game_end_reason"] for _, log, _ in games.values()} == {"bank", "bankrupt"}


def test_ruby_first_player_is_the_engines(games):
    """Ruby's first player is the first of the log's players; names and ids go by the engine's seats."""
    for played, log, _ in games.values():
        events, _ = redrive(played)
        view = _view(events[0].row, played.num_players, MAX_PLAYERS)
        assert log["players"][0]["id"] == TURN.priority_player(view) + 1 == TURN.active_player(view) + 1
        assert sorted(p["id"] for p in log["players"]) == list(range(1, played.num_players + 1))
        assert all(p["name"] == f"Seat {p['id']}" for p in log["players"])


def test_a_port_tile_on_a_turn_ruby_skips_replays(games):
    """The engine offers MF's owner a turn Ruby skips, and Ruby takes the tile while it waits on someone else."""
    _, log, _ = games[(3, 7)]
    progress = replay_extract("port", load_game_extract("export-3p-7", log))
    assert progress.stop is None and progress.private_moves["backed up to a skipped turn"]


def test_chat_lines_come_from_the_acting_seat_before_their_move(games):
    for played, log, _ in games.values():
        lines = [a for a in log["actions"] if a["type"] == "message"]
        assert len(lines) == len(played.actions)
        assert all(ACTION_PATTERN.match(a["message"]) for a in lines), [a["message"] for a in lines][:5]
        events, _ = redrive(played)
        searched = [e for e in events if e.searched is not None]
        for line, event in zip(lines, searched, strict=True):
            view = _view(event.row, played.num_players, MAX_PLAYERS)
            assert line["entity"] == TURN.active_player(view) + 1
            assert line["message"].startswith(K.DecisionPhase(event.phase).name.removeprefix("DPHASE_"))
            assert line["message"].count("*") == 1


def test_no_eval_leaves_the_same_game(games):
    played, log, _ = games[(3, 7)]
    plain, _ = compose(played, include_eval=False, created_at=1)
    without = [{k: v for k, v in a.items() if k != "id"} for a in log["actions"] if a["type"] != "message"]
    assert [{k: v for k, v in a.items() if k != "id"} for a in plain["actions"]] == without


def test_the_composer_checks_what_a_load_does_not():
    """The composer validates as Ruby's UI does, so it refuses a lay that ``Game.load`` would take:
    the first corp's first lay, turned until its track no longer connects."""
    played = random_game(3, 7)
    events, _ = redrive(played)
    intents = game_intents(played, events, MAX_PLAYERS)
    lay = next(intent for intent in intents if intent["type"] == "lay_tile" and intent["entity_type"] == "corporation")
    doc = {"id": "hs_x_1", "seed": 1, "description": "", "created_at": 1, "created_date": "2026-01-01",
           "players": [{"id": i + 1, "name": f"Seat {i + 1}"} for i in range(3)], "intents": intents}
    refusals = []
    for turn in range(1, 6):
        lay["rotation"] = (lay["rotation"] + 1) % 6
        result = subprocess.run(["ruby", str(selfplay_log.COMPOSER)], input=json.dumps(doc), capture_output=True,
                                encoding="utf-8", check=False)
        if result.returncode != 0 and "not legally rotated" in json.loads(result.stderr)["message"]:
            refusals.append(turn)
    assert refusals


def test_recording_a_game_leaves_it_as_play_game_plays_it(tmp_path, monkeypatch):
    """The tool's game with a real model is ``play_game``'s, decision for decision;
    it round-trips with its real chat lines, and converts again from its saved file."""
    torch.manual_seed(0)
    model = create_model(CONFIG)
    device = torch.device("cpu")
    played = play(model, CONFIG, num_players=3, seed=11, epoch=0, simulations=None, device=device)

    evaluator = NNEvaluator(GAME, model, device, num_players=MAX_PLAYERS, terminal_rank_weight=CONFIG.terminal_blend,
                            input_spec=get_model_input_spec(CONFIG))
    record = play_game(evaluator, CONFIG, game_seed=11, rng=np.random.default_rng(11),
                       state_pool=StatePool(GAME, 2 * (CONFIG.max_simulations + 1), GAME.state_size(MAX_PLAYERS)),
                       epoch_config=CONFIG.compute_epoch_config(0), num_players=3)
    events, terminal = redrive(played)
    assert played.net_worths == record.net_worths and len(played.actions) == record.total_moves
    assert np.array_equal(np.stack([e.row for e in events if e.searched is not None]), record.states)

    saved = tmp_path / "game.played.json"
    played.save(saved)
    output = tmp_path / "game.json"
    monkeypatch.setattr(sys, "argv", ["g1889_selfplay_log", "--played", str(saved), "--output", str(output)])
    selfplay_log.main()
    log = json.loads(output.read_text())
    assert sum(a["type"] == "message" for a in log["actions"]) == len(played.actions)
    assert check_export("export-real-3p-11", log, played.net_worths, terminal) == []
