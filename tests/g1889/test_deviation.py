"""The deviation test (``train/deviation_1889.py``): its positions are where withholding,
and only withholding, buys the next train; its branches play out from them untouched;
and ``play`` resumes, with ``report`` pairing the branches by position."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER
from g1889.entities.bank import BANK
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState
from games import get_game
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train.checkpoint import save_checkpoint
from train.config import TrainingConfig
from train.deviation_1889 import (
    BRANCHES,
    _apply_choice,
    branch_state,
    continuation_config,
    continue_game,
    find_positions,
    finance,
    finance_target,
    main,
    paired,
    position_state,
    read_results,
    save_positions,
    summarize,
)

from tests.g1889.phases.test_random_playouts import _choose

MAX_PLAYERS = 5
CONFIG = TrainingConfig(
    game="1889", model_path="nn/transformer-1889.py", num_players=0, min_players=3, max_players=MAX_PLAYERS,
    d_model=32, num_heads=4, num_layers=1, num_simulations=2, search_batch_size=2, num_workers=0,
    zero_sum_values=True, private_sale_same_president=True, train_sale_same_president=True,
    auction_pass_payout_cap=1,
)
GAME = get_game(CONFIG)
# Random playouts with a finance position each, and one without.
GAMES = [(3, 23), (4, 8), (5, 1), (5, 6), (3, 0)]


def random_rows(num_players: int, seed: int) -> list[np.ndarray]:
    """The rows of a random playout's decisions, as a buffer holds them."""
    rng, state = np.random.default_rng(seed), GAME.new_game(num_players, MAX_PLAYERS, seed)
    rows = []
    while True:
        rows.append(state._array.copy())
        if DRIVER.apply_action(state, _choose(state, rng, True)) == STATUS_GAME_OVER:
            return rows


@pytest.fixture(scope="module")
def buffer() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """The games' rows in write order, after the tail of a game the ring overwrote; and
    each game's rows, as (start, end)."""
    games = [random_rows(*key) for key in GAMES]
    first = next(i for i, row in enumerate(games[0])
                 if finance_target(GameState.from_array(row, GAMES[0][0], max_players=MAX_PLAYERS)))
    tail = games[0][first:]  # starts at a finance position, but not at a game's start
    rows, counts, spans = list(tail), [GAMES[0][0]] * len(tail), []
    for (num_players, _), game_rows in zip(GAMES, games):
        spans.append((len(rows), len(rows) + len(game_rows)))
        rows += game_rows
        counts += [num_players] * len(game_rows)
    return np.stack(rows), np.array(counts, dtype=np.int8), spans


@pytest.fixture(scope="module")
def positions(buffer):
    states, counts, _ = buffer
    return find_positions(GAME, states, counts, MAX_PLAYERS, per_cell=10, rng=np.random.default_rng(0))


ROOT = GAME.new_game(MAX_PLAYERS, MAX_PLAYERS, 0)


def test_positions_are_dividends_where_only_withholding_buys_the_train(buffer, positions):
    _, _, spans = buffer
    assert min(p.buffer_row for p in positions) >= spans[0][0]  # none from the overwritten tail
    games = [next(g for g, (start, end) in enumerate(spans) if start <= p.buffer_row < end) for p in positions]
    assert len(positions) == len(set(games)) == 4  # one a game; none in the game without
    for p, g in zip(positions, games):
        assert p.decisions_left == spans[g][1] - p.buffer_row
        paid = position_state(GAME, ROOT, p)
        _apply_choice(GAME, paid, K.DPHASE_DIVIDEND, lambda info: info.type == A.ACTION_PAYOUT)
        if GAME.decision_phase(paid) == K.DPHASE_BUY_TRAIN and TURN.operating_corp(paid) == p.corp:
            actions = np.empty(GAME.policy.max_action_size, dtype=np.uint16)
            kinds = {A.decode_action_py(K.DPHASE_BUY_TRAIN, int(a)).type
                     for a in actions[:GAME.enumerate_policy_actions(paid, actions)]}
            assert A.ACTION_BUY_DEPOT_TRAIN not in kinds
        financed = position_state(GAME, ROOT, p)
        depot = TRAINS.depot(financed, p.train)
        finance(GAME, financed, p)  # asserts the corp holds one more of the train
        assert TRAINS.depot(financed, p.train) == depot - 1
        assert not BANK.broken(financed)  # the withhold left the game's end open


def test_a_branch_plays_to_the_end_and_leaves_its_start_alone(positions):
    torch.manual_seed(0)
    evaluator = NNEvaluator(GAME, create_model(CONFIG), torch.device("cpu"), num_players=MAX_PLAYERS,
                            terminal_rank_weight=CONFIG.terminal_blend, input_spec=get_model_input_spec(CONFIG))
    config = continuation_config(CONFIG)
    epoch_config = config.compute_epoch_config(0)
    pool = StatePool(GAME, 2 * (CONFIG.max_simulations + 1), GAME.state_size(MAX_PLAYERS))
    p = positions[0]
    position = position_state(GAME, ROOT, p)
    for branch in BRANCHES:
        start = branch_state(GAME, ROOT, p, branch)
        assert (start.private_sale_same_president, start.train_sale_same_president, start.auction_pass_payout_cap) \
            == (True, True, 1)
        before = start._array.copy()
        first = continue_game(GAME, evaluator, config, epoch_config, start, 7, pool, position)
        assert np.array_equal(start._array, before)
        assert continue_game(GAME, evaluator, config, epoch_config, start, 7, pool, position) == first
        assert first["started_at_start"]
        assert len(first["net_worths"]) == p.num_players and abs(sum(first["values"])) < 1e-3
        assert min(first["trains_after"].values()) >= 0
        if branch == "model":  # its first decision is the position's
            assert first["first_phase"] == "DPHASE_DIVIDEND"
        else:  # counted from the position, its own purchase is one more than from its start
            name = D.TRAIN_NAMES[p.train]
            from_start = continue_game(GAME, evaluator, config, epoch_config, start, 7, pool)
            assert first["trains_after"][name] == from_start["trains_after"][name] + 1
            assert first["ors_left"] >= from_start["ors_left"]


def test_play_resumes_and_the_report_pairs_branches_by_position(tmp_path, positions):
    torch.manual_seed(0)
    checkpoint = tmp_path / "checkpoint_epoch_0001.pt"
    save_checkpoint(checkpoint, 0, create_model(CONFIG), {}, CONFIG, {}, {})
    positions_path, results = tmp_path / "positions.json", tmp_path / "results.jsonl"
    save_positions(positions_path, positions[:2], {})
    args = ["play", "--checkpoint", str(checkpoint), "--positions", str(positions_path), "--results", str(results),
            "--continuations", "2", "--deep", "2", "--device", "cpu"]
    main(args)
    lines = read_results(results)
    assert sorted((r["position"], r["kind"], r["k"]) for r in lines) == sorted(
        [(p.id, "readout", 0) for p in positions[:2]]
        + [(p.id, b, k) for p in positions[:2] for b in BRANCHES for k in range(2)])
    with results.open("a") as out:
        out.write('{"position": 0, "ki')  # a write cut off by a kill
    main(args)
    assert read_results(results) == lines  # nothing left to run, and the cut line gone

    rows = paired(positions[:2], lines)
    for row, p in zip(rows, positions[:2]):
        def mean(branch):
            return np.mean([r["values"][p.president] for r in lines if r["position"] == p.id and r["kind"] == branch])
        assert row["d_value"] == pytest.approx(mean("finance") - mean("model"))
        readout = next(r for r in lines if r["position"] == p.id and r["kind"] == "readout")
        assert row["predicted_deep"] == pytest.approx(readout["deep"]["finance"] - readout["deep"]["root"])
    assert paired(positions[:2], [r for r in lines if not (r["position"] == positions[0].id and r["kind"] == "finance")]) \
        == rows[1:]
    one_seed = paired(positions[:1], [r for r in lines if not (r["kind"] == "finance" and r["k"] == 1)])[0]
    p = positions[0]
    def value(branch, k):
        return next(r for r in lines if r["position"] == p.id and r["kind"] == branch and r["k"] == k)["values"][p.president]
    assert one_seed["continuations"] == 1
    assert one_seed["d_value"] == pytest.approx(value("finance", 0) - value("model", 0))  # pairs by seed
    with pytest.raises(ValueError, match="another positions file"):
        paired([replace(p, buffer_row=p.buffer_row + 1)], lines)
    summary = summarize(rows, resamples=200)
    mean, low, high = summary["groups"]["all"]["d_value"]
    assert low <= mean <= high and mean == pytest.approx(np.mean([r["d_value"] for r in rows]))
    main(["report", "--positions", str(positions_path), "--results", str(results), "--json", str(tmp_path / "s.json")])
    assert json.loads((tmp_path / "s.json").read_text())["summary"]["groups"]["all"]["positions"] == 2
