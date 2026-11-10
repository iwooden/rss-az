"""Self-play games started from supplied positions (1889/v1-design.md §2.1).

- ``play_game`` from a row: it begins there, counts its own decisions, reads both
  temperature schedules at the game's real move numbers, and explores first.
- The epoch's schedule of starts: the share of each player count's games, the
  draw, and its dependence on the seed and epoch alone.
- A run through ``train.main`` with a resume: the started games' rows carry the
  flag, their stats have a TensorBoard group of their own, and a replayed epoch
  draws the starts it drew.
- A worker takes a started game's task through the eval server.
- What the loader and the config refuse.

The positions are rows late in random playouts, so the games from them are short.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER
from g1889.entities.bank import BANK
from games import RELATION_RECORD_WIDTH, get_game
from games.g1889_starts import StartPositions, engine_options, save_start_positions, state_layout_hash
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train import main as main_module
from train import self_play
from train.checkpoint import load_checkpoint
from train.config import EpochConfig, TrainingConfig
from train.eval_server import EvaluationServer, SharedEvalBuffers
from train.replay_buffer import FLAG_FROM_POSITION, ReplayBuffer
from train.self_play import (
    build_epoch_player_count_schedule, build_epoch_start_schedule, play_game, self_play_worker,
)
from train.tb_reader import read_tb_scalars

from tests.g1889.phases.test_random_playouts import _choose
from tests.g1889.test_relation_transport import _get_result
from tests.g1889.test_training_run import _smoke_config, _train

REPO = Path(__file__).resolve().parents[2]
CHECKED_IN = REPO / "g1889/data/human_starts.npz"
MAX_PLAYERS = 5
CONFIG = TrainingConfig(
    game="1889", model_path="nn/transformer-1889.py", num_players=0, min_players=3, max_players=MAX_PLAYERS,
    d_model=32, num_heads=4, num_layers=1, num_simulations=2, search_batch_size=2, num_workers=0,
    zero_sum_values=True, temp_initial=1.0, temp_final=0.25, policy_target_temp_initial=1.0,
    policy_target_temp_final=0.5,
)
GAME = get_game(CONFIG)
ROOT = GAME.new_game(MAX_PLAYERS, MAX_PLAYERS, 0)
LEFT = (90, 60, 30)  # the positions' decisions left in their playouts


def _playout(num_players: int, seed: int) -> list[np.ndarray]:
    """The rows of a random playout's decisions under the flags, then its final row."""
    rng, state = np.random.default_rng(seed), GAME.new_game(num_players, MAX_PLAYERS, seed)
    rows = []
    while True:
        rows.append(state._array.copy())
        if DRIVER.apply_action(state, _choose(state, rng, True)) == STATUS_GAME_OVER:
            return [*rows, state._array.copy()]


def _positions(player_counts: tuple[int, ...] = (3, 4, 5)) -> StartPositions:
    """Three positions a player count, late in a playout each, as the builder
    saves them: with the extraction prologue's caches fresh."""
    records = np.empty((GAME.max_relation_edges, RELATION_RECORD_WIDTH), dtype=np.uint8)
    rows, index = [], []
    for num_players in player_counts:
        playout = _playout(num_players, num_players)[:-1]
        for left in LEFT:
            at = next(i for i in range(len(playout) - left, 0, -1)
                      if not BANK.broken(GAME.scratch_state(ROOT, playout[i])))
            GAME.fill_relations_sparse(GAME.scratch_state(ROOT, playout[at]), records, MAX_PLAYERS)
            rows.append(playout[at])
            index.append(at)
    zeros = np.zeros(len(rows), np.int8)
    return StartPositions(
        rows=np.stack(rows), num_players=np.repeat(np.array(player_counts, np.int8), len(LEFT)),
        source_game=np.repeat(np.arange(len(player_counts), dtype=np.int16), len(LEFT)),
        decision_index=np.array(index, np.int32), decisions_left=np.tile(np.array(LEFT, np.int32), len(player_counts)),
        game_phase=zeros, decision_phase=zeros, round_type=zeros,
        source_games=np.array([f"playout {n}" for n in player_counts]), phase_names=np.array(D.PHASE_NAMES),
        max_players=MAX_PLAYERS, layout_hash=state_layout_hash(),
        engine_options=json.dumps(engine_options(CONFIG), sort_keys=True),
    )


@pytest.fixture(scope="module")
def positions() -> StartPositions:
    return _positions()


@pytest.fixture(scope="module")
def starts_path(positions, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("starts") / "starts.npz"
    save_start_positions(path, positions)
    return path


def _evaluator(config: TrainingConfig = CONFIG) -> NNEvaluator:
    torch.manual_seed(0)
    return NNEvaluator(GAME, create_model(config), torch.device("cpu"), num_players=MAX_PLAYERS,
                       terminal_rank_weight=config.terminal_blend, input_spec=get_model_input_spec(config))


@pytest.fixture(scope="module")
def evaluator() -> NNEvaluator:
    return _evaluator()


def _pool() -> StatePool:
    return StatePool(GAME, 2 * (CONFIG.max_simulations + 1), GAME.state_size(MAX_PLAYERS))


def _linear(move: int, window: tuple[int, int], initial: float, final: float) -> float:
    """A schedule's value: ``initial`` through the window's start, ``final`` from its end."""
    return float(np.interp(move, window, (initial, final)))


# =============================================================================
# PLAY_GAME FROM A ROW
# =============================================================================


@pytest.mark.parametrize("windows_from", ["the epoch", "the config"])
def test_a_started_game_plays_from_its_row_at_its_real_move_numbers(positions, evaluator, monkeypatch, windows_from):
    """A 4-player position whose move number lies inside both temperature windows,
    so the opening's temperature, the schedule's, and one counted from the
    start would all differ."""
    at = int(np.flatnonzero(positions.num_players == 4)[0])  # the one with the most decisions left
    row, start_move = positions.rows[at], int(positions.decision_index[at])
    before = row.copy()
    window, explore = (start_move - 10, start_move + 30), 6
    lists = {"temp_anneal_start": 0, "temp_anneal_end": 0, "policy_target_temp_anneal_start": 0,
             "policy_target_temp_anneal_end": 0,
             "policy_target_temp_anneal_starts": [1, window[0], 1], "policy_target_temp_anneal_ends": [2, window[1], 2]}
    if windows_from == "the epoch":
        config = replace(CONFIG, **lists, temp_anneal_starts=[1, 1, 1], temp_anneal_ends=[2, 2, 2],
                         start_explore_moves=explore)
        epoch_config = EpochConfig(c_puct=1.5, value_blend_alpha=1.0, temp_anneal_windows={4: window})
    else:
        config = replace(CONFIG, **lists, temp_anneal_starts=[1, window[0], 1], temp_anneal_ends=[2, window[1], 2],
                         start_explore_moves=explore)
        epoch_config = EpochConfig(c_puct=1.5, value_blend_alpha=1.0)
    temperatures: list[float] = []  # each decision's target temperature, then its action temperature
    scale = self_play.scale_visit_counts_by_temperature

    def spy(counts, temperature):
        temperatures.append(temperature)
        return scale(counts, temperature)

    monkeypatch.setattr(self_play, "scale_visit_counts_by_temperature", spy)
    record = play_game(evaluator, config, game_seed=5, rng=np.random.default_rng(5), state_pool=_pool(),
                       epoch_config=epoch_config, num_players=4, start_row=row, start_move=start_move)

    assert np.array_equal(record.states[0], row) and np.array_equal(row, before)
    assert record.flags == FLAG_FROM_POSITION and record.start_move == start_move
    assert len(record.net_worths) == 4  # it reached the game's end
    # It counts the decisions it played: a row and two temperatures each.
    decisions = record.num_examples
    assert record.total_moves == decisions == len(temperatures) // 2 and decisions > 2 * explore
    assert decisions < start_move  # so a total counted from the start move would show
    assert record.policy_target_top1_fraction == pytest.approx(record.policy_targets.max(axis=1).mean(), rel=1e-5)
    assert record.policy_target_entropy_mean == pytest.approx(
        -(record.policy_targets * np.log(np.maximum(record.policy_targets, 1e-30))).sum(axis=1).mean(), rel=1e-4)

    target, action = temperatures[0::2], temperatures[1::2]
    assert target == pytest.approx([_linear(start_move + i, window, 1.0, 0.5) for i in range(decisions)])
    assert action == pytest.approx([
        1.0 if i < explore else _linear(start_move + i, window, 1.0, 0.25) for i in range(decisions)
    ])
    # The explored moves are inside the window, where the schedule is below 1.
    assert target[0] < 1.0 and action[explore] < 1.0 and action[explore - 1] == 1.0
    # The policy metrics' stages are the target schedule's, at the real moves.
    counted = record.policy_metrics.scalars()
    stages = {stage: int(counted.get(f"policy/stage/{stage}/decision_count", 0)
                         + counted.get(f"policy/stage/{stage}/forced_count", 0))
              for stage in ("pre_anneal", "anneal", "post_anneal")}
    in_window = min(decisions, window[1] - start_move)
    assert stages == {"pre_anneal": 0, "anneal": in_window, "post_anneal": decisions - in_window}


def test_an_opening_game_is_the_same_with_the_settings_on(evaluator, starts_path):
    """The settings act through the rows the main loop hands out, and no game from
    the opening explores."""
    def play(config: TrainingConfig):
        return play_game(evaluator, config, game_seed=3, rng=np.random.default_rng(3), state_pool=_pool(),
                         epoch_config=config.compute_epoch_config(0), num_players=3)

    off = play(CONFIG)
    on = play(replace(CONFIG, start_positions_path=str(starts_path), human_start_fraction=0.25, start_explore_moves=7))
    assert off.flags == on.flags == 0 and off.start_move == on.start_move == 0
    assert off.total_moves == on.total_moves and off.net_worths == on.net_worths
    for name in ("states", "phase_ids", "legal_masks", "policy_targets", "value_targets", "policy_priorities"):
        assert np.array_equal(getattr(off, name), getattr(on, name)), name
    assert off.game_stats == on.game_stats


def test_play_game_refuses_a_row_it_cannot_start_from(positions):
    row = positions.rows[0]  # 3 players
    final = _playout(3, 3)[-1]

    def play(**kwargs):
        return play_game(None, CONFIG, game_seed=0, rng=np.random.default_rng(0), **kwargs)

    with pytest.raises(ValueError, match="3-player position, but the game is assigned 4"):
        play(num_players=4, start_row=row)
    with pytest.raises(ValueError, match="not a state row"):
        play(num_players=3, start_row=row[:-1])
    with pytest.raises(ValueError, match="the game is over"):
        play(num_players=3, start_row=final)
    with pytest.raises(ValueError, match="needs a start_row"):
        play(num_players=3, start_move=5)
    with pytest.raises(ValueError, match="start_move must be >= 0"):
        play(num_players=3, start_row=row, start_move=-1)


# =============================================================================
# THE EPOCH'S STARTS
# =============================================================================


def _schedule(games: int) -> list[int]:
    return build_epoch_player_count_schedule(replace(CONFIG, games_per_epoch=games))


def test_a_share_of_each_player_counts_games_starts_from_its_own_positions():
    # 5, 7, and 3 positions at 3, 4, and 5 players.
    counts = np.array([3] * 5 + [4] * 7 + [5] * 3, np.int8)
    schedule = _schedule(120)
    drawn: dict[int, list[int]] = {3: [], 4: [], 5: []}
    for epoch in range(200):
        starts = build_epoch_start_schedule(schedule, counts, 0.25, seed=42, epoch=epoch)
        assert len(starts) == len(schedule)
        for num_players in (3, 4, 5):
            mine = [start for start, n in zip(starts, schedule) if n == num_players]
            started = [j for j, start in enumerate(mine) if start >= 0]
            # A quarter of its 40 games, evenly spaced: the last of every four.
            assert started == list(range(3, 40, 4))
            assert all(counts[mine[j]] == num_players for j in started)
            drawn[num_players] += [mine[j] for j in started]
    # Uniform over the count's positions: 2,000 draws each.
    for num_players, draws in drawn.items():
        values, tallies = np.unique(draws, return_counts=True)
        assert values.tolist() == np.flatnonzero(counts == num_players).tolist()
        expected = len(draws) / len(values)
        assert np.abs(tallies - expected).max() < 4 * np.sqrt(expected)


@pytest.mark.parametrize(("games", "fraction", "started"), [
    (12, 0.25, 1), (3, 0.25, 0), (6, 0.25, 1), (30, 0.25, 3), (9, 1.0, 3), (9, 0.5, 2), (300, 0.1, 10),
])
def test_the_share_rounds_to_the_nearest_game_and_spreads(games, fraction, started):
    """Per player count: ``fraction`` of its games to the nearest, halves up, with
    the gaps between them within a game of each other."""
    counts = np.array([3, 4, 5, 3, 4, 5], np.int8)
    schedule = _schedule(games)
    starts = build_epoch_start_schedule(schedule, counts, fraction, seed=1, epoch=0)
    for num_players in (3, 4, 5):
        at = [j for j, start in enumerate(s for s, n in zip(starts, schedule) if n == num_players) if start >= 0]
        assert len(at) == started
        gaps = np.diff([-1, *at])
        assert len(at) == 0 or gaps.max() - gaps.min() <= 1


def test_the_starts_follow_the_seed_and_epoch_and_nothing_else():
    counts = np.repeat(np.array([3, 4, 5], np.int8), 50)
    schedule = _schedule(60)

    def draw(seed: int, epoch: int, fraction: float = 0.25) -> list[int]:
        return build_epoch_start_schedule(schedule, counts, fraction, seed, epoch)

    assert draw(42, 7) == draw(42, 7)  # a resume, or an epoch replayed after a shutdown
    assert draw(42, 7) != draw(42, 8) and draw(42, 7) != draw(43, 7)
    # Off: every game from the opening, with or without positions.
    assert draw(42, 7, 0.0) == [-1] * 60
    assert build_epoch_start_schedule(schedule, None, 0.25, 42, 7) == [-1] * 60
    # A player count without positions plays from the opening; the others' draws stay.
    without_5 = build_epoch_start_schedule(schedule, counts[:100], 0.25, 42, 7)
    assert [s for s, n in zip(without_5, schedule) if n == 5] == [-1] * 20
    assert [s for s, n in zip(without_5, schedule) if n != 5] == [s for s, n in zip(draw(42, 7), schedule) if n != 5]


# =============================================================================
# A RUN
# =============================================================================


def test_a_run_starts_a_quarter_of_its_games_and_a_replayed_epoch_draws_the_same(
    tmp_path, monkeypatch, capsys, positions, starts_path,
):
    """Twelve games an epoch, four a player count. Epoch 0 starts one of each
    four. Epoch 1, with every game started, is shut down after its first game
    and replayed on resume."""
    fields_ = {
        **_smoke_config(tmp_path), "games_per_epoch": 12, "training_steps_per_epoch": 4,
        "start_positions_path": str(starts_path), "human_start_fraction": 0.25, "start_explore_moves": 5,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(fields_))
    config = TrainingConfig.from_json(json.dumps(fields_))
    checkpoints = tmp_path / "checkpoints"
    device = torch.device("cpu")
    player_counts = build_epoch_player_count_schedule(config)

    games: list[tuple[int, int, np.ndarray | None, int]] = []  # game seed, players, start row, start move
    records = []
    real_play_game = main_module.play_game

    def play_game_spy(evaluator, config, game_seed, rng, **kwargs):
        games.append((game_seed, kwargs["num_players"], kwargs["start_row"], kwargs["start_move"]))
        records.append(real_play_game(evaluator, config, game_seed, rng, **kwargs))
        return records[-1]

    monkeypatch.setattr(main_module, "play_game", play_game_spy)

    def run(*args: str, stop: bool = False) -> str:
        event = threading.Event()
        if stop:
            event.set()
        with monkeypatch.context() as patch:
            patch.setattr(main_module, "_start_shutdown_listener", lambda: event)
            _train(patch, "--config", str(config_path), *args)
        return capsys.readouterr().out

    def expected_starts(epoch: int, fraction: float) -> list[tuple[np.ndarray | None, int]]:
        drawn = build_epoch_start_schedule(player_counts, positions.num_players, fraction, config.seed, epoch)
        return [(None, 0) if i < 0 else (positions.rows[i], int(positions.decision_index[i])) for i in drawn]

    def assert_starts(played: list, expected: list) -> None:
        assert len(played) == len(expected)
        for (_, _, row, move), (want_row, want_move) in zip(played, expected, strict=True):
            assert move == want_move and (row is None) == (want_row is None)
            assert row is None or np.array_equal(row, want_row)

    # --- Epoch 0: a quarter of each player count's games ---
    output = run()
    assert f"Start positions: {starts_path}, 9 rows (3p 3, 4p 3, 5p 3); 25% of each" in output
    assert [n for _, n, _, _ in games] == player_counts
    assert_starts(games, expected_starts(0, 0.25))
    started = [row is not None for _, _, row, _ in games]
    assert [sum(s for s, n in zip(started, player_counts) if n == count) for count in (3, 4, 5)] == [1, 1, 1]
    for (_, num_players, row, _), record in zip(games, records):
        assert record.flags == (FLAG_FROM_POSITION if row is not None else 0)
        assert row is None or GAME.num_players_from_row(row) == num_players
    # The seeds are those of a run without starts.
    master = np.random.default_rng(config.seed)
    seeds = []
    for _ in range(12):
        seeds.append(int(master.integers(0, 2**31)))
        master.integers(0, 2**63)
    assert [seed for seed, _, _, _ in games] == seeds

    # The saved buffer: each started game's rows carry the flag, from its start row on.
    buffer = ReplayBuffer(GAME, config.buffer_capacity, GAME.state_size(MAX_PLAYERS), MAX_PLAYERS,
                          min_players=3, max_players=MAX_PLAYERS)
    assert buffer.load(checkpoints / "replay_buffer") == sum(r.num_examples for r in records)
    rows = buffer.rows_oldest_first()
    for game_id, ((_, num_players, row, _), record) in enumerate(zip(games, records)):
        of_game = rows["game_ids"] == game_id
        assert int(of_game.sum()) == record.num_examples
        assert (rows["player_counts"][of_game] == num_players).all()
        assert (rows["flags"][of_game] == (FLAG_FROM_POSITION if row is not None else 0)).all()
        if row is not None:
            assert np.array_equal(rows["states"][of_game][0], row)
    flagged = int((rows["flags"] == FLAG_FROM_POSITION).sum())
    assert 0 < flagged < len(rows["flags"])

    # The temperature windows' lengths are the opening games'.
    def lengths(of: list, started_too: bool = False) -> dict[int, float]:
        return {n: float(np.mean([r.total_moves for r in of if r.num_players == n
                                  and (started_too or not r.flags)])) for n in (3, 4, 5)}

    first = load_checkpoint(checkpoints / "checkpoint_epoch_0001.pt", device)
    opening_lengths = first.get("game_lengths")
    assert opening_lengths is not None and opening_lengths == pytest.approx(lengths(records))
    assert opening_lengths != pytest.approx(lengths(records, started_too=True))
    epoch_0 = list(records)

    # --- Epoch 1, every game started: one game, a shutdown, and the replay ---
    config_path.write_text(json.dumps({**fields_, "human_start_fraction": 1.0}))
    del games[:], records[:]
    assert "Trained: no" in run("--resume", "latest", "--num-epochs", "2", stop=True)
    (interrupted,) = games
    output = run("--resume", "latest", "--num-epochs", "2")
    assert "Resumed from epoch 0" in output and "100% of each" in output
    replayed = games[1:]
    assert_starts(replayed, expected_starts(1, 1.0))
    assert_starts([interrupted], expected_starts(1, 1.0)[:1])
    assert replayed[0][0] != interrupted[0]  # the same start, under the next seeds
    assert len({row.tobytes() for _, _, row, _ in replayed if row is not None}) > 3  # not one position throughout
    # No game from the opening: each player count keeps its length.
    second = load_checkpoint(checkpoints / "checkpoint_epoch_0002.pt", device)
    assert second.get("game_lengths") == opening_lengths

    # --- TensorBoard: the started games apart ---
    scalars = read_tb_scalars(str(tmp_path / "runs"))
    opening = [r for r in epoch_0 if not r.flags]
    began = [r for r in epoch_0 if r.flags]
    assert scalars["self_play_aggregate/total_examples"] == [(1, len(rows["flags"]) - flagged)]
    assert scalars["self_play_started/total_examples"][0] == (1, flagged)
    assert scalars["self_play_aggregate/game_length_mean"] == [
        (1, pytest.approx(np.mean([r.total_moves for r in opening])))]
    assert scalars["self_play_started/game_length_mean"] == [
        (1, pytest.approx(np.mean([r.total_moves for r in began]))),
        (2, pytest.approx(np.mean([r.total_moves for r in records[1:]]))),
    ]
    assert scalars["self_play_started/games"] == [(1, 3), (2, 12)]
    assert scalars["self_play_started/start_move_mean"] == [
        (1, pytest.approx(np.mean([r.start_move for r in began]))),
        (2, pytest.approx(np.mean([r.start_move for r in records[1:]]))),
    ]
    for n in (3, 4, 5):
        assert scalars[f"self_play_started_{n}p/games"] == [(1, 1), (2, 4)]
        assert scalars[f"self_play_{n}p/game_length_mean"] == [(1, pytest.approx(opening_lengths[n]))]
        (mine,) = [r for r in began if r.num_players == n]
        assert scalars[f"self_play_started_{n}p/start_move_mean"][0] == (1, mine.start_move)
        assert scalars[f"self_play_started_{n}p/game_length_mean"][0] == (1, mine.total_moves)
    for group in ("started", "started_3p", "started_4p", "started_5p"):
        for name in ("end_bank", "end_bankruptcy", "trains_bought_2", "policy/all/decision_count", "net_worth_1st"):
            assert [step for step, _ in scalars[f"self_play_{group}/{name}"]] == [1, 2], (group, name)
    # Every row reached replay.
    assert scalars["buffer/size"][0] == (1, len(rows["flags"]))


def test_a_worker_plays_a_started_games_task(positions):
    """The multi-worker path: the task carries the row and its move number."""
    config = replace(CONFIG, num_workers=1, dirichlet_epsilon=0.0)
    ctx = mp.get_context("spawn")
    torch.manual_seed(0)
    shared_bufs = SharedEvalBuffers(GAME, 1, batch_size=config.search_batch_size, num_players=MAX_PLAYERS,
                                    input_spec=get_model_input_spec(config))
    shared_bufs.init_bitmap([(0, 1)], ctx)
    server = EvaluationServer(create_model(config), torch.device("cpu"), shared_bufs, mp_context=ctx, no_compile=True,
                              batch_shape_mode="bucketed", max_batch_size=8)
    tasks, results = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=self_play_worker, args=(tasks, results, config, shared_bufs, 0), daemon=True)
    at = int(np.flatnonzero(positions.num_players == 5)[-1])
    row, start_move = positions.rows[at], int(positions.decision_index[at])
    try:
        server.start()
        assert server.wait_ready(timeout=120.0)
        worker.start()
        tasks.put((0, 1, 5, config.compute_epoch_config(0), row, start_move))
        record = _get_result(results, server, [worker])
        tasks.put(None)
        worker.join(timeout=10.0)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        server.stop()
    assert record.flags == FLAG_FROM_POSITION and record.start_move == start_move and record.num_players == 5
    assert np.array_equal(record.states[0], row) and record.total_moves == record.num_examples > 0


# =============================================================================
# REFUSALS
# =============================================================================


def test_the_adapter_loads_the_runs_positions_and_refuses_others(positions, starts_path, tmp_path, capsys):
    starts = main_module._load_start_positions(
        replace(CONFIG, start_positions_path=str(starts_path), human_start_fraction=0.25), GAME)
    assert starts is not None
    assert np.array_equal(starts.rows, positions.rows) and starts.rows.dtype == np.int16
    assert np.array_equal(starts.num_players, positions.num_players)
    assert np.array_equal(starts.start_moves, positions.decision_index)
    assert "No start positions" not in capsys.readouterr().out

    def load(**changes):
        config = replace(CONFIG, start_positions_path=str(starts_path), human_start_fraction=0.25, **changes)
        return main_module._load_start_positions(config, get_game(config))

    # Positions filtered under other engine flags may be ones these cannot continue from.
    for flags in ({"auction_pass_payout_cap": 0}, {"train_sale_same_president": False},
                  {"private_sale_same_president": False}):
        with pytest.raises(ValueError, match="filtered under engine flags"):
            load(**flags)
    with pytest.raises(ValueError, match="capacity 5, not the run's 4"):
        load(max_players=4)
    # A file without a player count's positions says so.
    path = tmp_path / "no_5.npz"
    save_start_positions(path, _positions((3, 4)))
    config = replace(CONFIG, start_positions_path=str(path), human_start_fraction=0.25)
    starts = main_module._load_start_positions(config, GAME)
    assert starts is not None and set(starts.num_players.tolist()) == {3, 4}
    assert "No start positions at 5 players: its games all start from the opening" in capsys.readouterr().out
    # Off: no path, or no share of the games.
    assert main_module._load_start_positions(CONFIG, GAME) is None
    assert main_module._load_start_positions(replace(config, human_start_fraction=0.0), GAME) is None


def test_a_game_without_start_positions_refuses_a_path(starts_path):
    config = TrainingConfig(start_positions_path=str(starts_path))  # RSS
    game = get_game(config)
    assert game.load_start_positions is None
    with pytest.raises(ValueError, match="game 'rss' has no start positions"):
        main_module._load_start_positions(config, game)
    with pytest.raises(ValueError, match="game 'rss' has no start positions"):
        main_module._load_start_positions(replace(config, human_start_fraction=0.25), game)


@pytest.mark.parametrize(("changes", "message"), [
    ({"human_start_fraction": 0.25}, "requires start_positions_path"),
    ({"human_start_fraction": 1.5, "start_positions_path": "x.npz"}, r"must be in \[0, 1\]"),
    ({"human_start_fraction": -0.1}, r"must be in \[0, 1\]"),
    ({"start_explore_moves": -1}, "must be an integer >= 0"),
    ({"start_explore_moves": 2.5}, "must be an integer >= 0"),
    ({"start_positions_path": 3}, "must be a path string"),
])
def test_the_settings_are_validated(changes, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**changes)
    assert (TrainingConfig().start_positions_path, TrainingConfig().human_start_fraction,
            TrainingConfig().start_explore_moves) == (None, 0.0, 20)


@pytest.mark.skipif(not CHECKED_IN.exists(), reason=f"no {CHECKED_IN.name} checked in")
def test_the_checked_in_file_loads_for_v0s_config_and_a_game_plays_from_it(evaluator):
    """The human positions, through the adapter under the run's flags: one of each
    player count plays to the end from its decision index."""
    v0 = TrainingConfig.from_json((REPO / "train_configs/1889-v0.json").read_text())
    load = get_game(v0).load_start_positions
    assert load is not None
    starts = load(str(CHECKED_IN))
    assert starts.rows.shape[1] == GAME.state_size(v0.effective_max_players)
    assert set(starts.num_players.tolist()) == set(v0.iter_player_counts())
    assert engine_options(v0) == engine_options(CONFIG)  # so CONFIG's model may play them
    rng = np.random.default_rng(0)
    for num_players in (3, 4, 5):
        at = int(rng.choice(np.flatnonzero(starts.num_players == num_players)))
        record = play_game(evaluator, CONFIG, game_seed=at, rng=np.random.default_rng(at), state_pool=_pool(),
                           num_players=num_players, start_row=starts.rows[at], start_move=int(starts.start_moves[at]))
        assert record.flags == FLAG_FROM_POSITION and np.array_equal(record.states[0], starts.rows[at])
        assert record.total_moves == record.num_examples > 0 and len(record.net_worths) == num_players
