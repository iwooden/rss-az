"""The 1889 self-play stats (1889/model-design.md §6.6, §8 row 8.3).

On random playouts at 3-5 players, driven as self-play drives the collector
(``observe`` at each returned decision, ``finish`` at the end), its numbers equal
what the driver's history and the terminal state say by other routes:

- the end reasons are the engine's, with the bankrupt player or broken bank to show;
- the ORs are the history's OR_START steps;
- the trains bought are the history's depot purchases, forced ones included;
  each rank's in play, and the game's last phase, follow from them;
- the floated corps are those that ever appear in an OR's order, and the final
  cash is their treasuries';
- the track passes are the returned TRACK decisions that pass, and the withholds
  the returned DIVIDEND decisions that withhold, by the game phase they came in.

Through ``play_game`` with a real model, the stats reach the record and
``train/main.py``'s TensorBoard scalars for each reporting group.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER
from g1889.entities.bank import BANK
from g1889.entities.corp import CORPS
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState
from games import STATUS_GAME_OVER, STATUS_OK, get_game
from games.g1889_selfplay import G1889SelfPlayStats
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train import self_play as self_play_module
from train.config import TrainingConfig
from train.main import _build_epoch_self_play_scalars, _SelfPlayMetricAccumulator
from train.self_play import GameRecord, play_game

from tests.g1889.phases.test_random_playouts import _choose

MAX_PLAYERS = 5
PLAYER_COUNTS = (3, 4, 5)
SEEDS = range(20)
MAX_DECISIONS = 3000

CONFIG = TrainingConfig(
    game="1889",
    model_path="nn/transformer-1889.py",
    num_players=0,
    min_players=3,
    max_players=MAX_PLAYERS,
    d_model=32,
    num_heads=4,
    num_layers=1,
    num_simulations=2,
    search_batch_size=2,
    num_workers=0,
    zero_sum_values=True,
)
GAME = get_game(CONFIG)
PHASE_NAMES = D.PHASE_NAMES
TRAIN_NAMES = D.TRAIN_NAMES


class Playout(NamedTuple):
    """A random game's terminal state, its collector's report, and what the history says."""

    state: GameState
    stats: dict[str, tuple[float, float]]
    or_starts: int
    bought: Counter[int]
    corps_in_an_order: set[int]
    track_decisions: int
    track_passes: int
    dividends: Counter[int]
    withholds: Counter[int]


def _play(num_players: int, seed: int) -> Playout:
    rng = np.random.default_rng(seed)
    state = GAME.new_game(num_players, MAX_PLAYERS, seed)
    view = GameState(num_players, max_players=MAX_PLAYERS)
    collector = GAME.new_selfplay_stats(num_players)
    scratch = np.empty(GAME.policy.max_action_size, dtype=np.uint16)
    or_starts = track_decisions = track_passes = 0
    bought: Counter[int] = Counter()
    corps_in_an_order: set[int] = set()
    dividends: Counter[int] = Counter()
    withholds: Counter[int] = Counter()

    for _ in range(MAX_DECISIONS):
        phase = GAME.decision_phase(state)
        n_legal = GAME.enumerate_policy_actions(state, scratch)
        legal = scratch[:n_legal].copy()
        action = _choose(state, rng, True)
        track_decisions += phase == K.DPHASE_TRACK
        track_passes += phase == K.DPHASE_TRACK and action == K.TRACK_PASS
        if phase == K.DPHASE_DIVIDEND:
            dividends[PHASE.current(state)] += 1
            withholds[PHASE.current(state)] += action == K.DIVIDEND_WITHHOLD
        # Priors and visits only pass through: the collector's report ignores them.
        collector.observe(state, legal, int(np.flatnonzero(legal == action)[0]), np.ones(n_legal) / n_legal,
                          np.ones(n_legal))
        history: list[tuple[np.ndarray, int, int]] = []
        status = DRIVER.apply_action(state, action, history)
        assert status in (STATUS_OK, STATUS_GAME_OVER)
        for row, row_phase, row_action in history:
            view.rebind(row, num_players, MAX_PLAYERS)
            if TURN.round(view) == K.ROUND_OPERATING:
                corps_in_an_order |= {TURN.or_order(view, i) for i in range(D.NUM_CORPS)} - {-1}
            if row_phase < 0:
                or_starts += row_action == K.STEP_OR_START
            elif row_phase == K.DPHASE_SELECT_TRAIN and TURN.pending_train_source(view) == K.TRAIN_SOURCE_DEPOT:
                info = A.decode_action_py(row_phase, row_action)
                bought[D.TRAIN_D if info.type == A.ACTION_TRADE_IN else info.train_type] += 1
        if status == STATUS_GAME_OVER:
            return Playout(state, collector.finish(state), or_starts, bought, corps_in_an_order,
                           track_decisions, track_passes, dividends, withholds)
    pytest.fail(f"{num_players}p seed {seed}: no game end within {MAX_DECISIONS} decisions")


@pytest.fixture(scope="module")
def playouts() -> dict[tuple[int, int], Playout]:
    return {(n, seed): _play(n, seed) for n in PLAYER_COUNTS for seed in SEEDS}


def _means(stats: dict[str, tuple[float, float]], prefix: str, names: tuple[str, ...]) -> list[float]:
    """The per-game scalars ``prefix + name``, each with denominator 1."""
    assert all(stats[f"{prefix}{name}"][1] == 1 for name in names)
    return [stats[f"{prefix}{name}"][0] for name in names]


def test_the_playouts_reach_what_the_stats_report(playouts):
    """Both ends, every last phase past the early ones, and every train type,
    so the checks below are not vacuous."""
    reasons = Counter(TURN.end_reason(p.state) for p in playouts.values())
    assert reasons[K.END_REASON_BANK] >= 10 and reasons[K.END_REASON_BANKRUPT] >= 10
    phases = Counter(PHASE.current(p.state) for p in playouts.values())
    assert all(phases[phase] for phase in (2, 3, 4, 5))
    assert all(sum(p.bought[t] for p in playouts.values()) for t in range(len(TRAIN_NAMES)))


def test_end_reasons_are_the_engines(playouts):
    for (n, seed), p in playouts.items():
        reason = TURN.end_reason(p.state)
        bankrupt = [i for i in range(n) if PLAYERS[i].bankrupt(p.state)]
        assert (p.stats["end_bank"][0], p.stats["end_bankruptcy"][0]) == (
            float(reason == K.END_REASON_BANK), float(reason == K.END_REASON_BANKRUPT),
        ), f"{n}p seed {seed}"
        assert p.stats["end_bank"][0] + p.stats["end_bankruptcy"][0] == 1
        if p.stats["end_bankruptcy"][0]:
            assert len(bankrupt) == 1
        else:
            assert not bankrupt and BANK.broken(p.state)


def test_ors_played_are_the_historys(playouts):
    for (n, seed), p in playouts.items():
        assert p.stats["ors_played"] == (p.or_starts, 1), f"{n}p seed {seed}"


def test_trains_bought_are_the_historys_and_fit_the_state(playouts):
    for (n, seed), p in playouts.items():
        msg = f"{n}p seed {seed}"
        bought = _means(p.stats, "trains_bought_", TRAIN_NAMES)
        assert bought == [p.bought[t] for t in range(len(TRAIN_NAMES))], msg
        # Each rank's last copy sells before the next rank's first; D waits only for phase 6.
        for t in range(D.TRAIN_D - 1):
            assert not bought[t + 1] or bought[t] == D.TRAIN_COUNT[t], msg
        # What was bought is in play, unless its rank has rusted.
        for t in range(len(TRAIN_NAMES)):
            in_play = TRAINS.pool(p.state, t) + sum(TRAINS.corp_trains(p.state, c, t) for c in range(D.NUM_CORPS))
            rusts_on = D.TRAIN_RUSTS_ON[t]
            assert in_play == (0 if rusts_on >= 0 and bought[rusts_on] else bought[t]), msg


def test_last_phase_is_one_hot_and_follows_from_the_trains_bought(playouts):
    for (n, seed), p in playouts.items():
        one_hot = _means(p.stats, "last_phase_", PHASE_NAMES)
        assert sum(one_hot) == 1 and one_hot[PHASE.current(p.state)] == 1, f"{n}p seed {seed}"
        # A phase starts with its train's first purchase, phase 2 with the game.
        bought = _means(p.stats, "trains_bought_", TRAIN_NAMES)
        assert one_hot.index(1) == max(t for t in range(len(TRAIN_NAMES)) if t == 0 or bought[t]), f"{n}p seed {seed}"


def test_corps_floated_are_those_that_entered_an_or(playouts):
    """A corp floats in an SR, and an OR follows every SR, so each floated corp operated or was about to."""
    for (n, seed), p in playouts.items():
        floated = {c for c in range(D.NUM_CORPS) if CORPS[c].floated(p.state)}
        assert p.corps_in_an_order == floated, f"{n}p seed {seed}"
        assert p.stats["corps_floated"] == (len(floated), 1), f"{n}p seed {seed}"


def test_track_pass_share_counts_the_returned_decisions(playouts):
    for (n, seed), p in playouts.items():
        assert p.stats["track_pass_share"] == (p.track_passes, p.track_decisions), f"{n}p seed {seed}"
    assert sum(p.track_decisions for p in playouts.values()) > 0


def _withhold_shares_by_phase(stats: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {name: stats[f"withhold_share_{name}"] for name in PHASE_NAMES if f"withhold_share_{name}" in stats}


def test_withhold_shares_count_the_returned_decisions_by_game_phase(playouts):
    for (n, seed), p in playouts.items():
        msg = f"{n}p seed {seed}"
        assert p.stats["withhold_share"] == (sum(p.withholds.values()), sum(p.dividends.values())), msg
        assert _withhold_shares_by_phase(p.stats) == {
            PHASE_NAMES[phase]: (p.withholds[phase], count) for phase, count in p.dividends.items()
        }, msg
    # Both choices, in more than one phase, and a phase some game never reached is absent.
    withheld = sum(sum(p.withholds.values()) for p in playouts.values())
    assert 0 < withheld < sum(sum(p.dividends.values()) for p in playouts.values())
    assert len({phase for p in playouts.values() for phase in p.dividends}) > 1
    assert any(len(_withhold_shares_by_phase(p.stats)) < len(PHASE_NAMES) for p in playouts.values())


def test_corp_cash_end_is_the_floated_corps_treasuries(playouts):
    for (n, seed), p in playouts.items():
        floated = [c for c in range(D.NUM_CORPS) if CORPS[c].floated(p.state)]
        assert p.stats["corp_cash_end"] == (sum(CORPS[c].cash(p.state) for c in floated), len(floated)), (
            f"{n}p seed {seed}"
        )


# =============================================================================
# THROUGH PLAY_GAME AND TENSORBOARD'S SCALARS
# =============================================================================


class SelfPlayGame(NamedTuple):
    record: GameRecord
    terminal: GameState  # a copy, from the collector's finish


@pytest.fixture(scope="module")
def games() -> dict[int, SelfPlayGame]:
    """One small-model game at each player count."""
    torch.manual_seed(0)
    model = create_model(CONFIG)
    evaluator = NNEvaluator(
        GAME, model, torch.device("cpu"), num_players=MAX_PLAYERS,
        terminal_rank_weight=CONFIG.terminal_blend, input_spec=get_model_input_spec(CONFIG),
    )
    pool = StatePool(GAME, 2 * (CONFIG.max_simulations + 1), GAME.state_size(MAX_PLAYERS))
    terminals: list[GameState] = []

    class SpiedStats(G1889SelfPlayStats):
        def finish(self, state: GameState) -> dict[str, tuple[float, float]]:
            terminals.append(_row_view(state._array.copy(), TURN.num_players(state)))
            return super().finish(state)

    spied = replace(GAME, new_selfplay_stats=SpiedStats)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(self_play_module, "get_game", lambda config: spied)
        return {
            n: SelfPlayGame(
                play_game(
                    evaluator, CONFIG, game_seed=n, rng=np.random.default_rng(n), state_pool=pool,
                    epoch_config=CONFIG.compute_epoch_config(0), num_players=n,
                ),
                terminals[-1],
            )
            for n in PLAYER_COUNTS
        }


def _row_view(row: np.ndarray, num_players: int) -> GameState:
    return GameState.from_array(row, num_players, max_players=MAX_PLAYERS)


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_a_self_play_record_reports_its_games_stats(games, num_players):
    record, terminal = games[num_players]
    stats = record.game_stats
    assert A.is_terminal_py(terminal)
    assert stats["end_bank"][0] == (TURN.end_reason(terminal) == K.END_REASON_BANK)
    assert stats["end_bankruptcy"][0] == (TURN.end_reason(terminal) == K.END_REASON_BANKRUPT)
    assert _means(stats, "last_phase_", PHASE_NAMES)[PHASE.current(terminal)] == 1

    # Every searched decision is a replay row, so the rows bound what the stats say.
    assert stats["track_pass_share"][1] == np.count_nonzero(record.phase_ids == K.DPHASE_TRACK)
    assert 0 <= stats["track_pass_share"][0] <= stats["track_pass_share"][1]
    views = [_row_view(row, num_players) for row in record.states]
    dividend_rows = Counter(
        PHASE.current(view)
        for view, phase in zip(views, record.phase_ids, strict=True)
        if phase == K.DPHASE_DIVIDEND
    )
    assert stats["withhold_share"][1] == sum(dividend_rows.values())
    assert {name: share[1] for name, share in _withhold_shares_by_phase(stats).items()} == {
        PHASE_NAMES[phase]: count for phase, count in dividend_rows.items()
    }
    assert stats["corp_cash_end"][1] == stats["corps_floated"][0]
    ors_seen = {(TURN.turn(v), TURN.or_num(v)) for v in views if TURN.round(v) == K.ROUND_OPERATING}
    assert stats["ors_played"][0] >= len(ors_seen)
    last_row = views[-1]
    bought = _means(stats, "trains_bought_", TRAIN_NAMES)
    for t in range(D.TRAIN_D):
        assert bought[t] >= D.TRAIN_COUNT[t] - TRAINS.depot(last_row, t)


def test_tensorboard_scalars_carry_the_stats_for_each_group(games):
    metrics = _SelfPlayMetricAccumulator()
    records: list[Any] = [games[n].record for n in PLAYER_COUNTS]
    for record in records:
        metrics.add_record(record)
    scalars = _build_epoch_self_play_scalars(metrics)
    always = {
        "end_bank", "end_bankruptcy", "ors_played", "track_pass_share", "withhold_share", "corps_floated",
        "corp_cash_end",
        *(f"last_phase_{name}" for name in PHASE_NAMES),
        *(f"trains_bought_{name}" for name in TRAIN_NAMES),
    }
    by_phase = {f"withhold_share_{name}" for name in PHASE_NAMES}
    for record in records:
        assert always <= set(record.game_stats) <= always | by_phase
    names = {name for record in records for name in record.game_stats}
    for name in names:
        numerator = sum(record.game_stats[name][0] for record in records if name in record.game_stats)
        denominator = sum(record.game_stats[name][1] for record in records if name in record.game_stats)
        assert scalars[f"self_play_aggregate/{name}"] == pytest.approx(numerator / max(denominator, 1))
        for record in records:
            tag = f"self_play_{record.num_players}p/{name}"
            if name in record.game_stats:
                own = record.game_stats[name]
                assert scalars[tag] == pytest.approx(own[0] / max(own[1], 1))
            else:
                assert tag not in scalars
