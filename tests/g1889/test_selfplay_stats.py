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

The stats added for v1 (1889/v1-design.md §4) equal what every step of the
driver's history says, forced and automated ones included (``_Truth``):

- the auction premium is the cash the players paid in steps that paid no pass
  payout, less the privates' face; SR1's cash is the players' as the auction ends;
- the pars and the tokens off a home are the terminal state's, the tokens counted
  on the map; the treasuries at float are the corps' as they float;
- a stranded SR is one whose SR_END leaves a corp parred but not floated, and
  SR2's sales are its SELL_AMOUNT decisions' shares;
- a phase starts in the OR the history's OR_START steps have reached when it
  starts, and MF's tile goes down in the round of the step that sets its flag.

The richest treasury when the 5s go on sale is the collector's one stat taken
at the next returned decision rather than at the event, and it matches that
definition. The two part when automated steps between them move cash: on the
v1 review's games, in 11 of 289 random playouts by ¥5-40 and 1 of 40 model
games by ¥25 (an OR's start paying privates' revenue to corps), and never in
100 human games. A game observed from a later decision reports only what
followed it.

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
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, PrivateFlag
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


def _players_cash(state: GameState) -> int:
    return sum(PLAYERS[i].cash(state) for i in range(TURN.num_players(state)))


def _tokens_off_home(state: GameState) -> int:
    """Station tokens on the map other than each corp's on its home hex."""
    on_map = sum(MAP.tokens(state, h).bit_count() for h in range(D.NUM_HEXES))
    return on_map - sum(MAP.tokens(state, D.CORP_HOME[c]) >> c & 1 for c in range(D.NUM_CORPS))


def _mf_laid(state: GameState) -> bool:
    return bool(PRIVATES[D.PRIVATE_MF].flags(state) & PrivateFlag.MF_PORT_USED)


def _richest(state: GameState) -> int:
    return max(CORPS[c].cash(state) for c in range(D.NUM_CORPS) if CORPS[c].floated(state))


class _Truth:
    """The v1 stats from every driver step after the first observed decision,
    forced and automated ones included. What a stat counts must all come after
    that decision, or the stat is absent."""

    def __init__(self, start: GameState, or_starts: int) -> None:
        owners = [PRIVATES[p].owner(start) for p in range(D.NUM_PRIVATES)]
        self.auction = TURN.round(start) == K.ROUND_AUCTION and set(owners) <= {K.OWNER_UNSOLD, K.OWNER_NOT_IN_PLAY}
        self.premium = -sum(D.PRIVATE_VALUE[p] for p in range(D.NUM_PRIVATES) if owners[p] == K.OWNER_UNSOLD)
        self.sr1_cash: int | None = None
        self.window: tuple[int, int] | None = None
        self.payouts_amid_sales = False
        self.parred = {c for c in range(D.NUM_CORPS) if CORPS[c].par_index(start) >= 0}
        self.floated = {c for c in range(D.NUM_CORPS) if CORPS[c].floated(start)}
        self.float_cash: list[int] = []
        self.srs_ended = self.srs_stranded = 0
        self.from_sr1 = TURN.turn(start) == 1
        self.sr2_sold = 0
        self.sr2_over = False
        self.tokens_from_start = _tokens_off_home(start) == 0
        self.watch_5s = TRAINS.depot(start, D.TRAIN_4) > 0
        self.richest_at_5s: int | None = None       # as the last 4 leaves the depot
        self.richest_at_5s_seen: int | None = None  # at the next returned decision
        self.or_starts = or_starts
        self.phase_ors: dict[int, int] = {}
        mf_owner = PRIVATES[D.PRIVATE_MF].owner(start)
        self.watch_mf = not _mf_laid(start) and (
            mf_owner == K.OWNER_UNSOLD or K.OWNER_PLAYER <= mf_owner < K.OWNER_CORP
        )
        self.mf: tuple[int, bool] | None = None

    def decision(self, state: GameState) -> None:
        if self.richest_at_5s is not None and self.richest_at_5s_seen is None:
            self.richest_at_5s_seen = _richest(state)
        # Whether pass payouts and sales came between the same two decisions, where cash cannot price the sales.
        payouts = TURN.auction_pass_payouts(state)
        unsold = sum(PRIVATES[p].owner(state) == K.OWNER_UNSOLD for p in range(D.NUM_PRIVATES))
        if self.window is not None:
            self.payouts_amid_sales |= payouts > self.window[0] and unsold < self.window[1]
        self.window = (payouts, unsold) if TURN.round(state) == K.ROUND_AUCTION else None

    def step(self, before: GameState, after: GameState, phase: int, action: int) -> None:
        if self.auction and TURN.round(before) == K.ROUND_AUCTION:
            if TURN.auction_pass_payouts(after) == TURN.auction_pass_payouts(before):  # a payout sells nothing
                self.premium += _players_cash(before) - _players_cash(after)
            if TURN.round(after) == K.ROUND_STOCK:
                self.sr1_cash = _players_cash(after)
        for c in range(D.NUM_CORPS):
            if c not in self.floated and CORPS[c].floated(after):
                self.floated.add(c)
                self.float_cash.append(CORPS[c].cash(after))
        if TURN.round(before) == K.ROUND_STOCK != TURN.round(after):
            self.srs_ended += 1
            self.srs_stranded += any(
                CORPS[c].par_index(after) >= 0 and not CORPS[c].floated(after) for c in range(D.NUM_CORPS)
            )
            self.sr2_over |= TURN.turn(before) == 2
        if phase == K.DPHASE_SELL_AMOUNT and TURN.turn(before) == 2:
            self.sr2_sold += action + 1  # SELL_AMOUNT's id is the shares less one
        if self.watch_5s and TRAINS.depot(after, D.TRAIN_4) == 0:
            self.watch_5s = False
            self.richest_at_5s = _richest(after)
        self.or_starts += phase < 0 and action == K.STEP_OR_START
        for p in range(PHASE.current(before) + 1, PHASE.current(after) + 1):
            self.phase_ors[p] = self.or_starts
        if self.watch_mf and not _mf_laid(before) and _mf_laid(after):
            self.mf = (TURN.turn(before), TURN.round(before) == K.ROUND_OPERATING)

    def stats(self, end: GameState) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        if self.auction:
            assert self.sr1_cash is not None
            out["auction_premium"] = (self.premium, 1)
            out["auction_sr1_cash"] = (self.sr1_cash, TURN.num_players(end))
        pars = [CORPS[c].par_price(end) for c in range(D.NUM_CORPS)
                if c not in self.parred and CORPS[c].par_index(end) >= 0]
        if pars:
            out["par_price"] = (sum(pars), len(pars))
            out["par_65_share"] = (pars.count(65), len(pars))
        if self.srs_ended:
            out["par_stranded_share"] = (self.srs_stranded, self.srs_ended)
        if self.float_cash:
            out["corp_cash_float"] = (sum(self.float_cash), len(self.float_cash))
        if self.richest_at_5s_seen is not None:
            out["corp_cash_max_5s"] = (self.richest_at_5s_seen, 1)
        if self.from_sr1 and self.sr2_over:
            out["sr2_shares_sold"] = (self.sr2_sold, 1)
        if self.tokens_from_start:
            out["tokens_placed"] = (_tokens_off_home(end), 1)
        if self.from_sr1:
            out.update({f"phase_start_or_{PHASE_NAMES[p]}": (ors, 1) for p, ors in self.phase_ors.items()})
        if self.watch_mf:
            out["mf_port_laid"] = (self.mf is not None, 1)
        if self.mf is not None:
            out["mf_port_sr"] = (self.mf[0], 1)
            out["mf_port_in_or"] = (self.mf[1], 1)
        return out


V1_STATS = (
    "auction_premium", "auction_sr1_cash", "par_price", "par_65_share", "par_stranded_share", "corp_cash_float",
    "corp_cash_max_5s", "sr2_shares_sold", "tokens_placed", *(f"phase_start_or_{p}" for p in PHASE_NAMES[1:]),
    "mf_port_laid", "mf_port_sr", "mf_port_in_or",
)


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
    decisions: int
    truth: _Truth


def _play(num_players: int, seed: int, first: int = 0) -> Playout:
    """A random playout, its collector observing from returned decision ``first`` on."""
    rng = np.random.default_rng(seed)
    state = GAME.new_game(num_players, MAX_PLAYERS, seed)
    view = GameState(num_players, max_players=MAX_PLAYERS)
    after = GameState(num_players, max_players=MAX_PLAYERS)
    collector = GAME.new_selfplay_stats(num_players)
    scratch = np.empty(GAME.policy.max_action_size, dtype=np.uint16)
    or_starts = track_decisions = track_passes = 0
    bought: Counter[int] = Counter()
    corps_in_an_order: set[int] = set()
    dividends: Counter[int] = Counter()
    withholds: Counter[int] = Counter()
    truth: _Truth | None = None

    for decision in range(MAX_DECISIONS):
        phase = GAME.decision_phase(state)
        n_legal = GAME.enumerate_policy_actions(state, scratch)
        legal = scratch[:n_legal].copy()
        action = _choose(state, rng, True)
        track_decisions += phase == K.DPHASE_TRACK
        track_passes += phase == K.DPHASE_TRACK and action == K.TRACK_PASS
        if phase == K.DPHASE_DIVIDEND:
            dividends[PHASE.current(state)] += 1
            withholds[PHASE.current(state)] += action == K.DIVIDEND_WITHHOLD
        if decision == first:
            truth = _Truth(state, or_starts)
        if truth is not None:
            truth.decision(state)
            # Priors and visits only pass through: the collector's report ignores them.
            collector.observe(state, legal, int(np.flatnonzero(legal == action)[0]), np.ones(n_legal) / n_legal,
                              np.ones(n_legal))
        history: list[tuple[np.ndarray, int, int]] = []
        status = DRIVER.apply_action(state, action, history)
        assert status in (STATUS_OK, STATUS_GAME_OVER)
        for k, (row, row_phase, row_action) in enumerate(history):
            view.rebind(row, num_players, MAX_PLAYERS)
            if TURN.round(view) == K.ROUND_OPERATING:
                corps_in_an_order |= {TURN.or_order(view, i) for i in range(D.NUM_CORPS)} - {-1}
            if row_phase < 0:
                or_starts += row_action == K.STEP_OR_START
            elif row_phase == K.DPHASE_SELECT_TRAIN and TURN.pending_train_source(view) == K.TRAIN_SOURCE_DEPOT:
                info = A.decode_action_py(row_phase, row_action)
                bought[D.TRAIN_D if info.type == A.ACTION_TRADE_IN else info.train_type] += 1
            if truth is not None:
                after.rebind(history[k + 1][0] if k + 1 < len(history) else state._array, num_players, MAX_PLAYERS)
                truth.step(view, after, row_phase, row_action)
        if status == STATUS_GAME_OVER:
            assert truth is not None, f"{num_players}p seed {seed}: the game ends before decision {first}"
            return Playout(state, collector.finish(state), or_starts, bought, corps_in_an_order,
                           track_decisions, track_passes, dividends, withholds, decision + 1, truth)
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


def _v1_stats(stats: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    return {name: stats[name] for name in V1_STATS if name in stats}


def test_the_playouts_reach_what_the_v1_stats_report(playouts):
    """Auctions whose pass payouts fall among sales between two decisions, SRs
    with and without a stranded par, SR2 sales, every phase, the 5s on sale, and
    MF's tile laid in an SR and in an OR, so the checks below are not vacuous."""
    truths = [p.truth for p in playouts.values()]
    assert any(t.payouts_amid_sales for t in truths)
    assert any(t.srs_stranded for t in truths) and any(t.srs_stranded < t.srs_ended for t in truths)
    assert sum(t.sr2_sold for t in truths) > 0
    assert set().union(*(t.phase_ors for t in truths)) == set(range(1, len(PHASE_NAMES)))
    assert any(t.richest_at_5s is not None for t in truths)
    assert {t.mf[1] for t in truths if t.mf is not None} == {False, True}


def test_v1_stats_are_the_historys(playouts):
    for (n, seed), p in playouts.items():
        assert _v1_stats(p.stats) == p.truth.stats(p.state), f"{n}p seed {seed}"


def test_the_richest_treasury_at_the_5s_is_mostly_as_the_last_4_sells(playouts):
    """It is read at the next returned decision; automated steps between can move it."""
    pairs = [(p.truth.richest_at_5s, p.stats["corp_cash_max_5s"][0])
             for p in playouts.values() if "corp_cash_max_5s" in p.stats]
    assert len(pairs) > len(playouts) // 2
    assert sum(at == seen for at, seen in pairs) >= 0.9 * len(pairs)


@pytest.mark.parametrize("num_players", PLAYER_COUNTS)
def test_a_game_observed_from_a_later_decision_reports_only_what_followed(playouts, num_players):
    """As a game started from a mid-game position is: what the first observed
    decision could not see is absent, not 0, and the rest counts from there."""
    for seed in range(5):
        full = playouts[(num_players, seed)]
        for share in (0.05, 0.3, 0.6, 0.9):
            p = _play(num_players, seed, int(share * full.decisions))
            msg = f"{num_players}p seed {seed} from decision {int(share * full.decisions)}"
            assert np.array_equal(p.state._array, full.state._array), msg  # the same game
            assert _v1_stats(p.stats) == p.truth.stats(p.state), msg
            if share >= 0.6:
                assert not {"auction_premium", "sr2_shares_sold", "phase_start_or_3"} & set(p.stats), msg


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
        assert always <= set(record.game_stats) <= always | by_phase | set(V1_STATS)
        # A game from the start reports these whatever it does.
        assert {"auction_premium", "par_stranded_share", "tokens_placed", "mf_port_laid"} <= set(record.game_stats)
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
