"""Adapter conformance: every registered game meets the ``Game`` contract.

The generic tests talk only to the adapter. The engine and self-play stats
checks run for every game; the extraction checks skip a game whose extraction
has not landed, whose adapter raises NotImplementedError for it. The RSS
migration tests pin the RSS adapter to the direct engine calls that
consumers made before pipeline-refactor.md step 2 moved them onto the
adapter, and its self-play stats to the scalars train/main.py logged before
step 3 moved them behind the hook. The 1889 tests pin its adapter to the
g1889 engine and its extraction sizes.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import torch

from games import (
    GAME_NAMES,
    RELATION_RECORD_WIDTH,
    STATUS_GAME_OVER,
    STATUS_INVALID,
    STATUS_OK,
    Game,
    get_game,
)
from train.config import TrainingConfig
from train.self_play import play_game

# RSS migration and 1889 checks only.
from core.actions import (
    enumerate_legal_actions_py,
    enumerate_policy_actions_py,
    get_decision_phase_py,
)
from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
)
from core.data import MAX_ACTION_SIZE, PHASE_ACTION_SIZES, DecisionPhase, GameConstants, GamePhases
from core.driver import STATUS_GAME_OVER_PY, STATUS_INVALID_PY, STATUS_OK_PY
from core.relations import (
    get_relation_coord_data,
    get_relation_coord_data_batch,
    get_relation_data,
    get_relation_data_batch,
)
from core.state import GameState, get_layout, get_storage_player_capacity
from core.token_data import (
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
    get_token_dim,
)
from entities.player import PLAYERS
from entities.turn import TURN
import g1889.constants as K1889
from g1889.actions import (
    ACTION_PASS as G1889_PASS,
    decode_action_py as g1889_decode,
    enumerate_legal_actions_py as g1889_legal,
)
from g1889.driver import (
    STATUS_GAME_OVER as G1889_GAME_OVER,
    STATUS_INVALID as G1889_INVALID,
    STATUS_OK as G1889_OK,
)
from g1889.state import GameState as G1889State, get_layout as g1889_layout
from games.rss_selfplay import StrategyTraceBuilder
from mcts.evaluator import BaseEvaluator, compute_terminal_values
from nn import policy_layout as legacy_policy_layout
from tests.test_actions_width import _state_acq_select_price


REPO_ROOT = Path(__file__).resolve().parents[1]

# Engine options of train_configs/v3-09-20-26.json.
V3_ENGINE_OPTIONS = {
    "v3_behavior": True,
    "acq_same_president": False,
    "max_acq_price_actions": 8,
}

# Replay and live play run 1889 with every policy flag off (engine-design.md
# §7); self-play keeps the config defaults.
G1889_FLAGS_OFF = {
    "private_sale_same_president": False,
    "train_sale_same_president": False,
    "auction_pass_payout_cap": 0,
}

# Case id -> (config fields, token layout version, or None for a game whose
# extraction has not landed). Every registered game needs at least one case.
CASES: dict[str, tuple[dict[str, Any], int | None]] = {
    "rss-v2": ({"game": "rss"}, 2),
    "rss-v3": ({"game": "rss", **V3_ENGINE_OPTIONS}, 3),
    "1889": ({"game": "1889"}, 1),
    "1889-flags-off": ({"game": "1889", **G1889_FLAGS_OFF}, 1),
}
RSS_CASES = [case_id for case_id, (fields, _) in CASES.items() if fields["game"] == "rss"]
G1889_CASES = [case_id for case_id, (fields, _) in CASES.items() if fields["game"] == "1889"]

# (num_players, max_players): exact-size rows, and padded rows that share one
# capacity as in mixed-player training.
PLAYER_SETUPS = ((3, 3), (3, 5), (4, 5), (5, 5))
SEEDS = range(3)
SAMPLE_EVERY = 5
MAX_DECISIONS = 5000


@dataclass
class Sample:
    """Live-state model inputs at one decision."""

    decision: int
    tokens: np.ndarray
    records: np.ndarray
    record_count: int


@dataclass
class Playout:
    """A uniform random playout through the policy action set."""

    num_players: int
    max_players: int
    final: Any  # the live state, now terminal; its runtime flags never change
    rows: list[np.ndarray] = field(default_factory=list)  # before each decision
    phases: list[int] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    active_players: list[int] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    # The self-play stats collector's report, observing every decision as
    # search's top choice under uniform priors.
    game_stats: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class Case:
    config: TrainingConfig
    game: Game
    layout_version: int | None  # None: no extraction, so no samples
    playouts: list[Playout]


def _token_buffer(game: Game, max_players: int, layout_version: int, n: int = 0) -> np.ndarray:
    shape = (game.num_tokens(max_players), game.token_dim(layout_version))
    return np.full((n, *shape) if n else shape, np.nan, dtype=np.float32)


def _dense_buffer(game: Game, max_players: int, n: int = 0) -> np.ndarray:
    num_tokens = game.num_tokens(max_players)
    shape = (game.num_relations, num_tokens, num_tokens)
    return np.full((n, *shape) if n else shape, 0xFF, dtype=np.uint8)


def _record_buffer(game: Game, n: int = 0) -> np.ndarray:
    shape = (game.max_relation_edges, RELATION_RECORD_WIDTH)
    return np.full((n, *shape) if n else shape, 0xFF, dtype=np.uint8)


def _materialize(
    records: np.ndarray, count: int, num_relations: int, num_tokens: int,
) -> np.ndarray:
    dense = np.zeros((num_relations, num_tokens, num_tokens), dtype=np.uint8)
    relation, query, key, value = records[:count].T
    dense[relation, query, key] = value
    return dense


def _play(
    game: Game, num_players: int, max_players: int, seed: int, layout_version: int | None,
) -> Playout:
    state = game.new_game(num_players, max_players, seed)
    playout = Playout(num_players, max_players, state)
    stats = game.new_selfplay_stats(num_players)
    rng = np.random.default_rng(seed)
    buf = np.empty(game.policy.max_action_size, dtype=np.uint16)
    for decision in range(MAX_DECISIONS):
        if game.is_terminal(state):
            playout.game_stats = stats.finish(state)
            return playout
        n = game.enumerate_policy_actions(state, buf)
        assert n > 0, f"seed {seed}: no policy actions at decision {decision}"
        playout.rows.append(state._array.copy())
        playout.phases.append(game.decision_phase(state))
        playout.actions.append(buf[:n].copy())
        playout.active_players.append(game.active_player(state))
        if layout_version is not None and decision % SAMPLE_EVERY == 0:
            tokens = _token_buffer(game, max_players, layout_version)
            game.fill_tokens(state, tokens, max_players=max_players, layout_version=layout_version)
            records = _record_buffer(game)
            count = game.fill_relations_sparse(state, records, max_players=max_players)
            playout.samples.append(Sample(decision, tokens, records, count))
        chosen = int(rng.integers(n))
        counts = np.ones(n, dtype=np.float32)
        counts[chosen] = 2.0
        stats.observe(state, buf[:n].copy(), chosen, np.full(n, 1.0 / n, dtype=np.float32), counts)
        status = game.apply_action(state, int(buf[chosen]))
        assert status in (STATUS_OK, STATUS_GAME_OVER), f"seed {seed}: status {status}"
        assert (status == STATUS_GAME_OVER) == game.is_terminal(state)
    pytest.fail(f"seed {seed}: no terminal state within {MAX_DECISIONS} decisions")


@cache
def _case(case_id: str) -> Case:
    fields, layout_version = CASES[case_id]
    config = TrainingConfig(**fields)
    game = get_game(config)
    playouts = [
        _play(game, num_players, max_players, seed, layout_version)
        for num_players, max_players in PLAYER_SETUPS
        for seed in SEEDS
    ]
    return Case(config, game, layout_version, playouts)


@pytest.fixture(params=list(CASES))
def case(request: pytest.FixtureRequest) -> Case:
    return _case(request.param)


@pytest.fixture
def extraction_case(case: Case) -> Case:
    """``case``, or a skip, with the adapter's reason, for a game whose
    extraction has not landed."""
    if case.layout_version is None:
        with pytest.raises(NotImplementedError) as missing:
            case.game.num_tokens(case.playouts[0].max_players)
        pytest.skip(str(missing.value))
    return case


@pytest.fixture(params=RSS_CASES)
def rss_case(request: pytest.FixtureRequest) -> Case:
    return _case(request.param)


@pytest.fixture(params=G1889_CASES)
def g1889_case(request: pytest.FixtureRequest) -> Case:
    return _case(request.param)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_every_registered_game_has_a_case():
    assert {fields["game"] for fields, _ in CASES.values()} == set(GAME_NAMES)


def test_get_game_rejects_unknown_names():
    config = TrainingConfig()
    config.game = "chess"
    with pytest.raises(ValueError, match="registered games: rss, 1889$"):
        get_game(config)


def test_importing_games_imports_no_engine():
    code = (
        "import sys, games; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'core', 'entities', 'phases', 'g1889'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# Generic conformance: adapter calls only
# ---------------------------------------------------------------------------


def test_policy_layout_is_consistent(case: Case):
    policy = case.game.policy
    assert case.game.name == case.config.game
    assert policy.offsets[0] == 0
    assert tuple(np.diff(policy.offsets)) == policy.phase_sizes
    assert policy.unified_dim == policy.offsets[-1]
    assert policy.max_action_size == max(policy.phase_sizes) <= 1 << 16
    assert len(set(policy.phase_names)) == policy.num_phases
    assert set(policy.phases_with_pass_slot) <= set(range(policy.num_phases))
    lut = policy.build_action_lut()
    assert lut.shape == (policy.num_phases, policy.max_action_size)
    for phase, size in enumerate(policy.phase_sizes):
        assert lut[phase, :size].tolist() == list(
            range(policy.offsets[phase], policy.offsets[phase + 1])
        )


def test_playouts_end_with_one_score_per_player(case: Case):
    for playout in case.playouts:
        assert case.game.is_terminal(playout.final)
        scores = case.game.final_scores(playout.final)
        assert len(scores) == playout.num_players
        assert all(isinstance(score, int) for score in scores)


def test_policy_actions_are_distinct_phase_local_ids(case: Case):
    policy = case.game.policy
    for playout in case.playouts:
        for phase, actions, player in zip(
            playout.phases, playout.actions, playout.active_players, strict=True,
        ):
            assert 0 <= phase < policy.num_phases
            assert len(np.unique(actions)) == len(actions)
            assert int(actions.max()) < policy.phase_sizes[phase]
            assert 0 <= player < playout.num_players


def test_rows_report_player_count_and_capacity(case: Case):
    game = case.game
    for playout in case.playouts:
        assert playout.final.max_players == playout.max_players
        for row in playout.rows[::SAMPLE_EVERY]:
            assert row.dtype == np.int16
            assert row.shape == (game.state_size(playout.max_players),)
            assert game.storage_player_capacity(row.shape[0]) == playout.max_players
            assert game.num_players_from_row(row) == playout.num_players


def test_every_policy_action_applies_on_a_rebound_scratch_state(case: Case):
    # Search expansion: copy the parent row, rebind one scratch state, apply.
    game = case.game
    for playout in case.playouts:
        row = playout.rows[0].copy()
        scratch = game.scratch_state(playout.final, row)
        for source, actions in zip(playout.rows, playout.actions, strict=True):
            for action in actions:
                row[:] = source
                scratch.rebind(row, playout.num_players, max_players=playout.max_players)
                status = game.apply_action(scratch, int(action))
                assert status in (STATUS_OK, STATUS_GAME_OVER)
                assert (status == STATUS_GAME_OVER) == game.is_terminal(scratch)


def test_selfplay_stats_report_numerator_denominator_pairs(case: Case):
    # train/main.py logs each name as sum(numerators) / max(sum(denominators), 1).
    for playout in case.playouts:
        for name, (numerator, denominator) in playout.game_stats.items():
            assert isinstance(name, str) and name
            assert math.isfinite(numerator) and math.isfinite(denominator)
            assert denominator >= 0


def _scratch_states(game: Game, playout: Playout, decisions) -> Iterator[tuple[int, Any]]:
    """For each decision, a fresh scratch state over a copy of the row before
    it, and one scratch state of the playout's rebound to another copy."""
    rebound = game.scratch_state(playout.final, playout.rows[0].copy())
    for decision in decisions:
        source = playout.rows[decision]
        row = source.copy()
        fresh = game.scratch_state(playout.final, row)
        assert np.shares_memory(fresh._array, row)
        rebound.rebind(source.copy(), playout.num_players, max_players=playout.max_players)
        yield decision, fresh
        yield decision, rebound


def test_scratch_state_reproduces_decisions(case: Case):
    game = case.game
    buf = np.empty(game.policy.max_action_size, dtype=np.uint16)
    for playout in case.playouts:
        for decision, scratch in _scratch_states(game, playout, range(0, len(playout.rows), SAMPLE_EVERY)):
            assert scratch.max_players == playout.max_players
            assert game.decision_phase(scratch) == playout.phases[decision]
            assert game.active_player(scratch) == playout.active_players[decision]
            n = game.enumerate_policy_actions(scratch, buf)
            np.testing.assert_array_equal(buf[:n], playout.actions[decision])


def test_scratch_state_reproduces_model_inputs(extraction_case: Case):
    case = extraction_case
    game = case.game
    assert case.layout_version is not None
    for playout in case.playouts:
        samples = {sample.decision: sample for sample in playout.samples}
        for decision, scratch in _scratch_states(game, playout, samples):
            sample = samples[decision]
            tokens = _token_buffer(game, playout.max_players, case.layout_version)
            game.fill_tokens(
                scratch, tokens,
                max_players=playout.max_players, layout_version=case.layout_version,
            )
            np.testing.assert_array_equal(tokens, sample.tokens)
            records = _record_buffer(game)
            count = game.fill_relations_sparse(
                scratch, records, max_players=playout.max_players,
            )
            assert count == sample.record_count
            np.testing.assert_array_equal(records, sample.records)


def test_dense_and_sparse_relations_encode_the_same_edges(extraction_case: Case):
    game = extraction_case.game
    for playout in extraction_case.playouts:
        num_tokens = game.num_tokens(playout.max_players)
        for sample in playout.samples:
            assert np.isfinite(sample.tokens).all()
            assert 0 <= sample.record_count <= game.max_relation_edges
            assert not sample.records[sample.record_count:].any()
            assert (sample.records[:sample.record_count, 1:3] < num_tokens).all()
            state = game.scratch_state(playout.final, playout.rows[sample.decision].copy())
            dense = _dense_buffer(game, playout.max_players)
            game.fill_relations_dense(state, dense, max_players=playout.max_players)
            assert np.count_nonzero(dense) == sample.record_count
            np.testing.assert_array_equal(
                _materialize(sample.records, sample.record_count, game.num_relations, num_tokens),
                dense,
            )


def test_batch_extraction_matches_single_states(extraction_case: Case):
    case = extraction_case
    game, layout_version = case.game, case.layout_version
    assert layout_version is not None
    for max_players in sorted({playout.max_players for playout in case.playouts}):
        # One batch per capacity, mixing player counts like a replay batch.
        playouts = [playout for playout in case.playouts if playout.max_players == max_players]
        pairs = [(playout, sample) for playout in playouts for sample in playout.samples[::4]]
        rows = [playout.rows[sample.decision].copy() for playout, sample in pairs]
        n = len(rows)

        tokens = _token_buffer(game, max_players, layout_version, n)
        game.fill_tokens_batch(rows, tokens, max_players=max_players, layout_version=layout_version)
        for i, (_, sample) in enumerate(pairs):
            np.testing.assert_array_equal(tokens[i], sample.tokens)
        dense = _dense_buffer(game, max_players, n)
        game.fill_relations_dense_batch(rows, dense, max_players=max_players)
        records = _record_buffer(game, n)
        game.fill_relations_sparse_batch(rows, records, max_players=max_players)

        for i, (playout, sample) in enumerate(pairs):
            np.testing.assert_array_equal(records[i], sample.records)
            single = _dense_buffer(game, max_players)
            state = game.scratch_state(playout.final, rows[i])
            game.fill_relations_dense(state, single, max_players=max_players)
            np.testing.assert_array_equal(dense[i], single)


# ---------------------------------------------------------------------------
# RSS migration: the adapter equals the direct engine calls it replaced in step 2
# ---------------------------------------------------------------------------


def test_rss_policy_layout_matches_legacy_constants():
    """RSS's layout is the engine's action table, and nn.policy_layout's legacy
    names, which the v2 and v3 model files import, are that layout."""
    policy = get_game(TrainingConfig()).policy
    assert policy.phase_sizes == tuple(PHASE_ACTION_SIZES)
    assert policy.num_phases == GameConstants.NUM_DECISION_PHASES
    assert policy.offsets == (0, *np.cumsum(PHASE_ACTION_SIZES).tolist())
    assert policy.max_action_size == MAX_ACTION_SIZE
    assert policy.phases_with_pass_slot == tuple(int(phase) for phase in (
        DecisionPhase.DPHASE_INVEST, DecisionPhase.DPHASE_BID, DecisionPhase.DPHASE_ACQ_SELECT_CORP,
        DecisionPhase.DPHASE_CLOSING, DecisionPhase.DPHASE_IPO,
    ))
    # The TensorBoard spellings of per-phase training and self-play metrics.
    assert policy.phase_names == (
        "invest", "bid", "acq_corp", "acq_offer", "close", "div", "issue",
        "ipo", "par", "acq_co", "acq_price",
    )
    lut = policy.build_action_lut()
    assert lut.shape == (GameConstants.NUM_DECISION_PHASES, MAX_ACTION_SIZE)
    for phase, size in enumerate(PHASE_ACTION_SIZES):
        assert lut[phase, :size].tolist() == list(range(policy.offsets[phase], policy.offsets[phase + 1]))
        assert not lut[phase, size:].any()

    assert legacy_policy_layout.NUM_PHASES == policy.num_phases
    assert legacy_policy_layout.PHASE_OFFSETS == policy.offsets
    assert legacy_policy_layout.UNIFIED_LOGIT_DIM == policy.unified_dim
    assert legacy_policy_layout.PHASES_WITH_PASS_SLOT == policy.phases_with_pass_slot
    assert torch.equal(legacy_policy_layout.build_action_lut(), lut)


def test_rss_conventions_match_engine_constants(rss_case: Case):
    game = rss_case.game
    assert (STATUS_OK, STATUS_GAME_OVER, STATUS_INVALID) == (
        STATUS_OK_PY, STATUS_GAME_OVER_PY, STATUS_INVALID_PY,
    )
    assert RELATION_RECORD_WIDTH == ATTENTION_RELATION_COORD_WIDTH
    assert game.num_relations == NUM_ATTENTION_RELATIONS
    assert game.max_relation_edges == MAX_ATTENTION_RELATION_EDGES
    for max_players in (3, 4, 5):
        assert game.num_tokens(max_players) == get_num_tokens(max_players)
        assert game.state_size(max_players) == get_layout(max_players).total_size
    for layout_version in (2, 3):
        assert game.token_dim(layout_version) == get_token_dim(layout_version)


def test_rss_engine_calls_match_direct_calls(rss_case: Case):
    game, cap = rss_case.game, rss_case.config.max_acq_price_actions
    adapter_buf = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    direct_buf = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    for playout in rss_case.playouts:
        assert playout.final.max_acq_price_actions == cap
        for row in playout.rows:
            state = game.scratch_state(playout.final, row.copy())
            capacity = row.shape[0]
            assert game.num_players_from_row(row) == TURN.get_num_players(state)
            assert game.storage_player_capacity(capacity) == get_storage_player_capacity(capacity)
            assert game.decision_phase(state) == get_decision_phase_py(state)
            assert game.active_player(state) == TURN.get_active_player(state)
            assert not game.is_terminal(state)
            n = game.enumerate_policy_actions(state, adapter_buf)
            assert n == enumerate_policy_actions_py(state, direct_buf)
            np.testing.assert_array_equal(adapter_buf[:n], direct_buf[:n])
            # The policy set is a subset of what the driver accepts.
            n_legal = enumerate_legal_actions_py(state, direct_buf)
            assert set(adapter_buf[:n].tolist()) <= set(direct_buf[:n_legal].tolist())
        assert game.is_terminal(playout.final)
        assert TURN.get_phase(playout.final) == GamePhases.PHASE_GAME_OVER

    # Random playouts rarely see a wide ACQ price range. The config's cap
    # reaches it through the root's runtime flags.
    wide = _state_acq_select_price()
    state = game.scratch_state(rss_case.playouts[0].final, wide._array.copy())
    n_legal = enumerate_legal_actions_py(state, direct_buf)
    assert n_legal > V3_ENGINE_OPTIONS["max_acq_price_actions"]
    assert game.enumerate_policy_actions(state, adapter_buf) == (cap or n_legal)


def test_rss_final_scores_are_the_net_worths_evaluate_terminal_reads(rss_case: Case):
    game = rss_case.game
    weight = rss_case.config.terminal_blend
    for playout in rss_case.playouts:
        state = playout.final
        scores = game.final_scores(state)
        assert scores == [PLAYERS[i].get_net_worth(state) for i in range(playout.num_players)]
        evaluator = BaseEvaluator(game, playout.max_players, terminal_rank_weight=weight)
        np.testing.assert_array_equal(
            compute_terminal_values(scores, playout.num_players, weight),
            evaluator.evaluate_terminal(state),
        )


# The self-play scalars train/main.py logged per reporting group before step 3
# moved them behind the stats hook, without their self_play_<group>/ prefix.
RSS_GAME_STATS = (
    "acq_decisions_per_phase", "acq_offers_per_phase", "acq_offer_acceptance_rate",
    "acq_cap_hits_per_phase", "acq_offer_accept_prior_mean",
    "acq_rejections_expected_fraction", "acq_rejections_surprise_fraction",
    "acq_offers_top_choice_fraction", "acq_top_choice_acceptance_rate",
    "invest_roundtrip_cap_hits_per_game", "total_shares", "total_companies",
    "avg_active_corp_price", "corps_in_receivership", "pct_games_max_price_corp",
)
# Logged by finishing rank, as <name>_1st, <name>_2nd, ...
RSS_RANK_STATS = (
    "invest_roundtrip_cap_hits", "shares", "companies", "pres_share_value",
    "nw_cash_pct", "nw_companies_pct", "nw_shares_pct",
)
RANK_LABELS = ("1st", "2nd", "3rd", "4th", "5th")
NUM_CORPS = int(GameConstants.NUM_CORPS)


def test_rss_selfplay_stats_report_the_scalars_main_logged(rss_case: Case):
    for playout in rss_case.playouts:
        n, state, stats = playout.num_players, playout.final, playout.game_stats
        ranked = sorted(range(n), key=lambda p: PLAYERS[p].get_net_worth(state), reverse=True)
        labels = RANK_LABELS[:n]
        assert set(stats) == set(RSS_GAME_STATS) | {
            f"{name}_{label}" for name in RSS_RANK_STATS for label in labels
        }
        # Everything but the acquisition rates is a per-game mean.
        assert all(den == 1 for name, (_, den) in stats.items() if not name.startswith("acq_"))
        decisions, phases = stats["acq_decisions_per_phase"]
        assert decisions >= phases
        # Each rank reports its player, the one with that finishing net worth.
        for label, player in zip(labels, ranked):
            assert stats[f"invest_roundtrip_cap_hits_{label}"][0] == (
                PLAYERS[player].get_invest_roundtrip_cap_hits(state)
            )
            assert stats[f"shares_{label}"][0] == sum(
                PLAYERS[player].get_shares(state, corp) for corp in range(NUM_CORPS)
            )
            if PLAYERS[player].get_net_worth(state) > 0:
                assert sum(
                    stats[f"nw_{part}_pct_{label}"][0] for part in ("cash", "companies", "shares")
                ) == pytest.approx(1.0)
        for total, per_rank in (
            ("invest_roundtrip_cap_hits_per_game", "invest_roundtrip_cap_hits"),
            ("total_shares", "shares"),
            ("total_companies", "companies"),
        ):
            assert stats[total][0] == sum(stats[f"{per_rank}_{label}"][0] for label in labels)


def test_rss_strategy_trace_is_the_rss_builder(rss_case: Case):
    new_trace = rss_case.game.new_strategy_trace
    assert new_trace is not None
    builder = new_trace(4)
    assert isinstance(builder, StrategyTraceBuilder)
    assert (builder.num_players, builder.unified_dim) == (4, rss_case.game.policy.unified_dim)


def _assert_same_fill(adapter_fill, direct_fill, source, buf: np.ndarray, **kwargs) -> None:
    expected = buf.copy()
    assert adapter_fill(source, buf, **kwargs) == direct_fill(source, expected, **kwargs)
    np.testing.assert_array_equal(buf, expected)


def test_rss_model_inputs_match_direct_extractors(rss_case: Case):
    game, layout_version = rss_case.game, rss_case.layout_version
    assert layout_version is not None
    for playout in rss_case.playouts:
        mp = playout.max_players
        rows = [playout.rows[sample.decision].copy() for sample in playout.samples]
        n = len(rows)
        _assert_same_fill(
            game.fill_tokens_batch, get_token_data_batch, rows,
            _token_buffer(game, mp, layout_version, n),
            max_players=mp, layout_version=layout_version,
        )
        _assert_same_fill(
            game.fill_relations_dense_batch, get_relation_data_batch, rows,
            _dense_buffer(game, mp, n), max_players=mp,
        )
        _assert_same_fill(
            game.fill_relations_sparse_batch, get_relation_coord_data_batch, rows,
            _record_buffer(game, n), max_players=mp,
        )
        for row in rows:
            state = game.scratch_state(playout.final, row)
            _assert_same_fill(
                game.fill_tokens, get_token_data, state,
                _token_buffer(game, mp, layout_version),
                max_players=mp, layout_version=layout_version,
            )
            _assert_same_fill(
                game.fill_relations_dense, get_relation_data, state,
                _dense_buffer(game, mp), max_players=mp,
            )
            _assert_same_fill(
                game.fill_relations_sparse, get_relation_coord_data, state,
                _record_buffer(game), max_players=mp,
            )


def test_rss_new_game_and_scratch_state_match_direct_construction(rss_case: Case):
    config, game = rss_case.config, rss_case.game
    flags = (
        "v3_behavior", "acq_same_president", "acq_offer_limits",
        "max_acq_price_actions", "allow_positive_income_closing", "step_mode",
    )

    # train/self_play.py play_game before step 2
    state = game.new_game(4, 5, 7)
    expected = GameState(
        4, max_players=5, v3_behavior=config.v3_behavior,
        acq_same_president=config.acq_same_president,
        max_acq_price_actions=config.max_acq_price_actions,
    )
    expected.initialize_game(4, seed=7, max_players=5)
    np.testing.assert_array_equal(state._array, expected._array)
    assert {f: getattr(state, f) for f in flags} == {f: getattr(expected, f) for f in flags}

    # mcts/search.py run_search before step 2: the scratch carries the root's
    # four runtime flags. Change every flag on the root so each one is observable.
    root = GameState.from_array(
        state._array, 4, max_players=5, v3_behavior=not config.v3_behavior,
    )
    for name in flags[1:]:
        if name != "max_acq_price_actions":
            setattr(root, name, not getattr(root, name))
    root.max_acq_price_actions = 6  # neither test config's cap
    scratch = game.scratch_state(root, state._array.copy())
    expected = GameState.from_buffer(
        state._array.copy(), 4, max_players=5, v3_behavior=root.v3_behavior,
    )
    expected.acq_same_president = root.acq_same_president
    expected.acq_offer_limits = root.acq_offer_limits
    expected.max_acq_price_actions = root.max_acq_price_actions
    assert {f: getattr(scratch, f) for f in flags} == {f: getattr(expected, f) for f in flags}
    np.testing.assert_array_equal(scratch._array, state._array)


# ---------------------------------------------------------------------------
# 1889: the adapter over the g1889 engine and its extraction
# ---------------------------------------------------------------------------


def test_g1889_conventions_match_engine_constants(g1889_case: Case):
    game = g1889_case.game
    assert (STATUS_OK, STATUS_GAME_OVER, STATUS_INVALID) == (G1889_OK, G1889_GAME_OVER, G1889_INVALID)
    policy = game.policy
    assert policy.phase_sizes == tuple(K1889.PHASE_ACTION_SIZES)
    assert (policy.unified_dim, policy.max_action_size) == (297, K1889.MAX_ACTION_SIZE)  # engine-design §5.2
    # As in RSS, the phases whose slot 0 passes before their other choices.
    assert set(policy.phases_with_pass_slot) == {
        phase for phase in range(policy.num_phases) if g1889_decode(phase, 0).type == G1889_PASS
    }
    for max_players in range(3, 7):  # engine-design §3: 760 + 18 per seat
        assert game.state_size(max_players) == g1889_layout(max_players).total_size == 760 + 18 * max_players


def test_g1889_new_game_and_scratch_state_carry_the_flags(g1889_case: Case):
    config, game = g1889_case.config, g1889_case.game
    flags = tuple(G1889_FLAGS_OFF)
    state = game.new_game(4, 5, 7)
    expected = G1889State(4, max_players=5, **{flag: getattr(config, flag) for flag in flags})
    expected.initialize_game(4, seed=7, max_players=5)
    np.testing.assert_array_equal(state._array, expected._array)
    assert {flag: getattr(state, flag) for flag in flags} == {flag: getattr(config, flag) for flag in flags}

    # The scratch carries the root's flags, whatever the config's.
    root = G1889State.from_array(state._array, 4, max_players=5)
    root.private_sale_same_president = not config.private_sale_same_president
    root.train_sale_same_president = not config.train_sale_same_president
    root.auction_pass_payout_cap = 3  # neither case's
    scratch = game.scratch_state(root, state._array.copy())
    assert {flag: getattr(scratch, flag) for flag in flags} == {flag: getattr(root, flag) for flag in flags}
    np.testing.assert_array_equal(scratch._array, state._array)


def test_g1889_policy_actions_are_engine_legal(g1889_case: Case):
    game = g1889_case.game
    buf = np.empty(game.policy.max_action_size, dtype=np.uint16)
    for playout in g1889_case.playouts:
        for row, actions in zip(playout.rows, playout.actions, strict=True):
            n = g1889_legal(game.scratch_state(playout.final, row.copy()), buf)
            assert set(actions.tolist()) <= set(buf[:n].tolist())


def test_g1889_has_no_selfplay_stats_or_strategy_trace(g1889_case: Case):
    assert all(playout.game_stats == {} for playout in g1889_case.playouts)
    assert g1889_case.game.new_strategy_trace is None
    with pytest.raises(ValueError, match="'1889' has no strategy trace"):
        play_game(None, g1889_case.config, 0, np.random.default_rng(0), collect_strategy_trace=True)


def test_g1889_extraction_sizes(g1889_case: Case):
    """extraction-design §5.1 and §6: 90 + max_players tokens of width 108, 25
    relations, and 2,048 records, above §6.3's worst case of 1,844."""
    game = g1889_case.game
    assert [game.num_tokens(max_players) for max_players in (3, 4, 5)] == [93, 94, 95]
    assert game.token_dim(1) == 108
    assert (game.num_relations, game.max_relation_edges) == (25, 2048)
