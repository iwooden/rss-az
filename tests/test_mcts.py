"""Tests for the transformer-refactor MCTS implementation.

Covers the four MCTS modules on the sparse-policy contract:

- ``mcts/mcts_core.pyx`` — Cython hot paths: ``select_child``, ``backup``,
  ``increment_visits``, ``virtual_backup``, ``expand_node_sparse``.
- ``mcts/node.py`` — ``MCTSNode`` with sparse expand + per-leaf pending
  phase/action context slots.
- ``mcts/evaluator.py`` — ``NNEvaluator`` / ``compute_terminal_values``
  over token buffers; evaluate returns the 5-tuple
  ``(sparse_priors, values, action_ids, n_legal, phase_id)``.
- ``mcts/search.py`` — ``run_search`` with sparse expand + batched
  leaf eval, ``StatePool`` (int16 compact rows), A0GB greedy leaf
  value, propagation lock/unlock, and subtree reuse.

All tests here target 3-player games (the NN/MCTS/training scope is 3-5p,
per CLAUDE.md; 3p is the canonical smoke config).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.actions import (
    enumerate_legal_actions_py,
    get_decision_phase_py,
    MAX_LEGAL_ACTIONS_PY,
)
from core.data import GamePhases, MAX_ACTION_SIZE
from core.driver import DRIVER
from core.state import GameState, get_layout
from entities.company import COMPANIES
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, compute_terminal_values, fill_token_buffer
from mcts.mcts_core import (
    backup,
    expand_node_sparse,
    gather_action_ids,
    gather_n_legals,
    gather_phase_ids,
    gather_states,
    increment_visits,
    scatter_results,
    select_child,
    virtual_backup,
)
from mcts.node import MCTSNode
from mcts.search import (
    StatePool,
    _add_dirichlet_noise,
    _collect_subtree_nodes,
    _propagate_lock,
    _propagate_unlock,
    _reset_root_for_reuse,
    get_action_probabilities,
    get_greedy_leaf_value,
    prepare_reuse_root,
    run_search,
)
from nn.transformer import RSSTransformerNet, TransformerConfig
from train.config import MCTSConfig


K_MAX = int(MAX_LEGAL_ACTIONS_PY)
NUM_PLAYERS = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game_state():
    """Fresh 3p game state at the INVEST opener (deterministic seed)."""
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=42)
    assert TURN.get_phase(state) == GamePhases.PHASE_INVEST
    return state


@pytest.fixture(scope="session")
def model():
    """Shared transformer model (random weights, reproducible seed)."""
    torch.manual_seed(0)
    cfg = TransformerConfig(num_players=NUM_PLAYERS)
    return RSSTransformerNet(cfg).to(torch.device("cpu"))


@pytest.fixture(scope="session")
def evaluator(model):
    """In-process NNEvaluator wrapped around the shared model."""
    return NNEvaluator(model, torch.device("cpu"), num_players=NUM_PLAYERS)


@pytest.fixture
def search_root(game_state, evaluator):
    """Root node after 20 sims — shared fixture for post-search probing."""
    config = MCTSConfig(num_simulations=20, num_players=NUM_PLAYERS)
    root = run_search(game_state, evaluator, config)
    return root, config


@pytest.fixture
def search_root_deep(game_state, evaluator):
    """Deeper root (100 sims) for A0GB traversal sanity."""
    config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)
    root = run_search(game_state, evaluator, config)
    return root, config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_expanded_node(
    active_player: int, num_actions: int, *,
    num_players: int = NUM_PLAYERS,
    priors: np.ndarray | None = None,
    default_value: np.ndarray | None = None,
    visit_counts: np.ndarray | None = None,
    value_sums: np.ndarray | None = None,
) -> MCTSNode:
    """Construct a synthetic MCTSNode mirroring the post-expand state.

    Post-expand contract (see ``expand_node_sparse``):
    - ``visit_counts`` is zeros ``(num_actions,)``
    - ``value_sums[i]`` broadcasts from ``default_value`` (FPU)
    - Node's own ``visit_count = 1`` after its NN eval is backed up

    PUCT treats ``visit_counts[i] == 0`` as FPU: Q = value_sums[i] (raw),
    denominator = 1. There is NO +1 virtual visit baked into the arrays.

    Callers that want to simulate "after N real visits on action i" should
    pass ``visit_counts[i] = N`` and a matching ``value_sums[i]`` (sum of
    the N backed-up values).
    """
    node = MCTSNode(active_player_id=active_player, num_players=num_players)
    node.legal_actions = np.arange(num_actions, dtype=np.int32)
    if priors is None:
        priors = np.full(num_actions, 1.0 / num_actions, dtype=np.float32)
    node.priors = priors.astype(np.float32)
    if default_value is None:
        default_value = np.zeros(num_players, dtype=np.float32)
    node.default_value = default_value.astype(np.float32)
    if visit_counts is None:
        visit_counts = np.zeros(num_actions, dtype=np.int32)
    node.visit_counts = visit_counts.astype(np.int32)
    if value_sums is None:
        value_sums = np.broadcast_to(
            default_value, (num_actions, num_players),
        ).astype(np.float32).copy()
    node.value_sums = value_sums.astype(np.float32)
    # Tree invariant: visit_count == 1 + sum(per-action visits).
    node.visit_count = 1 + int(visit_counts.sum())
    return node


# ---------------------------------------------------------------------------
# MCTSConfig
# ---------------------------------------------------------------------------

class TestMCTSConfig:
    def test_defaults(self):
        cfg = MCTSConfig(num_players=3)
        assert cfg.num_simulations == 800
        assert cfg.c_puct == 2.5
        assert cfg.dirichlet_alpha == 0.8
        assert cfg.dirichlet_epsilon == 0.25
        assert cfg.dirichlet_dynamic is True
        assert cfg.dirichlet_alpha_numerator == 10.0
        assert cfg.num_players == 3
        assert cfg.search_batch_size == 8

    def test_action_dim_is_max_action_size(self):
        """Post-refactor: dense pad width is player-count independent."""
        cfg3 = MCTSConfig(num_players=3)
        cfg5 = MCTSConfig(num_players=5)
        assert cfg3.action_dim == int(MAX_ACTION_SIZE)
        assert cfg5.action_dim == int(MAX_ACTION_SIZE)

    def test_validation_num_simulations(self):
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=0)
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=-1)

    def test_validation_search_batch_size(self):
        with pytest.raises(ValueError, match="search_batch_size"):
            MCTSConfig(search_batch_size=0)

    def test_validation_num_players(self):
        # NN/MCTS scope is 3-5 players only.
        with pytest.raises(ValueError, match="num_players"):
            MCTSConfig(num_players=2)
        with pytest.raises(ValueError, match="num_players"):
            MCTSConfig(num_players=6)

    def test_validation_c_puct(self):
        with pytest.raises(ValueError, match="c_puct"):
            MCTSConfig(c_puct=-0.1)
        MCTSConfig(c_puct=0)  # boundary is valid

    def test_validation_dirichlet_alpha(self):
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            MCTSConfig(dirichlet_alpha=0)
        with pytest.raises(ValueError, match="dirichlet_alpha"):
            MCTSConfig(dirichlet_alpha=-1)

    def test_validation_dirichlet_epsilon(self):
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            MCTSConfig(dirichlet_epsilon=-0.1)
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            MCTSConfig(dirichlet_epsilon=1.5)
        MCTSConfig(dirichlet_epsilon=0)
        MCTSConfig(dirichlet_epsilon=1)

    def test_validation_dirichlet_alpha_numerator(self):
        with pytest.raises(ValueError, match="dirichlet_alpha_numerator"):
            MCTSConfig(dirichlet_alpha_numerator=0)
        with pytest.raises(ValueError, match="dirichlet_alpha_numerator"):
            MCTSConfig(dirichlet_alpha_numerator=-1)


# ---------------------------------------------------------------------------
# Terminal value computation
# ---------------------------------------------------------------------------

class TestTerminalValues:
    def test_clear_ranking(self):
        # NW: P0=100, P1=300, P2=200. mean=200, max=300, scale=1.5.
        # Rank: P0=-1.0, P1=+1.0, P2=0.0.
        # Margin: P0=1.5*(100-200)/300=-0.5, P1=+0.5, P2=0.0.
        # Blend: [-0.75, +0.75, 0.0].
        vals = compute_terminal_values([100, 300, 200], 3)
        np.testing.assert_array_almost_equal(vals, [-0.75, 0.75, 0.0])

    def test_winner_has_highest_value(self):
        vals = compute_terminal_values([50, 200, 150], 3)
        assert vals[1] > vals[0]
        assert vals[1] > vals[2]

    def test_zero_net_worth_player(self):
        vals = compute_terminal_values([0, 100, 50], 3)
        np.testing.assert_array_almost_equal(vals, [-0.875, 0.875, 0.0])

    def test_tied_winners(self):
        # [200, 200, 100]: p0,p1 tie for 1st, avg(1,0)=0.5, p2 gets -1.
        # Margin: p0,p1=+0.25, p2=-0.5. Blend: [0.375, 0.375, -0.75].
        vals = compute_terminal_values([200, 200, 100], 3)
        np.testing.assert_array_almost_equal(vals, [0.375, 0.375, -0.75])

    def test_all_tied(self):
        vals = compute_terminal_values([100, 100, 100], 3)
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 0.0])

    def test_all_zero(self):
        # No net worth → uniformly zero (nothing to rank on).
        vals = compute_terminal_values([0, 0, 0], 3)
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 0.0])

    def test_continuous_gradient(self):
        """3rd place with higher NW gets a less negative reward."""
        vals_low = compute_terminal_values([100, 243, 261], 3)
        vals_high = compute_terminal_values([150, 243, 261], 3)
        assert vals_high[0] > vals_low[0]

    def test_values_bounded(self):
        for nw in ([500, 1, 0], [100, 100, 100], [1, 1000, 500],
                   [0, 0, 100], [80, 7, 3]):
            vals = compute_terminal_values(nw, 3)
            assert (vals >= -1.0 - 1e-6).all()
            assert (vals <= 1.0 + 1e-6).all()

    def test_rank_sharpness(self):
        """Overtaking a close competitor produces a meaningful reward gap.

        With pure margin the gap would be ~0.01. The rank component
        lifts it well above that.
        """
        vals = compute_terminal_values([200, 101, 100], 3)
        assert vals[1] - vals[2] > 0.4

    def test_zero_sum(self):
        for nw in ([100, 300, 200], [50, 200, 150], [200, 200, 100],
                   [100, 100, 100], [0, 100, 50], [200, 101, 100],
                   [500, 1, 0], [80, 7, 3]):
            vals = compute_terminal_values(nw, 3)
            assert abs(vals.sum()) < 0.01

    def test_rank_weight_extremes(self):
        """rank_weight=0.0 returns pure margin; 1.0 returns pure rank."""
        nw = [100, 300, 200]
        margin = compute_terminal_values(nw, 3, rank_weight=0.0)
        rank = compute_terminal_values(nw, 3, rank_weight=1.0)
        np.testing.assert_array_almost_equal(margin, [-0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(rank, [-1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Sparse expansion (mcts_core.expand_node_sparse)
# ---------------------------------------------------------------------------

class TestExpandNodeSparse:
    def test_basic_expand(self):
        node = MCTSNode(num_players=3)
        action_ids = np.array([7, 13, 42, 100], dtype=np.uint16)
        priors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        default_value = np.array([0.5, -0.3, -0.2], dtype=np.float32)

        expand_node_sparse(node, action_ids, 4, priors, default_value, 3)

        assert node.legal_actions is not None
        assert node.priors is not None
        assert node.visit_counts is not None
        assert node.value_sums is not None
        assert node.default_value is not None
        assert node.legal_actions.dtype == np.int32
        assert node.legal_actions.tolist() == [7, 13, 42, 100]
        assert node.priors.dtype == np.float32
        np.testing.assert_array_almost_equal(node.priors, [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_array_equal(node.visit_counts, np.zeros(4, dtype=np.int32))
        assert node.value_sums.shape == (4, 3)
        # FPU: each row initialized to default_value
        for i in range(4):
            np.testing.assert_array_almost_equal(
                node.value_sums[i], default_value,
            )
        np.testing.assert_array_almost_equal(node.default_value, default_value)

    def test_partial_n_ignores_tail(self):
        """Only the first n entries of action_ids/priors should be read."""
        node = MCTSNode(num_players=3)
        ids = np.array([1, 2, 3, 999, 999], dtype=np.uint16)
        priors = np.array([0.5, 0.3, 0.2, 9.9, 9.9], dtype=np.float32)
        default_value = np.zeros(3, dtype=np.float32)

        expand_node_sparse(node, ids, 3, priors, default_value, 3)

        assert node.legal_actions is not None
        assert node.priors is not None
        assert node.value_sums is not None
        assert node.legal_actions.tolist() == [1, 2, 3]
        np.testing.assert_array_almost_equal(node.priors, [0.5, 0.3, 0.2])
        assert node.value_sums.shape == (3, 3)

    def test_large_action_ids_preserve_full_width(self):
        """uint16 action ids up to ~14976 should round-trip into int32 legal_actions."""
        node = MCTSNode(num_players=3)
        ids = np.array([0, 14976], dtype=np.uint16)
        priors = np.array([0.5, 0.5], dtype=np.float32)
        dv = np.zeros(3, dtype=np.float32)
        expand_node_sparse(node, ids, 2, priors, dv, 3)
        assert node.legal_actions is not None
        assert node.legal_actions.tolist() == [0, 14976]


# ---------------------------------------------------------------------------
# Per-leaf gather / scatter IPC primitives (mcts_core)
# ---------------------------------------------------------------------------
#
# The eval server uses these to assemble a contiguous batch from per-worker
# shared-mem slots and scatter priors/values back. They're in mcts_core.pyx
# because they're on the MCTS hot path; they do not otherwise exercise the
# node or search machinery.

class TestGatherScatter:
    def test_gather_states_concatenates_per_request_rows(self):
        """Each (worker_idx, count) request contributes count rows starting at src[widx, 0]."""
        num_workers, batch, row = 2, 4, 6
        src = np.arange(
            num_workers * batch * row, dtype=np.float32,
        ).reshape(num_workers, batch, row)
        dst = np.zeros((num_workers * batch, row), dtype=np.float32)
        worker_indices = np.array([0, 1], dtype=np.int32)
        counts = np.array([3, 2], dtype=np.int32)

        total = gather_states(dst, src, worker_indices, counts, 2)

        assert total == 5
        np.testing.assert_array_equal(dst[:3], src[0, :3])
        np.testing.assert_array_equal(dst[3:5], src[1, :2])
        # Untouched tail stayed zero (memcpy writes only `total` rows).
        np.testing.assert_array_equal(dst[5:], 0)

    def test_gather_states_empty_requests(self):
        """num_requests=0 is a legal no-op that returns 0."""
        num_workers, batch, row = 2, 2, 3
        src = np.ones((num_workers, batch, row), dtype=np.float32)
        dst = np.zeros((num_workers * batch, row), dtype=np.float32)
        total = gather_states(
            dst, src,
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            0,
        )
        assert total == 0
        np.testing.assert_array_equal(dst, 0)

    def test_gather_phase_ids_respects_worker_order(self):
        """Scheduling order controls the output layout, not worker index."""
        num_workers, batch = 3, 4
        src = np.array(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            dtype=np.int8,
        )
        dst = np.zeros(num_workers * batch, dtype=np.int8)
        worker_indices = np.array([2, 0], dtype=np.int32)
        counts = np.array([3, 2], dtype=np.int32)

        total = gather_phase_ids(dst, src, worker_indices, counts, 2)

        assert total == 5
        assert dst[:3].tolist() == [8, 9, 10]  # worker 2, first 3
        assert dst[3:5].tolist() == [0, 1]     # worker 0, first 2

    def test_gather_action_ids_preserves_int16_values(self):
        """int16 rows copied verbatim; large ids up to 32767 round-trip."""
        num_workers, batch, k_max = 2, 3, 4
        src = np.arange(
            num_workers * batch * k_max, dtype=np.int16,
        ).reshape(num_workers, batch, k_max)
        dst = np.zeros((num_workers * batch, k_max), dtype=np.int16)
        worker_indices = np.array([1, 0], dtype=np.int32)
        counts = np.array([2, 1], dtype=np.int32)

        total = gather_action_ids(dst, src, worker_indices, counts, 2)

        assert total == 3
        np.testing.assert_array_equal(dst[:2], src[1, :2])
        np.testing.assert_array_equal(dst[2:3], src[0, :1])

    def test_gather_action_ids_handles_large_ids(self):
        """uint16-range ids (post-refactor MAX_ACTION_SIZE=14977) fit in int16."""
        k_max = 2
        src = np.array([[[14976, 32767]]], dtype=np.int16)
        dst = np.zeros((1, k_max), dtype=np.int16)
        total = gather_action_ids(
            dst, src,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            1,
        )
        assert total == 1
        assert dst[0, 0] == 14976
        assert dst[0, 1] == 32767

    def test_gather_n_legals_scalar_copy(self):
        """n_legal per leaf is a single int16 per batch slot."""
        num_workers, batch = 2, 3
        src = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)
        dst = np.zeros(num_workers * batch, dtype=np.int16)
        worker_indices = np.array([0, 1], dtype=np.int32)
        counts = np.array([2, 3], dtype=np.int32)

        total = gather_n_legals(dst, src, worker_indices, counts, 2)

        assert total == 5
        assert dst[:2].tolist() == [10, 20]
        assert dst[2:5].tolist() == [40, 50, 60]

    def test_scatter_results_round_trips_priors_and_values(self):
        """scatter_results is the inverse of the gather_* pass for eval outputs."""
        num_workers, batch, k_max, npl = 2, 3, 8, 3
        priors_row_bytes = k_max * 4  # float32

        src_priors_f = np.arange(
            num_workers * batch * k_max, dtype=np.float32,
        ).reshape(num_workers * batch, k_max)
        src_values = np.arange(
            num_workers * batch * npl, dtype=np.float32,
        ).reshape(num_workers * batch, npl)
        dst_priors_f = np.zeros((num_workers, batch, k_max), dtype=np.float32)
        dst_values = np.zeros((num_workers, batch, npl), dtype=np.float32)

        # Byte-level views — scatter_results takes char memoryviews so the
        # caller can plug any float dtype; the contents round-trip unchanged.
        src_priors_b = src_priors_f.view(np.int8).reshape(
            num_workers * batch, priors_row_bytes,
        )
        dst_priors_b = dst_priors_f.view(np.int8).reshape(
            num_workers, batch, priors_row_bytes,
        )

        worker_indices = np.array([1, 0], dtype=np.int32)
        counts = np.array([2, 1], dtype=np.int32)

        scatter_results(
            src_priors_b, src_values, dst_priors_b, dst_values,
            worker_indices, counts, 2, priors_row_bytes,
        )

        # Worker 1 got the first 2 source rows; worker 0 got the next 1.
        np.testing.assert_array_equal(dst_priors_f[1, :2], src_priors_f[:2])
        np.testing.assert_array_equal(dst_values[1, :2], src_values[:2])
        np.testing.assert_array_equal(dst_priors_f[0, :1], src_priors_f[2:3])
        np.testing.assert_array_equal(dst_values[0, :1], src_values[2:3])

    def test_scatter_results_leaves_unused_slots_zero(self):
        """scatter writes only `count` rows per worker; untouched slots stay zero."""
        num_workers, batch, k_max, npl = 2, 3, 4, 3
        priors_row_bytes = k_max * 4

        src_priors_f = np.full(
            (num_workers * batch, k_max), 0.25, dtype=np.float32,
        )
        src_values = np.full(
            (num_workers * batch, npl), 0.5, dtype=np.float32,
        )
        dst_priors_f = np.zeros((num_workers, batch, k_max), dtype=np.float32)
        dst_values = np.zeros((num_workers, batch, npl), dtype=np.float32)

        src_priors_b = src_priors_f.view(np.int8).reshape(
            num_workers * batch, priors_row_bytes,
        )
        dst_priors_b = dst_priors_f.view(np.int8).reshape(
            num_workers, batch, priors_row_bytes,
        )

        # Worker 0 gets only 1 row; slots [1:] must stay zero.
        scatter_results(
            src_priors_b, src_values, dst_priors_b, dst_values,
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32),
            1, priors_row_bytes,
        )
        np.testing.assert_array_equal(dst_priors_f[0, 1:], 0)
        np.testing.assert_array_equal(dst_values[0, 1:], 0)
        # Worker 1 entirely untouched.
        np.testing.assert_array_equal(dst_priors_f[1], 0)
        np.testing.assert_array_equal(dst_values[1], 0)


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_default_construction(self):
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.active_player_id == 0
        assert not node.is_terminal
        assert not node.expanded()
        assert node.state_idx == -1
        assert node.terminal_values is None
        assert node.legal_actions is None
        assert node.priors is None
        assert node.default_value is None
        assert node.visit_counts is None
        assert node.value_sums is None
        assert node._propagation_saved is None

    def test_pending_slots_default(self):
        """Per-leaf phase context slots start cleared."""
        node = MCTSNode()
        assert node.pending_n == 0
        assert node.pending_phase == -1

    def test_num_players_sets_value_sum_shape(self):
        for n in (3, 4, 5):
            node = MCTSNode(num_players=n)
            assert node.value_sum.shape == (n,)
            assert node.value_sum.dtype == np.float32
            assert np.all(node.value_sum == 0.0)

    def test_mean_value_zero_visits(self):
        node = MCTSNode(num_players=3)
        assert node.mean_value(0) == 0.0

    def test_mean_value_accumulation(self):
        node = MCTSNode(num_players=3)
        node.visit_count = 2
        node.value_sum = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        assert node.mean_value(0) == pytest.approx(0.5)
        assert node.mean_value(1) == pytest.approx(0.25)
        assert node.mean_value(2) == pytest.approx(-0.25)

    def test_expand_through_public_api(self):
        """node.expand() delegates to expand_node_sparse (Cython)."""
        node = MCTSNode(num_players=3)
        ids = np.array([3, 7, 11], dtype=np.uint16)
        priors = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        dv = np.array([0.1, -0.2, 0.1], dtype=np.float32)
        node.expand(ids, 3, priors, num_players=3, default_value=dv)

        assert node.expanded()
        assert node.legal_actions is not None
        assert node.priors is not None
        assert node.value_sums is not None
        assert node.legal_actions.tolist() == [3, 7, 11]
        np.testing.assert_array_almost_equal(node.priors, [0.5, 0.3, 0.2])
        assert len(node.children) == 0  # lazy — no child nodes yet
        for i in range(3):
            np.testing.assert_array_almost_equal(node.value_sums[i], dv)

    def test_pending_slots_writable(self):
        """search.py stashes pending_* on leaves between selection + expand."""
        node = MCTSNode()
        node.pending_n = 2
        node.pending_phase = 0  # DPHASE_INVEST
        assert node.pending_n == 2
        assert node.pending_phase == 0


# ---------------------------------------------------------------------------
# PUCT selection
# ---------------------------------------------------------------------------

class TestPUCTSelection:
    def test_selects_highest_prior_when_all_virtual(self):
        """With equal Q (FPU), PUCT picks the highest-prior action."""
        root = _make_expanded_node(
            active_player=0, num_actions=2,
            priors=np.array([0.1, 0.9], dtype=np.float32),
        )
        action, array_idx = select_child(root, c_puct=2.5)
        assert action == 1
        assert array_idx == 1

    def test_exploits_high_value_with_visits(self):
        """With enough visits, PUCT prefers the better Q (exploitation)."""
        default = np.zeros(3, dtype=np.float32)
        vs = np.array([
            [-25.0, 10.0, 15.0],   # Q(p0) = -25/50 = -0.5
            [25.0, -10.0, -15.0],  # Q(p0) = +25/50 = +0.5
        ], dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=2,
            priors=np.array([0.8, 0.2], dtype=np.float32),
            default_value=default,
            visit_counts=np.array([50, 50], dtype=np.int32),
            value_sums=vs,
        )
        action, _ = select_child(root, c_puct=1.0)
        assert action == 1  # player 0 prefers positive Q

    def test_single_legal_action(self):
        """One legal action → always selected."""
        root = _make_expanded_node(active_player=0, num_actions=1)
        root.legal_actions = np.array([7], dtype=np.int32)
        action, array_idx = select_child(root, c_puct=2.5)
        assert action == 7
        assert array_idx == 0

    def test_returns_correct_array_index(self):
        """array_idx indexes into legal_actions (non-contiguous ids)."""
        root = _make_expanded_node(
            active_player=0, num_actions=3,
            priors=np.array([0.1, 0.1, 0.8], dtype=np.float32),
        )
        root.legal_actions = np.array([10, 42, 99], dtype=np.int32)
        action, array_idx = select_child(root, c_puct=2.5)
        assert action == 99
        assert array_idx == 2
        assert root.legal_actions[array_idx] == action

    def test_fpu_prefers_visited_over_unvisited_in_losing_position(self):
        """FPU keeps search from wasting sims on unvisited actions in bad positions.

        Parent value = -0.8 (losing). A visited child at Q=-0.62 beats an
        unvisited child whose FPU = -0.8. Without FPU (default Q=0.0), the
        unvisited child would dominate exploitation.
        """
        default_val = np.array([-0.8, 0.3, 0.5], dtype=np.float32)
        # Action 0: 9 real visits, value_sums = sum of 9 backups = -5.58 → Q=-0.62
        # Action 1: 0 real visits, value_sums still = FPU default
        vs = np.array([
            [-5.58, 2.43, 2.43],  # Q(p0) = -5.58/9 = -0.62
            [-0.8, 0.3, 0.5],     # raw FPU, Q(p0) = -0.8
        ], dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=2,
            priors=np.array([0.5, 0.5], dtype=np.float32),
            default_value=default_val,
            visit_counts=np.array([9, 0], dtype=np.int32),
            value_sums=vs,
        )
        # Tiny c_puct so exploitation dominates.
        action, _ = select_child(root, c_puct=0.001)
        assert action == 0

    def test_player_specific_q(self):
        """Active player chooses based on its own column of value_sums."""
        # Action 0 great for p0, bad for p1. Action 1 reversed.
        vs = np.array([
            [10.0, -10.0, 0.0],  # +1.0 for p0, -1.0 for p1
            [-10.0, 10.0, 0.0],  # reverse
        ], dtype=np.float32)
        priors = np.array([0.5, 0.5], dtype=np.float32)
        visit_counts = np.array([10, 10], dtype=np.int32)
        a0, _ = select_child(
            _make_expanded_node(
                active_player=0, num_actions=2,
                priors=priors, visit_counts=visit_counts, value_sums=vs,
            ),
            c_puct=0.001,
        )
        a1, _ = select_child(
            _make_expanded_node(
                active_player=1, num_actions=2,
                priors=priors, visit_counts=visit_counts, value_sums=vs,
            ),
            c_puct=0.001,
        )
        assert a0 == 0
        assert a1 == 1


# ---------------------------------------------------------------------------
# Dirichlet noise
# ---------------------------------------------------------------------------

class TestDirichletNoise:
    def test_zero_epsilon_leaves_priors_unchanged(self):
        node = MCTSNode(num_players=3)
        node.priors = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        original = node.priors.copy()
        _add_dirichlet_noise(node, alpha=0.3, epsilon=0.0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(node.priors, original)

    def test_noise_modifies_priors_and_keeps_simplex(self):
        node = MCTSNode(num_players=3)
        node.priors = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        original = node.priors.copy()
        _add_dirichlet_noise(node, alpha=0.3, epsilon=0.25, rng=np.random.default_rng(42))
        assert not np.array_equal(node.priors, original)
        assert node.priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert (node.priors >= 0).all()

    def test_seeded_rng_is_reproducible(self):
        results = []
        for _ in range(2):
            node = MCTSNode(num_players=3)
            node.priors = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            _add_dirichlet_noise(node, alpha=0.3, epsilon=0.25,
                                 rng=np.random.default_rng(123))
            results.append(node.priors.copy())
        np.testing.assert_array_equal(results[0], results[1])

    def test_dynamic_alpha_matches_numerator_over_n(self):
        """Dynamic alpha = numerator / legal_count."""
        rng = np.random.default_rng(42)
        numerator = 10.0

        node = MCTSNode(num_players=3)
        node.priors = np.array([0.99, 0.01], dtype=np.float32)
        alpha = numerator / len(node.priors)
        assert alpha == pytest.approx(5.0)
        _add_dirichlet_noise(node, alpha=alpha, epsilon=0.25, rng=rng)
        assert node.priors.sum() == pytest.approx(1.0, abs=1e-5)
        # With alpha=5.0 on 2 actions the noise is nearly uniform, so the
        # previously tiny action gets a big boost.
        assert node.priors[1] > 0.05

    def test_noise_keeps_float32_dtype(self):
        """Priors stay float32 — the Cython select_child asserts this."""
        node = MCTSNode(num_players=3)
        node.priors = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        _add_dirichlet_noise(node, alpha=0.3, epsilon=0.25,
                             rng=np.random.default_rng(0))
        assert node.priors.dtype == np.float32


# ---------------------------------------------------------------------------
# Backup & visit counting (Cython hot paths)
# ---------------------------------------------------------------------------

class TestBackup:
    def test_increment_visits_walks_path_and_leaf(self):
        root = _make_expanded_node(active_player=0, num_actions=2)
        child = MCTSNode(active_player_id=1, num_players=3)
        child.visit_count = 0
        root.children[1] = child  # legal_actions[1] == action id 1

        pre_root = root.visit_count
        path = [(root, 1, 1)]
        increment_visits(path, child)

        assert child.visit_count == 1
        assert root.visit_count == pre_root + 1
        # Only array index 1 touched; index 0 stays at 0 (no real visits).
        assert root.visit_counts is not None
        assert int(root.visit_counts[0]) == 0
        assert int(root.visit_counts[1]) == 1

    def test_backup_first_visit_replaces_fpu(self):
        """On the first real visit, value_sums[array_idx] replaces FPU default."""
        default = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=2, default_value=default,
        )
        child = MCTSNode(active_player_id=1, num_players=3)
        root.children[1] = child

        path = [(root, 1, 1)]
        leaf_values = np.array([0.8, -0.3, -0.5], dtype=np.float32)
        increment_visits(path, child)
        backup(path, child, leaf_values)

        # Child accumulates leaf value
        np.testing.assert_array_almost_equal(child.value_sum, leaf_values)
        # Root value_sum accumulates (root had 0s before)
        np.testing.assert_array_almost_equal(root.value_sum, leaf_values)
        # Edge value_sums[1] REPLACED (first real visit) — NOT added to FPU
        assert root.value_sums is not None
        np.testing.assert_array_almost_equal(root.value_sums[1], leaf_values)
        # Untouched edge still at FPU default
        np.testing.assert_array_almost_equal(root.value_sums[0], default)

    def test_backup_second_visit_accumulates(self):
        """Subsequent visits accumulate into value_sums[array_idx]."""
        default = np.zeros(3, dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=1, default_value=default,
        )
        child = MCTSNode(active_player_id=1, num_players=3)
        root.children[0] = child

        path = [(root, 0, 0)]
        increment_visits(path, child)
        backup(path, child, np.array([0.5, 0.2, -0.7], dtype=np.float32))
        increment_visits(path, child)
        backup(path, child, np.array([0.3, 0.4, -0.7], dtype=np.float32))

        # Two visits → value_sums[0] == sum of both leaf values
        assert root.value_sums is not None
        np.testing.assert_array_almost_equal(
            root.value_sums[0], [0.8, 0.6, -1.4],
        )
        # value_sum accumulates identically
        np.testing.assert_array_almost_equal(root.value_sum, [0.8, 0.6, -1.4])
        assert child.visit_count == 2


class TestVirtualBackup:
    def test_virtual_backup_increments_and_sets_q(self):
        """virtual_backup echoes the child's mean Q to the root edge."""
        default = np.zeros(3, dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=2,
            visit_counts=np.array([0, 0], dtype=np.int32),
            default_value=default,
        )
        # Child has 4 real visits, mean Q = (2.0, -1.0, -1.0)/4 = (0.5, -0.25, -0.25)
        child = MCTSNode(active_player_id=1, num_players=3)
        child.visit_count = 4
        child.value_sum = np.array([2.0, -1.0, -1.0], dtype=np.float32)
        root.children[1] = child

        pre_rvc = root.visit_count
        virtual_backup(root, child, 1)

        assert root.visit_count == pre_rvc + 1
        assert root.visit_counts is not None
        assert root.value_sums is not None
        assert root.visit_counts[1] == 1
        # First virtual visit replaces FPU (vc goes 0→1)
        np.testing.assert_array_almost_equal(
            root.value_sums[1], [0.5, -0.25, -0.25],
        )
        # value_sum accumulated with child Q
        np.testing.assert_array_almost_equal(
            root.value_sum, [0.5, -0.25, -0.25],
        )

    def test_virtual_backup_second_call_accumulates(self):
        """Second virtual_backup adds to value_sums (not replace)."""
        default = np.zeros(3, dtype=np.float32)
        root = _make_expanded_node(
            active_player=0, num_actions=1,
            visit_counts=np.array([0], dtype=np.int32),
            default_value=default,
        )
        child = MCTSNode(active_player_id=1, num_players=3)
        child.visit_count = 2
        child.value_sum = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        root.children[0] = child

        virtual_backup(root, child, 0)
        virtual_backup(root, child, 0)

        # Each virtual_backup adds child Q = (0.5, 0.0, -0.5).
        # First replaces FPU; second accumulates.
        assert root.value_sums is not None
        assert root.visit_counts is not None
        np.testing.assert_array_almost_equal(
            root.value_sums[0], [1.0, 0.0, -1.0],
        )
        assert root.visit_counts[0] == 2


# ---------------------------------------------------------------------------
# StatePool
# ---------------------------------------------------------------------------

class TestStatePool:
    def test_alloc_stores_and_returns_index(self):
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        src = np.arange(total_size, dtype=np.int16)
        idx = pool.alloc(src)
        assert idx == 0
        assert pool._next == 1
        np.testing.assert_array_equal(pool.row(0), src)

    def test_alloc_from_row_copies_existing(self):
        """alloc_from_row copies an existing pool row to a new slot."""
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        src = np.arange(total_size, dtype=np.int16)
        pool.alloc(src)
        idx = pool.alloc_from_row(0)
        assert idx == 1
        np.testing.assert_array_equal(pool.row(1), src)

    def test_reset_zeros_cursor(self):
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        pool.alloc(np.zeros(total_size, dtype=np.int16))
        pool.reset()
        assert pool._next == 0

    def test_row_returns_view(self):
        """row() returns a view, not a copy — writes are visible."""
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        pool.alloc(np.zeros(total_size, dtype=np.int16))
        view = pool.row(0)
        view[3] = 123
        assert pool.states[0, 3] == 123

    def test_compact_preserves_data_and_updates_indices(self):
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        nodes = []
        for i in range(5):
            node = MCTSNode(num_players=3)
            node.state_idx = pool.alloc(
                np.full(total_size, i, dtype=np.int16),
            )
            nodes.append(node)

        # Keep nodes 1 and 3 (ascending order ensures safe in-place copy)
        retained = [nodes[1], nodes[3]]
        pool.compact(retained)

        assert pool._next == 2
        assert nodes[1].state_idx == 0
        assert nodes[3].state_idx == 1
        np.testing.assert_array_equal(pool.row(0), np.full(total_size, 1, dtype=np.int16))
        np.testing.assert_array_equal(pool.row(1), np.full(total_size, 3, dtype=np.int16))

    def test_compact_noop_when_already_contiguous(self):
        total_size = get_layout(3).total_size
        pool = StatePool(8, total_size)
        nodes = []
        for i in range(3):
            node = MCTSNode(num_players=3)
            node.state_idx = pool.alloc(
                np.full(total_size, i, dtype=np.int16),
            )
            nodes.append(node)

        pool.compact(nodes)

        assert pool._next == 3
        for i, node in enumerate(nodes):
            assert node.state_idx == i

    def test_ensure_pending_bufs_grows_on_demand(self):
        total_size = get_layout(3).total_size
        pool = StatePool(4, total_size)
        assert pool._pending_action_ids_buf is None
        pool.ensure_pending_bufs(4)
        assert pool._pending_action_ids_buf is not None
        assert pool._pending_action_ids_buf.shape == (4, K_MAX)
        assert pool._pending_n_buf is not None
        assert pool._pending_n_buf.shape == (4,)

        # Growing the batch reallocates
        pool.ensure_pending_bufs(8)
        assert pool._pending_action_ids_buf.shape == (8, K_MAX)

        # Shrinking doesn't reallocate
        buf = pool._pending_action_ids_buf
        pool.ensure_pending_bufs(2)
        assert pool._pending_action_ids_buf is buf


# ---------------------------------------------------------------------------
# NNEvaluator
# ---------------------------------------------------------------------------

class TestNNEvaluator:
    def test_evaluate_returns_sparse_5_tuple(self, game_state, evaluator):
        priors, values, action_ids, n_legal, phase_id = evaluator.evaluate(game_state)
        assert priors.shape == (n_legal,)
        assert priors.dtype == np.float32
        assert values.shape == (NUM_PLAYERS,)
        assert values.dtype == np.float32
        assert action_ids.shape == (n_legal,)
        assert action_ids.dtype == np.uint16
        assert 0 <= phase_id <= 7  # 8 decision phases

    def test_evaluate_priors_sum_to_one(self, game_state, evaluator):
        priors, _, _, _, _ = evaluator.evaluate(game_state)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert (priors >= 0).all()

    def test_evaluate_values_in_tanh_range(self, game_state, evaluator):
        _, values, _, _, _ = evaluator.evaluate(game_state)
        assert (values >= -1.0 - 1e-5).all()
        assert (values <= 1.0 + 1e-5).all()

    def test_evaluate_phase_id_matches_state(self, game_state, evaluator):
        _, _, _, _, phase_id = evaluator.evaluate(game_state)
        assert phase_id == get_decision_phase_py(game_state)

    def test_evaluate_action_ids_match_enumerator(self, game_state, evaluator):
        _, _, action_ids, n_legal, _ = evaluator.evaluate(game_state)
        scratch = np.empty(K_MAX, dtype=np.uint16)
        expected_n = enumerate_legal_actions_py(game_state, scratch)
        assert n_legal == expected_n
        np.testing.assert_array_equal(action_ids[:n_legal], scratch[:expected_n])

    def test_evaluate_terminal_values(self, evaluator):
        state = GameState(3)
        state.initialize_game(3, seed=42)
        TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        PLAYERS[0].set_net_worth(state, 500)
        PLAYERS[1].set_net_worth(state, 300)
        PLAYERS[2].set_net_worth(state, 100)
        vals = evaluator.evaluate_terminal(state)
        np.testing.assert_array_almost_equal(vals, [0.8, 0.0, -0.8])

    def test_evaluate_batch_empty(self, evaluator):
        assert evaluator.evaluate_batch([]) == []

    def test_evaluate_batch_single_matches_evaluate(self, game_state, evaluator):
        """Batch of 1 returns the same tuple shape as evaluate()."""
        single = evaluator.evaluate(game_state)
        batch = evaluator.evaluate_batch([game_state])
        assert len(batch) == 1
        p_s, v_s, ids_s, n_s, ph_s = single
        p_b, v_b, ids_b, n_b, ph_b = batch[0]
        np.testing.assert_array_almost_equal(p_b, p_s)
        np.testing.assert_array_almost_equal(v_b, v_s)
        np.testing.assert_array_equal(ids_b, ids_s)
        assert n_b == n_s
        assert ph_b == ph_s

    def test_evaluate_batch_multiple_states(self, game_state, evaluator):
        results = evaluator.evaluate_batch([game_state, game_state, game_state])
        assert len(results) == 3
        for priors, values, action_ids, n_legal, phase_id in results:
            assert priors.shape == (n_legal,)
            assert values.shape == (NUM_PLAYERS,)
            assert priors.sum() == pytest.approx(1.0, abs=1e-5)
            assert (values >= -1.0 - 1e-5).all() and (values <= 1.0 + 1e-5).all()

    def test_evaluate_leaves_matches_evaluate(self, game_state, evaluator):
        """evaluate_leaves should agree with evaluate on the same state."""
        priors_ref, values_ref, action_ids, n_legal, phase_id = (
            evaluator.evaluate(game_state)
        )
        # Build the packed inputs the way run_search does.
        state_arrays = [game_state._array]
        phase_ids = [phase_id]
        action_ids_buf = np.zeros((1, K_MAX), dtype=np.uint16)
        action_ids_buf[0, :n_legal] = action_ids
        n_legals = [n_legal]
        leaves = evaluator.evaluate_leaves(
            state_arrays, phase_ids, action_ids_buf, n_legals,
        )
        assert len(leaves) == 1
        priors, values = leaves[0]
        assert priors.shape == (n_legal,)
        np.testing.assert_array_almost_equal(priors, priors_ref)
        np.testing.assert_array_almost_equal(values, values_ref)

    def test_evaluate_leaves_empty(self, evaluator):
        result = evaluator.evaluate_leaves(
            [], [], np.empty((0, K_MAX), dtype=np.uint16), [],
        )
        assert result == []

    def test_model_num_players_mismatch_rejected(self, model):
        """Evaluator-model num_players mismatch raises at construction."""
        # model is 3p; request 4p — must fail early
        with pytest.raises(ValueError, match="num_players"):
            NNEvaluator(model, torch.device("cpu"), num_players=4)


# ---------------------------------------------------------------------------
# fill_token_buffer
# ---------------------------------------------------------------------------

class TestFillTokenBuffer:
    def test_buffer_matches_expected_shape(self, game_state, evaluator):
        """fill_token_buffer writes (num_tokens, token_dim) into its arg."""
        buf = np.zeros(
            (evaluator.num_tokens, evaluator.token_dim), dtype=np.float32,
        )
        fill_token_buffer(game_state, buf)
        # Not all-zero — the initialized state has populated tokens.
        assert np.any(buf != 0)


# ---------------------------------------------------------------------------
# Full search
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_search_visit_count_is_one_plus_sims(self, search_root):
        root, _ = search_root
        # 1 initial root eval + num_simulations leaf evals
        assert root.visit_count == 21
        assert root.expanded()
        assert len(root.children) > 0

    def test_probabilities_sum_to_one_and_nonneg(self, search_root):
        root, _ = search_root
        probs = get_action_probabilities(root, temperature=1.0)
        assert probs.shape == (int(MAX_ACTION_SIZE),)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

    def test_probabilities_greedy_puts_all_mass_on_one_action(self, search_root):
        root, _ = search_root
        probs = get_action_probabilities(root, temperature=0.0)
        assert probs.shape == (int(MAX_ACTION_SIZE),)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs == 1.0).sum() == 1

    def test_probabilities_zero_on_illegal_actions(self, search_root):
        """Dense probs vector has mass only at sparse legal ids."""
        root, _ = search_root
        probs = get_action_probabilities(root, temperature=1.0)
        assert root.legal_actions is not None
        legal_mask = np.zeros(int(MAX_ACTION_SIZE), dtype=bool)
        legal_mask[root.legal_actions] = True
        assert (probs[~legal_mask] == 0).all()

    def test_greedy_leaf_value_bounded(self, search_root_deep):
        root, _ = search_root_deep
        val = get_greedy_leaf_value(root, num_players=NUM_PLAYERS)
        assert val.shape == (NUM_PLAYERS,)
        assert (val >= -1.0).all() and (val <= 1.0).all()

    def test_greedy_leaf_follows_max_visits(self, search_root_deep):
        """A0GB traversal should follow the most-visited child at each level."""
        root, _ = search_root_deep
        node = root
        while node.expanded() and not node.is_terminal:
            assert node.visit_counts is not None and node.legal_actions is not None
            best_idx = int(np.argmax(node.visit_counts))
            if node.visit_counts[best_idx] == 0:
                break
            best_action = int(node.legal_actions[best_idx])
            if best_action not in node.children:
                break
            node = node.children[best_action]

        expected = node.value_sum / node.visit_count
        actual = get_greedy_leaf_value(root, num_players=NUM_PLAYERS)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_nodes_have_pool_state_indices(self, search_root):
        """Every visited node has a valid state_idx into the pool."""
        root, _ = search_root
        assert root.state_idx == 0  # root is the first alloc
        for child in root.children.values():
            assert child.state_idx >= 0

    def test_lazy_expansion_has_fewer_children_than_legal(self, search_root):
        """Children dict is sparse — only visited legal actions appear."""
        root, _ = search_root
        assert root.legal_actions is not None
        assert len(root.children) <= len(root.legal_actions)
        assert len(root.children) <= 20  # sim budget bound

    def test_pending_slots_cleared_post_expand(self, search_root):
        """After expand, pending_* slots are reset on every non-terminal node."""
        root, _ = search_root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.expanded() and not node.is_terminal:
                assert node.pending_n == 0
                assert node.pending_phase == -1
            stack.extend(node.children.values())

    def test_visit_count_invariant(self, search_root):
        """For every expanded node: visit_count == 1 + sum(visit_counts) - len(legal_actions) * 0.

        Children get 1 FPU virtual visit apiece on expand; real visits add to
        that. The Cython select_child treats FPU visits identically to real
        ones for the PUCT denominator, so `sum(visit_counts) + 1` equals
        `visit_count` on a fully-expanded node.
        """
        root, _ = search_root
        def check(node):
            if not node.expanded():
                return
            assert node.visit_counts is not None
            child_total = int(node.visit_counts.sum())
            assert node.visit_count == 1 + child_total, (
                f"visit_count={node.visit_count} != "
                f"1 + sum(child_visits)={1 + child_total}"
            )
            for c in node.children.values():
                check(c)
        check(root)

    def test_terminal_root_returns_immediately(self, evaluator):
        """Search on a game-over root returns with just the terminal eval."""
        state = GameState(3)
        state.initialize_game(3, seed=42)
        TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        PLAYERS[0].set_net_worth(state, 500)
        PLAYERS[1].set_net_worth(state, 300)
        PLAYERS[2].set_net_worth(state, 100)

        config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)
        root = run_search(state, evaluator, config)

        assert root.is_terminal
        assert not root.expanded()
        assert root.visit_count == 1
        assert root.terminal_values is not None
        np.testing.assert_array_almost_equal(
            root.terminal_values, [0.8, 0.0, -0.8],
        )

    def test_run_search_requires_root_state_when_no_reuse(self, evaluator):
        """Fresh search with no root_state and no reuse raises."""
        config = MCTSConfig(num_simulations=5, num_players=NUM_PLAYERS)
        with pytest.raises(ValueError, match="root_state"):
            run_search(None, evaluator, config)

    def test_run_search_requires_pool_with_reuse(self, game_state, evaluator):
        """reuse_root without a state_pool should fail fast."""
        config = MCTSConfig(num_simulations=5, num_players=NUM_PLAYERS)
        dummy = MCTSNode(num_players=NUM_PLAYERS)
        with pytest.raises(ValueError, match="state_pool"):
            run_search(
                game_state, evaluator, config, reuse_root=dummy,
                state_pool=None,
            )


# ---------------------------------------------------------------------------
# Batched search
# ---------------------------------------------------------------------------

class TestBatchedSearch:
    @pytest.mark.parametrize("bs", [1, 2, 4, 8])
    def test_visit_count_stable_across_batch_sizes(
        self, game_state, evaluator, bs,
    ):
        config = MCTSConfig(
            num_simulations=20, search_batch_size=bs, num_players=NUM_PLAYERS,
        )
        root = run_search(game_state, evaluator, config)
        assert root.visit_count == 21

    def test_batched_action_probs_valid(self, game_state, evaluator):
        config = MCTSConfig(
            num_simulations=20, search_batch_size=4, num_players=NUM_PLAYERS,
        )
        root = run_search(game_state, evaluator, config)
        probs = get_action_probabilities(root, temperature=1.0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

    def test_large_batch_clamped_by_num_sims(self, game_state, evaluator):
        """batch_size > num_simulations still completes."""
        config = MCTSConfig(
            num_simulations=5, search_batch_size=16, num_players=NUM_PLAYERS,
        )
        root = run_search(game_state, evaluator, config)
        assert root.visit_count == 6

    def test_batched_visit_count_invariant(self, game_state, evaluator):
        """Visit count invariant holds across batched search too."""
        for bs in (2, 4, 8, 16):
            config = MCTSConfig(
                num_simulations=40, search_batch_size=bs,
                num_players=NUM_PLAYERS,
            )
            root = run_search(game_state, evaluator, config)
            assert root.visit_count == 41

            def check(node):
                if not node.expanded():
                    return
                assert node.visit_counts is not None
                total = int(node.visit_counts.sum())
                assert node.visit_count == 1 + total
                for c in node.children.values():
                    check(c)
            check(root)

    def test_no_duplicate_states_per_batch(self, game_state, evaluator):
        """No two leaves in the same eval batch share a pool row."""

        class DupTracker:
            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.num_tokens = inner.num_tokens
                self.token_dim = inner.token_dim
                self.found_dup = False

            def evaluate(self, state):
                return self._inner.evaluate(state)

            def evaluate_leaves(
                self, state_arrays, phase_ids, action_ids_buf, n_legals,
            ):
                addrs = [a.ctypes.data for a in state_arrays]
                if len(addrs) != len(set(addrs)):
                    self.found_dup = True
                return self._inner.evaluate_leaves(
                    state_arrays, phase_ids, action_ids_buf, n_legals,
                )

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        tracker = DupTracker(evaluator)
        config = MCTSConfig(
            num_simulations=40, search_batch_size=16,
            dirichlet_epsilon=0.0, num_players=NUM_PLAYERS,
        )
        run_search(game_state, tracker, config)
        assert not tracker.found_dup, "Duplicate leaf in a batch"

    def test_leaf_lock_caps_first_batch_to_root_frontier(self, game_state, evaluator):
        """With batch_size > #legal at root, the first batch is the frontier size."""

        class BatchSizeTracker:
            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.num_tokens = inner.num_tokens
                self.token_dim = inner.token_dim
                self.sizes: list[int] = []

            def evaluate(self, state):
                return self._inner.evaluate(state)

            def evaluate_leaves(
                self, state_arrays, phase_ids, action_ids_buf, n_legals,
            ):
                self.sizes.append(len(state_arrays))
                return self._inner.evaluate_leaves(
                    state_arrays, phase_ids, action_ids_buf, n_legals,
                )

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        scratch = np.empty(K_MAX, dtype=np.uint16)
        legal_count = int(enumerate_legal_actions_py(game_state, scratch))
        tracker = BatchSizeTracker(evaluator)
        config = MCTSConfig(
            num_simulations=legal_count,
            search_batch_size=legal_count + 16,
            dirichlet_epsilon=0.0, num_players=NUM_PLAYERS,
        )
        run_search(game_state, tracker, config)
        assert len(tracker.sizes) >= 1
        # First batch fills up to the frontier size; propagation locks the
        # root and the batch ships.
        assert tracker.sizes[0] == legal_count


# ---------------------------------------------------------------------------
# Propagation lock / unlock
# ---------------------------------------------------------------------------

class TestPropagationLock:
    @staticmethod
    def _neg_inf():
        return np.full(NUM_PLAYERS, -np.inf, dtype=np.float32)

    def test_no_propagation_when_siblings_unlocked(self):
        """Locking one of two parent edges should NOT propagate."""
        neg_inf = self._neg_inf()
        root = _make_expanded_node(active_player=0, num_actions=2)
        parent = _make_expanded_node(
            active_player=1, num_actions=2,
            value_sums=np.array([
                [0.1, -0.05, -0.05], [0.2, -0.1, -0.1],
            ], dtype=np.float32),
        )
        root.children[0] = parent

        assert parent.value_sums is not None
        assert root.value_sums is not None
        parent.value_sums[0] = neg_inf
        _propagate_lock([(root, 0, 0), (parent, 0, 0)], neg_inf)

        assert root._propagation_saved is None
        assert not np.isinf(root.value_sums[0, 0])

    def test_propagation_when_all_siblings_locked(self):
        neg_inf = self._neg_inf()
        root = _make_expanded_node(active_player=0, num_actions=1)
        parent = _make_expanded_node(active_player=1, num_actions=2)
        root.children[0] = parent
        assert root.value_sums is not None
        assert parent.value_sums is not None
        root_q_before = root.value_sums[0].copy()

        parent.value_sums[0] = neg_inf
        _propagate_lock([(root, 0, 0), (parent, 0, 0)], neg_inf)
        assert root._propagation_saved is None  # 1 of 2 locked

        parent.value_sums[1] = neg_inf
        _propagate_lock([(root, 0, 0), (parent, 1, 1)], neg_inf)

        assert root._propagation_saved is not None
        assert 0 in root._propagation_saved
        np.testing.assert_array_equal(
            root._propagation_saved[0], root_q_before,
        )
        assert root.value_sums[0, 0] == -np.inf

    def test_multi_level_propagation_cascades(self):
        neg_inf = self._neg_inf()
        root = _make_expanded_node(active_player=0, num_actions=1)
        mid = _make_expanded_node(active_player=1, num_actions=1)
        parent = _make_expanded_node(active_player=0, num_actions=2)
        root.children[0] = mid
        mid.children[0] = parent

        assert parent.value_sums is not None
        assert root.value_sums is not None
        parent.value_sums[0] = neg_inf
        _propagate_lock(
            [(root, 0, 0), (mid, 0, 0), (parent, 0, 0)], neg_inf,
        )
        assert mid._propagation_saved is None

        parent.value_sums[1] = neg_inf
        _propagate_lock(
            [(root, 0, 0), (mid, 0, 0), (parent, 1, 1)], neg_inf,
        )
        assert mid._propagation_saved is not None
        assert 0 in mid._propagation_saved
        assert root._propagation_saved is not None
        assert 0 in root._propagation_saved
        assert root.value_sums[0, 0] == -np.inf

    def test_unlock_restores_values(self):
        neg_inf = self._neg_inf()
        root = _make_expanded_node(active_player=0, num_actions=1)
        parent = _make_expanded_node(
            active_player=1, num_actions=2,
            value_sums=np.array([
                [0.3, -0.1, -0.2], [0.5, -0.2, -0.3],
            ], dtype=np.float32),
        )
        root.children[0] = parent
        assert root.value_sums is not None
        assert parent.value_sums is not None
        root_q_before = root.value_sums[0].copy()
        parent_q_a = parent.value_sums[0].copy()

        path_a = [(root, 0, 0), (parent, 0, 0)]
        path_b = [(root, 0, 0), (parent, 1, 1)]
        parent.value_sums[0] = neg_inf
        _propagate_lock(path_a, neg_inf)
        parent.value_sums[1] = neg_inf
        _propagate_lock(path_b, neg_inf)
        assert root.value_sums[0, 0] == -np.inf

        # Unlock A: restore parent edge, then the propagation unlock
        parent.value_sums[0] = parent_q_a
        _propagate_unlock(path_a)

        np.testing.assert_array_equal(root.value_sums[0], root_q_before)
        assert root._propagation_saved is None
        # Parent B still locked (not restored by A's unlock)
        assert parent.value_sums[1, 0] == -np.inf

    def test_unlock_order_independent(self):
        """Unlocking in either order arrives at the same final state."""
        neg_inf = self._neg_inf()
        for order in [(0, 1), (1, 0)]:
            root = _make_expanded_node(active_player=0, num_actions=1)
            parent = _make_expanded_node(
                active_player=1, num_actions=2,
                value_sums=np.array([
                    [0.3, -0.1, -0.2], [0.5, -0.2, -0.3],
                ], dtype=np.float32),
            )
            root.children[0] = parent
            assert root.value_sums is not None
            assert parent.value_sums is not None
            root.value_sums[0] = np.array(
                [0.4, -0.2, -0.2], dtype=np.float32,
            )
            root_q_orig = root.value_sums[0].copy()
            parent_qs = [parent.value_sums[i].copy() for i in range(2)]

            paths = [
                [(root, 0, 0), (parent, 0, 0)],
                [(root, 0, 0), (parent, 1, 1)],
            ]
            for p in paths:
                aidx = p[-1][2]
                parent.value_sums[aidx] = neg_inf
                _propagate_lock(p, neg_inf)
            assert root.value_sums[0, 0] == -np.inf

            for i in order:
                parent.value_sums[paths[i][-1][2]] = parent_qs[i]
                _propagate_unlock(paths[i])

            np.testing.assert_array_equal(
                root.value_sums[0], root_q_orig,
            )
            assert root._propagation_saved is None
            for j in range(2):
                np.testing.assert_array_equal(
                    parent.value_sums[j], parent_qs[j],
                )

    def test_no_residual_saves_after_search(self, game_state, evaluator):
        """No node should leak a propagation save after run_search returns."""
        config = MCTSConfig(
            num_simulations=50, search_batch_size=8, num_players=NUM_PLAYERS,
        )
        root = run_search(game_state, evaluator, config)

        stack = [root]
        while stack:
            node = stack.pop()
            assert node._propagation_saved is None
            stack.extend(node.children.values())


# ---------------------------------------------------------------------------
# Subtree reuse
# ---------------------------------------------------------------------------

class TestSubtreeReuse:
    def test_collect_subtree_nodes_sorted_by_state_idx(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=30, num_players=NUM_PLAYERS)
        root = run_search(game_state, evaluator, config)
        nodes = _collect_subtree_nodes(root)
        for n in nodes:
            assert n.state_idx >= 0
        indices = [n.state_idx for n in nodes]
        assert indices == sorted(indices)
        assert root in nodes

    def test_reset_root_for_reuse_zeros_visits_preserves_children(self):
        """_reset_root_for_reuse zeros visit stats but keeps children + default."""
        default = np.array([0.1, -0.1, 0.0], dtype=np.float32)
        node = _make_expanded_node(
            active_player=0, num_actions=3, default_value=default,
            visit_counts=np.array([5, 2, 0], dtype=np.int32),
            value_sums=np.array([
                [1.0, 0.5, -0.5],
                [0.2, -0.1, -0.1],
                default,
            ], dtype=np.float32),
        )
        # Simulate post-search visit totals
        node.visit_count = 1 + 5 + 2 + 0
        node.value_sum = np.array([0.7, 0.1, -0.1], dtype=np.float32)
        # Add some bogus child entries to confirm they survive
        child_a = MCTSNode(num_players=3)
        child_a.visit_count = 5
        node.children[0] = child_a

        _reset_root_for_reuse(node)

        assert node.visit_count == 1
        assert node.visit_counts is not None
        assert (node.visit_counts == 0).all()
        assert node.value_sums is not None and node.default_value is not None
        assert node.legal_actions is not None
        # FPU broadcast
        for i in range(len(node.legal_actions)):
            np.testing.assert_array_equal(
                node.value_sums[i], node.default_value,
            )
        np.testing.assert_array_equal(node.value_sum, default)
        # Children preserved
        assert 0 in node.children
        assert node.children[0] is child_a
        assert child_a.visit_count == 5

    def test_prepare_reuse_root_none_for_missing_action(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=5, num_players=NUM_PLAYERS)
        total_size = get_layout(3).total_size
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)
        assert prepare_reuse_root(root, 99999, pool) is None

    def test_prepare_reuse_root_returns_child_with_reset_stats(
        self, game_state, evaluator,
    ):
        config = MCTSConfig(num_simulations=50, num_players=NUM_PLAYERS)
        total_size = get_layout(3).total_size
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])
        old_children = dict(root.children[best_action].children)

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None
        assert reuse.expanded()
        assert reuse.visit_count == 1
        assert reuse.visit_counts is not None
        assert (reuse.visit_counts == 0).all()
        # Child dict preserved
        assert reuse.children == old_children
        # Pool compacted to exactly the subtree size
        subtree_size = len(_collect_subtree_nodes(reuse))
        assert pool._next == subtree_size

    def test_prepare_reuse_root_none_for_terminal_child(self, game_state):
        """If the chosen child is terminal, no reuse is possible."""
        total_size = get_layout(3).total_size
        pool = StatePool(4, total_size)

        root = MCTSNode(num_players=NUM_PLAYERS)
        root.state_idx = pool.alloc(game_state._array)
        root.expand(
            np.array([7], dtype=np.uint16),
            1,
            np.array([1.0], dtype=np.float32),
            NUM_PLAYERS,
            np.zeros(NUM_PLAYERS, dtype=np.float32),
        )
        terminal_child = MCTSNode(num_players=NUM_PLAYERS, is_terminal=True)
        terminal_child.terminal_values = np.zeros(NUM_PLAYERS, dtype=np.float32)
        terminal_child.state_idx = pool.alloc_from_row(root.state_idx)
        root.children[7] = terminal_child

        assert prepare_reuse_root(root, 7, pool) is None


    def test_reuse_produces_valid_tree(self, game_state, evaluator):
        """End-to-end: search, reuse, search again, check new root."""
        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        next_state = GameState.from_array(game_state._array, NUM_PLAYERS)
        DRIVER.apply_action(next_state, best_action)

        root2 = run_search(
            next_state, evaluator, config, state_pool=pool, reuse_root=reuse,
        )
        assert root2 is reuse
        # Full sim budget after reset (virtual backups + real search)
        assert root2.visit_count >= config.num_simulations

        probs = get_action_probabilities(root2, temperature=1.0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        val = get_greedy_leaf_value(root2, num_players=NUM_PLAYERS)
        assert val.shape == (NUM_PLAYERS,)
        assert (val >= -1.0).all() and (val <= 1.0).all()

    def test_reuse_saves_nn_evaluations(self, game_state, evaluator):
        """Reuse should reduce NN forward passes vs. a fresh search."""

        class EvalCounter:
            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.num_tokens = inner.num_tokens
                self.token_dim = inner.token_dim
                self.count = 0

            def evaluate(self, state):
                self.count += 1
                return self._inner.evaluate(state)

            def evaluate_leaves(
                self, state_arrays, phase_ids, action_ids_buf, n_legals,
            ):
                self.count += len(state_arrays)
                return self._inner.evaluate_leaves(
                    state_arrays, phase_ids, action_ids_buf, n_legals,
                )

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)

        # Fresh baseline
        counter_fresh = EvalCounter(evaluator)
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, counter_fresh, config, state_pool=pool)
        fresh_evals = counter_fresh.count

        # Reuse
        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])
        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None
        next_state = GameState.from_array(game_state._array, NUM_PLAYERS)
        DRIVER.apply_action(next_state, best_action)

        counter_reuse = EvalCounter(evaluator)
        run_search(
            next_state, counter_reuse, config, state_pool=pool, reuse_root=reuse,
        )
        assert counter_reuse.count < fresh_evals, (
            f"Reuse ({counter_reuse.count}) should save vs fresh ({fresh_evals})"
        )

    def test_reuse_with_batched_search(self, game_state, evaluator):
        """Subtree reuse should work alongside batched leaf evaluation."""
        total_size = get_layout(3).total_size
        config = MCTSConfig(
            num_simulations=50, search_batch_size=4, num_players=NUM_PLAYERS,
        )
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])
        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        next_state = GameState.from_array(game_state._array, NUM_PLAYERS)
        DRIVER.apply_action(next_state, best_action)
        root2 = run_search(
            next_state, evaluator, config, state_pool=pool, reuse_root=reuse,
        )
        assert root2.visit_count >= config.num_simulations

    def test_multi_move_reuse_chain(self, game_state, evaluator):
        """Chained reuse over several moves exercises pool compaction."""
        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=50, num_players=NUM_PLAYERS)
        # 2*(num_sims+1) is the minimum safe pool size under reuse
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        rng = np.random.default_rng(42)

        state = GameState.from_array(game_state._array, NUM_PLAYERS)
        reuse_root = None
        moves_played = 0
        for _ in range(5):
            root = run_search(
                state, evaluator, config, rng=rng,
                state_pool=pool, reuse_root=reuse_root,
            )
            probs = get_action_probabilities(root, temperature=1.0)
            action = int(rng.choice(int(MAX_ACTION_SIZE), p=probs))
            status = DRIVER.apply_action(state, action)
            moves_played += 1
            if status == 2:  # STATUS_GAME_OVER
                break
            reuse_root = prepare_reuse_root(root, action, pool)

        assert moves_played >= 2

    def test_virtual_backup_matches_child_q_after_reuse(
        self, game_state, evaluator,
    ):
        """After reuse, root Q for caught-up actions should match child Q."""
        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)
        pool = StatePool(2 * (config.num_simulations + 1), total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])
        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        # Snapshot child Q before second search
        pre_child_qs = {
            a: c.value_sum.copy() / c.visit_count
            for a, c in reuse.children.items() if c.visit_count > 0
        }

        next_state = GameState.from_array(game_state._array, NUM_PLAYERS)
        DRIVER.apply_action(next_state, best_action)
        root2 = run_search(
            next_state, evaluator, config, state_pool=pool, reuse_root=reuse,
        )
        assert root2.visit_counts is not None and root2.value_sums is not None
        assert root2.legal_actions is not None

        for i, action in enumerate(root2.legal_actions):
            a = int(action)
            if a not in root2.children or a not in pre_child_qs:
                continue
            child = root2.children[a]
            if root2.visit_counts[i] == 0:
                continue
            root_q = root2.value_sums[i] / root2.visit_counts[i]
            child_q = child.value_sum / child.visit_count
            np.testing.assert_allclose(root_q, child_q, atol=0.01)

    def test_reuse_noise_changes_visit_distribution(self, game_state, evaluator):
        """Different Dirichlet seeds produce different post-reuse distributions."""
        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=100, num_players=NUM_PLAYERS)

        dists = []
        for seed in (42, 123):
            pool = StatePool(
                2 * (config.num_simulations + 1), total_size,
            )
            root = run_search(
                game_state, evaluator, config, state_pool=pool,
                rng=np.random.default_rng(seed),
            )
            assert root.visit_counts is not None and root.legal_actions is not None
            best_idx = int(np.argmax(root.visit_counts))
            best_action = int(root.legal_actions[best_idx])
            reuse = prepare_reuse_root(root, best_action, pool)
            assert reuse is not None

            next_state = GameState.from_array(game_state._array, NUM_PLAYERS)
            DRIVER.apply_action(next_state, best_action)
            root2 = run_search(
                next_state, evaluator, config, state_pool=pool,
                reuse_root=reuse, rng=np.random.default_rng(seed + 1000),
            )
            dists.append(get_action_probabilities(root2, temperature=1.0))

        l1 = np.abs(dists[0] - dists[1]).sum()
        assert l1 > 0.05, f"Reuse should see different noise (L1={l1:.4f})"
