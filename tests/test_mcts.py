"""Tests for MCTS search implementation."""

import numpy as np
import pytest
import torch

from core.actions import get_total_action_count, get_valid_action_mask
from core.data import GamePhases
from core.state import GameState
from entities.company import COMPANIES
from entities.turn import TURN
from train.config import MCTSConfig
from mcts.evaluator import (
    NNEvaluator,
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
    unrotate_values,
)
from mcts.node import MCTSNode
from mcts.search import (
    StatePool,
    _add_dirichlet_noise,
    _backup,
    _collect_subtree_nodes,
    get_action_probabilities,
    get_greedy_leaf_value,
    prepare_reuse_root,
    run_search,
    select_child,
)
from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig

# Computed action dimensions (single source of truth)
_ACT_3P = get_total_action_count(3)
_ACT_6P = get_total_action_count(6)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layout():
    return get_layout(3)


@pytest.fixture(scope="session")
def model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _layout = get_layout(3)
    cfg = RSSModelConfig(input_dim=_layout.visible_size)
    return RSSAlphaZeroNet(cfg).to(device)


@pytest.fixture(scope="session")
def evaluator(model):
    device = next(model.parameters()).device
    return NNEvaluator(model, device, num_players=3)


@pytest.fixture
def search_root(game_state, evaluator):
    """Root node after 20 simulations."""
    config = MCTSConfig(num_simulations=20)
    root = run_search(game_state, evaluator, config)
    return root, config


@pytest.fixture
def search_root_deep(game_state, evaluator):
    """Root node after 100 simulations for deeper exploration."""
    config = MCTSConfig(num_simulations=100)
    root = run_search(game_state, evaluator, config)
    return root, config


# ---------------------------------------------------------------------------
# MCTSConfig
# ---------------------------------------------------------------------------

class TestMCTSConfig:
    def test_action_dim_3p(self):
        cfg = MCTSConfig(num_players=3)
        assert cfg.action_dim == _ACT_3P

    def test_action_dim_6p(self):
        cfg = MCTSConfig(num_players=6)
        assert cfg.action_dim == _ACT_6P

    def test_defaults(self):
        cfg = MCTSConfig()
        assert cfg.num_simulations == 800
        assert cfg.c_puct == 2.5
        assert cfg.dirichlet_alpha == 0.8
        assert cfg.dirichlet_epsilon == 0.25
        assert cfg.dirichlet_dynamic is False
        assert cfg.dirichlet_alpha_numerator == 10.0
        assert cfg.num_players == 3
        assert cfg.search_batch_size == 1

    def test_validation_num_simulations(self):
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=0)
        with pytest.raises(ValueError, match="num_simulations"):
            MCTSConfig(num_simulations=-1)

    def test_validation_search_batch_size(self):
        with pytest.raises(ValueError, match="search_batch_size"):
            MCTSConfig(search_batch_size=0)

    def test_validation_num_players(self):
        with pytest.raises(ValueError, match="num_players"):
            MCTSConfig(num_players=1)

    def test_validation_c_puct(self):
        with pytest.raises(ValueError, match="c_puct"):
            MCTSConfig(c_puct=-0.1)
        MCTSConfig(c_puct=0)  # zero is valid

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
        MCTSConfig(dirichlet_epsilon=0)  # boundary valid
        MCTSConfig(dirichlet_epsilon=1)  # boundary valid

    def test_validation_dirichlet_alpha_numerator(self):
        with pytest.raises(ValueError, match="dirichlet_alpha_numerator"):
            MCTSConfig(dirichlet_alpha_numerator=0)
        with pytest.raises(ValueError, match="dirichlet_alpha_numerator"):
            MCTSConfig(dirichlet_alpha_numerator=-1)


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

class TestLayout:
    # Visible-size golden values are tested in test_state_layout.py.
    # Here we just check structural consistency.

    def test_visible_plus_hidden_equals_total(self):
        for n in [2, 3, 6]:
            layout = get_layout(n)
            assert layout.visible_size + layout.hidden_size == layout.total_size

    def test_player_stride_3p(self):
        layout = get_layout(3)
        assert layout.player_stride == 67  # 64 + num_players


# ---------------------------------------------------------------------------
# State rotation
# ---------------------------------------------------------------------------

class TestStateRotation:
    def test_no_rotation_for_player_0(self, game_state, layout):
        visible = game_state._array[:layout.visible_size].copy()
        rotated = rotate_visible_state(game_state._array, 0, 3)
        np.testing.assert_array_equal(visible, rotated)

    def test_player_blocks_rotated_by_1(self, game_state, layout):
        visible = game_state._array[:layout.visible_size]
        p_off = layout.players_offset
        stride = layout.player_stride
        p0 = visible[p_off:p_off + stride].copy()
        p1 = visible[p_off + stride:p_off + 2 * stride].copy()
        p2 = visible[p_off + 2 * stride:p_off + 3 * stride].copy()

        rotated = rotate_visible_state(game_state._array, 1, 3)

        # After rotation by 1: slot 0=P1, slot 1=P2, slot 2=P0
        np.testing.assert_array_equal(
            rotated[p_off:p_off + stride], p1
        )
        np.testing.assert_array_equal(
            rotated[p_off + stride:p_off + 2 * stride], p2
        )
        np.testing.assert_array_equal(
            rotated[p_off + 2 * stride:p_off + 3 * stride], p0
        )

    def test_player_blocks_rotated_by_2(self, game_state, layout):
        visible = game_state._array[:layout.visible_size]
        p_off = layout.players_offset
        stride = layout.player_stride
        p0 = visible[p_off:p_off + stride].copy()
        p1 = visible[p_off + stride:p_off + 2 * stride].copy()
        p2 = visible[p_off + 2 * stride:p_off + 3 * stride].copy()

        rotated = rotate_visible_state(game_state._array, 2, 3)

        # After rotation by 2: slot 0=P2, slot 1=P0, slot 2=P1
        np.testing.assert_array_equal(
            rotated[p_off:p_off + stride], p2
        )
        np.testing.assert_array_equal(
            rotated[p_off + stride:p_off + 2 * stride], p0
        )
        np.testing.assert_array_equal(
            rotated[p_off + 2 * stride:p_off + 3 * stride], p1
        )

    def test_non_player_data_unchanged(self, game_state, layout):
        visible = game_state._array[:layout.visible_size].copy()
        rotated = rotate_visible_state(game_state._array, 1, 3)

        # Phase + CoO (before players)
        np.testing.assert_array_equal(
            visible[:layout.players_offset],
            rotated[:layout.players_offset],
        )

        # FI and everything after players
        end_players = layout.players_offset + layout.players_size
        # Exclude turn state per-player fields from comparison
        fi_to_turn = visible[end_players:layout.turn_offset]
        fi_to_turn_rot = rotated[end_players:layout.turn_offset]
        np.testing.assert_array_equal(fi_to_turn, fi_to_turn_rot)

    def test_turn_per_player_fields_rotated(self, game_state, layout):
        """Verify auction_high_bidder, auction_starter, auction_passed are rotated."""
        # Set distinguishable per-player values so rotation is detectable
        # (default zeros would make np.roll a no-op)
        for field_offset in (layout.auction_high_bidder_offset,
                             layout.auction_starter_offset,
                             layout.auction_passed_offset):
            game_state._array[field_offset:field_offset + 3] = [1.0, 2.0, 3.0]

        for field_offset in (layout.auction_high_bidder_offset,
                             layout.auction_starter_offset,
                             layout.auction_passed_offset):
            rotated = rotate_visible_state(game_state._array, 1, 3)
            rotated_field = rotated[field_offset:field_offset + 3]
            # After rotation by 1: [1, 2, 3] -> [2, 3, 1]
            np.testing.assert_array_equal(rotated_field, [2.0, 3.0, 1.0])


# ---------------------------------------------------------------------------
# Value rotation
# ---------------------------------------------------------------------------

class TestValueRotation:
    def test_unrotate_player_0_identity(self):
        vals = np.array([0.5, 0.0, -0.5])
        result = unrotate_values(vals, 0)
        np.testing.assert_array_almost_equal(result, vals)

    def test_unrotate_player_1(self):
        # NN output: [v_p1, v_p2, v_p0]
        vals = np.array([0.5, 0.0, -0.5])
        result = unrotate_values(vals, 1)
        # Canonical: [v_p0, v_p1, v_p2] = [-0.5, 0.5, 0.0]
        np.testing.assert_array_almost_equal(result, [-0.5, 0.5, 0.0])

    def test_unrotate_player_2(self):
        vals = np.array([0.5, 0.0, -0.5])
        result = unrotate_values(vals, 2)
        # Canonical: [v_p0, v_p1, v_p2] = [0.0, -0.5, 0.5]
        np.testing.assert_array_almost_equal(result, [0.0, -0.5, 0.5])

    def test_round_trip(self):
        """rotate then unrotate should give back original canonical values."""
        canonical = np.array([1.0, 0.0, -1.0])
        for player_id in range(3):
            rotated = np.roll(canonical, -player_id)  # simulate rotation
            unrotated = unrotate_values(rotated, player_id)
            np.testing.assert_array_almost_equal(unrotated, canonical)


# ---------------------------------------------------------------------------
# Terminal values
# ---------------------------------------------------------------------------

class TestTerminalValues:
    def test_clear_ranking(self):
        # NW: P0=100, P1=300, P2=200, mean=200, max=300, scale=1.5
        # Rank: P0=-1.0, P1=+1.0, P2=0.0
        # Margin: P0=1.5*(100-200)/300=-0.5, P1=1.5*(300-200)/300=+0.5, P2=0.0
        # Blend: P0=-0.75, P1=+0.75, P2=0.0
        vals = compute_terminal_values([100, 300, 200], 3)
        np.testing.assert_array_almost_equal(vals, [-0.75, 0.75, 0.0])

    def test_winner_has_highest_value(self):
        vals = compute_terminal_values([50, 200, 150], 3)
        assert vals[1] > vals[0]
        assert vals[1] > vals[2]

    def test_zero_net_worth(self):
        # NW: [0, 100, 50], mean=50, max=100, scale=1.5
        # Rank: P0=-1.0, P1=+1.0, P2=0.0
        # Margin: P0=1.5*(0-50)/100=-0.75, P1=1.5*(100-50)/100=+0.75, P2=0.0
        # Blend: P0=-0.875, P1=+0.875, P2=0.0
        vals = compute_terminal_values([0, 100, 50], 3)
        np.testing.assert_array_almost_equal(vals, [-0.875, 0.875, 0.0])

    def test_tied_winners(self):
        # NW: [200, 200, 100], mean=166.67, max=200, scale=1.5
        # Rank: P0,P1 tie for 1st → avg(1.0,0.0)=0.5 each, P2=-1.0
        # Margin: P0=P1=1.5*(200-166.67)/200=+0.25, P2=1.5*(100-166.67)/200=-0.5
        # Blend: P0=P1=0.375, P2=-0.75
        vals = compute_terminal_values([200, 200, 100], 3)
        np.testing.assert_array_almost_equal(vals, [0.375, 0.375, -0.75])

    def test_all_tied(self):
        # All equal NW: rank=0 for all, margin=0 for all (zero deviation)
        vals = compute_terminal_values([100, 100, 100], 3)
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 0.0])

    def test_all_zero(self):
        vals = compute_terminal_values([0, 0, 0], 3)
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 0.0])

    def test_continuous_gradient(self):
        """3rd place with higher NW gets a less negative reward."""
        vals_low = compute_terminal_values([100, 243, 261], 3)
        vals_high = compute_terminal_values([150, 243, 261], 3)
        # P0 with $150 should get a better reward than P0 with $100
        assert vals_high[0] > vals_low[0]

    def test_values_bounded(self):
        """All rewards must be in [-1, +1]."""
        for nw in ([500, 1, 0], [100, 100, 100], [1, 1000, 500],
                   [0, 0, 100], [80, 7, 3]):
            vals = compute_terminal_values(nw, 3)
            assert (vals >= -1.0 - 1e-6).all() and (vals <= 1.0 + 1e-6).all(), (
                f"Out of bounds for {nw}: {vals}"
            )

    def test_rank_sharpness(self):
        """Overtaking a close competitor should produce meaningful reward gap."""
        # P1 barely beats P2: rank component creates large gap
        vals = compute_terminal_values([200, 101, 100], 3)
        gap = vals[1] - vals[2]
        # With pure ratio, gap would be ~0.01. Rank component ensures it's much larger.
        assert gap > 0.4

    def test_zero_sum(self):
        """Rewards should sum to zero across players."""
        for nw in ([100, 300, 200], [50, 200, 150], [200, 200, 100],
                   [100, 100, 100], [0, 100, 50], [200, 101, 100],
                   [500, 1, 0], [80, 7, 3]):
            vals = compute_terminal_values(nw, 3)
            assert abs(vals.sum()) < 0.01, f"Not zero-sum for {nw}: sum={vals.sum()}"


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

    def test_mean_value_zero_visits(self):
        node = MCTSNode(num_players=3)
        assert node.mean_value(0) == 0.0

    def test_mean_value(self):
        node = MCTSNode(num_players=3)
        node.visit_count = 2
        node.value_sum = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        assert node.mean_value(0) == pytest.approx(0.5)
        assert node.mean_value(1) == pytest.approx(0.25)

    def test_expand_sets_up_arrays(self):
        """expand() creates per-action arrays but no child nodes."""
        node = MCTSNode(num_players=3)
        priors = np.zeros(_ACT_3P, dtype=np.float32)
        mask = np.zeros(_ACT_3P, dtype=np.float32)
        priors[0] = 0.6
        priors[5] = 0.3
        priors[10] = 0.1
        mask[0] = 1.0
        mask[5] = 1.0
        mask[10] = 1.0
        default_val = np.array([0.2, -0.1, -0.1], dtype=np.float32)

        node.expand(priors, mask, num_players=3, default_value=default_val)

        assert node.expanded()
        assert len(node.children) == 0  # No children created
        assert node.legal_actions is not None
        assert node.priors is not None
        assert node.default_value is not None
        assert node.visit_counts is not None
        assert node.value_sums is not None
        np.testing.assert_array_equal(node.legal_actions, [0, 5, 10])
        assert node.priors[0] == pytest.approx(0.6)
        assert node.priors[1] == pytest.approx(0.3)
        assert node.priors[2] == pytest.approx(0.1)
        np.testing.assert_array_almost_equal(node.default_value, default_val)
        # Zero-init: visit_counts start at 0, value_sums at default_value (FPU)
        assert node.visit_counts.shape == (3,)
        assert (node.visit_counts == 0).all()
        assert node.value_sums.shape == (3, 3)  # 3 actions x 3 players
        for i in range(3):
            np.testing.assert_array_almost_equal(
                node.value_sums[i], default_val
            )


# ---------------------------------------------------------------------------
# PUCT selection
# ---------------------------------------------------------------------------

class TestPUCTSelection:
    def test_selects_highest_prior_when_unvisited(self):
        """With only virtual visits (equal Q), PUCT prefers highest prior."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 2  # sum of per-action visit counts (1+1)
        root.legal_actions = np.array([0, 1], dtype=np.int32)
        root.priors = np.array([0.1, 0.9], dtype=np.float32)
        default_val = np.zeros(3, dtype=np.float32)
        root.default_value = default_val
        # Virtual visits: visit_counts=1, value_sums=default_value
        root.visit_counts = np.ones(2, dtype=np.int32)
        root.value_sums = np.zeros((2, 3), dtype=np.float32)

        action, _ = select_child(root, c_puct=2.5)
        assert action == 1

    def test_exploits_high_value_with_visits(self):
        """With enough visits, should prefer high-value actions."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 102  # sum of per-action visit counts
        root.legal_actions = np.array([0, 1], dtype=np.int32)
        root.priors = np.array([0.8, 0.2], dtype=np.float32)
        root.default_value = np.zeros(3, dtype=np.float32)
        # 1 virtual + 50 real = 51 total per action
        root.visit_counts = np.array([51, 51], dtype=np.int32)
        root.value_sums = np.array([
            [-25.0, 10.0, 15.0],   # action 0: Q(p0) ≈ -0.49
            [25.0, -10.0, -15.0],  # action 1: Q(p0) ≈ +0.49
        ], dtype=np.float32)

        # Player 0 should prefer action 1 (higher Q)
        action, _ = select_child(root, c_puct=1.0)
        assert action == 1

    def test_single_legal_action(self):
        """With one legal action, PUCT must always select it."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 1
        root.legal_actions = np.array([7], dtype=np.int32)
        root.priors = np.array([1.0], dtype=np.float32)
        root.default_value = np.zeros(3, dtype=np.float32)
        root.visit_counts = np.ones(1, dtype=np.int32)
        root.value_sums = np.zeros((1, 3), dtype=np.float32)

        action, array_idx = select_child(root, c_puct=2.5)
        assert action == 7
        assert array_idx == 0

    def test_returns_correct_array_index(self):
        """The returned array_idx should index into legal_actions/priors/visit_counts."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 3  # sum of per-action visit counts (1+1+1)
        # Non-contiguous action indices to distinguish action from array index
        root.legal_actions = np.array([10, 42, 99], dtype=np.int32)
        root.priors = np.array([0.1, 0.1, 0.8], dtype=np.float32)
        root.default_value = np.zeros(3, dtype=np.float32)
        root.visit_counts = np.ones(3, dtype=np.int32)
        root.value_sums = np.zeros((3, 3), dtype=np.float32)

        action, array_idx = select_child(root, c_puct=2.5)
        # Highest prior is at array index 2 (action 99)
        assert action == 99
        assert array_idx == 2
        assert root.legal_actions[array_idx] == action

    def test_fpu_prefers_visited_over_unvisited_in_losing_position(self):
        """FPU prevents wasting simulations on unvisited actions in bad positions.

        With parent value = -0.8 (losing), a visited child at Q=-0.6
        (slightly better) should be preferred over an unvisited child
        whose FPU defaults to -0.8.
        """
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 11  # sum of per-action visit counts (10+1)
        default_val = np.array([-0.8, 0.3, 0.5], dtype=np.float32)
        root.legal_actions = np.array([0, 1], dtype=np.int32)
        root.priors = np.array([0.5, 0.5], dtype=np.float32)
        root.default_value = default_val
        # Action 0: 1 virtual + 9 real = 10 visits
        # Action 1: 1 virtual + 0 real = 1 visit (FPU only)
        root.visit_counts = np.array([10, 1], dtype=np.int32)
        root.value_sums = np.array([
            [-6.2, 2.7, 2.7],       # Q(p0) = -6.2/10 = -0.62
            [-0.8, 0.3, 0.5],       # Q(p0) = -0.8/1 = -0.8 (FPU)
        ], dtype=np.float32)

        # Tiny c_puct so exploitation dominates
        action, _ = select_child(root, c_puct=0.001)
        # Q(action 0) = -0.62 > Q(action 1) = -0.8 → prefer action 0
        # Without FPU: Q(unvisited) = 0.0 > -0.62 → would waste a sim on action 1
        assert action == 0


# ---------------------------------------------------------------------------
# Dirichlet noise
# ---------------------------------------------------------------------------

class TestDirichletNoise:
    def test_zero_epsilon_leaves_priors_unchanged(self):
        """With epsilon=0, priors should be unmodified."""
        node = MCTSNode(num_players=3)
        node.priors = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        original = node.priors.copy()
        rng = np.random.default_rng(42)

        _add_dirichlet_noise(node, alpha=0.3, epsilon=0.0, rng=rng)

        np.testing.assert_array_equal(node.priors, original)

    def test_noise_modifies_priors(self):
        """With epsilon>0, priors should change."""
        node = MCTSNode(num_players=3)
        node.priors = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        original = node.priors.copy()
        rng = np.random.default_rng(42)

        _add_dirichlet_noise(node, alpha=0.3, epsilon=0.25, rng=rng)

        assert not np.array_equal(node.priors, original)
        # Priors should still sum to ~1 (convex combination of two distributions)
        assert node.priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert (node.priors >= 0).all()

    def test_seeded_rng_is_reproducible(self):
        """Same seed should produce identical noised priors."""
        results = []
        for _ in range(2):
            node = MCTSNode(num_players=3)
            node.priors = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            rng = np.random.default_rng(123)
            _add_dirichlet_noise(node, alpha=0.3, epsilon=0.25, rng=rng)
            results.append(node.priors.copy())

        np.testing.assert_array_equal(results[0], results[1])

    def test_dynamic_alpha_varies_with_action_count(self):
        """Dynamic alpha should produce different noise for different action counts."""
        rng = np.random.default_rng(42)
        numerator = 10.0

        # 2 legal actions → alpha = 10/2 = 5.0 (very uniform)
        node2 = MCTSNode(num_players=3)
        node2.priors = np.array([0.99, 0.01], dtype=np.float32)
        alpha2 = numerator / len(node2.priors)
        assert alpha2 == pytest.approx(5.0)
        _add_dirichlet_noise(node2, alpha=alpha2, epsilon=0.25, rng=rng)
        assert node2.priors.sum() == pytest.approx(1.0, abs=1e-5)
        # With alpha=5.0 and 2 actions, noise is nearly uniform (~0.5 each),
        # so the rare action (0.01) gets a significant boost
        assert node2.priors[1] > 0.05

        # 20 legal actions → alpha = 10/20 = 0.5 (more concentrated)
        node20 = MCTSNode(num_players=3)
        priors20 = np.full(20, 0.05, dtype=np.float32)
        node20.priors = priors20
        alpha20 = numerator / len(node20.priors)
        assert alpha20 == pytest.approx(0.5)
        _add_dirichlet_noise(node20, alpha=alpha20, epsilon=0.25, rng=rng)
        assert node20.priors.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

class TestBackup:
    def test_backup_propagates_values(self):
        """Values should propagate from leaf through all ancestors.

        _backup only propagates values (value_sum, value_sums).
        Visit counts are handled separately by _increment_visits.
        """
        from mcts.search import _increment_visits

        # Build a 2-level tree: root -> child (via action 5, array_idx 1)
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 1
        root.legal_actions = np.array([3, 5], dtype=np.int32)
        root.priors = np.array([0.4, 0.6], dtype=np.float32)
        root.default_value = np.zeros(3, dtype=np.float32)
        root.visit_counts = np.ones(2, dtype=np.int32)
        root.value_sums = np.zeros((2, 3), dtype=np.float32)

        child = MCTSNode(active_player_id=1, num_players=3)
        child.visit_count = 0
        root.children[5] = child

        leaf_values = np.array([0.8, -0.3, -0.5], dtype=np.float32)
        path = [(root, 5, 1)]  # (parent, action, array_idx)

        # Increment visits (normally done at selection time)
        _increment_visits(path, child)

        assert child.visit_count == 1
        assert root.visit_count == 2
        assert root.visit_counts[1] == 2  # 1 FPU + 1 real
        assert root.visit_counts[0] == 1  # untouched

        # Backup values (no visit count changes)
        _backup(path, child, leaf_values)

        np.testing.assert_array_almost_equal(child.value_sum, leaf_values)
        np.testing.assert_array_almost_equal(root.value_sum, leaf_values)
        np.testing.assert_array_almost_equal(
            root.value_sums[1], leaf_values  # FPU zeros + leaf_values
        )
        np.testing.assert_array_almost_equal(
            root.value_sums[0], np.zeros(3)  # untouched
        )
        # Visit counts unchanged by backup
        assert child.visit_count == 1
        assert root.visit_count == 2
        assert root.visit_counts[1] == 2


# ---------------------------------------------------------------------------
# Full search
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_search_basic(self, search_root):
        root, _ = search_root

        assert root.visit_count == 21  # 1 initial + 20 simulations
        assert root.expanded()
        assert len(root.children) > 0

    def test_action_probabilities_sum_to_one(self, search_root):
        root, config = search_root
        probs = get_action_probabilities(root, temperature=1.0, action_dim=config.action_dim)

        assert probs.shape == (config.action_dim,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

    def test_action_probabilities_greedy(self, search_root):
        root, config = search_root
        probs = get_action_probabilities(root, temperature=0.0, action_dim=config.action_dim)

        # Greedy: exactly one action with probability 1.0
        assert probs.sum() == pytest.approx(1.0)
        assert (probs == 1.0).sum() == 1

    def test_greedy_leaf_value_bounded(self, search_root_deep):
        root, config = search_root_deep
        val = get_greedy_leaf_value(root, num_players=config.num_players)

        assert val.shape == (3,)
        assert (val >= -1.0).all()
        assert (val <= 1.0).all()

    def test_greedy_leaf_value_nonzero(self, search_root_deep):
        """With enough simulations, the greedy leaf should have non-trivial values."""
        root, config = search_root_deep
        val = get_greedy_leaf_value(root, num_players=config.num_players)

        # At least one value should be non-zero (random weights produce non-zero output)
        assert not np.allclose(val, 0.0)

    def test_greedy_leaf_follows_max_visits(self, search_root_deep):
        """A0GB traversal should follow the most-visited child at each level."""
        root, config = search_root_deep

        # Manually trace the greedy path using visit counts
        node = root
        while node.expanded() and not node.is_terminal:
            best_idx = int(np.argmax(node.visit_counts))
            if node.visit_counts[best_idx] == 0:
                break
            best_action = int(node.legal_actions[best_idx])
            if best_action not in node.children:
                break
            node = node.children[best_action]

        # Manual traversal should arrive at the same value as get_greedy_leaf_value
        expected = node.value_sum / node.visit_count
        actual = get_greedy_leaf_value(root, num_players=config.num_players)
        np.testing.assert_array_almost_equal(actual, expected)

    def test_nodes_have_state_indices(self, search_root):
        """Visited nodes in the tree should have valid state pool indices."""
        root, _ = search_root

        # Root has state index 0 (first allocated)
        assert root.state_idx == 0

        # All visited children have valid state indices
        for child in root.children.values():
            assert child.state_idx >= 0

    def test_lazy_expansion_fewer_children(self, search_root):
        """Lazy expansion creates fewer children than legal actions."""
        root, _ = search_root

        # Root has legal_actions array (all legal moves)
        assert root.legal_actions is not None
        # But children dict only has visited actions
        assert len(root.children) <= len(root.legal_actions)
        # With 20 sims, not all legal actions can be visited
        assert len(root.children) <= 20

    def test_terminal_children_have_cached_values(self, evaluator):
        """Terminal nodes in the search tree should have terminal_values cached."""
        # Build a state 1 move from GAME_OVER: remove all companies so the
        # only legal action is PASS, and a full round of passes ends the game.
        state = GameState(3)
        state.initialize_game(42)
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        config = MCTSConfig(num_simulations=10)
        root = run_search(state, evaluator, config)

        # Walk tree — must find at least one terminal node
        terminal_count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_terminal:
                terminal_count += 1
                assert node.terminal_values is not None
                assert node.terminal_values.shape == (3,)
            for child in node.children.values():
                stack.append(child)
        assert terminal_count > 0, "Search should reach terminal nodes near game end"

    def test_terminal_root_returns_immediately(self, evaluator):
        """Search on a game-over state should return without running simulations."""
        state = GameState(3)
        state.initialize_game(42)
        TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        state.set_player_net_worth(0, 500)
        state.set_player_net_worth(1, 300)
        state.set_player_net_worth(2, 100)

        config = MCTSConfig(num_simulations=100)
        root = run_search(state, evaluator, config)

        assert root.visit_count == 1  # Only the initial evaluation
        assert root.is_terminal
        assert not root.expanded()
        assert root.terminal_values is not None
        # NW: [500, 300, 100], mean=300, max=500, scale=1.5
        # Rank: [+1.0, 0.0, -1.0], Margin: [+0.6, 0.0, -0.6]
        # Blend: [+0.8, 0.0, -0.8]
        np.testing.assert_array_almost_equal(
            root.terminal_values, [0.8, 0.0, -0.8]
        )


# ---------------------------------------------------------------------------
# NNEvaluator
# ---------------------------------------------------------------------------

class TestNNEvaluator:
    def test_evaluate_shapes(self, game_state, evaluator):
        policy, values, mask = evaluator.evaluate(game_state)
        assert policy.shape == (_ACT_3P,)
        assert values.shape == (3,)
        assert mask.shape == (_ACT_3P,)

    def test_policy_is_valid_distribution(self, game_state, evaluator):
        policy, _, _ = evaluator.evaluate(game_state)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()

    def test_values_bounded(self, game_state, evaluator):
        _, values, _ = evaluator.evaluate(game_state)
        assert (values >= -1.0).all()
        assert (values <= 1.0).all()

    def test_policy_zero_on_illegal_actions(self, game_state, evaluator):
        """Policy should have zero probability on illegal actions."""
        policy, _, mask = evaluator.evaluate(game_state)

        illegal = mask == 0.0
        assert illegal.any(), "Need at least one illegal action for this test"
        assert (policy[illegal] == 0.0).all()

    def test_evaluate_terminal(self, evaluator):
        state = GameState(3)
        state.initialize_game(42)
        # Set some net worth values for testing
        state.set_player_net_worth(0, 500)
        state.set_player_net_worth(1, 300)
        state.set_player_net_worth(2, 100)

        vals = evaluator.evaluate_terminal(state)
        # NW: [500, 300, 100], mean=300, max=500, scale=1.5
        # Rank: [+1.0, 0.0, -1.0], Margin: [+0.6, 0.0, -0.6]
        # Blend: [+0.8, 0.0, -0.8]
        np.testing.assert_array_almost_equal(vals, [0.8, 0.0, -0.8])

    def test_evaluate_batch_single(self, game_state, evaluator):
        """Batch of 1 should match single evaluate."""
        single_policy, single_values, single_mask = evaluator.evaluate(game_state)
        batch_results = evaluator.evaluate_batch([game_state])

        assert len(batch_results) == 1
        np.testing.assert_array_almost_equal(batch_results[0][0], single_policy)
        np.testing.assert_array_almost_equal(batch_results[0][1], single_values)
        np.testing.assert_array_equal(batch_results[0][2], single_mask)

    def test_evaluate_batch_multiple(self, game_state, evaluator):
        """Batch of identical states should produce identical results."""
        results = evaluator.evaluate_batch([game_state, game_state])

        assert len(results) == 2
        np.testing.assert_array_almost_equal(results[0][0], results[1][0])
        np.testing.assert_array_almost_equal(results[0][1], results[1][1])

    def test_evaluate_batch_empty(self, evaluator):
        """Empty batch should return empty list."""
        assert evaluator.evaluate_batch([]) == []

    def test_evaluate_batch_shapes(self, game_state, evaluator):
        """Batch results should have correct shapes."""
        results = evaluator.evaluate_batch([game_state, game_state, game_state])

        assert len(results) == 3
        for policy, values, mask in results:
            assert policy.shape == (_ACT_3P,)
            assert values.shape == (3,)
            assert mask.shape == (_ACT_3P,)
            assert policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert (values >= -1.0).all()
            assert (values <= 1.0).all()



# ---------------------------------------------------------------------------
# Batched search
# ---------------------------------------------------------------------------

class TestBatchedSearch:
    def test_batched_search_visit_count(self, game_state, evaluator):
        """Batched search should produce correct total visit count."""
        config = MCTSConfig(num_simulations=20, search_batch_size=4)
        root = run_search(game_state, evaluator, config)

        assert root.visit_count == 21  # 1 initial + 20 simulations
        assert root.expanded()
        assert len(root.children) > 0

    def test_batched_search_action_probs(self, game_state, evaluator):
        """Batched search should produce valid action probabilities."""
        config = MCTSConfig(num_simulations=20, search_batch_size=4)
        root = run_search(game_state, evaluator, config)
        probs = get_action_probabilities(root, temperature=1.0, action_dim=config.action_dim)

        assert probs.shape == (config.action_dim,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

    def test_batched_search_greedy_value(self, game_state, evaluator):
        """Batched search should produce valid A0GB values."""
        config = MCTSConfig(num_simulations=50, search_batch_size=4)
        root = run_search(game_state, evaluator, config)
        val = get_greedy_leaf_value(root, num_players=config.num_players)

        assert val.shape == (3,)
        assert (val >= -1.0).all()
        assert (val <= 1.0).all()

    def test_batch_size_1_matches_unbatched(self, game_state, evaluator):
        """batch_size=1 should produce identical results to default."""
        seed = 42
        config_b1 = MCTSConfig(num_simulations=20, search_batch_size=1)
        config_default = MCTSConfig(num_simulations=20)

        root_b1 = run_search(
            game_state, evaluator, config_b1, rng=np.random.default_rng(seed)
        )
        root_default = run_search(
            game_state, evaluator, config_default, rng=np.random.default_rng(seed)
        )

        assert root_b1.visit_count == root_default.visit_count
        np.testing.assert_array_almost_equal(
            root_b1.value_sum, root_default.value_sum
        )

    def test_large_batch_size_clamped(self, game_state, evaluator):
        """batch_size > num_simulations should still work correctly."""
        config = MCTSConfig(num_simulations=5, search_batch_size=16)
        root = run_search(game_state, evaluator, config)

        assert root.visit_count == 6  # 1 initial + 5 simulations
        assert root.expanded()

    def test_various_batch_sizes(self, game_state, evaluator):
        """Search should work with various batch sizes."""
        for bs in [2, 3, 5, 8]:
            config = MCTSConfig(num_simulations=20, search_batch_size=bs)
            root = run_search(game_state, evaluator, config)
            assert root.visit_count == 21

    def test_terminal_root_with_batching(self, evaluator):
        """Batched search on game-over state should return immediately."""
        state = GameState(3)
        state.initialize_game(42)
        TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        state.set_player_net_worth(0, 500)
        state.set_player_net_worth(1, 300)
        state.set_player_net_worth(2, 100)

        config = MCTSConfig(num_simulations=20, search_batch_size=4)
        root = run_search(state, evaluator, config)

        assert root.visit_count == 1
        assert root.is_terminal

    def test_batched_search_no_duplicate_nodes_in_batch(self, game_state, evaluator):
        """Batched search must not queue the same node for evaluation twice.

        Uses a concentrated-policy evaluator to force narrow frontiers where
        duplicate selection is likely without the deduplication fix.
        """

        class ConcentratedEvaluator:
            """Returns 95% prior on one action to create narrow frontiers."""

            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.batch_call_sizes: list[int] = []

            def evaluate(self, state):
                policy, values, mask = self._inner.evaluate(state)
                return self._concentrate(policy), values, mask

            def evaluate_batch(self, states):
                results = self._inner.evaluate_batch(states)
                self.batch_call_sizes.append(len(states))
                return [(self._concentrate(p), v, m) for p, v, m in results]

            def evaluate_leaves(self, state_arrays, active_player_ids):
                results = self._inner.evaluate_leaves(state_arrays, active_player_ids)
                self.batch_call_sizes.append(len(state_arrays))
                return [(self._concentrate(p), v) for p, v in results]

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

            @staticmethod
            def _concentrate(policy):
                """Put 95% mass on the highest-prior action."""
                best = np.argmax(policy)
                new = np.full_like(policy, 0.05 / max(np.count_nonzero(policy) - 1, 1))
                new[policy == 0] = 0  # keep illegal actions at 0
                new[best] = 0.95
                # renormalize
                total = new.sum()
                if total > 0:
                    new /= total
                return new

        conc = ConcentratedEvaluator(evaluator)
        config = MCTSConfig(
            num_simulations=24, search_batch_size=8,
            dirichlet_epsilon=0.0,  # disable noise for deterministic priors
        )
        root = run_search(game_state, conc, config)

        # Verify visit count invariant on every expanded node.
        # The duplicate-leaf bug violates this: double expand() overwrites
        # per-action arrays while node-level visit_count retains both evals.
        def check_invariant(node: MCTSNode) -> None:
            if not node.expanded():
                return
            assert node.visit_counts is not None
            child_visits = int(node.visit_counts.sum())
            expected = 1 + child_visits  # 1 for this node's own eval
            assert node.visit_count == expected, (
                f"visit_count={node.visit_count} != "
                f"1 + sum(child_visits)={expected}"
            )
            for child in node.children.values():
                check_invariant(child)

        check_invariant(root)

    def test_leaf_lock_caps_batch_to_frontier_size(self, game_state, evaluator):
        """When batch_size > available frontier nodes, batch should be capped.

        At the root of a fresh search, the frontier is the set of legal
        actions (each leads to an unexpanded child). With batch_size larger
        than the number of legal actions, the leaf-lock mechanism should
        submit a batch equal to the frontier size, not the full batch_size.
        """

        class BatchSizeTracker:
            """Wraps evaluator to record batch sizes."""

            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.batch_sizes: list[int] = []

            def evaluate(self, state):
                return self._inner.evaluate(state)

            def evaluate_batch(self, states):
                self.batch_sizes.append(len(states))
                return self._inner.evaluate_batch(states)

            def evaluate_leaves(self, state_arrays, active_player_ids):
                self.batch_sizes.append(len(state_arrays))
                return self._inner.evaluate_leaves(state_arrays, active_player_ids)

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        # Count legal actions at the root to know the frontier size
        legal_count = int(get_valid_action_mask(game_state).sum())

        tracker = BatchSizeTracker(evaluator)
        # batch_size much larger than legal actions at root
        config = MCTSConfig(
            num_simulations=legal_count, search_batch_size=legal_count + 10,
            dirichlet_epsilon=0.0,
        )
        run_search(game_state, tracker, config)

        # First batch should be capped at the frontier size (legal actions)
        assert len(tracker.batch_sizes) >= 1
        assert tracker.batch_sizes[0] == legal_count, (
            f"First batch size {tracker.batch_sizes[0]} != "
            f"frontier size {legal_count}"
        )

    def test_leaf_lock_no_duplicate_evaluations(self, game_state, evaluator):
        """No node should appear in the same evaluation batch twice."""

        class DuplicateNodeTracker:
            """Wraps evaluator to check for duplicate nodes per batch.

            Uses the data pointer address of each state's backing array
            to distinguish nodes. Same pool row = same data pointer.
            """

            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.found_duplicates = False

            def evaluate(self, state):
                return self._inner.evaluate(state)

            def evaluate_batch(self, states):
                addrs = [s._array.ctypes.data for s in states]
                if len(addrs) != len(set(addrs)):
                    self.found_duplicates = True
                return self._inner.evaluate_batch(states)

            def evaluate_leaves(self, state_arrays, active_player_ids):
                addrs = [arr.ctypes.data for arr in state_arrays]
                if len(addrs) != len(set(addrs)):
                    self.found_duplicates = True
                return self._inner.evaluate_leaves(state_arrays, active_player_ids)

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        tracker = DuplicateNodeTracker(evaluator)
        config = MCTSConfig(
            num_simulations=40, search_batch_size=16,
            dirichlet_epsilon=0.0,
        )
        run_search(game_state, tracker, config)

        assert not tracker.found_duplicates, "Same node appeared twice in a batch"


# ---------------------------------------------------------------------------
# Subtree reuse
# ---------------------------------------------------------------------------

class TestSubtreeReuse:
    def test_pool_compact_preserves_data(self):
        """Compaction should copy states to the front and update indices."""
        pool = StatePool(10, 5)
        # Allocate 5 rows with distinct data
        nodes = []
        for i in range(5):
            node = MCTSNode(num_players=3)
            data = np.full(5, float(i), dtype=np.float32)
            node.state_idx = pool.alloc(data)
            nodes.append(node)

        # Keep only nodes 1, 3 (originally at indices 1, 3)
        retained = [nodes[1], nodes[3]]
        # Sort by state_idx (already sorted)
        pool.compact(retained)

        assert pool._next == 2
        assert nodes[1].state_idx == 0
        assert nodes[3].state_idx == 1
        np.testing.assert_array_equal(pool.row(0), np.full(5, 1.0))
        np.testing.assert_array_equal(pool.row(1), np.full(5, 3.0))

    def test_pool_compact_noop_when_contiguous(self):
        """Compaction should be a no-op when retained nodes are already at front."""
        pool = StatePool(10, 5)
        nodes = []
        for i in range(3):
            node = MCTSNode(num_players=3)
            node.state_idx = pool.alloc(np.full(5, float(i), dtype=np.float32))
            nodes.append(node)

        pool.compact(nodes)

        assert pool._next == 3
        for i, node in enumerate(nodes):
            assert node.state_idx == i

    def test_collect_subtree_nodes(self, game_state, evaluator):
        """Should collect all nodes with valid state_idx in the subtree."""
        config = MCTSConfig(num_simulations=30)
        root = run_search(game_state, evaluator, config)

        # Total nodes in tree (root + all descendants)
        all_nodes = _collect_subtree_nodes(root)

        # All should have valid state indices
        for node in all_nodes:
            assert node.state_idx >= 0

        # Should be sorted by state_idx
        indices = [n.state_idx for n in all_nodes]
        assert indices == sorted(indices)

        # Root should be in the list
        assert root in all_nodes

    def test_prepare_reuse_root_returns_child(self, game_state, evaluator):
        """Should return the chosen child with compacted pool and reset stats."""
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(102, total_size)
        config = MCTSConfig(num_simulations=50)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        # Find the most-visited action
        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        old_children = dict(root.children[best_action].children)

        reuse = prepare_reuse_root(root, best_action, pool)

        assert reuse is not None
        assert reuse.expanded()
        # Root stats should be reset for zero-visit-root reuse
        assert reuse.visit_count == 1
        assert reuse.visit_counts is not None
        assert (reuse.visit_counts == 0).all()
        # Children should be preserved
        assert reuse.children == old_children
        # Pool should be compacted to just the subtree size
        subtree_size = len(_collect_subtree_nodes(reuse))
        assert pool._next == subtree_size

    def test_prepare_reuse_root_none_for_missing_action(self, game_state, evaluator):
        """Should return None if action wasn't visited during search."""
        config = MCTSConfig(num_simulations=5)
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(6, total_size)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        # Use an action that's definitely not in the tree
        reuse = prepare_reuse_root(root, 999, pool)
        assert reuse is None

    def test_prepare_reuse_root_none_for_terminal(self, evaluator):
        """Should return None if chosen child is terminal."""
        # Create a near-terminal state
        state = GameState(3)
        state.initialize_game(42)
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(21, total_size)
        config = MCTSConfig(num_simulations=20)
        root = run_search(state, evaluator, config, state_pool=pool)

        # Find a terminal child if one exists
        for action_idx, child in root.children.items():
            if child.is_terminal:
                reuse = prepare_reuse_root(root, action_idx, pool)
                assert reuse is None
                break

    def test_reuse_search_produces_valid_tree(self, game_state, evaluator):
        """Search with reuse_root should produce a valid tree."""
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(202, total_size)
        config = MCTSConfig(num_simulations=100)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        # Find best action and prepare reuse
        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        # Apply action to get next state
        from core.driver import DRIVER
        next_state = GameState.from_array(game_state._array, 3)
        DRIVER.apply_action(next_state, best_action)

        # Run search with reuse
        root2 = run_search(next_state, evaluator, config, state_pool=pool, reuse_root=reuse)

        # Should be the same node object
        assert root2 is reuse
        # Should reach target sim count (full budget after reset)
        assert root2.visit_count >= config.num_simulations

        # Action probabilities should be valid
        probs = get_action_probabilities(root2, temperature=1.0, action_dim=config.action_dim)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

        # A0GB value should be valid
        val = get_greedy_leaf_value(root2, num_players=3)
        assert val.shape == (3,)
        assert (val >= -1.0).all()
        assert (val <= 1.0).all()

    def test_reuse_saves_simulations(self, game_state, evaluator):
        """Reused search should do fewer NN evaluations than fresh search."""

        class EvalCounter:
            """Wraps evaluator to count NN forward passes."""

            def __init__(self, inner):
                self._inner = inner
                self.num_players = inner.num_players
                self.eval_count = 0

            def evaluate(self, state):
                self.eval_count += 1
                return self._inner.evaluate(state)

            def evaluate_batch(self, states):
                self.eval_count += len(states)
                return self._inner.evaluate_batch(states)

            def evaluate_leaves(self, state_arrays, active_player_ids):
                self.eval_count += len(state_arrays)
                return self._inner.evaluate_leaves(state_arrays, active_player_ids)

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        # Fresh search
        counter_fresh = EvalCounter(evaluator)
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(202, total_size)
        config = MCTSConfig(num_simulations=100)
        root = run_search(game_state, counter_fresh, config, state_pool=pool)
        fresh_evals = counter_fresh.eval_count

        # Prepare reuse
        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        from core.driver import DRIVER
        next_state = GameState.from_array(game_state._array, 3)
        DRIVER.apply_action(next_state, best_action)

        # Reuse search
        counter_reuse = EvalCounter(evaluator)
        run_search(
            next_state, counter_reuse, config, state_pool=pool, reuse_root=reuse,
        )
        reuse_evals = counter_reuse.eval_count

        # Reuse should save evaluations
        assert reuse_evals < fresh_evals, (
            f"Reuse ({reuse_evals} evals) should save vs fresh ({fresh_evals} evals)"
        )

    def test_reuse_with_batched_search(self, game_state, evaluator):
        """Subtree reuse should work with batched leaf evaluation."""
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(102, total_size)
        config = MCTSConfig(num_simulations=50, search_batch_size=4)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        from core.driver import DRIVER
        next_state = GameState.from_array(game_state._array, 3)
        DRIVER.apply_action(next_state, best_action)

        root2 = run_search(
            next_state, evaluator, config, state_pool=pool, reuse_root=reuse,
        )

        assert root2.visit_count >= config.num_simulations
        probs = get_action_probabilities(root2, temperature=1.0, action_dim=config.action_dim)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_multi_move_reuse_chain(self, game_state, evaluator):
        """Subtree reuse should work across multiple consecutive moves."""
        from core.driver import DRIVER
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(102, total_size)
        config = MCTSConfig(num_simulations=50)
        rng = np.random.default_rng(42)

        state = GameState.from_array(game_state._array, 3)
        reuse_root = None

        for _ in range(5):  # Play 5 moves with reuse
            root = run_search(
                state, evaluator, config, rng=rng,
                state_pool=pool, reuse_root=reuse_root,
            )

            probs = get_action_probabilities(root, temperature=1.0, action_dim=config.action_dim)
            assert probs.sum() == pytest.approx(1.0, abs=1e-5)

            action = int(rng.choice(config.action_dim, p=probs))
            status = DRIVER.apply_action(state, action)
            if status == 2:  # STATUS_GAME_OVER
                break

            reuse_root = prepare_reuse_root(root, action, pool)

    def test_reuse_root_reset_preserves_children(self, game_state, evaluator):
        """After prepare_reuse_root, children dict should be preserved
        but root visit stats should be zeroed."""
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(202, total_size)
        config = MCTSConfig(num_simulations=100)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])
        child = root.children[best_action]

        # Snapshot child's subtree structure before reuse
        old_children_keys = set(child.children.keys())
        old_child_visits = {
            a: c.visit_count for a, c in child.children.items()
        }
        old_default_value = child.default_value.copy() if child.default_value is not None else None

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None
        assert reuse is child

        # Root stats should be reset
        assert reuse.visit_count == 1
        assert reuse.visit_counts is not None
        assert (reuse.visit_counts == 0).all()
        assert reuse.value_sums is not None
        assert reuse.default_value is not None
        assert reuse.legal_actions is not None
        # Each row of value_sums should equal default_value (FPU)
        for i in range(len(reuse.legal_actions)):
            np.testing.assert_array_equal(reuse.value_sums[i], reuse.default_value)
        np.testing.assert_array_equal(reuse.value_sum, reuse.default_value)

        # Children should be completely preserved
        assert set(reuse.children.keys()) == old_children_keys
        for a, c in reuse.children.items():
            assert c.visit_count == old_child_visits[a]
        # default_value should be unchanged
        assert old_default_value is not None
        np.testing.assert_array_equal(reuse.default_value, old_default_value)

    def test_virtual_backup_matches_child_q(self, game_state, evaluator):
        """After search with reuse, root Q for caught-up actions should
        match the child's Q value."""
        from core.state import get_layout
        total_size = get_layout(3).total_size
        pool = StatePool(202, total_size)
        config = MCTSConfig(num_simulations=100)
        root = run_search(game_state, evaluator, config, state_pool=pool)

        assert root.visit_counts is not None and root.legal_actions is not None
        best_idx = int(np.argmax(root.visit_counts))
        best_action = int(root.legal_actions[best_idx])

        reuse = prepare_reuse_root(root, best_action, pool)
        assert reuse is not None

        # Snapshot the children's Q values before the reuse search
        # (children that existed before reuse search starts)
        pre_search_child_qs = {}
        for action_idx, child in reuse.children.items():
            if child.visit_count > 0:
                pre_search_child_qs[action_idx] = (
                    child.value_sum.copy() / child.visit_count
                )

        from core.driver import DRIVER
        next_state = GameState.from_array(game_state._array, 3)
        DRIVER.apply_action(next_state, best_action)

        root2 = run_search(next_state, evaluator, config, state_pool=pool, reuse_root=reuse)
        assert root2.visit_counts is not None and root2.value_sums is not None
        assert root2.legal_actions is not None

        # For each action where the root caught up to the child's old visits,
        # the root's Q should be roughly equal to the child's Q.
        for i, action in enumerate(root2.legal_actions):
            action = int(action)
            if action not in root2.children or action not in pre_search_child_qs:
                continue
            child = root2.children[action]
            if root2.visit_counts[i] == 0:
                continue
            root_q = root2.value_sums[i] / root2.visit_counts[i]
            child_q = child.value_sum / child.visit_count
            np.testing.assert_allclose(root_q, child_q, atol=0.001)

    def test_reuse_noise_affects_distribution(self, game_state, evaluator):
        """Different RNG seeds should produce meaningfully different visit
        distributions when using subtree reuse, demonstrating that Dirichlet
        noise is effective."""
        from core.driver import DRIVER
        from core.state import get_layout
        total_size = get_layout(3).total_size
        config = MCTSConfig(num_simulations=100)

        distributions = []
        for seed in [42, 123]:
            pool = StatePool(202, total_size)
            root = run_search(
                game_state, evaluator, config, state_pool=pool,
                rng=np.random.default_rng(seed),
            )

            assert root.visit_counts is not None and root.legal_actions is not None
            best_idx = int(np.argmax(root.visit_counts))
            best_action = int(root.legal_actions[best_idx])

            reuse = prepare_reuse_root(root, best_action, pool)
            assert reuse is not None

            next_state = GameState.from_array(game_state._array, 3)
            DRIVER.apply_action(next_state, best_action)

            root2 = run_search(
                next_state, evaluator, config, state_pool=pool,
                reuse_root=reuse, rng=np.random.default_rng(seed + 1000),
            )

            probs = get_action_probabilities(
                root2, temperature=1.0, action_dim=config.action_dim,
            )
            distributions.append(probs)

        # The two distributions should differ due to different noise
        diff = np.abs(distributions[0] - distributions[1]).sum()
        assert diff > 0.05, (
            f"Noise should produce different distributions (L1 diff={diff:.4f})"
        )

    def test_reuse_pool_capacity_must_be_2x(self, game_state, evaluator):
        """Pool sized 2*(num_simulations+1) must suffice for subtree reuse.

        Regression test: train/main.py once allocated StatePool with only
        num_simulations+1 capacity. With subtree reuse, compacted nodes
        occupy the front of the pool and new allocations start after them,
        requiring up to 2*(num_simulations+1) total capacity.
        """
        from core.driver import DRIVER
        from core.state import get_layout

        num_sims = 50
        total_size = get_layout(3).total_size

        # Correctly sized pool: 2*(num_sims+1) — must not overflow
        pool = StatePool(2 * (num_sims + 1), total_size)
        config = MCTSConfig(num_simulations=num_sims)
        rng = np.random.default_rng(42)
        state = GameState.from_array(game_state._array, 3)
        reuse_root = None
        moves_played = 0

        for _ in range(20):
            root = run_search(
                state, evaluator, config, rng=rng,
                state_pool=pool, reuse_root=reuse_root,
            )
            probs = get_action_probabilities(root, temperature=1.0,
                                             action_dim=config.action_dim)
            action = int(rng.choice(config.action_dim, p=probs))
            status = DRIVER.apply_action(state, action)
            moves_played += 1
            if status == 2:
                break
            reuse_root = prepare_reuse_root(root, action, pool)

        assert moves_played >= 2, "Need multiple moves to exercise reuse"
