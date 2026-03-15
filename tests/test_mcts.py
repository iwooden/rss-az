"""Tests for MCTS search implementation."""

import numpy as np
import pytest
import torch

from core.actions import get_valid_action_mask
from core.data import GamePhases
from core.state import GameState
from entities.company import COMPANIES
from entities.turn import TURN
from mcts.config import MCTSConfig
from mcts.evaluator import (
    NNEvaluator,
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
    unrotate_values,
)
from mcts.node import MCTSNode
from mcts.search import (
    _add_dirichlet_noise,
    _backup,
    get_action_probabilities,
    get_greedy_leaf_value,
    run_search,
    select_child,
)
from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def layout():
    return get_layout(3)


@pytest.fixture(scope="session")
def model():
    return RSSAlphaZeroNet(RSSModelConfig())


@pytest.fixture(scope="session")
def evaluator(model):
    return NNEvaluator(model, torch.device("cpu"), num_players=3)


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
        assert cfg.action_dim == 246  # 186 + 3*20

    def test_action_dim_6p(self):
        cfg = MCTSConfig(num_players=6)
        assert cfg.action_dim == 306  # 186 + 6*20

    def test_defaults(self):
        cfg = MCTSConfig()
        assert cfg.num_simulations == 800
        assert cfg.c_puct == 2.5
        assert cfg.dirichlet_alpha == 0.3
        assert cfg.dirichlet_epsilon == 0.25
        assert cfg.temperature == 1.0
        assert cfg.num_players == 3
        assert cfg.search_batch_size == 1


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

class TestLayout:
    def test_visible_size_3p(self):
        layout = get_layout(3)
        assert layout.visible_size == 3023

    def test_visible_size_2p(self):
        layout = get_layout(2)
        assert layout.visible_size == 2943

    def test_visible_size_6p(self):
        layout = get_layout(6)
        assert layout.visible_size == 3275

    def test_player_stride_3p(self):
        layout = get_layout(3)
        assert layout.player_stride == 75  # 72 + num_players


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
        vals = compute_terminal_values([100, 300, 200], 3)
        np.testing.assert_array_almost_equal(vals, [-1.0, 1.0, 0.0])

    def test_first_place_tie(self):
        vals = compute_terminal_values([200, 200, 100], 3)
        # P0, P1 tie for 1st: avg(1.0, 0.0) = 0.5
        np.testing.assert_array_almost_equal(vals, [0.5, 0.5, -1.0])

    def test_last_place_tie(self):
        vals = compute_terminal_values([200, 100, 100], 3)
        # P1, P2 tie for 2nd: avg(0.0, -1.0) = -0.5
        np.testing.assert_array_almost_equal(vals, [1.0, -0.5, -0.5])

    def test_all_tied(self):
        vals = compute_terminal_values([100, 100, 100], 3)
        np.testing.assert_array_almost_equal(vals, [0.0, 0.0, 0.0])

    def test_values_sum_to_zero(self):
        """Ranking rewards should always sum to zero."""
        for nw in ([500, 300, 100], [100, 100, 200], [50, 50, 50]):
            vals = compute_terminal_values(nw, 3)
            assert abs(vals.sum()) < 1e-6


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class TestMCTSNode:
    def test_default_construction(self):
        node = MCTSNode()
        assert node.visit_count == 0
        assert node.prior == 0.0
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
        priors = np.zeros(246, dtype=np.float32)
        mask = np.zeros(246, dtype=np.float32)
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
        # Virtual visits: visit_counts start at 1, value_sums at default_value
        assert node.visit_counts.shape == (3,)
        assert (node.visit_counts == 1).all()
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

        # Manually trace the greedy path using real visit counts
        node = root
        while node.expanded() and not node.is_terminal:
            real_counts = node.visit_counts - 1
            best_idx = int(np.argmax(real_counts))
            if real_counts[best_idx] == 0:
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
        np.testing.assert_array_almost_equal(
            root.terminal_values, [1.0, 0.0, -1.0]
        )


# ---------------------------------------------------------------------------
# NNEvaluator
# ---------------------------------------------------------------------------

class TestNNEvaluator:
    def test_evaluate_shapes(self, game_state, evaluator):
        policy, values, mask = evaluator.evaluate(game_state)
        assert policy.shape == (246,)
        assert values.shape == (3,)
        assert mask.shape == (246,)

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
        np.testing.assert_array_almost_equal(vals, [1.0, 0.0, -1.0])

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
            assert policy.shape == (246,)
            assert values.shape == (3,)
            assert mask.shape == (246,)
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
            real_child_visits = int((node.visit_counts - 1).sum())
            expected = 1 + real_child_visits
            assert node.visit_count == expected, (
                f"visit_count={node.visit_count} != "
                f"1 + sum(real_child_visits)={expected}"
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

            def evaluate_terminal(self, state):
                return self._inner.evaluate_terminal(state)

        tracker = DuplicateNodeTracker(evaluator)
        config = MCTSConfig(
            num_simulations=40, search_batch_size=16,
            dirichlet_epsilon=0.0,
        )
        run_search(game_state, tracker, config)

        assert not tracker.found_duplicates, "Same node appeared twice in a batch"
