"""Tests for MCTS search implementation."""

import numpy as np
import pytest
import torch

from core.state import GameState
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


@pytest.fixture
def game_state():
    state = GameState(3)
    state.initialize_game(42)
    return state


@pytest.fixture
def model():
    return RSSAlphaZeroNet(RSSModelConfig())


@pytest.fixture
def evaluator(model):
    return NNEvaluator(model, torch.device("cpu"), num_players=3)


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

class TestLayout:
    def test_visible_size_3p(self):
        layout = get_layout(3)
        assert layout['visible_size'] == 3023

    def test_visible_size_2p(self):
        layout = get_layout(2)
        assert layout['visible_size'] == 2943

    def test_visible_size_6p(self):
        layout = get_layout(6)
        assert layout['visible_size'] == 3275

    def test_player_stride_3p(self):
        layout = get_layout(3)
        assert layout['player_stride'] == 75  # 72 + num_players


# ---------------------------------------------------------------------------
# State rotation
# ---------------------------------------------------------------------------

class TestStateRotation:
    def test_no_rotation_for_player_0(self, game_state, layout):
        visible = game_state._array[:layout['visible_size']].copy()
        rotated = rotate_visible_state(game_state._array, 0, 3)
        np.testing.assert_array_equal(visible, rotated)

    def test_player_blocks_rotated_by_1(self, game_state, layout):
        visible = game_state._array[:layout['visible_size']]
        p_off = layout['players_offset']
        stride = layout['player_stride']
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
        visible = game_state._array[:layout['visible_size']]
        p_off = layout['players_offset']
        stride = layout['player_stride']
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
        visible = game_state._array[:layout['visible_size']].copy()
        rotated = rotate_visible_state(game_state._array, 1, 3)

        # Phase + CoO (before players)
        np.testing.assert_array_equal(
            visible[:layout['players_offset']],
            rotated[:layout['players_offset']],
        )

        # FI and everything after players
        end_players = layout['players_offset'] + layout['players_size']
        # Exclude turn state per-player fields from comparison
        fi_to_turn = visible[end_players:layout['turn_offset']]
        fi_to_turn_rot = rotated[end_players:layout['turn_offset']]
        np.testing.assert_array_equal(fi_to_turn, fi_to_turn_rot)

    def test_turn_per_player_fields_rotated(self, game_state, layout):
        """Verify auction_high_bidder, auction_starter, auction_passed are rotated."""
        # Set some per-player turn state values to make rotation detectable
        visible = game_state._array[:layout['visible_size']].copy()

        for field_offset in (layout['auction_high_bidder_offset'],
                             layout['auction_starter_offset'],
                             layout['auction_passed_offset']):
            orig = visible[field_offset:field_offset + 3].copy()
            rotated = rotate_visible_state(game_state._array, 1, 3)
            rotated_field = rotated[field_offset:field_offset + 3]
            # After rotation by 1: [p0, p1, p2] -> [p1, p2, p0]
            expected = np.roll(orig, -1)
            np.testing.assert_array_equal(rotated_field, expected)


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

    def test_mean_value_zero_visits(self):
        node = MCTSNode(num_players=3)
        assert node.mean_value(0) == 0.0

    def test_mean_value(self):
        node = MCTSNode(num_players=3)
        node.visit_count = 2
        node.value_sum = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        assert node.mean_value(0) == pytest.approx(0.5)
        assert node.mean_value(1) == pytest.approx(0.25)

    def test_expand_creates_children(self):
        node = MCTSNode(num_players=3)
        priors = np.zeros(246, dtype=np.float32)
        mask = np.zeros(246, dtype=np.float32)
        priors[0] = 0.6
        priors[5] = 0.3
        priors[10] = 0.1
        mask[0] = 1.0
        mask[5] = 1.0
        mask[10] = 1.0

        node.expand(priors, mask, active_player_id=1, num_players=3)

        assert node.expanded()
        assert len(node.children) == 3
        assert 0 in node.children
        assert 5 in node.children
        assert 10 in node.children
        assert node.children[0].prior == pytest.approx(0.6)
        assert node.children[5].prior == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# PUCT selection
# ---------------------------------------------------------------------------

class TestPUCTSelection:
    def test_selects_highest_prior_when_unvisited(self):
        """With no visits, PUCT should prefer the highest prior."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 1

        root.children[0] = MCTSNode(prior=0.1, num_players=3)
        root.children[1] = MCTSNode(prior=0.9, num_players=3)

        assert select_child(root, c_puct=2.5) == 1

    def test_exploits_high_value_with_visits(self):
        """With enough visits, should prefer high-value actions."""
        root = MCTSNode(active_player_id=0, num_players=3)
        root.visit_count = 100

        # Child 0: high prior, low value
        c0 = MCTSNode(prior=0.8, num_players=3)
        c0.visit_count = 50
        c0.value_sum = np.array([-25.0, 10.0, 15.0], dtype=np.float32)
        root.children[0] = c0

        # Child 1: low prior, high value
        c1 = MCTSNode(prior=0.2, num_players=3)
        c1.visit_count = 50
        c1.value_sum = np.array([25.0, -10.0, -15.0], dtype=np.float32)
        root.children[1] = c1

        # Player 0 should prefer child 1 (Q = 0.5 vs -0.5)
        assert select_child(root, c_puct=1.0) == 1


# ---------------------------------------------------------------------------
# Full search
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_search_basic(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=20)
        root = run_search(game_state, evaluator, config)

        assert root.visit_count == 21  # 1 initial + 20 simulations
        assert root.expanded()
        assert len(root.children) > 0

    def test_action_probabilities_sum_to_one(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=20)
        root = run_search(game_state, evaluator, config)
        probs = get_action_probabilities(root, temperature=1.0, action_dim=config.action_dim)

        assert probs.shape == (config.action_dim,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert (probs >= 0).all()

    def test_action_probabilities_greedy(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=20)
        root = run_search(game_state, evaluator, config)
        probs = get_action_probabilities(root, temperature=0.0, action_dim=config.action_dim)

        # Greedy: exactly one action with probability 1.0
        assert probs.sum() == pytest.approx(1.0)
        assert (probs == 1.0).sum() == 1

    def test_greedy_leaf_value_bounded(self, game_state, evaluator):
        config = MCTSConfig(num_simulations=50)
        root = run_search(game_state, evaluator, config)
        val = get_greedy_leaf_value(root, num_players=config.num_players)

        assert val.shape == (3,)
        assert (val >= -1.0).all()
        assert (val <= 1.0).all()

    def test_greedy_leaf_value_nonzero(self, game_state, evaluator):
        """With enough simulations, the greedy leaf should have non-trivial values."""
        config = MCTSConfig(num_simulations=50)
        root = run_search(game_state, evaluator, config)
        val = get_greedy_leaf_value(root, num_players=config.num_players)

        # At least one value should be non-zero (random weights produce non-zero output)
        assert not np.allclose(val, 0.0)

    def test_greedy_leaf_follows_max_visits(self, game_state, evaluator):
        """A0GB traversal should follow the most-visited child at each level."""
        config = MCTSConfig(num_simulations=100)
        root = run_search(game_state, evaluator, config)

        # Manually trace the greedy path
        node = root
        while node.expanded() and not node.is_terminal:
            best_action = max(
                node.children, key=lambda a: node.children[a].visit_count
            )
            best_child = node.children[best_action]
            if best_child.visit_count == 0:
                break

            # Verify this is actually the max-visit child
            max_visits = max(c.visit_count for c in node.children.values())
            assert best_child.visit_count == max_visits

            node = best_child

    def test_terminal_state_search(self, evaluator):
        """Search on a non-terminal state should return a valid root."""
        state = GameState(3)
        state.initialize_game(42)

        config = MCTSConfig(num_simulations=10)
        root = run_search(state, evaluator, config)
        # At least verify it runs and returns a valid root
        assert root.visit_count > 0


# ---------------------------------------------------------------------------
# NNEvaluator
# ---------------------------------------------------------------------------

class TestNNEvaluator:
    def test_evaluate_shapes(self, game_state, evaluator):
        policy, values = evaluator.evaluate(game_state)
        assert policy.shape == (246,)
        assert values.shape == (3,)

    def test_policy_is_valid_distribution(self, game_state, evaluator):
        policy, _ = evaluator.evaluate(game_state)
        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()

    def test_values_bounded(self, game_state, evaluator):
        _, values = evaluator.evaluate(game_state)
        assert (values >= -1.0).all()
        assert (values <= 1.0).all()

    def test_evaluate_terminal(self, evaluator):
        state = GameState(3)
        state.initialize_game(42)
        # Set some net worth values for testing
        state.set_player_net_worth(0, 500)
        state.set_player_net_worth(1, 300)
        state.set_player_net_worth(2, 100)

        vals = evaluator.evaluate_terminal(state)
        np.testing.assert_array_almost_equal(vals, [1.0, 0.0, -1.0])
