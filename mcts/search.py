"""MCTS search algorithm with PUCT selection and A0GB value targets.

Implements AlphaZero-style MCTS for multiplayer games:
- Vectorized PUCT selection with FPU parent value initialization
- Lazy node expansion (children allocated on first visit only)
- Dirichlet noise at the root for exploration
- A0GB greedy backup for value targets (Willemsen et al., 2020)
- Pre-allocated state pool for zero per-node allocation
- Batched leaf evaluation with virtual loss for GPU throughput
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from train.config import MCTSConfig
from mcts.node import MCTSNode


class StatePool:
    """Pre-allocated matrix for MCTS node game states.

    Created once and reused across all MCTS searches for the training
    lifetime. Each row stores one node's full game state. reset() is
    called at the start of each run_search to reuse the same memory.
    """

    __slots__ = ("states", "_next")

    def __init__(self, capacity: int, state_size: int) -> None:
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self._next = 0

    def reset(self) -> None:
        """Reset the write cursor for a new search."""
        self._next = 0

    def alloc(self, source: np.ndarray) -> int:
        """Copy source array into the next pool row, return its index."""
        idx = self._next
        self.states[idx] = source
        self._next += 1
        return idx

    def alloc_from_row(self, src_idx: int) -> int:
        """Copy an existing pool row to the next slot, return the new index."""
        idx = self._next
        self.states[idx] = self.states[src_idx]
        self._next += 1
        return idx

    def row(self, idx: int) -> np.ndarray:
        """Return a view (not copy) of the given pool row."""
        return self.states[idx]


def select_child(node: MCTSNode, c_puct: float) -> tuple[int, int]:
    """Select the child action with the highest PUCT value.

    Uses vectorized UCB computation over per-action arrays:
        UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

    where Q(a) is the mean value for the active player at this node.
    Each action starts with a virtual visit at the parent's NN value (FPU),
    so visit_counts >= 1 and Q = value_sums / visit_counts always.

    Args:
        node: The parent node to select from. Must be expanded.
        c_puct: Exploration constant.

    Returns:
        Tuple of (action_index, array_index) where action_index is the game
        action and array_index is the position in node.legal_actions.
    """
    assert node.visit_counts is not None and node.value_sums is not None
    assert node.priors is not None and node.legal_actions is not None
    player = node.active_player_id
    sqrt_parent = math.sqrt(node.visit_count)
    vc = node.visit_counts

    # Q values: visit_counts >= 1 always (FPU virtual visit), no branching
    q = node.value_sums[:, player] / vc
    ucb = q + c_puct * node.priors * sqrt_parent / (1 + vc)
    best_idx = int(np.argmax(ucb))
    return int(node.legal_actions[best_idx]), best_idx


def _add_dirichlet_noise(
    node: MCTSNode, alpha: float, epsilon: float, rng: np.random.Generator,
) -> None:
    """Add Dirichlet noise to the root node's action priors.

    Modifies priors in-place:
        P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
    """
    assert node.priors is not None
    noise = rng.dirichlet([alpha] * len(node.priors))
    node.priors = (1 - epsilon) * node.priors + epsilon * noise


# Type alias for a selection path: list of (parent_node, action, array_idx)
_Path = list[tuple[MCTSNode, int, int]]


def _apply_virtual_loss(path: _Path, virtual_loss: np.ndarray) -> None:
    """Apply virtual loss along a selection path to discourage re-selection.

    Increments visit counts and adds a pessimistic value (all -1) to each
    node on the path. This makes the selected actions look worse to subsequent
    PUCT selections within the same batch.
    """
    for node, _, array_idx in path:
        assert node.visit_counts is not None and node.value_sums is not None
        node.visit_count += 1
        node.value_sum += virtual_loss
        node.visit_counts[array_idx] += 1
        node.value_sums[array_idx] += virtual_loss


def _undo_virtual_loss(path: _Path, virtual_loss: np.ndarray) -> None:
    """Remove virtual loss from a selection path before applying real backup."""
    for node, _, array_idx in path:
        assert node.visit_counts is not None and node.value_sums is not None
        node.visit_count -= 1
        node.value_sum -= virtual_loss
        node.visit_counts[array_idx] -= 1
        node.value_sums[array_idx] -= virtual_loss


def run_search(
    root_state: Any,
    evaluator: Any,
    config: MCTSConfig,
    rng: np.random.Generator | None = None,
    state_pool: StatePool | None = None,
) -> MCTSNode:
    """Run MCTS search from the given root state.

    Supports batched leaf evaluation: multiple leaves are selected per
    iteration (using virtual loss to avoid path collisions), evaluated
    in a single NN forward pass, then expanded and backed up.

    Args:
        root_state: GameState object to search from.
        evaluator: NNEvaluator for leaf evaluation.
        config: Search hyperparameters (search_batch_size controls batching).
        rng: Optional numpy random Generator for Dirichlet noise.
            If None, creates an unseeded generator.
        state_pool: Optional pre-allocated StatePool for node state storage.
            If None, a temporary pool is created. For training, pass a
            persistent pool to avoid per-search allocation.

    Returns:
        The root MCTSNode with search statistics populated.
    """
    from core.actions import get_valid_action_mask
    from core.data import GamePhases
    from core.driver import DRIVER, STATUS_GAME_OVER_PY
    from core.state import GameState

    num_players = config.num_players
    batch_size = config.search_batch_size

    # Set up state pool
    if state_pool is None:
        from core.state import get_layout
        total_size = get_layout(num_players).total_size
        state_pool = StatePool(config.num_simulations + 1, total_size)
    state_pool.reset()

    # Check if root state is terminal
    is_terminal = root_state.get_phase() == GamePhases.PHASE_GAME_OVER

    root = MCTSNode(
        prior=0.0,
        active_player_id=root_state.get_active_player(),
        num_players=num_players,
        is_terminal=is_terminal,
    )
    root.state_idx = state_pool.alloc(root_state._array)

    if is_terminal:
        values = evaluator.evaluate_terminal(root_state)
        root.terminal_values = values
        root.visit_count = 1
        root.value_sum += values
        return root

    # Evaluate root with NN
    policy_probs, root_values = evaluator.evaluate(root_state)
    mask = get_valid_action_mask(root_state)

    # Expand root (sets up per-action arrays, no children created)
    root.expand(policy_probs, mask, num_players=num_players,
                default_value=root_values)

    # Add Dirichlet noise at root
    if rng is None:
        rng = np.random.default_rng()
    if config.dirichlet_epsilon > 0:
        _add_dirichlet_noise(root, config.dirichlet_alpha, config.dirichlet_epsilon, rng)

    # Backup root evaluation
    root.visit_count = 1
    root.value_sum += root_values

    # Virtual loss vector: pessimistic for all players
    virtual_loss = np.full(num_players, -1.0, dtype=np.float32)

    # Run simulations in batches
    sim = 0
    while sim < config.num_simulations:
        # Collect a batch of leaves for NN evaluation
        pending: list[tuple[_Path, MCTSNode, GameState]] = []

        for _ in range(min(batch_size, config.num_simulations - sim)):
            sim += 1

            # Selection: traverse tree to find a leaf
            node = root
            path: _Path = []

            while node.expanded() and not node.is_terminal:
                action_idx, array_idx = select_child(node, config.c_puct)
                path.append((node, action_idx, array_idx))

                if action_idx in node.children:
                    # Follow existing child
                    node = node.children[action_idx]
                else:
                    # First visit: create child node with state from pool
                    assert node.priors is not None
                    child = MCTSNode(
                        prior=float(node.priors[array_idx]),
                        num_players=num_players,
                    )
                    # Copy parent state to new pool row, apply action in-place
                    child.state_idx = state_pool.alloc_from_row(node.state_idx)
                    child_gs = GameState.from_buffer(
                        state_pool.row(child.state_idx), num_players
                    )
                    status = DRIVER.apply_action(child_gs, action_idx)
                    child.active_player_id = child_gs.get_active_player()
                    if status == STATUS_GAME_OVER_PY:
                        child.is_terminal = True
                        child.terminal_values = evaluator.evaluate_terminal(child_gs)
                    node.children[action_idx] = child
                    node = child
                    break  # New child is unexpanded — it's the leaf

            # Terminal node: backup immediately (no NN needed)
            if node.is_terminal:
                assert node.terminal_values is not None
                _backup(path, node, node.terminal_values)
                continue

            # Non-terminal leaf: apply virtual loss and queue for batch eval
            _apply_virtual_loss(path, virtual_loss)

            leaf_gs = GameState.from_buffer(
                state_pool.row(node.state_idx), num_players
            )
            node.active_player_id = leaf_gs.get_active_player()
            pending.append((path, node, leaf_gs))

        # Batch evaluate all pending leaves
        if not pending:
            continue

        leaf_states = [gs for _, _, gs in pending]
        results = evaluator.evaluate_batch(leaf_states)

        for (path, node, _leaf_gs), (policy_probs, values) in zip(
            pending, results
        ):
            # Undo virtual loss before real backup
            _undo_virtual_loss(path, virtual_loss)

            # Expand the leaf (sets up arrays for future selection)
            leaf_mask = get_valid_action_mask(_leaf_gs)
            node.expand(policy_probs, leaf_mask, num_players=num_players,
                        default_value=values)

            # Real backup
            _backup(path, node, values)

    return root


def _backup(
    path: _Path,
    leaf: MCTSNode,
    values: np.ndarray,
) -> None:
    """Backpropagate values from a leaf up to the root.

    Updates both node-level aggregates (visit_count, value_sum) and
    per-action arrays (visit_counts, value_sums) on each parent.

    Args:
        path: List of (parent_node, action, array_idx) tuples from root
            to leaf's parent.
        leaf: The leaf node that was evaluated.
        values: Canonical per-player values from the leaf evaluation.
    """
    # Update leaf
    leaf.visit_count += 1
    leaf.value_sum += values

    # Walk back up the path (all nodes in path are expanded)
    for node, _, array_idx in reversed(path):
        assert node.visit_counts is not None and node.value_sums is not None
        node.visit_count += 1
        node.value_sum += values
        node.visit_counts[array_idx] += 1
        node.value_sums[array_idx] += values


def get_action_probabilities(
    root: MCTSNode,
    temperature: float,
    action_dim: int,
) -> np.ndarray:
    """Convert root visit counts to action probabilities.

    Args:
        root: Root node after search.
        temperature: Controls exploration.
            temperature=1.0: proportional to visit counts.
            temperature->0: deterministic (argmax).
        action_dim: Size of the action space.

    Returns:
        Probability distribution over actions, shape (action_dim,).
    """
    probs = np.zeros(action_dim, dtype=np.float32)

    if root.legal_actions is None:
        return probs
    assert root.visit_counts is not None

    # Real visit counts (subtract virtual FPU visit)
    real_counts = root.visit_counts - 1

    if temperature < 1e-8:
        # Greedy: pick the most-visited action
        best_idx = int(np.argmax(real_counts))
        probs[root.legal_actions[best_idx]] = 1.0
        return probs

    # Temperature-scaled visit counts
    counts = real_counts.astype(np.float32) ** (1.0 / temperature)
    total = counts.sum()
    if total > 0:
        counts /= total
    probs[root.legal_actions] = counts

    return probs


def get_greedy_leaf_value(root: MCTSNode, num_players: int) -> np.ndarray:
    """Compute the A0GB greedy backup value target.

    Starting from the root, follow the child with the highest visit count
    (greedy policy) at each level until reaching a leaf node (no children)
    or a terminal node. Return that node's value.

    At a leaf node with a single visit, this equals V_NN (the neural network's
    evaluation). At a terminal node, this equals the game outcome.

    Reference: Willemsen et al., "Value targets in off-policy AlphaZero:
    a new greedy backup" (ALA 2020 / Neural Computing and Applications, 2022).

    Args:
        root: Root node after search.
        num_players: Number of players in the game.

    Returns:
        Canonical per-player values at the greedy leaf, shape (num_players,).
    """
    node = root

    while node.expanded() and not node.is_terminal:
        assert node.visit_counts is not None and node.legal_actions is not None
        # Real visit counts (subtract virtual FPU visit)
        real_counts = node.visit_counts - 1
        # Follow the child with the most real visits (greedy)
        best_idx = int(np.argmax(real_counts))
        if real_counts[best_idx] == 0:
            # All children are unvisited — this node is the tree-edge leaf.
            # Its value_sum / visit_count equals V_NN from its single evaluation.
            break
        best_action = int(node.legal_actions[best_idx])
        if best_action not in node.children:
            break
        node = node.children[best_action]

    # Return the mean value at this node
    if node.visit_count == 0:
        return np.zeros(num_players, dtype=np.float32)

    return node.value_sum / node.visit_count
