"""MCTS search algorithm with PUCT selection and A0GB value targets.

Implements AlphaZero-style MCTS for multiplayer games:
- Vectorized PUCT selection with FPU parent value initialization
- Lazy node expansion (children allocated on first visit only)
- Dirichlet noise at the root for exploration
- A0GB greedy backup for value targets (Willemsen et al., 2020)
- Pre-allocated state pool for zero per-node allocation
- Batched leaf evaluation with leaf-lock deduplication for GPU throughput
- Subtree reuse: reuse the chosen child's subtree as the next search root
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from train.profile_stats import SearchStats

import numpy as np

from train.config import MCTSConfig
from mcts.node import MCTSNode
from mcts.mcts_core import select_child, backup as _backup, increment_visits as _increment_visits, virtual_backup as _virtual_backup, masked_softmax as _apply_mask_softmax


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

    def compact(self, nodes: list[MCTSNode]) -> None:
        """Compact pool to retain only states for the given nodes.

        Copies retained states to the front of the pool and updates
        each node's state_idx to its new position.

        Args:
            nodes: Nodes to retain, sorted by state_idx ascending.
                Ascending order ensures copies never overwrite unread sources.
        """
        for new_idx, node in enumerate(nodes):
            if new_idx != node.state_idx:
                self.states[new_idx] = self.states[node.state_idx]
            node.state_idx = new_idx
        self._next = len(nodes)

    def row(self, idx: int) -> np.ndarray:
        """Return a view (not copy) of the given pool row."""
        return self.states[idx]


def _add_dirichlet_noise(
    node: MCTSNode, alpha: float, epsilon: float, rng: np.random.Generator,
) -> None:
    """Add Dirichlet noise to the root node's action priors.

    Modifies priors in-place:
        P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
    """
    assert node.priors is not None
    noise = rng.dirichlet([alpha] * len(node.priors))
    node.priors = ((1 - epsilon) * node.priors + epsilon * noise).astype(np.float32)


# Type alias for a selection path: list of (parent_node, action, array_idx)
_Path = list[tuple[MCTSNode, int, int]]


def run_search(
    root_state: Any | None,
    evaluator: Any,
    config: MCTSConfig,
    rng: np.random.Generator | None = None,
    state_pool: StatePool | None = None,
    reuse_root: MCTSNode | None = None,
    profile: SearchStats | None = None,
) -> MCTSNode:
    """Run MCTS search from the given root state.

    Supports batched leaf evaluation: multiple leaves are selected per
    iteration and evaluated in a single NN forward pass for GPU throughput.

    Batching uses a leaf-lock mechanism to prevent duplicate evaluation:
    when a leaf is queued for evaluation, its Q value in the parent is set
    to -inf, ensuring PUCT never re-selects the same leaf within a batch.
    Visit counts along the selection path are incremented at selection time
    to gently nudge subsequent selections toward less-explored branches.
    After evaluation, the leaf lock is removed and values are backed up.

    Args:
        root_state: GameState object to search from. Required for fresh
            searches, ignored when reuse_root is provided (the reused
            subtree's pooled states are authoritative).
        evaluator: NNEvaluator for leaf evaluation.
        config: Search hyperparameters (search_batch_size controls batching).
        rng: Optional numpy random Generator for Dirichlet noise.
            If None, creates an unseeded generator.
        state_pool: Optional pre-allocated StatePool for node state storage.
            If None, a temporary pool is created. For training, pass a
            persistent pool to avoid per-search allocation.
            Required when reuse_root is provided.
        reuse_root: Optional pre-searched subtree root from the previous move.
            When provided, the root's visit statistics have been reset by
            prepare_reuse_root() so Dirichlet noise is fully effective.
            Actions whose children have existing visits trigger lightweight
            "virtual backups" (child Q echoed to root) until the root's
            per-action count catches up, then real search begins.
            The state_pool must already be compacted for this subtree.

    Returns:
        The root MCTSNode with search statistics populated.
    """
    from core.actions import get_valid_action_mask
    from core.data import GamePhases
    from core.driver import DRIVER, STATUS_GAME_OVER_PY, STATUS_INVALID_PY
    from core.state import GameState

    num_players = config.num_players
    batch_size = config.search_batch_size

    if reuse_root is not None:
        # Reuse requires the matching pool — a fresh pool would make the
        # reused nodes' state_idx values point into uninitialized memory.
        if state_pool is None:
            raise ValueError("state_pool is required when reuse_root is provided")

        # Reuse existing subtree — pool was already compacted and root
        # stats reset by prepare_reuse_root(), don't reset the pool.
        # Full sim budget: virtual backups for existing children are
        # near-free, real search runs once root catches up per action.
        root = reuse_root
        num_sims = config.num_simulations
    else:
        if root_state is None:
            raise ValueError("root_state is required for fresh searches")

        # Set up state pool
        if state_pool is None:
            from core.state import get_layout
            total_size = get_layout(num_players).total_size
            state_pool = StatePool(2 * (config.num_simulations + 1), total_size)
        # Fresh search — reset pool and build root from scratch
        state_pool.reset()

        # Check if root state is terminal
        is_terminal = root_state.get_phase() == GamePhases.PHASE_GAME_OVER

        root = MCTSNode(
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
        policy_probs, root_values, mask = evaluator.evaluate(root_state)

        # Expand root (sets up per-action arrays, no children created)
        root.expand(policy_probs, mask, num_players=num_players,
                    default_value=root_values)

        # Backup root evaluation
        root.visit_count = 1
        root.value_sum += root_values

        num_sims = config.num_simulations

    # Scratch GameState rebound to each pool row via rebind()
    scratch_gs = GameState.from_buffer(state_pool.row(0), num_players)

    # Add Dirichlet noise at root (fresh noise each search)
    if rng is None:
        rng = np.random.default_rng()
    if config.dirichlet_epsilon > 0:
        assert root.priors is not None
        if config.dirichlet_dynamic:
            alpha = config.dirichlet_alpha_numerator / len(root.priors)
        else:
            alpha = config.dirichlet_alpha
        _add_dirichlet_noise(root, alpha, config.dirichlet_epsilon, rng)

    # Leaf lock sentinel: -inf Q guarantees PUCT never re-selects a locked edge
    neg_inf_row = np.full(num_players, -np.inf, dtype=np.float32)

    if profile is not None:
        profile.num_searches += 1
    _t0 = _t1 = _t2 = 0.0  # profile timing scratch vars

    # Run simulations in batches
    sim = 0
    while sim < num_sims:
        if profile is not None:
            _t0 = perf_counter()

        # Collect a batch of leaves for NN evaluation.
        pending: list[tuple[_Path, MCTSNode]] = []
        pending_ids: set[int] = set()  # safety net for single-action parents
        saved_values: list[np.ndarray] = []  # saved parent Q rows for unlock

        while len(pending) < batch_size and sim < num_sims:
            # Selection: traverse tree to find a leaf
            node = root
            path: _Path = []
            is_root = True
            did_virtual_backup = False

            while node.expanded() and not node.is_terminal:
                assert node.visit_counts is not None
                action_idx, array_idx = select_child(node, config.c_puct)
                path.append((node, action_idx, array_idx))

                if action_idx in node.children:
                    child = node.children[action_idx]

                    # Virtual backup: at the root of a reused subtree,
                    # if the child has more real visits than the root
                    # has recorded, echo the child's mean Q to the root
                    # without traversing deeper. This lets Dirichlet noise
                    # meaningfully influence the root visit distribution.
                    if (is_root
                            and child.visit_count > 0
                            and node.visit_counts[array_idx] < child.visit_count):
                        _virtual_backup(node, child, array_idx)
                        sim += 1
                        did_virtual_backup = True
                        break

                    # Follow existing child
                    is_root = False
                    node = child
                else:
                    # First visit: create child node with state from pool.
                    # Rebind scratch GameState to avoid allocating a wrapper.
                    child = MCTSNode(
                        num_players=num_players,
                    )
                    child.state_idx = state_pool.alloc_from_row(node.state_idx)
                    scratch_gs.rebind(state_pool.row(child.state_idx))
                    status = DRIVER.apply_action(scratch_gs, action_idx)
                    assert status != STATUS_INVALID_PY, (
                        f"MCTS expansion got STATUS_INVALID for action {action_idx}"
                    )
                    child.active_player_id = scratch_gs.get_active_player()
                    if status == STATUS_GAME_OVER_PY:
                        child.is_terminal = True
                        child.terminal_values = evaluator.evaluate_terminal(
                            scratch_gs
                        )
                    else:
                        # Cache legal mask for later evaluation
                        child.pending_mask = get_valid_action_mask(scratch_gs)
                    node.children[action_idx] = child
                    node = child
                    break  # New child is unexpanded — it's the leaf

            # Virtual backup already handled — skip leaf processing
            if did_virtual_backup:
                continue

            # Terminal node: increment visits along path and backup values
            if node.is_terminal:
                assert node.terminal_values is not None
                sim += 1
                _increment_visits(path, node)
                _backup(path, node, node.terminal_values)
                continue

            # Non-terminal leaf: queue for batch evaluation.
            # If selection lands on an already-pending leaf, stop filling
            # this batch. `continue` would loop forever: neither sim nor
            # visit counts were updated, so PUCT would make identical
            # choices. Instead, evaluate the partial batch (unlocking
            # leaves), then the next iteration has fresh frontier.
            nid = id(node)
            if nid in pending_ids:
                break

            sim += 1
            _increment_visits(path, node)

            # Lock parent edge and queue for batch eval
            parent, _, parent_aidx = path[-1]
            assert parent.value_sums is not None
            saved_values.append(parent.value_sums[parent_aidx].copy())
            parent.value_sums[parent_aidx] = neg_inf_row

            pending.append((path, node))
            pending_ids.add(nid)

        # Batch evaluate all pending leaves using raw arrays (no GameState needed)
        if not pending:
            if profile is not None:
                profile.selection_secs += perf_counter() - _t0
            continue

        if profile is not None:
            _t1 = perf_counter()
            profile.selection_secs += _t1 - _t0

        leaf_arrays = [state_pool.row(node.state_idx) for _, node in pending]
        leaf_players = [node.active_player_id for _, node in pending]
        results = evaluator.evaluate_leaves(leaf_arrays, leaf_players)

        if profile is not None:
            _t2 = perf_counter()
            profile.eval_secs += _t2 - _t1

        for i, ((path, node), (logits, values)) in enumerate(
            zip(pending, results)
        ):
            # Unlock parent edge: restore saved Q values
            parent, _, parent_aidx = path[-1]
            assert parent.value_sums is not None
            parent.value_sums[parent_aidx] = saved_values[i]

            # Apply masked softmax and expand the leaf
            assert node.pending_mask is not None
            policy_probs = _apply_mask_softmax(logits, node.pending_mask)
            node.expand(policy_probs, node.pending_mask,
                        num_players=num_players, default_value=values)
            node.pending_mask = None  # dead after expansion

            # Backup values (visit counts already incremented at selection time)
            _backup(path, node, values)

        if profile is not None:
            profile.backup_secs += perf_counter() - _t2
            profile.num_eval_batches += 1
            profile.total_leaves += len(pending)

    return root


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

    counts = root.visit_counts

    if temperature < 1e-8:
        # Greedy: pick the most-visited action
        best_idx = int(np.argmax(counts))
        probs[root.legal_actions[best_idx]] = 1.0
        return probs

    # Temperature-scaled visit counts
    scaled = counts.astype(np.float32) ** (1.0 / temperature)
    total = scaled.sum()
    if total > 0:
        scaled /= total
    probs[root.legal_actions] = scaled

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
        best_idx = int(np.argmax(node.visit_counts))
        if node.visit_counts[best_idx] == 0:
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


def _collect_subtree_nodes(root: MCTSNode) -> list[MCTSNode]:
    """Collect all nodes in a subtree that have pool-allocated states.

    Returns nodes sorted by state_idx (ascending) for safe in-place
    compaction — copies never overwrite unread source rows.
    """
    nodes: list[MCTSNode] = []
    stack: list[MCTSNode] = [root]
    while stack:
        node = stack.pop()
        if node.state_idx >= 0:
            nodes.append(node)
        stack.extend(node.children.values())
    nodes.sort(key=lambda n: n.state_idx)
    return nodes


def _reset_root_for_reuse(node: MCTSNode) -> None:
    """Reset root-level visit stats for zero-visit-root reuse.

    Preserves children, priors, legal_actions, and default_value.
    Resets visit_counts, value_sums, visit_count, and value_sum
    so that Dirichlet noise has full effect on PUCT selection.

    During subsequent search, actions whose children have existing
    visit counts trigger "virtual backups" — the child's mean Q is
    backed up without tree traversal until the root's per-action
    visit count catches up to the child's visit count.
    """
    assert node.expanded()
    assert node.legal_actions is not None
    assert node.default_value is not None
    n = len(node.legal_actions)
    num_players = node.value_sum.shape[0]
    node.visit_count = 1
    node.value_sum = node.default_value.copy()
    node.visit_counts = np.zeros(n, dtype=np.int32)
    node.value_sums = np.broadcast_to(
        node.default_value, (n, num_players)
    ).astype(np.float32).copy()


def prepare_reuse_root(
    root: MCTSNode,
    action_idx: int,
    state_pool: StatePool,
) -> MCTSNode | None:
    """Extract the chosen child as a reuse root for the next search.

    Compacts the state pool to retain only the child's subtree states,
    then resets the child's root-level visit statistics so that
    Dirichlet noise has full effect during the next search.

    Args:
        root: The current search root.
        action_idx: The action that was chosen (played in the real game).
        state_pool: The state pool used during search.

    Returns:
        The child node ready for reuse, or None if the action has no
        child in the tree or the child is terminal.
    """
    child = root.children.get(action_idx)
    if child is None or child.is_terminal:
        return None

    # All non-terminal children are expanded after search completes
    assert child.expanded(), "Reuse root must be expanded"

    # Compact pool to retain only this subtree's states
    retained = _collect_subtree_nodes(child)
    state_pool.compact(retained)

    # Reset root-level stats so Dirichlet noise is effective
    _reset_root_for_reuse(child)

    return child
