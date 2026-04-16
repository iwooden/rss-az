from __future__ import annotations

import numpy as np
from mcts.mcts_core import expand_node_sparse as _expand_node_sparse


class MCTSNode:
    """A node in the MCTS search tree.

    Each node stores visit statistics, value estimates, and per-action arrays
    for vectorized PUCT selection. Values are stored in canonical player order
    (player 0, 1, 2, ...) so that backpropagation is straightforward — the new
    token transformer returns canonical-order values (no rotation), matching
    this layout directly.

    Expansion is lazy: expand() sets up per-action arrays (legal_actions,
    priors, visit_counts, value_sums) but does NOT create child MCTSNode
    objects. Children are created on first visit during PUCT selection and
    stored in the children dict.

    Attributes:
        visit_count: Number of times this node has been visited (N).
        value_sum: Cumulative canonical values per player, shape (num_players,).
        active_player_id: Canonical id of the player who acts at this node.
            Indexes value_sums[:, active_player_id] during PUCT selection.
        children: Mapping from action index to child MCTSNode (visited only).
        is_terminal: Whether this node represents a game-over state.
        state_idx: Index into the StatePool matrix (-1 if unassigned).
        terminal_values: Cached terminal values for game-over nodes.
        pending_n: Count of legal actions cached for this leaf at child
            creation. 0 until a batch evaluator fills in the buffer.
            The action ids themselves are packed directly into the shared
            (batch, K_MAX) eval buffer at creation time — no per-node copy.
        pending_phase: Decision phase id (0-7) of the state at child
            creation. Needed because the model dispatches per-leaf phase_ids.
            -1 until populated.
        legal_actions: Sorted int array of legal action indices, shape (N,).
        priors: NN policy priors for legal actions, shape (N,). Already
            softmax-normalized over the legal list.
        default_value: Parent's NN value used as FPU for unvisited actions.
        visit_counts: Per-action visit counts, shape (N,). Zero-initialized;
            unvisited actions use default_value via value_sums / max(1, vc).
        value_sums: Per-action cumulative values, shape (N, num_players).
            Initialized to default_value per row (FPU). On the first real
            visit, _backup replaces (not adds to) this FPU value.
    """

    __slots__ = (
        "visit_count",
        "value_sum",
        "active_player_id",
        "children",
        "is_terminal",
        "state_idx",
        "terminal_values",
        "pending_n",
        "pending_phase",
        "legal_actions",
        "priors",
        "default_value",
        "visit_counts",
        "value_sums",
        "_propagation_saved",
    )

    def __init__(
        self,
        active_player_id: int = 0,
        num_players: int = 3,
        is_terminal: bool = False,
    ) -> None:
        self.visit_count: int = 0
        self.value_sum: np.ndarray = np.zeros(num_players, dtype=np.float32)
        self.active_player_id: int = active_player_id
        self.children: dict[int, MCTSNode] = {}
        self.is_terminal: bool = is_terminal
        self.state_idx: int = -1
        self.terminal_values: np.ndarray | None = None
        self.pending_n: int = 0
        self.pending_phase: int = -1
        self.legal_actions: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.default_value: np.ndarray | None = None
        self.visit_counts: np.ndarray | None = None
        self.value_sums: np.ndarray | None = None
        self._propagation_saved: dict[int, np.ndarray] | None = None

    def mean_value(self, player_id: int) -> float:
        """Return the mean value estimate for the given player.

        Returns 0.0 if this node has never been visited.

        Not used in production search — kept for test assertions.
        """
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player_id] / self.visit_count)

    def expanded(self) -> bool:
        """Return True if this node has been expanded (has per-action arrays)."""
        return self.legal_actions is not None

    def expand(
        self,
        action_ids: np.ndarray,
        n: int,
        priors: np.ndarray,
        num_players: int,
        default_value: np.ndarray,
    ) -> None:
        """Expand this node from a sparse legal-action list + aligned priors.

        Does NOT create child MCTSNode objects — children are allocated lazily
        on first visit during selection.

        Args:
            action_ids: Legal phase-local action ids, shape (>= n,) uint16.
                Only the first n entries are read.
            n: Number of legal actions.
            priors: Softmax-normalized priors over the legal list,
                shape (>= n,) float32. Only the first n entries are read.
                The eval server / NNEvaluator has already gathered + softmaxed
                the dense model logits at these action ids.
            num_players: Number of players in the game.
            default_value: NN value output for this node, shape (num_players,).
                Used as FPU (First Play Urgency) virtual visit for each action.
        """
        _expand_node_sparse(self, action_ids, n, priors, default_value, num_players)
