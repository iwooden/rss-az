from __future__ import annotations

import numpy as np


class MCTSNode:
    """A node in the MCTS search tree.

    Each node stores visit statistics, value estimates, and per-action arrays
    for vectorized PUCT selection. Values are stored in canonical player order
    (player 0, 1, 2, ...) so that backpropagation is straightforward.

    Expansion is lazy: expand() sets up per-action arrays (legal_actions,
    priors, visit_counts, value_sums) but does NOT create child MCTSNode
    objects. Children are created on first visit during PUCT selection and
    stored in the children dict.

    Attributes:
        visit_count: Number of times this node has been visited (N).
        value_sum: Cumulative canonical values per player, shape (num_players,).
        prior: Policy prior P(a) assigned by the parent's NN evaluation.
        active_player_id: The player who acts at this node.
        children: Mapping from action index to child MCTSNode (visited only).
        is_terminal: Whether this node represents a game-over state.
        state: Game state array stored at this node, shape (total_size,).
        terminal_values: Cached terminal values for game-over nodes.
        legal_actions: Sorted int array of legal action indices, shape (N,).
        priors: NN policy priors for legal actions, shape (N,).
        default_value: Parent's NN value used as FPU virtual visit.
        visit_counts: Per-action visit counts, shape (N,). Initialized to 1
            (virtual visit from FPU) so Q = value_sums / visit_counts always.
        value_sums: Per-action cumulative values, shape (N, num_players).
            Initialized to default_value per row (FPU virtual visit).
    """

    __slots__ = (
        "visit_count",
        "value_sum",
        "prior",
        "active_player_id",
        "children",
        "is_terminal",
        "state",
        "terminal_values",
        "legal_actions",
        "priors",
        "default_value",
        "visit_counts",
        "value_sums",
    )

    def __init__(
        self,
        prior: float = 0.0,
        active_player_id: int = 0,
        num_players: int = 3,
        is_terminal: bool = False,
    ) -> None:
        self.visit_count: int = 0
        self.value_sum: np.ndarray = np.zeros(num_players, dtype=np.float32)
        self.prior: float = prior
        self.active_player_id: int = active_player_id
        self.children: dict[int, MCTSNode] = {}
        self.is_terminal: bool = is_terminal
        self.state: np.ndarray | None = None
        self.terminal_values: np.ndarray | None = None
        self.legal_actions: np.ndarray | None = None
        self.priors: np.ndarray | None = None
        self.default_value: np.ndarray | None = None
        self.visit_counts: np.ndarray | None = None
        self.value_sums: np.ndarray | None = None

    def mean_value(self, player_id: int) -> float:
        """Return the mean value estimate for the given player.

        Returns 0.0 if this node has never been visited.
        """
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player_id] / self.visit_count)

    def expanded(self) -> bool:
        """Return True if this node has been expanded (has per-action arrays)."""
        return self.legal_actions is not None

    def expand(
        self,
        policy_priors: np.ndarray,
        legal_mask: np.ndarray,
        num_players: int,
        default_value: np.ndarray,
    ) -> None:
        """Expand this node by setting up per-action arrays for PUCT selection.

        Does NOT create child MCTSNode objects — children are allocated lazily
        on first visit during selection.

        Args:
            policy_priors: NN policy output, shape (action_dim,). Values for
                legal actions are used as priors for the corresponding children.
            legal_mask: Binary mask, shape (action_dim,). 1.0 for legal actions.
            num_players: Number of players in the game.
            default_value: NN value output for this node, shape (num_players,).
                Used as FPU (First Play Urgency) virtual visit for each action.
        """
        actions = np.nonzero(legal_mask)[0].astype(np.int32)
        n = len(actions)
        self.legal_actions = actions
        self.priors = policy_priors[actions].astype(np.float32).copy()
        self.default_value = default_value.copy()
        # Virtual visit: each action starts with 1 visit at parent's value.
        # Q = value_sums / visit_counts is always valid (no division by zero).
        self.visit_counts = np.ones(n, dtype=np.int32)
        self.value_sums = np.broadcast_to(
            default_value, (n, num_players)
        ).astype(np.float32).copy()
