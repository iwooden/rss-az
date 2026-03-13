from __future__ import annotations

import numpy as np


class MCTSNode:
    """A node in the MCTS search tree.

    Each node stores visit statistics, value estimates, and the policy prior
    from the parent's NN evaluation. Values are stored in canonical player
    order (player 0, 1, 2, ...) so that backpropagation is straightforward.

    Attributes:
        visit_count: Number of times this node has been visited (N).
        value_sum: Cumulative canonical values per player, shape (num_players,).
        prior: Policy prior P(a) assigned by the parent's NN evaluation.
        active_player_id: The player who acts at this node.
        children: Mapping from action index to child MCTSNode.
        is_terminal: Whether this node represents a game-over state.
    """

    __slots__ = (
        "visit_count",
        "value_sum",
        "prior",
        "active_player_id",
        "children",
        "is_terminal",
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

    def mean_value(self, player_id: int) -> float:
        """Return the mean value estimate for the given player.

        Returns 0.0 if this node has never been visited.
        """
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player_id] / self.visit_count)

    def expanded(self) -> bool:
        """Return True if this node has been expanded (has children)."""
        return len(self.children) > 0

    def expand(
        self,
        policy_priors: np.ndarray,
        legal_mask: np.ndarray,
        active_player_id: int,
        num_players: int,
    ) -> None:
        """Expand this node by creating a child for each legal action.

        Args:
            policy_priors: NN policy output, shape (action_dim,). Values for
                legal actions are used as priors for the corresponding children.
            legal_mask: Binary mask, shape (action_dim,). 1.0 for legal actions.
            active_player_id: The player who will act at each child node.
                Note: this is the active player in the child states, which may
                differ from self.active_player_id.
            num_players: Number of players in the game.
        """
        legal_actions = np.nonzero(legal_mask)[0]
        for action_idx in legal_actions:
            self.children[int(action_idx)] = MCTSNode(
                prior=float(policy_priors[action_idx]),
                active_player_id=active_player_id,
                num_players=num_players,
            )
