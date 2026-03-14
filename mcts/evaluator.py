"""NN evaluator for MCTS leaf evaluation.

Handles state rotation (active player → slot 0), legal action masking,
neural network inference, and value un-rotation to canonical player order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from core.state import LayoutInfo, get_layout as _get_layout_uncached

# Cache layouts per player count
_layout_cache: dict[int, LayoutInfo] = {}


def get_layout(num_players: int) -> LayoutInfo:
    """Get cached layout for a given player count.

    Wraps core.state.get_layout() with caching since it crosses the
    Cython/Python boundary each call.
    """
    if num_players not in _layout_cache:
        _layout_cache[num_players] = _get_layout_uncached(num_players)
    return _layout_cache[num_players]


def rotate_visible_state(state_array: np.ndarray, active_player_id: int,
                         num_players: int) -> np.ndarray:
    """Return a copy of the visible state with player data rotated.

    After rotation, the active player's data is at slot 0 (what the NN expects).

    Rotates:
    - Player data blocks (contiguous, each player_stride floats)
    - Turn state per-player fields: auction_high_bidder, auction_starter,
      auction_passed (each num_players floats)

    Args:
        state_array: Full state array (visible + hidden).
        active_player_id: Canonical player ID (0 to num_players-1).
        num_players: Number of players in the game.

    Returns:
        Copy of visible state portion with rotation applied.
    """
    layout = get_layout(num_players)
    visible = state_array[:layout.visible_size].copy()

    if active_player_id == 0:
        return visible  # No rotation needed

    # Rotate player data blocks
    p_off = layout.players_offset
    stride = layout.player_stride
    # Extract all player blocks, then roll
    player_data = visible[p_off:p_off + layout.players_size].copy()
    player_blocks = player_data.reshape(num_players, stride)
    rotated_blocks = np.roll(player_blocks, -active_player_id, axis=0)
    visible[p_off:p_off + layout.players_size] = rotated_blocks.ravel()

    # Rotate per-player turn state fields
    for field_offset in (layout.auction_high_bidder_offset,
                         layout.auction_starter_offset,
                         layout.auction_passed_offset):
        field = visible[field_offset:field_offset + num_players].copy()
        visible[field_offset:field_offset + num_players] = np.roll(
            field, -active_player_id
        )

    return visible


def unrotate_values(values: np.ndarray, active_player_id: int) -> np.ndarray:
    """Convert NN value output (active player at index 0) to canonical order.

    The NN outputs values where index 0 = active player, index 1 = next player, etc.
    This rotates them back so index 0 = player 0, index 1 = player 1, etc.

    Args:
        values: Per-player values from NN, shape (num_players,).
        active_player_id: Canonical player ID of the active player.

    Returns:
        Values in canonical player order.
    """
    return np.roll(values, active_player_id)


def compute_terminal_values(net_worths: list[int], num_players: int) -> np.ndarray:
    """Compute canonical reward values for a terminal game state.

    Ranks players by net worth. Rewards are evenly distributed from
    +1.0 (1st place) to -1.0 (last place). Ties receive averaged rewards.

    For 3 players: 1st=1.0, 2nd=0.0, 3rd=-1.0

    Args:
        net_worths: List of net worth values per player (canonical order).
        num_players: Number of players.

    Returns:
        np.ndarray of shape (num_players,) with reward values per player.
    """
    # Rank rewards: evenly distributed from +1.0 to -1.0
    rank_rewards = np.linspace(1.0, -1.0, num_players)

    # Sort players by net worth descending, stable sort for tie consistency
    sorted_indices = np.argsort(net_worths)[::-1]  # descending

    values = np.zeros(num_players, dtype=np.float32)

    i = 0
    while i < num_players:
        # Find group of tied players
        j = i + 1
        while j < num_players and net_worths[sorted_indices[j]] == net_worths[sorted_indices[i]]:
            j += 1

        # Average reward for tied positions
        avg_reward = float(np.mean(rank_rewards[i:j]))
        for k in range(i, j):
            values[sorted_indices[k]] = avg_reward

        i = j

    return values


class NNEvaluator:
    """Wraps a neural network model for MCTS leaf evaluation.

    Handles state rotation, legal action masking, inference,
    and value un-rotation to canonical player order.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_players: int = 3) -> None:
        self.model = model
        self.device = device
        self.num_players = num_players
        self.layout = get_layout(num_players)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a game state with the neural network.

        Args:
            state: GameState object.

        Returns:
            Tuple of (policy_probs, canonical_values):
            - policy_probs: shape (action_dim,), softmax over legal actions
            - canonical_values: shape (num_players,), per-player values in
              canonical order (index 0 = player 0, etc.)
        """
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()

        # Rotate visible state so active player is at slot 0
        rotated_visible = rotate_visible_state(
            state._array, active_player, self.num_players
        )

        # Get legal action mask
        mask_np = get_valid_action_mask(state)

        # Convert to tensors
        x = torch.from_numpy(rotated_visible).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        # Forward pass
        policy_logits, value_output = self.model(x, legal_action_mask=mask)

        # Policy: softmax over masked logits
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        # Value: already has tanh applied in the model, un-rotate to canonical
        values_rotated = value_output.squeeze(0).cpu().numpy()
        canonical_values = unrotate_values(values_rotated, active_player)

        return policy_probs, canonical_values

    @torch.no_grad()
    def evaluate_batch(
        self, states: list[Any],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate multiple game states in a single NN forward pass.

        Args:
            states: List of GameState objects.

        Returns:
            List of (policy_probs, canonical_values) tuples, one per state.
        """
        from core.actions import get_valid_action_mask

        n = len(states)
        if n == 0:
            return []
        if n == 1:
            return [self.evaluate(states[0])]

        active_players = [s.get_active_player() for s in states]

        # Rotate and stack visible states
        rotated = np.stack([
            rotate_visible_state(s._array, ap, self.num_players)
            for s, ap in zip(states, active_players)
        ])
        masks = np.stack([get_valid_action_mask(s) for s in states])

        # Single forward pass for the whole batch
        x = torch.from_numpy(rotated).to(self.device)
        mask = torch.from_numpy(masks).to(self.device)
        policy_logits, value_output = self.model(x, legal_action_mask=mask)

        policy_probs = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        values = value_output.cpu().numpy()

        results: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical_values = unrotate_values(values[i], active_players[i])
            results.append((policy_probs[i], canonical_values))

        return results

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values from a game-over state.

        Args:
            state: GameState in GAME_OVER phase.

        Returns:
            Canonical values, shape (num_players,).
        """
        net_worths = [state.get_player_net_worth(i) for i in range(self.num_players)]
        return compute_terminal_values(net_worths, self.num_players)
