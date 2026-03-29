"""NN evaluator for MCTS leaf evaluation.

Handles state rotation (active player → slot 0), legal action masking,
neural network inference, and value un-rotation to canonical player order.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

from core.state import LayoutInfo, get_layout as _get_layout_uncached
from mcts.mcts_core import (
    rotate_visible_state_into as _rotate_cython,
    masked_softmax as _masked_softmax_cython,
    unrotate_values as _unrotate_cython,
)

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


def rotate_visible_state_into(
    dst: np.ndarray,
    state_array: np.ndarray,
    active_player_id: int,
    num_players: int,
) -> None:
    """Copy visible state into *dst* with player data rotated in-place.

    Delegates to a Cython implementation that uses memcpy and pointer
    arithmetic instead of numpy roll/copy/reshape.

    Args:
        dst: Destination array of shape ``(visible_size,)`` — written in-place.
        state_array: Full state array (visible + hidden).
        active_player_id: Canonical player ID (0 to num_players-1).
        num_players: Number of players in the game.
    """
    layout = get_layout(num_players)
    _rotate_cython(
        dst, state_array, active_player_id, num_players,
        layout.visible_size, layout.players_offset, layout.player_stride,
        layout.auction_high_bidder_offset,
        layout.auction_starter_offset,
        layout.auction_passed_offset,
    )


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
    dst = np.empty(layout.visible_size, dtype=state_array.dtype)
    rotate_visible_state_into(dst, state_array, active_player_id, num_players)
    return dst


def apply_mask_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply legal action mask and softmax to raw policy logits.

    Delegates to a Cython implementation that uses a single C loop
    instead of the previous torch tensor round-trip.

    Args:
        logits: Raw logits from NN, shape (action_dim,).
        mask: Binary float32 mask (1.0 = legal, 0.0 = illegal).

    Returns:
        Probability distribution over actions, shape (action_dim,).
    """
    return _masked_softmax_cython(logits, mask)


def unrotate_values(values: np.ndarray, active_player_id: int) -> np.ndarray:
    """Convert NN value output (active player at index 0) to canonical order.

    Delegates to a Cython implementation that avoids numpy dispatch
    overhead on the tiny (num_players,) array.

    Args:
        values: Per-player values from NN, shape (num_players,).
        active_player_id: Canonical player ID of the active player.

    Returns:
        Values in canonical player order.
    """
    vals = values if values.dtype == np.float32 else values.astype(np.float32)
    return _unrotate_cython(vals, active_player_id, len(vals))


def compute_terminal_values(
    net_worths: list[int], num_players: int, rank_weight: float = 0.5,
) -> np.ndarray:
    """Compute canonical reward values for a terminal game state.

    Blend of rank-based and zero-sum net-worth-deviation rewards. The rank
    component provides sharp signal at rank boundaries (overtaking an
    opponent matters a lot). The margin component provides continuous
    gradient within ranks (3rd place still has reason to improve).

    Both components are zero-sum across players, so the blended result is
    also zero-sum (better utilization of the tanh value head's [-1, +1]
    range). The margin uses a scale factor of n/(n-1) which guarantees
    the result stays in [-1, +1] when all net worths are non-negative
    (game rules ensure this — players cannot have negative net worth).

    When all players have zero net worth, all receive 0.0.

    Args:
        net_worths: List of net worth values per player (canonical order).
        num_players: Number of players.
        rank_weight: Weight for rank component (0.0 = pure margin,
            1.0 = pure rank). Default 0.5 (equal blend).

    Returns:
        np.ndarray of shape (num_players,) with reward values per player.
    """
    max_nw = max(net_worths)

    if max_nw == 0:
        return np.zeros(num_players, dtype=np.float32)

    # Rank component: evenly spaced from +1.0 to -1.0 by placement (zero-sum)
    rank_rewards = np.linspace(1.0, -1.0, num_players)
    sorted_indices = np.argsort(net_worths)[::-1]  # descending
    rank_values = np.zeros(num_players, dtype=np.float32)
    i = 0
    while i < num_players:
        j = i + 1
        while j < num_players and net_worths[sorted_indices[j]] == net_worths[sorted_indices[i]]:
            j += 1
        avg_reward = float(np.mean(rank_rewards[i:j]))
        for k in range(i, j):
            rank_values[sorted_indices[k]] = avg_reward
        i = j

    if rank_weight >= 1.0:
        return rank_values

    # Margin component: zero-sum net-worth deviation from mean, scaled by
    # n/(n-1) so the range is exactly [-1, +1] for any NW distribution
    mean_nw = sum(net_worths) / num_players
    scale = num_players / (num_players - 1)
    margin_values = np.array(
        [scale * (nw - mean_nw) / max_nw for nw in net_worths], dtype=np.float32
    )

    if rank_weight <= 0.0:
        return margin_values

    return rank_weight * rank_values + (1.0 - rank_weight) * margin_values


class BaseEvaluator:
    """Shared state and post-processing for MCTS evaluators.

    Subclassed by NNEvaluator (local model inference) and
    RemoteEvaluator (shared-memory IPC to eval server).
    """

    def __init__(self, num_players: int, terminal_rank_weight: float = 0.5) -> None:
        self.num_players = num_players
        self.terminal_rank_weight = terminal_rank_weight
        self.layout = get_layout(num_players)

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values from a game-over state.

        Args:
            state: GameState in GAME_OVER phase.

        Returns:
            Canonical values, shape (num_players,).
        """
        net_worths = [state.get_player_net_worth(i) for i in range(self.num_players)]
        return compute_terminal_values(
            net_worths, self.num_players, self.terminal_rank_weight
        )

    def _check_nan(self, logits: np.ndarray, values: np.ndarray) -> None:
        """Raise if NN output contains NaN."""
        has_nan_logits = np.isnan(logits).any()
        has_nan_values = np.isnan(values).any()
        if has_nan_logits or has_nan_values:
            parts: list[str] = []
            if has_nan_logits:
                nan_count = int(np.isnan(logits).sum())
                parts.append(f"logits ({nan_count}/{logits.size} NaN)")
            if has_nan_values:
                nan_count = int(np.isnan(values).sum())
                parts.append(f"values ({nan_count}/{values.size} NaN)")
            msg = f"NaN in NN eval output: {', '.join(parts)}"
            logger.error(msg)
            raise RuntimeError(msg)

    def _finalize_single(
        self, logits: np.ndarray, values: np.ndarray,
        mask: np.ndarray, active_player_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply mask+softmax and unrotate values for a single state."""
        self._check_nan(logits, values)
        policy_probs = apply_mask_softmax(logits, mask)
        canonical_values = unrotate_values(values, active_player_id)
        return policy_probs, canonical_values, mask

    def _finalize_batch(
        self, logits_np: np.ndarray, values_np: np.ndarray,
        masks: list[np.ndarray], active_player_ids: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Apply mask+softmax and unrotate values for a batch of states."""
        return [
            self._finalize_single(logits_np[i], values_np[i], masks[i],
                                  active_player_ids[i])
            for i in range(len(active_player_ids))
        ]

    def _finalize_leaves(
        self, logits_np: np.ndarray, values_np: np.ndarray,
        active_player_ids: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Unrotate values for a batch of leaves (logits returned raw)."""
        self._check_nan(logits_np, values_np)
        return [
            (logits_np[i], unrotate_values(values_np[i], active_player_ids[i]))
            for i in range(len(active_player_ids))
        ]


class NNEvaluator(BaseEvaluator):
    """Wraps a neural network model for MCTS leaf evaluation.

    Handles state rotation, legal action masking, inference,
    and value un-rotation to canonical player order.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_players: int = 3, *,
                 terminal_rank_weight: float = 0.5) -> None:
        super().__init__(num_players, terminal_rank_weight)
        self.model = model
        self.device = device
        self._autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
        self.model.eval()

        # Reusable rotation buffer for batch evaluation, grows as needed
        self._rotation_buf: np.ndarray | None = None

        # Validate model output dims match expected action space and num_players.
        # Catches misconfiguration before it reaches boundscheck=False Cython code.
        from core.actions import get_total_action_count
        expected_action_dim = get_total_action_count(num_players)
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            if getattr(cfg, "action_dim", expected_action_dim) != expected_action_dim:
                raise ValueError(
                    f"Model action_dim ({cfg.action_dim}) does not match "
                    f"expected action space for {num_players} players "
                    f"({expected_action_dim})"
                )
            if getattr(cfg, "value_dim", num_players) != num_players:
                raise ValueError(
                    f"Model value_dim ({cfg.value_dim}) does not match "
                    f"num_players ({num_players})"
                )

    def _get_rotation_buf(self, n: int) -> np.ndarray:
        """Return a (n, visible_size) rotation buffer, reusing if large enough."""
        buf = self._rotation_buf
        if buf is None or buf.shape[0] < n:
            buf = np.empty((n, self.layout.visible_size), dtype=np.float32)
            self._rotation_buf = buf
        return buf[:n]

    @torch.no_grad()
    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a game state with the neural network.

        Args:
            state: GameState object.

        Returns:
            Tuple of (policy_probs, canonical_values, legal_mask):
            - policy_probs: shape (action_dim,), softmax over legal actions
            - canonical_values: shape (num_players,), per-player values in
              canonical order (index 0 = player 0, etc.)
            - legal_mask: shape (action_dim,), binary mask of legal actions
        """
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()

        # Rotate visible state so active player is at slot 0
        rotated_visible = rotate_visible_state(
            state._array, active_player, self.num_players
        )

        # Get legal action mask
        mask_np = get_valid_action_mask(state)

        # Convert to tensor (no mask — applied CPU-side after inference)
        x = torch.from_numpy(rotated_visible).unsqueeze(0).to(self.device)

        # Forward pass (bfloat16 on CUDA for throughput)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x)

        # Raw logits + values to numpy, then finalize CPU-side
        logits = policy_logits.float().squeeze(0).cpu().numpy()
        values = value_output.float().squeeze(0).cpu().numpy()
        return self._finalize_single(logits, values, mask_np, active_player)

    @torch.no_grad()
    def evaluate_batch(
        self, states: list[Any],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate multiple game states in a single NN forward pass.

        Args:
            states: List of GameState objects.

        Returns:
            List of (policy_probs, canonical_values, legal_mask) tuples,
            one per state.
        """
        from core.actions import get_valid_action_mask

        n = len(states)
        if n == 0:
            return []
        if n == 1:
            return [self.evaluate(states[0])]

        active_players = [s.get_active_player() for s in states]

        # Rotate visible states into reusable buffer
        rotated = self._get_rotation_buf(n)
        for i, (s, ap) in enumerate(zip(states, active_players)):
            rotate_visible_state_into(rotated[i], s._array, ap, self.num_players)
        masks = [get_valid_action_mask(s) for s in states]

        # Single forward pass (no mask — applied CPU-side after inference)
        x = torch.from_numpy(rotated).to(self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x)

        logits = policy_logits.float().cpu().numpy()
        values = value_output.float().cpu().numpy()
        return self._finalize_batch(logits, values, masks, active_players)

    @torch.no_grad()
    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        active_player_ids: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate pre-computed leaf data in a single NN forward pass.

        Optimized for the MCTS hot loop: takes raw arrays instead of
        GameState objects, avoiding Python wrapper allocation. Returns
        raw logits — the caller applies masked softmax using the legal
        masks it already has on each node.

        Args:
            state_arrays: Raw state arrays (pool row views), each (total_size,).
            active_player_ids: Active player ID for each state.

        Returns:
            List of (logits, canonical_values) tuples. Logits are raw
            NN output; caller must apply masked softmax before use.
        """
        n = len(state_arrays)
        if n == 0:
            return []

        # Rotate visible states into reusable buffer
        rotated = self._get_rotation_buf(n)
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            rotate_visible_state_into(rotated[i], arr, ap, self.num_players)

        # Single forward pass
        x = torch.from_numpy(rotated).to(self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x)

        logits = policy_logits.float().cpu().numpy()
        values = value_output.float().cpu().numpy()
        return self._finalize_leaves(logits, values, active_player_ids)
