"""Pre-allocated ring buffer for storing and sampling training examples."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class TrainingExample(NamedTuple):
    """Single training example from self-play."""

    state: np.ndarray  # (visible_size,), float32 — rotated visible state
    legal_mask: np.ndarray  # (action_dim,), float32 — binary legal action mask
    policy_target: np.ndarray  # (action_dim,), float32 — MCTS visit probabilities
    value_target: np.ndarray  # (num_players,), float32 — A0GB values, active-player-first


class ReplayBuffer:
    """Pre-allocated numpy ring buffer for training examples.

    All arrays are allocated at full capacity upfront. Examples are written
    in a circular fashion, overwriting the oldest when full.
    """

    def __init__(
        self,
        capacity: int,
        visible_size: int,
        action_dim: int,
        num_players: int,
    ) -> None:
        self._capacity = capacity
        self._size = 0
        self._index = 0

        self._states = np.zeros((capacity, visible_size), dtype=np.float32)
        self._legal_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self._policy_targets = np.zeros((capacity, action_dim), dtype=np.float32)
        self._value_targets = np.zeros((capacity, num_players), dtype=np.float32)

    def add_examples(self, examples: list[TrainingExample]) -> None:
        """Add a batch of examples to the ring buffer."""
        n = len(examples)
        if n == 0:
            return

        # Stack into contiguous arrays
        states = np.stack([e.state for e in examples])
        legal_masks = np.stack([e.legal_mask for e in examples])
        policy_targets = np.stack([e.policy_target for e in examples])
        value_targets = np.stack([e.value_target for e in examples])

        if n >= self._capacity:
            # More examples than capacity — just keep the last `capacity` examples
            tail = n - self._capacity
            self._states[:] = states[tail:]
            self._legal_masks[:] = legal_masks[tail:]
            self._policy_targets[:] = policy_targets[tail:]
            self._value_targets[:] = value_targets[tail:]
            self._size = self._capacity
            self._index = 0
            return

        end = self._index + n
        if end <= self._capacity:
            # No wrap-around
            self._states[self._index : end] = states
            self._legal_masks[self._index : end] = legal_masks
            self._policy_targets[self._index : end] = policy_targets
            self._value_targets[self._index : end] = value_targets
        else:
            # Wrap around
            first = self._capacity - self._index
            self._states[self._index :] = states[:first]
            self._legal_masks[self._index :] = legal_masks[:first]
            self._policy_targets[self._index :] = policy_targets[:first]
            self._value_targets[self._index :] = value_targets[:first]

            remainder = n - first
            self._states[:remainder] = states[first:]
            self._legal_masks[:remainder] = legal_masks[first:]
            self._policy_targets[:remainder] = policy_targets[first:]
            self._value_targets[:remainder] = value_targets[first:]

        self._index = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict of torch tensors (CPU)."""
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return {
            "states": torch.from_numpy(self._states[indices].copy()),
            "legal_masks": torch.from_numpy(self._legal_masks[indices].copy()),
            "policy_targets": torch.from_numpy(self._policy_targets[indices].copy()),
            "value_targets": torch.from_numpy(self._value_targets[indices].copy()),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity
