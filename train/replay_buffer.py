"""Pre-allocated ring buffer for storing and sampling training examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


class TrainingExample(NamedTuple):
    """Single training example from self-play.

    Legacy dense-MLP schema — scheduled for replacement under rss-az-phli.2
    with the sparse (state int16, phase_id, action_ids, policy_target,
    value_target) contract produced by train.self_play.SelfPlayExample.
    """

    state: np.ndarray  # (state_size,), float32 — flat state vector
    legal_mask: np.ndarray  # (action_dim,), float32 — binary legal action mask
    policy_target: np.ndarray  # (action_dim,), float32 — MCTS visit probabilities
    value_target: np.ndarray  # (num_players,), float32 — A0GB values, active-player-first


class ReplayBuffer:
    """Pre-allocated numpy ring buffer for training examples.

    All arrays are allocated at full capacity upfront. Examples are written
    in a circular fashion, overwriting the oldest when full.

    Legacy dense-MLP schema — scheduled for replacement under rss-az-phli.2.
    """

    def __init__(
        self,
        capacity: int,
        state_size: int,
        action_dim: int,
        num_players: int,
    ) -> None:
        self._capacity = capacity
        self._size = 0
        self._index = 0

        self._states = np.zeros((capacity, state_size), dtype=np.float32)
        self._legal_masks = np.zeros((capacity, action_dim), dtype=np.float32)
        self._policy_targets = np.zeros((capacity, action_dim), dtype=np.float32)
        self._value_targets = np.zeros((capacity, num_players), dtype=np.float32)

    def add_stacked(
        self,
        states: np.ndarray,
        legal_masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> None:
        """Add pre-stacked arrays directly into the ring buffer.

        Arrays should have shape (n, feature_dim) where n is the number of
        examples. This avoids redundant np.stack when the caller already has
        contiguous arrays (e.g., from pre-stacking in worker processes).
        """
        n = states.shape[0]
        if n == 0:
            return

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
        """Sample a random batch. Returns dict of torch tensors (CPU).

        Raises ValueError if batch_size > current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return {
            "states": torch.from_numpy(self._states[indices]),
            "legal_masks": torch.from_numpy(self._legal_masks[indices]),
            "policy_targets": torch.from_numpy(self._policy_targets[indices]),
            "value_targets": torch.from_numpy(self._value_targets[indices]),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def save(self, directory: Path) -> None:
        """Save buffer contents to directory for later resume.

        Writes individual .npy files for each array (only the occupied portion)
        plus a metadata.json with size/index/capacity.
        """
        if self._size == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        n = self._size
        if n < self._capacity:
            # Partial buffer: data is contiguous in [0, n)
            np.save(directory / "states.npy", self._states[:n])
            np.save(directory / "legal_masks.npy", self._legal_masks[:n])
            np.save(directory / "policy_targets.npy", self._policy_targets[:n])
            np.save(directory / "value_targets.npy", self._value_targets[:n])
        else:
            # Full buffer: save entire arrays
            np.save(directory / "states.npy", self._states)
            np.save(directory / "legal_masks.npy", self._legal_masks)
            np.save(directory / "policy_targets.npy", self._policy_targets)
            np.save(directory / "value_targets.npy", self._value_targets)
        (directory / "metadata.json").write_text(
            json.dumps({"size": self._size, "index": self._index,
                        "capacity": self._capacity})
        )

    def load(self, directory: Path) -> int:
        """Load buffer contents from directory.

        Returns the number of examples loaded (0 if directory missing or
        capacity mismatch).
        """
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return 0
        meta = json.loads(metadata_path.read_text())
        saved_capacity: int = meta["capacity"]
        if saved_capacity != self._capacity:
            print(
                f"  Warning: replay buffer capacity mismatch "
                f"(saved={saved_capacity:,}, current={self._capacity:,}), "
                f"skipping load"
            )
            return 0
        states = np.load(directory / "states.npy")
        n = states.shape[0]
        self._states[:n] = states
        self._legal_masks[:n] = np.load(directory / "legal_masks.npy")
        self._policy_targets[:n] = np.load(directory / "policy_targets.npy")
        self._value_targets[:n] = np.load(directory / "value_targets.npy")
        self._size = meta["size"]
        self._index = meta["index"]
        return self._size
