"""Pre-allocated ring buffer for storing and sampling training examples.

Sparse schema — mirrors ``train.self_play.GameRecord``: raw compact int16
game states, per-leaf ``phase_id`` + ``n_legal``, zero-padded legal
``action_ids`` + sparse ``policy_target`` row (only ``[:n_legal]`` is
meaningful), and canonical-order per-player ``value_target``.

The trainer is responsible for calling ``core.token_data.get_token_data``
per sampled state at training time to materialize the
``(num_tokens, token_dim)`` float32 token buffer — cheap nogil Cython call,
fine in DataLoader workers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


class TrainingExample(NamedTuple):
    """Single sparse training example.

    Paired 1:1 with ``SelfPlayExample`` from ``train.self_play``.
    """

    state: np.ndarray  # (state_size_int16,) int16 — raw compact state
    phase_id: int  # decision phase 0-7
    n_legal: int  # number of legal actions at this state
    action_ids: np.ndarray  # (n_legal,) uint16 — phase-local legal ids
    policy_target: np.ndarray  # (n_legal,) float32 — MCTS visit probs
    value_target: np.ndarray  # (num_players,) float32 — canonical A0GB


class ReplayBuffer:
    """Pre-allocated numpy ring buffer for sparse training examples.

    All arrays are allocated at full capacity upfront. Examples are written
    in a circular fashion, overwriting the oldest when full.
    """

    def __init__(
        self,
        capacity: int,
        state_size_int16: int,
        num_players: int,
        k_max: int = 256,
    ) -> None:
        self._capacity = capacity
        self._state_size = state_size_int16
        self._num_players = num_players
        self._k_max = k_max
        self._size = 0
        self._index = 0

        self._states = np.zeros((capacity, state_size_int16), dtype=np.int16)
        self._phase_ids = np.zeros(capacity, dtype=np.int8)
        self._n_legals = np.zeros(capacity, dtype=np.int16)
        self._action_ids = np.zeros((capacity, k_max), dtype=np.uint16)
        self._policy_targets = np.zeros((capacity, k_max), dtype=np.float32)
        self._value_targets = np.zeros((capacity, num_players), dtype=np.float32)

    def add_stacked(
        self,
        states: np.ndarray,
        phase_ids: np.ndarray,
        n_legals: np.ndarray,
        action_ids: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> None:
        """Add pre-stacked arrays directly into the ring buffer.

        All arrays share a leading axis of length ``n``. The sparse fields
        (``action_ids``, ``policy_targets``) must already be zero-padded to
        ``k_max`` along their trailing axis.
        """
        n = states.shape[0]
        if n == 0:
            return

        if n >= self._capacity:
            # More examples than capacity — just keep the last `capacity` examples
            tail = n - self._capacity
            self._states[:] = states[tail:]
            self._phase_ids[:] = phase_ids[tail:]
            self._n_legals[:] = n_legals[tail:]
            self._action_ids[:] = action_ids[tail:]
            self._policy_targets[:] = policy_targets[tail:]
            self._value_targets[:] = value_targets[tail:]
            self._size = self._capacity
            self._index = 0
            return

        end = self._index + n
        if end <= self._capacity:
            # No wrap-around
            self._states[self._index : end] = states
            self._phase_ids[self._index : end] = phase_ids
            self._n_legals[self._index : end] = n_legals
            self._action_ids[self._index : end] = action_ids
            self._policy_targets[self._index : end] = policy_targets
            self._value_targets[self._index : end] = value_targets
        else:
            # Wrap around
            first = self._capacity - self._index
            self._states[self._index :] = states[:first]
            self._phase_ids[self._index :] = phase_ids[:first]
            self._n_legals[self._index :] = n_legals[:first]
            self._action_ids[self._index :] = action_ids[:first]
            self._policy_targets[self._index :] = policy_targets[:first]
            self._value_targets[self._index :] = value_targets[:first]

            remainder = n - first
            self._states[:remainder] = states[first:]
            self._phase_ids[:remainder] = phase_ids[first:]
            self._n_legals[:remainder] = n_legals[first:]
            self._action_ids[:remainder] = action_ids[first:]
            self._policy_targets[:remainder] = policy_targets[first:]
            self._value_targets[:remainder] = value_targets[first:]

        self._index = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict of torch tensors (CPU).

        The trainer runs ``get_token_data`` per state to produce the
        ``(num_tokens, token_dim)`` float32 token buffer — this keeps the
        replay buffer ~3× smaller (int16 vs float32 tokenized) and lets
        tokenization run inside DataLoader workers.

        Note: ``action_ids`` is returned as int16 (torch has no uint16).
        Values fit losslessly (max action id 14976 ≪ 32767); reinterpret
        as uint16 on the consumer side if needed.

        Raises ValueError if batch_size > current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return {
            "states": torch.from_numpy(self._states[indices]),
            "phase_ids": torch.from_numpy(self._phase_ids[indices]),
            "n_legals": torch.from_numpy(self._n_legals[indices]),
            "action_ids": torch.from_numpy(
                self._action_ids[indices].view(np.int16)
            ),
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
        plus a metadata.json with size/index/capacity/state_size/num_players/k_max.
        """
        if self._size == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        n = self._size
        if n < self._capacity:
            # Partial buffer: data is contiguous in [0, n)
            np.save(directory / "states.npy", self._states[:n])
            np.save(directory / "phase_ids.npy", self._phase_ids[:n])
            np.save(directory / "n_legals.npy", self._n_legals[:n])
            np.save(directory / "action_ids.npy", self._action_ids[:n])
            np.save(directory / "policy_targets.npy", self._policy_targets[:n])
            np.save(directory / "value_targets.npy", self._value_targets[:n])
        else:
            # Full buffer: save entire arrays
            np.save(directory / "states.npy", self._states)
            np.save(directory / "phase_ids.npy", self._phase_ids)
            np.save(directory / "n_legals.npy", self._n_legals)
            np.save(directory / "action_ids.npy", self._action_ids)
            np.save(directory / "policy_targets.npy", self._policy_targets)
            np.save(directory / "value_targets.npy", self._value_targets)
        (directory / "metadata.json").write_text(
            json.dumps({
                "size": self._size,
                "index": self._index,
                "capacity": self._capacity,
                "state_size": self._state_size,
                "num_players": self._num_players,
                "k_max": self._k_max,
            })
        )

    def load(self, directory: Path) -> int:
        """Load buffer contents from directory.

        Returns the number of examples loaded (0 if directory missing or
        shape mismatch).
        """
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return 0
        meta = json.loads(metadata_path.read_text())
        saved_capacity: int = meta["capacity"]
        saved_state_size: int = meta.get("state_size", -1)
        saved_num_players: int = meta.get("num_players", -1)
        saved_k_max: int = meta.get("k_max", -1)
        if (
            saved_capacity != self._capacity
            or saved_state_size != self._state_size
            or saved_num_players != self._num_players
            or saved_k_max != self._k_max
        ):
            print(
                f"  Warning: replay buffer shape mismatch "
                f"(saved cap/state/players/k_max="
                f"{saved_capacity:,}/{saved_state_size}/{saved_num_players}/{saved_k_max}, "
                f"current={self._capacity:,}/{self._state_size}/"
                f"{self._num_players}/{self._k_max}), skipping load"
            )
            return 0
        states = np.load(directory / "states.npy")
        n = states.shape[0]
        self._states[:n] = states
        self._phase_ids[:n] = np.load(directory / "phase_ids.npy")
        self._n_legals[:n] = np.load(directory / "n_legals.npy")
        self._action_ids[:n] = np.load(directory / "action_ids.npy")
        self._policy_targets[:n] = np.load(directory / "policy_targets.npy")
        self._value_targets[:n] = np.load(directory / "value_targets.npy")
        self._size = meta["size"]
        self._index = meta["index"]
        return self._size
