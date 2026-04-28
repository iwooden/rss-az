"""Pre-allocated ring buffer for storing and sampling training examples.

Dense unified-slot schema — mirrors ``train.self_play.GameRecord``: raw
compact int16 game states, per-row ``phase_id`` kept purely for per-phase
TB reporting, dense ``legal_mask`` + ``policy_target`` rows over
``UNIFIED_LOGIT_DIM`` unified logit slots, and canonical-order per-player
``value_target``.

The trainer is responsible for calling ``core.token_data.get_token_data``
and ``core.relations.get_relation_data`` per sampled state at training time
to materialize the ``(num_tokens, token_dim)`` float32 token buffer and
``(num_relations, num_tokens, num_tokens)`` uint8 relation planes. Keeping
replay in compact state form avoids storing derived NN inputs twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.relations import get_relation_data_batch
from core.token_data import get_num_tokens
from nn.transformer import UNIFIED_LOGIT_DIM


class TrainingExample(NamedTuple):
    """Single dense training example.

    Paired 1:1 with ``SelfPlayExample`` from ``train.self_play``.
    """

    state: np.ndarray  # (state_size_int16,) int16 — raw compact state
    phase_id: int  # decision phase 0-10 (TB reporting only)
    legal_mask: np.ndarray  # (UNIFIED_LOGIT_DIM,) uint8 — 1 = legal slot
    policy_target: np.ndarray  # (UNIFIED_LOGIT_DIM,) f32 — zero on illegal
    value_target: np.ndarray  # (num_players,) f32 — canonical A0GB


class ReplayBuffer:
    """Pre-allocated numpy ring buffer for dense training examples.

    All arrays are allocated at full capacity upfront. Examples are written
    in a circular fashion, overwriting the oldest when full.
    """

    def __init__(
        self,
        capacity: int,
        state_size_int16: int,
        num_players: int,
    ) -> None:
        self._capacity = capacity
        self._state_size = state_size_int16
        self._num_players = num_players
        self._unified_dim = int(UNIFIED_LOGIT_DIM)
        self._num_tokens = get_num_tokens(num_players)
        self._num_relations = NUM_ATTENTION_RELATIONS
        self._size = 0
        self._index = 0

        self._states = np.zeros((capacity, state_size_int16), dtype=np.int16)
        self._phase_ids = np.zeros(capacity, dtype=np.int8)
        self._legal_masks = np.zeros((capacity, self._unified_dim), dtype=np.uint8)
        self._policy_targets = np.zeros(
            (capacity, self._unified_dim), dtype=np.float32,
        )
        self._value_targets = np.zeros((capacity, num_players), dtype=np.float32)

    def add_stacked(
        self,
        states: np.ndarray,
        phase_ids: np.ndarray,
        legal_masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> None:
        """Add pre-stacked arrays directly into the ring buffer.

        All arrays share a leading axis of length ``n``. ``legal_masks``
        and ``policy_targets`` are dense over ``UNIFIED_LOGIT_DIM`` slots;
        ``policy_target`` is already zero on slots outside the legal set.
        """
        n = states.shape[0]
        if n == 0:
            return

        if n >= self._capacity:
            # More examples than capacity — just keep the last `capacity` examples
            tail = n - self._capacity
            self._states[:] = states[tail:]
            self._phase_ids[:] = phase_ids[tail:]
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
            self._phase_ids[self._index : end] = phase_ids
            self._legal_masks[self._index : end] = legal_masks
            self._policy_targets[self._index : end] = policy_targets
            self._value_targets[self._index : end] = value_targets
        else:
            # Wrap around
            first = self._capacity - self._index
            self._states[self._index :] = states[:first]
            self._phase_ids[self._index :] = phase_ids[:first]
            self._legal_masks[self._index :] = legal_masks[:first]
            self._policy_targets[self._index :] = policy_targets[:first]
            self._value_targets[self._index :] = value_targets[:first]

            remainder = n - first
            self._states[:remainder] = states[first:]
            self._phase_ids[:remainder] = phase_ids[first:]
            self._legal_masks[:remainder] = legal_masks[first:]
            self._policy_targets[:remainder] = policy_targets[first:]
            self._value_targets[:remainder] = value_targets[first:]

        self._index = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict of torch tensors (CPU).

        Relation planes are generated from the sampled compact states rather
        than stored in the ring buffer, matching the trainer hot path.

        Raises ValueError if batch_size > current buffer size.
        """
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        states = self._states[indices]
        relations = np.empty(
            (
                batch_size,
                self._num_relations,
                self._num_tokens,
                self._num_tokens,
            ),
            dtype=np.uint8,
        )
        get_relation_data_batch(
            [states[i] for i in range(batch_size)],
            self._num_players,
            relations,
        )
        return {
            "states": torch.from_numpy(states),
            "phase_ids": torch.from_numpy(self._phase_ids[indices]),
            "legal_masks": torch.from_numpy(self._legal_masks[indices]),
            "relations": torch.from_numpy(relations),
            "policy_targets": torch.from_numpy(self._policy_targets[indices]),
            "value_targets": torch.from_numpy(self._value_targets[indices]),
        }

    def sample_into(
        self,
        batch_size: int,
        rng: np.random.Generator,
        states_out: np.ndarray,
        phase_ids_out: np.ndarray,
        legal_masks_out: np.ndarray,
        policy_targets_out: np.ndarray,
        value_targets_out: np.ndarray,
        relations_out: np.ndarray | None = None,
    ) -> None:
        """Fill caller-provided arrays with a random batch.

        Unlike ``sample``, this writes directly into existing buffers —
        used by the trainer to fill pinned host scratch, so the
        subsequent H→D copy is genuinely async. Integer outputs may be
        wider than the stored dtype (e.g. int64); widening happens
        during the fancy-index copy. If ``relations_out`` is supplied,
        relation planes are generated from the sampled states into that
        caller-owned scratch buffer.
        """
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        states_out[:] = self._states[indices]
        phase_ids_out[:] = self._phase_ids[indices]
        legal_masks_out[:] = self._legal_masks[indices]
        policy_targets_out[:] = self._policy_targets[indices]
        value_targets_out[:] = self._value_targets[indices]
        if relations_out is not None:
            get_relation_data_batch(
                [states_out[i] for i in range(batch_size)],
                self._num_players,
                relations_out,
            )

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def save(self, directory: Path) -> None:
        """Save buffer contents to directory for later resume.

        Writes individual .npy files for each array (only the occupied portion)
        plus a metadata.json with size/index/capacity/state_size/num_players.
        """
        if self._size == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        n = self._size
        if n < self._capacity:
            # Partial buffer: data is contiguous in [0, n)
            np.save(directory / "states.npy", self._states[:n])
            np.save(directory / "phase_ids.npy", self._phase_ids[:n])
            np.save(directory / "legal_masks.npy", self._legal_masks[:n])
            np.save(directory / "policy_targets.npy", self._policy_targets[:n])
            np.save(directory / "value_targets.npy", self._value_targets[:n])
        else:
            # Full buffer: save entire arrays
            np.save(directory / "states.npy", self._states)
            np.save(directory / "phase_ids.npy", self._phase_ids)
            np.save(directory / "legal_masks.npy", self._legal_masks)
            np.save(directory / "policy_targets.npy", self._policy_targets)
            np.save(directory / "value_targets.npy", self._value_targets)
        (directory / "metadata.json").write_text(
            json.dumps({
                "size": self._size,
                "index": self._index,
                "capacity": self._capacity,
                "state_size": self._state_size,
                "num_players": self._num_players,
                "unified_dim": self._unified_dim,
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
        saved_unified_dim: int = meta.get("unified_dim", -1)
        if (
            saved_capacity != self._capacity
            or saved_state_size != self._state_size
            or saved_num_players != self._num_players
            or saved_unified_dim != self._unified_dim
        ):
            print(
                f"  Warning: replay buffer shape mismatch "
                f"(saved cap/state/players/unified="
                f"{saved_capacity:,}/{saved_state_size}/{saved_num_players}/{saved_unified_dim}, "
                f"current={self._capacity:,}/{self._state_size}/"
                f"{self._num_players}/{self._unified_dim}), skipping load"
            )
            return 0
        states = np.load(directory / "states.npy")
        n = states.shape[0]
        self._states[:n] = states
        self._phase_ids[:n] = np.load(directory / "phase_ids.npy")
        self._legal_masks[:n] = np.load(directory / "legal_masks.npy")
        self._policy_targets[:n] = np.load(directory / "policy_targets.npy")
        self._value_targets[:n] = np.load(directory / "value_targets.npy")
        self._size = meta["size"]
        self._index = meta["index"]
        return self._size
