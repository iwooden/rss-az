"""Pre-allocated ring buffer for storing and sampling training examples.

Dense unified-slot schema — mirrors ``train.self_play.GameRecord``: raw
compact int16 game states, per-row ``phase_id`` kept purely for per-phase
TB reporting, dense ``legal_mask`` + ``policy_target`` rows over
``UNIFIED_LOGIT_DIM`` unified logit slots, and canonical-order per-player
``value_target``.

The trainer is responsible for materializing model-family-specific inputs per
sampled state at training time: transformer token/relation tensors or ResNet
dense vectors. Keeping replay in compact state form avoids storing derived NN
inputs twice and keeps canonical value targets shared by both model families.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.relations import get_relation_data_batch
from core.state import get_layout
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
    player_count: int  # actual player count for this state


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
        *,
        min_players: int = 0,
        max_players: int = 0,
    ) -> None:
        self._capacity = capacity
        self._state_size = state_size_int16
        self._max_players = max_players or num_players
        self._min_players = (
            min_players
            or (
                min(num_players, self._max_players)
                if num_players
                else self._max_players
            )
        )
        self._num_players = self._max_players
        if not 3 <= self._min_players <= self._max_players <= 5:
            raise ValueError(
                "ReplayBuffer player range must be within 3-5 and min <= max; "
                f"got {self._min_players}-{self._max_players}"
            )
        expected_state_size = get_layout(self._max_players).total_size
        if state_size_int16 != expected_state_size:
            raise ValueError(
                f"state_size_int16 {state_size_int16} does not match "
                f"get_layout({self._max_players}).total_size "
                f"({expected_state_size})"
            )
        self._unified_dim = int(UNIFIED_LOGIT_DIM)
        self._num_tokens = get_num_tokens(self._max_players)
        self._num_relations = NUM_ATTENTION_RELATIONS
        self._size = 0
        self._index = 0

        self._states = np.zeros((capacity, state_size_int16), dtype=np.int16)
        self._phase_ids = np.zeros(capacity, dtype=np.int8)
        self._player_counts = np.zeros(capacity, dtype=np.uint8)
        self._legal_masks = np.zeros((capacity, self._unified_dim), dtype=np.uint8)
        self._policy_targets = np.zeros(
            (capacity, self._unified_dim), dtype=np.float32,
        )
        self._value_targets = np.zeros(
            (capacity, self._max_players), dtype=np.float32,
        )

    def add_stacked(
        self,
        states: np.ndarray,
        phase_ids: np.ndarray,
        legal_masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        player_counts: np.ndarray | None = None,
        num_players: int | None = None,
    ) -> None:
        """Add pre-stacked arrays directly into the ring buffer.

        All arrays share a leading axis of length ``n``. ``legal_masks``
        and ``policy_targets`` are dense over ``UNIFIED_LOGIT_DIM`` slots;
        ``policy_target`` is already zero on slots outside the legal set.
        ``value_targets`` may be actual-width ``(n, num_players)`` or
        max-width ``(n, max_players)``. Padded player slots are zeroed.
        """
        n = states.shape[0]
        if n == 0:
            return
        player_counts_arr = self._resolve_player_counts(
            n, value_targets, player_counts, num_players,
        )
        padded_value_targets = self._pad_value_targets(
            value_targets, player_counts_arr,
        )

        if n >= self._capacity:
            # More examples than capacity — just keep the last `capacity` examples
            tail = n - self._capacity
            self._states[:] = states[tail:]
            self._phase_ids[:] = phase_ids[tail:]
            self._player_counts[:] = player_counts_arr[tail:]
            self._legal_masks[:] = legal_masks[tail:]
            self._policy_targets[:] = policy_targets[tail:]
            self._value_targets[:] = padded_value_targets[tail:]
            self._size = self._capacity
            self._index = 0
            return

        end = self._index + n
        if end <= self._capacity:
            # No wrap-around
            self._states[self._index : end] = states
            self._phase_ids[self._index : end] = phase_ids
            self._player_counts[self._index : end] = player_counts_arr
            self._legal_masks[self._index : end] = legal_masks
            self._policy_targets[self._index : end] = policy_targets
            self._value_targets[self._index : end] = padded_value_targets
        else:
            # Wrap around
            first = self._capacity - self._index
            self._states[self._index :] = states[:first]
            self._phase_ids[self._index :] = phase_ids[:first]
            self._player_counts[self._index :] = player_counts_arr[:first]
            self._legal_masks[self._index :] = legal_masks[:first]
            self._policy_targets[self._index :] = policy_targets[:first]
            self._value_targets[self._index :] = padded_value_targets[:first]

            remainder = n - first
            self._states[:remainder] = states[first:]
            self._phase_ids[:remainder] = phase_ids[first:]
            self._player_counts[:remainder] = player_counts_arr[first:]
            self._legal_masks[:remainder] = legal_masks[first:]
            self._policy_targets[:remainder] = policy_targets[first:]
            self._value_targets[:remainder] = padded_value_targets[first:]

        self._index = end % self._capacity
        self._size = min(self._size + n, self._capacity)

    def _resolve_player_counts(
        self,
        n: int,
        value_targets: np.ndarray,
        player_counts: np.ndarray | None,
        num_players: int | None,
    ) -> np.ndarray:
        if player_counts is not None and num_players is not None:
            raise ValueError("Pass either player_counts or num_players, not both")
        if value_targets.ndim != 2:
            raise ValueError("value_targets must be a 2D array")
        if player_counts is None:
            actual = num_players if num_players is not None else value_targets.shape[1]
            player_counts_arr = np.full(n, actual, dtype=np.uint8)
        else:
            player_counts_arr = np.asarray(player_counts, dtype=np.uint8)
            if player_counts_arr.shape != (n,):
                raise ValueError(
                    f"player_counts shape {player_counts_arr.shape} != ({n},)"
                )

        if np.any(player_counts_arr < self._min_players) or np.any(
            player_counts_arr > self._max_players
        ):
            raise ValueError(
                f"player_counts must be within configured range "
                f"{self._min_players}-{self._max_players}"
            )
        return player_counts_arr

    def _pad_value_targets(
        self,
        value_targets: np.ndarray,
        player_counts: np.ndarray,
    ) -> np.ndarray:
        n = value_targets.shape[0]
        if value_targets.ndim != 2 or n != player_counts.shape[0]:
            raise ValueError(
                "value_targets must be a 2D array with the same leading "
                "dimension as player_counts"
            )
        if value_targets.shape[1] > self._max_players:
            raise ValueError(
                f"value_targets width {value_targets.shape[1]} exceeds "
                f"max_players {self._max_players}"
            )
        padded = np.zeros((n, self._max_players), dtype=np.float32)
        for i, actual in enumerate(player_counts):
            actual_int = int(actual)
            if value_targets.shape[1] < actual_int:
                raise ValueError(
                    f"value_targets width {value_targets.shape[1]} is too "
                    f"small for player_counts[{i}]={actual_int}"
                )
            padded[i, :actual_int] = value_targets[i, :actual_int]
        return padded

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict of torch tensors (CPU).

        Relation planes are generated from the sampled compact states rather
        than stored in the ring buffer for transformer-oriented callers.

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
            relations,
            max_players=self._max_players,
        )
        return {
            "states": torch.from_numpy(states),
            "phase_ids": torch.from_numpy(self._phase_ids[indices]),
            "player_counts": torch.from_numpy(self._player_counts[indices]),
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
        player_counts_out: np.ndarray | None = None,
    ) -> None:
        """Fill caller-provided arrays with a random batch.

        Unlike ``sample``, this writes directly into existing buffers —
        used by the trainer to fill pinned host scratch, so the
        subsequent H→D copy is genuinely async. Integer outputs may be
        wider than the stored dtype (e.g. int64); widening happens
        during the fancy-index copy. If ``relations_out`` is supplied,
        relation planes are generated from the sampled states into that
        caller-owned scratch buffer; ResNet callers leave it as ``None``.
        """
        if batch_size > self._size:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({self._size})"
            )
        indices = rng.choice(self._size, size=batch_size, replace=False)
        states_out[:] = self._states[indices]
        phase_ids_out[:] = self._phase_ids[indices]
        if player_counts_out is not None:
            player_counts_out[:] = self._player_counts[indices]
        legal_masks_out[:] = self._legal_masks[indices]
        policy_targets_out[:] = self._policy_targets[indices]
        value_targets_out[:] = self._value_targets[indices]
        if relations_out is not None:
            get_relation_data_batch(
                [states_out[i] for i in range(batch_size)],
                relations_out,
                max_players=self._max_players,
            )

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def save(self, directory: Path) -> None:
        """Save buffer contents to directory for later resume.

        Writes individual .npy files for each array (only the occupied portion)
        plus metadata with size/index/capacity/player range/shape details.
        """
        if self._size == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        n = self._size
        if n < self._capacity:
            # Partial buffer: data is contiguous in [0, n)
            np.save(directory / "states.npy", self._states[:n])
            np.save(directory / "phase_ids.npy", self._phase_ids[:n])
            np.save(directory / "player_counts.npy", self._player_counts[:n])
            np.save(directory / "legal_masks.npy", self._legal_masks[:n])
            np.save(directory / "policy_targets.npy", self._policy_targets[:n])
            np.save(directory / "value_targets.npy", self._value_targets[:n])
        else:
            # Full buffer: save entire arrays
            np.save(directory / "states.npy", self._states)
            np.save(directory / "phase_ids.npy", self._phase_ids)
            np.save(directory / "player_counts.npy", self._player_counts)
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
                "min_players": self._min_players,
                "max_players": self._max_players,
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
        saved_min_players: int = meta.get("min_players", -1)
        saved_max_players: int = meta.get("max_players", -1)
        saved_unified_dim: int = meta.get("unified_dim", -1)
        saved_size: int = meta.get("size", -1)
        saved_index: int = meta.get("index", -1)
        player_counts_path = directory / "player_counts.npy"
        if (
            saved_capacity != self._capacity
            or saved_state_size != self._state_size
            or saved_num_players != self._num_players
            or saved_min_players != self._min_players
            or saved_max_players != self._max_players
            or saved_unified_dim != self._unified_dim
            or not 0 <= saved_size <= self._capacity
            or not 0 <= saved_index < self._capacity
            or not player_counts_path.exists()
        ):
            print(
                f"  Warning: replay buffer shape mismatch "
                f"(saved cap/state/players/min/max/unified="
                f"{saved_capacity:,}/{saved_state_size}/{saved_num_players}/"
                f"{saved_min_players}/{saved_max_players}/{saved_unified_dim}, "
                f"current={self._capacity:,}/{self._state_size}/"
                f"{self._num_players}/{self._min_players}/{self._max_players}/"
                f"{self._unified_dim}), skipping load"
            )
            return 0
        states = np.load(directory / "states.npy")
        n = states.shape[0]
        if n != saved_size:
            print(
                f"  Warning: replay buffer size mismatch "
                f"(metadata size={saved_size}, states rows={n}), skipping load"
            )
            return 0
        phase_ids = np.load(directory / "phase_ids.npy")
        player_counts = np.load(player_counts_path)
        legal_masks = np.load(directory / "legal_masks.npy")
        policy_targets = np.load(directory / "policy_targets.npy")
        value_targets = np.load(directory / "value_targets.npy")
        expected_shapes = {
            "states.npy": (n, self._state_size),
            "phase_ids.npy": (n,),
            "player_counts.npy": (n,),
            "legal_masks.npy": (n, self._unified_dim),
            "policy_targets.npy": (n, self._unified_dim),
            "value_targets.npy": (n, self._max_players),
        }
        actual_shapes = {
            "states.npy": states.shape,
            "phase_ids.npy": phase_ids.shape,
            "player_counts.npy": player_counts.shape,
            "legal_masks.npy": legal_masks.shape,
            "policy_targets.npy": policy_targets.shape,
            "value_targets.npy": value_targets.shape,
        }
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                print(
                    f"  Warning: replay buffer array shape mismatch "
                    f"({name} saved={actual_shapes[name]}, current={expected}), "
                    "skipping load"
                )
                return 0
        self._states[:n] = states
        self._phase_ids[:n] = phase_ids
        self._player_counts[:n] = player_counts
        self._legal_masks[:n] = legal_masks
        self._policy_targets[:n] = policy_targets
        self._value_targets[:n] = value_targets
        self._size = saved_size
        self._index = saved_index
        return self._size
