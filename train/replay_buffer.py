"""Pre-allocated ring buffer for storing and sampling training examples.

Dense unified-slot schema — mirrors ``train.self_play.GameRecord``: raw
compact int16 game states, per-row ``phase_id`` kept purely for per-phase
TB reporting, dense ``legal_mask`` + ``policy_target`` rows over the game's
``policy.unified_dim`` unified logit slots, and canonical-order per-player
``value_target``. Each row also carries a sampling priority,
KL(policy target || unnoised prior), used when prioritized replay is on, and
its game: the game's id, flags, and final net worths.

Games are written whole and in id order, and the buffer records the first game
id of each epoch, so the newest ``buffer_epochs`` epochs are the ring's newest
run of rows: the window that sampling draws from.

The trainer materializes token and relation inputs per sampled state at
training time. Keeping replay in compact state form avoids storing derived NN
inputs twice.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from games import RELATION_RECORD_WIDTH

if TYPE_CHECKING:
    from games import Game

# Bit 0 of a row's flags: its game started from a supplied position.
FLAG_FROM_POSITION = 1

# What ``load`` gives a buffer saved without these arrays: unknown.
_UNKNOWN = {"priorities": np.nan, "game_ids": -1, "flags": 0, "net_worths": -1}


class TrainingExample(NamedTuple):
    """Single dense training example.

    Paired 1:1 with ``SelfPlayExample`` from ``train.self_play``.
    """

    state: np.ndarray  # (state_size_int16,) int16 — raw compact state
    phase_id: int  # decision phase 0-10 (TB reporting only)
    legal_mask: np.ndarray  # (unified_dim,) uint8 — 1 = legal slot
    policy_target: np.ndarray  # (unified_dim,) f32 — zero on illegal
    value_target: np.ndarray  # (num_players,) f32 — canonical outcome/A0GB blend
    player_count: int  # actual player count for this state


def _write_order(size: int, index: int, capacity: int) -> np.ndarray:
    """Positions of a ring's rows, oldest first.

    A full ring's next write, ``index``, is its oldest row. Until it fills, its
    rows run from position 0.
    """
    positions = np.arange(size)
    return np.roll(positions, -index) if size == capacity else positions


class ReplayBuffer:
    """Pre-allocated numpy ring buffer for dense training examples.

    All arrays are allocated at full capacity upfront. Examples are written
    in a circular fashion, overwriting the oldest when full.
    """

    def __init__(
        self,
        game: Game,
        capacity: int,
        state_size_int16: int,
        num_players: int,
        *,
        min_players: int = 0,
        max_players: int = 0,
        priority_fraction: float = 0.0,
        priority_exponent: float = 0.5,
        buffer_epochs: int = 0,
    ) -> None:
        if buffer_epochs < 0:
            raise ValueError(f"buffer_epochs must be >= 0, got {buffer_epochs}")
        if not 0.0 <= priority_fraction < 1.0:
            raise ValueError(
                f"priority_fraction must be in [0, 1), got {priority_fraction}"
            )
        if priority_exponent < 0:
            raise ValueError(
                f"priority_exponent must be >= 0, got {priority_exponent}"
            )
        self._priority_fraction = priority_fraction
        self._priority_exponent = priority_exponent
        self._buffer_epochs = buffer_epochs
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
        expected_state_size = game.state_size(self._max_players)
        if state_size_int16 != expected_state_size:
            raise ValueError(
                f"state_size_int16 {state_size_int16} does not match "
                f"game.state_size({self._max_players}) "
                f"({expected_state_size})"
            )
        self._game_name = game.name
        self._unified_dim = game.policy.unified_dim
        # A sampled row's relations: planes, or records the caller flattens.
        if game.relation_format == "flat":
            self._relation_shape = (game.max_relation_edges, RELATION_RECORD_WIDTH)
            self._fill_relations_batch = game.fill_relations_sparse_batch
        else:
            num_tokens = game.num_tokens(self._max_players)
            self._relation_shape = (game.num_relations, num_tokens, num_tokens)
            self._fill_relations_batch = game.fill_relations_dense_batch
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
        # Raw KL priorities; NaN marks rows saved before priorities existed.
        self._priorities = np.full(capacity, np.nan, dtype=np.float32)
        # Each row's game: its serial number in the run, never decreasing in
        # write order, its flags (FLAG_*), and its final net worths in
        # canonical player order, -1 past the game's players. Rows saved
        # before these existed have id -1 and net worths -1.
        self._game_ids = np.full(capacity, -1, dtype=np.int32)
        self._flags = np.zeros(capacity, dtype=np.uint8)
        self._net_worths = np.full((capacity, self._max_players), -1, dtype=np.int32)
        self._next_game_id = 0
        # (epoch, its first game id) for each epoch begun, oldest first.
        self._epochs: list[tuple[int, int]] = []
        self._window_warned = False
        # The window and the sampling weights derived from its priorities,
        # rebuilt after writes.
        self._priority_version = 0
        self._window_version = -1
        self._window_rows = 0
        self._window_epochs = 0
        self._sampler_version = -1
        self._sampler_rel = np.empty(0, dtype=np.float32)
        self._sampler_cdf = np.empty(0, dtype=np.float64)

    def _columns(self) -> dict[str, np.ndarray]:
        """Every per-row array, keyed as ``add_stacked`` takes it and as
        ``save`` names its file. Whatever moves rows moves all of these."""
        return {
            "states": self._states,
            "phase_ids": self._phase_ids,
            "legal_masks": self._legal_masks,
            "policy_targets": self._policy_targets,
            "value_targets": self._value_targets,
            "player_counts": self._player_counts,
            "priorities": self._priorities,
            "game_ids": self._game_ids,
            "flags": self._flags,
            "net_worths": self._net_worths,
        }

    def begin_epoch(self, epoch: int) -> None:
        """Record that ``epoch``'s games come next.

        A resume that replays a partly played epoch finds it already begun and
        keeps its start, so the games it played before stay with it. An epoch
        that got no game gives up its entry.
        """
        self._window_warned = False
        if self._epochs and self._epochs[-1][0] == epoch:
            return
        if self._epochs and self._epochs[-1][1] == self._next_game_id:
            self._epochs.pop()
        self._epochs.append((epoch, self._next_game_id))

    @property
    def epochs(self) -> list[tuple[int, int]]:
        """``(epoch, its first game id)`` for each epoch begun, oldest first."""
        return list(self._epochs)

    def add_stacked(
        self,
        states: np.ndarray,
        phase_ids: np.ndarray,
        legal_masks: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        player_counts: np.ndarray | None = None,
        num_players: int | None = None,
        priorities: np.ndarray | None = None,
        game_ids: np.ndarray | None = None,
        flags: np.ndarray | int = 0,
        net_worths: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        """Add pre-stacked arrays directly into the ring buffer.

        All arrays share a leading axis of length ``n``. ``legal_masks``
        and ``policy_targets`` are dense over ``unified_dim`` slots;
        ``policy_target`` is already zero on slots outside the legal set.
        ``value_targets`` may be actual-width ``(n, num_players)`` or
        max-width ``(n, max_players)``. Padded player slots are zeroed.
        ``priorities`` are per-row KL(policy target || unnoised prior); rows
        added without them sample at the average priority.

        Without ``game_ids`` the rows are one game, which takes the buffer's
        next id; ``flags`` and ``net_worths`` (canonical player order, at the
        game's width or the buffer's) are then the game's. A copy of saved
        rows passes all three per row, in the order they were written.
        """
        n = states.shape[0]
        if n == 0:
            return
        game_ids_arr = (
            np.full(n, self._next_game_id, dtype=np.int32) if game_ids is None
            else np.asarray(game_ids, dtype=np.int32)
        )
        if game_ids_arr.shape != (n,):
            raise ValueError(f"game_ids shape {game_ids_arr.shape} != ({n},)")
        net_worths_arr = np.full((n, self._max_players), -1, dtype=np.int32)
        if net_worths is not None:
            given = np.atleast_2d(np.asarray(net_worths, dtype=np.int32))
            net_worths_arr[:, : given.shape[1]] = given
        priorities_arr = (
            np.full(n, np.nan, dtype=np.float32) if priorities is None
            else np.asarray(priorities, dtype=np.float32)
        )
        if priorities_arr.shape != (n,):
            raise ValueError(f"priorities shape {priorities_arr.shape} != ({n},)")
        self._priority_version += 1
        player_counts_arr = self._resolve_player_counts(
            n, value_targets, player_counts, num_players,
        )
        padded_value_targets = self._pad_value_targets(
            value_targets, player_counts_arr,
        )

        self._next_game_id = max(self._next_game_id, int(game_ids_arr[-1]) + 1)
        self._warn_if_window_loses_rows(n)

        # Only the newest ``capacity`` rows of a larger add are kept.
        kept = min(n, self._capacity)
        if n >= self._capacity:
            self._index = 0
        first = min(kept, self._capacity - self._index)
        incoming = {
            "states": states,
            "phase_ids": phase_ids,
            "legal_masks": legal_masks,
            "policy_targets": policy_targets,
            "value_targets": padded_value_targets,
            "player_counts": player_counts_arr,
            "priorities": priorities_arr,
            "game_ids": game_ids_arr,
            "flags": np.broadcast_to(np.asarray(flags, dtype=np.uint8), (n,)),
            "net_worths": net_worths_arr,
        }
        for name, column in self._columns().items():
            rows = incoming[name][n - kept :]
            column[self._index : self._index + first] = rows[:first]
            column[: kept - first] = rows[first:]  # the wrap-around
        self._index = (self._index + kept) % self._capacity
        self._size = min(self._size + n, self._capacity)

    def _warn_if_window_loses_rows(self, incoming: int) -> None:
        """Warn, once an epoch, when ``incoming`` rows will overwrite rows of
        the newest ``buffer_epochs`` epochs: the window outgrew the capacity."""
        evicted = min(self._size + incoming - self._capacity, self._size)
        if evicted <= 0 or not self._buffer_epochs or self._window_warned:
            return
        # With fewer epochs begun, rows from before the record count as one.
        if len(self._epochs) >= self._buffer_epochs:
            newest_evicted = (self._index - self._size + evicted - 1) % self._capacity
            window_start = self._epochs[-self._buffer_epochs][1]
            if self._game_ids[newest_evicted] < window_start:
                return
        self._window_warned = True
        print(
            f"  Warning: the newest {self._buffer_epochs} epochs' rows exceed "
            f"buffer_capacity ({self._capacity:,}); dropping their oldest"
        )

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

    def _window(self) -> tuple[int, int]:
        """The sampling window: its rows, the buffer's newest, and its epochs.

        The window is the newest ``buffer_epochs`` epochs that still hold rows,
        or every row when ``buffer_epochs`` is 0. Rows from before the first
        recorded epoch, as in a buffer saved without the record, are one epoch.
        """
        if self._window_version != self._priority_version:
            n = self._size
            held: list[int] = []  # the first game id of each epoch with rows
            if n:
                oldest = int(self._game_ids[(self._index - n) % self._capacity])
                newest = int(self._game_ids[self._index - 1])
                starts = [start for _, start in self._epochs if start <= newest]
                held = [
                    start for i, start in enumerate(starts)
                    if i + 1 == len(starts) or starts[i + 1] > oldest
                ]
                if not held or oldest < held[0]:
                    held.insert(0, oldest)
            window = held[-self._buffer_epochs :]  # all of them when it is 0
            self._window_rows = n if len(window) == len(held) else int(
                np.count_nonzero(self._game_ids[:n] >= window[0])
            )
            self._window_epochs = len(window)
            self._window_version = self._priority_version
        return self._window_rows, self._window_epochs

    @property
    def window_size(self) -> int:
        """Rows sampling draws from; ``len(self)`` unless ``buffer_epochs`` is set."""
        return self._window()[0]

    @property
    def window_epochs(self) -> int:
        """Epochs with rows in the sampling window."""
        return self._window()[1]

    def _in_window(self, column: np.ndarray) -> np.ndarray:
        """``column``'s rows in the sampling window, oldest first when the
        window is a part of the buffer and in ring order when it is all of it."""
        rows = self.window_size
        if rows == self._size:
            return column[:rows]
        return column.take(
            np.arange(self._index - rows, self._index), axis=0, mode="wrap",
        )

    def _sampling_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rel, cdf)`` over the window's rows for priority sampling.

        Row weight is ``priority ** exponent``; rows without a priority take
        the mean weight. ``rel`` is each weight relative to the mean (its
        prioritized sampling rate versus uniform), ``cdf`` the running sum.
        """
        if self._sampler_version != self._priority_version:
            kl = self._in_window(self._priorities).astype(np.float64)
            n = len(kl)
            known = np.isfinite(kl)
            weights = np.empty(n, dtype=np.float64)
            weights[known] = np.power(kl[known], self._priority_exponent)
            weights[~known] = weights[known].mean() if known.any() else 1.0
            total = float(weights.sum())
            if total <= 0.0:
                weights.fill(1.0)
                total = float(n)
            self._sampler_rel = (weights * (n / total)).astype(np.float32)
            self._sampler_cdf = np.cumsum(weights)
            self._sampler_version = self._priority_version
        return self._sampler_rel, self._sampler_cdf

    def _sample_indices(
        self, batch_size: int, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Draw batch indices and their importance weights versus uniform.

        Rows come from the sampling window (``_window``). Without
        prioritization this is one uniform draw without replacement
        and the weights are ``None``. Otherwise ``round(fraction * B)`` rows
        come from the priority distribution (with replacement) and the rest
        uniformly. Each row's weight, uniform probability over its mixture
        probability, makes weighted batch averages estimate uniform ones.
        """
        rows = self.window_size
        if batch_size > rows:
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({rows})"
            )
        weights = None
        if self._priority_fraction == 0.0:
            indices = rng.choice(rows, size=batch_size, replace=False)
        else:
            k = round(self._priority_fraction * batch_size)
            fraction = k / batch_size
            rel, cdf = self._sampling_weights()
            uniform = rng.choice(rows, size=batch_size - k, replace=False)
            prioritized = np.minimum(
                np.searchsorted(cdf, rng.random(k) * cdf[-1], side="right"),
                rows - 1,
            )
            indices = np.concatenate((uniform, prioritized))
            weights = (
                1.0 / ((1.0 - fraction) + fraction * rel[indices])
            ).astype(np.float32)
        # A window short of the buffer is the rows up to the write head.
        if rows < self._size:
            indices = (indices + self._index - rows) % self._capacity
        return indices, weights

    def priority_stats(self) -> dict[str, float]:
        """Priority diagnostics for Tensorboard (empty while the buffer is),
        over the rows sampling draws from.

        ``priority_kl_mean`` averages known priorities. With prioritization
        on, ``priority_max_rate`` is the most-sampled row's rate versus
        uniform and ``priority_ess_fraction`` the effective sample size of
        the sampling distribution as a share of those rows.
        """
        if self._size == 0:
            return {}
        kl = self._in_window(self._priorities)
        known = np.isfinite(kl)
        stats = {"priority_unknown_fraction": float(1.0 - known.mean())}
        if known.any():
            stats["priority_kl_mean"] = float(kl[known].mean())
        if self._priority_fraction > 0.0:
            rel, _ = self._sampling_weights()
            f = self._priority_fraction
            rate = (1.0 - f) + f * rel.astype(np.float64)
            stats["priority_max_rate"] = float(rate.max())
            stats["priority_ess_fraction"] = float(1.0 / np.mean(rate**2))
        return stats

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch. Returns dict of torch tensors (CPU).

        Relation inputs are generated from the sampled compact states rather
        than stored in the ring buffer for transformer-oriented callers:
        ``(B, R, T, T)`` planes, or for a flat game ``(B, E, 4)`` records, which
        the caller flattens.

        ``is_weights`` are the rows' importance weights versus uniform
        sampling (all 1 without prioritization).

        Raises ValueError if batch_size > current buffer size.
        """
        indices, weights = self._sample_indices(batch_size, rng)
        if weights is None:
            weights = np.ones(batch_size, dtype=np.float32)
        states = self._states[indices]
        relations = np.empty(
            (batch_size, *self._relation_shape), dtype=np.uint8,
        )
        self._fill_relations_batch(
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
            "is_weights": torch.from_numpy(weights),
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
        is_weights_out: np.ndarray | None = None,
    ) -> None:
        """Fill caller-provided arrays with a random batch.

        Unlike ``sample``, this writes directly into existing buffers —
        used by the trainer to fill pinned host scratch, so the
        subsequent H→D copy is genuinely async. Integer outputs may be
        wider than the stored dtype (e.g. int64); widening happens
        during the fancy-index copy. If ``relations_out`` is supplied,
        relation inputs are generated from the sampled states into that
        caller-owned scratch buffer: planes, or for a flat game ``(B, E, 4)``
        records, which the caller flattens. ``is_weights_out`` receives
        importance weights versus uniform sampling (all 1 without
        prioritization).
        """
        indices, weights = self._sample_indices(batch_size, rng)
        if is_weights_out is not None:
            is_weights_out[:] = 1.0 if weights is None else weights
        states_out[:] = self._states[indices]
        phase_ids_out[:] = self._phase_ids[indices]
        if player_counts_out is not None:
            player_counts_out[:] = self._player_counts[indices]
        legal_masks_out[:] = self._legal_masks[indices]
        policy_targets_out[:] = self._policy_targets[indices]
        value_targets_out[:] = self._value_targets[indices]
        if relations_out is not None:
            self._fill_relations_batch(
                [states_out[i] for i in range(batch_size)],
                relations_out,
                max_players=self._max_players,
            )

    def rows_oldest_first(self) -> dict[str, np.ndarray]:
        """Copies of the occupied rows' arrays, in write order, keyed as
        ``add_stacked`` takes them (``player_counts`` for the counts)."""
        order = _write_order(self._size, self._index, self._capacity)
        return {name: column[order] for name, column in self._columns().items()}

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    def save(self, directory: Path) -> None:
        """Save buffer contents to directory for later resume.

        Writes individual .npy files for each array (only the occupied portion)
        plus metadata with the game, size/index/capacity/player range/shape
        details, the next game id, and the epoch record.
        """
        if self._size == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        for name, column in self._columns().items():
            np.save(directory / f"{name}.npy", column[: self._size])
        (directory / "metadata.json").write_text(
            json.dumps({
                "game": self._game_name,
                "size": self._size,
                "index": self._index,
                "capacity": self._capacity,
                "state_size": self._state_size,
                "num_players": self._num_players,
                "min_players": self._min_players,
                "max_players": self._max_players,
                "unified_dim": self._unified_dim,
                "next_game_id": self._next_game_id,
                "epochs": self._epochs,
            })
        )

    def load(self, directory: Path) -> int:
        """Load buffer contents from directory.

        Returns the number of examples loaded (0 if directory missing, another
        game's, or shape mismatch). A buffer saved at another capacity gives
        its newest rows that fit, in order. Metadata without a game is RSS's.
        A buffer saved before priorities, game ids, flags, or net worths
        existed loads them as unknown; one saved without the epoch record
        loads as rows from before any epoch.
        """
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return 0
        meta = json.loads(metadata_path.read_text())
        saved_game: str = meta.get("game", "rss")
        if saved_game != self._game_name:
            print(
                f"  Warning: replay buffer is from game {saved_game!r}, "
                f"not {self._game_name!r}, skipping load"
            )
            return 0
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
            saved_state_size != self._state_size
            or saved_num_players != self._num_players
            or saved_min_players != self._min_players
            or saved_max_players != self._max_players
            or saved_unified_dim != self._unified_dim
            or not 0 <= saved_size <= saved_capacity
            or not 0 <= saved_index < saved_capacity
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
        saved: dict[str, np.ndarray] = {}
        for name, column in self._columns().items():
            path = directory / f"{name}.npy"
            saved[name] = (
                np.load(path) if name not in _UNKNOWN or path.exists()
                else np.full(
                    (saved_size, *column.shape[1:]), _UNKNOWN[name], dtype=column.dtype,
                )
            )
        n = saved["states"].shape[0]
        if n != saved_size:
            print(
                f"  Warning: replay buffer size mismatch "
                f"(metadata size={saved_size}, states rows={n}), skipping load"
            )
            return 0
        for name, column in self._columns().items():
            expected = (n, *column.shape[1:])
            if saved[name].shape != expected:
                print(
                    f"  Warning: replay buffer array shape mismatch "
                    f"({name}.npy saved={saved[name].shape}, current={expected}), "
                    "skipping load"
                )
                return 0
        rows = slice(None)
        kept = n
        if saved_capacity == self._capacity:
            self._index = saved_index
        else:
            kept = min(n, self._capacity)
            rows = _write_order(n, saved_index, saved_capacity)[n - kept:]
            self._index = kept % self._capacity
            print(
                f"  Replay buffer capacity changed "
                f"({saved_capacity:,} -> {self._capacity:,}): "
                f"keeping the newest {kept:,} of {n:,} rows"
            )
        for name, column in self._columns().items():
            column[:kept] = saved[name][rows]
        self._next_game_id = meta.get("next_game_id", 0)
        self._epochs = [(epoch, start) for epoch, start in meta.get("epochs", [])]
        if self._buffer_epochs and "epochs" not in meta:
            print(
                f"  Replay buffer has no epoch record: its {kept:,} rows count "
                "as one epoch toward buffer_epochs"
            )
        self._priority_version += 1
        self._size = kept
        return self._size
