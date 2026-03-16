"""Per-game cache for NN evaluation results.

Avoids redundant GPU forward passes when MCTS explores states that were
already evaluated in earlier searches within the same game. This is the
lightweight alternative to subtree reuse: each move starts with a fresh
MCTS tree (so Dirichlet noise is fully effective), but cached NN results
mean we don't pay full GPU cost to rebuild it.

Backed by pre-allocated numpy matrices with a dict-based hash index.
The cache stores (policy, value) tuples keyed by a 128-bit MD5 hash of
the state array. At 131K entries the collision probability is ~10^-28.

Memory per entry: ~1.1KB (984B policy + 12B values + ~100B index overhead).
At 131K entries: ~140MB per worker. With 24 workers: ~3.4GB total.
"""

from __future__ import annotations

import hashlib

import numpy as np


class EvalCache:
    """Matrix-backed NN evaluation cache with hash index.

    Stores policy and value outputs keyed by state array hash.
    Pre-allocates numpy matrices and doubles capacity when full.
    """

    __slots__ = (
        "_policies", "_values", "_index",
        "_count", "_capacity", "_action_dim", "_num_players",
    )

    def __init__(
        self,
        action_dim: int,
        num_players: int,
        initial_capacity: int = 2048,
    ) -> None:
        self._action_dim = action_dim
        self._num_players = num_players
        self._capacity = initial_capacity
        self._count = 0
        self._policies = np.zeros((initial_capacity, action_dim), dtype=np.float32)
        self._values = np.zeros((initial_capacity, num_players), dtype=np.float32)
        self._index: dict[bytes, int] = {}

    def lookup(self, state_array: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Look up cached evaluation for a state.

        Args:
            state_array: Full game state array (visible + hidden).

        Returns:
            (policy_probs, canonical_values) if cached, else None.
            Returned arrays are views into the backing matrices.
        """
        key = hashlib.md5(state_array).digest()
        idx = self._index.get(key)
        if idx is None:
            return None
        return self._policies[idx], self._values[idx]

    def store(
        self,
        state_array: np.ndarray,
        policy: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Store an evaluation result.

        Args:
            state_array: Full game state array (the hash key source).
            policy: Policy probabilities, shape (action_dim,).
            values: Canonical per-player values, shape (num_players,).
        """
        key = hashlib.md5(state_array).digest()
        if key in self._index:
            return
        if self._count >= self._capacity:
            self._grow()
        idx = self._count
        self._policies[idx] = policy
        self._values[idx] = values
        self._index[key] = idx
        self._count += 1

    def clear(self) -> None:
        """Clear all cached entries for a new game.

        Resets the write cursor and index but keeps allocated matrices.
        """
        self._count = 0
        self._index.clear()

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return self._count

    def _grow(self) -> None:
        """Double the backing matrix capacity."""
        new_cap = self._capacity * 2
        new_policies = np.zeros((new_cap, self._action_dim), dtype=np.float32)
        new_values = np.zeros((new_cap, self._num_players), dtype=np.float32)
        new_policies[:self._count] = self._policies[:self._count]
        new_values[:self._count] = self._values[:self._count]
        self._policies = new_policies
        self._values = new_values
        self._capacity = new_cap
