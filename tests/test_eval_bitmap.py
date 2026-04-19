"""Direct tests for the eval-server signaling bitmap primitives.

The multi-word submitted bitmap in ``mcts/mcts_core.pyx`` has only ever been
production-smoked by running self-play. These tests exercise the Cython
primitives directly with synthetic memoryviews to pin down correctness of
word indexing, drain ordering, per-word doorbell semantics, and cache-line
padded layout.

Layout under test (from ``SharedEvalBuffers.init_bitmap``):
    shape ``(num_servers * W, 8)`` uint64. Word ``w`` of server ``s`` lives
    at row ``s*W + w``, column 0. Columns 1..7 are dead padding (one cache
    line per word at 8-byte words × 8 = 64 bytes).
"""

from __future__ import annotations

import numpy as np

from mcts.mcts_core import (
    server_drain_bitmap,
    server_peek_bitmap,
    worker_publish_request,
)


def _make_bitmap(num_servers: int, num_words: int) -> np.ndarray:
    """Allocate a cache-line-padded bitmap matching the production shape."""
    return np.zeros((num_servers * num_words, 8), dtype=np.uint64)


def _drain_one(
    masks: np.ndarray,
    counts: np.ndarray,
    server_id: int,
    partition_start: int,
    num_words: int,
    capacity: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    widx = np.empty(capacity, dtype=np.int32)
    cnt = np.empty(capacity, dtype=np.int32)
    n = server_drain_bitmap(
        masks, counts, widx, cnt, server_id, partition_start, num_words,
    )
    return n, widx[:n].copy(), cnt[:n].copy()


# ---------------------------------------------------------------------------
# Single-server, single-word (W=1) — structural identity to the pre-refactor
# code path. Every test here should pass trivially if the new code preserves
# the old semantics for partition_size <= 64.
# ---------------------------------------------------------------------------

def test_publish_bit_zero_word_zero():
    """local_idx=0 → word 0, bit 0. First publish flips 0→nonzero."""
    masks = _make_bitmap(1, 1)
    counts = np.zeros(1, dtype=np.int32)
    became_nonempty = worker_publish_request(
        masks, counts, worker_idx=0, server_id=0, local_idx=0,
        state_count=7, num_words=1,
    )
    assert became_nonempty
    assert counts[0] == 7
    assert masks[0, 0] == 1

    n, widx, cnt = _drain_one(masks, counts, 0, 0, 1, capacity=1)
    assert n == 1
    assert widx[0] == 0
    assert cnt[0] == 7
    # Drain clears the word.
    assert masks[0, 0] == 0


def test_publish_bit_63_word_zero():
    """local_idx=63 → word 0, bit 63 (top bit of the first word)."""
    masks = _make_bitmap(1, 1)
    counts = np.zeros(64, dtype=np.int32)
    became_nonempty = worker_publish_request(
        masks, counts, worker_idx=63, server_id=0, local_idx=63,
        state_count=3, num_words=1,
    )
    assert became_nonempty
    assert masks[0, 0] == (np.uint64(1) << np.uint64(63))

    n, widx, _ = _drain_one(masks, counts, 0, 0, 1, capacity=1)
    assert n == 1
    assert widx[0] == 63


def test_drain_empty_bitmap_returns_zero():
    masks = _make_bitmap(1, 1)
    counts = np.zeros(1, dtype=np.int32)
    n, _, _ = _drain_one(masks, counts, 0, 0, 1, capacity=8)
    assert n == 0


def test_peek_empty_and_nonempty_single_word():
    masks = _make_bitmap(1, 1)
    counts = np.zeros(1, dtype=np.int32)
    assert not server_peek_bitmap(masks, 0, 1)
    worker_publish_request(
        masks, counts, worker_idx=0, server_id=0, local_idx=5,
        state_count=1, num_words=1,
    )
    assert server_peek_bitmap(masks, 0, 1)


def test_second_publish_same_word_does_not_signal():
    """With the word already nonzero, became_nonempty must be False."""
    masks = _make_bitmap(1, 1)
    counts = np.zeros(64, dtype=np.int32)
    first = worker_publish_request(
        masks, counts, worker_idx=0, server_id=0, local_idx=0,
        state_count=1, num_words=1,
    )
    second = worker_publish_request(
        masks, counts, worker_idx=1, server_id=0, local_idx=1,
        state_count=1, num_words=1,
    )
    assert first
    assert not second


# ---------------------------------------------------------------------------
# Multi-word (W >= 2) — the new code path that lifts the 64-worker cap.
# ---------------------------------------------------------------------------

def test_publish_local_idx_64_lands_in_word_one():
    """local_idx=64 → word 1, bit 0. Must not collide with local_idx=0."""
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    became_nonempty = worker_publish_request(
        masks, counts, worker_idx=64, server_id=0, local_idx=64,
        state_count=11, num_words=2,
    )
    assert became_nonempty
    # Word 0 untouched, word 1 has bit 0 set.
    assert masks[0, 0] == 0
    assert masks[1, 0] == 1

    n, widx, cnt = _drain_one(masks, counts, 0, 0, 2, capacity=1)
    assert n == 1
    assert widx[0] == 64
    assert cnt[0] == 11


def test_publish_local_idx_127_lands_in_word_one_top_bit():
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    worker_publish_request(
        masks, counts, worker_idx=127, server_id=0, local_idx=127,
        state_count=2, num_words=2,
    )
    assert masks[0, 0] == 0
    assert masks[1, 0] == (np.uint64(1) << np.uint64(63))

    n, widx, _ = _drain_one(masks, counts, 0, 0, 2, capacity=1)
    assert n == 1
    assert widx[0] == 127


def test_drain_across_multiple_words():
    """Publishes into word 0 and word 1 both come out of a single drain,
    with worker indices recovered correctly."""
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    # Fan out across both words: one in word 0 (local_idx=3), one in word 1
    # (local_idx=64+5 = 69).
    worker_publish_request(
        masks, counts, worker_idx=3, server_id=0, local_idx=3,
        state_count=10, num_words=2,
    )
    worker_publish_request(
        masks, counts, worker_idx=69, server_id=0, local_idx=69,
        state_count=20, num_words=2,
    )

    n, widx, cnt = _drain_one(masks, counts, 0, 0, 2, capacity=8)
    assert n == 2
    # Drain walks word 0 first, then word 1 (ctz within each word).
    assert widx.tolist() == [3, 69]
    assert cnt.tolist() == [10, 20]


def test_peek_fires_when_only_word_one_is_set():
    """peek must short-circuit on *any* non-empty word, not just word 0."""
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    worker_publish_request(
        masks, counts, worker_idx=80, server_id=0, local_idx=80,
        state_count=1, num_words=2,
    )
    assert masks[0, 0] == 0  # word 0 is empty
    assert masks[1, 0] != 0  # word 1 has the bit
    assert server_peek_bitmap(masks, 0, 2)


def test_per_word_became_nonempty_over_signals_across_words():
    """Per-word rule: publishing to an empty word returns True even if a
    sibling word is already non-empty. This is the bounded over-signal
    documented in more-workers.md; it's correctness-safe because
    ``mp.Event.set`` is idempotent.
    """
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    # Fill word 0 first.
    worker_publish_request(
        masks, counts, worker_idx=0, server_id=0, local_idx=0,
        state_count=1, num_words=2,
    )
    # Publishing into (still-empty) word 1 must return True, even though
    # the server-visible "is there any work" state didn't change.
    became_nonempty = worker_publish_request(
        masks, counts, worker_idx=64, server_id=0, local_idx=64,
        state_count=1, num_words=2,
    )
    assert became_nonempty


def test_drain_returns_zero_across_empty_words():
    """Multi-word empty drain returns 0 (all W exchange-to-zero hits 0)."""
    masks = _make_bitmap(1, 4)
    counts = np.zeros(256, dtype=np.int32)
    n, _, _ = _drain_one(masks, counts, 0, 0, 4, capacity=8)
    assert n == 0


# ---------------------------------------------------------------------------
# Multi-server cache-line-padded layout: server s's words live at rows
# [s*W, s*W+W); writes by server s's workers must not touch server s'≠s's
# words.
# ---------------------------------------------------------------------------

def test_multi_server_isolation_word_zero():
    """Server 0 and server 1 share no bits: each server's publish only
    sets bits within its own row range."""
    num_words = 2
    masks = _make_bitmap(num_servers=2, num_words=num_words)
    counts = np.zeros(256, dtype=np.int32)

    # Server 0 publishes local_idx=5 (partition_start=0, worker_idx=5).
    worker_publish_request(
        masks, counts, worker_idx=5, server_id=0, local_idx=5,
        state_count=1, num_words=num_words,
    )
    # Server 1 publishes local_idx=70 (partition_start=128, worker_idx=198).
    worker_publish_request(
        masks, counts, worker_idx=198, server_id=1, local_idx=70,
        state_count=1, num_words=num_words,
    )

    # Server 0 rows: 0, 1. Server 1 rows: 2, 3.
    assert masks[0, 0] != 0
    assert masks[1, 0] == 0  # server 0's word 1 untouched
    assert masks[2, 0] == 0  # server 1's word 0 untouched (local_idx 70 → word 1)
    assert masks[3, 0] != 0  # server 1's word 1 carries local_idx 70

    # Peek each server separately.
    assert server_peek_bitmap(masks, 0, num_words)
    assert server_peek_bitmap(masks, 1, num_words)

    # Drain each server and verify isolation + correct worker_idx recovery.
    n0, w0, _ = _drain_one(masks, counts, 0, partition_start=0,
                           num_words=num_words, capacity=8)
    n1, w1, _ = _drain_one(masks, counts, 1, partition_start=128,
                           num_words=num_words, capacity=8)
    assert n0 == 1 and w0[0] == 5
    assert n1 == 1 and w1[0] == 198


def test_padding_columns_are_untouched_by_publish():
    """Publish must only touch column 0 of the word's row. The dead
    padding columns (1..7) must remain zero, since a future refactor that
    accidentally writes into them would silently break cache-line isolation.
    """
    masks = _make_bitmap(num_servers=2, num_words=2)
    counts = np.zeros(256, dtype=np.int32)
    worker_publish_request(
        masks, counts, worker_idx=0, server_id=0, local_idx=0,
        state_count=1, num_words=2,
    )
    worker_publish_request(
        masks, counts, worker_idx=100, server_id=1, local_idx=36,
        state_count=1, num_words=2,
    )
    assert np.all(masks[:, 1:] == 0)


def test_full_word_drain_recovers_all_64_bits():
    """Every local_idx in [0, 64) publishes into word 0 and drains back."""
    masks = _make_bitmap(1, 2)
    counts = np.zeros(128, dtype=np.int32)
    for i in range(64):
        counts[i] = i + 1
        worker_publish_request(
            masks, counts, worker_idx=i, server_id=0, local_idx=i,
            state_count=i + 1, num_words=2,
        )
    n, widx, cnt = _drain_one(masks, counts, 0, 0, 2, capacity=128)
    assert n == 64
    assert widx.tolist() == list(range(64))
    assert cnt.tolist() == [i + 1 for i in range(64)]
