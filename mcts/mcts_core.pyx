# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython implementations of MCTS hot functions.

Replaces pure-Python/numpy versions in search.py and evaluator.py,
eliminating numpy/torch dispatch overhead for small arrays where the
overhead dominates actual computation.
"""

from libc.math cimport sqrtf, expf
from libc.string cimport memcpy

cdef extern from "<assert.h>" nogil:
    void assert_c "assert" (bint expression)

import numpy as np


# ---------------------------------------------------------------------------
# Shared-memory signaling primitives (eval server <-> worker communication)
# ---------------------------------------------------------------------------
# GCC built-in atomics provide correct memory ordering on both x86 (TSO) and
# ARM (weak ordering, needs explicit barriers).
#
# Protocol uses a per-server uint64 bitmap where bit i means local worker i
# has a pending request. Workers publish via atomic fetch-or (release),
# servers claim all pending work via atomic exchange (acquire). This gives
# O(1) empty check and O(k) drain for k ready workers.

from libc.stdint cimport uint64_t

cdef extern from *:
    """
    #include <stdint.h>
    #define ATOMIC_FETCH_OR_U64(ptr, val) __atomic_fetch_or(ptr, val, __ATOMIC_RELEASE)
    #define ATOMIC_EXCHANGE_U64(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_ACQUIRE)
    #define ATOMIC_LOAD_U64(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define CTZ64(x) __builtin_ctzll(x)
    """
    uint64_t ATOMIC_FETCH_OR_U64(uint64_t* ptr, uint64_t val) nogil
    uint64_t ATOMIC_EXCHANGE_U64(uint64_t* ptr, uint64_t val) nogil
    uint64_t ATOMIC_LOAD_U64(uint64_t* ptr) nogil
    int CTZ64(uint64_t x) nogil


def worker_publish_request(
    uint64_t[:] submitted_masks,
    int[:] counts,
    int worker_idx,
    int server_id,
    int local_idx,
    int state_count,
):
    """Worker-side: write count, atomically set bit in server's submitted mask.

    The release fence on fetch_or ensures the server sees state data and
    count writes that preceded this call.

    Returns True if the server's mask transitioned from 0 -> non-zero
    (caller should set the server's doorbell event to wake it).
    """
    counts[worker_idx] = state_count
    cdef uint64_t bit = <uint64_t>1 << <uint64_t>local_idx
    cdef uint64_t old_mask = ATOMIC_FETCH_OR_U64(&submitted_masks[server_id], bit)
    return old_mask == 0


def server_drain_bitmap(
    uint64_t[:] submitted_masks,
    int[:] counts,
    int[:] out_worker_indices,
    int[:] out_counts,
    int server_id,
    int partition_start,
):
    """Server-side: atomically claim all pending requests from the bitmap.

    Exchanges the server's submitted_mask with 0 (acquire), then iterates
    set bits via ctz to build worker_indices and counts arrays. O(k) in
    the number of ready workers.

    Returns the number of requests found.
    """
    cdef uint64_t mask = ATOMIC_EXCHANGE_U64(&submitted_masks[server_id], 0)
    cdef int n = 0
    cdef int local_idx, worker_idx
    with nogil:
        while mask != 0:
            local_idx = CTZ64(mask)
            worker_idx = partition_start + local_idx
            out_worker_indices[n] = worker_idx
            out_counts[n] = counts[worker_idx]
            n = n + 1
            mask = mask & (mask - 1)  # clear lowest set bit
    return n


def server_peek_bitmap(uint64_t[:] submitted_masks, int server_id):
    """Server-side: acquire-load the submitted mask without modifying it.

    Used for the lost-wakeup recheck: after clearing the doorbell event,
    peek to see if new work arrived before sleeping.
    """
    return ATOMIC_LOAD_U64(&submitted_masks[server_id]) != 0


cdef (int, int) _select_child_impl(
    const int[:] legal_actions, const float[:] priors,
    const int[:] visit_counts, const float[:, :] value_sums,
    int active_player_id, int parent_visit_count, float c_puct,
) noexcept nogil:
    """Select the child action with the highest PUCT value.

    Pure C loop — no Python objects, no numpy dispatch.

    UCB(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

    Q(a) = value_sums[a, active_player] / max(1, visit_counts[a])
    Unvisited actions (vc=0) use the FPU default stored in value_sums.

    Returns (action_index, array_index).
    """
    cdef int n = legal_actions.shape[0]
    cdef float sqrt_parent = sqrtf(<float>parent_visit_count)
    cdef float best_ucb = -1e30
    cdef int best_idx = 0
    cdef int i, vc
    cdef float q, ucb

    for i in range(n):
        vc = visit_counts[i]
        if vc > 0:
            q = value_sums[i, active_player_id] / <float>vc
        else:
            q = value_sums[i, active_player_id]
        ucb = q + c_puct * priors[i] * sqrt_parent / (1.0 + <float>vc)
        if ucb > best_ucb:
            best_ucb = ucb
            best_idx = i

    return legal_actions[best_idx], best_idx


def select_child(node, float c_puct):
    """Select the child action with the highest PUCT value.

    Drop-in replacement for search.py:select_child. Extracts typed
    memoryviews from MCTSNode attributes and delegates to the nogil
    C implementation.

    Args:
        node: MCTSNode (must be expanded).
        c_puct: Exploration constant.

    Returns:
        Tuple of (action_index, array_index).
    """
    cdef const int[:] legal_actions = node.legal_actions
    cdef const float[:] priors = node.priors
    cdef const int[:] vc = node.visit_counts
    cdef const float[:, :] vs = node.value_sums
    cdef int player = node.active_player_id
    cdef int parent_vc = node.visit_count

    cdef int action_idx, array_idx
    action_idx, array_idx = _select_child_impl(
        legal_actions, priors, vc, vs, player, parent_vc, c_puct,
    )
    return action_idx, array_idx


cdef void _backup_node(
    float[:] value_sum, float[:, :] value_sums,
    const int[:] visit_counts, int array_idx,
    const float[:] values, int num_players,
) noexcept nogil:
    """Update a single node's value_sum and value_sums during backup.

    On the first real visit (visit_counts[array_idx] == 1), replaces the
    FPU default value instead of adding to it. Otherwise accumulates.
    """
    cdef int p
    cdef bint first_visit = (visit_counts[array_idx] == 1)

    for p in range(num_players):
        value_sum[p] = value_sum[p] + values[p]
        if first_visit:
            value_sums[array_idx, p] = values[p]
        else:
            value_sums[array_idx, p] = value_sums[array_idx, p] + values[p]


cdef void _virtual_backup_node(
    float[:] value_sum, float[:, :] value_sums,
    int[:] visit_counts, int array_idx,
    const float[:] child_q, int num_players,
) noexcept nogil:
    """Update root stats for one virtual backup during subtree reuse catch-up.

    Like _backup_node but also increments visit_counts[array_idx] (the root
    manages its own visit count increment in the caller). On the first visit
    (visit_counts goes from 0 to 1), replaces the FPU default; otherwise adds.
    """
    cdef int p
    visit_counts[array_idx] = visit_counts[array_idx] + 1
    cdef bint first_visit = (visit_counts[array_idx] == 1)

    for p in range(num_players):
        value_sum[p] = value_sum[p] + child_q[p]
        if first_visit:
            value_sums[array_idx, p] = child_q[p]
        else:
            value_sums[array_idx, p] = value_sums[array_idx, p] + child_q[p]


def virtual_backup(root, child, int array_idx):
    """Perform one virtual backup at the root during subtree reuse.

    Computes the child's mean Q value and updates the root's visit counts,
    value_sum, and value_sums using typed memoryviews and a nogil C helper.
    The child node is not modified.

    Args:
        root: MCTSNode (must be expanded). Modified in place.
        child: Child MCTSNode with existing visits.
        array_idx: Index into root's per-action arrays for this child.
    """
    cdef int num_players = root.value_sum.shape[0]
    child_q = child.value_sum / child.visit_count
    cdef float[:] child_q_view = child_q
    cdef float[:] root_value_sum = root.value_sum
    cdef float[:, :] vs = root.value_sums
    cdef int[:] vc = root.visit_counts

    _virtual_backup_node(root_value_sum, vs, vc, array_idx, child_q_view, num_players)
    root.visit_count += 1


def backup(list path, leaf, values_np):
    """Backpropagate values from a leaf up to the root.

    Drop-in replacement for search.py:_backup. Iterates the Python path
    list in reverse; per-node work uses typed memoryviews and nogil.

    Args:
        path: List of (parent_node, action, array_idx) tuples.
        leaf: The leaf MCTSNode that was evaluated.
        values_np: Canonical per-player values, shape (num_players,).
    """
    cdef float[:] values = values_np
    cdef int num_players = values.shape[0]
    cdef int array_idx
    cdef float[:] value_sum
    cdef float[:, :] value_sums
    cdef const int[:] visit_counts

    # Update leaf value_sum
    cdef float[:] leaf_vs = leaf.value_sum
    cdef int p
    for p in range(num_players):
        leaf_vs[p] = leaf_vs[p] + values[p]

    # Backup through path in reverse
    cdef int i
    for i in range(len(path) - 1, -1, -1):
        node = path[i][0]
        array_idx = path[i][2]
        value_sum = node.value_sum
        value_sums = node.value_sums
        visit_counts = node.visit_counts
        _backup_node(value_sum, value_sums, visit_counts, array_idx, values, num_players)


def increment_visits(list path, leaf):
    """Increment visit counts along a selection path and on the leaf.

    Drop-in replacement for search.py:_increment_visits. Uses typed
    memoryviews for the inner visit_counts array update.

    Args:
        path: List of (parent_node, action, array_idx) tuples.
        leaf: The selected leaf MCTSNode.
    """
    leaf.visit_count += 1
    cdef int array_idx
    cdef int[:] vc
    cdef int i
    for i in range(len(path)):
        node = path[i][0]
        array_idx = path[i][2]
        node.visit_count += 1
        vc = node.visit_counts
        vc[array_idx] = vc[array_idx] + 1


# ---------------------------------------------------------------------------
# State rotation (replaces numpy roll/copy in evaluator.py)
# ---------------------------------------------------------------------------

cdef void _rotate_visible_state(
    float* dst, const float* src,
    int visible_size, int active_player_id, int num_players,
    int players_offset, int player_stride,
    int field0_offset, int field1_offset, int field2_offset,
) noexcept nogil:
    """Copy visible state from src to dst with player rotation.

    Copies the full visible region, then overwrites player blocks and
    per-player turn fields in rotated order. Reads from src only, so
    there is no aliasing issue even though dst starts as a copy of src.
    """
    # Bulk copy entire visible state
    memcpy(dst, src, visible_size * sizeof(float))

    if active_player_id == 0:
        return

    # Overwrite player data blocks in rotated order.
    # np.roll(blocks, -active_player_id, axis=0) means:
    #   dst_slot[i] = src_slot[(i + active_player_id) % num_players]
    cdef int i, src_player
    cdef int byte_stride = player_stride * sizeof(float)
    for i in range(num_players):
        src_player = (i + active_player_id) % num_players
        memcpy(
            dst + players_offset + i * player_stride,
            src + players_offset + src_player * player_stride,
            byte_stride,
        )

    # Rotate 3 per-player turn state fields (each num_players floats)
    cdef int field_offsets[3]
    field_offsets[0] = field0_offset
    field_offsets[1] = field1_offset
    field_offsets[2] = field2_offset
    cdef int f, off
    for f in range(3):
        off = field_offsets[f]
        for i in range(num_players):
            src_player = (i + active_player_id) % num_players
            dst[off + i] = src[off + src_player]


def rotate_visible_state_into(
    float[:] dst, const float[:] src,
    int active_player_id, int num_players,
    int visible_size, int players_offset, int player_stride,
    int field0_offset, int field1_offset, int field2_offset,
):
    """Copy visible state into dst with player rotation.

    Drop-in replacement for evaluator.rotate_visible_state_into.
    Uses memcpy and pointer arithmetic instead of numpy roll/copy.

    Args:
        dst: Destination buffer, shape (visible_size,).
        src: Full state array (visible + hidden).
        active_player_id: Canonical player ID (0 to num_players-1).
        num_players: Number of players.
        visible_size: Size of visible state region.
        players_offset: Offset to player data blocks.
        player_stride: Floats per player block.
        field0_offset: Offset to auction_high_bidder per-player field.
        field1_offset: Offset to auction_starter per-player field.
        field2_offset: Offset to auction_passed per-player field.
    """
    _rotate_visible_state(
        &dst[0], &src[0],
        visible_size, active_player_id, num_players,
        players_offset, player_stride,
        field0_offset, field1_offset, field2_offset,
    )


# ---------------------------------------------------------------------------
# Masked softmax (replaces torch round-trip in evaluator.py)
# ---------------------------------------------------------------------------

cdef void _masked_softmax(
    float* out, const float* logits, const float* mask, int n,
) noexcept nogil:
    """Numerically stable masked softmax: mask, shift, exp, normalize.

    Illegal actions (mask <= 0) get probability 0.
    """
    cdef int i
    cdef float max_val = -1e30
    cdef float v

    # Find max of legal logits for numerical stability
    for i in range(n):
        if mask[i] > 0.0:
            v = logits[i]
            if v > max_val:
                max_val = v

    # Compute exp(logit - max) for legal actions, sum
    cdef float total = 0.0
    for i in range(n):
        if mask[i] > 0.0:
            out[i] = expf(logits[i] - max_val)
            total = total + out[i]
        else:
            out[i] = 0.0

    # Normalize
    if total > 0.0:
        for i in range(n):
            out[i] = out[i] / total


def masked_softmax(const float[:] logits, const float[:] mask):
    """Apply legal action mask and softmax to raw policy logits.

    Drop-in replacement for evaluator.apply_mask_softmax. Uses a single
    C loop instead of torch tensor round-trip.

    Args:
        logits: Raw logits from NN, shape (action_dim,).
        mask: Binary float32 mask (1.0 = legal, 0.0 = illegal).

    Returns:
        Probability distribution over actions, shape (action_dim,).
    """
    cdef int n = logits.shape[0]
    assert mask.shape[0] == n, f"logits length ({n}) != mask length ({mask.shape[0]})"
    result = np.empty(n, dtype=np.float32)
    cdef float[:] out = result
    _masked_softmax(&out[0], &logits[0], &mask[0], n)
    return result


# ---------------------------------------------------------------------------
# Value un-rotation (replaces np.roll on tiny arrays in evaluator.py)
# ---------------------------------------------------------------------------

def gather_states(
    float[:, :] dst,
    float[:, :, :] src,
    const int[:] worker_indices,
    const int[:] counts,
    int num_requests,
):
    """Gather input states from per-worker shared memory into a contiguous buffer.

    Replaces the Python loop in eval_server._eval_server_serve:
        for widx, n in batch_info:
            pin_s_np[total_n:total_n + n] = w_states_np[widx][:n]

    Args:
        dst: Contiguous destination buffer, shape (max_batch, visible_size).
        src: Per-worker input states, shape (num_workers, batch_size, visible_size).
        worker_indices: Worker index per request, shape (num_requests,).
        counts: Number of states per request, shape (num_requests,).
        num_requests: Number of requests in this batch.

    Returns:
        Total number of states gathered.
    """
    cdef int vis = dst.shape[1]
    cdef int row_bytes = vis * sizeof(float)
    cdef int total = 0
    cdef int i, n, widx
    with nogil:
        for i in range(num_requests):
            widx = worker_indices[i]
            n = counts[i]
            assert_c(total + n <= <int>dst.shape[0])
            memcpy(&dst[total, 0], &src[widx, 0, 0], n * row_bytes)
            total = total + n
    return total


def scatter_results(
    const char[:, :] src_logits,
    const float[:, :] src_values,
    char[:, :, :] dst_logits,
    float[:, :, :] dst_values,
    const int[:] worker_indices,
    const int[:] counts,
    int num_requests,
    int logit_row_bytes,
):
    """Scatter inference results from contiguous buffers to per-worker shared memory.

    Replaces the Python loop in eval_server._eval_server_serve:
        for widx, n in batch_info:
            w_logits[widx][:n].copy_(pin_log[offset:offset + n])
            w_values[widx][:n].copy_(pin_val[offset:offset + n])

    Uses char (byte) views for logits to stay dtype-agnostic at the Cython level.

    Args:
        src_logits: Contiguous logit buffer as bytes, shape (max_batch, logit_row_bytes).
        src_values: Contiguous value buffer, shape (max_batch, num_players).
        dst_logits: Per-worker logit slots as bytes, shape (num_workers, batch_size, logit_row_bytes).
        dst_values: Per-worker value slots, shape (num_workers, batch_size, num_players).
        worker_indices: Worker index per request, shape (num_requests,).
        counts: Number of states per request, shape (num_requests,).
        num_requests: Number of requests in this batch.
        logit_row_bytes: Bytes per logit row (action_dim * 2 for bf16).
    """
    cdef int npl = src_values.shape[1]
    cdef int val_row_bytes = npl * sizeof(float)
    cdef int offset = 0
    cdef int i, n, widx
    with nogil:
        for i in range(num_requests):
            widx = worker_indices[i]
            n = counts[i]
            assert_c(offset + n <= <int>src_logits.shape[0])
            memcpy(&dst_logits[widx, 0, 0], &src_logits[offset, 0], n * logit_row_bytes)
            memcpy(&dst_values[widx, 0, 0], &src_values[offset, 0], n * val_row_bytes)
            offset = offset + n


def unrotate_values(const float[:] values, int active_player_id, int num_players):
    """Convert NN values (active player at index 0) to canonical order.

    Drop-in replacement for evaluator.unrotate_values.
    Equivalent to np.roll(values, active_player_id) without numpy dispatch.

    Args:
        values: Per-player values from NN, shape (num_players,).
        active_player_id: Canonical player ID of the active player.
        num_players: Number of players.

    Returns:
        Values in canonical player order, shape (num_players,).
    """
    result = np.empty(num_players, dtype=np.float32)
    cdef float[:] out = result
    cdef int i
    for i in range(num_players):
        out[(i + active_player_id) % num_players] = values[i]
    return result
