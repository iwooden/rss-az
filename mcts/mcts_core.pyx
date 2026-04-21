"""Cython implementations of MCTS hot functions.

Replaces pure-Python/numpy versions in search.py and evaluator.py,
eliminating numpy/torch dispatch overhead for small arrays where the
overhead dominates actual computation.
"""

from libc.math cimport sqrtf, isinf, INFINITY
from libc.stdint cimport int8_t, int16_t, int64_t, uint8_t, uint16_t, uint64_t
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
# Protocol uses a per-server bitmap of ``W = ceil(partition_size / 64)``
# uint64 words. Bit ``b`` of word ``w`` means local worker ``w*64 + b`` has
# a pending request. Each word is padded to its own 64-byte cache line to
# prevent false sharing between (server, word) pairs at large partition
# sizes: the shared tensor has shape ``(num_servers * W, 8)`` uint64 and
# word ``w`` of server ``s`` lives at row ``s*W + w``, column 0. Columns
# 1..7 are dead padding.
#
# Workers publish via atomic fetch-or (release) on their own word; servers
# claim all pending work via atomic exchange (acquire) per word. Empty
# check is O(W) acquire loads (short-circuits on first non-empty word);
# drain is O(W + k ready) with a ctz loop per word.

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
    uint64_t[:, :] submitted_masks,
    int[:] counts,
    int worker_idx,
    int server_id,
    int local_idx,
    int state_count,
    int num_words,
):
    """Worker-side: write count, atomically set bit in server's submitted mask.

    The release fence on fetch_or ensures the server sees state data and
    count writes that preceded this call.

    ``submitted_masks`` is the cache-line-padded ``(num_servers * W, 8)``
    tensor; word ``w`` of server ``s`` lives at row ``s*W + w``, column 0.

    Returns True if *this worker's word* transitioned from 0 -> non-zero
    (caller should set the server's doorbell event to wake it). At W > 1
    this may over-signal when a sibling word was already non-zero — bounded
    by W×, cheaper than maintaining a separate cross-word pending counter,
    and correctness-safe because ``mp.Event.set`` is idempotent.
    """
    counts[worker_idx] = state_count
    cdef int word_idx = local_idx >> 6
    cdef int bit_offset = local_idx & 63
    cdef uint64_t bit = <uint64_t>1 << <uint64_t>bit_offset
    cdef int row = server_id * num_words + word_idx
    cdef uint64_t old_word = ATOMIC_FETCH_OR_U64(&submitted_masks[row, 0], bit)
    return old_word == 0


def server_drain_bitmap(
    uint64_t[:, :] submitted_masks,
    int[:] counts,
    int[:] out_worker_indices,
    int[:] out_counts,
    int server_id,
    int partition_start,
    int num_words,
):
    """Server-side: atomically claim all pending requests from the bitmap.

    Exchanges each of the server's W words with 0 (acquire), then iterates
    set bits via ctz to build worker_indices and counts arrays. O(W + k)
    where k is the number of ready workers.

    Returns the number of requests found.
    """
    cdef int n = 0
    cdef int w, bit_offset, local_idx, worker_idx
    cdef int base = server_id * num_words
    cdef uint64_t mask
    with nogil:
        for w in range(num_words):
            mask = ATOMIC_EXCHANGE_U64(&submitted_masks[base + w, 0], 0)
            while mask != 0:
                bit_offset = CTZ64(mask)
                local_idx = (w << 6) + bit_offset
                worker_idx = partition_start + local_idx
                out_worker_indices[n] = worker_idx
                out_counts[n] = counts[worker_idx]
                n = n + 1
                mask = mask & (mask - 1)  # clear lowest set bit
    return n


def server_peek_bitmap(
    uint64_t[:, :] submitted_masks,
    int server_id,
    int num_words,
):
    """Server-side: acquire-load the submitted words without modifying them.

    Used for the lost-wakeup recheck: after clearing the doorbell event,
    peek to see if new work arrived before sleeping. Short-circuits on the
    first non-empty word.
    """
    cdef int w
    cdef int base = server_id * num_words
    for w in range(num_words):
        if ATOMIC_LOAD_U64(&submitted_masks[base + w, 0]) != 0:
            return True
    return False


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


# descend_path outcome codes. Exported so search.py can dispatch.
DESCEND_NEED_NEW_CHILD = 0
DESCEND_VIRTUAL_BACKUP = 1
DESCEND_EXISTING_LEAF = 2


def descend_path(root, float c_puct, list path_out):
    """PUCT-descend from ``root`` until the descent must stop.

    Fuses the per-level selection loop from ``run_search``. Each descent
    step calls the C PUCT impl directly, appends ``(node, action_idx,
    array_idx)`` to ``path_out``, and uses ``dict.get`` (one lookup)
    rather than ``in``-then-``__getitem__`` (two lookups).

    Args:
        root: The search root MCTSNode.
        c_puct: PUCT exploration constant.
        path_out: Mutable list; one tuple appended per descent level.
            Caller creates and owns it (cleared per iteration).

    Returns:
        ``(outcome, node, action_idx, array_idx)`` where outcome is one of:

        - DESCEND_NEED_NEW_CHILD: ``node`` is the parent needing a new
          child at ``(action_idx, array_idx)``. Its entry is the last
          element of ``path_out``.
        - DESCEND_VIRTUAL_BACKUP: ``node`` is the reuse-root whose child
          at ``(action_idx, array_idx)`` needs a virtual backup. Only
          fires at depth 0 (``is_root == True``). Its entry is the last
          element of ``path_out``.
        - DESCEND_EXISTING_LEAF: ``node`` is the leaf (terminal or
          already queued-for-eval). ``action_idx`` and ``array_idx`` are
          -1. The leaf's parent is the last element of ``path_out``.
    """
    cdef int action_idx, array_idx
    cdef int parent_vc, active_player_id, child_vc
    cdef bint is_root = True
    cdef const int[:] legal_actions_view
    cdef const float[:] priors_view
    cdef const int[:] vc_view
    cdef const float[:, :] vs_view
    cdef object child

    node = root
    while node.expanded() and not node.is_terminal:
        legal_actions_view = node.legal_actions
        priors_view = node.priors
        vc_view = node.visit_counts
        vs_view = node.value_sums
        active_player_id = node.active_player_id
        parent_vc = node.visit_count

        action_idx, array_idx = _select_child_impl(
            legal_actions_view, priors_view, vc_view, vs_view,
            active_player_id, parent_vc, c_puct,
        )
        path_out.append((node, action_idx, array_idx))

        child = node.children.get(action_idx)
        if child is None:
            return (DESCEND_NEED_NEW_CHILD, node, action_idx, array_idx)

        if is_root:
            child_vc = child.visit_count
            # Virtual backup: reuse-root hasn't caught up to child's real visits
            if child_vc > 0 and vc_view[array_idx] < child_vc:
                return (DESCEND_VIRTUAL_BACKUP, node, action_idx, array_idx)
            is_root = False

        node = child

    return (DESCEND_EXISTING_LEAF, node, -1, -1)


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


cdef bint _all_col0_neg_inf(const float[:, :] value_sums) noexcept nogil:
    """Return 1 iff value_sums[i, 0] == -inf for every row i.

    Used to detect the 'all sibling edges locked' condition in MCTS leaf
    batching without paying numpy dispatch for a <~K-element reduction.
    """
    cdef int n = value_sums.shape[0]
    cdef int i
    cdef float v
    for i in range(n):
        v = value_sums[i, 0]
        if not (isinf(v) and v < 0.0):
            return 0
    return 1


def all_col0_neg_inf(const float[:, :] value_sums):
    """Python-callable wrapper around _all_col0_neg_inf."""
    return _all_col0_neg_inf(value_sums) != 0


cdef void _save_and_lock_row(
    float[:, :] vs, float[:, :] saved_vs, unsigned char[:] saved_mask,
    int aidx, int npl,
) noexcept nogil:
    """Copy vs[aidx] into saved_vs[aidx], overwrite vs[aidx] with -inf, mark locked."""
    cdef int k
    memcpy(&saved_vs[aidx, 0], &vs[aidx, 0], npl * <int>sizeof(float))
    for k in range(npl):
        vs[aidx, k] = -INFINITY
    saved_mask[aidx] = 1


cdef void _restore_row(
    float[:, :] vs, float[:, :] saved_vs, unsigned char[:] saved_mask,
    int aidx, int npl,
) noexcept nogil:
    """Copy saved_vs[aidx] back into vs[aidx], clear mask bit."""
    memcpy(&vs[aidx, 0], &saved_vs[aidx, 0], npl * <int>sizeof(float))
    saved_mask[aidx] = 0


def propagate_lock(list path):
    """Walk the selection path from leaf upward, locking ancestors whose
    outgoing edges are all now locked (via leaf-lock or prior propagation).

    Drop-in replacement for search.py::_propagate_lock. The saved Q row and
    the lock flag for each propagation-locked edge live in two parallel
    arrays on the ancestor MCTSNode (``saved_value_sums``, ``saved_mask``),
    lazily allocated on first use. The inner save/restore is a nogil memcpy
    plus a small loop — no Python dict churn in the hot path.

    Args:
        path: List of (parent_node, action, array_idx) tuples describing
            the selection path whose leaf was just locked. Entries are
            processed from ``path[-1]`` up to ``path[0]``.
    """
    cdef int path_len = len(path)
    cdef int j, aidx, n_rows, npl
    cdef float[:, :] vs_j
    cdef float[:, :] vs_anc
    cdef float[:, :] saved_vs
    cdef unsigned char[:] saved_mask

    for j in range(path_len - 1, -1, -1):
        node_j = path[j][0]
        vs_j = node_j.value_sums
        if not _all_col0_neg_inf(vs_j):
            return
        if j == 0:
            return  # root fully locked — can't propagate further
        ancestor = path[j - 1][0]
        aidx = <int>path[j - 1][2]
        vs_anc = ancestor.value_sums
        npl = vs_anc.shape[1]

        if ancestor.saved_mask is None:
            n_rows = vs_anc.shape[0]
            ancestor.saved_value_sums = np.empty((n_rows, npl), dtype=np.float32)
            ancestor.saved_mask = np.zeros(n_rows, dtype=np.uint8)
        saved_mask = ancestor.saved_mask
        if saved_mask[aidx] == 0:
            saved_vs = ancestor.saved_value_sums
            _save_and_lock_row(vs_anc, saved_vs, saved_mask, aidx, npl)
            ancestor.saved_count = ancestor.saved_count + 1


def propagate_unlock(list path):
    """Walk the selection path from leaf's parent upward, restoring any
    propagation-locked ancestor edges.

    Drop-in replacement for search.py::_propagate_unlock. Order-independent:
    the first unlock in a fully-locked subtree restores the full ancestor
    chain; later unlocks on sibling paths find those edges already restored
    and stop immediately.

    Args:
        path: List of (parent_node, action, array_idx) tuples — same path
            layout produced at selection time. Entries are processed from
            ``path[-2]`` up to ``path[0]``.
    """
    cdef int path_len = len(path)
    cdef int j, aidx, npl
    cdef float[:, :] vs
    cdef float[:, :] saved_vs
    cdef unsigned char[:] saved_mask

    for j in range(path_len - 2, -1, -1):
        node_j = path[j][0]
        if node_j.saved_count == 0:
            return
        aidx = <int>path[j][2]
        saved_mask = node_j.saved_mask
        if saved_mask[aidx] == 0:
            return
        vs = node_j.value_sums
        saved_vs = node_j.saved_value_sums
        npl = vs.shape[1]
        _restore_row(vs, saved_vs, saved_mask, aidx, npl)
        node_j.saved_count = node_j.saved_count - 1


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
# Eval-server shared-memory gather/scatter helpers
# ---------------------------------------------------------------------------
# Both gather helpers share the same skeleton: a nogil loop that walks the
# (worker_indices, counts) request list and memcpys each request's contiguous
# row block from the per-worker shared-mem slot into a flat batch buffer.
# They differ only by dtype and row width. scatter_results is the inverse.

def gather_states(
    float[:, :] dst,
    float[:, :, :] src,
    const int[:] worker_indices,
    const int[:] counts,
    int num_requests,
):
    """Gather per-leaf NN input rows into a contiguous batch buffer.

    Row width is `num_tokens * token_dim` floats — callers reshape the logical
    ``(W, B, num_tokens, token_dim)`` buffer to 3-D ``(W, B, row_floats)``
    before calling; the memcpy is the same.

    Args:
        dst: Contiguous destination buffer, shape (max_batch, row_floats).
        src: Per-worker input rows, shape (num_workers, batch_size, row_floats).
        worker_indices: Worker index per request, shape (num_requests,).
        counts: Number of rows per request, shape (num_requests,).
        num_requests: Number of requests in this batch.

    Returns:
        Total number of rows gathered.
    """
    cdef int row_floats = dst.shape[1]
    cdef int row_bytes = row_floats * sizeof(float)
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


def gather_masks(
    uint8_t[:, :] dst,
    const uint8_t[:, :, :] src,
    const int[:] worker_indices,
    const int[:] counts,
    int num_requests,
):
    """Gather per-leaf ``(UNIFIED_LOGIT_DIM,)`` legal-action masks into a
    contiguous batch buffer.

    Row width is ``UNIFIED_LOGIT_DIM`` bytes (uint8: 1 == legal slot). This
    replaces the old per-phase (phase_ids, action_ids, n_legals) triple — a
    dense 170-byte mask carries the same legality information without the
    per-phase LUT gather on the GPU.

    Args:
        dst: Contiguous destination buffer, shape (max_batch, UNIFIED_LOGIT_DIM) uint8.
        src: Per-worker mask slots, shape (num_workers, batch_size, UNIFIED_LOGIT_DIM) uint8.
        worker_indices: Worker index per request, shape (num_requests,).
        counts: Number of rows per request, shape (num_requests,).
        num_requests: Number of requests in this batch.

    Returns:
        Total number of mask rows gathered.
    """
    cdef int u = dst.shape[1]
    cdef int row_bytes = u * <int>sizeof(uint8_t)
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
    const char[:, :] src_priors,
    const float[:, :] src_values,
    char[:, :, :] dst_priors,
    float[:, :, :] dst_values,
    const int[:] worker_indices,
    const int[:] counts,
    int num_requests,
    int priors_row_bytes,
):
    """Scatter sparse-prior + canonical-value results back to per-worker shared memory.

    The eval server has already masked + softmaxed the dense model logits
    on the GPU before copy-back, so the scattered row is a dense
    UNIFIED_LOGIT_DIM-wide prior vector (illegal slots carry ~0 mass) that
    the worker reads directly — no further transform needed.

    Priors use a byte (char) view so the caller can plug in any float dtype
    (f32 in the canonical path) without branching here.

    Args:
        src_priors: Contiguous prior buffer as bytes, shape (max_batch, priors_row_bytes).
        src_values: Contiguous value buffer, shape (max_batch, num_players).
        dst_priors: Per-worker prior slots as bytes, shape (num_workers, batch_size, priors_row_bytes).
        dst_values: Per-worker value slots, shape (num_workers, batch_size, num_players).
        worker_indices: Worker index per request, shape (num_requests,).
        counts: Number of rows per request, shape (num_requests,).
        num_requests: Number of requests in this batch.
        priors_row_bytes: Bytes per priors row (UNIFIED_LOGIT_DIM * 4 for f32).
    """
    cdef int npl = src_values.shape[1]
    cdef int val_row_bytes = npl * <int>sizeof(float)
    cdef int offset = 0
    cdef int i, n, widx
    with nogil:
        for i in range(num_requests):
            widx = worker_indices[i]
            n = counts[i]
            assert_c(offset + n <= <int>src_priors.shape[0])
            memcpy(&dst_priors[widx, 0, 0], &src_priors[offset, 0], n * priors_row_bytes)
            memcpy(&dst_values[widx, 0, 0], &src_values[offset, 0], n * val_row_bytes)
            offset = offset + n


# ---------------------------------------------------------------------------
# Sparse node expansion
# ---------------------------------------------------------------------------

def expand_node_sparse(
    node,
    const uint16_t[:] action_ids,
    int n,
    const float[:] priors_legal,
    const float[:] default_value,
    int num_players,
):
    """Set up per-action arrays on an MCTSNode from a sparse legal-action list.

    Replaces the dense-mask path: callers pass the already-enumerated legal
    phase-local action ids plus their softmax-normalized priors, aligned 1:1.
    Skips the 14977-wide mask scan of the dense form.

    Args:
        node: MCTSNode to expand (modified in place).
        action_ids: Legal phase-local action ids, shape (≥ n,) uint16 — only
            first n entries are read.
        n: Number of legal actions.
        priors_legal: Softmax-normalized priors over the legal list,
            shape (≥ n,) float32 — only first n entries are read.
        default_value: FPU default value (NN value for this node), shape (num_players,) float32.
        num_players: Number of players in the game.
    """
    actions = np.empty(n, dtype=np.int32)
    priors = np.empty(n, dtype=np.float32)
    visit_counts = np.zeros(n, dtype=np.int32)
    value_sums = np.empty((n, num_players), dtype=np.float32)
    dv = np.empty(num_players, dtype=np.float32)

    cdef int[:] a_view = actions
    cdef float[:] p_view = priors
    cdef float[:, :] vs_view = value_sums
    cdef float[:] dv_view = dv

    cdef int i, p

    # Copy sparse legal list + aligned priors
    for i in range(n):
        a_view[i] = <int>action_ids[i]
        p_view[i] = priors_legal[i]

    # Copy default_value and broadcast it into each value_sums row
    for p in range(num_players):
        dv_view[p] = default_value[p]
    for i in range(n):
        for p in range(num_players):
            vs_view[i, p] = default_value[p]

    # Set node attributes
    node.legal_actions = actions
    node.priors = priors
    node.default_value = dv
    node.visit_counts = visit_counts
    node.value_sums = value_sums
