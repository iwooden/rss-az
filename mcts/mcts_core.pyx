# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython implementations of MCTS selection and backup hot functions.

These replace the pure-Python/numpy versions in search.py, eliminating
numpy dispatch overhead (~15-20us/call) for arrays of 20-100 elements
where the overhead dominates actual computation.
"""

from libc.math cimport sqrtf

import numpy as np


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
