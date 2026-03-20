# cython: language_level=3
"""Declarations for MCTS hot-path functions."""

cdef (int, int) _select_child_impl(
    const int[:] legal_actions, const float[:] priors,
    const int[:] visit_counts, const float[:, :] value_sums,
    int active_player_id, int parent_visit_count, float c_puct,
) noexcept nogil

cdef void _backup_node(
    float[:] value_sum, float[:, :] value_sums,
    const int[:] visit_counts, int array_idx,
    const float[:] values, int num_players,
) noexcept nogil

cdef void _virtual_backup_node(
    float[:] value_sum, float[:, :] value_sums,
    int[:] visit_counts, int array_idx,
    const float[:] child_q, int num_players,
) noexcept nogil
