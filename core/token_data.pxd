"""Shared extraction API; each version owns its feature constants and kernels.

See token_data_v2.pxd and token_data_v3.pxd for model-specific declarations.
The default layout is v2. Both layouts currently share token row order.
"""

from core.state cimport GameState

cpdef int get_num_tokens(int max_players) noexcept nogil
cpdef int get_token_dim(int layout_version=*) except -1
cpdef object get_token_widths(int max_players, int layout_version=*)

cpdef void get_token_data(
    GameState state, float[:, ::1] buffer, int max_players=*, int layout_version=*,
)

# Supports both (state_arrays, buffer) and (state_arrays, num_players, buffer).
cpdef void get_token_data_batch(
    list state_arrays, object arg2, object arg3=*, int max_players=*, int layout_version=*,
)
