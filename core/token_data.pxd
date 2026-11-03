"""Shared extraction API; each version owns its feature constants and kernels.

See token_data_v2.pxd and token_data_v3.pxd for model-specific declarations.
Every call names its layout. Both layouts currently share token row order.
get_token_data and get_token_data_batch are def functions (keyword-only
layout_version), so they have no declarations here.
"""

from core.state cimport GameState

cpdef int get_num_tokens(int max_players) noexcept nogil
cpdef int get_token_dim(int layout_version) except -1
cpdef object get_token_widths(int max_players, int layout_version)
