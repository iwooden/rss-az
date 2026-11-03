"""Dispatch engine observations to the model's version-specific extractor.

Feature layouts, normalization, and fill kernels belong to token_data_v2 or
token_data_v3. Dispatch happens once per public call, including whole batches.
Every call names its layout: there is no default. Models and consumers import
a layout's constants from its own module (``core.token_data_v2`` or
``core.token_data_v3``).
"""

from core.state cimport GameState
from core cimport token_data_v2 as v2, token_data_v3 as v3


cpdef int get_num_tokens(int max_players) noexcept nogil:
    """Shared token count; both current layouts use the same entity rows."""
    return v2.get_num_tokens(max_players)


cpdef int get_token_dim(int layout_version) except -1:
    if layout_version == 2:
        return <int>v2.TokenDataSize.TOKEN_DIM
    if layout_version == 3:
        return <int>v3.TokenDataSize.TOKEN_DIM
    raise ValueError(f"Unsupported token layout version: {layout_version}")


cpdef object get_token_widths(int max_players, int layout_version):
    if layout_version == 2:
        return v2.get_token_widths(max_players)
    if layout_version == 3:
        return v3.get_token_widths(max_players)
    raise ValueError(f"Unsupported token layout version: {layout_version}")


# def rather than cpdef: cpdef cannot take the keyword-only layout_version,
# and nothing calls these from C.
def get_token_data(
    GameState state, float[:, ::1] buffer, int max_players=0, *, int layout_version,
):
    """Fill an observation using the selected layout's input contract."""
    if layout_version == 2:
        v2.get_token_data(state, buffer, max_players)
    elif layout_version == 3:
        v3.get_token_data(state, buffer, max_players)
    else:
        raise ValueError(f"Unsupported token layout version: {layout_version}")


def get_token_data_batch(
    list state_arrays, object arg2, object arg3=None,
    int max_players=0, *, int layout_version,
):
    """Dispatch once for the entire batch; reconstruction stays in the extractor."""
    if layout_version == 2:
        v2.get_token_data_batch(state_arrays, arg2, arg3, max_players)
    elif layout_version == 3:
        v3.get_token_data_batch(state_arrays, arg2, arg3, max_players)
    else:
        raise ValueError(f"Unsupported token layout version: {layout_version}")
