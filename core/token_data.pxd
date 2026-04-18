"""
Declaration file for token data extraction.

``get_token_data`` is the sole engine→NN interface: it fills a
(num_tokens, TOKEN_DIM) float32 buffer with normalized per-token
features from a compact GameState. Feature layout per token type is
documented in ``token-data.md`` and matches the order expected by
``nn/transformer.py``.
"""

from core.state cimport GameState


# Maximum feature count across all token types (== raw token_dim input
# to ``nn/transformer.py``). All tokens are zero-padded to this width.
# See the companion .pyx for the per-token feature layout and counts.
cpdef enum TokenDataSize:
    TOKEN_DIM = 97


# Number of tokens for a given player count (num_players + 53 fixed entities).
cpdef int get_num_tokens(int num_players) noexcept nogil


# Fill a (num_tokens, TOKEN_DIM) float32 memoryview with per-token features.
# The buffer is zeroed by the function; phase-specific tokens remain zero
# when the current engine phase does not match. Requires a C-contiguous
# float32 memoryview sized for at least (num_players + 53, TOKEN_DIM).
cpdef void get_token_data(GameState state, float[:, ::1] buffer)


# Batched variant: fill ``buffer[i]`` from ``state_arrays[i]`` for i in [0, n).
# Reuses one scratch GameState across all rows via rebind, so the outer
# function call amortizes per-state Python dispatch + GameState construction
# over a single entry. Cache refresh + ``_fill_buffer`` run together in one
# nogil block per row via ``refresh_player_cache_if_dirty``; only ``rebind``
# itself stays GIL-held (Python-level validation + ``_array`` attr write).
# Requires a C-contiguous (n, num_players + 53, TOKEN_DIM) float32 buffer.
cpdef void get_token_data_batch(
    list state_arrays, int num_players, float[:, :, ::1] buffer,
)
