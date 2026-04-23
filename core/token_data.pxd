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
# Equals ``max(TokenWidth.*)``; currently pinned by ``TW_CORP = 85``.
# See the companion .pyx for the per-token feature layout and counts.
cpdef enum TokenDataSize:
    TOKEN_DIM = 85


# Non-padded feature width per token type (single source of truth for the
# per-type projection input sizes on the model side; each token in the
# (num_tokens, TOKEN_DIM) buffer uses only the first ``TW_*`` slots of its
# row, with the rest zero-padded up to TOKEN_DIM). The corresponding
# ``_fill_*_token`` helper in ``token_data.pyx`` writes exactly
# ``TW_*`` slots — keep these in sync with the OFF_* layout constants
# inside those helpers. Used by ``get_token_widths`` to build the
# per-position widths array that matches ``_fill_buffer``'s layout.
cpdef enum TokenWidth:
    TW_MARKET_INFO           = 54
    TW_COMPANY               = 26
    TW_FI                    = 38
    TW_GLOBAL_INFO           = 23
    TW_INVEST                = 17
    TW_AUCTION               = 13
    TW_DIVIDEND              = 34
    TW_ISSUE                 = 9
    TW_PAR                   = 50
    TW_ACQ_SELECT_COMPANY    = 36
    TW_ACQ_OFFER             = 11
    TW_ACQ_PRICE             = 3
    TW_CORP                  = 85
    TW_PLAYER                = 80


# Number of tokens for a given player count (num_players + 55 fixed entities).
cpdef int get_num_tokens(int num_players) noexcept nogil


# Per-position non-padded feature widths matching ``_fill_buffer``'s layout.
# Returns a ``(num_players + 55,)`` uint8 numpy array; each entry is the
# width of the corresponding buffer row (<= TOKEN_DIM). The model can use
# this to slice ``buffer[i, :widths[i]]`` into the per-type projection.
cpdef object get_token_widths(int num_players)


# Fill a (num_tokens, TOKEN_DIM) float32 memoryview with per-token features.
# The buffer is zeroed by the function; phase-specific tokens remain zero
# when the current engine phase does not match. Requires a C-contiguous
# float32 memoryview sized for at least (num_players + 55, TOKEN_DIM).
cpdef void get_token_data(GameState state, float[:, ::1] buffer)


# Batched variant: fill ``buffer[i]`` from ``state_arrays[i]`` for i in [0, n).
# Reuses one scratch GameState across all rows via rebind, so the outer
# function call amortizes per-state Python dispatch + GameState construction
# over a single entry. Cache refresh + ``_fill_buffer`` run together in one
# nogil block per row via ``refresh_player_cache_if_dirty``; only ``rebind``
# itself stays GIL-held (Python-level validation + ``_array`` attr write).
# Requires a C-contiguous (n, num_players + 55, TOKEN_DIM) float32 buffer.
cpdef void get_token_data_batch(
    list state_arrays, int num_players, float[:, :, ::1] buffer,
)
