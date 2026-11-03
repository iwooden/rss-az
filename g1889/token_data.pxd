"""Shikoku 1889 token extraction (1889/extraction-design.md §4-5).

``get_token_data`` fills a ``(get_num_tokens(max_players), TOKEN_DIM)``
float32 buffer from a state and ``get_token_data_batch`` one per raw row; both
name their layout, and 1 is the only one. Every row type is zero-padded to
TOKEN_DIM; ``get_token_widths`` gives each row's width. The slots of each row
type are in the .pyx. get_token_data and get_token_data_batch are def
functions (keyword-only layout_version), so they have no declarations here.

``extraction_prologue`` refreshes the caches every fill reads, writing them
into the row (§4); the relation fills run it too. ``tile_candidates`` is the
candidate list behind CANDIDATE rows and relations 21-24 (§7.1).
"""

from g1889.actions cimport LayCandidate
from g1889.data.tables cimport NUM_HEXES
from g1889.state cimport LAYOUT, GameState

cpdef enum TokenDataSize:
    TOKEN_DIM = 108                 # the widest row, CANDIDATE

# The first row of each type (§5.1). Players come last, so every other index
# is the same at every capacity.
cpdef enum TokenRow:
    ROW_GLOBAL = 0
    ROW_PRICE = 1
    ROW_DIVIDEND = 2
    ROW_CORP = 3                    # NUM_CORPS rows
    ROW_PRIVATE = 10                # NUM_PRIVATES
    ROW_TRAIN = 17                  # NUM_TRAIN_TYPES
    ROW_HEX = 23                    # NUM_HEXES
    ROW_CANDIDATE = 75              # MAX_LAY_CANDIDATES
    ROW_PLAYER = 90                 # max_players

# Each row type's width; the slots past it are zero.
cpdef enum TokenWidth:
    TW_GLOBAL = 43
    TW_PRICE = 24
    TW_DIVIDEND = 8
    TW_CORP = 51
    TW_PRIVATE = 18
    TW_TRAIN = 13
    TW_HEX = 105
    TW_CANDIDATE = 108
    TW_PLAYER = 38

# The tile block (§5.7): a tile as laid, at the same slots of HEX and CANDIDATE rows.
cpdef enum TileBlock:
    TILE_BLOCK = 2
    TW_TILE_BLOCK = 76

cpdef int get_num_tokens(int max_players) noexcept nogil
cpdef int get_token_dim(int layout_version) except -1
cpdef object get_token_widths(int max_players, int layout_version)

# Shared with the relation fills: the prologue, the player-count checks
# (3 <= num_players <= max_players <= 5; max_players 0 is the row's count),
# the batches' row view, and the candidate list.
cdef void extraction_prologue(GameState state) noexcept nogil
cdef int resolve_max_players(int num_players, int max_players, str func) except -1
cdef GameState bind_row(GameState view, object row, int max_players, str func)
cdef int tile_candidates(GameState state, int phase, int* hex, LayCandidate* buf) noexcept nogil


cdef inline int graph_entry(GameState state, int corp, int hex) noexcept nogil:
    """The corp's GRAPH entry for the hex, read raw: fresh after the prologue."""
    return <int>state._data[LAYOUT.graph_offset + corp * <int>NUM_HEXES + hex]
