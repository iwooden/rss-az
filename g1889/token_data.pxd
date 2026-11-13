"""Shikoku 1889 token extraction (1889/extraction-design.md §4-5, 1889/v1-design.md §5).

``get_token_data`` fills a ``(get_num_tokens(max_players), get_token_dim(layout))``
float32 buffer from a state and ``get_token_data_batch`` one per raw row; both
name their layout: 1, which ``nn/transformer-1889.py`` reads and which is
pinned (``tests/g1889/test_layout1_parity.py``), or 2. The rows are the same
in both, so token indices in relation records do not depend on the layout.
Every row type is zero-padded to the layout's width, and
``get_token_widths`` gives each row's own. The slots of each row type are in
the .pyx and in ``token-data-1889.md``. get_token_data and
get_token_data_batch are def functions (keyword-only layout_version), so
they have no declarations here.

Layout 3 (1889/v2-design.md, ``nn/transformer-1889-v2.py``) is not a token
buffer: ``fill_inputs`` and ``fill_inputs_batch`` write one packed float
vector a state, the rows, pair blocks, and choice table at the
``InputOffset3`` offsets, and ``get_input_parts`` / ``get_input_size``
describe it. ``get_tile_table``, ``get_choice_slots``, and
``get_named_rows`` are the model's static tables.

``extraction_prologue`` refreshes the caches every fill reads, writing them
into the row (§4); the relation fills run it too. Layout 2's token fills
also refresh the hypothetical route maxes (``routes_refresh_hypotheticals``).
``tile_candidates`` is the candidate list behind CANDIDATE rows and
relations 21-24 (§7.1).
"""

from g1889.actions cimport LayCandidate
from g1889.data.tables cimport NUM_HEXES
from g1889.state cimport LAYOUT, GameState

cpdef enum TokenDataSize:
    TOKEN_DIM = 108                 # layout 1's widest row, CANDIDATE
    TOKEN_DIM_2 = 109               # layout 2's

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

# Each row type's width in layout 1; the slots past it are zero.
cpdef enum TokenWidth:
    TW_GLOBAL = 44
    TW_PRICE = 24
    TW_DIVIDEND = 8
    TW_CORP = 51
    TW_PRIVATE = 18
    TW_TRAIN = 13
    TW_HEX = 105
    TW_CANDIDATE = 108
    TW_PLAYER = 38

# The same in layout 2.
cpdef enum TokenWidth2:
    TW2_GLOBAL = 90
    TW2_PRICE = 30
    TW2_DIVIDEND = 13
    TW2_CORP = 77
    TW2_PRIVATE = 18
    TW2_TRAIN = 15
    TW2_HEX = 105
    TW2_CANDIDATE = 109
    TW2_PLAYER = 40

# The tile block (§5.7): a tile as laid, at the same slots of HEX and CANDIDATE rows.
cpdef enum TileBlock:
    TILE_BLOCK = 2
    TW_TILE_BLOCK = 76

# Layout 3 (token-data-1889.md, "Layout 3"): one packed float vector a state,
# its parts at the offsets below. Each row type's width:
cpdef enum TokenWidth3:
    TW3_GLOBAL = 135
    TW3_CORP = 76
    TW3_PRIVATE = 15
    TW3_CENTER = 21
    TW3_PLAYER = 20

# Each row type's first trunk row; players come last.
cpdef enum TrunkRow3:
    TRUNK3_GLOBAL = 0
    TRUNK3_CORP = 1                 # NUM_CORPS rows
    TRUNK3_PRIVATE = 8              # NUM_PRIVATES
    TRUNK3_CENTER = 15              # NUM_RCS
    TRUNK3_PLAYER = 40              # max_players

# Each pair block's channels, its last axis.
cpdef enum PairChannels3:
    CH3_PLAYER_CORP = 3             # shares, president, sold this SR
    CH3_PLAYER_PRIVATE = 2          # owns, bid
    CH3_CORP_PRIVATE = 1            # owns
    CH3_CORP_CENTER = 4             # token, reaches, on its routes, build distance
    CH3_CENTER_CENTER = 1           # build distance

# The choice table: a row a phase-local action, each row its integer columns,
# the shared block, then the decision's phase block.
cpdef enum ChoiceTable3:
    CHOICE3_ROWS = 46               # MAX_ACTION_SIZE
    CHOICE3_WIDTH = 47
    CHOICE3_HEX = 0                 # hex code
    CHOICE3_TILE = 1                # tile code
    CHOICE3_ROTATION = 2
    CHOICE3_SHARED = 3              # S0-S6
    CHOICE3_PHASE = 10              # the phase block's slot 0
    CHOICE3_PHASE_WIDTH = 37        # the widest phase block, TILE's

# The parts' offsets. The players' three parts come last, each max_players
# long: PLAYER rows at IN3_PLAYERS, PLAYER x CORP after them, then PLAYER x
# PRIVATE, so every other offset is the same at every capacity.
cpdef enum InputOffset3:
    IN3_GLOBAL = 0
    IN3_CORP = 135
    IN3_PRIVATE = 667
    IN3_CENTER = 772
    IN3_CORP_PRIVATE = 1297
    IN3_CORP_CENTER = 1346
    IN3_CENTER_CENTER = 2046
    IN3_CHOICES = 2671
    IN3_PLAYERS = 4833
    IN3_PER_PLAYER = 55             # TW3_PLAYER + NUM_CORPS * 3 + NUM_PRIVATES * 2

# ``get_named_rows``' entries besides a trunk row.
cpdef enum NamedRow3:
    NAMED3_NONE = -1
    NAMED3_TOPMOST = -2             # the PRIVATE row PRIVATE 2 selects (AUCTION's buy)

cpdef int get_num_tokens(int max_players) noexcept nogil
cpdef int get_token_dim(int layout_version) except -1
cpdef object get_token_widths(int max_players, int layout_version)
cdef int input_size3(int max_players) noexcept nogil

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
