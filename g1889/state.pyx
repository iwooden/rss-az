"""State layout and GameState for the 1889 engine; see state.pxd.

Layout: [global | corps | privates | hexes | supply | trains | graph | players].
Entity handles in ``g1889.entities`` own the semantics of every block;
``initialize_game`` reproduces the Ruby engine's new-game setup.
"""

from collections import namedtuple

from libc.stdint cimport int16_t, uint64_t
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC
cimport numpy as cnp
import numpy as np

from g1889.constants cimport OWNER_NOT_IN_PLAY, OWNER_UNSOLD, ROUND_AUCTION, STEP_AUCTION
from g1889.data.tables cimport (
    BANK_CASH, CORP_TOKENS, MAX_PLAYERS, MIN_PLAYERS, NUM_CORPS, NUM_HEXES, NUM_PRIVATES,
    NUM_SUPPLY_TILES, NUM_TRAIN_TYPES, PHASE_ORS, PRIVATES_IN_PLAY, START_CASH, TILE_COUNT,
    TRAIN_COUNT,
)

cnp.import_array()


# =============================================================================
# LAYOUT
# =============================================================================

cdef inline int _take(int* offset, int width) noexcept nogil:
    """Claim ``width`` slots at ``offset`` and return the first."""
    cdef int start = offset[0]
    offset[0] += width
    return start


cdef StateLayout compute_layout() noexcept nogil:
    cdef StateLayout layout
    cdef int o = 0
    layout.global_offset = _take(&o, compute_global_offsets().size)
    layout.corps_offset = _take(&o, compute_corp_field_offsets().size * <int>NUM_CORPS)
    layout.privates_offset = _take(&o, compute_private_field_offsets().size * <int>NUM_PRIVATES)
    layout.hexes_offset = _take(&o, compute_hex_field_offsets().size * <int>NUM_HEXES)
    layout.supply_offset = _take(&o, <int>NUM_SUPPLY_TILES)
    layout.trains_offset = _take(&o, compute_train_offsets().size)
    layout.graph_offset = _take(&o, <int>NUM_CORPS * <int>NUM_HEXES)
    layout.players_offset = o
    return layout


cdef GlobalOffsets compute_global_offsets() noexcept nogil:
    cdef GlobalOffsets g
    cdef int o = 0
    g.num_players = _take(&o, 1)
    g.round = _take(&o, 1)
    g.step = _take(&o, 1)
    g.return_step = _take(&o, 1)
    g.game_phase = _take(&o, 1)
    g.turn = _take(&o, 1)
    g.or_num = _take(&o, 1)
    g.or_count = _take(&o, 1)
    g.active_player = _take(&o, 1)
    g.priority_player = _take(&o, 1)
    g.bank_cash = _take(&o, 1)
    g.bank_broken = _take(&o, 1)
    g.end_reason = _take(&o, 1)
    g.auction_turn = _take(&o, 1)
    g.auction_private = _take(&o, 1)
    g.auction_discount = _take(&o, 1)
    g.auction_pass_payouts = _take(&o, 1)
    g.sr_bought = _take(&o, 1)
    g.sr_sold_mask = _take(&o, 1)
    g.sr_sold_then_bought = _take(&o, 1)
    g.operating_corp = _take(&o, 1)
    g.or_index = _take(&o, 1)
    g.or_order = _take(&o, <int>NUM_CORPS)
    g.route_revenue = _take(&o, 1)
    g.pending_hex = _take(&o, 1)
    g.pending_corp = _take(&o, 1)
    g.pending_private = _take(&o, 1)
    g.pending_train_source = _take(&o, 1)
    g.pending_seller = _take(&o, 1)
    g.ebuy_sold_mask = _take(&o, 1)
    g.ebuy_sold_any = _take(&o, 1)
    g.market_stamp = _take(&o, 1)
    g.conn_dirty = _take(&o, 1)
    g.route_dirty = _take(&o, 1)
    g.size = o
    return g


cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil:
    cdef CorpFieldOffsets c
    cdef int o = 0
    c.par_index = _take(&o, 1)
    c.floated = _take(&o, 1)
    c.cash = _take(&o, 1)
    c.ipo_shares = _take(&o, 1)
    c.pool_shares = _take(&o, 1)
    c.president = _take(&o, 1)
    c.market_row = _take(&o, 1)
    c.market_col = _take(&o, 1)
    c.market_stamp = _take(&o, 1)
    c.tokens_left = _take(&o, 1)
    c.trains = _take(&o, <int>NUM_TRAIN_TYPES)
    c.last_revenue = _take(&o, 1)
    c.route_cache = _take(&o, 1)
    c.route_available = _take(&o, 1)
    c.size = o
    return c


cdef PrivateFieldOffsets compute_private_field_offsets() noexcept nogil:
    cdef PrivateFieldOffsets p
    cdef int o = 0
    p.owner = _take(&o, 1)
    p.flags = _take(&o, 1)
    p.size = o
    return p


cdef HexFieldOffsets compute_hex_field_offsets() noexcept nogil:
    cdef HexFieldOffsets h
    cdef int o = 0
    h.tile = _take(&o, 1)
    h.rotation = _take(&o, 1)
    h.tokens = _take(&o, 1)
    h.size = o
    return h


cdef TrainOffsets compute_train_offsets() noexcept nogil:
    cdef TrainOffsets t
    cdef int o = 0
    t.depot = _take(&o, <int>NUM_TRAIN_TYPES)
    t.pool = _take(&o, <int>NUM_TRAIN_TYPES)
    t.size = o
    return t


cdef PlayerFieldOffsets compute_player_field_offsets() noexcept nogil:
    cdef PlayerFieldOffsets p
    cdef int o = 0
    p.cash = _take(&o, 1)
    p.shares = _take(&o, <int>NUM_CORPS)
    p.sold_mask = _take(&o, 1)
    p.bids = _take(&o, <int>NUM_PRIVATES)
    p.passed = _take(&o, 1)
    p.bankrupt = _take(&o, 1)
    p.size = o
    return p


LAYOUT = compute_layout()
GLOBALS = compute_global_offsets()
CORP_FIELDS = compute_corp_field_offsets()
PRIVATE_FIELDS = compute_private_field_offsets()
HEX_FIELDS = compute_hex_field_offsets()
TRAIN_FIELDS = compute_train_offsets()
PLAYER_FIELDS = compute_player_field_offsets()


# =============================================================================
# PYTHON LAYOUT ACCESSORS
# =============================================================================

cdef dict _LAYOUT_DICT = LAYOUT
LayoutInfo = namedtuple('LayoutInfo', ('total_size', 'max_players', 'player_size', *_LAYOUT_DICT))


def _named(name, dict offsets):
    return namedtuple(name, offsets.keys())(**offsets)


# Offsets within each block; each tuple's ``size`` is the block (or stride) width.
_FIELDS = {
    'global': _named('GlobalFields', GLOBALS),
    'corp': _named('CorpFields', CORP_FIELDS),
    'private': _named('PrivateFields', PRIVATE_FIELDS),
    'hex': _named('HexFields', HEX_FIELDS),
    'train': _named('TrainFields', TRAIN_FIELDS),
    'player': _named('PlayerFields', PLAYER_FIELDS),
}


def get_layout(int max_players):
    """Block offsets and the row width at player capacity ``max_players``."""
    return LayoutInfo(total_size=_expected_size(max_players), max_players=max_players,
                      player_size=PLAYER_FIELDS.size, **_LAYOUT_DICT)


def get_global_fields():
    return _FIELDS['global']


def get_corp_fields():
    return _FIELDS['corp']


def get_private_fields():
    return _FIELDS['private']


def get_hex_fields():
    return _FIELDS['hex']


def get_train_fields():
    return _FIELDS['train']


def get_player_fields():
    return _FIELDS['player']


cpdef int get_storage_player_capacity(int state_size_int16) except -1:
    """Return the player capacity implied by a row width."""
    cdef int player_region = state_size_int16 - LAYOUT.players_offset
    assert player_region >= 0 and player_region % PLAYER_FIELDS.size == 0, \
        f"row width {state_size_int16} is not {LAYOUT.players_offset} + {PLAYER_FIELDS.size} * capacity"
    cdef int capacity = player_region // PLAYER_FIELDS.size
    assert <int>MIN_PLAYERS <= capacity <= <int>MAX_PLAYERS, \
        f"row width {state_size_int16} implies capacity {capacity}, expected {MIN_PLAYERS}-{MAX_PLAYERS}"
    return capacity


# =============================================================================
# STORAGE
# =============================================================================

cdef inline int _expected_size(int max_players) noexcept nogil:
    return LAYOUT.players_offset + PLAYER_FIELDS.size * max_players


cdef int _validate_storage_players(int num_players, int max_players) except -1:
    assert <int>MIN_PLAYERS <= num_players <= <int>MAX_PLAYERS, \
        f"num_players must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {num_players}"
    if max_players == 0:
        max_players = num_players
    assert num_players <= max_players <= <int>MAX_PLAYERS, \
        f"max_players must be {num_players}-{MAX_PLAYERS}, got {max_players}"
    return max_players


cdef void _seed_zeroed_storage(GameState state, int num_players) noexcept:
    """Write the non-zero values an empty row needs: the player count, the bank's
    cash, printed tiles, privates out of play, and the -1 sentinels."""
    cdef int16_t* d = state._data
    cdef int16_t* g = d + LAYOUT.global_offset
    cdef int16_t* corp
    cdef int i
    g[GLOBALS.num_players] = <int16_t>num_players
    g[GLOBALS.bank_cash] = <int16_t>BANK_CASH
    g[GLOBALS.return_step] = -1
    g[GLOBALS.auction_private] = -1
    g[GLOBALS.operating_corp] = -1
    g[GLOBALS.pending_hex] = -1
    g[GLOBALS.pending_corp] = -1
    g[GLOBALS.pending_private] = -1
    g[GLOBALS.pending_train_source] = -1
    g[GLOBALS.pending_seller] = -1
    for i in range(<int>NUM_CORPS):
        g[GLOBALS.or_order + i] = -1
        corp = d + LAYOUT.corps_offset + i * CORP_FIELDS.size
        corp[CORP_FIELDS.par_index] = -1
        corp[CORP_FIELDS.president] = -1
        corp[CORP_FIELDS.market_row] = -1
        corp[CORP_FIELDS.market_col] = -1
    for i in range(<int>NUM_PRIVATES):
        d[LAYOUT.privates_offset + i * PRIVATE_FIELDS.size + PRIVATE_FIELDS.owner] = <int16_t>OWNER_NOT_IN_PLAY
    for i in range(<int>NUM_HEXES):
        d[LAYOUT.hexes_offset + i * HEX_FIELDS.size + HEX_FIELDS.tile] = <int16_t>(<int>NUM_SUPPLY_TILES + i)


cdef void _reset_storage(GameState state, int num_players, int max_players) except *:
    """Zero the row at capacity ``max_players`` and seed it, reusing the current
    buffer when its width already matches so a wrapped buffer stays zero-copy."""
    cdef int expected = _expected_size(max_players)
    cdef cnp.ndarray arr = None
    if state._array is not None:
        arr = np.asarray(state._array)
        if not (arr.dtype == np.int16 and arr.ndim == 1 and <int>arr.shape[0] == expected
                and arr.flags['C_CONTIGUOUS']):
            arr = None
    if arr is None:
        arr = np.zeros(expected, dtype=np.int16)
    else:
        arr.fill(0)
    state._array = arr
    state._data = <int16_t*>cnp.PyArray_DATA(arr)
    state.max_players = max_players
    _seed_zeroed_storage(state, num_players)


cdef _check_row(cnp.ndarray arr, int num_players, int max_players):
    cdef int expected = _expected_size(max_players)
    assert arr.dtype == np.int16, f"expected an int16 row, got {arr.dtype}"
    assert arr.ndim == 1 and <int>arr.shape[0] == expected, \
        f"expected a 1-D row of width {expected}, got shape {(<object>arr).shape}"
    cdef int stored = <int>arr[LAYOUT.global_offset + GLOBALS.num_players]
    assert stored == num_players, f"row num_players {stored} != claimed {num_players}"


cdef _bind_buffer(GameState state, object buffer, int num_players, int max_players):
    """Point ``state`` at ``buffer`` (zero-copy); shared by from_buffer and rebind."""
    cdef cnp.ndarray buf = np.asarray(buffer)
    max_players = _validate_storage_players(num_players, max_players)
    _check_row(buf, num_players, max_players)
    assert buf.flags['C_CONTIGUOUS'] and buf.flags['WRITEABLE'], "row must be C-contiguous and writeable"
    state._array = buf
    state._data = <int16_t*>cnp.PyArray_DATA(buf)
    state.max_players = max_players


cdef inline int _seeded_priority(int seed, int num_players) noexcept nogil:
    """The seed's first player: splitmix64's finalizer, reduced mod num_players."""
    cdef uint64_t x = <uint64_t>seed + 0x9E3779B97F4A7C15ULL
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL
    return <int>((x ^ (x >> 31)) % <uint64_t>num_players)


# =============================================================================
# GAME STATE
# =============================================================================

cdef class GameState:
    """A 1889 row plus the policy flags. Layout offsets are module constants;
    the entity handles hold all game logic."""

    def __cinit__(self, int num_players, bint _alloc=True, int max_players=0,
                  bint private_sale_same_president=False, bint train_sale_same_president=False,
                  int auction_pass_payout_cap=0):
        max_players = _validate_storage_players(num_players, max_players)
        self.max_players = max_players
        self.private_sale_same_president = private_sale_same_president
        self.train_sale_same_president = train_sale_same_president
        self.auction_pass_payout_cap = auction_pass_payout_cap
        if _alloc:
            _reset_storage(self, num_players, max_players)

    @staticmethod
    def from_array(array, int num_players, int max_players=0):
        """Copy ``array`` into a new GameState; the flags start off."""
        cdef cnp.ndarray arr = np.asarray(array)
        max_players = _validate_storage_players(num_players, max_players)
        _check_row(arr, num_players, max_players)
        state = GameState(num_players, max_players=max_players)
        np.copyto(state._array, arr)
        return state

    @staticmethod
    def from_buffer(buffer, int num_players, int max_players=0):
        """Wrap ``buffer`` as the backing row without copying; the flags start off."""
        state = GameState(num_players, _alloc=False, max_players=max_players)
        _bind_buffer(state, buffer, num_players, max_players)
        return state

    def rebind(self, buffer, int num_players, int max_players=0):
        """Repoint this wrapper at another row, keeping its flags (MCTS pool rows)."""
        _bind_buffer(self, buffer, num_players, max_players)

    @property
    def auction_pass_payout_cap(self):
        """All-pass payouts after which a player who can buy the topmost private
        may not pass, nor bid themselves out of buying it (policy only); 0
        disables the cap."""
        return self._auction_pass_payout_cap

    @auction_pass_payout_cap.setter
    def auction_pass_payout_cap(self, int value):
        assert value >= 0, f"auction_pass_payout_cap must be >= 0, got {value}"
        self._auction_pass_payout_cap = value

    cpdef void initialize_game(self, int num_players, int seed=-1, int max_players=0,
                               int priority_player=-1):
        """Set up a new game as the Ruby engine does (``Game::Base#initialize``).

        Canonical player order is seat order. ``priority_player`` -1 lets the
        seed pick who acts first (replay passes 0, the first seat); ``seed`` -1
        uses the clock. ``max_players`` 0 keeps the current capacity when it
        fits ``num_players``.
        """
        cdef timespec ts
        cdef int i
        # Entity modules cimport this one, so they are imported at call time.
        from g1889.entities.bank import BANK, BANK_ACCOUNT
        from g1889.entities.corp import CORPS, SHARES_PER_CORP
        from g1889.entities.hexmap import MAP
        from g1889.entities.private import PRIVATES
        from g1889.entities.trains import TRAINS
        from g1889.entities.turn import TURN

        if max_players == 0 and self.max_players >= num_players:
            max_players = self.max_players
        _reset_storage(self, num_players, _validate_storage_players(num_players, max_players))

        if priority_player < 0:
            if seed < 0:
                clock_gettime(CLOCK_MONOTONIC, &ts)
                seed = <int>(ts.tv_sec ^ ts.tv_nsec)
            priority_player = _seeded_priority(seed, num_players)
        assert priority_player < num_players, \
            f"priority_player {priority_player} out of range for {num_players} players"

        # Bank and starting cash (init_starting_cash).
        for i in range(num_players):
            BANK.transfer(self, BANK_ACCOUNT, i, START_CASH[num_players])
        # Privates in play for the player count (min_players; SIR leaves 2-player games).
        for i in range(<int>NUM_PRIVATES):
            if PRIVATES_IN_PLAY[num_players] >> i & 1:
                PRIVATES[i].set_owner(self, OWNER_UNSOLD)
        # Corps start unparred with every share in the IPO and the charter's
        # tokens. Home-token reservations are implicit: HEX_HOME_CORP marks the
        # home city, which keeps a slot free until the home token is placed.
        for i in range(<int>NUM_CORPS):
            CORPS[i].set_ipo_shares(self, SHARES_PER_CORP)
            CORPS[i].set_tokens_left(self, CORP_TOKENS[i])
        # Tile supply and train depot; the map keeps its printed tiles.
        for i in range(<int>NUM_SUPPLY_TILES):
            MAP.set_supply(self, i, TILE_COUNT[i])
        for i in range(<int>NUM_TRAIN_TYPES):
            TRAINS.set_depot(self, i, TRAIN_COUNT[i])
        # Caches start valid: no tokens and no trains. Phase 2 sets one OR per set;
        # or_num is 1 outside ORs, as Ruby's round_num.
        TURN.set_round(self, ROUND_AUCTION)
        TURN.set_step(self, STEP_AUCTION)
        TURN.set_turn(self, 1)
        TURN.set_or_num(self, 1)
        TURN.set_or_count(self, PHASE_ORS[0])
        TURN.set_priority_player(self, priority_player)
        TURN.set_active_player(self, priority_player)
        TURN.set_auction_turn(self, priority_player)
