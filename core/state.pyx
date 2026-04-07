# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation (compact layout for transformer architecture).

Single contiguous int16 array — no visible/hidden split. All values stored as
raw signed integers (no normalization divisors). The NN never reads this array
directly; get_token_data() extracts per-entity features into float eval buffers.

Layout:
  [metadata | fi | company_incomes | market | corps | turn |
   deck | company_tracking | players]

The players section is the only one that still varies with player count, so
it lives at the very end of the buffer. Every other offset is constant
across all player counts and is exposed through module-level constants
(`LAYOUT`, `TURN_OFFSETS`, `PLAYER_FIELDS`, `CORP_FIELDS`) that entity
handles and token extraction can `cimport` directly. The only
num_players-dependent quantity is the total buffer size, computed inline
as `LAYOUT.players_offset + LAYOUT.player_stride * num_players` at the
small handful of sites that need it (allocation and length validation).
All per-player tracking (cash, shares, presidencies, share buys/sells,
the auction-passed flag) lives inside one player_stride block, so
`_player_ptr(i)` reaches everything for player `i` in a single pointer
hop.
"""

cimport cython
from libc.stdint cimport int16_t
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC
cimport numpy as cnp
import numpy as np

from collections import namedtuple

from core.data cimport (
    GameConstants,
    GamePhases,
    CORP_SHARE_COUNT,
)

# Python-accessible namedtuples for layout introspection

PlayerFields = namedtuple('PlayerFields', [
    'cash', 'net_worth', 'liquidity', 'turn_order',
    'owned_shares', 'is_president', 'round_trips', 'income',
    'share_buys', 'share_sells', 'auction_passed',
])

CorpFields = namedtuple('CorpFields', [
    'active', 'cash', 'unissued_shares', 'issued_shares', 'bank_shares',
    'income', 'stars', 'share_price', 'acquisition_proceeds',
    'in_receivership', 'price_index', 'pending_price_move',
    'raw_revenue', 'synergy_income', 'coo_cost', 'ability_income',
])

TurnFields = namedtuple('TurnFields', [
    'end_card_flipped', 'consecutive_passes', 'cards_remaining',
    'auction_price', 'auction_company', 'auction_high_bidder',
    'auction_starter',
    'dividend_remaining', 'issue_remaining', 'ipo_remaining',
])

LayoutInfo = namedtuple('LayoutInfo', [
    # Sizes
    'total_size', 'player_stride', 'corp_stride',
    # Metadata
    'active_player_offset', 'num_players_offset',
    'phase_offset', 'coo_level_offset', 'turn_number_offset',
    # Sections
    'players_offset', 'fi_offset',
    'company_incomes_offset', 'market_offset', 'corps_offset', 'turn_offset',
    'deck_top_offset', 'deck_order_offset',
    'company_locations_offset', 'company_owner_ids_offset',
    # Convenience
    'num_players',
])

# Import entity modules for their global instances
from entities import company as company_module
from entities import deck as deck_module

cnp.import_array()


# =============================================================================
# LAYOUT COMPUTATION
# =============================================================================

cdef StateLayout compute_layout() noexcept nogil:
    """Compute the fixed state-buffer layout.

    Every offset in the returned struct is constant across all player
    counts. The only num_players-dependent value is the total buffer
    size, computed inline as
        LAYOUT.players_offset + LAYOUT.player_stride * num_players
    at the few sites that need it (allocation and length validation).

    Section sizes (player stride, corp stride, turn block) are derived
    from the dedicated offset-computation functions so this layout
    function never needs to know the field list.

    The players section lives at the **end** of the buffer because it
    is the only section whose size depends on `num_players`. Every
    other offset stays constant, which lets entity handles, token
    extraction, and tests reference offsets directly from the module-
    level `LAYOUT` constant without re-computing per game.
    """
    cdef StateLayout layout
    cdef int offset = 0

    cdef PlayerFieldOffsets player_fields = compute_player_field_offsets()
    cdef CorpFieldOffsets corp_fields = compute_corp_field_offsets()
    cdef TurnStateOffsets turn_offsets = compute_turn_offsets()

    layout.player_stride = player_fields.stride
    layout.corp_stride = corp_fields.stride

    # --- Metadata (5 slots) ---
    layout.active_player_offset = offset
    offset += 1
    layout.num_players_offset = offset
    offset += 1
    layout.phase_offset = offset
    offset += 1
    layout.coo_level_offset = offset
    offset += 1
    layout.turn_number_offset = offset
    offset += 1

    # --- Foreign Investor: cash, income (ownership lives in
    # company_locations / company_owner_ids) ---
    layout.fi_offset = offset
    offset += 2

    # --- Company adjusted incomes (36 raw integers) ---
    layout.company_incomes_offset = offset
    offset += GameConstants.NUM_COMPANIES

    # --- Market availability (27 flags) ---
    layout.market_offset = offset
    offset += GameConstants.NUM_MARKET_SPACES

    # --- Corporations ---
    layout.corps_offset = offset
    offset += layout.corp_stride * GameConstants.NUM_CORPS

    # --- Turn state ---
    layout.turn_offset = offset
    offset += turn_offsets.size

    # --- Deck ---
    layout.deck_top_offset = offset
    offset += 1
    layout.deck_order_offset = offset
    offset += GameConstants.MAX_DECK_SIZE  # 36 company IDs

    # --- Company tracking (location enums + owner IDs) ---
    layout.company_locations_offset = offset
    offset += GameConstants.NUM_COMPANIES  # 36 CompanyLocation enum values
    layout.company_owner_ids_offset = offset
    offset += GameConstants.NUM_COMPANIES  # 36 owner IDs (-1 = none)

    # --- Players (LAST: only num_players-dependent section). The actual
    # length of this region depends on num_players and is *not* recorded
    # here — callers compute total_size on demand from
    # players_offset + player_stride * num_players.
    layout.players_offset = offset

    return layout


# =============================================================================
# TURN STATE SUB-OFFSETS
# =============================================================================

cdef TurnStateOffsets compute_turn_offsets() noexcept nogil:
    """Compute sub-offsets within turn state section.

    The turn block is fixed-size — it no longer scales with player count
    now that the per-player auction-passed flag lives inside each player's
    block. The final `t.size` field is the total length of the turn block
    and is used by compute_layout to size the section without duplicating
    the field list.
    """
    cdef TurnStateOffsets t
    cdef int offset = 0

    t.end_card_flipped = offset
    offset += 1
    t.consecutive_passes = offset
    offset += 1
    t.cards_remaining = offset
    offset += 1

    # Auction
    t.auction_price = offset
    offset += 1
    t.auction_company = offset
    offset += 1
    t.auction_high_bidder = offset
    offset += 1
    t.auction_starter = offset
    offset += 1

    # Phase remaining tracking
    t.dividend_remaining = offset
    offset += GameConstants.NUM_CORPS
    t.issue_remaining = offset
    offset += GameConstants.NUM_CORPS
    t.ipo_remaining = offset
    offset += GameConstants.NUM_COMPANIES

    t.size = offset
    return t


# =============================================================================
# PLAYER FIELD OFFSETS (within player stride)
# =============================================================================

cdef PlayerFieldOffsets compute_player_field_offsets() noexcept nogil:
    """Compute field offsets within a player's data block.

    Player stride is fixed across player counts — turn_order is a single
    integer, not a one-hot. All per-player tracking (including share
    buy/sell counts for the current turn) lives inside the player block,
    so `_player_ptr(i)` reaches everything for player `i` in one pointer
    hop. The final `p.stride` field is the total block size, used by
    compute_layout to size the players section.
    """
    cdef PlayerFieldOffsets p
    cdef int offset = 0

    p.cash = offset
    offset += 1
    p.net_worth = offset
    offset += 1
    p.liquidity = offset
    offset += 1
    p.turn_order = offset
    offset += 1  # single integer (not num_players one-hot)
    p.owned_shares = offset
    offset += GameConstants.NUM_CORPS
    p.is_president = offset
    offset += GameConstants.NUM_CORPS
    p.round_trips = offset
    offset += 1
    p.income = offset
    offset += 1
    p.share_buys = offset
    offset += GameConstants.NUM_CORPS
    p.share_sells = offset
    offset += GameConstants.NUM_CORPS
    p.auction_passed = offset
    offset += 1

    p.stride = offset
    return p


# =============================================================================
# CORP FIELD OFFSETS (within corp stride)
# =============================================================================

cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil:
    """Compute field offsets within a corp's data block.

    The final `c.stride` field is the total block size, used by
    compute_layout to size the corps section.
    """
    cdef CorpFieldOffsets c
    cdef int offset = 0

    c.active = offset
    offset += 1
    c.cash = offset
    offset += 1
    c.unissued_shares = offset
    offset += 1
    c.issued_shares = offset
    offset += 1
    c.bank_shares = offset
    offset += 1
    c.income = offset
    offset += 1
    c.stars = offset
    offset += 1
    c.share_price = offset
    offset += 1
    c.acquisition_proceeds = offset
    offset += 1
    c.in_receivership = offset
    offset += 1
    c.price_index = offset
    offset += 1
    c.pending_price_move = offset
    offset += 1
    c.raw_revenue = offset
    offset += 1
    c.synergy_income = offset
    offset += 1
    c.coo_cost = offset
    offset += 1
    c.ability_income = offset
    offset += 1

    c.stride = offset
    return c


# =============================================================================
# PYTHON-ACCESSIBLE LAYOUT ACCESSORS
# =============================================================================

def get_layout(int num_players):
    """Python-accessible layout offsets. Single source of truth.

    Returns a LayoutInfo namedtuple with all sizes, offsets, and strides
    needed by Python code (tests, evaluator). Cython code should
    `cimport LAYOUT` directly for nogil performance.

    Only ``total_size`` depends on ``num_players``; every other field is
    a constant pulled straight from the module-level ``LAYOUT`` struct.
    """
    cdef int total_size = LAYOUT.players_offset + LAYOUT.player_stride * num_players
    return LayoutInfo(
        total_size=total_size,
        player_stride=LAYOUT.player_stride,
        corp_stride=LAYOUT.corp_stride,
        active_player_offset=LAYOUT.active_player_offset,
        num_players_offset=LAYOUT.num_players_offset,
        phase_offset=LAYOUT.phase_offset,
        coo_level_offset=LAYOUT.coo_level_offset,
        turn_number_offset=LAYOUT.turn_number_offset,
        players_offset=LAYOUT.players_offset,
        fi_offset=LAYOUT.fi_offset,
        company_incomes_offset=LAYOUT.company_incomes_offset,
        market_offset=LAYOUT.market_offset,
        corps_offset=LAYOUT.corps_offset,
        turn_offset=LAYOUT.turn_offset,
        deck_top_offset=LAYOUT.deck_top_offset,
        deck_order_offset=LAYOUT.deck_order_offset,
        company_locations_offset=LAYOUT.company_locations_offset,
        company_owner_ids_offset=LAYOUT.company_owner_ids_offset,
        num_players=num_players,
    )


def get_player_fields():
    """Python-accessible player field sub-offsets within each player's data block.

    Returns a PlayerFields namedtuple with relative offsets (add to
    players_offset + p * player_stride to get absolute position).
    """
    return PlayerFields(
        cash=PLAYER_FIELDS.cash,
        net_worth=PLAYER_FIELDS.net_worth,
        liquidity=PLAYER_FIELDS.liquidity,
        turn_order=PLAYER_FIELDS.turn_order,
        owned_shares=PLAYER_FIELDS.owned_shares,
        is_president=PLAYER_FIELDS.is_president,
        round_trips=PLAYER_FIELDS.round_trips,
        income=PLAYER_FIELDS.income,
        share_buys=PLAYER_FIELDS.share_buys,
        share_sells=PLAYER_FIELDS.share_sells,
        auction_passed=PLAYER_FIELDS.auction_passed,
    )


def get_corp_fields():
    """Python-accessible corp field sub-offsets within each corp's data block.

    Returns a CorpFields namedtuple with relative offsets (add to
    corps_offset + c * corp_stride to get absolute position).
    """
    return CorpFields(
        active=CORP_FIELDS.active,
        cash=CORP_FIELDS.cash,
        unissued_shares=CORP_FIELDS.unissued_shares,
        issued_shares=CORP_FIELDS.issued_shares,
        bank_shares=CORP_FIELDS.bank_shares,
        income=CORP_FIELDS.income,
        stars=CORP_FIELDS.stars,
        share_price=CORP_FIELDS.share_price,
        acquisition_proceeds=CORP_FIELDS.acquisition_proceeds,
        in_receivership=CORP_FIELDS.in_receivership,
        price_index=CORP_FIELDS.price_index,
        pending_price_move=CORP_FIELDS.pending_price_move,
        raw_revenue=CORP_FIELDS.raw_revenue,
        synergy_income=CORP_FIELDS.synergy_income,
        coo_cost=CORP_FIELDS.coo_cost,
        ability_income=CORP_FIELDS.ability_income,
    )


def get_turn_fields():
    """Python-accessible turn state sub-offsets within the turn block.

    Returns a TurnFields namedtuple with relative offsets (add to
    turn_offset to get absolute position). The turn block is fixed-size —
    no num_players argument needed.
    """
    return TurnFields(
        end_card_flipped=TURN_OFFSETS.end_card_flipped,
        consecutive_passes=TURN_OFFSETS.consecutive_passes,
        cards_remaining=TURN_OFFSETS.cards_remaining,
        auction_price=TURN_OFFSETS.auction_price,
        auction_company=TURN_OFFSETS.auction_company,
        auction_high_bidder=TURN_OFFSETS.auction_high_bidder,
        auction_starter=TURN_OFFSETS.auction_starter,
        dividend_remaining=TURN_OFFSETS.dividend_remaining,
        issue_remaining=TURN_OFFSETS.issue_remaining,
        ipo_remaining=TURN_OFFSETS.ipo_remaining,
    )


# =============================================================================
# MODULE-LEVEL LAYOUT CONSTANTS
# =============================================================================
#
# Computed once at import time. Every offset is constant across all player
# counts (only the total buffer size depends on num_players, computed
# inline where needed). Other modules cimport these directly from
# `core.state` rather than reaching through a GameState instance.

LAYOUT = compute_layout()
TURN_OFFSETS = compute_turn_offsets()
PLAYER_FIELDS = compute_player_field_offsets()
CORP_FIELDS = compute_corp_field_offsets()


# =============================================================================
# GAME STATE CLASS
# =============================================================================

cdef class GameState:
    """
    Game state container (compact layout).

    Holds the raw memory buffer plus the player count. Layout offsets are
    *not* stored per-instance — they're constants on the module
    (`LAYOUT`, `TURN_OFFSETS`, `PLAYER_FIELDS`, `CORP_FIELDS`). The only
    num_players-dependent value is the total buffer size, computed inline
    where needed. Logic is delegated to Entity handles and Phase classes.
    """
    def __cinit__(self, unsigned int num_players, bint _alloc=True):
        if num_players < 2 or num_players > GameConstants.MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")

        self._num_players = num_players
        self.step_mode = False

        if not _alloc:
            # Caller will set _array and _data (used by from_buffer)
            return

        cdef int total_size = LAYOUT.players_offset + LAYOUT.player_stride * <int>num_players

        # Allocate array (zero-initialized)
        self._array = np.zeros(total_size, dtype=np.int16)
        self._data = <int16_t*>cnp.PyArray_DATA(self._array)

        # Initialize constant metadata
        self._data[LAYOUT.num_players_offset] = <int16_t>num_players

        # Initialize company owner_ids to -1 (no owner when in deck)
        # Company locations are already 0 (LOC_DECK) from zero-initialization
        cdef int i
        for i in range(<int>GameConstants.NUM_COMPANIES):
            self._data[LAYOUT.company_owner_ids_offset + i] = -1

    @staticmethod
    def from_array(array, int num_players):
        """Reconstruct GameState from raw numpy array.

        Args:
            array: numpy int16 array (will be copied)
            num_players: number of players (required to compute layout)

        Returns:
            New GameState with copied array data
        """
        state = GameState(num_players)
        cdef cnp.ndarray arr = np.asarray(array)
        cdef int expected_size = LAYOUT.players_offset + LAYOUT.player_stride * num_players
        if arr.dtype != np.int16:
            raise ValueError(f"Expected int16 array, got {arr.dtype}")
        if arr.ndim != 1 or <int>arr.shape[0] != expected_size:
            raise ValueError(
                f"Expected 1-D array of length {expected_size}, "
                f"got ndim={arr.ndim} len={<int>arr.shape[0]}"
            )
        np.copyto(state._array, arr)
        return state

    @staticmethod
    def from_buffer(buffer, int num_players):
        """Wrap an existing numpy array as backing store (zero-copy).

        The GameState will read/write directly into the provided buffer.
        No array allocation occurs. Caller must ensure the buffer outlives
        the GameState.

        Note: unlike the default constructor, this path does NOT seed
        ``company_owner_ids`` to -1 — the buffer is assumed to already
        contain valid game state (e.g. cloned from another GameState via
        the MCTS state pool). Passing in a freshly zero-initialized
        buffer would silently mark every company as owned by player 0.

        Args:
            buffer: numpy int16 array of correct size (not copied)
            num_players: number of players (required for layout lookup)

        Returns:
            GameState backed by the provided buffer
        """
        state = GameState(num_players, _alloc=False)
        cdef cnp.ndarray buf = np.asarray(buffer)
        cdef int expected_size = LAYOUT.players_offset + LAYOUT.player_stride * num_players
        if buf.dtype != np.int16:
            raise ValueError(f"Expected int16 array, got {buf.dtype}")
        if buf.ndim != 1 or <int>buf.shape[0] != expected_size:
            raise ValueError(
                f"Expected 1-D array of length {expected_size}, "
                f"got ndim={buf.ndim} len={<int>buf.shape[0]}"
            )
        if not buf.flags['C_CONTIGUOUS']:
            raise ValueError("Buffer must be C-contiguous")
        state._array = buf
        state._data = <int16_t*>cnp.PyArray_DATA(buf)
        return state

    def rebind(self, buffer):
        """Rebind this GameState to a different backing buffer (zero-copy).

        Avoids allocating a new GameState wrapper when only the underlying
        data changes. Used in MCTS search hot paths.
        """
        self._array = buffer
        self._data = <int16_t*>cnp.PyArray_DATA(buffer)

    # =========================================================================
    # INTERNAL POINTER ACCESS
    # =========================================================================

    cdef int16_t* _player_ptr(self, int player_id) noexcept nogil:
        """Get pointer to player data block."""
        return self._data + LAYOUT.players_offset + (player_id * LAYOUT.player_stride)

    cdef int16_t* _corp_ptr(self, int corp_id) noexcept nogil:
        """Get pointer to corporation data block."""
        return self._data + LAYOUT.corps_offset + (corp_id * LAYOUT.corp_stride)

    cdef int16_t* _turn_ptr(self) noexcept nogil:
        """Get pointer to turn state."""
        return self._data + LAYOUT.turn_offset

    cdef int _get_active_player(self) noexcept nogil:
        """Get active player ID from metadata."""
        return self._data[LAYOUT.active_player_offset]

    cdef void _set_active_player(self, int player_id) noexcept nogil:
        """Set active player ID in metadata."""
        self._data[LAYOUT.active_player_offset] = <int16_t>player_id

    cpdef int get_active_player(self):
        """Get active player ID (Python-accessible)."""
        return self._get_active_player()

    cpdef void set_active_player(self, int player_id):
        """Set active player ID (Python-accessible)."""
        self._set_active_player(player_id)

    cpdef int get_num_players(self):
        """Get number of players (Python-accessible)."""
        return self._num_players

    # =========================================================================
    # GAME INITIALIZATION
    # =========================================================================

    cpdef void initialize_game(self, int seed=-1):
        """
        Initialize a new game with all starting state.

        Args:
            seed: Random seed for deck shuffling. If -1, uses current time.

        Sets up players, FI, corporations, market, deck, and turn state.
        All entity handles are stateless and read offsets directly from
        the module-level layout constants, so no per-handle initialization
        step is required.
        """
        cdef int i, corp_id, company_id
        cdef int actual_seed
        cdef int starting_cash
        cdef timespec ts
        cdef int16_t* turn
        cdef int16_t* player
        cdef int16_t* corp

        # 1. Set player starting state (raw integers, no normalization)
        starting_cash = 25 if self._num_players == 6 else 30
        for i in range(self._num_players):
            player = self._player_ptr(i)
            player[PLAYER_FIELDS.cash] = <int16_t>starting_cash
            player[PLAYER_FIELDS.net_worth] = <int16_t>starting_cash
            player[PLAYER_FIELDS.liquidity] = <int16_t>starting_cash
            player[PLAYER_FIELDS.turn_order] = <int16_t>i

        # 2. Set Foreign Investor state (raw integers)
        self._data[LAYOUT.fi_offset] = 4       # cash
        self._data[LAYOUT.fi_offset + 1] = 5   # income (base +5)

        # 3. Initialize corporations (only non-zero: unissued shares)
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            corp = self._corp_ptr(corp_id)
            corp[CORP_FIELDS.unissued_shares] = <int16_t>CORP_SHARE_COUNT[corp_id]

        # 4. Initialize market - all spaces available
        for i in range(<int>GameConstants.NUM_MARKET_SPACES):
            self._data[LAYOUT.market_offset + i] = 1

        # 5. Build and shuffle deck (also marks excluded companies as LOC_EXCLUDED)
        if seed < 0:
            clock_gettime(CLOCK_MONOTONIC, &ts)
            actual_seed = <int>(ts.tv_sec ^ ts.tv_nsec)
        else:
            actual_seed = seed
        deck_module.DECK.setup(self, self._num_players, actual_seed)

        # 6. Draw initial companies (move_to_auction clears the revealed flag)
        for i in range(self._num_players):
            company_id = deck_module.DECK.draw(self)
            company_module.COMPANIES[company_id].move_to_auction(self)

        # 7. Set turn state (raw integers)
        self._data[LAYOUT.phase_offset] = <int16_t>GamePhases.PHASE_INVEST
        self._data[LAYOUT.coo_level_offset] = 1
        self._data[LAYOUT.turn_number_offset] = 1

        # 8. Initialize "no selection" auction sentinels to -1
        turn = self._turn_ptr()
        turn[TURN_OFFSETS.auction_company] = -1
        turn[TURN_OFFSETS.auction_high_bidder] = -1
        turn[TURN_OFFSETS.auction_starter] = -1

        # 9. Set active player
        self._set_active_player(0)
