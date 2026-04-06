# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation (compact layout for transformer architecture).

Single contiguous int16 array — no visible/hidden split. All values stored as
raw signed integers (no normalization divisors). The NN never reads this array
directly; get_token_data() extracts per-entity features into float eval buffers.

Layout:
  [metadata | players | fi | company_incomes | market | corps | turn |
   deck | company_tracking]

The players section is the only one that still varies with player count;
all per-player tracking (cash, shares, presidencies, share buys/sells) lives
inside one player_stride block, so `_player_ptr(i)` reaches everything for
player `i` in a single pointer hop.
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
    'share_buys', 'share_sells',
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
    'auction_starter', 'auction_passed',
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
from entities import player as player_module
from entities import fi as fi_module
from entities import corp as corp_module
from entities import company as company_module
from entities import market as market_module
from entities import turn as turn_module
from entities import deck as deck_module

cnp.import_array()


# =============================================================================
# LAYOUT COMPUTATION
# =============================================================================

cdef StateLayout compute_layout(int num_players) noexcept nogil:
    """Compute complete state layout for given player count.

    All values are stored as raw integers/flags — no normalization.
    No visible/hidden distinction.

    Section sizes (player stride, corp stride, turn block) are derived from
    the dedicated offset-computation functions to avoid any duplication.
    Adding a new field in one of those functions automatically resizes the
    enclosing section here.
    """
    cdef StateLayout layout
    cdef int offset = 0

    # Pull strides/sizes from the per-section computations so this layout
    # function never needs to know the field list.
    cdef PlayerFieldOffsets player_fields = compute_player_field_offsets()
    cdef CorpFieldOffsets corp_fields = compute_corp_field_offsets()
    cdef TurnStateOffsets turn_offsets = compute_turn_offsets(num_players)

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

    # --- Players ---
    layout.players_offset = offset
    offset += layout.player_stride * num_players

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

    layout.total_size = offset
    return layout


# =============================================================================
# TURN STATE SUB-OFFSETS
# =============================================================================

cdef TurnStateOffsets compute_turn_offsets(int num_players) noexcept nogil:
    """Compute sub-offsets within turn state section.

    The final `t.size` field is the total length of the turn block — used
    by compute_layout to size the section without duplicating the field
    list.
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
    t.auction_passed = offset
    offset += num_players

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
    use the cdef structs directly for nogil performance.
    """
    cdef StateLayout layout = compute_layout(num_players)
    return LayoutInfo(
        total_size=layout.total_size,
        player_stride=layout.player_stride,
        corp_stride=layout.corp_stride,
        active_player_offset=layout.active_player_offset,
        num_players_offset=layout.num_players_offset,
        phase_offset=layout.phase_offset,
        coo_level_offset=layout.coo_level_offset,
        turn_number_offset=layout.turn_number_offset,
        players_offset=layout.players_offset,
        fi_offset=layout.fi_offset,
        company_incomes_offset=layout.company_incomes_offset,
        market_offset=layout.market_offset,
        corps_offset=layout.corps_offset,
        turn_offset=layout.turn_offset,
        deck_top_offset=layout.deck_top_offset,
        deck_order_offset=layout.deck_order_offset,
        company_locations_offset=layout.company_locations_offset,
        company_owner_ids_offset=layout.company_owner_ids_offset,
        num_players=num_players,
    )


def get_player_fields():
    """Python-accessible player field sub-offsets within each player's data block.

    Returns a PlayerFields namedtuple with relative offsets (add to
    players_offset + p * player_stride to get absolute position).
    """
    cdef PlayerFieldOffsets p = compute_player_field_offsets()
    return PlayerFields(
        cash=p.cash,
        net_worth=p.net_worth,
        liquidity=p.liquidity,
        turn_order=p.turn_order,
        owned_shares=p.owned_shares,
        is_president=p.is_president,
        round_trips=p.round_trips,
        income=p.income,
        share_buys=p.share_buys,
        share_sells=p.share_sells,
    )


def get_corp_fields():
    """Python-accessible corp field sub-offsets within each corp's data block.

    Returns a CorpFields namedtuple with relative offsets (add to
    corps_offset + c * corp_stride to get absolute position).
    """
    cdef CorpFieldOffsets c = compute_corp_field_offsets()
    return CorpFields(
        active=c.active,
        cash=c.cash,
        unissued_shares=c.unissued_shares,
        issued_shares=c.issued_shares,
        bank_shares=c.bank_shares,
        income=c.income,
        stars=c.stars,
        share_price=c.share_price,
        acquisition_proceeds=c.acquisition_proceeds,
        in_receivership=c.in_receivership,
        price_index=c.price_index,
        pending_price_move=c.pending_price_move,
        raw_revenue=c.raw_revenue,
        synergy_income=c.synergy_income,
        coo_cost=c.coo_cost,
        ability_income=c.ability_income,
    )


def get_turn_fields(int num_players):
    """Python-accessible turn state sub-offsets within the turn block.

    Returns a TurnFields namedtuple with relative offsets (add to
    turn_offset to get absolute position).
    """
    cdef TurnStateOffsets t = compute_turn_offsets(num_players)
    return TurnFields(
        end_card_flipped=t.end_card_flipped,
        consecutive_passes=t.consecutive_passes,
        cards_remaining=t.cards_remaining,
        auction_price=t.auction_price,
        auction_company=t.auction_company,
        auction_high_bidder=t.auction_high_bidder,
        auction_starter=t.auction_starter,
        auction_passed=t.auction_passed,
        dividend_remaining=t.dividend_remaining,
        issue_remaining=t.issue_remaining,
        ipo_remaining=t.ipo_remaining,
    )


# =============================================================================
# CACHED LAYOUT TABLES (computed once, reused by all GameState instances)
# =============================================================================

# Indexed by num_players (slots 0-1 unused, 2-6 valid)
cdef StateLayout _cached_layouts[7]
cdef TurnStateOffsets _cached_turns[7]
cdef PlayerFieldOffsets _cached_player_fields  # Fixed stride, same for all player counts
cdef CorpFieldOffsets _cached_corp_fields


cdef void _populate_layout_cache() noexcept nogil:
    cdef int n
    for n in range(2, 7):
        _cached_layouts[n] = compute_layout(n)
        _cached_turns[n] = compute_turn_offsets(n)


_cached_player_fields = compute_player_field_offsets()
_cached_corp_fields = compute_corp_field_offsets()
_populate_layout_cache()


# =============================================================================
# GAME STATE CLASS
# =============================================================================

cdef class GameState:
    """
    Game state container (compact layout).

    Holds the raw memory buffer and layout information.
    All values stored as raw integers/flags — no normalization.
    Logic is delegated to Entity handles and Phase classes.
    """
    def __cinit__(self, unsigned int num_players, bint _alloc=True):
        if num_players < 2 or num_players > GameConstants.MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")

        self._num_players = num_players
        self.step_mode = False

        # Look up precomputed layouts
        self._layout = _cached_layouts[num_players]
        self._turn_offsets = _cached_turns[num_players]
        self._player_fields = _cached_player_fields
        self._corp_fields = _cached_corp_fields

        if not _alloc:
            # Caller will set _array and _data (used by from_buffer)
            return

        # Allocate array (zero-initialized)
        self._array = np.zeros(self._layout.total_size, dtype=np.int16)
        self._data = <int16_t*>cnp.PyArray_DATA(self._array)

        # Initialize constant metadata
        self._data[self._layout.num_players_offset] = <int16_t>num_players

        # Initialize company owner_ids to -1 (no owner when in deck)
        # Company locations are already 0 (LOC_DECK) from zero-initialization
        cdef int i
        for i in range(<int>GameConstants.NUM_COMPANIES):
            self._data[self._layout.company_owner_ids_offset + i] = -1

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
        if arr.dtype != np.int16:
            raise ValueError(f"Expected int16 array, got {arr.dtype}")
        if arr.ndim != 1 or <int>arr.shape[0] != state._layout.total_size:
            raise ValueError(
                f"Expected 1-D array of length {state._layout.total_size}, "
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
        if buf.dtype != np.int16:
            raise ValueError(f"Expected int16 array, got {buf.dtype}")
        if buf.ndim != 1 or <int>buf.shape[0] != state._layout.total_size:
            raise ValueError(
                f"Expected 1-D array of length {state._layout.total_size}, "
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
        return self._data + self._layout.players_offset + (player_id * self._layout.player_stride)

    cdef int16_t* _corp_ptr(self, int corp_id) noexcept nogil:
        """Get pointer to corporation data block."""
        return self._data + self._layout.corps_offset + (corp_id * self._layout.corp_stride)

    cdef int16_t* _turn_ptr(self) noexcept nogil:
        """Get pointer to turn state."""
        return self._data + self._layout.turn_offset

    cdef int _get_active_player(self) noexcept nogil:
        """Get active player ID from metadata."""
        return self._data[self._layout.active_player_offset]

    cdef void _set_active_player(self, int player_id) noexcept nogil:
        """Set active player ID in metadata."""
        self._data[self._layout.active_player_offset] = <int16_t>player_id

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
        Entity handles must be updated to match the compact layout before
        this method produces correct results.
        """
        cdef int i, corp_id, company_id
        cdef int actual_seed
        cdef int starting_cash
        cdef timespec ts
        cdef int16_t* turn
        cdef int16_t* player
        cdef int16_t* corp

        # 1. Initialize all entity handles (they cache offsets from layout)
        for i in range(self._num_players):
            player_module.PLAYERS[i].initialize(self)
        fi_module.FI.initialize(self)
        for c in corp_module.CORPS:
            c.initialize(self)
        for c in company_module.COMPANIES:
            c.initialize(self)
        market_module.MARKET.initialize(self)
        turn_module.TURN.initialize(self)
        deck_module.DECK.initialize(self)

        # 2. Set player starting state (raw integers, no normalization)
        starting_cash = 25 if self._num_players == 6 else 30
        for i in range(self._num_players):
            player = self._player_ptr(i)
            player[self._player_fields.cash] = <int16_t>starting_cash
            player[self._player_fields.net_worth] = <int16_t>starting_cash
            player[self._player_fields.liquidity] = <int16_t>starting_cash
            player[self._player_fields.turn_order] = <int16_t>i

        # 3. Set Foreign Investor state (raw integers)
        self._data[self._layout.fi_offset] = 4       # cash
        self._data[self._layout.fi_offset + 1] = 5   # income (base +5)

        # 4. Initialize corporations (only non-zero: unissued shares)
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            corp = self._corp_ptr(corp_id)
            corp[self._corp_fields.unissued_shares] = <int16_t>CORP_SHARE_COUNT[corp_id]

        # 5. Initialize market - all spaces available
        for i in range(<int>GameConstants.NUM_MARKET_SPACES):
            self._data[self._layout.market_offset + i] = 1

        # 6. Build and shuffle deck (also marks excluded companies as LOC_EXCLUDED)
        if seed < 0:
            clock_gettime(CLOCK_MONOTONIC, &ts)
            actual_seed = <int>(ts.tv_sec ^ ts.tv_nsec)
        else:
            actual_seed = seed
        deck_module.DECK.setup(self, self._num_players, actual_seed)

        # 7. Draw initial companies (move_to_auction clears the revealed flag)
        for i in range(self._num_players):
            company_id = deck_module.DECK.draw(self)
            company_module.COMPANIES[company_id].move_to_auction(self)

        # 8. Set turn state (raw integers)
        self._data[self._layout.phase_offset] = <int16_t>GamePhases.PHASE_INVEST
        self._data[self._layout.coo_level_offset] = 1
        self._data[self._layout.turn_number_offset] = 1

        # 9. Initialize "no selection" auction sentinels to -1
        turn = self._turn_ptr()
        turn[self._turn_offsets.auction_company] = -1
        turn[self._turn_offsets.auction_high_bidder] = -1
        turn[self._turn_offsets.auction_starter] = -1

        # 10. Set active player
        self._set_active_player(0)
