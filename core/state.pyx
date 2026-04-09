# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation (compact layout for transformer architecture).

Single contiguous int16 array — no visible/hidden split. All values stored as
raw signed integers (no normalization divisors). The NN never reads this array
directly; get_token_data() extracts per-entity features into float eval buffers.

Layout:
  [fi | companies | market | corps | turn | deck | players]

The five metadata slots (active_player, num_players, phase, coo_level,
turn_number) live at the front of the turn block. The companies section
holds three parallel 36-slot sub-arrays (adjusted incomes, location
enums, owner ids) reachable via `LAYOUT.companies_offset +
COMPANY_OFFSETS.<field>`. The deck section holds the top-of-deck index
and the 36-slot order array reachable via `LAYOUT.deck_offset +
DECK_OFFSETS.<field>`. `StateLayout` only describes section offsets,
never scalar slots.

The players section is the only one that still varies with player count, so
it lives at the very end of the buffer. Every other offset is constant
across all player counts and is exposed through module-level constants
(`LAYOUT`, `TURN_OFFSETS`, `PLAYER_FIELDS`, `CORP_FIELDS`,
`COMPANY_OFFSETS`, `DECK_OFFSETS`) that entity handles and token
extraction can `cimport` directly. The only
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
    'income', 'total_stars', 'cash_stars', 'company_stars',
    'share_price', 'acquisition_proceeds',
    'in_receivership', 'price_index', 'pending_price_move',
    'raw_revenue', 'synergy_income', 'coo_cost', 'ability_income',
])

CompanyFields = namedtuple('CompanyFields', [
    'incomes', 'locations', 'owner_ids',
])

DeckFields = namedtuple('DeckFields', [
    'top', 'order',
])

TurnFields = namedtuple('TurnFields', [
    'active_player', 'num_players',
    'phase', 'coo_level', 'turn_number',
    'end_card_flipped', 'consecutive_passes', 'cards_remaining',
    'auction_price', 'auction_company', 'auction_high_bidder',
    'auction_starter',
    'dividend_remaining', 'issue_remaining', 'ipo_remaining',
])

LayoutInfo = namedtuple('LayoutInfo', [
    # Sizes
    'total_size', 'player_stride', 'corp_stride',
    # Sections
    'players_offset', 'fi_offset',
    'companies_offset', 'market_offset', 'corps_offset', 'turn_offset',
    'deck_offset',
    # Convenience
    'num_players',
])

# Import entity modules for their global instances
from entities import company as company_module
from entities import deck as deck_module
from entities import turn as turn_module

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
    cdef CompanyOffsets company_offsets = compute_company_offsets()
    cdef DeckOffsets deck_offsets = compute_deck_offsets()

    layout.player_stride = player_fields.stride
    layout.corp_stride = corp_fields.stride

    # --- Foreign Investor: cash, income ---
    layout.fi_offset = offset
    offset += 2

    # --- Companies (adjusted incomes + locations + owner_ids) ---
    layout.companies_offset = offset
    offset += company_offsets.size

    # --- Market availability (27 flags) ---
    layout.market_offset = offset
    offset += GameConstants.NUM_MARKET_SPACES

    # --- Corporations ---
    layout.corps_offset = offset
    offset += layout.corp_stride * GameConstants.NUM_CORPS

    # --- Turn state ---
    layout.turn_offset = offset
    offset += turn_offsets.size

    # --- Deck (top index + order array) ---
    layout.deck_offset = offset
    offset += deck_offsets.size

    # --- Players (LAST: only num_players-dependent section). The actual
    # length of this region depends on num_players and is *not* recorded
    # here — callers compute total_size on demand from
    # players_offset + player_stride * num_players. ---
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

    # Metadata (formerly the top-of-buffer "metadata" section)
    t.active_player = offset
    offset += 1
    t.num_players = offset
    offset += 1
    t.phase = offset
    offset += 1
    t.coo_level = offset
    offset += 1
    t.turn_number = offset
    offset += 1

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

    # Internal derived-cache dirty mask (one bit per player).
    t.player_cache_dirty = offset
    offset += 1
    # Internal derived-cache dirty mask (one bit per corporation).
    t.corp_cache_dirty = offset
    offset += 1

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
# COMPANY SECTION SUB-OFFSETS
# =============================================================================

cdef CompanyOffsets compute_company_offsets() noexcept nogil:
    """Compute sub-offsets within the companies section.

    The section is laid out as three parallel 36-slot sub-arrays:
    adjusted incomes, location enums, and owner IDs (player_id /
    corp_id / -1). Keeping them contiguous lets ``StateLayout`` describe
    a single ``companies_offset`` instead of three independent slots.
    The final ``c.size`` field is the total length of the companies
    block and is used by compute_layout to size the section.
    """
    cdef CompanyOffsets c
    cdef int offset = 0

    c.incomes = offset
    offset += GameConstants.NUM_COMPANIES
    c.locations = offset
    offset += GameConstants.NUM_COMPANIES
    c.owner_ids = offset
    offset += GameConstants.NUM_COMPANIES

    c.size = offset
    return c


# =============================================================================
# DECK SECTION SUB-OFFSETS
# =============================================================================

cdef DeckOffsets compute_deck_offsets() noexcept nogil:
    """Compute sub-offsets within the deck section.

    The section is laid out as a single top-of-deck index slot followed
    by a 36-slot order array (shuffled company IDs). Keeping them
    grouped lets ``StateLayout`` describe a single ``deck_offset``
    instead of two independent slots. The final ``d.size`` field is the
    total length of the deck block and is used by compute_layout to
    size the section.
    """
    cdef DeckOffsets d
    cdef int offset = 0

    d.top = offset
    offset += 1
    d.order = offset
    offset += GameConstants.MAX_DECK_SIZE  # 36 company IDs

    d.size = offset
    return d


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
    c.total_stars = offset
    offset += 1
    c.cash_stars = offset
    offset += 1
    c.company_stars = offset
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
        players_offset=LAYOUT.players_offset,
        fi_offset=LAYOUT.fi_offset,
        companies_offset=LAYOUT.companies_offset,
        market_offset=LAYOUT.market_offset,
        corps_offset=LAYOUT.corps_offset,
        turn_offset=LAYOUT.turn_offset,
        deck_offset=LAYOUT.deck_offset,
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
        total_stars=CORP_FIELDS.total_stars,
        cash_stars=CORP_FIELDS.cash_stars,
        company_stars=CORP_FIELDS.company_stars,
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


def get_company_fields():
    """Python-accessible companies-section sub-offsets.

    Returns a CompanyFields namedtuple with relative offsets (add to
    companies_offset to get absolute position). The companies section
    is fixed-size — no num_players argument needed.
    """
    return CompanyFields(
        incomes=COMPANY_OFFSETS.incomes,
        locations=COMPANY_OFFSETS.locations,
        owner_ids=COMPANY_OFFSETS.owner_ids,
    )


def get_deck_fields():
    """Python-accessible deck-section sub-offsets.

    Returns a DeckFields namedtuple with relative offsets (add to
    deck_offset to get absolute position). The deck section is
    fixed-size — no num_players argument needed.
    """
    return DeckFields(
        top=DECK_OFFSETS.top,
        order=DECK_OFFSETS.order,
    )


def get_turn_fields():
    """Python-accessible turn state sub-offsets within the turn block.

    Returns a TurnFields namedtuple with relative offsets (add to
    turn_offset to get absolute position). The turn block is fixed-size —
    no num_players argument needed.
    """
    return TurnFields(
        active_player=TURN_OFFSETS.active_player,
        num_players=TURN_OFFSETS.num_players,
        phase=TURN_OFFSETS.phase,
        coo_level=TURN_OFFSETS.coo_level,
        turn_number=TURN_OFFSETS.turn_number,
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
COMPANY_OFFSETS = compute_company_offsets()
DECK_OFFSETS = compute_deck_offsets()


# =============================================================================
# GAME STATE CLASS
# =============================================================================

cdef inline int _expected_size(int num_players) noexcept nogil:
    return LAYOUT.players_offset + LAYOUT.player_stride * num_players


cdef void _seed_zeroed_storage(GameState state, int num_players):
    """Seed zero-initialized storage with the non-zero sentinels we require."""
    cdef int i
    cdef int owner_ids_base = LAYOUT.companies_offset + COMPANY_OFFSETS.owner_ids

    # num_players is canonical engine metadata and must always match the
    # backing-buffer size this GameState was initialized for.
    state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players] = <int16_t>num_players

    # Company locations default to LOC_DECK from zero-init, but owner_ids must
    # be -1 or a freshly allocated state would read as "all companies owned by
    # player 0".
    for i in range(<int>GameConstants.NUM_COMPANIES):
        state._data[owner_ids_base + i] = -1


cdef void _reset_storage(GameState state, int num_players):
    """Reset this GameState to a zeroed buffer for `num_players`.

    Reuses the current backing buffer when it already has the right size so a
    wrapped external buffer keeps its zero-copy behavior. If the requested
    player count needs a different size, a fresh numpy buffer is allocated and
    bound instead.
    """
    cdef int expected_size = _expected_size(num_players)
    cdef cnp.ndarray arr

    if state._array is not None:
        arr = np.asarray(state._array)
        if (arr.dtype == np.int16
                and arr.ndim == 1
                and <int>arr.shape[0] == expected_size
                and arr.flags['C_CONTIGUOUS']):
            arr.fill(0)
            state._array = arr
            state._data = <int16_t*>cnp.PyArray_DATA(arr)
            _seed_zeroed_storage(state, num_players)
            return

    arr = np.zeros(expected_size, dtype=np.int16)
    state._array = arr
    state._data = <int16_t*>cnp.PyArray_DATA(arr)
    _seed_zeroed_storage(state, num_players)


cdef inline _bind_buffer(GameState state, object buffer, int num_players):
    """Validate `buffer` and point `state` at it as its backing store.

    Shared body of ``from_buffer`` (which constructs a new wrapper) and
    ``rebind`` (which repoints an existing wrapper). Validation is
    ``assert``-based so the checks compile out under ``python -O`` for
    the MCTS hot path. The buffer is wrapped via ``np.asarray``, which
    is a no-op when the caller already passes in a numpy array.
    """
    cdef cnp.ndarray buf = np.asarray(buffer)
    cdef int expected_size = _expected_size(num_players)
    cdef int16_t* data
    cdef int canonical_num_players
    assert buf.dtype == np.int16, \
        f"Expected int16 array, got {buf.dtype}"
    assert buf.ndim == 1 and <int>buf.shape[0] == expected_size, \
        f"Expected 1-D array of length {expected_size}, " \
        f"got ndim={buf.ndim} len={<int>buf.shape[0]}"
    assert buf.flags['C_CONTIGUOUS'], "Buffer must be C-contiguous"
    data = <int16_t*>cnp.PyArray_DATA(buf)
    canonical_num_players = <int>data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
    assert canonical_num_players == num_players, \
        f"buffer canonical num_players {canonical_num_players} != claimed {num_players}"
    state._array = buf
    state._data = data


cdef class GameState:
    """
    Game state container (compact layout).

    Holds the raw memory buffer plus the player count. Layout offsets are
    *not* stored per-instance — they're constants on the module
    (`LAYOUT`, `TURN_OFFSETS`, `PLAYER_FIELDS`, `CORP_FIELDS`,
    `COMPANY_OFFSETS`, `DECK_OFFSETS`). The only num_players-dependent
    value is the total buffer size, computed inline where needed. Logic
    is delegated to Entity handles and Phase classes.
    """
    def __cinit__(self, unsigned int num_players, bint _alloc=True):
        if num_players < 2 or num_players > GameConstants.MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")

        self.step_mode = False

        if not _alloc:
            # Caller will set _array and _data (used by from_buffer). The
            # canonical num_players slot inside the turn block is assumed
            # to already be populated in the supplied buffer.
            return

        _reset_storage(self, <int>num_players)

    @staticmethod
    def from_array(array, int num_players):
        """Reconstruct GameState from raw numpy array.

        Args:
            array: numpy int16 array (will be copied)
            num_players: number of players (required to compute layout)

        Returns:
            New GameState with copied array data
        """
        cdef cnp.ndarray arr = np.asarray(array)
        cdef int expected_size = _expected_size(num_players)
        cdef int16_t* data
        cdef int canonical_num_players
        if arr.dtype != np.int16:
            raise ValueError(f"Expected int16 array, got {arr.dtype}")
        if arr.ndim != 1 or <int>arr.shape[0] != expected_size:
            raise ValueError(
                f"Expected 1-D array of length {expected_size}, "
                f"got ndim={arr.ndim} len={<int>arr.shape[0]}"
            )
        data = <int16_t*>cnp.PyArray_DATA(arr)
        canonical_num_players = <int>data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
        if canonical_num_players != num_players:
            raise ValueError(
                f"array canonical num_players {canonical_num_players} != claimed {num_players}"
            )
        state = GameState(num_players)
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
        the MCTS state pool). The canonical ``turn.num_players`` slot in
        that buffer must already match the claimed ``num_players``.

        Args:
            buffer: numpy int16 array of correct size (not copied)
            num_players: number of players (required for layout lookup)

        Returns:
            GameState backed by the provided buffer
        """
        state = GameState(num_players, _alloc=False)
        _bind_buffer(state, buffer, num_players)
        return state

    def rebind(self, buffer, int num_players):
        """Repoint this GameState at a different backing buffer (zero-copy).

        Reuses the existing wrapper instead of allocating a new one. Used
        in MCTS search hot paths to swap a scratch GameState across rows
        of a state pool. The new buffer may have a different player count
        than the current one — caller passes `num_players` explicitly so
        the size validation matches the new buffer.
        """
        _bind_buffer(self, buffer, num_players)

    # =========================================================================
    # GAME INITIALIZATION
    # =========================================================================

    cpdef void initialize_game(self, int num_players, int seed=-1):
        """
        Initialize a new game with all starting state.

        Args:
            num_players: Player count for the new game.
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
        if num_players < 2 or num_players > <int>GameConstants.MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")

        _reset_storage(self, num_players)

        # 1. Set player starting state (raw integers, no normalization)
        starting_cash = 25 if num_players == 6 else 30
        for i in range(num_players):
            player = self._data + LAYOUT.players_offset + i * LAYOUT.player_stride
            player[PLAYER_FIELDS.cash] = <int16_t>starting_cash
            player[PLAYER_FIELDS.net_worth] = <int16_t>starting_cash
            player[PLAYER_FIELDS.liquidity] = <int16_t>starting_cash
            player[PLAYER_FIELDS.turn_order] = <int16_t>i

        # 2. Set Foreign Investor state (raw integers)
        self._data[LAYOUT.fi_offset] = 4       # cash
        self._data[LAYOUT.fi_offset + 1] = 5   # income (base +5)

        # 3. Initialize corporations (only non-zero: unissued shares)
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            corp = self._data + LAYOUT.corps_offset + corp_id * LAYOUT.corp_stride
            corp[CORP_FIELDS.unissued_shares] = <int16_t>CORP_SHARE_COUNT[corp_id]

        # 4. Initialize market - all spaces available
        for i in range(<int>GameConstants.NUM_MARKET_SPACES):
            self._data[LAYOUT.market_offset + i] = 1

        # 5. Seed baseline CoO through the TurnState entity so the adjusted
        #    company-income cache and all dependent income fields start from a
        #    coherent state before any setup-time draws/transfers occur.
        turn_module.TURN.set_coo_level(self, 1)

        # 6. Build and shuffle deck (also marks excluded companies as LOC_EXCLUDED)
        if seed < 0:
            clock_gettime(CLOCK_MONOTONIC, &ts)
            actual_seed = <int>(ts.tv_sec ^ ts.tv_nsec)
        else:
            actual_seed = seed
        deck_module.DECK.setup(self, num_players, actual_seed)

        # 7. Draw initial companies (move_to_auction clears the revealed flag)
        for i in range(num_players):
            company_id = deck_module.DECK.draw(self)
            company_module.COMPANIES[company_id].move_to_auction(self)

        # 8. Set remaining turn-state scalars (CoO already seeded above and may
        #    have advanced during the initial setup draws).
        turn = self._data + LAYOUT.turn_offset
        turn[TURN_OFFSETS.phase] = <int16_t>GamePhases.PHASE_INVEST
        turn[TURN_OFFSETS.turn_number] = 1

        # 9. Initialize "no selection" auction sentinels to -1
        turn[TURN_OFFSETS.auction_company] = -1
        turn[TURN_OFFSETS.auction_high_bidder] = -1
        turn[TURN_OFFSETS.auction_starter] = -1

        # 10. Set active player
        turn_module.TURN.set_active_player(self, 0)

        # Player finance caches are already valid in the freshly seeded state:
        # players have starting cash, no shares, and no owned companies.
        turn[TURN_OFFSETS.player_cache_dirty] = 0
        # Corporation derived caches are also valid in the freshly seeded
        # state: all corps start inactive with zeroed derived fields.
        turn[TURN_OFFSETS.corp_cache_dirty] = 0
