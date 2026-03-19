# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation.

The state is a single contiguous float array organized as:
  [VISIBLE STATE (for NN)][HIDDEN STATE (truncated before NN)]

All game logic operates directly on the array - no Python object overhead.
The visible state is presented to the NN with player rotation (active player first).
"""

cimport cython
from libc.math cimport lround
from libc.string cimport memcpy, memset
from posix.time cimport clock_gettime, timespec, CLOCK_MONOTONIC
cimport numpy as cnp
import numpy as np

from collections import namedtuple

from core.data cimport get_adjusted_company_income
from core.data cimport (
    get_adjusted_company_income,
    GameConstants,
    GamePhases,
    CASH_DIVISOR,
    INCOME_DIVISOR,
    PRICE_DIVISOR,
    SHARE_DIVISOR,
    COMPANY_STAR_DIVISOR,
    MARKET_PRICES,
    get_corp_share_count,
    COMPANY_STARS,
    COMPANY_LOW_PRICE,
    COMPANY_FACE_VALUE,
    COMPANY_HIGH_PRICE,
    COMPANY_INCOME,
    get_company_stars,
    get_company_face_value,
    get_company_low_price,
    get_company_high_price,
    get_market_price,
)

LayoutInfo = namedtuple('LayoutInfo', [
    # Sizes
    'visible_size', 'hidden_size', 'total_size',
    'player_stride', 'players_size', 'fi_size',
    'corp_stride', 'corps_size', 'turn_size',
    'auction_slot_info_size', 'invest_impacts_size', 'phase_size', 'coo_size',
    'companies_size', 'company_incomes_size', 'market_size',
    # Visible offsets
    'phase_offset', 'coo_offset', 'players_offset', 'fi_offset',
    'auction_companies_offset', 'revealed_companies_offset',
    'removed_companies_offset', 'company_incomes_offset',
    'market_offset', 'corps_offset', 'turn_offset',
    'auction_slot_info_offset',
    'invest_impacts_offset',
    # Per-player turn field offsets (absolute, for rotation)
    'auction_high_bidder_offset', 'auction_starter_offset',
    'auction_passed_offset',
    # Active company offsets (absolute, within turn state)
    'active_company_offset',
    'active_company_info_offset',
    # Active corp offsets (absolute, within turn state)
    'active_corp_offset',
    'active_corp_info_offset',
    'active_corp_companies_offset',
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
from entities import offer as offer_module
from entities.company cimport get_auction_company_for_slot

cnp.import_array()

# Buffer size constants for hidden state offer buffers
# Note: These are duplicated in phases/acquisition.pyx and phases/closing.pyx
# because DEF (compile-time constant) is required for static array sizing and
# cannot be imported across Cython modules.
DEF OFFER_BUFFER_SIZE = 250
DEF CLOSE_OFFER_BUFFER_SIZE = 100

# =============================================================================
# STATE LAYOUT STRUCTURE
# =============================================================================
# Please refer to VECTORS.md for the detailed layout specification.
#
# =============================================================================


# =============================================================================
# LAYOUT COMPUTATION
# =============================================================================

cdef StateLayout compute_layout(int num_players) noexcept nogil:
    """Compute complete state layout for given player count."""
    cdef StateLayout layout
    cdef int offset = 0

    # Phase one-hot
    layout.phase_size = GameConstants.NUM_PHASES
    layout.phase_offset = offset
    offset += layout.phase_size

    # Cost of ownership one-hot
    layout.coo_size = GameConstants.NUM_COO_LEVELS
    layout.coo_offset = offset
    offset += layout.coo_size

    # Players
    layout.player_stride = (
        1 +                 # cash
        1 +                 # net_worth
        num_players +       # turn_order one-hot
        1 +                 # is_auction_high_bidder
        GameConstants.NUM_COMPANIES +     # owned_companies
        GameConstants.NUM_CORPS +         # owned_shares
        GameConstants.NUM_CORPS +         # is_president
        GameConstants.NUM_CORPS +         # share_buys
        GameConstants.NUM_CORPS +         # share_sells
        1 +                 # acquisition_proceeds
        1                   # income
    )
    layout.players_offset = offset
    layout.players_size = layout.player_stride * num_players
    offset += layout.players_size

    # Foreign investor
    layout.fi_size = 2 + GameConstants.NUM_COMPANIES # cash, income, owned companies
    layout.fi_offset = offset
    offset += layout.fi_size

    # Company locations (3 arrays)
    layout.companies_size = GameConstants.NUM_COMPANIES * 3
    layout.auction_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.revealed_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.removed_companies_offset = offset
    offset += GameConstants.NUM_COMPANIES

    # Company adjusted incomes (dynamic based on CoO level)
    layout.company_incomes_size = GameConstants.NUM_COMPANIES
    layout.company_incomes_offset = offset
    offset += GameConstants.NUM_COMPANIES

    # Market availability
    layout.market_size = GameConstants.NUM_MARKET_SPACES
    layout.market_offset = offset
    offset += layout.market_size

    # Corporations
    layout.corp_stride = (
        1 +                 # active
        1 +                 # cash
        1 +                 # unissued_shares
        1 +                 # issued_shares
        1 +                 # bank_shares
        1 +                 # income
        1 +                 # stars
        1 +                 # share_price
        1 +                 # acquisition_proceeds
        1 +                 # in_receivership
        GameConstants.NUM_MARKET_SPACES + # price_index one-hot
        GameConstants.NUM_COMPANIES +     # owned_companies
        GameConstants.NUM_COMPANIES       # acquisition_companies
    )
    layout.corps_offset = offset
    layout.corps_size = layout.corp_stride * GameConstants.NUM_CORPS
    offset += layout.corps_size

    # Turn state
    layout.turn_size = (
        1 +                 # turn_number
        1 +                 # end_card_flipped
        1 +                 # consecutive_passes (for INVEST phase)
        # Auction
        1 +                 # auction_price
        num_players +       # auction_high_bidder
        num_players +       # auction_starter
        num_players +       # auction_passed
        # Dividends
        GameConstants.MAX_DIVIDEND +      # dividend_impact
        GameConstants.NUM_CORPS +         # dividend_remaining
        # Issue
        GameConstants.NUM_CORPS +         # issue_remaining
        # IPO
        GameConstants.NUM_COMPANIES +     # ipo_remaining
        # Acquisition offers
        1 +                 # acq_is_fi_offer
        GameConstants.NUM_COMPANIES +     # acq_synergy_values
        # Active company
        GameConstants.NUM_COMPANIES +     # active_company (one-hot)
        5 +                              # active_company_info (stars, low, face, high, income)
        # Active corp
        GameConstants.NUM_CORPS +         # active_corp (one-hot)
        3 +                              # active_corp_info (income, stars, share_price)
        GameConstants.NUM_COMPANIES +     # active_corp_companies (owned company flags)
        # Deck
        1                                # cards_remaining
    )
    layout.turn_offset = offset
    offset += layout.turn_size

    # Auction slot info (5 scalars per slot: stars, low, face, high, income)
    layout.auction_slot_info_size = 5 * num_players
    layout.auction_slot_info_offset = offset
    offset += layout.auction_slot_info_size

    # Invest phase impacts (8 buy + 8 sell = 16 floats)
    layout.invest_impacts_size = 2 * GameConstants.NUM_CORPS
    layout.invest_impacts_offset = offset
    offset += layout.invest_impacts_size

    layout.visible_size = offset

    # Hidden state layout:
    # [0] active_player
    # [1] num_players
    # [2] deck_top
    # [3..38] deck_order (36 slots)
    # [39] phase (compact)
    # [40] coo_level (compact)
    # [41] auction_company (compact)
    # [42] auction_high_bidder (compact)
    # [43] auction_starter (compact)
    # [44..51] corp_price_indices (8 slots)
    # [52] offer_count (number of offers in buffer)
    # [53] offer_index (current offer being processed)
    # [54..803] offer_buffer (250 offers * 3 floats: owner_type, corp_id, company_id)
    # [804] close_offer_count (number of close offers)
    # [805] close_offer_index (current close offer being processed)
    # [806..1105] close_offer_buffer (100 offers * 3 floats: owner_type, owner_id, company_id)
    # [1106] acq_active_corp (compact, for O(1) access)
    # [1107] acq_target_company (compact, for O(1) access)
    # [1108] closing_company (compact, for O(1) access)
    # [1109] dividend_corp (compact, for O(1) access)
    # [1110] issue_corp (compact, for O(1) access)
    # [1111] ipo_company (compact, for O(1) access)
    layout.hidden_active_player_offset = offset
    offset += 1
    layout.hidden_num_players_offset = offset
    offset += 1
    layout.hidden_deck_top_offset = offset
    offset += 1
    layout.hidden_deck_order_offset = offset
    offset += GameConstants.MAX_DECK_SIZE
    layout.hidden_phase_offset = offset
    offset += 1
    layout.hidden_coo_level_offset = offset
    offset += 1
    layout.hidden_auction_company_offset = offset
    offset += 1
    layout.hidden_auction_high_bidder_offset = offset
    offset += 1
    layout.hidden_auction_starter_offset = offset
    offset += 1
    layout.hidden_corp_price_indices_offset = offset
    offset += GameConstants.NUM_CORPS
    layout.hidden_offer_count_offset = offset
    offset += 1
    layout.hidden_offer_index_offset = offset
    offset += 1
    layout.hidden_offer_buffer_offset = offset
    offset += OFFER_BUFFER_SIZE * 3  # 250 offers * 3 floats per offer (owner_type, corp_id, company_id)
    # Close offer buffer (100 offers max)
    # Each offer: owner_type (0=player, 1=corp), owner_id, company_id
    layout.hidden_close_offer_count_offset = offset
    offset += 1
    layout.hidden_close_offer_index_offset = offset
    offset += 1
    layout.hidden_close_offer_buffer_offset = offset
    offset += CLOSE_OFFER_BUFFER_SIZE * 3  # 100 offers * 3 floats per offer
    # O(1) access for one-hot fields (avoids scanning visible one-hot arrays)
    layout.hidden_acq_active_corp_offset = offset
    offset += 1
    layout.hidden_acq_target_company_offset = offset
    offset += 1
    layout.hidden_closing_company_offset = offset
    offset += 1
    layout.hidden_dividend_corp_offset = offset
    offset += 1
    layout.hidden_issue_corp_offset = offset
    offset += 1
    layout.hidden_ipo_company_offset = offset
    offset += 1
    # Company location tracking (O(1) clearing without scanning visible state)
    # [862..897] company_locations (36 floats: CompanyLocation enum per company)
    # [898..933] company_owner_ids (36 floats: player_id or corp_id, -1 when N/A)
    layout.hidden_company_locations_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.hidden_company_owner_ids_offset = offset
    offset += GameConstants.NUM_COMPANIES
    layout.hidden_size = offset - layout.visible_size

    layout.total_size = layout.visible_size + layout.hidden_size

    return layout


# =============================================================================
# TURN STATE SUB-OFFSETS
# =============================================================================

cdef TurnStateOffsets compute_turn_offsets(int num_players) noexcept nogil:
    """Compute sub-offsets within turn state section."""
    cdef TurnStateOffsets t
    cdef int offset = 0

    t.turn_number = offset
    offset += 1
    t.end_card_flipped = offset
    offset += 1
    t.consecutive_passes = offset
    offset += 1

    # Auction
    t.auction_price = offset
    offset += 1
    t.auction_high_bidder = offset
    offset += num_players
    t.auction_starter = offset
    offset += num_players
    t.auction_passed = offset
    offset += num_players

    # Dividends
    t.dividend_impact = offset
    offset += GameConstants.MAX_DIVIDEND
    t.dividend_remaining = offset
    offset += GameConstants.NUM_CORPS

    # Issue
    t.issue_remaining = offset
    offset += GameConstants.NUM_CORPS

    # IPO
    t.ipo_remaining = offset
    offset += GameConstants.NUM_COMPANIES

    # Acquisition offers
    t.acq_is_fi_offer = offset
    offset += 1
    t.acq_synergy_values = offset
    offset += GameConstants.NUM_COMPANIES

    # Active company: one-hot (36) + contextual info (5 scalars)
    t.active_company = offset
    offset += GameConstants.NUM_COMPANIES
    t.active_company_info = offset
    offset += 5

    # Active corp: one-hot (8) + contextual info (3 scalars) + owned companies (36 flags)
    t.active_corp = offset
    offset += GameConstants.NUM_CORPS
    t.active_corp_info = offset
    offset += 3
    t.active_corp_companies = offset
    offset += GameConstants.NUM_COMPANIES

    # Cards remaining in deck
    t.cards_remaining = offset
    offset += 1

    return t


# =============================================================================
# PLAYER FIELD OFFSETS (within player stride)
# =============================================================================

cdef PlayerFieldOffsets compute_player_field_offsets(int num_players) noexcept nogil:
    """Compute field offsets within a player's data block."""
    cdef PlayerFieldOffsets p
    cdef int offset = 0

    p.cash = offset
    offset += 1
    p.net_worth = offset
    offset += 1
    p.turn_order = offset
    offset += num_players
    p.is_auction_high_bidder = offset
    offset += 1
    p.owned_companies = offset
    offset += GameConstants.NUM_COMPANIES
    p.owned_shares = offset
    offset += GameConstants.NUM_CORPS
    p.is_president = offset
    offset += GameConstants.NUM_CORPS
    p.share_buys = offset
    offset += GameConstants.NUM_CORPS
    p.share_sells = offset
    offset += GameConstants.NUM_CORPS
    p.acquisition_proceeds = offset
    offset += 1
    p.income = offset

    return p


# =============================================================================
# CORP FIELD OFFSETS (within corp stride)
# =============================================================================

def get_layout(int num_players):
    """Python-accessible layout offsets. Single source of truth.

    Returns a LayoutInfo namedtuple with all sizes, offsets, and strides
    needed by Python code (MCTS evaluator, tests). Cython code should
    continue using the cdef structs directly for nogil performance.
    """
    cdef StateLayout layout = compute_layout(num_players)
    cdef TurnStateOffsets turn = compute_turn_offsets(num_players)
    return LayoutInfo(
        visible_size=layout.visible_size,
        hidden_size=layout.hidden_size,
        total_size=layout.total_size,
        player_stride=layout.player_stride,
        players_size=layout.players_size,
        fi_size=layout.fi_size,
        corp_stride=layout.corp_stride,
        corps_size=layout.corps_size,
        turn_size=layout.turn_size,
        auction_slot_info_size=layout.auction_slot_info_size,
        invest_impacts_size=layout.invest_impacts_size,
        phase_size=layout.phase_size,
        coo_size=layout.coo_size,
        companies_size=layout.companies_size,
        company_incomes_size=layout.company_incomes_size,
        market_size=layout.market_size,
        phase_offset=layout.phase_offset,
        coo_offset=layout.coo_offset,
        players_offset=layout.players_offset,
        fi_offset=layout.fi_offset,
        auction_companies_offset=layout.auction_companies_offset,
        revealed_companies_offset=layout.revealed_companies_offset,
        removed_companies_offset=layout.removed_companies_offset,
        company_incomes_offset=layout.company_incomes_offset,
        market_offset=layout.market_offset,
        corps_offset=layout.corps_offset,
        turn_offset=layout.turn_offset,
        auction_slot_info_offset=layout.auction_slot_info_offset,
        invest_impacts_offset=layout.invest_impacts_offset,
        auction_high_bidder_offset=layout.turn_offset + turn.auction_high_bidder,
        auction_starter_offset=layout.turn_offset + turn.auction_starter,
        auction_passed_offset=layout.turn_offset + turn.auction_passed,
        active_company_offset=layout.turn_offset + turn.active_company,
        active_company_info_offset=layout.turn_offset + turn.active_company_info,
        active_corp_offset=layout.turn_offset + turn.active_corp,
        active_corp_info_offset=layout.turn_offset + turn.active_corp_info,
        active_corp_companies_offset=layout.turn_offset + turn.active_corp_companies,
        num_players=num_players,
    )


cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil:
    """Compute field offsets within a corp's data block."""
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
    offset += GameConstants.NUM_MARKET_SPACES
    c.owned_companies = offset
    offset += GameConstants.NUM_COMPANIES
    c.acquisition_companies = offset

    return c

# =============================================================================
# CACHED LAYOUT TABLES (computed once, reused by all GameState instances)
# =============================================================================

# Indexed by num_players (slots 0-1 unused, 2-6 valid)
cdef StateLayout _cached_layouts[7]
cdef TurnStateOffsets _cached_turns[7]
cdef PlayerFieldOffsets _cached_player_fields[7]
cdef CorpFieldOffsets _cached_corp_fields[1]

cdef int _npl_init
_cached_corp_fields[0] = compute_corp_field_offsets()
for _npl_init in range(2, 7):
    _cached_layouts[_npl_init] = compute_layout(_npl_init)
    _cached_turns[_npl_init] = compute_turn_offsets(_npl_init)
    _cached_player_fields[_npl_init] = compute_player_field_offsets(_npl_init)


# =============================================================================
# GAME STATE CLASS
# =============================================================================

cdef class GameState:
    """
    Game state container.

    Holds the raw memory buffer and layout information.
    Logic is delegated to Entity handles and Phase classes.
    """
    def __cinit__(self, unsigned int num_players, bint _alloc=True):
        if num_players < 2 or num_players > GameConstants.MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")

        self._num_players = num_players

        # Look up precomputed layouts (struct copy, ~160 bytes)
        self._layout = _cached_layouts[num_players]
        self._turn_offsets = _cached_turns[num_players]
        self._player_fields = _cached_player_fields[num_players]
        self._corp_fields = _cached_corp_fields[0]
        self._turn = self._turn_offsets

        if not _alloc:
            # Caller will set _array and _data (used by from_buffer)
            return

        # Allocate array (zero-initialized)
        self._array = np.zeros(self._layout.total_size, dtype=np.float32)
        self._data = <float*>cnp.PyArray_DATA(self._array)

        # Initialize constant hidden state fields
        self._data[self._layout.hidden_num_players_offset] = <float>num_players

        # Initialize company owner_ids to -1 (no owner when in deck)
        # Company locations are already 0 (LOC_DECK) from zero-initialization
        cdef int i
        for i in range(<int>GameConstants.NUM_COMPANIES):
            self._data[self._layout.hidden_company_owner_ids_offset + i] = -1.0

    @staticmethod
    def from_array(array, int num_players):
        """Reconstruct GameState from raw numpy array.

        Args:
            array: numpy float32 array (will be copied)
            num_players: number of players (required to compute layout)

        Returns:
            New GameState with copied array data
        """
        state = GameState(num_players)
        cdef cnp.ndarray arr = np.asarray(array)
        if arr.dtype != np.float32:
            raise ValueError(f"Expected float32 array, got {arr.dtype}")
        if arr.ndim != 1 or <int>arr.shape[0] != state._layout.total_size:
            py_shape = np.PyArray_DIMS(arr)
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

        Args:
            buffer: numpy float32 array of correct size (not copied)
            num_players: number of players (required for layout lookup)

        Returns:
            GameState backed by the provided buffer
        """
        state = GameState(num_players, _alloc=False)
        cdef cnp.ndarray buf = np.asarray(buffer)
        if buf.dtype != np.float32:
            raise ValueError(f"Expected float32 array, got {buf.dtype}")
        if buf.ndim != 1 or <int>buf.shape[0] != state._layout.total_size:
            raise ValueError(
                f"Expected 1-D array of length {state._layout.total_size}, "
                f"got ndim={buf.ndim} len={<int>buf.shape[0]}"
            )
        if not buf.flags['C_CONTIGUOUS']:
            raise ValueError("Buffer must be C-contiguous")
        state._array = buf
        state._data = <float*>cnp.PyArray_DATA(buf)
        return state

    def rebind(self, buffer):
        """Rebind this GameState to a different backing buffer (zero-copy).

        Avoids allocating a new GameState wrapper when only the underlying
        data changes. Used in MCTS search hot paths to eliminate per-node
        Python object allocation.

        Args:
            buffer: numpy float32 array of correct size (not copied)
        """
        self._array = buffer
        self._data = <float*>cnp.PyArray_DATA(buffer)

    # =========================================================================
    # INTERNAL POINTER ACCESS
    # =========================================================================

    cdef float* _player_ptr(self, int player_id) noexcept nogil:
        """Get pointer to player data block."""
        return self._data + self._layout.players_offset + (player_id * self._layout.player_stride)

    cdef float* _corp_ptr(self, int corp_id) noexcept nogil:
        """Get pointer to corporation data block."""
        return self._data + self._layout.corps_offset + (corp_id * self._layout.corp_stride)

    cdef float* _turn_ptr(self) noexcept nogil:
        """Get pointer to turn state."""
        return self._data + self._layout.turn_offset

    cdef int _get_active_player(self) noexcept nogil:
        """Get active player ID from hidden state."""
        return <int>self._data[self._layout.hidden_active_player_offset]

    cdef void _set_active_player(self, int player_id) noexcept nogil:
        """Set active player ID in hidden state."""
        self._data[self._layout.hidden_active_player_offset] = <float>player_id

    cpdef int get_active_player(self):
        """Get active player ID (Python-accessible)."""
        return self._get_active_player()

    cpdef int get_num_players(self):
        """Get number of players (Python-accessible)."""
        return self._num_players

    # =========================================================================
    # PHASE ACCESS
    # =========================================================================

    cpdef int get_phase(self):
        """Get current phase from hidden state."""
        return <int>self._data[self._layout.hidden_phase_offset]

    # Note: set_phase() removed - use TurnState.set_phase() to avoid duplication

    # =========================================================================
    # PLAYER ACCESS
    # =========================================================================

    cpdef int get_player_cash(self, int player_id):
        """Get player's cash (denormalized to integer dollars)."""
        cdef float* player = self._player_ptr(player_id)
        return <int>(player[self._player_fields.cash] * CASH_DIVISOR + 0.5)

    cpdef void set_player_cash(self, int player_id, int cash):
        """Set player's cash."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.cash] = <float>cash / CASH_DIVISOR

    cpdef int get_player_net_worth(self, int player_id):
        """Get player's net worth."""
        cdef float* player = self._player_ptr(player_id)
        return <int>(player[self._player_fields.net_worth] * CASH_DIVISOR + 0.5)

    cpdef void set_player_net_worth(self, int player_id, int net_worth):
        """Set player's net worth."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.net_worth] = <float>net_worth / CASH_DIVISOR

    # =========================================================================
    # CORPORATION ACCESS
    # =========================================================================

    cdef bint _is_corp_active(self, int corp_id) noexcept nogil:
        """Check if corporation is active (nogil version)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.active] == 1.0

    cpdef bint is_corp_active(self, int corp_id):
        """Check if corporation is active."""
        return self._is_corp_active(corp_id)

    cpdef void set_corp_active(self, int corp_id, bint active):
        """Set corporation active status."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.active] = 1.0 if active else 0.0

    cpdef int get_corp_cash(self, int corp_id):
        """Get corporation's cash."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>lround(corp[self._corp_fields.cash] * CASH_DIVISOR)

    cpdef void set_corp_cash(self, int corp_id, int cash):
        """Set corporation's cash."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.cash] = <float>cash / CASH_DIVISOR

    cpdef int get_corp_bank_shares(self, int corp_id):
        """Get corporation's bank shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.bank_shares] * SHARE_DIVISOR + 0.5)

    cpdef void set_corp_bank_shares(self, int corp_id, int shares):
        """Set corporation's bank shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.bank_shares] = <float>shares / SHARE_DIVISOR

    cpdef int get_corp_unissued_shares(self, int corp_id):
        """Get corporation's unissued shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.unissued_shares] * SHARE_DIVISOR + 0.5)

    cpdef void set_corp_unissued_shares(self, int corp_id, int shares):
        """Set corporation's unissued shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.unissued_shares] = <float>shares / SHARE_DIVISOR

    cpdef int get_corp_issued_shares(self, int corp_id):
        """Get corporation's issued shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.issued_shares] * SHARE_DIVISOR + 0.5)

    cpdef void set_corp_issued_shares(self, int corp_id, int shares):
        """Set corporation's issued shares."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.issued_shares] = <float>shares / SHARE_DIVISOR

    cdef int _get_corp_share_price(self, int corp_id) noexcept nogil:
        """Get corporation's share price (nogil version)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.share_price] * PRICE_DIVISOR + 0.5)

    cpdef int get_corp_share_price(self, int corp_id):
        """Get corporation's share price."""
        return self._get_corp_share_price(corp_id)

    cpdef void set_corp_share_price(self, int corp_id, int price):
        """Set corporation's share price."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.share_price] = <float>price / PRICE_DIVISOR

    cpdef int get_corp_price_index(self, int corp_id):
        """Get corporation's market price index from hidden state."""
        return <int>self._data[self._layout.hidden_corp_price_indices_offset + corp_id]

    cpdef void set_corp_price_index(self, int corp_id, int index):
        """Set corporation's market price index in hidden and one-hot."""
        cdef float* corp = self._corp_ptr(corp_id)
        cdef int i
        # Update hidden compact value
        self._data[self._layout.hidden_corp_price_indices_offset + corp_id] = <float>index
        # Update one-hot encoding
        for i in range(<int>GameConstants.NUM_MARKET_SPACES):
            corp[self._corp_fields.price_index + i] = 1.0 if i == index else 0.0

    cpdef bint is_corp_in_receivership(self, int corp_id):
        """Check if corporation is in receivership."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.in_receivership] == 1.0

    cpdef void set_corp_in_receivership(self, int corp_id, bint in_recv):
        """Set corporation receivership status."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.in_receivership] = 1.0 if in_recv else 0.0

    cdef bint _corp_owns_company(self, int corp_id, int company_id) noexcept nogil:
        """Check if corp owns a company (nogil version)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.owned_companies + company_id] == 1.0

    cpdef bint corp_owns_company(self, int corp_id, int company_id):
        """Check if corp owns a company."""
        return self._corp_owns_company(corp_id, company_id)

    cpdef void set_corp_owns_company(self, int corp_id, int company_id, bint owns):
        """Set corp company ownership."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.owned_companies + company_id] = 1.0 if owns else 0.0

    # =========================================================================
    # MARKET ACCESS
    # =========================================================================

    cpdef bint is_market_space_available(self, int index):
        """Check if market space is available."""
        return self._data[self._layout.market_offset + index] == 1.0

    cpdef void set_market_space_available(self, int index, bint available):
        """Set market space availability."""
        self._data[self._layout.market_offset + index] = 1.0 if available else 0.0

    # =========================================================================
    # COMPANY ACCESS
    # =========================================================================

    cdef bint _is_company_for_auction(self, int company_id) noexcept nogil:
        """Check if company is available for auction (nogil version)."""
        return self._data[self._layout.auction_companies_offset + company_id] == 1.0

    cpdef bint is_company_for_auction(self, int company_id):
        """Check if company is available for auction."""
        return self._is_company_for_auction(company_id)

    cpdef void set_company_for_auction(self, int company_id, bint for_auction):
        """Set company auction availability."""
        self._data[self._layout.auction_companies_offset + company_id] = 1.0 if for_auction else 0.0

    # =========================================================================
    # AUCTION STATE ACCESS
    # =========================================================================

    cpdef int get_auction_company(self):
        """Get current auction company from hidden state."""
        return <int>self._data[self._layout.hidden_auction_company_offset]

    # Note: set_auction_company() removed - use TurnState.set_auction_company()

    cpdef int get_auction_price(self):
        """Get current auction price."""
        cdef float* turn = self._turn_ptr()
        return <int>(turn[self._turn_offsets.auction_price] * PRICE_DIVISOR + 0.5)

    # Note: set_auction_price() removed - use TurnState.set_auction_price()

    # =========================================================================
    # ACQUISITION STATE ACCESS
    # =========================================================================

    cdef int _get_acq_active_corp(self) noexcept nogil:
        """Get active corp in acquisition phase (nogil version)."""
        return <int>self._data[self._layout.hidden_acq_active_corp_offset]

    cpdef int get_acq_active_corp(self):
        """Get active corp in acquisition phase."""
        return self._get_acq_active_corp()

    # Note: set_acq_active_corp() removed - use TurnState.set_acq_active_corp()

    cdef int _get_acq_target_company(self) noexcept nogil:
        """Get target company in acquisition phase (nogil version)."""
        return <int>self._data[self._layout.hidden_acq_target_company_offset]

    cpdef int get_acq_target_company(self):
        """Get target company in acquisition phase."""
        return self._get_acq_target_company()

    # Note: set_acq_target_company() removed - use TurnState.set_acq_target_company()

    cpdef bint is_acq_fi_offer(self):
        """Check if acquisition is an FI offer."""
        cdef float* turn = self._turn_ptr()
        return turn[self._turn_offsets.acq_is_fi_offer] == 1.0

    # Note: set_acq_fi_offer() removed - use TurnState.set_acq_fi_offer()

    # =========================================================================
    # DIVIDEND STATE ACCESS
    # =========================================================================

    cdef int _get_dividend_corp(self) noexcept nogil:
        """Get current dividend corp (nogil version)."""
        return <int>self._data[self._layout.hidden_dividend_corp_offset]

    cpdef int get_dividend_corp(self):
        """Get current dividend corp."""
        return self._get_dividend_corp()

    # Note: set_dividend_corp() removed - use TurnState.set_dividend_corp()

    # =========================================================================
    # ISSUE STATE ACCESS
    # =========================================================================

    cdef int _get_issue_corp(self) noexcept nogil:
        """Get current issue corp (nogil version)."""
        return <int>self._data[self._layout.hidden_issue_corp_offset]

    cpdef int get_issue_corp(self):
        """Get current issue corp."""
        return self._get_issue_corp()

    # Note: set_issue_corp() removed - use TurnState.set_issue_corp()

    # =========================================================================
    # IPO STATE ACCESS
    # =========================================================================

    cdef int _get_ipo_company(self) noexcept nogil:
        """Get current IPO company (nogil version)."""
        return <int>self._data[self._layout.hidden_ipo_company_offset]

    cpdef int get_ipo_company(self):
        """Get current IPO company."""
        return self._get_ipo_company()

    # Note: set_ipo_company() removed - use TurnState.set_ipo_company()

    # =========================================================================
    # CLOSING STATE ACCESS
    # =========================================================================

    cdef int _get_current_closing_company(self) noexcept nogil:
        """Get current company being closed (nogil version)."""
        return <int>self._data[self._layout.hidden_closing_company_offset]

    cpdef int get_current_closing_company(self):
        """Get current company being closed."""
        return self._get_current_closing_company()

    # Note: set_current_closing_company() removed - use TurnState.set_closing_company()

    # =========================================================================
    # AUCTION SLOT INFO
    # =========================================================================

    cpdef void _populate_auction_slot_info(self):
        """
        Populate auction slot info block with company data for each auction slot.

        For each slot i (0..num_players-1), writes 5 normalized scalars:
        stars, low_price, face_value, high_price, adjusted_income.
        Empty slots are zero-filled.

        Called whenever the auction row changes:
        - Game initialization (after initial draw)
        - WRAP_UP (after revealed->auction transition)
        - BID resolution (auction winner removes a company)
        """
        cdef int slot, company_id, base
        cdef int coo_level = <int>self._data[self._layout.hidden_coo_level_offset]

        for slot in range(self._num_players):
            base = self._layout.auction_slot_info_offset + slot * 5
            company_id = get_auction_company_for_slot(self, slot)
            if company_id >= 0:
                self._data[base + 0] = <float>get_company_stars(company_id) / COMPANY_STAR_DIVISOR
                self._data[base + 1] = <float>get_company_low_price(company_id) / PRICE_DIVISOR
                self._data[base + 2] = <float>get_company_face_value(company_id) / PRICE_DIVISOR
                self._data[base + 3] = <float>get_company_high_price(company_id) / PRICE_DIVISOR
                self._data[base + 4] = <float>get_adjusted_company_income(company_id, coo_level) / INCOME_DIVISOR
            else:
                self._data[base + 0] = 0.0
                self._data[base + 1] = 0.0
                self._data[base + 2] = 0.0
                self._data[base + 3] = 0.0
                self._data[base + 4] = 0.0

    # =========================================================================
    # INVEST PHASE IMPACTS
    # =========================================================================

    cpdef void _populate_invest_impacts(self):
        """
        Compute buy/sell net worth impact for the active player for each corp.

        Layout: [buy_impact_0..buy_impact_7, sell_impact_0..sell_impact_7]
        Each normalized by PRICE_DIVISOR.

        Net worth delta from buying or selling one share:
          delta = current_shares * (new_price - old_price)

        Buy: new_price = price at next higher available market space
        Sell: new_price = price at next lower available market space

        Invalid actions (no bank shares, can't afford, no player shares) → 0.
        """
        cdef int corp_id, player_id, shares, current_index, new_index
        cdef int old_price, new_price, impact, buy_base, sell_base
        cdef int player_cash

        player_id = self._get_active_player()
        player_cash = player_module.PLAYERS[player_id].get_cash(self)
        buy_base = self._layout.invest_impacts_offset
        sell_base = buy_base + <int>GameConstants.NUM_CORPS

        for corp_id in range(<int>GameConstants.NUM_CORPS):
            self._data[buy_base + corp_id] = 0.0
            self._data[sell_base + corp_id] = 0.0

            if not corp_module.CORPS[corp_id].is_active(self):
                continue

            current_index = corp_module.CORPS[corp_id].get_price_index(self)
            old_price = get_market_price(current_index)
            shares = player_module.PLAYERS[player_id].get_shares(self, corp_id)

            # Buy impact: need bank shares available and player can afford
            if corp_module.CORPS[corp_id].get_bank_shares(self) > 0:
                new_index = market_module.MARKET.find_next_higher_space(self, current_index)
                new_price = get_market_price(new_index)
                if player_cash >= new_price:
                    impact = shares * (new_price - old_price)
                    self._data[buy_base + corp_id] = <float>impact / PRICE_DIVISOR

            # Sell impact: need player to own shares
            if shares > 0:
                new_index = market_module.MARKET.find_next_lower_space(self, current_index)
                new_price = get_market_price(new_index)
                impact = shares * (new_price - old_price)
                self._data[sell_base + corp_id] = <float>impact / PRICE_DIVISOR

    cpdef void _clear_invest_impacts(self):
        """Zero all invest impact slots."""
        cdef int i
        cdef int base = self._layout.invest_impacts_offset
        for i in range(self._layout.invest_impacts_size):
            self._data[base + i] = 0.0

    # =========================================================================
    # ACTIVE COMPANY CONTEXTUAL INFO
    # =========================================================================

    cpdef void set_active_company(self, int company_id):
        """
        Set active company info scalars in turn state.

        Writes stars, low_price, face_value, high_price, adjusted_income
        for the given company. The one-hot is managed by the per-phase
        set/clear methods (set_auction_company, set_acq_target_company, etc.).
        """
        cdef int base = self._layout.turn_offset + self._turn_offsets.active_company_info
        cdef int coo_level = <int>self._data[self._layout.hidden_coo_level_offset]
        self._data[base + 0] = <float>get_company_stars(company_id) / COMPANY_STAR_DIVISOR
        self._data[base + 1] = <float>get_company_low_price(company_id) / PRICE_DIVISOR
        self._data[base + 2] = <float>get_company_face_value(company_id) / PRICE_DIVISOR
        self._data[base + 3] = <float>get_company_high_price(company_id) / PRICE_DIVISOR
        self._data[base + 4] = <float>get_adjusted_company_income(company_id, coo_level) / INCOME_DIVISOR

    cpdef void clear_active_company(self):
        """Clear active company info scalars (zero-fill 5 floats).

        The one-hot is cleared by the per-phase clear methods.
        """
        cdef int base = self._layout.turn_offset + self._turn_offsets.active_company_info
        self._data[base + 0] = 0.0
        self._data[base + 1] = 0.0
        self._data[base + 2] = 0.0
        self._data[base + 3] = 0.0
        self._data[base + 4] = 0.0

    # =========================================================================
    # ACTIVE CORP CONTEXTUAL INFO
    # =========================================================================

    cpdef void set_active_corp(self, int corp_id):
        """
        Set active corp info and owned companies in turn state.

        Writes income, stars, and share_price (all normalized) from the corp's
        data block, plus copies the 36-element owned_companies flags.
        The one-hot is managed by the per-phase set/clear methods
        (set_dividend_corp, set_issue_corp, set_acq_active_corp).
        """
        cdef int info_base = self._layout.turn_offset + self._turn_offsets.active_corp_info
        cdef int companies_base = self._layout.turn_offset + self._turn_offsets.active_corp_companies
        cdef float* corp = self._corp_ptr(corp_id)
        cdef int i
        # Income, stars, and share_price (already normalized in corp data block)
        self._data[info_base + 0] = corp[self._corp_fields.income]
        self._data[info_base + 1] = corp[self._corp_fields.stars]
        self._data[info_base + 2] = corp[self._corp_fields.share_price]
        # Copy owned company flags (36 floats)
        for i in range(<int>GameConstants.NUM_COMPANIES):
            self._data[companies_base + i] = corp[self._corp_fields.owned_companies + i]

    cpdef void clear_active_corp(self):
        """Clear active corp info and owned companies (zero-fill).

        The one-hot is cleared by the per-phase clear methods.
        """
        cdef int info_base = self._layout.turn_offset + self._turn_offsets.active_corp_info
        cdef int companies_base = self._layout.turn_offset + self._turn_offsets.active_corp_companies
        cdef int i
        self._data[info_base + 0] = 0.0
        self._data[info_base + 1] = 0.0
        self._data[info_base + 2] = 0.0
        for i in range(<int>GameConstants.NUM_COMPANIES):
            self._data[companies_base + i] = 0.0

    # =========================================================================
    # GAME INITIALIZATION
    # =========================================================================

    cpdef void initialize_game(self, int seed=-1):
        """
        Initialize a new game with all starting state.

        Args:
            seed: Random seed for deck shuffling. If -1 (default), uses current time.

        This sets up:
        - Players with starting cash, turn order, and cleared ownership
        - Foreign Investor with starting cash and no companies
        - All corporations as inactive with unissued shares
        - All market spaces available
        - Deck built and shuffled, with initial companies drawn
        - Turn state for phase 1, turn 1, active player 0
        """
        cdef int i, corp_id, company_id
        cdef int actual_seed
        cdef int starting_cash
        cdef timespec ts

        # 1. Initialize all entity handles FIRST
        for i in range(self._num_players):
            player_module.PLAYERS[i].initialize(self)
        fi_module.FI.initialize(self)
        for corp in corp_module.CORPS:
            corp.initialize(self)
        for company in company_module.COMPANIES:
            company.initialize(self)
        market_module.MARKET.initialize(self)
        turn_module.TURN.initialize(self)
        deck_module.DECK.initialize(self)
        offer_module.ACQ_OFFERS.initialize(self)
        offer_module.CLOSE_OFFERS.initialize(self)

        # 2. Set player starting state (array starts as zeros, only set non-zero values)
        starting_cash = 25 if self._num_players == 6 else 30
        for i in range(self._num_players):
            player_module.PLAYERS[i].set_cash(self, starting_cash)
            player_module.PLAYERS[i].set_turn_order(self, i)
            player_module.PLAYERS[i].set_net_worth(self, starting_cash)

        # 3. Set Foreign Investor state
        fi_module.FI.set_cash(self, 4)
        fi_module.FI.set_income(self, 5)  # FI base income (+5, no companies yet)

        # 4. Initialize corporations (only non-zero: unissued shares)
        for corp in corp_module.CORPS:
            corp.set_unissued_shares(self, get_corp_share_count(corp.corp_id))

        # 5. Initialize market - all spaces available
        for i in range(<int>GameConstants.NUM_MARKET_SPACES):
            market_module.MARKET.set_space_available(self, i, True)

        # 6. Build and shuffle deck
        if seed < 0:
            clock_gettime(CLOCK_MONOTONIC, &ts)
            actual_seed = <int>(ts.tv_sec ^ ts.tv_nsec)
        else:
            actual_seed = seed
        deck_module.DECK.setup(self, self._num_players, actual_seed)

        # 7. Draw initial companies (move_to_auction clears the revealed flag set by draw)
        for i in range(self._num_players):
            company_id = deck_module.DECK.draw(self)
            company_module.COMPANIES[company_id].move_to_auction(self)

        # 8. Mark excluded companies
        # In games with < 6 players, some companies aren't included in the deck.
        # Their hidden location defaults to LOC_DECK (0) from the zero-initialized
        # array, but they're not actually in the deck. Mark them as excluded in
        # hidden state only (visible state stays untouched to avoid leaking deck
        # composition to the NN).
        deck_companies = set(deck_module.DECK.get_order(self))
        for i in range(<int>GameConstants.NUM_COMPANIES):
            if (company_module.COMPANIES[i].get_location(self) == 0  # LOC_DECK
                    and i not in deck_companies):
                company_module.COMPANIES[i].exclude_from_game(self)

        # 9. Set turn state (non-zero values only)
        turn_module.TURN.set_phase(self, GamePhases.PHASE_INVEST)
        turn_module.TURN.set_coo_level(self, 1)
        turn_module.TURN.set_turn_number(self, 1)

        # Clear one-hot encodings (sets compact storage to -1.0 for "no selection")
        turn_module.TURN.clear_auction_company(self)
        turn_module.TURN.clear_auction_high_bidder(self)
        turn_module.TURN.clear_auction_starter(self)
        turn_module.TURN.clear_dividend_corp(self)
        turn_module.TURN.clear_issue_corp(self)
        turn_module.TURN.clear_ipo_company(self)
        turn_module.TURN.clear_acq_active_corp(self)
        turn_module.TURN.clear_acq_target_company(self)
        turn_module.TURN.clear_closing_company(self)

        # 10. Clear active corp info (one-hot cleared above via clear_*_corp)
        self.clear_active_corp()

        # 11. Populate auction slot info for initial auction row
        self._populate_auction_slot_info()

        # Set active player
        self._set_active_player(0)

        # 12. Populate invest impacts for first player
        self._populate_invest_impacts()