# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game state implementation.

The state is a single contiguous float array organized as:
  [VISIBLE STATE (for NN)][HIDDEN STATE (truncated before NN)]

All game logic operates directly on the array - no Python object overhead.
The visible state is presented to the NN with player rotation (active player first).
"""

cimport cython
from libc.string cimport memcpy, memset
cimport numpy as cnp
import numpy as np

from data cimport get_adjusted_company_income

cnp.import_array()

# =============================================================================
# CONSTANTS
# =============================================================================

DEF NUM_COMPANIES = 36
DEF NUM_CORPS = 8
DEF NUM_MARKET_SPACES = 27
DEF NUM_PHASES = 11  # Including GAME_OVER
DEF NUM_COO_LEVELS = 7  # Cost of ownership levels 1-7
DEF MAX_PLAYERS = 6
DEF MAX_DECK_SIZE = 36
DEF NUM_PAR_PRICES = 14
DEF MAX_DIVIDEND = 26  # Max dividend per share option

# Phase indices
DEF PHASE_INVEST = 0
DEF PHASE_BID_IN_AUCTION = 1
DEF PHASE_WRAP_UP = 2
DEF PHASE_ACQUISITION = 3
DEF PHASE_CLOSING = 4
DEF PHASE_INCOME = 5
DEF PHASE_DIVIDENDS = 6
DEF PHASE_END_CARD = 7
DEF PHASE_ISSUE_SHARES = 8
DEF PHASE_IPO = 9
DEF PHASE_GAME_OVER = 10

# Normalization divisors
DEF CASH_DIVISOR = 200.0
DEF SHARE_DIVISOR = 7.0
DEF STAR_DIVISOR = 20.0
DEF MAX_ROUNDTRIPS = 2.0
DEF INCOME_DIVISOR = 10.0  # Company adjusted income range is roughly -10 to +10

# Market prices (27 spaces, index 0 = bankruptcy)
cdef int[27] MARKET_PRICES = [
    0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16,
    18, 20, 22, 24, 27, 30, 33, 37, 41, 45,
    50, 55, 61, 68, 75
]

# =============================================================================
# STATE LAYOUT STRUCTURE
# =============================================================================
#
# VISIBLE STATE (fed to NN, with player rotation):
# ------------------------------------------------
# phase[12]                    - one-hot phase
# coo[7]                       - one-hot cost of ownership level
#
# PLAYERS (repeated num_players times):
#   cash[1]                    - normalized
#   net_worth[1]               - normalized
#   turn_order[num_players]    - one-hot position
#   is_auction_high_bidder[1]  - auction flag
#   owned_companies[36]        - flags
#   owned_shares[8]            - normalized per corp
#   is_president[8]            - flags
#   share_buys[8]              - round-trip tracking
#   share_sells[8]             - round-trip tracking
#
# FOREIGN INVESTOR:
#   fi_cash[1]                 - normalized
#   fi_companies[36]           - flags
#
# COMPANY LOCATIONS:
#   companies_for_auction[36]  - flags
#   companies_revealed[36]     - flags (unavailable this turn)
#   companies_removed[36]      - flags (out of game)
#
# MARKET:
#   market_available[27]       - flags (1=available, 0=taken)
#
# CORPS (repeated 8 times):
#   active[1]                  - flag
#   cash[1]                    - normalized
#   unissued_shares[1]         - normalized
#   issued_shares[1]           - normalized
#   bank_shares[1]             - normalized
#   income[1]                  - normalized (derived)
#   stars[1]                   - normalized (derived)
#   share_price[1]             - normalized
#   acquisition_proceeds[1]    - normalized
#   in_receivership[1]         - flag
#   price_index[27]            - one-hot market position
#   owned_companies[36]        - flags
#   acquisition_companies[36]  - flags (pending this phase)
#
# TURN STATE:
#   turn_number[1]             - normalized
#   end_card_flipped[1]        - flag
#   consecutive_passes[1]      - normalized (passes / num_players)
#
#   # Auction (BID_IN_AUCTION phase, -1 otherwise)
#   auction_company[36]        - one-hot
#   auction_price[1]           - normalized
#   auction_high_bidder[num_players] - one-hot
#   auction_starter[num_players]     - one-hot
#   auction_passed[num_players]      - flags
#
#
#   # Dividends (DIVIDENDS phase, -1 otherwise)
#   dividend_corp[8]           - one-hot current
#   dividend_impact[26]        - price impact per level
#   dividend_remaining[8]      - flags corps left
#
#   # Issue (ISSUE_SHARES phase, -1 otherwise)
#   issue_corp[8]              - one-hot current
#   issue_remaining[8]         - flags corps left
#
#   # IPO (IPO phase, -1 otherwise)
#   ipo_company[36]            - one-hot current
#   ipo_remaining[36]          - flags companies left
#
#   # Acquisition (ACQUISITION phase, -1 otherwise)
#   acq_active_corp[8]         - one-hot buying corp
#   acq_target_company[36]     - one-hot target company
#   acq_is_fi_offer[1]         - flag (1=FI target, 0=player/corp target)
#
# STATIC COMPANY DATA (constant, for NN reference):
#   For each company[36]:
#     stars[1]                 - normalized
#     low_price[1]             - normalized
#     face_value[1]            - normalized
#     high_price[1]            - normalized
#     synergies[36]            - flags
#
# HIDDEN STATE (truncated before NN):
# ------------------------------------
# active_player[1]             - canonical player index
# num_players[1]               - player count (for variable games)
# deck_top[1]                  - index of top card (-1 if empty)
# deck_order[36]               - company IDs in order (-1 for empty slots)
#
# =============================================================================


# =============================================================================
# LAYOUT COMPUTATION
# =============================================================================

cdef struct StateLayout:
    # Visible section sizes
    int phase_size
    int coo_size
    int player_stride
    int players_size
    int fi_size
    int companies_size
    int market_size
    int corp_stride
    int corps_size
    int turn_size
    int static_size
    int visible_size

    # Hidden section sizes
    int hidden_size

    # Total
    int total_size

    # Offsets (into visible section)
    int phase_offset
    int coo_offset
    int players_offset
    int fi_offset
    int auction_companies_offset
    int revealed_companies_offset
    int removed_companies_offset
    int market_offset
    int corps_offset
    int turn_offset
    int static_offset

    # Offsets (into hidden section, relative to visible_size)
    int hidden_active_player_offset
    int hidden_num_players_offset
    int hidden_deck_top_offset
    int hidden_deck_order_offset
    # Compact storage for frequently-accessed one-hot fields (performance optimization)
    int hidden_phase_offset
    int hidden_coo_level_offset
    int hidden_auction_company_offset
    int hidden_auction_high_bidder_offset
    int hidden_auction_starter_offset
    int hidden_corp_price_indices_offset  # 8 ints for corp price indices


cdef StateLayout compute_layout(int num_players) noexcept nogil:
    """Compute complete state layout for given player count."""
    cdef StateLayout layout
    cdef int offset = 0

    # Phase one-hot
    layout.phase_size = NUM_PHASES
    layout.phase_offset = offset
    offset += layout.phase_size

    # Cost of ownership one-hot
    layout.coo_size = NUM_COO_LEVELS
    layout.coo_offset = offset
    offset += layout.coo_size

    # Players
    layout.player_stride = (
        1 +                 # cash
        1 +                 # net_worth
        num_players +       # turn_order one-hot
        1 +                 # is_auction_high_bidder
        NUM_COMPANIES +     # owned_companies
        NUM_CORPS +         # owned_shares
        NUM_CORPS +         # is_president
        NUM_CORPS +         # share_buys
        NUM_CORPS           # share_sells
    )
    layout.players_offset = offset
    layout.players_size = layout.player_stride * num_players
    offset += layout.players_size

    # Foreign investor
    layout.fi_size = 1 + NUM_COMPANIES
    layout.fi_offset = offset
    offset += layout.fi_size

    # Company locations (3 arrays)
    layout.companies_size = NUM_COMPANIES * 3
    layout.auction_companies_offset = offset
    offset += NUM_COMPANIES
    layout.revealed_companies_offset = offset
    offset += NUM_COMPANIES
    layout.removed_companies_offset = offset
    offset += NUM_COMPANIES

    # Company adjusted incomes (dynamic based on CoO level)
    layout.company_incomes_size = NUM_COMPANIES
    layout.company_incomes_offset = offset
    offset += NUM_COMPANIES

    # Market availability
    layout.market_size = NUM_MARKET_SPACES
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
        NUM_MARKET_SPACES + # price_index one-hot
        NUM_COMPANIES +     # owned_companies
        NUM_COMPANIES       # acquisition_companies
    )
    layout.corps_offset = offset
    layout.corps_size = layout.corp_stride * NUM_CORPS
    offset += layout.corps_size

    # Turn state
    layout.turn_size = (
        1 +                 # turn_number
        1 +                 # end_card_flipped
        1 +                 # consecutive_passes
        # Auction
        NUM_COMPANIES +     # auction_company
        1 +                 # auction_price
        num_players +       # auction_high_bidder
        num_players +       # auction_starter
        num_players +       # auction_passed
        # Dividends
        NUM_CORPS +         # dividend_corp
        MAX_DIVIDEND +      # dividend_impact
        NUM_CORPS +         # dividend_remaining
        # Issue
        NUM_CORPS +         # issue_corp
        NUM_CORPS +         # issue_remaining
        # IPO
        NUM_COMPANIES +     # ipo_company
        NUM_COMPANIES +     # ipo_remaining
        # Acquisition offers
        NUM_CORPS +         # acq_active_corp
        NUM_COMPANIES +     # acq_target_company
        1                   # acq_is_fi_offer
    )
    layout.turn_offset = offset
    offset += layout.turn_size

    # Static company data
    layout.static_size = NUM_COMPANIES * (4 + NUM_COMPANIES)  # stars, low, face, high, synergies
    layout.static_offset = offset
    offset += layout.static_size

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
    layout.hidden_active_player_offset = 0
    layout.hidden_num_players_offset = 1
    layout.hidden_deck_top_offset = 2
    layout.hidden_deck_order_offset = 3
    layout.hidden_phase_offset = 3 + MAX_DECK_SIZE  # 39
    layout.hidden_coo_level_offset = 3 + MAX_DECK_SIZE + 1  # 40
    layout.hidden_auction_company_offset = 3 + MAX_DECK_SIZE + 2  # 41
    layout.hidden_auction_high_bidder_offset = 3 + MAX_DECK_SIZE + 3  # 42
    layout.hidden_auction_starter_offset = 3 + MAX_DECK_SIZE + 4  # 43
    layout.hidden_corp_price_indices_offset = 3 + MAX_DECK_SIZE + 5  # 44
    layout.hidden_size = 3 + MAX_DECK_SIZE + 5 + NUM_CORPS  # 52 total

    layout.total_size = layout.visible_size + layout.hidden_size

    return layout


# =============================================================================
# TURN STATE SUB-OFFSETS
# =============================================================================

cdef struct TurnStateOffsets:
    int turn_number
    int end_card_flipped
    int consecutive_passes
    # Auction
    int auction_company
    int auction_price
    int auction_high_bidder
    int auction_starter
    int auction_passed
    # Dividends
    int dividend_corp
    int dividend_impact
    int dividend_remaining
    # Issue
    int issue_corp
    int issue_remaining
    # IPO
    int ipo_company
    int ipo_remaining
    # Acquisition offers
    int acq_active_corp
    int acq_target_company
    int acq_is_fi_offer
    # Closing phase
    int closing_company


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
    t.auction_company = offset
    offset += NUM_COMPANIES
    t.auction_price = offset
    offset += 1
    t.auction_high_bidder = offset
    offset += num_players
    t.auction_starter = offset
    offset += num_players
    t.auction_passed = offset
    offset += num_players

    # Dividends
    t.dividend_corp = offset
    offset += NUM_CORPS
    t.dividend_impact = offset
    offset += MAX_DIVIDEND
    t.dividend_remaining = offset
    offset += NUM_CORPS

    # Issue
    t.issue_corp = offset
    offset += NUM_CORPS
    t.issue_remaining = offset
    offset += NUM_CORPS

    # IPO
    t.ipo_company = offset
    offset += NUM_COMPANIES
    t.ipo_remaining = offset
    offset += NUM_COMPANIES

    # Acquisition offers
    t.acq_active_corp = offset
    offset += NUM_CORPS
    t.acq_target_company = offset
    offset += NUM_COMPANIES
    t.acq_is_fi_offer = offset
    offset += 1

    # Closing phase
    t.closing_company = offset
    offset += NUM_COMPANIES

    return t


# =============================================================================
# PLAYER FIELD OFFSETS (within player stride)
# =============================================================================

cdef struct PlayerFieldOffsets:
    int cash
    int net_worth
    int turn_order
    int is_auction_high_bidder
    int owned_companies
    int owned_shares
    int is_president
    int share_buys
    int share_sells


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
    offset += NUM_COMPANIES
    p.owned_shares = offset
    offset += NUM_CORPS
    p.is_president = offset
    offset += NUM_CORPS
    p.share_buys = offset
    offset += NUM_CORPS
    p.share_sells = offset

    return p


# =============================================================================
# CORP FIELD OFFSETS (within corp stride)
# =============================================================================

cdef struct CorpFieldOffsets:
    int active
    int cash
    int unissued_shares
    int issued_shares
    int bank_shares
    int income
    int stars
    int share_price
    int acquisition_proceeds
    int in_receivership
    int price_index
    int owned_companies
    int acquisition_companies


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
    offset += NUM_MARKET_SPACES
    c.owned_companies = offset
    offset += NUM_COMPANIES
    c.acquisition_companies = offset

    return c


# =============================================================================
# GAMESTATE CLASS
# =============================================================================

cdef class GameState:
    """
    The game state as a contiguous float array.

    All game logic operates directly on self._data.
    Visible state can be passed to NN (with rotation).
    Hidden state contains deck and internal bookkeeping.

    Attributes are declared in state.pxd for cimport access.
    """

    def __cinit__(self, int num_players):
        if num_players < 2 or num_players > MAX_PLAYERS:
            raise ValueError(f"num_players must be 2-{MAX_PLAYERS}")

        self._num_players = num_players

        # Compute layout
        self._layout = compute_layout(num_players)
        self._turn = compute_turn_offsets(num_players)
        self._player_fields = compute_player_field_offsets(num_players)
        self._corp_fields = compute_corp_field_offsets()

        # Allocate array
        self._array = np.zeros(self._layout.total_size, dtype=np.float32)
        self._data = <float*>cnp.PyArray_DATA(self._array)

        # Initialize hidden state
        self._data[self._layout.visible_size + self._layout.hidden_num_players_offset] = <float>num_players

        # Initialize deck as empty
        self._set_deck_top(-1)
        cdef int i
        for i in range(MAX_DECK_SIZE):
            self._data[self._layout.visible_size + self._layout.hidden_deck_order_offset + i] = -1.0

        # Initialize one-hot visible fields and compact hidden storage together
        # Phase: initially unset (-1 = no phase, one-hot all zeros)
        self._data[self._layout.visible_size + self._layout.hidden_phase_offset] = -1.0

        # CoO level: set one-hot[0] = 1.0 (level 1) and compact = 1
        self._data[self._layout.coo_offset] = 1.0
        self._data[self._layout.visible_size + self._layout.hidden_coo_level_offset] = 1.0

        # Initialize company adjusted incomes for CoO level 1
        self.update_all_company_incomes()

        # Auction fields: set to -1 (no auction) in both visible and hidden
        # Note: one-hot uses -1.0 to indicate "not in auction mode"
        self._data[self._layout.visible_size + self._layout.hidden_auction_company_offset] = -1.0
        self._data[self._layout.visible_size + self._layout.hidden_auction_high_bidder_offset] = -1.0
        self._data[self._layout.visible_size + self._layout.hidden_auction_starter_offset] = -1.0

        # Corp price indices: set to -1 (no price card / inactive) in compact storage
        for i in range(NUM_CORPS):
            self._data[self._layout.visible_size + self._layout.hidden_corp_price_indices_offset + i] = -1.0

        # Initialize market spaces as all available
        for i in range(NUM_MARKET_SPACES):
            self._data[self._layout.market_offset + i] = 1.0

    def __init__(self, int num_players):
        pass  # All init in __cinit__

    # =========================================================================
    # HIDDEN STATE ACCESSORS
    # =========================================================================

    cdef inline int _get_active_player(self) noexcept nogil:
        return <int>self._data[self._layout.visible_size + self._layout.hidden_active_player_offset]

    cdef inline void _set_active_player(self, int player_id) noexcept nogil:
        self._data[self._layout.visible_size + self._layout.hidden_active_player_offset] = <float>player_id

    cdef inline int _get_deck_top(self) noexcept nogil:
        return <int>self._data[self._layout.visible_size + self._layout.hidden_deck_top_offset]

    cdef inline void _set_deck_top(self, int top) noexcept nogil:
        self._data[self._layout.visible_size + self._layout.hidden_deck_top_offset] = <float>top

    cdef inline int _get_deck_company(self, int index) noexcept nogil:
        return <int>self._data[self._layout.visible_size + self._layout.hidden_deck_order_offset + index]

    cdef inline void _set_deck_company(self, int index, int company_id) noexcept nogil:
        self._data[self._layout.visible_size + self._layout.hidden_deck_order_offset + index] = <float>company_id

    # =========================================================================
    # PHASE ACCESSORS
    # =========================================================================

    cdef inline int get_phase(self) noexcept nogil:
        """Get current phase index (reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_phase_offset]

    cdef inline void set_phase(self, int phase) noexcept nogil:
        """Set current phase (updates both one-hot visible and compact hidden)."""
        cdef int i
        cdef float* phase_ptr = self._data + self._layout.phase_offset
        # Update one-hot visible state
        for i in range(NUM_PHASES):
            phase_ptr[i] = 0.0
        if phase >= 0 and phase < NUM_PHASES:
            phase_ptr[phase] = 1.0
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_phase_offset] = <float>phase

    # =========================================================================
    # COST OF OWNERSHIP ACCESSORS
    # =========================================================================

    cdef inline int get_coo_level(self) noexcept nogil:
        """Get cost of ownership level (1-7, reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_coo_level_offset]

    cdef inline void set_coo_level(self, int level) noexcept nogil:
        """Set cost of ownership level (1-7, updates both one-hot visible and compact hidden)."""
        cdef int i
        cdef float* coo_ptr = self._data + self._layout.coo_offset
        # Update one-hot visible state
        for i in range(NUM_COO_LEVELS):
            coo_ptr[i] = 0.0
        if level >= 1 and level <= NUM_COO_LEVELS:
            coo_ptr[level - 1] = 1.0
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_coo_level_offset] = <float>level
        # Update all company adjusted incomes based on new CoO level
        self.update_all_company_incomes()

    # =========================================================================
    # POINTER ACCESSORS (for fast field access)
    # =========================================================================

    cdef inline float* _player_ptr(self, int player_id) noexcept nogil:
        """Get pointer to start of player's data block."""
        return self._data + self._layout.players_offset + player_id * self._layout.player_stride

    cdef inline float* _corp_ptr(self, int corp_id) noexcept nogil:
        """Get pointer to start of corp's data block."""
        return self._data + self._layout.corps_offset + corp_id * self._layout.corp_stride

    cdef inline float* _turn_ptr(self) noexcept nogil:
        """Get pointer to turn state section."""
        return self._data + self._layout.turn_offset

    cdef inline float* _hidden_price_indices_ptr(self) noexcept nogil:
        """Get pointer to hidden corp price indices storage."""
        return self._data + self._layout.visible_size + self._layout.hidden_corp_price_indices_offset

    # =========================================================================
    # TURN STATE ACCESSORS
    # =========================================================================

    cdef inline int get_turn_number(self) noexcept nogil:
        """Get current turn number."""
        cdef float* turn = self._turn_ptr()
        # Denormalize: stored as turn / 50 (games rarely exceed 50 turns)
        return <int>(turn[self._turn.turn_number] * 50.0 + 0.5)

    cdef inline void set_turn_number(self, int turn_num) noexcept nogil:
        """Set current turn number (normalized)."""
        cdef float* turn = self._turn_ptr()
        turn[self._turn.turn_number] = <float>turn_num / 50.0

    cdef inline int get_consecutive_passes(self) noexcept nogil:
        """Get consecutive passes count."""
        cdef float* turn = self._turn_ptr()
        return <int>(turn[self._turn.consecutive_passes] * self._num_players + 0.5)

    cdef inline void set_consecutive_passes(self, int count) noexcept nogil:
        """Set consecutive passes count (normalized by num_players)."""
        cdef float* turn = self._turn_ptr()
        turn[self._turn.consecutive_passes] = <float>count / <float>self._num_players

    cdef inline void increment_consecutive_passes(self) noexcept nogil:
        """Increment consecutive passes."""
        self.set_consecutive_passes(self.get_consecutive_passes() + 1)

    cdef inline void clear_consecutive_passes(self) noexcept nogil:
        """Reset consecutive passes to 0."""
        self.set_consecutive_passes(0)

    # =========================================================================
    # AUCTION STATE ACCESSORS
    # =========================================================================

    cdef inline int get_auction_company(self) noexcept nogil:
        """Get auction company ID, or -1 if no auction (reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_auction_company_offset]

    cdef inline void set_auction_company(self, int company_id) noexcept nogil:
        """Set auction company (updates both one-hot visible and compact hidden)."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.auction_company
        # Update one-hot visible state
        cdef float val = -1.0 if company_id < 0 else 0.0
        for i in range(NUM_COMPANIES):
            ptr[i] = val
        if company_id >= 0 and company_id < NUM_COMPANIES:
            ptr[company_id] = 1.0
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_auction_company_offset] = <float>company_id

    cdef inline int get_auction_price(self) noexcept nogil:
        """Get current auction price."""
        cdef float* turn = self._turn_ptr()
        cdef float val = turn[self._turn.auction_price]
        if val < 0:
            return -1
        return <int>(val * CASH_DIVISOR + 0.5)

    cdef inline void set_auction_price(self, int price) noexcept nogil:
        """Set auction price, or -1 to clear."""
        cdef float* turn = self._turn_ptr()
        if price < 0:
            turn[self._turn.auction_price] = -1.0
        else:
            turn[self._turn.auction_price] = <float>price / CASH_DIVISOR

    cdef inline int get_auction_high_bidder(self) noexcept nogil:
        """Get auction high bidder player ID, or -1 (reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_auction_high_bidder_offset]

    cdef inline void set_auction_high_bidder(self, int player_id) noexcept nogil:
        """Set auction high bidder (updates both one-hot visible and compact hidden)."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.auction_high_bidder
        # Update one-hot visible state
        cdef float val = -1.0 if player_id < 0 else 0.0
        for i in range(self._num_players):
            ptr[i] = val
        if player_id >= 0 and player_id < self._num_players:
            ptr[player_id] = 1.0
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_auction_high_bidder_offset] = <float>player_id

    cdef inline int get_auction_starter(self) noexcept nogil:
        """Get auction starter player ID, or -1 (reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_auction_starter_offset]

    cdef inline void set_auction_starter(self, int player_id) noexcept nogil:
        """Set auction starter (updates both one-hot visible and compact hidden)."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.auction_starter
        # Update one-hot visible state
        cdef float val = -1.0 if player_id < 0 else 0.0
        for i in range(self._num_players):
            ptr[i] = val
        if player_id >= 0 and player_id < self._num_players:
            ptr[player_id] = 1.0
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_auction_starter_offset] = <float>player_id

    cdef inline bint get_auction_passed(self, int player_id) noexcept nogil:
        """Check if player has left the auction."""
        cdef float* ptr = self._turn_ptr() + self._turn.auction_passed
        return ptr[player_id] == 1.0

    cdef inline void set_auction_passed(self, int player_id, bint passed) noexcept nogil:
        """Set whether player has left auction."""
        cdef float* ptr = self._turn_ptr() + self._turn.auction_passed
        ptr[player_id] = 1.0 if passed else 0.0

    cdef inline void clear_auction_state(self) noexcept nogil:
        """Clear all auction state (set to -1)."""
        cdef int i
        self.set_auction_company(-1)
        self.set_auction_price(-1)
        self.set_auction_high_bidder(-1)
        self.set_auction_starter(-1)
        cdef float* ptr = self._turn_ptr() + self._turn.auction_passed
        for i in range(self._num_players):
            ptr[i] = -1.0

    cdef inline void init_auction_passed(self) noexcept nogil:
        """Initialize auction passed flags to 0 (no one has left yet)."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.auction_passed
        for i in range(self._num_players):
            ptr[i] = 0.0

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    cdef inline void copy_from(self, GameState other) noexcept nogil:
        """Fast copy of entire state (visible + hidden)."""
        memcpy(self._data, other._data, self._layout.total_size * sizeof(float))

    cdef inline bint is_game_over(self) noexcept nogil:
        """Check if game is over."""
        return self.get_phase() == PHASE_GAME_OVER

    cdef inline void advance_active_player(self) noexcept nogil:
        """Move to next player in turn order."""
        cdef int current = self._get_active_player()
        self._set_active_player((current + 1) % self._num_players)

    # =========================================================================
    # PLAYER TURN ORDER AND CASH
    # =========================================================================

    cdef inline int get_player_turn_order(self, int player_id) noexcept nogil:
        """Get player's position in turn order (0 = first)."""
        cdef float* player = self._player_ptr(player_id)
        cdef int i
        for i in range(self._num_players):
            if player[self._player_fields.turn_order + i] == 1.0:
                return i
        return -1

    cdef inline void set_player_turn_order(self, int player_id, int position) noexcept nogil:
        """Set player's position in turn order (one-hot)."""
        cdef float* player = self._player_ptr(player_id)
        cdef int i
        for i in range(self._num_players):
            player[self._player_fields.turn_order + i] = 0.0
        if position >= 0 and position < self._num_players:
            player[self._player_fields.turn_order + position] = 1.0

    cdef inline int get_player_cash(self, int player_id) noexcept nogil:
        """Get player's cash."""
        cdef float* player = self._player_ptr(player_id)
        return <int>(player[self._player_fields.cash] * CASH_DIVISOR + 0.5)

    cdef inline void set_player_cash(self, int player_id, int cash) noexcept nogil:
        """Set player's cash."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.cash] = <float>cash / CASH_DIVISOR

    # =========================================================================
    # FOREIGN INVESTOR
    # =========================================================================

    cdef inline int get_fi_cash(self) noexcept nogil:
        """Get foreign investor's cash."""
        return <int>(self._data[self._layout.fi_offset] * CASH_DIVISOR + 0.5)

    cdef inline void set_fi_cash(self, int cash) noexcept nogil:
        """Set foreign investor's cash."""
        self._data[self._layout.fi_offset] = <float>cash / CASH_DIVISOR

    cdef inline void add_fi_cash(self, int amount) noexcept nogil:
        """Add to foreign investor's cash."""
        cdef int current = self.get_fi_cash()
        self.set_fi_cash(current + amount)

    cdef inline bint fi_owns_company(self, int company_id) noexcept nogil:
        """Check if FI owns a company."""
        return self._data[self._layout.fi_offset + 1 + company_id] == 1.0

    cdef inline void set_fi_owns_company(self, int company_id, bint owns) noexcept nogil:
        """Set whether FI owns a company."""
        self._data[self._layout.fi_offset + 1 + company_id] = 1.0 if owns else 0.0

    # =========================================================================
    # COMPANY LOCATIONS
    # =========================================================================

    cdef inline bint is_company_for_auction(self, int company_id) noexcept nogil:
        """Check if company is available for auction."""
        return self._data[self._layout.auction_companies_offset + company_id] == 1.0

    cdef inline void set_company_for_auction(self, int company_id, bint available) noexcept nogil:
        """Set whether company is available for auction."""
        self._data[self._layout.auction_companies_offset + company_id] = 1.0 if available else 0.0

    cdef inline bint is_company_revealed(self, int company_id) noexcept nogil:
        """Check if company is revealed (drawn this turn, unavailable)."""
        return self._data[self._layout.revealed_companies_offset + company_id] == 1.0

    cdef inline void set_company_revealed(self, int company_id, bint revealed) noexcept nogil:
        """Set whether company is revealed."""
        self._data[self._layout.revealed_companies_offset + company_id] = 1.0 if revealed else 0.0

    cdef inline void draw_company_to_revealed(self) noexcept nogil:
        """Draw top company from deck to revealed pile."""
        cdef int deck_top = self._get_deck_top()
        if deck_top < 0:
            return  # Deck empty

        cdef int company_id = self._get_deck_company(deck_top)
        if company_id < 0:
            return  # Invalid

        # Add to revealed
        self.set_company_revealed(company_id, True)

        # Update deck top
        self._set_deck_top(deck_top - 1)

    cdef inline void move_revealed_to_auction(self) noexcept nogil:
        """Move all revealed companies to available for auction."""
        cdef int i
        for i in range(NUM_COMPANIES):
            if self.is_company_revealed(i):
                self.set_company_for_auction(i, True)
                self.set_company_revealed(i, False)

    # =========================================================================
    # PLAYER COMPANY OWNERSHIP
    # =========================================================================

    cdef inline bint player_owns_company(self, int player_id, int company_id) noexcept nogil:
        """Check if player owns a company."""
        cdef float* player = self._player_ptr(player_id)
        return player[self._player_fields.owned_companies + company_id] == 1.0

    cdef inline void set_player_owns_company(self, int player_id, int company_id, bint owns) noexcept nogil:
        """Set whether player owns a company."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.owned_companies + company_id] = 1.0 if owns else 0.0

    cdef inline bint is_player_president(self, int player_id, int corp_id) noexcept nogil:
        """Check if player is president of a corp."""
        cdef float* player = self._player_ptr(player_id)
        return player[self._player_fields.is_president + corp_id] == 1.0

    cdef inline void set_player_president(self, int player_id, int corp_id, bint is_pres) noexcept nogil:
        """Set whether player is president of a corp."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.is_president + corp_id] = 1.0 if is_pres else 0.0

    cdef inline int get_player_shares(self, int player_id, int corp_id) noexcept nogil:
        """Get number of shares player owns in a corp."""
        cdef float* player = self._player_ptr(player_id)
        return <int>(player[self._player_fields.owned_shares + corp_id] * SHARE_DIVISOR + 0.5)

    cdef inline void add_player_cash(self, int player_id, int amount) noexcept nogil:
        """Add to player's cash."""
        cdef int current = self.get_player_cash(player_id)
        self.set_player_cash(player_id, current + amount)

    # =========================================================================
    # CORPORATION ACCESSORS
    # =========================================================================

    cdef inline bint is_corp_active(self, int corp_id) noexcept nogil:
        """Check if corp is active (has been IPO'd)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.active] == 1.0

    cdef inline void set_corp_active(self, int corp_id, bint active) noexcept nogil:
        """Set whether corp is active."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.active] = 1.0 if active else 0.0

    cdef inline int get_corp_cash(self, int corp_id) noexcept nogil:
        """Get corp's cash."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.cash] * CASH_DIVISOR + 0.5)

    cdef inline void set_corp_cash(self, int corp_id, int cash) noexcept nogil:
        """Set corp's cash."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.cash] = <float>cash / CASH_DIVISOR

    cdef inline void add_corp_cash(self, int corp_id, int amount) noexcept nogil:
        """Add to corp's cash."""
        cdef int current = self.get_corp_cash(corp_id)
        self.set_corp_cash(corp_id, current + amount)

    cdef inline bint is_corp_in_receivership(self, int corp_id) noexcept nogil:
        """Check if corp is in receivership."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.in_receivership] == 1.0

    cdef inline void set_corp_in_receivership(self, int corp_id, bint in_recv) noexcept nogil:
        """Set whether corp is in receivership."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.in_receivership] = 1.0 if in_recv else 0.0

    cdef inline int get_corp_price_index(self, int corp_id) noexcept nogil:
        """Get corp's market price index (reads from compact hidden storage)."""
        return <int>self._data[self._layout.visible_size + self._layout.hidden_corp_price_indices_offset + corp_id]

    cdef inline void set_corp_price_index(self, int corp_id, int index) noexcept nogil:
        """Set corp's market price index (updates one-hot visible, compact hidden, and share_price)."""
        cdef float* corp = self._corp_ptr(corp_id)
        cdef int i
        # Update one-hot visible state
        for i in range(NUM_MARKET_SPACES):
            corp[self._corp_fields.price_index + i] = 0.0
        if index >= 0 and index < NUM_MARKET_SPACES:
            corp[self._corp_fields.price_index + index] = 1.0
            # Update share_price field
            corp[self._corp_fields.share_price] = <float>MARKET_PRICES[index] / CASH_DIVISOR
        else:
            # No price card means 75
            corp[self._corp_fields.share_price] = 75.0 / CASH_DIVISOR
        # Update compact hidden storage
        self._data[self._layout.visible_size + self._layout.hidden_corp_price_indices_offset + corp_id] = <float>index

    cdef inline int get_corp_share_price(self, int corp_id) noexcept nogil:
        """Get corp's share price (reads from visible share_price field)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.share_price] * CASH_DIVISOR + 0.5)

    cdef inline bint corp_owns_company(self, int corp_id, int company_id) noexcept nogil:
        """Check if corp owns a company."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.owned_companies + company_id] == 1.0

    cdef inline void set_corp_owns_company(self, int corp_id, int company_id, bint owns) noexcept nogil:
        """Set whether corp owns a company."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.owned_companies + company_id] = 1.0 if owns else 0.0

    cdef inline int get_corp_company_count(self, int corp_id) noexcept nogil:
        """Count how many companies corp owns."""
        cdef float* corp = self._corp_ptr(corp_id)
        cdef int count = 0
        cdef int i
        for i in range(NUM_COMPANIES):
            if corp[self._corp_fields.owned_companies + i] == 1.0:
                count += 1
        return count

    cdef inline bint corp_has_acquisition_company(self, int corp_id, int company_id) noexcept nogil:
        """Check if corp has company in acquisition pile (pending)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return corp[self._corp_fields.acquisition_companies + company_id] == 1.0

    cdef inline void set_corp_acquisition_company(self, int corp_id, int company_id, bint has) noexcept nogil:
        """Set whether corp has company in acquisition pile."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.acquisition_companies + company_id] = 1.0 if has else 0.0

    cdef inline int get_corp_acquisition_proceeds(self, int corp_id) noexcept nogil:
        """Get corp's pending acquisition proceeds."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.acquisition_proceeds] * CASH_DIVISOR + 0.5)

    cdef inline void set_corp_acquisition_proceeds(self, int corp_id, int amount) noexcept nogil:
        """Set corp's pending acquisition proceeds."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.acquisition_proceeds] = <float>amount / CASH_DIVISOR

    cdef inline void add_corp_acquisition_proceeds(self, int corp_id, int amount) noexcept nogil:
        """Add to corp's pending acquisition proceeds."""
        cdef int current = self.get_corp_acquisition_proceeds(corp_id)
        self.set_corp_acquisition_proceeds(corp_id, current + amount)

    # =========================================================================
    # ACQUISITION OFFER STATE
    # =========================================================================

    cdef inline int get_acq_active_corp(self) noexcept nogil:
        """Get active buying corp in acquisition offer, or -1."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.acq_active_corp
        for i in range(NUM_CORPS):
            if ptr[i] == 1.0:
                return i
        return -1

    cdef inline void set_acq_active_corp(self, int corp_id) noexcept nogil:
        """Set active buying corp (one-hot), or -1 to clear."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.acq_active_corp
        cdef float val = -1.0 if corp_id < 0 else 0.0
        for i in range(NUM_CORPS):
            ptr[i] = val
        if corp_id >= 0 and corp_id < NUM_CORPS:
            ptr[corp_id] = 1.0

    cdef inline int get_acq_target_company(self) noexcept nogil:
        """Get target company in acquisition offer, or -1."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.acq_target_company
        for i in range(NUM_COMPANIES):
            if ptr[i] == 1.0:
                return i
        return -1

    cdef inline void set_acq_target_company(self, int company_id) noexcept nogil:
        """Set target company (one-hot), or -1 to clear."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.acq_target_company
        cdef float val = -1.0 if company_id < 0 else 0.0
        for i in range(NUM_COMPANIES):
            ptr[i] = val
        if company_id >= 0 and company_id < NUM_COMPANIES:
            ptr[company_id] = 1.0

    cdef inline bint is_acq_fi_offer(self) noexcept nogil:
        """Check if current acquisition offer is for FI company."""
        cdef float* ptr = self._turn_ptr() + self._turn.acq_is_fi_offer
        return ptr[0] == 1.0

    cdef inline void set_acq_is_fi_offer(self, bint is_fi) noexcept nogil:
        """Set whether current offer is for FI company."""
        cdef float* ptr = self._turn_ptr() + self._turn.acq_is_fi_offer
        ptr[0] = 1.0 if is_fi else 0.0

    cdef inline void clear_acq_offer(self) noexcept nogil:
        """Clear acquisition offer state."""
        self.set_acq_active_corp(-1)
        self.set_acq_target_company(-1)
        cdef float* ptr = self._turn_ptr() + self._turn.acq_is_fi_offer
        ptr[0] = -1.0

    cdef inline void finalize_acquisitions(self) noexcept nogil:
        """Move acquisition companies to owned and proceeds to cash for all corps."""
        cdef int corp_id, company_id
        cdef float* corp
        for corp_id in range(NUM_CORPS):
            if not self.is_corp_active(corp_id):
                continue
            corp = self._corp_ptr(corp_id)
            # Move acquisition proceeds to cash
            self.add_corp_cash(corp_id, self.get_corp_acquisition_proceeds(corp_id))
            self.set_corp_acquisition_proceeds(corp_id, 0)
            # Move acquisition companies to owned
            for company_id in range(NUM_COMPANIES):
                if self.corp_has_acquisition_company(corp_id, company_id):
                    self.set_corp_owns_company(corp_id, company_id, True)
                    self.set_corp_acquisition_company(corp_id, company_id, False)

    # =========================================================================
    # CLOSING PHASE OFFER STATE
    # =========================================================================

    cdef inline int get_current_closing_company(self) noexcept nogil:
        """Get current company offered for closing, or -1 if none."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.closing_company
        for i in range(NUM_COMPANIES):
            if ptr[i] == 1.0:
                return i
        return -1

    cdef inline void set_current_closing_company(self, int company_id) noexcept nogil:
        """Set current company for closing offer (one-hot), or -1 to clear."""
        cdef int i
        cdef float* ptr = self._turn_ptr() + self._turn.closing_company
        cdef float val = -1.0 if company_id < 0 else 0.0
        for i in range(NUM_COMPANIES):
            ptr[i] = val
        if company_id >= 0 and company_id < NUM_COMPANIES:
            ptr[company_id] = 1.0

    cdef inline void clear_closing_company(self) noexcept nogil:
        """Clear closing company state."""
        self.set_current_closing_company(-1)

    # =========================================================================
    # CORPORATION SHARE ACCESSORS
    # =========================================================================

    cdef inline int get_corp_issued_shares(self, int corp_id) noexcept nogil:
        """Get corp's issued shares count."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.issued_shares] * SHARE_DIVISOR + 0.5)

    cdef inline void set_corp_issued_shares(self, int corp_id, int shares) noexcept nogil:
        """Set corp's issued shares count."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.issued_shares] = <float>shares / SHARE_DIVISOR

    cdef inline int get_corp_bank_shares(self, int corp_id) noexcept nogil:
        """Get corp's bank shares (issued but not player-owned)."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.bank_shares] * SHARE_DIVISOR + 0.5)

    cdef inline void set_corp_bank_shares(self, int corp_id, int shares) noexcept nogil:
        """Set corp's bank shares count."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.bank_shares] = <float>shares / SHARE_DIVISOR

    cdef inline int get_corp_unissued_shares(self, int corp_id) noexcept nogil:
        """Get corp's unissued shares count."""
        cdef float* corp = self._corp_ptr(corp_id)
        return <int>(corp[self._corp_fields.unissued_shares] * SHARE_DIVISOR + 0.5)

    cdef inline void set_corp_unissued_shares(self, int corp_id, int shares) noexcept nogil:
        """Set corp's unissued shares count."""
        cdef float* corp = self._corp_ptr(corp_id)
        corp[self._corp_fields.unissued_shares] = <float>shares / SHARE_DIVISOR

    # =========================================================================
    # REMOVED COMPANIES (closed/out of game)
    # =========================================================================

    cdef inline bint is_company_removed(self, int company_id) noexcept nogil:
        """Check if company has been removed from game."""
        return self._data[self._layout.removed_companies_offset + company_id] == 1.0

    cdef inline void set_company_removed(self, int company_id, bint removed) noexcept nogil:
        """Set whether company has been removed from game."""
        self._data[self._layout.removed_companies_offset + company_id] = 1.0 if removed else 0.0

    # =========================================================================
    # COMPANY ADJUSTED INCOMES
    # =========================================================================

    cdef inline int get_company_adjusted_income(self, int company_id) noexcept nogil:
        """Get company's adjusted income (base income minus CoO cost)."""
        return <int>(self._data[self._layout.company_incomes_offset + company_id] * INCOME_DIVISOR + 0.5)

    cdef inline void update_all_company_incomes(self) noexcept nogil:
        """Update all company incomes based on current CoO level."""
        cdef int company_id
        cdef int coo_level = self.get_coo_level()
        cdef int adjusted_income
        for company_id in range(NUM_COMPANIES):
            adjusted_income = get_adjusted_company_income(company_id, coo_level)
            self._data[self._layout.company_incomes_offset + company_id] = <float>adjusted_income / INCOME_DIVISOR

    # =========================================================================
    # MARKET SPACE AVAILABILITY
    # =========================================================================

    cdef inline bint is_market_space_available(self, int index) noexcept nogil:
        """Check if market space is available (not taken by a corp)."""
        return self._data[self._layout.market_offset + index] == 1.0

    cdef inline void set_market_space_available(self, int index, bint available) noexcept nogil:
        """Set whether market space is available."""
        self._data[self._layout.market_offset + index] = 1.0 if available else 0.0

    # =========================================================================
    # CORPORATION BANKRUPTCY
    # =========================================================================

    cdef inline void bankrupt_corp(self, int corp_id) noexcept nogil:
        """Handle corporation bankruptcy - reset corp state."""
        cdef float* corp = self._corp_ptr(corp_id)
        cdef int i, price_index

        # Get current price index to free market space
        price_index = self.get_corp_price_index(corp_id)
        if price_index >= 0:
            self.set_market_space_available(price_index, True)

        # Reset corp state
        self.set_corp_active(corp_id, False)
        self.set_corp_cash(corp_id, 0)
        self.set_corp_in_receivership(corp_id, False)
        self.set_corp_issued_shares(corp_id, 0)
        self.set_corp_bank_shares(corp_id, 0)
        # Note: unissued_shares will need to be reset based on corp share count
        # For now, set to 0 - the caller should set properly
        self.set_corp_unissued_shares(corp_id, 0)
        self.set_corp_acquisition_proceeds(corp_id, 0)

        # Clear price index (no market card)
        for i in range(NUM_MARKET_SPACES):
            corp[self._corp_fields.price_index + i] = 0.0

        # Clear share price
        corp[self._corp_fields.share_price] = 0.0

        # Remove all owned companies
        for i in range(NUM_COMPANIES):
            if self.corp_owns_company(corp_id, i):
                self.set_corp_owns_company(corp_id, i, False)
                self.set_company_removed(i, True)

        # Clear acquisition companies
        for i in range(NUM_COMPANIES):
            self.set_corp_acquisition_company(corp_id, i, False)

        # Clear player presidencies for this corp
        for i in range(self._num_players):
            self.set_player_president(i, corp_id, False)

    # =========================================================================
    # END CARD STATE
    # =========================================================================

    cdef inline bint get_end_card_flipped(self) noexcept nogil:
        """Check if end card has been flipped."""
        cdef float* turn = self._turn_ptr()
        return turn[self._turn.end_card_flipped] == 1.0

    cdef inline void set_end_card_flipped(self, bint flipped) noexcept nogil:
        """Set end card flipped state."""
        cdef float* turn = self._turn_ptr()
        turn[self._turn.end_card_flipped] = 1.0 if flipped else 0.0

    # =========================================================================
    # PLAYER NET WORTH
    # =========================================================================

    cdef inline int get_player_net_worth(self, int player_id) noexcept nogil:
        """Get player's stored net worth."""
        cdef float* player = self._player_ptr(player_id)
        return <int>(player[self._player_fields.net_worth] * CASH_DIVISOR + 0.5)

    cdef inline void set_player_net_worth(self, int player_id, int net_worth) noexcept nogil:
        """Set player's net worth."""
        cdef float* player = self._player_ptr(player_id)
        player[self._player_fields.net_worth] = <float>net_worth / CASH_DIVISOR

    # =========================================================================
    # NN INPUT GENERATION
    # =========================================================================

    def get_nn_input(self):
        """
        Get state formatted for neural network input.

        Returns a numpy array with:
        1. Only visible state (hidden truncated)
        2. Player data rotated so active player is first
        3. Player-relative fields (auction_high_bidder, etc.) rotated
        """
        cdef int active = self._get_active_player()

        # If active player is 0, no rotation needed
        if active == 0:
            return self._array[:self._layout.visible_size].copy()

        # Need to rotate - create output array
        cdef cnp.ndarray result = np.empty(self._layout.visible_size, dtype=np.float32)
        cdef float* out = <float*>cnp.PyArray_DATA(result)

        self._rotate_for_nn(out, active)
        return result

    cdef void _rotate_for_nn(self, float* out, int active_player) noexcept nogil:
        """Rotate state so active_player becomes player 0."""
        cdef int i, src_player, dst_player
        cdef int offset

        # Copy non-player data unchanged (phase, coo)
        memcpy(out, self._data, (self._layout.phase_size + self._layout.coo_size) * sizeof(float))
        offset = self._layout.phase_size + self._layout.coo_size

        # Rotate player data
        for i in range(self._num_players):
            src_player = (active_player + i) % self._num_players
            memcpy(
                out + self._layout.players_offset + i * self._layout.player_stride,
                self._data + self._layout.players_offset + src_player * self._layout.player_stride,
                self._layout.player_stride * sizeof(float)
            )

        # Copy FI, companies, company incomes, market, corps unchanged
        cdef int post_players_start = self._layout.fi_offset
        cdef int post_players_size = (
            self._layout.fi_size +
            self._layout.companies_size +
            self._layout.company_incomes_size +
            self._layout.market_size +
            self._layout.corps_size
        )
        memcpy(
            out + post_players_start,
            self._data + post_players_start,
            post_players_size * sizeof(float)
        )

        # Copy turn state, but rotate player-indexed fields
        self._rotate_turn_state(out, active_player)

        # Copy static data unchanged
        memcpy(
            out + self._layout.static_offset,
            self._data + self._layout.static_offset,
            self._layout.static_size * sizeof(float)
        )

    cdef void _rotate_turn_state(self, float* out, int active_player) noexcept nogil:
        """Copy turn state with player indices rotated."""
        cdef float* src = self._data + self._layout.turn_offset
        cdef float* dst = out + self._layout.turn_offset
        cdef int i, src_idx

        # Copy scalar fields unchanged
        dst[self._turn.turn_number] = src[self._turn.turn_number]
        dst[self._turn.end_card_flipped] = src[self._turn.end_card_flipped]
        dst[self._turn.consecutive_passes] = src[self._turn.consecutive_passes]

        # Auction company (not player-indexed)
        memcpy(
            dst + self._turn.auction_company,
            src + self._turn.auction_company,
            NUM_COMPANIES * sizeof(float)
        )

        # Auction price
        dst[self._turn.auction_price] = src[self._turn.auction_price]

        # Rotate player-indexed auction fields
        for i in range(self._num_players):
            src_idx = (active_player + i) % self._num_players
            dst[self._turn.auction_high_bidder + i] = src[self._turn.auction_high_bidder + src_idx]
            dst[self._turn.auction_starter + i] = src[self._turn.auction_starter + src_idx]
            dst[self._turn.auction_passed + i] = src[self._turn.auction_passed + src_idx]

        # Copy remaining turn state (not player-indexed)
        cdef int remaining_start = self._turn.auction_passed + self._num_players
        cdef int remaining_size = self._layout.turn_size - remaining_start
        memcpy(
            dst + remaining_start,
            src + remaining_start,
            remaining_size * sizeof(float)
        )

    # =========================================================================
    # PYTHON PROPERTIES
    # =========================================================================

    @property
    def num_players(self):
        return self._num_players

    @property
    def size(self):
        return self._layout.total_size

    @property
    def visible_size(self):
        return self._layout.visible_size

    @property
    def active_player(self):
        return self._get_active_player()

    @active_player.setter
    def active_player(self, int value):
        self._set_active_player(value)

    @property
    def phase(self):
        return self.get_phase()

    @phase.setter
    def phase(self, int value):
        self.set_phase(value)

    @property
    def coo_level(self):
        return self.get_coo_level()

    @coo_level.setter
    def coo_level(self, int value):
        self.set_coo_level(value)

    @property
    def turn_number(self):
        return self.get_turn_number()

    @turn_number.setter
    def turn_number(self, int value):
        self.set_turn_number(value)

    @property
    def consecutive_passes(self):
        return self.get_consecutive_passes()

    @consecutive_passes.setter
    def consecutive_passes(self, int value):
        self.set_consecutive_passes(value)

    def as_numpy(self):
        """Return full state as numpy array (includes hidden)."""
        return self._array

    def as_tensor(self):
        """Return full state as PyTorch tensor (includes hidden)."""
        import torch
        return torch.from_numpy(self._array)

    def clone(self):
        """Create a deep copy of this state."""
        cdef GameState new_state = GameState(self._num_players)
        new_state.copy_from(self)
        return new_state

    def is_terminal(self):
        return self.is_game_over()

    # Acquisition state Python accessors
    def get_acq_active_corp_py(self):
        """Python accessor for active buying corp."""
        return self.get_acq_active_corp()

    def get_acq_target_company_py(self):
        """Python accessor for target company."""
        return self.get_acq_target_company()

    def is_acq_fi_offer_py(self):
        """Python accessor for FI offer flag."""
        return self.is_acq_fi_offer()

    # Player turn order Python accessor
    def set_player_turn_order_py(self, int player_id, int position):
        """Python accessor to set player turn order."""
        self.set_player_turn_order(player_id, position)

    # Corporation share Python accessors
    def get_corp_issued_shares_py(self, int corp_id):
        """Python accessor for corp issued shares."""
        return self.get_corp_issued_shares(corp_id)

    def set_corp_issued_shares_py(self, int corp_id, int shares):
        """Python accessor to set corp issued shares."""
        self.set_corp_issued_shares(corp_id, shares)

    def get_corp_bank_shares_py(self, int corp_id):
        """Python accessor for corp bank shares."""
        return self.get_corp_bank_shares(corp_id)

    def set_corp_bank_shares_py(self, int corp_id, int shares):
        """Python accessor to set corp bank shares."""
        self.set_corp_bank_shares(corp_id, shares)

    def get_corp_unissued_shares_py(self, int corp_id):
        """Python accessor for corp unissued shares."""
        return self.get_corp_unissued_shares(corp_id)

    def set_corp_unissued_shares_py(self, int corp_id, int shares):
        """Python accessor to set corp unissued shares."""
        self.set_corp_unissued_shares(corp_id, shares)

    def get_corp_price_index_py(self, int corp_id):
        """Python accessor for corp price index."""
        return self.get_corp_price_index(corp_id)

    def set_corp_price_index_py(self, int corp_id, int index):
        """Python accessor to set corp price index (updates both one-hot and compact storage)."""
        self.set_corp_price_index(corp_id, index)

    # Removed companies Python accessors
    def is_company_removed_py(self, int company_id):
        """Python accessor to check if company is removed."""
        return self.is_company_removed(company_id)

    def set_company_removed_py(self, int company_id, bint removed):
        """Python accessor to set company removed flag."""
        self.set_company_removed(company_id, removed)

    # Company income Python accessors
    def get_company_adjusted_income_py(self, int company_id):
        """Python accessor to get company's adjusted income."""
        return self.get_company_adjusted_income(company_id)

    # Market space Python accessors
    def is_market_space_available_py(self, int index):
        """Python accessor to check if market space is available."""
        return self.is_market_space_available(index)

    def set_market_space_available_py(self, int index, bint available):
        """Python accessor to set market space availability."""
        self.set_market_space_available(index, available)

    # Bankruptcy Python accessor
    def bankrupt_corp_py(self, int corp_id):
        """Python accessor for corp bankruptcy."""
        self.bankrupt_corp(corp_id)

    # End card state Python accessors
    def get_end_card_flipped_py(self):
        """Python accessor to check if end card is flipped."""
        return self.get_end_card_flipped()

    def set_end_card_flipped_py(self, bint flipped):
        """Python accessor to set end card flipped state."""
        self.set_end_card_flipped(flipped)

    # Player net worth Python accessors
    def get_player_net_worth_py(self, int player_id):
        """Python accessor for player net worth."""
        return self.get_player_net_worth(player_id)

    def set_player_net_worth_py(self, int player_id, int net_worth):
        """Python accessor to set player net worth."""
        self.set_player_net_worth(player_id, net_worth)

    # =========================================================================
    # INVARIANT CHECKING PYTHON ACCESSORS
    # =========================================================================

    # Player accessors
    def get_player_cash_py(self, int player_id):
        """Python accessor for player cash."""
        return self.get_player_cash(player_id)

    def get_player_shares_py(self, int player_id, int corp_id):
        """Python accessor for player shares in a corp."""
        return self.get_player_shares(player_id, corp_id)

    def is_player_president_py(self, int player_id, int corp_id):
        """Python accessor to check if player is president of a corp."""
        return self.is_player_president(player_id, corp_id)

    # Corp accessors
    def get_corp_cash_py(self, int corp_id):
        """Python accessor for corp cash."""
        return self.get_corp_cash(corp_id)

    def is_corp_active_py(self, int corp_id):
        """Python accessor to check if corp is active."""
        return self.is_corp_active(corp_id)

    def is_corp_in_receivership_py(self, int corp_id):
        """Python accessor to check if corp is in receivership."""
        return self.is_corp_in_receivership(corp_id)

    def get_corp_acquisition_proceeds_py(self, int corp_id):
        """Python accessor for corp acquisition proceeds."""
        return self.get_corp_acquisition_proceeds(corp_id)

    # Company location accessors
    def is_company_for_auction_py(self, int company_id):
        """Python accessor to check if company is available for auction."""
        return self.is_company_for_auction(company_id)

    def is_company_revealed_py(self, int company_id):
        """Python accessor to check if company is revealed."""
        return self.is_company_revealed(company_id)

    def player_owns_company_py(self, int player_id, int company_id):
        """Python accessor to check if player owns a company."""
        return self.player_owns_company(player_id, company_id)

    def corp_owns_company_py(self, int corp_id, int company_id):
        """Python accessor to check if corp owns a company."""
        return self.corp_owns_company(corp_id, company_id)

    def corp_has_acquisition_company_py(self, int corp_id, int company_id):
        """Python accessor to check if corp has a company in its acquisition pile."""
        return self.corp_has_acquisition_company(corp_id, company_id)

    def fi_owns_company_py(self, int company_id):
        """Python accessor to check if FI owns a company."""
        return self.fi_owns_company(company_id)

    def is_company_in_deck_py(self, int company_id):
        """Python accessor to check if a company is still in the deck (not yet drawn)."""
        cdef int deck_top = self._get_deck_top()
        cdef int i
        if deck_top < 0:
            return False  # Deck is empty
        for i in range(deck_top + 1):
            if self._get_deck_company(i) == company_id:
                return True
        return False

    # FI accessors
    def get_fi_cash_py(self):
        """Python accessor for FI cash."""
        return self.get_fi_cash()

    # Auction state accessors
    def get_auction_company_py(self):
        """Python accessor for current auction company."""
        return self.get_auction_company()

    def get_auction_price_py(self):
        """Python accessor for current auction price."""
        return self.get_auction_price()

    def get_auction_high_bidder_py(self):
        """Python accessor for current auction high bidder."""
        return self.get_auction_high_bidder()

    def get_auction_starter_py(self):
        """Python accessor for auction starter."""
        return self.get_auction_starter()

    def setup_new_game(self, shuffle_seed=None):
        """
        Set up a new game with proper initial state.

        This initializes:
        - Phase to INVEST
        - Player cash to 30
        - Player turn order (0, 1, 2, ...)
        - Shuffled company deck
        - Initial companies in auction

        Args:
            shuffle_seed: Optional random seed for deck shuffling
        """
        import random
        from data import py_get_company_stars

        if shuffle_seed is not None:
            random.seed(shuffle_seed)

        # Set phase to INVEST
        self.set_phase(PHASE_INVEST)

        # Initialize players
        for player_id in range(self._num_players):
            self.set_player_cash(player_id, 30)
            self.set_player_turn_order(player_id, player_id)
            self.set_player_net_worth(player_id, 30)

        # Build deck using same logic as Python Game.make_deck
        # Groups: [0-5 reds, 6-13 oranges, 14-21 yellows, 22-29 greens, 30-35 blues]
        group_ranges = [
            (0, 6),    # Reds (1 star) - companies 0-5, last is 5
            (6, 14),   # Oranges (2 stars) - companies 6-13, last is 13
            (14, 22),  # Yellows (3 stars) - companies 14-21, last is 21
            (22, 30),  # Greens (4 stars) - companies 22-29, last is 29
            (30, 36),  # Blues (5 stars) - companies 30-35, last is 35
        ]

        deck = []
        excluded = []
        for start, end in group_ranges:
            # Last company in each group is always the last in range - 1
            last_company = end - 1

            # Get non-last companies
            non_last = list(range(start, end - 1))
            random.shuffle(non_last)

            # Take up to num_players non-last companies
            selected = non_last[:self._num_players]

            # Track excluded companies (those not selected for the deck)
            excluded.extend(non_last[self._num_players:])

            # Add the last company
            selected.append(last_company)

            # Shuffle the group
            random.shuffle(selected)

            deck.extend(selected)

        # Mark excluded companies as removed
        for company_id in excluded:
            self.set_company_removed(company_id, True)

        # Reverse so we pop from the end (deck[0] is bottom)
        deck.reverse()

        # Store deck in hidden state
        self._set_deck_top(len(deck) - 1)
        for i, company_id in enumerate(deck):
            self._set_deck_company(i, company_id)

        # Draw initial companies to auction
        for _ in range(self._num_players):
            self.draw_company_to_revealed()
        self.move_revealed_to_auction()

        # Initialize FI cash
        self.set_fi_cash(0)

    def get_final_scores(self):
        """
        Get player IDs and net worths for game-over scoring.

        Returns list of (player_id, net_worth) tuples sorted by net worth descending.
        Ties broken by player order (lower player order wins).
        """
        cdef list scores = []
        cdef int player_id
        for player_id in range(self._num_players):
            scores.append((
                player_id,
                self.get_player_net_worth(player_id),
                self.get_player_turn_order(player_id)
            ))
        # Sort by net worth descending, then by turn order ascending for ties
        scores.sort(key=lambda x: (-x[1], x[2]))
        return [(s[0], s[1]) for s in scores]


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

def get_state_size(int num_players):
    """Get total state size for given player count."""
    cdef StateLayout layout = compute_layout(num_players)
    return layout.total_size

def get_visible_size(int num_players):
    """Get visible state size (for NN input) for given player count."""
    cdef StateLayout layout = compute_layout(num_players)
    return layout.visible_size

def get_market_price(int index):
    """Get share price at market index."""
    if index < 0 or index >= NUM_MARKET_SPACES:
        return -1
    return MARKET_PRICES[index]

def get_market_index(int price):
    """Get market index for a share price."""
    cdef int i
    for i in range(NUM_MARKET_SPACES):
        if MARKET_PRICES[i] == price:
            return i
    return -1

# Phase constants for Python access
PHASE_NAMES = [
    "INVEST", "BID_IN_AUCTION", "WRAP_UP", "ACQUISITION",
    "CLOSING", "INCOME", "DIVIDENDS", "END_CARD",
    "ISSUE_SHARES", "IPO", "GAME_OVER"
]
