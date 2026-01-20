# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Turn state entity implementation.

Provides clean getter/setter access to turn-specific state including:
- Phase and cost-of-ownership level
- Turn number and end card status
- Auction, dividends, issue, IPO, acquisition, and closing phase tracking
"""

from state cimport GameState, StateLayout, TurnStateOffsets
from data cimport GameConstants, GamePhases, CASH_DIVISOR


cdef class TurnState:
    """
    Entity handle for accessing turn state.

    There is only one TurnState instance, created at module load.
    Offsets are computed on first access to a GameState via initialize().
    All methods take GameState as first argument for stateless operation.
    """

    def __cinit__(self):
        self._num_players = 0
        self._phase_offset = 0
        self._coo_offset = 0
        self._turn_offset = 0
        self._hidden_phase_offset = 0
        self._hidden_coo_level_offset = 0
        self._hidden_auction_company_offset = 0
        self._hidden_auction_high_bidder_offset = 0
        self._hidden_auction_starter_offset = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this TurnState instance.
        """
        cdef StateLayout layout = state._layout
        cdef TurnStateOffsets turn = state._turn_offsets

        self._num_players = state._num_players

        # Phase & CoO are at the start of the state vector (one-hot encoded)
        self._phase_offset = layout.phase_offset
        self._coo_offset = layout.coo_offset

        # Hidden state offsets (compact integer storage for fast access)
        # These are already absolute offsets (computed continuing from visible_size)
        self._hidden_phase_offset = layout.hidden_phase_offset
        self._hidden_coo_level_offset = layout.hidden_coo_level_offset
        self._hidden_auction_company_offset = layout.hidden_auction_company_offset
        self._hidden_auction_high_bidder_offset = layout.hidden_auction_high_bidder_offset
        self._hidden_auction_starter_offset = layout.hidden_auction_starter_offset

        # Turn state base offset
        self._turn_offset = layout.turn_offset

        # Cache absolute offsets for turn state fields
        self._turn_number_offset = self._turn_offset + turn.turn_number
        self._end_card_flipped_offset = self._turn_offset + turn.end_card_flipped
        self._consecutive_passes_offset = self._turn_offset + turn.consecutive_passes

        # Auction
        self._auction_company_offset = self._turn_offset + turn.auction_company
        self._auction_price_offset = self._turn_offset + turn.auction_price
        self._auction_high_bidder_offset = self._turn_offset + turn.auction_high_bidder
        self._auction_starter_offset = self._turn_offset + turn.auction_starter
        self._auction_passed_offset = self._turn_offset + turn.auction_passed

        # Dividends
        self._dividend_corp_offset = self._turn_offset + turn.dividend_corp
        self._dividend_impact_offset = self._turn_offset + turn.dividend_impact
        self._dividend_remaining_offset = self._turn_offset + turn.dividend_remaining

        # Issue
        self._issue_corp_offset = self._turn_offset + turn.issue_corp
        self._issue_remaining_offset = self._turn_offset + turn.issue_remaining

        # IPO
        self._ipo_company_offset = self._turn_offset + turn.ipo_company
        self._ipo_remaining_offset = self._turn_offset + turn.ipo_remaining

        # Acquisition
        self._acq_active_corp_offset = self._turn_offset + turn.acq_active_corp
        self._acq_target_company_offset = self._turn_offset + turn.acq_target_company
        self._acq_is_fi_offer_offset = self._turn_offset + turn.acq_is_fi_offer

        # Closing
        self._closing_company_offset = self._turn_offset + turn.closing_company

    # =========================================================================
    # PHASE (one-hot, 11 values: 0-10)
    # =========================================================================

    cpdef int get_phase(self, GameState state):
        """Get current game phase (0-10). Uses hidden compact storage for O(1) access."""
        return <int>state._data[self._hidden_phase_offset]

    cpdef void set_phase(self, GameState state, int phase):
        """Set current game phase. Updates both one-hot and hidden compact storage."""
        cdef int i
        # Clear one-hot encoding
        for i in range(GameConstants.NUM_PHASES):
            state._data[self._phase_offset + i] = 0.0
        # Set one-hot and hidden compact value
        if phase >= 0 and phase < GameConstants.NUM_PHASES:
            state._data[self._phase_offset + phase] = 1.0
            state._data[self._hidden_phase_offset] = <float>phase

    # =========================================================================
    # COST OF OWNERSHIP LEVEL (one-hot, 7 values: 1-7 in game terms)
    # =========================================================================

    cpdef int get_coo_level(self, GameState state):
        """Get cost of ownership level (1-7 in game terms). Uses hidden compact storage for O(1) access."""
        return <int>state._data[self._hidden_coo_level_offset]

    cpdef void set_coo_level(self, GameState state, int level):
        """Set cost of ownership level (1-7 in game terms). Updates both one-hot and hidden compact storage."""
        cdef int i
        # Clear one-hot encoding
        for i in range(GameConstants.NUM_COO_LEVELS):
            state._data[self._coo_offset + i] = 0.0
        # Set one-hot and hidden compact value
        if level >= 1 and level <= GameConstants.NUM_COO_LEVELS:
            state._data[self._coo_offset + level - 1] = 1.0
            state._data[self._hidden_coo_level_offset] = <float>level

    # =========================================================================
    # TURN NUMBER
    # =========================================================================

    cpdef int get_turn_number(self, GameState state):
        """Get current turn number."""
        return <int>(state._data[self._turn_number_offset] * 50.0 + 0.5)

    cpdef void set_turn_number(self, GameState state, int turn):
        """Set current turn number."""
        state._data[self._turn_number_offset] = <float>turn / 50.0

    # =========================================================================
    # END CARD FLIPPED
    # =========================================================================

    cpdef bint is_end_card_flipped(self, GameState state):
        """Check if the end card has been flipped."""
        return state._data[self._end_card_flipped_offset] == 1.0

    cpdef void set_end_card_flipped(self, GameState state, bint flipped):
        """Set whether the end card has been flipped."""
        state._data[self._end_card_flipped_offset] = 1.0 if flipped else 0.0

    # =========================================================================
    # CONSECUTIVE PASSES (INVEST phase)
    # =========================================================================

    cpdef int get_consecutive_passes(self, GameState state):
        """Get number of consecutive passes in INVEST phase."""
        return <int>(state._data[self._consecutive_passes_offset] * self._num_players + 0.5)

    cpdef void set_consecutive_passes(self, GameState state, int passes):
        """Set number of consecutive passes."""
        state._data[self._consecutive_passes_offset] = <float>passes / self._num_players

    cpdef void increment_consecutive_passes(self, GameState state):
        """Increment consecutive pass counter."""
        cdef int current = self.get_consecutive_passes(state)
        self.set_consecutive_passes(state, current + 1)

    cpdef void clear_consecutive_passes(self, GameState state):
        """Clear consecutive pass counter (called when any non-pass action taken)."""
        state._data[self._consecutive_passes_offset] = 0.0

    # =========================================================================
    # AUCTION STATE
    # =========================================================================

    cpdef int get_auction_company(self, GameState state):
        """Get company being auctioned. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_auction_company_offset]

    cpdef void set_auction_company(self, GameState state, int company_id):
        """Set company being auctioned. Updates both one-hot and hidden compact storage."""
        cdef int i
        # Clear one-hot encoding
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._auction_company_offset + i] = 0.0
        # Set one-hot and hidden compact value
        if company_id >= 0 and company_id < GameConstants.NUM_COMPANIES:
            state._data[self._auction_company_offset + company_id] = 1.0
        state._data[self._hidden_auction_company_offset] = <float>company_id

    cpdef void clear_auction_company(self, GameState state):
        """Clear auction company (no active auction). Updates both one-hot and hidden storage."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._auction_company_offset + i] = 0.0
        state._data[self._hidden_auction_company_offset] = -1.0

    cpdef int get_auction_price(self, GameState state):
        """Get current auction price."""
        cdef float val = state._data[self._auction_price_offset]
        if val < 0:
            return -1
        return <int>(val * CASH_DIVISOR + 0.5)

    cpdef void set_auction_price(self, GameState state, int price):
        """Set current auction price."""
        if price < 0:
            state._data[self._auction_price_offset] = -1.0
        else:
            state._data[self._auction_price_offset] = <float>price / CASH_DIVISOR

    cpdef int get_auction_high_bidder(self, GameState state):
        """Get current high bidder. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_auction_high_bidder_offset]

    cpdef void set_auction_high_bidder(self, GameState state, int player_id):
        """Set current high bidder. Updates both one-hot and hidden compact storage."""
        cdef int i
        # Clear one-hot encoding
        for i in range(self._num_players):
            state._data[self._auction_high_bidder_offset + i] = 0.0
        # Set one-hot and hidden compact value
        if player_id >= 0 and player_id < self._num_players:
            state._data[self._auction_high_bidder_offset + player_id] = 1.0
        state._data[self._hidden_auction_high_bidder_offset] = <float>player_id

    cpdef void clear_auction_high_bidder(self, GameState state):
        """Clear high bidder. Updates both one-hot and hidden storage."""
        cdef int i
        for i in range(self._num_players):
            state._data[self._auction_high_bidder_offset + i] = 0.0
        state._data[self._hidden_auction_high_bidder_offset] = -1.0

    cpdef int get_auction_starter(self, GameState state):
        """Get auction starter. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_auction_starter_offset]

    cpdef void set_auction_starter(self, GameState state, int player_id):
        """Set auction starter. Updates both one-hot and hidden compact storage."""
        cdef int i
        # Clear one-hot encoding
        for i in range(self._num_players):
            state._data[self._auction_starter_offset + i] = 0.0
        # Set one-hot and hidden compact value
        if player_id >= 0 and player_id < self._num_players:
            state._data[self._auction_starter_offset + player_id] = 1.0
        state._data[self._hidden_auction_starter_offset] = <float>player_id

    cpdef void clear_auction_starter(self, GameState state):
        """Clear auction starter. Updates both one-hot and hidden storage."""
        cdef int i
        for i in range(self._num_players):
            state._data[self._auction_starter_offset + i] = 0.0
        state._data[self._hidden_auction_starter_offset] = -1.0

    cpdef bint has_player_passed_auction(self, GameState state, int player_id):
        """Check if player has passed (left) the auction."""
        if player_id < 0 or player_id >= self._num_players:
            return False
        return state._data[self._auction_passed_offset + player_id] == 1.0

    cpdef void set_player_passed_auction(self, GameState state, int player_id, bint passed):
        """Set whether player has passed the auction."""
        if player_id >= 0 and player_id < self._num_players:
            state._data[self._auction_passed_offset + player_id] = 1.0 if passed else 0.0

    cpdef void clear_auction_passed(self, GameState state):
        """Clear all auction passed flags."""
        cdef int i
        for i in range(self._num_players):
            state._data[self._auction_passed_offset + i] = 0.0

    # =========================================================================
    # DIVIDENDS STATE
    # =========================================================================

    cpdef int get_dividend_corp(self, GameState state):
        """Get current dividend corp (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            if state._data[self._dividend_corp_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_dividend_corp(self, GameState state, int corp_id):
        """Set current dividend corp (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._dividend_corp_offset + i] = 0.0
        if corp_id >= 0 and corp_id < GameConstants.NUM_CORPS:
            state._data[self._dividend_corp_offset + corp_id] = 1.0

    cpdef void clear_dividend_corp(self, GameState state):
        """Clear dividend corp."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._dividend_corp_offset + i] = 0.0

    cpdef int get_dividend_impact(self, GameState state, int level):
        """Get dividend impact at given level (0-25)."""
        if level < 0 or level >= GameConstants.MAX_DIVIDEND:
            return 0
        return <int>(state._data[self._dividend_impact_offset + level] + 0.5)

    cpdef void set_dividend_impact(self, GameState state, int level, int impact):
        """Set dividend impact at given level."""
        if level >= 0 and level < GameConstants.MAX_DIVIDEND:
            state._data[self._dividend_impact_offset + level] = <float>impact

    cpdef bint is_dividend_remaining(self, GameState state, int corp_id):
        """Check if corp still needs dividend processing."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return False
        return state._data[self._dividend_remaining_offset + corp_id] == 1.0

    cpdef void set_dividend_remaining(self, GameState state, int corp_id, bint remaining):
        """Set whether corp needs dividend processing."""
        if corp_id >= 0 and corp_id < GameConstants.NUM_CORPS:
            state._data[self._dividend_remaining_offset + corp_id] = 1.0 if remaining else 0.0

    # =========================================================================
    # ISSUE STATE
    # =========================================================================

    cpdef int get_issue_corp(self, GameState state):
        """Get current issue corp (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            if state._data[self._issue_corp_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_issue_corp(self, GameState state, int corp_id):
        """Set current issue corp (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._issue_corp_offset + i] = 0.0
        if corp_id >= 0 and corp_id < GameConstants.NUM_CORPS:
            state._data[self._issue_corp_offset + corp_id] = 1.0

    cpdef void clear_issue_corp(self, GameState state):
        """Clear issue corp."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._issue_corp_offset + i] = 0.0

    cpdef bint is_issue_remaining(self, GameState state, int corp_id):
        """Check if corp still needs issue processing."""
        if corp_id < 0 or corp_id >= GameConstants.NUM_CORPS:
            return False
        return state._data[self._issue_remaining_offset + corp_id] == 1.0

    cpdef void set_issue_remaining(self, GameState state, int corp_id, bint remaining):
        """Set whether corp needs issue processing."""
        if corp_id >= 0 and corp_id < GameConstants.NUM_CORPS:
            state._data[self._issue_remaining_offset + corp_id] = 1.0 if remaining else 0.0

    # =========================================================================
    # IPO STATE
    # =========================================================================

    cpdef int get_ipo_company(self, GameState state):
        """Get current IPO company (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            if state._data[self._ipo_company_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_ipo_company(self, GameState state, int company_id):
        """Set current IPO company (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._ipo_company_offset + i] = 0.0
        if company_id >= 0 and company_id < GameConstants.NUM_COMPANIES:
            state._data[self._ipo_company_offset + company_id] = 1.0

    cpdef void clear_ipo_company(self, GameState state):
        """Clear IPO company."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._ipo_company_offset + i] = 0.0

    cpdef bint is_ipo_remaining(self, GameState state, int company_id):
        """Check if company still needs IPO processing."""
        if company_id < 0 or company_id >= GameConstants.NUM_COMPANIES:
            return False
        return state._data[self._ipo_remaining_offset + company_id] == 1.0

    cpdef void set_ipo_remaining(self, GameState state, int company_id, bint remaining):
        """Set whether company needs IPO processing."""
        if company_id >= 0 and company_id < GameConstants.NUM_COMPANIES:
            state._data[self._ipo_remaining_offset + company_id] = 1.0 if remaining else 0.0

    # =========================================================================
    # ACQUISITION STATE
    # =========================================================================

    cpdef int get_acq_active_corp(self, GameState state):
        """Get active acquiring corp (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            if state._data[self._acq_active_corp_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_acq_active_corp(self, GameState state, int corp_id):
        """Set active acquiring corp (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._acq_active_corp_offset + i] = 0.0
        if corp_id >= 0 and corp_id < GameConstants.NUM_CORPS:
            state._data[self._acq_active_corp_offset + corp_id] = 1.0

    cpdef void clear_acq_active_corp(self, GameState state):
        """Clear active acquiring corp."""
        cdef int i
        for i in range(GameConstants.NUM_CORPS):
            state._data[self._acq_active_corp_offset + i] = 0.0

    cpdef int get_acq_target_company(self, GameState state):
        """Get target company for acquisition (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            if state._data[self._acq_target_company_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_acq_target_company(self, GameState state, int company_id):
        """Set target company for acquisition (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._acq_target_company_offset + i] = 0.0
        if company_id >= 0 and company_id < GameConstants.NUM_COMPANIES:
            state._data[self._acq_target_company_offset + company_id] = 1.0

    cpdef void clear_acq_target_company(self, GameState state):
        """Clear target company."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._acq_target_company_offset + i] = 0.0

    cpdef bint is_acq_fi_offer(self, GameState state):
        """Check if current acquisition is from Foreign Investor."""
        return state._data[self._acq_is_fi_offer_offset] == 1.0

    cpdef void set_acq_fi_offer(self, GameState state, bint is_fi):
        """Set whether current acquisition is from Foreign Investor."""
        state._data[self._acq_is_fi_offer_offset] = 1.0 if is_fi else 0.0

    # =========================================================================
    # CLOSING STATE
    # =========================================================================

    cpdef int get_closing_company(self, GameState state):
        """Get company being offered for closing (one-hot, returns -1 if none)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            if state._data[self._closing_company_offset + i] == 1.0:
                return i
        return -1

    cpdef void set_closing_company(self, GameState state, int company_id):
        """Set company being offered for closing (one-hot encoded)."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._closing_company_offset + i] = 0.0
        if company_id >= 0 and company_id < GameConstants.NUM_COMPANIES:
            state._data[self._closing_company_offset + company_id] = 1.0

    cpdef void clear_closing_company(self, GameState state):
        """Clear closing company."""
        cdef int i
        for i in range(GameConstants.NUM_COMPANIES):
            state._data[self._closing_company_offset + i] = 0.0


# =============================================================================
# GLOBAL TURN STATE INSTANCE
# =============================================================================

# Single TurnState instance
TURN = TurnState()
