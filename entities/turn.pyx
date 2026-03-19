# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Turn state entity implementation.

Provides clean getter/setter access to turn-specific state including:
- Phase and cost-of-ownership level
- Turn number and end card status
- Auction, dividends, issue, IPO, acquisition, and closing phase tracking
"""

from core.state cimport GameState, StateLayout, TurnStateOffsets
from libc.math cimport lround
from core.data cimport GameConstants, GamePhases, CASH_DIVISOR, IMPACT_DIVISOR, get_adjusted_company_income, COMPANY_SYNERGY
from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
from entities.encoding cimport (
    set_one_hot, clear_one_hot,
    set_one_hot_with_compact, clear_one_hot_with_compact,
)

# Use constants from GameConstants (imported above)


# =============================================================================
# LOW-LEVEL NOGIL ACCESSORS
# =============================================================================

cdef struct TurnOffsets:
    # Offsets within turn state data block in the state vector
    int acq_is_fi_offer
    int active_corp


cdef TurnOffsets get_turn_offsets(int num_players) noexcept nogil:
    """
    Compute field offsets within turn data block.

    The turn state is stored as a contiguous float array with the following layout:
    - turn_number (1)
    - end_card_flipped (1)
    - consecutive_passes (1)
    - auction_price (1)
    - auction_high_bidder (num_players)
    - auction_starter (num_players)
    - auction_passed (num_players)
    - dividend_impact (26)
    - dividend_remaining (8)
    - issue_remaining (8)
    - ipo_remaining (36)
    - acq_is_fi_offer (1)
    - acq_synergy_values (36)
    - active_company (36)
    - active_company_info (5)
    - active_corp (8)
    - active_corp_info (3)
    - active_corp_companies (36)
    """
    cdef TurnOffsets t
    cdef int offset = 0

    # Skip turn_number (1), end_card_flipped (1), consecutive_passes (1)
    offset += 3

    # Skip auction_price (1)
    offset += 1
    # Skip auction_high_bidder (num_players), auction_starter (num_players), auction_passed (num_players)
    offset += num_players * 3

    # Skip dividend_impact (26)
    offset += GameConstants.MAX_DIVIDEND
    # Skip dividend_remaining (8)
    offset += GameConstants.NUM_CORPS

    # Skip issue_remaining (8)
    offset += GameConstants.NUM_CORPS

    # Skip ipo_remaining (36)
    offset += GameConstants.NUM_COMPANIES

    t.acq_is_fi_offer = offset
    offset += 1
    # Skip acq_synergy_values (36)
    offset += GameConstants.NUM_COMPANIES
    # Skip active_company (36)
    offset += GameConstants.NUM_COMPANIES
    # Skip active_company_info (5)
    offset += 5

    t.active_corp = offset

    return t


cdef inline int get_active_corp_nogil(float* turn, TurnOffsets* t) noexcept nogil:
    """Get active corp (scan one-hot, returns -1 if none)."""
    cdef int i
    for i in range(<int>GameConstants.NUM_CORPS):
        if turn[t.active_corp + i] == 1.0:
            return i
    return -1


cdef inline bint is_acq_fi_offer_nogil(float* turn, TurnOffsets* t) noexcept nogil:
    """Check if current acquisition is from Foreign Investor."""
    return turn[t.acq_is_fi_offer] == 1.0


# =============================================================================
# HIGH-LEVEL ENTITY CLASS
# =============================================================================

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
        # Hidden offsets for O(1) nogil access
        self._hidden_acq_active_corp_offset = 0
        self._hidden_acq_target_company_offset = 0
        self._hidden_acq_is_fi_offer_offset = 0
        self._hidden_dividend_corp_offset = 0
        self._hidden_issue_corp_offset = 0
        self._hidden_ipo_company_offset = 0
        self._hidden_closing_company_offset = 0

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

        # Hidden state offsets (compact integer storage for fast O(1) access)
        # These are already absolute offsets (computed continuing from visible_size)
        self._hidden_phase_offset = layout.hidden_phase_offset
        self._hidden_coo_level_offset = layout.hidden_coo_level_offset
        self._hidden_auction_company_offset = layout.hidden_auction_company_offset
        self._hidden_auction_high_bidder_offset = layout.hidden_auction_high_bidder_offset
        self._hidden_auction_starter_offset = layout.hidden_auction_starter_offset
        self._hidden_acq_active_corp_offset = layout.hidden_acq_active_corp_offset
        self._hidden_acq_target_company_offset = layout.hidden_acq_target_company_offset
        self._hidden_dividend_corp_offset = layout.hidden_dividend_corp_offset
        self._hidden_issue_corp_offset = layout.hidden_issue_corp_offset
        self._hidden_ipo_company_offset = layout.hidden_ipo_company_offset
        self._hidden_closing_company_offset = layout.hidden_closing_company_offset

        # Turn state base offset
        self._turn_offset = layout.turn_offset

        # Cache absolute offsets for turn state fields
        self._turn_number_offset = self._turn_offset + turn.turn_number
        self._end_card_flipped_offset = self._turn_offset + turn.end_card_flipped
        self._consecutive_passes_offset = self._turn_offset + turn.consecutive_passes

        # Auction
        self._auction_price_offset = self._turn_offset + turn.auction_price
        self._auction_high_bidder_offset = self._turn_offset + turn.auction_high_bidder
        self._auction_starter_offset = self._turn_offset + turn.auction_starter
        self._auction_passed_offset = self._turn_offset + turn.auction_passed

        # Dividends
        self._dividend_impact_offset = self._turn_offset + turn.dividend_impact
        self._dividend_remaining_offset = self._turn_offset + turn.dividend_remaining

        # Issue
        self._issue_remaining_offset = self._turn_offset + turn.issue_remaining

        # IPO
        self._ipo_remaining_offset = self._turn_offset + turn.ipo_remaining

        # Acquisition
        self._acq_is_fi_offer_offset = self._turn_offset + turn.acq_is_fi_offer
        # Note: acq_is_fi_offer is a single float (not one-hot), so use visible offset directly
        self._hidden_acq_is_fi_offer_offset = self._acq_is_fi_offer_offset
        self._acq_synergy_values_offset = self._turn_offset + turn.acq_synergy_values

        # Active company one-hot (shared by auction/acq/closing/ipo)
        self._active_company_offset = self._turn_offset + turn.active_company

        # Active corp one-hot (shared by dividend/issue/acq)
        self._active_corp_offset = self._turn_offset + turn.active_corp

    # =========================================================================
    # PHASE (one-hot, 11 values: 0-10)
    # =========================================================================

    cpdef int get_phase(self, GameState state):
        """Get current game phase (0-10). Uses hidden compact storage for O(1) access."""
        return <int>state._data[self._hidden_phase_offset]

    cpdef void set_phase(self, GameState state, int phase):
        """Set current game phase. Updates both one-hot and hidden compact storage."""
        set_one_hot_with_compact(
            state._data, self._phase_offset, GameConstants.NUM_PHASES,
            self._hidden_phase_offset, phase
        )

    # =========================================================================
    # COST OF OWNERSHIP LEVEL (one-hot, 7 values: 1-7 in game terms)
    # =========================================================================

    cpdef int get_coo_level(self, GameState state):
        """Get cost of ownership level (1-7 in game terms). Uses hidden compact storage for O(1) access."""
        return <int>state._data[self._hidden_coo_level_offset]

    cpdef void set_coo_level(self, GameState state, int level):
        """
        Set cost of ownership level (1-7 in game terms).

        Updates both one-hot and hidden compact storage, and recalculates
        all company adjusted incomes based on the new CoO level.

        Note: This method doesn't use set_one_hot_with_compact because coo_level
        uses 1-indexed game values (1-7) while the one-hot is 0-indexed (0-6).
        The hidden compact stores the 1-indexed value for direct use in game logic.
        """
        set_one_hot(state._data, self._coo_offset, GameConstants.NUM_COO_LEVELS, level - 1)
        if 1 <= level <= GameConstants.NUM_COO_LEVELS:
            state._data[self._hidden_coo_level_offset] = <float>level
            # Update all company adjusted incomes for the new CoO level
            self._update_all_company_incomes(state, level)
            # Recalculate all incomes (they depend on adjusted company incomes)
            self._update_all_corp_incomes(state)
            self._update_all_player_incomes(state)
            fi_module.FI.calculate_income(state)
            # Update auction slot info (income values depend on CoO)
            state._populate_auction_slot_info()

    cdef void _update_all_company_incomes(self, GameState state, int coo_level):
        """
        Update all 36 company adjusted incomes based on current CoO level.

        This is called automatically when CoO level changes, ensuring the
        company_incomes state array always reflects the current CoO.
        """
        cdef int company_id, adjusted_income
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            adjusted_income = get_adjusted_company_income(company_id, coo_level)
            company_module.COMPANIES[company_id].set_adjusted_income(state, adjusted_income)

    cdef void _update_all_corp_incomes(self, GameState state):
        """Recalculate income for all active corporations."""
        cdef int corp_id
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            if corp_module.CORPS[corp_id].is_active(state):
                corp_module.CORPS[corp_id].calculate_income(state)

    cdef void _update_all_player_incomes(self, GameState state):
        """Recalculate income for all players."""
        cdef int player_id
        for player_id in range(self._num_players):
            player_module.PLAYERS[player_id].calculate_income(state)

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
        """Set company being auctioned. Updates active_company one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_auction_company_offset, company_id
        )

    cpdef void clear_auction_company(self, GameState state):
        """Clear auction company. Updates active_company one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_auction_company_offset
        )

    cpdef int get_auction_price(self, GameState state):
        """Get current auction price. Returns 0 when no auction is active."""
        cdef float val = state._data[self._auction_price_offset]
        if val <= 0:
            return 0
        return <int>(val * CASH_DIVISOR + 0.5)

    cpdef void set_auction_price(self, GameState state, int price):
        """Set current auction price. Use 0 or negative to clear."""
        if price <= 0:
            state._data[self._auction_price_offset] = 0.0
        else:
            state._data[self._auction_price_offset] = <float>price / CASH_DIVISOR

    cpdef int get_auction_high_bidder(self, GameState state):
        """Get current high bidder. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_auction_high_bidder_offset]

    cpdef void set_auction_high_bidder(self, GameState state, int player_id):
        """Set current high bidder. Updates both one-hot and hidden compact storage."""
        set_one_hot_with_compact(
            state._data, self._auction_high_bidder_offset, self._num_players,
            self._hidden_auction_high_bidder_offset, player_id
        )

    cpdef void clear_auction_high_bidder(self, GameState state):
        """Clear high bidder. Updates both one-hot and hidden storage."""
        clear_one_hot_with_compact(
            state._data, self._auction_high_bidder_offset, self._num_players,
            self._hidden_auction_high_bidder_offset
        )

    cpdef int get_auction_starter(self, GameState state):
        """Get auction starter. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_auction_starter_offset]

    cpdef void set_auction_starter(self, GameState state, int player_id):
        """Set auction starter. Updates both one-hot and hidden compact storage."""
        set_one_hot_with_compact(
            state._data, self._auction_starter_offset, self._num_players,
            self._hidden_auction_starter_offset, player_id
        )

    cpdef void clear_auction_starter(self, GameState state):
        """Clear auction starter. Updates both one-hot and hidden storage."""
        clear_one_hot_with_compact(
            state._data, self._auction_starter_offset, self._num_players,
            self._hidden_auction_starter_offset
        )

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
        """Get current dividend corp. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_dividend_corp_offset]

    cpdef void set_dividend_corp(self, GameState state, int corp_id):
        """Set current dividend corp. Updates active_corp one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_dividend_corp_offset, corp_id
        )

    cpdef void clear_dividend_corp(self, GameState state):
        """Clear dividend corp. Updates active_corp one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_dividend_corp_offset
        )

    cpdef int get_dividend_impact(self, GameState state, int level):
        """Get dividend impact at given level (0-25). Returns denormalized index delta."""
        if level < 0 or level >= GameConstants.MAX_DIVIDEND:
            return 0
        return <int>lround(state._data[self._dividend_impact_offset + level] * IMPACT_DIVISOR)

    cpdef void set_dividend_impact(self, GameState state, int level, int impact):
        """Set dividend impact at given level. Normalizes by IMPACT_DIVISOR for NN."""
        if level >= 0 and level < GameConstants.MAX_DIVIDEND:
            state._data[self._dividend_impact_offset + level] = <float>impact / IMPACT_DIVISOR

    cpdef void clear_dividend_impacts(self, GameState state):
        """Zero all 26 dividend impact slots."""
        cdef int i
        for i in range(<int>GameConstants.MAX_DIVIDEND):
            state._data[self._dividend_impact_offset + i] = 0.0

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
        """Get current issue corp. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_issue_corp_offset]

    cpdef void set_issue_corp(self, GameState state, int corp_id):
        """Set current issue corp. Updates active_corp one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_issue_corp_offset, corp_id
        )

    cpdef void clear_issue_corp(self, GameState state):
        """Clear issue corp. Updates active_corp one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_issue_corp_offset
        )

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
        """Get current IPO company. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_ipo_company_offset]

    cpdef void set_ipo_company(self, GameState state, int company_id):
        """Set current IPO company. Updates active_company one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_ipo_company_offset, company_id
        )

    cpdef void clear_ipo_company(self, GameState state):
        """Clear IPO company. Updates active_company one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_ipo_company_offset
        )

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
    # NOGIL ACCESSORS (for mask generation in actions.pyx)
    # Uses hidden state offsets for O(1) access instead of scanning one-hot arrays.
    # =========================================================================

    cdef inline int _get_acq_active_corp_nogil(self, float* data) noexcept nogil:
        """Get active acquiring corp from hidden state. O(1) access."""
        return <int>data[self._hidden_acq_active_corp_offset]

    cdef inline int _get_acq_target_company_nogil(self, float* data) noexcept nogil:
        """Get target company for acquisition from hidden state. O(1) access."""
        return <int>data[self._hidden_acq_target_company_offset]

    cdef inline bint _is_acq_fi_offer_nogil(self, float* data) noexcept nogil:
        """Check if current acquisition is from Foreign Investor. O(1) access."""
        return data[self._hidden_acq_is_fi_offer_offset] == 1.0

    cdef inline int _get_dividend_corp_nogil(self, float* data) noexcept nogil:
        """Get current dividend corp from hidden state. O(1) access."""
        return <int>data[self._hidden_dividend_corp_offset]

    cdef inline int _get_issue_corp_nogil(self, float* data) noexcept nogil:
        """Get current issue corp from hidden state. O(1) access."""
        return <int>data[self._hidden_issue_corp_offset]

    cdef inline int _get_ipo_company_nogil(self, float* data) noexcept nogil:
        """Get current IPO company from hidden state. O(1) access."""
        return <int>data[self._hidden_ipo_company_offset]

    cdef inline int _get_closing_company_nogil(self, float* data) noexcept nogil:
        """Get company being offered for closing from hidden state. O(1) access."""
        return <int>data[self._hidden_closing_company_offset]

    cdef inline int _get_auction_price_nogil(self, float* data) noexcept nogil:
        """Get current auction price from cached offset. Returns 0 when no auction."""
        cdef float val = data[self._auction_price_offset]
        if val <= 0:
            return 0
        return <int>(val * CASH_DIVISOR + 0.5)

    cdef inline int _get_coo_level_nogil(self, float* data) noexcept nogil:
        """Get cost of ownership level from hidden state. O(1) access."""
        return <int>data[self._hidden_coo_level_offset]

    # =========================================================================
    # ACQUISITION STATE
    # =========================================================================

    cpdef int get_acq_active_corp(self, GameState state):
        """Get active acquiring corp. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_acq_active_corp_offset]

    cpdef void set_acq_active_corp(self, GameState state, int corp_id):
        """Set active acquiring corp. Updates active_corp one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_acq_active_corp_offset, corp_id
        )

    cpdef void clear_acq_active_corp(self, GameState state):
        """Clear active acquiring corp. Updates active_corp one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS,
            self._hidden_acq_active_corp_offset
        )

    cpdef int get_acq_target_company(self, GameState state):
        """Get target company for acquisition. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_acq_target_company_offset]

    cpdef void set_acq_target_company(self, GameState state, int company_id):
        """Set target company. Updates active_company one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_acq_target_company_offset, company_id
        )

    cpdef void clear_acq_target_company(self, GameState state):
        """Clear target company. Updates active_company one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_acq_target_company_offset
        )

    cpdef bint is_acq_fi_offer(self, GameState state):
        """Check if current acquisition is from Foreign Investor."""
        return state._data[self._acq_is_fi_offer_offset] == 1.0

    cpdef void set_acq_fi_offer(self, GameState state, bint is_fi):
        """Set whether current acquisition is from Foreign Investor."""
        state._data[self._acq_is_fi_offer_offset] = 1.0 if is_fi else 0.0

    # =========================================================================
    # ACQUISITION SYNERGY VALUES
    # =========================================================================

    cpdef void populate_acq_synergy_values(self, GameState state, int corp_id, int target_company_id):
        """Compute and set synergy values for the current acquisition offer.

        For each company i, if the buying corp owns company i, sets the value to
        (COMPANY_SYNERGY[i][target] + COMPANY_SYNERGY[target][i]) / CASH_DIVISOR.
        Otherwise 0.
        """
        cdef int i, bonus
        cdef int offset = self._acq_synergy_values_offset
        cdef float* corp = state._corp_ptr(corp_id)
        cdef int owned_offset = state._corp_fields.owned_companies
        for i in range(<int>GameConstants.NUM_COMPANIES):
            if corp[owned_offset + i] == 1.0:
                bonus = COMPANY_SYNERGY[i][target_company_id] + COMPANY_SYNERGY[target_company_id][i]
                state._data[offset + i] = <float>bonus / CASH_DIVISOR
            else:
                state._data[offset + i] = 0.0

    cpdef void clear_acq_synergy_values(self, GameState state):
        """Clear all synergy values to 0 (non-active)."""
        cdef int i
        cdef int offset = self._acq_synergy_values_offset
        for i in range(<int>GameConstants.NUM_COMPANIES):
            state._data[offset + i] = 0.0

    cpdef float get_acq_synergy_value(self, GameState state, int company_id):
        """Get synergy value for a company (for testing). Returns raw float."""
        return state._data[self._acq_synergy_values_offset + company_id]

    # =========================================================================
    # CLOSING STATE
    # =========================================================================

    cpdef int get_closing_company(self, GameState state):
        """Get company being offered for closing. Uses hidden compact storage for O(1) access. Returns -1 if none."""
        return <int>state._data[self._hidden_closing_company_offset]

    cpdef void set_closing_company(self, GameState state, int company_id):
        """Set company being offered for closing. Updates active_company one-hot and hidden compact."""
        set_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_closing_company_offset, company_id
        )

    cpdef void clear_closing_company(self, GameState state):
        """Clear closing company. Updates active_company one-hot and hidden compact."""
        clear_one_hot_with_compact(
            state._data, self._active_company_offset, GameConstants.NUM_COMPANIES,
            self._hidden_closing_company_offset
        )

    # =========================================================================
    # ACTIVE CORP ONE-HOT (no hidden compact — for CLOSING phase)
    # =========================================================================

    cpdef void set_active_corp_one_hot(self, GameState state, int corp_id):
        """Set active corp one-hot only (no hidden compact update).

        Used by CLOSING phase where there is no per-phase hidden compact
        for the corp. Other phases (DIVIDENDS, ISSUE, ACQ) use their own
        set_*_corp methods which also update hidden compact storage.
        """
        set_one_hot(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS, corp_id
        )

    cpdef void clear_active_corp_one_hot(self, GameState state):
        """Clear active corp one-hot only (no hidden compact update)."""
        clear_one_hot(
            state._data, self._active_corp_offset, GameConstants.NUM_CORPS
        )

    # =========================================================================
    # TURN ORDER NAVIGATION
    # =========================================================================

    cpdef int find_player_at_position(self, GameState state, int position):
        """
        Find player_id with given turn order position.

        Args:
            state: Game state
            position: Turn order position to find (0-indexed)

        Returns:
            player_id or -1 if not found
        """
        cdef int player_id
        for player_id in range(state._num_players):
            if player_module.PLAYERS[player_id].get_turn_order(state) == position:
                return player_id
        return -1

    cpdef void advance_to_next_bidder(self, GameState state):
        """
        Advance active player to next non-passed bidder in turn order.

        Used during auction to skip players who have left the auction.
        """
        cdef int current_player = state._get_active_player()
        cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
        cdef int next_position, candidate
        cdef int checked = 0

        while checked < state._num_players:
            next_position = (current_position + 1) % state._num_players
            candidate = self.find_player_at_position(state, next_position)

            if not self.has_player_passed_auction(state, candidate):
                state._set_active_player(candidate)
                return

            current_position = next_position
            checked += 1

        # Should never reach here - means all players passed

    cpdef void set_active_player_after(self, GameState state, int player_id):
        """
        Set active player to next player after given player in turn order.

        Args:
            state: Game state
            player_id: Player whose turn just finished
        """
        cdef int position = player_module.PLAYERS[player_id].get_turn_order(state)
        cdef int next_position = (position + 1) % state._num_players
        cdef int next_player = self.find_player_at_position(state, next_position)
        state._set_active_player(next_player)


# =============================================================================
# GLOBAL TURN STATE INSTANCE
# =============================================================================

# Single TurnState instance
TURN = TurnState()
