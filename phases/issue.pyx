# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Issue Shares phase implementation.

In this phase, corporations (in descending share price order):
1. May issue one share from unissued stack
2. Share goes to bank (bank_shares += 1)
3. Share price moves down by 1 (skip taken spaces), unless Stock Masters
4. Corp receives the new share price as cash

Corps in receivership MUST issue if they have unissued shares.
After all corps processed, transition to IPO phase.
"""

cimport cython
from state cimport (
    GameState, PHASE_ISSUE_SHARES, PHASE_IPO,
    NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES
)
from data cimport (
    get_market_price, get_corp_share_count, get_company_face_value,
    CORP_SM
)

# Import shared helpers
from helpers.player cimport (
    PlayerOffsets, get_player_offsets,
    player_owns_company,
    update_all_player_net_worths
)
from helpers.corp cimport (
    CorpOffsets, get_corp_offsets,
    is_corp_active, get_corp_cash, set_corp_cash,
    get_corp_share_price, get_corp_price_index, set_corp_price_index,
    get_corp_issued_shares, set_corp_issued_shares,
    get_corp_bank_shares, set_corp_bank_shares,
    get_corp_unissued_shares, set_corp_unissued_shares,
    is_corp_in_receivership,
    set_active_player_to_president,
    handle_corp_bankruptcy
)
from helpers.turn cimport (
    IssueTurnOffsets, get_issue_turn_offsets
)
from helpers.market cimport (
    find_next_lower_price_index
)


# =============================================================================
# ISSUE PHASE CLASS
# =============================================================================

cdef class IssuePhase:
    """
    Manages the Issue Shares phase.

    Corps issue shares in descending share price order.
    Receivership corps must issue if able.
    Stock Masters (SM) doesn't lose share price when issuing.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players
        self._co = get_corp_offsets()
        self._po = get_player_offsets(num_players)
        self._ito = get_issue_turn_offsets(num_players)

    # =========================================================================
    # POINTER HELPERS
    # =========================================================================

    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil:
        return state._corp_ptr(corp_id)

    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil:
        return state._player_ptr(player_id)

    cdef float* _get_turn(self, GameState state) noexcept nogil:
        return state._turn_ptr()

    cdef float* _get_market(self, GameState state) noexcept nogil:
        return state._data + state._layout.market_offset

    # =========================================================================
    # PHASE SETUP
    # =========================================================================

    cpdef void setup_issue_phase(self, GameState state):
        """
        Initialize issue phase.

        Note: issue_remaining flags should already be set by End Card phase.
        This just starts processing the first corp.
        """
        state.set_phase(PHASE_ISSUE_SHARES)
        self.advance_to_next_corp(state)

    cpdef void advance_to_next_corp(self, GameState state):
        """
        Advance to the next corp in issue order (descending share price).

        If a corp is in receivership and has unissued shares, auto-issue.
        Otherwise, wait for president's action.
        """
        cdef int corp_id, best_corp, best_price, price
        cdef float* turn = self._get_turn(state)
        cdef float* corp
        cdef int unissued

        # Clear current issue corp
        for corp_id in range(NUM_CORPS):
            turn[self._ito.issue_corp + corp_id] = 0.0

        # Find highest share price corp among remaining that can issue
        best_corp = -1
        best_price = -1

        for corp_id in range(NUM_CORPS):
            # Check issue_remaining flag (1.0 = can issue, 0.0 = fully issued, -1.0 = inactive)
            if turn[self._ito.issue_remaining + corp_id] != 1.0:
                continue

            corp = self._get_corp(state, corp_id)
            price = get_corp_share_price(corp, &self._co)

            if price > best_price:
                best_price = price
                best_corp = corp_id

        if best_corp < 0:
            # No more corps to process
            self._transition_to_ipo(state)
            return

        # Set current issue corp (one-hot)
        turn[self._ito.issue_corp + best_corp] = 1.0

        # Check if corp is in receivership
        corp = self._get_corp(state, best_corp)
        if is_corp_in_receivership(corp, &self._co):
            # Receivership corps must issue if able
            unissued = get_corp_unissued_shares(corp, &self._co)
            if unissued > 0:
                self.do_issue(state)
            else:
                # Can't issue, mark as done and move on
                turn[self._ito.issue_remaining + best_corp] = 0.0
                self.advance_to_next_corp(state)
            return

        # Normal corp - set active player to president and wait for action
        set_active_player_to_president(state, best_corp, self._num_players)

    # =========================================================================
    # ACTION VALIDATION
    # =========================================================================

    cpdef bint can_issue(self, GameState state):
        """Check if current corp can issue a share."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return False

        cdef float* corp = self._get_corp(state, corp_id)
        cdef int unissued = get_corp_unissued_shares(corp, &self._co)

        return unissued > 0

    cpdef bint can_pass(self, GameState state):
        """Check if current corp can pass (skip issuing)."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return False

        cdef float* corp = self._get_corp(state, corp_id)
        cdef int unissued

        # Receivership corps cannot pass if they can issue
        if is_corp_in_receivership(corp, &self._co):
            unissued = get_corp_unissued_shares(corp, &self._co)
            return unissued == 0

        return True

    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================

    cpdef void do_issue(self, GameState state):
        """Execute share issue for current corp."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return

        if not self.can_issue(state):
            return

        cdef float* turn = self._get_turn(state)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)
        cdef float* hidden_price_indices = state._hidden_price_indices_ptr()
        cdef int issued = get_corp_issued_shares(corp, &self._co)
        cdef int bank_shares = get_corp_bank_shares(corp, &self._co)
        cdef int unissued = get_corp_unissued_shares(corp, &self._co)
        cdef int current_index = get_corp_price_index(corp, &self._co)
        cdef int new_index, new_price, corp_cash

        # Issue the share
        set_corp_issued_shares(corp, &self._co, issued + 1)
        set_corp_bank_shares(corp, &self._co, bank_shares + 1)
        set_corp_unissued_shares(corp, &self._co, unissued - 1)

        # Move share price down (unless Stock Masters)
        if corp_id != CORP_SM:
            if current_index == NUM_MARKET_SPACES - 1:
                # At 75 (index 26, off-market), find highest available below 75
                new_index = find_next_lower_price_index(market, NUM_MARKET_SPACES - 1)
            else:
                new_index = find_next_lower_price_index(market, current_index)

            # Check for bankruptcy
            if new_index == 0:
                handle_corp_bankruptcy(state, corp_id, current_index, self._num_players)
                # Mark as done (bankruptcy already handled)
                turn[self._ito.issue_remaining + corp_id] = 0.0
                # Update net worths after bankruptcy
                update_all_player_net_worths(state, self._num_players)
                self.advance_to_next_corp(state)
                return

            # Update market space and price index
            # Don't release index 0 (inactive) or 26 (multiple corps can be at $75)
            if current_index > 0 and current_index < NUM_MARKET_SPACES - 1:
                state.set_market_space_available(current_index, True)
            # Don't take index 26 ($75) - multiple corps can be there
            if new_index < NUM_MARKET_SPACES - 1:
                state.set_market_space_available(new_index, False)
            set_corp_price_index(corp, &self._co, new_index, hidden_price_indices, corp_id)
            new_price = get_market_price(new_index)
        else:
            # Stock Masters: price doesn't change
            new_price = get_corp_share_price(corp, &self._co)

        # Corp receives the (new) share price as cash
        corp_cash = get_corp_cash(corp, &self._co)
        set_corp_cash(corp, &self._co, corp_cash + new_price)

        # Mark this corp as done
        turn[self._ito.issue_remaining + corp_id] = 0.0

        # Update all player net worths (share prices changed)
        update_all_player_net_worths(state, self._num_players)

        # Advance to next corp
        self.advance_to_next_corp(state)

    cpdef void do_pass(self, GameState state):
        """Skip issuing for current corp."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return

        if not self.can_pass(state):
            return

        cdef float* turn = self._get_turn(state)

        # Mark this corp as done (chose not to issue)
        turn[self._ito.issue_remaining + corp_id] = 0.0

        # Advance to next corp
        self.advance_to_next_corp(state)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    cpdef int get_current_corp(self, GameState state):
        """Get the current corp issuing, or -1 if none."""
        cdef float* turn = self._get_turn(state)
        cdef int corp_id

        for corp_id in range(NUM_CORPS):
            if turn[self._ito.issue_corp + corp_id] == 1.0:
                return corp_id

        return -1

    cpdef list get_valid_actions(self, GameState state):
        """Get list of valid action names."""
        cdef list result = []

        if self.can_issue(state):
            result.append("issue")

        if self.can_pass(state):
            result.append("pass")

        return result

    # =========================================================================
    # PHASE TRANSITION
    # =========================================================================

    cdef void _transition_to_ipo(self, GameState state) noexcept:
        """Transition to IPO phase."""
        cdef float* turn = self._get_turn(state)
        cdef int i, company_id
        cdef float* player

        # Clear issue state
        for i in range(NUM_CORPS):
            turn[self._ito.issue_corp + i] = -1.0
            turn[self._ito.issue_remaining + i] = -1.0

        # Set up IPO state
        # Clear ipo_company
        for i in range(NUM_COMPANIES):
            turn[self._ito.ipo_company + i] = -1.0

        # Mark player-owned companies in ipo_remaining (descending face value order)
        # For now, just mark all player-owned companies; the IPO phase will handle ordering
        for company_id in range(NUM_COMPANIES):
            turn[self._ito.ipo_remaining + company_id] = -1.0

        # Find companies owned by players
        for company_id in range(NUM_COMPANIES):
            for player_id in range(self._num_players):
                player = self._get_player(state, player_id)
                if player_owns_company(player, &self._po, company_id):
                    turn[self._ito.ipo_remaining + company_id] = 1.0
                    break

        state.set_phase(PHASE_IPO)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create IssuePhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = IssuePhase(num_players)
    return _phase_handlers[num_players]


def get_constants():
    """Get constants for Python tests."""
    return {
        'CORP_SM': CORP_SM,
    }
