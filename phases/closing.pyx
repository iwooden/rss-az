# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Closing phase implementation (offer-based).

In this phase:
1. Companies with negative income are offered for closing one at a time
2. Player chooses: close the offered company, or pass
3. After all players pass or no more closeable, auto-closures happen
4. Transition to Income phase
"""

cimport cython
from state cimport GameState, PHASE_CLOSING, PHASE_INCOME, NUM_COMPANIES, NUM_CORPS
from data cimport (
    get_adjusted_company_income, get_company_income, get_company_face_value,
    get_company_stars, get_cost_of_ownership, CORP_JS
)

# Import shared helpers
from helpers.corp cimport get_president_of_corp, find_corp_owning_company

# =============================================================================
# CONSTANTS (offer-based: close or pass)
# =============================================================================

DEF CLOSING_ACTION_CLOSE = 0
DEF CLOSING_ACTION_PASS = 1
DEF CLOSING_ACTION_MAX = 2

# Receivership auto-close thresholds
DEF RECEIVERSHIP_RED_COO_THRESHOLD = 4    # Close red if CoO >= 4
DEF RECEIVERSHIP_ORANGE_COO_THRESHOLD = 7  # Close orange if CoO >= 7


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef bint company_has_negative_income(GameState state, int company_id) noexcept nogil:
    """Check if company has negative adjusted income at current CoO level."""
    cdef int coo_level = state.get_coo_level()
    cdef int adjusted_income = get_adjusted_company_income(company_id, coo_level)
    return adjusted_income < 0


cdef void close_company_for_corp(GameState state, int corp_id, int company_id) noexcept nogil:
    """Close a company owned by a corp. Handles JS special ability."""
    cdef int income_bonus

    # JS gets 2x printed income when closing
    if corp_id == CORP_JS:
        income_bonus = get_company_income(company_id) * 2
        state.add_corp_cash(corp_id, income_bonus)

    # Remove company from corp and mark as removed
    state.set_corp_owns_company(corp_id, company_id, False)
    state.set_company_removed(company_id, True)


cdef void close_company_for_player(GameState state, int player_id, int company_id) noexcept nogil:
    """Close a company owned by a player."""
    state.set_player_owns_company(player_id, company_id, False)
    state.set_company_removed(company_id, True)


cdef void close_company_for_fi(GameState state, int company_id) noexcept nogil:
    """Close a company owned by FI."""
    state.set_fi_owns_company(company_id, False)
    state.set_company_removed(company_id, True)


# =============================================================================
# CLOSING PHASE CLASS
# =============================================================================

cdef class ClosingPhase:
    """
    Manages the Closing phase (offer-based).

    Companies with negative income are offered one at a time.
    Player chooses to close or pass on each offer.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players

    cpdef void setup_closing(self, GameState state):
        """Initialize closing phase."""
        state.set_phase(PHASE_CLOSING)
        state.clear_closing_company()

        # Find first closeable company (start from company 0)
        if not self._find_next_closeable(state, 0):
            # No closeable companies, do auto-closures and transition
            self.auto_close_and_transition(state)

    cpdef bint setup_next_closeable(self, GameState state):
        """
        Find and set up the next closeable company offer.
        Wrapper that starts from company 0.
        """
        return self._find_next_closeable(state, 0)

    cdef bint _find_next_closeable(self, GameState state, int start_company):
        """
        Find next closeable company starting from start_company.

        Searches companies in order by company_id. For each company, checks:
        - If owned by a player with negative income: offer to that player
        - If owned by a corp (with 2+ companies) with negative income: offer to president

        Returns True if an offer was set up, False if no more.
        """
        cdef int company_id, player_id, corp_id, company_count

        for company_id in range(start_company, NUM_COMPANIES):
            if not company_has_negative_income(state, company_id):
                continue

            # Check if owned by any player
            for player_id in range(self._num_players):
                if state.player_owns_company(player_id, company_id):
                    # Player can always close their own negative-income company
                    state._set_active_player(player_id)
                    state.set_current_closing_company(company_id)
                    return True

            # Check if owned by any corp
            for corp_id in range(NUM_CORPS):
                if not state.is_corp_active(corp_id):
                    continue
                if not state.corp_owns_company(corp_id, company_id):
                    continue

                # Corp must have 2+ companies to close any
                company_count = state.get_corp_company_count(corp_id)
                if company_count < 2:
                    continue

                # Offer to president
                player_id = get_president_of_corp(state, corp_id, self._num_players)
                if player_id >= 0:
                    state._set_active_player(player_id)
                    state.set_current_closing_company(company_id)
                    return True

        # No more closeable companies
        state.clear_closing_company()
        return False

    cpdef bint is_waiting_for_action(self, GameState state):
        """Check if we're waiting for a player action."""
        if state.get_phase() != PHASE_CLOSING:
            return False
        return state.get_current_closing_company() >= 0

    cpdef bint can_do_action(self, GameState state, int action):
        """Check if action is valid."""
        if state.get_phase() != PHASE_CLOSING:
            return False

        if action < 0 or action >= CLOSING_ACTION_MAX:
            return False

        cdef int company_id = state.get_current_closing_company()
        if company_id < 0:
            return False

        # Both close and pass are valid when there's an offer
        return True

    cpdef void do_action(self, GameState state, int action):
        """Execute a closing action."""
        cdef int player_id, company_id, corp_id, next_company

        if not self.can_do_action(state, action):
            return

        player_id = state._get_active_player()
        company_id = state.get_current_closing_company()
        next_company = company_id + 1  # Start search from next company

        if action == CLOSING_ACTION_PASS:
            # Player passes - keep this company, move to next offer
            state.clear_closing_company()
            if not self._find_next_closeable(state, next_company):
                self.auto_close_and_transition(state)
            return

        # CLOSING_ACTION_CLOSE - close the current company
        # Determine owner (player or corp)
        if state.player_owns_company(player_id, company_id):
            close_company_for_player(state, player_id, company_id)
        else:
            corp_id = find_corp_owning_company(state, player_id, company_id)
            if corp_id >= 0:
                close_company_for_corp(state, corp_id, company_id)

        # Find next closeable company
        state.clear_closing_company()
        if not self._find_next_closeable(state, next_company):
            self.auto_close_and_transition(state)

    cpdef list get_valid_actions(self, GameState state):
        """Get list of valid action IDs."""
        cdef list result = []
        if state.get_current_closing_company() >= 0:
            result.append(CLOSING_ACTION_CLOSE)
            result.append(CLOSING_ACTION_PASS)
        return result

    cpdef void auto_close_and_transition(self, GameState state):
        """
        Handle automatic closures and transition to Income phase.

        1. FI closes all negative-income companies
        2. Receivership corps close per rules
        3. Players force-close to prevent bankruptcy
        4. Transition to Income
        """
        cdef int company_id, corp_id, player_id
        cdef int coo_level = state.get_coo_level()

        # 1. FI auto-closes negative-income companies
        for company_id in range(NUM_COMPANIES):
            if state.fi_owns_company(company_id):
                if company_has_negative_income(state, company_id):
                    close_company_for_fi(state, company_id)

        # 2. Receivership corps auto-close
        for corp_id in range(NUM_CORPS):
            if not state.is_corp_active(corp_id):
                continue
            if not state.is_corp_in_receivership(corp_id):
                continue

            # Process in ascending face value (keep highest)
            self._auto_close_receivership_corp(state, corp_id, coo_level)

        # 3. Force-close player companies to prevent bankruptcy
        for player_id in range(self._num_players):
            self._force_close_player_companies(state, player_id, coo_level)

        # 4. Transition to Income phase
        state.clear_closing_company()
        state.set_phase(PHASE_INCOME)

    cdef void _auto_close_receivership_corp(self, GameState state, int corp_id, int coo_level) noexcept:
        """Auto-close companies for a receivership corp per rules."""
        cdef int company_id, stars, coo_cost
        cdef list owned_companies = []

        # Collect companies (ascending face value)
        for company_id in range(NUM_COMPANIES):
            if state.corp_owns_company(corp_id, company_id):
                owned_companies.append(company_id)

        # Sort by face value ascending
        owned_companies.sort(key=lambda c: get_company_face_value(c))

        # Check each company for auto-close condition
        for company_id in owned_companies:
            # Keep at least one company
            if state.get_corp_company_count(corp_id) <= 1:
                break

            stars = get_company_stars(company_id)
            coo_cost = get_cost_of_ownership(coo_level, stars)

            # Red (stars=1): close if CoO >= 4
            if stars == 1 and coo_cost >= RECEIVERSHIP_RED_COO_THRESHOLD:
                close_company_for_corp(state, corp_id, company_id)
            # Orange (stars=2): close if CoO >= 7
            elif stars == 2 and coo_cost >= RECEIVERSHIP_ORANGE_COO_THRESHOLD:
                close_company_for_corp(state, corp_id, company_id)

    cdef void _force_close_player_companies(self, GameState state, int player_id, int coo_level) noexcept:
        """Force-close player companies to prevent bankruptcy from negative income."""
        cdef int company_id, company_income
        cdef int player_cash = state.get_player_cash(player_id)
        cdef int total_income = 0
        cdef list player_companies = []

        # Calculate total income from player's companies
        for company_id in range(NUM_COMPANIES):
            if state.player_owns_company(player_id, company_id):
                player_companies.append(company_id)
                total_income += get_adjusted_company_income(company_id, coo_level)

        # Check if player would go bankrupt
        if player_cash + total_income >= 0:
            return  # No problem

        # Sort by adjusted income ascending (close worst first)
        player_companies.sort(key=lambda c: get_adjusted_company_income(c, coo_level))

        # Close companies until player won't go bankrupt
        for company_id in player_companies:
            if player_cash + total_income >= 0:
                break

            # Remove this company's income from total
            company_income = get_adjusted_company_income(company_id, coo_level)
            total_income -= company_income
            close_company_for_player(state, player_id, company_id)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create ClosingPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = ClosingPhase(num_players)
    return _phase_handlers[num_players]


def get_action_constants():
    """Get action constants for Python tests."""
    return {
        'CLOSING_ACTION_CLOSE': CLOSING_ACTION_CLOSE,
        'CLOSING_ACTION_PASS': CLOSING_ACTION_PASS,
        'CLOSING_ACTION_MAX': CLOSING_ACTION_MAX,
    }
