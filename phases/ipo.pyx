# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
IPO phase implementation.

In this phase, in descending Face Value order, private companies may form corporations:
1. Player removes company from their assets
2. Selects an inactive corp and a valid par price for company's star tier
3. If par_price >= face_value: 2 shares issued (1 to player, 1 to bank)
   If par_price < face_value: 4 shares issued (2 to player, 2 to bank)
4. Player pays corp: (player_shares * par_price) - face_value
5. Bank pays corp: bank_shares * par_price
6. Company becomes subsidiary of corp

After all companies processed (or no valid IPO possible), transition to INVEST phase.
"""

cimport cython
from state cimport (
    GameState, PHASE_IPO, PHASE_INVEST,
    NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES
)
from data cimport (
    get_market_price, get_market_index, get_corp_share_count,
    get_company_face_value, get_company_stars,
    is_valid_par_price, get_par_price,
    NUM_PAR_PRICES
)

# Import shared helpers
from helpers.player cimport (
    PlayerOffsets, get_player_offsets,
    get_player_cash, set_player_cash,
    get_player_shares, set_player_shares,
    player_owns_company, set_player_owns_company,
    set_player_president,
    calculate_player_net_worth, update_all_player_net_worths
)
from helpers.corp cimport (
    CorpOffsets, get_corp_offsets,
    is_corp_active, set_corp_active,
    set_corp_cash, set_corp_price_index,
    set_corp_issued_shares, set_corp_bank_shares, set_corp_unissued_shares,
    set_corp_owns_company
)
from helpers.turn cimport (
    IssueTurnOffsets, get_issue_turn_offsets
)


# =============================================================================
# IPO PHASE CLASS
# =============================================================================

cdef class IPOPhase:
    """
    Manages the IPO phase.

    Companies are processed in descending face value order.
    For each company, if the owner can afford any valid IPO, they may:
    - Pass (skip IPO'ing this company)
    - IPO with a specific corp and par price
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
        return state._market_ptr()

    # =========================================================================
    # PHASE SETUP
    # =========================================================================

    cpdef void setup_ipo_phase(self, GameState state):
        """
        Initialize IPO phase.

        Note: ipo_remaining flags should already be set by Issue phase.
        This just starts processing the first company.
        """
        state.set_phase(PHASE_IPO)
        self.advance_to_next_company(state)

    cpdef void advance_to_next_company(self, GameState state):
        """
        Advance to the next company in IPO order (descending face value).

        Find the highest face value company that:
        1. Is in ipo_remaining (player-owned)
        2. Has an owner who can afford at least one valid IPO

        If no such company exists, transition to INVEST phase.
        """
        cdef int company_id, owner_id
        cdef float* turn = self._get_turn(state)
        cdef int i

        # Clear current IPO company
        for i in range(NUM_COMPANIES):
            turn[self._ito.ipo_company + i] = 0.0

        # Check if any inactive corp exists
        if not self._has_any_inactive_corp(state):
            self._transition_to_invest(state)
            return

        # Process companies in descending face value order (highest index first)
        for company_id in range(NUM_COMPANIES - 1, -1, -1):
            # Check if company is in ipo_remaining
            if turn[self._ito.ipo_remaining + company_id] != 1.0:
                continue

            # Find owner
            owner_id = self._find_company_owner(state, company_id)
            if owner_id < 0:
                # No owner found - remove from ipo_remaining and continue
                turn[self._ito.ipo_remaining + company_id] = 0.0
                continue

            # Check if owner can afford any valid IPO
            if not self._can_player_afford_any_ipo(state, owner_id, company_id):
                # Owner can't afford - remove from ipo_remaining and continue
                turn[self._ito.ipo_remaining + company_id] = 0.0
                continue

            # Found a valid company - set it as current and set active player
            turn[self._ito.ipo_company + company_id] = 1.0
            state._set_active_player(owner_id)
            return

        # No more valid companies
        self._transition_to_invest(state)

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    cdef int _find_company_owner(self, GameState state, int company_id) noexcept nogil:
        """Find the player who owns this company, or -1 if none."""
        cdef int player_id
        cdef float* player

        for player_id in range(self._num_players):
            player = self._get_player(state, player_id)
            if player_owns_company(player, &self._po, company_id):
                return player_id

        return -1

    cdef bint _has_any_inactive_corp(self, GameState state) noexcept nogil:
        """Check if any corporation is inactive (available for IPO)."""
        cdef int corp_id
        cdef float* corp

        for corp_id in range(NUM_CORPS):
            corp = self._get_corp(state, corp_id)
            if not is_corp_active(corp, &self._co):
                return True

        return False

    cdef bint _can_player_afford_any_ipo(self, GameState state, int player_id, int company_id) noexcept nogil:
        """Check if player can afford at least one valid IPO for this company."""
        cdef float* player = self._get_player(state, player_id)
        cdef float* market = self._get_market(state)
        cdef int player_cash = get_player_cash(player, &self._po)
        cdef int face_value = get_company_face_value(company_id)
        cdef int star_tier = get_company_stars(company_id)
        cdef int par_index, par_price, cost, market_index

        # Check each valid par price for this company's star tier
        for par_index in range(NUM_PAR_PRICES):
            if not is_valid_par_price(star_tier, par_index):
                continue

            par_price = get_par_price(par_index)
            market_index = get_market_index(par_price)

            # Check if market space is available
            if market_index < 0 or market[market_index] != 1.0:
                continue

            # Calculate cost to player
            cost = self._calculate_ipo_cost(face_value, par_price)

            if player_cash >= cost:
                return True

        return False

    cdef int _calculate_ipo_cost(self, int face_value, int par_price) noexcept nogil:
        """
        Calculate how much player pays to IPO.

        If par_price >= face_value: player gets 1 share, pays (1 * par_price) - face_value
        If par_price < face_value: player gets 2 shares, pays (2 * par_price) - face_value
        """
        cdef int player_shares
        if par_price >= face_value:
            player_shares = 1
        else:
            player_shares = 2
        return (player_shares * par_price) - face_value

    # =========================================================================
    # ACTION VALIDATION
    # =========================================================================

    cpdef bint can_pass(self, GameState state):
        """Check if player can pass (skip IPO'ing current company)."""
        return self.get_current_company(state) >= 0

    cpdef bint can_ipo(self, GameState state, int corp_id, int par_index):
        """Check if player can IPO with given corp and par price."""
        cdef int company_id = self.get_current_company(state)
        if company_id < 0:
            return False

        # Check corp is inactive
        cdef float* corp = self._get_corp(state, corp_id)
        if is_corp_active(corp, &self._co):
            return False

        # Check par price is valid for company's star tier
        cdef int star_tier = get_company_stars(company_id)
        if not is_valid_par_price(star_tier, par_index):
            return False

        # Check market space is available
        cdef int par_price = get_par_price(par_index)
        cdef int market_index = get_market_index(par_price)
        cdef float* market = self._get_market(state)

        if market_index < 0 or market[market_index] != 1.0:
            return False

        # Check player can afford
        cdef int player_id = state._get_active_player()
        cdef float* player = self._get_player(state, player_id)
        cdef int player_cash = get_player_cash(player, &self._po)
        cdef int face_value = get_company_face_value(company_id)
        cdef int cost = self._calculate_ipo_cost(face_value, par_price)

        return player_cash >= cost

    # =========================================================================
    # ACTION EXECUTION
    # =========================================================================

    cpdef void do_pass(self, GameState state):
        """Skip IPO'ing current company."""
        cdef int company_id = self.get_current_company(state)
        if company_id < 0:
            return

        cdef float* turn = self._get_turn(state)

        # Remove from ipo_remaining
        turn[self._ito.ipo_remaining + company_id] = 0.0

        # Advance to next company
        self.advance_to_next_company(state)

    cpdef void do_ipo(self, GameState state, int corp_id, int par_index):
        """Execute IPO with given corp and par price."""
        cdef int company_id = self.get_current_company(state)
        if company_id < 0:
            return

        if not self.can_ipo(state, corp_id, par_index):
            return

        cdef int player_id = state._get_active_player()
        cdef float* turn = self._get_turn(state)
        cdef float* player = self._get_player(state, player_id)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)
        cdef float* hidden_price_indices = state._hidden_price_indices_ptr()

        cdef int par_price = get_par_price(par_index)
        cdef int market_index = get_market_index(par_price)
        cdef int face_value = get_company_face_value(company_id)
        cdef int total_shares = get_corp_share_count(corp_id)

        # Determine share distribution
        cdef int player_shares, bank_shares, issued_shares
        if par_price >= face_value:
            player_shares = 1
            bank_shares = 1
        else:
            player_shares = 2
            bank_shares = 2
        issued_shares = player_shares + bank_shares

        # Calculate payments
        cdef int player_cost = (player_shares * par_price) - face_value
        cdef int bank_payment = bank_shares * par_price
        cdef int corp_cash = player_cost + bank_payment

        # Remove company from player
        set_player_owns_company(player, &self._po, company_id, False)

        # Set up corp
        set_corp_active(corp, &self._co, True)
        set_corp_price_index(corp, &self._co, market_index, hidden_price_indices, corp_id)
        set_corp_cash(corp, &self._co, corp_cash)
        set_corp_issued_shares(corp, &self._co, issued_shares)
        set_corp_bank_shares(corp, &self._co, bank_shares)
        set_corp_unissued_shares(corp, &self._co, total_shares - issued_shares)
        set_corp_owns_company(corp, &self._co, company_id, True)

        # Mark market space as taken
        market[market_index] = 0.0

        # Give player their shares and set as president
        set_player_shares(player, &self._po, corp_id, player_shares)
        set_player_president(player, &self._po, corp_id, True)

        # Deduct player's payment
        cdef int current_cash = get_player_cash(player, &self._po)
        set_player_cash(player, &self._po, current_cash - player_cost)

        # Remove from ipo_remaining
        turn[self._ito.ipo_remaining + company_id] = 0.0

        # Update all player net worths
        update_all_player_net_worths(state, self._num_players)

        # Advance to next company
        self.advance_to_next_company(state)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    cpdef int get_current_company(self, GameState state):
        """Get the current company for IPO, or -1 if none."""
        cdef float* turn = self._get_turn(state)
        cdef int company_id

        for company_id in range(NUM_COMPANIES):
            if turn[self._ito.ipo_company + company_id] == 1.0:
                return company_id

        return -1

    cpdef int get_current_company_owner(self, GameState state):
        """Get the owner of the current company, or -1 if none."""
        cdef int company_id = self.get_current_company(state)
        if company_id < 0:
            return -1
        return self._find_company_owner(state, company_id)

    cpdef list get_valid_ipo_options(self, GameState state):
        """
        Get list of valid (corp_id, par_index) tuples for current company.

        Returns empty list if no current company or no valid options.
        """
        cdef list result = []
        cdef int corp_id, par_index

        for corp_id in range(NUM_CORPS):
            for par_index in range(NUM_PAR_PRICES):
                if self.can_ipo(state, corp_id, par_index):
                    result.append((corp_id, par_index))

        return result

    # =========================================================================
    # PHASE TRANSITION
    # =========================================================================

    cdef void _transition_to_invest(self, GameState state) noexcept:
        """Transition to INVEST phase for the next turn."""
        cdef float* turn = self._get_turn(state)
        cdef int i
        cdef int first_player
        cdef int current_turn

        # Clear IPO state
        for i in range(NUM_COMPANIES):
            turn[self._ito.ipo_company + i] = -1.0
            turn[self._ito.ipo_remaining + i] = -1.0

        # Increment turn number (we're starting a new turn)
        current_turn = state.get_turn_number()
        state.set_turn_number(current_turn + 1)

        # Set phase to INVEST
        state.set_phase(PHASE_INVEST)

        # Set active player to first player in turn order (position 0)
        first_player = state.get_player_at_turn_order(0)
        state._set_active_player(first_player)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create IPOPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = IPOPhase(num_players)
    return _phase_handlers[num_players]


def get_constants():
    """Get constants for Python tests."""
    return {
        'NUM_PAR_PRICES': NUM_PAR_PRICES,
        'NUM_CORPS': NUM_CORPS,
    }
