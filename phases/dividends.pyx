# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Dividends phase implementation.

In this phase, corporations (in descending share price order):
1. Choose dividend amount (0 to share_price // 3)
2. Pay dividends to shareholders
3. Adjust share price based on stars vs target stars

Corps in receivership or with insufficient cash auto-pay 0.
After all corps have paid, transition to END_CARD phase.
"""

cimport cython
from state cimport (
    GameState, PHASE_DIVIDENDS, PHASE_END_CARD,
    NUM_COMPANIES, NUM_CORPS, NUM_MARKET_SPACES
)
from data cimport (
    get_company_stars, get_market_price, get_corp_share_count,
    CORP_SI, MAX_DIVIDEND
)

# Import shared helpers
from helpers.player cimport (
    PlayerOffsets, get_player_offsets,
    get_player_shares, add_player_cash
)
from helpers.corp cimport (
    CorpOffsets, get_corp_offsets,
    is_corp_active, get_corp_cash, set_corp_cash,
    get_corp_issued_shares, get_corp_share_price,
    get_corp_price_index, set_corp_price_index,
    is_corp_in_receivership, corp_owns_company,
    set_active_player_to_president,
    calculate_corp_company_stars, calculate_target_stars,
    handle_corp_bankruptcy
)
from helpers.turn cimport (
    DividendTurnOffsets, get_dividend_turn_offsets
)
from helpers.market cimport (
    find_adjusted_price_index
)


# =============================================================================
# DIVIDENDS PHASE CLASS
# =============================================================================

cdef class DividendsPhase:
    """
    Manages the Dividends phase.

    Corps pay dividends in descending share price order.
    Presidents choose dividend amount (0 to share_price // 3).
    Share price adjusts based on stars vs target stars.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players
        self._po = get_player_offsets(num_players)
        self._co = get_corp_offsets()
        self._dto = get_dividend_turn_offsets(num_players)

    # =========================================================================
    # POINTER HELPERS
    # =========================================================================

    cdef float* _get_player(self, GameState state, int player_id) noexcept nogil:
        """Get pointer to player data."""
        return state._player_ptr(player_id)

    cdef float* _get_corp(self, GameState state, int corp_id) noexcept nogil:
        """Get pointer to corp data."""
        return state._corp_ptr(corp_id)

    cdef float* _get_turn(self, GameState state) noexcept nogil:
        """Get pointer to turn state."""
        return state._turn_ptr()

    cdef float* _get_market(self, GameState state) noexcept nogil:
        """Get pointer to market data."""
        return state._data + state._layout.market_offset

    # =========================================================================
    # PHASE SETUP
    # =========================================================================

    cpdef void setup_dividends(self, GameState state):
        """
        Initialize dividends phase.

        Sets up dividend_remaining flags for all active corps,
        then processes the first corp (highest share price).
        """
        cdef int corp_id
        cdef float* turn = self._get_turn(state)
        cdef float* corp
        cdef list active_corps = []

        state.set_phase(PHASE_DIVIDENDS)

        # Clear dividend state
        for corp_id in range(NUM_CORPS):
            turn[self._dto.dividend_corp + corp_id] = 0.0
            turn[self._dto.dividend_remaining + corp_id] = 0.0

        # Mark active corps as remaining
        for corp_id in range(NUM_CORPS):
            corp = self._get_corp(state, corp_id)
            if is_corp_active(corp, &self._co):
                turn[self._dto.dividend_remaining + corp_id] = 1.0
                active_corps.append(corp_id)

        if not active_corps:
            # No active corps, skip to END_CARD
            self._transition_to_end_card(state)
            return

        # Process first corp (highest share price)
        self.advance_to_next_corp(state)

    cpdef void advance_to_next_corp(self, GameState state):
        """
        Advance to the next corp in dividend order (descending share price).

        If a corp is in receivership or can't afford any dividend, auto-pay 0.
        Otherwise, wait for player action.
        """
        cdef int corp_id, best_corp, best_price, price
        cdef float* turn = self._get_turn(state)
        cdef float* corp
        cdef int issued_shares, cash

        # Clear current dividend corp
        for corp_id in range(NUM_CORPS):
            turn[self._dto.dividend_corp + corp_id] = 0.0

        # Find highest share price corp among remaining
        best_corp = -1
        best_price = -1

        for corp_id in range(NUM_CORPS):
            if turn[self._dto.dividend_remaining + corp_id] != 1.0:
                continue

            corp = self._get_corp(state, corp_id)
            price = get_corp_share_price(corp, &self._co)

            if price > best_price:
                best_price = price
                best_corp = corp_id

        if best_corp < 0:
            # No more corps to process
            self._transition_to_end_card(state)
            return

        # Set up state for this corp
        self._setup_dividend_state(state, best_corp)

        # Check if corp needs auto-processing
        corp = self._get_corp(state, best_corp)

        # Receivership corps always pay 0
        if is_corp_in_receivership(corp, &self._co):
            self.do_dividend(state, 0)
            return

        # Corps that can't afford even $1 per share pay 0
        issued_shares = get_corp_issued_shares(corp, &self._co)
        cash = get_corp_cash(corp, &self._co)
        if cash < issued_shares:
            self.do_dividend(state, 0)
            return

        # Otherwise, set active player to president and wait for action
        set_active_player_to_president(state, best_corp, self._num_players)

    cdef void _setup_dividend_state(self, GameState state, int corp_id) noexcept:
        """Set up dividend_corp and calculate dividend_impact for NN."""
        cdef float* turn = self._get_turn(state)
        cdef float* corp = self._get_corp(state, corp_id)
        cdef int i, issued_shares, cash, company_stars, target
        cdef int new_cash, new_stars, diff

        # Set current dividend corp (one-hot)
        turn[self._dto.dividend_corp + corp_id] = 1.0

        # Calculate dividend_impact for each possible dividend level
        issued_shares = get_corp_issued_shares(corp, &self._co)
        cash = get_corp_cash(corp, &self._co)

        # Get company stars (without cash contribution)
        company_stars = calculate_corp_company_stars(corp, &self._co)

        # Add SI bonus if applicable
        if corp_id == CORP_SI:
            company_stars += 2

        target = calculate_target_stars(corp, &self._co)

        for i in range(MAX_DIVIDEND):
            new_cash = cash - (i * issued_shares)
            if new_cash < 0:
                turn[self._dto.dividend_impact + i] = -1.0
            else:
                new_stars = company_stars + (new_cash // 10)
                diff = new_stars - target
                # Scale from [-2, 2] to [-0.5, 0.5]
                # Clamp diff to reasonable range first
                if diff < -2:
                    diff = -2
                elif diff > 2:
                    diff = 2
                turn[self._dto.dividend_impact + i] = <float>diff / 4.0

    # =========================================================================
    # ACTION VALIDATION AND EXECUTION
    # =========================================================================

    cpdef bint can_do_dividend(self, GameState state, int amount):
        """Check if the given dividend amount is valid."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return False

        if amount < 0:
            return False

        cdef float* corp = self._get_corp(state, corp_id)
        cdef int share_price = get_corp_share_price(corp, &self._co)
        cdef int max_dividend = share_price // 3

        if amount > max_dividend:
            return False

        # Check corp can afford it
        cdef int issued_shares = get_corp_issued_shares(corp, &self._co)
        cdef int total_payout = amount * issued_shares
        cdef int cash = get_corp_cash(corp, &self._co)

        if total_payout > cash:
            return False

        return True

    cpdef void do_dividend(self, GameState state, int amount):
        """Execute dividend payout for current corp."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return

        if not self.can_do_dividend(state, amount):
            return

        # Pay dividends
        self._pay_dividends(state, corp_id, amount)

        # Adjust share price
        self._adjust_share_price(state, corp_id)

        # Mark this corp as done
        cdef float* turn = self._get_turn(state)
        turn[self._dto.dividend_remaining + corp_id] = 0.0

        # Advance to next corp
        self.advance_to_next_corp(state)

    cdef void _pay_dividends(self, GameState state, int corp_id, int amount) noexcept:
        """Pay dividends to player shareholders."""
        cdef float* corp = self._get_corp(state, corp_id)
        cdef int issued_shares = get_corp_issued_shares(corp, &self._co)
        cdef int total_payout = amount * issued_shares
        cdef int player_id, player_shares, player_payout
        cdef float* player

        # Deduct from corp cash
        cdef int corp_cash = get_corp_cash(corp, &self._co)
        set_corp_cash(corp, &self._co, corp_cash - total_payout)

        # Pay each player based on their shares
        for player_id in range(self._num_players):
            player = self._get_player(state, player_id)
            player_shares = get_player_shares(player, &self._po, corp_id)
            if player_shares > 0:
                player_payout = amount * player_shares
                add_player_cash(player, &self._po, player_payout)

        # Note: Bank shares don't need payment (bank has infinite money)

    cdef void _adjust_share_price(self, GameState state, int corp_id) noexcept:
        """Adjust share price based on stars vs target stars."""
        cdef float* corp = self._get_corp(state, corp_id)
        cdef float* market = self._get_market(state)
        cdef float* hidden_price_indices = state._hidden_price_indices_ptr()
        cdef int current_index = get_corp_price_index(corp, &self._co)
        cdef int corp_stars = self.calculate_corp_stars(state, corp_id)
        cdef int target = calculate_target_stars(corp, &self._co)
        cdef int diff = corp_stars - target
        cdef int movement, new_index

        # Clamp movement to [-2, 2]
        if diff <= -2:
            movement = -2
        elif diff == -1:
            movement = -1
        elif diff == 0:
            movement = 0
        elif diff == 1:
            movement = 1
        else:
            movement = 2

        if movement == 0:
            return  # No change

        # Handle case where corp is at price 75 (index 26, on or off market)
        if current_index == NUM_MARKET_SPACES - 1:
            # At 75, can only go down
            if movement > 0:
                return  # Already at max, can't go higher
            # Find available space going down from index 26
            new_index = find_adjusted_price_index(market, NUM_MARKET_SPACES - 1, movement)
        else:
            new_index = find_adjusted_price_index(market, current_index, movement)

        # Check for bankruptcy
        if new_index == 0:
            handle_corp_bankruptcy(state, corp_id, current_index, self._num_players)
            return

        # Check if going off the top (to price 75, index 26)
        if new_index == NUM_MARKET_SPACES - 1:
            # Release old card, move to price 75 (index 26, off-market)
            # Don't take the market space at 26 - multiple corps can be there
            if current_index > 0 and current_index < NUM_MARKET_SPACES - 1:
                state.set_market_space_available(current_index, True)
            set_corp_price_index(corp, &self._co, NUM_MARKET_SPACES - 1, hidden_price_indices, corp_id)
            return

        # Normal case: release old card, take new card
        if current_index > 0 and current_index < NUM_MARKET_SPACES - 1:
            state.set_market_space_available(current_index, True)
        state.set_market_space_available(new_index, False)
        set_corp_price_index(corp, &self._co, new_index, hidden_price_indices, corp_id)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    cpdef list get_valid_actions(self, GameState state):
        """Get list of valid dividend amounts."""
        cdef list result = []
        cdef int i
        cdef int max_div = self.get_max_dividend(state)

        for i in range(max_div + 1):
            if self.can_do_dividend(state, i):
                result.append(i)

        return result

    cpdef int get_max_dividend(self, GameState state):
        """Get maximum dividend amount for current corp."""
        cdef int corp_id = self.get_current_corp(state)
        if corp_id < 0:
            return 0

        cdef float* corp = self._get_corp(state, corp_id)
        cdef int share_price = get_corp_share_price(corp, &self._co)
        return share_price // 3

    cpdef int get_current_corp(self, GameState state):
        """Get the current corp paying dividends, or -1 if none."""
        cdef float* turn = self._get_turn(state)
        cdef int corp_id

        for corp_id in range(NUM_CORPS):
            if turn[self._dto.dividend_corp + corp_id] == 1.0:
                return corp_id

        return -1

    # =========================================================================
    # STAR CALCULATIONS
    # =========================================================================

    cpdef int calculate_corp_stars(self, GameState state, int corp_id):
        """
        Calculate corporation's current stars.

        Stars = company stars + cash // 10 + SI bonus
        """
        cdef float* corp = self._get_corp(state, corp_id)
        cdef int total = calculate_corp_company_stars(corp, &self._co)
        cdef int cash = get_corp_cash(corp, &self._co)

        total += cash // 10

        # SI (Stars, Inc.) gets +2 stars
        if corp_id == CORP_SI:
            total += 2

        return total

    cpdef int calculate_target_stars(self, GameState state, int corp_id):
        """
        Calculate target stars for share price.

        Target = round(issued_shares * share_price / 10)
        """
        cdef float* corp = self._get_corp(state, corp_id)
        return calculate_target_stars(corp, &self._co)

    # =========================================================================
    # PHASE TRANSITION
    # =========================================================================

    cdef void _transition_to_end_card(self, GameState state) noexcept:
        """Transition to END_CARD phase."""
        cdef float* turn = self._get_turn(state)
        cdef int i

        # Clear dividend state
        for i in range(NUM_CORPS):
            turn[self._dto.dividend_corp + i] = -1.0
            turn[self._dto.dividend_remaining + i] = -1.0
        for i in range(MAX_DIVIDEND):
            turn[self._dto.dividend_impact + i] = -1.0

        state.set_phase(PHASE_END_CARD)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create DividendsPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = DividendsPhase(num_players)
    return _phase_handlers[num_players]


def get_action_constants():
    """Get action constants for Python tests."""
    return {
        'MAX_DIVIDEND': MAX_DIVIDEND,
    }
