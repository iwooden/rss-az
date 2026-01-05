# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Acquisition phase implementation.

The Acquisition phase (Phase 3) handles corporations buying companies:
1. FI offers: Corporations offered FI companies in priority order
   - OS first (pays face value), then by share price descending (pay high price)
   - Receivership corps auto-buy
2. General acquisitions: (buyer_corp, target_company) tuples offered in order
   - Order by buyer share price (desc), then target face value (desc)
   - Targets: president's private companies, or sibling corp's companies (2+ companies)
   - Price: 0-49 offset from low_price, or pass

At phase end, acquisition_companies move to owned_companies,
acquisition_proceeds become cash.
"""

cimport cython
from state cimport GameState, NUM_COMPANIES, NUM_CORPS
from data cimport (
    get_company_face_value, get_company_low_price, get_company_high_price
)

# Import shared helpers
from helpers.corp cimport get_president_of_corp

# Phase constants
DEF PHASE_ACQUISITION = 3
DEF PHASE_CLOSING = 4

# Corp constants
DEF CORP_OS = 2  # Overseas Trading

# Note: Action constants are defined in acquisition.pxd as cdef enum
# ACQ_ACTION_PASS = 50, ACQ_FI_ACTION_BUY = 0, ACQ_FI_ACTION_PASS = 1

# Re-export as Python module variables for test access
def get_action_constants():
    """Get action constants as a dict for Python access."""
    return {
        'ACQ_ACTION_PASS': ACQ_ACTION_PASS,
        'ACQ_FI_ACTION_BUY': ACQ_FI_ACTION_BUY,
        'ACQ_FI_ACTION_PASS': ACQ_FI_ACTION_PASS,
    }


# =============================================================================
# OFFER TUPLE STRUCTURE
# =============================================================================

cdef struct AcqOffer:
    int corp_id
    int company_id
    int corp_share_price  # For sorting
    int company_face_value  # For sorting
    bint is_fi_offer


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef bint corp_can_afford_fi_company(GameState state, int corp_id, int company_id) noexcept nogil:
    """Check if corp can afford to buy FI company."""
    cdef int price
    cdef int corp_cash = state.get_corp_cash(corp_id)

    if corp_id == CORP_OS:
        price = get_company_face_value(company_id)
    else:
        price = get_company_high_price(company_id)

    return corp_cash >= price


cdef int get_fi_purchase_price(int corp_id, int company_id) noexcept nogil:
    """Get the price a corp would pay for an FI company."""
    if corp_id == CORP_OS:
        return get_company_face_value(company_id)
    return get_company_high_price(company_id)


cdef void execute_fi_purchase(GameState state, int corp_id, int company_id) noexcept nogil:
    """Execute purchase of FI company by corp."""
    cdef int price = get_fi_purchase_price(corp_id, company_id)

    # Corp pays, FI receives
    state.add_corp_cash(corp_id, -price)
    state.add_fi_cash(price)

    # Company moves to corp's acquisition pile
    state.set_fi_owns_company(company_id, False)
    state.set_corp_acquisition_company(corp_id, company_id, True)


cdef void execute_corp_purchase(GameState state, int buyer_corp, int company_id, int price) noexcept nogil:
    """Execute purchase of company by corp from player or other corp."""
    cdef int seller_player, seller_corp, i

    # Find who owns the company
    # Check players first
    for i in range(state._num_players):
        if state.player_owns_company(i, company_id):
            # Player sells to corp
            state.set_player_owns_company(i, company_id, False)
            state.add_player_cash(i, price)
            state.add_corp_cash(buyer_corp, -price)
            state.set_corp_acquisition_company(buyer_corp, company_id, True)
            return

    # Check corps
    for i in range(NUM_CORPS):
        if i == buyer_corp:
            continue
        if state.corp_owns_company(i, company_id):
            # Corp sells to corp - seller gets acquisition_proceeds
            state.set_corp_owns_company(i, company_id, False)
            state.add_corp_acquisition_proceeds(i, price)
            state.add_corp_cash(buyer_corp, -price)
            state.set_corp_acquisition_company(buyer_corp, company_id, True)
            return


# =============================================================================
# FI OFFER GENERATION
# =============================================================================

cdef bint find_next_fi_offer(GameState state, int* out_corp, int* out_company) noexcept nogil:
    """
    Find the next FI offer to make.

    Iterates through FI companies (face value descending), then through corps
    in priority order (OS first, then share price descending).

    Returns True if an offer was found, False if no more FI offers.
    Sets out_corp and out_company to the offer details.
    """
    cdef int company_id, corp_id, i
    cdef int best_company = -1
    cdef int best_face_value = -1

    # Find FI companies, sorted by face value descending
    # We iterate in order and track the best not-yet-processed
    cdef int[36] fi_companies
    cdef int fi_count = 0

    for company_id in range(NUM_COMPANIES):
        if state.fi_owns_company(company_id):
            fi_companies[fi_count] = company_id
            fi_count += 1

    if fi_count == 0:
        return False

    # Sort FI companies by face value descending (simple insertion sort)
    cdef int j, temp
    for i in range(1, fi_count):
        j = i
        while j > 0 and get_company_face_value(fi_companies[j]) > get_company_face_value(fi_companies[j-1]):
            temp = fi_companies[j]
            fi_companies[j] = fi_companies[j-1]
            fi_companies[j-1] = temp
            j -= 1

    # Build corp priority list: OS first (if active), then by share price desc
    cdef int[8] corp_priority
    cdef int[8] corp_prices
    cdef int corp_count = 0

    # Add OS first if active
    if state.is_corp_active(CORP_OS):
        corp_priority[corp_count] = CORP_OS
        corp_prices[corp_count] = 9999  # Highest priority
        corp_count += 1

    # Add other active corps
    for corp_id in range(NUM_CORPS):
        if corp_id == CORP_OS:
            continue
        if state.is_corp_active(corp_id):
            corp_priority[corp_count] = corp_id
            corp_prices[corp_count] = state.get_corp_share_price(corp_id)
            corp_count += 1

    # Sort by price descending (insertion sort, skip OS at position 0)
    cdef int start_idx = 1 if (corp_count > 0 and corp_priority[0] == CORP_OS) else 0
    for i in range(start_idx + 1, corp_count):
        j = i
        while j > start_idx and corp_prices[j] > corp_prices[j-1]:
            temp = corp_priority[j]
            corp_priority[j] = corp_priority[j-1]
            corp_priority[j-1] = temp
            temp = corp_prices[j]
            corp_prices[j] = corp_prices[j-1]
            corp_prices[j-1] = temp
            j -= 1

    # Find first valid offer
    for i in range(fi_count):
        company_id = fi_companies[i]
        for j in range(corp_count):
            corp_id = corp_priority[j]
            if corp_can_afford_fi_company(state, corp_id, company_id):
                out_corp[0] = corp_id
                out_company[0] = company_id
                return True

    return False


# =============================================================================
# GENERAL ACQUISITION OFFER GENERATION
# =============================================================================

cdef bint is_valid_acquisition_target(GameState state, int buyer_corp, int company_id) noexcept nogil:
    """Check if company is a valid acquisition target for buyer corp."""
    cdef int president = get_president_of_corp(state, buyer_corp, state._num_players)
    if president < 0:
        return False  # Corp in receivership can't do general acquisitions

    cdef int low_price = get_company_low_price(company_id)
    cdef int corp_cash = state.get_corp_cash(buyer_corp)

    if corp_cash < low_price:
        return False

    # Check if president owns the company
    if state.player_owns_company(president, company_id):
        return True

    # Check if another corp (same president) owns it, and has 2+ companies
    cdef int other_corp
    for other_corp in range(NUM_CORPS):
        if other_corp == buyer_corp:
            continue
        if not state.is_corp_active(other_corp):
            continue
        if get_president_of_corp(state, other_corp, state._num_players) != president:
            continue
        if state.corp_owns_company(other_corp, company_id):
            if state.get_corp_company_count(other_corp) >= 2:
                return True

    return False


cdef bint find_next_general_offer(GameState state, int* out_corp, int* out_company) noexcept nogil:
    """
    Find the next general acquisition offer to make.

    Iterates through (buyer_corp, target_company) tuples ordered by:
    - Buyer share price descending
    - Target face value descending

    Returns True if an offer was found, False if no more offers.
    """
    cdef int corp_id, company_id, i, j
    cdef int best_corp = -1
    cdef int best_company = -1
    cdef int best_corp_price = -1
    cdef int best_face_value = -1

    # Iterate through corps by share price descending
    cdef int[8] corp_order
    cdef int[8] corp_prices
    cdef int corp_count = 0

    for corp_id in range(NUM_CORPS):
        if not state.is_corp_active(corp_id):
            continue
        if state.is_corp_in_receivership(corp_id):
            continue  # Receivership corps don't do general acquisitions
        corp_order[corp_count] = corp_id
        corp_prices[corp_count] = state.get_corp_share_price(corp_id)
        corp_count += 1

    # Sort by price descending
    cdef int temp
    for i in range(1, corp_count):
        j = i
        while j > 0 and corp_prices[j] > corp_prices[j-1]:
            temp = corp_order[j]
            corp_order[j] = corp_order[j-1]
            corp_order[j-1] = temp
            temp = corp_prices[j]
            corp_prices[j] = corp_prices[j-1]
            corp_prices[j-1] = temp
            j -= 1

    # For each corp, find valid targets ordered by face value descending
    for i in range(corp_count):
        corp_id = corp_order[i]
        best_company = -1
        best_face_value = -1

        for company_id in range(NUM_COMPANIES):
            if is_valid_acquisition_target(state, corp_id, company_id):
                if get_company_face_value(company_id) > best_face_value:
                    best_face_value = get_company_face_value(company_id)
                    best_company = company_id

        if best_company >= 0:
            out_corp[0] = corp_id
            out_company[0] = best_company
            return True

    return False


# =============================================================================
# ACQUISITION PHASE HANDLER
# =============================================================================

cdef class AcquisitionPhase:
    """
    Acquisition phase handler.

    Call setup_next_offer() to find and set up the next offer.
    If it returns True, call get_valid_actions() and do_action().
    Repeat until setup_next_offer() returns False.

    Attributes declared in acquisition.pxd.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players

    cpdef bint setup_next_offer(self, GameState state):
        """
        Find and set up the next acquisition offer.

        Returns True if an offer was set up (waiting for action).
        Returns False if no more offers (phase should end).

        For receivership corps buying from FI, auto-executes the purchase.
        """
        cdef int corp_id, company_id, president

        if state.get_phase() != PHASE_ACQUISITION:
            return False

        # Clear any previous offer
        state.clear_acq_offer()

        # First, handle FI offers
        while find_next_fi_offer(state, &corp_id, &company_id):
            # Check if receivership corp - auto-buy
            if state.is_corp_in_receivership(corp_id):
                execute_fi_purchase(state, corp_id, company_id)
                continue  # Look for next offer

            # Non-receivership corp - set up offer for NN decision
            state.set_acq_active_corp(corp_id)
            state.set_acq_target_company(company_id)
            state.set_acq_is_fi_offer(True)

            # Set active player to corp's president
            president = get_president_of_corp(state, corp_id, self._num_players)
            if president >= 0:
                state._set_active_player(president)

            return True

        # No more FI offers, try general acquisitions
        if find_next_general_offer(state, &corp_id, &company_id):
            state.set_acq_active_corp(corp_id)
            state.set_acq_target_company(company_id)
            state.set_acq_is_fi_offer(False)

            # Set active player to corp's president
            president = get_president_of_corp(state, corp_id, self._num_players)
            if president >= 0:
                state._set_active_player(president)

            return True

        # No more offers - phase ends
        return False

    cpdef bint is_waiting_for_action(self, GameState state):
        """Check if there's a pending offer waiting for action."""
        return state.get_acq_active_corp() >= 0

    cpdef bint can_do_action(self, GameState state, int action):
        """Check if action is valid for current offer."""
        cdef int corp_id = state.get_acq_active_corp()
        cdef int company_id = state.get_acq_target_company()

        if corp_id < 0 or company_id < 0:
            return False

        if state.is_acq_fi_offer():
            # FI offer: only buy (0) or pass (1)
            return action == ACQ_FI_ACTION_BUY or action == ACQ_FI_ACTION_PASS

        # General acquisition: price offset (0-49) or pass (50)
        if action == ACQ_ACTION_PASS:
            return True

        if action < 0 or action >= ACQ_ACTION_PASS:
            return False

        # Check if price is valid
        cdef int low_price = get_company_low_price(company_id)
        cdef int high_price = get_company_high_price(company_id)
        cdef int price = low_price + action

        if price > high_price:
            return False

        cdef int corp_cash = state.get_corp_cash(corp_id)
        return corp_cash >= price

    cpdef void do_action(self, GameState state, int action):
        """Execute the chosen action for current offer."""
        cdef int corp_id = state.get_acq_active_corp()
        cdef int company_id = state.get_acq_target_company()
        cdef int price

        if corp_id < 0 or company_id < 0:
            raise ValueError("No active acquisition offer")

        if state.is_acq_fi_offer():
            if action == ACQ_FI_ACTION_BUY:
                execute_fi_purchase(state, corp_id, company_id)
            # else: pass, do nothing
        else:
            if action != ACQ_ACTION_PASS:
                # Execute purchase at given price offset
                price = get_company_low_price(company_id) + action
                execute_corp_purchase(state, corp_id, company_id, price)
            # else: pass, do nothing

        # Clear offer state
        state.clear_acq_offer()

    cpdef list get_valid_actions(self, GameState state):
        """Get list of valid actions for current offer."""
        cdef list actions = []
        cdef int i

        if not self.is_waiting_for_action(state):
            return actions

        if state.is_acq_fi_offer():
            # FI offers: buy or pass
            if self.can_do_action(state, ACQ_FI_ACTION_BUY):
                actions.append(ACQ_FI_ACTION_BUY)
            actions.append(ACQ_FI_ACTION_PASS)
        else:
            # General: price offsets + pass
            for i in range(ACQ_ACTION_PASS):
                if self.can_do_action(state, i):
                    actions.append(i)
            actions.append(ACQ_ACTION_PASS)

        return actions

    cpdef void transition_to_closing(self, GameState state):
        """Finalize acquisitions and transition to Closing phase."""
        # Move acquisition companies to owned, proceeds to cash
        state.finalize_acquisitions()

        # Clear offer state
        state.clear_acq_offer()

        # Transition to Closing
        state.set_phase(PHASE_CLOSING)
        state._set_active_player(0)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create AcquisitionPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = AcquisitionPhase(num_players)
    return _phase_handlers[num_players]


def handle_acquisition_phase(GameState state):
    """
    Run the acquisition phase until complete.

    This is a convenience function for testing. In actual training,
    the MCTS will call setup_next_offer(), get_valid_actions(), and do_action().
    """
    handler = get_phase_handler(state._num_players)

    while handler.setup_next_offer(state):
        # For testing, just pass on everything
        if state.is_acq_fi_offer():
            handler.do_action(state, ACQ_FI_ACTION_PASS)
        else:
            handler.do_action(state, ACQ_ACTION_PASS)

    handler.transition_to_closing(state)
