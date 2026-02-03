# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
ACQUISITION phase: Corps acquire companies from FI, other corps, and players.

DESIGN: One-by-One Offer Presentation
=====================================
Instead of exposing all acquisition opportunities simultaneously (which would
create a huge action space of corp × company × price combinations), offers are:

1. Generated once at phase entry into a hidden buffer (_generate_offers)
2. Sorted by priority: OS→FI, Corp→FI by price, Corp→Corp, Corp→Player
3. Presented one at a time via visible state (acq_active_corp, acq_target_company)
4. Advanced sequentially after accept/pass (_advance_to_next_offer)
5. Re-validated dynamically since earlier buys may invalidate later offers

This keeps the action space constant (51 prices + 2 FI actions + pass = 54 actions)
regardless of how many potential acquisitions exist in a given game state.

Receivership corps (no president) are handled automatically:
- Auto-buy from FI at face value if affordable (RECV-01)
- Auto-pass on non-FI offers (RECV-02: can only buy from FI)

See CLAUDE.md "Offer Buffer Pattern" for full documentation.
"""

from core.state cimport GameState
from core.driver cimport _is_game_terminal
from core.data cimport GamePhases, GameConstants, CorpIndices, get_company_face_value, get_company_low_price
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_ACQ_PRICE, ACTION_ACQ_FI_HIGH, ACTION_ACQ_FI_FACE
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
from core.data import CORP_NAMES, CORP_NAME_TO_ID

# Buffer size constant (DEF required for static array sizing - cannot be imported)
DEF OFFER_BUFFER_SIZE = 250

# Company location constants (from entities/company.pxd)
DEF LOC_PLAYER = 3
DEF LOC_FI = 4
DEF LOC_CORP = 5


# =============================================================================
# QUICKSORT HELPERS
# =============================================================================

cdef void _qsort_desc_3(int* keys, int* arr1, int* arr2, int lo, int hi) noexcept nogil:
    """
    Quicksort 3 parallel arrays by keys in descending order.

    Used for FI offers sorted by corp share price (descending).
    Arrays: keys=prices, arr1=corp_ids, arr2=company_ids
    """
    if lo >= hi:
        return

    cdef int pivot = keys[(lo + hi) // 2]
    cdef int i = lo
    cdef int j = hi
    cdef int tmp

    while i <= j:
        while keys[i] > pivot:
            i += 1
        while keys[j] < pivot:
            j -= 1
        if i <= j:
            # Swap all three arrays
            tmp = keys[i]; keys[i] = keys[j]; keys[j] = tmp
            tmp = arr1[i]; arr1[i] = arr1[j]; arr1[j] = tmp
            tmp = arr2[i]; arr2[i] = arr2[j]; arr2[j] = tmp
            i += 1
            j -= 1

    _qsort_desc_3(keys, arr1, arr2, lo, j)
    _qsort_desc_3(keys, arr1, arr2, i, hi)


cdef void _qsort_price_fv_4(int* prices, int* fvs, int* arr1, int* arr2, int lo, int hi) noexcept nogil:
    """
    Quicksort 4 parallel arrays by (price DESC, face_value DESC).

    Used for corp-corp and corp-player offers.
    Arrays: prices=sort key 1, fvs=sort key 2, arr1=corp_ids, arr2=company_ids

    Comparison: (price_a, fv_a) < (price_b, fv_b) if:
      - price_a < price_b, OR
      - price_a == price_b AND fv_a < fv_b
    (We want higher prices first, then higher face values)
    """
    if lo >= hi:
        return

    cdef int pivot_price = prices[(lo + hi) // 2]
    cdef int pivot_fv = fvs[(lo + hi) // 2]
    cdef int i = lo
    cdef int j = hi
    cdef int tmp

    while i <= j:
        # Move i right while element should come before pivot
        while prices[i] > pivot_price or (prices[i] == pivot_price and fvs[i] > pivot_fv):
            i += 1
        # Move j left while element should come after pivot
        while prices[j] < pivot_price or (prices[j] == pivot_price and fvs[j] < pivot_fv):
            j -= 1
        if i <= j:
            # Swap all four arrays
            tmp = prices[i]; prices[i] = prices[j]; prices[j] = tmp
            tmp = fvs[i]; fvs[i] = fvs[j]; fvs[j] = tmp
            tmp = arr1[i]; arr1[i] = arr1[j]; arr1[j] = tmp
            tmp = arr2[i]; arr2[i] = arr2[j]; arr2[j] = tmp
            i += 1
            j -= 1

    _qsort_price_fv_4(prices, fvs, arr1, arr2, lo, j)
    _qsort_price_fv_4(prices, fvs, arr1, arr2, i, hi)


def qsort_desc_3_py(list keys, list arr1, list arr2):
    """Python wrapper for testing _qsort_desc_3."""
    cdef int n = len(keys)
    if n == 0:
        return keys, arr1, arr2

    cdef int c_keys[256]
    cdef int c_arr1[256]
    cdef int c_arr2[256]
    cdef int i

    for i in range(n):
        c_keys[i] = keys[i]
        c_arr1[i] = arr1[i]
        c_arr2[i] = arr2[i]

    if n > 1:
        _qsort_desc_3(c_keys, c_arr1, c_arr2, 0, n - 1)

    return ([c_keys[i] for i in range(n)],
            [c_arr1[i] for i in range(n)],
            [c_arr2[i] for i in range(n)])


def qsort_price_fv_4_py(list prices, list fvs, list arr1, list arr2):
    """Python wrapper for testing _qsort_price_fv_4."""
    cdef int n = len(prices)
    if n == 0:
        return prices, fvs, arr1, arr2

    cdef int c_prices[256]
    cdef int c_fvs[256]
    cdef int c_arr1[256]
    cdef int c_arr2[256]
    cdef int i

    for i in range(n):
        c_prices[i] = prices[i]
        c_fvs[i] = fvs[i]
        c_arr1[i] = arr1[i]
        c_arr2[i] = arr2[i]

    if n > 1:
        _qsort_price_fv_4(c_prices, c_fvs, c_arr1, c_arr2, 0, n - 1)

    return ([c_prices[i] for i in range(n)],
            [c_fvs[i] for i in range(n)],
            [c_arr1[i] for i in range(n)],
            [c_arr2[i] for i in range(n)])


# =============================================================================
# OFFER GENERATION HELPERS
# =============================================================================

cdef int _collect_fi_offers(GameState state, int* corp_ids, int* company_ids) noexcept:
    """
    Collect Corp->FI offers. Returns count.
    OS->FI offers come first (by face value DESC), then other corps by descending share price.
    Within each corp's offers, companies are sorted by face value DESC (most expensive first).
    Only active corps that can afford at least one FI company are included.
    """
    cdef int count = 0
    cdef int company_id, corp_id
    cdef int corp_cash, high_price
    cdef int temp_count = 0
    cdef int temp_corp_ids[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_prices[OFFER_BUFFER_SIZE]
    cdef int temp_fvs[OFFER_BUFFER_SIZE]
    cdef int i, face_value

    # First pass: OS->FI offers (OS always first if can afford)
    # Iterate in descending face value order (most expensive first)
    for company_id in range(<int>GameConstants.NUM_COMPANIES - 1, -1, -1):
        if fi_module.FI.owns_company(state, company_id):
            high_price = company_module.COMPANIES[company_id].get_high_price()

            # Check if OS can afford (OS always first if affordable)
            if corp_module.CORPS[CorpIndices.CORP_OS].is_active(state):
                corp_cash = corp_module.CORPS[CorpIndices.CORP_OS].get_cash(state)
                if corp_cash >= high_price and count < OFFER_BUFFER_SIZE:
                    corp_ids[count] = CorpIndices.CORP_OS
                    company_ids[count] = company_id
                    count += 1

    # Second pass: collect other corps (not OS) into temp arrays
    # Iterate in descending face value order
    for company_id in range(<int>GameConstants.NUM_COMPANIES - 1, -1, -1):
        if fi_module.FI.owns_company(state, company_id):
            high_price = company_module.COMPANIES[company_id].get_high_price()
            face_value = get_company_face_value(company_id)

            # Check all other corps (skip OS)
            for corp_id in range(<int>GameConstants.NUM_CORPS):
                if corp_id == CorpIndices.CORP_OS:
                    continue

                if corp_module.CORPS[corp_id].is_active(state):
                    corp_cash = corp_module.CORPS[corp_id].get_cash(state)
                    if corp_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                        temp_corp_ids[temp_count] = corp_id
                        temp_company_ids[temp_count] = company_id
                        temp_prices[temp_count] = corp_module.CORPS[corp_id].get_share_price(state)
                        temp_fvs[temp_count] = face_value
                        temp_count += 1

    # Quicksort by (share price DESC, face value DESC)
    if temp_count > 1:
        _qsort_price_fv_4(temp_prices, temp_fvs, temp_corp_ids, temp_company_ids, 0, temp_count - 1)

    # Append sorted non-OS offers to output
    for i in range(temp_count):
        if count < OFFER_BUFFER_SIZE:
            corp_ids[count] = temp_corp_ids[i]
            company_ids[count] = temp_company_ids[i]
            count += 1

    return count


cdef int _collect_corp_corp_offers(GameState state, int* corp_ids, int* company_ids) noexcept:
    """
    Collect Corp->Corp offers where same player is president of both.
    Sorted by (buyer share price DESC, target face value DESC).

    Note: Receivership corps are automatically excluded as sellers because
    get_president_id returns -1 for receivership, which never matches
    any player_id (0 to num_players-1). This implements RECV-02.
    """
    cdef int count = 0
    cdef int player_id, buyer_corp, seller_corp, company_id
    cdef int buyer_cash, high_price, buyer_price, face_value
    cdef int temp_count = 0
    cdef int temp_buyer_corps[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_buyer_prices[OFFER_BUFFER_SIZE]
    cdef int temp_face_values[OFFER_BUFFER_SIZE]
    cdef int i

    # For each player, find corps they control and generate offers
    for player_id in range(state._num_players):
        # Find all corps this player is president of
        for buyer_corp in range(<int>GameConstants.NUM_CORPS):
            if not corp_module.CORPS[buyer_corp].is_active(state):
                continue
            if corp_module.CORPS[buyer_corp].get_president_id(state) != player_id:
                continue

            buyer_cash = corp_module.CORPS[buyer_corp].get_cash(state)
            buyer_price = corp_module.CORPS[buyer_corp].get_share_price(state)

            # Find all other corps this player is president of
            for seller_corp in range(<int>GameConstants.NUM_CORPS):
                if seller_corp == buyer_corp:
                    continue
                if not corp_module.CORPS[seller_corp].is_active(state):
                    continue
                if corp_module.CORPS[seller_corp].get_president_id(state) != player_id:
                    continue

                # Find companies owned by seller corp
                for company_id in range(<int>GameConstants.NUM_COMPANIES):
                    if corp_module.CORPS[seller_corp].owns_company(state, company_id):
                        high_price = company_module.COMPANIES[company_id].get_high_price()
                        face_value = get_company_face_value(company_id)

                        if buyer_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                            temp_buyer_corps[temp_count] = buyer_corp
                            temp_company_ids[temp_count] = company_id
                            temp_buyer_prices[temp_count] = buyer_price
                            temp_face_values[temp_count] = face_value
                            temp_count += 1

    # Quicksort by (buyer price DESC, face value DESC)
    if temp_count > 1:
        _qsort_price_fv_4(temp_buyer_prices, temp_face_values, temp_buyer_corps, temp_company_ids, 0, temp_count - 1)

    # Copy sorted results to output
    for i in range(temp_count):
        if count < OFFER_BUFFER_SIZE:
            corp_ids[count] = temp_buyer_corps[i]
            company_ids[count] = temp_company_ids[i]
            count += 1

    return count


cdef int _collect_player_private_offers(GameState state, int* corp_ids, int* company_ids) noexcept:
    """
    Collect Corp->Player private company offers.
    All corps controlled by a player can bid on each private company owned by that player.
    Sorted by (buyer share price DESC, target face value DESC).
    """
    cdef int count = 0
    cdef int player_id, corp_id, company_id
    cdef int corp_cash, high_price, corp_price, face_value
    cdef int temp_count = 0
    cdef int temp_corp_ids[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_corp_prices[OFFER_BUFFER_SIZE]
    cdef int temp_face_values[OFFER_BUFFER_SIZE]
    cdef int i

    # For each player, find their private companies and corps they control
    for player_id in range(state._num_players):
        # Find all private companies owned by this player
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if player_module.PLAYERS[player_id].owns_company(state, company_id):
                high_price = company_module.COMPANIES[company_id].get_high_price()
                face_value = get_company_face_value(company_id)

                # Find all corps this player is president of
                for corp_id in range(<int>GameConstants.NUM_CORPS):
                    if not corp_module.CORPS[corp_id].is_active(state):
                        continue
                    if corp_module.CORPS[corp_id].get_president_id(state) != player_id:
                        continue

                    corp_cash = corp_module.CORPS[corp_id].get_cash(state)
                    corp_price = corp_module.CORPS[corp_id].get_share_price(state)

                    if corp_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                        temp_corp_ids[temp_count] = corp_id
                        temp_company_ids[temp_count] = company_id
                        temp_corp_prices[temp_count] = corp_price
                        temp_face_values[temp_count] = face_value
                        temp_count += 1

    # Quicksort by (corp price DESC, face value DESC)
    if temp_count > 1:
        _qsort_price_fv_4(temp_corp_prices, temp_face_values, temp_corp_ids, temp_company_ids, 0, temp_count - 1)

    # Copy sorted results to output
    for i in range(temp_count):
        if count < OFFER_BUFFER_SIZE:
            corp_ids[count] = temp_corp_ids[i]
            company_ids[count] = temp_company_ids[i]
            count += 1

    return count


# =============================================================================
# MAIN OFFER GENERATION
# =============================================================================

cdef void _generate_offers(GameState state) noexcept:
    """
    Generate all valid acquisition offers and store in hidden buffer.

    Priority order:
    1. OS->FI offers by face value DESC (OFFER-02)
    2. Other Corp->FI offers by (share price DESC, face value DESC) (OFFER-03)
    3. Corp->Corp offers by (buyer price DESC, face value DESC) (OFFER-04)
    4. Corp->Player private offers by (buyer price DESC, face value DESC) (OFFER-05)

    Stores in hidden state buffer: [offer_count][offer_index][buffer...]
    """
    cdef int temp_corp_ids[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int offer_count = 0
    cdef int count, i, base

    # Initialize offer count and index to 0
    state._data[state._layout.hidden_offer_count_offset] = 0.0
    state._data[state._layout.hidden_offer_index_offset] = 0.0

    # Collect FI offers (OS first, then others by price)
    count = _collect_fi_offers(state, temp_corp_ids, temp_company_ids)
    for i in range(count):
        if offer_count < OFFER_BUFFER_SIZE:
            base = state._layout.hidden_offer_buffer_offset + (offer_count * 2)
            state._data[base] = <float>temp_corp_ids[i]
            state._data[base + 1] = <float>temp_company_ids[i]
            offer_count += 1

    # Collect Corp->Corp offers
    count = _collect_corp_corp_offers(state, temp_corp_ids, temp_company_ids)
    for i in range(count):
        if offer_count < OFFER_BUFFER_SIZE:
            base = state._layout.hidden_offer_buffer_offset + (offer_count * 2)
            state._data[base] = <float>temp_corp_ids[i]
            state._data[base + 1] = <float>temp_company_ids[i]
            offer_count += 1

    # Collect Corp->Player private offers
    count = _collect_player_private_offers(state, temp_corp_ids, temp_company_ids)
    for i in range(count):
        if offer_count < OFFER_BUFFER_SIZE:
            base = state._layout.hidden_offer_buffer_offset + (offer_count * 2)
            state._data[base] = <float>temp_corp_ids[i]
            state._data[base + 1] = <float>temp_company_ids[i]
            offer_count += 1

    # Write final offer count
    state._data[state._layout.hidden_offer_count_offset] = <float>offer_count


def generate_offers_py(GameState state):
    """Python wrapper for testing offer generation."""
    _generate_offers(state)


cpdef tuple get_offer_at(GameState state, int index):
    """Get (corp_id, company_id) at buffer index, or (-1, -1) if invalid."""
    cdef int count = <int>state._data[state._layout.hidden_offer_count_offset]
    if index < 0 or index >= count:
        return (-1, -1)
    cdef int base = state._layout.hidden_offer_buffer_offset + (index * 2)
    return (<int>state._data[base], <int>state._data[base + 1])


cpdef int get_offer_count(GameState state):
    """Get number of offers in buffer."""
    return <int>state._data[state._layout.hidden_offer_count_offset]


# =============================================================================
# OFFER VALIDATION AND PRESENTATION
# =============================================================================

cdef bint _is_offer_valid(GameState state, int corp_id, int company_id) noexcept:
    """
    Check if offer is still valid for presentation.

    Invalid if:
    - Seller no longer owns the company (was acquired earlier this phase)
    - Corp doesn't have enough cash for minimum price
    - Ownership relationship no longer valid (president changed)

    Returns True if offer is valid.
    """
    cdef int price, corp_cash
    cdef int location, owner_id, buyer_president, seller_president

    # Check if FI still owns it (for FI offers)
    if fi_module.FI.owns_company(state, company_id):
        # FI offer: check cash (OS pays face, others pay high)
        corp_cash = corp_module.CORPS[corp_id].get_cash(state)
        if corp_id == CorpIndices.CORP_OS:
            price = get_company_face_value(company_id)
        else:
            price = company_module.COMPANIES[company_id].get_high_price()
        return corp_cash >= price

    # Non-FI: check current location and ownership relationship
    location = company_module.COMPANIES[company_id].get_location(state)
    owner_id = company_module.COMPANIES[company_id].get_owner_id(state)
    buyer_president = corp_module.CORPS[corp_id].get_president_id(state)

    # Check buyer can afford minimum price (low_price for non-FI)
    corp_cash = corp_module.CORPS[corp_id].get_cash(state)
    price = get_company_low_price(company_id)
    if corp_cash < price:
        return False

    if location == LOC_CORP:
        # Corp-corp: seller corp's president must match buyer's president
        seller_president = corp_module.CORPS[owner_id].get_president_id(state)
        if seller_president != buyer_president or seller_president < 0:
            return False
    elif location == LOC_PLAYER:
        # Player-corp: player must be buyer corp's president
        if owner_id != buyer_president:
            return False
    else:
        # Company not owned by corp or player (acquired, removed, etc.)
        return False

    return True


cdef void _present_current_offer(GameState state) noexcept:
    """
    Update visible state to reflect current offer in buffer.

    For receivership corps:
    - FI offers: auto-execute buy if affordable at HIGH price, else auto-pass
    - Non-FI offers: auto-pass (receivership can only buy from FI per RULES.md)

    Loops until a player-president offer is found or offers exhausted.

    STATE-01: Sets visible acquisition state for current offer.
    STATE-04: Clears acq_active_corp when no more offers.
    RECV-01: Receivership corps auto-buy FI offers if affordable at HIGH price.
    RECV-03: Auto-buy executes within this loop (no player action).
    """
    cdef int count = <int>state._data[state._layout.hidden_offer_count_offset]
    cdef int index = <int>state._data[state._layout.hidden_offer_index_offset]
    cdef int corp_id, company_id, president, base
    cdef int high_price, corp_cash
    cdef bint is_fi_offer

    while index < count:
        base = state._layout.hidden_offer_buffer_offset + (index * 2)
        corp_id = <int>state._data[base]
        company_id = <int>state._data[base + 1]

        # Skip invalid offers (already acquired, insufficient cash, etc.)
        if not _is_offer_valid(state, corp_id, company_id):
            index += 1
            state._data[state._layout.hidden_offer_index_offset] = <float>index
            continue

        # Check if buying corp is in receivership
        if corp_module.CORPS[corp_id].is_in_receivership(state):
            is_fi_offer = fi_module.FI.owns_company(state, company_id)

            # Receivership corps only buy from FI (per RULES.md)
            if is_fi_offer:
                # Per RULES.md: FI "Only sells at maximum allowed price"
                # Only OS can pay face value - receivership pays high price
                high_price = company_module.COMPANIES[company_id].get_high_price()
                corp_cash = corp_module.CORPS[corp_id].get_cash(state)

                if corp_cash >= high_price:
                    # Auto-execute: buy at high price
                    _execute_receivership_fi_buy(state, corp_id, company_id)
            # else: auto-pass by falling through

            # Advance to next offer (auto-pass for both unaffordable FI and non-FI)
            index += 1
            state._data[state._layout.hidden_offer_index_offset] = <float>index
            continue

        # Found player-president offer - set visible state and return
        turn_module.TURN.set_acq_active_corp(state, corp_id)
        turn_module.TURN.set_acq_target_company(state, company_id)
        turn_module.TURN.set_acq_fi_offer(state, fi_module.FI.owns_company(state, company_id))

        president = corp_module.CORPS[corp_id].get_president_id(state)
        state._set_active_player(president if president >= 0 else 0)
        return

    # No more valid offers (STATE-04)
    turn_module.TURN.clear_acq_active_corp(state)
    turn_module.TURN.clear_acq_target_company(state)
    turn_module.TURN.set_acq_fi_offer(state, False)


cdef void _advance_to_next_offer(GameState state) noexcept:
    """
    Advance offer index and present next offer.

    Called after accept or pass on current offer.
    """
    cdef int index = <int>state._data[state._layout.hidden_offer_index_offset]
    state._data[state._layout.hidden_offer_index_offset] = <float>(index + 1)
    _present_current_offer(state)


def present_current_offer_py(GameState state):
    """Python wrapper for testing."""
    _present_current_offer(state)


def advance_to_next_offer_py(GameState state):
    """Python wrapper for testing."""
    _advance_to_next_offer(state)


cpdef int get_offer_index(GameState state):
    """Get current offer index."""
    return <int>state._data[state._layout.hidden_offer_index_offset]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

cdef bint _is_target_already_acquired(GameState state, int company_id) noexcept:
    """
    Check if target company is already in any corp's acquisition_companies.

    VALID-04: Defensive check - offer generation filters this, but re-verify at action time.
    Returns True if already acquired, False otherwise.
    """
    cdef int check_corp
    for check_corp in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[check_corp].has_acquisition_company(state, company_id):
            return True
    return False


cdef int _count_seller_companies(GameState state, int seller_corp_id, int target_company_id) noexcept:
    """
    Count companies seller retains after selling target.

    Counts:
    - Companies in seller's owned_companies (excluding target)
    - Companies in seller's acquisition_companies (excluding target)

    Returns total count.
    """
    cdef int count = 0
    cdef int company_id

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_id == target_company_id:
            continue
        if corp_module.CORPS[seller_corp_id].owns_company(state, company_id):
            count += 1
        if corp_module.CORPS[seller_corp_id].has_acquisition_company(state, company_id):
            count += 1

    return count


cdef bint _validate_price_action(GameState state, int price) noexcept:
    """
    Validate price-based acquisition action.

    Checks:
    - VALID-01: Price in [low_price, high_price] range
    - VALID-02: Corp has sufficient cash
    - VALID-03: Corp seller retains >= 1 company after sale
    - VALID-04: Target not already acquired
    - VALID-05: Target not already in buyer's owned_companies
    - VALID-06: Same-president (guaranteed by offer generation, no runtime check)

    Returns True if valid, False otherwise.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef bint is_fi = turn_module.TURN.is_acq_fi_offer(state)
    cdef int low_price, high_price, corp_cash
    cdef int location, seller_corp_id

    # Get price bounds
    low_price = get_company_low_price(company_id)
    high_price = company_module.COMPANIES[company_id].get_high_price()

    # VALID-01: Price in range
    if price < low_price or price > high_price:
        return False

    # VALID-02: Corp can afford
    corp_cash = corp_module.CORPS[corp_id].get_cash(state)
    if corp_cash < price:
        return False

    # VALID-04: Target not already acquired
    if _is_target_already_acquired(state, company_id):
        return False

    # VALID-05: Target not already in buyer's owned_companies
    if corp_module.CORPS[corp_id].owns_company(state, company_id):
        return False

    # VALID-03: Seller retains >= 1 company (only for corp sellers, not FI or players)
    if not is_fi:
        location = company_module.COMPANIES[company_id].get_location(state)
        if location == LOC_CORP:
            seller_corp_id = company_module.COMPANIES[company_id].get_owner_id(state)
            if _count_seller_companies(state, seller_corp_id, company_id) < 1:
                return False

    return True


cdef bint _validate_fi_buy_high(GameState state) noexcept:
    """
    Validate FI Buy High action (non-OS corps buying at high price).

    Checks:
    - Defensive: is_acq_fi_offer is True
    - Corp is not OS (OS must use face value)
    - VALID-02: Corp has sufficient cash for high_price
    - VALID-04: Target not already acquired

    Returns True if valid, False otherwise.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef bint is_fi = turn_module.TURN.is_acq_fi_offer(state)

    # Defensive: Must be FI offer
    if not is_fi:
        return False

    # OS cannot use FI Buy High
    if corp_id == CorpIndices.CORP_OS:
        return False

    # VALID-02: Corp can afford high price
    cdef int high_price = company_module.COMPANIES[company_id].get_high_price()
    cdef int corp_cash = corp_module.CORPS[corp_id].get_cash(state)
    if corp_cash < high_price:
        return False

    # VALID-04: Target not already acquired
    if _is_target_already_acquired(state, company_id):
        return False

    return True


cdef bint _validate_fi_buy_face(GameState state) noexcept:
    """
    Validate FI Buy Face action (OS only, buying at face value).

    Checks:
    - Defensive: is_acq_fi_offer is True
    - Corp is OS (only OS uses face value)
    - VALID-02: Corp has sufficient cash for face_value
    - VALID-04: Target not already acquired

    Returns True if valid, False otherwise.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef bint is_fi = turn_module.TURN.is_acq_fi_offer(state)

    # Defensive: Must be FI offer
    if not is_fi:
        return False

    # Only OS can use FI Buy Face
    if corp_id != CorpIndices.CORP_OS:
        return False

    # VALID-02: Corp can afford face value
    cdef int face_value = get_company_face_value(company_id)
    cdef int corp_cash = corp_module.CORPS[corp_id].get_cash(state)
    if corp_cash < face_value:
        return False

    # VALID-04: Target not already acquired
    if _is_target_already_acquired(state, company_id):
        return False

    return True


# =============================================================================
# ACTION HANDLERS
# =============================================================================

cdef void _handle_accept_price(GameState state, int price) noexcept:
    """
    Execute price-based acquisition (non-FI offers).

    Transfers:
    - Money from buyer corp to seller (corp or player)
    - Company to buyer's acquisition zone

    Then advances to next offer.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef int location, seller_id, current_proceeds

    # Determine seller from company location
    location = company_module.COMPANIES[company_id].get_location(state)
    seller_id = company_module.COMPANIES[company_id].get_owner_id(state)

    # Buyer pays
    corp_module.CORPS[corp_id].add_cash(state, -price)

    # Seller receives (to acquisition_proceeds)
    if location == LOC_CORP:
        # Corp seller: use get+set pattern (no add_acquisition_proceeds method)
        current_proceeds = corp_module.CORPS[seller_id].get_acquisition_proceeds(state)
        corp_module.CORPS[seller_id].set_acquisition_proceeds(state, current_proceeds + price)
    elif location == LOC_PLAYER:
        # Player seller: has add_acquisition_proceeds method
        player_module.PLAYERS[seller_id].add_acquisition_proceeds(state, price)

    # Transfer company to buyer's acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)

    # Advance to next offer
    _advance_to_next_offer(state)


cdef void _handle_fi_buy_high(GameState state) noexcept:
    """
    Execute FI purchase at high price (non-OS corps).

    Transfers:
    - high_price from buyer corp to FI
    - Company to buyer's acquisition zone

    Then advances to next offer.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef int high_price = company_module.COMPANIES[company_id].get_high_price()

    # Transfer money: buyer -> FI
    corp_module.CORPS[corp_id].add_cash(state, -high_price)
    fi_module.FI.add_cash(state, high_price)

    # Transfer company to buyer's acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)

    # Advance to next offer
    _advance_to_next_offer(state)


cdef void _handle_fi_buy_face(GameState state) noexcept:
    """
    Execute FI purchase at face value (OS only).

    Transfers:
    - face_value from OS to FI
    - Company to OS's acquisition zone

    Then advances to next offer.
    """
    cdef int corp_id = turn_module.TURN.get_acq_active_corp(state)
    cdef int company_id = turn_module.TURN.get_acq_target_company(state)
    cdef int face_value = get_company_face_value(company_id)

    # Transfer money: OS -> FI
    corp_module.CORPS[corp_id].add_cash(state, -face_value)
    fi_module.FI.add_cash(state, face_value)

    # Transfer company to OS's acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)

    # Advance to next offer
    _advance_to_next_offer(state)


cdef void _handle_pass(GameState state) noexcept:
    """
    Pass on current offer, advance to next.

    Pass permanently skips current offer - index advances and next offer is presented.
    """
    _advance_to_next_offer(state)


cdef void _execute_receivership_fi_buy(GameState state, int corp_id, int company_id) noexcept:
    """
    Execute FI purchase for receivership corp at HIGH price.

    Per RULES.md: FI "Only sells at maximum allowed price".
    Only OS has the special ability to pay face value - receivership corps pay full price.
    This is called from _present_current_offer for receivership auto-buy.
    Does NOT advance offer index - caller handles that.
    """
    cdef int high_price = company_module.COMPANIES[company_id].get_high_price()

    # Transfer money: corp -> FI
    corp_module.CORPS[corp_id].add_cash(state, -high_price)
    fi_module.FI.add_cash(state, high_price)

    # Transfer company to corp's acquisition zone
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)


# =============================================================================
# PHASE ENTRY SETUP
# =============================================================================

cpdef void setup_acquisition_phase(GameState state):
    """
    Set up ACQUISITION phase at entry.

    Called from WRAP_UP before transitioning to ACQUISITION.
    Per CONTEXT.md: "Offer buffer populated at phase entry (not lazily)"

    Steps:
    1. Clear offer buffer index to 0
    2. Generate all offers into buffer (OFFER-01)
    3. Present first valid offer (or clear state if none)

    Does NOT clear acquisition zones - that happens at phase EXIT (after merge).
    """
    # Reset offer tracking
    state._data[state._layout.hidden_offer_index_offset] = 0.0
    state._data[state._layout.hidden_offer_count_offset] = 0.0

    # Generate offers (populates buffer)
    _generate_offers(state)

    # Present first offer (or clear if none)
    _present_current_offer(state)


def setup_acquisition_phase_py(GameState state):
    """Python wrapper for testing."""
    setup_acquisition_phase(state)


def apply_acquisition_action_py(GameState state, int action_type, int amount=0):
    """Python wrapper for testing."""
    cdef ActionInfo info
    info.action_type = action_type
    info.amount = amount
    return apply_acquisition_action(state, &info)


def merge_acquisition_zones_py(GameState state):
    """Python wrapper for testing zone merging."""
    _merge_acquisition_zones(state)


def transition_to_closing_py(GameState state):
    """Python wrapper for testing phase transition."""
    _transition_to_closing(state)


# =============================================================================
# ZONE MERGING (Phase 14)
# =============================================================================

cdef void _merge_player_proceeds(GameState state) noexcept:
    """
    Merge player acquisition_proceeds into cash, then clear.

    FLOW-04: Player proceeds merge at phase end.
    """
    cdef int player_id, proceeds
    for player_id in range(state._num_players):
        proceeds = player_module.PLAYERS[player_id].get_acquisition_proceeds(state)
        if proceeds > 0:
            player_module.PLAYERS[player_id].add_cash(state, proceeds)
            player_module.PLAYERS[player_id].clear_acquisition_proceeds(state)


cdef void _merge_corp_proceeds(GameState state) noexcept:
    """
    Merge corp acquisition_proceeds into cash, then clear.

    FLOW-04: Corp proceeds merge at phase end.
    """
    cdef int corp_id, proceeds
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[corp_id].is_active(state):
            proceeds = corp_module.CORPS[corp_id].get_acquisition_proceeds(state)
            if proceeds > 0:
                corp_module.CORPS[corp_id].add_cash(state, proceeds)
                corp_module.CORPS[corp_id].set_acquisition_proceeds(state, 0)


cdef void _merge_corp_companies(GameState state) noexcept:
    """
    Transfer acquisition_companies to owned_companies, then clear.

    FLOW-03: Companies in acquisition zone merge to owned at phase end.

    Uses Company.transfer_to_corp() which:
    - Clears acquisition_company flag (via clear_location)
    - Sets owned_company flag
    - Updates company location and owner
    """
    cdef int corp_id, company_id
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[corp_id].is_active(state):
            for company_id in range(<int>GameConstants.NUM_COMPANIES):
                if corp_module.CORPS[corp_id].has_acquisition_company(state, company_id):
                    # transfer_to_corp handles all state updates:
                    # - clears acquisition flag (via clear_location)
                    # - sets owned_company flag
                    # - updates location to LOC_CORP and owner_id
                    company_module.COMPANIES[company_id].transfer_to_corp(state, corp_id)


cdef void _merge_acquisition_zones(GameState state) noexcept:
    """
    Merge all acquisition zones into final state.

    Called before transitioning to CLOSING phase.
    Order: proceeds first (both player and corp), then companies.
    """
    _merge_player_proceeds(state)
    _merge_corp_proceeds(state)
    _merge_corp_companies(state)


cdef void _transition_to_closing(GameState state) noexcept:
    """
    Complete ACQUISITION phase and transition to CLOSING.

    FLOW-02: Transition when no more valid offers.
    DRIVER-03: Phase handler transitions internally.

    Steps:
    1. Merge all acquisition zones (proceeds + companies)
    2. Transition to CLOSING phase (auto-close executes first)
    """
    # Merge before leaving ACQUISITION
    _merge_acquisition_zones(state)

    # Transition to CLOSING (auto-close will execute, then offers in Phase 17)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_CLOSING)


# =============================================================================
# MAIN ACTION HANDLER
# =============================================================================

cdef int apply_acquisition_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply ACQUISITION phase action to state.

    Action types:
    - ACTION_ACQ_PRICE: Buy at low_price + info.amount
    - ACTION_ACQ_FI_HIGH: Buy FI company at high price (non-OS)
    - ACTION_ACQ_FI_FACE: Buy FI company at face value (OS only)
    - ACTION_PASS: Decline current offer

    Returns: 0=success, 1=invalid
    """
    cdef int company_id, low_price, price

    if info.action_type == ACTION_ACQ_PRICE:
        # Calculate actual price from offset
        company_id = turn_module.TURN.get_acq_target_company(state)
        low_price = get_company_low_price(company_id)
        price = low_price + info.amount

        # Validate and execute
        if not _validate_price_action(state, price):
            return 1
        _handle_accept_price(state, price)
        return 0

    elif info.action_type == ACTION_ACQ_FI_HIGH:
        if not _validate_fi_buy_high(state):
            return 1
        _handle_fi_buy_high(state)
        return 0

    elif info.action_type == ACTION_ACQ_FI_FACE:
        if not _validate_fi_buy_face(state):
            return 1
        _handle_fi_buy_face(state)
        return 0

    elif info.action_type == ACTION_PASS:
        # Pass is always valid
        _handle_pass(state)
        return 0

    # Unknown action type
    return 1


# =============================================================================
# MAIN PHASE HANDLER (STUB)
# =============================================================================

cdef int apply_acquisition_stub(GameState state) noexcept:
    """
    Stub: ACQUISITION immediately transitions to new INVEST turn.

    When ACQUISITION is fully implemented, this will be replaced with:
    - FI purchase logic (Phase 10)
    - Corp acquisition offers
    - Company availability updates

    For now, just increment turn number and start new INVEST.
    Handles terminal state detection to prevent infinite loops.
    """
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    cdef int i

    # Check for terminal state before transitioning to INVEST
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        return 0

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Transition to new INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    return 0
