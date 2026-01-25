# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""ACQUISITION phase stub - transitions immediately to INVEST."""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants, get_company_face_value
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
from core.data import CORP_NAMES, CORP_NAME_TO_ID

# Constants
DEF OFFER_BUFFER_SIZE = 250
DEF OS_CORP_ID = 2  # OS is index 2 in CORP_NAMES


# =============================================================================
# OFFER GENERATION HELPERS
# =============================================================================

cdef int _get_corp_president(GameState state, int corp_id) noexcept:
    """Get player_id of corp president, or -1 if in receivership."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].is_president_of(state, corp_id):
            return player_id
    return -1


cdef int _collect_fi_offers(GameState state, int* corp_ids, int* company_ids) noexcept:
    """
    Collect Corp->FI offers. Returns count.
    OS->FI offers come first, then other corps by descending share price.
    Only active corps that can afford at least one FI company are included.
    """
    cdef int count = 0
    cdef int company_id, corp_id
    cdef int corp_cash, high_price
    cdef int temp_count = 0
    cdef int temp_corp_ids[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_prices[OFFER_BUFFER_SIZE]
    cdef int i, j, best_idx
    cdef int best_price, curr_price
    cdef int swap_corp, swap_company, swap_price

    # First pass: OS->FI offers (OS always first if can afford)
    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            high_price = company_module.COMPANIES[company_id].get_high_price()

            # Check if OS can afford (OS always first if affordable)
            if corp_module.CORPS[CORP_NAMES[OS_CORP_ID]].is_active(state):
                corp_cash = corp_module.CORPS[CORP_NAMES[OS_CORP_ID]].get_cash(state)
                if corp_cash >= high_price and count < OFFER_BUFFER_SIZE:
                    corp_ids[count] = OS_CORP_ID
                    company_ids[count] = company_id
                    count += 1

    # Second pass: collect other corps (not OS) into temp arrays
    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            high_price = company_module.COMPANIES[company_id].get_high_price()

            # Check all other corps (skip OS)
            for corp_id in range(GameConstants.NUM_CORPS):
                if corp_id == OS_CORP_ID:
                    continue

                if corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
                    corp_cash = corp_module.CORPS[CORP_NAMES[corp_id]].get_cash(state)
                    if corp_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                        temp_corp_ids[temp_count] = corp_id
                        temp_company_ids[temp_count] = company_id
                        temp_prices[temp_count] = corp_module.CORPS[CORP_NAMES[corp_id]].get_share_price(state)
                        temp_count += 1

    # Selection sort by descending share price (like wrap_up.pyx)
    for i in range(temp_count):
        best_idx = i
        best_price = temp_prices[i]

        for j in range(i + 1, temp_count):
            curr_price = temp_prices[j]
            if curr_price > best_price:
                best_idx = j
                best_price = curr_price

        # Swap to front
        if best_idx != i:
            swap_corp = temp_corp_ids[i]
            temp_corp_ids[i] = temp_corp_ids[best_idx]
            temp_corp_ids[best_idx] = swap_corp

            swap_company = temp_company_ids[i]
            temp_company_ids[i] = temp_company_ids[best_idx]
            temp_company_ids[best_idx] = swap_company

            swap_price = temp_prices[i]
            temp_prices[i] = temp_prices[best_idx]
            temp_prices[best_idx] = swap_price

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
    Sorted by (buyer share price DESC, target face value ASC).
    """
    cdef int count = 0
    cdef int player_id, buyer_corp, seller_corp, company_id
    cdef int buyer_cash, high_price, buyer_price, face_value
    cdef int temp_count = 0
    cdef int temp_buyer_corps[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_buyer_prices[OFFER_BUFFER_SIZE]
    cdef int temp_face_values[OFFER_BUFFER_SIZE]
    cdef int i, j, best_idx
    cdef int best_price, best_fv, curr_price, curr_fv
    cdef int swap_buyer, swap_company, swap_price, swap_fv

    # For each player, find corps they control and generate offers
    for player_id in range(state._num_players):
        # Find all corps this player is president of
        for buyer_corp in range(GameConstants.NUM_CORPS):
            if not corp_module.CORPS[CORP_NAMES[buyer_corp]].is_active(state):
                continue
            if _get_corp_president(state, buyer_corp) != player_id:
                continue

            buyer_cash = corp_module.CORPS[CORP_NAMES[buyer_corp]].get_cash(state)
            buyer_price = corp_module.CORPS[CORP_NAMES[buyer_corp]].get_share_price(state)

            # Find all other corps this player is president of
            for seller_corp in range(GameConstants.NUM_CORPS):
                if seller_corp == buyer_corp:
                    continue
                if not corp_module.CORPS[CORP_NAMES[seller_corp]].is_active(state):
                    continue
                if _get_corp_president(state, seller_corp) != player_id:
                    continue

                # Find companies owned by seller corp
                for company_id in range(GameConstants.NUM_COMPANIES):
                    if corp_module.CORPS[CORP_NAMES[seller_corp]].owns_company(state, company_id):
                        high_price = company_module.COMPANIES[company_id].get_high_price()
                        face_value = get_company_face_value(company_id)

                        if buyer_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                            temp_buyer_corps[temp_count] = buyer_corp
                            temp_company_ids[temp_count] = company_id
                            temp_buyer_prices[temp_count] = buyer_price
                            temp_face_values[temp_count] = face_value
                            temp_count += 1

    # Selection sort by (buyer price DESC, face value ASC)
    for i in range(temp_count):
        best_idx = i
        best_price = temp_buyer_prices[i]
        best_fv = temp_face_values[i]

        for j in range(i + 1, temp_count):
            curr_price = temp_buyer_prices[j]
            curr_fv = temp_face_values[j]

            # Higher buyer price wins, or if equal, lower face value wins
            if (curr_price > best_price or
                (curr_price == best_price and curr_fv < best_fv)):
                best_idx = j
                best_price = curr_price
                best_fv = curr_fv

        # Swap to front
        if best_idx != i:
            swap_buyer = temp_buyer_corps[i]
            temp_buyer_corps[i] = temp_buyer_corps[best_idx]
            temp_buyer_corps[best_idx] = swap_buyer

            swap_company = temp_company_ids[i]
            temp_company_ids[i] = temp_company_ids[best_idx]
            temp_company_ids[best_idx] = swap_company

            swap_price = temp_buyer_prices[i]
            temp_buyer_prices[i] = temp_buyer_prices[best_idx]
            temp_buyer_prices[best_idx] = swap_price

            swap_fv = temp_face_values[i]
            temp_face_values[i] = temp_face_values[best_idx]
            temp_face_values[best_idx] = swap_fv

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
    Sorted by (buyer share price DESC, target face value ASC).
    """
    cdef int count = 0
    cdef int player_id, corp_id, company_id
    cdef int corp_cash, high_price, corp_price, face_value
    cdef int temp_count = 0
    cdef int temp_corp_ids[OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[OFFER_BUFFER_SIZE]
    cdef int temp_corp_prices[OFFER_BUFFER_SIZE]
    cdef int temp_face_values[OFFER_BUFFER_SIZE]
    cdef int i, j, best_idx
    cdef int best_price, best_fv, curr_price, curr_fv
    cdef int swap_corp, swap_company, swap_price, swap_fv

    # For each player, find their private companies and corps they control
    for player_id in range(state._num_players):
        # Find all private companies owned by this player
        for company_id in range(GameConstants.NUM_COMPANIES):
            if player_module.PLAYERS[player_id].owns_company(state, company_id):
                high_price = company_module.COMPANIES[company_id].get_high_price()
                face_value = get_company_face_value(company_id)

                # Find all corps this player is president of
                for corp_id in range(GameConstants.NUM_CORPS):
                    if not corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
                        continue
                    if _get_corp_president(state, corp_id) != player_id:
                        continue

                    corp_cash = corp_module.CORPS[CORP_NAMES[corp_id]].get_cash(state)
                    corp_price = corp_module.CORPS[CORP_NAMES[corp_id]].get_share_price(state)

                    if corp_cash >= high_price and temp_count < OFFER_BUFFER_SIZE:
                        temp_corp_ids[temp_count] = corp_id
                        temp_company_ids[temp_count] = company_id
                        temp_corp_prices[temp_count] = corp_price
                        temp_face_values[temp_count] = face_value
                        temp_count += 1

    # Selection sort by (corp price DESC, face value ASC)
    for i in range(temp_count):
        best_idx = i
        best_price = temp_corp_prices[i]
        best_fv = temp_face_values[i]

        for j in range(i + 1, temp_count):
            curr_price = temp_corp_prices[j]
            curr_fv = temp_face_values[j]

            # Higher corp price wins, or if equal, lower face value wins
            if (curr_price > best_price or
                (curr_price == best_price and curr_fv < best_fv)):
                best_idx = j
                best_price = curr_price
                best_fv = curr_fv

        # Swap to front
        if best_idx != i:
            swap_corp = temp_corp_ids[i]
            temp_corp_ids[i] = temp_corp_ids[best_idx]
            temp_corp_ids[best_idx] = swap_corp

            swap_company = temp_company_ids[i]
            temp_company_ids[i] = temp_company_ids[best_idx]
            temp_company_ids[best_idx] = swap_company

            swap_price = temp_corp_prices[i]
            temp_corp_prices[i] = temp_corp_prices[best_idx]
            temp_corp_prices[best_idx] = swap_price

            swap_fv = temp_face_values[i]
            temp_face_values[i] = temp_face_values[best_idx]
            temp_face_values[best_idx] = swap_fv

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
    1. OS->FI offers (OFFER-02)
    2. Other Corp->FI offers by descending share price (OFFER-03)
    3. Corp->Corp offers by (buyer price DESC, face value ASC) (OFFER-04)
    4. Corp->Player private offers by (buyer price DESC, face value ASC) (OFFER-05)

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
# HELPER FUNCTIONS
# =============================================================================

cdef bint _is_game_terminal(GameState state) noexcept:
    """
    Check if the game has reached a terminal state.

    Terminal state occurs when:
    1. No companies are available for auction, AND
    2. No corporations are active

    This prevents infinite INVEST->WRAP_UP->ACQUISITION loops when
    all companies are removed from the game.
    """
    cdef int company_id, corp_id
    cdef bint has_auction_companies = False
    cdef bint has_active_corps = False

    # Check for any companies available for auction
    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_for_auction(state):
            has_auction_companies = True
            break

    # Check for any active corporations
    for corp_id in range(GameConstants.NUM_CORPS):
        if corp_module.CORPS[CORP_NAMES[corp_id]].is_active(state):
            has_active_corps = True
            break

    # Terminal if no auction companies AND no active corps
    return not has_auction_companies and not has_active_corps


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

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to new INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    return 0
