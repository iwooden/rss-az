# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""CLOSING phase handler implementation."""

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases, PHASE_GAME_OVER,
    get_cost_of_ownership, get_company_income, get_company_stars, get_company_face_value
)
from entities import turn as turn_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
from entities import player as player_module

# Constants
DEF CLOSE_OFFER_BUFFER_SIZE = 100
DEF OWNER_PLAYER = 0  # Owner type for player-owned companies
DEF OWNER_CORP = 1    # Owner type for corp-owned companies


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef bint _is_game_terminal(GameState state) noexcept:
    """
    Check if the game has reached a terminal state.

    Terminal state occurs when:
    1. No companies are available for auction, AND
    2. No corporations are active

    This prevents infinite INVEST->WRAP_UP->ACQUISITION->CLOSING loops when
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
        if corp_module.CORPS[corp_id].is_active(state):
            has_active_corps = True
            break

    # Terminal if no auction companies AND no active corps
    return not has_auction_companies and not has_active_corps

cdef void _close_company(GameState state, int company_id, int owner_type, int owner_id) noexcept:
    """
    Close a company and handle cleanup.

    Steps:
    1. Clear ownership from previous owner (FI or corp)
    2. Apply Junkyard Scrappers bonus (2x printed income to JS cash)
    3. Remove company from game

    Args:
        state: Game state
        company_id: Company to close (0-35)
        owner_type: LOC_FI (4) or LOC_CORP (5)
        owner_id: Corp ID if owner_type is LOC_CORP, -1 otherwise
    """
    cdef int printed_income = get_company_income(company_id)

    # Clear ownership before removal
    if owner_type == 4:  # LOC_FI
        fi_module.FI.set_owns_company(state, company_id, False)
    elif owner_type == 5:  # LOC_CORP
        corp_module.CORPS[owner_id].set_owns_company(state, company_id, False)

    # Junkyard Scrappers (corp_id 0) bonus: 2x printed income
    if corp_module.CORPS[0].is_active(state):
        corp_module.CORPS[0].add_cash(state, printed_income * 2)

    # Remove company from game
    company_module.COMPANIES[company_id].remove_from_game(state)


cdef bint _has_negative_adjusted_income(GameState state, int company_id) noexcept:
    """Check if company has negative adjusted income (eligible for close offer)."""
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int base_income = get_company_income(company_id)
    cdef int stars = get_company_stars(company_id)
    cdef int coo_value = get_cost_of_ownership(coo_level, stars)
    return (base_income - coo_value) < 0  # NEGATIVE only, not zero


cdef int _get_corp_president(GameState state, int corp_id) noexcept:
    """Get player_id of corp president, or -1 if in receivership."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].is_president_of(state, corp_id):
            return player_id
    return -1


cdef int _collect_player_close_offers(
    GameState state,
    int* owner_types, int* owner_ids, int* company_ids, int* face_values,
    int start_idx
) noexcept:
    """
    Collect close offers for player-owned private companies.
    Returns count of offers added (starting from start_idx).
    """
    cdef int count = 0
    cdef int player_id, company_id, idx

    for player_id in range(state._num_players):
        for company_id in range(GameConstants.NUM_COMPANIES):
            if not player_module.PLAYERS[player_id].owns_company(state, company_id):
                continue
            if not _has_negative_adjusted_income(state, company_id):
                continue

            idx = start_idx + count
            if idx >= CLOSE_OFFER_BUFFER_SIZE:
                return count

            owner_types[idx] = OWNER_PLAYER
            owner_ids[idx] = player_id
            company_ids[idx] = company_id
            face_values[idx] = get_company_face_value(company_id)
            count += 1

    return count


cdef int _collect_corp_close_offers(
    GameState state,
    int* owner_types, int* owner_ids, int* company_ids, int* face_values,
    int start_idx
) noexcept:
    """
    Collect close offers for corp-owned companies where player is president.
    Excludes receivership corps (no president) and FI (handled by auto-close).
    Returns count of offers added.
    """
    cdef int count = 0
    cdef int corp_id, company_id, president, idx

    for corp_id in range(GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue

        # Skip receivership corps (no president = excluded from offers)
        president = _get_corp_president(state, corp_id)
        if president < 0:
            continue

        for company_id in range(GameConstants.NUM_COMPANIES):
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue
            if not _has_negative_adjusted_income(state, company_id):
                continue

            idx = start_idx + count
            if idx >= CLOSE_OFFER_BUFFER_SIZE:
                return count

            owner_types[idx] = OWNER_CORP
            owner_ids[idx] = corp_id
            company_ids[idx] = company_id
            face_values[idx] = get_company_face_value(company_id)
            count += 1

    return count


cdef void _sort_close_offers_by_face_value(
    int* owner_types, int* owner_ids, int* company_ids, int* face_values,
    int count
) noexcept:
    """Sort close offers by face value ascending (lowest first)."""
    cdef int i, j, best_idx, best_fv, curr_fv
    cdef int swap_ot, swap_oi, swap_cid, swap_fv

    for i in range(count):
        best_idx = i
        best_fv = face_values[i]

        for j in range(i + 1, count):
            curr_fv = face_values[j]
            if curr_fv < best_fv:  # Lower face value wins
                best_idx = j
                best_fv = curr_fv

        if best_idx != i:
            # Swap all four arrays
            swap_ot = owner_types[i]
            owner_types[i] = owner_types[best_idx]
            owner_types[best_idx] = swap_ot

            swap_oi = owner_ids[i]
            owner_ids[i] = owner_ids[best_idx]
            owner_ids[best_idx] = swap_oi

            swap_cid = company_ids[i]
            company_ids[i] = company_ids[best_idx]
            company_ids[best_idx] = swap_cid

            swap_fv = face_values[i]
            face_values[i] = face_values[best_idx]
            face_values[best_idx] = swap_fv


cdef void _generate_close_offers(GameState state) noexcept:
    """
    Generate all close offers and store in hidden buffer.

    Offers sorted by face value ascending (lowest first).
    Only companies with negative adjusted income.
    Only from players and player-presided corps (not FI, not receivership).

    Buffer layout: [count][index][offer0_owner_type][offer0_owner_id][offer0_company_id]...
    """
    cdef int temp_owner_types[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_owner_ids[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_company_ids[CLOSE_OFFER_BUFFER_SIZE]
    cdef int temp_face_values[CLOSE_OFFER_BUFFER_SIZE]
    cdef int offer_count = 0
    cdef int i, base

    # Initialize counters
    state._data[state._layout.hidden_close_offer_count_offset] = 0.0
    state._data[state._layout.hidden_close_offer_index_offset] = 0.0

    # Collect offers from players (private companies)
    offer_count = _collect_player_close_offers(
        state, temp_owner_types, temp_owner_ids, temp_company_ids, temp_face_values, 0
    )

    # Collect offers from player-presided corps
    offer_count += _collect_corp_close_offers(
        state, temp_owner_types, temp_owner_ids, temp_company_ids, temp_face_values, offer_count
    )

    # Sort by face value ascending
    _sort_close_offers_by_face_value(
        temp_owner_types, temp_owner_ids, temp_company_ids, temp_face_values, offer_count
    )

    # Write to buffer
    for i in range(offer_count):
        base = state._layout.hidden_close_offer_buffer_offset + (i * 3)
        state._data[base] = <float>temp_owner_types[i]
        state._data[base + 1] = <float>temp_owner_ids[i]
        state._data[base + 2] = <float>temp_company_ids[i]

    state._data[state._layout.hidden_close_offer_count_offset] = <float>offer_count


cdef int _count_corp_companies(GameState state, int corp_id, int exclude_company_id) noexcept:
    """
    Count companies corp retains after excluding target.
    Used to enforce last-company rule.
    Returns count of companies excluding the specified company_id.
    """
    cdef int count = 0
    cdef int company_id

    for company_id in range(GameConstants.NUM_COMPANIES):
        if company_id == exclude_company_id:
            continue
        if corp_module.CORPS[corp_id].owns_company(state, company_id):
            count += 1

    return count


cdef bint _is_close_offer_valid(GameState state, int owner_type, int owner_id, int company_id) noexcept:
    """
    Check if close offer is still valid for presentation.

    Invalid if:
    - Company already closed earlier in this phase (removed from game)
    - Owner no longer owns company
    - Corp owner would have 0 companies after close (last-company rule)

    Returns True if offer is valid, False otherwise.
    """
    # Check company still exists (not already closed)
    if company_module.COMPANIES[company_id].is_removed(state):
        return False

    # Check ownership unchanged
    if owner_type == OWNER_PLAYER:
        if not player_module.PLAYERS[owner_id].owns_company(state, company_id):
            return False
    elif owner_type == OWNER_CORP:
        if not corp_module.CORPS[owner_id].owns_company(state, company_id):
            return False

        # Corp last-company rule: can't close if corp would have 0 companies
        if _count_corp_companies(state, owner_id, company_id) < 1:
            return False

    return True


cdef void _transition_to_income(GameState state) noexcept:
    """
    Complete CLOSING phase and transition to INCOME.

    Called when no more close offers exist.
    """
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    cdef int i

    # Check for terminal state
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to INCOME phase
    # Note: INCOME phase not implemented yet, using INVEST as temporary target
    # This will be updated when INCOME phase is implemented
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)


cdef void _present_next_close_offer(GameState state) noexcept:
    """
    Advance to next valid offer and update visible state.

    Loops until valid offer found or offers exhausted.
    When no more offers, clears closing_company and transitions to INCOME.
    """
    cdef int count = <int>state._data[state._layout.hidden_close_offer_count_offset]
    cdef int index = <int>state._data[state._layout.hidden_close_offer_index_offset]
    cdef int owner_type, owner_id, company_id, president, base

    while index < count:
        base = state._layout.hidden_close_offer_buffer_offset + (index * 3)
        owner_type = <int>state._data[base]
        owner_id = <int>state._data[base + 1]
        company_id = <int>state._data[base + 2]

        # Check if offer still valid (dynamic re-validation)
        if not _is_close_offer_valid(state, owner_type, owner_id, company_id):
            index += 1
            state._data[state._layout.hidden_close_offer_index_offset] = <float>index
            continue

        # Found valid offer - set visible state
        turn_module.TURN.set_closing_company(state, company_id)

        # Determine active player (owner for player, president for corp)
        if owner_type == OWNER_PLAYER:
            state._set_active_player(owner_id)
        elif owner_type == OWNER_CORP:
            president = _get_corp_president(state, owner_id)
            state._set_active_player(president if president >= 0 else 0)
        return

    # No more valid offers - clear state and transition
    turn_module.TURN.clear_closing_company(state)
    _transition_to_income(state)


cdef void _process_fi_auto_close(GameState state) noexcept:
    """
    Execute FI auto-close logic.

    FI closes companies with negative adjusted income (income - CoO < 0).
    Only applies to companies owned by FI.
    """
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int company_id
    cdef int base_income, stars, coo_value, adjusted_income
    cdef int num_to_close = 0
    cdef int[36] companies_to_close  # Track which companies to close

    # First pass: identify companies to close
    for company_id in range(GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            base_income = get_company_income(company_id)
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)
            adjusted_income = base_income - coo_value

            # Close if NEGATIVE (not zero)
            if adjusted_income < 0:
                companies_to_close[num_to_close] = company_id
                num_to_close += 1

    # Second pass: close identified companies
    for company_id in range(num_to_close):
        _close_company(state, companies_to_close[company_id], 4, -1)  # LOC_FI = 4


cdef void _process_receivership_auto_close(GameState state) noexcept:
    """
    Execute receivership corporation auto-close logic.

    Receivership corps close companies based on CoO thresholds:
    - Red (1 star): close if CoO >= $4
    - Orange (2 stars): close if CoO >= $7
    - Yellow/Green/Blue (3-5 stars): NEVER auto-close

    Protection: Highest face value company in each receivership corp is protected.
    Vintage Machinery (corp 6): Apply $10 CoO reduction before threshold check.
    """
    cdef int corp_id, company_id, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef bint is_vm
    cdef int protected_company, max_face_value, face_value
    cdef int num_to_close
    cdef int[36] companies_to_close

    for corp_id in range(GameConstants.NUM_CORPS):
        # Skip inactive corps and non-receivership corps
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not corp_module.CORPS[corp_id].is_in_receivership(state):
            continue

        # Check if this is Vintage Machinery (corp_id 6)
        is_vm = (corp_id == 6)

        # Find highest face value company (protected)
        protected_company = -1
        max_face_value = -1
        for company_id in range(GameConstants.NUM_COMPANIES):
            if corp_module.CORPS[corp_id].owns_company(state, company_id):
                face_value = get_company_face_value(company_id)
                if face_value > max_face_value:
                    max_face_value = face_value
                    protected_company = company_id

        # Identify companies to close
        num_to_close = 0
        for company_id in range(GameConstants.NUM_COMPANIES):
            # Skip if not owned by this corp
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue

            # Skip protected company
            if company_id == protected_company:
                continue

            # Get CoO value
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)

            # Apply Vintage Machinery reduction
            if is_vm:
                coo_value = max(0, coo_value - 10)

            # Check thresholds
            if stars == 1 and coo_value >= 4:  # Red
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            elif stars == 2 and coo_value >= 7:  # Orange
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            # Yellow (3), Green (4), Blue (5): never auto-close

        # Close identified companies
        for company_id in range(num_to_close):
            _close_company(state, companies_to_close[company_id], 5, corp_id)  # LOC_CORP = 5


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_closing_auto(GameState state) noexcept:
    """
    Execute auto-close logic for FI and receivership corps.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. FI closes companies with negative adjusted income
    2. Receivership corps close red/orange companies above CoO thresholds
    3. Junkyard Scrappers receives 2x printed income for each closure
    4. Transition to INVEST (Phase 16 temporary - Phase 17 will add offer logic)

    Returns: 0 always (deterministic, no failure modes)
    """
    cdef int current_turn, i

    _process_fi_auto_close(state)
    _process_receivership_auto_close(state)

    # Check for terminal state after auto-close
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return 0

    # TEMPORARY (Phase 16): Transition to INVEST
    # Phase 17 will add offer-based closing logic here instead
    current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to new INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    return 0


def apply_closing_auto_py(GameState state):
    """Python wrapper for testing."""
    return apply_closing_auto(state)
