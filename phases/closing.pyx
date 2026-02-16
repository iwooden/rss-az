# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
CLOSING phase: Companies with negative adjusted income can be closed.

DESIGN: Two-Stage Close with One-by-One Offers
==============================================
Stage 1 - Auto-close (deterministic, no player input):
  - FI closes companies with negative adjusted income (income - CoO < 0)
  - Receivership corps close red/orange companies above CoO thresholds:
    * Red (1 star): close if CoO >= $4
    * Orange (2 stars): close if CoO >= $7
    * Yellow/Green/Blue: never auto-close
  - Highest face value company in each receivership corp is protected

Stage 2 - Offer-based close (player decisions):
  - Only for player-owned privates and player-presided corps
  - Uses same one-by-one pattern as acquisition.pyx:
    1. Generate offers into hidden buffer (_generate_close_offers)
    2. Sort by face value ascending (cheapest first)
    3. Present one at a time via closing_company state
    4. Player chooses CLOSE or PASS for each
  - Dynamic re-validation: skip offers where company already closed

Mandatory close (after all offers processed):
  - If player would have negative income+cash, force-close their cheapest
    negative-income private company, repeating until safe

Action space: just 2 actions (CLOSE, PASS) - offers presented sequentially.

See CLAUDE.md "Offer Buffer Pattern" for full documentation.
"""

from core.state cimport GameState
from core.driver cimport _is_game_terminal
from core.data cimport (
    GameConstants, GamePhases, CorpIndices, PHASE_GAME_OVER, PHASE_INCOME,
    get_cost_of_ownership, get_company_income, get_company_stars, get_company_face_value
)
from core.actions cimport ActionInfo, ACTION_CLOSE, ACTION_PASS
from entities.company cimport LOC_PLAYER, LOC_FI, LOC_CORP
from entities.offer cimport OfferBuffer
from entities.offer import CLOSE_OFFERS as _close_offers_py
cdef OfferBuffer _buf = _close_offers_py
from entities import turn as turn_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
from entities import player as player_module

# Buffer size constant (DEF required for static array sizing - cannot be imported)
DEF CLOSE_OFFER_BUFFER_SIZE = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _close_company(GameState state, int company_id, int owner_type, int owner_id) noexcept:
    """
    Close a company and handle cleanup.

    Steps:
    1. Apply Junkyard Scrappers bonus (2x printed income to JS cash)
    2. Remove company from game (clears ownership automatically)
    3. Recalculate stars if owned by a corp (company removed, cash may have changed)

    Args:
        state: Game state
        company_id: Company to close (0-35)
        owner_type: LOC_FI (4) or LOC_CORP (5)
        owner_id: Corp ID if owner_type is LOC_CORP, -1 otherwise
    """
    cdef int printed_income = get_company_income(company_id)

    # Junkyard Scrappers bonus: 2x printed income only when JS closes its own company
    if owner_type == LOC_CORP and owner_id == CorpIndices.CORP_JS:
        corp_module.CORPS[owner_id].add_cash(state, printed_income * 2)

    # Remove company from game (stars auto-updated via remove_from_game and set_cash)
    company_module.COMPANIES[company_id].remove_from_game(state)


cdef void _close_player_company(GameState state, int company_id, int player_id) noexcept:
    """
    Close a player-owned private company during mandatory close.

    remove_from_game handles clearing ownership automatically.
    """
    company_module.COMPANIES[company_id].remove_from_game(state)


cdef void _process_mandatory_close(GameState state) noexcept:
    """
    Auto-close player private companies to prevent negative cash in INCOME.

    Called at phase end, before transition to INCOME.
    Iterates players by ID order. For each player with income + cash < 0:
    1. Find cheapest (lowest face value) negative-income private company
    2. Close it
    3. Recheck income + cash
    4. Repeat until income + cash >= 0

    Per CONTEXT.md: CoO is fixed at phase start, no re-evaluation during loop.
    Per CONTEXT.md: Players CAN end up with zero companies (no minimum retention).

    Note: JS bonus does NOT apply here - mandatory close only affects player-owned
    private companies, and JS bonus only triggers when JS closes its own subsidiaries.
    """
    cdef int player_id, company_id, income, cash
    cdef int cheapest_company, cheapest_fv, fv
    cdef int coo_level = turn_module.TURN.get_coo_level(state)

    # Iterate players by player ID order (0, 1, 2, ...)
    for player_id in range(state._num_players):
        # While player has negative total (income + cash)
        while True:
            income = player_module.PLAYERS[player_id].get_income(state)
            cash = player_module.PLAYERS[player_id].get_cash(state)

            if income + cash >= 0:
                break  # Player is safe

            # Find cheapest negative-income company owned by player
            cheapest_company = -1
            cheapest_fv = 999999  # Large sentinel

            for company_id in range(<int>GameConstants.NUM_COMPANIES):
                if not player_module.PLAYERS[player_id].owns_company(state, company_id):
                    continue

                # Check if negative income (CLO-14 targets negative-income companies)
                if company_module.COMPANIES[company_id].get_adjusted_income(state) >= 0:
                    continue

                fv = get_company_face_value(company_id)
                if fv < cheapest_fv:
                    cheapest_fv = fv
                    cheapest_company = company_id

            if cheapest_company < 0:
                # No more negative-income companies to close
                # Per CONTEXT.md: impossible to still be negative after closing ALL negative-income privates
                break

            # Close the company (CLO-15: cheapest first)
            _close_player_company(state, cheapest_company, player_id)


cdef bint _has_negative_adjusted_income(GameState state, int company_id) noexcept:
    """Check if company has negative adjusted income (eligible for close offer)."""
    return company_module.COMPANIES[company_id].get_adjusted_income(state) < 0


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
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if not player_module.PLAYERS[player_id].owns_company(state, company_id):
                continue
            if not _has_negative_adjusted_income(state, company_id):
                continue

            idx = start_idx + count
            if idx >= CLOSE_OFFER_BUFFER_SIZE:
                return count

            owner_types[idx] = LOC_PLAYER
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

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue

        # Skip receivership corps (no president = excluded from offers)
        president = corp_module.CORPS[corp_id].get_president_id(state)
        if president < 0:
            continue

        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue
            if not _has_negative_adjusted_income(state, company_id):
                continue

            idx = start_idx + count
            if idx >= CLOSE_OFFER_BUFFER_SIZE:
                return count

            owner_types[idx] = LOC_CORP
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
    cdef int i
    cdef float* data = state._data

    _buf.reset(data)

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
        _buf.append(data, i,
                   temp_owner_types[i], temp_owner_ids[i], temp_company_ids[i])

    _buf.set_count(data, offer_count)


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
    if owner_type == LOC_PLAYER:
        if not player_module.PLAYERS[owner_id].owns_company(state, company_id):
            return False
    elif owner_type == LOC_CORP:
        if not corp_module.CORPS[owner_id].owns_company(state, company_id):
            return False

        # Corp last-company rule: can't close if corp would have 0 companies
        # Check count >= 2 since closing one would leave at least 1
        if corp_module.CORPS[owner_id].count_companies(state) < 2:
            return False

    return True


cdef void _transition_to_income(GameState state) noexcept:
    """
    Complete CLOSING phase and transition to INCOME.

    Called when no more close offers exist.
    """
    # Check for terminal state
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return

    # Transition to INCOME phase
    turn_module.TURN.set_phase(state, PHASE_INCOME)


cdef void _present_next_close_offer(GameState state) noexcept:
    """
    Advance to next valid offer and update visible state.

    Loops until valid offer found or offers exhausted.
    When no more offers, clears closing_company and transitions to INCOME.
    """
    cdef float* data = state._data
    cdef int count = _buf.get_count(data)
    cdef int index = _buf.get_index(data)
    cdef int owner_type, owner_id, company_id, president, base

    while index < count:
        base = _buf.offer_base(index)
        owner_type = <int>data[base]
        owner_id = <int>data[base + 1]
        company_id = <int>data[base + 2]

        # Check if offer still valid (dynamic re-validation)
        if not _is_close_offer_valid(state, owner_type, owner_id, company_id):
            index += 1
            _buf.set_index(data, index)
            continue

        # Found valid offer - update net worths before presenting decision
        # (catches auto-close changes and prior close decisions)
        player_module.update_all_net_worths(state)

        # Set visible state
        turn_module.TURN.set_closing_company(state, company_id)

        # Determine active player (owner for player, president for corp)
        if owner_type == LOC_PLAYER:
            state._set_active_player(owner_id)
        elif owner_type == LOC_CORP:
            president = corp_module.CORPS[owner_id].get_president_id(state)
            state._set_active_player(president if president >= 0 else 0)
        return

    # No more valid offers - process mandatory close then transition
    turn_module.TURN.clear_closing_company(state)
    _process_mandatory_close(state)  # CLO-14, CLO-15: mandatory close before transition
    _transition_to_income(state)


cdef void _process_fi_auto_close(GameState state) noexcept:
    """
    Execute FI auto-close logic.

    FI closes companies with negative adjusted income (income - CoO < 0).
    Only applies to companies owned by FI.
    """
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int company_id
    cdef int num_to_close = 0
    cdef int[36] companies_to_close  # Track which companies to close

    # First pass: identify companies to close
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if fi_module.FI.owns_company(state, company_id):
            # Close if NEGATIVE adjusted income (not zero)
            if company_module.COMPANIES[company_id].get_adjusted_income(state) < 0:
                companies_to_close[num_to_close] = company_id
                num_to_close += 1

    # Second pass: close identified companies
    for company_id in range(num_to_close):
        _close_company(state, companies_to_close[company_id], LOC_FI, -1)


cdef void _process_receivership_auto_close(GameState state) noexcept:
    """
    Execute receivership corporation auto-close logic.

    Receivership corps close companies based on CoO thresholds:
    - Red (1 star): close if CoO >= $4
    - Orange (2 stars): close if CoO >= $7
    - Yellow/Green/Blue (3-5 stars): NEVER auto-close

    Protection: Highest face value company in each receivership corp is protected.

    Note: Corporation special abilities (like VM's CoO reduction) do NOT apply here.
    Per RULES.md, VM's ability is for income calculation only. Receivership corps
    follow simple deterministic rules without special ability considerations.
    """
    cdef int corp_id, company_id, stars, coo_value
    cdef int coo_level = turn_module.TURN.get_coo_level(state)
    cdef int protected_company, max_face_value, face_value
    cdef int num_to_close
    cdef int[36] companies_to_close

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        # Skip inactive corps and non-receivership corps
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not corp_module.CORPS[corp_id].is_in_receivership(state):
            continue

        # Find highest face value company (protected)
        protected_company = -1
        max_face_value = -1
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if corp_module.CORPS[corp_id].owns_company(state, company_id):
                face_value = get_company_face_value(company_id)
                if face_value > max_face_value:
                    max_face_value = face_value
                    protected_company = company_id

        # Identify companies to close
        num_to_close = 0
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            # Skip if not owned by this corp
            if not corp_module.CORPS[corp_id].owns_company(state, company_id):
                continue

            # Skip protected company
            if company_id == protected_company:
                continue

            # Get CoO value and check thresholds
            stars = get_company_stars(company_id)
            coo_value = get_cost_of_ownership(coo_level, stars)

            if stars == 1 and coo_value >= 4:  # Red
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            elif stars == 2 and coo_value >= 7:  # Orange
                companies_to_close[num_to_close] = company_id
                num_to_close += 1
            # Yellow (3), Green (4), Blue (5): never auto-close

        # Close identified companies
        for company_id in range(num_to_close):
            _close_company(state, companies_to_close[company_id], LOC_CORP, corp_id)


# =============================================================================
# ACTION HANDLERS
# =============================================================================

cdef void _handle_close_accept(GameState state) noexcept:
    """
    Accept current close offer: close the company and advance to next offer.

    The _close_company helper handles:
    - Clearing ownership
    - Junkyard Scrappers bonus (2x printed income)
    - Removing company from game
    """
    cdef float* data = state._data
    cdef int company_id = turn_module.TURN.get_closing_company(state)
    cdef int index = _buf.get_index(data)
    cdef int base, owner_type, owner_id

    if company_id < 0:
        return  # No active offer

    # Get current offer details
    base = _buf.offer_base(index)
    owner_type = <int>data[base]
    owner_id = <int>data[base + 1]

    # Close the company (remove_from_game handles clearing ownership)
    if owner_type == LOC_PLAYER:
        company_module.COMPANIES[company_id].remove_from_game(state)
    elif owner_type == LOC_CORP:
        _close_company(state, company_id, LOC_CORP, owner_id)

    # Advance to next offer
    _buf.advance(data)
    _present_next_close_offer(state)


cdef void _handle_close_pass(GameState state) noexcept:
    """
    Pass on current close offer: keep company, advance to next offer.
    """
    _buf.advance(state._data)
    _present_next_close_offer(state)


cdef int apply_closing_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply CLOSING phase player action.

    Action types:
    - ACTION_CLOSE: Accept offer, close company
    - ACTION_PASS: Reject offer, keep company

    Returns: 0=success, 1=invalid
    """
    cdef int company_id

    if info.action_type == ACTION_CLOSE:
        # Validate offer still active
        company_id = turn_module.TURN.get_closing_company(state)
        if company_id < 0:
            return 1  # No active offer

        _handle_close_accept(state)
        return 0

    elif info.action_type == ACTION_PASS:
        # Validate offer still active
        company_id = turn_module.TURN.get_closing_company(state)
        if company_id < 0:
            return 1  # No active offer

        _handle_close_pass(state)
        return 0

    return 1  # Unknown action type


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_closing_auto(GameState state) noexcept:
    """
    Execute auto-close logic and setup offer-based closing.

    Steps:
    1. FI closes companies with negative adjusted income (CLO-01)
    2. Receivership corps close red/orange companies above CoO thresholds (CLO-02, CLO-03, CLO-04)
    3. Generate close offers for player/corp decisions
    4. Present first offer (or transition to INCOME if none)

    Returns: 0 always (deterministic entry, no failure modes)
    """
    _process_fi_auto_close(state)
    _process_receivership_auto_close(state)

    # Check for terminal state after auto-close
    if _is_game_terminal(state):
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return 0

    # Generate close offers (Phase 17)
    _generate_close_offers(state)

    # Present first offer (or transition to INCOME if none)
    _present_next_close_offer(state)

    return 0


def apply_closing_auto_py(GameState state):
    """Python wrapper for testing."""
    return apply_closing_auto(state)


def apply_closing_action_py(GameState state, int action_type):
    """Python wrapper for testing closing actions."""
    cdef ActionInfo info
    info.action_type = action_type
    return apply_closing_action(state, &info)


def get_close_offer_count_py(GameState state):
    """Get number of close offers in buffer."""
    return _buf.get_count(state._data)


def get_close_offer_index_py(GameState state):
    """Get current close offer index."""
    return _buf.get_index(state._data)


def get_close_offer_py(GameState state, int index):
    """Get close offer at index as (owner_type, owner_id, company_id) tuple."""
    cdef int base = _buf.offer_base(index)
    return (
        <int>state._data[base],
        <int>state._data[base + 1],
        <int>state._data[base + 2]
    )


def generate_close_offers_py(GameState state):
    """Python wrapper for offer generation (for testing)."""
    _generate_close_offers(state)


def process_mandatory_close_py(GameState state):
    """Python wrapper for mandatory close (for testing)."""
    _process_mandatory_close(state)
