# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""WRAP_UP phase handler implementation."""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants, get_company_face_value
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import fi as fi_module
from entities import deck as deck_module
from phases import acquisition as acquisition_module


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef int _find_cheapest_affordable_available(GameState state) noexcept:
    """
    Find the cheapest available company that FI can afford.

    Iterates companies in ascending face value order (company_id 0-35).
    Returns first affordable company (guaranteed cheapest due to iteration order).

    Returns:
        company_id (0-35) if affordable company found, -1 if none
    """
    cdef int company_id
    cdef int fi_cash = fi_module.FI.get_cash(state)
    cdef int face_value

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_for_auction(state):
            face_value = get_company_face_value(company_id)
            if face_value <= fi_cash:
                return company_id  # First affordable = cheapest

    return -1  # No affordable companies


cdef void _fi_purchase_company(GameState state, int company_id) noexcept:
    """
    Execute FI purchase of a single company.

    Steps:
    1. Deduct face value from FI cash
    2. Transfer company to FI
    3. Draw replacement card from deck
    4. Mark replacement as revealed (unavailable)

    Args:
        state: Game state
        company_id: Company to purchase (0-35)
    """
    cdef int face_value, new_company

    face_value = get_company_face_value(company_id)
    fi_module.FI.add_cash(state, -face_value)
    company_module.COMPANIES[company_id].transfer_to_fi(state)

    new_company = deck_module.DECK.draw(state)
    if new_company >= 0:
        company_module.COMPANIES[new_company].set_revealed(state, True)


cdef void _process_fi_purchases(GameState state) noexcept:
    """
    Execute FI purchase loop.

    FI repeatedly purchases cheapest affordable available company at face value
    until no affordable companies remain. Each purchase draws a replacement card
    that becomes unavailable (revealed).

    Uses while-loop with re-query pattern (no snapshotting) to handle dynamic
    company availability changes during purchases.
    """
    cdef int company_id

    while True:
        company_id = _find_cheapest_affordable_available(state)
        if company_id < 0:
            break
        _fi_purchase_company(state, company_id)


cdef void _make_all_revealed_available(GameState state) noexcept:
    """
    Make all revealed companies available for auction.

    Converts all LOC_REVEALED companies to LOC_AUCTION (available).
    This prepares the company pool for the next INVEST round.
    """
    cdef int company_id

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_revealed(state):
            company_module.COMPANIES[company_id].move_to_auction(state)


cdef void _reorder_players_by_cash(GameState state) noexcept:
    """
    Reorder players by descending cash with old position tie-breaking.

    Algorithm:
    1. Collect (cash, old_position, player_id) tuples
    2. Selection sort by (-cash, old_position): higher cash wins, equal cash -> lower old position wins
    3. Assign new turn order positions
    4. Set active player to new position 0
    """
    cdef int num_players = state._num_players
    cdef int[6] cash_values       # Max 6 players
    cdef int[6] old_positions
    cdef int[6] player_ids
    cdef int i, j, best_idx, temp_id
    cdef int best_cash, best_pos, curr_cash, curr_pos

    # Gather current state - all cdef vars declared at function start
    for i in range(num_players):
        player_ids[i] = i
        cash_values[i] = player_module.PLAYERS[i].get_cash(state)
        old_positions[i] = player_module.PLAYERS[i].get_turn_order(state)

    # Selection sort by (-cash, old_position) - stable for ties
    for i in range(num_players):
        best_idx = i
        best_cash = cash_values[player_ids[i]]
        best_pos = old_positions[player_ids[i]]

        for j in range(i + 1, num_players):
            curr_cash = cash_values[player_ids[j]]
            curr_pos = old_positions[player_ids[j]]

            # Higher cash wins, or if equal, lower old position wins
            if (curr_cash > best_cash or
                (curr_cash == best_cash and curr_pos < best_pos)):
                best_idx = j
                best_cash = curr_cash
                best_pos = curr_pos

        # Swap to front
        if best_idx != i:
            temp_id = player_ids[i]
            player_ids[i] = player_ids[best_idx]
            player_ids[best_idx] = temp_id

    # Apply new turn order
    for i in range(num_players):
        player_module.PLAYERS[player_ids[i]].set_turn_order(state, i)

    # Set active player to new position 0 (REORDER-03)
    state._set_active_player(player_ids[0])


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Reorder players by descending cash (tie-break by old position)
    2. Set active player to new position 0
    3. Clear consecutive passes for next INVEST round
    4. FI purchases cheapest available companies at face value
    5. All revealed companies become available for auction
    6. Set up ACQUISITION phase (generate offers, present first)
    7. Transition to ACQUISITION

    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)

    # Phase 10: FI purchases and availability transition
    _process_fi_purchases(state)
    _make_all_revealed_available(state)

    # Set up acquisition phase (generate offers, present first)
    acquisition_module.setup_acquisition_phase(state)

    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0


def apply_wrap_up_py(GameState state):
    """Python wrapper for testing."""
    return apply_wrap_up(state)
