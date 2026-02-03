# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""END_CARD phase handler implementation.

This is a deterministic non-player phase (0 actions) that checks game-ending
conditions and handles the end card flip mechanic.

Checks executed in order:
1. If 75 share price reached by any corp → GAME_OVER
2. If no unowned companies left → flip end card, set CoO to 7
3. If end card already flipped → GAME_OVER
4. Otherwise → transition to ISSUE_SHARES
"""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants
from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities.company cimport LOC_DECK, LOC_AUCTION, LOC_REVEALED
from phases.issue cimport setup_issue_phase


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef bint _check_75_price_reached(GameState state) noexcept:
    """
    Check if any active corporation has reached the $75 price (index 26).

    Returns: True if game should end due to 75 price reached
    """
    cdef int corp_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        corp = corp_module.CORPS[corp_id]
        if corp.is_active(state) and corp.get_price_index(state) == 26:
            return True
    return False


cdef bint _check_no_unowned_companies(GameState state) noexcept:
    """
    Check if there are no unowned companies remaining.

    Unowned = in deck, auction slots, or revealed (drawn but not auctionable).

    Returns: True if no unowned companies exist
    """
    cdef int company_id, loc

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        company = company_module.COMPANIES[company_id]
        loc = company.get_location(state)
        if loc == LOC_DECK or loc == LOC_AUCTION or loc == LOC_REVEALED:
            return False
    return True


cdef void _flip_end_card(GameState state) noexcept:
    """
    Flip the end card and set CoO level to 7.

    This happens when there are no unowned companies remaining.
    The game will end at the next END_CARD phase check.
    """
    turn_module.TURN.set_end_card_flipped(state, True)
    turn_module.TURN.set_coo_level(state, <int>GameConstants.COO_LEVEL_END_CARD_FLIPPED)


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_end_card(GameState state) noexcept:
    """
    Execute END_CARD phase logic.

    This is a deterministic non-player phase with 0 actions.
    Checks are executed in order per RULES.md:

    1. If 75 share price reached → GAME_OVER
    2. If no unowned companies → flip end card, set CoO to 7
    3. If end card already flipped → GAME_OVER
    4. Otherwise → ISSUE_SHARES

    Returns: 0 always (deterministic, no failure modes)
    """
    # Check 1: 75 share price reached
    if _check_75_price_reached(state):
        turn_module.TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        return 0

    # Check 2: No unowned companies → flip end card
    if _check_no_unowned_companies(state):
        _flip_end_card(state)

    # Check 3: End card already flipped → game over
    if turn_module.TURN.is_end_card_flipped(state):
        turn_module.TURN.set_phase(state, GamePhases.PHASE_GAME_OVER)
        return 0

    # Check 4: Normal transition to ISSUE_SHARES phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)
    setup_issue_phase(state)
    return 0


def apply_end_card_py(GameState state):
    """Python wrapper for testing."""
    return apply_end_card(state)
