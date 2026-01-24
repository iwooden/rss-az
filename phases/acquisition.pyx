# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""ACQUISITION phase stub - transitions immediately to INVEST."""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from core.data import CORP_NAMES


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
