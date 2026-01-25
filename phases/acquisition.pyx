# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""ACQUISITION phase stub - transitions immediately to INVEST."""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants, get_company_face_value, get_company_low_price
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
# HELPER FUNCTIONS
# =============================================================================

cdef int _get_corp_president(GameState state, int corp_id) noexcept:
    """
    Find the player with the most shares in a corporation.

    Returns player_id of president, or -1 if corp is in receivership.
    """
    cdef int player_id, max_shares, shares, president
    max_shares = 0
    president = -1

    for player_id in range(state._num_players):
        shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
        if shares > max_shares:
            max_shares = shares
            president = player_id

    return president


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
# OFFER STATE PRESENTATION
# =============================================================================

cdef void _present_current_offer(GameState state) noexcept:
    """
    Update visible state to reflect current offer in buffer.

    Reads offer at current index from hidden buffer.
    Sets acq_active_corp, acq_target_company, acq_is_fi_offer.
    Sets active_player to president of buying corp (or -1 for receivership).

    STATE-01: Sets visible acquisition state for current offer.
    STATE-04: Clears acq_active_corp when no more offers.
    """
    cdef int count = <int>state._data[state._layout.hidden_offer_count_offset]
    cdef int index = <int>state._data[state._layout.hidden_offer_index_offset]
    cdef int corp_id, company_id, president

    # No more offers (STATE-04)
    if index >= count:
        turn_module.TURN.clear_acq_active_corp(state)
        turn_module.TURN.clear_acq_target_company(state)
        turn_module.TURN.set_acq_fi_offer(state, False)
        return

    # Read current offer from buffer
    cdef int base = state._layout.hidden_offer_buffer_offset + (index * 2)
    corp_id = <int>state._data[base]
    company_id = <int>state._data[base + 1]

    # Set visible state (STATE-01)
    turn_module.TURN.set_acq_active_corp(state, corp_id)
    turn_module.TURN.set_acq_target_company(state, company_id)
    turn_module.TURN.set_acq_fi_offer(state, fi_module.FI.owns_company(state, company_id))

    # Set active player to president of buying corp
    president = _get_corp_president(state, corp_id)
    state._set_active_player(president if president >= 0 else 0)


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
