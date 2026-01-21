# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""INVEST phase handler implementation."""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo, ActionType,
    ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE
)
from entities import turn as turn_module
from entities import player as player_module
from entities.company cimport get_auction_company_for_slot
from core.data cimport GamePhases, get_company_face_value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef int _find_player_at_position(GameState state, int position) noexcept:
    """Find player_id with given turn order position."""
    cdef int player_id
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].get_turn_order(state) == position:
            return player_id
    return -1


cdef void _advance_active_player(GameState state) noexcept:
    """Advance to next player in turn order."""
    cdef int current_player = state._get_active_player()
    cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
    cdef int next_position = (current_position + 1) % state._num_players
    cdef int next_player = _find_player_at_position(state, next_position)
    state._set_active_player(next_player)


cdef void _advance_to_next_bidder(GameState state) noexcept:
    """Advance to next non-passed bidder in turn order."""
    cdef int current_player = state._get_active_player()
    cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
    cdef int next_position, candidate
    cdef int checked = 0

    while checked < state._num_players:
        next_position = (current_position + 1) % state._num_players
        candidate = _find_player_at_position(state, next_position)

        if not turn_module.TURN.has_player_passed_auction(state, candidate):
            state._set_active_player(candidate)
            return

        current_position = next_position
        checked += 1

    # Should never reach here - means all players passed


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply INVEST phase action to state.

    Returns: 0=success, 1=invalid
    """
    cdef int company_id, face_value, bid_price, player_id

    if info.action_type == ACTION_PASS:
        # Increment consecutive_passes counter
        turn_module.TURN.increment_consecutive_passes(state)

        # Check if all players have passed
        if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
            # Transition to WRAP_UP phase
            turn_module.TURN.set_phase(state, GamePhases.PHASE_WRAP_UP)
        else:
            # Advance to next player in turn order
            _advance_active_player(state)

        return 0

    elif info.action_type == ACTION_AUCTION:
        # Get company_id from auction slot
        company_id = get_auction_company_for_slot(state, info.slot)
        if company_id < 0:
            return 1  # Invalid slot

        # Calculate bid price: face value + bid amount
        face_value = get_company_face_value(company_id)
        bid_price = face_value + info.amount

        # Get starter player
        player_id = state._get_active_player()

        # Initialize auction state
        turn_module.TURN.set_auction_company(state, company_id)
        turn_module.TURN.set_auction_price(state, bid_price)
        turn_module.TURN.set_auction_high_bidder(state, player_id)
        turn_module.TURN.set_auction_starter(state, player_id)
        turn_module.TURN.clear_auction_passed(state)

        # Clear consecutive passes (INV-02)
        turn_module.TURN.clear_consecutive_passes(state)

        # Transition to BID_IN_AUCTION phase
        turn_module.TURN.set_phase(state, GamePhases.PHASE_BID_IN_AUCTION)

        # Advance to next bidder (skipping passed players)
        _advance_to_next_bidder(state)

        return 0

    elif info.action_type == ACTION_BUY_SHARE:
        # TODO Phase 4: Buy share logic
        return 0

    elif info.action_type == ACTION_SELL_SHARE:
        # TODO Phase 4: Sell share logic
        return 0

    return 1  # Invalid action type for INVEST phase
