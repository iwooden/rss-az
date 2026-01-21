# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""BID_IN_AUCTION phase handler implementation."""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo, ActionType,
    ACTION_LEAVE_AUCTION, ACTION_RAISE_BID
)
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import deck as deck_module
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


cdef int _count_active_bidders(GameState state) noexcept:
    """Count players who haven't left auction."""
    cdef int count = 0
    cdef int player_id
    for player_id in range(state._num_players):
        if not turn_module.TURN.has_player_passed_auction(state, player_id):
            count += 1
    return count


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


cdef void _set_active_player_after(GameState state, int player_id) noexcept:
    """Set active player to next player after given player in turn order."""
    cdef int position = player_module.PLAYERS[player_id].get_turn_order(state)
    cdef int next_position = (position + 1) % state._num_players
    cdef int next_player = _find_player_at_position(state, next_position)
    state._set_active_player(next_player)


cdef void _resolve_auction(GameState state) noexcept:
    """Complete auction resolution sequence."""
    cdef int winner_id = turn_module.TURN.get_auction_high_bidder(state)
    cdef int starter_id = turn_module.TURN.get_auction_starter(state)
    cdef int company_id = turn_module.TURN.get_auction_company(state)
    cdef int price = turn_module.TURN.get_auction_price(state)
    cdef int new_company

    # Winner pays bid price (BID-06)
    player_module.PLAYERS[winner_id].add_cash(state, -price)

    # Winner receives company (BID-07)
    company_module.COMPANIES[company_id].transfer_to_player(state, winner_id)

    # Update winner's net worth (BID-12)
    player_module.PLAYERS[winner_id].update_net_worth(state)

    # Draw new company to auction row (BID-09)
    new_company = deck_module.DECK.draw(state)
    if new_company >= 0:
        company_module.COMPANIES[new_company].move_to_auction(state)

    # Clear auction state (BID-08)
    turn_module.TURN.clear_auction_company(state)
    turn_module.TURN.clear_auction_high_bidder(state)
    turn_module.TURN.clear_auction_starter(state)
    turn_module.TURN.clear_auction_passed(state)
    turn_module.TURN.set_auction_price(state, -1)

    # Return to INVEST phase (BID-10)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    # Next player after starter (BID-11)
    _set_active_player_after(state, starter_id)


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_bid_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply BID_IN_AUCTION phase action to state.

    Returns: 0=success, 1=invalid
    """
    cdef int player_id, company_id, new_bid, active_bidders

    if info.action_type == ACTION_LEAVE_AUCTION:
        # Get current player
        player_id = state._get_active_player()

        # Set passed flag (BID-01)
        turn_module.TURN.set_player_passed_auction(state, player_id, True)

        # Count remaining bidders
        active_bidders = _count_active_bidders(state)

        if active_bidders == 1:
            # Auction resolves (BID-05)
            _resolve_auction(state)
        else:
            # Continue auction - advance to next bidder (BID-02)
            _advance_to_next_bidder(state)

        return 0

    elif info.action_type == ACTION_RAISE_BID:
        # Get auction company
        company_id = turn_module.TURN.get_auction_company(state)

        # Calculate new bid: face value + info.amount + 1
        # info.amount is 0-18, representing bids from face+1 to face+19
        new_bid = get_company_face_value(company_id) + info.amount + 1

        # Get current player
        player_id = state._get_active_player()

        # Update auction state (BID-03)
        turn_module.TURN.set_auction_price(state, new_bid)
        turn_module.TURN.set_auction_high_bidder(state, player_id)

        # Advance to next bidder (BID-04)
        _advance_to_next_bidder(state)

        return 0

    return 1  # Invalid action type for BID phase
