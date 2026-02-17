# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""INVEST phase handler implementation."""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo, ActionType,
    ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE
)
from entities import turn as turn_module
from entities import player as player_module
from entities import corp as corp_module
from entities import market as market_module
from entities import company as company_module
from entities.company cimport get_auction_company_for_slot
from core.data cimport GamePhases, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_WRAP_UP, PHASE_GAME_OVER, get_company_face_value, get_market_price, GameConstants
from core.data import CORP_NAMES


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Note: Presidency and receivership are now automatically recalculated
# whenever shares change via player.set_shares(). See entities/player.pyx
# _recalculate_presidency() for the implementation.

cdef void _advance_active_player(GameState state) noexcept:
    """Advance to next player in turn order."""
    cdef int current_player = state._get_active_player()
    cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
    cdef int next_position = (current_position + 1) % state._num_players
    cdef int next_player = turn_module.TURN.find_player_at_position(state, next_position)
    state._set_active_player(next_player)


cdef void _handle_buy_share(GameState state, int corp_id) noexcept:
    """
    Handle buy share action.

    Per RULES.md: Price moves BEFORE payment (player pays new price)

    Sequence:
    1. Find new price after price movement (skipping occupied spaces)
    2. Update market space availability
    3. Transfer money (player pays new price to corp)
    4. Transfer share (bank to player)
    5. Track buy (for training loop prevention)
    6. Update net worth
    7. Reset consecutive passes
    8. Advance to next player
    """
    cdef int player_id, current_index, new_index, new_price
    cdef int bank_shares, player_shares
    cdef object corp  # Corporation entity

    # Get active player
    player_id = state._get_active_player()

    # Get corp by name lookup (existing pattern from corp.pyx)
    corp = corp_module.CORPS[corp_id]

    # Get current price index and find new index
    current_index = corp.get_price_index(state)
    new_index = market_module.MARKET.find_next_higher_space(state, current_index)
    new_price = get_market_price(new_index)

    # Update market space availability
    market_module.MARKET.set_space_available(state, current_index, True)  # Free old
    corp.set_price_index(state, new_index)  # Updates price too
    if new_index != 26:  # Price 75 is always available, don't mark occupied
        market_module.MARKET.set_space_available(state, new_index, False)

    # Transfer money: player pays to bank (INV-07)
    # Per RULES.md: "Player pays new share price to Bank" - money leaves circulation
    player_module.PLAYERS[player_id].add_cash(state, -new_price)

    # Transfer share (INV-09)
    # set_shares() automatically adjusts bank shares, presidency, and receivership
    player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares + 1)

    # Round-trip tracking (INV-16)
    player_module.PLAYERS[player_id].increment_share_buys(state, corp_id)

    # Update net worth for all players (price movement affects all shareholders)
    player_module.update_all_net_worths(state)

    # Reset consecutive passes (INV-02)
    turn_module.TURN.clear_consecutive_passes(state)

    # Check for $75 game end - immediate after buy completes (RULES.md line 346)
    if new_index == 26:
        turn_module.TURN.set_phase(state, PHASE_GAME_OVER)
        return

    # Advance active player
    _advance_active_player(state)


cdef void _handle_sell_share(GameState state, int corp_id) noexcept:
    """
    Handle sell share action.

    Per RULES.md: Price moves down FIRST, then player receives NEW (lower) price.

    Sequence:
    1. Transfer share (player to bank)
    2. Move price down (skipping occupied spaces)
    3. Pay player the NEW (lower) price
    4. If price reaches 0, execute bankruptcy and return
    5. Track sell (for training loop prevention)
    6. Update net worth
    7. Reset consecutive passes
    8. Advance to next player
    """
    cdef int player_id, current_index, new_index, sell_price
    cdef int bank_shares, player_shares
    cdef object corp  # Corporation entity

    # Get active player
    player_id = state._get_active_player()

    # Get corp by name lookup
    corp = corp_module.CORPS[corp_id]

    # Transfer share first (INV-12)
    # set_shares() automatically adjusts bank shares, presidency, and receivership
    player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares - 1)

    # Move price down (INV-13) - BEFORE paying player
    current_index = corp.get_price_index(state)
    new_index = market_module.MARKET.find_next_lower_space(state, current_index)
    market_module.MARKET.set_space_available(state, current_index, True)  # Free old
    corp.set_price_index(state, new_index)  # Updates price

    # Pay player the NEW (lower) price (INV-11)
    sell_price = get_market_price(new_index)
    player_module.PLAYERS[player_id].add_cash(state, sell_price)

    # Check for bankruptcy (INV-22)
    if new_index == 0:
        corp.go_bankrupt(state)  # go_bankrupt updates all net worths
        # Reset consecutive passes (INV-02)
        turn_module.TURN.clear_consecutive_passes(state)
        # Advance active player
        _advance_active_player(state)
        return  # Skip remaining steps - corp is gone

    # Occupy new space (non-bankruptcy case)
    market_module.MARKET.set_space_available(state, new_index, False)
    # Note: set_shares() above already updated receivership and presidency

    # Round-trip tracking (INV-16)
    player_module.PLAYERS[player_id].increment_share_sells(state, corp_id)

    # Update net worth for all players (price movement affects all shareholders)
    player_module.update_all_net_worths(state)

    # Reset consecutive passes (INV-02)
    turn_module.TURN.clear_consecutive_passes(state)

    # Advance active player
    _advance_active_player(state)


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_invest_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply INVEST phase action to state.

    Returns: 0=success, 1=invalid
    """
    cdef int company_id, face_value, bid_price, player_id, i

    if info.action_type == ACTION_PASS:
        # Increment consecutive_passes counter
        turn_module.TURN.increment_consecutive_passes(state)

        # Check if all players have passed
        if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
            # Clear buy/sell tracking at end of INVEST phase (bookkeeping only,
            # avoids exposing stale data to model in subsequent phases)
            for i in range(state._num_players):
                player_module.PLAYERS[i].clear_roundtrip_tracking(state)

            # All players passed - transition to WRAP_UP phase
            turn_module.TURN.set_phase(state, PHASE_WRAP_UP)
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
        # Note: auction_passed flags are cleared at auction END (bid.pyx),
        # not at start - they're initialized cleared and stay cleared between auctions

        # Clear consecutive passes (INV-02)
        turn_module.TURN.clear_consecutive_passes(state)

        # Transition to BID_IN_AUCTION phase
        turn_module.TURN.set_phase(state, PHASE_BID_IN_AUCTION)

        # Advance to next bidder (skipping passed players)
        turn_module.TURN.advance_to_next_bidder(state)

        return 0

    elif info.action_type == ACTION_BUY_SHARE:
        _handle_buy_share(state, info.corp_id)
        return 0

    elif info.action_type == ACTION_SELL_SHARE:
        _handle_sell_share(state, info.corp_id)
        return 0

    return 1  # Invalid action type for INVEST phase
