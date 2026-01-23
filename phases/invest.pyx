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
from core.data cimport GamePhases, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_WRAP_UP, PHASE_GAME_OVER, get_company_face_value, get_market_price, get_corp_share_count, GameConstants
from core.data import CORP_NAMES


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _check_receivership(GameState state, int corp_id) noexcept:
    """
    Check if corporation enters or exits receivership.

    Receivership = all player-owned shares are 0 (bank owns all issued shares).
    """
    cdef int player_id, total_player_shares
    cdef object corp

    corp = corp_module.CORPS[CORP_NAMES[corp_id]]
    total_player_shares = 0

    for player_id in range(state._num_players):
        total_player_shares += player_module.PLAYERS[player_id].get_shares(state, corp_id)

    if total_player_shares == 0:
        corp.set_in_receivership(state, True)
        # Clear all president flags - no president in receivership
        for player_id in range(state._num_players):
            player_module.PLAYERS[player_id].set_president_of(state, corp_id, False)
    else:
        corp.set_in_receivership(state, False)


cdef void _check_presidency(GameState state, int corp_id) noexcept:
    """
    Check if presidency should transfer.

    President = player with most shares. Tie-breaking: incumbent keeps it.
    Per CONTEXT.md: "current president keeps it when shares are equal"
    """
    cdef int player_id, shares, max_shares, president_id, current_president, incumbent_shares
    cdef object corp

    corp = corp_module.CORPS[CORP_NAMES[corp_id]]

    # Skip if in receivership (no president)
    if corp.is_in_receivership(state):
        return

    # Find current president
    current_president = -1
    for player_id in range(state._num_players):
        if player_module.PLAYERS[player_id].is_president_of(state, corp_id):
            current_president = player_id
            break

    # Find player with most shares
    # Incumbent advantage: on ties, incumbent keeps presidency
    # We handle this by initializing with incumbent's shares if they exist
    max_shares = 0
    president_id = -1

    # First pass: find the maximum share count
    for player_id in range(state._num_players):
        shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
        if shares > max_shares:
            max_shares = shares

    # Second pass: find winner (incumbent wins ties)
    # If incumbent has max_shares, they keep it
    # Otherwise, first player with max_shares wins
    if max_shares > 0:
        if current_president >= 0:
            incumbent_shares = player_module.PLAYERS[current_president].get_shares(state, corp_id)
            if incumbent_shares == max_shares:
                # Incumbent ties for max - they keep presidency
                president_id = current_president

        # If incumbent doesn't have max shares, find first player who does
        if president_id < 0:
            for player_id in range(state._num_players):
                shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
                if shares == max_shares:
                    president_id = player_id
                    break

    # Update if changed (and someone has shares)
    if president_id >= 0 and president_id != current_president:
        if current_president >= 0:
            player_module.PLAYERS[current_president].set_president_of(state, corp_id, False)
        player_module.PLAYERS[president_id].set_president_of(state, corp_id, True)
    elif president_id < 0 and current_president >= 0:
        # No one has shares but there was a president - clear (shouldn't happen if receivership check ran first)
        player_module.PLAYERS[current_president].set_president_of(state, corp_id, False)


cdef void _execute_bankruptcy(GameState state, int corp_id) noexcept:
    """
    Execute bankruptcy procedure for a corporation (INV-22 through INV-27).

    Triggered when share price drops to index 0. This is a complete reset:
    - All owned companies removed from game
    - All shares returned to unissued (cleared from players)
    - Corp cash returned to bank (set to 0)
    - Market space freed
    - Corp deactivated and available for future IPO

    Args:
        state: Game state to modify
        corp_id: Corporation that went bankrupt
    """
    cdef int company_id, player_id, current_index
    cdef object corp

    # Get corp entity
    corp = corp_module.CORPS[CORP_NAMES[corp_id]]

    # Step 1: Remove all owned companies from game (INV-23)
    for company_id in range(GameConstants.NUM_COMPANIES):
        if corp.owns_company(state, company_id):
            company_module.COMPANIES[company_id].remove_from_game(state)
            corp.set_owns_company(state, company_id, False)

    # Step 2: Return all shares to unissued - clear player shares first (INV-24)
    for player_id in range(state._num_players):
        player_module.PLAYERS[player_id].set_shares(state, corp_id, 0)
        player_module.PLAYERS[player_id].set_president_of(state, corp_id, False)

    # Step 3: Reset corp share counts (INV-24)
    corp.set_unissued_shares(state, get_corp_share_count(corp_id))
    corp.set_issued_shares(state, 0)
    corp.set_bank_shares(state, 0)

    # Step 4: Return money to bank - clear corp cash (INV-25)
    corp.set_cash(state, 0)

    # Step 5: Free market space if needed (INV-26)
    # The sell handler already freed the old space, but verify current state
    current_index = corp.get_price_index(state)
    if current_index > 0:
        market_module.MARKET.set_space_available(state, current_index, True)

    # Step 6: Deactivate corp and clear remaining state (INV-27)
    corp.set_active(state, False)
    corp.set_price_index(state, 0)
    corp.set_in_receivership(state, False)
    corp.set_income(state, 0)
    corp.set_stars(state, 0)
    corp.set_acquisition_proceeds(state, 0)

    # Step 7: Clear acquisition company flags
    for company_id in range(GameConstants.NUM_COMPANIES):
        corp.set_acquisition_company(state, company_id, False)


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

    Per CONTEXT.md: Price moves BEFORE payment (player pays new price)

    Sequence:
    1. Find new price after price movement (skipping occupied spaces)
    2. Update market space availability
    3. Transfer money (player pays new price to corp)
    4. Transfer share (bank to player)
    5. Track round-trip
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
    corp = corp_module.CORPS[CORP_NAMES[corp_id]]

    # Get current price index and find new index
    current_index = corp.get_price_index(state)
    new_index = market_module.MARKET.find_next_higher_space(state, current_index)
    new_price = get_market_price(new_index)

    # Update market space availability
    market_module.MARKET.set_space_available(state, current_index, True)  # Free old
    corp.set_price_index(state, new_index)  # Updates price too
    if new_index != 26:  # Price 75 is always available, don't mark occupied
        market_module.MARKET.set_space_available(state, new_index, False)

    # Transfer money (INV-07, INV-08)
    player_module.PLAYERS[player_id].add_cash(state, -new_price)
    corp.add_cash(state, new_price)

    # Transfer share (INV-09)
    bank_shares = corp.get_bank_shares(state)
    corp.set_bank_shares(state, bank_shares - 1)
    player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares + 1)

    # Check receivership exit (INV-21 - buying from receivership clears it)
    # Per CONTEXT.md: shares are fungible, no special "president share" handling
    # Buyer becomes president simply by having the most shares (the only player with shares)
    _check_receivership(state, corp_id)

    # Check presidency (INV-18, INV-19)
    _check_presidency(state, corp_id)

    # Round-trip tracking (INV-16)
    player_module.PLAYERS[player_id].increment_share_buys(state, corp_id)

    # Update net worth for all players (price movement affects all shareholders)
    player_module.update_all_net_worths(state)

    # Reset consecutive passes (INV-02)
    turn_module.TURN.clear_consecutive_passes(state)

    # Advance active player
    _advance_active_player(state)


cdef void _handle_sell_share(GameState state, int corp_id) noexcept:
    """
    Handle sell share action.

    Per CONTEXT.md: Sell receives current price, then price moves down

    Sequence:
    1. Get current price (before movement)
    2. Transfer money (corp pays sell price to player)
    3. Transfer share (player to bank)
    4. Move price down (skipping occupied spaces)
    5. If price reaches 0, execute bankruptcy and return
    6. Track round-trip
    7. Update net worth
    8. Reset consecutive passes
    9. Advance to next player
    """
    cdef int player_id, current_index, new_index, sell_price
    cdef int bank_shares, player_shares
    cdef object corp  # Corporation entity

    # Get active player
    player_id = state._get_active_player()

    # Get corp by name lookup
    corp = corp_module.CORPS[CORP_NAMES[corp_id]]

    # Get current price BEFORE movement (INV-11)
    current_index = corp.get_price_index(state)
    sell_price = get_market_price(current_index)

    # Transfer money (INV-11)
    player_module.PLAYERS[player_id].add_cash(state, sell_price)

    # Transfer share (INV-12)
    player_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, player_shares - 1)
    bank_shares = corp.get_bank_shares(state)
    corp.set_bank_shares(state, bank_shares + 1)

    # Move price down (INV-13)
    new_index = market_module.MARKET.find_next_lower_space(state, current_index)
    market_module.MARKET.set_space_available(state, current_index, True)  # Free old
    corp.set_price_index(state, new_index)  # Updates price

    # Check for bankruptcy (INV-22)
    if new_index == 0:
        _execute_bankruptcy(state, corp_id)
        # Update net worth for all players (bankruptcy affects all shareholders)
        player_module.update_all_net_worths(state)
        # Reset consecutive passes (INV-02)
        turn_module.TURN.clear_consecutive_passes(state)
        # Advance active player
        _advance_active_player(state)
        return  # Skip remaining steps - corp is gone

    # Occupy new space (non-bankruptcy case)
    market_module.MARKET.set_space_available(state, new_index, False)

    # Check receivership (INV-20) - must check before presidency
    _check_receivership(state, corp_id)

    # Check presidency (INV-18, INV-19) - only if not in receivership
    if not corp.is_in_receivership(state):
        _check_presidency(state, corp_id)

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
    cdef int company_id, face_value, bid_price, player_id

    if info.action_type == ACTION_PASS:
        # Increment consecutive_passes counter
        turn_module.TURN.increment_consecutive_passes(state)

        # Check if all players have passed
        if turn_module.TURN.get_consecutive_passes(state) >= state._num_players:
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
        turn_module.TURN.clear_auction_passed(state)

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
