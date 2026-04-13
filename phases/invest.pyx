"""INVEST phase handler.

Handles the four INVEST actions: PASS, AUCTION (start), BUY_SHARE, SELL_SHARE.

- PASS: increments consecutive_passes. Once all players have passed in a row,
  clears per-player round-trip tracking scratch and transitions to
  ``PHASE_WRAP_UP``.
- AUCTION: seeds auction state (company, price, high bidder, starter),
  clears per-player has_passed flags, transitions to ``PHASE_BID``,
  and advances past the starter to the next bidder.
- BUY_SHARE: moves corp price up to the next available index, charges the
  player the *new* (higher) price, transfers the share, tracks the buy.
  Buying the $75 slot (market index 26) immediately transitions to
  ``PHASE_GAME_OVER`` (RULES.md game-end condition #1).
- SELL_SHARE: transfers the share first (so presidency / receivership update
  before downstream reads), moves the corp price down, pays the player the
  *new* (lower) price, tracks the sell. Hitting market index 0 triggers
  ``corp.go_bankrupt()`` which fully cleans up.

All state access goes through entity handles — the handler imports no
layout constants and never indexes ``state._data`` directly. Cache
invalidation (player finance, corp derived income/stars) happens inside
the entity handle methods, so there is no manual net-worth or income
refresh here.
"""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_AUCTION,
    ACTION_BUY_SHARE,
    ACTION_SELL_SHARE,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    COMPANY_FACE_VALUE,
    MARKET_PRICES,
)
from entities.company cimport company_is_for_auction

# Late Python-level entity imports. ``phases/`` sits above ``entities/`` in
# the dependency DAG (nobody imports phases), so there is no cycle to work
# around — we can import the singleton modules at module load and dispatch
# through them for every call.
from entities import turn as turn_module
from entities import player as player_module
from entities import corp as corp_module
from entities import market as market_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef inline void _advance_active_player(GameState state) noexcept:
    """Advance the active-player slot to the next player in turn order."""
    cdef int current = turn_module.TURN.get_active_player(state)
    turn_module.TURN.set_active_player_after(state, current)


# =============================================================================
# ACTION-SPECIFIC HANDLERS
# =============================================================================

cdef void _handle_pass(GameState state) noexcept:
    """INVEST pass: bump consecutive_passes; end phase if all have passed."""
    cdef int i, num_players

    turn_module.TURN.increment_consecutive_passes(state)
    num_players = turn_module.TURN.get_num_players(state)

    if turn_module.TURN.get_consecutive_passes(state) >= num_players:
        # All players passed consecutively. INVEST owns the per-player
        # round-trip scratch fields (share_buys / share_sells),
        # so clear them here before handing control to WRAP_UP — the token
        # features for later phases should not see stale INVEST-turn data.
        for i in range(num_players):
            player_module.PLAYERS[i].clear_roundtrip_tracking(state)
        turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_WRAP_UP)
        return

    _advance_active_player(state)


cdef void _handle_auction(GameState state, int company_id, int bid_offset) noexcept:
    """INVEST auction start: set up auction context, transition to BID."""
    cdef int starter = turn_module.TURN.get_active_player(state)
    cdef int face_value = COMPANY_FACE_VALUE[company_id]
    cdef int bid_price = face_value + bid_offset

    # Defensive invariant: the enumerator only emits auctions for
    # LOC_AUCTION companies. If this fires, the enumerator and the
    # handler have fallen out of sync.
    assert company_is_for_auction(state, company_id), \
        f"_handle_auction: company {company_id} is not LOC_AUCTION"

    # Seed auction context. ``active_company`` is reused by BID as the
    # auction target — no dedicated ``auction_company`` slot exists in
    # the turn block.
    turn_module.TURN.set_active_company(state, company_id)
    turn_module.TURN.set_auction_price(state, bid_price)
    turn_module.TURN.set_auction_high_bidder(state, starter)
    turn_module.TURN.set_auction_starter(state, starter)

    # Fresh auction: nobody has left yet. Clear any leftover has_passed
    # flags defensively — BID treats has_passed as "left the auction",
    # and INVEST itself does not use has_passed so a stray set flag from
    # a previous auction or an earlier test setup must not carry over.
    turn_module.TURN.clear_passed_flags(state)

    # Non-pass action resets the pass counter.
    turn_module.TURN.clear_consecutive_passes(state)

    # Transition to BID. The starter has already placed the opening bid
    # (recorded in auction_price / auction_high_bidder), so control moves
    # to the next non-passed bidder. With freshly-cleared has_passed flags
    # this is just the next player in turn order.
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_BID)
    turn_module.TURN.advance_to_next_bidder(state)


cdef void _handle_buy_share(GameState state, int corp_id) noexcept:
    """INVEST buy-share: move price up, charge new price, transfer share."""
    cdef int player_id = turn_module.TURN.get_active_player(state)
    cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1  # 26 == $75
    cdef int current_index, new_index, new_price, current_shares

    # Price moves BEFORE payment — player pays the *new* (higher) price.
    # RULES.md "Buy One Share": corp returns its card, takes the next
    # higher available card, and the player pays the new price.
    current_index = corp_module.CORPS[corp_id].get_price_index(state)
    new_index = market_module.MARKET.find_next_higher_space(state, current_index)

    # Free the old slot first, then occupy the new one (unless the new
    # slot is the always-shared $75 sentinel at index 26).
    market_module.MARKET.set_space_available(state, current_index, True)
    corp_module.CORPS[corp_id].set_price_index(state, new_index)
    if new_index != max_index:
        market_module.MARKET.set_space_available(state, new_index, False)

    new_price = MARKET_PRICES[new_index]

    # Pay new price to bank (money leaves circulation). Cache dirty bits
    # are flipped inside add_cash / set_shares / set_price_index, so there
    # is no manual net-worth or income refresh here.
    player_module.PLAYERS[player_id].add_cash(state, -new_price)

    # Transfer the share. set_shares auto-adjusts bank_shares and
    # recalculates presidency / receivership for this corp.
    current_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, current_shares + 1)

    # Round-trip tracking (INVEST scratch).
    player_module.PLAYERS[player_id].increment_share_buys(state, corp_id)

    # Non-pass action resets the pass counter.
    turn_module.TURN.clear_consecutive_passes(state)

    # $75 game-end trigger: firing the moment a corp lands on index 26.
    # No active-player advance — the driver will see PHASE_GAME_OVER and
    # stop the game loop.
    if new_index == max_index:
        turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_GAME_OVER)
        return

    _advance_active_player(state)


cdef void _handle_sell_share(GameState state, int corp_id) noexcept:
    """INVEST sell-share: transfer share, move price down, pay new price."""
    cdef int player_id = turn_module.TURN.get_active_player(state)
    cdef int current_index, new_index, sell_price, current_shares

    # Peek at the price-index transition first so we can short-circuit into
    # bankruptcy without any intermediate state mutation. go_bankrupt() is
    # the single source of truth for bankruptcy cleanup (market-space free,
    # company removal, share return, corp deactivation); we call it with
    # the corp still sitting at its current pre-sell state so it can free
    # the right market slot on its own.
    current_index = corp_module.CORPS[corp_id].get_price_index(state)
    new_index = market_module.MARKET.find_next_lower_space(state, current_index)

    if new_index == 0:
        # Bankruptcy. Skip the share transfer and the $0 payment —
        # go_bankrupt() zeros all player shares for this corp as part of
        # its normal teardown, and MARKET_PRICES[0] == 0 so there is no
        # cash to pay. No round-trip tracking: the corp no longer exists
        # and there is no corp_id to track against.
        corp_module.CORPS[corp_id].go_bankrupt(state)
        turn_module.TURN.clear_consecutive_passes(state)
        _advance_active_player(state)
        return

    # Normal case: transfer the share, move the price, pay the player.
    #
    # Transfer the share FIRST. Important because set_shares(0) may flip
    # the corp into receivership when the last president sells out; that
    # flip has to be visible before any downstream reads look at the corp.
    current_shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
    player_module.PLAYERS[player_id].set_shares(state, corp_id, current_shares - 1)

    # Move price down BEFORE paying the player — the player receives the
    # *new* (lower) price, not the old one.
    market_module.MARKET.set_space_available(state, current_index, True)
    corp_module.CORPS[corp_id].set_price_index(state, new_index)
    market_module.MARKET.set_space_available(state, new_index, False)

    sell_price = MARKET_PRICES[new_index]
    player_module.PLAYERS[player_id].add_cash(state, sell_price)

    # Round-trip tracking (INVEST scratch).
    player_module.PLAYERS[player_id].increment_share_sells(state, corp_id)

    turn_module.TURN.clear_consecutive_passes(state)
    _advance_active_player(state)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_invest_action(GameState state, ActionInfo* info) noexcept:
    """Apply an INVEST-phase action to ``state``.

    ``info`` is assumed to be a legal INVEST action produced by
    ``decode_action(DPHASE_INVEST, action_id)`` after the id was yielded
    by ``_enumerate_invest``. Illegal actions are a driver bug and are
    caught by the assertion on ``action_type`` below.
    """
    cdef int action_type = info.action_type

    if action_type == ACTION_PASS:
        _handle_pass(state)
    elif action_type == ACTION_AUCTION:
        _handle_auction(state, info.company_id, info.amount)
    elif action_type == ACTION_BUY_SHARE:
        _handle_buy_share(state, info.corp_id)
    elif action_type == ACTION_SELL_SHARE:
        _handle_sell_share(state, info.corp_id)
    else:
        raise AssertionError(
            f"apply_invest_action: illegal action_type {action_type}"
        )
