"""BID phase handler.

Handles the two BID actions: LEAVE (pass-class) and RAISE.

- LEAVE: mark the active player ``has_passed`` and either resolve the
  auction (when only one bidder remains) or advance control to the next
  non-passed bidder in turn order.
- RAISE: set the new auction price to ``face_value + 1 + bid_offset``,
  update the high bidder to the raising player, and advance control to
  the next non-passed bidder. The raising player is never marked passed.

When the auction resolves, the winner pays the final bid to the bank,
receives the company, a replacement card is drawn (``LOC_REVEALED`` —
becomes auctionable in ``WRAP_UP``), every auction scratch field in the
turn block is cleared, the phase transitions back to ``PHASE_INVEST``,
and control goes to the player *after* the auction starter (per RULES
§Auction step 5) — not the winner.

All state access goes through entity handles. The handler does not
import layout constants and never indexes ``state._data`` directly.
"""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_RAISE,
)
from core.data cimport (
    GamePhases,
    AUCTION_CAP,
    COMPANY_FACE_VALUE,
)

# Late Python-level entity imports, same pattern as phases/invest.pyx.
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import deck as deck_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef int _count_active_bidders(GameState state) noexcept:
    """Return the number of players who have not yet left the auction."""
    cdef int i
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int count = 0
    for i in range(num_players):
        if not player_module.PLAYERS[i].has_passed(state):
            count += 1
    return count


cdef void _resolve_auction(GameState state) noexcept:
    """Finish the auction, hand the company to the winner, hand control back.

    Winner pays the final bid to the bank, receives the company, the deck
    draws a replacement into ``LOC_REVEALED`` (``WRAP_UP`` later flips it
    to ``LOC_AUCTION``), every auction scratch field in the turn block is
    cleared, the phase transitions back to ``PHASE_INVEST``, and control
    goes to the player *after* the starter. An empty deck at resolution
    time is legal — ``Deck.draw`` returns -1 and mutates nothing.
    """
    cdef int winner = turn_module.TURN.get_auction_high_bidder(state)
    cdef int starter = turn_module.TURN.get_auction_starter(state)
    cdef int company_id = turn_module.TURN.get_active_company(state)
    cdef int price = turn_module.TURN.get_auction_price(state)

    # Defensive invariant: the driver must have seeded a live auction
    # before dispatching any BID action. Both fields being valid is the
    # precondition for _handle_leave / _handle_raise calling us.
    assert winner >= 0, "_resolve_auction: high bidder unset"
    assert company_id >= 0, "_resolve_auction: active_company unset"

    # Pay the bid to the bank (money leaves circulation). Cache dirty
    # bits are flipped inside add_cash / transfer_to_player, so there is
    # no manual net-worth refresh here.
    player_module.PLAYERS[winner].add_cash(state, -price)
    company_module.COMPANIES[company_id].transfer_to_player(state, winner)

    # Draw the replacement card. It lands in LOC_REVEALED; WRAP_UP will
    # flip it to LOC_AUCTION. An empty deck is not an error — draw()
    # returns -1 and mutates nothing in that case.
    deck_module.DECK.draw(state)

    # Clear all auction scratch fields in the turn block.
    turn_module.TURN.clear_active_company(state)
    turn_module.TURN.set_auction_price(state, 0)
    turn_module.TURN.clear_auction_high_bidder(state)
    turn_module.TURN.clear_auction_starter(state)
    # Defensive clear — no later phase currently reads has_passed, but
    # keep it clean so a stray flag can't leak into a future phase.
    turn_module.TURN.clear_passed_flags(state)

    # Hand control back to INVEST. Next action goes to the player *after*
    # the starter (RULES.md §Auction step 5) — not the winner.
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_INVEST)
    turn_module.TURN.set_active_player_after(state, starter)


# =============================================================================
# ACTION-SPECIFIC HANDLERS
# =============================================================================

cdef void _handle_leave(GameState state) noexcept:
    """Active player leaves the auction; resolve or advance to next bidder."""
    cdef int pid = turn_module.TURN.get_active_player(state)

    player_module.PLAYERS[pid].set_has_passed(state, True)

    if _count_active_bidders(state) == 1:
        _resolve_auction(state)
        return

    # At least two bidders remain, one of whom is not the just-left
    # player, so advance_to_next_bidder is guaranteed to find a target.
    turn_module.TURN.advance_to_next_bidder(state)


cdef void _handle_raise(GameState state, int bid_offset) noexcept:
    """Active player raises to ``face_value + 1 + bid_offset``."""
    cdef int pid = turn_module.TURN.get_active_player(state)
    cdef int company_id = turn_module.TURN.get_active_company(state)
    cdef int new_bid

    # Defensive invariants (compile out under python -O). These catch
    # driver/enumerator drift, not rule-level legality.
    assert 0 <= bid_offset < <int>AUCTION_CAP - 1, \
        f"_handle_raise: bid_offset {bid_offset} out of [0, {<int>AUCTION_CAP - 1})"
    assert company_id >= 0, "_handle_raise: active_company unset"

    new_bid = COMPANY_FACE_VALUE[company_id] + bid_offset + 1

    assert new_bid > turn_module.TURN.get_auction_price(state), \
        f"_handle_raise: new bid {new_bid} not greater than current"
    assert new_bid <= player_module.PLAYERS[pid].get_cash(state), \
        f"_handle_raise: new bid {new_bid} exceeds player {pid} cash"

    turn_module.TURN.set_auction_price(state, new_bid)
    turn_module.TURN.set_auction_high_bidder(state, pid)

    # The raising player stays in the auction — has_passed is untouched.
    # The previous high bidder is still in, so there is always a distinct
    # next non-passed player for advance_to_next_bidder to land on.
    turn_module.TURN.advance_to_next_bidder(state)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_bid_action(GameState state, ActionInfo* info) noexcept:
    """Apply a BID-phase action to ``state``.

    ``info`` is assumed to be a legal BID action produced by
    ``decode_action(DPHASE_BID, action_id)`` after the id was yielded by
    ``_enumerate_bid``. Illegal actions are a driver bug and trip the
    assertion on ``action_type`` below.
    """
    cdef int action_type = info.action_type

    if action_type == ACTION_PASS:
        _handle_leave(state)
    elif action_type == ACTION_RAISE:
        _handle_raise(state, info.amount)
    else:
        raise AssertionError(
            f"apply_bid_action: illegal action_type {action_type}"
        )
