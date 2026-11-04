"""Private auction handlers; see auction.pxd.

Ruby's ``WaterfallAuction`` (``step/waterfall_auction.rb``) is the oracle.
On a normal turn the turn player passes, buys the topmost private at its
price, or selects a private to bid on and then names the amount in BID; a bid
on the topmost private is a purchase at the bid (private 0's only through
replay's ``DRIVER.override_topmost_bid``). After each sale the new
topmost private resolves in place (``resolve_bids``): without bids normal
turns resume; one bid wins it; two or more open a closed auction, in which the
lowest bidder raises in BID or drops out (BID's slot 0) until one bid is
left. Chains of sales end at a private without bids, or with the auction.

Fields:

- ``auction_turn`` is the normal turn (Ruby's ``entity_index``). It advances
  after every normal turn, and a closed auction leaves it alone;
- ``active_player`` is the actor: the turn player, or the lowest bidder in a
  closed auction;
- ``auction_private`` is the private in a closed auction, else -1;
- ``pending_private`` is BID's private outside a closed auction;
- ``priority_player`` is Ruby's ``priority_deal_player``: the auction's first
  player until a turn player buys the topmost, then the player after the last
  such buyer.
  Bid resolutions and the free A leave it alone. At the end of the auction it
  is ``auction_turn``, who opens the first SR (``reorder_players``).
"""

from g1889.actions cimport ACTION_BUY_TOPMOST, ACTION_PASS, ACTION_SELECT_BID, ActionInfo, price_amount
from g1889.constants cimport AUCTION_DISCOUNT_STEP, OWNER_UNSOLD, ROUND_AUCTION, STEP_AUCTION, STEP_BID
from g1889.data.tables cimport PRIVATE_TR
from g1889.entities.player cimport Player, all_passed, clear_passed, player_bid
from g1889.entities.private cimport (
    Private, pay_revenue, private_bid_count, private_lowest_bidder, private_owner, private_price,
    topmost_private,
)
from g1889.entities.turn cimport TurnState, turn_active_player, turn_auction_private, turn_pending_private
from g1889.phases.stock cimport start_stock_round
from g1889.state cimport GameState

from g1889.entities import player as _player, private as _private, turn as _turn

cdef list PLAYERS = _player.PLAYERS
cdef list PRIVATES = _private.PRIVATES
cdef TurnState TURN = _turn.TURN


cdef inline Player _player_at(int player):
    return <Player>PLAYERS[player]


cdef inline Private _private_at(int private):
    return <Private>PRIVATES[private]


# =============================================================================
# RESOLUTION
# =============================================================================

cdef void _end_auction(GameState state) except *:
    """Every private is sold: the turn player takes priority and opens the
    first SR, and the round's passes clear (``process_single_action``)."""
    cdef int first = TURN.auction_turn(state)
    TURN.set_auction_private(state, -1)
    clear_passed(state)
    TURN.set_priority_player(state, first)
    start_stock_round(state)


cdef void _resolve(GameState state) except *:
    """``resolve_bids``: sell the topmost private to its only bidder while it
    has one, then wait for a normal turn (no bids) or open a closed auction."""
    cdef int p, count, bidder
    while True:
        p = topmost_private(state)
        if p < 0:
            _end_auction(state)
            return
        count = private_bid_count(state, p)
        if count != 1:
            break
        bidder = private_lowest_bidder(state, p)
        _private_at(p).sell_to_player(state, bidder, player_bid(state, bidder, p))
    if count == 0:
        TURN.set_auction_private(state, -1)
        TURN.set_step(state, STEP_AUCTION)
        TURN.set_active_player(state, TURN.auction_turn(state))
    else:
        TURN.set_auction_private(state, p)
        TURN.set_step(state, STEP_BID)
        TURN.set_active_player(state, private_lowest_bidder(state, p))


cdef void _end_turn(GameState state) except *:
    """Ruby's ``next_entity_index!`` after a normal turn, then the resolution."""
    TURN.set_auction_turn(state, TURN.next_player(state, TURN.auction_turn(state)))
    _resolve(state)


cdef void _buy(GameState state, int player, int private, int price) except *:
    """The turn player buys the topmost private (``placement_bid`` on it)."""
    _player_at(player).set_passed(state, False)
    _private_at(private).sell_to_player(state, player, price)
    TURN.set_priority_player(state, TURN.next_player(state, player))   # last_to_act
    _end_turn(state)


cdef void _all_passed(GameState state) except *:
    """``all_passed!``: while A is unsold its price drops by ¥5, and at ¥0 the
    player after the last passer takes it free, which moves the turn on once
    more. Otherwise every owned private pays out. The passes clear."""
    if private_owner(state, PRIVATE_TR) == OWNER_UNSOLD:
        TURN.set_auction_discount(state, TURN.auction_discount(state) + <int>AUCTION_DISCOUNT_STEP)
        if private_price(state, PRIVATE_TR) <= 0:
            TURN.set_auction_turn(state, TURN.next_player(state, TURN.auction_turn(state)))
            _private_at(PRIVATE_TR).sell_to_player(state, TURN.auction_turn(state), 0)
    else:
        pay_revenue(state)
        TURN.set_auction_pass_payouts(state, TURN.auction_pass_payouts(state) + 1)
    clear_passed(state)


# =============================================================================
# HANDLERS
# =============================================================================

cdef void select_bid(GameState state, int private) except *:
    """Open BID on ``private``, from AUCTION's slot or replay's override."""
    TURN.set_pending_private(state, private)
    TURN.set_step(state, STEP_BID)


cdef void apply_auction(GameState state, const ActionInfo* info) except *:
    """A normal turn: pass, buy the topmost at its price, or select a private for BID."""
    cdef int player = turn_active_player(state)
    cdef int top
    if info.type == ACTION_PASS:
        _player_at(player).set_passed(state, True)
        if all_passed(state):
            _all_passed(state)
        _end_turn(state)
    elif info.type == ACTION_BUY_TOPMOST:
        top = topmost_private(state)
        _buy(state, player, top, private_price(state, top))
    else:
        assert info.type == ACTION_SELECT_BID, f"apply_auction: action type {info.type}"
        select_bid(state, info.private)


cdef void apply_bid(GameState state, const ActionInfo* info) except *:
    """In a closed auction the lowest bidder drops out, losing the bid
    (``pass_auction``), or raises. Otherwise bid the amount on the selected
    private; on the topmost, buy it at the bid."""
    cdef int player = turn_active_player(state)
    cdef int p = turn_auction_private(state)
    cdef int amount = 0 if info.type == ACTION_PASS else price_amount(state, info)
    if p >= 0:
        if info.type != ACTION_PASS:
            _player_at(player).set_passed(state, False)
        _player_at(player).set_bid(state, p, amount)  # 0 drops out
        _resolve(state)
        return
    p = turn_pending_private(state)
    TURN.set_pending_private(state, -1)
    if p == topmost_private(state):
        _buy(state, player, p, amount)
        return
    _player_at(player).set_passed(state, False)
    _player_at(player).set_bid(state, p, amount)
    _end_turn(state)


# =============================================================================
# PYTHON TEST ENTRY POINT
# =============================================================================

def setup_auction_py(GameState state, int turn_player):
    """Enter the auction at ``turn_player``'s normal turn and resolve the
    topmost private as after a sale, from bids and owners set up by hand."""
    TURN.set_round(state, ROUND_AUCTION)
    TURN.set_pending_private(state, -1)
    TURN.set_auction_turn(state, turn_player)
    _resolve(state)
