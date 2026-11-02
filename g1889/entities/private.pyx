"""PRIVATES entity; see private.pxd."""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_CORP, OWNER_NOT_IN_PLAY, OWNER_PLAYER, OWNER_UNSOLD
from g1889.data.tables cimport (
    HEX_BLOCKER, MIN_BID_INCREMENT, NUM_CORPS, NUM_PRIVATES, PRIVATE_REVENUE, PRIVATE_TR, PRIVATE_UTF,
    PRIVATE_VALUE, UTF_LATE_PHASE, UTF_LATE_REVENUE,
)
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.player cimport Player, player_bid
from g1889.entities.turn cimport turn_auction_discount, turn_auction_private, turn_num_players
from g1889.state cimport GLOBALS, LAYOUT, PRIVATE_FIELDS, GameState

from g1889.data.tables import PrivateId
from g1889.entities import bank as _bank, player as _player

cdef Bank BANK = _bank.BANK
cdef list PLAYERS = _player.PLAYERS


cdef inline int _slot(int private, int field) noexcept nogil:
    return LAYOUT.privates_offset + private * PRIVATE_FIELDS.size + field


cdef int private_owner(GameState state, int private) noexcept nogil:
    return <int>state._data[_slot(private, PRIVATE_FIELDS.owner)]


cdef int private_revenue(GameState state, int private) noexcept nogil:
    """Revenue paid to the owner; a player-owned UTF pays ¥50 from phase 5."""
    cdef int owner = private_owner(state, private)
    if (private == <int>PRIVATE_UTF and owner >= <int>OWNER_PLAYER and owner < <int>OWNER_CORP
            and state._data[LAYOUT.global_offset + GLOBALS.game_phase] >= <int>UTF_LATE_PHASE):
        return <int>UTF_LATE_REVENUE
    return PRIVATE_REVENUE[private]


cdef bint private_blocks_hex(GameState state, int hex) noexcept nogil:
    """Whether the hex's blocker private keeps it from being laid on: TR
    blocks K4 and ER blocks C4 while a player owns them (``blocks_hexes`` with
    ``owner_type: player``; ``Tracker#ability_blocking_hex``)."""
    cdef int private = HEX_BLOCKER[hex]
    cdef int owner
    if private < 0:
        return False
    owner = private_owner(state, private)
    return <int>OWNER_PLAYER <= owner < <int>OWNER_CORP


# =============================================================================
# AUCTION QUERIES
# =============================================================================

cdef int topmost_private(GameState state) noexcept nogil:
    """The first unsold private (the only one that can be bought outright), or -1."""
    cdef int p
    for p in range(<int>NUM_PRIVATES):
        if private_owner(state, p) == <int>OWNER_UNSOLD:
            return p
    return -1


cdef int private_price(GameState state, int private) noexcept nogil:
    """Face value less the auction discount, which only A takes (``min_bid``)."""
    return PRIVATE_VALUE[private] - (turn_auction_discount(state) if private == <int>PRIVATE_TR else 0)


cdef int private_high_bid(GameState state, int private) noexcept nogil:
    """The highest bid on ``private``, 0 without bids."""
    cdef int i, high = 0
    for i in range(turn_num_players(state)):
        high = max(high, player_bid(state, i, private))
    return high


cdef int private_bid_count(GameState state, int private) noexcept nogil:
    cdef int i, count = 0
    for i in range(turn_num_players(state)):
        count += player_bid(state, i, private) > 0
    return count


cdef int private_lowest_bidder(GameState state, int private) noexcept nogil:
    """The player with the lowest bid on ``private``, or -1. Each bid tops the
    last by ¥5, so bids on one private never tie."""
    cdef int i, bid, low = 0, bidder = -1
    for i in range(turn_num_players(state)):
        bid = player_bid(state, i, private)
        if bid > 0 and (bidder < 0 or bid < low):
            low, bidder = bid, i
    return bidder


cdef int private_min_bid(GameState state, int private) noexcept nogil:
    """The least bid on unsold ``private`` (``WaterfallAuction#min_bid``): the
    topmost's price outside a closed auction, where a bid buys it; otherwise
    the high bid, or the price without bids, plus MIN_BID_INCREMENT."""
    cdef int high
    if private == topmost_private(state) and turn_auction_private(state) < 0:
        return private_price(state, private)
    high = private_high_bid(state, private)
    return (high if high > 0 else private_price(state, private)) + <int>MIN_BID_INCREMENT


# =============================================================================
# HANDLES
# =============================================================================

cdef class Private:
    """Stateless handle over one private's block."""

    def __cinit__(self, int private_id, str sym):
        self.private_id = private_id
        self.sym = sym

    cpdef int owner(self, GameState state):
        return private_owner(state, self.private_id)

    cpdef void set_owner(self, GameState state, int owner):
        """Set the owner code: not in play, closed, unsold, a player, or a corp."""
        cdef int num_players = <int>state._data[LAYOUT.global_offset + GLOBALS.num_players]
        assert (<int>OWNER_NOT_IN_PLAY <= owner <= <int>OWNER_UNSOLD
                or <int>OWNER_PLAYER <= owner < <int>OWNER_PLAYER + num_players
                or <int>OWNER_CORP <= owner < <int>OWNER_CORP + <int>NUM_CORPS), f"owner code {owner}"
        state._data[_slot(self.private_id, PRIVATE_FIELDS.owner)] = <int16_t>owner

    cpdef int flags(self, GameState state):
        return <int>state._data[_slot(self.private_id, PRIVATE_FIELDS.flags)]

    cpdef void set_flags(self, GameState state, int flags):
        state._data[_slot(self.private_id, PRIVATE_FIELDS.flags)] = <int16_t>flags

    cpdef int revenue(self, GameState state):
        return private_revenue(state, self.private_id)

    cpdef bint blocks_hex(self, GameState state, int hex):
        """Whether this private blocks lays on ``hex`` now."""
        return HEX_BLOCKER[hex] == self.private_id and private_blocks_hex(state, hex)

    cpdef int price(self, GameState state):
        return private_price(state, self.private_id)

    cpdef int high_bid(self, GameState state):
        return private_high_bid(state, self.private_id)

    cpdef int bid_count(self, GameState state):
        return private_bid_count(state, self.private_id)

    cpdef int lowest_bidder(self, GameState state):
        return private_lowest_bidder(state, self.private_id)

    cpdef int min_bid(self, GameState state):
        assert self.owner(state) == <int>OWNER_UNSOLD, f"private {self.sym} is not for sale"
        return private_min_bid(state, self.private_id)

    cpdef void sell_to_player(self, GameState state, int player, int price):
        """The bank sells this unsold private to ``player`` for ``price``, ¥0
        for the free A. Its bids lapse with the sale (``accept_bid``)."""
        cdef int i
        assert self.owner(state) == <int>OWNER_UNSOLD, f"private {self.sym} is not for sale"
        assert price >= 0, f"negative price {price}"
        if price:
            BANK.transfer(state, <int>OWNER_PLAYER + player, <int>BANK_ACCOUNT, price)
        self.set_owner(state, <int>OWNER_PLAYER + player)
        for i in range(turn_num_players(state)):
            (<Player>PLAYERS[i]).set_bid(state, self.private_id, 0)


cpdef int topmost(GameState state):
    return topmost_private(state)


cpdef void pay_revenue(GameState state):
    """Every owned private pays its revenue from the bank to its owner
    (``payout_companies``). Owner codes are cash accounts. Ruby pays in player
    order; the bank only pays out, so the order cannot change the result."""
    cdef int p, owner
    for p in range(<int>NUM_PRIVATES):
        owner = private_owner(state, p)
        if owner >= <int>OWNER_PLAYER:
            BANK.transfer(state, <int>BANK_ACCOUNT, owner, private_revenue(state, p))


PRIVATES = [Private(p.value, p.name.removeprefix('PRIVATE_')) for p in PrivateId]
PRIVATES_BY_SYM = {private.sym: private for private in PRIVATES}
