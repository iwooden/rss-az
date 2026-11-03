"""PLAYERS entity; see player.pxd."""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_PLAYER, SHARES_PER_CORP
from g1889.data.tables cimport CERT_LIMIT, MAX_OWNERSHIP_PERCENT, MAX_PLAYERS, NUM_CORPS, NUM_PRIVATES, PRIVATE_VALUE
from g1889.entities.market cimport market_counts_for_limit, market_price, market_unlimited
from g1889.entities.turn cimport turn_num_players
from g1889.state cimport CORP_FIELDS, LAYOUT, PLAYER_FIELDS, PRIVATE_FIELDS, GameState

# Holdings are counted in 10% shares.
cdef enum:
    SHARE_PERCENT = 10


cdef inline int _slot(int player, int field) noexcept nogil:
    return LAYOUT.players_offset + player * PLAYER_FIELDS.size + field


cdef int player_cash(GameState state, int player) noexcept nogil:
    return <int>state._data[_slot(player, PLAYER_FIELDS.cash)]


cdef int player_shares(GameState state, int player, int corp) noexcept nogil:
    return <int>state._data[_slot(player, PLAYER_FIELDS.shares) + corp]


cdef int player_bid(GameState state, int player, int private) noexcept nogil:
    return <int>state._data[_slot(player, PLAYER_FIELDS.bids) + private]


cdef int player_committed(GameState state, int player) noexcept nogil:
    """Cash held by the player's open bids (``committed_cash``)."""
    cdef int total = 0
    cdef int p
    for p in range(<int>NUM_PRIVATES):
        total += player_bid(state, player, p)
    return total


cdef int player_available(GameState state, int player, int private) noexcept nogil:
    """Cash the player may bid or spend on ``private``: cash less their bids on
    other privates (``max_bid``). ``private`` -1 subtracts every bid."""
    cdef int own = player_bid(state, player, private) if private >= 0 else 0
    return player_cash(state, player) - player_committed(state, player) + own


cdef bint player_passed(GameState state, int player) noexcept nogil:
    return state._data[_slot(player, PLAYER_FIELDS.passed)] != 0


cdef int player_sold_mask(GameState state, int player) noexcept nogil:
    """Corps the player has sold this SR (Ruby's ``players_sold``)."""
    return <int>state._data[_slot(player, PLAYER_FIELDS.sold_mask)]


# The corp president and private owner fields are read raw here: CORPS and
# PRIVATES cimport this module, so it cannot cimport theirs.

cdef int player_cert_count(GameState state, int player) noexcept nogil:
    """Certificates toward the limit (``num_certs``): each private the player
    owns, and their certificates of every corp outside the yellow and orange
    zones, where the President's Certificate counts once."""
    cdef int count = 0
    cdef int c, p, shares
    for p in range(<int>NUM_PRIVATES):
        count += state._data[LAYOUT.privates_offset + p * PRIVATE_FIELDS.size + PRIVATE_FIELDS.owner] == \
            <int>OWNER_PLAYER + player
    for c in range(<int>NUM_CORPS):
        shares = player_shares(state, player, c)
        if shares and market_counts_for_limit(state, c):
            count += shares - (state._data[LAYOUT.corps_offset + c * CORP_FIELDS.size + CORP_FIELDS.president]
                               == player)
    return count


cdef bint player_can_hold(GameState state, int player, int corp, int shares) noexcept nogil:
    """Whether the player may hold ``shares`` more of ``corp`` (``holding_ok?``):
    at most 60% outside the orange zone."""
    return (market_unlimited(state, corp)
            or SHARE_PERCENT * (player_shares(state, player, corp) + shares) <= <int>MAX_OWNERSHIP_PERCENT)


cdef bint player_can_gain(GameState state, int player, int corp, int shares) noexcept nogil:
    """Whether the player may take ``shares`` more of ``corp`` (``can_gain?``):
    they may hold them, and are below the certificate limit unless the corp's
    shares do not count toward it. An exchange skips the limit
    (``can_exchange_dr``)."""
    return player_can_hold(state, player, corp, shares) and (
        not market_counts_for_limit(state, corp)
        or player_cert_count(state, player) < CERT_LIMIT[turn_num_players(state)])


cdef int player_value(GameState state, int player) noexcept nogil:
    """``Player#value`` (``player.rb:28``), what Ruby's ``result`` scores:
    cash, the player's shares at the market price (the President's
    Certificate counts twice; an unparred corp's are worth nothing), and the
    face value of each open private they own. A bankrupt player keeps the
    shares Ruby could not sell, and scores the same way."""
    cdef int value = player_cash(state, player)
    cdef int c, p
    for c in range(<int>NUM_CORPS):
        value += player_shares(state, player, c) * market_price(state, c)
    for p in range(<int>NUM_PRIVATES):
        if state._data[LAYOUT.privates_offset + p * PRIVATE_FIELDS.size + PRIVATE_FIELDS.owner] == \
                <int>OWNER_PLAYER + player:
            value += PRIVATE_VALUE[p]
    return value


cdef int _write_slot(GameState state, int player, int field) except -1:
    cdef int num_players = turn_num_players(state)
    assert player < num_players, f"player {player} not in a {num_players}-player game"
    return _slot(player, field)


cdef class Player:
    """Stateless handle over one player's block."""

    def __cinit__(self, int player_id):
        self.player_id = player_id

    cpdef int cash(self, GameState state):
        return player_cash(state, self.player_id)

    cpdef int shares(self, GameState state, int corp):
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        return player_shares(state, self.player_id, corp)

    cpdef void set_shares(self, GameState state, int corp, int shares):
        assert 0 <= corp < <int>NUM_CORPS and 0 <= shares <= <int>SHARES_PER_CORP, f"shares {shares} of corp {corp}"
        state._data[_write_slot(state, self.player_id, PLAYER_FIELDS.shares) + corp] = <int16_t>shares

    cpdef int sold_mask(self, GameState state):
        return <int>state._data[_slot(self.player_id, PLAYER_FIELDS.sold_mask)]

    cpdef void set_sold_mask(self, GameState state, int mask):
        assert 0 <= mask < 1 << <int>NUM_CORPS, f"sold mask {mask:#x}"
        state._data[_write_slot(state, self.player_id, PLAYER_FIELDS.sold_mask)] = <int16_t>mask

    cpdef int cert_count(self, GameState state):
        return player_cert_count(state, self.player_id)

    cpdef int value(self, GameState state):
        return player_value(state, self.player_id)

    cpdef bint can_gain(self, GameState state, int corp, int shares=1):
        assert 0 <= corp < <int>NUM_CORPS and shares > 0, f"{shares} shares of corp {corp}"
        return player_can_gain(state, self.player_id, corp, shares)

    cpdef int bid(self, GameState state, int private):
        assert 0 <= private < <int>NUM_PRIVATES, f"private {private} out of range"
        return player_bid(state, self.player_id, private)

    cpdef void set_bid(self, GameState state, int private, int amount):
        assert 0 <= private < <int>NUM_PRIVATES and amount >= 0, f"bid {amount} on private {private}"
        state._data[_write_slot(state, self.player_id, PLAYER_FIELDS.bids) + private] = <int16_t>amount

    cpdef int committed(self, GameState state):
        return player_committed(state, self.player_id)

    cpdef int available(self, GameState state, int private=-1):
        assert -1 <= private < <int>NUM_PRIVATES, f"private {private} out of range"
        return player_available(state, self.player_id, private)

    cpdef void clear_bids(self, GameState state):
        cdef int base = _write_slot(state, self.player_id, PLAYER_FIELDS.bids)
        cdef int p
        for p in range(<int>NUM_PRIVATES):
            state._data[base + p] = 0

    cpdef bint passed(self, GameState state):
        return player_passed(state, self.player_id)

    cpdef void set_passed(self, GameState state, bint passed):
        state._data[_write_slot(state, self.player_id, PLAYER_FIELDS.passed)] = passed

    cpdef bint bankrupt(self, GameState state):
        return state._data[_slot(self.player_id, PLAYER_FIELDS.bankrupt)] != 0

    cpdef void set_bankrupt(self, GameState state, bint bankrupt):
        state._data[_write_slot(state, self.player_id, PLAYER_FIELDS.bankrupt)] = bankrupt


cpdef bint all_passed(GameState state):
    """Whether every player's passed flag is set (``entities.all?(&:passed?)``)."""
    cdef int i
    for i in range(turn_num_players(state)):
        if not player_passed(state, i):
            return False
    return True


cpdef void clear_passed(GameState state):
    cdef int i
    for i in range(turn_num_players(state)):
        state._data[_slot(i, PLAYER_FIELDS.passed)] = 0


PLAYERS = [Player(i) for i in range(<int>MAX_PLAYERS)]
