"""CORPS entity; see corp.pxd."""

from libc.stdint cimport int16_t

from g1889.constants cimport OWNER_CORP, OWNER_PLAYER
from g1889.data.tables cimport (
    CORP_TOKEN_PRICE, CORP_TOKENS, FLOAT_PERCENT, MARKET_SHARE_LIMIT, NUM_CORPS, NUM_PAR_PRICES, PAR_COL,
    PAR_PRICE, PAR_ROW,
)
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.market cimport Market, market_price
from g1889.entities.player cimport Player, player_shares
from g1889.entities.turn cimport turn_num_players
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, GameState

from g1889.data.tables import CorpId
from g1889.entities import bank as _bank, market as _market, player as _player

cdef Bank BANK = _bank.BANK
cdef Market MARKET = _market.MARKET
cdef list PLAYERS = _player.PLAYERS

# Shares of a corp that may sit in the pool, and that leave the IPO to float it.
cdef int POOL_LIMIT = <int>MARKET_SHARE_LIMIT * <int>SHARES_PER_CORP // 100
cdef int FLOAT_SHARES = <int>FLOAT_PERCENT * <int>SHARES_PER_CORP // 100


cdef inline int _slot(int corp, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + field


cdef inline int _get(GameState state, int corp, int field) noexcept nogil:
    return <int>state._data[_slot(corp, field)]


cdef inline void _set(GameState state, int corp, int field, int value) noexcept nogil:
    state._data[_slot(corp, field)] = <int16_t>value


cdef int corp_par_index(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.par_index)


cdef int corp_par_price(GameState state, int corp) noexcept nogil:
    """The par price, or 0 while unparred."""
    cdef int par_index = corp_par_index(state, corp)
    return PAR_PRICE[par_index] if par_index >= 0 else 0


cdef bint corp_floated(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.floated) != 0


cdef int corp_cash(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.cash)


cdef int corp_ipo_shares(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.ipo_shares)


cdef int corp_pool_shares(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.pool_shares)


cdef int corp_president(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.president)


cdef int corp_tokens_left(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.tokens_left)


cdef int corp_token_price(GameState state, int corp) noexcept nogil:
    """The price of the corp's next charter token (``next_token``), which the
    Token step charges; the first, the home token, is free. 0 with none left."""
    cdef int left = corp_tokens_left(state, corp)
    return CORP_TOKEN_PRICE[corp][CORP_TOKENS[corp] - left] if left > 0 else 0


cdef void invalidate_route(GameState state, int corp) noexcept nogil:
    cdef int slot = LAYOUT.global_offset + GLOBALS.route_dirty
    state._data[slot] = <int16_t>(state._data[slot] | (1 << corp))


cdef void invalidate_all_routes(GameState state) noexcept nogil:
    state._data[LAYOUT.global_offset + GLOBALS.route_dirty] = <int16_t>((1 << <int>NUM_CORPS) - 1)


# =============================================================================
# SHARE RULES
# =============================================================================

cdef int corp_successor(GameState state, int corp) noexcept nogil:
    """The president after a transfer (``SharePool#transfer_shares``): the
    president keeps the corp while nobody holds more, since ties never change
    it. Otherwise the largest holder with at least the President's
    Certificate's two shares takes it, the nearest clockwise from the
    president first; with none, the president stays."""
    cdef int president = corp_president(state, corp)
    cdef int n = turn_num_players(state)
    cdef int most = 0
    cdef int i, d
    for i in range(n):
        most = max(most, player_shares(state, i, corp))
    if player_shares(state, president, corp) == most or most < <int>PRESIDENT_SHARES:
        return president
    for d in range(1, n):
        i = (president + d) % n
        if player_shares(state, i, corp) == most:
            return i
    return president


cdef int corp_max_sale(GameState state, int player, int corp) noexcept nogil:
    """The largest block ``player`` may sell to the pool; every smaller block is
    legal too. That is their holding, up to the pool's room (rules §2.3). A
    president may sell into the President's Certificate only when another
    player holds two shares to take it (``ShareBundle#can_dump?``), so
    otherwise keeps two shares. An unparred corp's shares cannot be sold
    (``all_bundles_for_corporation``)."""
    cdef int held = player_shares(state, player, corp)
    cdef int most_other = 0
    cdef int i
    if corp_par_index(state, corp) < 0:
        return 0
    if corp_president(state, corp) == player:
        for i in range(turn_num_players(state)):
            if i != player:
                most_other = max(most_other, player_shares(state, i, corp))
        if most_other < <int>PRESIDENT_SHARES:
            held -= <int>PRESIDENT_SHARES
    return max(0, min(held, POOL_LIMIT - corp_pool_shares(state, corp)))


# =============================================================================
# HANDLES
# =============================================================================

cdef class Corporation:
    """Stateless handle over one corp's block."""

    def __cinit__(self, int corp_id, str sym):
        self.corp_id = corp_id
        self.sym = sym

    cpdef int par_index(self, GameState state):
        return corp_par_index(state, self.corp_id)

    cpdef void set_par_index(self, GameState state, int par_index):
        assert -1 <= par_index < <int>NUM_PAR_PRICES, f"par index {par_index} out of range"
        _set(state, self.corp_id, CORP_FIELDS.par_index, par_index)

    cpdef bint floated(self, GameState state):
        return corp_floated(state, self.corp_id)

    cpdef void set_floated(self, GameState state, bint floated):
        _set(state, self.corp_id, CORP_FIELDS.floated, floated)

    cpdef int cash(self, GameState state):
        return corp_cash(state, self.corp_id)

    cpdef int ipo_shares(self, GameState state):
        return corp_ipo_shares(state, self.corp_id)

    cpdef void set_ipo_shares(self, GameState state, int shares):
        assert 0 <= shares <= <int>SHARES_PER_CORP, f"IPO shares {shares} out of range"
        _set(state, self.corp_id, CORP_FIELDS.ipo_shares, shares)

    cpdef int pool_shares(self, GameState state):
        return corp_pool_shares(state, self.corp_id)

    cpdef void set_pool_shares(self, GameState state, int shares):
        assert 0 <= shares <= POOL_LIMIT, f"pool shares {shares} out of range"
        _set(state, self.corp_id, CORP_FIELDS.pool_shares, shares)

    cpdef int president(self, GameState state):
        return corp_president(state, self.corp_id)

    cpdef void set_president(self, GameState state, int player):
        cdef int num_players = <int>state._data[LAYOUT.global_offset + GLOBALS.num_players]
        assert -1 <= player < num_players, f"president {player} out of range"
        _set(state, self.corp_id, CORP_FIELDS.president, player)

    cpdef int tokens_left(self, GameState state):
        return corp_tokens_left(state, self.corp_id)

    cpdef void set_tokens_left(self, GameState state, int tokens):
        assert 0 <= tokens <= CORP_TOKENS[self.corp_id], f"tokens {tokens} exceed the charter"
        _set(state, self.corp_id, CORP_FIELDS.tokens_left, tokens)

    cpdef int token_price(self, GameState state):
        return corp_token_price(state, self.corp_id)

    cpdef int last_revenue(self, GameState state):
        return _get(state, self.corp_id, CORP_FIELDS.last_revenue)

    cpdef void set_last_revenue(self, GameState state, int revenue):
        _set(state, self.corp_id, CORP_FIELDS.last_revenue, revenue)

    cpdef bint route_dirty(self, GameState state):
        return (state._data[LAYOUT.global_offset + GLOBALS.route_dirty] >> self.corp_id) & 1

    # Stock round. Every payment goes to or from the bank (rules §6.2).

    cpdef int par_price(self, GameState state):
        return corp_par_price(state, self.corp_id)

    cdef void _move_to_player(self, GameState state, int player, int shares, int price) except *:
        """``player`` pays ``price`` for ``shares`` from the IPO, or receives it
        for selling ``shares`` (negative) to the pool."""
        cdef Player holder = <Player>PLAYERS[player]
        if shares > 0:
            BANK.transfer(state, <int>OWNER_PLAYER + player, <int>BANK_ACCOUNT, price)
        else:
            BANK.transfer(state, <int>BANK_ACCOUNT, <int>OWNER_PLAYER + player, price)
        holder.set_shares(state, self.corp_id, player_shares(state, player, self.corp_id) + shares)

    cpdef void par(self, GameState state, int player, int par_index):
        """``player`` starts the corp at par ``par_index``: its token goes to the
        bottom of the par cell's stack, and the player buys the President's
        Certificate from the IPO for twice the par (``process_par``)."""
        cdef int corp = self.corp_id
        assert corp_par_index(state, corp) < 0 and 0 <= par_index < <int>NUM_PAR_PRICES, \
            f"cannot par {self.sym} at index {par_index}"
        self.set_par_index(state, par_index)
        MARKET.set_position(state, corp, PAR_ROW[par_index], PAR_COL[par_index])
        self.set_president(state, player)
        self.set_ipo_shares(state, corp_ipo_shares(state, corp) - <int>PRESIDENT_SHARES)
        self._move_to_player(state, player, <int>PRESIDENT_SHARES, <int>PRESIDENT_SHARES * PAR_PRICE[par_index])

    cpdef void buy_share(self, GameState state, int player, bint from_pool):
        """``player`` buys a share from the IPO at par or from the pool at the
        market price. A buyer who now holds more than the president takes the
        presidency, and the corp floats once half its shares have left the IPO
        (``SharePool#buy_shares``)."""
        cdef int corp = self.corp_id
        if from_pool:
            assert corp_pool_shares(state, corp) > 0, f"no {self.sym} share in the pool"
            self.set_pool_shares(state, corp_pool_shares(state, corp) - 1)
            self._move_to_player(state, player, 1, market_price(state, corp))
        else:
            assert corp_par_index(state, corp) >= 0 and corp_ipo_shares(state, corp) > 0, \
                f"no {self.sym} share in the IPO"
            self.set_ipo_shares(state, corp_ipo_shares(state, corp) - 1)
            self._move_to_player(state, player, 1, corp_par_price(state, corp))
        self.set_president(state, corp_successor(state, corp))
        self.float_if_ready(state)

    cpdef void exchange_share(self, GameState state, int player):
        """``player`` takes a share from the IPO free, for a private (DR's
        exchange: ``SharePool#buy_shares`` with ``exchange``). The corp need
        not be parred, as Ruby's ``available_share`` takes any share but the
        President's Certificate: an unparred corp's share has no price, and
        nobody becomes president, as no holder has the certificate's two
        shares. Once parred, a holder of more than the president takes the
        presidency, and the corp floats once half its shares have left the IPO,
        mid-OR too; the OR's order is fixed, so it operates from the next OR."""
        cdef int corp = self.corp_id
        cdef bint parred = corp_par_index(state, corp) >= 0
        cdef Player holder = <Player>PLAYERS[player]
        assert corp_ipo_shares(state, corp) > (0 if parred else <int>PRESIDENT_SHARES), \
            f"no {self.sym} share in the IPO to exchange"
        self.set_ipo_shares(state, corp_ipo_shares(state, corp) - 1)
        holder.set_shares(state, corp, player_shares(state, player, corp) + 1)
        if parred:
            self.set_president(state, corp_successor(state, corp))
            self.float_if_ready(state)

    cpdef void sell_shares(self, GameState state, int player, int count):
        """``player`` sells ``count`` shares to the pool at the market price, paid
        by the bank. The presidency passes if the sale leaves another player
        holding more, then the price drops one row per share
        (``sell_shares_and_change_price``, ``SELL_MOVEMENT = :down_share``)."""
        cdef int corp = self.corp_id
        cdef int i
        assert 0 < count <= corp_max_sale(state, player, corp), f"cannot sell {count} {self.sym} shares"
        self.set_pool_shares(state, corp_pool_shares(state, corp) + count)
        self._move_to_player(state, player, -count, count * market_price(state, corp))
        self.set_president(state, corp_successor(state, corp))
        for i in range(count):
            MARKET.move_down(state, corp)

    cpdef bint float_if_ready(self, GameState state):
        """Float once at most half the shares are left in the IPO: the bank pays
        the corp ten times its par (full capitalization). Returns whether it
        floated now."""
        cdef int corp = self.corp_id
        if (corp_floated(state, corp) or corp_par_index(state, corp) < 0
                or corp_ipo_shares(state, corp) > <int>SHARES_PER_CORP - FLOAT_SHARES):
            return False
        self.set_floated(state, True)
        BANK.transfer(state, <int>BANK_ACCOUNT, <int>OWNER_CORP + corp,
                      <int>SHARES_PER_CORP * corp_par_price(state, corp))
        return True

    cpdef int max_sale(self, GameState state, int player):
        return corp_max_sale(state, player, self.corp_id)

    cpdef bint sold_out(self, GameState state):
        """Whether players hold every share (``sold_out?``)."""
        return corp_ipo_shares(state, self.corp_id) == 0 and corp_pool_shares(state, self.corp_id) == 0


CORPS = [Corporation(c.value, c.name.removeprefix('CORP_')) for c in CorpId]
CORPS_BY_SYM = {corp.sym: corp for corp in CORPS}

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in CorpShares.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
