"""CORPS entity; see corp.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport CORP_TOKENS, NUM_CORPS, NUM_PAR_PRICES
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, GameState

from g1889.data.tables import CorpId


cdef inline int _slot(int corp, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + field


cdef inline int _get(GameState state, int corp, int field) noexcept nogil:
    return <int>state._data[_slot(corp, field)]


cdef inline void _set(GameState state, int corp, int field, int value) noexcept nogil:
    state._data[_slot(corp, field)] = <int16_t>value


cdef int corp_par_index(GameState state, int corp) noexcept nogil:
    return _get(state, corp, CORP_FIELDS.par_index)


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


cdef void invalidate_route(GameState state, int corp) noexcept nogil:
    cdef int slot = LAYOUT.global_offset + GLOBALS.route_dirty
    state._data[slot] = <int16_t>(state._data[slot] | (1 << corp))


cdef void invalidate_all_routes(GameState state) noexcept nogil:
    state._data[LAYOUT.global_offset + GLOBALS.route_dirty] = <int16_t>((1 << <int>NUM_CORPS) - 1)


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
        assert 0 <= shares <= <int>SHARES_PER_CORP, f"pool shares {shares} out of range"
        _set(state, self.corp_id, CORP_FIELDS.pool_shares, shares)

    cpdef int president(self, GameState state):
        return corp_president(state, self.corp_id)

    cpdef void set_president(self, GameState state, int player):
        cdef int num_players = <int>state._data[LAYOUT.global_offset + GLOBALS.num_players]
        assert -1 <= player < num_players, f"president {player} out of range"
        _set(state, self.corp_id, CORP_FIELDS.president, player)

    cpdef int tokens_left(self, GameState state):
        return _get(state, self.corp_id, CORP_FIELDS.tokens_left)

    cpdef void set_tokens_left(self, GameState state, int tokens):
        assert 0 <= tokens <= CORP_TOKENS[self.corp_id], f"tokens {tokens} exceed the charter"
        _set(state, self.corp_id, CORP_FIELDS.tokens_left, tokens)

    cpdef int last_revenue(self, GameState state):
        return _get(state, self.corp_id, CORP_FIELDS.last_revenue)

    cpdef void set_last_revenue(self, GameState state, int revenue):
        _set(state, self.corp_id, CORP_FIELDS.last_revenue, revenue)

    cpdef bint route_dirty(self, GameState state):
        return (state._data[LAYOUT.global_offset + GLOBALS.route_dirty] >> self.corp_id) & 1


CORPS = [Corporation(c.value, c.name.removeprefix('CORP_')) for c in CorpId]
CORPS_BY_SYM = {corp.sym: corp for corp in CORPS}

# Python mirrors: a bare assignment would be shadowed by the cdef names.
globals()['SHARES_PER_CORP'] = <int>SHARES_PER_CORP
