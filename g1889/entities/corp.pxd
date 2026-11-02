"""CORPS: the corp blocks' share, president, token, and revenue fields.

MARKET owns ``market_row``/``market_col``/``market_stamp`` and TRAINS the
train counts. ``route_cache`` and ``route_available`` are caches behind the
global ``route_dirty`` and ``conn_dirty`` masks; their accessors arrive with
GRAPH and the route solver.
"""

from g1889.state cimport GameState

cpdef enum CorpShares:
    SHARES_PER_CORP = 10           # 10% units; the President's Certificate counts 2

cdef int corp_par_index(GameState state, int corp) noexcept nogil
cdef bint corp_floated(GameState state, int corp) noexcept nogil
cdef int corp_cash(GameState state, int corp) noexcept nogil
cdef int corp_ipo_shares(GameState state, int corp) noexcept nogil
cdef int corp_pool_shares(GameState state, int corp) noexcept nogil
cdef int corp_president(GameState state, int corp) noexcept nogil

cdef void invalidate_route(GameState state, int corp) noexcept nogil
cdef void invalidate_all_routes(GameState state) noexcept nogil


cdef class Corporation:
    cdef readonly int corp_id
    cdef readonly str sym

    cpdef int par_index(self, GameState state)
    cpdef void set_par_index(self, GameState state, int par_index)
    cpdef bint floated(self, GameState state)
    cpdef void set_floated(self, GameState state, bint floated)
    cpdef int cash(self, GameState state)
    cpdef int ipo_shares(self, GameState state)
    cpdef void set_ipo_shares(self, GameState state, int shares)
    cpdef int pool_shares(self, GameState state)
    cpdef void set_pool_shares(self, GameState state, int shares)
    cpdef int president(self, GameState state)
    cpdef void set_president(self, GameState state, int player)
    cpdef int tokens_left(self, GameState state)
    cpdef void set_tokens_left(self, GameState state, int tokens)
    cpdef int last_revenue(self, GameState state)
    cpdef void set_last_revenue(self, GameState state, int revenue)
    cpdef bint route_dirty(self, GameState state)
