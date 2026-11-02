"""CORPS: the corp blocks' share, president, token, and revenue fields.

MARKET owns ``market_row``/``market_col``/``market_stamp`` and TRAINS the
train counts. ``route_cache`` and ``route_available`` are caches behind the
global ``route_dirty`` and ``conn_dirty`` masks: GRAPH fills and reads
``route_available`` (``GRAPH.route_available``), and the route solver owns
``route_cache``.

Shares are 10% units, and the President's Certificate counts 2. Ruby moves
certificates in a presidency change, but each holder's units stay the same
(the new president's two shares go to the old one, or to the pool when the
old one sold the certificate), so a change only rewrites ``president``.
"""

from g1889.state cimport GameState

cpdef enum CorpShares:
    SHARES_PER_CORP = 10           # 10% units; the President's Certificate counts 2
    PRESIDENT_SHARES = 2

cdef int corp_par_index(GameState state, int corp) noexcept nogil
cdef int corp_par_price(GameState state, int corp) noexcept nogil
cdef bint corp_floated(GameState state, int corp) noexcept nogil
cdef int corp_cash(GameState state, int corp) noexcept nogil
cdef int corp_ipo_shares(GameState state, int corp) noexcept nogil
cdef int corp_pool_shares(GameState state, int corp) noexcept nogil
cdef int corp_president(GameState state, int corp) noexcept nogil
cdef int corp_tokens_left(GameState state, int corp) noexcept nogil
cdef int corp_token_price(GameState state, int corp) noexcept nogil

# Share rules (rules §6.3, §6.5)
cdef int corp_successor(GameState state, int corp) noexcept nogil
cdef int corp_max_sale(GameState state, int player, int corp) noexcept nogil
cdef int corp_emergency_sale(GameState state, int player, int corp) noexcept nogil

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
    cpdef int token_price(self, GameState state)
    cpdef int last_revenue(self, GameState state)
    cpdef void set_last_revenue(self, GameState state, int revenue)
    cpdef bint route_dirty(self, GameState state)

    # Stock round
    cdef void _move_to_player(self, GameState state, int player, int shares, int price) except *
    cpdef int par_price(self, GameState state)
    cpdef void par(self, GameState state, int player, int par_index)
    cpdef void buy_share(self, GameState state, int player, bint from_pool)
    cpdef void exchange_share(self, GameState state, int player)
    cpdef void sell_shares(self, GameState state, int player, int count)
    cpdef bint float_if_ready(self, GameState state)
    cpdef int max_sale(self, GameState state, int player)
    cpdef int emergency_sale(self, GameState state, int player)
    cpdef bint sold_out(self, GameState state)
