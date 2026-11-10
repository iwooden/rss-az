"""MAP: each hex's tile, canonical rotation, and token mask, and the tile supply.

A lay or a token makes stale the connectivity and routes of the corps whose
networks touch the hex (``map_touch_mask``), and a token its own corp's; the
raw setters, which may put anything anywhere, every corp's.

Home reservations are implicit (state.pxd): a hex whose ``HEX_HOME_CORP`` has
not placed its token keeps one slot for it, as Ruby's ``City#reservations``
do (``part/city.rb``: ``get_slot``, ``tokenable?``). A reservation keeps the
slot from other corps' tokens but never blocks a route (``City#blocks?``).
"""

from libc.stdint cimport uint32_t

from g1889.data.tables cimport RC_CITY, TILE_RC, TILE_SLOTS
from g1889.state cimport GameState

cdef int hex_tile(GameState state, int hex) noexcept nogil
cdef int hex_rotation(GameState state, int hex) noexcept nogil
cdef int hex_tokens(GameState state, int hex) noexcept nogil
cdef uint32_t hex_paths(GameState state, int hex) noexcept nogil
cdef int supply_count(GameState state, int tile) noexcept nogil
cdef void invalidate_connectivity(GameState state) noexcept nogil

# The corps whose connectivity or routes may depend on the hex, as a corp mask.
cdef int map_touch_mask(GameState state, int hex) noexcept nogil
# For each corp, an upper bound on the revenue centers GRAPH reaches, from the same search.
cdef void map_center_bounds(GameState state, int* counts) noexcept nogil

# MAP.lay's and MAP.place_token's writes, without their checks, for nogil
# callers that apply legal changes (extraction's hypotheticals). ``touch`` is
# the hex's ``map_touch_mask`` before the change: the corps marked stale. A
# caller that tries several changes on copies of one state computes it once,
# and one that throws the changed state away may pass only the corps whose
# caches it will read there.
cdef void map_put_tile(GameState state, int hex, int tile, int rotation, int touch) noexcept nogil
cdef void map_put_token(GameState state, int hex, int corp, int touch) noexcept nogil

# Cities, revenue, and terrain
cdef int map_token_count(GameState state, int hex) noexcept nogil
cdef int map_free_slots(GameState state, int hex) noexcept nogil
cdef bint map_has_slot_for(GameState state, int hex, int corp) noexcept nogil
cdef bint map_blocked_for(GameState state, int hex, int corp) noexcept nogil
cdef int map_revenue(GameState state, int hex, int phase, bint is_d) noexcept nogil
cdef int map_lay_cost(GameState state, int hex, int corp) noexcept nogil


cdef inline int _count(int mask) noexcept nogil:
    """The tokens in a token mask."""
    cdef int n = 0
    while mask:
        mask &= mask - 1
        n += 1
    return n


cdef inline bint tile_blocks(int tile, int tokens, int corp) noexcept nogil:
    """``City#blocks?`` for ``tile`` holding the token mask ``tokens``: a city
    whose slots are all filled, none by ``corp``. Reservations never block."""
    return TILE_RC[tile] == <int>RC_CITY and not (tokens >> corp & 1) and _count(tokens) >= TILE_SLOTS[tile]


cdef class HexMap:
    cpdef int tile(self, GameState state, int hex)
    cpdef int rotation(self, GameState state, int hex)
    cpdef void set_tile(self, GameState state, int hex, int tile, int rotation)
    cpdef int tokens(self, GameState state, int hex)
    cpdef void set_tokens(self, GameState state, int hex, int mask)
    cpdef int supply(self, GameState state, int tile)
    cpdef void set_supply(self, GameState state, int tile, int count)
    cpdef int conn_dirty(self, GameState state)
    cpdef int touch_mask(self, GameState state, int hex)

    # Queries
    cpdef int free_slots(self, GameState state, int hex)
    cpdef bint has_slot_for(self, GameState state, int hex, int corp)
    cpdef bint blocked_for(self, GameState state, int hex, int corp)
    cpdef int revenue(self, GameState state, int hex, int phase, bint is_d=*)
    cpdef int lay_cost(self, GameState state, int hex, int corp)

    # Mutations
    cpdef void lay(self, GameState state, int hex, int tile, int rotation)
    cpdef void lay_track(self, GameState state, int corp, int hex, int tile, int rotation)
    cpdef void place_token(self, GameState state, int hex, int corp)
    cpdef void buy_token(self, GameState state, int hex, int corp)
