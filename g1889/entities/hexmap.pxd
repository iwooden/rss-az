"""MAP: each hex's tile, canonical rotation, and token mask, and the tile supply.

Any change to the map makes every corp's connectivity and routes stale.
"""

from g1889.state cimport GameState

cdef int hex_tile(GameState state, int hex) noexcept nogil
cdef int hex_rotation(GameState state, int hex) noexcept nogil
cdef int hex_tokens(GameState state, int hex) noexcept nogil
cdef int supply_count(GameState state, int tile) noexcept nogil
cdef void invalidate_connectivity(GameState state) noexcept nogil


cdef class HexMap:
    cpdef int tile(self, GameState state, int hex)
    cpdef int rotation(self, GameState state, int hex)
    cpdef void set_tile(self, GameState state, int hex, int tile, int rotation)
    cpdef int tokens(self, GameState state, int hex)
    cpdef void set_tokens(self, GameState state, int hex, int mask)
    cpdef int supply(self, GameState state, int tile)
    cpdef void set_supply(self, GameState state, int tile, int count)
    cpdef int conn_dirty(self, GameState state)
