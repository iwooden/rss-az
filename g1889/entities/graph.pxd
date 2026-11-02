"""GRAPH: each corp's connectivity, cached in the graph block behind ``conn_dirty``.

Entry (c, h), at ``graph_offset + c * NUM_HEXES + h``, holds bits 0-5
(``GRAPH_SIDES``), the sides of h the corp reaches (Ruby's
``connected_hexes[h]``), which a lay's exits must meet, and bit 6
(``GRAPH_NODE``), whether it reaches h's revenue center (``connected_nodes``),
which a token needs. The refresh also writes the corp's ``route_available``:
some token's walk reaches two or more revenue centers (Ruby's
``route_train_purchase``). ``MAP`` writes mark every corp dirty; the getters
refresh lazily, and ``graph_refresh`` can run in a nogil prologue.
"""

from g1889.state cimport GameState

cpdef enum GraphBits:
    GRAPH_SIDES = 0x3F
    GRAPH_NODE = 0x40

cdef void graph_refresh(GameState state, int corp) noexcept nogil
cdef int graph_sides(GameState state, int corp, int hex) noexcept nogil
cdef bint graph_node(GameState state, int corp, int hex) noexcept nogil
cdef bint graph_route_available(GameState state, int corp) noexcept nogil
cdef bint graph_walks_new_track(GameState state, int corp, int hex, int tile, int rotation) noexcept nogil


cdef class Graph:
    cpdef bint dirty(self, GameState state, int corp)
    cpdef void refresh(self, GameState state, int corp)
    cpdef int sides_reached(self, GameState state, int corp, int hex)
    cpdef bint node_reached(self, GameState state, int corp, int hex)
    cpdef bint route_available(self, GameState state, int corp)
    cpdef bint walks_new_track(self, GameState state, int corp, int hex, int tile, int rotation)
