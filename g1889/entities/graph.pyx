"""GRAPH entity; see graph.pxd.

Connectivity is Ruby's ``Graph#compute`` (``graph.rb``), whose walks
(``part/node.rb``, ``part/path.rb``) the engine replicates step for step,
because the order they visit paths in decides what they reach. For each of
the corp's token cities, in Ruby's hex order (``HEX_WALK_ORDER``), a walk
starts with the other token cities marked visited:

- **Node walk.** Skip a visited node; otherwise mark it and walk each of its
  paths in the tile's order (``TILE_PATHS``).
- **Path walk.** Skip a visited path, or one with a side in use on the walk's
  stack (the edge counter). Mark it and record it (below). If it touches a
  node other than the walk's origin that does not block the corp
  (``Node#blocks?``: an offboard, or a city full of other corps' tokens),
  walk that node first. Then cross each side but the one it entered by, if
  the side is open, and walk the paths of the hex across at that side, in
  order. A walk never reverses at an edge: it takes only the paths across it.
- **Marks.** A walk's marks are permanent, so a path is walked once, from
  whichever end the walk meets first, and a loop is followed one way round.
  Past an exit that two paths of the tile share (``Tile#converging_exit?``,
  the merge of a junction tile's branches), the walk is "converging", and it
  clears each path's and node's mark once it has explored past it.

Recording a path the first time any token's walk meets it marks its sides on
its hex and the hex across each, and its node; a token hex reaches all its
open sides and its node. ``route_available`` holds if one token's walk meets
paths, not met before, that touch two or more revenue centers.
"""

from libc.stdint cimport int16_t, uint8_t, uint16_t, uint32_t, uint64_t

from g1889.data.tables cimport (
    ENDPOINT_PAIRS, HEX_EXIT_MASK, HEX_NEIGHBOR, HEX_WALK_ORDER, NODE, NUM_CORPS, NUM_HEXES, NUM_SIDES, PAIR_A,
    PAIR_B, RC_OFFBOARD, TILE_MASK, TILE_PATH_COUNT, TILE_PATHS, TILE_RC,
)
from g1889.entities.hexmap cimport hex_paths, hex_rotation, hex_tile, hex_tokens, tile_blocks
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, GameState


cdef struct _Walk:
    int corp
    # A hypothetical lay (``graph_walks_new_track``): hex ``lay_hex`` holds
    # (lay_tile, lay_rotation), and the walks stop once one meets a path of it
    # outside ``lay_old_paths``. ``lay_hex`` is -1 otherwise.
    int lay_hex
    int lay_tile
    int lay_rotation
    uint32_t lay_old_paths
    bint found
    uint16_t cache[52]              # NUM_HEXES graph entries
    uint8_t recorded[52]            # paths recorded, by tile slot (Graph#compute's ``paths``)
    uint8_t visited[52]             # this token walk's visited paths, by tile slot
    uint8_t in_use[52]              # sides crossed on the walk's stack (the edge counter)
    uint64_t visited_nodes          # this token walk's visited nodes, by hex
    uint64_t token_nodes            # nodes this token walk recorded first (``local_nodes``)
    int token_node_count


cdef inline int _entry(int corp, int hex) noexcept nogil:
    return LAYOUT.graph_offset + corp * <int>NUM_HEXES + hex


cdef inline int _tile(GameState state, _Walk* w, int hex) noexcept nogil:
    return w.lay_tile if hex == w.lay_hex else hex_tile(state, hex)


cdef inline int _rotation(GameState state, _Walk* w, int hex) noexcept nogil:
    return w.lay_rotation if hex == w.lay_hex else hex_rotation(state, hex)


cdef inline bint _shared(uint32_t paths) noexcept nogil:
    """Two or more paths: at an exit, ``Tile#converging_exit?``."""
    return (paths & (paths - 1)) != 0


cdef inline bint _blocks(GameState state, _Walk* w, int hex, int tile) noexcept nogil:
    """``Node#blocks?``: an offboard, or a city whose slots hold only other corps' tokens."""
    if TILE_RC[tile] == <int>RC_OFFBOARD:
        return True
    return tile_blocks(tile, hex_tokens(state, hex), w.corp)


cdef inline void _record(_Walk* w, int hex, int slot, int pair) noexcept nogil:
    """The graph's block: a newly met path marks its sides here and across, and its node."""
    cdef int end, side
    if w.lay_hex >= 0:
        if hex == w.lay_hex and not (w.lay_old_paths >> pair & 1):
            w.found = True
        return
    if w.recorded[hex] >> slot & 1:
        return
    w.recorded[hex] |= <uint8_t>(1 << slot)
    for end in range(2):
        side = PAIR_A[pair] if end == 0 else PAIR_B[pair]
        if side == <int>NODE:
            w.cache[hex] |= <uint16_t>GRAPH_NODE
            if not (w.token_nodes >> hex & 1):
                w.token_nodes |= (<uint64_t>1) << hex
                w.token_node_count += 1
            continue
        w.cache[hex] |= <uint16_t>(1 << side)
        if HEX_EXIT_MASK[hex] >> side & 1:
            w.cache[HEX_NEIGHBOR[hex][side]] |= <uint16_t>(1 << ((side + 3) % <int>NUM_SIDES))


cdef void _walk_node(GameState state, _Walk* w, int hex, bint converging) noexcept nogil:
    """``Node#walk``: the node's paths in tile order."""
    cdef int tile, rotation, slot
    if w.found or w.visited_nodes >> hex & 1:
        return
    w.visited_nodes |= (<uint64_t>1) << hex
    tile = _tile(state, w, hex)
    rotation = _rotation(state, w, hex)
    for slot in range(TILE_PATH_COUNT[tile]):
        if PAIR_B[TILE_PATHS[tile][rotation][slot]] == <int>NODE:
            _walk_path(state, w, hex, slot, -1, converging, hex)
    if converging:
        w.visited_nodes &= ~((<uint64_t>1) << hex)


cdef void _walk_path(GameState state, _Walk* w, int hex, int slot, int entry, bint converging,
                     int origin) noexcept nogil:
    """``Path#walk`` of path ``slot`` of the hex, entered at side ``entry`` (-1
    from its node), in a walk that started at node ``origin``."""
    cdef int tile = _tile(state, w, hex)
    cdef int rotation = _rotation(state, w, hex)
    cdef int pair = TILE_PATHS[tile][rotation][slot]
    cdef int a = PAIR_A[pair]
    cdef int b = PAIR_B[pair]
    cdef int sides = (1 << a) | ((1 << b) if b != <int>NODE else 0)
    cdef int end, side, across, side_across, tile_across, rotation_across, k, pair_across
    cdef uint32_t paths_here
    cdef bint onward
    if w.found or w.visited[hex] >> slot & 1 or w.in_use[hex] & sides:
        return
    w.visited[hex] |= <uint8_t>(1 << slot)
    _record(w, hex, slot, pair)
    if b == <int>NODE and hex != origin and not _blocks(state, w, hex, tile):
        _walk_node(state, w, hex, converging)
    paths_here = TILE_MASK[tile][rotation]
    for end in range(2):
        side = a if end == 0 else b
        if side == <int>NODE or side == entry or not (HEX_EXIT_MASK[hex] >> side & 1):
            continue
        w.in_use[hex] |= <uint8_t>(1 << side)
        across = HEX_NEIGHBOR[hex][side]
        side_across = (side + 3) % <int>NUM_SIDES
        tile_across = _tile(state, w, across)
        rotation_across = _rotation(state, w, across)
        onward = converging or _shared(paths_here & ENDPOINT_PAIRS[side])
        for k in range(TILE_PATH_COUNT[tile_across]):
            pair_across = TILE_PATHS[tile_across][rotation_across][k]
            if PAIR_A[pair_across] == side_across or PAIR_B[pair_across] == side_across:
                _walk_path(state, w, across, k, side_across, onward, origin)
        w.in_use[hex] &= <uint8_t>~(1 << side)
    if converging:
        w.visited[hex] &= <uint8_t>~(1 << slot)


cdef bint _walk_tokens(GameState state, _Walk* w) noexcept nogil:
    """Every token's walk, in Ruby's hex order; whether one recorded two or
    more revenue centers first."""
    cdef uint64_t tokens = 0
    cdef bint route = False
    cdef int i, h, k
    for h in range(<int>NUM_HEXES):
        w.cache[h] = 0
        w.recorded[h] = 0
        if hex_tokens(state, h) >> w.corp & 1:
            tokens |= (<uint64_t>1) << h
            w.cache[h] = <uint16_t>(HEX_EXIT_MASK[h] | <int>GRAPH_NODE)
    for i in range(<int>NUM_HEXES):
        h = HEX_WALK_ORDER[i]
        if not (tokens >> h & 1):
            continue
        for k in range(<int>NUM_HEXES):
            w.visited[k] = 0
            w.in_use[k] = 0
        w.visited_nodes = tokens & ~((<uint64_t>1) << h)
        w.token_nodes = 0
        w.token_node_count = 0
        _walk_node(state, w, h, False)
        if w.found:
            break
        route = route or w.token_node_count >= 2
    return route


cdef void graph_refresh(GameState state, int corp) noexcept nogil:
    """Recompute the corp's entries and ``route_available`` if they are stale."""
    cdef int dirty_slot = LAYOUT.global_offset + GLOBALS.conn_dirty
    cdef int dirty = state._data[dirty_slot]
    cdef _Walk w
    cdef bint route
    cdef int h
    if not (dirty >> corp & 1):
        return
    w.corp = corp
    w.lay_hex = -1
    w.found = False
    route = _walk_tokens(state, &w)
    for h in range(<int>NUM_HEXES):
        state._data[_entry(corp, h)] = <int16_t>w.cache[h]
    state._data[LAYOUT.corps_offset + corp * CORP_FIELDS.size + CORP_FIELDS.route_available] = route
    state._data[dirty_slot] = <int16_t>(dirty & ~(1 << corp))


cdef bint graph_walks_new_track(GameState state, int corp, int hex, int tile, int rotation) noexcept nogil:
    """Whether the corp's walks, were ``tile`` laid on the hex at ``rotation``,
    would meet a path of it the hex's current tile lacks: the "new track"
    half of the semi-restrictive rule (``check_track_restrictions!`` reads
    ``connected_paths`` after the lay). The walks stop at the first such path."""
    cdef _Walk w
    w.corp = corp
    w.lay_hex = hex
    w.lay_tile = tile
    w.lay_rotation = rotation
    w.lay_old_paths = hex_paths(state, hex)
    w.found = False
    _walk_tokens(state, &w)
    return w.found


cdef int graph_sides(GameState state, int corp, int hex) noexcept nogil:
    graph_refresh(state, corp)
    return state._data[_entry(corp, hex)] & <int>GRAPH_SIDES


cdef bint graph_node(GameState state, int corp, int hex) noexcept nogil:
    graph_refresh(state, corp)
    return (state._data[_entry(corp, hex)] & <int>GRAPH_NODE) != 0


cdef bint graph_route_available(GameState state, int corp) noexcept nogil:
    graph_refresh(state, corp)
    return state._data[LAYOUT.corps_offset + corp * CORP_FIELDS.size + CORP_FIELDS.route_available] != 0


cdef inline int _check(int corp, int hex) except -1:
    assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
    assert 0 <= hex < <int>NUM_HEXES, f"hex {hex} out of range"
    return 0


cdef class Graph:
    """Stateless handle over the graph block."""

    cpdef bint dirty(self, GameState state, int corp):
        _check(corp, 0)
        return state._data[LAYOUT.global_offset + GLOBALS.conn_dirty] >> corp & 1

    cpdef void refresh(self, GameState state, int corp):
        _check(corp, 0)
        graph_refresh(state, corp)

    cpdef int sides_reached(self, GameState state, int corp, int hex):
        """Side mask of the hex the corp reaches (Ruby ``connected_hexes``)."""
        _check(corp, hex)
        return graph_sides(state, corp, hex)

    cpdef bint node_reached(self, GameState state, int corp, int hex):
        """Whether the corp reaches the hex's revenue center (``connected_nodes``)."""
        _check(corp, hex)
        return graph_node(state, corp, hex)

    cpdef bint route_available(self, GameState state, int corp):
        """Whether some token's walk reaches two or more revenue centers (Ruby's
        ``route_train_purchase``, which ``must_buy_train?`` reads)."""
        _check(corp, 0)
        return graph_route_available(state, corp)

    cpdef bint walks_new_track(self, GameState state, int corp, int hex, int tile, int rotation):
        _check(corp, hex)
        return graph_walks_new_track(state, corp, hex, tile, rotation)


# _Walk's arrays are sized with literals, and its node masks are 64 bits.
assert <int>NUM_HEXES == 52

GRAPH = Graph()

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in GraphBits.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
