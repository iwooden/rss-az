"""Route solver; see solver.pxd.

The rules are those ``Route#revenue`` checks (``route.rb``,
``game/base.rb:1434-1625``) on the walks of ``Node#walk`` and ``Path#walk``
(``part/node.rb``, ``part/path.rb``), which Ruby's ``AutoRouter``
(``auto_router.rb``) searches. A route:

- has at least 2 stops and at most the train's distance. Every revenue
  center costs 1 (``visit_cost``), towns included, and D's 999 is no limit;
- counts every center it visits (``compute_stops`` keeps every visit for a
  numeric distance), each once (``check_cycles!``);
- includes a city with one of the corp's tokens (``check_route_token``);
- reaches an offboard, or a city whose slots hold only other corps' tokens,
  only at an end (``check_connected``, ``Node#blocks?``);
- crosses no hex side twice (``check_overlap``, keyed by hex and side; its
  town and node-pair keys never add a constraint in 1889). That holds
  within the route and across the corp's routes, so trains share no track
  but may share stops.

A center pays ``map_revenue``: its tile's value, an offboard's by phase, and
the D value to a D train (``route_base_revenue``).

**Segments.** Every path of a hex with a revenue center ends at the center,
and no other hex has one (``tests/g1889/test_routes.py``), so a route is a
sequence of centers joined by segments: track between two centers through
hexes without one. A segment walk takes the paths of the hex across a side
at that side only (a walk never reverses at a junction) and crosses no side
twice; each segment keeps its half-edges (hex * 6 + side) as a bitset.

**Routes.** Each route is generated once, from the lowest-numbered token
city on it, the anchor (lower token cities are off limits): first a right
ray from the anchor, then a left ray, the two sharing no center or
half-edge, with the left ray's first segment numbered above the right ray's,
which picks one of the route's two orientations. A ray may end at a
blocking center but not pass it. The routes, up to the longest train's
distance, form a forest stored in pre-order: a route's subtree holds the
routes that extend it, each of which keeps its track and adds stops.

**Trains.** The most trains k on earn, with some track taken, is the
better of train k running nothing and train k running a route r, which
earns r's revenue plus what trains k + 1 on earn with r's track taken too.
The search walks train k's forest, siblings in descending order of their
subtree's best revenue to train k. What the later trains earn can only drop
down a subtree, since each route keeps its parent's track, so the value at
a parent bounds its subtree: a subtree is cut once its best revenue plus
that bound cannot beat the best total, and the later trains' routes at the
parent are still their best at a child whose segment they miss. A call
needs the exact value only above a floor, and returns early below it.
Identical trains are interchangeable, so the next one takes a route only
from the first one's tree (its root at the anchor) on; every pair is still
tried once at least.
"""

from libc.stdint cimport int8_t, int16_t, int32_t, uint16_t, uint32_t, uint64_t
from libc.stdio cimport fprintf, stderr
from libc.stdlib cimport abort, free, malloc, realloc
from libc.string cimport memcpy, memset

from g1889.data.tables cimport (
    D_TRAIN, HEX_EXIT_MASK, HEX_NEIGHBOR, HEX_RC, MAX_ROUTE_HEXES, NODE, NUM_CORPS, NUM_HEXES, NUM_RCS, NUM_SIDES,
    NUM_TRAIN_TYPES, PAIR_A, PAIR_B, RC_HEX, RC_OFFBOARD, TILE_PATH_COUNT, TILE_PATHS, TILE_RC, TRAIN_DISTANCE,
)
from g1889.entities.hexmap cimport hex_rotation, hex_tile, hex_tokens, map_blocked_for, map_revenue
from g1889.entities.phase cimport game_phase, next_rust_phase, phase_advance, phase_on_train_bought
from g1889.entities.trains cimport (
    corp_train_count, corp_train_total, pool_count, trains_cheapest_type, trains_limit, trains_set_count,
    trains_set_pool,
)
from g1889.entities.turn cimport turn_num_players
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, ROUTE_HEX_WORDS, GameState, copy_row


cdef enum:
    EDGE_WORDS = 5                  # half-edges hex * 6 + side: 312 bits
    NO_NODE = -1
    SORTED_SIBLINGS = 64            # siblings a walk orders by their best revenue; more go in pre-order


cdef struct _Segment:
    uint64_t edges[5]               # EDGE_WORDS
    int a, b                        # its centers, a < b
    int first_hex, num_hexes        # its hexes from a's to b's, in _Solver.hexes


cdef struct _Node:                  # a route, and the root of the routes that extend it
    int32_t parent                  # the route it extends by one segment, or NO_NODE
    int32_t subtree_end             # one past its subtree's last node
    int32_t segment                 # the segment it adds
    int8_t end                      # the center it adds
    int8_t left                     # whether it extends the left ray
    int8_t anchor
    int8_t stops
    int16_t revenue[2]              # to other trains, to D trains
    int16_t best[4]                 # MAX_ROUTES: the most train k earns on a route in the subtree, or -1


cdef struct _Solver:
    uint32_t tokens                 # centers with the corp's token
    uint32_t blocks                 # centers a route may end at but not pass
    int value[2][25]                # NUM_RCS: each center's pay to other trains and to D trains
    int max_stops                   # the longest train's distance, at most NUM_RCS
    _Segment* segments
    int num_segments, segment_cap
    int8_t* hexes
    int num_hexes, hex_cap
    int adjacency_start[26]         # NUM_RCS + 1: the segments at each center, CSR
    int* adjacency
    _Node* nodes
    int num_nodes, node_cap
    # The route being extended
    int anchor
    int right                       # the right ray's last node
    int right_first                 # the right ray's first segment
    uint32_t visited
    uint64_t used[5]
    int stops
    int revenue[2]


cdef struct _Search:
    int num_trains
    int kind[4]                     # MAX_ROUTES, longest train first: 1 for a D train
    int distance[4]
    bint repeats[5]                 # train k has train k - 1's type (False past the last)
    int next_type[4]                # the first train after k of another type
    int bound[5]                    # the sum of the best single routes of trains k on


cdef void* _reserve(void* buffer, int* capacity, int count, size_t item) noexcept nogil:
    """``buffer`` with room for ``count`` items, doubling ``capacity`` as needed."""
    if count <= capacity[0]:
        return buffer
    capacity[0] = max(max(2 * capacity[0], count), 64)
    buffer = realloc(buffer, capacity[0] * item)
    if buffer == NULL:
        fprintf(stderr, b"solve_routes: out of memory\n")
        abort()
    return buffer


cdef inline bint _has(const uint64_t* bits, int i) noexcept nogil:
    return (bits[i >> 6] >> (i & 63)) & 1


cdef inline void _toggle(uint64_t* bits, int i) noexcept nogil:
    bits[i >> 6] ^= (<uint64_t>1) << (i & 63)


cdef inline bint _overlaps(const uint64_t* a, const uint64_t* b) noexcept nogil:
    return ((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3]) | (a[4] & b[4])) != 0


cdef inline void _union(uint64_t* out, const uint64_t* a, const uint64_t* b) noexcept nogil:
    cdef int i
    for i in range(<int>EDGE_WORDS):
        out[i] = a[i] | b[i]


# =============================================================================
# SEGMENTS
# =============================================================================

cdef void _add_segment(_Solver* s, int a, int b, const uint64_t* used, const int8_t* trail,
                       int length) noexcept nogil:
    cdef _Segment* segment
    s.segments = <_Segment*>_reserve(s.segments, &s.segment_cap, s.num_segments + 1, sizeof(_Segment))
    s.hexes = <int8_t*>_reserve(s.hexes, &s.hex_cap, s.num_hexes + length, sizeof(int8_t))
    segment = &s.segments[s.num_segments]
    memcpy(segment.edges, used, sizeof(segment.edges))
    segment.a = a
    segment.b = b
    segment.first_hex = s.num_hexes
    segment.num_hexes = length
    memcpy(&s.hexes[s.num_hexes], trail, length)
    s.num_hexes += length
    s.num_segments += 1


cdef void _follow(GameState state, _Solver* s, int origin, int hex, int side, uint64_t* used, int8_t* trail,
                  int length) noexcept nogil:
    """Cross ``side`` of ``hex``, already in ``used``, and take each path of
    the hex across at that side (``Path#walk``). A path to its center closes a
    segment from center ``origin``, kept from its lower center; any other
    leaves through its other side unless the walk has crossed it."""
    cdef int across, entry, tile, rotation, k, pair, exit_
    if not (HEX_EXIT_MASK[hex] >> side & 1):
        return
    across = HEX_NEIGHBOR[hex][side]
    entry = (side + 3) % <int>NUM_SIDES
    if _has(used, across * <int>NUM_SIDES + entry):
        return
    if length == <int>MAX_ROUTE_HEXES:
        fprintf(stderr, b"solve_routes: a segment crosses more than MAX_ROUTE_HEXES %d hexes\n",
                <int>MAX_ROUTE_HEXES)
        abort()
    tile = hex_tile(state, across)
    rotation = hex_rotation(state, across)
    _toggle(used, across * <int>NUM_SIDES + entry)
    trail[length] = <int8_t>across
    for k in range(TILE_PATH_COUNT[tile]):
        pair = TILE_PATHS[tile][rotation][k]
        if PAIR_A[pair] == entry:
            exit_ = PAIR_B[pair]
        elif PAIR_B[pair] == entry:
            exit_ = PAIR_A[pair]
        else:
            continue
        if exit_ == <int>NODE:
            if origin < HEX_RC[across]:
                _add_segment(s, origin, HEX_RC[across], used, trail, length + 1)
        elif not _has(used, across * <int>NUM_SIDES + exit_):
            _toggle(used, across * <int>NUM_SIDES + exit_)
            _follow(state, s, origin, across, exit_, used, trail, length + 1)
            _toggle(used, across * <int>NUM_SIDES + exit_)
    _toggle(used, across * <int>NUM_SIDES + entry)


cdef void _find_segments(GameState state, _Solver* s) noexcept nogil:
    """Every segment on the map, then the segments at each center."""
    cdef uint64_t used[5]
    cdef int8_t trail[104]          # MAX_ROUTE_HEXES
    cdef int cursor[25]             # NUM_RCS
    cdef int v, hex, tile, rotation, k, side, i
    cdef _Segment* segment
    memset(used, 0, sizeof(used))
    for v in range(<int>NUM_RCS):
        hex = RC_HEX[v]
        tile = hex_tile(state, hex)
        rotation = hex_rotation(state, hex)
        trail[0] = <int8_t>hex
        for k in range(TILE_PATH_COUNT[tile]):
            side = PAIR_A[TILE_PATHS[tile][rotation][k]]  # each path joins a side to the center
            _toggle(used, hex * <int>NUM_SIDES + side)
            _follow(state, s, v, hex, side, used, trail, 1)
            _toggle(used, hex * <int>NUM_SIDES + side)
    memset(s.adjacency_start, 0, sizeof(s.adjacency_start))
    for i in range(s.num_segments):
        s.adjacency_start[s.segments[i].a + 1] += 1
        s.adjacency_start[s.segments[i].b + 1] += 1
    for v in range(<int>NUM_RCS):
        s.adjacency_start[v + 1] += s.adjacency_start[v]
        cursor[v] = s.adjacency_start[v]
    s.adjacency = <int*>malloc(max(2 * s.num_segments, 1) * sizeof(int))
    if s.adjacency == NULL:
        fprintf(stderr, b"solve_routes: out of memory\n")
        abort()
    for i in range(s.num_segments):
        segment = &s.segments[i]
        s.adjacency[cursor[segment.a]] = i
        s.adjacency[cursor[segment.b]] = i
        cursor[segment.a] += 1
        cursor[segment.b] += 1


# =============================================================================
# ROUTES
# =============================================================================

cdef inline int _other(_Solver* s, int segment, int center) noexcept nogil:
    return s.segments[segment].b if s.segments[segment].a == center else s.segments[segment].a


cdef inline bint _open(_Solver* s, int segment, int end) noexcept nogil:
    return not (s.visited >> end & 1) and not _overlaps(s.segments[segment].edges, s.used)


cdef inline void _enter(_Solver* s, int segment, int end) noexcept nogil:
    cdef const uint64_t* edges = s.segments[segment].edges
    cdef int i
    for i in range(<int>EDGE_WORDS):
        s.used[i] |= edges[i]
    s.visited |= (<uint32_t>1) << end
    s.stops += 1
    s.revenue[0] += s.value[0][end]
    s.revenue[1] += s.value[1][end]


cdef inline void _leave(_Solver* s, int segment, int end) noexcept nogil:
    cdef const uint64_t* edges = s.segments[segment].edges
    cdef int i
    for i in range(<int>EDGE_WORDS):
        s.used[i] &= ~edges[i]
    s.visited &= ~((<uint32_t>1) << end)
    s.stops -= 1
    s.revenue[0] -= s.value[0][end]
    s.revenue[1] -= s.value[1][end]


cdef int _add_node(_Solver* s, int parent, int segment, int end, bint left) noexcept nogil:
    """Store the route being extended, which adds ``segment`` to route ``parent``."""
    cdef _Node* node
    s.nodes = <_Node*>_reserve(s.nodes, &s.node_cap, s.num_nodes + 1, sizeof(_Node))
    node = &s.nodes[s.num_nodes]
    node.parent = parent
    node.subtree_end = s.num_nodes + 1
    node.segment = segment
    node.end = <int8_t>end
    node.left = left
    node.anchor = <int8_t>s.anchor
    node.stops = <int8_t>s.stops
    node.revenue[0] = <int16_t>s.revenue[0]
    node.revenue[1] = <int16_t>s.revenue[1]
    s.num_nodes += 1
    return s.num_nodes - 1


cdef void _extend_left(_Solver* s, int center, int parent) noexcept nogil:
    """Store each extension of the left ray, at ``center``, by one segment,
    and extend it further; ``parent`` is the route extended."""
    cdef int i, segment, end, node
    for i in range(s.adjacency_start[center], s.adjacency_start[center + 1]):
        segment = s.adjacency[i]
        end = _other(s, segment, center)
        if (parent == s.right and segment <= s.right_first) or not _open(s, segment, end):
            continue
        _enter(s, segment, end)
        node = _add_node(s, parent, segment, end, True)
        if s.stops < s.max_stops and not (s.blocks >> end & 1):
            _extend_left(s, end, node)
        s.nodes[node].subtree_end = s.num_nodes
        _leave(s, segment, end)


cdef void _extend_right(_Solver* s, int center, int parent) noexcept nogil:
    """Store each extension of the right ray by one segment, then each left
    ray it allows, and extend the right ray further."""
    cdef int i, segment, end, node
    for i in range(s.adjacency_start[center], s.adjacency_start[center + 1]):
        segment = s.adjacency[i]
        end = _other(s, segment, center)
        if not _open(s, segment, end):
            continue
        _enter(s, segment, end)
        node = _add_node(s, parent, segment, end, False)
        if parent == NO_NODE:
            s.right_first = segment
        if s.stops < s.max_stops:
            s.right = node
            _extend_left(s, s.anchor, node)
            if not (s.blocks >> end & 1):
                _extend_right(s, end, node)
        s.nodes[node].subtree_end = s.num_nodes
        _leave(s, segment, end)


cdef void _find_routes(_Solver* s) noexcept nogil:
    cdef int anchor
    for anchor in range(<int>NUM_RCS):
        if not (s.tokens >> anchor & 1):
            continue
        s.anchor = anchor
        s.visited = (s.tokens & (((<uint32_t>1) << anchor) - 1)) | ((<uint32_t>1) << anchor)
        memset(s.used, 0, sizeof(s.used))
        s.stops = 1
        s.revenue[0] = s.value[0][anchor]
        s.revenue[1] = s.value[1][anchor]
        _extend_right(s, anchor, NO_NODE)


# =============================================================================
# TRAINS
# =============================================================================

cdef struct _Later:                 # what the trains after k earn after a route, and the routes they take
    int value                       # at most this; exactly this when exact
    bint exact
    int picks[4]                    # MAX_ROUTES: when exact, their routes (-1 for none)
    uint64_t edges[5]               # and those routes' track


cdef void _route_edges(_Solver* s, int node, uint64_t* edges) noexcept nogil:
    """Add route ``node``'s track to ``edges``."""
    while node != NO_NODE:
        _union(edges, edges, s.segments[s.nodes[node].segment].edges)
        node = s.nodes[node].parent


cdef int _walk(_Solver* s, _Search* q, int k, const uint64_t* used, int after, int root, int lo, int hi,
               const uint64_t* parent_edges, const _Later* later, int best, int* picks) noexcept nogil:
    """Search train ``k``'s routes in sibling subtrees ``lo`` to ``hi`` of
    the tree from node ``root`` (NO_NODE: they are roots), whose parent's
    track is ``parent_edges``, best subtree first (``_visit``). ``later``
    bounds what the trains after k earn after any of them, and ``best`` is
    the best total so far; returns the new best, with ``picks`` updated."""
    cdef int order[64]              # SORTED_SIBLINGS
    cdef int count = 0
    cdef int i = lo
    cdef int j
    while i < hi and count < <int>SORTED_SIBLINGS:
        j = count
        while j > 0 and s.nodes[order[j - 1]].best[k] < s.nodes[i].best[k]:
            order[j] = order[j - 1]
            j -= 1
        order[j] = i
        count += 1
        i = s.nodes[i].subtree_end
    if i < hi:                      # too many to order: pre-order
        i = lo
        while i < hi:
            best = _visit(s, q, k, used, after, root, i, parent_edges, later, best, picks)
            i = s.nodes[i].subtree_end
        return best
    for j in range(count):
        best = _visit(s, q, k, used, after, root, order[j], parent_edges, later, best, picks)
    return best


cdef int _visit(_Solver* s, _Search* q, int k, const uint64_t* used, int after, int root, int i,
                const uint64_t* parent_edges, const _Later* later, int best, int* picks) noexcept nogil:
    """Give train ``k`` route ``i``, unless it shares track with ``used`` or
    is numbered ``after`` or below, and search its subtree. What the later
    trains earn after its parent bounds what they earn after it, and is
    exact when their routes there still fit. A next train of k's type takes
    a route in i's tree or a later one."""
    cdef uint64_t edges[5]
    cdef uint64_t taken[5]
    cdef _Later here
    cdef _Node* node = &s.nodes[i]
    cdef int cap = node.best[k]
    cdef int next_after, floor, j
    if cap < 0 or cap + later.value <= best or node.subtree_end <= after + 1 \
            or _overlaps(s.segments[node.segment].edges, used):
        return best
    if root == NO_NODE:
        root = i
    next_after = root - 1 if q.repeats[k + 1] else NO_NODE
    _union(edges, parent_edges, s.segments[node.segment].edges)
    here = later[0]
    if i > after:
        if not (later.exact and not _overlaps(s.segments[node.segment].edges, later.edges)
                and not (0 <= later.picks[k + 1] <= next_after)):
            _union(taken, used, edges)
            floor = best - cap
            here.value = _solve(s, q, k + 1, taken, next_after, floor, here.picks)
            here.exact = here.value > floor
            memset(here.edges, 0, sizeof(here.edges))
            for j in range(k + 1, q.num_trains):
                if here.exact and here.picks[j] >= 0:
                    _route_edges(s, here.picks[j], here.edges)
        if node.revenue[q.kind[k]] + here.value > best:
            best = node.revenue[q.kind[k]] + here.value
            picks[k] = i
            for j in range(k + 1, q.num_trains):
                picks[j] = here.picks[j]
    elif here.exact and _overlaps(s.segments[node.segment].edges, here.edges):
        here.exact = False          # still a bound
    if cap + here.value > best and i + 1 < node.subtree_end:
        best = _walk(s, q, k, used, after, root, i + 1, node.subtree_end, edges, &here, best, picks)
    return best


cdef int _solve(_Solver* s, _Search* q, int k, const uint64_t* used, int after, int floor,
                int* picks) noexcept nogil:
    """The most trains ``k`` on earn with the half-edges ``used`` taken, when
    that is above ``floor``; otherwise at most ``floor``. Train k runs
    nothing, nor then do the trains of its type after it, or a route
    numbered above ``after`` (from train k - 1's tree on, when it has k's
    type, so each pair of identical trains' routes is tried at least once).
    ``picks[k:]`` gets the routes of a result above ``floor``, -1 for none."""
    cdef uint64_t empty[5]
    cdef _Later later
    cdef int best = max(floor, 0)
    cdef int k2, j
    if k == q.num_trains:
        return best
    for j in range(k, q.num_trains):
        picks[j] = -1
        later.picks[j] = -1
    k2 = q.next_type[k]
    later.value = _solve(s, q, k2, used, NO_NODE, best, later.picks)
    later.exact = later.value > best and k2 == k + 1
    memset(later.edges, 0, sizeof(later.edges))
    if later.value > best:
        best = later.value
        for j in range(k2, q.num_trains):
            picks[j] = later.picks[j]
            if later.picks[j] >= 0:
                _route_edges(s, later.picks[j], later.edges)
    if k2 != k + 1:                 # the trains of k's type ran nothing: bound what they earn
        later.value = q.bound[k + 1]
    memset(empty, 0, sizeof(empty))
    return _walk(s, q, k, used, after, NO_NODE, 0, s.num_nodes, empty, &later, best, picks)


cdef void _set_bests(_Solver* s, _Search* q) noexcept nogil:
    """Each node's best revenue to each train over its subtree, within the
    train's distance, and the trains' best single routes."""
    cdef int i, k, parent
    cdef _Node* node
    for i in range(s.num_nodes):
        node = &s.nodes[i]
        for k in range(q.num_trains):
            node.best[k] = node.revenue[q.kind[k]] if node.stops <= q.distance[k] else -1
    for i in range(s.num_nodes - 1, -1, -1):
        parent = s.nodes[i].parent
        for k in range(q.num_trains):
            if parent != NO_NODE:
                s.nodes[parent].best[k] = max(s.nodes[parent].best[k], s.nodes[i].best[k])
            elif s.nodes[i].best[k] > q.bound[k]:
                q.bound[k] = s.nodes[i].best[k]
    for k in range(q.num_trains - 1, -1, -1):
        q.bound[k] += q.bound[k + 1]


# =============================================================================
# OUTPUT
# =============================================================================

cdef int _append(_Solver* s, int segment, int start, int8_t* hexes, int count) noexcept nogil:
    """Append the segment's hexes after the first, walking it from center ``start``."""
    cdef _Segment* seg = &s.segments[segment]
    cdef int k
    if count + seg.num_hexes - 1 > <int>MAX_ROUTE_HEXES:
        fprintf(stderr, b"solve_routes: a route crosses more than MAX_ROUTE_HEXES %d hexes\n",
                <int>MAX_ROUTE_HEXES)
        abort()
    for k in range(1, seg.num_hexes):
        hexes[count] = s.hexes[seg.first_hex + (k if seg.a == start else seg.num_hexes - 1 - k)]
        count += 1
    return count


cdef int _write_hexes(_Solver* s, int node, int8_t* hexes) noexcept nogil:
    """Route ``node``'s hexes from the end of its left ray, through the
    anchor, to the end of its right ray."""
    cdef int path[25]               # NUM_RCS: the route's nodes, last first: its left ray, then its right
    cdef int n = 0
    cdef int left = 0
    cdef int count = 1
    cdef int i
    while node != NO_NODE:
        path[n] = node
        n += 1
        node = s.nodes[node].parent
    while left < n and s.nodes[path[left]].left:
        left += 1
    hexes[0] = <int8_t>RC_HEX[s.nodes[path[0]].end if left else s.nodes[path[0]].anchor]
    for i in range(left):           # toward the anchor
        count = _append(s, s.nodes[path[i]].segment, s.nodes[path[i]].end, hexes, count)
    for i in range(n - 1, left - 1, -1):
        count = _append(s, s.nodes[path[i]].segment, _other(s, s.nodes[path[i]].segment, s.nodes[path[i]].end),
                        hexes, count)
    return count


cdef int solve_routes(GameState state, int corp, RouteSet* out) noexcept nogil:
    """The corp's maximum revenue over its trains this turn; with ``out``, the
    routes that earn it, one per train with a route, longest train first."""
    cdef _Solver s
    cdef _Search q
    cdef int train[4]               # MAX_ROUTES
    cdef int picks[4]
    cdef uint64_t empty[5]
    cdef int phase = game_phase(state)
    cdef int t, n, v, hex, k, best
    memset(&s, 0, sizeof(s))
    memset(&q, 0, sizeof(q))
    memset(empty, 0, sizeof(empty))
    if out != NULL:
        out.revenue = 0
        out.num_routes = 0
    for t in range(<int>NUM_TRAIN_TYPES - 1, -1, -1):
        for n in range(corp_train_count(state, corp, t)):
            if q.num_trains == <int>MAX_ROUTES:
                fprintf(stderr, b"solve_routes: corp %d has more than MAX_ROUTES %d trains\n", corp, <int>MAX_ROUTES)
                abort()
            k = q.num_trains
            train[k] = t
            q.kind[k] = t == <int>D_TRAIN
            q.distance[k] = min(<int>TRAIN_DISTANCE[t], <int>NUM_RCS)
            q.repeats[k] = k > 0 and train[k - 1] == t
            s.max_stops = max(s.max_stops, q.distance[k])
            q.num_trains += 1
    for k in range(q.num_trains - 1, -1, -1):
        q.next_type[k] = q.next_type[k + 1] if k + 1 < q.num_trains and q.repeats[k + 1] else k + 1
    for v in range(<int>NUM_RCS):
        hex = RC_HEX[v]
        if hex_tokens(state, hex) >> corp & 1:
            s.tokens |= (<uint32_t>1) << v
        if TILE_RC[hex_tile(state, hex)] == <int>RC_OFFBOARD or map_blocked_for(state, hex, corp):
            s.blocks |= (<uint32_t>1) << v
        for k in range(2):
            s.value[k][v] = map_revenue(state, hex, phase, k)
    if q.num_trains == 0 or s.tokens == 0 or s.max_stops < 2:
        return 0

    _find_segments(state, &s)
    _find_routes(&s)
    _set_bests(&s, &q)
    for k in range(q.num_trains):
        picks[k] = -1
    best = _solve(&s, &q, 0, empty, NO_NODE, 0, picks)
    if out != NULL and best > 0:
        out.revenue = best
        for k in range(q.num_trains):
            if picks[k] < 0:
                continue
            out.train_type[out.num_routes] = train[k]
            out.route_revenue[out.num_routes] = s.nodes[picks[k]].revenue[q.kind[k]]
            out.num_hexes[out.num_routes] = _write_hexes(&s, picks[k], out.hexes[out.num_routes])
            out.num_routes += 1

    free(s.segments)
    free(s.hexes)
    free(s.adjacency)
    free(s.nodes)
    return best


# =============================================================================
# CACHE AND PYTHON ENTRY POINTS
# =============================================================================

cdef inline int _corp_slot(int corp, int field) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + field


cdef int route_max(GameState state, int corp) noexcept nogil:
    """The corp's maximum revenue, solved again only when ``route_dirty`` marks
    its ``route_cache`` stale; the solve also writes ``route_hexes``."""
    cdef int dirty_slot = LAYOUT.global_offset + GLOBALS.route_dirty
    cdef int cache_slot = _corp_slot(corp, CORP_FIELDS.route_cache)
    cdef int hexes_slot = _corp_slot(corp, CORP_FIELDS.route_hexes)
    cdef int dirty = state._data[dirty_slot]
    cdef uint16_t words[4]          # ROUTE_HEX_WORDS
    cdef RouteSet routes
    cdef int i, k, hex
    if dirty >> corp & 1:
        state._data[cache_slot] = <int16_t>solve_routes(state, corp, &routes)
        memset(words, 0, sizeof(words))
        for i in range(routes.num_routes):
            for k in range(routes.num_hexes[i]):
                hex = routes.hexes[i][k]
                words[hex >> 4] |= <uint16_t>(1 << (hex & 15))
        for i in range(<int>ROUTE_HEX_WORDS):
            state._data[hexes_slot + i] = <int16_t>words[i]
        state._data[dirty_slot] = <int16_t>(dirty & ~(1 << corp))
    return state._data[cache_slot]


cdef bint route_hex(GameState state, int corp, int hex) noexcept nogil:
    """Whether the hex is on one of the routes that earn the corp's route
    max, read raw from ``route_hexes``: fresh after ``route_max``."""
    return <uint16_t>state._data[_corp_slot(corp, CORP_FIELDS.route_hexes) + (hex >> 4)] >> (hex & 15) & 1


cdef int route_max_after_purchase(GameState state, GameState scratch, int corp, int train_type,
                                  int trade_in) noexcept nogil:
    """See solver.pxd. The purchase is the engine's: the traded train goes to
    the pool first (``TRAINS.trade_in_for_d``), where a 4 rusts as phase D
    begins, then ``phase_on_train_bought``. The caller keeps the corp within
    ``MAX_ROUTES`` trains."""
    copy_row(scratch, state)
    if trade_in >= 0:
        trains_set_count(scratch, corp, trade_in, corp_train_count(scratch, corp, trade_in) - 1)
        trains_set_pool(scratch, trade_in, pool_count(scratch, trade_in) + 1)
    trains_set_count(scratch, corp, train_type, corp_train_count(scratch, corp, train_type) + 1)
    phase_on_train_bought(scratch, train_type)
    return solve_routes(scratch, corp, NULL)


cdef void routes_refresh_hypotheticals(GameState state, GameState scratch, int corp) noexcept nogil:
    """Recompute the corp's ``route_after_rust`` and ``route_with_next`` if
    ``hyp_dirty`` marks them stale (solver.pxd), each on ``scratch``, a state
    of ``state``'s capacity. Refreshes the route cache first."""
    cdef int dirty_slot = LAYOUT.global_offset + GLOBALS.hyp_dirty
    cdef int dirty = state._data[dirty_slot]
    cdef int now = route_max(state, corp)
    cdef int rust = next_rust_phase(game_phase(state))
    cdef int next_type = trains_cheapest_type(state)
    cdef int held = corp_train_total(state, corp)
    cdef int after_rust = now, with_next = now
    if not (dirty >> corp & 1):
        return
    if rust >= 0 and held:
        copy_row(scratch, state)
        while game_phase(scratch) < rust:
            phase_advance(scratch)
        after_rust = solve_routes(scratch, corp, NULL)
    if next_type >= 0 and held < trains_limit(state):
        with_next = route_max_after_purchase(state, scratch, corp, next_type, -1)
    state._data[_corp_slot(corp, CORP_FIELDS.route_after_rust)] = <int16_t>after_rust
    state._data[_corp_slot(corp, CORP_FIELDS.route_with_next)] = <int16_t>with_next
    state._data[dirty_slot] = <int16_t>(dirty & ~(1 << corp))


cdef inline int _check(int corp) except -1:
    assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
    return 0


cpdef int max_route_revenue(GameState state, int corp):
    """The corp's maximum revenue over its trains, cached in ``route_cache``."""
    _check(corp)
    return route_max(state, corp)


cpdef list best_routes_py(GameState state, int corp):
    """The routes ``solve_routes`` picks, one dict per train with a route,
    longest train first: ``train`` (its type), ``revenue``, ``hexes`` (hex
    indices in route order, each stop once), and ``stops`` (the hexes of its
    revenue centers, in order). The cache is left as it is."""
    cdef RouteSet out
    cdef int i, k
    _check(corp)
    solve_routes(state, corp, &out)
    routes = []
    for i in range(out.num_routes):
        hexes = [<int>out.hexes[i][k] for k in range(out.num_hexes[i])]
        routes.append({'train': out.train_type[i], 'revenue': out.route_revenue[i], 'hexes': hexes,
                       'stops': [h for h in hexes if HEX_RC[h] >= 0]})
    return routes


cpdef list route_hexes_py(GameState state, int corp):
    """The hexes on the routes that earn the corp's route max, from the cache."""
    _check(corp)
    route_max(state, corp)
    return [h for h in range(<int>NUM_HEXES) if route_hex(state, corp, h)]


cpdef tuple route_hypotheticals_py(GameState state, int corp):
    """The corp's (route max after the next rust, route max with the next
    train), from the cache (``routes_refresh_hypotheticals``)."""
    _check(corp)
    cdef GameState scratch = GameState.from_array(state._array, turn_num_players(state), state.max_players)
    routes_refresh_hypotheticals(state, scratch, corp)
    return (<int>state._data[_corp_slot(corp, CORP_FIELDS.route_after_rust)],
            <int>state._data[_corp_slot(corp, CORP_FIELDS.route_with_next)])


# The solver's arrays are sized with literals: by the largest train limit
# (RouteSet, _Node, _Search), and by NUM_RCS, whose centers fit a 32-bit mask.
assert <int>NUM_RCS == 25 and <int>MAX_ROUTE_HEXES == 104
assert 64 * <int>EDGE_WORDS >= <int>NUM_HEXES * <int>NUM_SIDES
assert <int>ROUTE_HEX_WORDS == 4  # route_max's buffer

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in RouteLimits.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
