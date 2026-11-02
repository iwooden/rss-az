"""A slow, independent route search, the route solver's reference (1889/engine-design.md §8).

It walks track path by path from every revenue center, as Ruby's
``Node#walk`` and ``Path#walk`` do with backtracking (``part/node.rb``,
``part/path.rb``), and keeps every walk that ``Route#revenue`` accepts
(``route.rb``): two or more stops, a city with the corp's token, no hex side
crossed twice, and no pass through an offboard or a city blocked for the
corp. A walk enters a hex across a side only by the paths at that side, so
it never reverses at a junction. Then it tries every assignment of those
routes to the corp's trains, each within its distance and sharing no hex
side with the others, cut only by the plain bound of each train's best
single route. It shares nothing with ``g1889/routes/solver.pyx`` but the
entity reads.
"""

from __future__ import annotations

from typing import NamedTuple

import g1889.data.tables as D
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.trains import TRAINS


class Route(NamedTuple):
    hexes: tuple[int, ...]         # every hex in route order, each stop once
    stops: tuple[int, ...]         # the revenue centers' hexes, in order
    edges: frozenset[tuple[int, int]]  # (hex, side) of every path end at a side


def _paths(state, h: int) -> list[tuple[int, int]]:
    tile, rotation = MAP.tile(state, h), MAP.rotation(state, h)
    return [(D.PAIR_A[p], D.PAIR_B[p]) for p in D.TILE_PATHS[tile][rotation][:D.TILE_PATH_COUNT[tile]]]


def normal(hexes) -> tuple[int, ...]:
    """A route's hexes in the lower of its two orientations."""
    return min(tuple(hexes), tuple(reversed(hexes)))


def legal_routes(state, corp: int) -> dict[tuple[int, ...], Route]:
    """Every route the corp may run, of any length, keyed by ``normal`` hexes."""
    paths = [_paths(state, h) for h in range(D.NUM_HEXES)]
    routes: dict[tuple[int, ...], Route] = {}

    def blocks(h: int) -> bool:
        return D.TILE_RC[MAP.tile(state, h)] == D.RC_OFFBOARD or MAP.blocked_for(state, h, corp)

    def cross(h: int, side: int, hexes: tuple, stops: tuple, edges: frozenset) -> None:
        """Leave ``h`` through ``side``, whose path end is in ``edges``."""
        if not D.HEX_EXIT_MASK[h] >> side & 1:
            return
        n, entry = D.HEX_NEIGHBOR[h][side], (side + 3) % D.NUM_SIDES
        if (n, entry) in edges:
            return
        for a, b in paths[n]:
            if entry not in (a, b):
                continue
            other = b if a == entry else a
            if other != D.NODE:
                if (n, other) not in edges:
                    cross(n, other, hexes + (n,), stops, edges | {(n, entry), (n, other)})
                continue
            if n in stops:
                continue
            here = Route(hexes + (n,), stops + (n,), edges | {(n, entry)})
            if any(MAP.tokens(state, s) >> corp & 1 for s in here.stops):
                routes.setdefault(normal(here.hexes), here)
            if not blocks(n):
                for a2, b2 in paths[n]:
                    if b2 == D.NODE and a2 != entry and (n, a2) not in here.edges:
                        cross(n, a2, here.hexes, here.stops, here.edges | {(n, a2)})

    for x in range(D.NUM_HEXES):
        for a, b in paths[x]:
            if b == D.NODE:
                cross(x, a, (x,), (x,), frozenset({(x, a)}))
    return routes


def revenue(state, stops, train: int) -> int:
    phase = PHASE.current(state)
    return sum(MAP.revenue(state, h, phase, train == D.TRAIN_D) for h in stops)


def trains_of(state, corp: int) -> list[int]:
    return [t for t in range(D.NUM_TRAIN_TYPES) for _ in range(TRAINS.corp_trains(state, corp, t))]


def best_revenue(state, corp: int) -> int:
    """The most the corp's trains earn together."""
    trains = trains_of(state, corp)
    routes = legal_routes(state, corp).values()
    options = [sorted(((revenue(state, r.stops, t), r.edges) for r in routes if len(r.stops) <= D.TRAIN_DISTANCE[t]),
                      key=lambda option: -option[0]) for t in trains]
    rest = [sum(o[0][0] for o in options[i:] if o) for i in range(len(options) + 1)]
    best = 0

    def search(i: int, used: frozenset, total: int) -> None:
        nonlocal best
        if i == len(trains):
            best = max(best, total)
            return
        search(i + 1, used, total)
        for value, edges in options[i]:
            if total + value + rest[i + 1] <= best:
                break
            if edges.isdisjoint(used):
                search(i + 1, used | edges, total + value)

    search(0, frozenset(), 0)
    return best
