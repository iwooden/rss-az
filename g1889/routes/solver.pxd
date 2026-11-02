"""The route solver (1889/engine-design.md §8): a corp's maximum revenue over
its trains, and the routes that earn it.

``solve_routes`` is exact and needs no GIL, so extraction can run it in a
nogil prologue; ``out`` may be NULL when only the revenue is wanted.
``max_route_revenue`` (``route_max`` without the GIL) caches the result in
the corp's ``route_cache`` behind ``route_dirty``, which MAP, TRAINS, and
PHASE writes set (``entities/corp.pxd``). ``best_routes_py`` returns the
routes for tests, analysis, and live play.
"""

from libc.stdint cimport int8_t

from g1889.state cimport GameState

cpdef enum RouteLimits:
    MAX_ROUTES = 4                  # the largest train limit


cdef struct RouteSet:
    int revenue
    int num_routes                  # at most MAX_ROUTES, one per train that earns
    int train_type[4]
    int route_revenue[4]
    int num_hexes[4]
    int8_t hexes[4][104]            # MAX_ROUTE_HEXES: each route's hexes in order, its stops once
                                    # each; a crossing tile's hex can appear twice

cdef int solve_routes(GameState state, int corp, RouteSet* out) noexcept nogil
cdef int route_max(GameState state, int corp) noexcept nogil

cpdef int max_route_revenue(GameState state, int corp)
cpdef list best_routes_py(GameState state, int corp)
