"""The route solver (1889/engine-design.md §8): a corp's maximum revenue over
its trains, and the routes that earn it.

``solve_routes`` is exact and needs no GIL, so extraction can run it in a
nogil prologue; ``out`` may be NULL when only the revenue is wanted.
``max_route_revenue`` (``route_max`` without the GIL) caches the result in
the corp's ``route_cache``, and the hexes of the routes that earn it in
``route_hexes`` (``route_hex``), behind ``route_dirty``, which MAP, TRAINS,
and PHASE writes set (``entities/corp.pxd``). ``best_routes_py`` returns the
routes for tests, analysis, and live play.

Two hypothetical route maxes are cached beside it, behind ``hyp_dirty``
(``routes_refresh_hypotheticals``; 1889/v1-design.md §5.9), each solved on a
scratch copy of the row:

- ``route_after_rust``: with no purchase, once the next phase that rusts a
  rank has started (``next_rust_phase``; the phases before it start on the
  way). The route max now when no rust is left, in phase D.
- ``route_with_next``: with one more train of the cheapest type on sale
  (``trains_cheapest_type``) and the phase that purchase starts
  (``phase_on_train_bought``). The route max now when the corp has no room
  under the train limit.

Only extraction reads them. They go stale with the route cache, and when the
cheapest train on sale changes type (``trains_set_depot``, ``trains_set_pool``).

``solve_routes_via`` is for a change that only adds routes, all through one
hex: a lay, or the corp's own token. From the route max before the change it
searches only sets of routes with one through the hex, on one forest for the
corp's trains and, if asked, for one more train too. Extraction's candidate
and token rows use it.
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
# After a lay or the corp's own token at ``hex``, from the route max before it.
cdef int solve_routes_via(GameState state, int corp, int hex, int floor, int extra_type, int extra_floor,
                          int* with_extra) noexcept nogil
cdef int route_max(GameState state, int corp) noexcept nogil
cdef bint route_hex(GameState state, int corp, int hex) noexcept nogil

# A corp's route max with one more train of ``train_type``, or with a D for a
# traded ``trade_in`` (-1 for none), and the phase the purchase starts: tried
# on ``scratch``, a state of ``state``'s capacity, which is left changed.
cdef int route_max_after_purchase(GameState state, GameState scratch, int corp, int train_type,
                                  int trade_in) noexcept nogil
cdef void routes_refresh_hypotheticals(GameState state, GameState scratch, int corp) noexcept nogil

cpdef int max_route_revenue(GameState state, int corp)
cpdef list best_routes_py(GameState state, int corp)
cpdef list route_hexes_py(GameState state, int corp)
cpdef tuple route_hypotheticals_py(GameState state, int corp)
