"""
Market entity declarations.

The market is a flat array of 27 availability flags inside the compact
GameState — one per share-price space ($0 / $5 / ... / $75). Static price
data lives in core.data and is read directly via cimported arrays.

The handle is stateless: every read derives its slot inline from
``LAYOUT.market_offset`` on ``core.state``. There is no per-instance
offset cache and no initialize() step.
"""

from libc.stdint cimport int16_t
from core.state cimport GameState


cdef void copy_market_availability(GameState state, int16_t* out_flags) noexcept nogil


# Free-function price-movement helpers, cimportable from nogil code outside
# the Market class (e.g. ``core/token_data.pyx``). The class methods on
# ``Market`` delegate to these so there is a single source of truth.
cdef int market_find_next_higher_space(GameState state, int current_index) noexcept nogil
cdef int market_find_next_lower_space(GameState state, int current_index) noexcept nogil


cdef class Market:
    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef bint _is_space_available(self, GameState state, int index) noexcept nogil
    cdef void _set_space_available(self, GameState state, int index, bint available) noexcept nogil
    cdef int _find_next_higher_space(self, GameState state, int current_index) noexcept nogil
    cdef int _find_next_lower_space(self, GameState state, int current_index) noexcept nogil

    # Space availability (Python-accessible wrappers)
    cpdef bint is_space_available(self, GameState state, int index)
    cpdef void set_space_available(self, GameState state, int index, bint available)

    # Price lookups (read directly from core.data static arrays)
    cpdef int get_price_at_index(self, int index)
    cpdef int get_index_for_price(self, int price)

    # Price movement helpers (Python-accessible wrappers)
    cpdef int find_next_higher_space(self, GameState state, int current_index)
    cpdef int find_next_lower_space(self, GameState state, int current_index)
