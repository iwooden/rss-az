"""TRAINS: depot and pool counts by type, each corp's trains, and the moves
between them: buys, the D trade-in, discards, and rusting."""

from g1889.state cimport GameState

cdef int depot_count(GameState state, int train_type) noexcept nogil
cdef int pool_count(GameState state, int train_type) noexcept nogil
cdef int corp_train_count(GameState state, int corp, int train_type) noexcept nogil
cdef int corp_train_total(GameState state, int corp) noexcept nogil
cdef int trains_depot_types(GameState state) noexcept nogil
cdef int trains_cheapest_price(GameState state) noexcept nogil
cdef int trains_limit(GameState state) noexcept nogil
cdef int trains_first_crowded(GameState state) noexcept nogil
cdef int trade_in_price(int train_type) noexcept nogil


cdef class Trains:
    cpdef int depot(self, GameState state, int train_type)
    cpdef void set_depot(self, GameState state, int train_type, int count)
    cpdef int pool(self, GameState state, int train_type)
    cpdef void set_pool(self, GameState state, int train_type, int count)
    cpdef int corp_trains(self, GameState state, int corp, int train_type)
    cpdef void set_corp_trains(self, GameState state, int corp, int train_type, int count)
    cpdef int corp_total(self, GameState state, int corp)
    cpdef int depot_types(self, GameState state)
    cpdef int cheapest_price(self, GameState state)
    cpdef int limit(self, GameState state)
    cpdef bint crowded(self, GameState state, int corp)
    cpdef int first_crowded(self, GameState state)

    # Moves (rules §8.8, §9); BANK moves the money
    cdef void _give(self, GameState state, int corp, int train_type) except *
    cpdef void buy_from_depot(self, GameState state, int corp, int train_type)
    cpdef void buy_from_pool(self, GameState state, int corp, int train_type)
    cpdef void buy_from_corp(self, GameState state, int corp, int seller, int train_type, int price)
    cpdef void trade_in_for_d(self, GameState state, int corp, int train_type)
    cpdef void discard(self, GameState state, int corp, int train_type)
    cpdef void rust(self, GameState state, int train_type)
