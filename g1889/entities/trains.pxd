"""TRAINS: depot and pool counts by type, and each corp's trains."""

from g1889.state cimport GameState

cdef int depot_count(GameState state, int train_type) noexcept nogil
cdef int pool_count(GameState state, int train_type) noexcept nogil
cdef int corp_train_count(GameState state, int corp, int train_type) noexcept nogil
cdef int corp_train_total(GameState state, int corp) noexcept nogil
cdef int trains_depot_types(GameState state) noexcept nogil
cdef bint trains_outside_depot(GameState state) noexcept nogil


cdef class Trains:
    cpdef int depot(self, GameState state, int train_type)
    cpdef void set_depot(self, GameState state, int train_type, int count)
    cpdef int pool(self, GameState state, int train_type)
    cpdef void set_pool(self, GameState state, int train_type, int count)
    cpdef int corp_trains(self, GameState state, int corp, int train_type)
    cpdef void set_corp_trains(self, GameState state, int corp, int train_type, int count)
    cpdef int corp_total(self, GameState state, int corp)
    cpdef int depot_types(self, GameState state)
