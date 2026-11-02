"""TRAINS entity; see trains.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport NUM_CORPS, NUM_TRAIN_TYPES, TRAIN_AVAILABLE_ON
from g1889.entities.corp cimport invalidate_route
from g1889.entities.phase cimport game_phase
from g1889.state cimport CORP_FIELDS, LAYOUT, TRAIN_FIELDS, GameState


cdef inline int _corp_slot(int corp, int train_type) noexcept nogil:
    return LAYOUT.corps_offset + corp * CORP_FIELDS.size + CORP_FIELDS.trains + train_type


cdef int depot_count(GameState state, int train_type) noexcept nogil:
    """Copies left in the depot; -1 is unlimited (D)."""
    return <int>state._data[LAYOUT.trains_offset + TRAIN_FIELDS.depot + train_type]


cdef int pool_count(GameState state, int train_type) noexcept nogil:
    return <int>state._data[LAYOUT.trains_offset + TRAIN_FIELDS.pool + train_type]


cdef int corp_train_count(GameState state, int corp, int train_type) noexcept nogil:
    return <int>state._data[_corp_slot(corp, train_type)]


cdef int corp_train_total(GameState state, int corp) noexcept nogil:
    cdef int t, total = 0
    for t in range(<int>NUM_TRAIN_TYPES):
        total += corp_train_count(state, corp, t)
    return total


cdef int trains_depot_types(GameState state) noexcept nogil:
    """The train types the depot sells now, as a type mask (``Depot#depot_trains``
    without the pool): the lowest rank left, and any later rank whose
    ``available_on`` phase has come (D from phase 6)."""
    cdef int phase = game_phase(state)
    cdef int mask = 0
    cdef int t
    for t in range(<int>NUM_TRAIN_TYPES):
        if depot_count(state, t) == 0:
            continue
        if mask == 0 or 0 <= TRAIN_AVAILABLE_ON[t] <= phase:
            mask |= 1 << t
    return mask


cdef bint trains_outside_depot(GameState state) noexcept nogil:
    """Whether the pool or any corp holds a train."""
    cdef int t, c
    for t in range(<int>NUM_TRAIN_TYPES):
        if pool_count(state, t):
            return True
        for c in range(<int>NUM_CORPS):
            if corp_train_count(state, c, t):
                return True
    return False


cdef inline int _check(int train_type, int count, int lowest) except -1:
    assert 0 <= train_type < <int>NUM_TRAIN_TYPES and count >= lowest, f"{count} trains of type {train_type}"
    return 0


cdef class Trains:
    """Stateless handle over the trains block and the corps' trains."""

    cpdef int depot(self, GameState state, int train_type):
        return depot_count(state, train_type)

    cpdef void set_depot(self, GameState state, int train_type, int count):
        _check(train_type, count, -1)
        state._data[LAYOUT.trains_offset + TRAIN_FIELDS.depot + train_type] = <int16_t>count

    cpdef int pool(self, GameState state, int train_type):
        return pool_count(state, train_type)

    cpdef void set_pool(self, GameState state, int train_type, int count):
        _check(train_type, count, 0)
        state._data[LAYOUT.trains_offset + TRAIN_FIELDS.pool + train_type] = <int16_t>count

    cpdef int corp_trains(self, GameState state, int corp, int train_type):
        return corp_train_count(state, corp, train_type)

    cpdef void set_corp_trains(self, GameState state, int corp, int train_type, int count):
        """Set a corp's count of one train type; the corp's route cache goes stale."""
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        _check(train_type, count, 0)
        state._data[_corp_slot(corp, train_type)] = <int16_t>count
        invalidate_route(state, corp)

    cpdef int corp_total(self, GameState state, int corp):
        assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
        return corp_train_total(state, corp)

    cpdef int depot_types(self, GameState state):
        return trains_depot_types(state)


TRAINS = Trains()
