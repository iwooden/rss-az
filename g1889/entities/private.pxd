"""PRIVATES: the private blocks (owner code and flags)."""

from g1889.state cimport GameState

cdef int private_owner(GameState state, int private) noexcept nogil
cdef int private_revenue(GameState state, int private) noexcept nogil


cdef class Private:
    cdef readonly int private_id
    cdef readonly str sym

    cpdef int owner(self, GameState state)
    cpdef void set_owner(self, GameState state, int owner)
    cpdef int flags(self, GameState state)
    cpdef void set_flags(self, GameState state, int flags)
    cpdef int revenue(self, GameState state)
