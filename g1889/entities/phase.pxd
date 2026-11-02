"""PHASE: the game phase (``GamePhase``)."""

from g1889.state cimport GameState

cdef int game_phase(GameState state) noexcept nogil


cdef class Phase:
    cpdef int current(self, GameState state)
    cpdef void set_current(self, GameState state, int phase)
