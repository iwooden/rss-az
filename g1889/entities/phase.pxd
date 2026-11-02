"""PHASE: the game phase (``GamePhase``) and its start's effects."""

from g1889.state cimport GameState

cdef int game_phase(GameState state) noexcept nogil


cdef class Phase:
    cpdef int current(self, GameState state)
    cpdef void set_current(self, GameState state, int phase)
    cpdef void advance(self, GameState state)
    cpdef bint on_train_bought(self, GameState state, int train_type)
