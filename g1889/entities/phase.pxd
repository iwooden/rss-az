"""PHASE: the game phase (``GamePhase``) and its start's effects."""

from g1889.state cimport GameState

cdef int game_phase(GameState state) noexcept nogil
cdef int next_rust_phase(int phase) noexcept nogil

# PHASE.advance and PHASE.on_train_bought without the GIL, for a purchase
# tried on a scratch state (extraction's hypotheticals).
cdef void phase_advance(GameState state) noexcept nogil
cdef bint phase_on_train_bought(GameState state, int train_type) noexcept nogil


cdef class Phase:
    cpdef int current(self, GameState state)
    cpdef void set_current(self, GameState state, int phase)
    cpdef void advance(self, GameState state)
    cpdef bint on_train_bought(self, GameState state, int train_type)
