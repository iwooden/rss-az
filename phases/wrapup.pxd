# cython: language_level=3
"""
Declaration file for wrap up phase.
"""

from cython_core.state cimport GameState

cdef class WrapUpPhase:
    cdef int _num_players
    cpdef void execute(self, GameState state)
