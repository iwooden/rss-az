# cython: language_level=3
from core.state cimport GameState

cdef int apply_temp_end_turn(GameState state) noexcept
