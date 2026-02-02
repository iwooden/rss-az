# cython: language_level=3
from core.state cimport GameState

cdef int apply_income(GameState state) noexcept
