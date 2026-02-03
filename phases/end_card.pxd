# cython: language_level=3
from core.state cimport GameState

cdef int apply_end_card(GameState state) noexcept
