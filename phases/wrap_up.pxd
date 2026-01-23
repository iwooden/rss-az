# cython: language_level=3
"""WRAP_UP phase handler declarations."""

from core.state cimport GameState

cdef int apply_wrap_up(GameState state) noexcept
