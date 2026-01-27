# cython: language_level=3
"""CLOSING phase handler declarations."""

from core.state cimport GameState

cdef int apply_closing_auto(GameState state) noexcept
