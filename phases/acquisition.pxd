# cython: language_level=3
"""ACQUISITION phase stub declarations."""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_acquisition_action(GameState state, ActionInfo* info) noexcept
cdef void _transition_to_closing(GameState state) noexcept
