# cython: language_level=3
"""CLOSING phase handler declarations."""

from core.state cimport GameState

cdef int apply_closing_auto(GameState state) noexcept
cdef void _generate_close_offers(GameState state) noexcept
cdef bint _has_negative_adjusted_income(GameState state, int company_id) noexcept
cdef int _get_corp_president(GameState state, int corp_id) noexcept
