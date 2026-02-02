# cython: language_level=3
"""CLOSING phase handler declarations."""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef int apply_closing_auto(GameState state) noexcept
cdef int apply_closing_action(GameState state, ActionInfo* info) noexcept
cdef void _handle_close_accept(GameState state) noexcept
cdef void _handle_close_pass(GameState state) noexcept
cdef void _generate_close_offers(GameState state) noexcept
cdef bint _has_negative_adjusted_income(GameState state, int company_id) noexcept
cdef void _present_next_close_offer(GameState state) noexcept
cdef bint _is_close_offer_valid(GameState state, int owner_type, int owner_id, int company_id) noexcept
cdef int _count_corp_companies(GameState state, int corp_id, int exclude_company_id) noexcept
cdef void _transition_to_income(GameState state) noexcept
