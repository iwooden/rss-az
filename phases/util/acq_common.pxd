"""Shared ACQ helper declarations.

Cimported by all four ACQ sub-phase modules
(``acq_select_corp``, ``acq_select_company``, ``acq_select_price``,
``acq_offer``). Implementations live in ``acq_common.pyx`` — these
helpers are semantically "ACQ common," not owned by any single
sub-phase.
"""

from core.state cimport GameState

cdef void _clear_acquisition_context(GameState state) noexcept
cdef void _clear_acq_offer_flags(GameState state) noexcept
cdef void _clear_acq_pair(GameState state) noexcept
cdef void _handle_player_fi_buy(GameState state, int corp_id, int company_id) noexcept
cdef int _get_fi_purchase_price(int corp_id, int company_id) noexcept
cdef int _find_first_preemptor(GameState state, int company_id, int original_corp) noexcept
cdef int _find_first_active_player(GameState state) noexcept
cdef void _set_first_acquisition_player_or_closing(GameState state) noexcept
cdef void _advance_to_next_player(GameState state) noexcept
cdef void _execute_fi_buy(GameState state, int corp_id, int company_id) noexcept
cdef int _find_most_expensive_affordable_fi_company(
    GameState state, int corp_id,
) noexcept
cdef bint _find_receivership_forced_buy(
    GameState state, int* out_corp, int* out_company,
) noexcept
cdef bint _process_receivership_forced_buys(GameState state) noexcept
cdef void _enter_acq_offer(
    GameState state, int offered_corp, int company_id, int price,
    int original_corp, int deciding_player,
) noexcept
cdef void _resume_acquisition_after_offer(GameState state, int original_corp) noexcept
cdef void _execute_acq_transfer(
    GameState state, int buyer_corp, int company_id, int price, int loc,
) noexcept
cdef void _merge_acquisition_zones(GameState state) noexcept
cdef void _transition_to_closing(GameState state) noexcept
