"""ACQ_SELECT_CORP phase handler declarations.

First leg of the three-step ACQ flow (SELECT_CORP → SELECT_COMPANY →
SELECT_PRICE). Hosts the shared ACQ helpers that ``acq_offer.pyx`` and
``acq_select_price.pyx`` cimport — having one owner module for the
helpers is simpler than a dedicated ``acquisition_common.pyx``.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_acquisition_phase(GameState state) noexcept
cdef void apply_acq_select_corp_action(GameState state, ActionInfo* info) noexcept

# Shared helpers (cimported by acq_offer.pyx and acq_select_price.pyx)
cdef void _clear_acquisition_context(GameState state) noexcept
cdef void _resume_acquisition_after_offer(GameState state, int original_corp) noexcept
cdef void _execute_fi_buy(GameState state, int corp_id, int company_id) noexcept
cdef int _get_fi_purchase_price(int corp_id, int company_id) noexcept
cdef int _find_first_preemptor(GameState state, int company_id, int original_corp) noexcept
cdef int _find_first_active_player(GameState state) noexcept
cdef void _enter_acq_offer(
    GameState state, int offered_corp, int company_id, int price,
    int original_corp, int deciding_player,
) noexcept
