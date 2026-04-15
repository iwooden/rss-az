"""ACQUISITION phase handler declarations.

Two entry points: ``setup_acquisition_phase`` initializes phase context,
``apply_acquisition_action`` dispatches player PASS/ACQ_PRICE/FI_BUY decisions.
Shared helpers are exposed for ``phases/acq_offer.pyx`` to cimport.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_acquisition_phase(GameState state) noexcept
cdef void apply_acquisition_action(GameState state, ActionInfo* info) noexcept

# Shared helpers (cimported by acq_offer.pyx)
cdef void _clear_acquisition_context(GameState state) noexcept
cdef void _resume_acquisition_after_offer(GameState state, int original_corp) noexcept
cdef void _execute_fi_buy(GameState state, int corp_id, int company_id) noexcept
cdef int _get_fi_purchase_price(int corp_id, int company_id) noexcept
cdef int _find_first_preemptor(GameState state, int company_id) noexcept
cdef int _find_first_active_player(GameState state) noexcept
