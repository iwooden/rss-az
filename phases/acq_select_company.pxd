"""ACQ_SELECT_COMPANY phase handler declarations.

Middle leg of the three-step ACQ flow. Entered only via
``apply_acq_select_corp_action`` after active_corp is seeded; on
resolution transitions to PHASE_ACQ_SELECT_PRICE with active_company set.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void apply_acq_select_company_action(GameState state, ActionInfo* info) noexcept
