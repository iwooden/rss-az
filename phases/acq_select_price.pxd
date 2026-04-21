"""ACQ_SELECT_PRICE phase handler declarations.

Final leg of the three-step ACQ flow. Entered only via
``apply_acq_select_company_action`` for non-FI targets, after active_corp
and active_company are both seeded. On resolution either enters ACQ_OFFER
(cross-president) or executes the acquisition directly and walks back to
PHASE_ACQ_SELECT_CORP with the same player active (stay-on-same-player).
FI purchases bypass this phase entirely and execute in SELECT_COMPANY.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void apply_acq_select_price_action(GameState state, ActionInfo* info) noexcept
