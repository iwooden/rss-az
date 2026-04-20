"""ACQ_SELECT_CORP phase handler declarations.

First leg of the three-step ACQ flow (SELECT_CORP → SELECT_COMPANY →
SELECT_PRICE). Shared ACQ helpers live in ``phases.util.acq_common`` and
are cimported directly by each sub-phase that needs them.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_acquisition_phase(GameState state) noexcept
cdef void apply_acq_select_corp_action(GameState state, ActionInfo* info) noexcept
