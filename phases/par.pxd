"""PAR phase handler declarations.

PAR is the price-select half of the Form Corporation flow. Entered only via
``apply_ipo_action`` after a corp has been selected; on resolution the handler
clears ``active_corp`` and hands control back to IPO (or INVEST) through
``phases.ipo._advance_to_next_company``.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void apply_par_action(GameState state, ActionInfo* info) noexcept
