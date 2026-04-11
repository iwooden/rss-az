# cython: language_level=3
"""ACQ_OFFER phase handler declarations.

Single entry point: ``apply_acq_offer_action`` dispatches PASS/ACCEPT
decisions. No setup function — ACQ_OFFER is entered mid-action from
``apply_acquisition_action`` or as a cross-president offer.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void apply_acq_offer_action(GameState state, ActionInfo* info) noexcept
