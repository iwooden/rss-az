"""ACQ_SELECT_COMPANY phase handler.

Middle of the three-step ACQ flow: SELECT_CORP → SELECT_COMPANY →
SELECT_PRICE. No pass here — SELECT_CORP already committed the player to
picking a target.

Dispatch:
  - LOC_FI target: run the FI buy (with preemption) directly. FI purchases
    have a fixed price (face for OS, high otherwise), so the model never
    sees a price decision for them — SELECT_PRICE is skipped.
  - LOC_CORP / LOC_PLAYER target: stamp active_company and transition to
    PHASE_ACQ_SELECT_PRICE.

Reference: RULES.md Acquisition procedure. See phase-refactor.md for the
split rationale.
"""

from core.state cimport GameState
from core.data cimport GamePhases
from core.actions cimport ActionInfo, ACTION_ACQ_SELECT_COMPANY
from entities.company cimport LOC_FI, company_location
from phases.util.acq_common cimport _handle_player_fi_buy

from entities import turn as turn_module


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_acq_select_company_action(GameState state, ActionInfo* info) noexcept:
    """Stamp active_company and dispatch.

    FI targets execute immediately (preemption via ACQ_OFFER if applicable);
    non-FI targets advance to SELECT_PRICE.
    """
    assert info.action_type == <int>ACTION_ACQ_SELECT_COMPANY, \
        f"apply_acq_select_company_action: unexpected type {info.action_type}"
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    cdef int company_id = info.company_id
    assert corp_id >= 0, "apply_acq_select_company_action: active_corp unset"

    turn_module.TURN.set_active_company(state, company_id)

    if company_location(state, company_id) == <int>LOC_FI:
        _handle_player_fi_buy(state, corp_id, company_id)
    else:
        turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_SELECT_PRICE)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def apply_acq_select_company_action_py(GameState state, int phase_id, int action_id):
    from core.actions import decode_action_py
    info_tuple = decode_action_py(phase_id, action_id)
    cdef ActionInfo info
    info.phase = info_tuple.phase
    info.action_type = info_tuple.action_type
    info.corp_id = info_tuple.corp_id
    info.company_id = info_tuple.company_id
    info.amount = info_tuple.amount
    apply_acq_select_company_action(state, &info)
