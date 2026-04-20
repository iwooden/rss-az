"""ACQ_SELECT_CORP phase handler.

First of three sub-phases in the ACQ flow:

  ACQ_SELECT_CORP    → pick corp (or pass)
  ACQ_SELECT_COMPANY → pick target company
  ACQ_SELECT_PRICE   → pick price offset (or FI_BUY)

Active flags: entry seeds ``active_corp=-1, active_company=-1``. Selecting
a corp sets ``active_corp`` and transitions to PHASE_ACQ_SELECT_COMPANY.
The SELECT_PRICE handler clears both fields and walks back here after a
successful acquisition (stay-on-same-player).

Shared ACQ helpers (preemptor scan, FI buy, receivership forced-buy
loop, phase-exit cleanup, ACQ_OFFER push/resume) live in
``phases.util.acq_common``.

Reference: RULES.md Acquisition procedure. See phase-refactor.md for the
split rationale.
"""

from core.state cimport GameState
from core.data cimport GamePhases
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_ACQ_SELECT_CORP,
)
from phases.util.acq_common cimport (
    _advance_to_next_player,
    _process_receivership_forced_buys,
    _set_first_acquisition_player_or_closing,
)

from entities import turn as turn_module
from entities import player as player_module


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_acquisition_phase(GameState state) noexcept:
    """Initialize ACQ phase context. Called by WRAP_UP."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_SELECT_CORP)
    turn_module.TURN.clear_acquisition_context(state)
    turn_module.TURN.clear_passed_flags(state)

    if _process_receivership_forced_buys(state):
        return
    _set_first_acquisition_player_or_closing(state)


cdef void apply_acq_select_corp_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch a SELECT_CORP action. Assumes legality (driver guarantees).

    PASS: mark current player passed, advance.
    SELECT_CORP: seed active_corp, transition to SELECT_COMPANY.
    """
    cdef int pid
    if info.action_type == <int>ACTION_PASS:
        pid = turn_module.TURN.get_active_player(state)
        player_module.PLAYERS[pid].set_has_passed(state, True)
        _advance_to_next_player(state)
        return

    assert info.action_type == <int>ACTION_ACQ_SELECT_CORP, \
        f"apply_acq_select_corp_action: unexpected type {info.action_type}"
    turn_module.TURN.set_active_corp(state, info.corp_id)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_SELECT_COMPANY)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_acquisition_phase_py(GameState state):
    setup_acquisition_phase(state)


def apply_acq_select_corp_action_py(GameState state, int phase_id, int action_id):
    from core.actions import decode_action_py
    info_tuple = decode_action_py(phase_id, action_id)
    cdef ActionInfo info
    info.phase = info_tuple.phase
    info.action_type = info_tuple.action_type
    info.corp_id = info_tuple.corp_id
    info.company_id = info_tuple.company_id
    info.amount = info_tuple.amount
    apply_acq_select_corp_action(state, &info)
