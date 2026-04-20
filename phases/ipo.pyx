"""IPO phase handler.

PHASE_IPO (Phase 10) is the corp-select half of the Form Corporation flow.
Companies are processed in descending face-value order. For each company the
owner gets: pass, or select one of the inactive corps. Action space is
9 = 1 pass + 8 corps. Par-price selection is PHASE_PAR; after PAR resolves,
``_advance_to_next_company`` walks to the next company (back to IPO) or
transitions to INVEST when none remain.

Active flags: IPO entry sets ``active_company`` and leaves ``active_corp=-1``.
Selecting a corp sets ``active_corp`` and switches to PHASE_PAR; the PAR
handler clears ``active_corp`` and runs ``_advance_to_next_company``.

Reference: RULES.md "Form Corporation" procedure.

All state access goes through entity handles.
"""

from core.state cimport GameState
from core.data cimport GameConstants, GamePhases
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_IPO
from entities.company cimport (
    LOC_PLAYER,
    company_location,
    company_owner_id,
    company_face_value,
)

from entities import turn as turn_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _init_ipo_remaining(GameState state) noexcept:
    """Mark ipo_remaining = True for every company at LOC_PLAYER."""
    cdef int company_id
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        turn_module.TURN.set_ipo_remaining(
            state, company_id,
            company_location(state, company_id) == <int>LOC_PLAYER,
        )


cdef int _find_next_ipo_company(GameState state) noexcept:
    """Find the remaining company with the highest face value.

    Returns company_id or -1 if none remain.
    """
    cdef int company_id, best_id, best_fv, fv
    best_id = -1
    best_fv = -1
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if not turn_module.TURN.is_ipo_remaining(state, company_id):
            continue
        # Double-check still player-owned
        if company_location(state, company_id) != <int>LOC_PLAYER:
            turn_module.TURN.set_ipo_remaining(state, company_id, False)
            continue
        fv = company_face_value(company_id)
        if fv > best_fv:
            best_fv = fv
            best_id = company_id
    return best_id


cdef void _transition_out_of_ipo(GameState state) noexcept:
    """End-of-phase cleanup; starts a new turn."""
    # Clear IPO state
    turn_module.TURN.clear_active_company(state)

    # Increment turn number
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Set active player to position 0 (start of new turn)
    cdef int first_player = turn_module.TURN.find_player_at_position(state, 0)
    turn_module.TURN.set_active_player(state, first_player)

    # Transition to INVEST
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_INVEST)


cdef void _advance_to_next_company(GameState state) noexcept:
    """Find next company or transition out of IPO.

    Called by ``setup_ipo_phase`` on entry, by the IPO PASS branch, and by
    the PAR handler after a float completes. On finding a company the engine
    is left in PHASE_IPO with ``active_company`` and ``active_player`` seeded.
    """
    cdef int company_id = _find_next_ipo_company(state)
    if company_id < 0:
        _transition_out_of_ipo(state)
        return

    # Set up for this company's owner's decision
    turn_module.TURN.set_active_company(state, company_id)
    cdef int player_id = company_owner_id(state, company_id)
    turn_module.TURN.set_active_player(state, player_id)
    # IPO is a per-company decision; ensure the engine phase is PHASE_IPO
    # (PAR handler walks us back here after a float).
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_IPO)


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_ipo_phase(GameState state) noexcept:
    """Initialize IPO phase: set remaining flags and find first company."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_IPO)
    _init_ipo_remaining(state)
    turn_module.TURN.clear_active_company(state)
    _advance_to_next_company(state)


cdef void apply_ipo_action(GameState state, ActionInfo* info) noexcept:
    """Apply an IPO corp-select decision for the active company.

    PASS: drop the active company from the remaining set and advance.
    ACTION_IPO: set ``active_corp`` and switch to PHASE_PAR. The float
    (share transfer, cash flows, market claim) happens in the PAR handler.
    """
    cdef int company_id
    if info.action_type == <int>ACTION_PASS:
        company_id = turn_module.TURN.get_active_company(state)
        assert company_id >= 0, "apply_ipo_action: no active company"
        turn_module.TURN.set_ipo_remaining(state, company_id, False)
        _advance_to_next_company(state)
        return

    assert info.action_type == <int>ACTION_IPO, f"apply_ipo_action: unexpected type {info.action_type}"
    turn_module.TURN.set_active_corp(state, info.corp_id)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_PAR)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_ipo_phase_py(GameState state):
    """Python-accessible shim around the cdef ``setup_ipo_phase``."""
    setup_ipo_phase(state)


def apply_ipo_action_py(GameState state, int action_type, int corp_id=-1):
    """Python-accessible shim around the cdef ``apply_ipo_action``."""
    cdef ActionInfo ai
    ai.phase = 7  # DPHASE_IPO
    ai.action_type = action_type
    ai.corp_id = corp_id
    ai.company_id = -1
    ai.amount = -1
    apply_ipo_action(state, &ai)
