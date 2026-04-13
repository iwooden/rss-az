"""IPO phase handler.

PHASE_IPO (Phase 10) is a decision phase where each player-owned company
(LOC_PLAYER) may be floated into a new corporation. Companies are processed
in descending face-value order. For each company, the owner gets a single
decision: pass, or float a corp at a valid par price. The action space is
113 = 1 pass + 8x14 corp x par_index (merged IPO+PAR -- no separate PAR phase).

After all companies are processed, the engine starts a new turn and
transitions to PHASE_INVEST.

Reference: RULES.md "Form Corporation" procedure.

All state access goes through entity handles.
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases,
    PAR_PRICE_VALID,
    COMPANY_STARS,
)
from entities.corp cimport _simulate_float
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_IPO
from entities.company cimport (
    LOC_PLAYER,
    company_location,
    company_owner_id,
    company_face_value,
)

from entities import turn as turn_module
from entities import corp as corp_module
from entities import player as player_module
from entities import market as market_module


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


cdef void _process_ipo(GameState state, int corp_id, int par_index) noexcept:
    """Execute the Form Corporation procedure for the active IPO company."""
    cdef int company_id = turn_module.TURN.get_ipo_company(state)
    cdef int player_id = company_owner_id(state, company_id)
    cdef int face_value = company_face_value(company_id)

    # Canonical float simulation — returns everything derived from
    # (face_value, par_index). The same helper drives the extractor's
    # par-token preview so they can't drift.
    cdef int float_shares, market_index, player_payment, corp_cash, issued
    (float_shares, market_index, player_payment, corp_cash, issued) = (
        _simulate_float(face_value, par_index)
    )

    # Float corp (handles: activate, transfer company, claim market space,
    # set price index, distribute shares, set presidency). Issued shares
    # are set by ``float_corp`` itself from ``float_shares``.
    corp_module.CORPS[corp_id].float_corp(state, player_id, company_id, market_index, float_shares)

    corp_module.CORPS[corp_id].set_cash(state, corp_cash)
    player_module.PLAYERS[player_id].add_cash(state, -player_payment)

    # Clear from remaining
    turn_module.TURN.set_ipo_remaining(state, company_id, False)


cdef void _transition_out_of_ipo(GameState state) noexcept:
    """End-of-phase cleanup; starts a new turn."""
    # Clear IPO state
    turn_module.TURN.clear_ipo_company(state)

    # Increment turn number
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Set active player to position 0 (start of new turn)
    cdef int first_player = turn_module.TURN.find_player_at_position(state, 0)
    turn_module.TURN.set_active_player(state, first_player)

    # Transition to INVEST
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_INVEST)


cdef void _advance_to_next_company(GameState state) noexcept:
    """Find next company or transition out of IPO."""
    cdef int company_id = _find_next_ipo_company(state)
    if company_id < 0:
        _transition_out_of_ipo(state)
        return

    # Set up for this company's owner's decision
    turn_module.TURN.set_ipo_company(state, company_id)
    cdef int player_id = company_owner_id(state, company_id)
    turn_module.TURN.set_active_player(state, player_id)


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_ipo_phase(GameState state) noexcept:
    """Initialize IPO phase: set remaining flags and find first company."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_IPO)
    _init_ipo_remaining(state)
    turn_module.TURN.clear_ipo_company(state)
    _advance_to_next_company(state)


cdef void apply_ipo_action(GameState state, ActionInfo* info) noexcept:
    """Apply an IPO decision for the active company."""
    cdef int company_id
    if info.action_type == <int>ACTION_PASS:
        company_id = turn_module.TURN.get_ipo_company(state)
        assert company_id >= 0, "apply_ipo_action: no active company"
        turn_module.TURN.set_ipo_remaining(state, company_id, False)
        _advance_to_next_company(state)
        return

    assert info.action_type == <int>ACTION_IPO, f"apply_ipo_action: unexpected type {info.action_type}"
    _process_ipo(state, info.corp_id, info.amount)
    _advance_to_next_company(state)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_ipo_phase_py(GameState state):
    """Python-accessible shim around the cdef ``setup_ipo_phase``."""
    setup_ipo_phase(state)


def apply_ipo_action_py(GameState state, int action_type, int corp_id=-1, int par_index=-1):
    """Python-accessible shim around the cdef ``apply_ipo_action``."""
    cdef ActionInfo ai
    ai.phase = 7  # DPHASE_IPO
    ai.action_type = action_type
    ai.corp_id = corp_id
    ai.company_id = -1
    ai.amount = par_index
    apply_ipo_action(state, &ai)
