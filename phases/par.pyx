"""PAR phase handler.

PHASE_PAR (Phase 12) is the price-select half of the Form Corporation flow.
Action space: 14 par indices, no pass. Legality (enforced upstream by
``_enumerate_par``): par is valid for the active company's star tier, a market
slot is available, and the active player can afford the required payment.

The float (share transfer, cash flows, market claim, presidency) executes
here via ``_simulate_float`` + ``corp.float_corp`` — the same canonical helper
the extractor uses for its par-token preview, so legality and resolution can
not drift. After the float, ``active_corp`` is cleared and control returns to
IPO via ``_advance_to_next_company`` (or transitions to INVEST when no
player-owned companies remain).

Reference: RULES.md "Form Corporation" procedure.

All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo, ACTION_PAR
from entities.corp cimport _simulate_float
from entities.company cimport company_face_value
from phases.ipo cimport _advance_to_next_company

from entities import turn as turn_module
from entities import corp as corp_module
from entities import player as player_module


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_par_action(GameState state, ActionInfo* info) noexcept:
    """Execute the Form Corporation procedure at the chosen par price.

    Reads ``active_company`` and ``active_corp`` from the turn block (seeded
    by ``apply_ipo_action``), runs the float, drops the company from the
    remaining set, clears ``active_corp``, and advances to the next
    player-owned company (or transitions out of IPO).
    """
    assert info.action_type == <int>ACTION_PAR, \
        f"apply_par_action: unexpected type {info.action_type}"

    cdef int company_id = turn_module.TURN.get_active_company(state)
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    assert company_id >= 0, "apply_par_action: no active company"
    assert corp_id >= 0, "apply_par_action: no active corp"

    cdef int par_index = info.amount
    cdef int player_id = turn_module.TURN.get_active_player(state)
    cdef int face_value = company_face_value(company_id)

    cdef int float_shares, market_index, player_payment, corp_cash, issued
    (float_shares, market_index, player_payment, corp_cash, issued) = (
        _simulate_float(face_value, par_index)
    )

    # Float corp (handles: activate, transfer company, claim market space,
    # set price index, distribute shares, set presidency). Issued shares
    # are set by ``float_corp`` itself from ``float_shares``.
    corp_module.CORPS[corp_id].float_corp(
        state, player_id, company_id, market_index, float_shares
    )

    corp_module.CORPS[corp_id].set_cash(state, corp_cash)
    player_module.PLAYERS[player_id].add_cash(state, -player_payment)

    # Clear remaining flag and the per-PAR active_corp, then walk back to
    # IPO for the next player-owned company (or INVEST when done).
    turn_module.TURN.set_ipo_remaining(state, company_id, False)
    turn_module.TURN.clear_active_corp(state)
    _advance_to_next_company(state)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def apply_par_action_py(GameState state, int par_index):
    """Python-accessible shim around the cdef ``apply_par_action``."""
    cdef ActionInfo ai
    ai.phase = 8  # DPHASE_PAR
    ai.action_type = ACTION_PAR
    ai.corp_id = -1
    ai.company_id = -1
    ai.amount = par_index
    apply_par_action(state, &ai)
