"""INCOME phase handler.

INCOME (Phase 5 / ``PHASE_INCOME = 6``) is an automated engine phase
with no player decisions. Per RULES.md §Phase 5 and §Collect Income:

1. Players collect income from privately-owned companies.
2. Corporations collect income (revenue - CoO + synergy + ability).
3. FI collects stored income (includes +5 base bonus).
4. Any corp with negative income whose cash drops below zero goes
   bankrupt.

Players cannot go bankrupt — CLOSING's mandatory close ensures
``income + cash >= 0`` before we get here.

All state access goes through entity handles.
"""

from core.state cimport GameState
from core.data cimport GameConstants, GamePhases

# Late Python-level entity imports, same pattern as phases/end_card.pyx.
from entities import turn as turn_module
from entities import player as player_module
from entities import corp as corp_module
from entities import fi as fi_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _collect_player_income(GameState state) noexcept:
    """Add each player's income to their cash.

    Player income comes only from privately-owned companies (LOC_PLAYER).
    The cached value is kept current by the dirty-bit mechanism — no
    explicit recalculation here.
    """
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int i

    for i in range(num_players):
        player_module.PLAYERS[i].add_cash(
            state, player_module.PLAYERS[i].get_income(state))


cdef void _collect_corp_income(GameState state) noexcept:
    """Add each active corp's income to its cash; bankrupt if insolvent.

    A corp goes bankrupt only when it has negative income AND its cash
    drops below zero after application. Corps with negative income but
    enough cash reserve survive — they just pay the bank.

    Iteration order (ascending corp_id) is deterministic. Bankruptcy
    side-effects (company removal, cache invalidation) don't
    retroactively change income already applied earlier in the loop.
    """
    cdef int corp_id
    cdef int income

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        income = corp_module.CORPS[corp_id].get_income(state)
        corp_module.CORPS[corp_id].add_cash(state, income)
        if income < 0 and corp_module.CORPS[corp_id].get_cash(state) < 0:
            corp_module.CORPS[corp_id].go_bankrupt(state)


cdef void _collect_fi_income(GameState state) noexcept:
    """Apply FI income to FI cash.

    FI income is always non-negative after CLOSING (mandatory close
    removes negative-income FI companies). The +5 base bonus ensures
    income >= 5 even with no companies.
    """
    fi_module.FI.apply_income(state)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_income(GameState state) noexcept:
    """Execute INCOME phase logic end-to-end.

    Automated — no action input, no return value. The driver calls
    this when ``TURN.get_phase(state) == PHASE_INCOME`` and then
    continues dispatching to the next phase without presenting an
    action to any player.
    """
    _collect_player_income(state)
    _collect_corp_income(state)
    _collect_fi_income(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_DIVIDENDS)


# =============================================================================
# PYTHON TEST WRAPPER
# =============================================================================

def apply_income_py(GameState state):
    """Python-accessible shim around the cdef ``apply_income``.

    The core handler is cdef-only so the driver can dispatch to it on
    the nogil hot path. Smoke tests and scratch scripts need a Python
    entry point — this is it.
    """
    apply_income(state)
