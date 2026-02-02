# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""INCOME phase handler implementation."""

from core.state cimport GameState
from core.data cimport GamePhases, GameConstants
from entities import turn as turn_module
from entities import player as player_module
from entities import corp as corp_module
from entities import fi as fi_module


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _apply_income_to_corps(GameState state) noexcept:
    """
    Apply income to all active corporations, handling bankruptcy.

    Process order: corp_id 0-7
    Bankruptcy check: immediately after each corp's income application.
    Per CONTEXT.md: multiple corps can go bankrupt in same INCOME phase.
    """
    cdef int corp_id, income

    for corp_id in range(GameConstants.NUM_CORPS):
        corp = corp_module.CORPS[corp_id]

        if not corp.is_active(state):
            continue

        # Calculate and apply income (uses existing methods from Phase 22)
        income = corp.calculate_income(state)
        corp.apply_income(state, income)

        # Check bankruptcy immediately after application
        if corp.get_cash(state) < 0:
            corp.go_bankrupt(state)


cdef void _apply_income_to_fi(GameState state) noexcept:
    """Apply income to Foreign Investor."""
    cdef int income = fi_module.FI.calculate_income(state)
    fi_module.FI.apply_income(state, income)


cdef void _apply_income_to_players(GameState state) noexcept:
    """
    Apply income to all players.

    Per CONTEXT.md: Players CAN have negative income (fine).
    Per CONTEXT.md: Players CANNOT have negative cash after income.
    Assertion added to catch violations (should never happen due to CLOSING phase).
    """
    cdef int player_id, income, cash_after

    for player_id in range(state._num_players):
        player = player_module.PLAYERS[player_id]

        # Calculate and apply income (uses existing methods from Phase 22)
        income = player.get_income(state)
        player.add_cash(state, income)

        # Assert player cash non-negative (CLOSING should have prevented this)
        cash_after = player.get_cash(state)
        assert cash_after >= 0, f"Player {player_id} has negative cash {cash_after} after income"


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_income(GameState state) noexcept:
    """
    Execute INCOME phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Apply income to all corporations (with bankruptcy handling)
    2. Apply income to Foreign Investor
    3. Apply income to all players
    4. Transition to TEMP_END_TURN

    Per CONTEXT.md: Entity processing order doesn't matter (independent).
    Per CONTEXT.md: Check bankruptcy immediately per-corp after income.

    Returns: 0 always (deterministic, no failure modes)
    """
    # Apply income to all entities
    _apply_income_to_corps(state)
    _apply_income_to_fi(state)
    _apply_income_to_players(state)

    # Transition to TEMP_END_TURN phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_TEMP_END_TURN)
    return 0


def apply_income_py(GameState state):
    """Python wrapper for testing."""
    return apply_income(state)
