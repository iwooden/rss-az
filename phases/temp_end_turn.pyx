# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
TEMP_END_TURN phase handler implementation.

TEMPORARY PHASE: This phase consolidates end-of-turn bookkeeping while other
phases (DIVIDENDS, ISSUE_SHARES, IPO) are being implemented. Once all phases
exist, the turn increment logic should move to END_CARD phase or wherever
the game rules specify.

Current flow: INCOME -> TEMP_END_TURN -> INVEST
Target flow: INCOME -> DIVIDENDS -> END_CARD -> ISSUE_SHARES -> IPO -> INVEST
"""

from core.state cimport GameState
from core.data cimport GamePhases
from entities import turn as turn_module


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_temp_end_turn(GameState state) noexcept:
    """
    Execute TEMP_END_TURN phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Increment turn number
    2. Transition to INVEST

    NOTE: Roundtrip clearing happens in INVEST phase (before WRAP_UP transition),
    NOT here. Per CONTEXT.md: "Roundtrip info only relevant in INVEST phase -
    clearing it elsewhere pollutes state vector for model."

    Returns: 0 always (deterministic, no failure modes)
    """
    cdef int current_turn = turn_module.TURN.get_turn_number(state)

    # Increment turn number (end of turn bookkeeping)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Transition to INVEST phase (start new turn)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)
    return 0


def apply_temp_end_turn_py(GameState state):
    """Python wrapper for testing."""
    return apply_temp_end_turn(state)
