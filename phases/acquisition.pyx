# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""ACQUISITION phase stub - transitions immediately to INVEST."""

from core.state cimport GameState
from core.data cimport GamePhases
from entities import turn as turn_module
from entities import player as player_module


# =============================================================================
# MAIN PHASE HANDLER (STUB)
# =============================================================================

cdef int apply_acquisition_stub(GameState state) noexcept:
    """
    Stub: ACQUISITION immediately transitions to INVEST.

    When ACQUISITION is fully implemented, this will be replaced with:
    - FI purchase logic (Phase 10)
    - Corp acquisition offers
    - Company availability updates

    Turn number increments in IPO phase (last phase of game turn), not here.
    """
    cdef int i

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    return 0
