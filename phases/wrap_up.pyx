# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""WRAP_UP phase handler implementation."""

from core.state cimport GameState
from core.data cimport GamePhases
from entities import turn as turn_module
from entities import player as player_module


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _reorder_players_by_cash(GameState state) noexcept:
    """
    Reorder players by descending cash with old position tie-breaking.

    Algorithm:
    1. Collect (cash, old_position, player_id) tuples
    2. Selection sort by (-cash, old_position): higher cash wins, equal cash -> lower old position wins
    3. Assign new turn order positions
    4. Set active player to new position 0
    """
    cdef int num_players = state._num_players
    cdef int[6] cash_values       # Max 6 players
    cdef int[6] old_positions
    cdef int[6] player_ids
    cdef int i, j, best_idx, temp_id
    cdef int best_cash, best_pos, curr_cash, curr_pos

    # Gather current state - all cdef vars declared at function start
    for i in range(num_players):
        player_ids[i] = i
        cash_values[i] = player_module.PLAYERS[i].get_cash(state)
        old_positions[i] = player_module.PLAYERS[i].get_turn_order(state)

    # Selection sort by (-cash, old_position) - stable for ties
    for i in range(num_players):
        best_idx = i
        best_cash = cash_values[player_ids[i]]
        best_pos = old_positions[player_ids[i]]

        for j in range(i + 1, num_players):
            curr_cash = cash_values[player_ids[j]]
            curr_pos = old_positions[player_ids[j]]

            # Higher cash wins, or if equal, lower old position wins
            if (curr_cash > best_cash or
                (curr_cash == best_cash and curr_pos < best_pos)):
                best_idx = j
                best_cash = curr_cash
                best_pos = curr_pos

        # Swap to front
        if best_idx != i:
            temp_id = player_ids[i]
            player_ids[i] = player_ids[best_idx]
            player_ids[best_idx] = temp_id

    # Apply new turn order
    for i in range(num_players):
        player_module.PLAYERS[player_ids[i]].set_turn_order(state, i)

    # Set active player to new position 0 (REORDER-03)
    state._set_active_player(player_ids[0])


# =============================================================================
# MAIN PHASE HANDLER
# =============================================================================

cdef int apply_wrap_up(GameState state) noexcept:
    """
    Execute WRAP_UP phase logic.

    This is a deterministic non-player phase with 0 actions.
    Steps:
    1. Reorder players by descending cash (tie-break by old position)
    2. Set active player to new position 0
    3. Clear consecutive passes for next INVEST round
    4. Transition to ACQUISITION (which stubs to INVEST)

    Returns: 0 always (deterministic, no failure modes)
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
