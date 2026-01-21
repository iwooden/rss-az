# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game driver for action dispatch and legal move generation.

The GameDriver routes actions to phase-specific handlers and provides
legal move generation for the neural network.
"""

from core.state cimport GameState
from core.actions cimport (
    ActionLayout, ActionInfo, compute_action_layout, decode_action
)
from core.actions import get_valid_action_mask
from core.data cimport GamePhases, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_GAME_OVER
from core.driver cimport ActionStatus, STATUS_OK, STATUS_INVALID, STATUS_GAME_OVER
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action


cdef class GameDriver:
    """
    Game driver for dispatching actions to phase handlers.

    Stateless singleton - all state is in the GameState object.
    Following the entity handle pattern from entities/turn.pyx.
    """

    def __cinit__(self):
        """Initialize driver (no state needed - stateless pattern)."""
        pass

    cpdef int apply_action(self, GameState state, int action_idx):
        """
        Apply action to game state by dispatching to appropriate phase handler.

        Args:
            state: GameState object to modify
            action_idx: Index into action vector (0 to total_actions-1)

        Returns:
            STATUS_OK (0) if action applied successfully
            STATUS_INVALID (1) if action is invalid for current state
            STATUS_GAME_OVER (2) if game ended after this action
        """
        cdef int num_players = state._num_players
        cdef ActionLayout layout = compute_action_layout(num_players)
        cdef ActionInfo info
        cdef int result

        # Validate action index is in bounds
        if action_idx < 0 or action_idx >= layout.total_size:
            return STATUS_INVALID

        # Get valid action mask and check if this action is legal
        cdef object mask = get_valid_action_mask(state)
        if mask[action_idx] != 1.0:
            return STATUS_INVALID

        # Decode action
        info = decode_action(&layout, action_idx)

        # Dispatch based on current phase
        cdef int phase = state.get_phase()

        if phase == PHASE_INVEST:
            result = apply_invest_action(state, &info)
        elif phase == PHASE_BID_IN_AUCTION:
            result = apply_bid_action(state, &info)
        else:
            # Other phases not yet implemented (stubs for Phase 3+)
            return STATUS_INVALID

        # Check if game ended after action
        if state.get_phase() == PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        return result

    cpdef object get_legal_moves(self, GameState state):
        """
        Get legal action mask for current state.

        Wraps get_valid_action_mask() for convenient access.

        Returns:
            Numpy float32 array where 1.0 = valid action, 0.0 = invalid
        """
        return get_valid_action_mask(state)


# Global singleton instance (stateless pattern)
DRIVER = GameDriver()
