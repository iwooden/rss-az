# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
End Card phase implementation.

This phase is automatic - no player actions.

Flow:
1. Check if any corp has share price 75 -> game ends
2. Check if end card already flipped -> game ends
3. Check if no auction companies and deck empty -> flip end card, set CoO to 7
4. Set up issue phase state (corps that can still issue, sorted by share price)
5. Transition to Issue Shares phase
"""

cimport cython
from cython_core.state cimport (
    GameState, PHASE_END_CARD, PHASE_ISSUE_SHARES, PHASE_GAME_OVER,
    NUM_COMPANIES, NUM_CORPS
)
from cython_core.data cimport get_corp_share_count, MAX_SHARE_PRICE, COO_LEVEL_END_CARD_FLIPPED

# Import shared helpers
from cython_core.helpers.turn cimport IssueTurnOffsets, get_issue_turn_offsets


# =============================================================================
# END CARD PHASE CLASS
# =============================================================================

cdef class EndCardPhase:
    """
    Handles the End Card phase.

    This is fully automatic - checks for game end conditions,
    potentially flips the end card, then transitions to Issue Shares.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players

    cpdef void handle_end_card_phase(self, GameState state):
        """
        Main entry point - process entire End Card phase.
        """
        if state.get_phase() != PHASE_END_CARD:
            return

        # Check for game end conditions
        if self._check_game_end(state):
            return  # Game ended

        # Check if we need to flip the end card
        if self._should_flip_end_card(state):
            self._flip_end_card(state)

        # Set up issue phase and transition
        self._setup_issue_phase(state)

    # =========================================================================
    # GAME END CHECKS
    # =========================================================================

    cdef bint _check_game_end(self, GameState state) noexcept:
        """
        Check for game end conditions.

        Returns True if game ended, False otherwise.
        """
        # Condition 1: Any corp at 75 share price
        if self._any_corp_at_max_price(state):
            self._end_game(state)
            return True

        # Condition 2: End card already flipped
        if self._is_end_card_flipped(state):
            self._end_game(state)
            return True

        return False

    cdef bint _any_corp_at_max_price(self, GameState state) noexcept nogil:
        """Check if any active corp has share price 75."""
        cdef int corp_id
        for corp_id in range(NUM_CORPS):
            if state.is_corp_active(corp_id):
                if state.get_corp_share_price(corp_id) == MAX_SHARE_PRICE:
                    return True
        return False

    cdef bint _is_end_card_flipped(self, GameState state) noexcept nogil:
        """Check if end card has been flipped."""
        cdef float* turn = state._turn_ptr()
        # end_card_flipped is at offset 1 in turn state
        return turn[1] == 1.0

    cdef void _end_game(self, GameState state) noexcept:
        """End the game."""
        state.set_phase(PHASE_GAME_OVER)

    # =========================================================================
    # END CARD FLIP
    # =========================================================================

    cdef bint _should_flip_end_card(self, GameState state) noexcept nogil:
        """
        Check if end card should be flipped.

        Flip if no companies for auction AND deck is empty.
        """
        if self._has_any_auction_companies(state):
            return False
        if not self._is_deck_empty(state):
            return False
        return True

    cdef void _flip_end_card(self, GameState state) noexcept:
        """Flip the end card and update cost of ownership."""
        cdef float* turn = state._turn_ptr()

        # Set end_card_flipped flag
        turn[1] = 1.0

        # Update cost of ownership to level 7 (10 cost for R/O/Y/G)
        state.set_coo_level(COO_LEVEL_END_CARD_FLIPPED)

    cdef bint _has_any_auction_companies(self, GameState state) noexcept nogil:
        """Check if there are any companies available for auction."""
        cdef int company_id
        for company_id in range(NUM_COMPANIES):
            if state.is_company_for_auction(company_id):
                return True
        return False

    cdef bint _is_deck_empty(self, GameState state) noexcept nogil:
        """Check if the company deck is empty."""
        return state._get_deck_top() < 0

    # =========================================================================
    # ISSUE PHASE SETUP
    # =========================================================================

    cdef void _setup_issue_phase(self, GameState state) noexcept:
        """
        Set up state for the Issue Shares phase.

        - Clear issue_corp (one-hot for current corp)
        - Set issue_remaining flags for corps that can still issue
        - Corps are processed in ascending share price order
        """
        cdef IssueTurnOffsets offsets = get_issue_turn_offsets(self._num_players)
        cdef float* turn = state._turn_ptr()
        cdef int corp_id
        cdef int issued_shares, total_shares

        # Clear issue_corp (set to -1 to indicate no corp selected yet)
        for corp_id in range(NUM_CORPS):
            turn[offsets.issue_corp + corp_id] = -1.0

        # Set issue_remaining flags for active corps that can still issue
        for corp_id in range(NUM_CORPS):
            if not state.is_corp_active(corp_id):
                turn[offsets.issue_remaining + corp_id] = -1.0
                continue

            issued_shares = state.get_corp_issued_shares(corp_id)
            total_shares = get_corp_share_count(corp_id)

            if issued_shares < total_shares:
                turn[offsets.issue_remaining + corp_id] = 1.0
            else:
                turn[offsets.issue_remaining + corp_id] = 0.0

        # Transition to Issue Shares phase
        state.set_phase(PHASE_ISSUE_SHARES)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create EndCardPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = EndCardPhase(num_players)
    return _phase_handlers[num_players]


def get_constants():
    """Get constants for Python tests."""
    return {
        'MAX_SHARE_PRICE': MAX_SHARE_PRICE,
        'COO_LEVEL_END_CARD_FLIPPED': COO_LEVEL_END_CARD_FLIPPED,
    }
