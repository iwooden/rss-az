# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Wrap Up phase implementation.

The Wrap Up phase (Phase 2) is automatic:
1. Determine new player order by cash (descending), ties broken by old order
2. Foreign Investor buys cheapest available companies at face value
   - For each purchase, draw and reveal new company (unavailable this turn)
3. All revealed companies become available for auction
4. Transition to Acquisition phase

No player decisions in this phase.
"""

cimport cython
from state cimport GameState, NUM_COMPANIES
from data cimport get_company_face_value

# Phase constants
DEF PHASE_WRAP_UP = 2
DEF PHASE_ACQUISITION = 3


# =============================================================================
# PLAYER ORDER SORTING
# =============================================================================

cdef struct PlayerSortKey:
    int player_id
    int cash
    int old_order


cdef void sort_players_by_cash(GameState state) noexcept:
    """
    Sort players by cash (descending), ties broken by old turn order.
    Updates the turn_order one-hot for each player.
    """
    cdef int num_players = state._num_players
    cdef PlayerSortKey[6] keys  # Max 6 players
    cdef int i, j
    cdef PlayerSortKey temp

    # Collect current data
    for i in range(num_players):
        keys[i].player_id = i
        keys[i].cash = state.get_player_cash(i)
        keys[i].old_order = state.get_player_turn_order(i)

    # Simple insertion sort (small N, stable sort)
    for i in range(1, num_players):
        j = i
        while j > 0:
            # Compare: higher cash first, then lower old_order for ties
            if (keys[j].cash > keys[j-1].cash or
                (keys[j].cash == keys[j-1].cash and keys[j].old_order < keys[j-1].old_order)):
                temp = keys[j]
                keys[j] = keys[j-1]
                keys[j-1] = temp
                j -= 1
            else:
                break

    # Update turn order for each player
    for i in range(num_players):
        state.set_player_turn_order(keys[i].player_id, i)


# =============================================================================
# FOREIGN INVESTOR BUYING
# =============================================================================

cdef int find_cheapest_auction_company(GameState state) noexcept nogil:
    """Find the cheapest company available for auction. Returns -1 if none."""
    cdef int i
    cdef int cheapest_id = -1
    cdef int cheapest_value = 999999

    for i in range(NUM_COMPANIES):
        if state.is_company_for_auction(i):
            face_value = get_company_face_value(i)
            if face_value < cheapest_value:
                cheapest_value = face_value
                cheapest_id = i

    return cheapest_id


cdef void fi_buy_companies(GameState state) noexcept:
    """Foreign Investor buys cheapest available companies at face value."""
    cdef int company_id
    cdef int face_value
    cdef int fi_cash

    while True:
        # Find cheapest available
        company_id = find_cheapest_auction_company(state)
        if company_id < 0:
            break  # No companies available

        face_value = get_company_face_value(company_id)
        fi_cash = state.get_fi_cash()

        if fi_cash < face_value:
            break  # Can't afford

        # FI buys the company
        state.set_fi_owns_company(company_id, True)
        state.set_company_for_auction(company_id, False)
        state.add_fi_cash(-face_value)

        # Draw new company to revealed (unavailable this turn)
        state.draw_company_to_revealed()


# =============================================================================
# WRAP UP PHASE HANDLER
# =============================================================================

cdef class WrapUpPhase:
    """
    Wrap Up phase handler.

    This phase is automatic - no player decisions.
    Call execute() to run the entire phase.

    Attributes declared in wrapup.pxd.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players

    cpdef void execute(self, GameState state):
        """Execute the entire Wrap Up phase."""
        if state.get_phase() != PHASE_WRAP_UP:
            raise ValueError("Not in Wrap Up phase")

        # 1. Determine new player order
        sort_players_by_cash(state)

        # 2. Foreign Investor buys companies
        fi_buy_companies(state)

        # 3. Revealed companies become available
        state.move_revealed_to_auction()

        # 4. Transition to Acquisition phase
        state.set_phase(PHASE_ACQUISITION)
        state._set_active_player(0)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

cdef dict _phase_handlers = {}

def get_phase_handler(int num_players):
    """Get or create WrapUpPhase handler for player count."""
    if num_players not in _phase_handlers:
        _phase_handlers[num_players] = WrapUpPhase(num_players)
    return _phase_handlers[num_players]


def handle_wrap_up(GameState state):
    """Convenience function to execute wrap up phase."""
    handler = get_phase_handler(state._num_players)
    handler.execute(state)
