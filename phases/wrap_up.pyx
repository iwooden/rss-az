"""WRAP_UP phase handler.

WRAP_UP is an automated engine phase with no decision action. Per
RULES.md §Phase 2 it executes, in order:

1. Reorder players by descending cash, tie-breaking on ascending old
   turn order (stable on ties because every position is unique).
2. Clear the INVEST consecutive-pass counter so the next INVEST round
   starts fresh. (INVEST's pass handler transitions out *without*
   clearing the counter — see ``phases/invest.pyx::_handle_pass``.)
3. Foreign Investor purchase loop: buy every cheapest affordable
   LOC_AUCTION company at face value, drawing a replacement after each
   purchase. Drawn replacements land in LOC_REVEALED (``DECK.draw``
   also bumps CoO on color-tier boundaries or empty-deck transition).
4. Flip every LOC_REVEALED company to LOC_AUCTION — this includes both
   pre-existing cards revealed during INVEST auctions and new
   replacements drawn by the FI loop above.
5. Transition to PHASE_ACQUISITION. Phase-entry setup (active-player,
   remaining masks) is owned by ``phases/acquisition.pyx``; WRAP_UP
   only flips the phase enum.

All state access goes through entity handles — the handler imports no
layout constants and never indexes ``state._data`` directly. Cache
invalidation (player finance, FI income, corp derived income/stars,
company adjusted incomes) happens inside the entity handle methods, so
there is no manual refresh here.
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants,
    GamePhases,
    COMPANY_FACE_VALUE,
)
from phases.acquisition cimport setup_acquisition_phase

# Late Python-level entity imports, same pattern as phases/invest.pyx.
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import fi as fi_module
from entities import deck as deck_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _reorder_players_by_cash(GameState state) noexcept:
    """Reorder players by descending cash, ascending old turn order.

    Selection sort on (-cash, +old_position). Stable on ties because
    every ``turn_order`` value is a unique permutation index, so the
    secondary key never ties. 6-wide stack arrays size for MAX_PLAYERS;
    the 3p prototype only fills the first three slots.
    """
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int cash_values[6]
    cdef int old_positions[6]
    cdef int player_ids[6]
    cdef int i, j, best_idx, temp_id
    cdef int best_cash, best_pos, curr_cash, curr_pos

    # Snapshot current cash and turn order. Sorting operates on
    # player_ids[] so we can read cash/position via indirection without
    # shuffling the two input arrays.
    for i in range(num_players):
        player_ids[i] = i
        cash_values[i] = player_module.PLAYERS[i].get_cash(state)
        old_positions[i] = player_module.PLAYERS[i].get_turn_order(state)

    # Selection sort: for each output slot, scan the remainder for the
    # best (-cash, +old_position) candidate and swap it to the front.
    for i in range(num_players):
        best_idx = i
        best_cash = cash_values[player_ids[i]]
        best_pos = old_positions[player_ids[i]]

        for j in range(i + 1, num_players):
            curr_cash = cash_values[player_ids[j]]
            curr_pos = old_positions[player_ids[j]]

            # Higher cash wins; on cash tie, lower old position wins.
            if (curr_cash > best_cash
                    or (curr_cash == best_cash and curr_pos < best_pos)):
                best_idx = j
                best_cash = curr_cash
                best_pos = curr_pos

        if best_idx != i:
            temp_id = player_ids[i]
            player_ids[i] = player_ids[best_idx]
            player_ids[best_idx] = temp_id

    # Write new turn-order values. The i-th ranked player goes to
    # position i.
    for i in range(num_players):
        player_module.PLAYERS[player_ids[i]].set_turn_order(state, i)


cdef int _find_cheapest_affordable_available(GameState state) noexcept:
    """Return company_id of the cheapest LOC_AUCTION company FI can afford.

    Returns -1 if none exist. Company indices are sorted by ascending
    face value globally (see ``core/data.pyx::COMPANY_FACE_VALUE``),
    so the first hit in the 0..35 scan is the cheapest.
    """
    cdef int fi_cash = fi_module.FI.get_cash(state)
    cdef int company_id
    cdef int face_value

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_for_auction(state):
            face_value = COMPANY_FACE_VALUE[company_id]
            if face_value <= fi_cash:
                return company_id

    return -1


cdef void _fi_purchase_company(GameState state, int company_id) noexcept:
    """Execute a single FI face-value purchase of ``company_id``.

    Deducts face value from FI cash, transfers the company to LOC_FI
    (``transfer_to_fi`` cascades ``FI.calculate_income``), and draws a
    replacement card. An empty deck at draw time is legal — ``DECK.draw``
    returns -1 and mutates nothing in that case. Drawing a card that
    crosses a color-tier boundary (or exhausts the deck) bumps the
    cost-of-ownership level inside ``DECK.draw``; that cascade is
    handled by ``TURN.set_coo_level``.
    """
    cdef int face_value = COMPANY_FACE_VALUE[company_id]

    fi_module.FI.add_cash(state, -face_value)
    company_module.COMPANIES[company_id].transfer_to_fi(state)
    deck_module.DECK.draw(state)


cdef void _fi_purchase_loop(GameState state) noexcept:
    """FI buys cheapest affordable available companies until none remain.

    Re-query pattern (no snapshot): every purchase draws a replacement
    that lands in LOC_REVEALED, so the set of LOC_AUCTION candidates
    shrinks monotonically and the loop is guaranteed to terminate.
    """
    cdef int company_id

    while True:
        company_id = _find_cheapest_affordable_available(state)
        if company_id < 0:
            return
        _fi_purchase_company(state, company_id)


cdef void _reveal_all_to_auction(GameState state) noexcept:
    """Flip every LOC_REVEALED company to LOC_AUCTION."""
    cdef int company_id

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_revealed(state):
            company_module.COMPANIES[company_id].move_to_auction(state)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_wrap_up(GameState state) noexcept:
    """Execute WRAP_UP phase logic end-to-end.

    Automated — no action input, no return value. The driver calls
    this when ``TURN.get_phase(state) == PHASE_WRAP_UP`` and then
    continues dispatching to the next phase without presenting an
    action to any player.
    """
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)
    _fi_purchase_loop(state)
    _reveal_all_to_auction(state)

    # Hand off to ACQUISITION. Phase-entry setup (active-player,
    # passed-flags) is owned by ``phases/acquisition.pyx``.
    setup_acquisition_phase(state)


# =============================================================================
# PYTHON TEST WRAPPER
# =============================================================================

def apply_wrap_up_py(GameState state):
    """Python-accessible shim around the cdef ``apply_wrap_up``.

    The core handler is cdef-only so the driver can dispatch to it on
    the nogil hot path. Smoke tests and scratch scripts need a Python
    entry point — this is it.
    """
    apply_wrap_up(state)
