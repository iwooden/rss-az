"""END_CARD phase handler.

END_CARD is an automated engine phase with no decision action. Per
RULES.md §Phase 7 and §Game End Conditions it runs the following
ordered checks:

1. If any **active** corp sits at the $75 market slot (``price_index``
   26) → transition to PHASE_GAME_OVER.
2. If the end card was already flipped on a previous turn →
   transition to PHASE_GAME_OVER.
3. If no unowned companies remain (nothing left in DECK / AUCTION /
   REVEALED) → flip the end card and bump cost-of-ownership to level
   7, then continue.
4. Otherwise → transition to PHASE_ISSUE_SHARES.

Case (3) only **flips** the end card; the game actually ends the next
time END_CARD runs and case (2) fires.

All state access goes through entity handles. ``set_coo_level(7)``
cascades through every company's adjusted income, every active corp's
cache, every player's finance cache, and the FI income — all handled
inside ``turn.pyx::set_coo_level``, so there is no manual refresh
here.

Phase-entry setup for ISSUE_SHARES (initializing the per-corp
``issue_remaining`` mask) is owned by ``phases/issue.pyx`` — END_CARD
only flips the phase enum.
"""

from core.state cimport GameState
from core.data cimport GameConstants, GamePhases
from entities.company cimport LOC_DECK, LOC_AUCTION, LOC_REVEALED

# Late Python-level entity imports, same pattern as phases/invest.pyx.
from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef bint _any_corp_at_max_price(GameState state) noexcept:
    """True if any active corp is parked on the $75 market slot.

    $75 is the top market index (``NUM_MARKET_SPACES - 1`` == 26). A
    corp only gets there via INVEST buy-share — which already
    transitions to PHASE_GAME_OVER directly — or via a post-dividend
    price move (which does not short-circuit). This check catches the
    latter path.
    """
    cdef int max_index = <int>GameConstants.NUM_MARKET_SPACES - 1
    cdef int corp_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if corp_module.CORPS[corp_id].get_price_index(state) == max_index:
            return True
    return False


cdef bint _no_unowned_companies(GameState state) noexcept:
    """True if no company is still in the deck, auction pool, or revealed.

    Unowned = LOC_DECK (undrawn) or LOC_AUCTION (available) or
    LOC_REVEALED (drawn mid-turn, not yet available). Everything else
    — LOC_PLAYER, LOC_FI, LOC_CORP, LOC_CORP_ACQ, LOC_REMOVED,
    LOC_EXCLUDED — counts as "taken out of the unowned pool" for the
    purpose of this check.
    """
    cdef int company_id
    cdef int loc

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        loc = company_module.COMPANIES[company_id].get_location(state)
        if loc == <int>LOC_DECK \
                or loc == <int>LOC_AUCTION \
                or loc == <int>LOC_REVEALED:
            return False
    return True


cdef void _flip_end_card(GameState state) noexcept:
    """Flip the end card and cascade CoO to level 7.

    ``set_coo_level`` already re-derives every company's adjusted
    income and invalidates corp/player caches, so this is the only
    bookkeeping needed.
    """
    turn_module.TURN.set_end_card_flipped(state, True)
    turn_module.TURN.set_coo_level(
        state, <int>GameConstants.COO_LEVEL_END_CARD_FLIPPED)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_end_card(GameState state) noexcept:
    """Execute END_CARD phase logic end-to-end.

    Automated — no action input, no return value. The driver calls
    this when ``TURN.get_phase(state) == PHASE_END_CARD`` and then
    continues dispatching to the next phase without presenting an
    action to any player.
    """
    # (1) Any active corp at $75 ends the game immediately.
    if _any_corp_at_max_price(state):
        turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_GAME_OVER)
        return

    # (2) End card already flipped on a previous turn ends the game.
    if turn_module.TURN.is_end_card_flipped(state):
        turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_GAME_OVER)
        return

    # (3) No unowned companies left — flip the end card and continue.
    # The game will end the next time END_CARD runs and case (2) fires.
    if _no_unowned_companies(state):
        _flip_end_card(state)

    # (4) Normal transition to ISSUE_SHARES. Phase-entry setup for
    # ISSUE_SHARES is owned by ``phases/issue.pyx``.
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ISSUE_SHARES)


# =============================================================================
# PYTHON TEST WRAPPER
# =============================================================================

def apply_end_card_py(GameState state):
    """Python-accessible shim around the cdef ``apply_end_card``.

    The core handler is cdef-only so the driver can dispatch to it on
    the nogil hot path. Smoke tests and scratch scripts need a Python
    entry point — this is it.
    """
    apply_end_card(state)
