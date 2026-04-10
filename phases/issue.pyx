"""ISSUE phase handler.

ISSUE_SHARES (Phase 9 / ``PHASE_ISSUE_SHARES = 9``) is a decision phase where
each active corporation's president may issue one share from the unissued
stack. Corps are processed in descending share-price order. Receivership corps
auto-issue if they have unissued shares. After all corps are processed, the
engine transitions to ``PHASE_IPO``.

Action space: 2 actions -- ``pass(0)`` + ``issue(1)``.

Reference: RULES.md Phase 8, Issue Share, Sell One Share.

All state access goes through entity handles.
"""

from core.state cimport GameState
from core.data cimport GameConstants, GamePhases, CorpIndices, MARKET_PRICES
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_ISSUE

# Late Python-level entity imports, same pattern as phases/dividends.pyx.
from entities import turn as turn_module
from entities import corp as corp_module
from entities import market as market_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _init_issue_remaining(GameState state) noexcept:
    """Set issue_remaining flags: True for active corps, False otherwise."""
    cdef int corp_id
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        turn_module.TURN.set_issue_remaining(
            state, corp_id,
            corp_module.CORPS[corp_id].is_active(state),
        )


cdef int _find_next_issue_corp(GameState state) noexcept:
    """Find the remaining active corp with the highest price_index.

    Returns corp_id or -1 if none remain.
    """
    cdef int corp_id, best_id, best_price, price
    best_id = -1
    best_price = -1
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not turn_module.TURN.is_issue_remaining(state, corp_id):
            continue
        price = corp_module.CORPS[corp_id].get_price_index(state)
        if price > best_price:
            best_price = price
            best_id = corp_id
    return best_id


cdef void _issue_one_share(GameState state, int corp_id) noexcept:
    """Execute the issue procedure: share transfer + price move + cash.

    Stock Masters (CORP_SM) receives current share price with no price drop.
    All other corps move to the next lower available market space and receive
    the new (lower) share price. Bankruptcy occurs if new price reaches 0.
    """
    cdef int unissued = corp_module.CORPS[corp_id].get_unissued_shares(state)
    assert unissued > 0, f"_issue_one_share: corp {corp_id} has no unissued shares"

    # 1. Transfer share: unissued -> issued + bank
    corp_module.CORPS[corp_id].set_unissued_shares(state, unissued - 1)
    corp_module.CORPS[corp_id].set_issued_shares(
        state, corp_module.CORPS[corp_id].get_issued_shares(state) + 1,
    )
    corp_module.CORPS[corp_id].set_bank_shares(
        state, corp_module.CORPS[corp_id].get_bank_shares(state) + 1,
    )

    # 2. Price adjustment + payment
    cdef int current_index, new_index

    if corp_id == <int>CorpIndices.CORP_SM:
        # Stock Masters: no price change, receive current price
        corp_module.CORPS[corp_id].add_cash(
            state, corp_module.CORPS[corp_id].get_share_price(state),
        )
    else:
        current_index = corp_module.CORPS[corp_id].get_price_index(state)

        # Find next lower available space
        new_index = market_module.MARKET.find_next_lower_space(state, current_index)

        # Free old space (index 26 is shared $75 slot, never freed)
        if current_index < 26:
            market_module.MARKET.set_space_available(state, current_index, True)
        # Claim new space
        if new_index < 26:
            market_module.MARKET.set_space_available(state, new_index, False)

        corp_module.CORPS[corp_id].set_price_index(state, new_index)

        if new_index == 0:
            corp_module.CORPS[corp_id].go_bankrupt(state)
            return

        # Corp receives new (lower) share price
        corp_module.CORPS[corp_id].add_cash(state, MARKET_PRICES[new_index])


cdef void _advance_to_next_corp(GameState state) noexcept:
    """Find the next corp to process. Auto-process receivership and no-shares.

    Sets active_corp/active_player for the next player-controlled corp with
    unissued shares, or transitions to PHASE_IPO if all corps are done.
    """
    cdef int corp_id, unissued
    while True:
        corp_id = _find_next_issue_corp(state)
        if corp_id == -1:
            # All done -> transition to IPO
            turn_module.TURN.clear_active_corp(state)
            turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_IPO)
            return

        unissued = corp_module.CORPS[corp_id].get_unissued_shares(state)

        if unissued == 0:
            # Nothing to issue -> auto-skip
            turn_module.TURN.set_issue_remaining(state, corp_id, False)
            continue

        if corp_module.CORPS[corp_id].is_in_receivership(state):
            # Receivership: must issue
            _issue_one_share(state, corp_id)
            turn_module.TURN.set_issue_remaining(state, corp_id, False)
            continue

        # Player-controlled with unissued shares -> decision
        turn_module.TURN.set_active_corp(state, corp_id)
        turn_module.TURN.set_active_player(
            state, corp_module.CORPS[corp_id].get_president_id(state),
        )
        return


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_issue_phase(GameState state) noexcept:
    """Initialize ISSUE phase: set remaining flags and find first corp."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ISSUE_SHARES)
    _init_issue_remaining(state)
    turn_module.TURN.clear_active_corp(state)
    _advance_to_next_corp(state)


cdef void apply_issue_action(GameState state, ActionInfo* info) noexcept:
    """Apply an issue decision for the active corp."""
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    assert corp_id >= 0, f"apply_issue_action: no active corp"
    if info.action_type == <int>ACTION_ISSUE:
        _issue_one_share(state, corp_id)
    turn_module.TURN.set_issue_remaining(state, corp_id, False)
    _advance_to_next_corp(state)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_issue_phase_py(GameState state):
    """Python-accessible shim around the cdef ``setup_issue_phase``."""
    setup_issue_phase(state)


def apply_issue_action_py(GameState state, int action_type):
    """Python-accessible shim around the cdef ``apply_issue_action``."""
    cdef ActionInfo ai
    ai.phase = 6  # DPHASE_ISSUE
    ai.action_type = action_type
    ai.corp_id = -1
    ai.company_id = -1
    ai.amount = 0
    apply_issue_action(state, &ai)
