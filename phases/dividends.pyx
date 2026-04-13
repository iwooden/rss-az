"""DIVIDENDS phase handler.

DIVIDENDS (Phase 7 / ``PHASE_DIVIDENDS = 7``) is a decision phase where each
active corporation's president chooses a dividend amount per share. Corps are
processed in descending share-price order. Receivership corps auto-pay 0 and
are skipped (no player decision). After all corps are processed, the engine
transitions to ``PHASE_END_CARD``.

Action space: 26 actions (amounts 0-25, no pass action). ``action_id == amount``.

Reference: RULES.md Phase 6, Pay Dividends, Adjust Share Price, Target Stars,
Maximum Dividend Per Share.

Repeated corp reads go through entity-owned primitives; semantic mutations
remain on entity handles.
"""

from core.state cimport GameState
from core.data cimport GameConstants, GamePhases
from core.actions cimport ActionInfo, ACTION_DIVIDEND
from entities.corp cimport (
    corp_is_active,
    corp_issued_shares,
    corp_price_index,
    corp_is_in_receivership,
    corp_president_id,
    corp_pending_price_move,
)

# Late Python-level entity imports, same pattern as phases/income.pyx.
from entities import turn as turn_module
from entities import corp as corp_module
from entities import player as player_module
from entities import market as market_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _init_dividend_remaining(GameState state) noexcept:
    """Set dividend_remaining flags: True for active corps, False otherwise."""
    cdef int corp_id
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        turn_module.TURN.set_dividend_remaining(
            state, corp_id,
            corp_is_active(state, corp_id),
        )


cdef int _find_next_dividend_corp(GameState state) noexcept:
    """Find the remaining active corp with the highest price_index.

    Returns corp_id or -1 if none remain.
    """
    cdef int corp_id, best_id, best_price, price
    best_id = -1
    best_price = -1
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        if not turn_module.TURN.is_dividend_remaining(state, corp_id):
            continue
        price = corp_price_index(state, corp_id)
        if price > best_price:
            best_price = price
            best_id = corp_id
    return best_id


cdef void _pay_dividends(GameState state, int corp_id, int amount_per_share) noexcept:
    """Pay shareholders and deduct from corp cash.

    Bank shares "receive" dividends too (money leaves the corp) but the
    bank total is not tracked — only player payouts matter.
    """
    cdef int num_players, p, shares
    if amount_per_share <= 0:
        return
    num_players = turn_module.TURN.get_num_players(state)
    for p in range(num_players):
        shares = player_module.PLAYERS[p].get_shares(state, corp_id)
        if shares > 0:
            player_module.PLAYERS[p].add_cash(state, amount_per_share * shares)
    corp_module.CORPS[corp_id].add_cash(
        state, -(amount_per_share * corp_issued_shares(state, corp_id)),
    )


cdef int _find_target_index(GameState state, int current_index, int move) noexcept:
    """Compute target market index after price movement with slide logic.

    Slides through occupied spaces in the direction of movement. Returns 0
    for bankruptcy (slide to or past index 0). Index 26 ($75) is always
    available (shared slot).
    """
    cdef int target = current_index + move
    if target <= 0:
        return 0
    if target >= 26:
        return 26
    # Slide through occupied spaces in the direction of movement.
    if move > 0:
        while not market_module.MARKET.is_space_available(state, target):
            target += 1
            if target >= 26:
                return 26
    else:
        while not market_module.MARKET.is_space_available(state, target):
            target -= 1
            if target <= 0:
                return 0
    return target


cdef void _adjust_share_price(GameState state, int corp_id) noexcept:
    """Adjust corp's market position based on pending price move.

    Handles space availability (freeing old, claiming new) and bankruptcy
    when the target reaches index 0.
    """
    cdef int current_index = corp_price_index(state, corp_id)
    cdef int move = corp_pending_price_move(state, corp_id)
    cdef int target_index

    if move == 0:
        return

    target_index = _find_target_index(state, current_index, move)

    if target_index == 0:
        corp_module.CORPS[corp_id].go_bankrupt(state)
        return

    # Free old space (index 26 is always-available shared slot).
    if current_index < 26:
        market_module.MARKET.set_space_available(state, current_index, True)
    # Claim new space.
    if target_index < 26:
        market_module.MARKET.set_space_available(state, target_index, False)

    corp_module.CORPS[corp_id].set_price_index(state, target_index)


cdef void _advance_to_next_corp(GameState state) noexcept:
    """Find the next corp to process. Auto-process receivership corps.

    Sets active_corp/active_player for the next player-controlled corp,
    or transitions to PHASE_END_CARD if all corps are done.
    """
    cdef int corp_id
    while True:
        corp_id = _find_next_dividend_corp(state)
        if corp_id == -1:
            # All done.
            turn_module.TURN.clear_active_corp(state)
            turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_END_CARD)
            return

        if corp_is_in_receivership(state, corp_id):
            # Receivership: auto-pay 0, adjust price, clear flag, loop.
            _adjust_share_price(state, corp_id)
            turn_module.TURN.set_dividend_remaining(state, corp_id, False)
            continue

        # Player-controlled: set active context and return for decision.
        turn_module.TURN.set_active_corp(state, corp_id)
        turn_module.TURN.set_active_player(
            state, corp_president_id(state, corp_id),
        )
        return


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_dividends_phase(GameState state) noexcept:
    """Initialize DIVIDENDS phase: set remaining flags and find first corp."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_DIVIDENDS)
    _init_dividend_remaining(state)
    turn_module.TURN.clear_active_corp(state)
    _advance_to_next_corp(state)


cdef void apply_dividend_action(GameState state, ActionInfo* info) noexcept:
    """Apply a dividend amount decision for the active corp."""
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    cdef int amount = info.amount
    assert corp_id >= 0, f"apply_dividend_action: no active corp"
    _pay_dividends(state, corp_id, amount)
    _adjust_share_price(state, corp_id)
    turn_module.TURN.set_dividend_remaining(state, corp_id, False)
    _advance_to_next_corp(state)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_dividends_phase_py(GameState state):
    """Python-accessible shim around the cdef ``setup_dividends_phase``."""
    setup_dividends_phase(state)


def apply_dividend_action_py(GameState state, int amount):
    """Python-accessible shim around the cdef ``apply_dividend_action``."""
    cdef ActionInfo info
    info.phase = 5  # DPHASE_DIVIDENDS
    info.action_type = <int>ACTION_DIVIDEND
    info.corp_id = -1
    info.company_id = -1
    info.amount = amount
    apply_dividend_action(state, &info)
