# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
DIVIDENDS phase: Corporations pay dividends and adjust share prices.

DESIGN: Processing Order and Receivership Auto-Handling
========================================================
Processing order: Descending share price (highest first).
For each active corporation:
1. President chooses dividend per share (0 to max)
2. Corporation pays dividend × issued_shares from treasury
3. Each shareholder receives dividend × their_shares
4. Corporation adjusts share price based on owned stars vs required stars

Receivership corporations:
- Auto-processed in the loop (no player decision)
- Always pay $0 dividend
- Price adjusts normally based on stars

Phase transitions:
- After all corps processed → END_CARD (which routes to ISSUE_SHARES/IPO/INVEST or GAME_OVER)

Action space: 26 actions (dividend amounts 0-25 per share)
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases, CorpIndices,
    PHASE_END_CARD,
    get_required_stars, get_max_dividend, MARKET_PRICES
)
from core.actions cimport ActionInfo, ACTION_DIVIDEND
from entities import turn as turn_module
from entities import corp as corp_module
from entities import player as player_module
from entities import market as market_module


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _init_dividend_remaining(GameState state) noexcept:
    """
    Set dividend_remaining flags for all active corporations.

    Called at phase entry to mark which corps need processing.
    """
    cdef int corp_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[corp_id].is_active(state):
            turn_module.TURN.set_dividend_remaining(state, corp_id, True)
        else:
            turn_module.TURN.set_dividend_remaining(state, corp_id, False)


cdef int _find_next_dividend_corp(GameState state) noexcept:
    """
    Find the next corporation to process dividends.

    Processing order: Descending share price (highest price_index first).
    Only considers corps with dividend_remaining flag set.

    Returns:
        corp_id of next corp to process, or -1 if none remaining
    """
    cdef int corp_id, price_index
    cdef int best_corp = -1
    cdef int best_price_index = -1

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not turn_module.TURN.is_dividend_remaining(state, corp_id):
            continue
        if not corp_module.CORPS[corp_id].is_active(state):
            continue

        price_index = corp_module.CORPS[corp_id].get_price_index(state)
        if price_index > best_price_index:
            best_price_index = price_index
            best_corp = corp_id

    return best_corp


cdef void _pay_dividends(GameState state, int corp_id, int amount_per_share) noexcept:
    """
    Distribute dividends to all shareholders.

    Per RULES.md line 403: Pay dividend to owner of each issued share.
    - Players receive dividend × their shares
    - Bank shares: dividend goes to bank (no action needed, just deducted from corp)

    Args:
        state: Game state
        corp_id: Corporation paying dividends
        amount_per_share: Dividend amount per share
    """
    cdef int player_id, shares, payment
    cdef int total_payment = 0
    cdef int issued_shares = corp_module.CORPS[corp_id].get_issued_shares(state)

    if amount_per_share <= 0:
        return  # No dividend to pay

    # Pay each player their share of dividends
    for player_id in range(state._num_players):
        shares = player_module.PLAYERS[player_id].get_shares(state, corp_id)
        if shares > 0:
            payment = amount_per_share * shares
            player_module.PLAYERS[player_id].add_cash(state, payment)
            total_payment += payment

    # Bank shares also receive payment (just deducted from corp)
    # Total deduction = amount_per_share × issued_shares
    # (Bank shares = issued - player_owned, but we just pay total)
    cdef int total_cost = amount_per_share * issued_shares
    corp_module.CORPS[corp_id].add_cash(state, -total_cost)


cdef int _calculate_price_move(int owned_stars, int required_stars) noexcept:
    """
    Calculate price movement based on star comparison.

    Per RULES.md lines 318-323:
    - diff >= 2: up 2 tiers (+2)
    - diff == 1: up 1 tier (+1)
    - diff == 0: no change (0)
    - diff == -1: down 1 tier (-1)
    - diff <= -2: down 2 tiers (-2)

    Returns:
        Number of tiers to move (-2, -1, 0, +1, +2)
    """
    cdef int diff = owned_stars - required_stars

    if diff >= 2:
        return 2
    elif diff == 1:
        return 1
    elif diff == 0:
        return 0
    elif diff == -1:
        return -1
    else:  # diff <= -2
        return -2


cdef int _find_target_index(GameState state, int current_index, int move) noexcept:
    """
    Find target price index after movement, sliding past occupied spaces.

    Per RULES.md line 324: Take target share price card (skip if in use,
    continue in same direction).

    The target is computed as a fixed offset from the current index. If that
    target space is occupied, the corp slides further in the same direction
    until a free space is found. Occupied spaces between current and target
    are ignored (they don't consume movement steps).

    Args:
        state: Game state
        current_index: Current price index
        move: Number of tiers to move (positive = up, negative = down)

    Returns:
        Target price index (0 = bankruptcy, 26 = $75 max)
    """
    if move == 0:
        return current_index

    cdef int target = current_index + move
    cdef int direction = 1 if move > 0 else -1

    # Bounds checking
    if target <= 0:
        return 0  # Bankruptcy
    if target >= 26:
        return 26  # Max price ($75), multiple corps can share

    # If target is occupied, slide further in the same direction
    while not market_module.MARKET.is_space_available(state, target):
        target += direction
        if target <= 0:
            return 0
        if target >= 26:
            return 26

    return target


cdef void _adjust_share_price(GameState state, int corp_id) noexcept:
    """
    Adjust corporation share price based on owned vs required stars.

    Per RULES.md lines 311-327:
    1. Calculate required stars for current price/shares
    2. Calculate owned stars (companies + cash/10 + SI bonus)
    3. Compare to determine movement
    4. Move to target, skipping occupied spaces
    5. If target is 0: Go Bankrupt
    6. If no higher card when rising: take no card (price = $75)

    Args:
        state: Game state
        corp_id: Corporation to adjust
    """
    cdef int current_index = corp_module.CORPS[corp_id].get_price_index(state)
    cdef int issued_shares = corp_module.CORPS[corp_id].get_issued_shares(state)
    cdef int owned_stars = corp_module.CORPS[corp_id].get_stars(state)
    cdef int required_stars = get_required_stars(current_index, issued_shares)
    cdef int move = _calculate_price_move(owned_stars, required_stars)
    cdef int target_index

    if move == 0:
        return  # No change

    # Find target index (skipping occupied spaces)
    target_index = _find_target_index(state, current_index, move)

    # Check for bankruptcy (price drops to 0)
    if target_index == 0:
        corp_module.CORPS[corp_id].go_bankrupt(state)
        return

    # Free old space (unless at $75 which can be shared)
    if current_index < 26:
        market_module.MARKET.set_space_available(state, current_index, True)

    # Take new space (unless at $75)
    if target_index < 26:
        market_module.MARKET.set_space_available(state, target_index, False)

    # Update corporation price
    corp_module.CORPS[corp_id].set_price_index(state, target_index)


cdef void _process_receivership_corp(GameState state, int corp_id) noexcept:
    """
    Auto-process a receivership corporation.

    Receivership corps pay $0 dividend and adjust price normally.
    No player decision needed.

    Args:
        state: Game state
        corp_id: Receivership corp to process
    """
    # Pay $0 dividend (no-op, but clear any acquisition proceeds)
    # _pay_dividends(state, corp_id, 0)  # No payment needed

    # Adjust share price based on stars
    _adjust_share_price(state, corp_id)

    # Mark as processed
    turn_module.TURN.set_dividend_remaining(state, corp_id, False)


cdef void _compute_dividend_impacts(GameState state, int corp_id) noexcept:
    """
    Precompute price index impact for each valid dividend level.

    For each level 0..max_dividend, simulates:
    1. New corp cash after paying level * issued_shares
    2. New cash stars from reduced cash
    3. New total stars → price move → target index via _find_target_index
    4. Stores (target_index - current_index) normalized by IMPACT_DIVISOR

    Invalid levels (beyond what the corp can pay) are left at 0.0.
    """
    cdef int current_index = corp_module.CORPS[corp_id].get_price_index(state)
    cdef int issued_shares = corp_module.CORPS[corp_id].get_issued_shares(state)
    cdef int corp_cash = corp_module.CORPS[corp_id].get_cash(state)
    cdef int current_stars = corp_module.CORPS[corp_id].get_stars(state)
    cdef int required_stars = get_required_stars(current_index, issued_shares)

    # Decompose current stars into non-cash component
    cdef int current_cash_stars = corp_cash // 10 if corp_cash > 0 else 0
    cdef int si_bonus = 2 if corp_id == <int>CorpIndices.CORP_SI else 0
    cdef int base_stars = current_stars - current_cash_stars - si_bonus

    # Compute max valid dividend level (same logic as action mask)
    cdef int card_max = get_max_dividend(current_index)
    cdef int afford_max
    if issued_shares > 0:
        afford_max = corp_cash // issued_shares
    else:
        afford_max = 0
    cdef int max_div = card_max if card_max < afford_max else afford_max
    if max_div >= <int>GameConstants.MAX_DIVIDEND:
        max_div = <int>GameConstants.MAX_DIVIDEND - 1

    cdef int level, new_cash, new_cash_stars, new_total_stars, move, target_index, impact

    # Clear all slots first (invalid levels stay at 0.0)
    turn_module.TURN.clear_dividend_impacts(state)

    for level in range(max_div + 1):
        new_cash = corp_cash - (level * issued_shares)
        new_cash_stars = new_cash // 10 if new_cash > 0 else 0
        new_total_stars = base_stars + new_cash_stars + si_bonus
        move = _calculate_price_move(new_total_stars, required_stars)
        target_index = _find_target_index(state, current_index, move)
        impact = target_index - current_index
        turn_module.TURN.set_dividend_impact(state, level, impact)


cdef void _transition_out_of_dividends(GameState state) noexcept:
    """
    Transition out of DIVIDENDS phase.

    Always transitions to END_CARD - let it handle game-over checks.
    """
    # Clear dividend impact and corp
    turn_module.TURN.clear_dividend_impacts(state)
    turn_module.TURN.clear_dividend_corp(state)
    state.clear_active_corp()

    # Transition to END_CARD phase (handles game-over logic)
    turn_module.TURN.set_phase(state, PHASE_END_CARD)


cdef void _advance_to_next_corp(GameState state) noexcept:
    """
    Find next corp to process or transition out of phase.

    Auto-processes receivership corps in a loop until a player-controlled
    corp is found or all corps are done.
    """
    cdef int corp_id, president_id

    while True:
        corp_id = _find_next_dividend_corp(state)

        if corp_id < 0:
            # No more corps to process
            _transition_out_of_dividends(state)
            return

        # Check if receivership (no president)
        if corp_module.CORPS[corp_id].is_in_receivership(state):
            _process_receivership_corp(state, corp_id)
            continue  # Loop to find next

        # Player-controlled corp - update net worths before presenting decision
        # (catches INCOME cash changes, receivership price adjustments, prior dividends)
        player_module.update_all_net_worths(state)

        # Set up for player decision
        turn_module.TURN.set_dividend_corp(state, corp_id)
        state.set_active_corp(corp_id)
        _compute_dividend_impacts(state, corp_id)
        president_id = corp_module.CORPS[corp_id].get_president_id(state)
        state._set_active_player(president_id)
        return


# =============================================================================
# ACTION HANDLER
# =============================================================================

cdef int apply_dividend_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply DIVIDENDS phase player action.

    Action type: ACTION_DIVIDEND
    Amount encoded in info.amount (0-25 per share)

    Steps:
    1. Pay dividends to shareholders
    2. Adjust share price
    3. Mark corp as processed
    4. Advance to next corp

    Returns: 0=success, 1=invalid
    """
    cdef int corp_id = turn_module.TURN.get_dividend_corp(state)
    cdef int amount = info.amount

    if corp_id < 0:
        return 1  # No active corp

    if info.action_type != ACTION_DIVIDEND:
        return 1  # Wrong action type

    # Validate amount (should already be constrained by mask, but double-check)
    if amount < 0 or amount >= <int>GameConstants.MAX_DIVIDEND:
        return 1

    # Pay dividends (stars auto-updated via set_cash)
    _pay_dividends(state, corp_id, amount)

    # Adjust share price
    _adjust_share_price(state, corp_id)

    # Mark as processed
    turn_module.TURN.set_dividend_remaining(state, corp_id, False)

    # Advance to next corp
    _advance_to_next_corp(state)

    return 0


# =============================================================================
# PHASE ENTRY POINT
# =============================================================================

cpdef void setup_dividends_phase(GameState state):
    """
    Initialize DIVIDENDS phase.

    Called from INCOME phase transition.
    Sets up dividend_remaining flags and advances to first corp.
    """
    # Initialize remaining flags for all active corps
    _init_dividend_remaining(state)

    # Clear any previous dividend corp
    turn_module.TURN.clear_dividend_corp(state)

    # Find and set up first corp (or transition out if none)
    _advance_to_next_corp(state)


# =============================================================================
# PYTHON WRAPPERS (for testing)
# =============================================================================

def setup_dividends_phase_py(GameState state):
    """Python wrapper for setup_dividends_phase."""
    setup_dividends_phase(state)


def apply_dividend_action_py(GameState state, int amount):
    """Python wrapper for apply_dividend_action."""
    cdef ActionInfo info
    info.action_type = ACTION_DIVIDEND
    info.amount = amount
    return apply_dividend_action(state, &info)


def find_next_dividend_corp_py(GameState state):
    """Python wrapper for _find_next_dividend_corp."""
    return _find_next_dividend_corp(state)


def calculate_price_move_py(int owned_stars, int required_stars):
    """Python wrapper for _calculate_price_move."""
    return _calculate_price_move(owned_stars, required_stars)


def find_target_index_py(GameState state, int current_index, int move):
    """Python wrapper for _find_target_index."""
    return _find_target_index(state, current_index, move)


def compute_dividend_impacts_py(GameState state, int corp_id):
    """Python wrapper for _compute_dividend_impacts."""
    _compute_dividend_impacts(state, corp_id)
