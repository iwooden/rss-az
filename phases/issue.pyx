# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
ISSUE_SHARES phase: Corporations may issue one share from their unissued stack.

DESIGN: Processing Order and Receivership Auto-Handling
========================================================
Processing order: Descending share price (highest first).
For each active corporation:
1. President decides: Issue or Pass
2. If Issue: Execute Issue One Share procedure
3. Mark share price card as horizontal (implicit, no state needed)

Receivership corporations:
- Auto-processed in the loop (no player decision)
- MUST issue if they have unissued shares
- Auto-pass if no unissued shares

Stock Masters (CORP_SM) special ability:
- Share price does NOT change when issuing
- Receives current share price as proceeds

Normal corporations:
- Price drops first (find next lower available space)
- Receives NEW (lower) price as proceeds
- If price drops to 0: Go Bankrupt

Phase transitions:
- After all corps processed -> IPO phase
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases, CorpIndices,
    PHASE_IPO, get_market_price
)
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_ISSUE
from entities import turn as turn_module
from entities import corp as corp_module
from entities import player as player_module
from entities import market as market_module
from phases.ipo cimport setup_ipo_phase


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _init_issue_remaining(GameState state) noexcept:
    """
    Set issue_remaining flags for all active corporations.

    Called at phase entry to mark which corps need processing.
    """
    cdef int corp_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[corp_id].is_active(state):
            turn_module.TURN.set_issue_remaining(state, corp_id, True)
        else:
            turn_module.TURN.set_issue_remaining(state, corp_id, False)


cdef int _find_next_issue_corp(GameState state) noexcept:
    """
    Find the next corporation to process issue shares.

    Processing order: Descending share price (highest price_index first).
    Only considers corps with issue_remaining flag set.

    Returns:
        corp_id of next corp to process, or -1 if none remaining
    """
    cdef int corp_id, price_index
    cdef int best_corp = -1
    cdef int best_price_index = -1

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not turn_module.TURN.is_issue_remaining(state, corp_id):
            continue
        if not corp_module.CORPS[corp_id].is_active(state):
            continue

        price_index = corp_module.CORPS[corp_id].get_price_index(state)
        if price_index > best_price_index:
            best_price_index = price_index
            best_corp = corp_id

    return best_corp


cdef void _process_issue_share(GameState state, int corp_id) noexcept:
    """
    Execute Issue One Share procedure for a corporation.

    Per RULES.md Issue One Share:
    - Same as Sell One Share except:
      - Acting entity is corporation (not player)
      - Share comes from unissued stack (turn face up)
      - Stock Masters special: price does NOT change; receives current price

    Steps:
    1. Get current price index and unissued shares
    2. If Stock Masters: proceeds = current price (no movement)
    3. Else: Find next lower space, move price, proceeds = new price
    4. If price drops to 0: Go Bankrupt
    5. Transfer share: unissued--, issued++, bank_shares++
    6. Pay corp the proceeds

    Args:
        state: Game state
        corp_id: Corporation issuing the share
    """
    cdef int current_index = corp_module.CORPS[corp_id].get_price_index(state)
    cdef int unissued = corp_module.CORPS[corp_id].get_unissued_shares(state)
    cdef int issued = corp_module.CORPS[corp_id].get_issued_shares(state)
    cdef int bank_shares = corp_module.CORPS[corp_id].get_bank_shares(state)
    cdef int new_index, proceeds

    if unissued <= 0:
        return  # No shares to issue

    # Stock Masters special ability: price doesn't change
    if corp_id == CorpIndices.CORP_SM:
        proceeds = get_market_price(current_index)
        # No price movement
    else:
        # Normal corp: find next lower available space
        new_index = market_module.MARKET.find_next_lower_space(state, current_index)

        # Check for bankruptcy (price drops to 0)
        if new_index == 0:
            corp_module.CORPS[corp_id].go_bankrupt(state)
            return

        # Move price: free old space, take new space
        market_module.MARKET.set_space_available(state, current_index, True)
        market_module.MARKET.set_space_available(state, new_index, False)
        corp_module.CORPS[corp_id].set_price_index(state, new_index)

        # Proceeds are the NEW (lower) price
        proceeds = get_market_price(new_index)

    # Transfer share: unissued -> issued -> bank
    corp_module.CORPS[corp_id].set_unissued_shares(state, unissued - 1)
    corp_module.CORPS[corp_id].set_issued_shares(state, issued + 1)
    corp_module.CORPS[corp_id].set_bank_shares(state, bank_shares + 1)

    # Pay the corporation
    corp_module.CORPS[corp_id].add_cash(state, proceeds)


cdef void _process_receivership_corp(GameState state, int corp_id) noexcept:
    """
    Auto-process a receivership corporation.

    Receivership corps MUST issue if they have unissued shares.
    If no unissued shares, they auto-pass.

    Args:
        state: Game state
        corp_id: Receivership corp to process
    """
    cdef int unissued = corp_module.CORPS[corp_id].get_unissued_shares(state)

    if unissued > 0:
        # Must issue
        _process_issue_share(state, corp_id)

    # Mark as processed (whether issued or passed)
    turn_module.TURN.set_issue_remaining(state, corp_id, False)


cdef void _transition_out_of_issue(GameState state) noexcept:
    """
    Transition out of ISSUE_SHARES phase.

    Transitions to IPO phase for player-owned companies to form corporations.
    """
    # Clear issue corp
    turn_module.TURN.clear_issue_corp(state)

    # Transition to IPO phase
    turn_module.TURN.set_phase(state, PHASE_IPO)
    setup_ipo_phase(state)


cdef void _advance_to_next_corp(GameState state) noexcept:
    """
    Find next corp to process or transition out of phase.

    Auto-processes receivership corps in a loop until a player-controlled
    corp is found or all corps are done.
    """
    cdef int corp_id, president_id

    while True:
        corp_id = _find_next_issue_corp(state)

        if corp_id < 0:
            # No more corps to process
            _transition_out_of_issue(state)
            return

        # Check if receivership (no president)
        if corp_module.CORPS[corp_id].is_in_receivership(state):
            _process_receivership_corp(state, corp_id)
            continue  # Loop to find next

        # Player-controlled corp - update net worths before presenting decision
        # (catches prior issue price drops and receivership auto-issues)
        player_module.update_all_net_worths(state)

        # Set up for player decision
        turn_module.TURN.set_issue_corp(state, corp_id)
        president_id = corp_module.CORPS[corp_id].get_president_id(state)
        state._set_active_player(president_id)
        return


# =============================================================================
# ACTION HANDLER
# =============================================================================

cdef int apply_issue_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply ISSUE_SHARES phase player action.

    Action types:
    - ACTION_PASS: President declines to issue
    - ACTION_ISSUE: President issues one share

    Steps:
    1. If ISSUE: Execute Issue One Share procedure
    2. Mark corp as processed
    3. Advance to next corp

    Returns: 0=success, 1=invalid
    """
    cdef int corp_id = turn_module.TURN.get_issue_corp(state)

    if corp_id < 0:
        return 1  # No active corp

    if info.action_type == ACTION_ISSUE:
        # Validate: must have unissued shares
        if corp_module.CORPS[corp_id].get_unissued_shares(state) <= 0:
            return 1  # Can't issue without shares
        _process_issue_share(state, corp_id)
    elif info.action_type != ACTION_PASS:
        return 1  # Invalid action type

    # Mark as processed
    turn_module.TURN.set_issue_remaining(state, corp_id, False)

    # Advance to next corp
    _advance_to_next_corp(state)

    return 0


# =============================================================================
# PHASE ENTRY POINT
# =============================================================================

cpdef void setup_issue_phase(GameState state):
    """
    Initialize ISSUE_SHARES phase.

    Called from END_CARD phase transition.
    Sets up issue_remaining flags and advances to first corp.
    """
    # Initialize remaining flags for all active corps
    _init_issue_remaining(state)

    # Clear any previous issue corp
    turn_module.TURN.clear_issue_corp(state)

    # Find and set up first corp (or transition out if none)
    _advance_to_next_corp(state)


# =============================================================================
# PYTHON WRAPPERS (for testing)
# =============================================================================

def setup_issue_phase_py(GameState state):
    """Python wrapper for setup_issue_phase."""
    setup_issue_phase(state)


def apply_issue_action_py(GameState state, bint issue):
    """
    Python wrapper for apply_issue_action.

    Args:
        state: Game state
        issue: True to issue, False to pass
    """
    cdef ActionInfo info
    info.action_type = ACTION_ISSUE if issue else ACTION_PASS
    return apply_issue_action(state, &info)


def find_next_issue_corp_py(GameState state):
    """Python wrapper for _find_next_issue_corp."""
    return _find_next_issue_corp(state)


def process_issue_share_py(GameState state, int corp_id):
    """Python wrapper for _process_issue_share."""
    _process_issue_share(state, corp_id)
