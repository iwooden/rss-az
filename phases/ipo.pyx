# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
IPO phase: Player-owned companies may form corporations.

DESIGN: Processing Order and Form Corporation
==============================================
Processing order: Descending face value (highest first).
For each player-owned company:
1. Company owner decides: Form Corporation (choose corp + par) or Pass
2. If IPO: Execute Form Corporation procedure

Form Corporation procedure:
1. Player selects available corp charter and valid par price for company's color
2. Share distribution based on face value vs par price:
   - FV > par: player gets 2 shares, bank gets 2 shares
   - FV <= par: player gets 1 share, bank gets 1 share
3. Player pays corp: (player_shares * par_price) - face_value
4. Bank pays corp: bank_shares * par_price
5. Company becomes subsidiary of new corporation

Phase transitions:
- After all companies processed -> INVEST (new turn)
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants, GamePhases,
    PHASE_INVEST,
    get_company_face_value, get_company_stars,
    get_par_price, get_par_index_for_slot, get_market_index,
    get_corp_share_count
)
from core.actions cimport ActionInfo, ACTION_PASS, ACTION_IPO
from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module
from entities import market as market_module
from entities.company cimport LOC_PLAYER


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

cdef void _init_ipo_remaining(GameState state) noexcept:
    """
    Set ipo_remaining flags for all player-owned companies.

    Called at phase entry to mark which companies need processing.
    Only player-owned companies are eligible for IPO.
    """
    cdef int company_id, location

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        location = company_module.COMPANIES[company_id].get_location(state)
        if location == LOC_PLAYER:
            turn_module.TURN.set_ipo_remaining(state, company_id, True)
        else:
            turn_module.TURN.set_ipo_remaining(state, company_id, False)


cdef int _find_next_ipo_company(GameState state) noexcept:
    """
    Find the next company to process for IPO.

    Processing order: Descending face value (highest first).
    Only considers companies with ipo_remaining flag set.

    Returns:
        company_id of next company to process, or -1 if none remaining
    """
    cdef int company_id, face_value
    cdef int best_company = -1
    cdef int best_face_value = -1

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if not turn_module.TURN.is_ipo_remaining(state, company_id):
            continue

        # Verify still player-owned (ownership can change mid-phase theoretically)
        if company_module.COMPANIES[company_id].get_location(state) != LOC_PLAYER:
            turn_module.TURN.set_ipo_remaining(state, company_id, False)
            continue

        face_value = get_company_face_value(company_id)
        if face_value > best_face_value:
            best_face_value = face_value
            best_company = company_id

    return best_company


cdef void _process_ipo(GameState state, int corp_id, int par_slot) noexcept:
    """
    Execute Form Corporation procedure.

    Per RULES.md Form Corporation:
    1. Select unused charter card (corp_id)
    2. Select valid par price for company's color (par_slot)
    3. Share distribution:
       - FV > par: player 2, bank 2
       - FV <= par: player 1, bank 1
    4. Player pays: (player_shares * par_price) - face_value
    5. Bank pays: bank_shares * par_price (to corp)
    6. Company becomes subsidiary

    Args:
        state: Game state
        corp_id: Selected corporation charter
        par_slot: Selected par price slot for this star tier
    """
    # Get current IPO company info
    cdef int company_id = turn_module.TURN.get_ipo_company(state)
    cdef int player_id = company_module.COMPANIES[company_id].get_owner_id(state)
    cdef int star_tier = get_company_stars(company_id)
    cdef int face_value = get_company_face_value(company_id)

    # Get par price from slot
    cdef int par_index = get_par_index_for_slot(star_tier, par_slot)
    cdef int par_price = get_par_price(par_index)
    cdef int market_index = get_market_index(par_price)

    # Calculate share distribution (player and bank each get float_shares)
    cdef int float_shares
    if face_value > par_price:
        float_shares = 2
    else:
        float_shares = 1

    # Float the corporation (transfers company, sets up corp, gives player shares)
    corp_module.CORPS[corp_id].float_corp(state, player_id, company_id, market_index, float_shares)

    # Phase-specific: calculate and apply payments
    cdef int player_payment = (float_shares * par_price) - face_value
    cdef int bank_payment = float_shares * par_price
    cdef int corp_cash = player_payment + bank_payment

    corp_module.CORPS[corp_id].set_cash(state, corp_cash)
    player_module.PLAYERS[player_id].add_cash(state, -player_payment)

    # Clear from remaining
    turn_module.TURN.set_ipo_remaining(state, company_id, False)


cdef void _transition_out_of_ipo(GameState state) noexcept:
    """
    Transition out of IPO phase.

    Completes the turn cycle by incrementing turn number and transitioning to INVEST.

    NOTE: Roundtrip clearing happens in INVEST phase (before WRAP_UP transition),
    NOT here. Per RULES.md: Roundtrip info only relevant in INVEST phase -
    clearing it elsewhere pollutes state vector for model.
    """
    # Clear IPO company and active company
    turn_module.TURN.clear_ipo_company(state)
    state.clear_active_company()

    # Increment turn number (end of turn bookkeeping)
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Set active player to position 0 in turn order (start of new turn)
    cdef int first_player = turn_module.TURN.find_player_at_position(state, 0)
    state._set_active_player(first_player)

    # Update net worths before INVEST (catches dividend/issue price changes
    # that happened during the last corp processed in those phases)
    player_module.update_all_net_worths(state)

    # Transition to INVEST phase (start new turn)
    turn_module.TURN.set_phase(state, PHASE_INVEST)

    # Compute buy/sell impacts for the first player
    state._populate_invest_impacts()


cdef void _advance_to_next_company(GameState state) noexcept:
    """
    Find next company to process or transition out of phase.

    Sets up state for next company's owner to make IPO decision,
    or transitions to next phase if no companies remain.
    """
    cdef int company_id = _find_next_ipo_company(state)
    cdef int player_id

    if company_id < 0:
        # No more companies to process
        _transition_out_of_ipo(state)
        return

    # Update net worths before presenting decision
    # (catches INCOME cash changes, prior IPO payments, and share acquisitions)
    player_module.update_all_net_worths(state)

    # Set up for company owner's decision
    turn_module.TURN.set_ipo_company(state, company_id)
    state.set_active_company(company_id)
    player_id = company_module.COMPANIES[company_id].get_owner_id(state)
    state._set_active_player(player_id)


# =============================================================================
# ACTION HANDLER
# =============================================================================

cdef int apply_ipo_action(GameState state, ActionInfo* info) noexcept:
    """
    Apply IPO phase player action.

    Action types:
    - ACTION_PASS: Owner declines to IPO this company
    - ACTION_IPO: Owner forms corporation (corp_id, par_slot)

    Steps:
    1. If IPO: Execute Form Corporation procedure
    2. Mark company as processed
    3. Advance to next company

    Returns: 0=success, 1=invalid
    """
    cdef int company_id = turn_module.TURN.get_ipo_company(state)
    cdef int star_tier
    cdef int par_index

    if company_id < 0:
        return 1  # No active company

    if info.action_type == ACTION_IPO:
        # Validate corp is available (not already active)
        if corp_module.CORPS[info.corp_id].is_active(state):
            return 1  # Corp already in use

        # Validate par_index bounds (memory safety)
        star_tier = get_company_stars(company_id)
        par_index = get_par_index_for_slot(star_tier, info.slot)
        if par_index < 0 or par_index >= <int>GameConstants.NUM_PAR_PRICES:
            return 1  # Invalid par slot

        # Process the IPO
        _process_ipo(state, info.corp_id, info.slot)
    elif info.action_type == ACTION_PASS:
        # Mark as processed
        turn_module.TURN.set_ipo_remaining(state, company_id, False)
    else:
        return 1  # Invalid action type

    # Advance to next company
    _advance_to_next_company(state)

    return 0


# =============================================================================
# PHASE ENTRY POINT
# =============================================================================

cpdef void setup_ipo_phase(GameState state):
    """
    Initialize IPO phase.

    Called from ISSUE_SHARES phase transition.
    Sets up ipo_remaining flags and advances to first company.
    """
    # Initialize remaining flags for all player-owned companies
    _init_ipo_remaining(state)

    # Clear any previous IPO company
    turn_module.TURN.clear_ipo_company(state)

    # Find and set up first company (or transition out if none)
    _advance_to_next_company(state)


# =============================================================================
# PYTHON WRAPPERS (for testing)
# =============================================================================

def setup_ipo_phase_py(GameState state):
    """Python wrapper for setup_ipo_phase."""
    setup_ipo_phase(state)


def apply_ipo_action_py(GameState state, int corp_id, int par_slot):
    """
    Python wrapper for apply_ipo_action with IPO action.

    Args:
        state: Game state
        corp_id: Corporation to form
        par_slot: Par price slot for company's star tier
    """
    cdef ActionInfo info
    info.action_type = ACTION_IPO
    info.corp_id = corp_id
    info.slot = par_slot
    return apply_ipo_action(state, &info)


def apply_ipo_pass_py(GameState state):
    """Python wrapper for apply_ipo_action with PASS action."""
    cdef ActionInfo info
    info.action_type = ACTION_PASS
    return apply_ipo_action(state, &info)


def find_next_ipo_company_py(GameState state):
    """Python wrapper for _find_next_ipo_company."""
    return _find_next_ipo_company(state)


def process_ipo_py(GameState state, int corp_id, int par_slot):
    """Python wrapper for _process_ipo."""
    _process_ipo(state, corp_id, par_slot)
