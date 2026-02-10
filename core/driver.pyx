# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game driver for action dispatch and legal move generation.

The GameDriver routes actions to phase-specific handlers and provides
legal move generation for the neural network.

Auto-applies forced actions iteratively until a choice is needed (2+ legal
actions) or the game ends. Ensures neural network only sees decision states.
"""

from core.state cimport GameState
from core.actions cimport (
    ActionLayout, ActionInfo, compute_action_layout, decode_action,
    get_total_actions_for_players, _fill_action_mask, _is_action_valid_in_buffer,
    _mask_buffer
)
from core.actions import get_valid_action_mask
from core.data cimport GamePhases, GameConstants, PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_GAME_OVER, PHASE_WRAP_UP, PHASE_ACQUISITION, PHASE_CLOSING, PHASE_INCOME, PHASE_DIVIDENDS, PHASE_END_CARD, PHASE_ISSUE_SHARES, PHASE_IPO
from core.driver cimport ActionStatus, STATUS_OK, STATUS_INVALID, STATUS_GAME_OVER, ForcedActionResult
from phases.invest cimport apply_invest_action
from phases.bid cimport apply_bid_action
from phases.wrap_up cimport apply_wrap_up
from phases.acquisition cimport apply_acquisition_action, _transition_to_closing
from phases.closing cimport apply_closing_auto, apply_closing_action
from phases.income cimport apply_income
from phases.dividends cimport apply_dividend_action
from phases.end_card cimport apply_end_card
from phases.issue cimport apply_issue_action
from phases.ipo cimport apply_ipo_action
from entities import turn as turn_module
from entities import company as company_module
from entities import corp as corp_module


class ForcedActionLoopError(RuntimeError):
    """Raised when forced action loop exceeds iteration limit."""
    pass


class ZeroLegalActionsError(RuntimeError):
    """Raised when zero legal actions exist outside GAME_OVER phase."""
    pass


# Maximum iterations for auto-apply loop (prevents infinite loops from bugs)
DEF MAX_FORCED_ITERATIONS = 100

# Sentinel action values for non-player phases (negative to distinguish from real actions)
DEF ACTION_WRAP_UP_SENTINEL = -100
DEF ACTION_ACQUISITION_SENTINEL = -101
DEF ACTION_CLOSING_SENTINEL = -102
DEF ACTION_INCOME_SENTINEL = -103
DEF ACTION_END_CARD_SENTINEL = -105


cdef bint _is_game_terminal(GameState state) noexcept:
    """
    Check if the game has reached a terminal state.

    Terminal state occurs when:
    1. No companies are available for auction, AND
    2. No corporations are active

    This prevents infinite INVEST->WRAP_UP->ACQUISITION->CLOSING loops when
    all companies are removed from the game.
    """
    cdef int company_id, corp_id
    cdef bint has_auction_companies = False
    cdef bint has_active_corps = False

    # Check for any companies available for auction
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].is_for_auction(state):
            has_auction_companies = True
            break

    # Check for any active corporations
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_module.CORPS[corp_id].is_active(state):
            has_active_corps = True
            break

    # Terminal if no auction companies AND no active corps
    return not has_auction_companies and not has_active_corps


cdef bint _is_non_player_phase_check(GameState state, int phase) noexcept:
    """
    Check if phase has no player actions (deterministic execution).

    ACQUISITION is a hybrid: non-player when no offers exist, player when offers exist.
    """
    if phase == PHASE_WRAP_UP:
        return True

    if phase == PHASE_ACQUISITION:
        # ACQUISITION with no active corp = no offers = non-player phase
        return turn_module.TURN.get_acq_active_corp(state) == -1

    if phase == PHASE_CLOSING:
        # CLOSING is hybrid: non-player when no offers, player when offers exist
        # closing_company == -1 means no active offer (auto-close mode)
        # closing_company >= 0 means offer active (player decision mode)
        return turn_module.TURN.get_closing_company(state) == -1

    if phase == PHASE_INCOME:
        return True

    if phase == PHASE_END_CARD:
        return True

    return False


cdef void _execute_non_player_phase(GameState state, object history):
    """Execute deterministic non-player phase and record to history."""
    cdef int phase = state.get_phase()
    cdef int sentinel

    if phase == PHASE_WRAP_UP:
        sentinel = ACTION_WRAP_UP_SENTINEL
    elif phase == PHASE_ACQUISITION:
        sentinel = ACTION_ACQUISITION_SENTINEL
    elif phase == PHASE_CLOSING:
        sentinel = ACTION_CLOSING_SENTINEL
    elif phase == PHASE_INCOME:
        sentinel = ACTION_INCOME_SENTINEL
    elif phase == PHASE_END_CARD:
        sentinel = ACTION_END_CARD_SENTINEL
    else:
        return  # Unknown non-player phase

    # Record state BEFORE execution (matches player action pattern)
    if history is not None:
        history.append((state._array.copy(), sentinel))

    # Execute phase logic
    if phase == PHASE_WRAP_UP:
        apply_wrap_up(state)
    elif phase == PHASE_ACQUISITION:
        _transition_to_closing(state)
    elif phase == PHASE_CLOSING:
        apply_closing_auto(state)
    elif phase == PHASE_INCOME:
        apply_income(state)
    elif phase == PHASE_END_CARD:
        apply_end_card(state)


cdef ForcedActionResult _check_forced_action(GameState state) noexcept:
    """
    Check legal action count and find single action if forced.

    Fills the static _mask_buffer directly without numpy allocation.

    Returns:
        ForcedActionResult with count (0, 1, or 2+) and action_idx
        - count=0: no legal actions (error condition)
        - count=1: exactly one action (forced), action_idx set
        - count=2+: multiple actions (choice needed), action_idx=-1
    """
    cdef ForcedActionResult result
    cdef int total = get_total_actions_for_players(state._num_players)
    cdef int i

    _fill_action_mask(state)

    result.action_idx = -1
    result.count = 0

    for i in range(total):
        if _mask_buffer[i] == 1.0:
            result.count += 1
            if result.count == 1:
                result.action_idx = i
            elif result.count == 2:
                result.action_idx = -1  # Not forced
                return result  # Early exit - no need to count higher

    return result


cdef class GameDriver:
    """
    Game driver for dispatching actions to phase handlers.

    Stateless singleton - all state is in the GameState object.
    Following the entity handle pattern from entities/turn.pyx.
    """

    def __cinit__(self):
        """Initialize driver (no state needed - stateless pattern)."""
        pass

    cdef int _dispatch_action(self, GameState state, int action_idx, object history):
        """
        Decode and dispatch action without validation.

        Internal helper for pre-validated actions (e.g. forced actions found by
        _check_forced_action). Skips mask generation entirely.

        Args:
            state: GameState object to modify
            action_idx: Index into action vector (already validated)
            history: Optional list to append (state.copy(), action) tuple

        Returns:
            STATUS_OK (0) if action applied successfully
            STATUS_INVALID (1) if dispatch fails
        """
        cdef ActionLayout layout = compute_action_layout(state._num_players)
        cdef ActionInfo info
        cdef int result

        # Append to history if provided (before applying action)
        if history is not None:
            history.append((state._array.copy(), action_idx))

        # Decode action
        info = decode_action(&layout, action_idx)

        # Dispatch based on current phase
        cdef int phase = state.get_phase()

        if phase == PHASE_INVEST:
            result = apply_invest_action(state, &info)
        elif phase == PHASE_BID_IN_AUCTION:
            result = apply_bid_action(state, &info)
        elif phase == PHASE_ACQUISITION:
            result = apply_acquisition_action(state, &info)
        elif phase == PHASE_CLOSING:
            result = apply_closing_action(state, &info)
        elif phase == PHASE_DIVIDENDS:
            result = apply_dividend_action(state, &info)
        elif phase == PHASE_ISSUE_SHARES:
            result = apply_issue_action(state, &info)
        elif phase == PHASE_IPO:
            result = apply_ipo_action(state, &info)
        else:
            return STATUS_INVALID

        return result

    cdef int _apply_single_action(self, GameState state, int action_idx, object history):
        """
        Apply one action without auto-continuation.

        Validates action legality via _fill_action_mask (no numpy allocation),
        then dispatches. Used for the initial user action in apply_action().

        Args:
            state: GameState object to modify
            action_idx: Index into action vector (0 to total_actions-1)
            history: Optional list to append (state.copy(), action) tuple

        Returns:
            STATUS_OK (0) if action applied successfully
            STATUS_INVALID (1) if action is invalid for current state
        """
        cdef int total_actions = get_total_actions_for_players(state._num_players)

        # Validate action index is in bounds
        if action_idx < 0 or action_idx >= total_actions:
            return STATUS_INVALID

        # Fill mask buffer and check legality (no numpy allocation)
        _fill_action_mask(state)
        if not _is_action_valid_in_buffer(action_idx, total_actions):
            return STATUS_INVALID

        return self._dispatch_action(state, action_idx, history)

    cpdef int apply_action(self, GameState state, int action_idx, object history=None):
        """
        Apply action to game state, auto-applying forced actions until choice needed.

        This method applies the user's action, then iteratively auto-applies any
        forced actions (states with exactly one legal action) until a decision
        point is reached (2+ legal actions) or the game ends.

        Args:
            state: GameState object to modify
            action_idx: Index into action vector (0 to total_actions-1)
            history: Optional list to collect (state.copy(), action) tuples

        Returns:
            STATUS_OK (0) if action applied successfully (2+ choices now available)
            STATUS_INVALID (1) if action is invalid for current state
            STATUS_GAME_OVER (2) if game ended after this action

        Raises:
            ForcedActionLoopError: If auto-apply exceeds 100 iterations
            ZeroLegalActionsError: If zero legal actions exist outside GAME_OVER
        """
        cdef int result, iterations
        cdef ForcedActionResult forced

        # Apply the user's action
        result = self._apply_single_action(state, action_idx, history)
        if result != STATUS_OK:
            return result
        if state.get_phase() == PHASE_GAME_OVER:
            return STATUS_GAME_OVER

        # Auto-apply forced actions
        iterations = 0
        while iterations < MAX_FORCED_ITERATIONS:
            forced = _check_forced_action(state)

            if forced.count == 0:
                # Check if this is a non-player phase (0 actions is valid)
                if _is_non_player_phase_check(state, state.get_phase()):
                    _execute_non_player_phase(state, history)
                    if state.get_phase() == PHASE_GAME_OVER:
                        return STATUS_GAME_OVER
                    iterations += 1
                    continue  # Re-check after phase execution
                raise ZeroLegalActionsError("Zero legal actions in non-terminal state")

            if forced.count >= 2:
                return STATUS_OK  # Choice needed - return to caller

            # Exactly 1 action - auto-apply (already validated by _check_forced_action)
            result = self._dispatch_action(state, forced.action_idx, history)
            if result != STATUS_OK:
                return result
            if state.get_phase() == PHASE_GAME_OVER:
                return STATUS_GAME_OVER

            iterations += 1

        raise ForcedActionLoopError(f"Forced action loop exceeded {MAX_FORCED_ITERATIONS} iterations")

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

# Python-accessible status code constants
STATUS_OK_PY = STATUS_OK
STATUS_INVALID_PY = STATUS_INVALID
STATUS_GAME_OVER_PY = STATUS_GAME_OVER
