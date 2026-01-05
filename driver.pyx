# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Game driver implementation.

Main entry point for applying actions to game state.
Handles action decoding, phase dispatch, and automatic phase transitions.
"""

cimport cython

from state cimport (
    GameState,
    PHASE_INVEST, PHASE_BID_IN_AUCTION, PHASE_WRAP_UP, PHASE_ACQUISITION,
    PHASE_CLOSING, PHASE_INCOME, PHASE_DIVIDENDS, PHASE_END_CARD,
    PHASE_ISSUE_SHARES, PHASE_IPO, PHASE_GAME_OVER
)
from actions cimport (
    ActionLayout, ActionInfo,
    compute_action_layout, decode_action,
    ACTION_PASS, ACTION_AUCTION, ACTION_BUY_SHARE, ACTION_SELL_SHARE,
    ACTION_LEAVE_AUCTION, ACTION_RAISE_BID,
    ACTION_ACQ_PRICE, ACTION_ACQ_FI_HIGH, ACTION_ACQ_FI_FACE,
    ACTION_CLOSE, ACTION_DIVIDEND, ACTION_ISSUE, ACTION_IPO
)
from data cimport get_par_index_for_slot, get_company_stars
from helpers.company cimport get_auction_company_for_slot

# Phase handlers
from phases.invest cimport InvestPhase
from phases.acquisition cimport (
    AcquisitionPhase, ACQ_ACTION_PASS, ACQ_FI_ACTION_BUY, ACQ_FI_ACTION_PASS
)
from phases.closing cimport (
    ClosingPhase, CLOSING_ACTION_CLOSE, CLOSING_ACTION_PASS
)
from phases.dividends cimport DividendsPhase
from phases.issue cimport IssuePhase
from phases.ipo cimport IPOPhase
from phases.wrapup cimport WrapUpPhase
from phases.income cimport IncomePhase
from phases.endcard cimport EndCardPhase


# =============================================================================
# DRIVER CLASS
# =============================================================================

cdef class GameDriver:
    """
    Main game driver that handles action application.

    Instantiated once per player count, caches phase handlers.
    """

    def __cinit__(self, int num_players):
        self._num_players = num_players
        self._layout = compute_action_layout(num_players)

        # Create phase handlers
        self._invest = InvestPhase(num_players)
        self._acquisition = AcquisitionPhase(num_players)
        self._closing = ClosingPhase(num_players)
        self._dividends = DividendsPhase(num_players)
        self._issue = IssuePhase(num_players)
        self._ipo = IPOPhase(num_players)
        self._wrapup = WrapUpPhase(num_players)
        self._income = IncomePhase(num_players)
        self._endcard = EndCardPhase(num_players)

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    cpdef void apply_action(self, GameState state, int action_idx):
        """
        Apply an action to the game state.

        Args:
            state: Current game state
            action_idx: Action index from NN output

        This function:
        1. Decodes the action index to ActionInfo
        2. Dispatches to the appropriate phase handler
        3. Runs any automatic phases until player input needed
        """
        cdef ActionInfo info = decode_action(&self._layout, action_idx)

        # Dispatch the action
        self._dispatch_action(state, &info)

        # Run automatic phases
        self._run_automatic_phases(state)

    # =========================================================================
    # ACTION DISPATCH
    # =========================================================================

    cdef void _dispatch_action(self, GameState state, ActionInfo* info) noexcept:
        """Dispatch action to appropriate phase handler."""
        cdef int phase = info.phase

        if phase == PHASE_INVEST:
            self._dispatch_invest(state, info)
        elif phase == PHASE_BID_IN_AUCTION:
            self._dispatch_bid(state, info)
        elif phase == PHASE_ACQUISITION:
            self._dispatch_acquisition(state, info)
        elif phase == PHASE_CLOSING:
            self._dispatch_closing(state, info)
        elif phase == PHASE_DIVIDENDS:
            self._dispatch_dividends(state, info)
        elif phase == PHASE_ISSUE_SHARES:
            self._dispatch_issue(state, info)
        elif phase == PHASE_IPO:
            self._dispatch_ipo(state, info)

    cdef void _dispatch_invest(self, GameState state, ActionInfo* info) noexcept:
        """Handle INVEST phase actions."""
        cdef int action_type = info.action_type
        cdef int company_id

        if action_type == ACTION_PASS:
            self._invest.do_pass(state)

        elif action_type == ACTION_AUCTION:
            # Map slot to company_id
            company_id = get_auction_company_for_slot(state, info.slot)
            if company_id >= 0:
                # info.amount holds bid_offset
                self._invest.do_start_auction(state, company_id, info.amount)

        elif action_type == ACTION_BUY_SHARE:
            self._invest.do_buy_share(state, info.corp_id)

        elif action_type == ACTION_SELL_SHARE:
            self._invest.do_sell_share(state, info.corp_id)

    cdef void _dispatch_bid(self, GameState state, ActionInfo* info) noexcept:
        """Handle BID_IN_AUCTION phase actions."""
        cdef int action_type = info.action_type

        if action_type == ACTION_LEAVE_AUCTION:
            self._invest.do_leave_auction(state)

        elif action_type == ACTION_RAISE_BID:
            # bid_offset in amount field: represents new_bid = face + amount + 1
            # (since offset 0 = face+1, offset 18 = face+19)
            self._invest.do_raise_bid(state, info.amount + 1)

    cdef void _dispatch_acquisition(self, GameState state, ActionInfo* info) noexcept:
        """Handle ACQUISITION phase actions."""
        cdef int action_type = info.action_type
        cdef bint is_fi = state.is_acq_fi_offer()

        if action_type == ACTION_PASS:
            if is_fi:
                self._acquisition.do_action(state, ACQ_FI_ACTION_PASS)
            else:
                self._acquisition.do_action(state, ACQ_ACTION_PASS)

        elif action_type == ACTION_ACQ_PRICE:
            # General acquisition: price offset in amount field
            self._acquisition.do_action(state, info.amount)

        elif action_type == ACTION_ACQ_FI_HIGH:
            # FI offer: buy at high price
            self._acquisition.do_action(state, ACQ_FI_ACTION_BUY)

        elif action_type == ACTION_ACQ_FI_FACE:
            # FI offer: buy at face value (OS only)
            self._acquisition.do_action(state, ACQ_FI_ACTION_BUY)

    cdef void _dispatch_closing(self, GameState state, ActionInfo* info) noexcept:
        """Handle CLOSING phase actions."""
        cdef int action_type = info.action_type

        if action_type == ACTION_CLOSE:
            self._closing.do_action(state, CLOSING_ACTION_CLOSE)
        elif action_type == ACTION_PASS:
            self._closing.do_action(state, CLOSING_ACTION_PASS)

    cdef void _dispatch_dividends(self, GameState state, ActionInfo* info) noexcept:
        """Handle DIVIDENDS phase actions."""
        # Amount contains dividend amount (0-25)
        self._dividends.do_dividend(state, info.amount)

    cdef void _dispatch_issue(self, GameState state, ActionInfo* info) noexcept:
        """Handle ISSUE_SHARES phase actions."""
        cdef int action_type = info.action_type

        if action_type == ACTION_PASS:
            self._issue.do_pass(state)
        elif action_type == ACTION_ISSUE:
            self._issue.do_issue(state)

    cdef void _dispatch_ipo(self, GameState state, ActionInfo* info) noexcept:
        """Handle IPO phase actions."""
        cdef int action_type = info.action_type
        cdef int company_id, star_tier, par_index

        if action_type == ACTION_PASS:
            self._ipo.do_pass(state)

        elif action_type == ACTION_IPO:
            # corp_id is in info.corp_id
            # par_slot is in info.slot - needs mapping to par_index
            company_id = self._ipo.get_current_company(state)
            if company_id >= 0:
                star_tier = get_company_stars(company_id)
                par_index = get_par_index_for_slot(star_tier, info.slot)
                if par_index >= 0:
                    self._ipo.do_ipo(state, info.corp_id, par_index)

    # =========================================================================
    # AUTOMATIC PHASE HANDLING
    # =========================================================================

    cdef void _run_automatic_phases(self, GameState state) noexcept:
        """
        Run automatic phases until reaching a phase that requires player input.

        Automatic phases:
        - WRAP_UP: Player order + FI buys + reveal companies
        - INCOME: All entities collect income
        - END_CARD: Check game end, flip card if needed

        Also handles setup for phases with automatic sub-steps:
        - ACQUISITION: setup_next_offer (auto-buys for receivership)
        - CLOSING: setup_next_closeable (auto-closures)
        - DIVIDENDS: advance_to_next_corp (auto-pays for receivership)
        - ISSUE_SHARES: advance_to_next_corp (auto-issues for receivership)
        - IPO: advance_to_next_company (skips unaffordable)
        """
        cdef int phase
        cdef int iterations = 0
        cdef int MAX_ITERATIONS = 100  # Safety limit

        while iterations < MAX_ITERATIONS:
            iterations += 1
            phase = state.get_phase()

            # Game over - stop
            if phase == PHASE_GAME_OVER:
                break

            # Fully automatic phases
            if phase == PHASE_WRAP_UP:
                self._wrapup.execute(state)
                continue

            if phase == PHASE_INCOME:
                self._income.handle_income_phase(state)
                continue

            if phase == PHASE_END_CARD:
                self._endcard.handle_end_card_phase(state)
                continue

            # Phases with automatic sub-steps - check if waiting for player
            if phase == PHASE_ACQUISITION:
                if not self._acquisition.is_waiting_for_action(state):
                    # Need to setup next offer
                    if not self._acquisition.setup_next_offer(state):
                        self._acquisition.transition_to_closing(state)
                        continue
                break  # Waiting for player action

            if phase == PHASE_CLOSING:
                if not self._closing.is_waiting_for_action(state):
                    # Need to setup next closeable
                    if not self._closing.setup_next_closeable(state):
                        # No more closeables, transition to income
                        state.set_phase(PHASE_INCOME)
                        continue
                break  # Waiting for player action

            if phase == PHASE_DIVIDENDS:
                if self._dividends.get_current_corp(state) < 0:
                    # No current corp, advance to next
                    self._dividends.advance_to_next_corp(state)
                    # Check if we transitioned out
                    if state.get_phase() != PHASE_DIVIDENDS:
                        continue
                break  # Waiting for player action

            if phase == PHASE_ISSUE_SHARES:
                if self._issue.get_current_corp(state) < 0:
                    # No current corp, advance to next
                    self._issue.advance_to_next_corp(state)
                    # Check if we transitioned out
                    if state.get_phase() != PHASE_ISSUE_SHARES:
                        continue
                break  # Waiting for player action

            if phase == PHASE_IPO:
                if self._ipo.get_current_company(state) < 0:
                    # No current company, advance to next
                    self._ipo.advance_to_next_company(state)
                    # Check if we transitioned out
                    if state.get_phase() != PHASE_IPO:
                        continue
                break  # Waiting for player action

            # Phases that require player input
            if phase in (PHASE_INVEST, PHASE_BID_IN_AUCTION):
                break

            # Unknown phase - break to avoid infinite loop
            break


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

cdef dict _drivers = {}


cpdef GameDriver get_driver(int num_players):
    """Get or create a GameDriver for the given player count."""
    if num_players not in _drivers:
        _drivers[num_players] = GameDriver(num_players)
    return _drivers[num_players]


cpdef void apply_action(GameState state, int action_idx):
    """
    Apply an action to the game state.

    Convenience function that handles driver caching internally.

    Args:
        state: Current game state
        action_idx: Action index from NN output
    """
    cdef GameDriver driver = get_driver(state._num_players)
    driver.apply_action(state, action_idx)
