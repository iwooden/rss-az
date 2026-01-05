# cython: language_level=3
"""
Game driver declarations.

Provides the main entry point for applying actions to game state.
"""

from state cimport GameState
from actions cimport ActionLayout, ActionInfo

# Phase handler imports
from phases.invest cimport InvestPhase
from phases.acquisition cimport AcquisitionPhase
from phases.closing cimport ClosingPhase
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
    # Configuration
    cdef int _num_players
    cdef ActionLayout _layout

    # Debug mode
    cdef public bint debug
    cdef public list _history

    # Cached phase handlers
    cdef InvestPhase _invest
    cdef AcquisitionPhase _acquisition
    cdef ClosingPhase _closing
    cdef DividendsPhase _dividends
    cdef IssuePhase _issue
    cdef IPOPhase _ipo
    cdef WrapUpPhase _wrapup
    cdef IncomePhase _income
    cdef EndCardPhase _endcard

    # Main entry point
    cpdef void apply_action(self, GameState state, int action_idx)

    # Debug methods
    cpdef void enable_debug(self)
    cpdef void disable_debug(self)
    cpdef void clear_history(self)
    cpdef list get_history(self)
    cpdef str dump_history(self)

    # Internal dispatch methods
    cdef void _dispatch_action(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_invest(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_bid(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_acquisition(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_closing(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_dividends(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_issue(self, GameState state, ActionInfo* info) noexcept
    cdef void _dispatch_ipo(self, GameState state, ActionInfo* info) noexcept

    # Automatic phase handling
    cdef void _run_automatic_phases(self, GameState state) noexcept


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

# Convenience function for getting/creating driver
cpdef GameDriver get_driver(int num_players)

# Direct apply function (creates/reuses driver internally)
cpdef void apply_action(GameState state, int action_idx)
