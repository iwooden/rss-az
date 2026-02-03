# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Foreign Investor entity implementation.

Provides clean getter/setter access to the Foreign Investor state in the game
state vector. The FI is a special entity that buys companies at high price
and holds them until corporations acquire them.
"""

from libc.math cimport lround

from core.state cimport GameState, StateLayout
from core.data cimport GameConstants, CASH_DIVISOR


cdef class ForeignInvestor:
    """
    Entity handle for accessing Foreign Investor state.

    There is only one FI instance, created at module load.
    Offsets are computed on first access to a GameState via initialize().
    All methods take GameState as first argument for stateless operation.
    """

    def __cinit__(self):
        self._cash_offset = 0
        self._owned_companies_offset = 0

    cpdef void initialize(self, GameState state):
        """
        Initialize offsets from state layout. Call once when starting a new game.

        This must be called before using any other methods on this ForeignInvestor instance.
        """
        cdef StateLayout layout = state._layout

        # FI layout: [cash][owned_companies x 36]
        self._cash_offset = layout.fi_offset
        self._owned_companies_offset = layout.fi_offset + 1

    # =========================================================================
    # CASH OPERATIONS
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Get FI's cash (integer dollars)."""
        return <int>(state._data[self._cash_offset] * CASH_DIVISOR + 0.5)

    cpdef void set_cash(self, GameState state, int cash):
        """Set FI's cash (integer dollars)."""
        state._data[self._cash_offset] = <float>cash / CASH_DIVISOR

    cpdef void add_cash(self, GameState state, int amount):
        """Add to FI's cash (can be negative to subtract)."""
        cdef int current = self.get_cash(state)
        self.set_cash(state, current + amount)

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Check if FI owns a company."""
        return state._data[self._owned_companies_offset + company_id] == 1.0

    cpdef void set_owns_company(self, GameState state, int company_id, bint owns):
        """Set whether FI owns a company."""
        state._data[self._owned_companies_offset + company_id] = 1.0 if owns else 0.0

    # =========================================================================
    # INCOME CALCULATION
    # =========================================================================

    cpdef int calculate_income(self, GameState state):
        """
        Calculate total income for Foreign Investor.

        Uses the cached company_incomes array (updated when CoO changes).
        FI always receives +5 base income bonus.

        Returns:
            Total income (always positive due to CLOSING phase)
        """
        cdef int total = 0
        cdef int company_id
        cdef int company_incomes_offset = state._layout.company_incomes_offset
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if state._data[self._owned_companies_offset + company_id] == 1.0:
                total += <int>lround(state._data[company_incomes_offset + company_id] * CASH_DIVISOR)
        # FI always gets +5 bonus (RULES.md line 354)
        return total + 5

    # =========================================================================
    # INCOME APPLICATION
    # =========================================================================

    cpdef void apply_income(self, GameState state, int income):
        """Apply calculated income to FI cash."""
        self.add_cash(state, income)


# =============================================================================
# GLOBAL FOREIGN INVESTOR INSTANCE
# =============================================================================

# Single FI instance
FI = ForeignInvestor()
