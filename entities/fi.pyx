# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Foreign Investor entity implementation.

Provides clean getter/setter access to the Foreign Investor state in the game
state vector. The FI is a special entity that buys companies at high price
and holds them until corporations acquire them.
"""

from core.state cimport GameState, StateLayout
from core.data cimport (
    CASH_DIVISOR, GameConstants,
    get_company_income, get_company_stars, get_cost_of_ownership
)
from entities import turn as turn_module


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

        Formula: sum(printed_income - CoO) + 5
        FI always receives +5 base income bonus.

        Returns:
            Total income (always positive due to CLOSING phase)
        """
        cdef int company_id, base_income, stars, coo_value
        cdef int coo_level = turn_module.TURN.get_coo_level(state)
        cdef int total = 0

        for company_id in range(GameConstants.NUM_COMPANIES):
            if self.owns_company(state, company_id):
                base_income = get_company_income(company_id)
                stars = get_company_stars(company_id)
                coo_value = get_cost_of_ownership(coo_level, stars)
                total += base_income - coo_value

        # FI always gets +5 bonus (RULES.md line 354)
        total += 5

        return total


# =============================================================================
# GLOBAL FOREIGN INVESTOR INSTANCE
# =============================================================================

# Single FI instance
FI = ForeignInvestor()
