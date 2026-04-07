# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Foreign Investor entity implementation.

The FI is a special entity that buys companies (at face value when OS has
the priority slot, otherwise at high price) and holds them until
corporations acquire them. In the compact state layout the FI block is
just two raw int16 slots — cash at ``LAYOUT.fi_offset`` and income at
``LAYOUT.fi_offset + 1``. Ownership of companies lives in the shared
company_locations / company_owner_ids arrays via LOC_FI, so this handle
no longer iterates a per-FI ownership flag array.

The handle is fully stateless: every read computes its slot inline from
the module-level ``LAYOUT`` constant on ``core.state``. No per-instance
offset cache, no initialize() step.
"""

from libc.stdint cimport int16_t

from core.state cimport GameState, LAYOUT
from core.data cimport GameConstants
from entities.company cimport LOC_FI


cdef class ForeignInvestor:
    """
    Entity handle for accessing Foreign Investor state.

    A single global instance is created at module load. The handle has
    no per-instance state — all methods take a GameState as the first
    argument and read offsets directly from the module-level ``LAYOUT``
    constant.
    """

    # =========================================================================
    # CASH (low-level, nogil)
    # =========================================================================

    cdef inline int _get_cash(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.fi_offset]

    cdef inline void _set_cash(self, GameState state, int cash) noexcept nogil:
        state._data[LAYOUT.fi_offset] = <int16_t>cash

    # =========================================================================
    # CASH (Python-accessible wrappers)
    # =========================================================================

    cpdef int get_cash(self, GameState state):
        """Return FI's cash (raw integer dollars)."""
        return self._get_cash(state)

    cpdef void set_cash(self, GameState state, int cash):
        """Set FI's cash (raw integer dollars)."""
        self._set_cash(state, cash)

    cpdef void add_cash(self, GameState state, int amount):
        """Add to FI's cash (negative `amount` subtracts)."""
        self._set_cash(state, self._get_cash(state) + amount)

    # =========================================================================
    # COMPANY OWNERSHIP
    # =========================================================================

    cpdef bint owns_company(self, GameState state, int company_id):
        """Return True if FI currently owns the given company.

        Backed by the shared company_locations array — there is no
        FI-side ownership flag to keep in sync.
        """
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return state._data[LAYOUT.company_locations_offset + company_id] == <int>LOC_FI

    # =========================================================================
    # INCOME (low-level, nogil)
    # =========================================================================

    cdef inline int _get_income(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.fi_offset + 1]

    cdef inline void _set_income(self, GameState state, int income) noexcept nogil:
        state._data[LAYOUT.fi_offset + 1] = <int16_t>income

    # =========================================================================
    # INCOME (Python-accessible wrappers)
    # =========================================================================

    cpdef int get_income(self, GameState state):
        """Return FI's stored income (raw integer dollars)."""
        return self._get_income(state)

    cpdef void set_income(self, GameState state, int income):
        """Set FI's income (raw integer dollars)."""
        self._set_income(state, income)

    cpdef int calculate_income(self, GameState state):
        """Recalculate, store, and return FI's total income.

        Sums adjusted incomes of every company currently in LOC_FI from
        the shared company_incomes array (kept up to date by the engine
        as ownership and CoO change). FI always receives a +5 base income
        bonus per RULES.md (line 354).

        Returns:
            Total income (always positive due to CLOSING phase rules).
        """
        cdef int total = 0
        cdef int company_id
        cdef int company_incomes_offset = LAYOUT.company_incomes_offset
        cdef int company_locations_offset = LAYOUT.company_locations_offset
        cdef int loc_fi = <int>LOC_FI
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            if state._data[company_locations_offset + company_id] == loc_fi:
                total += <int>state._data[company_incomes_offset + company_id]
        # FI always gets +5 bonus (RULES.md line 354)
        total += 5
        self._set_income(state, total)
        return total

    # =========================================================================
    # INCOME APPLICATION
    # =========================================================================

    cpdef void apply_income(self, GameState state):
        """Apply FI's stored income to its cash.

        Relies on the contract that `calculate_income()` is called whenever
        an event changes FI's income (company transfers in/out, CoO level
        changes, etc.), so the stored value is always current.
        """
        self.add_cash(state, self._get_income(state))


# =============================================================================
# GLOBAL FOREIGN INVESTOR INSTANCE
# =============================================================================

# Single FI instance
FI = ForeignInvestor()
