# cython: language_level=3
"""
Foreign Investor entity declarations.
"""

from core.state cimport GameState


cdef class ForeignInvestor:
    cdef int _cash_offset           # Offset to FI cash in state array
    cdef int _owned_companies_offset  # Offset to FI owned companies

    # Initialization
    cpdef void initialize(self, GameState state)

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Company ownership
    cpdef bint owns_company(self, GameState state, int company_id)
    cpdef void set_owns_company(self, GameState state, int company_id, bint owns)

    # Income calculation
    cpdef int calculate_income(self, GameState state)
