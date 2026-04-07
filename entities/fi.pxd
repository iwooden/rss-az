# cython: language_level=3
"""
Foreign Investor entity declarations.

In the compact GameState the FI block is just two raw int16 slots — cash
and income. Company ownership is tracked entirely through the shared
company_locations / company_owner_ids arrays (LOC_FI), so the FI handle no
longer carries its own ownership flags.
"""

from core.state cimport GameState


cdef class ForeignInvestor:
    # Cached absolute offsets into the compact state array.
    cdef int _cash_offset
    cdef int _income_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Low-level (nogil) accessors used by hot paths inside the engine.
    cdef int _get_cash(self, GameState state) noexcept nogil
    cdef void _set_cash(self, GameState state, int cash) noexcept nogil
    cdef int _get_income(self, GameState state) noexcept nogil
    cdef void _set_income(self, GameState state, int income) noexcept nogil

    # Cash (Python-accessible wrappers)
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Company ownership (use Company.transfer_to_fi() to set)
    cpdef bint owns_company(self, GameState state, int company_id)

    # Income (Python-accessible wrappers)
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef int calculate_income(self, GameState state)
    cpdef void apply_income(self, GameState state)
