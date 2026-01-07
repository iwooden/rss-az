# cython: language_level=3
"""
Corporation entity declarations.
"""

from state cimport GameState


cdef class Corporation:
    cdef readonly int corp_id
    cdef readonly str name
    cdef int _base_offset      # Cached offset to this corp's data in state array
    cdef int _num_players      # Cached player count

    # Hidden state offset for fast price index access
    cdef int _hidden_price_index_offset

    # Field offsets within corp stride (cached on first use)
    cdef int _active_offset
    cdef int _cash_offset
    cdef int _unissued_shares_offset
    cdef int _issued_shares_offset
    cdef int _bank_shares_offset
    cdef int _income_offset
    cdef int _stars_offset
    cdef int _share_price_offset
    cdef int _acquisition_proceeds_offset
    cdef int _in_receivership_offset
    cdef int _price_index_offset
    cdef int _owned_companies_offset
    cdef int _acquisition_companies_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Active status
    cpdef bint is_active(self, GameState state)
    cpdef void set_active(self, GameState state, bint active)

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Share tracking
    cpdef int get_unissued_shares(self, GameState state)
    cpdef void set_unissued_shares(self, GameState state, int shares)
    cpdef int get_issued_shares(self, GameState state)
    cpdef void set_issued_shares(self, GameState state, int shares)
    cpdef int get_bank_shares(self, GameState state)
    cpdef void set_bank_shares(self, GameState state, int shares)

    # Income and stars
    cpdef int get_income(self, GameState state)
    cpdef void set_income(self, GameState state, int income)
    cpdef int get_stars(self, GameState state)
    cpdef void set_stars(self, GameState state, int stars)

    # Share price
    cpdef int get_share_price(self, GameState state)
    cpdef void set_share_price(self, GameState state, int price)
    cpdef int get_price_index(self, GameState state)
    cpdef void set_price_index(self, GameState state, int index)

    # Acquisition proceeds
    cpdef int get_acquisition_proceeds(self, GameState state)
    cpdef void set_acquisition_proceeds(self, GameState state, int proceeds)

    # Receivership
    cpdef bint is_in_receivership(self, GameState state)
    cpdef void set_in_receivership(self, GameState state, bint in_recv)

    # Company ownership
    cpdef bint owns_company(self, GameState state, int company_id)
    cpdef void set_owns_company(self, GameState state, int company_id, bint owns)

    # Acquisition pile (companies pending integration)
    cpdef bint has_acquisition_company(self, GameState state, int company_id)
    cpdef void set_acquisition_company(self, GameState state, int company_id, bint has)
