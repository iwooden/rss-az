# cython: language_level=3
"""
Player entity declarations.
"""

from state cimport GameState


cdef class Player:
    cdef readonly int player_id
    cdef int _base_offset      # Cached offset to this player's data in state array
    cdef int _num_players      # Cached player count
    cdef bint _initialized     # Whether offsets have been computed

    # Field offsets within player stride (cached on first use)
    cdef int _cash_offset
    cdef int _net_worth_offset
    cdef int _turn_order_offset
    cdef int _owned_companies_offset
    cdef int _owned_shares_offset
    cdef int _is_president_offset
    cdef int _share_buys_offset
    cdef int _share_sells_offset

    # Initialization
    cpdef void initialize(self, GameState state)

    # Cash
    cpdef int get_cash(self, GameState state)
    cpdef void set_cash(self, GameState state, int cash)
    cpdef void add_cash(self, GameState state, int amount)

    # Net worth
    cpdef int get_net_worth(self, GameState state)
    cpdef void set_net_worth(self, GameState state, int net_worth)
    # TODO: calculate_net_worth() - requires Corp entity for share prices

    # Turn order
    cpdef int get_turn_order(self, GameState state)
    cpdef void set_turn_order(self, GameState state, int order)

    # Company ownership
    cpdef bint owns_company(self, GameState state, int company_id)
    cpdef void set_owns_company(self, GameState state, int company_id, bint owns)

    # Corporation shares
    cpdef int get_shares(self, GameState state, int corp_id)
    cpdef void set_shares(self, GameState state, int corp_id, int shares)

    # President status
    cpdef bint is_president_of(self, GameState state, int corp_id)
    cpdef void set_president_of(self, GameState state, int corp_id, bint is_pres)

    # Round-trip tracking (invest phase)
    cpdef int get_share_buys(self, GameState state, int corp_id)
    cpdef void increment_share_buys(self, GameState state, int corp_id)
    cpdef int get_share_sells(self, GameState state, int corp_id)
    cpdef void increment_share_sells(self, GameState state, int corp_id)
    cpdef int get_roundtrips(self, GameState state, int corp_id)
    cpdef void clear_roundtrip_tracking(self, GameState state)


# Global player instances are exposed as a Python list in player.pyx
