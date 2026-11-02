"""PLAYERS: the player blocks. Cash moves only through ``BANK.transfer``.

Holdings are in 10% shares, the President's Certificate counting 2; it is one
certificate toward the limit.
"""

from g1889.state cimport GameState

cdef int player_cash(GameState state, int player) noexcept nogil
cdef int player_shares(GameState state, int player, int corp) noexcept nogil
cdef int player_bid(GameState state, int player, int private) noexcept nogil
cdef int player_committed(GameState state, int player) noexcept nogil
cdef int player_available(GameState state, int player, int private) noexcept nogil
cdef bint player_passed(GameState state, int player) noexcept nogil
cdef int player_sold_mask(GameState state, int player) noexcept nogil

# Stock round limits (rules §6.2, §6.8)
cdef int player_cert_count(GameState state, int player) noexcept nogil
cdef bint player_can_gain(GameState state, int player, int corp, int shares) noexcept nogil

# Ruby's Player#value, which final_scores reports
cdef int player_value(GameState state, int player) noexcept nogil


cdef class Player:
    cdef readonly int player_id

    cpdef int cash(self, GameState state)
    cpdef int shares(self, GameState state, int corp)
    cpdef void set_shares(self, GameState state, int corp, int shares)
    cpdef int sold_mask(self, GameState state)
    cpdef void set_sold_mask(self, GameState state, int mask)
    cpdef int cert_count(self, GameState state)
    cpdef bint can_gain(self, GameState state, int corp, int shares=*)
    cpdef int value(self, GameState state)

    # Auction bids, at most one per private; 0 when none
    cpdef int bid(self, GameState state, int private)
    cpdef void set_bid(self, GameState state, int private, int amount)
    cpdef int committed(self, GameState state)
    cpdef int available(self, GameState state, int private=*)
    cpdef void clear_bids(self, GameState state)

    cpdef bint passed(self, GameState state)
    cpdef void set_passed(self, GameState state, bint passed)
    cpdef bint bankrupt(self, GameState state)
    cpdef void set_bankrupt(self, GameState state, bint bankrupt)


# Passed flags of every player
cpdef bint all_passed(GameState state)
cpdef void clear_passed(GameState state)
