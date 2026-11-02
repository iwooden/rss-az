"""PRIVATES: the private blocks (owner code and flags), the auction's
queries over the players' bids on each private, and the sales to corps and
DR's exchange."""

from g1889.state cimport GameState

# Bits of a private's ``flags``.
cpdef enum PrivateFlag:
    MF_PORT_USED = 1                # MF's port tile is laid (its ability is used up)

cdef int private_owner(GameState state, int private) noexcept nogil
cdef int private_flags(GameState state, int private) noexcept nogil
cdef int private_revenue(GameState state, int private) noexcept nogil
cdef bint private_blocks_hex(GameState state, int hex) noexcept nogil

# Auction (Ruby's WaterfallAuction): unsold privates sell top-down.
cdef int topmost_private(GameState state) noexcept nogil
cdef int private_price(GameState state, int private) noexcept nogil
cdef int private_high_bid(GameState state, int private) noexcept nogil
cdef int private_bid_count(GameState state, int private) noexcept nogil
cdef int private_lowest_bidder(GameState state, int private) noexcept nogil
cdef int private_min_bid(GameState state, int private) noexcept nogil


cdef class Private:
    cdef readonly int private_id
    cdef readonly str sym

    cpdef int owner(self, GameState state)
    cpdef void set_owner(self, GameState state, int owner)
    cpdef int flags(self, GameState state)
    cpdef void set_flags(self, GameState state, int flags)
    cpdef void close(self, GameState state)
    cpdef int revenue(self, GameState state)
    cpdef bint blocks_hex(self, GameState state, int hex)

    cpdef int price(self, GameState state)
    cpdef int high_bid(self, GameState state)
    cpdef int bid_count(self, GameState state)
    cpdef int lowest_bidder(self, GameState state)
    cpdef int min_bid(self, GameState state)
    cpdef void sell_to_player(self, GameState state, int player, int price)

    # Operating rounds (phases 3-4) and DR's exchange
    cpdef void sell_to_corp(self, GameState state, int corp, int price)
    cpdef void exchange(self, GameState state)


cpdef int topmost(GameState state)
cpdef void pay_revenue(GameState state)
