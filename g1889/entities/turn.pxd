"""TURN: the global block's game, auction, round, and pending fields.

BANK owns the bank fields, PHASE ``game_phase``, MARKET ``market_stamp``, and
the corp and map entities the dirty masks. Setters write raw values; the
phase handlers own the round and step transitions.
"""

from g1889.state cimport GameState

cdef int turn_num_players(GameState state) noexcept nogil
cdef int turn_round(GameState state) noexcept nogil
cdef int turn_step(GameState state) noexcept nogil
cdef int turn_active_player(GameState state) noexcept nogil
cdef int turn_auction_private(GameState state) noexcept nogil
cdef int turn_auction_discount(GameState state) noexcept nogil
cdef int turn_auction_pass_payouts(GameState state) noexcept nogil
cdef int turn_pending_private(GameState state) noexcept nogil
cdef int turn_pending_corp(GameState state) noexcept nogil
cdef int turn_pending_hex(GameState state) noexcept nogil
cdef int turn_operating_corp(GameState state) noexcept nogil
cdef int turn_route_revenue(GameState state) noexcept nogil
cdef int turn_turn(GameState state) noexcept nogil
cdef bint turn_sr_bought(GameState state) noexcept nogil
cdef int turn_sr_sold_mask(GameState state) noexcept nogil
cdef bint turn_sr_sold_then_bought(GameState state) noexcept nogil


cdef class TurnState:
    cpdef int num_players(self, GameState state)
    cpdef int next_player(self, GameState state, int player)

    # Game
    cpdef int round(self, GameState state)
    cpdef void set_round(self, GameState state, int value)
    cpdef int step(self, GameState state)
    cpdef void set_step(self, GameState state, int value)
    cpdef int return_step(self, GameState state)
    cpdef void set_return_step(self, GameState state, int value)
    cpdef int turn(self, GameState state)
    cpdef void set_turn(self, GameState state, int value)
    cpdef int or_num(self, GameState state)
    cpdef void set_or_num(self, GameState state, int value)
    cpdef int or_count(self, GameState state)
    cpdef void set_or_count(self, GameState state, int value)
    cpdef int active_player(self, GameState state)
    cpdef void set_active_player(self, GameState state, int value)
    cpdef int priority_player(self, GameState state)
    cpdef void set_priority_player(self, GameState state, int value)
    cpdef int end_reason(self, GameState state)
    cpdef void set_end_reason(self, GameState state, int value)

    # Auction
    cpdef int auction_turn(self, GameState state)
    cpdef void set_auction_turn(self, GameState state, int value)
    cpdef int auction_private(self, GameState state)
    cpdef void set_auction_private(self, GameState state, int value)
    cpdef int auction_discount(self, GameState state)
    cpdef void set_auction_discount(self, GameState state, int value)
    cpdef int auction_pass_payouts(self, GameState state)
    cpdef void set_auction_pass_payouts(self, GameState state, int value)

    # Stock round: this turn's purchase, its sales, and whether a purchase
    # followed a sale (which ends selling for the turn)
    cpdef int sr_bought(self, GameState state)
    cpdef void set_sr_bought(self, GameState state, int value)
    cpdef int sr_sold_mask(self, GameState state)
    cpdef void set_sr_sold_mask(self, GameState state, int value)
    cpdef int sr_sold_then_bought(self, GameState state)
    cpdef void set_sr_sold_then_bought(self, GameState state, int value)

    # Operating round
    cpdef int operating_corp(self, GameState state)
    cpdef void set_operating_corp(self, GameState state, int value)
    cpdef int or_index(self, GameState state)
    cpdef void set_or_index(self, GameState state, int value)
    cpdef int or_order(self, GameState state, int index)
    cpdef void set_or_order(self, GameState state, int index, int corp)
    cpdef int route_revenue(self, GameState state)
    cpdef void set_route_revenue(self, GameState state, int value)

    # Pending choice
    cpdef int pending_hex(self, GameState state)
    cpdef void set_pending_hex(self, GameState state, int value)
    cpdef int pending_corp(self, GameState state)
    cpdef void set_pending_corp(self, GameState state, int value)
    cpdef int pending_private(self, GameState state)
    cpdef void set_pending_private(self, GameState state, int value)
    cpdef int pending_train_source(self, GameState state)
    cpdef void set_pending_train_source(self, GameState state, int value)
    cpdef int pending_seller(self, GameState state)
    cpdef void set_pending_seller(self, GameState state, int value)

    # Emergency money
    cpdef int ebuy_sold_mask(self, GameState state)
    cpdef void set_ebuy_sold_mask(self, GameState state, int value)
    cpdef int ebuy_sold_any(self, GameState state)
    cpdef void set_ebuy_sold_any(self, GameState state, int value)
