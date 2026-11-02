"""TURN entity; see turn.pxd."""

from libc.stdint cimport int16_t

from g1889.constants cimport NUM_END_REASONS, NUM_ROUNDS, NUM_STEPS
from g1889.data.tables cimport NUM_CORPS, NUM_HEXES, NUM_PRIVATES
from g1889.state cimport GameState, GLOBALS, LAYOUT


cdef inline int _get(GameState state, int field) noexcept nogil:
    return <int>state._data[LAYOUT.global_offset + field]


cdef inline void _set(GameState state, int field, int value) noexcept nogil:
    state._data[LAYOUT.global_offset + field] = <int16_t>value


cdef int turn_num_players(GameState state) noexcept nogil:
    return _get(state, GLOBALS.num_players)


cdef int turn_round(GameState state) noexcept nogil:
    return _get(state, GLOBALS.round)


cdef int turn_step(GameState state) noexcept nogil:
    return _get(state, GLOBALS.step)


cdef int turn_active_player(GameState state) noexcept nogil:
    return _get(state, GLOBALS.active_player)


cdef inline int _check(int value, int lo, int hi, str name) except -1:
    assert lo <= value < hi, f"{name} {value} out of range [{lo}, {hi})"
    return 0


cdef class TurnState:
    """Stateless handle over the global block's turn fields."""

    cpdef int num_players(self, GameState state):
        return turn_num_players(state)

    cpdef int next_player(self, GameState state, int player):
        """The next player clockwise. Bankruptcy ends a 1889 game, so none is skipped."""
        return (player + 1) % turn_num_players(state)

    # Game

    cpdef int round(self, GameState state):
        return turn_round(state)

    cpdef void set_round(self, GameState state, int value):
        _check(value, 0, <int>NUM_ROUNDS, 'round')
        _set(state, GLOBALS.round, value)

    cpdef int step(self, GameState state):
        return turn_step(state)

    cpdef void set_step(self, GameState state, int value):
        _check(value, 0, <int>NUM_STEPS, 'step')
        _set(state, GLOBALS.step, value)

    cpdef int return_step(self, GameState state):
        return _get(state, GLOBALS.return_step)

    cpdef void set_return_step(self, GameState state, int value):
        _check(value, -1, <int>NUM_STEPS, 'return_step')
        _set(state, GLOBALS.return_step, value)

    cpdef int turn(self, GameState state):
        return _get(state, GLOBALS.turn)

    cpdef void set_turn(self, GameState state, int value):
        _set(state, GLOBALS.turn, value)

    cpdef int or_num(self, GameState state):
        return _get(state, GLOBALS.or_num)

    cpdef void set_or_num(self, GameState state, int value):
        _set(state, GLOBALS.or_num, value)

    cpdef int or_count(self, GameState state):
        return _get(state, GLOBALS.or_count)

    cpdef void set_or_count(self, GameState state, int value):
        _set(state, GLOBALS.or_count, value)

    cpdef int active_player(self, GameState state):
        return turn_active_player(state)

    cpdef void set_active_player(self, GameState state, int value):
        _check(value, 0, turn_num_players(state), 'active_player')
        _set(state, GLOBALS.active_player, value)

    cpdef int priority_player(self, GameState state):
        return _get(state, GLOBALS.priority_player)

    cpdef void set_priority_player(self, GameState state, int value):
        _check(value, 0, turn_num_players(state), 'priority_player')
        _set(state, GLOBALS.priority_player, value)

    cpdef int end_reason(self, GameState state):
        return _get(state, GLOBALS.end_reason)

    cpdef void set_end_reason(self, GameState state, int value):
        _check(value, 0, <int>NUM_END_REASONS, 'end_reason')
        _set(state, GLOBALS.end_reason, value)

    # Auction

    cpdef int auction_turn(self, GameState state):
        return _get(state, GLOBALS.auction_turn)

    cpdef void set_auction_turn(self, GameState state, int value):
        _check(value, 0, turn_num_players(state), 'auction_turn')
        _set(state, GLOBALS.auction_turn, value)

    cpdef int auction_private(self, GameState state):
        return _get(state, GLOBALS.auction_private)

    cpdef void set_auction_private(self, GameState state, int value):
        _check(value, -1, <int>NUM_PRIVATES, 'auction_private')
        _set(state, GLOBALS.auction_private, value)

    cpdef int auction_discount(self, GameState state):
        return _get(state, GLOBALS.auction_discount)

    cpdef void set_auction_discount(self, GameState state, int value):
        _set(state, GLOBALS.auction_discount, value)

    cpdef int auction_pass_payouts(self, GameState state):
        return _get(state, GLOBALS.auction_pass_payouts)

    cpdef void set_auction_pass_payouts(self, GameState state, int value):
        _set(state, GLOBALS.auction_pass_payouts, value)

    # Stock round

    cpdef int sr_bought(self, GameState state):
        return _get(state, GLOBALS.sr_bought)

    cpdef void set_sr_bought(self, GameState state, int value):
        _set(state, GLOBALS.sr_bought, value)

    cpdef int sr_sold_mask(self, GameState state):
        return _get(state, GLOBALS.sr_sold_mask)

    cpdef void set_sr_sold_mask(self, GameState state, int value):
        _set(state, GLOBALS.sr_sold_mask, value)

    cpdef int sr_sold_then_bought(self, GameState state):
        return _get(state, GLOBALS.sr_sold_then_bought)

    cpdef void set_sr_sold_then_bought(self, GameState state, int value):
        _set(state, GLOBALS.sr_sold_then_bought, value)

    # Operating round

    cpdef int operating_corp(self, GameState state):
        return _get(state, GLOBALS.operating_corp)

    cpdef void set_operating_corp(self, GameState state, int value):
        _check(value, -1, <int>NUM_CORPS, 'operating_corp')
        _set(state, GLOBALS.operating_corp, value)

    cpdef int or_index(self, GameState state):
        return _get(state, GLOBALS.or_index)

    cpdef void set_or_index(self, GameState state, int value):
        _set(state, GLOBALS.or_index, value)

    cpdef int or_order(self, GameState state, int index):
        _check(index, 0, <int>NUM_CORPS, 'or_order index')
        return _get(state, GLOBALS.or_order + index)

    cpdef void set_or_order(self, GameState state, int index, int corp):
        _check(index, 0, <int>NUM_CORPS, 'or_order index')
        _check(corp, -1, <int>NUM_CORPS, 'or_order corp')
        _set(state, GLOBALS.or_order + index, corp)

    cpdef int route_revenue(self, GameState state):
        return _get(state, GLOBALS.route_revenue)

    cpdef void set_route_revenue(self, GameState state, int value):
        _set(state, GLOBALS.route_revenue, value)

    # Pending choice

    cpdef int pending_hex(self, GameState state):
        return _get(state, GLOBALS.pending_hex)

    cpdef void set_pending_hex(self, GameState state, int value):
        _check(value, -1, <int>NUM_HEXES, 'pending_hex')
        _set(state, GLOBALS.pending_hex, value)

    cpdef int pending_corp(self, GameState state):
        return _get(state, GLOBALS.pending_corp)

    cpdef void set_pending_corp(self, GameState state, int value):
        _check(value, -1, <int>NUM_CORPS, 'pending_corp')
        _set(state, GLOBALS.pending_corp, value)

    cpdef int pending_private(self, GameState state):
        return _get(state, GLOBALS.pending_private)

    cpdef void set_pending_private(self, GameState state, int value):
        _check(value, -1, <int>NUM_PRIVATES, 'pending_private')
        _set(state, GLOBALS.pending_private, value)

    cpdef int pending_train_source(self, GameState state):
        return _get(state, GLOBALS.pending_train_source)

    cpdef void set_pending_train_source(self, GameState state, int value):
        _set(state, GLOBALS.pending_train_source, value)

    cpdef int pending_seller(self, GameState state):
        return _get(state, GLOBALS.pending_seller)

    cpdef void set_pending_seller(self, GameState state, int value):
        _check(value, -1, turn_num_players(state), 'pending_seller')
        _set(state, GLOBALS.pending_seller, value)

    # Emergency money

    cpdef int ebuy_sold_mask(self, GameState state):
        return _get(state, GLOBALS.ebuy_sold_mask)

    cpdef void set_ebuy_sold_mask(self, GameState state, int value):
        _set(state, GLOBALS.ebuy_sold_mask, value)

    cpdef int ebuy_sold_any(self, GameState state):
        return _get(state, GLOBALS.ebuy_sold_any)

    cpdef void set_ebuy_sold_any(self, GameState state, int value):
        _set(state, GLOBALS.ebuy_sold_any, value)


TURN = TurnState()
