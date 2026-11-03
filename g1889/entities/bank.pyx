"""BANK entity; see bank.pxd."""

from libc.stdint cimport INT16_MAX, INT16_MIN, int16_t

from g1889.constants cimport OWNER_CORP, OWNER_PLAYER
from g1889.data.tables cimport NUM_CORPS
from g1889.entities.turn cimport turn_num_players
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, PLAYER_FIELDS, GameState


cdef int bank_cash(GameState state) noexcept nogil:
    return <int>state._data[LAYOUT.global_offset + GLOBALS.bank_cash]


cdef bint bank_broken(GameState state) noexcept nogil:
    return state._data[LAYOUT.global_offset + GLOBALS.bank_broken] != 0


cdef int _cash_slot(GameState state, int account) except -1:
    if account == BANK_ACCOUNT:
        return LAYOUT.global_offset + GLOBALS.bank_cash
    if <int>OWNER_PLAYER <= account < <int>OWNER_PLAYER + turn_num_players(state):
        return LAYOUT.players_offset + (account - <int>OWNER_PLAYER) * PLAYER_FIELDS.size + PLAYER_FIELDS.cash
    assert <int>OWNER_CORP <= account < <int>OWNER_CORP + <int>NUM_CORPS, f"no cash account {account}"
    return LAYOUT.corps_offset + (account - <int>OWNER_CORP) * CORP_FIELDS.size + CORP_FIELDS.cash


cdef class Bank:
    """Stateless handle over the bank fields."""

    cpdef int cash(self, GameState state):
        return bank_cash(state)

    cpdef bint broken(self, GameState state):
        return bank_broken(state)

    cpdef void set_broken(self, GameState state):
        """Break the bank; it stays broken (``bank.rb``)."""
        state._data[LAYOUT.global_offset + GLOBALS.bank_broken] = 1

    cpdef void transfer(self, GameState state, int src, int dst, int amount):
        """Move ``amount`` from ``src`` to ``dst``. Players and corps cannot go
        below zero; a payment that would take the bank below zero breaks it and
        is still made (``bank.rb:26-39``)."""
        cdef int src_slot = _cash_slot(state, src)
        cdef int dst_slot = _cash_slot(state, dst)
        assert src != dst, f"transfer from account {src} to itself"
        assert amount >= 0, f"negative transfer {amount}"
        cdef int src_cash = <int>state._data[src_slot] - amount
        cdef int dst_cash = <int>state._data[dst_slot] + amount
        assert src_cash >= (INT16_MIN if src == BANK_ACCOUNT else 0), \
            f"account {src} cannot pay {amount} from {state._data[src_slot]}"
        assert dst_cash <= INT16_MAX, f"account {dst} cash {dst_cash} leaves int16"
        if src_cash < 0:
            self.set_broken(state)
        state._data[src_slot] = <int16_t>src_cash
        state._data[dst_slot] = <int16_t>dst_cash


BANK = Bank()

# Python mirrors: a bare assignment would be shadowed by the cdef names.
for _name, _member in Account.__members__.items():
    globals()[_name] = int(_member)
del _name, _member
