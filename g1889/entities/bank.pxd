"""BANK: the bank's cash and broken flag, and the only code that moves cash.

Accounts are owner codes: player i is ``OWNER_PLAYER + i``, corp c is
``OWNER_CORP + c``, and the bank is ``BANK_ACCOUNT``.
"""

from g1889.state cimport GameState

cpdef enum Account:
    BANK_ACCOUNT = -1

cdef int bank_cash(GameState state) noexcept nogil
cdef bint bank_broken(GameState state) noexcept nogil


cdef class Bank:
    cpdef int cash(self, GameState state)
    cpdef bint broken(self, GameState state)
    cpdef void set_broken(self, GameState state)
    cpdef void transfer(self, GameState state, int src, int dst, int amount)
