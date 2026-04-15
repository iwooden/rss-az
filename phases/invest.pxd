"""INVEST phase handler declarations.

Handles the four INVEST actions: PASS, AUCTION (start), BUY_SHARE, SELL_SHARE.
All state access goes through entity handles. Illegal actions are a driver
bug — the driver owns legality gating via ``_enumerate_invest`` in
``core/actions.pyx``, so the handler takes an ``ActionInfo*`` and assumes
it is legal.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo


cdef void apply_invest_action(GameState state, ActionInfo* info) noexcept
