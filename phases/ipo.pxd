"""IPO phase handler declarations.

Two entry points: ``setup_ipo_phase`` initializes the per-company remaining
flags and finds the first company, ``apply_ipo_action`` dispatches IPO
decisions (PASS or corp-select). Par-price selection is DPHASE_PAR; the PAR
handler calls ``_advance_to_next_company`` to walk back to IPO on the next
player-owned company (or transition to INVEST when none remain).
All state access goes through entity handles.
"""

from core.state cimport GameState
from core.actions cimport ActionInfo

cdef void setup_ipo_phase(GameState state) noexcept
cdef void apply_ipo_action(GameState state, ActionInfo* info) noexcept
cdef void _advance_to_next_company(GameState state) noexcept
