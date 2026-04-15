"""INCOME phase handler declarations.

INCOME is an automated engine phase with no decision action. The
handler collects income for all entities (players, corporations, FI),
checks for corp bankruptcy, and transitions to PHASE_DIVIDENDS.
All state access goes through entity handles.
"""

from core.state cimport GameState


cdef void apply_income(GameState state) noexcept
