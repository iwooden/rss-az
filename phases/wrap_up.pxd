"""WRAP_UP phase handler declarations.

WRAP_UP is an automated engine phase with no decision action. The
handler reorders players by cash, runs the Foreign Investor purchase
loop, flips every LOC_REVEALED company to LOC_AUCTION, and transitions
to PHASE_ACQUISITION. All state access goes through entity handles.
"""

from core.state cimport GameState


cdef void apply_wrap_up(GameState state) noexcept
