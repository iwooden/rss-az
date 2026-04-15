"""END_CARD phase handler declarations.

END_CARD is an automated engine phase with no decision action. The
handler runs the four game-end checks per RULES.md §Phase 7 and either
transitions to PHASE_GAME_OVER, flips the end card and continues, or
transitions to PHASE_ISSUE_SHARES. All state access goes through
entity handles.
"""

from core.state cimport GameState


cdef void apply_end_card(GameState state) noexcept
