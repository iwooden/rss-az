# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-24)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Planning next milestone

## Current Position

Milestone: v3.0 WRAP_UP Phase — SHIPPED
Phase: Complete (11 phases total through v3.0)
Plan: N/A
Status: Ready for next milestone
Last activity: 2026-01-24 — v3.0 milestone completed and archived

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##########] 100%

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |
| v3.0 | WRAP_UP Phase | 9-11 + 10.1 | 6 | 2026-01-24 |

See `.planning/milestones/` for full archives.

## Next Milestone

**Candidates for v4.0:**
- Remaining game phases: CLO (Close of Market), INC (Income), DIV (Dividend), END (End of Turn), ISS (Issue Stock), IPO (Initial Public Offering)
- Full game loop completion

## Accumulated Context

### Key Patterns (from v1-v3.0)

**Cython patterns:**
- Entity initialization order - Initialize all handles before setting state
- Module import pattern - Avoid circular imports with `from entities import X as X_module`
- Stateless singleton pattern - GameDriver follows entity handle design
- Phase handler pattern - cdef noexcept functions for zero overhead
- Cdef variable declaration - Declare all cdef vars at function start
- Auto-apply loop pattern - Iterative forced action application until 2+ choices
- Early-exit counting - Stop at count=2 instead of counting all actions
- Non-player phase pattern - 0 actions valid for deterministic phases (WRAP_UP, ACQUISITION)
- Sentinel action values - Negative integers (-100, -101) for non-player phase history

**Game logic patterns:**
- Turn order navigation - Find player at position, advance with wraparound
- Auction resolution sequence - pay -> transfer -> update -> draw -> cleanup -> transition
- Price movement - find_next_higher/lower_space skips occupied spaces
- Index 26 always available - Price $75 never marked occupied
- Bankruptcy inline execution - Execute immediately during sell, no deferral
- Two-pass presidency algorithm - Find max shares first, then check incumbent for tie-breaking
- Receivership before presidency - Check receivership first, skip presidency if in receivership
- Terminal state detection - Check for playable game state before transitioning to INVEST
- Player reordering - Selection sort by (-cash, old_position) for descending cash with tie-breaking
- FI purchase loop - While-loop with re-query, ascending company_id for cheapest-first

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Test fixture pattern - Phase-specific fixtures return state in target phase
- Shared conftest pattern - Centralized fixtures and assertion helpers
- Integration test consolidation - Cross-phase tests in test_integration.py
- History tracking pattern - pass history=[] for full chain observation
- State snapshot pattern - get_state_at(index) reconstructs GameState from history

### Pending Todos

None.

### Blockers/Concerns

None - ready for next milestone.

## Session Continuity

Last session: 2026-01-24
Stopped at: v3.0 milestone completion
Resume file: None
Next action: /gsd:new-milestone for v4.0
