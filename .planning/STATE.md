# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Planning next milestone

## Current Position

Milestone: v2.1 complete - planning next
Phase: 8 of 8 complete
Status: Milestone shipped
Last activity: 2026-01-23 - v2.1 milestone complete

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] 100%

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |

See `.planning/milestones/` for full archives.

## Accumulated Context

### Key Patterns (from v1, v2, and v2.1)

**Cython patterns:**
- Entity initialization order - Initialize all handles before setting state
- Module import pattern - Avoid circular imports with `from entities import X as X_module`
- Stateless singleton pattern - GameDriver follows entity handle design
- Phase handler pattern - cdef noexcept functions for zero overhead
- Cdef variable declaration - Declare all cdef vars at function start
- Auto-apply loop pattern - Iterative forced action application until 2+ choices
- Early-exit counting - Stop at count=2 instead of counting all actions

**Game logic patterns:**
- Turn order navigation - Find player at position, advance with wraparound
- Auction resolution sequence - pay -> transfer -> update -> draw -> cleanup -> transition
- Price movement - find_next_higher/lower_space skips occupied spaces
- Index 26 always available - Price $75 never marked occupied
- Bankruptcy inline execution - Execute immediately during sell, no deferral
- Two-pass presidency algorithm - Find max shares first, then check incumbent for tie-breaking
- Receivership before presidency - Check receivership first, skip presidency if in receivership

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Test fixture pattern - Phase-specific fixtures return state in target phase
- Shared conftest pattern - Centralized fixtures and assertion helpers
- Integration test structure - assert_invariants -> apply_action_and_verify -> verify outcome -> assert_invariants
- Parametrized player counts (3, 6) - Boundary verification
- History tracking pattern - pass history=[] to DRIVER.apply_action for full chain observation
- State snapshot pattern - get_state_at(index) reconstructs GameState from history tuple
- Explicit history assertions - assert len(result.history) == 1 for no-auto-apply verification
- Test categorization - 3 categories: no changes, explicit assertions, edge cases

### Pending Todos

None.

### Blockers/Concerns

None - v2.1 milestone shipped. Ready for next milestone.

## Session Continuity

Last session: 2026-01-23
Stopped at: v2.1 milestone complete
Resume file: None
Next action: Start next milestone with /gsd:new-milestone
