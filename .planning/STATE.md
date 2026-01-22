# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-21)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** v2.1 Forced Action Auto-Application

## Current Position

Milestone: v2.1 Forced Action Auto-Application
Phase: 7 - Core Implementation (pending)
Status: Roadmap complete, ready for planning
Last activity: 2026-01-21 - Roadmap created

Progress: v1 [##########] | v2 [##########] | v2.1 [░░░░░░░░░░] 0%

## Phase Overview

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 7 | Core Implementation | 12 | Pending |
| 8 | Test Updates | 7 | Pending |

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |

See `.planning/milestones/` for full archives.

## Accumulated Context

### Key Patterns (from v1 and v2)

**Cython patterns:**
- Entity initialization order - Initialize all handles before setting state
- Module import pattern - Avoid circular imports with `from entities import X as X_module`
- Stateless singleton pattern - GameDriver follows entity handle design
- Phase handler pattern - cdef noexcept functions for zero overhead
- Cdef variable declaration - Declare all cdef vars at function start

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

### Pending Todos

None.

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-01-21
Stopped at: Roadmap creation complete
Resume file: .planning/ROADMAP.md
Next action: /gsd:plan-phase 7
