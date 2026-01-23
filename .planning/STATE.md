# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Phase 9 - WRAP_UP Core Logic

## Current Position

Milestone: v3.0 WRAP_UP Phase
Phase: 9 of 11 (WRAP_UP Core Logic)
Plan: 2 of 2
Status: Phase complete
Last activity: 2026-01-23 — Completed 09-02-PLAN.md

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##        ] 22%

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |

See `.planning/milestones/` for full archives.

## v3.0 Roadmap Summary

**Phases:** 3 (9-11)
**Requirements:** 18 total
- Player Reordering: 3 requirements (REORDER-01 to REORDER-03)
- Foreign Investor Purchases: 7 requirements (FI-01 to FI-07)
- Company Availability: 1 requirement (AVAIL-01)
- Phase Transitions: 4 requirements (PHASE-01 to PHASE-04)
- Testing: 3 requirements (TEST-01 to TEST-03)

**Phase structure:**
- Phase 9: WRAP_UP Core Logic — Player reordering + phase transitions (7 requirements)
- Phase 10: FI Purchase Logic — Foreign Investor purchases + company availability (8 requirements)
- Phase 11: Test Updates — Fix tests + add verification tests (3 requirements)

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

### Key Decisions for v3.0

**From research (2026-01-22):**
- WRAP_UP is fully deterministic (zero player choices)
- Implement as atomic operation that gets discrete state history entry
- Loosen 0-action invariant for non-player phases
- Use while-loop with re-query for FI purchases (no snapshotting)
- All entity interfaces already exist (no new methods needed)

**From 09-01 (2026-01-23):**
- Selection sort for player reordering (stable, explicit tie-breaking at 6 players max)
- Turn number increment in ACQUISITION (final phase before INVEST)
- setup.py auto-discovery handles new phase modules (no manual edits needed)

**From 09-02 (2026-01-23):**
- Sentinel action values (negative integers) for non-player phase history entries
- Non-player phases execute automatically in auto-apply loop with history recording
- Complete phase flow: INVEST → WRAP_UP → ACQUISITION → INVEST (new turn)

### Pending Todos

None.

### Blockers/Concerns

**Phase 9 complete - ready for Phase 10:**
- WRAP_UP and ACQUISITION phases integrated and working
- Phase flow tested (9 test failures expected - covered in Phase 11)

**Known issues for Phase 11 (test updates):**
- 9 tests in test_invest.py expect GAME_OVER after all-pass but now get STATUS_OK (new turn)
- Tests need updates to verify WRAP_UP → ACQUISITION → INVEST flow
- Tests need verification of sentinel action history entries

## Session Continuity

Last session: 2026-01-23
Stopped at: Completed 09-02-PLAN.md (Phase 9 complete)
Resume file: None
Next action: Phase 10 - FI Purchase Logic
