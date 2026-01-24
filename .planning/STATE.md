# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-23)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Phase 11 - Test Updates

## Current Position

Milestone: v3.0 WRAP_UP Phase
Phase: 11 of 11 (Test Updates)
Plan: 1 of 1
Status: Phase complete - bugs block milestone
Last activity: 2026-01-24 — Added failing tests for WRAP_UP bugs

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [#########!] 90% (4 failing tests)

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

**From 10-01 (2026-01-23):**
- FI purchase loop uses while-loop with re-query pattern (no snapshotting)
- Purchase iteration in ascending company_id order (0-35) guarantees cheapest-first
- Availability transition after all FI purchases (revealed → auction state change)

**From 11-01 (2026-01-24):**
- Simplified test strategy when implementation has critical bugs - document bugs, test what can be verified
- Test files should document known bugs in header comments
- Phase transition tests can verify flow even when intermediate logic has bugs

### Pending Todos

None.

### Blockers/Concerns

**Phase 11 complete - WRAP_UP testing reveals critical bugs:**
- 190 tests total: 186 passing, 4 failing
- test_invest.py updated for WRAP_UP flow (9 tests fixed)
- test_wrap_up.py created with 18 tests (14 passing, 4 failing)
- **Failing tests document bugs** - this is correct behavior

**CRITICAL BUGS in WRAP_UP implementation (4 failing tests):**

**Bug 1: FI cash becomes 0 after purchases**
- Test: `TestFICashPreservation::test_fi_cash_preserved_when_no_purchases`
- FI cash incorrectly zeroed after WRAP_UP regardless of purchases
- Example: FI has 50 cash, no purchases, ends with 0 cash

**Bug 2: Player cash becomes 0 for players 1+ after WRAP_UP**
- Tests: `TestPlayerCashPreservation::test_player_cash_preserved_*` (3 tests)
- After WRAP_UP cycle, players 1+ have cash=0 regardless of initial value
- Example: Set cash [20, 30, 25], after WRAP_UP: [20, 0, 0]

**Bug 3: Infinite loop when no companies available**
- ForcedActionLoopError when FI purchase loop has no companies to buy
- Discovered via failing `test_fi_cash_preserved_when_no_purchases`

**v3.0 cannot ship until bugs are fixed** - tests will remain failing as documentation

## Session Continuity

Last session: 2026-01-24
Stopped at: Added proper failing tests for WRAP_UP bugs
Resume file: None
Next action: Fix WRAP_UP bugs (investigate state array corruption in phases 9-10)
