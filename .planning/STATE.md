# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-28)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** v6.0 INCOME phase - Phase 22 (Income Calculation)

## Current Position

Milestone: v6.0 INCOME Phase
Phase: 22 of 23 (Income Calculation)
Plan: 1 of 2 in progress
Status: Phase 22 in progress
Last activity: 2026-01-29 — Completed 22-01-PLAN.md

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##########] | v4.0 [##########] | v5.0 [##########] | v5.1 [##########] | v6.0 [##________]

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |
| v3.0 | WRAP_UP Phase | 9-11 + 10.1 | 6 | 2026-01-24 |
| v4.0 | ACQUISITION Phase | 12-15 | 13 | 2026-01-26 |
| v5.0 | CLOSING Phase | 15.1, 16-19 | 14 | 2026-01-27 |
| v5.1 | nogil Optimization | 20 | 3 | 2026-01-28 |

See `.planning/milestones/` for full archives.

## Accumulated Context

### Key Patterns (from v1-v5.1)

**Cython patterns:**
- Non-player phase pattern - 0 actions valid for deterministic phases (WRAP_UP pattern)
- Entity handle pattern for state access
- Phase handler pattern - cdef noexcept functions for zero overhead
- Low-level nogil accessor pattern - Offsets struct + get_offsets() + cdef inline accessors
- Player income calculation pattern - Player method returns sum of adjusted income from owned privates only (18-01)
- Pair counting pattern - i<j nested loop generates unique pairs without duplicates (21-01)
- Entity income calculation pattern - cpdef int calculate_income(GameState) for Corp/FI (22-01)

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Integration test consolidation - Cross-phase tests in test_integration.py
- TDD RED-GREEN-REFACTOR - test commit, feat commit, refactor commit (optional)

### Pending Todos

None.

### Blockers/Concerns

**INCOME Phase temporary transition:** Phase 17 transitions CLOSING -> INVEST as temporary workaround. **TO BE FIXED IN Phase 23 (TRN-01).**

**Remaining Game Phases after v6.0:** DIV (DIVIDENDS), END (END_GAME), ISS (ISSUE_SHARES), IPO (INITIAL_PUBLIC_OFFERING)

## Session Continuity

Last session: 2026-01-29
Stopped at: Completed 22-01-PLAN.md
Resume file: None
Next action: Plan and execute Phase 22-02 (Corporation Special Abilities)
