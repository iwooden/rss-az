# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-02)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Planning next milestone (v7.0 DIVIDENDS phase)

## Current Position

Milestone: v6.0 INCOME Phase (COMPLETE)
Phase: N/A — milestone complete
Plan: N/A
Status: Ready to start v7.0
Last activity: 2026-02-02 — v6.0 milestone archived

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##########] | v4.0 [##########] | v5.0 [##########] | v5.1 [##########] | v6.0 [##########]

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
| v6.0 | INCOME Phase | 21-23 | 7 | 2026-02-02 |

See `.planning/milestones/` for full archives.

## Accumulated Context

### Key Patterns (from v1-v6.0)

**Cython patterns:**
- Non-player phase pattern - 0 actions valid for deterministic phases (WRAP_UP pattern)
- Entity handle pattern for state access
- Phase handler pattern - cdef noexcept functions for zero overhead
- Low-level nogil accessor pattern - Offsets struct + get_offsets() + cdef inline accessors
- Player income calculation pattern - Player method returns sum of adjusted income from owned privates only (18-01)
- Pair counting pattern - i<j nested loop generates unique pairs without duplicates (21-01)
- Entity income calculation pattern - cpdef int calculate_income(GameState) for Corp/FI (22-01)
- Special ability dispatch pattern - if/elif chain using CorpIndices enum for clarity (22-02)
- Income flow separation pattern - calculate_income (pure) vs apply_income (mutation) (22-03)
- Negative cash rounding pattern - +0.5 for positive, -0.5 for negative (22-03)
- Bankruptcy delegation pattern - phases call Corp.go_bankrupt() instead of inline logic (23-01)
- Income application pattern - INCOME phase applies per-entity income with immediate bankruptcy check (23-02)
- Temporary phase consolidation pattern - TEMP_END_TURN consolidates end-of-turn bookkeeping (23-03)

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Integration test consolidation - Cross-phase tests in test_integration.py
- TDD RED-GREEN-REFACTOR - test commit, feat commit, refactor commit (optional)

### Pending Todos

None.

### Blockers/Concerns

**Remaining Game Phases after v6.0:** DIV (DIVIDENDS), END (END_CARD), ISS (ISSUE_SHARES), IPO (INITIAL_PUBLIC_OFFERING)

**TEMP_END_TURN is temporary:** Once all remaining phases are implemented, turn increment logic should move to the appropriate phase per game rules.

## Session Continuity

Last session: 2026-02-02
Stopped at: v6.0 milestone complete and archived
Resume file: None
Next action: Start v7.0 milestone with /gsd:new-milestone
