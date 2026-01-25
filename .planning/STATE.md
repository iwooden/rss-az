# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-24)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** v4.0 ACQUISITION Phase - Phase 13 (Actions & Validation)

## Current Position

Milestone: v4.0 ACQUISITION Phase
Phase: 13 of 15 (Actions & Validation)
Plan: 02 of 02
Status: Phase complete
Last activity: 2026-01-25 - Completed 13-02-PLAN.md (action handler integration)

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##########] | v4.0 [#####     ] 33%

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |
| v3.0 | WRAP_UP Phase | 9-11 + 10.1 | 6 | 2026-01-24 |

See `.planning/milestones/` for full archives.

## Current Milestone: v4.0 ACQUISITION Phase

**Goal:** Implement ACQUISITION phase with AlphaZero-optimized mechanics

**Phases:**
- Phase 12: Offer Infrastructure (9 requirements) ✓
- Phase 13: Actions & Validation (10 requirements) ✓
- Phase 14: Flow & Integration (10 requirements)
- Phase 15: Testing (7 requirements)

**Key design decisions:**
- Same-president trade restriction (no inter-player negotiation)
- Offer-based flow with sorted priority presentation
- Receivership auto-buy integrated into offer loop
- Acquisition proceeds zone prevents re-acquisition loops

## Accumulated Context

### Key Patterns (from v1-v3.0)

**Cython patterns:**
- Entity initialization order - Initialize all handles before setting state
- Module import pattern - Avoid circular imports with `from entities import X as X_module`
- Phase handler pattern - cdef noexcept functions for zero overhead
- Non-player phase pattern - 0 actions valid for deterministic phases
- While-loop re-query pattern for dynamic state iteration

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Integration test consolidation - Cross-phase tests in test_integration.py

### Recent Decisions

| ID | Decision | Plan | Impact |
|----|----------|------|--------|
| offer-buffer-size | 250 offer slots (500 floats) | 12-01 | Hidden state buffer for pre-computed offers |
| acquisition-proceeds-normalization | Use CASH_DIVISOR | 12-01 | Consistent with other cash fields |
| use-is-president-of | Use player.is_president_of() for president lookup | 12-02 | Cleaner API than share counting |
| os-first-priority | OS->FI offers always appear first | 12-02 | OS has special acquisition rights per rules |
| president-detection-method | Find president by max share count | 12-03 | Simpler than is_president_of lookup |
| validation-skip-strategy | While-loop skip in _present_current_offer | 12-03 | Auto-skip invalid offers during presentation |
| receivership-active-player | Set active_player to 0 when president is -1 | 12-03 | Safe fallback for receivership corps |
| restore-missing-functions | Restored presentation functions from git history | 12-04 | Functions lost in commit bdba0d1 (12-02 overwrote 12-03) |
| python-wrapper-for-testing | Added apply_wrap_up_py wrapper | 12-04 | Enable integration tests for cdef functions |
| corp-acquisition-proceeds-pattern | Corp sellers use get+set pattern for acquisition_proceeds | 13-01 | Corp entity lacks add_acquisition_proceeds method |
| valid-06-guaranteed | Same-president guaranteed by offer generation | 13-01 | No runtime re-check needed in validation |
| validation-test-deferral | Test scaffolding now, implementation in 13-02 | 13-01 | Validation functions are cdef, need Python wrapper first |
| acquisition-hybrid-phase | ACQUISITION is non-player when no offers, player when offers exist | 13-02 | Prevents ZeroLegalActionsError on empty buffer |
| action-constants-exposure | Action enums exposed as *_PY suffixed constants | 13-02 | Enables Python test imports |
| company-location-constants | Fixed LOC_PLAYER=3, LOC_FI=4, LOC_CORP=5 | 13-02 | Corrected from wrong values (1,0,2) |

### Pending Todos

None.

### Blockers/Concerns

None - Phase 13 complete. Ready for Phase 14 (Flow & Integration).

## Session Continuity

Last session: 2026-01-25
Stopped at: Completed 13-02-PLAN.md
Resume file: None
Next action: Proceed to Phase 14 planning
