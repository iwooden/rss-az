# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-26)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** v5.0 CLOSING Phase

## Current Position

Milestone: v5.0 CLOSING Phase
Phase: 19 of 19 (Testing and Integration)
Plan: Ready to plan
Status: Phase 18 verified, ready to plan Phase 19
Last activity: 2026-01-27 — Phase 18 verified (3/3 criteria passed)

Progress: v1 [██████████] | v2 [██████████] | v2.1 [██████████] | v3.0 [██████████] | v4.0 [██████████] | v5.0 [██████████]

## Archived Milestones

| Version | Name | Phases | Plans | Shipped |
|---------|------|--------|-------|---------|
| v1 | Game State Init | 1 | 1 | 2026-01-20 |
| v2 | INVEST & BID_IN_AUCTION | 2-6 | 12 | 2026-01-21 |
| v2.1 | Forced Action Auto-Application | 7-8 | 3 | 2026-01-23 |
| v3.0 | WRAP_UP Phase | 9-11 + 10.1 | 6 | 2026-01-24 |
| v4.0 | ACQUISITION Phase | 12-15 | 13 | 2026-01-26 |

See `.planning/milestones/` for full archives.

## Accumulated Context

### Key Patterns (from v1-v4.0)

**Cython patterns:**
- Entity initialization order - Initialize all handles before setting state
- Module import pattern - Avoid circular imports with `from entities import X as X_module`
- Phase handler pattern - cdef noexcept functions for zero overhead
- Non-player phase pattern - 0 actions valid for deterministic phases
- Hybrid phase pattern - Non-player when no offers, player when offers exist
- While-loop re-query pattern for dynamic state iteration
- Acquisition zone pattern - Pending state during phase, merge at end
- One-hot encoding helpers pattern - cdef inline noexcept nogil functions on raw float* for zero overhead (15.1-01)
- Encoding helper application pattern - Replace inline one-hot loops with reusable helpers from encoding.pyx (15.1-04)
- Module-level buffer pattern - Pre-allocated static buffers with memset clearing for GIL-protected single-threaded operations (15.1-03)
- Phase dispatch pattern - Single helper function to deduplicate phase-based branching logic (15.1-03)
- Status code export pattern - Export Cython enum values as *_PY constants for Python access (15.1-05)
- Two-pass closing pattern - Identify companies to close, then close them to avoid state mutation during iteration (16-01)
- Temporary transition pattern - Document phase transitions that will change in future phases with TEMPORARY comments (16-02)
- Hidden buffer pattern - Pre-generate offers in hidden state buffer, present one at a time (17-01)
- Dynamic re-validation pattern - Validate offers at presentation time, not generation time, to handle state changes (17-02)
- Hybrid phase detection via state field - Use existing state field (closing_company == -1) to distinguish non-player/player modes (17-02)
- Player income calculation pattern - Player method returns sum of adjusted income from owned privates only (18-01)
- Mandatory close pattern - Iterate players by ID, close cheapest negative-income company until income + cash >= 0 (18-01)
- Phase-end protection pattern - Mandatory close before transition prevents bankruptcy in next phase (18-01)

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Integration test consolidation - Cross-phase tests in test_integration.py
- Invariant verification after every action
- Status code imports - Phase tests import from conftest.py, core tests import from core.driver (15.1-05)
- Python wrapper pattern - Expose internal state for testing (*_py functions) (17-03)
- Mandatory close test pattern - Set high CoO level, give negative-income companies, verify close order (18-02)
- Phase transition test pattern - Call apply_*_auto_py directly rather than driver loop (18-02)

### Pending Todos

None.

### Blockers/Concerns

**INCOME Phase temporary transition:** Phase 17 transitions CLOSING -> INVEST as temporary workaround until INCOME phase is implemented. Documented in _transition_to_income() function.

**nogil for mask functions (15.1-03):** Task 1 deferred - requires ~15 GameState accessor methods to have nogil versions. Will be addressed in Phase 15.1-04 as comprehensive GameState nogil refactoring.

**Remaining Phases After v5.0:** INC (INCOME), DIV (DIVIDENDS), END (END_GAME), ISS (ISSUE_SHARES), IPO (INITIAL_PUBLIC_OFFERING)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 003 | Principal engineer code review | 2026-01-26 | d06a4e6 | [003-principal-engineer-code-review](./quick/003-principal-engineer-code-review/) |

## Session Continuity

Last session: 2026-01-27
Stopped at: Completed 18-02-PLAN.md (Phase 18 complete)
Resume file: None
Next action: /gsd:discuss-phase 19
