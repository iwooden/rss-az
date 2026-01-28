# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-28)

**Core value:** Fast, reproducible game simulation for AI training with full rules compliance
**Current focus:** Planning next milestone (v6.0)

## Current Position

Milestone: v5.1 complete, planning v6.0
Phase: Not started
Plan: Not started
Status: Ready to plan
Last activity: 2026-01-28 — v5.1 milestone archived

Progress: v1 [##########] | v2 [##########] | v2.1 [##########] | v3.0 [##########] | v4.0 [##########] | v5.0 [##########] | v5.1 [##########]

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
- Exception co-location pattern - Define exceptions in the module that uses them rather than separate modules (quick-004)
- Low-level nogil accessor pattern - Offsets struct + get_offsets() + cdef inline accessors on raw float* for GIL-free state access (20-01)
- Mask refactoring pattern - Compute offsets once at function start, use raw pointer accessors throughout to eliminate cpdef calls (20-02)
- Inline nogil accessor pattern - cdef inline return_type func_nogil(...) noexcept nogil wraps state access for GIL-free calls (20-03)

**Testing patterns:**
- Per-task atomic commits - feat/test prefixes for git bisect
- Integration test consolidation - Cross-phase tests in test_integration.py
- Invariant verification after every action
- Status code imports - Phase tests import from conftest.py, core tests import from core.driver (15.1-05)
- Python wrapper pattern - Expose internal state for testing (*_py functions) (17-03)
- Mandatory close test pattern - Set high CoO level, give negative-income companies, verify close order (18-02)
- Phase transition test pattern - Call apply_*_auto_py directly rather than driver loop (18-02)
- Edge case test pattern - Boundary condition tests with explicit assertions in TestClosingEdgeCases class (19-01)
- Parameterized player count testing - Use @pytest.mark.parametrize("num_players", [3, 6]) for phase verification (19-01)
- Company ownership in tests - Use COMPANIES[x].transfer_to_player() for proper state, not PLAYERS[x].set_owns_company() (19-02)
- Full turn cycle testing - Track turn number increment after CLOSING->INVEST transition (19-02)

### Pending Todos

None.

### Blockers/Concerns

**INCOME Phase temporary transition:** Phase 17 transitions CLOSING -> INVEST as temporary workaround until INCOME phase is implemented. Documented in _transition_to_income() function.

**Remaining Game Phases for v6.0:** INC (INCOME), DIV (DIVIDENDS), END (END_GAME), ISS (ISSUE_SHARES), IPO (INITIAL_PUBLIC_OFFERING)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 003 | Principal engineer code review | 2026-01-26 | d06a4e6 | [003-principal-engineer-code-review](./quick/003-principal-engineer-code-review/) |
| 004 | Remove unused src/ directory | 2026-01-28 | 923b163 | [004-investigate-unused-exceptions-in-src](./quick/004-investigate-unused-exceptions-in-src/) |
| 005 | Fix JS scrapping bonus bug | 2026-01-28 | 2b35acf | [005-fix-js-scrapping-bonus-bug](./quick/005-fix-js-scrapping-bonus-bug/) |

## Session Continuity

Last session: 2026-01-28 23:30:00Z
Stopped at: v5.1 milestone archived
Resume file: None
Next action: /gsd:new-milestone to start v6.0
