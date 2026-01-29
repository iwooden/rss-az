---
phase: 22-income-calculation
plan: 02
subsystem: entities
tags: [cython, income, special-abilities, tdd]

# Dependency graph
requires:
  - phase: 22-01-base-income
    provides: Corporation.calculate_income foundation with synergy bonuses
provides:
  - Corporation special ability modifiers for PR, DA, S, VM
  - Complete income calculation formula ready for Phase 23 INCOME handler
affects: [23-income-phase-handler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD RED-GREEN cycle for entity method enhancement
    - Special ability dispatch pattern using CorpIndices enum

key-files:
  created: []
  modified:
    - entities/corp.pyx
    - tests/phases/test_income.py

key-decisions:
  - "VM ability applied BEFORE base calculation (modifies CoO, not final income)"
  - "DA ability uses printed income of highest FV company (not adjusted income)"
  - "S ability uses synergy_markers // 2 (integer division, rounds down)"
  - "PR ability simply adds company_count (no complex logic needed)"
  - "Non-income abilities (JS, OS, SM, SI) correctly ignored in calculate_income"

patterns-established:
  - "Special ability dispatch: if/elif chain using CorpIndices enum for clarity"
  - "Order matters: VM first (modifies CoO), then PR/DA/S (modify final income)"

# Metrics
duration: 3min
completed: 2026-01-29
---

# Phase 22 Plan 02: Corporation Special Abilities Summary

**Enhanced Corporation.calculate_income with special ability bonuses for PR, DA, S, and VM corporations using TDD**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-29T01:17:25Z
- **Completed:** 2026-01-29T01:20:43Z
- **Tasks:** 1 TDD task (2 commits: test, feat)
- **Files modified:** 2

## Accomplishments

- PR (Prussian Railway) receives +1 per company owned
- DA (Doppler AG) receives +printed income of highest FV company (doubles that company's income)
- S (Synergistic) receives +synergy_markers // 2 bonus
- VM (Vintage Machinery) reduces total CoO by up to 10 (applied before base calculation)
- All 8 special ability tests pass
- Full test suite passes (335 tests, up from 327)

## Task Commits

TDD RED-GREEN cycle completed:

1. **RED: Failing tests** - `77e7d74` (test)
   - 8 test cases for CSA-01 through CSA-04
   - Tests for PR (0 companies, 3 companies)
   - Tests for DA (multiple companies with different FVs)
   - Tests for S (4 markers, 5 markers - rounds down)
   - Tests for VM (CoO < 10, CoO > 10)
   - Test for non-income ability corps (JS)
   - Expected failures: 4 (PR, DA, S missing implementation)

2. **GREEN: Implementation** - `46eeb07` (feat)
   - Added CorpIndices and get_company_face_value imports
   - Enhanced calculate_income to track highest_fv and highest_fv_income
   - Applied VM ability first (reduces CoO before subtraction)
   - Applied PR/DA/S abilities after base calculation
   - Fixed test calculations to account for synergy bonuses
   - All 335 tests pass

**Plan metadata:** (this commit - docs)

_Note: REFACTOR phase skipped - implementation clean on first pass_

## Files Created/Modified

- `entities/corp.pyx` - Enhanced calculate_income with special ability dispatch logic
- `tests/phases/test_income.py` - Added TestCorpSpecialAbilities class with 8 test cases

## Decisions Made

**1. VM ability applied BEFORE base calculation**
- Rationale: VM modifies CoO itself, not the final income. Must be applied before CoO subtraction.
- Implementation: `if self.corp_id == CorpIndices.CORP_VM: total_coo = max(0, total_coo - 10)` before calculating total_income
- Impact: Correct order of operations per game rules

**2. DA ability uses printed income, not adjusted income**
- Rationale: Per RULES.md and CONTEXT.md, DA doubles PRINTED income (before CoO), not adjusted income
- Implementation: Track highest_fv_income during first pass before CoO subtraction
- Impact: Accurate bonus calculation matching game rules

**3. S ability uses synergy_markers // 2 (integer division)**
- Rationale: Per game rules, bonus is half the marker count, rounded down
- Implementation: `total_income += synergy_markers // 2`
- Impact: 4 markers -> +2, 5 markers -> +2 (correct rounding behavior)

**4. If/elif chain pattern for special abilities**
- Rationale: Clear, explicit dispatch that's easy to verify against game rules
- Alternative considered: Dictionary dispatch (rejected - less clear in Cython context)
- Impact: Readable code that matches RULES.md structure directly

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Ready for Phase 23:** INCOME phase handler
- All income calculation logic complete (base + synergy + abilities)
- Corp.calculate_income and FI.calculate_income fully tested
- Player.get_income already implemented (Phase 18)
- Phase 23 just needs to call these methods and update state

**Testing coverage:**
- 8 synergy tests (Phase 21)
- 7 base income tests (Phase 22-01)
- 8 special ability tests (Phase 22-02)
- Total: 23 income calculation tests

**No blockers** - implementation complete and tested

---
*Phase: 22-income-calculation*
*Completed: 2026-01-29*
