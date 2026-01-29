---
phase: 22-income-calculation
plan: 03
subsystem: game-logic
tags: [cython, income, cash-management, float-normalization, testing]

# Dependency graph
requires:
  - phase: 22-01
    provides: calculate_income methods for Corp and FI
  - phase: 22-02
    provides: Corporation special ability logic in calculate_income
provides:
  - apply_income methods for Corp and FI wrapping add_cash
  - Income application tests verifying positive/negative cash changes
  - Bug fix: Corporation.get_cash rounding for negative values
affects: [23-income-phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Income flow separation: calculate_income (pure) vs apply_income (mutation)"
    - "Entity income application pattern: Corp/FI use apply_income(), Player uses add_cash() directly"
    - "Negative cash rounding pattern: +0.5 for positive, -0.5 for negative"

key-files:
  created: []
  modified:
    - entities/corp.pyx
    - entities/corp.pxd
    - entities/fi.pyx
    - entities/fi.pxd
    - tests/phases/test_income.py

key-decisions:
  - "apply_income wraps add_cash to separate income calculation (pure) from cash mutation (side effect)"
  - "Player doesn't need apply_income - Phase 23 will call player.add_cash(player.get_income()) directly"
  - "Fixed Corporation.get_cash to properly round negative values (-0.5 for negative, +0.5 for positive)"

patterns-established:
  - "Income application pattern: entity.apply_income(state, entity.calculate_income(state))"

# Metrics
duration: 3min
completed: 2026-01-29
---

# Phase 22 Plan 03: Income Application Summary

**Income application methods with proper negative cash handling for corporation bankruptcy support**

## Performance

- **Duration:** 3 min
- **Started:** 2026-01-29T01:22:48Z
- **Completed:** 2026-01-29T01:25:48Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Income application methods separate calculation from mutation
- All Phase 22 income requirements (INC-01 through INC-05, SYN-03, CSA-01 through CSA-04) implemented and tested
- Bug fix enables corporation bankruptcy (negative cash allowed)
- Full test suite passes: 340 tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement apply_income methods** - `513512f` (feat)
2. **Task 2: Write income application tests** - `bf2eecc` (test)
3. **Task 3: Final verification and cleanup** - `2ebf8c8` (docs)

## Files Created/Modified
- `entities/corp.pyx` - Added apply_income() method, fixed get_cash() negative rounding
- `entities/corp.pxd` - Declared apply_income() method
- `entities/fi.pyx` - Added apply_income() method
- `entities/fi.pxd` - Declared apply_income() method
- `tests/phases/test_income.py` - Added TestIncomeApplication class with 5 tests, updated docstring

## Decisions Made

**1. Income flow separation pattern**
- Rationale: calculate_income (pure function, testable) vs apply_income (side effect, simple wrapper)
- Benefits: Clear separation of concerns, easier testing, matches existing codebase patterns

**2. Player doesn't get apply_income method**
- Rationale: Player already has get_income() and add_cash() - no need for wrapper
- Phase 23 will call: `player.add_cash(state, player.get_income(state))`

**3. Corporation.get_cash rounding fix**
- Rationale: +0.5 rounding breaks for negative values (rounds toward zero instead of nearest)
- Fix: Conditional rounding (+0.5 for positive, -0.5 for negative)
- Impact: Enables corporation bankruptcy (negative cash allowed per CONTEXT.md)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Corporation.get_cash rounding for negative values**
- **Found during:** Task 2 (test_corp_can_go_negative test)
- **Issue:** get_cash() used +0.5 rounding for all values, causing incorrect negative rounding (-5 became -4)
- **Root cause:** `<int>(value * DIVISOR + 0.5)` rounds toward zero for negatives
- **Fix:** Conditional rounding: `if val >= 0: +0.5 else: -0.5`
- **Files modified:** entities/corp.pyx (lines 89-95, 183-188)
- **Verification:** test_corp_can_go_negative now passes (cash=-5 correctly returned)
- **Committed in:** bf2eecc (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix essential for corporation bankruptcy support (Phase 23 requirement). No scope creep.

## Issues Encountered

**Cython method declaration requirement**
- Issue: Adding apply_income() to .pyx without .pxd declaration caused compile error
- Resolution: Added cpdef declarations to entities/corp.pxd and entities/fi.pxd
- Learning: All cpdef methods must be declared in .pxd interface file

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 23 (INCOME Phase Handler):**
- ✓ Corporation.calculate_income() returns income with all modifiers
- ✓ Corporation.apply_income() applies income to cash
- ✓ ForeignInvestor.calculate_income() returns income with +5 bonus
- ✓ ForeignInvestor.apply_income() applies income to cash
- ✓ Player.get_income() returns income from privates
- ✓ All entities support negative income/cash flow
- ✓ Full test coverage: synergy, base income, special abilities, application

**Phase 23 implementation pattern:**
```cython
# For each corporation:
income = corp.calculate_income(state)
corp.apply_income(state, income)

# For FI:
income = FI.calculate_income(state)
FI.apply_income(state, income)

# For each player:
income = player.get_income(state)
player.add_cash(state, income)
```

**No blockers.** Phase 23 can implement INCOME phase handler with deterministic 0-action pattern (WRAP_UP pattern).

---
*Phase: 22-income-calculation*
*Completed: 2026-01-29*
