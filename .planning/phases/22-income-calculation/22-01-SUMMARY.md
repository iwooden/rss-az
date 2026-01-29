---
phase: 22-income-calculation
plan: 01
subsystem: entities
tags: [cython, income, synergy, cost-of-ownership, tdd]

# Dependency graph
requires:
  - phase: 21-synergy-infrastructure
    provides: compute_synergy_bonuses function for pair counting
provides:
  - Corporation.calculate_income method with synergy bonuses
  - ForeignInvestor.calculate_income method with +5 bonus
  - Foundation for Phase 23 INCOME phase handler
affects: [22-02-corporation-abilities, 23-income-phase-handler]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Entity income calculation method pattern (calculate_income)
    - TDD: RED-GREEN-REFACTOR for entity methods

key-files:
  created: []
  modified:
    - entities/corp.pyx
    - entities/corp.pxd
    - entities/fi.pyx
    - entities/fi.pxd
    - core/data.pxd
    - tests/phases/test_income.py

key-decisions:
  - "Corporation income includes synergy bonuses from Phase 21"
  - "ForeignInvestor receives fixed +5 bonus per game rules"
  - "Special abilities deferred to Phase 22-02 per plan scope"

patterns-established:
  - "Income calculation method pattern: cpdef int calculate_income(self, GameState state)"
  - "TDD commit pattern: test commit (RED) → feat commit (GREEN) → docs commit (metadata)"

# Metrics
duration: 5min
completed: 2026-01-29
---

# Phase 22 Plan 01: Base Income Calculation Summary

**Corporation and ForeignInvestor income methods with synergy integration and CoO deductions using TDD**

## Performance

- **Duration:** 5 min
- **Started:** 2026-01-29T01:09:07Z
- **Completed:** 2026-01-29T01:13:59Z
- **Tasks:** 1 TDD task (3 commits: test, feat, docs)
- **Files modified:** 6

## Accomplishments

- Corporation.calculate_income returns base income - CoO + synergy bonuses
- ForeignInvestor.calculate_income returns base income - CoO + 5 fixed bonus
- All 7 new income tests pass (Corp: 4 tests, FI: 3 tests)
- Full test suite passes (327 tests)

## Task Commits

Each TDD phase was committed atomically following RED-GREEN pattern:

1. **RED: Failing tests** - `69055eb` (test)
   - 4 Corporation income tests
   - 3 ForeignInvestor income tests
   - Fixed initial CoO level assertion (starts at 1, not 0)

2. **GREEN: Implementation** - `9876382` (feat)
   - Corporation.calculate_income with synergy integration
   - ForeignInvestor.calculate_income with +5 bonus
   - Exposed compute_synergy_bonuses in core/data.pxd
   - Added method declarations in corp.pxd and fi.pxd

**Plan metadata:** (this commit - docs)

_Note: REFACTOR phase skipped - implementation was clean on first pass_

## Files Created/Modified

- `entities/corp.pyx` - Added calculate_income method with synergy bonuses
- `entities/corp.pxd` - Added calculate_income declaration
- `entities/fi.pyx` - Added calculate_income method with +5 bonus
- `entities/fi.pxd` - Added calculate_income declaration
- `core/data.pxd` - Exposed compute_synergy_bonuses for entity modules
- `tests/phases/test_income.py` - Added TestCorpBaseIncome and TestFIIncome classes

## Decisions Made

**1. Expose compute_synergy_bonuses via cdef declaration**
- Rationale: Function is `cdef inline` in data.pyx, not accessible from other .pyx files without declaration in .pxd
- Impact: Enables corp.pyx to call synergy calculation without code duplication

**2. Fixed initial CoO level test assertion**
- Found: Game starts at CoO level 1, not 0 (per initialize_game behavior)
- Impact: Test assertions now match actual game state initialization

**3. Deferred special abilities to Phase 22-02**
- Rationale: Plan explicitly scopes this task to base income only
- Implementation includes clear comments noting special abilities are deferred
- Impact: Clean separation of concerns, Phase 22-02 will extend this foundation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**1. Compilation error: compute_synergy_bonuses not accessible**
- Problem: Function is `cdef inline` in data.pyx, no declaration in data.pxd
- Solution: Added `cdef (int, int) compute_synergy_bonuses(int*, int) noexcept nogil` to data.pxd
- Resolution: Build succeeded, tests passed

**2. Method declaration missing in .pxd files**
- Problem: Cython requires cpdef method declarations in .pxd when class is used across modules
- Solution: Added `cpdef int calculate_income(self, GameState state)` to corp.pxd and fi.pxd
- Resolution: Build succeeded, all tests pass

## Next Phase Readiness

**Ready for Phase 22-02:** Corporation special abilities
- Corporation.calculate_income foundation in place
- Synergy bonuses correctly integrated
- Test infrastructure ready for special ability test cases

**Pattern established for Phase 23:** INCOME phase handler
- Entity income calculation methods follow Player.get_income pattern
- Can be called during phase execution to compute and apply income

**No blockers** - implementation complete and tested

---
*Phase: 22-income-calculation*
*Completed: 2026-01-29*
