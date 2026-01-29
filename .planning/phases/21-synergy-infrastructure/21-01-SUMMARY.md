---
phase: 21-synergy-infrastructure
plan: 01
subsystem: corporation-income
tags: [synergy, cython, nogil, tdd]
requires:
  - 20-03 # nogil accessor patterns
provides:
  - compute_synergy_bonuses # Core function for corporation synergy calculation
affects:
  - 22 # Will be used for corporation income calculation
tech-stack:
  added: []
  patterns:
    - "Pair counting with i<j loop to count each pair once"
    - "Bidirectional synergy summing (A->B + B->A)"
key-files:
  created:
    - tests/phases/test_income.py
  modified:
    - core/data.pyx
decisions:
  - id: SYN-01
    choice: "Use i<j loop pattern for pair counting"
    rationale: "Ensures each pair counted exactly once per RULES.md line 569"
    alternatives: "Track seen pairs in set (slower, needs GIL)"
  - id: SYN-02
    choice: "Sum bonuses from both directions for each pair"
    rationale: "A->B and B->A are separate bonuses per game rules"
    alternatives: "Max of two directions (incorrect per rules)"
metrics:
  duration: "2m 49s"
  completed: "2026-01-29"
---

# Phase 21 Plan 01: Synergy Pair Calculation Summary

**One-liner:** nogil synergy pair counter for corporation income with bidirectional bonus summing

## What Was Built

Implemented `compute_synergy_bonuses` function that takes an array of company IDs and returns both total synergy income and number of synergy marker pairs for a corporation.

**Key implementation details:**
- `cdef inline (int, int) compute_synergy_bonuses(int* company_ids, int num_companies) noexcept nogil`
- Returns tuple: (total_income, marker_count)
- Uses i<j loop pattern to count each pair exactly once
- Sums bonuses from both directions (A->B and B->A) for each pair
- Marker count increments only if at least one direction has synergy

**Test coverage:**
- 8 comprehensive test cases in `TestSynergyCalculation` (tests/phases/test_income.py)
- Covers: 0 companies, 1 company, 2 companies (various synergy patterns), 3+ companies
- Tests order independence (CDG,MAD vs MAD,CDG give same result)
- All 320 tests pass

## Technical Foundation

### nogil Safety Pattern

```cython
cdef inline (int, int) compute_synergy_bonuses(
    int* company_ids,
    int num_companies
) noexcept nogil:
    # Direct COMPANY_SYNERGY array access, no Python calls
    # Pure C-level computation
```

This function can be called from tight loops during income calculation without GIL overhead.

### Pair Counting Algorithm

```cython
for i in range(num_companies):
    for j in range(i + 1, num_companies):  # i<j ensures each pair once
        has_synergy = 0

        # Check both directions
        bonus_a_to_b = COMPANY_SYNERGY[company_ids[i]][company_ids[j]]
        if bonus_a_to_b > 0:
            total_income += bonus_a_to_b
            has_synergy = 1

        bonus_b_to_a = COMPANY_SYNERGY[company_ids[j]][company_ids[i]]
        if bonus_b_to_a > 0:
            total_income += bonus_b_to_a
            has_synergy = 1

        if has_synergy:
            marker_count += 1
```

**Why this works:**
- Outer loop: i from 0 to N-1
- Inner loop: j from i+1 to N
- This generates all unique pairs: (0,1), (0,2), ..., (0,N-1), (1,2), (1,3), ..., (N-2,N-1)
- Total pairs = N*(N-1)/2 (combinations formula)

### Example Calculation

Corporation owns: [DR, WT, BY]

**Synergy data:**
- DR→WT: 2
- DR→BY: 2
- BY→WT: 2

**Pair enumeration:**
1. i=0 (DR), j=1 (WT): DR→WT=2, WT→DR=0 → income +2, markers +1
2. i=0 (DR), j=2 (BY): DR→BY=2, BY→DR=0 → income +2, markers +1
3. i=1 (WT), j=2 (BY): WT→BY=0, BY→WT=2 → income +2, markers +1

**Result:** (6, 3) ✓

## Decisions Made

### SYN-01: Pair Counting Strategy

**Decision:** Use i<j nested loop pattern

**Context:** Need to count each company pair exactly once per RULES.md line 569

**Options considered:**
1. ✅ **i<j loop pattern** - Simple, fast, no memory allocation
2. ❌ Set-based tracking - Requires GIL, slower, more complex

**Rationale:** The i<j pattern is a standard algorithm for generating unique pairs. It's provably correct (generates exactly C(n,2) pairs), requires no additional memory, and stays nogil-safe.

### SYN-02: Bidirectional Income Summing

**Decision:** Sum bonuses from both directions (A→B + B→A)

**Context:** Game rules specify synergies are directional

**Options considered:**
1. ✅ **Sum both directions** - Matches RULES.md semantics
2. ❌ Take max of two directions - Simpler but incorrect
3. ❌ Only count initiating direction - Loses half the income

**Rationale:** Testing revealed that synergies in Rolling Stock Stars are **asymmetric**. CDG→MAD gives 16, but MAD→CDG gives 0. The game rules count both directions' bonuses when computing corporation income, so we must check both.

### SYN-03: Test Data Discovery

**Issue encountered:** Initial test assumed E↔BR was bidirectional (both give 8)

**Reality check:** Inspection of COMPANY_SYNERGY data showed:
- E→BR: 8 ✓
- BR→E: 0 ✗

**Systematic search:** Python script verified NO bidirectional synergies exist in the game

**Resolution:** Updated test to use DR/PKP asymmetric pair, renamed test case to reflect reality

**Impact:** This validates the importance of summing both directions - if we only checked one way, we'd miss legitimate income.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test data assumption incorrect**

- **Found during:** Task 1 (RED phase)
- **Issue:** Test assumed E↔BR was bidirectional synergy (both 8)
- **Fix:** Changed test to use DR→PKP (4), PKP→DR (0) asymmetric case
- **Files modified:** tests/phases/test_income.py
- **Commit:** 553e431 (part of feat commit, not separate)
- **Reasoning:** Discovered during test execution that no bidirectional synergies exist in actual game data

## Verification Results

✅ **Build succeeds:** `python3 setup.py build_ext --inplace` (exit 0)

✅ **Synergy tests pass:** `pytest tests/phases/test_income.py -v` (8/8 passed)

✅ **Full test suite passes:** `pytest tests/ -v` (320/320 passed)

✅ **Function is nogil-safe:** grep confirms `noexcept nogil` signature

✅ **py_compute_synergy_bonuses importable:** Python wrapper available for testing

## Future Integration Points

**Phase 22 (Corporation Income):** Will call `compute_synergy_bonuses` during income calculation:

```cython
# Pseudocode for future Phase 22
cdef int calculate_corp_income(Corporation* corp) noexcept nogil:
    cdef int base_income = sum_company_incomes(corp.companies, corp.num_companies)
    cdef int synergy_income, synergy_markers
    synergy_income, synergy_markers = compute_synergy_bonuses(
        corp.companies,
        corp.num_companies
    )

    # Synergistic ability: +1 per synergy marker
    if corp.ability == SYNERGISTIC:
        base_income += synergy_markers

    return base_income + synergy_income
```

**Phase 23 (INCOME Phase):** Uses corporation income to update player/corp cash

## Next Phase Readiness

**Ready to proceed:** ✅

**Blockers:** None

**Prerequisites for Phase 22:**
- ✅ COMPANY_SYNERGY matrix (exists since v1)
- ✅ get_company_synergy accessor (exists)
- ✅ compute_synergy_bonuses function (implemented this plan)

**Concerns:** None - implementation is straightforward and well-tested

## Task Breakdown

| Task | Type | Commit | Files |
|------|------|--------|-------|
| 1. RED - Write failing tests | test | b04b3a1 | tests/phases/test_income.py |
| 2. GREEN - Implement function | feat | 553e431 | core/data.pyx, tests/phases/test_income.py (fix) |

**TDD cycle notes:**
- RED: ImportError for py_compute_synergy_bonuses (expected)
- GREEN: All 8 tests pass, 320 total tests pass
- REFACTOR: Not needed - implementation was clean on first pass

## Learnings

1. **Game data validation matters:** Always verify test assumptions against actual data
2. **Asymmetric relationships:** Rolling Stock Stars synergies are NOT symmetric - this validates the need to check both directions
3. **i<j pattern:** Standard algorithm for unique pair generation works perfectly for nogil Cython
4. **TDD caught the bug:** Writing tests first revealed the bidirectional assumption before it could cause issues

---

**Status:** ✅ Complete
**Phase Progress:** 1/1 plans complete
**Next:** Phase 22 - Corporation Income Calculation
