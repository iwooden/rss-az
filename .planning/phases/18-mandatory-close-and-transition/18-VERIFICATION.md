---
phase: 18-mandatory-close-and-transition
verified: 2026-01-27T21:53:59Z
status: passed
score: 10/10 must-haves verified
---

# Phase 18: Mandatory Close and Transition Verification Report

**Phase Goal:** Auto-close at phase end protects players from negative cash, then transition to INCOME
**Verified:** 2026-01-27T21:53:59Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Players with negative total income have privates auto-closed until income non-negative | ✓ VERIFIED | `_process_mandatory_close()` iterates players, closes cheapest negative-income companies while `income + cash < 0` (lines 132-164) |
| 2 | Cheapest negative-income company closed first during mandatory closing | ✓ VERIFIED | Loop finds `cheapest_fv` by comparing face values (lines 141-156), calls `_close_player_company()` on cheapest (line 164) |
| 3 | Phase transitions to INCOME when all offers processed and mandatory closes complete | ✓ VERIFIED | `_present_next_close_offer()` calls `_process_mandatory_close()` then `_transition_to_income()` when no more offers (lines 450-453) |

**Score:** 3/3 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `entities/player.pyx` | get_income() method | ✓ VERIFIED | Lines 444-463: Calculates sum of adjusted income from owned companies (470 lines total) |
| `entities/player.pxd` | get_income() declaration | ✓ VERIFIED | Line 132: `cpdef int get_income(self, GameState state)` |
| `phases/closing.pyx` | _process_mandatory_close() | ✓ VERIFIED | Lines 112-164: Mandatory close logic (721 lines total) |
| `phases/closing.pyx` | _close_player_company() | ✓ VERIFIED | Lines 88-109: Player company close with JS bonus |
| `phases/closing.pyx` | Integration into transition | ✓ VERIFIED | Line 452: Called before `_transition_to_income()` |
| `phases/closing.pyx` | Python test wrapper | ✓ VERIFIED | Lines 719-721: `process_mandatory_close_py()` |
| `tests/test_mandatory_close.py` | Test coverage | ✓ VERIFIED | 14 tests covering all requirements (261 lines) |

**Artifacts:** 7/7 verified

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `phases/closing.pyx` | `entities/player.pyx` | `get_income()` call | ✓ WIRED | Line 135: `player_module.PLAYERS[player_id].get_income(state)` |
| `phases/closing.pyx` | `_close_player_company()` | Direct call | ✓ WIRED | Line 164: `_close_player_company(state, cheapest_company, player_id)` |
| `phases/closing.pyx` | `_process_mandatory_close()` | Phase transition | ✓ WIRED | Line 452: Called before `_transition_to_income()` when offers exhausted |
| `_process_mandatory_close()` | Face value comparison | `get_company_face_value()` | ✓ WIRED | Line 153: Used to find cheapest company |
| `_process_mandatory_close()` | Negative income check | `get_adjusted_company_income()` | ✓ WIRED | Line 150: Filters to only negative-income companies |
| `_close_player_company()` | JS bonus | `corp_module.CORPS[0]` | ✓ WIRED | Lines 105-106: Applies 2x printed income bonus |
| `tests/test_mandatory_close.py` | `process_mandatory_close_py()` | Import and call | ✓ WIRED | Line 16 import, 7 test calls |

**Links:** 7/7 wired correctly

### Requirements Coverage

| Requirement | Status | Supporting Evidence |
|-------------|--------|-------------------|
| CLO-14: Auto-close player private companies if total income would cause negative cash | ✓ SATISFIED | Implementation: lines 112-164 in closing.pyx. Tests: `test_mandatory_close_triggered_negative_total`, `test_mandatory_close_multiple_companies` (lines 96-160) |
| CLO-15: Close cheapest negative-income company first when mandatory closing | ✓ SATISFIED | Implementation: lines 141-156 find cheapest by face value. Test: `test_mandatory_close_cheapest_first` (lines 116-140) |
| CLO-16: Phase transitions to INCOME when all offers processed and mandatory closes complete | ✓ SATISFIED | Implementation: lines 450-453 call mandatory close then transition. Tests: `test_phase_transitions_after_mandatory_close`, `test_closing_flow_with_mandatory_close_triggered` (lines 220-261) |

**Coverage:** 3/3 requirements satisfied

### Anti-Patterns Found

None found. Code quality is high.

**Checked files:**
- `entities/player.pyx` (470 lines) - No TODOs, stubs, or placeholders
- `phases/closing.pyx` (721 lines) - No TODOs, stubs, or placeholders  
- `tests/test_mandatory_close.py` (261 lines) - No TODOs, stubs, or placeholders

### Test Execution

```
pytest tests/test_mandatory_close.py -v --tb=short

14 tests PASSED:

TestPlayerIncome (5 tests):
- test_get_income_no_companies
- test_get_income_single_company
- test_get_income_multiple_companies
- test_get_income_negative_company
- test_get_income_excludes_corp_subsidiaries

TestMandatoryClose (7 tests):
- test_mandatory_close_not_triggered_positive_total
- test_mandatory_close_triggered_negative_total
- test_mandatory_close_cheapest_first
- test_mandatory_close_multiple_companies
- test_mandatory_close_js_bonus
- test_mandatory_close_only_negative_income_companies
- test_mandatory_close_player_order

TestClosingPhaseTransition (2 tests):
- test_phase_transitions_after_mandatory_close
- test_closing_flow_with_mandatory_close_triggered
```

## Detailed Verification

### Truth 1: Players with negative total income have privates auto-closed

**Implementation:** `phases/closing.pyx` lines 112-164

```cython
cdef void _process_mandatory_close(GameState state) noexcept:
    for player_id in range(state._num_players):
        while True:
            income = player_module.PLAYERS[player_id].get_income(state)
            cash = player_module.PLAYERS[player_id].get_cash(state)
            
            if income + cash >= 0:
                break  # Player is safe
            
            # Find and close cheapest negative-income company
            ...
            _close_player_company(state, cheapest_company, player_id)
```

**Evidence:**
- Iterates all players (line 132)
- Checks `income + cash < 0` (line 138)
- Closes companies until condition satisfied (lines 134-164)
- Stops when `income + cash >= 0` (lines 138-139)

**Test coverage:**
- `test_mandatory_close_triggered_negative_total` - Verifies close happens when total negative
- `test_mandatory_close_multiple_companies` - Verifies iterates until safe
- `test_mandatory_close_not_triggered_positive_total` - Verifies no close when safe

### Truth 2: Cheapest negative-income company closed first

**Implementation:** `phases/closing.pyx` lines 141-164

```cython
# Find cheapest negative-income company owned by player
cheapest_company = -1
cheapest_fv = 999999  # Large sentinel

for company_id in range(GameConstants.NUM_COMPANIES):
    if not player_module.PLAYERS[player_id].owns_company(state, company_id):
        continue
    
    # Check if negative income
    if get_adjusted_company_income(company_id, coo_level) >= 0:
        continue
    
    fv = get_company_face_value(company_id)
    if fv < cheapest_fv:
        cheapest_fv = fv
        cheapest_company = company_id

# Close the company (CLO-15: cheapest first)
_close_player_company(state, cheapest_company, player_id)
```

**Evidence:**
- Scans all owned companies (lines 145-146)
- Filters to negative income only (lines 149-151)
- Compares face values to find minimum (lines 153-156)
- Closes cheapest found (line 164)

**Test coverage:**
- `test_mandatory_close_cheapest_first` - Gives player two negative-income companies with different face values ($1 and $3), verifies $1 company closed first

### Truth 3: Phase transitions after mandatory close

**Implementation:** `phases/closing.pyx` lines 450-453

```cython
# No more valid offers - process mandatory close then transition
turn_module.TURN.clear_closing_company(state)
_process_mandatory_close(state)  # CLO-14, CLO-15: mandatory close before transition
_transition_to_income(state)
```

**Evidence:**
- Called in `_present_next_close_offer()` when no more offers
- Explicit comment documenting purpose (line 452)
- Mandatory close runs before transition (line 452)
- Transition to INCOME follows (line 453)

**Test coverage:**
- `test_phase_transitions_after_mandatory_close` - Verifies phase changes to INVEST after auto-close
- `test_closing_flow_with_mandatory_close_triggered` - Integration test showing full flow: offers → pass → mandatory close → transition

### Artifact Quality Check

**Player.get_income() (entities/player.pyx lines 444-463):**
- Length: 20 lines (substantive)
- No stub patterns (no TODO, placeholder, empty returns)
- Has export: `cpdef int get_income` (line 444)
- Imported/used: Called in closing.pyx line 135
- Logic: Iterates owned companies, sums adjusted income (base - CoO)
- Excludes corp subsidiaries correctly (only checks `self.owns_company()`)

**_process_mandatory_close() (phases/closing.pyx lines 112-164):**
- Length: 53 lines (substantive)
- No stub patterns
- Clear logic: Nested loop structure (players → while negative → scan companies)
- Edge case handling: Breaks if no negative-income companies found (line 158)
- CoO level captured once at start (line 129) - correct per requirements

**_close_player_company() (phases/closing.pyx lines 88-109):**
- Length: 22 lines (substantive)
- No stub patterns
- Steps: Clear ownership → JS bonus → remove from game
- JS bonus correctly applied: 2x printed income to corp 0 if active (lines 105-106)

**Tests (tests/test_mandatory_close.py):**
- Length: 261 lines across 14 tests
- No stub patterns
- Comprehensive coverage: income calculation (5 tests), mandatory close logic (7 tests), phase transition (2 tests)
- All tests pass

## Summary

Phase 18 goal **ACHIEVED**. All three observable truths verified in implementation and tests.

**Implementation quality:**
- All artifacts exist and are substantive (no stubs)
- All key links properly wired
- Clean code with no anti-patterns
- Comprehensive test coverage (14 tests, all passing)

**Requirements satisfied:**
- CLO-14: Auto-close on negative total income ✓
- CLO-15: Cheapest-first close order ✓
- CLO-16: Phase transition after mandatory close ✓

**Key implementation details confirmed:**
1. `Player.get_income()` calculates sum of adjusted income from owned privates only
2. Mandatory close iterates players, closes cheapest negative-income companies until safe
3. CoO level fixed at phase start (no re-evaluation during loop)
4. Junkyard Scrappers bonus (2x printed income) applies to mandatory closes
5. Players can end with zero companies (no minimum retention)
6. Mandatory close executes before phase transition to INCOME

**Phase ready to proceed:** No gaps, no blockers, all tests passing.

---

_Verified: 2026-01-27T21:53:59Z_
_Verifier: Claude (gsd-verifier)_
