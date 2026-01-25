---
phase: 13
plan: 02
subsystem: acquisition-actions
tags: [cython, actions, driver-integration, testing]
completed: 2026-01-25

dependencies:
  requires:
    - "13-01: Validation helpers and action handlers"
    - "Phase 12: Offer infrastructure"
  provides:
    - "Main action handler with 4 action types"
    - "Driver integration for ACQUISITION phase"
    - "Hybrid phase handling (non-player when no offers)"
    - "Integration test suite"
  affects:
    - "Phase 14: Flow & Integration will use apply_acquisition_action"
    - "Future phases can reference action handler pattern"

tech-stack:
  added: []
  patterns:
    - "Action handler dispatch pattern"
    - "Hybrid phase detection (player vs non-player based on state)"
    - "Python wrapper pattern for testing Cython functions"
    - "Action constant exposure via *_PY suffix"

key-files:
  created: []
  modified:
    - path: "phases/acquisition.pyx"
      changes:
        - "Added apply_acquisition_action with 4-way dispatch"
        - "Added apply_acquisition_action_py wrapper"
        - "Fixed company location constants (LOC_PLAYER=3, LOC_FI=4, LOC_CORP=5)"
    - path: "phases/acquisition.pxd"
      changes:
        - "Added apply_acquisition_action declaration"
    - path: "core/driver.pyx"
      changes:
        - "Updated import to include apply_acquisition_action"
        - "Added ACQUISITION phase dispatch"
        - "Implemented hybrid phase check (_is_non_player_phase_check)"
        - "Imported turn_module for offer state access"
    - path: "core/actions.pyx"
      changes:
        - "Exposed action type constants as *_PY versions for Python tests"
    - path: "tests/test_acquisition.py"
      changes:
        - "Implemented TestValidation class (8 tests)"
        - "Implemented TestActionIntegration class (4 tests)"

decisions:
  - id: DEC-13-02-01
    what: "ACQUISITION as hybrid phase"
    why: "Empty offer buffer (no corps/companies) should auto-transition to INVEST, but offers require player decisions"
    how: "_is_non_player_phase_check returns True when acq_active_corp == -1"
    impact: "Prevents ZeroLegalActionsError on empty offer buffer, preserves existing WRAP_UP behavior"

  - id: DEC-13-02-02
    what: "Action constants exposed with _PY suffix"
    why: "Cython enum values not directly importable from Python"
    how: "MODULE_ACTION_TYPE_PY = ACTION_TYPE pattern in actions.pyx"
    impact: "Tests can import ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE"

  - id: DEC-13-02-03
    what: "Fixed company location constants"
    why: "acquisition.pyx had wrong LOC_* values (0,1,2 instead of 3,4,5)"
    how: "Corrected to match entities/company.pxd enum"
    impact: "Money now transfers correctly to players (was failing with location mismatch)"

metrics:
  duration: "~45 minutes"
  commits: 3
  tests_added: 12
  tests_passing: "214 passed, 1 skipped"
---

# Phase 13 Plan 02: Action Handler Integration Summary

**One-liner:** Applied ACQUISITION action handler with 4-way dispatch (price/FI-high/FI-face/pass), integrated with driver using hybrid phase detection for empty offer handling.

## What Was Built

### 1. Main Action Handler (apply_acquisition_action)

Four action types fully implemented:

- **ACTION_ACQ_PRICE**: Price-based acquisition
  - Calculates price as `low_price + info.amount`
  - Validates via `_validate_price_action`
  - Executes via `_handle_accept_price`

- **ACTION_ACQ_FI_HIGH**: FI purchase at high price (non-OS corps)
  - Validates corp is not OS
  - Validates cash and availability
  - Executes via `_handle_fi_buy_high`

- **ACTION_ACQ_FI_FACE**: FI purchase at face value (OS only)
  - Validates corp is OS
  - Validates cash and availability
  - Executes via `_handle_fi_buy_face`

- **ACTION_PASS**: Decline current offer
  - Always valid
  - Executes via `_handle_pass`

Returns: 0 = success, 1 = invalid

### 2. Driver Integration

**Action dispatch** added to `core/driver.pyx`:
```cython
elif phase == PHASE_ACQUISITION:
    result = apply_acquisition_action(state, &info)
```

**Hybrid phase handling**:
- ACQUISITION treated as **non-player phase** when `acq_active_corp == -1` (no offers)
- Treated as **player phase** when offers exist
- Prevents `ZeroLegalActionsError` on empty offer buffer
- Preserves existing WRAP_UP → ACQUISITION → INVEST cycle

### 3. Integration Test Suite

**TestValidation** (8 tests):
- Price validation (in-range, below low, above high)
- Cash sufficiency check
- OS vs non-OS FI purchase rules
- Already-acquired and already-owned checks

**TestActionIntegration** (4 tests):
- Full money transfer flow (corp → player)
- FI high-price purchase
- FI face-value purchase (OS)
- Pass action offer advancement

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed company location constants**
- **Found during:** Task 3 - test_accept_price_action failing
- **Issue:** acquisition.pyx had LOC_PLAYER=1, LOC_FI=0, LOC_CORP=2 (wrong)
- **Root cause:** Didn't reference entities/company.pxd enum (actual: 3, 4, 5)
- **Fix:** Updated to LOC_PLAYER=3, LOC_FI=4, LOC_CORP=5
- **Files modified:** phases/acquisition.pyx
- **Commit:** 2d99359
- **Impact:** Money transfers now work (player proceeds credited correctly)

**2. [Rule 3 - Blocking] Added hybrid phase detection**
- **Found during:** Task 3 - full test suite failing with ZeroLegalActionsError
- **Issue:** ACQUISITION with no offers (empty buffer) has zero legal actions, causing driver error
- **Root cause:** Removed ACQUISITION from _is_non_player_phase, but empty buffer case not handled
- **Fix:** Implemented _is_non_player_phase_check with state-dependent logic
- **Files modified:** core/driver.pyx
- **Commit:** 2d99359
- **Impact:** Empty offer buffer auto-transitions to INVEST via stub

**3. [Rule 2 - Missing Critical] Exposed action constants for Python**
- **Found during:** Task 3 - test imports failing
- **Issue:** ACTION_ACQ_PRICE etc. are Cython enums, not importable from Python
- **Fix:** Added ACTION_*_PY module-level constants in core/actions.pyx
- **Files modified:** core/actions.pyx
- **Commit:** 2d99359
- **Impact:** Tests can import action types

## Testing Results

### Test Counts
- **Total:** 214 passed, 1 skipped
- **New tests:** 12 (8 validation + 4 integration)
- **Regressions:** 0

### Validation Coverage

| Requirement | Test | Status |
|-------------|------|--------|
| VALID-01 | test_price_in_range_succeeds | ✓ |
| VALID-01 | test_price_below_low_rejected | ✓ |
| VALID-01 | test_price_above_high_rejected | ✓ |
| VALID-02 | test_insufficient_cash_rejected | Skipped† |
| VALID-04 | test_target_already_acquired_rejected | ✓ |
| VALID-05 | test_target_already_owned_rejected | ✓ |
| FI rules | test_fi_buy_high_rejects_os_corp | ✓ |
| FI rules | test_fi_buy_face_rejects_non_os_corp | ✓ |

† Skipped: Offer generation filters out unaffordable offers, so no offer to test

### Integration Tests

| Action | Test | Verification |
|--------|------|-------------|
| Price accept | test_accept_price_action | Money transfer, acquisition zone |
| FI high | test_fi_buy_high_action | Corp → FI transfer, acquisition zone |
| FI face | test_fi_buy_face_action | OS → FI at face value |
| Pass | test_pass_action | Offer index advances |

## Architecture

### Action Flow

```
Driver.apply_action(action_idx)
  ↓
[Phase check: PHASE_ACQUISITION]
  ↓
apply_acquisition_action(state, &info)
  ↓
[Dispatch on info.action_type]
  ↓
ACTION_ACQ_PRICE → _validate_price_action → _handle_accept_price
ACTION_ACQ_FI_HIGH → _validate_fi_buy_high → _handle_fi_buy_high
ACTION_ACQ_FI_FACE → _validate_fi_buy_face → _handle_fi_buy_face
ACTION_PASS → _handle_pass
```

### Hybrid Phase Detection

```
_is_non_player_phase_check(state, phase)
  ↓
[Phase == ACQUISITION?]
  ↓
YES → Check acq_active_corp:
        -1 → Non-player (no offers, run stub)
        ≥0 → Player (offers exist, wait for action)
NO → Check other phases
```

## Next Phase Readiness

**Phase 14 (Flow & Integration) can now:**
- Use apply_acquisition_action for action execution
- Reference hybrid phase pattern for other state-dependent phases
- Build on integration test patterns

**Outstanding work:**
- Phase flow completion (offer loop with auto-advances)
- End-of-phase cleanup (merge acquisition zones)
- Terminal state handling for no-companies case

**No blockers or concerns.**
