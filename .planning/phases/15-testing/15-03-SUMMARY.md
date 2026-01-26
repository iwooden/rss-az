---
phase: 15-testing
plan: 03
subsystem: testing
tags: [pytest, integration-tests, phase-transitions, invariants, acquisition]

# Dependency graph
requires:
  - phase: 15-01
    provides: Test organization (phases/ vs root), fixture re-exports
  - phase: 15-02
    provides: ACQUISITION validation boundary and edge case tests
  - phase: 12-14
    provides: ACQUISITION phase implementation (offer generation, actions, flow)
provides:
  - TestAcquisitionIntegration class with 7 end-to-end tests
  - Cross-phase flow verification (INVEST->WRAP_UP->ACQUISITION->INVEST)
  - Zone merging integration tests (proceeds + companies)
  - Multi-action sequence invariant verification
affects: [future-integration-tests]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Integration test pattern: apply_action_and_verify for invariant checking"
    - "Phase transition testing through full driver flow"
    - "Zone merge verification at phase boundaries"

key-files:
  created: []
  modified:
    - tests/test_integration.py

key-decisions:
  - "Use acq_pass action index for ACQUISITION phase pass (not pass_invest)"
  - "Use is_in_corp_acquisition(state, corp_id) to check acquisition zone"
  - "Use get_owner_id(state) for company ownership checks"

patterns-established:
  - "Integration tests verify invariants after each action"
  - "Zone merge tests set up proceeds + companies, then verify merge"
  - "Multi-action tests use iteration limits to prevent infinite loops"

# Metrics
duration: 2.5min
completed: 2026-01-26
---

# Phase 15 Plan 03: ACQUISITION Integration Tests Summary

**Cross-phase flow verification covering INVEST->WRAP_UP->ACQUISITION->INVEST with zone merging and invariant maintenance**

## Performance

- **Duration:** 2.5 min
- **Started:** 2026-01-26T21:48:23Z
- **Completed:** 2026-01-26T21:50:53Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments
- TestAcquisitionIntegration class with 7 comprehensive integration tests
- Phase transition tests verify WRAP_UP->ACQUISITION->INVEST flow
- Action tests verify accept/pass maintain invariants through full driver
- Zone merge test verifies FLOW-03 (proceeds) and FLOW-04 (companies)

## Task Commits

Each task was committed atomically:

1. **Tasks 1-3: Add ACQUISITION integration tests** - `32950be` (test)

**Plan metadata:** (to be committed)

## Files Created/Modified
- `tests/test_integration.py` - Added TestAcquisitionIntegration class with 7 tests

## Test Coverage

**Test 1: test_wrap_up_to_acquisition_maintains_invariants**
- Verifies INVEST->WRAP_UP->ACQUISITION->INVEST flow
- Fresh game (no offers) completes ACQUISITION immediately
- Turn number increments correctly

**Test 2: test_acquisition_to_invest_new_turn**
- Direct ACQUISITION->INVEST transition
- Turn number increments
- Invariants maintained

**Test 3: test_full_turn_cycle_with_acquisition**
- Multi-turn cycle (turn 1->2->3)
- Each cycle: pass all players, WRAP_UP, ACQUISITION, back to INVEST
- Invariants maintained throughout

**Test 4: test_acquisition_accept_maintains_invariants**
- Sets up player private->corp offer
- Applies accept action through driver
- Verifies invariants after acceptance

**Test 5: test_acquisition_pass_maintains_invariants**
- Sets up valid offer
- Applies pass action (acq_pass, not pass_invest)
- Verifies invariants maintained

**Test 6: test_multiple_acquisitions_maintain_invariants**
- Sets up multiple offers
- Accept first, pass remaining
- Verifies invariants after each action
- Phase eventually completes

**Test 7: test_zone_merge_at_phase_transition**
- Sets up acquisition proceeds (player + corp)
- Sets up acquisition zone company
- Calls transition_to_closing_py
- Verifies proceeds merged to cash
- Verifies company merged to owned_companies
- Verifies all zones cleared

## Decisions Made

**acq_pass vs pass_invest**
- ACQUISITION phase has its own pass action at layout['acq_pass']
- Cannot use layout['pass_invest'] (wrong phase, invalid action)

**Company ownership API**
- Use is_in_corp_acquisition(state, corp_id) to check acquisition zone
- Use get_owner_id(state) to get owner ID
- No is_in_acquisition_zone() method exists (needs corp_id parameter)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

**Action index confusion (resolved)**
- Initially tried to use layout['pass_invest'] for ACQUISITION pass
- Fixed by using layout['acq_pass'] (correct ACQUISITION pass action)

**Company API method names (resolved)**
- Initial use of non-existent methods (get_owner_corp_id, is_in_acquisition_zone)
- Fixed by using correct API: get_owner_id(), is_in_corp_acquisition(corp_id)

## Next Phase Readiness

**Phase 15 complete (TEST-06 satisfied)**
- All 7 test requirements satisfied (TEST-01 through TEST-07)
- Total: 254 tests passing (including 7 new integration tests)
- ACQUISITION phase fully tested at unit and integration levels

**v4.0 ACQUISITION Milestone status:**
- Phase 12: Offer Infrastructure ✓
- Phase 13: Actions & Validation ✓
- Phase 14: Flow & Integration ✓
- Phase 15: Testing ✓

**Remaining work:**
- CLOSING phase not yet implemented (transitions to INVEST as workaround)
- When CLOSING is added, integration tests will automatically cover it

---
*Phase: 15-testing*
*Completed: 2026-01-26*
