# Roadmap: v2.1 Forced Action Auto-Application

**Created:** 2026-01-21
**Milestone:** v2.1
**Depth:** Quick
**Phases:** 7-8 (continues from v2 which ended at Phase 6)

---

## Overview

Implement iterative auto-application of forced actions in GameDriver. When exactly one legal action exists, apply it automatically and continue until 2+ choices are available or the game ends. This ensures the neural network only sees states with real decisions. Two phases: core implementation with history tracking (Phase 7), then test infrastructure updates (Phase 8).

---

## Phase 7: Core Implementation

**Goal:** GameDriver auto-applies forced actions iteratively, with optional history tracking for test observability.

**Dependencies:** None (builds on existing GameDriver from v2)

**Requirements:**
- HELP-01: ForcedActionResult struct
- HELP-02: _check_forced_action() function
- HELP-03: _apply_single_action() function
- LOOP-01: Iterative auto-apply when 1 legal action
- LOOP-02: Exit loop when 2+ actions available
- LOOP-03: Exit loop when GAME_OVER
- LOOP-04: Iteration limit guard (100 max)
- LOOP-05: Zero legal actions raises error
- HIST-01: Optional history parameter
- HIST-02: Append (state.copy(), action) tuples
- HIST-03: History includes auto-applied actions
- HIST-04: No overhead when history is None

**Success Criteria:**
1. User calls apply_action() with a forced state and receives next state with 2+ legal actions (or GAME_OVER)
2. User can pass history=[] to apply_action() and inspect all intermediate states and actions after call
3. User cannot trigger infinite loop - iteration limit raises clear error
4. User receives error if state has zero legal actions (outside GAME_OVER)

---

## Phase 8: Test Updates

**Goal:** All existing tests pass with auto-apply behavior; new tests verify forced action chains and edge cases.

**Dependencies:** Phase 7 complete

**Requirements:**
- TEST-01: apply_and_track() fixture in conftest.py
- TEST-02: Helper provides access to full action history
- TEST-03: Helper allows intermediate state inspection
- TUPD-01: Categorize existing tests by auto-apply impact
- TUPD-02: Update tests asserting intermediate states
- TUPD-03: Add forced action chain tests
- TUPD-04: Add phase transition during auto-apply tests
- TUPD-05: Add iteration limit guard test
- TUPD-06: Add zero legal actions error test

**Success Criteria:**
1. All 170+ existing tests pass after auto-apply integration
2. User can use apply_and_track() helper to verify intermediate states in tests
3. Test suite covers forced action chains (multiple sequential auto-applies)
4. Test suite covers edge cases: phase transitions, iteration limit, zero actions error

---

## Progress

| Phase | Name | Requirements | Status |
|-------|------|--------------|--------|
| 7 | Core Implementation | 12 | Pending |
| 8 | Test Updates | 7 | Pending |

**Total:** 19 requirements across 2 phases

---

## Coverage Verification

| Requirement | Phase | Mapped |
|-------------|-------|--------|
| HELP-01 | 7 | Yes |
| HELP-02 | 7 | Yes |
| HELP-03 | 7 | Yes |
| LOOP-01 | 7 | Yes |
| LOOP-02 | 7 | Yes |
| LOOP-03 | 7 | Yes |
| LOOP-04 | 7 | Yes |
| LOOP-05 | 7 | Yes |
| HIST-01 | 7 | Yes |
| HIST-02 | 7 | Yes |
| HIST-03 | 7 | Yes |
| HIST-04 | 7 | Yes |
| TEST-01 | 8 | Yes |
| TEST-02 | 8 | Yes |
| TEST-03 | 8 | Yes |
| TUPD-01 | 8 | Yes |
| TUPD-02 | 8 | Yes |
| TUPD-03 | 8 | Yes |
| TUPD-04 | 8 | Yes |
| TUPD-05 | 8 | Yes |
| TUPD-06 | 8 | Yes |

**Mapped:** 19/19 (100%)
**Orphaned:** 0

---

*Roadmap created: 2026-01-21*
