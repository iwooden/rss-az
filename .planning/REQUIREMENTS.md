# Requirements: v2.1 Forced Action Auto-Application

**Defined:** 2026-01-21
**Core Value:** Never present model with 0 or 1 legal actions — auto-apply forced actions until real choice exists

## v2.1 Requirements

Requirements for forced action auto-application. Each maps to roadmap phases.

### Helper Infrastructure

- [ ] **HELP-01**: `ForcedActionResult` struct with `count` (int) and `action_idx` (int) fields
- [ ] **HELP-02**: `_check_forced_action(state)` cdef function returns ForcedActionResult distinguishing 0, 1, and 2+ legal actions
- [ ] **HELP-03**: `_apply_single_action(state, action, history)` cdef function applies one action without auto-advance loop

### Core Loop

- [ ] **LOOP-01**: `apply_action()` iteratively auto-applies when exactly 1 legal action exists
- [ ] **LOOP-02**: Loop exits when 2+ legal actions available
- [ ] **LOOP-03**: Loop exits when game phase is GAME_OVER
- [ ] **LOOP-04**: Iteration limit guard (100 max) raises error if exceeded
- [ ] **LOOP-05**: Zero legal actions (outside GAME_OVER) raises error

### History Tracking

- [ ] **HIST-01**: `apply_action()` accepts optional `history` parameter (list, default None)
- [ ] **HIST-02**: When history is not None, append `(state.copy(), action)` before each action applied
- [ ] **HIST-03**: History includes both explicit and auto-applied actions
- [ ] **HIST-04**: No overhead when history is None (production mode)

### Test Infrastructure

- [ ] **TEST-01**: `apply_and_track()` fixture/helper in conftest.py that manages history list
- [ ] **TEST-02**: Helper provides access to full action history after apply
- [ ] **TEST-03**: Helper allows inspection of intermediate states

### Test Updates

- [ ] **TUPD-01**: Categorize existing tests by auto-apply impact
- [ ] **TUPD-02**: Update tests that assert on intermediate states to use history helper
- [ ] **TUPD-03**: Add tests for forced action chains (multiple auto-applies in sequence)
- [ ] **TUPD-04**: Add tests for phase transition during auto-apply (BID → INVEST)
- [ ] **TUPD-05**: Add test for iteration limit guard (defensive)
- [ ] **TUPD-06**: Add test for zero legal actions error

## Future Requirements

(None — this is a focused enhancement milestone)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Action replay/undo | Not needed for training, adds complexity |
| Toggleable auto-apply | History parameter provides test observability |
| Async/callback hooks | Over-engineering for current needs |
| Performance profiling | Defer until bottleneck identified |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HELP-01 | Phase 7 | Pending |
| HELP-02 | Phase 7 | Pending |
| HELP-03 | Phase 7 | Pending |
| LOOP-01 | Phase 7 | Pending |
| LOOP-02 | Phase 7 | Pending |
| LOOP-03 | Phase 7 | Pending |
| LOOP-04 | Phase 7 | Pending |
| LOOP-05 | Phase 7 | Pending |
| HIST-01 | Phase 7 | Pending |
| HIST-02 | Phase 7 | Pending |
| HIST-03 | Phase 7 | Pending |
| HIST-04 | Phase 7 | Pending |
| TEST-01 | Phase 8 | Pending |
| TEST-02 | Phase 8 | Pending |
| TEST-03 | Phase 8 | Pending |
| TUPD-01 | Phase 8 | Pending |
| TUPD-02 | Phase 8 | Pending |
| TUPD-03 | Phase 8 | Pending |
| TUPD-04 | Phase 8 | Pending |
| TUPD-05 | Phase 8 | Pending |
| TUPD-06 | Phase 8 | Pending |

**Coverage:**
- v2.1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-01-21*
*Last updated: 2026-01-21 after initial definition*
