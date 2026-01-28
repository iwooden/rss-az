# Roadmap: Rolling Stock Stars v5.1 nogil Optimization

## Overview

The v5.1 milestone addresses deferred tech debt from Phase 15.1: enabling `nogil` on mask generation functions for true thread-level parallelization. This requires extending the existing low-level pointer-based accessor pattern (established in player.pyx) to corp and turn operations, then updating mask functions to use these consistently.

## Milestones

- ✅ **v1.0** - Game State Init - Phases 1 (shipped 2026-01-20)
- ✅ **v2.0** - INVEST & BID_IN_AUCTION - Phases 2-6 (shipped 2026-01-21)
- ✅ **v2.1** - Forced Action Auto-Application - Phases 7-8 (shipped 2026-01-23)
- ✅ **v3.0** - WRAP_UP Phase - Phases 9-11 + 10.1 (shipped 2026-01-24)
- ✅ **v4.0** - ACQUISITION Phase - Phases 12-15 (shipped 2026-01-26)
- ✅ **v5.0** - CLOSING Phase - Phases 16-19 (shipped 2026-01-27)
- 🔄 **v5.1** - nogil Optimization - Phase 20

## Phases

- [x] **Phase 15.1: Code Quality Refactoring** - DRY fixes, performance optimizations, architecture cleanup (INSERTED)
- [x] **Phase 16: Auto-Close Logic** - FI and receivership corps auto-close at phase start
- [x] **Phase 17: Offer-Based Close Flow** - Player decisions on negative-income companies
- [x] **Phase 18: Mandatory Close and Transition** - Auto-close at phase end, transition to INCOME
- [x] **Phase 19: Testing and Integration** - Comprehensive test coverage
- [ ] **Phase 20: nogil Mask Optimization** - Enable GIL-free mask generation for parallelization

## Phase Details

### Phase 15.1: Code Quality Refactoring (INSERTED)
**Goal**: Fix DRY violations, performance anti-patterns, and architecture issues before implementing CLOSING phase
**Depends on**: Phase 15 (ACQUISITION complete)
**Requirements**: None (refactoring phase)
**Success Criteria** (what must be TRUE):
  1. One-hot encoding helpers extracted to `entities/encoding.pyx` - all 15+ call sites refactored
  2. `nogil` added to all `_fill_*_mask()` functions in `actions.pyx`
  3. Mask buffer pre-allocated instead of `np.zeros()` per call
  4. Phase dispatch extracted to single helper function (removes 14 LOC duplication)
  5. CORPS changed from dict to list pattern (consistent with PLAYERS, COMPANIES)
  6. Normalization helpers created for cash/share conversions
  7. Test fixtures and status codes consolidated
  8. All existing tests pass (no regressions)
**Plans**: 5 plans

Plans:
- [x] 15.1-01-PLAN.md — Create encoding helpers (set/get/clear one-hot)
- [x] 15.1-02-PLAN.md — Change CORPS from dict to list pattern
- [x] 15.1-03-PLAN.md — Optimize actions.pyx (buffer, dispatch) [nogil deferred]
- [x] 15.1-04-PLAN.md — Apply encoding helpers to entities
- [x] 15.1-05-PLAN.md — Consolidate test infrastructure

### Phase 16: Auto-Close Logic
**Goal**: FI and receivership corps automatically close unprofitable companies at phase start
**Depends on**: Phase 15 (ACQUISITION complete)
**Requirements**: CLO-01, CLO-02, CLO-03, CLO-04
**Success Criteria** (what must be TRUE):
  1. FI closes any company where Cost of Ownership >= Income
  2. Receivership corp closes red companies when CoO >= 4
  3. Receivership corp closes orange companies when CoO >= 7
  4. Receivership corp always retains highest face value company (never closes last company)
**Plans**: 2 plans

Plans:
- [x] 16-01-PLAN.md — Create closing.pyx/pxd with auto-close logic
- [x] 16-02-PLAN.md — Driver integration and tests

### Phase 17: Offer-Based Close Flow
**Goal**: Players can decide to close or keep negative-income companies via offer system
**Depends on**: Phase 16
**Requirements**: CLO-05, CLO-06, CLO-07, CLO-08, CLO-09, CLO-10, CLO-11, CLO-12, CLO-13
**Success Criteria** (what must be TRUE):
  1. Offers generated only for companies with negative adjusted income
  2. Offers sorted by face value ascending (lowest value offered first)
  3. Player-owned privates and corp subsidiaries (same-president) included in offers
  4. Offer validation: corp closing offers invalid if corp would have 0 companies after close
  5. Dynamic re-validation: prior acceptance can invalidate later offers (corp down to 1 company)
  6. Accept action removes company from game; Pass action keeps company
  7. Junkyard Scrappers receives 2x printed income as bonus when closing
**Plans**: 3 plans

Plans:
- [x] 17-01-PLAN.md — State infrastructure and offer generation
- [x] 17-02-PLAN.md — Driver integration and action handlers
- [x] 17-03-PLAN.md — Comprehensive tests for CLO-05 through CLO-13

### Phase 18: Mandatory Close and Transition
**Goal**: Auto-close at phase end protects players from negative cash, then transition to INCOME
**Depends on**: Phase 17
**Requirements**: CLO-14, CLO-15, CLO-16
**Success Criteria** (what must be TRUE):
  1. Players with negative total income have privates auto-closed until income non-negative
  2. Cheapest negative-income company closed first during mandatory closing
  3. Phase transitions to INCOME when all offers processed and mandatory closes complete
**Plans**: 2 plans

Plans:
- [x] 18-01-PLAN.md — Player income method and mandatory close logic
- [x] 18-02-PLAN.md — Comprehensive tests for CLO-14, CLO-15, CLO-16

### Phase 19: Testing and Integration
**Goal**: Comprehensive test coverage validates CLOSING phase correctness
**Depends on**: Phase 18
**Requirements**: None (testing phase)
**Success Criteria** (what must be TRUE):
  1. Unit tests cover all 16 requirements individually
  2. Integration tests verify ACQUISITION -> CLOSING -> INCOME flow
  3. Edge case tests cover empty offers, all-pass, multi-close scenarios
  4. All existing tests pass (no regressions)
**Plans**: 2 plans

Plans:
- [x] 19-01-PLAN.md — Edge case tests for CLOSING phase
- [x] 19-02-PLAN.md — Integration tests for ACQUISITION -> CLOSING -> INCOME flow

### Phase 20: nogil Mask Optimization
**Goal**: Enable `nogil` on all mask generation functions for true thread-level parallelization
**Depends on**: Phase 19 (v5.0 complete)
**Requirements**: None (optimization phase, closes deferred tech debt from 15.1)
**Success Criteria** (what must be TRUE):
  1. Low-level nogil accessors created for corp operations in `entities/corp.pyx`
  2. Low-level nogil accessors created for turn state in `entities/turn.pyx`
  3. All 7 `_fill_*_mask()` functions use low-level accessors (no `state.get_*()` calls)
  4. All 7 `_fill_*_mask()` functions have `nogil` added to signature
  5. `_fill_mask_for_phase()` dispatch function has `nogil` added
  6. All existing tests pass (no regressions)
  7. Benchmark shows no performance regression (games/minute stable or improved)
**Plans**: 3 plans

Plans:
- [ ] 20-01-PLAN.md — Create low-level nogil accessors for corp and turn
- [ ] 20-02-PLAN.md — Refactor mask functions to use low-level accessors
- [ ] 20-03-PLAN.md — Add nogil to mask functions and verify

## Progress

**Execution Order:** 15.1 -> 16 -> 17 -> 18 -> 19 -> 20

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 15.1. Code Quality Refactoring | v5.0 | 5/5 | ✓ Complete | 2026-01-26 |
| 16. Auto-Close Logic | v5.0 | 2/2 | ✓ Complete | 2026-01-27 |
| 17. Offer-Based Close Flow | v5.0 | 3/3 | ✓ Complete | 2026-01-27 |
| 18. Mandatory Close and Transition | v5.0 | 2/2 | ✓ Complete | 2026-01-27 |
| 19. Testing and Integration | v5.0 | 2/2 | ✓ Complete | 2026-01-27 |
| 20. nogil Mask Optimization | v5.1 | 0/3 | Not Started | - |

---
*Roadmap created: 2026-01-26*
*Last updated: 2026-01-28*
*Phase 15.1 inserted: 2026-01-26 (code quality refactoring before CLOSING implementation)*
*Phase 15.1 planned: 2026-01-26 (5 plans in 2 waves)*
*Phase 15.1 complete: 2026-01-26 (7/8 criteria verified, nogil deferred)*
*Phase 16 planned: 2026-01-26 (2 plans in 2 waves)*
*Phase 16 complete: 2026-01-27 (4/4 criteria verified)*
*Phase 17 planned: 2026-01-27 (3 plans in 3 waves)*
*Phase 17 complete: 2026-01-27 (7/7 criteria verified)*
*Phase 18 planned: 2026-01-27 (2 plans in 2 waves)*
*Phase 18 complete: 2026-01-27 (3/3 criteria verified)*
*Phase 19 planned: 2026-01-27 (2 plans in 1 wave)*
*Phase 19 complete: 2026-01-27 (4/4 criteria verified)*
*Phase 20 added: 2026-01-28 (closes deferred nogil tech debt from 15.1)*
