# Roadmap: Rolling Stock Stars - Cython Game Engine

## Milestones

- ✅ **v1** - Phase 1 (shipped 2026-01-20)
- ✅ **v2** - Phases 2-6 (shipped 2026-01-21)
- ✅ **v2.1** - Phases 7-8 (shipped 2026-01-23)
- 🚧 **v3.0 WRAP_UP Phase** - Phases 9-11 (in progress)

## Overview

v3.0 implements the WRAP_UP phase, which executes deterministically at end of each INVEST round. The phase reorders players by descending cash, executes Foreign Investor purchases of available companies at face value, and transitions all unavailable companies to available. Implementation follows established phase handler patterns with zero new dependencies.

## Phases

**Phase Numbering:**
- Integer phases (9, 10, 11): Planned milestone work
- Decimal phases (e.g., 9.1): Urgent insertions (marked with INSERTED)

---

<details>
<summary>✅ v1 - Game State Initialization (Phase 1) - SHIPPED 2026-01-20</summary>

### Phase 1: Game State Initialization
**Goal**: GameState.initialize_game() produces valid starting state
**Plans**: 1 plan

Plans:
- [x] 01-01: Initialize game state with all entities

</details>

<details>
<summary>✅ v2 - INVEST & BID_IN_AUCTION (Phases 2-6) - SHIPPED 2026-01-21</summary>

### Phase 2: GameDriver Foundation
**Goal**: GameDriver dispatches actions to phase handlers
**Plans**: 2 plans

Plans:
- [x] 02-01: GameDriver core dispatch
- [x] 02-02: Action encoding and validation

### Phase 3: INVEST Core & Auction Flow
**Goal**: INVEST phase handles pass, start auction, share transactions
**Plans**: 3 plans

Plans:
- [x] 03-01: Pass and start auction
- [x] 03-02: Buy shares
- [x] 03-03: Sell shares

### Phase 4: BID_IN_AUCTION Implementation
**Goal**: BID_IN_AUCTION phase handles bidding and auction resolution
**Plans**: 3 plans

Plans:
- [x] 04-01: Leave auction and raise bid
- [x] 04-02: Auction resolution
- [x] 04-03: Round-trip limit enforcement

### Phase 5: Bankruptcy & Receivership
**Goal**: Corporations can go bankrupt and enter receivership
**Plans**: 2 plans

Plans:
- [x] 05-01: Bankruptcy at price 0
- [x] 05-02: Receivership when all shares sold

### Phase 6: Presidency Transfer
**Goal**: Presidency transfers with incumbent advantage
**Plans**: 2 plans

Plans:
- [x] 06-01: Two-pass presidency algorithm
- [x] 06-02: Incumbent tie-breaking

</details>

<details>
<summary>✅ v2.1 - Forced Action Auto-Application (Phases 7-8) - SHIPPED 2026-01-23</summary>

### Phase 7: Core Implementation
**Goal**: Auto-apply forced actions when only 1 legal action exists
**Plans**: 2 plans

Plans:
- [x] 07-01: Auto-apply loop with 0-action validation
- [x] 07-02: History tracking for auto-applied actions

### Phase 8: Test Updates
**Goal**: Update tests to account for auto-advancement behavior
**Plans**: 1 plan

Plans:
- [x] 08-01: Fix tests with explicit history assertions
- [x] 08-02: Add auto-apply behavior tests

</details>

---

## 🚧 v3.0 WRAP_UP Phase (Phases 9-11) - IN PROGRESS

**Milestone Goal:** Implement deterministic WRAP_UP phase that reorders players by cash and executes Foreign Investor purchases at end of INVEST round.

### Phase 9: WRAP_UP Core Logic ✓
**Goal**: Deterministic player reordering and phase transitions
**Depends on**: Phase 8 (v2.1)
**Requirements**: REORDER-01, REORDER-02, REORDER-03, PHASE-01, PHASE-02, PHASE-03, PHASE-04
**Success Criteria** (what must be TRUE):
  1. Players are reordered by descending cash with tie-breaking by old turn order after all players pass in INVEST
  2. Active player is updated to new position 0 after reordering
  3. WRAP_UP phase transitions to new INVEST turn with incremented turn number
  4. WRAP_UP execution creates discrete state history entry (not absorbed into INVEST)
  5. GameDriver allows 0 legal actions for non-player phases without error
**Plans**: 2 plans
**Completed**: 2026-01-23

Plans:
- [x] 09-01-PLAN.md — Create WRAP_UP and ACQUISITION phase handlers
- [x] 09-02-PLAN.md — Integrate handlers into INVEST and GameDriver

### Phase 10: Foreign Investor Purchase Logic ✓
**Goal**: Foreign Investor purchases cheapest available companies at face value
**Depends on**: Phase 9
**Requirements**: FI-01, FI-02, FI-03, FI-04, FI-05, FI-06, FI-07, AVAIL-01
**Success Criteria** (what must be TRUE):
  1. FI purchases cheapest available company at face value in ascending order
  2. After each purchase, new card is drawn and company becomes unavailable
  3. Purchase loop stops when FI cannot afford any remaining available company
  4. Edge cases handled correctly (FI 0 cash, empty deck, no available companies)
  5. After FI purchases complete, all unavailable companies become available
**Plans**: 1 plan
**Completed**: 2026-01-23

Plans:
- [x] 10-01-PLAN.md — Implement FI purchase loop and availability transition

### Phase 10.1: Fix WRAP_UP Bugs (INSERTED) ✓
**Goal**: Fix player_stride calculation bug causing player 1+ and FI data corruption
**Depends on**: Phase 10
**Requirements**: Fixes for REORDER-01, REORDER-02, REORDER-03, FI-01 through FI-07
**Success Criteria** (what must be TRUE):
  1. Player cash preserved through WRAP_UP cycle (players 1+ no longer zeroed)
  2. FI cash correctly calculated after purchases (remainder preserved, not zeroed)
  3. No infinite loop when no companies available for FI purchase
  4. All 4 failing tests in test_wrap_up.py pass
**Plans**: 1 plan
**Completed**: 2026-01-24

Plans:
- [x] 10.1-01-PLAN.md — Add is_auction_high_bidder to player_stride

**Root cause:** compute_layout() missing is_auction_high_bidder in player_stride calculation (stride=73 but should be 74). This causes player 1+ base offsets to drift by 1 float per player, corrupting all data access.

### Phase 11: Test Updates
**Goal**: Fix existing tests and add WRAP_UP verification tests
**Depends on**: Phase 10.1
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Existing INVEST/auction tests pass with auto-continue behavior through WRAP_UP
  2. Test utilities include set_phase() method for manual phase manipulation
  3. Player order verification tests confirm reordering correctness
**Plans**: 2 plans

Plans:
- [x] 11-01-PLAN.md — Fix test failures and add WRAP_UP verification tests
- [ ] 11-02-PLAN.md — Factor out integration tests to test_integration.py

## Progress

**Execution Order:** 9 → 10 → 10.1 → 11

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Game State Init | v1 | 1/1 | Complete | 2026-01-20 |
| 2. GameDriver Foundation | v2 | 2/2 | Complete | 2026-01-21 |
| 3. INVEST Core & Auction Flow | v2 | 3/3 | Complete | 2026-01-21 |
| 4. BID_IN_AUCTION Implementation | v2 | 3/3 | Complete | 2026-01-21 |
| 5. Bankruptcy & Receivership | v2 | 2/2 | Complete | 2026-01-21 |
| 6. Presidency Transfer | v2 | 2/2 | Complete | 2026-01-21 |
| 7. Core Implementation | v2.1 | 2/2 | Complete | 2026-01-23 |
| 8. Test Updates | v2.1 | 2/2 | Complete | 2026-01-23 |
| 9. WRAP_UP Core Logic | v3.0 | 2/2 | Complete | 2026-01-23 |
| 10. FI Purchase Logic | v3.0 | 1/1 | Complete | 2026-01-23 |
| 10.1 Fix WRAP_UP Bugs | v3.0 | 1/1 | Complete | 2026-01-24 |
| 11. Test Updates | v3.0 | 1/2 | In Progress | - |

---
*Last updated: 2026-01-24*
