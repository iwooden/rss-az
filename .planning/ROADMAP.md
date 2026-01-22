# Roadmap: Rolling Stock Stars - v2 INVEST & BID_IN_AUCTION

## Milestones

- ✅ **v1 Game State Initialization** - Phase 1 (shipped 2026-01-20)
- 🚧 **v2 INVEST & BID_IN_AUCTION** - Phases 2-6 (in progress)

## Phases

<details>
<summary>v1 Game State Initialization (Phase 1) - SHIPPED 2026-01-20</summary>

### Phase 1: Game State Initialization
**Goal**: Valid starting game state for 3-6 players with reproducible seeds
**Plans**: 1 plan

Plans:
- [x] 01-01: Initialize game state with seed-based deck shuffle

</details>

### v2 INVEST & BID_IN_AUCTION (In Progress)

**Milestone Goal:** Implement INVEST and BID_IN_AUCTION phase actions with game driver architecture, full share trading mechanics, and corporation lifecycle management.

- [x] **Phase 2: Infrastructure Setup** - Game driver dispatch and phase module structure
- [x] **Phase 3: INVEST Core & Auction Flow** - Pass, start auction, and full auction cycle
- [x] **Phase 4: Share Trading** - Buy/sell shares with price movement and round-trip limits
- [x] **Phase 5: Presidency & Bankruptcy** - Corporation ownership changes and bankruptcy handling
- [x] **Phase 6: Integration & Tests** - Comprehensive test coverage and edge case validation

## Phase Details

### Phase 2: Infrastructure Setup
**Goal**: Game driver can dispatch actions to phase handlers and generate legal move masks
**Depends on**: Phase 1 (game state initialization)
**Requirements**: DRV-01, DRV-02, DRV-03, DRV-04
**Success Criteria** (what must be TRUE):
  1. GameDriver.apply_action(state, action_idx) routes to correct phase handler
  2. GameDriver.get_legal_moves(state) returns valid action mask for current phase
  3. Action decoding uses existing decode_action() without modification
  4. All dispatch functions maintain noexcept nogil for performance
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md — Create GameDriver class and phase handler stubs
- [x] 02-02-PLAN.md — Test GameDriver dispatch and validation

### Phase 3: INVEST Core & Auction Flow
**Goal**: Players can pass, start auctions, bid, and complete full auction cycles
**Depends on**: Phase 2
**Requirements**: INV-01, INV-02, INV-03, INV-04, INV-04a, INV-05, INV-06, BID-01, BID-02, BID-03, BID-04, BID-05, BID-06, BID-07, BID-08, BID-09, BID-10, BID-11, BID-12
**Success Criteria** (what must be TRUE):
  1. Player can pass and consecutive passes tracked correctly
  2. Player can start auction for available company at chosen price
  3. Players can leave auction or raise bid in proper rotation
  4. Auction resolves correctly when one bidder remains (winner pays, gets company)
  5. Turn returns to player after auction starter (not winner) when auction completes
**Plans**: 3 plans

Plans:
- [x] 03-01-PLAN.md — Implement INVEST phase pass and start auction
- [x] 03-02-PLAN.md — Implement BID_IN_AUCTION phase handler
- [x] 03-03-PLAN.md — Test coverage for INVEST and BID phases

### Phase 4: Share Trading
**Goal**: Players can buy and sell shares with proper price movement and trading limits
**Depends on**: Phase 3
**Requirements**: INV-07, INV-08, INV-09, INV-10, INV-11, INV-12, INV-13, INV-14, INV-15, INV-16, INV-17
**Success Criteria** (what must be TRUE):
  1. Player can buy share (cash deducted, share transferred, price moves up)
  2. Player can sell share (cash received, share transferred, price moves down)
  3. Price movement skips occupied market spaces
  4. Round-trip limit (2 per corp per turn) prevents excessive trading
  5. Player net worth updates after each buy/sell action
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Implement buy/sell share handlers with price movement
- [x] 04-02-PLAN.md — Add round-trip limits to mask and test coverage

### Phase 5: Presidency & Bankruptcy
**Goal**: Corporation ownership transfers correctly and bankruptcy procedure completes cleanly
**Depends on**: Phase 4
**Requirements**: INV-18, INV-19, INV-20, INV-21, INV-22, INV-23, INV-24, INV-25, INV-26, INV-27
**Success Criteria** (what must be TRUE):
  1. Presidency transfers to player with most shares (incumbent keeps on tie)
  2. Receivership flag set when all player shares sold
  3. Buying from receivership exits receivership and sets buyer as president
  4. Corporation bankruptcy triggers when price drops to 0
  5. Bankruptcy procedure removes companies, returns shares/money/price card, corp available for future IPO
**Plans**: 2 plans

Plans:
- [x] 05-01-PLAN.md — Implement bankruptcy procedure and integrate into sell handler
- [x] 05-02-PLAN.md — Implement presidency/receivership checks and test coverage

### Phase 6: Integration & Tests
**Goal**: Comprehensive test coverage validates all phase logic and edge cases
**Depends on**: Phases 2-5
**Requirements**: TST-01, TST-02, TST-03, TST-04
**Success Criteria** (what must be TRUE):
  1. Test suite in tests/phases/ directory covers all implemented actions
  2. Common game scenarios from rules documented and tested
  3. Edge cases (bankruptcy cascade, presidency change, receivership) have dedicated tests
  4. Action mask matches valid actions after every state change
**Plans**: 3 plans

Plans:
- [x] 06-01-PLAN.md — Migrate tests to tests/phases/ and create shared assertion helpers
- [x] 06-02-PLAN.md — Add INVEST integration tests and edge case coverage
- [x] 06-03-PLAN.md — Add BID_IN_AUCTION integration tests and edge case coverage

## Progress

**Execution Order:** Phases 2 → 3 → 4 → 5 → 6

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Game State Init | v1 | 1/1 | Complete | 2026-01-20 |
| 2. Infrastructure Setup | v2 | 2/2 | Complete | 2026-01-21 |
| 3. INVEST Core & Auction | v2 | 3/3 | Complete | 2026-01-21 |
| 4. Share Trading | v2 | 2/2 | Complete | 2026-01-21 |
| 5. Presidency & Bankruptcy | v2 | 2/2 | Complete | 2026-01-21 |
| 6. Integration & Tests | v2 | 3/3 | Complete | 2026-01-21 |

---
*Roadmap created: 2026-01-20*
*v2 requirements: 48 total, 48 mapped*
