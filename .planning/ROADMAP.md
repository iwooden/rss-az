# Roadmap: v4.0 ACQUISITION Phase

## Overview

This milestone implements the ACQUISITION phase with AlphaZero-optimized mechanics: offer-based flow with sorted priority presentation, same-president trade restrictions, receivership auto-buy integration, and acquisition proceeds zone to prevent re-acquisition loops. The phase follows the established non-player phase pattern where some transitions happen automatically (receivership auto-buy) while player decisions occur for offers where the buyer's president is a player.

## Milestones

- [x] **v1** - Game State Init (Phase 1) - shipped 2026-01-20
- [x] **v2** - INVEST & BID_IN_AUCTION (Phases 2-6) - shipped 2026-01-21
- [x] **v2.1** - Forced Action Auto-Application (Phases 7-8) - shipped 2026-01-23
- [x] **v3.0** - WRAP_UP Phase (Phases 9-11 + 10.1) - shipped 2026-01-24
- [ ] **v4.0** - ACQUISITION Phase (Phases 12-15) - in progress

## Phases

**Phase Numbering:**
- Continues from v3.0's Phase 11
- Integer phases (12, 13, 14, 15): Planned milestone work
- Decimal phases (12.1, etc.): Urgent insertions if needed

- [x] **Phase 12: Offer Infrastructure** - Offer generation, sorting, and state management
- [x] **Phase 13: Actions & Validation** - Accept/pass actions with full validation rules
- [ ] **Phase 14: Flow & Integration** - Receivership auto-buy, phase transitions, driver integration
- [ ] **Phase 15: Testing** - Unit, integration, and edge case tests

## Phase Details

### Phase 12: Offer Infrastructure
**Goal**: Generate and present acquisition offers in correct priority order with proper state tracking
**Depends on**: v3.0 (WRAP_UP transitions to ACQUISITION)
**Requirements**: OFFER-01, OFFER-02, OFFER-03, OFFER-04, OFFER-05, STATE-01, STATE-02, STATE-03, STATE-04
**Success Criteria** (what must be TRUE):
  1. Calling offer generation returns offers in correct priority order (OS->FI first, then by share price, then corp-to-corp, then player privates)
  2. Active offer state (acq_active_corp, acq_target_company, acq_is_fi_offer) reflects current offer
  3. Acquisition zones (acquisition_companies, acquisition_proceeds) accumulate purchases within phase
  4. When no offers exist, acq_active_corp is set to -1
**Plans**: 4 plans in 3 waves

Plans:
- [x] 12-01-PLAN.md — State infrastructure (hidden offer buffer, player acquisition_proceeds)
- [x] 12-02-PLAN.md — Offer generation and sorting logic
- [x] 12-03-PLAN.md — Offer state presentation (visible state from buffer)
- [x] 12-04-PLAN.md — Phase entry integration (WRAP_UP -> ACQUISITION)

### Phase 13: Actions & Validation
**Goal**: Players can accept or pass on acquisition offers with full validation
**Depends on**: Phase 12
**Requirements**: ACTION-01, ACTION-02, ACTION-03, ACTION-04, VALID-01, VALID-02, VALID-03, VALID-04, VALID-05, VALID-06
**Success Criteria** (what must be TRUE):
  1. Player can accept acquisition at any valid price within the company's price span
  2. FI Buy High and FI Buy Face actions execute correctly for FI offers
  3. Pass action advances to next offer without modifying state
  4. Invalid actions are rejected (wrong price, insufficient cash, would leave seller with 0 companies, target already acquired, same-president violation)
**Plans**: 2 plans in 2 waves

Plans:
- [x] 13-01-PLAN.md — Validation helpers and action handlers
- [x] 13-02-PLAN.md — Main handler and driver integration

### Phase 14: Flow & Integration
**Goal**: Phase executes correctly with receivership auto-buy and proper transitions
**Depends on**: Phase 13
**Requirements**: RECV-01, RECV-02, RECV-03, FLOW-01, FLOW-02, FLOW-03, FLOW-04, DRIVER-01, DRIVER-02, DRIVER-03
**Success Criteria** (what must be TRUE):
  1. Receivership corps automatically buy affordable FI offers without player action
  2. Receivership corps cannot sell companies (no offers generated for receivership sellers)
  3. Phase transitions to CLOSING when no more valid offers exist
  4. Acquisition zones merge into owned_companies and corp cash at phase end
  5. Action mask returns valid price options for player-president offers
**Plans**: 3 plans in 3 waves

Plans:
- [ ] 14-01-PLAN.md — Receivership auto-buy in presentation loop
- [ ] 14-02-PLAN.md — Zone merging and phase transition to CLOSING
- [ ] 14-03-PLAN.md — Integration tests for flow and receivership

### Phase 15: Testing
**Goal**: Comprehensive test coverage validates ACQUISITION phase correctness
**Depends on**: Phase 14
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-05, TEST-06, TEST-07
**Success Criteria** (what must be TRUE):
  1. Unit tests verify offer generation produces correct priority ordering
  2. Unit tests verify each action type (price accept, FI high, FI face, pass) behaves correctly
  3. Unit tests verify all validation rules reject invalid actions appropriately
  4. Integration tests verify INVEST->WRAP_UP->ACQUISITION->CLOSING flow works end-to-end
  5. Edge case tests verify behavior with no valid offers, all receivership, empty FI
**Plans**: TBD

Plans:
- [ ] 15-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 12 -> 13 -> 14 -> 15

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 12. Offer Infrastructure | v4.0 | 4/4 | Complete | 2026-01-25 |
| 13. Actions & Validation | v4.0 | 2/2 | Complete | 2026-01-25 |
| 14. Flow & Integration | v4.0 | 0/3 | Planned | - |
| 15. Testing | v4.0 | 0/? | Not started | - |
