# Roadmap: Rolling Stock Stars v5.0 CLOSING Phase

## Overview

The CLOSING phase implementation extends the game engine with company closure mechanics. Players and corporations can close (remove) negative-income companies from the game, with auto-close rules for FI and receivership corps at phase start, offer-based player decisions during the phase, and mandatory auto-close at phase end to protect players from negative cash in INCOME. This milestone follows the hybrid phase pattern established in ACQUISITION.

## Milestones

- ✅ **v1.0** - Game State Init - Phases 1 (shipped 2026-01-20)
- ✅ **v2.0** - INVEST & BID_IN_AUCTION - Phases 2-6 (shipped 2026-01-21)
- ✅ **v2.1** - Forced Action Auto-Application - Phases 7-8 (shipped 2026-01-23)
- ✅ **v3.0** - WRAP_UP Phase - Phases 9-11 + 10.1 (shipped 2026-01-24)
- ✅ **v4.0** - ACQUISITION Phase - Phases 12-15 (shipped 2026-01-26)
- 🚧 **v5.0** - CLOSING Phase - Phases 16-19 (in progress)

## Phases

- [ ] **Phase 16: Auto-Close Logic** - FI and receivership corps auto-close at phase start
- [ ] **Phase 17: Offer-Based Close Flow** - Player decisions on negative-income companies
- [ ] **Phase 18: Mandatory Close and Transition** - Auto-close at phase end, transition to INCOME
- [ ] **Phase 19: Testing and Integration** - Comprehensive test coverage

## Phase Details

### Phase 16: Auto-Close Logic
**Goal**: FI and receivership corps automatically close unprofitable companies at phase start
**Depends on**: Phase 15 (ACQUISITION complete)
**Requirements**: CLO-01, CLO-02, CLO-03, CLO-04
**Success Criteria** (what must be TRUE):
  1. FI closes any company where Cost of Ownership >= Income
  2. Receivership corp closes red companies when CoO >= 4
  3. Receivership corp closes orange companies when CoO >= 7
  4. Receivership corp always retains highest face value company (never closes last company)
**Plans**: TBD

Plans:
- [ ] 16-01: TBD

### Phase 17: Offer-Based Close Flow
**Goal**: Players can decide to close or keep negative-income companies via offer system
**Depends on**: Phase 16
**Requirements**: CLO-05, CLO-06, CLO-07, CLO-08, CLO-09, CLO-10, CLO-11, CLO-12, CLO-13
**Success Criteria** (what must be TRUE):
  1. Offers generated only for companies with negative adjusted income
  2. Offers sorted by face value descending (highest value offered first)
  3. Player-owned privates and corp subsidiaries (same-president) included in offers
  4. Offer validation: corp closing offers invalid if corp would have 0 companies after close
  5. Dynamic re-validation: prior acceptance can invalidate later offers (corp down to 1 company)
  6. Accept action removes company from game; Pass action keeps company
  7. Junkyard Scrappers receives 2x printed income as bonus when closing
**Plans**: TBD

Plans:
- [ ] 17-01: TBD
- [ ] 17-02: TBD

### Phase 18: Mandatory Close and Transition
**Goal**: Auto-close at phase end protects players from negative cash, then transition to INCOME
**Depends on**: Phase 17
**Requirements**: CLO-14, CLO-15, CLO-16
**Success Criteria** (what must be TRUE):
  1. Players with negative total income have privates auto-closed until income non-negative
  2. Cheapest negative-income company closed first during mandatory closing
  3. Phase transitions to INCOME when all offers processed and mandatory closes complete
**Plans**: TBD

Plans:
- [ ] 18-01: TBD

### Phase 19: Testing and Integration
**Goal**: Comprehensive test coverage validates CLOSING phase correctness
**Depends on**: Phase 18
**Requirements**: None (testing phase)
**Success Criteria** (what must be TRUE):
  1. Unit tests cover all 16 requirements individually
  2. Integration tests verify ACQUISITION -> CLOSING -> INCOME flow
  3. Edge case tests cover empty offers, all-pass, multi-close scenarios
  4. All existing tests pass (no regressions)
**Plans**: TBD

Plans:
- [ ] 19-01: TBD

## Progress

**Execution Order:** 16 -> 17 -> 18 -> 19

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 16. Auto-Close Logic | v5.0 | 0/? | Not started | - |
| 17. Offer-Based Close Flow | v5.0 | 0/? | Not started | - |
| 18. Mandatory Close and Transition | v5.0 | 0/? | Not started | - |
| 19. Testing and Integration | v5.0 | 0/? | Not started | - |

---
*Roadmap created: 2026-01-26*
*Last updated: 2026-01-26*
