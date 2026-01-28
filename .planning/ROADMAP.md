# Roadmap: Rolling Stock Stars v6.0 INCOME Phase

## Overview

Implement the INCOME phase where all entities (players, corporations, Foreign Investor) collect income from owned companies, applying Cost of Ownership deductions, synergy bonuses, and corporation special abilities. This phase follows CLOSING and precedes DIVIDENDS (temporary transition to INVEST until DIVIDENDS is implemented).

## Milestones

- v1.0 through v5.1: See `.planning/milestones/` archives
- **v6.0 INCOME Phase** - Phases 21-23 (in progress)

## Phases

**Phase Numbering:**
- Continues from v5.1 (Phase 20 was last)
- Integer phases (21, 22, 23): Planned milestone work
- Decimal phases (21.1, etc.): Urgent insertions if needed

- [ ] **Phase 21: Synergy Infrastructure** - Data structures for synergy pair identification
- [ ] **Phase 22: Income Calculation** - Core income flow with CoO, synergies, and special abilities
- [ ] **Phase 23: Phase Integration** - INCOME phase handler with transitions and bankruptcy

## Phase Details

### Phase 21: Synergy Infrastructure
**Goal**: Synergy pairs can be identified between companies owned by the same corporation
**Depends on**: Nothing (first phase of v6.0)
**Requirements**: SYN-01, SYN-02
**Success Criteria** (what must be TRUE):
  1. Given a corporation, all synergy pairs among its companies are identified
  2. Each synergy pair is counted exactly once (A-B counted once, not A-B and B-A)
  3. Synergy count for a corporation with no synergies returns 0
  4. Synergy count works with 0, 1, or many companies owned
**Plans**: 1 plan

Plans:
- [ ] 21-01-PLAN.md — TDD: Synergy pair identification and counting

### Phase 22: Income Calculation
**Goal**: All entities can calculate their total income with all modifiers applied
**Depends on**: Phase 21 (synergy infrastructure)
**Requirements**: INC-01, INC-02, INC-03, INC-04, INC-05, SYN-03, CSA-01, CSA-02, CSA-03, CSA-04
**Success Criteria** (what must be TRUE):
  1. Entity income sums printed income from all owned companies minus Cost of Ownership
  2. Foreign Investor receives +5 base income bonus on top of company income
  3. Corporation with PR ability receives +1 per company owned
  4. Corporation with DA ability doubles income of highest face value company
  5. Corporation with S ability receives +1 per 2 synergy markers (rounded down)
  6. Corporation with VM ability reduces total CoO by up to 10 (min 0)
  7. Positive income adds to entity cash, negative income subtracts from entity cash
**Plans**: TBD

Plans:
- [ ] 22-01: Base income calculation (INC-01, INC-02, INC-03)
- [ ] 22-02: Corporation special abilities (CSA-01, CSA-02, CSA-03, CSA-04, SYN-03)
- [ ] 22-03: Income application (INC-04, INC-05)

### Phase 23: Phase Integration
**Goal**: INCOME phase executes as non-player phase with correct transitions and bankruptcy handling
**Depends on**: Phase 22 (income calculation)
**Requirements**: INC-06, TRN-01, TRN-02, TRN-03, TRN-04
**Success Criteria** (what must be TRUE):
  1. CLOSING transitions to INCOME (not INVEST)
  2. INCOME executes as non-player phase (0 valid actions, auto-executes)
  3. Corporation that cannot pay negative income executes bankruptcy procedure
  4. INCOME increments turn counter and clears roundtrip flags (end-of-turn logic)
  5. INCOME transitions to INVEST (temporary until DIVIDENDS implemented)
**Plans**: TBD

Plans:
- [ ] 23-01: Remove CLOSING temporary transition (TRN-01)
- [ ] 23-02: INCOME phase handler (TRN-02, TRN-03, TRN-04)
- [ ] 23-03: Corporation bankruptcy on negative income (INC-06)

## Progress

**Execution Order:** 21 -> 22 -> 23

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 21. Synergy Infrastructure | 0/1 | Planned | - |
| 22. Income Calculation | 0/3 | Not started | - |
| 23. Phase Integration | 0/3 | Not started | - |

---
*Created: 2026-01-28*
*Milestone: v6.0 INCOME Phase*
