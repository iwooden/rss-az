# Requirements: Rolling Stock Stars v5.0

**Defined:** 2026-01-26
**Core Value:** Fast, reproducible game simulation for AI training with full rules compliance

## v5.0 Requirements

Requirements for CLOSING phase implementation. Each maps to roadmap phases.

### Auto-Close (Phase Start)

- [x] **CLO-01**: FI closes companies where Cost of Ownership >= Income
- [x] **CLO-02**: Receivership corps close red companies if CoO >= 4
- [x] **CLO-03**: Receivership corps close orange companies if CoO >= 7
- [x] **CLO-04**: Receivership corps always keep highest face value company (can't close if only 1 company)

### Offer Generation

- [x] **CLO-05**: Only offer companies with negative adjusted income
- [x] **CLO-06**: Sort offers by face value ascending (lowest first)
- [x] **CLO-07**: Include player-owned privates in offers
- [x] **CLO-08**: Include corp subsidiaries (same-president) in offers

### Offer Validation

- [x] **CLO-09**: Corp closing offer invalid if corp would have 0 companies after close
- [x] **CLO-10**: Re-validate offers as they are processed (prior acceptance can invalidate later offers)

### Actions

- [x] **CLO-11**: Accept action closes the company (removes from game)
- [x] **CLO-12**: Pass action keeps the company
- [x] **CLO-13**: Junkyard Scrappers receives 2x printed income as scrapping bonus when closing

### Mandatory Close (Phase End)

- [x] **CLO-14**: Auto-close player private companies if total income would cause negative cash in INCOME
- [x] **CLO-15**: Close cheapest negative-income company first when mandatory closing

### Phase Transition

- [x] **CLO-16**: Transition to INCOME phase when all offers processed

## Future Requirements

Deferred to subsequent milestones.

### INCOME Phase

- **INC-01**: Players, corps, FI collect income (sum of company incomes - CoO)
- **INC-02**: FI receives +5 base income
- **INC-03**: Corp synergy income calculation
- **INC-04**: Corp special abilities (PR +1/company, DA double highest, S synergy bonus, VM -10 CoO)
- **INC-05**: Corp bankruptcy when can't pay negative income

### DIVIDENDS Phase

- **DIV-01**: Corps pay dividends in share price order (descending)
- **DIV-02**: Dividend amount selection (0 to max per share price card)
- **DIV-03**: Share price adjustment based on star requirements

### Other Phases

- **END-01**: End card detection and game termination
- **ISS-01**: Share issuance in share price order
- **IPO-01**: Private company IPO with corporation formation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Inter-player closing negotiation | Phase is deterministic/offer-based per AlphaZero design |
| Multiple companies per close action | Rules specify one company at a time |
| Closing profitable companies | Only negative-income companies strategically relevant |
| Corp bankruptcy during CLOSING | Corps go bankrupt during INCOME, not CLOSING |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLO-01 | Phase 16 | Complete |
| CLO-02 | Phase 16 | Complete |
| CLO-03 | Phase 16 | Complete |
| CLO-04 | Phase 16 | Complete |
| CLO-05 | Phase 17 | Complete |
| CLO-06 | Phase 17 | Complete |
| CLO-07 | Phase 17 | Complete |
| CLO-08 | Phase 17 | Complete |
| CLO-09 | Phase 17 | Complete |
| CLO-10 | Phase 17 | Complete |
| CLO-11 | Phase 17 | Complete |
| CLO-12 | Phase 17 | Complete |
| CLO-13 | Phase 17 | Complete |
| CLO-14 | Phase 18 | Complete |
| CLO-15 | Phase 18 | Complete |
| CLO-16 | Phase 18 | Complete |

**Coverage:**
- v5.0 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

---
*Requirements defined: 2026-01-26*
*Last updated: 2026-01-27 after Phase 18 complete*
