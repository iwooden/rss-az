# Requirements — v3.0 WRAP_UP Phase

## Milestone Requirements

### Player Reordering

- [ ] **REORDER-01**: Reorder players by descending cash at start of WRAP_UP
- [ ] **REORDER-02**: Preserve old turn order for tie-breaking (equal cash → lower old position wins)
- [ ] **REORDER-03**: Update active player to new position 0 after reordering

### Foreign Investor Purchases

- [ ] **FI-01**: FI buys cheapest available company at face value
- [ ] **FI-02**: Process purchases in ascending face value order (cheapest first)
- [ ] **FI-03**: Draw new card after each purchase and mark unavailable
- [ ] **FI-04**: Stop purchasing when FI cannot afford any remaining available company
- [ ] **FI-05**: Handle edge case: FI has 0 cash (skip purchase loop)
- [ ] **FI-06**: Handle edge case: deck empty after purchase (no new card drawn)
- [ ] **FI-07**: Handle edge case: no available companies (skip purchase loop)

### Company Availability

- [ ] **AVAIL-01**: After FI purchases complete, all unavailable companies become available

### Phase Transitions

- [ ] **PHASE-01**: Trigger WRAP_UP when all players pass consecutively in INVEST
- [ ] **PHASE-02**: WRAP_UP transitions to next phase (ACQUISITION when implemented, else new INVEST turn)
- [ ] **PHASE-03**: Loosen 0-action invariant: allow 0 actions for non-player phases (WRAP_UP, INCOME, etc.)
- [ ] **PHASE-04**: WRAP_UP gets discrete state history entry (not absorbed into INVEST transition)

### Testing

- [ ] **TEST-01**: Fix existing INVEST/auction tests that now auto-continue past WRAP_UP
- [ ] **TEST-02**: Add `set_phase()` method to Turn entity for test utilities
- [ ] **TEST-03**: Add player order verification tests (INVEST → WRAP_UP → set_phase(INVEST) → verify order)

---

## Future Requirements

(None — all WRAP_UP features are table stakes for this milestone)

---

## Out of Scope

- **ACQUISITION phase** — Next milestone (v3.1 or v4.0)
- **Other game phases** — CLO, INC, DIV, END, ISS, IPO deferred to future milestones
- **FI auction fallback** — Edge case where no player bids, defer

---

## Traceability

| REQ-ID | Phase | Plan |
|--------|-------|------|
| REORDER-01 | TBD | — |
| REORDER-02 | TBD | — |
| REORDER-03 | TBD | — |
| FI-01 | TBD | — |
| FI-02 | TBD | — |
| FI-03 | TBD | — |
| FI-04 | TBD | — |
| FI-05 | TBD | — |
| FI-06 | TBD | — |
| FI-07 | TBD | — |
| AVAIL-01 | TBD | — |
| PHASE-01 | TBD | — |
| PHASE-02 | TBD | — |
| PHASE-03 | TBD | — |
| PHASE-04 | TBD | — |
| TEST-01 | TBD | — |
| TEST-02 | TBD | — |
| TEST-03 | TBD | — |

---
*18 requirements across 5 categories*
