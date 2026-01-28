# Requirements: Rolling Stock Stars v6.0

**Defined:** 2026-01-28
**Core Value:** Fast, reproducible game simulation for AI training with full rules compliance

## v6.0 Requirements

Requirements for INCOME phase implementation. Enables income collection for all entities with synergy and corporation special abilities.

### Base Income Collection

- [ ] **INC-01**: Each entity (player, corp, FI) sums income from all owned companies
- [ ] **INC-02**: Cost of Ownership deducted from each company's income based on star color and CoO level
- [ ] **INC-03**: Foreign Investor receives +5● base income bonus
- [ ] **INC-04**: Positive total income → entity receives amount from Bank
- [ ] **INC-05**: Negative total income → entity pays amount to Bank
- [ ] **INC-06**: Corporation that cannot pay negative income executes bankruptcy procedure

### Synergy Income

- [ ] **SYN-01**: Synergy pairs identified between companies owned by same corporation
- [ ] **SYN-02**: Each synergy pair counted once (if A synergizes with B, count once not twice)
- [ ] **SYN-03**: Synergy income added to corporation's total before special abilities

### Corporation Special Abilities

- [ ] **CSA-01**: Prussian Railway (PR) receives +1● per company owned
- [ ] **CSA-02**: Doppler AG (DA) doubles printed income of highest Face Value company
- [ ] **CSA-03**: Synergistic (S) receives +1● per 2 synergy markers (rounded down)
- [ ] **CSA-04**: Vintage Machinery (VM) reduces total Cost of Ownership by up to 10● (min 0●)

### Phase Transitions

- [ ] **TRN-01**: CLOSING transitions to INCOME (remove temporary end-of-turn logic from CLOSING)
- [ ] **TRN-02**: INCOME has temporary end-of-turn logic (turn increment, roundtrip clear)
- [ ] **TRN-03**: INCOME transitions to INVEST (temporary until DIVIDENDS implemented)
- [ ] **TRN-04**: INCOME is non-player phase (deterministic, follows WRAP_UP pattern)

## Future Requirements

Deferred to v7.0+ milestones.

### DIVIDENDS Phase

- **DIV-01**: Corporations pay dividends in share price order (descending)
- **DIV-02**: Dividend per share ≥ 0●, ≤ maximum on share price card
- **DIV-03**: Corporation must have enough cash for (dividend × issued shares)
- **DIV-04**: Share price adjustment based on star requirements

### ISSUE_SHARES Phase

- **ISS-01**: Corporations may issue one share in share price order
- **ISS-02**: Receivership corporations must issue if shares available
- **ISS-03**: Stock Masters special ability (no price change on issue)

### IPO Phase

- **IPO-01**: Private companies may form corporations
- **IPO-02**: IPO price based on company color
- **IPO-03**: Share distribution between player and Bank

### END_GAME Phase

- **END-01**: Game end detection (75● share price, game end card)
- **END-02**: Final scoring calculation

## Out of Scope

Explicitly excluded from v6.0.

| Feature | Reason |
|---------|--------|
| DIVIDENDS phase | Separate milestone after INCOME |
| ISSUE_SHARES phase | Separate milestone |
| IPO phase | Separate milestone |
| END_GAME detection | Separate milestone |
| Synergy marker UI tracking | Engine only, visual markers not needed |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INC-01 | Phase [N] | Pending |
| INC-02 | Phase [N] | Pending |
| INC-03 | Phase [N] | Pending |
| INC-04 | Phase [N] | Pending |
| INC-05 | Phase [N] | Pending |
| INC-06 | Phase [N] | Pending |
| SYN-01 | Phase [N] | Pending |
| SYN-02 | Phase [N] | Pending |
| SYN-03 | Phase [N] | Pending |
| CSA-01 | Phase [N] | Pending |
| CSA-02 | Phase [N] | Pending |
| CSA-03 | Phase [N] | Pending |
| CSA-04 | Phase [N] | Pending |
| TRN-01 | Phase [N] | Pending |
| TRN-02 | Phase [N] | Pending |
| TRN-03 | Phase [N] | Pending |
| TRN-04 | Phase [N] | Pending |

**Coverage:**
- v6.0 requirements: 17 total
- Mapped to phases: 0
- Unmapped: 17 ⚠️

---
*Requirements defined: 2026-01-28*
*Last updated: 2026-01-28 after initial definition*
