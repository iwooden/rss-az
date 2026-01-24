# Requirements: v4.0 ACQUISITION Phase

## Overview

AlphaZero-optimized ACQUISITION phase with offer-based flow, same-president restrictions, sorted priority presentation, and receivership auto-execution.

**Key Design Decisions:**
- No inter-player negotiation (same-president restriction)
- Offer-based accept/reject flow (reduced action space)
- Sorted presentation implicitly handles FI priority rules
- Acquisition proceeds zone prevents re-acquisition loops

---

## v4.0 Requirements

### Offer Generation & Priority

- [ ] **OFFER-01**: Generate valid acquisition offers in sorted priority order
- [ ] **OFFER-02**: OS→FI offers come first (OS always has FI priority)
- [ ] **OFFER-03**: Other Corp→FI offers sorted by descending share price
- [ ] **OFFER-04**: Corp→Corp offers (same president) sorted by (buyer share price, target face value)
- [ ] **OFFER-05**: Corp→Player private company offers sorted by (buyer share price, target face value)

### Offer State Management

- [ ] **STATE-01**: Set acq_active_corp, acq_target_company, acq_is_fi_offer for current offer
- [ ] **STATE-02**: Track acquisition_companies per corp (pending acquisitions)
- [ ] **STATE-03**: Track acquisition_proceeds per corp (pending cash)
- [ ] **STATE-04**: Clear acq_active_corp (-1) when no more offers

### Action Handling

- [ ] **ACTION-01**: Accept acquisition at price within [low_price, high_price] range
- [ ] **ACTION-02**: FI Buy High action (buy FI company at max price)
- [ ] **ACTION-03**: FI Buy Face action (OS only, buy FI company at face value)
- [ ] **ACTION-04**: Pass action (decline current offer, advance to next)

### Validation

- [ ] **VALID-01**: Price must be within company's [low_price, high_price] span
- [ ] **VALID-02**: Buyer corp must have sufficient cash
- [ ] **VALID-03**: Seller corp must keep ≥1 company (owned + acquisition_companies after sale)
- [ ] **VALID-04**: Target company cannot already be in acquisition_companies
- [ ] **VALID-05**: Target company cannot be in buyer's owned_companies
- [ ] **VALID-06**: Same-president requirement for corp-to-corp and corp-to-player offers

### Receivership Integration

- [ ] **RECV-01**: Receivership corps auto-buy FI offers if affordable
- [ ] **RECV-02**: Receivership corps cannot sell companies
- [ ] **RECV-03**: Auto-buy executes within offer advancement loop (no player action)

### Phase Flow

- [ ] **FLOW-01**: Advance to next offer after each action (accept or pass)
- [ ] **FLOW-02**: Transition to CLOSING when no more valid offers
- [ ] **FLOW-03**: Merge acquisition_companies into owned_companies at phase end
- [ ] **FLOW-04**: Merge acquisition_proceeds into corp cash at phase end

### Driver Integration

- [ ] **DRIVER-01**: Remove ACQUISITION from _is_non_player_phase()
- [ ] **DRIVER-02**: Action mask returns valid price options for player-president offers
- [ ] **DRIVER-03**: Phase handler transitions to CLOSING internally when no more offers

### Testing

- [ ] **TEST-01**: Unit tests for offer generation and priority sorting
- [ ] **TEST-02**: Unit tests for each action type (price accept, FI high, FI face, pass)
- [ ] **TEST-03**: Unit tests for all validation rules (price range, cash, minimum companies, etc.)
- [ ] **TEST-04**: Unit tests for receivership auto-buy behavior
- [ ] **TEST-05**: Unit tests for phase flow (offer advancement, phase transition, proceeds merge)
- [ ] **TEST-06**: Integration tests covering INVEST→WRAP_UP→ACQUISITION→CLOSING flow
- [ ] **TEST-07**: Edge case tests (no valid offers, all corps in receivership, empty FI, etc.)

---

## Traceability

| Requirement | Phase | Plan |
|-------------|-------|------|
| OFFER-01 | TBD | TBD |
| OFFER-02 | TBD | TBD |
| OFFER-03 | TBD | TBD |
| OFFER-04 | TBD | TBD |
| OFFER-05 | TBD | TBD |
| STATE-01 | TBD | TBD |
| STATE-02 | TBD | TBD |
| STATE-03 | TBD | TBD |
| STATE-04 | TBD | TBD |
| ACTION-01 | TBD | TBD |
| ACTION-02 | TBD | TBD |
| ACTION-03 | TBD | TBD |
| ACTION-04 | TBD | TBD |
| VALID-01 | TBD | TBD |
| VALID-02 | TBD | TBD |
| VALID-03 | TBD | TBD |
| VALID-04 | TBD | TBD |
| VALID-05 | TBD | TBD |
| VALID-06 | TBD | TBD |
| RECV-01 | TBD | TBD |
| RECV-02 | TBD | TBD |
| RECV-03 | TBD | TBD |
| FLOW-01 | TBD | TBD |
| FLOW-02 | TBD | TBD |
| FLOW-03 | TBD | TBD |
| FLOW-04 | TBD | TBD |
| DRIVER-01 | TBD | TBD |
| DRIVER-02 | TBD | TBD |
| DRIVER-03 | TBD | TBD |
| TEST-01 | TBD | TBD |
| TEST-02 | TBD | TBD |
| TEST-03 | TBD | TBD |
| TEST-04 | TBD | TBD |
| TEST-05 | TBD | TBD |
| TEST-06 | TBD | TBD |
| TEST-07 | TBD | TBD |

---

## Out of Scope

- Inter-player acquisition negotiation — simplified for AlphaZero training
- FI intervention/preemption mechanics — handled via sorted offer priority
- Price negotiation within spans — fixed accept/reject model
