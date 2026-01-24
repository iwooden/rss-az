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
- [ ] **OFFER-02**: OS->FI offers come first (OS always has FI priority)
- [ ] **OFFER-03**: Other Corp->FI offers sorted by descending share price
- [ ] **OFFER-04**: Corp->Corp offers (same president) sorted by (buyer share price, target face value)
- [ ] **OFFER-05**: Corp->Player private company offers sorted by (buyer share price, target face value)

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
- [ ] **VALID-03**: Seller corp must keep >=1 company (owned + acquisition_companies after sale)
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
- [ ] **TEST-06**: Integration tests covering INVEST->WRAP_UP->ACQUISITION->CLOSING flow
- [ ] **TEST-07**: Edge case tests (no valid offers, all corps in receivership, empty FI, etc.)

---

## Traceability

| Requirement | Phase | Plan | Status |
|-------------|-------|------|--------|
| OFFER-01 | 12 | TBD | Pending |
| OFFER-02 | 12 | TBD | Pending |
| OFFER-03 | 12 | TBD | Pending |
| OFFER-04 | 12 | TBD | Pending |
| OFFER-05 | 12 | TBD | Pending |
| STATE-01 | 12 | TBD | Pending |
| STATE-02 | 12 | TBD | Pending |
| STATE-03 | 12 | TBD | Pending |
| STATE-04 | 12 | TBD | Pending |
| ACTION-01 | 13 | TBD | Pending |
| ACTION-02 | 13 | TBD | Pending |
| ACTION-03 | 13 | TBD | Pending |
| ACTION-04 | 13 | TBD | Pending |
| VALID-01 | 13 | TBD | Pending |
| VALID-02 | 13 | TBD | Pending |
| VALID-03 | 13 | TBD | Pending |
| VALID-04 | 13 | TBD | Pending |
| VALID-05 | 13 | TBD | Pending |
| VALID-06 | 13 | TBD | Pending |
| RECV-01 | 14 | TBD | Pending |
| RECV-02 | 14 | TBD | Pending |
| RECV-03 | 14 | TBD | Pending |
| FLOW-01 | 14 | TBD | Pending |
| FLOW-02 | 14 | TBD | Pending |
| FLOW-03 | 14 | TBD | Pending |
| FLOW-04 | 14 | TBD | Pending |
| DRIVER-01 | 14 | TBD | Pending |
| DRIVER-02 | 14 | TBD | Pending |
| DRIVER-03 | 14 | TBD | Pending |
| TEST-01 | 15 | TBD | Pending |
| TEST-02 | 15 | TBD | Pending |
| TEST-03 | 15 | TBD | Pending |
| TEST-04 | 15 | TBD | Pending |
| TEST-05 | 15 | TBD | Pending |
| TEST-06 | 15 | TBD | Pending |
| TEST-07 | 15 | TBD | Pending |

---

## Out of Scope

- Inter-player acquisition negotiation - simplified for AlphaZero training
- FI intervention/preemption mechanics - handled via sorted offer priority
- Price negotiation within spans - fixed accept/reject model
