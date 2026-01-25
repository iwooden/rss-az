# Phase 13: Actions & Validation - Context

**Gathered:** 2026-01-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement accept/pass actions for acquisition offers with full validation. Players (as corp presidents) can accept offers at valid prices or pass. Invalid actions return error code. This phase covers the action mechanics — receivership auto-buy and phase transitions are Phase 14.

</domain>

<decisions>
## Implementation Decisions

### Action format
- Use existing action definitions from VECTORS.md:
  - `+0 to +50`: Price Offer — `price = low_price + (idx - base)` for non-FI offers
  - `+51`: FI Buy High — Buy FI company at high price (non-OS corps)
  - `+52`: FI Buy Face — Buy FI company at face (OS only)
  - `+53`: Pass — Decline acquisition
- Seller is implied from current offer state (acq_target_company)
- Action mask only allows valid actions — never 0 valid actions (offers filtered at generation)
- Invalid actions return `1` (matching INVEST phase pattern), driver handles error

### FI offer handling
- **OS buying from FI**: Only FI Buy Face (+52) is valid, never FI Buy High
  - OS always uses its face-value privilege, no option to overpay
  - Applies whether OS is player-controlled or in receivership
- **Other corps buying from FI**: Only FI Buy High (+51) is valid
- Price range actions (+0 to +50) are NOT valid for FI offers
- FI receives the price paid (face or high depending on action)

### Validation rules
- All actions must be affordable — no negative cash balances ever
- FI Buy Face: valid only if `corp_cash >= face_value`
- FI Buy High: valid only if `corp_cash >= high_price`
- Price offers: valid only if `corp_cash >= (low_price + price_offset)`
- Corp seller must retain at least 1 company (in owned_companies OR acquisition_companies combined)
- Players can sell all their privates — no minimum ownership requirement
- Acquisitions are one-way: corps buy from players/FI/other corps, never reverse

### Pass behavior
- Pass permanently skips the current offer — hidden offer_index advances
- Pass immediately presents the next offer (internal loop, not separate driver call)
- Passed offers are never re-offered this phase

### Claude's Discretion
- Internal function organization
- Helper function naming
- Test structure for validation edge cases

</decisions>

<specifics>
## Specific Ideas

- Match INVEST phase pattern for invalid action handling (return 1)
- Phase 12 already filters unaffordable offers — action mask validation is a secondary check
- Receivership auto-buy logic is Phase 14 scope, but validation groundwork applies

</specifics>

<deferred>
## Deferred Ideas

- Receivership auto-buy mechanics — Phase 14
- Phase transitions (ACQUISITION → CLOSING) — Phase 14
- Acquisition zone merge at phase end — Phase 14

</deferred>

---

*Phase: 13-actions-validation*
*Context gathered: 2026-01-25*
