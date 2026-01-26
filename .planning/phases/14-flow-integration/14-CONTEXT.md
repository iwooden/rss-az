# Phase 14: Flow & Integration - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate receivership auto-buy logic, phase transitions (ACQUISITION → CLOSING), and driver integration with action masking. The offer infrastructure (Phase 12) and action handlers (Phase 13) are complete — this phase wires them together into a working flow.

</domain>

<decisions>
## Implementation Decisions

### Receivership auto-buy
- Receivership corps MUST buy from FI if they can afford the face value price (per RULES.md)
- Offers are processed one at a time in priority order — receivership corps follow the same offer ordering as everyone else
- When an offer is presented and the buying corp is in receivership:
  - If corp can afford the required price → auto-execute immediately (no player involvement)
  - If corp cannot afford → auto-pass (skip the offer)
- Receivership handling happens in the offer presentation loop BEFORE driver involvement
- Driver only sees states where a player can act (player-president offers)

### Phase transition triggers
- Phase transitions immediately to CLOSING when all offers are exhausted (buffer empty or all remaining invalid)
- No explicit "end phase" action needed
- Offer buffer is generated once at phase start, invalid offers are skipped at presentation time (no regeneration)

### Zone merging at phase end
- On phase exit (transition to CLOSING), merge acquisition zones into final state:
  - `acquisition_proceeds[player]` → add to `player.cash`, then zero out
  - `acquisition_proceeds[corp]` → add to `corp.cash`, then zero out
  - `acquisition_companies[corp]` → merge into `owned_companies[corp]`, then clear
- Both player and corp proceeds/companies must be merged and cleared

### Action mask behavior
- FI offers: OS buys at face value (special ability), all other corps buy at high price — price offsets NOT valid
- Non-FI offers: Valid actions are price offsets from 0 (low price) up to `high_price - low_price`
- Affordability checked against corp cash
- PASS is always a valid action (declines offer, won't be offered again this phase)
- Receivership offers never appear in action mask — handled before driver sees them

### Error handling
- Invalid offers after buffer generation (e.g., target already acquired) → silently skip
- Zone merge encountering inconsistent state → hard error (indicates bug)
  - Company doesn't exist → error
  - Company already in owned_companies → error (invariant: company can only be in one location)
- Buffer size (250 offers) is the safety limit — exceeding it indicates a bug needing investigation

### Claude's Discretion
- Exact implementation of offer presentation loop
- Helper function organization
- Test coverage strategy for edge cases

</decisions>

<specifics>
## Specific Ideas

- Follow the existing forced-action auto-apply pattern from driver.pyx (lines 203-230)
- Receivership auto-buy mirrors the non-player phase pattern (deterministic, no mask)
- "If corp can't afford any action, don't make the offer" — offer filtering handles affordability

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-flow-integration*
*Context gathered: 2026-01-26*
