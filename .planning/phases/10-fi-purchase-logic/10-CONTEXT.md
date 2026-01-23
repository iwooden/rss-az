# Phase 10: FI Purchase Logic - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Foreign Investor purchases cheapest available companies at face value during WRAP_UP, then all unavailable companies become available. This is deterministic game logic with no player decisions. FI purchases are integrated into the existing WRAP_UP phase handler from Phase 9.

</domain>

<decisions>
## Implementation Decisions

### Purchase Order
- FI buys in ascending face value order (rules specify this)
- All company face values are unique — no tie-breaking needed
- Re-check available companies after each purchase (not snapshot)
- Always buy cheapest affordable company (if FI skipped X because too expensive, re-check X after buying cheaper companies)
- Newly drawn companies are unavailable during this WRAP_UP (same as auction — reuse existing draw logic)

### Edge Case Handling
- If FI has 0 cash: Claude decides whether to skip loop or let it naturally terminate (optimization choice)
- If deck is empty: no card drawn after purchase, FI continues buying remaining available companies
- If no available companies: skip gracefully, proceed directly to availability transition
- If no affordable companies: silent proceed — no history logging for non-mutations

### History Recording
- All WRAP_UP operations (player reorder + FI purchases + availability transition) combine into single `(state, ACTION_WRAP_UP_SENTINEL)` history entry
- State captured BEFORE any WRAP_UP logic runs
- FI purchases do NOT get individual history entries — batched into phase

### Availability Transition
- Always runs at end of WRAP_UP, regardless of whether FI made purchases
- Loop through all `LOC_REVEALED` companies and call `move_to_auction()` on each
- No special ordering needed — company_id order implicitly matches face value order for auction slot representation

### Claude's Discretion
- Whether to short-circuit FI purchase loop when cash is 0
- Implementation details of the re-check loop (while vs iteration pattern)

</decisions>

<specifics>
## Specific Ideas

- FI purchase loop can potentially reuse code from auction resolution (drawing new card into revealed/unavailable state)
- The `get_auction_company_for_slot` function already handles face value ordering via company_id iteration
- `transfer_to_fi()` and `move_to_auction()` methods already exist on Company entity

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-fi-purchase-logic*
*Context gathered: 2026-01-23*
