# Phase 17: Offer-Based Close Flow - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Player decision system for closing negative-income companies via structured offers. Players and corps they preside over receive offers to close (remove from game) companies with negative adjusted income. FI and receivership corps are excluded (handled by Phase 16 auto-close). This follows the hybrid phase pattern from ACQUISITION.

</domain>

<decisions>
## Implementation Decisions

### Offer Ordering Logic
- Face values are unique — no tie-breaking needed
- Sort by face value **ascending** (lowest first) — cheapest companies offered first
- This aligns with Phase 18 mandatory closing which also closes cheapest first
- Rationale: Consistent ordering makes bankruptcy prevention logic simpler

### Hybrid Phase Mechanics
- Follow ACQUISITION pattern: non-player phase when no offers exist, player phase when offers exist
- Offers generated immediately after auto-close completes (same entry point)
- All offers generated and ordered upfront, presented one at a time
- Re-validate each offer before presenting (acceptance can invalidate later offers)
- One offer at a time — player acts, then next offer presented

### President Control Scope
- Offers include: player's private companies + companies owned by corps player presides over
- If player is president of multiple corps, all companies in single pool (mixed with privates)
- Receivership corps: excluded from offers (no president, handled by auto-close only)
- FI-owned companies: excluded from offers (handled by auto-close only)

### Action Validation Edge Cases
- Corp last-company rule: don't offer closing if corp has exactly 1 company remaining
- Check actual remaining count at offer presentation time, not at generation time
- If first of two corp offers accepted, second becomes invalid (skip it)
- If first of two corp offers passed, second remains valid
- Players CAN close their last private company (unlike corps)

### Claude's Discretion
- Internal data structures for offer queue
- Specific validation function organization
- Error handling for unexpected states

</decisions>

<specifics>
## Specific Ideas

- "The ascending order (cheapest first) will make it somewhat easier to handle the enforcement of company closures to avoid player bankruptcy" — user rationale for sorting choice
- Follow ACQUISITION patterns for hybrid phase detection and offer processing
- Use existing `_close_company()` from Phase 16 for actual closure (already handles JS bonus)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 17-offer-based-close-flow*
*Context gathered: 2026-01-27*
