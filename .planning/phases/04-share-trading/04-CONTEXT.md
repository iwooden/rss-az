# Phase 4: Share Trading - Context

**Gathered:** 2026-01-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Players can buy and sell shares during INVEST phase with proper price movement and trading limits. This phase implements the core share trading mechanics within the existing INVEST phase flow. Presidency changes and bankruptcy are handled in Phase 5.

</domain>

<decisions>
## Implementation Decisions

### Price Movement Logic
- Price moves BEFORE payment (player pays new price)
- When price rises, skip occupied market spaces to find next available
- Price 75 is special: multiple corps can share it (never considered "occupied")
- Price 0 triggers immediate bankruptcy
- If no lower space available when selling, price goes to 0 (bankruptcy)
- Buy validity requires computing actual new price (after skipping occupied) to check affordability
- Sell is always valid if player owns shares (worst case: bankruptcy)
- Stock Masters special (no price move) only applies to Issue Share (Phase 8), NOT player buy/sell

### Round-trip Tracking
- Custom rule to prevent AI training loops (not in official rules)
- Round-trip = 1 buy + 1 sell (any order) for a (player, corp) pair
- Counter = min(buys, sells) per player per corp
- Limit = MAX_ROUNDTRIPS (2) - once counter hits 2, BOTH buy AND sell blocked for that corp
- Clear counters at END of INVEST phase
- State storage already exists: `share_buys[NUM_CORPS]`, `share_sells[NUM_CORPS]` per player

### Buy Availability
- Source: Bank only (issued shares held by bank)
- Cannot buy if bank has no shares of that corp
- No certificate limit (players can hold any number of shares)
- Any player can buy any corp's shares (may trigger presidency change in Phase 5)
- Buy validity requires:
  1. Bank has >= 1 share of that corp
  2. Player can afford new price (after finding next-available higher space)
  3. Round-trip limit not exceeded: min(buys, sells) < 2

### Sell Availability
- Shares are fungible (no distinct "president share" card)
- Player can sell any share they own
- Selling last share triggers receivership (Phase 5 handling)
- Sell validity requires:
  1. Player owns >= 1 share of that corp
  2. Round-trip limit not exceeded: min(buys, sells) < 2

### Action Encoding
- Existing structure: `buy_share_base + corp_id` and `sell_share_base + corp_id` (8 each)
- Round-trip check must be added to mask generation (`_fill_invest_mask`)
- Price 75 must be treated as always available in buy price calculation

### Claude's Discretion
- Exact implementation of "skip occupied spaces" loop
- Helper function organization
- Test structure for edge cases

</decisions>

<specifics>
## Specific Ideas

- Game end triggers immediately when price reaches 75 via buy (not at Phase 7)
- Inactive corps use existing convention in code for price tracking (check current implementation)
- Follow existing entity patterns (cdef noexcept nogil functions for performance)

</specifics>

<deferred>
## Deferred Ideas

- Presidency change mechanics - Phase 5
- Bankruptcy procedure - Phase 5
- Receivership handling - Phase 5
- Net worth recalculation timing - may be Phase 5 or 6

</deferred>

---

*Phase: 04-share-trading*
*Context gathered: 2026-01-20*
