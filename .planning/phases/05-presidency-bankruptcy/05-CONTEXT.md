# Phase 5: Presidency & Bankruptcy - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Corporation ownership transfers correctly and bankruptcy procedure completes cleanly during INVEST phase share trading. Includes presidency changes after buy/sell actions, receivership entry/exit, and price-drop bankruptcy handling.

**Out of scope:** Income-phase bankruptcy (Phase 5 game phase), receivership auto-actions for phases 3/4/8.

</domain>

<decisions>
## Implementation Decisions

### Presidency Transfer
- Check for presidency transfer **after each share trade** (not batched to end of turn)
- Tie-breaking: **Current president keeps it** when shares are equal
- Presidency is mandatory — players cannot refuse
- Logic simplifies to: find player with most shares; if tied with incumbent, incumbent keeps

### Receivership Mechanics
- **Entry trigger:** All players combined have 0 shares (Bank owns all issued shares)
- **Exit trigger:** Player buys a share from receivership corp (normal "most shares = president" logic applies)
- **State tracking:** Use existing `in_receivership` flag in corp state (offset 9 in corp stride)
- Shares are **fungible** — no special "President's Share" card treatment
- **Deferred:** Receivership auto-actions for phases 3, 4, 8 — implement in later milestone

### Bankruptcy Procedure
- **Trigger:** Share price drops to 0 during sell action (price-drop bankruptcy)
- **Timing:** Execute bankruptcy **immediately inline** during sell handler (no deferral)
- **Company removal:** Just removed from game — no closing procedure, no Junkyard Scrappers bonus
- **Corp reset:** Reset to initial state (same as pre-IPO state after game initialization)
- Steps: remove companies → collect all shares (reset to unissued) → return money to bank → return price card to row → clear corp state

### Order of Operations (Sell Action)
1. Transfer share from player to bank
2. Move price down to next available space
3. Pay player the new (lower) price
4. **Check if price = 0 → Bankruptcy** (if yes, execute bankruptcy procedure)
5. **Check receivership** (if total player shares = 0, set in_receivership)
6. **Check presidency** (only if not in receivership, find player with most shares)
7. Update player net worth

### State Updates
- State vector must be **accurate after every action** for model training
- No batching or deferring updates — immediate state consistency
- Buy/sell may require updating multiple state locations:
  - Player share counts
  - Player cash and net worth
  - Corp bank_shares
  - Corp in_receivership flag
  - Player is_president flags
  - Market availability (if bankruptcy returns price card)

### Claude's Discretion
- Exact order of state updates within each handler (as long as final state is correct)
- Helper function organization
- Test case prioritization

</decisions>

<specifics>
## Specific Ideas

- "Think of receivership check first, presidency check second — if receivership triggers, skip presidency check (no president in receivership)"
- "Players should always be able to sell — standard logic applies regardless of resulting state"
- "Affordability check already exists in action mask — applies equally to receivership corps"
- "State vector is optimized for model, not game engine — may cache in hidden section, update multiple locations per action"

</specifics>

<deferred>
## Deferred Ideas

- Receivership auto-actions for Phase 3 (acquisition), Phase 4 (closing), Phase 8 (issue) — future milestone
- Income-phase bankruptcy (corp can't pay negative income) — future milestone
- Cascading bankruptcies — not applicable in INVEST phase (one corp per sell action)

</deferred>

---

*Phase: 05-presidency-bankruptcy*
*Context gathered: 2026-01-21*
