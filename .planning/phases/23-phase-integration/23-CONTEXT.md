# Phase 23: Phase Integration - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

INCOME phase executes as non-player phase with correct transitions and bankruptcy handling. Integrates income calculation (Phase 22) into the game loop. Creates TEMP_END_TURN phase for consolidated end-of-turn bookkeeping.

</domain>

<decisions>
## Implementation Decisions

### Entity processing order
- Order doesn't matter — entities are independent
- Use existing `calculate_income()` and `apply_income()` methods from entities folder
- Check bankruptcy immediately per-corp after income application (before next entity)
- Multiple corps can go bankrupt in same INCOME phase — handle each in sequence

### Player income handling
- Players CAN have negative income (that's fine)
- Players CANNOT have negative cash balance after income
- Add assertion that fails if player cash < 0 after income application
- Balance of $0 is allowed

### Bankruptcy procedure
- Check if bankruptcy logic already exists in INVEST phase
- If exists: refactor to `entities/corp.pyx` as reusable method
- Single implementation used by both INVEST and INCOME phases
- NO duplicate bankruptcy code
- Per RULES.md (lines 378-385): remove companies, collect shares to charter, return money to Bank, return share price card

### End-of-turn consolidation
- Create TEMP_END_TURN phase for end-of-turn bookkeeping
- Move turn increment to TEMP_END_TURN (remove from ACQUISITION and CLOSING)
- Roundtrip clear should happen at end of INVEST phase (before WRAP_UP), NOT at turn end
- Add comments explaining TEMP_END_TURN is temporary while phases are built out

### Phase transitions
- CLOSING → INCOME (remove temporary CLOSING → INVEST)
- INCOME → TEMP_END_TURN
- TEMP_END_TURN → INVEST

### Phase execution pattern
- Follow WRAP_UP pattern: non-player phase, 0 valid actions, auto-executes
- Single entry point (apply_income) with internal helpers
- Use existing entity methods: Corp.calculate_income(), Corp.apply_income(), FI.calculate_income(), FI.apply_income(), Player.get_income(), Player.add_cash()
- Do NOT write duplicate income calculation code

### Claude's Discretion
- Internal helper function naming and structure
- Exact iteration order over entities (since order doesn't matter)
- TEMP_END_TURN phase file location and structure

</decisions>

<specifics>
## Specific Ideas

- "Ensure all previous phases don't handle any end-of-turn logic" — clean separation
- "If you can't find existing methods, notify me instead of writing duplicate code"
- Roundtrip info only relevant in INVEST phase — clearing it elsewhere pollutes state vector for model

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 23-phase-integration*
*Context gathered: 2026-01-29*
