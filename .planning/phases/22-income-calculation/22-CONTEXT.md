# Phase 22: Income Calculation - Context

**Gathered:** 2026-01-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Calculate and apply income for all entities (players, corporations, Foreign Investor) with Cost of Ownership deductions, synergy bonuses, and corporation special abilities. Phase 21 provides synergy infrastructure. Phase 23 handles the INCOME phase handler and bankruptcy.

</domain>

<decisions>
## Implementation Decisions

### Modifier Ordering
- Formula: `printed_income - CoO + synergy + special_abilities = total_income`
- Sum printed income first (gross), then subtract CoO, then add bonuses
- Synergy is added AFTER CoO calculation — not subject to Cost of Ownership
- Special abilities (PR, DA, S, VM) order doesn't matter — all additive to final sum
- VM operates on **total** CoO, which is why gross income is calculated first

### Entity Differences
- Players: `sum(income) - sum(CoO)` — no synergies, no special abilities
- Foreign Investor: `sum(income) - sum(CoO) + 5` — fixed +5 bonus, no synergies
- Corporations: Full calculation with synergies and special abilities
- Implement corp income as method in `/entities/corp.pyx` for reusability
- Implement FI income as method in `/entities/fi.pyx` to match pattern
- Reuse Phase 18's player income pattern, ensure CoO is handled correctly

### Ability Edge Cases
- **DA (Doppler AG):** Doubles **printed** income (before CoO) of highest face value company. Face values are unique — ties impossible.
- **VM (Vintage Machinery):** Reduces total CoO by `min(total_coo, 10)` — cannot make CoO negative
- **S (Synergistic):** Use synergy count from Phase 21's `compute_synergy_bonuses()`, divide by 2 (rounded down)
- **PR (Prussian Railway):** +1 per company owned — straightforward count

### Cash Application
- Positive income: add to entity cash
- Negative income: subtract from entity cash (can go negative for corps)
- Players and FI cannot go bankrupt — if their balance would go negative, treat as error (CLOSING phase prevents this)
- Corporations can have negative cash after income — Phase 23 detects and handles bankruptcy

### Claude's Discretion
- Architecture choice: single function with branches vs separate functions per entity type (note: corps are fundamentally different from player/FI)
- Internal calculation order within the formula
- Helper function decomposition

</decisions>

<specifics>
## Specific Ideas

- "Sum of printed income first makes it architecturally easier to apply VM special ability"
- Corp income as entity method in corp.pyx enables state vector updates later
- Negative cash is acceptable within the automated INCOME phase — no intermediate state returned to driver

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 22-income-calculation*
*Context gathered: 2026-01-29*
