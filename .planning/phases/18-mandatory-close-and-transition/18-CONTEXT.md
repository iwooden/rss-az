# Phase 18: Mandatory Close and Transition - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Auto-close player-owned private companies at phase end to prevent player bankruptcy in INCOME phase. After all offers processed and mandatory closes complete, transition to INCOME (currently stubbed as INVEST). Mandatory close applies only to player privates, not corporation subsidiaries.

</domain>

<decisions>
## Implementation Decisions

### Income Calculation
- Player income = sum of adjusted income from private companies only
- Adjusted income = base income - Cost of Ownership
- Check: player income + player cash < 0 triggers mandatory close
- Add `get_income()` method to `entities/player.pyx` for this calculation
- CoO is fixed at phase start; no re-evaluation during mandatory close loop

### Mandatory Close Logic
- Iterate players by player ID order
- For each player with income + cash < 0:
  - Close cheapest (lowest face value) negative-income private company
  - Recheck income + cash after each close
  - Stop when income + cash >= 0
- Junkyard Scrappers bonus applies to mandatory closes (2x printed income)
- Player can end up with zero companies (no minimum retention rule for players)

### Edge Cases
- Impossible for player to still be negative after closing ALL negative-income privates
  - By definition, removing all negative income sources results in non-negative total income
- Face values are unique, so no tie-breaking needed (use existing face value ascending logic)

### Transition Behavior
- Mandatory close happens at phase end (after all offers processed)
- Reuse existing `_transition_to_income()` function from Phase 17
- Keep temporary transition: CLOSING -> INVEST (INCOME phase implemented later)
- Immediate transition after mandatory close loop completes

### Claude's Discretion
- Whether mandatory close is integrated into offer handler or separate handler
- Any state cleanup before transition (likely minimal/none based on "immediate" decision)
- Implementation approach for get_income() method

</decisions>

<specifics>
## Specific Ideas

- "This bankruptcy/auto-close operates in established offer order with re-validation" — similar pattern to ACQUISITION phase
- get_income() method on Player entity mirrors how Corp income will work (preparation for INCOME phase)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-mandatory-close-and-transition*
*Context gathered: 2026-01-27*
