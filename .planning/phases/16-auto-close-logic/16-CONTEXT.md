# Phase 16: Auto-Close Logic - Context

**Gathered:** 2026-01-26
**Status:** Ready for planning

<domain>
## Phase Boundary

FI and receivership corps automatically close unprofitable companies at CLOSING phase start. This is deterministic logic with no player input. After auto-close completes, flow continues to offer-based closing (Phase 17).

</domain>

<decisions>
## Implementation Decisions

### FI Auto-Close Rules
- FI closes companies with **negative adjusted income** (income - CoO < 0)
- FI does **NOT** close companies with zero adjusted income (only strictly negative)
- FI has **no retention rule** - closes ALL negative-income companies
- FI **can** end up with zero companies after auto-close
- Close order doesn't matter since all eligible companies close anyway

### Receivership Auto-Close Rules
- Receivership checks actual **CoO value** (not level) against thresholds:
  - Red companies: auto-close when CoO for red >= $4
  - Orange companies: auto-close when CoO for orange >= $7
  - Yellow/Green/Blue companies: **NEVER** auto-closed by receivership
- **Highest face value company is ALWAYS protected** regardless of color/threshold
- Face values are unique - ties impossible
- A receivership corp **cannot** end up with zero companies (game invariant)
- Vintage Machinery's CoO reduction ability **applies** when checking thresholds

### State Changes
- Closed companies are **removed entirely** from game (not tracked in discard)
- Update `companies_removed` vector (36 flags) when companies close
- Junkyard Scrappers receives **2x printed income** bonus on ALL closes (including auto-close)

### Transition
- After FI and receivership auto-close completes, **immediately** proceed to offer-based closing (Phase 17)
- No intermediate steps between auto-close and offer generation

### Claude's Discretion
- Implementation structure (single pass vs separate FI/receivership passes)
- Internal ordering of receivership corps if multiple exist
- How to structure the close operation helper

</decisions>

<specifics>
## Specific Ideas

- Follow existing Cython patterns from ACQUISITION phase for phase handler structure
- Use `companies_removed` vector from VECTORS.md for tracking closed companies
- Reference RULES.md Section "Phase 4: Closing" and "Receivership Automatic Actions" for source truth

</specifics>

<deferred>
## Deferred Ideas

- Offer ordering by ascending face value (Phase 17) - ascending order reduces round-trips to model, closes cheaper companies first
- Mandatory close interleaving with offers (Phase 17/18) - auto-close when player would go bankrupt, then resume offers

</deferred>

---

*Phase: 16-auto-close-logic*
*Context gathered: 2026-01-26*
