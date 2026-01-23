# Phase 9: WRAP_UP Core Logic - Context

**Gathered:** 2026-01-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement deterministic WRAP_UP phase that reorders players by descending cash and handles phase transitions. Fix existing INVEST phase to properly transition to WRAP_UP per game rules (instead of looping back to INVEST). WRAP_UP then transitions to ACQUISITION (the actual next phase per rules).

</domain>

<decisions>
## Implementation Decisions

### History entry format
- History is `(state, action)` tuples — no schema changes needed
- WRAP_UP creates single tuple entry; state snapshot captures the result (new player order)
- Action value for WRAP_UP: Claude's discretion based on existing action encoding patterns

### Auto-apply integration
- INVEST transitions to WRAP_UP when last player passes (per game rules)
- Existing auto-apply infrastructure detects WRAP_UP phase and executes logic
- WRAP_UP is non-player phase with 0 legal actions — loosen invariant for this case
- After WRAP_UP completes, transition to ACQUISITION (actual next phase per rules)

### Phase transition cleanup
- Fix INVEST to transition to WRAP_UP (not loop back to INVEST)
- WRAP_UP transitions to ACQUISITION (even though ACQUISITION is stub/unimplemented)
- Use `set_phase` utility for tests that need specific phase setups
- Future phases won't require modifying previous phase transition logic

### Claude's Discretion
- Sentinel/constant value for WRAP_UP action encoding
- How to detect "0 actions allowed" non-player phases cleanly
- ACQUISITION stub implementation (minimal, just enough to not error)

</decisions>

<specifics>
## Specific Ideas

- "Keep things clean and separated" — INVEST handles INVEST, WRAP_UP handles WRAP_UP
- Leverage existing auto-apply loop rather than special-casing WRAP_UP execution
- Fix the hacky loop-back pattern so future phase implementations are additive

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-wrap-up-core-logic*
*Context gathered: 2026-01-23*
