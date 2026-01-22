# Phase 7: Core Implementation - Context

**Gathered:** 2026-01-21
**Status:** Ready for planning

<domain>
## Phase Boundary

GameDriver auto-applies forced actions iteratively. When exactly one legal action exists, apply it automatically and continue until 2+ choices are available or game ends. This ensures the neural network only sees states with real decisions. Includes optional history tracking for test observability.

</domain>

<decisions>
## Implementation Decisions

### Error signaling
- Custom exception `ForcedActionLoopError` for iteration limit (100 loops)
- Minimal info in exception: iteration count and message, not full state
- Separate custom exception `ZeroLegalActionsError` for zero legal actions case
- Exceptions defined in separate `src/exceptions.py` module

### History API
- Optional list parameter: `apply_action(state, action, history=None)`
- Pass `[]` to collect history; pass `None` for no overhead
- If list is non-empty, append to it (don't replace)
- Each tuple contains `(state.copy(), action)` — state before action was applied
- Include ALL actions: user's initial action + all auto-applied actions
- `state.copy()` creates independent numpy array snapshot

### Loop guard
- Hardcoded constant `MAX_FORCED_ITERATIONS = 100`
- Not configurable via parameter
- Error message states the problem factually: "Forced action loop exceeded 100 iterations"
- Always raises exception — no warning mode

### Edge case handling
- GAME_OVER reached during auto-apply: return final state normally (caller checks phase)
- Phase transitions during auto-apply: no special handling, normal game flow
- Called with 2+ legal actions: apply normally, may trigger auto-apply chain afterward
- Auto-apply check happens after applying user's action, not before

### Claude's Discretion
- Internal helper function structure (`_check_forced_action`, `_apply_single_action`)
- ForcedActionResult struct design details
- Performance optimizations for the loop

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-core-implementation*
*Context gathered: 2026-01-21*
