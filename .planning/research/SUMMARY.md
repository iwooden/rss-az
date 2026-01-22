# Research Summary: v2.1 Forced Action Auto-Application

**Synthesized:** 2026-01-21
**Source Files:** FORCED_ACTION_STACK.md, FORCED_ACTION_FEATURES.md, ARCHITECTURE.md, PITFALLS.md

---

## Executive Summary

The forced action auto-application feature should implement an iterative while-loop inside `GameDriver.apply_action()` that automatically applies actions when exactly one legal action exists, continuing until 2+ choices are available or the game ends. The existing `get_forced_action()` function in `core/actions.pyx` already provides the detection mechanism; the primary work is integrating an auto-apply loop into the driver with proper termination guards and a new helper struct that distinguishes 0 from 2+ legal actions (which the current API cannot do).

The implementation is low-risk with HIGH confidence across all research areas. The main technical challenges are: (1) preventing infinite loops via iteration limits, (2) distinguishing zero-action error states from multi-action choice states, and (3) updating tests that assert on intermediate states. Performance impact is negligible since early-exit counting is O(k) where k is the position of the second valid action.

---

## Key Findings

### Implementation Approach (from ARCHITECTURE.md)

**Recommended Pattern:** Iterative loop inside `apply_action()` using a new `ForcedActionResult` struct.

```cython
cdef struct ForcedActionResult:
    int action_idx   # -1 if not forced
    int legal_count  # 0, 1, or 2+ (stop counting at 2)
```

**Why iterative, not recursive:**
- Stack safety for long forced chains (auction resolution -> all passes -> wrap-up)
- Cython optimizes iterative loops better
- No risk of stack overflow

**File Modifications Required:**
| File | Change |
|------|--------|
| `core/driver.pyx` | Add `_check_forced_action()` helper, modify `apply_action()` to loop |
| `core/driver.pxd` | Declare `ForcedActionResult` struct, new cdef helpers |

**No changes needed to:** phase handlers, actions.pyx, entity accessors.

### Edge Cases (from FORCED_ACTION_FEATURES.md)

**Critical Edge Cases:**

1. **Phase Transition Chains (EC-01):** Action causes phase change, new phase has only one legal action. Loop must continue across phase boundaries.

2. **Auction Resolution with Single Bidder (EC-02):** All but one player leave auction. Resolution is forced, then INVEST may also have only one valid action.

3. **All Players Must Pass (EC-03):** Each player's only legal action is PASS. Loop N times, then transition to WRAP_UP. Do not batch.

4. **Bankruptcy During Forced Action (EC-04):** Forced sell triggers bankruptcy (price hits 0). Bankruptcy executes inline, then loop continues.

5. **Infinite Loop Prevention (EC-06):** Add iteration limit of 100. Should never trigger in correct implementation.

6. **Zero Legal Actions:** Should never occur outside GAME_OVER. Return STATUS_INVALID and log error if encountered.

### AlphaZero Integration (from FORCED_ACTION_FEATURES.md)

- **Skip training samples** for forced states (policy has no signal to learn)
- **Skip MCTS search** for forced moves during self-play
- **Neural network only sees states** with 2+ legal actions
- **Value propagation** from final state after forced sequence completes

### Critical Pitfalls (from PITFALLS.md)

**Top 5 Pitfalls to Avoid:**

| Pitfall | Risk | Prevention |
|---------|------|------------|
| 1. Infinite Loop | HIGH | Iteration limit (100), state hash cycle detection in debug |
| 2. Zero Actions = No Forced | HIGH | Explicit count check; distinguish 0 from 2+ |
| 3. State Corruption Mid-Loop | MEDIUM | Ensure phase handlers are atomic; no partial mutations |
| 4. Recursive Depth Explosion | HIGH | Iterative only; phase handlers never call `apply_action()` |
| 5. Test Brittleness | MEDIUM | Update tests to assert behavior, not intermediate state |

**Test Update Strategy:** Categorize existing tests into:
- **Category A:** Behavior tests (no changes needed)
- **Category B:** Intermediate state tests (may need auto-apply disabled or assertion updates)
- **Category C:** End-state tests (minor assertion updates)

### Stack Recommendations (from FORCED_ACTION_STACK.md)

- **No new dependencies required** - existing Cython/NumPy stack sufficient
- **Keep float32 mask format** - required for neural network compatibility
- **Use `noexcept` for internal helpers** - no GIL overhead
- **Early-exit loop is optimal** - O(k) where k is position of second valid action
- **Performance impact negligible** - counting loop not the bottleneck

---

## Implications for Roadmap

### Suggested Phase Structure

**Phase 1: Helper Infrastructure** (Low Risk)
- Add `ForcedActionResult` struct to `core/driver.pxd`
- Add `_check_forced_action()` cdef function
- Add `_apply_single_action()` cdef function (extract current `apply_action()` body)
- Build and unit test helpers in isolation

**Phase 2: Loop Integration** (Medium Risk)
- Modify `apply_action()` to use iterative auto-apply pattern
- Add iteration limit guard (100 iterations)
- Handle zero-action case explicitly
- Manual testing with various game scenarios

**Phase 3: Test Updates** (Medium Risk)
- Categorize all existing phase tests
- Update tests asserting intermediate state
- Add new tests for:
  - Forced action sequences (chain of single-action phases)
  - Long forced chains (10+ auto-applies)
  - Zero legal actions detection (if reachable)
  - Seed reproducibility with auto-apply

### Rationale for Phase Order

1. **Helpers first** - Can be tested independently without breaking existing behavior
2. **Loop integration second** - After helpers verified, integration is mechanical
3. **Tests last** - Existing tests may fail after Phase 2; batch test updates together

### Research Flags

| Phase | Needs Research? | Reason |
|-------|-----------------|--------|
| Phase 1 | NO | Standard pattern, well-documented in ARCHITECTURE.md |
| Phase 2 | NO | Clear implementation from research; existing codebase patterns |
| Phase 3 | NO | Test patterns documented in PITFALLS.md |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Based on existing codebase patterns and official Cython docs |
| Features | HIGH | Verified against existing `get_forced_action()` implementation |
| Architecture | HIGH | Direct codebase analysis; clear modification points |
| Pitfalls | HIGH | Based on proven game engine patterns and codebase structure |
| Overall | HIGH | Well-scoped feature; existing infrastructure handles most complexity |

### Gaps to Address

1. **Value Propagation Detail:** Exact mechanism for value targets through forced sequences may need refinement during AlphaZero integration (post-v2.1).

2. **WRAP_UP Phase:** Auto-apply behavior at WRAP_UP boundary depends on WRAP_UP implementation (not yet built). Current scope: treat WRAP_UP as terminal for auto-apply purposes.

3. **Performance Baseline:** Should measure games/minute before implementation to quantify any regression.

---

## Sources

### Primary (HIGH confidence)
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` lines 520-567 - existing `get_forced_action()` implementation
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - current `apply_action()` structure (93 lines)
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - phase handler patterns
- `/home/icebreaker/rss-az-cython2/phases/bid.pyx` - auction resolution patterns
- [Cython 3.x Documentation](https://cython.readthedocs.io/) - noexcept, nogil, memoryviews

### Secondary (MEDIUM confidence)
- [alpha-zero-general MCTS.py](https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py) - forced move skipping patterns
- [Game Programming Patterns: State Machine](https://gameprogrammingpatterns.com/state.html) - infinite loop concerns
- [Board Game Arena Forums](https://forum.boardgamearena.com/) - auto-skip turn discussions

---

*Research synthesis: 2026-01-21*
