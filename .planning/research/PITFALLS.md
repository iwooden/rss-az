# Pitfalls Research: Forced Action Auto-Application

**Domain:** Iterative forced action auto-application in game engine
**Project:** Rolling Stock Stars Cython engine (v2.1 milestone)
**Researched:** 2026-01-21
**Focus:** Common mistakes when implementing auto-apply loop for single legal actions

---

## Critical Pitfalls

Mistakes that cause system hangs, infinite loops, or state corruption. Must be addressed before implementation.

---

### 1. Infinite Loop from Self-Reinforcing Forced Actions

**Risk:** Auto-application creates a cycle where applying action A produces a state where action B is the only option, and B produces a state where A is the only option, looping forever.

**Why it happens in this codebase:**
- Phase transitions (INVEST -> BID_IN_AUCTION -> INVEST) could create cycles
- Auction resolution returns to INVEST with next player - if that player has only one valid action, loop continues
- Bankruptcy cascades that reset state to configurations with forced actions
- State where every player only has "pass" available, triggering WRAP_UP repeatedly

**Warning Signs:**
- Application hangs during random game simulations
- CPU usage spikes to 100% on `apply_action()` call
- Tests timeout without assertion failure
- Benchmarks show 0 games/minute completion rate

**Prevention:**
1. **Iteration limit with emergency exit:**
   ```python
   MAX_AUTO_ADVANCES = 100  # Reasonable upper bound
   iterations = 0
   while is_forced_action(state) and iterations < MAX_AUTO_ADVANCES:
       apply_single_action(state, forced_action_idx)
       iterations += 1
   if iterations >= MAX_AUTO_ADVANCES:
       raise RuntimeError("Forced action loop limit exceeded")
   ```

2. **State cycle detection (hash-based):**
   - Track last N state hashes in auto-apply loop
   - If same state hash appears twice, we have a cycle
   - Cost: O(1) per iteration with hash table

3. **Debug logging in development builds:**
   - Log each auto-applied action with phase and action type
   - Makes cycles immediately visible in test output

**Affected Area:** `GameDriver.apply_action()` main loop implementation

---

### 2. Zero Legal Actions Treated as "No Forced Action"

**Risk:** When zero legal actions exist (should be impossible in non-terminal states), the auto-apply loop exits thinking there's no forced action, leaving the game in an unplayable state.

**Why it happens in this codebase:**
- Current `get_forced_action()` returns `(-1, False)` for both 0 and 2+ actions
- Mask generation bug could produce all-zero mask
- Phase handler transitions to unexpected phase with no valid actions
- Bankruptcy procedure leaves corp state that blocks all actions

**Warning Signs:**
- `get_valid_action_mask()` returns all zeros outside GAME_OVER phase
- Tests fail with "no valid actions" assertion
- Game state stuck without GAME_OVER flag set

**Prevention:**
1. **Explicit zero-action validation:**
   ```python
   count = count_valid_actions(mask)
   if count == 0 and phase != PHASE_GAME_OVER:
       raise RuntimeError(f"Zero legal actions in phase {phase}")
   elif count == 1:
       # Auto-apply
   else:
       # Return to caller for decision
   ```

2. **Invariant check after every action:**
   - Add to existing `assert_invariants()`: verify at least one valid action exists
   - Run invariant checks in test builds

3. **Separate helper functions:**
   - `_count_legal_actions(state)` returns count
   - `_find_single_legal_action(state)` returns index or -1
   - Different return values for 0 vs 2+ cases

**Affected Area:** Helper functions for forced action detection, validation logic in `apply_action()`

---

### 3. State Corruption During Mid-Loop Interruption

**Risk:** If an exception or error occurs mid-way through the auto-apply loop, the game state is left in a partially-applied state that violates invariants.

**Why it happens in this codebase:**
- Auto-apply loop applies action 1, then action 2 fails with invalid state
- State is now after action 1 but before expected state after action 2
- Subsequent operations see corrupted intermediate state
- Tests may pass locally but fail on different seeds

**Warning Signs:**
- Share counts don't sum correctly after certain action sequences
- Player cash goes negative unexpectedly
- Presidency flags inconsistent with share ownership
- Non-deterministic test failures ("flaky tests")

**Prevention:**
1. **Copy-on-advance pattern (expensive but safe):**
   - Clone state before entering auto-apply loop
   - Restore from clone if any error occurs
   - Trade-off: memory allocation overhead

2. **All-or-nothing validation (preferred):**
   - Validate each action can succeed BEFORE applying
   - Since actions are pre-validated via mask, this is already done
   - Ensure phase handlers never fail after mask validation passes

3. **Atomic phase handler design:**
   - Each phase handler must complete all mutations or none
   - Never exit early after partial state modification
   - Pattern from existing code: compute all values first, write all at end

**Affected Area:** All phase handlers in `phases/*.pyx`, error handling in `apply_action()`

---

### 4. Recursive Auto-Application Depth Explosion

**Risk:** Auto-apply calls itself recursively, or a phase handler triggers auto-apply again, leading to stack overflow.

**Why it happens in this codebase:**
- Phase handler for action A calls `apply_action()` directly for cleanup actions
- Nested auto-apply loops compound iteration counts
- Bankruptcy handler might try to auto-apply subsequent forced actions

**Warning Signs:**
- Stack overflow errors or segfaults
- Maximum recursion depth exceeded
- Unpredictable behavior on deep action chains

**Prevention:**
1. **Iterative-only design (no recursion):**
   - Use `while` loop, never recursive calls
   - Phase handlers never call `apply_action()` themselves
   - All cascading logic handled in single flat loop

2. **Depth tracking (if recursion unavoidable):**
   ```python
   cdef int _apply_action_impl(state, action_idx, int depth):
       if depth > MAX_DEPTH:
           raise RecursionError("Auto-apply depth exceeded")
       # ... apply and recurse with depth + 1
   ```

3. **Clear separation of concerns:**
   - `apply_action()` handles auto-apply loop
   - Phase handlers ONLY modify state for their one action
   - No phase handler calls back into driver

**Affected Area:** `GameDriver.apply_action()` structure, phase handler boundaries

---

## Moderate Pitfalls

Mistakes that cause test failures or implementation delays.

---

### 5. Test Brittleness from Unexpected State Advancement

**Risk:** Existing tests assert specific intermediate states that are now auto-skipped, causing mass test failures.

**Why it happens in this codebase:**
- Tests like `test_pass_increments_consecutive_passes` check state after one action
- With auto-apply, that "one action" might trigger 3 more auto-applied actions
- Test expects player 0 active, but auto-apply advanced to player 1
- Assertions on intermediate state (auction company, bid price) may be bypassed

**Specific tests at risk (from codebase analysis):**
- `test_pass_advances_active_player` - may advance further than expected
- `test_start_auction_sets_*` - auction might resolve immediately if all but one player pass
- `test_leave_auction_returns_ok` - auction might auto-resolve before assertion
- Any test checking `TURN.get_consecutive_passes()` mid-sequence

**Warning Signs:**
- Many tests fail with "unexpected active player" or "unexpected phase"
- Tests pass in isolation but fail in specific orders
- Assertions about intermediate state fail

**Prevention:**
1. **Test behavior, not intermediate state:**
   - Instead of: "after pass, consecutive_passes == 1"
   - Test: "after N players pass, phase is WRAP_UP"
   - Focus on final outcomes, not step-by-step state

2. **Create explicit "no auto-apply" test mode:**
   - Add flag to GameDriver: `auto_apply_enabled: bool = True`
   - Tests that need to check intermediate state disable auto-apply
   - Document which tests require this mode

3. **Update test assertions to be auto-apply aware:**
   - Use `assert phase in [INVEST, WRAP_UP]` instead of exact match
   - Check "player changed" instead of "player is 1"
   - Use delta assertions: "consecutive_passes increased"

4. **Categorize tests before implementation:**
   - **Category A:** Behavior tests (need no changes)
   - **Category B:** Intermediate state tests (need auto-apply disabled)
   - **Category C:** End-state tests (may need assertion updates)

**Affected Area:** All tests in `tests/phases/`, test fixtures in `conftest.py`

---

### 6. Active Player Tracking Confusion

**Risk:** Auto-apply changes active player multiple times in one `apply_action()` call, confusing tests and callers that expect a single player change.

**Why it happens in this codebase:**
- INVEST pass advances to next player
- If next player also has only "pass" valid, they auto-pass
- Original caller expected player 0 -> player 1, got player 0 -> player 2
- Return value doesn't indicate how many players were auto-advanced

**Warning Signs:**
- Tests fail with "expected player 1, got player 2"
- Turn order appears to skip players
- Assertions on "next player" logic fail

**Prevention:**
1. **Document return semantics clearly:**
   - `apply_action()` returns status of FINAL state, not each intermediate
   - Add method `get_last_applied_action_count()` for debugging

2. **Provide action history for debugging:**
   - Track list of auto-applied actions in debug builds
   - Available via `state.get_auto_applied_actions()` for tests

3. **Update test philosophy:**
   - Don't assert on which player is active
   - Assert on game outcomes and final state
   - Use property-based tests that work regardless of player advancement

**Affected Area:** Test assertions involving active player, `apply_action()` documentation

---

### 7. Performance Regression from Repeated Mask Generation

**Risk:** Auto-apply loop calls `get_valid_action_mask()` on every iteration, multiplying mask generation cost.

**Why it happens in this codebase:**
- Current `get_forced_action()` generates full mask to check action count
- 10 forced actions in a row = 10 full mask generations
- Mask generation is O(num_companies + num_corps) per call
- Self-play throughput drops significantly

**Warning Signs:**
- Benchmark shows 2x-10x slowdown after auto-apply implementation
- Profiler shows `get_valid_action_mask()` as top hotspot
- Games per minute drops below acceptable threshold

**Prevention:**
1. **Optimized forced action detection:**
   ```cython
   cdef tuple _count_and_find_single_action(GameState state):
       """Count valid actions; if exactly 1, return its index."""
       # Early exit as soon as count > 1 (no need to scan full mask)
       count = 0
       single_idx = -1
       for i in range(total_actions):
           if is_action_valid(state, i):  # Inline validity check
               count += 1
               if count == 1:
                   single_idx = i
               elif count > 1:
                   return (count, -1)  # Multiple actions, no forced
       return (count, single_idx)
   ```

2. **Early termination in mask generation:**
   - Add parameter: `stop_after_n: int = 0` (0 = generate full mask)
   - For forced action check, call with `stop_after_n = 2`
   - Returns immediately after finding 2 valid actions

3. **Cache mask within auto-apply loop:**
   - Generate mask once per iteration
   - Reuse for both count check and action application
   - Invalidate only after state mutation

**Affected Area:** `core/actions.pyx` `get_forced_action()`, new helper functions

---

### 8. Phase Transition Edge Cases

**Risk:** Auto-apply encounters unexpected phase during iteration, causing invalid action dispatch.

**Why it happens in this codebase:**
- BID_IN_AUCTION has single bidder -> auto-resolves to INVEST
- INVEST has all passes -> transitions to WRAP_UP
- WRAP_UP may have forced actions (not yet implemented)
- Auto-apply doesn't know how to handle WRAP_UP phase

**Warning Signs:**
- `STATUS_INVALID` returned from auto-apply loop
- "Invalid phase for action dispatch" errors
- Game appears stuck at phase boundary

**Prevention:**
1. **Phase-aware loop termination:**
   ```python
   TERMINAL_PHASES = {PHASE_GAME_OVER, PHASE_WRAP_UP}
   while count == 1 and phase not in TERMINAL_PHASES:
       apply_action(...)
       phase = state.get_phase()
   ```

2. **Explicit handling for each phase:**
   - Define which phases support auto-apply
   - Define which phases are "terminal" for auto-apply purposes
   - WRAP_UP is terminal until its handlers are implemented

3. **Graceful degradation:**
   - If phase not recognized, exit auto-apply loop
   - Return current state to caller
   - Log warning for debugging

**Affected Area:** `apply_action()` loop condition, phase constant definitions

---

## Minor Pitfalls

Issues that cause confusion or require small fixes.

---

### 9. Return Value Ambiguity

**Risk:** `apply_action()` returns STATUS_OK after auto-applying 5 actions, but caller doesn't know this.

**Prevention:**
- Document clearly: "STATUS_OK means final state is valid after all auto-applications"
- Consider returning count of applied actions for debugging
- Add optional callback for observing auto-applied actions

**Affected Area:** `apply_action()` documentation and return semantics

---

### 10. Seed Reproducibility with Auto-Apply

**Risk:** Auto-apply changes the sequence of random calls (if any), breaking seed reproducibility.

**Prevention:**
- Auto-apply should not introduce any new random calls
- Verify same seed produces same game with auto-apply enabled/disabled
- Add regression test for seed determinism

**Affected Area:** Tests verifying seed reproducibility

---

### 11. Debug Output Overwhelming

**Risk:** Logging every auto-applied action produces massive output during training.

**Prevention:**
- Use compile-time flag for verbose auto-apply logging
- Production builds: no logging in auto-apply loop
- Test builds: optional verbose mode

**Affected Area:** Logging infrastructure, build configuration

---

## Test Update Strategies

### Strategy 1: Behavior-Focused Assertions

**Before (brittle):**
```python
def test_pass_advances_active_player(game_state):
    initial_player = game_state.get_active_player()
    DRIVER.apply_action(game_state, layout['pass_invest'])
    new_player = game_state.get_active_player()
    assert new_player == (initial_player + 1) % 3  # Assumes exactly one advance
```

**After (resilient):**
```python
def test_pass_advances_active_player(game_state):
    initial_player = game_state.get_active_player()
    DRIVER.apply_action(game_state, layout['pass_invest'])
    new_player = game_state.get_active_player()
    # Player changed (may have advanced multiple times due to auto-apply)
    assert new_player != initial_player or game_state.get_phase() == GamePhases.PHASE_WRAP_UP
```

### Strategy 2: Disable Auto-Apply for Specific Tests

```python
@pytest.fixture
def game_state_no_auto():
    """State with auto-apply disabled for intermediate state testing."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    # If GameDriver supports it:
    # DRIVER.set_auto_apply(False)
    return state

def test_intermediate_state(game_state_no_auto):
    # Test can now check intermediate state
    pass
```

### Strategy 3: Delta Assertions

**Before:**
```python
assert TURN.get_consecutive_passes(game_state) == 1
```

**After:**
```python
passes_before = TURN.get_consecutive_passes(game_state)
DRIVER.apply_action(game_state, layout['pass_invest'])
passes_after = TURN.get_consecutive_passes(game_state)
assert passes_after > passes_before or game_state.get_phase() != GamePhases.PHASE_INVEST
```

### Strategy 4: End-State Testing

Focus tests on final outcomes rather than intermediate states:

```python
def test_all_players_pass_ends_invest(game_state):
    """Eventually transitions to WRAP_UP when all pass."""
    layout = get_action_layout(3)
    # Apply passes until phase changes (auto-apply handles the rest)
    for _ in range(10):  # Safety limit
        if game_state.get_phase() != GamePhases.PHASE_INVEST:
            break
        DRIVER.apply_action(game_state, layout['pass_invest'])
    assert game_state.get_phase() == GamePhases.PHASE_WRAP_UP
```

### Test Categorization Checklist

Before implementing auto-apply, categorize all tests:

| Test File | Test Count | Category A (behavior) | Category B (intermediate) | Category C (needs update) |
|-----------|------------|----------------------|--------------------------|---------------------------|
| test_invest.py | ~40 | Review | Review | Review |
| test_bid_in_auction.py | ~30 | Review | Review | Review |
| test_driver.py | ~15 | Review | Review | Review |

---

## Performance Considerations

### Baseline Metrics (before auto-apply)
- Games per minute: (measure current benchmark)
- Mask generation time: (profile)
- Actions per game average: (measure)

### Expected Impact
- Additional mask generations per `apply_action()` call: 0-20 (depends on forced action chains)
- Overhead per forced action: ~1 mask generation
- Mitigation: Early termination optimization can reduce to O(1) for non-forced cases

### Monitoring
- Add benchmark comparison: with vs without auto-apply
- Track average auto-applies per user action
- Alert if performance drops >20% from baseline

### Optimization Priority
1. First: Correct implementation with iteration limit
2. Then: Early termination for count > 1 detection
3. Optional: Mask caching within auto-apply loop

---

## Sources

### General Game Engine Patterns
- [Game Programming Patterns: State Machine](https://gameprogrammingpatterns.com/state.html) - State machine design, infinite loop concerns
- [Game Programming Patterns: Game Loop](https://gameprogrammingpatterns.com/game-loop.html) - Loop iteration safety patterns

### Board Game Frameworks
- [boardgame.io: Automatic Skip Turn Issue](https://github.com/boardgameio/boardgame.io/issues/859) - Turn skipping implementation patterns
- [Board Game Arena Forum: Automatic Skip](https://forum.boardgamearena.com/viewtopic.php?t=22661) - User preference for auto-skip

### Testing Patterns
- [xUnit Patterns: Fragile Test](http://xunitpatterns.com/Fragile%20Test.html) - Avoiding brittle tests
- [Preventing Brittle Tests](http://blog.wingman-sw.com/preventing-brittle-tests) - Behavior vs implementation testing

### Performance
- [GameDev.net: Game Loop Optimization](https://www.gamedev.net/forums/topic/692878-can-my-game-loop-be-optimized/) - Tight loop performance
- [Wayline: Fix Game Loop Performance](https://www.wayline.io/blog/fix-game-loop-performance) - Iteration limit patterns

### Iteration Safety
- [How to prevent infinite loop in C++](https://labex.io/tutorials/cpp-how-to-prevent-infinite-loop-in-c-436657) - Safety counter patterns
- [GameDev.net: Infinite Loops and State Machines](https://www.gamedev.net/forums/topic/670940-infinite-loops-and-state-machines/) - State machine loop prevention

### Project-Specific
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Existing `get_forced_action()` implementation
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - Current `apply_action()` structure
- `/home/icebreaker/rss-az-cython2/tests/phases/conftest.py` - Existing test patterns

---

*Research confidence: HIGH for pitfalls 1-4, 5-7 (verified against codebase structure and common patterns); MEDIUM for pitfalls 8-11 (extrapolated from general best practices)*
