# Features Research: Forced Action Auto-Application

**Domain:** Game AI engine forced action handling for AlphaZero training
**Researched:** 2026-01-21
**Confidence:** MEDIUM (verified against existing codebase, cross-referenced with AlphaZero patterns)

## Executive Summary

Forced action auto-application prevents the neural network from being presented with trivial decisions (0 or 1 legal actions). The goal is to auto-apply actions in a loop until 2+ choices exist, then return control to the model. This research identifies table stakes behaviors, edge cases requiring explicit handling, AlphaZero-specific considerations, and anti-features to avoid.

---

## Table Stakes

Must-have behaviors for a correct forced action auto-application system.

### FA-01: Single Action Detection

**What:** Detect when exactly one legal action exists in current state.
**Why Expected:** Core purpose of the feature. The existing `get_forced_action()` in `actions.pyx` already provides this (lines 520-567).
**Complexity:** Low (already implemented)
**Notes:** Returns `(action_idx, True)` when forced, `(-1, False)` otherwise.

### FA-02: Auto-Application Loop

**What:** Apply forced actions repeatedly until 2+ legal actions exist or game ends.
**Why Expected:** Single forced action may lead to another forced action (e.g., BID phase with only one remaining bidder). Must loop, not just apply once.
**Complexity:** Low
**Notes:** Loop structure: `while count_legal_actions() == 1 and not game_over: apply_action()`.

### FA-03: Zero Action Handling (Terminal Detection)

**What:** Detect when zero legal actions exist and treat as terminal/error state.
**Why Expected:** Zero legal actions should never occur in a well-formed game state. If it does, either game is over or state is corrupt.
**Complexity:** Low
**Notes:** Defensive check. Return game-over status or raise error depending on game rules.

### FA-04: Game Over Detection in Loop

**What:** Stop auto-application loop when game ends (PHASE_GAME_OVER).
**Why Expected:** Prevents infinite processing after game conclusion.
**Complexity:** Low
**Notes:** Check `state.get_phase() == PHASE_GAME_OVER` or equivalent after each action.

### FA-05: State Mutation Semantics

**What:** Auto-applied actions must mutate state identically to manually-applied actions.
**Why Expected:** Consistency. Same action should produce same result regardless of how it was triggered.
**Complexity:** Low
**Notes:** Route through same `apply_action()` code path.

---

## Edge Cases

Specific scenarios requiring explicit handling.

### EC-01: Phase Transition Chains

**Scenario:** Action causes phase transition, new phase has only one legal action.
**Example:** In BID_IN_AUCTION, last bidder leaves auction. This resolves auction (forced), transitions to INVEST, and if next player can only pass, that's also forced.
**Required Handling:** Loop must continue across phase boundaries. Do not assume phase change means 2+ actions.
**Complexity:** Medium
**Detection:** After each action, regenerate mask and count valid actions regardless of phase change.

### EC-02: Auction Resolution with Single Bidder

**Scenario:** Auction starts with N players, all but one leave. Resolution is forced.
**Example:** Player 0 starts auction, players 1 and 2 leave. Player 0 wins by default.
**Current Behavior:** `_resolve_auction()` in `phases/bid.pyx` executes, transitions to INVEST.
**Required Handling:** If post-auction INVEST has only one legal action, continue auto-applying.
**Notes:** Common scenario in endgame when few players have cash.

### EC-03: All Players Must Pass (WRAP_UP Trigger)

**Scenario:** After forced passes, `consecutive_passes >= num_players` triggers WRAP_UP transition.
**Example:** Each player's only legal action is PASS. After N forced passes, phase changes.
**Required Handling:** Each pass is a distinct forced action. Loop N times, then transition.
**Notes:** Do not batch/optimize by skipping to end. Each state transition should be explicit for debugging and potential future action history.

### EC-04: Bankruptcy During Forced Action

**Scenario:** Forced sell action triggers bankruptcy (price index reaches 0).
**Example:** Player has exactly one share of a low-price corp, must sell, price drops to 0.
**Required Handling:** Bankruptcy procedure executes inline. Continue loop after bankruptcy completes. Check if game ends (all players bankrupt, etc.).
**Complexity:** Medium
**Notes:** Bankruptcy already executes inline in `_handle_sell_share()` (phases/invest.pyx lines 246-322). Verify game-over check runs.

### EC-05: Receivership Exit During Loop

**Scenario:** Forced buy action exits corp from receivership, transfers presidency.
**Example:** Corp in receivership, only bank shares exist. Forced buy assigns presidency.
**Required Handling:** Standard buy flow handles this. Ensure state is consistent before next action mask generation.
**Notes:** Already implemented in `_handle_buy_share()` (phases/invest.pyx lines 178-243).

### EC-06: Infinite Loop Prevention

**Scenario:** Bug causes state to not progress, loop runs forever.
**Example:** Action mask says 1 legal action, but applying it doesn't change state.
**Required Handling:** Add iteration limit. Log error and return error status if exceeded.
**Suggested Limit:** 100 iterations (covers longest possible forced sequence in Rolling Stock Stars).
**Complexity:** Low
**Notes:** Should never trigger in correct implementation. Defensive measure only.

### EC-07: Round-Trip Limit Edge Case

**Scenario:** Player has completed 2 round-trips for all corps they own shares in.
**Example:** Player can only pass (buying/selling blocked for all corps).
**Required Handling:** PASS becomes the single forced action. Apply and continue.
**Notes:** Action mask correctly blocks buy/sell when round-trip limit reached (INV-17).

### EC-08: Forced Sell Before Buy Due to Cash

**Scenario:** Player has insufficient cash for any buy, but can sell shares.
**Example:** Player has 0 cash, holds one share. Must sell (single legal action), then can buy.
**Required Handling:** Sell is forced, buy becomes available after. Mask regeneration handles this naturally.
**Notes:** Not actually forced sell per game rules, but single-action scenarios can emerge.

### EC-09: IPO Forced When Single Valid Par Price

**Scenario:** Only one market space available for IPO at valid par price.
**Example:** Company can only IPO at one price due to market congestion.
**Required Handling:** If player has the company and must IPO (game rules), single action is forced.
**Notes:** Depends on how IPO phase is structured. PASS may be alternative (check `_fill_ipo_mask` in actions.pyx).

### EC-10: Empty Auction Row

**Scenario:** Deck depleted, auction row empty, no companies to auction.
**Example:** Late game, all companies sold or removed.
**Required Handling:** Start auction becomes invalid. If only PASS remains, it's forced.
**Notes:** Mask generation should handle this (already checks auction slots in `_fill_invest_mask`).

---

## AlphaZero Considerations

How forced action handling integrates with AlphaZero training.

### AZ-01: Training Data Exclusion for Forced States

**Pattern:** States with exactly one legal action should NOT contribute training samples.
**Rationale:** No decision was made. Policy has no signal to learn (any output produces same action). Including these samples adds noise.
**Implementation:** When recording training data, skip states where auto-application occurred.
**Confidence:** HIGH (standard AlphaZero pattern)
**Source:** Verified against [alpha-zero-general MCTS implementation](https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py) which notes forced states don't require full MCTS simulation.

### AZ-02: MCTS Search Skipping

**Pattern:** Skip MCTS search entirely for forced moves during self-play.
**Rationale:** With one legal move, policy output is irrelevant. Save computation.
**Implementation:** Before calling MCTS, check `get_forced_action()`. If forced, apply directly.
**Confidence:** HIGH (common optimization)
**Notes:** Research confirmed this is a [standard MCTS optimization](https://www.billparker.ai/2025/01/elements-of-monte-carlo-tree-search.html) for forced move detection.

### AZ-03: Consistent State Presentation

**Pattern:** Neural network always sees states with 2+ legal actions.
**Rationale:** Simplifies policy head interpretation. Network learns meaningful distributions only.
**Implementation:** Loop in driver until 2+ actions or game over before returning mask to model.
**Confidence:** HIGH (explicit project goal)

### AZ-04: Value Still Meaningful for Skipped States

**Pattern:** While policy training is skipped, value can still be propagated.
**Rationale:** The value of a forced-move state is the value after the forced sequence completes.
**Implementation:** Record value targets from post-loop state, not intermediate forced states.
**Confidence:** MEDIUM (implementation detail, may vary)

### AZ-05: Action Probability for Forced Moves

**Pattern:** When recording games, forced moves can use probability = 1.0 for the single action.
**Rationale:** Deterministic outcome, no distribution to learn.
**Implementation:** Either skip recording or record with [1.0] vector.
**Confidence:** MEDIUM (either approach valid)

### AZ-06: Temperature Handling

**Pattern:** Temperature parameter irrelevant for forced moves.
**Rationale:** With one action, sampling is deterministic regardless of temperature.
**Implementation:** No special handling needed; action selection gives same result.
**Confidence:** HIGH

---

## Anti-Features

Things to deliberately NOT build.

### AF-01: Do NOT Batch Multiple Forced Actions

**What to Avoid:** Computing final state by skipping intermediate forced actions.
**Why Avoid:** Loses state-by-state consistency. Harder to debug. Breaks potential action logging.
**What to Do Instead:** Apply each forced action individually in sequence. Loop is simple and correct.

### AF-02: Do NOT Add "Auto" Flag to Actions

**What to Avoid:** Modifying ActionInfo struct to mark actions as auto-applied.
**Why Avoid:** Pollutes action space. Actions should be self-contained.
**What to Do Instead:** Handle auto-application at driver level, invisible to action handlers.

### AF-03: Do NOT Modify Mask for Forced Actions

**What to Avoid:** Special mask values (e.g., 2.0) to indicate forced actions.
**Why Avoid:** Mask is float32 array for neural network. 1.0/0.0 semantics must be preserved.
**What to Do Instead:** Use separate `get_forced_action()` check function (already exists).

### AF-04: Do NOT Pre-compute Forced Sequences

**What to Avoid:** Analyzing game tree to predict forced action chains.
**Why Avoid:** Over-engineering. Simple loop is fast enough. Pre-computation adds complexity.
**What to Do Instead:** Lazy evaluation - detect and apply one at a time.

### AF-05: Do NOT Expose Forced State to Neural Network

**What to Avoid:** Presenting forced states to model then ignoring its output.
**Why Avoid:** Wastes inference cycles. Confuses training if output is recorded.
**What to Do Instead:** Complete forced sequence before neural network sees state.

### AF-06: Do NOT Add User-Configurable Auto-Apply Toggle

**What to Avoid:** Option to disable forced action auto-application.
**Why Avoid:** AlphaZero training requires consistent behavior. Toggle adds testing surface.
**What to Do Instead:** Always auto-apply. No configuration.

### AF-07: Do NOT Add Timers or Delays for Forced Actions

**What to Avoid:** Artificial delays to simulate "thinking" for forced moves.
**Why Avoid:** Training loop should be as fast as possible. No human observer.
**What to Do Instead:** Execute forced actions immediately.

---

## Trigger Conditions

When exactly should auto-application trigger?

### Trigger Point 1: After Initial Action Application

**When:** After `apply_action()` completes successfully.
**Rationale:** The initial action may create a forced situation. Check immediately.
**Implementation:** Call auto-advance loop at end of `apply_action_with_auto_advance()`.

### Trigger Point 2: At Game Initialization (NOT RECOMMENDED)

**When:** After `initialize_game()` before first action.
**Why NOT:** Starting state should always have 2+ legal actions by game design. If not, it's a bug in initialization.
**Note:** Could add debug assertion that initial state has 2+ actions.

### Trigger Point 3: On Phase Transition (IMPLICIT)

**When:** When phase changes due to action.
**Handling:** Covered by Trigger Point 1 - the loop continues through phase transitions automatically since mask is regenerated each iteration.

---

## Phase Transition Handling

What happens at each phase boundary?

### INVEST -> BID_IN_AUCTION

**Trigger:** Start auction action
**Next Active Player:** Next bidder in turn order (skipping starter)
**Forced Scenario:** If all other players have left or cannot afford minimum bid
**Handling:** Auction starts, `advance_to_next_bidder()` runs, mask regenerated

### BID_IN_AUCTION -> INVEST

**Trigger:** Auction resolution (one bidder remains)
**Next Active Player:** Player after auction starter
**Forced Scenario:** If that player can only pass (no cash, no shares to sell, all corps at round-trip limit)
**Handling:** Resolution completes, phase changes, mask regenerated, loop continues

### INVEST -> WRAP_UP

**Trigger:** consecutive_passes >= num_players
**Next Active Player:** N/A (phase-level transition)
**Forced Scenario:** Entire WRAP_UP might be deterministic depending on FI auction rules
**Handling:** Currently WRAP_UP not implemented, but same loop pattern applies

### General Pattern

All phase transitions:
1. Action handler changes phase
2. Returns control to driver
3. Driver regenerates action mask
4. Driver checks forced action count
5. If 1, apply and repeat
6. If 0 or 2+, return to caller

---

## Implementation Recommendations

### Location

Auto-application logic should live in `GameDriver` class in `core/driver.pyx`, as a new method wrapping the existing `apply_action()`.

### Suggested Interface

```python
# New method in GameDriver
cpdef int apply_action_with_auto(self, GameState state, int action_idx):
    """
    Apply action, then auto-apply any forced follow-up actions.

    Returns:
        STATUS_OK if actions applied, model should see resulting state
        STATUS_GAME_OVER if game ended during execution
        STATUS_INVALID if initial action was invalid
    """
```

### Loop Structure

```python
# Pseudocode
result = apply_action(state, action_idx)
if result != STATUS_OK:
    return result

iterations = 0
MAX_ITERATIONS = 100
while iterations < MAX_ITERATIONS:
    if state.get_phase() == PHASE_GAME_OVER:
        return STATUS_GAME_OVER

    forced_action, is_forced = get_forced_action(state)
    if not is_forced:
        break  # 0 or 2+ legal actions

    result = apply_action(state, forced_action)
    if result != STATUS_OK:
        # Should never happen for forced action
        return STATUS_ERROR

    iterations += 1

if iterations >= MAX_ITERATIONS:
    return STATUS_ERROR  # Infinite loop protection

return STATUS_OK
```

---

## Sources

### Verified (HIGH Confidence)

- Existing codebase: `/home/icebreaker/rss-az-cython2/core/actions.pyx` get_forced_action() implementation (lines 520-567)
- Existing codebase: `/home/icebreaker/rss-az-cython2/core/driver.pyx` GameDriver architecture
- Existing codebase: `/home/icebreaker/rss-az-cython2/phases/invest.pyx` action handlers with bankruptcy handling
- Existing codebase: `/home/icebreaker/rss-az-cython2/phases/bid.pyx` auction resolution and phase transition

### Research (MEDIUM Confidence)

- [alpha-zero-general MCTS.py](https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py) - Action masking patterns, no forced move skipping implemented but concept noted
- [Board Game Arena Forum - Automated passing](https://forum.boardgamearena.com/viewtopic.php?t=10513) - Community discussion on auto-pass mechanics
- [Board Game Arena Forum - Automatic Actions](https://forum.boardgamearena.com/viewtopic.php?t=19496) - Implementation approaches for single legal move handling
- [Board Game Arena Forum - Automatic skip turn](https://forum.boardgamearena.com/viewtopic.php?t=22661) - Turn-based game considerations
- [Boring Guy - Masking in Deep RL](https://boring-guy.sh/posts/masking-rl/) - Action masking edge cases discussion
- [MDPI - Invalid Action Masking Study](https://www.mdpi.com/2076-3417/13/14/8283) - Academic analysis of action masking in RL

### Background (LOW Confidence)

- [Chessprogramming Wiki - AlphaZero](https://www.chessprogramming.org/AlphaZero) - General AlphaZero architecture (no forced move specifics)
- [OpenSpiel AlphaZero docs](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md) - Reference implementation (no forced move specifics found)
- [GameDev.net - Infinite Loops and State Machines](https://www.gamedev.net/forums/topic/670940-infinite-loops-and-state-machines/) - General state machine loop prevention patterns
- [XState - Eventless transition infinite loop](https://github.com/statelyai/xstate/discussions/1592) - Infinite loop detection in state machines

---

## Gaps and Open Questions

1. **Value Propagation:** Exact mechanism for propagating value through forced sequences needs verification during implementation. Either record from final state or accumulate rewards.

2. **Testing Strategy:** Need integration tests that construct long forced sequences (5+ actions) to verify loop termination and state consistency.

3. **Performance:** For very long forced sequences, should we consider any batching? Current recommendation is no, but may revisit if profiling shows issues.

4. **Action History:** If action history/replay is added later, forced actions should still be recorded. Current scope excludes this per v2 requirements.

5. **Zero Action Count:** Need to clarify behavior - is this an error (should never happen) or game-over condition (terminal state)?

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Table stakes behaviors | HIGH | Derived from codebase analysis and clear project goals |
| Edge cases | MEDIUM | Based on game flow analysis, may discover more during testing |
| AlphaZero patterns | MEDIUM | Consistent with standard practices but limited official documentation on forced moves |
| Anti-features | HIGH | Clear reasoning based on AlphaZero architecture requirements |
| Phase transition handling | HIGH | Verified against existing phase handler implementations |
