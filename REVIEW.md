# Code Review (High-Impact Findings)

## 1) End-card flow ends the game one phase too early (rules mismatch)
`phases/end_card.pyx` currently flips the end card and then immediately checks `is_end_card_flipped`, which causes same-phase `GAME_OVER` (`apply_end_card`, lines ~95-101).

Per `RULES.md` and the game-end clarifications, the game should end when Phase 7 starts with the end card already flipped; the first flip should arm game end for the *next* END_CARD check, not terminate immediately.

This is a core game-flow correctness bug that can invalidate rollout outcomes and training targets near endgame.

:::task-stub{title="Fix END_CARD sequencing so first flip does not immediately end the game"}
In `phases/end_card.pyx::apply_end_card`, capture whether the end card was already flipped at phase entry (before any mutation).

Use that pre-check state for game-over condition #3, instead of checking after a potential `_flip_end_card(state)` call in the same phase.

Suggested structure:
1. Evaluate condition #1 (75 card in use) as-is.
2. Read `was_flipped = TURN.is_end_card_flipped(state)` once.
3. If no unowned companies and `not was_flipped`, call `_flip_end_card(state)` and continue to ISSUE_SHARES.
4. If `was_flipped`, transition to `PHASE_GAME_OVER`.

Update `tests/phases/test_end_card.py` so `test_all_companies_owned_flips_end_card` expects flip + continuation (not immediate game over), and add/adjust a follow-up check that game ends on the next END_CARD execution.
:::

## 2) Receivership “must issue” rule is violated when issuing would hit price 0
In `phases/issue.pyx::_process_issue_share`, if `new_index == 0`, the function calls `go_bankrupt()` and returns before decrementing unissued shares / incrementing issued+bank shares (lines ~129-145).

Rules require receivership corps to issue if possible even if it causes bankruptcy. Current behavior skips the actual issuance side effects on bankruptcy path, creating state incoherence between rules and share counts.

:::task-stub{title="Apply issue-share state transitions even on bankruptcy path"}
In `phases/issue.pyx::_process_issue_share`, refactor the non-Stock-Masters branch so the share issuance state changes are always applied when `unissued > 0`, including when the post-drop target is index 0.

Implementation guidance:
1. Compute `new_index` and proceeds as today.
2. Perform issuance bookkeeping (`unissued--`, `issued++`, `bank_shares++`) before final bankruptcy exit.
3. Preserve rule semantics for proceeds and bankruptcy:
   - For price-to-0 path, ensure corporation goes bankrupt after required state transitions.
   - Confirm whether proceeds at index 0 should be applied prior to bankruptcy according to your engine conventions; keep behavior consistent with Sell/Issue rules and existing bankruptcy cleanup.
4. Ensure market occupancy / price-index transitions remain coherent for the bankruptcy path.

Add tests in `tests/phases/test_issue.py` for a receivership corp with one unissued share and a forced drop to 0, asserting that issuance counters changed before/with bankruptcy handling.
:::

## 3) Overseas Trading special ability is dropped in receivership FI auto-buy path
In acquisition auto-processing for receivership (`phases/acquisition.pyx`, lines ~525-537 and `_execute_receivership_fi_buy`), all receivership FI buys are forced to high price.

But OS’s special ability is unconditional (“always pays face value to FI”), and should still apply in receivership. Current logic overcharges OS and changes acquisition feasibility/priorities.

:::task-stub{title="Honor Overseas Trading face-value FI pricing during receivership auto-buys"}
Update receivership FI auto-buy handling in `phases/acquisition.pyx` so corp-specific FI pricing is applied:

1. In `_present_current_offer`, when `is_in_receivership` and `is_fi_offer`, branch pricing by corp:
   - `CorpIndices.CORP_OS`: affordability and payment use `face_value`.
   - others: affordability and payment use `high_price`.
2. Update `_execute_receivership_fi_buy` (or split into two helpers) to support both pricing modes cleanly.
3. Keep the existing ordering/priority behavior unchanged; only pricing/affordability should differ for OS.

Add a targeted test in `tests/phases/test_acquisition.py` where OS is in receivership and can afford face but not high; assert that it auto-buys and pays face value.
Also adjust existing receivership tests that currently encode “high only” for all corps.
:::

## 4) Action-mask generation is duplicated in forced-action loop (performance hotspot)
`core/driver.pyx` computes full legal mask in `_check_forced_action()` and then recomputes it in `_apply_single_action()` for each forced step (lines ~163 and ~221).

For MCTS-heavy workloads, this repeats expensive mask generation at every deterministic transition and can significantly increase CPU time.

:::task-stub{title="Eliminate redundant mask recomputation during forced-action auto-apply"}
Refactor `core/driver.pyx` to avoid generating legal-action masks twice per forced step.

One approach:
1. Extend `_check_forced_action` to optionally return/reuse the mask (or at least the verified single action index and a generation token).
2. Add an internal `_apply_single_action_verified(...)` path that skips mask validation when caller has already validated legality from the current unchanged state.
3. Keep public safety semantics unchanged for external `apply_action` calls.

Validation:
- Preserve exact behavior for invalid actions and phase transitions.
- Add/adjust unit tests around forced single-action chains to ensure no functional regressions.
- Optionally add a micro-benchmark script (not committed) to compare per-move overhead before/after.
:::

## 5) Test suite currently codifies at least one incorrect rule outcome and misses critical edge-case coverage
Two high-impact coverage issues:
- `tests/phases/test_end_card.py` expects immediate game end on first end-card flip (incorrect per rules flow above).
- No explicit test for OS-in-receivership FI face-value exception; current receivership tests encode the opposite assumption.

These gaps make regressions likely in exactly the rule-sensitive areas that affect training run validity.

:::task-stub{title="Realign rule-sensitive tests with authoritative RULES.md endgame and OS receivership behavior"}
Update tests under `tests/phases/` to match authoritative rules and protect against future drift:

1. `test_end_card.py`:
   - Change first-flip expectation to “flip now, game continues”.
   - Add test verifying game ends when END_CARD is entered with pre-flipped card.
2. `test_acquisition.py`:
   - Add OS-in-receivership FI affordability case (face affordable, high not affordable) and assert auto-buy at face.
   - Keep non-OS receivership high-price behavior tests.
3. `test_issue.py`:
   - Add forced-issue-to-bankruptcy state-coherence checks for share counters.

Include state-coherence assertions (visible + hidden where relevant) so mutations are validated holistically, not just by phase field.
:::

---

## Additional Medium-Priority Findings

## 6) Dividend legality is enforced by mask, but handler lacks full defensive validation
`phases/dividends.pyx::apply_dividend_action` validates only `0 <= amount < MAX_DIVIDEND`, then unconditionally pays (`_pay_dividends`) without re-checking per-corp max dividend (`get_max_dividend`) or treasury sufficiency. Today this is mostly protected by the action mask, but it weakens phase atomicity/invariant safety if mask logic changes or direct wrappers are used.

Why this matters: a single mask regression could allow illegal dividend payments and silently create negative corp cash states during DIVIDENDS, with no explicit rejection in phase logic.

:::task-stub{title="Make dividend handler self-validating for rule-critical constraints"}
In `phases/dividends.pyx::apply_dividend_action`, add explicit checks before `_pay_dividends`:
1. `amount <= get_max_dividend(current_price_index)`
2. `corp_cash >= amount * issued_shares`
3. Keep rejecting non-dividend actions and out-of-range amounts.

Then add tests that call handler/wrapper paths directly with intentionally invalid dividend amounts and insufficient cash, asserting clean rejection (`invalid`) and unchanged state.
:::

## 7) Global C RNG (`srand`/`rand`) in deck setup is process-global and non-thread-safe
`entities/deck.pyx::setup` seeds and uses libc global RNG (`srand(seed)`, `rand()`). In parallel self-play contexts sharing a process (threads), this can cause cross-game RNG interference and non-reproducible deck outcomes.

Why this matters: reproducibility and statistical independence are critical in training data generation; hidden cross-instance coupling is hard to detect.

:::task-stub{title="Replace process-global RNG with per-state/per-instance RNG"}
Refactor `entities/deck.pyx` shuffle logic to use a local RNG stream (e.g., xorshift/PCG state stored in hidden state or per-GameState field) rather than libc global RNG.

Requirements:
1. Deterministic per game seed.
2. No shared global mutable RNG state.
3. Keep current deck composition rules unchanged.

Add tests verifying:
- Same seed => identical deck for same player count.
- Different seeds => variation.
- Two game instances initialized interleaved do not affect each other’s deck order.
:::

## 8) Forced-action loop repeatedly allocates Python/Numpy masks (extra GC/alloc pressure)
Beyond duplicate logical checks (issue #4), `core/driver.pyx` currently materializes `get_valid_action_mask(state)` multiple times per move path (`_check_forced_action`, `_apply_single_action`). This creates avoidable Python object churn on hot paths.

Why this matters: hundreds of MCTS nodes × many plies amplifies allocation overhead and can dominate wall time in Python/Cython boundaries.

:::task-stub{title="Introduce low-allocation legality checks for driver hot paths"}
Add an internal nogil-friendly legality/count path in `core/actions.pyx` that can:
1. Count legal actions and optionally return first/second legal indices.
2. Check legality of one action index without allocating a full numpy mask.

Use this in `core/driver.pyx` forced-action logic, keeping public `get_valid_action_mask` for training/policy consumers.
:::

## 9) Correction: broad tie-break concern is mostly invalid; keep only the $75 shared-price edge case
After re-checking `RULES.md` and implementation behavior, the broad tie-break concern should be narrowed:
- Most market prices are unique because corporations occupy unique market cards.
- However, **75● is a special shared state** (corp has no share-price card at cap), so multiple corporations can still have the same effective share price.

So the original “many implicit tie-breaks” framing was too broad. The only meaningful tie-risk to retain is deterministic handling when **2+ corps are simultaneously at 75●**.

:::task-stub{title="Replace generic tie-break work with focused 75-price tie tests"}
1. Update review/test plans to remove generic tie-break work for normal market prices.
2. Add targeted tests for `DIVIDENDS`/`ISSUE` ordering when multiple active corps are at 75●.
3. Document expected deterministic order for this specific shared-price case (e.g., corp_id ascending), so rollouts remain reproducible.
:::

## 10) State-coherence testing is strong but still misses transfer helper invariants at API boundaries
The engine heavily relies on entity transfer helpers (e.g., `Company.transfer_to_*`) to keep visible flags and hidden location mirrors coherent. Existing tests cover many phase outcomes, but there is limited direct invariant testing of these transfer APIs under chained transitions and edge reuse.

Why this matters: subtle desync between visible ownership and hidden O(1) mirrors can corrupt masks/offers and produce hard-to-debug downstream errors.

:::task-stub{title="Add direct transfer-API coherence tests (visible + hidden mirror)"}
Add focused tests (likely in `tests/test_state_layout.py` or a new `tests/test_company_transfers.py`) that:
1. Execute all transfer permutations (`deck/revealed/auction/player/corp/acq/fi/removed`).
2. Assert exactly one visible location flag is set (or none for deck where applicable).
3. Assert hidden location/owner mirror matches visible owner after each step.
4. Assert idempotency/safety on repeated transfer attempts.
:::
