# Project Research Summary

**Project:** Rolling Stock Stars Cython Engine - WRAP_UP Phase (v3.0)
**Domain:** High-performance board game engine for AlphaZero self-play training
**Researched:** 2026-01-22
**Confidence:** HIGH

## Executive Summary

The WRAP_UP phase is a deterministic end-of-turn transition phase that executes four sequential operations: (1) reorder players by descending cash with tie-breaking by old turn order, (2) redistribute turn order cards (conceptual, handled by reordering), (3) Foreign Investor purchases available companies at face value in ascending order, and (4) flip all unavailable companies to available. The phase requires no new dependencies and maps cleanly to existing Cython patterns, with all infrastructure already in place from v2.1 implementation.

The recommended approach is to implement WRAP_UP as a fully atomic, deterministic operation with zero player actions. This means the entire phase executes in a single function call when all players pass in INVEST phase, using C stdlib qsort for player reordering and sequential iteration for FI purchases. The phase handler follows the established `cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept` pattern, delegating all state manipulation to existing entity handles (Player, ForeignInvestor, Company, TurnState).

The primary risk is state corruption during the FI purchase loop if using a cached list while modifying availability. This is mitigated by using a while-loop that re-queries available companies each iteration — no snapshotting needed since we always operate on current state. Secondary risks include player reordering tie-breaking errors (prevented by stable sort with explicit tie-breaking logic) and phase transition timing issues (prevented by making WRAP_UP fully atomic).

## Key Findings

### Recommended Stack

The WRAP_UP phase requires **zero new dependencies**. All algorithms can be implemented using existing Cython patterns and NumPy utilities already in the codebase. The phase involves deterministic algorithms (player sorting, FI purchasing iteration) that map cleanly to the established `cdef noexcept` handler pattern.

**Core technologies (already present):**
- **Cython 3.0+**: Phase handler compilation and sorting algorithms — optimal for performance-critical sorting and iteration with `noexcept nogil` support
- **NumPy 2.0+**: State array operations via memory views — already used for efficient data access throughout engine
- **libc.stdlib**: C standard library qsort for player reordering — fast in-place sorting for small N (≤6 players), stable when properly implemented

**What NOT to add:**
- Python `sorted()` — GIL acquisition overhead; incompatible with `noexcept nogil`
- NumPy `argsort()` — Returns new array allocation; we need in-place reordering
- Custom sorting algorithms — libc.stdlib.qsort is optimal for small N

### Expected Features

The WRAP_UP phase is entirely deterministic with no player choices—all actions are forced by game rules. It has four sequential sub-features that execute atomically.

**Must have (table stakes):**
- **Player order recalculation** — Descending cash order with tie-breaking by old order (Medium complexity)
- **Foreign Investor purchases** — Ascending face value order, buy all affordable companies (High complexity, multi-step)
- **Unavailable companies become available** — Flip all `revealed_companies` flags to `auction_companies` (Low complexity)

**Implementation note:**
- Turn order card redistribution is conceptual (physical game mechanic) — already handled by player reordering state update, no separate implementation needed

**Defer (not applicable):**
- None. All WRAP_UP features are MVP. Phase is atomic and tightly-scoped, nothing can be deferred.

**Critical edge cases to handle:**
- FI cannot afford any companies — skip purchase loop, proceed to availability flip
- No available companies — skip purchase loop normally
- Tie-breaking in player order — use old turn order (lower old position wins tie)
- Deck exhausted during FI purchases — purchase completes, no new company revealed
- All players tied at 0 cash — old order preserved (stable sort)

**Anti-features (explicitly avoid):**
- Player choices during WRAP_UP — phase is fully deterministic
- FI purchases at price spans — always use face value exactly
- FI purchases unavailable companies — only purchase from auction row
- Revealed companies stay revealed across turns — all must become available
- FI purchases in descending order — rules specify ascending face value order

### Architecture Approach

The WRAP_UP phase integrates seamlessly with the existing Cython game engine architecture. It follows established patterns from INVEST and BID_IN_AUCTION phases: a stateless phase handler function (`apply_wrap_up_action`) dispatched by GameDriver, with entity handles managing all state mutations. The phase is fully deterministic, meaning it executes as a single atomic operation when triggered, requiring zero NN evaluations.

**Major components:**
1. **Phase Handler** (`phases/wrap_up.pyx`) — Stateless function following `cdef int apply_wrap_up_action(GameState state, ActionInfo* info) noexcept` pattern, delegates all state manipulation to entity handles
2. **GameDriver Dispatch** (`core/driver.pyx`) — Add WRAP_UP case to existing phase routing logic, auto-apply loop continues through deterministic phase
3. **Action Encoding** (skip or minimal) — WRAP_UP may expose only ACTION_PASS since phase is deterministic, or execute entirely within INVEST handler
4. **Entity Integration** (no changes needed) — ForeignInvestor, Company, Player, and TurnState entities already provide complete interface for WRAP_UP operations
5. **Phase Transitions** — INVEST → WRAP_UP (trigger: consecutive_passes >= num_players), WRAP_UP → ACQUISITION (if FI owns companies) or INVEST (new turn if FI owns nothing)

**Key architectural decision:**
Implement WRAP_UP as **fully atomic, deterministic operation** rather than action-based phase. This means:
- Legal actions: Only ACTION_PASS (or none if executed within INVEST handler)
- Auto-apply: Entire WRAP_UP phase executes in single function call
- NN never sees WRAP_UP state — completely transparent to training loop
- Simpler implementation, faster training (no NN evaluation overhead)

### Critical Pitfalls

Research identified 11 domain-specific pitfalls. Top 5 critical risks:

1. **Player Reordering Tie-Breaking Corruption** — Incorrect tie-breaking implementation creates non-deterministic behavior. Use stable sort with explicit tie-breaking: sort by (-cash, old_turn_order) in single pass. Capture old state before mutations, apply new positions in separate phase. Test with equal cash scenarios and seed reproducibility.

2. **Foreign Investor Purchase Loop State Corruption** — State modifications during iteration (marking companies unavailable, drawing new cards) can corrupt subsequent queries if using a cached list. Use a while-loop that re-queries available companies each iteration: find cheapest affordable, execute atomic purchase, loop until none affordable. No snapshotting needed.

3. **Company Availability State Confusion** — Losing track of which companies should flip from unavailable to available. Need clear state machine: either add explicit `unavailable_companies` flag or document invariant (revealed AND !for_auction AND !owned => unavailable). Test before/after WRAP_UP to verify correct flipping.

4. **Phase Transition Timing with Auto-Apply** — WRAP_UP phase ending but auto-apply loop not transitioning properly. Implement WRAP_UP as atomic operation within INVEST handler (when consecutive_passes >= num_players), execute entire phase, then transition to next phase. No intermediate legal action generation.

5. **Determinism Violation from Sorting Instability** — C qsort() not guaranteed stable, breaks seed reproducibility. Use Python's `sorted()` which guarantees stable sort, or implement explicit tie-breaking in comparison function. Add seed-based reproducibility tests to CI.

**Additional moderate risks:**
- Entity handle offset invalidation after reordering — update active_player to new position 0
- FI cash underflow — strict affordability checks with assertions (fi_cash >= face_value)
- Missing auction state cleanup — clear consecutive_passes and auction flags

## Implications for Roadmap

Based on research, WRAP_UP implementation should be structured as 2-3 focused phases:

### Phase 1: WRAP_UP Core Logic (Player Reordering + Phase Transitions)
**Rationale:** Foundation phase that establishes deterministic execution pattern. Player reordering is the simplest algorithmic piece and validates the atomic operation approach. Phase transition logic determines whether to go to ACQUISITION or start new turn, which is critical for integration.

**Delivers:**
- Player reordering algorithm (descending cash, tie-break by old order)
- Phase transition logic (INVEST → WRAP_UP → ACQUISITION or new turn)
- Atomic execution pattern (entire phase in one function call)
- Active player update after reordering
- Turn number increment and state reset for new turns

**Addresses features:**
- TS-01: Player order recalculation
- TS-02: Turn order card redistribution (automatic via reordering)

**Avoids pitfalls:**
- Pitfall 1: Player reordering tie-breaking corruption
- Pitfall 4: Phase transition timing with auto-apply
- Pitfall 5: Forced action interaction misunderstanding
- Pitfall 8: Determinism violation from sorting instability

**Integration points:**
- Modify `phases/invest.pyx`: trigger WRAP_UP when all players pass
- Create `phases/wrap_up.pyx`: implement reordering and phase transition
- Update `core/driver.pyx`: add WRAP_UP dispatch case

**Estimated complexity:** Medium (40-60 LOC for reordering, 20-30 LOC for transitions)

---

### Phase 2: FI Purchase Logic
**Rationale:** Most complex algorithmic piece with highest risk of state corruption if iteration pattern is wrong. Use while-loop with re-query each iteration — straightforward and safe. Building on Phase 1's foundation ensures phase transitions work before adding purchase complexity.

**Delivers:**
- FI company purchase loop (ascending face value order)
- Affordability checking and cash deduction
- Company ownership transfer to FI
- Deck drawing and company unavailability marking
- Purchase termination logic (out of cash or no affordable companies)

**Addresses features:**
- TS-03: Foreign Investor purchases
- EC-01: FI cannot afford any companies
- EC-02: No available companies
- EC-05: Deck exhausted during FI purchases

**Avoids pitfalls:**
- Pitfall 2: Foreign Investor purchase loop state corruption
- Pitfall 3: Company availability state confusion
- Pitfall 7: FI cash underflow
- Pitfall 9: Off-by-one in face value ordering

**Dependencies:**
- Requires entity methods: FI.get_cash(), FI.add_cash(), Company.transfer_to_fi(), Company.is_for_auction(), Company.get_face_value()
- All methods already exist (verified in entities/)

**Estimated complexity:** High (80-120 LOC with careful iteration pattern)

---

### Phase 3: Company Availability State Transition
**Rationale:** Simplest operation but depends on understanding company state machine from Phase 2. Flips all unavailable companies to available at end of WRAP_UP. Can be implemented quickly once FI purchase logic establishes clear availability semantics.

**Delivers:**
- Batch flip of unavailable → available companies
- State invariant validation (no orphaned companies)
- Integration with FI purchase logic (newly drawn companies handled correctly)

**Addresses features:**
- TS-04: Unavailable companies become available
- AF-04: Revealed companies stay revealed across turns (anti-feature to avoid)

**Avoids pitfalls:**
- Pitfall 3: Company availability state confusion
- Pitfall 10: Forgetting to clear auction state

**Dependencies:**
- Requires clear definition of "unavailable" state from Phase 2
- Entity methods: Company.is_revealed(), Company.move_to_auction()

**Estimated complexity:** Low (20-30 LOC, simple iteration)

---

### Phase Ordering Rationale

**Why this sequence:**
1. **Phase 1 (Core + Transitions)** establishes atomic execution pattern and validates phase transition logic without complex state mutations. This de-risks the architectural decision (atomic vs. action-based).
2. **Phase 2 (FI Purchases)** is the highest-risk component with most state mutations. Implementing after Phase 1 ensures phase boundaries work correctly before adding complexity.
3. **Phase 3 (Availability Flip)** is trivial but semantically depends on Phase 2's state machine. Implementing last ensures clear understanding of company lifecycle.

**Alternative considered:**
- Combine Phase 2 and Phase 3 into single implementation phase
- **Rejected:** Combining risks rushing through state machine design, which is the most error-prone aspect per pitfall research

**Dependencies discovered:**
- All entity interfaces already complete (no new methods needed)
- State array layout sufficient (existing fields support all operations)
- No new dependencies or libraries required
- INVEST phase already has stub for WRAP_UP transition (just needs implementation)

**Grouping by architecture:**
- Phase 1: Phase handler pattern + transition logic
- Phase 2: Entity handle delegation + iteration pattern
- Phase 3: Batch state update pattern

### Research Flags

**Phases needing NO additional research (standard patterns):**
- **Phase 1 (Core + Transitions):** Player sorting is well-understood algorithm, phase transition logic follows existing INVEST→BID_IN_AUCTION pattern
- **Phase 2 (FI Purchases):** Iteration pattern established, entity handle methods documented, no external dependencies
- **Phase 3 (Availability Flip):** Simple batch update, no unknowns

**All phases can proceed to planning without `/gsd:research-phase`.**

Research was comprehensive enough to identify all architectural decisions, pitfalls, and implementation patterns. The WRAP_UP phase maps directly to existing codebase patterns with zero unknowns.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | All algorithms implementable with existing Cython + NumPy. No new dependencies verified. |
| Features | **HIGH** | RULES.md provides explicit specification (lines 132-138). All table stakes features identified. |
| Architecture | **HIGH** | Entity interfaces verified complete in codebase. Phase handler pattern established from INVEST/BID_IN_AUCTION. |
| Pitfalls | **HIGH** | 11 pitfalls identified from codebase analysis and domain research. Prevention strategies specified. |

**Overall confidence:** **HIGH**

All research files have high-confidence sources (official RULES.md, existing codebase patterns, verified entity interfaces). No external APIs, sparse documentation, or ambiguous requirements.

### Gaps to Address

**Architectural decision to confirm during planning:**
- **WRAP_UP execution model:** Implement as atomic operation (recommended) vs. action-based phase with forced actions. Research strongly recommends atomic, but stakeholder preference should be confirmed.
  - **If atomic:** Execute entire WRAP_UP within INVEST handler when all players pass
  - **If action-based:** Create ACTION_WRAP_UP_PASS action, expose to auto-apply loop
  - **Recommendation:** Atomic model is simpler, faster (no NN evaluation), and matches physical game behavior

**Implementation detail to resolve in Phase 2:**
- **Company "unavailable" state representation:** Add explicit `unavailable_companies` flag to state array (cleaner) vs. infer from existing flags (revealed AND !for_auction AND !owned => unavailable)
  - **If explicit flag:** Requires state array layout change (1 bit per company = 36 floats or packed bits)
  - **If inferred:** No state changes, but requires clear invariant documentation and enforcement
  - **Recommendation:** Use inferred state if possible (state array already tight), add explicit flag only if invariant proves fragile during testing

**Testing strategy to validate:**
- **Seed-based reproducibility:** Add CI test that runs same game with same seed 100 times, verify identical outcomes (validates stable sort and determinism)
- **Invariant checks:** Add assertions for state consistency (FI cash >= 0, no orphaned companies, turn order valid)

## Sources

### Primary (HIGH confidence)
- **RULES.md** (lines 132-138) — Official game rules for Phase 2: Wrap-up, authoritative specification
- **core/driver.pyx** (lines 1-191) — GameDriver dispatch pattern, auto-apply loop implementation
- **phases/invest.pyx** (lines 1-392) — Phase handler pattern, INVEST→WRAP_UP transition stub
- **phases/bid.pyx** (lines 1-118) — Reference phase handler implementation
- **entities/fi.pyx** (lines 1-75) — ForeignInvestor entity complete interface (get_cash, add_cash, owns_company, set_owns_company)
- **entities/company.pyx** (lines 1-384) — Company entity transfer operations (transfer_to_fi, is_for_auction, get_face_value)
- **entities/turn.pyx** (lines 1-150+) — TurnState entity phase management (set_phase, increment_turn_number, clear_consecutive_passes)
- **entities/player.pyx** (lines 1-200) — Player turn_order implementation (get_turn_order, set_turn_order)
- **core/state.pyx** (lines 1-828) — State layout verification, player turn_order storage (one-hot encoding)
- **core/data.pxd** (lines 1-96) — GamePhases enum (PHASE_WRAP_UP = 2 already defined)

### Secondary (MEDIUM confidence)
- **.planning/PROJECT.md** — Architecture patterns, entity handle pattern documentation
- **.planning/research/FORCED_ACTION_FEATURES.md** — Auto-apply behavior specification (v2.1)
- **.planning/phases/03-invest-core-auction-flow/03-RESEARCH.md** — INVEST phase design patterns
- **.planning/phases/07-core-implementation/07-RESEARCH.md** — Auto-apply loop implementation details

### Research Coverage
- **STACK.md:** Analyzed existing stack, confirmed no new dependencies needed
- **FEATURES.md:** Mapped RULES.md to implementation features, identified table stakes and edge cases
- **ARCHITECTURE.md:** Verified entity interfaces complete, mapped integration points, identified phase transition flow
- **WRAP_UP_PITFALLS.md:** Identified 11 domain-specific pitfalls with prevention strategies and testing requirements

---
*Research completed: 2026-01-22*
*Ready for roadmap: yes*
