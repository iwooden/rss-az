# Research Summary: V2 Milestone - Action Dispatch & Phase Implementation

**Project:** Rolling Stock Stars Cython Engine
**Milestone:** V2 - INVEST/BID_IN_AUCTION Phase Implementation
**Synthesized:** 2026-01-20

---

## Executive Summary

This milestone adds action dispatch and phase logic to the existing high-performance Cython game engine. The recommended approach is to **extend existing patterns** rather than introduce new abstractions. The codebase already contains the foundational components (ActionLayout, ActionInfo structs, entity accessors, GamePhases enum) that the new phase implementations will build upon.

The primary technical approach is **enum-switch dispatch**: actions are decoded using the existing `decode_action()` function, then routed to phase-specific handlers based on the current game phase. All handlers must be `noexcept nogil` to maintain the 10,000+ games/minute benchmark. No new dependencies are required; the existing Cython/NumPy stack is sufficient.

The highest-risk areas are: (1) keeping legal move masks synchronized with actual action validity, (2) partial state updates leaving the game in an inconsistent state, and (3) complex cascading logic around presidency changes and bankruptcy. These risks are mitigated by establishing a single source of truth for validity checks, atomic state update patterns, and comprehensive test coverage for edge cases.

---

## Key Findings

### From STACK.md: Technology Recommendations

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| Dependencies | **None required** | Existing Cython 3.2.4 + NumPy 2.4.0 stack is sufficient |
| Dispatch pattern | **Enum-switch** | Compiles to C switch, O(1) dispatch, matches existing codebase |
| State machine | **Implicit in phase enum** | No external FSM library needed; state tracked in GameState array |
| Phase handlers | **cdef noexcept nogil** | All hot-path functions must release GIL |

**Do NOT add:** transitions, python-statemachine, or any Python-level dispatch libraries. These would break nogil and degrade performance.

### From FEATURES.md: Feature Priorities

**Table Stakes (must implement):**
- Pass action with consecutive pass tracking
- Buy/sell share with price movement
- Start auction action (triggers phase transition)
- Leave auction and raise bid actions
- Auction resolution (winner pays, company transferred)
- Active player rotation
- Phase transition logic

**Differentiators (valuable for training quality):**
- Round-trip limits (MAX_ROUNDTRIPS=2 per corp per turn)
- Change of presidency on share transfer
- Corporation bankruptcy on price reaching 0
- Receivership when all shares sold

**Anti-features (explicitly avoid):**
- Undo/redo stack, event logging, UI state
- Human-readable action names in hot path
- Defensive bounds checking in nogil code
- Dynamic allocation during game loop

### From ARCHITECTURE.md: Structure and Patterns

**New files to create:**
```
phases/
    __init__.pyx, __init__.pxd    # Package init
    invest.pyx, invest.pxd         # INVEST + BID_IN_AUCTION handlers

core/
    driver.pyx, driver.pxd         # GameDriver dispatch class
```

**Key patterns to follow:**
1. Phase handler signature: `cpdef void phase_apply_action(GameState state, int action_type, int slot, int corp_id, int amount)`
2. Entity access via global instances: `PLAYERS[i].get_cash(state)`, `TURN.set_phase(state, phase)`
3. Phase transitions via `TURN.set_phase()` with proper state cleanup
4. Driver dispatch: decode action, route by phase, call handler

**Anti-patterns to avoid:**
- Storing state in phase modules (breaks thread-safety)
- Direct array access `state._data[offset]` (use entity accessors)
- Phase logic influencing mask generation (creates circular dependency)
- Python-level loops in hot paths

### From PITFALLS.md: Critical Risks

**Top 5 pitfalls with prevention:**

| Pitfall | Severity | Prevention |
|---------|----------|------------|
| **Legal mask out of sync with action validity** | Critical | Single source of truth for validity checks; both mask and apply call same cdef function |
| **Partial state updates** | Critical | Compute all values first, then write all fields atomically at end |
| **Cascading bankruptcy corruption** | Critical | Staged procedure (close companies, then shares, then presidency, then net worth); breadth-first not depth-first |
| **Presidency change logic errors** | Critical | Separate `recalculate_presidency(state, corp_id)` function; test all tie-break scenarios |
| **Auction sub-phase state leakage** | Critical | Clear all auction state on both entry AND exit from sub-phase |

**Phase-specific warnings:**
- Game Driver: Mask/action synchronization
- Buy/Sell Share: Round-trip limits, presidency transfer
- Start Auction: Clear all auction state on entry
- BID_IN_AUCTION exits: Clear state on EVERY exit path
- Bankruptcy: Staged processing, watch for recursive triggers

---

## Implications for Roadmap

Based on combined research, the implementation should be structured as follows:

### Suggested Phase Structure

**Phase 1: Infrastructure Setup**
- Create `phases/` module structure with .pxd and .pyx files
- Create `core/driver.pyx` GameDriver class shell
- Establish shared validity check functions (single source of truth pattern)

**Rationale:** Foundation must exist before any action handlers. The driver and phase structure enable incremental testing.

**Delivers:** Compilable phase module, driver skeleton, shared utility functions

**Features:** None (infrastructure only)

**Pitfalls to avoid:** Entity handle initialization order

---

**Phase 2: INVEST Pass Action**
- Implement pass action handler
- Implement consecutive pass tracking
- Implement phase-end detection (all players passed)
- Implement active player rotation

**Rationale:** Simplest action type, validates entire dispatch pathway, enables testing of turn mechanics.

**Delivers:** Working pass action, turn rotation, phase end detection

**Features:** Pass action, consecutive pass tracking, active player rotation, phase end detection

**Pitfalls to avoid:** Missing active player updates, implicit phase transition assumptions

---

**Phase 3: Auction Flow**
- Implement start auction action (INVEST -> BID_IN_AUCTION transition)
- Implement leave auction action
- Implement raise bid action
- Implement auction resolution (winner determination, company transfer)
- Implement return to INVEST phase

**Rationale:** Completes the second phase type, tests sub-phase transitions, relatively contained logic.

**Delivers:** Full auction cycle from start to resolution

**Features:** Start auction, leave auction, raise bid, auction resolution, company transfer, phase transition back to INVEST

**Pitfalls to avoid:** Auction sub-phase state leakage (clear on all exit paths), auction starter tracking for turn resumption

---

**Phase 4: Share Trading**
- Implement buy share action
- Implement sell share action
- Implement price movement (up on buy, down on sell)
- Implement round-trip limit tracking and enforcement

**Rationale:** Complex state mutations with market price dependencies. Build on working pass/auction to validate integration.

**Delivers:** Complete share trading mechanics

**Features:** Buy share, sell share, price movement, round-trip limits

**Pitfalls to avoid:** Partial state updates, round-trip limit enforcement bugs, net worth update timing

---

**Phase 5: Presidency & Bankruptcy**
- Implement presidency change logic (on share transfer)
- Implement receivership detection
- Implement corporation bankruptcy (price reaches 0)
- Implement cascading state cleanup

**Rationale:** Most complex logic, deferred to last phase. Builds on working share trading.

**Delivers:** Complete corporation lifecycle management

**Features:** Change of presidency, receivership, corporation bankruptcy

**Pitfalls to avoid:** Presidency logic errors (tie-breaking, receivership edge cases), cascading bankruptcy state corruption

---

### Research Flags

| Phase | Needs `/gsd:research-phase`? | Notes |
|-------|------------------------------|-------|
| Phase 1: Infrastructure | No | Standard patterns, well-documented in codebase |
| Phase 2: Pass Action | No | Simple, follows existing patterns |
| Phase 3: Auction Flow | No | Game rules clear, architectural patterns established |
| Phase 4: Share Trading | Maybe | Market price movement logic may need clarification |
| Phase 5: Presidency & Bankruptcy | Yes | 18xx rules complex; presidency tie-breaking and bankruptcy cascade need detailed research |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | **HIGH** | No new dependencies; extends proven patterns |
| Features | **HIGH** | Table stakes derived from existing action mask code; game rules documented |
| Architecture | **HIGH** | Patterns match existing codebase; clear integration points |
| Pitfalls | **HIGH** for codebase-specific, **MEDIUM** for 18xx domain rules | Domain pitfalls based on general 18xx complexity |

### Gaps to Address During Planning

1. **Market price movement**: Exact algorithm for finding next available market space when spaces are occupied
2. **Presidency tie-breaking**: Confirm exact rules for tie-breaking by turn order
3. **Bankruptcy cascade**: Document exact order of operations when bankruptcy triggers additional bankruptcies
4. **FI auction fallback**: Rules when no players bid (FI gets company at face value) - deferred to v2+

---

## Sources

### Stack Research
- Cython documentation (official docs, HIGH confidence)
- Existing codebase patterns in `core/actions.pyx`, `entities/*.pyx` (HIGH confidence)
- Performance benchmarks from `setup.py benchmark` (validated)

### Feature Research
- Rolling Stock Stars How to Play (boardgameblitz.com)
- Daemon18xx GitHub - 18XX rules engine patterns
- 18XX with Ambie: Dumping Companies & Hostile Takeovers
- Simple Alpha Zero action dispatch patterns
- Game Programming Patterns: State
- Existing codebase: `core/actions.pyx`, `VECTORS.md`

### Architecture Research
- `/home/icebreaker/rss-az-cython2/core/state.pyx` (GameState implementation)
- `/home/icebreaker/rss-az-cython2/core/actions.pyx` (Action layout and mask generation)
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` (TurnState entity)
- `/home/icebreaker/rss-az-cython2/.planning/codebase/ARCHITECTURE.md`
- `/home/icebreaker/rss-az-cython2/.planning/codebase/CONVENTIONS.md`

### Pitfall Research
- Game Programming Patterns: State Machine Pattern
- Cython GIL and nogil documentation
- 1830 Rules Clarifications (18xx.net)
- PettingZoo Chess (AlphaZero legal move masking)
- State Machines for Game Dev (numberanalytics.com)
- Existing codebase: `core/actions.pyx`, `core/state.pyx`

---

*Research synthesis: 2026-01-20*
*Overall confidence: HIGH - Mature codebase with established patterns, clear game rules, well-documented pitfalls*
