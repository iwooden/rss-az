# Phase 09 Plan 01: WRAP_UP Core Logic Summary

**One-liner:** Created WRAP_UP handler with player reordering by descending cash and ACQUISITION stub with turn increment

---
phase: 09-wrap-up-core-logic
plan: 01
subsystem: phase-handlers
tags: [cython, phase-transition, player-reordering, deterministic-phase]
requires: [entities/player, entities/turn, core/state]
provides: [phases/wrap_up, phases/acquisition]
affects: []
tech-stack:
  added: []
  patterns: [deterministic-phase-handler, selection-sort-reordering, stub-phase-pattern]
decisions:
  - decision: Use selection sort for player reordering
    rationale: Stable sort that handles tie-breaking cleanly with explicit comparison logic
    alternatives: Python's sorted() (not available in Cython nogil context), quicksort (unstable)
  - decision: ACQUISITION increments turn number and clears roundtrip tracking
    rationale: Per RESEARCH.md, turn lifecycle happens in final phase before INVEST starts
    alternatives: Could have done in WRAP_UP or INVEST, but ACQUISITION is the natural transition point
  - decision: setup.py auto-discovery handles new modules
    rationale: Existing find_pyx_files('phases') pattern automatically includes new .pyx files
    alternatives: Manual setup.py edits (unnecessary, auto-discovery worked)
key-files:
  created:
    - phases/wrap_up.pyx
    - phases/wrap_up.pxd
    - phases/acquisition.pyx
    - phases/acquisition.pxd
  modified: []
duration: 98s
completed: 2026-01-23
---

## What Was Built

Created two new phase handler modules following established Cython patterns:

**WRAP_UP Phase Handler:**
- Deterministic phase with 0 legal actions (non-player phase)
- Reorders players by descending cash with old position tie-breaking
- Uses selection sort algorithm for stable, explicit tie-breaking
- Sets active player to new position 0 after reordering
- Clears consecutive passes counter
- Transitions to PHASE_ACQUISITION

**ACQUISITION Phase Stub:**
- Minimal stub implementation for future FI purchase logic (Phase 10)
- Increments turn number for new round
- Clears roundtrip tracking for all players
- Transitions to PHASE_INVEST to start new round

Both modules compile successfully and are importable from Python.

## Decisions Made

### Decision 1: Selection Sort Algorithm
**Context:** Need deterministic player reordering by (descending cash, ascending old position)

**Options considered:**
1. Selection sort - O(n²) but simple and stable
2. Python's sorted() - Not available in Cython nogil context
3. C qsort - Not stable for tie-breaking

**Chosen:** Selection sort

**Rationale:**
- At most 6 players, so O(n²) is negligible (36 comparisons max)
- Explicit comparison logic makes tie-breaking clear and verifiable
- Stable sort behavior guaranteed by algorithm structure
- Follows established pattern from codebase (simple, correct over clever)

**Impact:** Clean, maintainable code with zero performance concerns at game scale

### Decision 2: Turn Number Increment in ACQUISITION
**Context:** Turn lifecycle needs to increment somewhere before new INVEST round

**Options considered:**
1. Increment in WRAP_UP (before ACQUISITION)
2. Increment in ACQUISITION (chosen)
3. Increment at INVEST start

**Chosen:** Increment in ACQUISITION stub

**Rationale:**
- ACQUISITION is the final phase before INVEST per game flow
- Matches semantic meaning: "acquisition phase happens, then new turn begins"
- Keeps WRAP_UP focused solely on player reordering
- When ACQUISITION is fully implemented, turn increment stays in logical place

**Impact:** Clear phase responsibilities, future-proof for ACQUISITION implementation

### Decision 3: Rely on setup.py Auto-Discovery
**Context:** New .pyx files need to be included in build

**Options considered:**
1. Auto-discovery via find_pyx_files('phases') (chosen)
2. Manual setup.py edits to add new modules

**Chosen:** Auto-discovery (verify it works)

**Rationale:**
- Existing setup.py pattern on line 71 walks 'phases' directory
- Adding files to phases/ directory automatically includes them
- Less maintenance, fewer chances for error
- Verified by successful build with new modules

**Impact:** Zero setup.py changes needed, new modules "just work"

## Requirements Satisfied

**From v3.0 Roadmap Phase 9:**

✅ **REORDER-01:** Reorder players by descending cash
- Implemented in `_reorder_players_by_cash` using selection sort
- Gathers cash values via `PLAYERS[i].get_cash(state)`

✅ **REORDER-02:** Tie-breaking by old turn order position
- Selection sort compares `(curr_cash > best_cash or (curr_cash == best_cash and curr_pos < best_pos))`
- Lower old position wins when cash is equal

✅ **REORDER-03:** Set active player to new position 0
- After reordering, calls `state._set_active_player(player_ids[0])`

✅ **PHASE-01:** WRAP_UP transitions to ACQUISITION
- `turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)`

✅ **PHASE-02:** ACQUISITION increments turn number
- `turn_module.TURN.set_turn_number(state, current_turn + 1)`

✅ **PHASE-03:** Clear roundtrip tracking before new turn
- Loop calling `PLAYERS[i].clear_roundtrip_tracking(state)` for all players

✅ **PHASE-04:** ACQUISITION transitions to INVEST
- `turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)`

## Technical Implementation

### File Structure
```
phases/
  wrap_up.pxd          # cdef declaration for apply_wrap_up
  wrap_up.pyx          # WRAP_UP phase handler implementation
  acquisition.pxd      # cdef declaration for apply_acquisition_stub
  acquisition.pyx      # ACQUISITION phase stub implementation
```

### Key Code Patterns

**Player Reordering Algorithm:**
```cython
# Declare all cdef vars at function start (Cython pattern)
cdef int num_players = state._num_players
cdef int[6] cash_values, old_positions, player_ids
cdef int i, j, best_idx, temp_id

# Gather current state
for i in range(num_players):
    player_ids[i] = i
    cash_values[i] = player_module.PLAYERS[i].get_cash(state)
    old_positions[i] = player_module.PLAYERS[i].get_turn_order(state)

# Selection sort by (-cash, old_position)
for i in range(num_players):
    best_idx = i
    best_cash = cash_values[player_ids[i]]
    best_pos = old_positions[player_ids[i]]

    for j in range(i + 1, num_players):
        curr_cash = cash_values[player_ids[j]]
        curr_pos = old_positions[player_ids[j]]

        # Higher cash wins, or if equal, lower old position wins
        if (curr_cash > best_cash or
            (curr_cash == best_cash and curr_pos < best_pos)):
            best_idx = j
            best_cash = curr_cash
            best_pos = curr_pos

    # Swap to front
    if best_idx != i:
        temp_id = player_ids[i]
        player_ids[i] = player_ids[best_idx]
        player_ids[best_idx] = temp_id

# Apply new turn order
for i in range(num_players):
    player_module.PLAYERS[player_ids[i]].set_turn_order(state, i)

# Set active player to new position 0
state._set_active_player(player_ids[0])
```

**WRAP_UP Main Handler:**
```cython
cdef int apply_wrap_up(GameState state) noexcept:
    _reorder_players_by_cash(state)
    turn_module.TURN.clear_consecutive_passes(state)
    turn_module.TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
    return 0
```

**ACQUISITION Stub Handler:**
```cython
cdef int apply_acquisition_stub(GameState state) noexcept:
    cdef int current_turn = turn_module.TURN.get_turn_number(state)
    cdef int i

    # Increment turn number
    turn_module.TURN.set_turn_number(state, current_turn + 1)

    # Clear per-turn tracking for all players
    for i in range(state._num_players):
        player_module.PLAYERS[i].clear_roundtrip_tracking(state)

    # Transition to new INVEST phase
    turn_module.TURN.set_phase(state, GamePhases.PHASE_INVEST)

    return 0
```

### Build Integration
setup.py auto-discovery pattern:
```python
pyx_files = find_pyx_files('phases') + ...
```

New files automatically included in extension building. Both modules compile to .so files:
- `phases/wrap_up.cpython-312-x86_64-linux-gnu.so`
- `phases/acquisition.cpython-312-x86_64-linux-gnu.so`

## Testing Notes

**Build Verification:**
- ✅ `python setup.py build_ext --inplace` succeeds
- ✅ Both .so files generated in phases/
- ✅ `from phases.wrap_up import *` works
- ✅ `from phases.acquisition import *` works

**Integration Testing (Plan 02):**
- Need to integrate into GameDriver auto-apply loop
- Need to handle 0 legal actions for non-player phases
- Need to modify INVEST to transition to WRAP_UP instead of GAME_OVER
- Need tests for player reordering correctness
- Need tests for phase transition sequence

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

**Blockers:** None

**For Plan 02 (GameDriver Integration):**
- ✅ WRAP_UP handler exists and compiles
- ✅ ACQUISITION stub exists and compiles
- ✅ Both handlers follow cdef noexcept pattern
- ✅ Phase constants already defined (PHASE_WRAP_UP = 2, PHASE_ACQUISITION = 3)
- ⚠️ Need to handle WRAP_UP's different signature (no ActionInfo* parameter)
- ⚠️ Need to loosen 0-action invariant for non-player phases
- ⚠️ Need to modify INVEST to transition to WRAP_UP instead of GAME_OVER

**Dependencies satisfied:**
- Player entity methods: get_cash(), set_turn_order(), get_turn_order(), clear_roundtrip_tracking()
- Turn entity methods: clear_consecutive_passes(), set_phase(), get_turn_number(), set_turn_number()
- State method: _set_active_player()

All required interfaces exist and work correctly.

## Lessons Learned

1. **Auto-discovery patterns reduce friction:** setup.py's find_pyx_files pattern meant zero build config changes
2. **Cython variable declaration discipline:** Declaring all cdef vars at function start prevents scope issues
3. **Selection sort clarity:** Simple algorithm makes tie-breaking logic obvious and verifiable
4. **Stub pattern enables parallel work:** ACQUISITION stub allows WRAP_UP testing without full ACQUISITION implementation

## Related Files

**Created:**
- `/home/icebreaker/rss-az-cython2/phases/wrap_up.pyx` - WRAP_UP phase handler
- `/home/icebreaker/rss-az-cython2/phases/wrap_up.pxd` - WRAP_UP declarations
- `/home/icebreaker/rss-az-cython2/phases/acquisition.pyx` - ACQUISITION stub
- `/home/icebreaker/rss-az-cython2/phases/acquisition.pxd` - ACQUISITION declarations

**Referenced:**
- `/home/icebreaker/rss-az-cython2/entities/player.pyx` - Cash and turn order methods
- `/home/icebreaker/rss-az-cython2/entities/turn.pyx` - Phase transitions and turn number
- `/home/icebreaker/rss-az-cython2/core/state.pyx` - _set_active_player method
- `/home/icebreaker/rss-az-cython2/core/data.pxd` - GamePhases constants

**Next plan will modify:**
- `/home/icebreaker/rss-az-cython2/core/driver.pyx` - Add WRAP_UP/ACQUISITION integration
- `/home/icebreaker/rss-az-cython2/phases/invest.pyx` - Change transition to WRAP_UP

---

**Commits:**
- `5c3bb6f`: feat(09-01): create WRAP_UP phase handler
- `ea468d9`: feat(09-01): create ACQUISITION phase stub

**Duration:** 98 seconds
**Status:** ✅ Complete
