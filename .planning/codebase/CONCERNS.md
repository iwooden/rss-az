# Codebase Concerns

**Analysis Date:** 2026-01-20

## Performance & Optimization

### Bounds Checking Disabled Across All Cython Code

**Issue:** All `.pyx` files globally disable Cython's runtime bounds checking via compiler directives `boundscheck=False, wraparound=False`. This is a performance optimization for the hot path, but creates risk of silent memory corruption if array indices are computed incorrectly.

**Files affected:**
- `core/state.pyx`
- `core/data.pyx`
- `actions.pyx`
- `entities/*.pyx` (all entity files)
- `helpers/*.pyx` (all helper files)

**Impact:** Array index out-of-bounds errors will cause undefined behavior (segmentation faults, memory corruption) rather than clean Python exceptions. Makes debugging difficult and can silently corrupt game state.

**Current mitigation:** Extensive use of `noexcept nogil` declarations throughout codebase ensures compile-time verification. Manual bounds checking done in critical paths like `decode_action()` (`actions.pyx:162-163`).

**Improvement path:**
1. Document the index computation contracts precisely (why indices are safe)
2. Add assertions in debug builds to verify indices stay within bounds
3. Consider re-enabling bounds checking in non-critical entity getters/setters
4. Add comprehensive test coverage for edge cases (2-6 player counts)

---

## Memory Management

### No Manual Memory Cleanup in Entity Handles

**Issue:** Entity classes (`Player`, `Company`, `Corp`, `Turn`, `Deck`, etc. in `entities/` and `helpers/`) cache offset calculations in `__cinit__` and `initialize()`. If `initialize()` is never called, or called with incompatible state, cached offsets become invalid.

**Files:**
- `entities/player.pyx:25-50` - caches offsets in `initialize()`
- `entities/company.pyx:40-71` - caches offsets with location scanning
- `entities/corp.pyx:31-60` - similar pattern
- `entities/turn.pyx:35-80` - similar pattern

**Impact:** Using an entity handle without calling `initialize()` will access memory at offset 0 (the first element of state vector), likely reading/writing wrong game state. Silent corruption.

**Current mitigation:** Documentation states "This must be called before using any other methods." Global singleton instances (e.g., `PLAYERS = [Player(i) for i in range(6)]` in `entities/player.pyx:189`) are initialized once at game start.

**Risk:** New code paths that create entities and forget to call `initialize()` will fail silently. No compile-time checking.

**Improvement path:**
1. Add runtime guard in entity methods to detect uninitialized state (e.g., check if `_base_offset == 0` and `_num_players == 0`)
2. Throw exception or abort if used uninitialized
3. Consider making `initialize()` a required argument to `__init__` via Python wrapper

---

## State Layout Complexity

### Complex Offset Computation With Manual State Layout

**Issue:** Game state is a single float32 array with manually computed offsets. Layout computation happens at game start (`compute_layout()`, `compute_turn_offsets()`, etc. in `core/state.pyx:42-322`). Any error in offset calculations causes systematic corruption of ALL game state.

**Files:**
- `core/state.pyx:42-189` - offset computation (147 lines)
- `VECTORS.md` - manually maintained documentation of layout

**Impact:** Off-by-one errors in offset computation affect all subsequent state access. The computed offsets are cached in `StateLayout` struct and used throughout codebase with no validation.

**Current mitigation:**
- Offsets are computed once at `__cinit__` and cached
- Layout documented in `VECTORS.md`
- The `compute_*` functions are marked `noexcept nogil`

**Risk:** Changes to state layout (new fields, reordering) require manual updates to:
1. `compute_layout()`
2. Sub-offset functions (`compute_turn_offsets()`, `compute_player_field_offsets()`)
3. `VECTORS.md` documentation
4. All entity `initialize()` methods
5. Action decoding logic in `actions.pyx`

Easy to miss steps and introduce subtle bugs.

**Improvement path:**
1. Add validation in `compute_layout()` that cross-checks computed sizes against expected totals
2. Add unit tests that verify offset computations produce expected state sizes for each player count
3. Consider auto-generating offsets from a data schema
4. Add test that recreates layout, checks consistency

---

## Error Handling & Validation

### Minimal Input Validation

**Issue:** Only one explicit validation in entire codebase:

```python
# core/state.pyx:335
if num_players < 2 or num_players > GameConstants.MAX_PLAYERS:
    raise ValueError(f"num_players must be 2-{GameConstants.MAX_PLAYERS}")
```

Everything else returns sentinel values (-1) or undefined behavior.

**Files affected:** All core modules lack assertions/validation:
- `core/data.pyx` - lookup functions return -1 on invalid input
- `actions.pyx` - action decoding returns partially-filled `ActionInfo` struct on invalid action index
- Entity methods assume valid corp_id/player_id/company_id

**Impact:**
- Silent failures in game logic
- Corrupted state propagates undetected
- Difficult to debug which function produced bad state

**Examples of risky patterns:**
- `core/data.pyx:225` - `get_market_index(int price)` returns -1 if price not in market
- `entities/turn.pyx:207` - `get_auction_company()` returns -1 if not found, but callers may not check
- `actions.pyx:162-163` - bounds check returns partially-initialized ActionInfo

**Improvement path:**
1. Add precondition checks at function entry points for critical functions
2. Decide: fail-fast (raise exceptions) vs. silent sentinel values; be consistent
3. Add debug-mode assertions that validate invariants
4. Add test cases for invalid inputs to each public function

---

## Testing & Verification

### Test Directory Empty / Tests Deleted

**Issue:** `tests/` directory exists but is empty. CLAUDE.md references `pytest tests/test_invest.py -v` but no test files exist. Recent git commits show "delete tests" (commit 315a129).

**Files:**
- `tests/` - completely empty
- `CLAUDE.md:18-19` - references test files that don't exist

**Impact:**
- No automated verification of game logic correctness
- No regression detection when refactoring
- Complex state machine (game phases) has no recorded test cases
- Layout computation changes could introduce silent bugs

**Risk areas without tests:**
- State layout computation (`core/state.pyx:42-322`)
- Phase transitions and turn logic
- Player/corp/company state mutations
- Action validation and decoding
- Edge cases: 2-player vs 6-player games

**Improvement path:**
1. Restore test suite or rebuild from scratch
2. Prioritize:
   - Layout computation verification
   - Action decoding round-trip tests
   - Phase transition state machine
   - Entity offset caching correctness
3. Aim for coverage of:
   - Each player count (2, 3, 4, 5, 6)
   - Boundary conditions (max values, -1 sentinel handling)
   - State consistency checks

---

## Incomplete Features

### Missing calculate_net_worth() Implementation

**Issue:** `entities/player.pyx:81-82` has TODO:

```python
# TODO: calculate_net_worth() - requires Corp entity for share prices
# Net worth = cash + sum(company face values) + sum(shares * share_price)
```

**Files:** `entities/player.pyx:73-82`

**Impact:** Net worth is only stored as a value (`set_net_worth()`), never calculated from current game state. If stored net worth becomes stale, it will diverge from actual net worth. Player state could be inconsistent.

**Current usage:** Net worth appears to be set manually during game phases (stored but never updated). No clear code path that recalculates it.

**Risk:** Reporting player scores based on stale net worth value. Tie-breaking logic might use wrong values.

**Improvement path:**
1. Implement `calculate_net_worth()` requiring read access to corp share prices and company ownership
2. Define when net worth should be recalculated (at phase end? turn end?)
3. Add test that verifies calculated net worth matches stored value at key points

---

## Complex Data Structures

### Dense One-Hot Encoding for State Representation

**Issue:** State layout uses extensive one-hot encoding for categorical fields:

**Files:** `core/state.pyx` and throughout

**Examples:**
- Phase: 11-element one-hot array (only one bit set)
- CoO level: 7-element one-hot
- Auction high bidder: `num_players`-element one-hot
- Auction company: 36-element one-hot
- Many others (see `VECTORS.md`)

**Impact:**
- Wasteful use of float32 state space (should use compact int)
- Hidden state stores compact versions anyway (`hidden_phase_offset`, `hidden_coo_level_offset`, etc.)
- Redundancy between visible and hidden representations creates synchronization risk

**Example of redundancy:**
- Visible state: 11-element phase one-hot (11 floats)
- Hidden state: phase as compact value (1 float)
- Both must be kept in sync, doubling maintenance burden

**Improvement path:**
1. Audit hidden state usage - is it fully utilized?
2. Consider whether visible state truly needs one-hot, or if NN can handle compact representation
3. Add invariant checks that hidden and visible representations match
4. Document the "why" for one-hot encoding (if there's a good reason)

---

## Integration Risks

### Entity Global Singleton Pattern Assumes Single Game Instance

**Issue:** Global PLAYERS list initialized at module import:

```python
# entities/player.pyx:189
PLAYERS = [Player(i) for i in range(6)]
```

Similar pattern for all entity types. Each entity caches state offsets in `initialize()`.

**Impact:** If running multiple concurrent game instances with different player counts, the cached offsets become wrong. The offsets are player-count-dependent (see `compute_layout(num_players)` in `core/state.pyx:42`).

**Current assumption:** Single game instance at a time. Works for AlphaZero self-play (sequential games), but problematic for:
- Parallel self-play (multiple games in threads)
- API server serving multiple game instances
- Testing multiple games simultaneously

**Risk:** Silent state corruption if two games with different player counts try to use the same PLAYERS entity simultaneously.

**Files affected:**
- `entities/player.pyx:189` - PLAYERS global
- `entities/*.pyx` - all entity modules have similar globals
- `core/state.pyx` - state layout cached per GameState instance (OK), but entities assume shared globals

**Improvement path:**
1. Document: "Not thread-safe. Designed for single sequential game instance per process"
2. Or: restructure to pass player_count through entity method calls (remove state-dependent caching)
3. Or: make entity pools thread-local
4. Add test that verifies multi-instance isolation

---

## Floating Point Precision

### Float32 Quantization for Integer State

**Issue:** Integer game values (cash, share counts, etc.) are stored as float32 after division by divisor constants, then retrieved by multiplying back:

**Files:**
- `entities/player.pyx:56-67` - cash stored as `cash / CASH_DIVISOR`
- `entities/player.pyx:122-128` - shares stored as `shares / SHARE_DIVISOR`
- Divisors: `CASH_DIVISOR=200.0`, `SHARE_DIVISOR=7.0`

**Impact:** Repeated round-trip quantization can accumulate rounding errors:

```cython
# Simplified example
state._data[offset] = <float>100 / 7.0  # ~14.285714...
retrieved = <int>round(state._data[offset] * 7.0)  # Back to 100, but could be off by 1
```

float32 has ~7 decimal digits precision. For cash values up to $1M, precision loss could be significant.

**Risk:**
- Large values: cash > 1M or share counts > 1000000
- Repeated transactions in a single game could accumulate drift
- Player wealth calculations could be off by small amounts

**Current impact:** Likely minimal for typical game scales, but undocumented assumption about max values.

**Improvement path:**
1. Document max cash/share values where float32 precision is safe
2. Add assertions in setters to detect values outside safe range
3. Consider using fixed-point (integer) representation or higher precision
4. Add test that verifies round-trip quantization for edge cases

---

## Technical Debt Summary

| Concern | Severity | Impact | Effort to Fix |
|---------|----------|--------|----------------|
| Bounds checking disabled | High | Silent corruption possible | Medium |
| No entity initialization guards | High | Use-before-init corruption | Medium |
| Empty test directory | High | No regression detection | High |
| Complex layout offsets | Medium | Maintenance burden | Medium |
| Missing net_worth calculation | Medium | Stale player scores | Low |
| Missing input validation | Medium | Silent failures | Medium |
| Thread-unsafe global entities | Low | Blocks parallel usage | High |
| Float32 precision risk | Low | Rounding errors at scale | Low |

---

*Concerns audit: 2026-01-20*
