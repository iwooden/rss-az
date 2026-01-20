# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Memory-optimized monolithic state array with entity handle accessors and phase-based action generation.

**Key Characteristics:**
- Single contiguous `float32` array holds all game state (visible to NN + hidden internal)
- No Python object overhead—all logic operates directly on memory via C pointers (Cython nogil)
- Entity handles (`Player`, `Corporation`, `TurnState`, etc.) provide stateless accessors to state array
- Action space computed dynamically based on phase and valid moves
- Hidden state section untouched by NN (not provided as input)
- Automatic normalization/denormalization using divisors (cash, shares, stars, income)

## Layers

**Core State Layer:**
- Purpose: Manage raw float32 array allocation and layout computation
- Location: `core/state.pyx`, `core/state.pxd`
- Contains: `GameState` class (array holder + layout metadata), state layout structs
- Depends on: `core.data` (constants, offset computation)
- Used by: All entity modules, phase handlers, action generators

**Data Constants Layer:**
- Purpose: Static game data and normalization constants
- Location: `core/data.pyx`, `core/data.pxd`
- Contains: Company face/low/high prices, star ratings, market prices, corp names, normalization divisors (CASH_DIVISOR=200.0, SHARE_DIVISOR=7.0, etc.)
- Depends on: NumPy for imports
- Used by: Core state layout, entities, helpers, action validation

**Entity Handle Layer:**
- Purpose: Provide clean getter/setter interface to state array sections
- Location: `entities/` directory (`player.pyx`, `corp.pyx`, `turn.pyx`, `company.pyx`, `fi.pyx`, `market.pyx`, `deck.pyx`)
- Contains: One class per entity type (e.g., `Player`, `Corporation`, `TurnState`)
- Depends on: `core.state` (for GameState access), `core.data` (for constants)
- Used by: Phase handlers, action generators, game loop

**Helper Functions Layer:**
- Purpose: Low-level C pointer accessors for performance-critical nogil code sections
- Location: `helpers/` directory (`player.pyx`, `corp.pyx`, `turn.pyx`, `market.pyx`, `company.pyx`)
- Contains: Inline nogil functions operating on raw float pointers (e.g., `get_player_cash()`, `set_corp_active()`)
- Depends on: `core.data` (constants and offset structs)
- Used by: Action validation, fast path computations during NN inference

**Action Space Layer:**
- Purpose: Translate NN output indices to game actions and validate legal moves
- Location: `actions.pyx`, `actions.pxd`
- Contains: `ActionLayout` (offset computation for each phase), `ActionInfo` (decoded action with params), action masking functions
- Depends on: `core.state`, `entities`, `helpers`, `core.data`
- Used by: Game loop for action execution

## Data Flow

**Game Initialization:**

1. Python creates `GameState(num_players=N)` in `core/state.pyx`
2. Constructor computes layout: phase offsets, player stride, corp stride, turn offsets
3. Allocates `float32` array of total size (e.g., 3072 for 3 players)
4. Entity handles (`PLAYERS[]`, `CORPS[]`, `TURN`, `MARKET`, etc.) are initialized once at module load
5. Each entity calls `initialize(state)` to cache absolute offsets for fast repeated access

**Action Generation & Validation:**

1. NN produces output of shape `(batch, num_actions)` where `num_actions = 186 + (num_players * 20)`
2. Softmax applied externally; NN output fed to `ActionLayout.decode(action_index)` in `actions.pyx`
3. Action masking function generates valid action mask based on:
   - Current phase (via `TurnState.get_phase()`)
   - Player state (cash, companies, shares) via entity getters
   - Corp state (active, cash, owned companies) via entity getters
   - Turn-specific state (auction status, dividend corp, etc.)
4. Only actions with mask=1 are legal; invalid actions forced to "pass"
5. Decoded `ActionInfo` routed to phase handler

**State Mutation:**

1. Phase handler (not yet in scope) reads game state via entity handles
2. Handler performs business logic: transfer cash, flip company ownership bits, update share counts
3. All mutations go through entity setters (e.g., `Player.add_cash(state, amount)`)
4. Setters denormalize and write directly to state array at computed offset
5. For dual-encoded fields (hidden + visible one-hot), both updated atomically (e.g., `TurnState.set_phase()`)

**NN Inference Ready:**

1. Visible section of state array extracted (first `visible_size` floats)
2. Player rotation applied (active player becomes index 0, others shifted)
3. Array passed directly to PyTorch without serialization
4. NN output decoded as described in "Action Generation" section

**State Management:**

- All state is mutable in-place within the single array
- Entity handles cache offsets to avoid recomputation
- No object allocations during game play (nogil code paths)
- Hidden state (deck order, compact phase, compact corp prices) updated internally but never exposed to NN
- Visible state always kept in sync with hidden state (dual encoding for critical fields)

## Key Abstractions

**GameState:**
- Purpose: Memory container for entire game state
- Examples: `core/state.pyx` lines 326-550
- Pattern: Cython cdef class with `float*` pointer to numpy array; layout structures computed at construction

**Entity Handles:**
- Purpose: Stateless accessors providing intuitive API to state sections
- Examples: `entities/player.pyx` (Player class), `entities/corp.pyx` (Corporation class)
- Pattern: Each handle caches absolute field offsets in `initialize(state)` method; getter/setter methods take `GameState` argument

**Action Decoding:**
- Purpose: Map NN output index to playable action
- Examples: `actions.pyx` lines 73-150 (layout computation)
- Pattern: Offset-based lookup; phase determines which section of action space is active

**Normalization/Denormalization:**
- Purpose: Convert between integer game values and float32 neural network representation
- Examples: `core/data.pyx` (CASH_DIVISOR, SHARE_DIVISOR, etc.), `entities/player.pyx` lines 56-62
- Pattern: Divide by divisor on read (get_), multiply on write (set_)

**Offset Computation:**
- Purpose: Precompute array indices for all fields at game start
- Examples: `core/state.pyx` lines 42-189 (state layout), `entities/player.pyx` lines 30-50 (field offsets)
- Pattern: Recursive struct-based layout; each layer knows its size and computes its children's offsets

## Entry Points

**Python Game Instantiation:**
- Location: `core/__init__.py`, `core/state.pyx`
- Triggers: `GameState(num_players)` constructor call
- Responsibilities: Allocate state array, compute all layout structures, return initialized state container

**Action Submission:**
- Location: `actions.pyx` (action masking and decoding functions)
- Triggers: NN produces action index during inference
- Responsibilities: Validate action legality via mask, decode to `ActionInfo` with parameters, check bounds

**NN Inference Preparation:**
- Location: Implicit in `core/state.pyx` (getter methods for visible section)
- Triggers: Game loop requests state for NN input
- Responsibilities: Extract visible state array section, apply player rotation, return as numpy array

## Error Handling

**Strategy:** Bounds checking and validation in Cython compile directives disabled for performance (boundscheck=False); guards moved to high-level game loop.

**Patterns:**

- **Invalid Action:** NN output mapped to nearest valid action via action mask (invalid actions become "pass")
- **Out-of-Range Access:** Prevented by compile-time array bounds in entity methods; no runtime checks in nogil code
- **Layout Mismatch:** Caught at `initialize()` time when entity caches are computed; fails loudly if state layout incompatible
- **State Corruption:** Prevented by single-writer pattern (only entity setters write to array); no concurrent access

## Cross-Cutting Concerns

**Logging:** Not used in core state/entity layers. Phase handlers and game loop handle logging via Python print (outside nogil).

**Validation:**
- Action legality: Checked via `generate_action_mask()` in `actions.pyx`
- State consistency: No validation inside nogil paths; assumed valid by construction
- Player count: Validated at `GameState.__cinit__()` (2 to `MAX_PLAYERS`)

**Normalization:**
- Applied automatically in entity getters/setters
- Divisors centralized in `core/data.pyx` (CASH_DIVISOR=200.0, SHARE_DIVISOR=7.0, STAR_DIVISOR=20.0, MAX_ROUNDTRIPS=2.0)
- Inverse operation (denorm) applied on write; forward operation (norm) on read

**Memory Management:**
- NumPy array owned by `GameState`, held via `float*` pointer
- Python GC controls array lifetime
- No manual deallocation needed; Cython handles cleanup

---

*Architecture analysis: 2026-01-20*
