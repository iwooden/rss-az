# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Single-Vector State Machine with Entity Accessor Layer

This is a high-performance game engine for the "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. The entire game state is stored as a single contiguous `float32` array that can be passed directly to PyTorch without serialization overhead. The architecture separates concerns into three layers:

1. **Core State Layer** - Raw float array data storage and layout computation
2. **Entity Accessor Layer** - Lightweight wrapper objects for game entities (Player, Corporation, Company, etc.) that provide clean getter/setter access to state fields
3. **Action Decoding Layer** - Action space management, validation masks, and action decoding

**Key Characteristics:**
- Single contiguous state vector enables zero-copy transfer to neural networks
- Cython implementation with `nogil` performance-critical sections for training speed
- Stateless entity accessors (all methods take GameState as parameter) for thread-safety
- Two-tier state layout: visible (for NN input) + hidden (for internal logic only)
- Strict separation between performance-critical code (Cython) and Python interfaces
- No per-game-loop allocations; all memory pre-allocated at GameState construction

## Layers

**Core State Layer:**
- Purpose: Manage the raw float32 state array, compute memory layout offsets, and provide low-level C-speed access
- Location: `core/state.pyx`, `core/state.pxd`
- Contains: `GameState` class (array container + layout metadata), `StateLayout` struct, `TurnStateOffsets` struct, `PlayerFieldOffsets` struct, `CorpFieldOffsets` struct, layout computation functions
- Depends on: NumPy for array allocation, `core/data.pyx` for game constants
- Used by: Entity accessor layer, actions module

**Game Constants & Static Data Layer:**
- Purpose: Static game data (company stats, synergies, price tables) and normalization constants
- Location: `core/data.pyx`, `core/data.pxd`
- Contains: Game constants enums (`GameConstants`, `GamePhases`, `CorpIndices`), company arrays (`COMPANY_FACE_VALUE`, `COMPANY_STARS`, `COMPANY_SYNERGY`), market prices table, normalization divisors (`CASH_DIVISOR=200.0`, `SHARE_DIVISOR=7.0`, `STAR_DIVISOR=20.0`, `INCOME_DIVISOR=10.0`, `MAX_ROUNDTRIPS=2.0`), accessor functions for company data
- Depends on: Nothing
- Used by: All other layers

**Entity Accessor Layer:**
- Purpose: Provide stateless, high-level getter/setter access to game entities while maintaining O(1) field access
- Location: `entities/` directory
- Contains: `Player`, `Corporation`, `Company`, `ForeignInvestor`, `Market`, `TurnState`, `Deck` classes, each with both low-level `cdef` nogil functions (for performance-critical code) and high-level `cpdef` Python-callable methods
- Depends on: Core state layer for field offset computation, `core/data.pyx` for normalization constants
- Used by: Game logic, action generation

**Action Decoding Layer:**
- Purpose: Decode neural network output indices into structured action information, generate valid action masks
- Location: `core/actions.pyx`, `core/actions.pxd` (note: not found in current exploration, may be inferred from VECTORS.md)
- Contains: `ActionLayout` struct, action decoding functions, mask generation
- Depends on: Core state layer for phase/entity state
- Used by: RL training loop to interpret NN outputs

## Data Flow

**Game Initialization:**

1. User creates `GameState(num_players=N)` in `core/state.pyx`
2. GameState computes `StateLayout` based on player count (size varies with N)
3. GameState allocates float32 NumPy array of exact required size (2993 to 3321 floats depending on players)
4. Entity singletons (`PLAYERS`, `CORPS`, `TURN`, `MARKET`, etc.) call `initialize(state)` to cache their field offsets
5. Game is ready for play

**During Game Loop:**

1. Game logic reads current phase/turn state via `TURN.get_phase(state)`, `TURN.get_coo_level(state)`
2. Logic queries player/corp/company state via entity getters: `PLAYERS[i].get_cash(state)`, `CORPS[j].is_active(state)`
3. RL agent reads visible state via `state.get_nn_input()` (rotated to active player first)
4. Agent produces action index via NN forward pass
5. Action index decoded via action module into `ActionInfo` struct (phase, action_type, slot, corp_id, amount)
6. Game logic applies the action, modifying state fields via entity setters
7. Repeat until game over

**State Vector Representation:**

The state array is organized as: `[VISIBLE STATE (for NN)] [HIDDEN STATE (internal only)]`

**Visible section** (size varies by num_players, e.g. 3020 for 3 players):
- Phase (11 one-hot): Indices 0-10, one value = 1.0 indicating current phase
- CoO Level (7 one-hot): Indices 11-17, cost of ownership level
- Players (N times, stride=71+N): Cash, net_worth, turn_order, is_auction_high_bidder, owned_companies (36 flags), owned_shares (8 normalized), is_president (8 flags), share_buys (8 normalized), share_sells (8 normalized)
- Foreign Investor (37 floats): Cash + owned_companies (36 flags)
- Company Locations (108 floats): 3 arrays of 36 each (companies_for_auction, companies_revealed, companies_removed)
- Company Adjusted Incomes (36 floats): Normalized, updated when CoO changes
- Market Availability (27 floats): Flags for available share price slots
- Corporations (8 times, stride=109): Active flag, cash, shares (unissued/issued/bank), income, stars, share_price, acquisition_proceeds, in_receivership flag, price_index (27 one-hot), owned_companies (36 flags), acquisition_companies (36 flags)
- Turn State (251+3N floats): Turn number, end_card_flipped, consecutive_passes, auction state (company, price, high_bidder, starter, passed_flags), dividend state (corp, impacts, remaining), issue state (corp, remaining), IPO state (company, remaining), acquisition state (active_corp, target_company, is_fi_offer), closing state (company)
- Static Company Data (1440 floats): 36 companies × 40 floats each (stars, low_price, face_value, high_price, synergies[36])

**Hidden section** (size=52, offset=visible_size):
- active_player (1): Current player index
- num_players (1): Player count
- deck_top (1): Index of top card in deck (-1 if empty)
- deck_order (36): Company IDs in draw order
- phase (1): Compact phase storage (mirrors visible one-hot)
- coo_level (1): Compact CoO storage (mirrors visible one-hot)
- auction_company (1): Compact auction company ID
- auction_high_bidder (1): Compact high bidder player ID
- auction_starter (1): Compact auction starter player ID
- corp_price_indices (8): Compact price index per corp (mirrors visible one-hot)

**State Management:**

State is purely functional within a single turn cycle. No global mutable state—everything lives in the float array. Entity accessors are stateless; they only cache offsets on first `initialize()` call. This allows:
- Multiple concurrent game states without cross-contamination
- Efficient batching for parallel self-play
- Thread-safe read-only state access in RL training loops

## Key Abstractions

**GameState:**
- Purpose: Single source of truth for entire game state; memory container
- Examples: `core/state.pyx` (lines 110-189 for class definition)
- Pattern: Cython `cdef class` wrapping NumPy `float32` array. Layout computed once at init via `StateLayout` struct. All state mutations go through explicitly declared methods. No Python object overhead during game play.

**Entity Accessors (Player, Corporation, Company, TurnState, etc.):**
- Purpose: Stateless API for specific game entities; provide getter/setter access without object overhead
- Examples: `entities/player.pyx`, `entities/corp.pyx`, `entities/company.pyx`, `entities/turn.pyx`
- Pattern: Cython `cdef class` with methods taking GameState parameter. Each entity caches its base offset from layout on `initialize()` call. Provides both low-level `cdef` nogil functions (for performance-critical code like action validation) and high-level `cpdef` Python-callable methods (for game logic).

**ActionLayout:**
- Purpose: Map action vector indices to phase-specific action meanings
- Examples: Defined in `core/actions.pxd` (lines 50-83)
- Pattern: Layout struct with phase boundaries (invest_start, bid_start, acquisition_start, etc.) and sub-offsets. Pre-computed at game start. Decoding: action index → phase offset → phase-specific offset → ActionInfo struct.

**StateLayout & Offset Structs:**
- Purpose: Pre-compute all field offsets so lookups are O(1)
- Examples: `StateLayout`, `PlayerFieldOffsets`, `CorpFieldOffsets`, `TurnStateOffsets` in `core/state.pxd`
- Pattern: Offset structs define relative positions within each entity block (e.g. Player.cash=0, Player.net_worth=1). StateLayout adds absolute positions. Entities cache both during `initialize()`.

## Entry Points

**GameState Constructor:**
- Location: `core/state.pyx:GameState.__cinit__()`
- Triggers: User calls `GameState(num_players=N)`
- Responsibilities: Allocate float32 array, compute StateLayout, initialize to game-start state, return initialized GameState

**get_nn_input():**
- Location: `core/state.pyx:GameState.get_nn_input()` (inferred from architecture)
- Triggers: RL training loop needs NN input
- Responsibilities: Return visible state portion of array, with player rotation (active player first). Size = `visible_size` floats.

**get_valid_action_mask():**
- Location: `core/actions.pyx` (inferred from VECTORS.md)
- Triggers: RL training loop needs to filter invalid actions
- Responsibilities: Generate boolean mask of valid actions for current phase/state. Called before NN forward pass.

**decode_action():**
- Location: `core/actions.pyx:decode_action()` (inferred)
- Triggers: RL agent selects an action index from NN output
- Responsibilities: Convert action index to ActionInfo struct with phase, action_type, slot/corp_id, amount. Ready for game logic.

## Error Handling

**Strategy:** Minimal defensive checks in Cython core (boundscheck=False, nonecheck=False for speed). Validation delegated to entity setter methods and action mask generation.

**Patterns:**
- Entity setters clamp or default invalid input: `set_corp_active()` converts boolean to 1.0 or 0.0
- Action mask generation filters invalid actions at source: only return indices for legal moves in current phase
- State layout offsets pre-computed at init time and never recalculated, so bounds errors caught at GameState construction
- No runtime assertions in nogil code paths; assumes valid input from Python layer

## Cross-Cutting Concerns

**Logging:** Not implemented. Game runs at high speed for RL training; logging would bottleneck performance.

**Validation:** Two levels:
  1. **Action legality**: Enforced via action masks (no validation needed during apply)
  2. **State consistency**: Bounds checks and type conversions in entity getters/setters (int ↔ normalized float)

**Normalization:** Constants defined in `core/data.pyx`: `CASH_DIVISOR=200.0`, `SHARE_DIVISOR=7.0`, `STAR_DIVISOR=20.0`, `INCOME_DIVISOR=10.0`, `MAX_ROUNDTRIPS=2.0`. All state storage uses these divisors; getters/setters automatically convert to/from integers.

**Memory Management:** NumPy array owned by GameState, held via `float*` pointer. Python GC controls array lifetime. No manual deallocation; Cython handles cleanup on object destruction.

---

*Architecture analysis: 2026-01-20*
