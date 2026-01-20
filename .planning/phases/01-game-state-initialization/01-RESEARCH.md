# Phase 1: Game State Initialization - Research

**Researched:** 2026-01-20
**Domain:** Cython state initialization, deck building, entity management
**Confidence:** HIGH

## Summary

Game state initialization for this Cython-based board game engine follows a specific pattern: the GameState class allocates a contiguous float32 array in `__cinit__`, and initialization logic must properly set up all entity state using existing entity accessor methods. The deck building algorithm is complex with player-count-dependent rules already implemented in `Deck.setup()`. Companies are tracked by location flags across multiple state arrays (auction, revealed, removed, player-owned, FI-owned, corp-owned, or in-deck by absence).

The existing codebase provides complete entity accessor APIs (Player, ForeignInvestor, Corporation, Market, TurnState, Deck) that must be used rather than direct state array manipulation. All entities use singleton instances (PLAYERS, FI, CORPS, MARKET, TURN, DECK) that require `initialize()` to be called before use.

**Primary recommendation:** Implement as a `cpdef void initialize_game(self, int seed=-1)` method on GameState that calls entity `initialize()` methods, sets initial state via entity APIs, runs deck setup, draws initial companies, and configures turn state. Do NOT manipulate the float array directly - use the entity handle methods.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Cython | 3.x | Performance-critical game state | Zero-overhead C extension for AlphaZero training |
| NumPy | Latest | Float32 array storage | Efficient contiguous memory, PyTorch-compatible |
| libc.stdlib | Standard | RNG (rand, srand) | Already in use for deck shuffling, nogil compatible |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| libc.string | Standard | memset for bulk zeroing | Only if batch clearing needed (current code uses loops) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Cython | Pure Python | 10-100x slower, unacceptable for AlphaZero self-play |
| libc RNG | numpy.random | Requires GIL, breaks nogil performance |
| Direct array access | Python properties | Already using entity handles - this is the pattern |

**Installation:**
```bash
# Already installed - no new dependencies needed
# Existing: cython, numpy (see setup.py)
```

## Architecture Patterns

### Recommended Project Structure
Initialization should be implemented as a GameState method, not a standalone module:

```
core/
├── state.pyx           # Add initialize_game() here
├── state.pxd           # Add method declaration
└── data.pyx            # Already has static data

entities/
├── deck.pyx            # Already has setup() method - reuse
├── player.pyx          # Already has entity handles
├── fi.pyx              # Already has entity handles
├── corp.pyx            # Already has entity handles
├── market.pyx          # Already has entity handles
└── turn.pyx            # Already has entity handles
```

### Pattern 1: Entity Handle Initialization
**What:** All entity handles must be initialized before use
**When to use:** Start of initialize_game() method
**Example:**
```python
# Source: entities/player.pyx lines 242-263
cpdef void initialize_game(self, int seed=-1):
    """Initialize a new game from scratch."""
    cdef int i

    # Initialize all entity handles first
    for i in range(self._num_players):
        PLAYERS[i].initialize(self)
    FI.initialize(self)
    for corp in CORPS.values():
        corp.initialize(self)
    MARKET.initialize(self)
    TURN.initialize(self)
    DECK.initialize(self)

    # Now can safely use entity methods
    # ...
```

### Pattern 2: Entity State Modification via Handles
**What:** Use entity handle methods, not direct array access
**When to use:** All state modifications in initialization
**Example:**
```python
# Source: entities/player.pyx lines 268-279, entities/fi.pyx lines 43-54
# GOOD: Use entity handles
for i in range(self._num_players):
    starting_cash = 25 if self._num_players == 6 else 30
    PLAYERS[i].set_cash(self, starting_cash)
    PLAYERS[i].set_turn_order(self, i)

FI.set_cash(self, 4)

# BAD: Direct array manipulation
# self._data[offset] = value  # Don't do this!
```

### Pattern 3: Deck Building
**What:** Use existing Deck.setup() method - don't reimplement
**When to use:** Deck initialization phase
**Example:**
```python
# Source: entities/deck.pyx lines 118-183
# Deck building is already implemented with complex player-count logic
if seed < 0:
    # Use current time or other seed source
    seed = <int>time(NULL)

DECK.setup(self, self._num_players, seed)

# Then draw initial companies
cdef int company_id
for i in range(self._num_players):
    company_id = DECK.draw(self)
    if company_id >= 0:
        self.set_company_for_auction(company_id, True)
```

### Pattern 4: Corporation Reset
**What:** Clear all corporation state to inactive/unissued
**When to use:** Corporation initialization
**Example:**
```python
# Source: entities/corp.pyx lines 66-117, core/data.pyx lines 132
cdef Corporation corp
cdef int corp_id, total_shares

for corp_id in range(GameConstants.NUM_CORPS):
    corp = list(CORPS.values())[corp_id]

    # Mark inactive (no IPO yet)
    corp.set_active(self, False)

    # Clear cash
    corp.set_cash(self, 0)

    # Reset shares to unissued
    total_shares = get_corp_share_count(corp_id)
    corp.set_unissued_shares(self, total_shares)
    corp.set_issued_shares(self, 0)
    corp.set_bank_shares(self, 0)

    # Clear receivership
    corp.set_in_receivership(self, False)

    # Clear companies (both owned and acquisition)
    for company_id in range(GameConstants.NUM_COMPANIES):
        corp.set_owns_company(self, company_id, False)
        corp.set_acquisition_company(self, company_id, False)
```

### Pattern 5: Market Initialization
**What:** Mark all 27 market spaces as available
**When to use:** Market setup phase
**Example:**
```python
# Source: entities/market.pyx lines 39-48
cdef int i
for i in range(GameConstants.NUM_MARKET_SPACES):
    MARKET.set_space_available(self, i, True)
```

### Anti-Patterns to Avoid
- **Direct array manipulation:** Always use entity handles, not `self._data[offset] = value`
- **Forgetting entity initialization:** Must call `entity.initialize(self)` before using entity methods
- **Reimplementing deck logic:** Deck.setup() already handles all player-count edge cases
- **Mixing initialization order:** Initialize entity handles first, then use their methods
- **Skipping company location clearing:** Companies can be in multiple states (player, FI, corp, auction, revealed, removed)

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Deck building algorithm | Custom shuffling logic | `Deck.setup()` | Complex player-count rules (4p adds 5 orange not 4, 5p adds 7 not 5), color stacking order, "last" card inclusion |
| RNG seeding | Python random | `libc.stdlib.srand()` | Already used by Deck, consistent with existing code, nogil compatible |
| Entity state access | Direct float array indexing | Entity handle methods (Player.set_cash, etc.) | Handles normalization, bounds checking, one-hot encoding updates |
| Company location tracking | Custom flags | Company.transfer_to_X() methods | Atomically clears old location and sets new one, maintains cached location state |
| Share count initialization | Manual calculation | `get_corp_share_count(corp_id)` | Per-corp share counts differ (7,7,6,6,5,5,4,4) |

**Key insight:** The entity handle pattern exists to prevent direct state manipulation. All initialization must flow through entity APIs even though they add function call overhead - this ensures consistency and correctness.

## Common Pitfalls

### Pitfall 1: Forgetting Entity Handle Initialization
**What goes wrong:** Calling entity methods before `entity.initialize(state)` results in reading/writing to offset 0 (wrong memory location)
**Why it happens:** Entity handles compute offsets during initialize(), but are created as singletons at module load
**How to avoid:** First loop calls `initialize()` on all entities, second loop sets state
**Warning signs:** All entity reads return 0, all writes have no effect, segfaults on complex state access

### Pitfall 2: Inconsistent Company Location State
**What goes wrong:** A company is marked as both "in auction" and "player owned" simultaneously, or not marked anywhere
**Why it happens:** Setting location flags without clearing previous location, or forgetting to mark drawn companies
**How to avoid:** Use `Company.transfer_to_X()` methods which atomically clear and set, or manually clear before setting
**Warning signs:** Companies duplicated in UI, actions fail with "company not found", deck appears to have wrong count

### Pitfall 3: Wrong Starting Cash for 6-Player Games
**What goes wrong:** All players receive 30 coins instead of 25 for 6-player games
**Why it happens:** Forgetting the 6-player exception rule (RULES.md line 92)
**How to avoid:** Conditional check `starting_cash = 25 if num_players == 6 else 30`
**Warning signs:** 6-player games have economic imbalance, test failures for 6p starting state

### Pitfall 4: Phase/CoO Level Off-By-One Errors
**What goes wrong:** Phase set to 0 when it should be PHASE_INVEST (also 0), but CoO level set to 0 when it should be 1
**Why it happens:** Phase enum starts at 0, but CoO levels are 1-7 in game terms
**How to avoid:** Use `TURN.set_phase(self, GamePhases.PHASE_INVEST)` and `TURN.set_coo_level(self, 1)` not raw integers
**Warning signs:** Cost of ownership calculations incorrect, companies have wrong adjusted income

### Pitfall 5: Not Clearing Auction/Dividend/IPO State
**What goes wrong:** Game starts with stale auction state from previous game, causing invalid actions
**Why it happens:** TurnState fields are not automatically cleared when phase changes
**How to avoid:** Explicitly clear all phase-specific state: `TURN.clear_auction_company()`, dividend_corp, issue_corp, ipo_company, etc.
**Warning signs:** Action masking allows invalid actions, game starts mid-auction, test assertions fail

### Pitfall 6: Forgetting to Draw Initial Companies
**What goes wrong:** Game starts with no companies available for auction
**Why it happens:** RULES.md step 7 (draw N companies) is separate from deck building step 6
**How to avoid:** After `DECK.setup()`, loop `num_players` times calling `DECK.draw()` and `set_company_for_auction()`
**Warning signs:** INVEST phase has no valid actions, auction actions fail, UI shows empty market

## Code Examples

Verified patterns from the existing codebase:

### Initializing All Entity Handles
```python
# Pattern used throughout entities/*.pyx
cpdef void initialize_game(self, int seed=-1):
    """Initialize a new game state from scratch."""
    cdef int i
    cdef Corporation corp

    # CRITICAL: Initialize all entity handles before using them
    for i in range(self._num_players):
        PLAYERS[i].initialize(self)

    FI.initialize(self)

    for corp in CORPS.values():
        corp.initialize(self)

    MARKET.initialize(self)
    TURN.initialize(self)
    DECK.initialize(self)

    # Now entity methods are safe to call
```

### Setting Player Starting State
```python
# Source: entities/player.pyx accessor methods
cdef int i
cdef int starting_cash

# Player count determines starting cash (RULES.md line 92)
starting_cash = 25 if self._num_players == 6 else 30

for i in range(self._num_players):
    PLAYERS[i].set_cash(self, starting_cash)
    PLAYERS[i].set_turn_order(self, i)  # Linear order: 0=first
    PLAYERS[i].set_net_worth(self, starting_cash)

    # Clear ownership
    for company_id in range(GameConstants.NUM_COMPANIES):
        PLAYERS[i].set_owns_company(self, company_id, False)

    for corp_id in range(GameConstants.NUM_CORPS):
        PLAYERS[i].set_shares(self, corp_id, 0)
        PLAYERS[i].set_president_of(self, corp_id, False)
```

### Deck Setup and Initial Draw
```python
# Source: entities/deck.pyx lines 118-245
from libc.time cimport time

cdef int actual_seed
cdef int company_id
cdef int i

# Use provided seed or generate from time
if seed < 0:
    actual_seed = <int>time(NULL)
else:
    actual_seed = seed

# Build and shuffle deck (handles all player-count edge cases)
DECK.setup(self, self._num_players, actual_seed)

# Draw N companies and mark for auction (RULES.md step 7)
for i in range(self._num_players):
    company_id = DECK.draw(self)
    if company_id >= 0:
        # Use GameState method, not Company handle
        self.set_company_for_auction(company_id, True)
```

### Clearing Corporation State
```python
# Source: entities/corp.pyx accessor methods, core/data.pyx line 132
from core.data cimport get_corp_share_count

cdef Corporation corp
cdef int corp_id, company_id
cdef int total_shares

for corp_id in range(GameConstants.NUM_CORPS):
    corp = list(CORPS.values())[corp_id]

    corp.set_active(self, False)
    corp.set_cash(self, 0)
    corp.set_in_receivership(self, False)

    # Each corp has different total shares (7,7,6,6,5,5,4,4)
    total_shares = get_corp_share_count(corp_id)
    corp.set_unissued_shares(self, total_shares)
    corp.set_issued_shares(self, 0)
    corp.set_bank_shares(self, 0)

    # Clear all companies
    for company_id in range(GameConstants.NUM_COMPANIES):
        corp.set_owns_company(self, company_id, False)
        corp.set_acquisition_company(self, company_id, False)
```

### Setting Initial Turn State
```python
# Source: entities/turn.pyx accessor methods
from core.data cimport GamePhases

# Set phase to INVEST (RULES.md Phase 1)
TURN.set_phase(self, GamePhases.PHASE_INVEST)

# Set CoO level to 1 (game starts at level 1)
TURN.set_coo_level(self, 1)

# Set turn number to 1
TURN.set_turn_number(self, 1)

# Clear end card status
TURN.set_end_card_flipped(self, False)

# Clear consecutive passes
TURN.set_consecutive_passes(self, 0)

# Clear all phase-specific state
TURN.clear_auction_company()
TURN.clear_auction_high_bidder()
TURN.clear_auction_starter()
TURN.clear_auction_passed()
TURN.clear_dividend_corp()
TURN.clear_issue_corp()
TURN.clear_ipo_company()
TURN.clear_acq_active_corp()
TURN.clear_acq_target_company()
TURN.set_acq_fi_offer(self, False)
TURN.clear_closing_company()

# Set active player to player 0
self._set_active_player(0)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Python dict-based state | Single float32 array | Initial design | Zero-copy NN input, 10-100x faster |
| Direct array manipulation | Entity handle pattern | Current architecture | Type safety, encapsulation, normalization handled |
| Per-game RNG instances | Seeded libc rand | Deck implementation | Reproducible training, nogil compatible |
| Python random.shuffle | Fisher-Yates in C | Deck implementation | Faster, reproducible with seed |

**Deprecated/outdated:**
- None - this is a new feature in an existing architecture

## Open Questions

Things that couldn't be fully resolved:

1. **Active player rotation for NN input**
   - What we know: VECTORS.md mentions "player rotation (active player first)" for NN presentation
   - What's unclear: Whether initialize_game() should set up rotation or if that's handled elsewhere
   - Recommendation: Set active_player to 0 in initialization, let phase logic handle rotation on demand

2. **Company adjusted income initialization**
   - What we know: VECTORS.md line 114 says adjusted incomes are "automatically updated whenever the CoO level changes"
   - What's unclear: Whether `set_coo_level()` already updates company incomes or if initialization must do it manually
   - Recommendation: Check TurnState.set_coo_level() implementation; if it doesn't update, manually set incomes after setting CoO level

3. **Hidden vs visible state initialization**
   - What we know: State has hidden section (active_player, deck_order, etc.) and visible section (for NN)
   - What's unclear: Whether hidden state initialization is automatic or manual
   - Recommendation: Hidden state like deck_order is written by Deck.setup(), active_player by _set_active_player() - no special handling needed

## Sources

### Primary (HIGH confidence)
- `core/state.pyx` - GameState class structure, __cinit__ implementation, entity pointer methods
- `core/data.pyx` - Static game data arrays, COMPANY_* constants, get_corp_share_count()
- `entities/deck.pyx` - Deck.setup() implementation with player-count rules (lines 118-245)
- `entities/player.pyx` - Player entity handle pattern, cash/shares/company methods
- `entities/fi.pyx` - ForeignInvestor entity handle, cash/company methods
- `entities/corp.pyx` - Corporation entity handle, share/company/active methods
- `entities/market.pyx` - Market entity handle, space availability methods
- `entities/turn.pyx` - TurnState entity handle, phase/CoO/auction/dividend state methods
- `RULES.md` - Official game setup rules (lines 90-113), starting cash (line 92), deck building (lines 99-111)
- `VECTORS.md` - State vector layout, normalization constants, field offsets
- `.planning/REQUIREMENTS.md` - v1 requirements for initialization phase

### Secondary (MEDIUM confidence)
- None required - all findings verified with primary sources

### Tertiary (LOW confidence)
- None - complete information available in codebase

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All dependencies already in use, verified in setup.py and imports
- Architecture: HIGH - Entity handle pattern used throughout codebase, verified in entities/*.pyx
- Pitfalls: HIGH - Derived from actual code structure and common initialization mistakes

**Research date:** 2026-01-20
**Valid until:** 60 days (stable codebase, no external dependencies changing)
