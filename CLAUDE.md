# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous float32 array that can be passed directly to PyTorch without serialization overhead.

**Key characteristics:**
- 2-6 player support with dynamic state sizing
- ~3500-3800 floats per game state (varies by player count)
- No Python object overhead in hot paths (nogil execution)
- Benchmark target: thousands of games per minute

## Directory Structure

```
/home/icebreaker/rss-az-cython2/
├── core/                  # Low-level game engine and action handling
│   ├── state.pyx/pxd     # GameState class - central float32 array
│   ├── data.pyx/pxd      # Static game data (companies, corps, market)
│   ├── actions.pyx/pxd   # Action space layout and decoding
│   └── driver.pyx        # GameDriver for action dispatch
├── entities/              # Entity handles for clean state access
│   ├── player.pyx        # Player entity (cash, shares, companies)
│   ├── corp.pyx          # Corporation entity (IPO'd companies)
│   ├── turn.pyx          # Turn state entity (phase tracking)
│   ├── company.pyx       # Company entity (auction deck)
│   ├── deck.pyx          # Company deck management
│   ├── market.pyx        # Share price market spaces
│   ├── fi.pyx            # Foreign investor entity
│   └── encoding.pyx      # One-hot encoding utilities
├── phases/                # Game phase handlers
│   ├── invest.pyx        # Investment phase (buy/sell/auction)
│   ├── bid.pyx           # Bid in auction phase
│   ├── acquisition.pyx   # Acquisition offers phase
│   ├── closing.pyx       # Company closing phase
│   ├── income.pyx        # Income payment phase
│   ├── wrap_up.pyx       # Turn wrap-up (FI buying)
│   └── temp_end_turn.pyx # End turn transition
├── tests/                 # Test suite organized by phase
│   ├── conftest.py       # Pytest fixtures
│   └── phases/           # Phase-specific tests
├── setup.py              # Cython build configuration
├── RULES.md              # Complete game rules (24KB)
├── VECTORS.md            # State/action vector documentation
└── RSS.pdf               # Original board game rulebook
```

## Architecture Overview

### Entity Handles Pattern

Global singleton instances provide clean access to state array regions:

```python
PLAYERS = [Player(i) for i in range(6)]  # Player handles
CORPS = [Corporation(i) for i in range(8)]  # Corporation handles
TURN = TurnState()  # Turn tracking
FI = ForeignInvestor()  # Foreign investor
```

Each entity provides:
- **cdef methods** (nogil): Direct pointer arithmetic, max performance
- **cpdef methods**: Python-accessible wrappers for testing
- **Access pattern**: `entity.get_field(state)` reads from cached offset

### Low-level vs High-level Access

- **Low-level** (`cdef` nogil): Performance-critical loops, action validation
- **High-level** (cpdef/def): Python code, testing, debugging
- High-level wraps low-level (no code duplication)

## Key Modules

### GameState (`core/state.pyx`)

Central data structure: single contiguous float32 numpy array.

**Two-part layout:**
- **Visible state**: NN input (player-rotated so active player first)
- **Hidden state**: Internal bookkeeping (truncated before NN sees state)

**Hidden state purposes:**
- **Information hiding**: Data the model shouldn't see (deck order, canonical active player)
- **Bookkeeping**: Offer buffers for acquisition/closing phases
- **Performance**: Compact storage for O(1) access to one-hot values (phase, auction company, corp price indices, etc.)

**Sizes by player count:**
| Players | Visible | Hidden | Total |
|---------|---------|--------|-------|
| 2 | 2943 | 862 | 3805 |
| 3 | 3023 | 862 | 3885 |
| 6 | 3275 | 862 | 4137 |

### Actions (`core/actions.pyx`)

Dynamic action space: `186 + (num_players * 20)` total actions.
- 3 players: 246 actions
- 6 players: 306 actions

**Action layout by phase:**
- INVEST: 1 pass + auction slots + 8 buy + 8 sell
- BID: 1 leave + 19 raise bid amounts
- ACQUISITION: 51 price offsets + 2 FI actions + 1 pass
- CLOSING: 1 close + 1 pass
- DIVIDENDS: 26 dividend amounts
- ISSUE: 1 issue + 1 pass
- IPO: 1 pass + 64 corp/par combinations

### Driver (`core/driver.pyx`)

Stateless game loop engine:
- `apply_action(state, action_idx, history)`: Main entry point
- Auto-applies forced actions when only 1 legal choice
- Returns: `STATUS_OK`, `STATUS_INVALID`, or `STATUS_GAME_OVER`

### Data (`core/data.pyx`)

Static game constants:
- 36 companies, 8 corporations, 27 market prices
- **Normalization divisors:**
  - `CASH_DIVISOR = 200.0` (prices, cash)
  - `SHARE_DIVISOR = 7.0` (share counts)
  - `STAR_DIVISOR = 20.0` (star ratings)
  - `MAX_ROUNDTRIPS = 2.0` (buy/sell tracking)

## State Representation

**Normalization strategy:**
- Integers divided by divisor before storage
- Retrieved by multiplying back: `<int>(value * DIVISOR + 0.5)`

**One-hot encodings for:**
- Phase (11 values)
- Cost of ownership (7 levels)
- Turn order (per player)
- Price index (27 market spaces per corp)

**State layout:**
```
[Phase (11) | CoO Level (7) | Players (repeated) | FI (37) | Companies (108) |
 Company Incomes (36) | Market (27) | Corporations (872) | Turn (complex) |
 Static Data (400) | HIDDEN: Active Player, Deck, Offer Buffers]
```

**Player stride** = `72 + num_players` floats per player

**Corporation stride** = 109 floats per corp

See `VECTORS.md` for exact offsets.

## Game Flow & Phases

**11 Phases** (indices 0-10):
| Index | Phase | Description |
|-------|-------|-------------|
| 0 | INVEST | Buy/sell shares, start auctions |
| 1 | BID_IN_AUCTION | Bidding for a company |
| 2 | WRAP_UP | FI buying companies at face value |
| 3 | ACQUISITION | Corps acquiring companies |
| 4 | CLOSING | Player-owned companies closing |
| 5 | INCOME | Dividend payments |
| 6 | DIVIDENDS | Dividend calculation |
| 7 | END_CARD | Game end triggered |
| 8 | ISSUE_SHARES | Corp issuing shares |
| 9 | IPO | Company → Corporation conversion |
| 10 | GAME_OVER | Terminal state |

**Automated phases** (no player input): WRAP_UP, INCOME, TEMP_END_TURN

### Offer Buffer Pattern (acquisition.pyx, closing.pyx)

Both ACQUISITION and CLOSING phases use a **one-by-one offer presentation** pattern to keep the action space small. Instead of exposing all possible offers simultaneously (which would create a combinatorial explosion), offers are:

1. **Generated once** at phase entry into a hidden state buffer
2. **Sorted** by priority rules (varies by phase)
3. **Presented one at a time** to the active player
4. **Advanced sequentially** after each accept/pass decision
5. **Re-validated dynamically** since earlier decisions may invalidate later offers

**Why this pattern?**
- Action space stays constant regardless of game state complexity
- Model sees one decision at a time (cleaner learning signal)
- Priority ordering is deterministic (no hidden information)
- Buffer lives in hidden state (not visible to NN, but persists across actions)

#### ACQUISITION Phase Offers

**Priority order** (RULES-compliant):
1. **OS→FI**: OS corporation buys from Foreign Investor at face value (special ability), sorted by face value DESC
2. **Other Corp→FI**: By (share price DESC, face value DESC) - higher-valued corps and more expensive companies first
3. **Corp→Corp**: Same president controls buyer and seller, sorted by (buyer price DESC, face value DESC)
4. **Corp→Player**: President's corp buying their private companies, sorted by (buyer price DESC, face value DESC)

**Hidden buffer layout:**
```
[offer_count][offer_index][corp_id₀, company_id₀][corp_id₁, company_id₁]...
```

**Receivership handling**: Corps without a president (in receivership) have automated behavior:
- Auto-buy from FI at HIGH price if affordable (most expensive company first)
- Auto-pass on all other offers (receivership can only buy from FI)

**Actions**: 51 price offsets (low to high), FI_HIGH, FI_FACE (OS only), PASS

#### CLOSING Phase Offers

**Two-stage process:**
1. **Auto-close** (deterministic, no player input):
   - FI closes companies with negative adjusted income
   - Receivership corps close red/orange companies above CoO thresholds
2. **Offer-based close** (player decisions):
   - Only for player-owned and player-presided corps
   - Sorted by face value ascending (cheapest first)

**Hidden buffer layout:**
```
[offer_count][offer_index][owner_type₀, owner_id₀, company_id₀]...
```
Where `owner_type` is OWNER_PLAYER (0) or OWNER_CORP (1).

**Validation rules:**
- Company not already closed this phase
- Ownership unchanged since buffer generation
- Corp last-company rule: can't close if corp would have 0 companies

**Mandatory close** (after all offers processed): If a player would have negative income+cash, their cheapest negative-income private company is force-closed repeatedly until safe.

**Actions**: CLOSE, PASS

## Code Conventions

### Naming
- `corp_id`, `company_id`, `player_id` = indices (0-7, 0-35, 0-5)
- `CORPS[corp_id]`, `PLAYERS[player_id]` = global singletons
- `PHASE_*` constants from `GamePhases` enum

### Normalization
```cython
# Store: divide by divisor
state[offset] = cash / CASH_DIVISOR

# Retrieve: multiply and round
cash = <int>(state[offset] * CASH_DIVISOR + 0.5)
```

### One-hot patterns
```cython
# Find set bit
for i in range(N):
    if array[offset + i] == 1.0:
        return i

# Set bit
set_one_hot(array, offset, index, size)
```

### Pointer safety
- All `nogil` functions take pointers or direct offsets
- GameState provides `_player_ptr()`, `_corp_ptr()` helpers
- Entities compute offsets once and cache them

## Build Commands

```bash
# Build Cython extensions (required before running any Python code)
python setup.py build_ext --inplace

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_invest.py -v

# Clean build artifacts (.c, .so, .html, build/, *.egg-info)
python setup.py clean
```

## Testing Approach

**Organization:** Tests in `tests/`, phase tests in `tests/phases/`

**Key fixtures** (conftest.py):
- `game_state`: 3-player initialized game (seed=42)
- `invest_state`, `bid_state`, `trade_state`: Pre-configured states
- `apply_and_track`: Helper for action history and invariants

**Test patterns:**
- Validate action effects on state
- Check mask validity after actions
- Verify phase transitions
- Assert game invariants (cash conservation, share counts)

**Status codes:** STATUS_OK (0), STATUS_INVALID (1), STATUS_GAME_OVER (2)

## Key Files by Task

| Task | Primary Files | Secondary Files |
|------|---------------|-----------------|
| Add game rule | `phases/*.pyx` | `core/data.pyx`, `RULES.md` |
| Modify action space | `core/actions.pyx` | `core/driver.pyx`, `phases/*.pyx` |
| Debug state | `core/state.pyx`, `VECTORS.md` | Entity files |
| Optimize performance | Any `.pyx` | Check compiler directives, nogil |
| Add phase | Create `phases/new.pyx` | `core/driver.pyx`, `core/actions.pyx` |
| Fix bug | Tests first | Phase/entity files |

## Documentation

- **CLAUDE.md**: This file - agent instructions
- **RULES.md**: Complete game rules (24KB)
- **VECTORS.md**: State/action vector layouts with exact offsets
- **RSS.pdf**: Original board game rulebook

---

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Understanding Game Rules

**Before working on any game logic**, read `RULES.md` to understand the correct behavior. This is the authoritative source for how the game should work. Common areas where rules matter:
- Share buying/selling price calculations
- Phase transitions and action ordering
- Dividend payments and constraints
- Acquisition and closing logic
- Receivership and bankruptcy handling

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Agent Work Standards

When working on any task, create beads issues (`bd create`) for:
- Bugs or problems discovered
- Rule discrepancies or ambiguities
- Missing functionality or gaps
- Performance concerns
- Code quality issues
- Anything that needs follow-up but is out of scope for current task

**No insights or work should be lost.** When in doubt, create an issue.

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
