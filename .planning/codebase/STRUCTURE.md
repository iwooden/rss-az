# Codebase Structure

**Analysis Date:** 2026-01-20

## Directory Layout

```
rss-az-cython2/
├── core/                   # Game state container, constants, actions, data layout
│   ├── __init__.py        # Module exports (GameState, game data constants)
│   ├── __init__.pxd       # Cython declarations for core exports
│   ├── state.pyx          # GameState class and state layout computation
│   ├── state.pxd          # GameState declaration and StateLayout structs
│   ├── data.pyx           # Static game data, constants, normalization divisors
│   ├── data.pxd           # Data declarations and game constants enums
│   ├── actions.pyx        # Action space layout, action decoding, mask generation
│   └── actions.pxd        # ActionLayout struct and action decoding declarations
├── entities/              # Entity handle classes (Player, Corp, Turn, etc.)
│   ├── __init__.pyx       # Package init, creates singleton entity instances
│   ├── __init__.pxd       # Entity declarations
│   ├── player.pyx         # Player entity (cash, shares, companies, net_worth)
│   ├── player.pxd         # Player class and PlayerOffsets struct
│   ├── corp.pyx           # Corporation entity (active, cash, shares, prices)
│   ├── corp.pxd           # Corporation class and CorpFieldOffsets struct
│   ├── turn.pyx           # TurnState entity (phase, auction, dividend, IPO, etc.)
│   ├── turn.pxd           # TurnState class and TurnStateOffsets struct
│   ├── company.pyx        # Company entity (location, ownership, transfers)
│   ├── company.pxd        # Company class and company location enums
│   ├── fi.pyx             # ForeignInvestor entity (cash, owned_companies)
│   ├── fi.pxd             # ForeignInvestor class declaration
│   ├── market.pyx         # Market entity (available share price slots)
│   ├── market.pxd         # Market class declaration
│   ├── deck.pyx           # Deck entity (draw order, top card)
│   └── deck.pxd           # Deck class declaration
├── phases/                # Phase handler implementations (empty, reserved)
├── tests/                 # Test files (empty, reserved)
├── setup.py               # Cython build configuration
├── CLAUDE.md              # Developer instructions
├── RULES.md               # Rolling Stock Stars board game rules reference
└── VECTORS.md             # State and action vector layout specification
```

## Directory Purposes

**core/**
- Purpose: Game state container, static data, action decoding, and normalization constants
- Contains: Primary `GameState` class managing the float32 array, layout structs for offset computation, static company/corp/market data, action space layout and decoding logic
- Key files: `core/state.pyx` (GameState class), `core/data.pyx` (constants and game data), `core/actions.pyx` (action decoding)

**entities/**
- Purpose: Lightweight entity handle classes providing clean getter/setter interface to game state
- Contains: One class per major game entity (Player, Corporation, TurnState, Company, ForeignInvestor, Market, Deck), each with high-level Python methods and low-level nogil C functions
- Key files: `entities/player.pyx` (Player entity), `entities/corp.pyx` (Corporation entity), `entities/turn.pyx` (TurnState entity with all phase tracking)

**phases/**
- Purpose: Phase handler implementations (stub directory, currently empty)
- Contains: Currently no files; reserved for future phase-specific game logic
- Key files: None yet

**tests/**
- Purpose: Unit and integration tests (stub directory, currently empty)
- Contains: Currently no files; reserved for test suite
- Key files: None yet

## Key File Locations

**Entry Points:**
- `core/__init__.py`: Module exports GameState and re-exports from `core.state` and `core.data`
- `core/state.pyx:GameState.__cinit__()`: Constructor allocates state array, computes all layout structures
- `setup.py`: Build entry point; configures Cython compilation for all .pyx files in `core/`, `entities/`, `phases/`

**Configuration:**
- `setup.py`: Compiler directives (boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, overflowcheck=False for maximum performance)
- `VECTORS.md`: Complete specification of state array layout (visible vs hidden sections, field offsets, normalization constants, action space layout)
- `RULES.md`: Rolling Stock Stars game rules (reference for phase transitions, action legality, game mechanics)

**Core Logic:**
- `core/state.pyx`: GameState class (array container, layout computation), StateLayout computation, TurnStateOffsets, PlayerFieldOffsets, CorpFieldOffsets structs
- `core/data.pyx`: Company prices, star ratings, income, synergy arrays; market prices table; normalization divisors; corporation names; accessor functions
- `core/actions.pyx`: Action space layout, action mask generation, action decoding from index to ActionInfo

**State Access:**
- `entities/player.pyx`: Player entity (getters/setters for cash, net_worth, owned_companies, owned_shares, presidency, round-trip tracking)
- `entities/corp.pyx`: Corporation entity (getters/setters for active, cash, unissued/issued/bank shares, share_price, owned_companies, acquisition_companies, receivership)
- `entities/turn.pyx`: TurnState entity (phase, CoO level, turn number, auction state, dividend state, IPO state, acquisition state, closing state, end_card_flipped)
- `entities/company.pyx`: Company entity (location tracking, ownership, transfers between locations)

**Game Data:**
- `core/data.pyx:COMPANY_NAMES`: String names of 36 companies
- `core/data.pyx:COMPANY_FACE_VALUE`: Face value per company
- `core/data.pyx:COMPANY_STARS`: Star rating per company (1-5)
- `core/data.pyx:COMPANY_INCOME`: Base income per company
- `core/data.pyx:COMPANY_SYNERGY`: Synergy matrix (36×36) for income calculations
- `core/data.pyx:MARKET_PRICES`: Price table for 27 market slots (0→0, 1→5, 2→6, ..., 26→75)
- `core/data.pyx:CORP_NAMES`: Names of 8 corporations

**Testing:**
- `tests/`: Currently empty (no test files present)

## Naming Conventions

**Files:**
- `.pyx` (Cython implementation): `entity.pyx`, `data.pyx`
- `.pxd` (Cython declaration/header): `entity.pxd`, `data.pxd` (defines structs, declares functions for cimport)
- `.py` (Pure Python): `__init__.py`, `setup.py`

**Classes:**
- Entity handles: PascalCase singular: `Player`, `Corporation`, `TurnState`, `Company`, `ForeignInvestor`, `Market`, `Deck`
- Singleton instances: UPPERCASE: `PLAYERS` (list of 6 Player instances), `CORPS` (list of 8 Corporation instances), `TURN`, `MARKET`, `COMPANIES` (list of 36 Company instances), `COMPANIES_BY_NAME` (dict), `DECK`, `FI`

**Functions:**
- Getters: `get_<field>` or `is_<property>`: `get_cash()`, `get_net_worth()`, `is_corp_active()`
- Setters: `set_<field>`: `set_cash()`, `set_corp_active()`
- Computed fields: `add_cash()`, `bankrupt_corp()`, `transfer_ownership()`
- Low-level nogil: `_<function>`: `_get_player_cash()`, `_set_corp_active()` (for internal use in nogil paths)

**Variables:**
- Private fields: `_offset`, `_base_offset`, `_num_players` (cached offsets in entity instances)
- Const arrays: UPPERCASE: `COMPANY_NAMES`, `COMPANY_FACE_VALUE`, `MARKET_PRICES`, `CORP_NAMES`

**Types:**
- Struct names: Suffix `Offsets` or `Layout`: `StateLayout`, `TurnStateOffsets`, `PlayerFieldOffsets`, `CorpFieldOffsets`, `ActionLayout`, `ActionInfo`
- Enum names: `GameConstants`, `GamePhases`, `CorpIndices`
- Constants: UPPERCASE with `_`: `CASH_DIVISOR`, `SHARE_DIVISOR`, `STAR_DIVISOR`, `NUM_COMPANIES`, `NUM_CORPS`, `NUM_PHASES`, `NUM_PAR_PRICES`

## Where to Add New Code

**New Game Feature (e.g., new mechanic or phase transition rule):**
- State storage: Add fields to appropriate offset struct in `core/state.pxd` (e.g., add to `TurnStateOffsets` for turn-based state), update layout computation in `core/state.pyx`
- Entity access: Add getters/setters in `entities/<entity>.pyx` (e.g., `TurnState.get_new_field()`)
- Low-level helpers: Add nogil functions in entity .pyx if used in action validation
- Action handling: Add action type to `core/actions.pxd` enum if user-triggered, add decoding logic in `core/actions.pyx`
- Phase logic: Implement in `phases/<phase>.pyx` once phase handlers are created

**New Entity Type (e.g., a new game object with state):**
- Create `entities/<name>.pyx` and `entities/<name>.pxd`
- Define field offsets struct in .pxd, declare class in .pxd
- Implement class with `initialize()` method to cache offsets, getter/setter methods in .pyx
- Re-export in `entities/__init__.pyx` by importing and creating singleton instance
- Add layout computation to `core/state.pyx` if state is part of the main vector

**Utilities/Shared Functions:**
- Shared constants: Add to `core/data.pyx` (e.g., new price table, new corporation names)
- Shared accessor functions: Add to appropriate entity module (e.g., helper function for company lookup)
- Offset computation: Add struct + compute function to `core/state.pyx` if adds new state section

**Tests:**
- Test files: `tests/test_<module>.py` (e.g., `tests/test_player.py`, `tests/test_actions.py`)
- Use pytest; follow pattern once first test is created
- Build before testing: `python setup.py build_ext --inplace`
- Run: `pytest tests/` or `pytest tests/test_module.py -v`

## Special Directories

**.planning/codebase/**
- Purpose: GSD codebase analysis documents (ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md, STACK.md, INTEGRATIONS.md)
- Generated: Yes (by GSD analyzer)
- Committed: Yes

**build/**
- Purpose: Cython build artifacts (.c files, .so files, .egg-info directory)
- Generated: Yes (by `setup.py build_ext --inplace`)
- Committed: No (.gitignore)

**__pycache__/**
- Purpose: Python bytecode cache for .py files
- Generated: Yes (by Python interpreter)
- Committed: No (.gitignore)

**.venv/**
- Purpose: Python virtual environment
- Generated: Yes (by `python -m venv .venv`)
- Committed: No (.gitignore)

**.pytest_cache/**
- Purpose: Pytest test discovery and result cache
- Generated: Yes (by pytest)
- Committed: No (.gitignore)

## File Size & Complexity Indicators

**Large/Complex files:**
- `core/state.pyx` (~670 lines): GameState class + layout computation for all state sections
- `entities/turn.pyx` (~440+ lines): Turn state entity with phase tracking (auction, dividend, IPO, acquisition, closing)
- `entities/company.pyx` (~330+ lines): Company entity with location tracking and atomic transfers
- `core/actions.pyx` (~400+ lines inferred): Action space layout, mask generation, decoding
- `entities/corp.pyx` (~265+ lines): Corporation entity with share and price tracking
- `entities/player.pyx` (~200+ lines): Player entity with cash, shares, companies, presidents

## Import & Initialization Pattern

**Module initialization (one-time at import):**
1. `core/__init__.py`: Imports `GameState` from `core.state`, imports data constants from `core.data`
2. `entities/__init__.pyx`: Imports all entity modules (`player`, `corp`, `turn`, `company`, `fi`, `market`, `deck`), creates singleton instances for each entity (`PLAYERS=[Player(0), ..., Player(5)]`, `CORPS=[Corporation(0), ..., Corporation(7)]`, `TURN=TurnState()`, etc.), re-exports all
3. `core/actions.pyx`: Imports from `core.state`, `entities` for action decoding

**Per-game initialization:**
1. User creates `GameState(num_players=N)`
2. Game code imports entity singletons: `from entities import PLAYERS, CORPS, TURN, MARKET, COMPANIES`
3. Each entity calls `PLAYERS[i].initialize(state)`, `CORPS[j].initialize(state)`, etc. (typically done automatically by GameState constructor or explicitly by user)
4. Game loop uses entities for all state access: `PLAYERS[i].get_cash(state)`, `TURN.set_phase(state, PHASE_INVEST)`, etc.

## Migration Path for New Code

**From monolithic to modular:** The architecture is modular by design. If a single entity becomes too large:
- Split into separate concerns: e.g., `player.pyx` handles cash/net_worth, create `player_portfolio.pyx` for shares/companies
- Keep interface consistent: add methods to main entity that delegate to split modules
- Share offset structs in .pxd to avoid duplication

**Performance optimization:** If code becomes a bottleneck:
- Add `cdef` nogil version of method alongside `cpdef` Python version
- Move hot loops to nogil paths
- Profile first: use `python setup.py benchmark` to measure games/sec

---

*Structure analysis: 2026-01-20*
