# Codebase Structure

**Analysis Date:** 2026-01-20

## Directory Layout

```
rss-az-cython2/
├── core/                   # Game state container, constants, data layout
│   ├── __init__.py        # Module exports (GameState, game data)
│   ├── __init__.pxd       # Cython declarations
│   ├── state.pyx          # GameState class and layout computation
│   ├── state.pxd          # State class declaration and struct definitions
│   ├── data.pyx           # Static game data (companies, corps, prices, constants)
│   └── data.pxd           # Data declarations and normalization constants
├── entities/              # Entity handles (Player, Corp, Turn, etc.)
│   ├── __init__.pyx       # Package initialization, entity instantiation
│   ├── __init__.pxd       # Entity declarations
│   ├── player.pyx         # Player entity (cash, shares, companies, net_worth)
│   ├── player.pxd         # Player class declaration
│   ├── corp.pyx           # Corporation entity (active, cash, shares, prices)
│   ├── corp.pxd           # Corporation class declaration
│   ├── turn.pyx           # TurnState entity (phase, auction, dividend, IPO, etc.)
│   ├── turn.pxd           # TurnState class declaration
│   ├── company.pyx        # Company entity (location, ownership, auction)
│   ├── company.pxd        # Company class declaration
│   ├── fi.pyx             # ForeignInvestor entity (cash, companies)
│   ├── fi.pxd             # ForeignInvestor class declaration
│   ├── market.pyx         # Market entity (available spaces)
│   ├── market.pxd         # Market class declaration
│   ├── deck.pyx           # Deck entity (card order, draw state)
│   └── deck.pxd           # Deck class declaration
├── helpers/               # Low-level nogil accessor functions
│   ├── __init__.py        # Package initialization
│   ├── __init__.pxd       # Helper declarations
│   ├── player.pyx         # Player field offsets, cash, shares, companies
│   ├── player.pxd         # PlayerOffsets struct
│   ├── corp.pyx           # Corp field offsets, cash, shares, companies
│   ├── corp.pxd           # CorpOffsets struct
│   ├── turn.pyx           # Turn state field accessors
│   ├── turn.pxd           # Turn field declarations
│   ├── market.pyx         # Market space accessors
│   ├── market.pxd         # Market declarations
│   └── company.pyx        # Company location queries
├── phases/                # Phase handlers (empty in current codebase)
├── tests/                 # Test files (currently empty)
├── setup.py               # Cython build configuration
├── CLAUDE.md              # Developer instructions
├── RULES.md               # Rolling Stock Stars board game rules
└── VECTORS.md             # State and action vector layout specification
```

## Directory Purposes

**core/**
- Purpose: Game state container, static data, and normalization constants
- Contains: Primary `GameState` class managing the float32 array, layout structs for offset computation, static company/corp/market data
- Key files: `core/state.pyx` (GameState class, 671 lines), `core/data.pyx` (constants and game data)

**entities/**
- Purpose: Lightweight entity handle classes providing clean getter/setter interface
- Contains: One class per major game entity (Player, Corporation, TurnState, Company, ForeignInvestor, Market, Deck)
- Key files: `entities/player.pyx` (Player class), `entities/corp.pyx` (Corporation class), `entities/turn.pyx` (TurnState class, 440+ lines with all phase tracking)

**helpers/**
- Purpose: Low-level nogil accessor functions for performance-critical code paths
- Contains: Inline C functions operating on raw `float*` pointers, offset computation structs
- Key files: `helpers/corp.pyx` (corp accessor functions, 300+ lines), `helpers/player.pyx` (player accessors)

**phases/**
- Purpose: Phase handler implementations (stub directory, not yet populated)
- Contains: Currently empty
- Key files: None yet

**tests/**
- Purpose: Unit/integration tests
- Contains: Currently empty
- Key files: None yet

## Key File Locations

**Entry Points:**
- `core/__init__.py`: Module exports, imports `GameState` from `core.state`
- `core/state.pyx`: `GameState` class constructor (lines 333-353); initializes state array, computes all layout structures
- `setup.py`: Build entry point; configures Cython compilation for all .pyx files

**Configuration:**
- `setup.py`: Compiler directives (boundscheck=False, wraparound=False, cdivision=True, nonecheck=False for max performance)
- `VECTORS.md`: Complete specification of state array layout (visible vs hidden sections, normalization, field offsets)
- `RULES.md`: Rolling Stock Stars game rules (reference for phase transitions and action validation)

**Core Logic:**
- `core/state.pyx`: GameState class and layout computation (state layout struct, turn offsets, player offsets, corp offsets)
- `core/data.pyx`: Company prices, star ratings, market prices, normalization divisors, corp names
- `actions.pyx`: Action space layout, action masking, action decoding (186 + num_players*20 action count)

**State Access:**
- `entities/player.pyx`: Player entity (getters/setters for cash, net_worth, companies, shares, presidencies, round-trip tracking)
- `entities/corp.pyx`: Corporation entity (getters/setters for active, cash, shares, prices, owned companies, receivership status)
- `entities/turn.pyx`: TurnState entity (phase, CoO level, turn number, auction state, dividend state, IPO state, acquisition state, closing state)

**Testing:**
- `tests/`: Currently empty (no test files present)

## Naming Conventions

**Files:**
- `.pyx` (Cython implementation): `entity.pyx`, `data.pyx`
- `.pxd` (Cython declaration/header): `entity.pxd`, `data.pxd`
- `.py` (Pure Python): `__init__.py`, `setup.py`

**Classes:**
- Entity handles: PascalCase singular: `Player`, `Corporation`, `TurnState`, `Company`, `ForeignInvestor`, `Market`, `Deck`
- Singleton instances: UPPERCASE: `PLAYERS` (list of Player instances), `CORPS`, `TURN`, `MARKET`, `COMPANIES`, `DECK`, `FI`

**Functions:**
- Getters: `get_<field>` or `is_<property>`: `get_cash()`, `get_net_worth()`, `is_corp_active()`
- Setters: `set_<field>`: `set_cash()`, `set_corp_active()`
- Computed fields: `add_cash()`, `bankrupt_corp()`

**Variables:**
- Private fields: `_offset`, `_base_offset`, `_num_players` (cached offsets in entity handles)
- Const arrays: UPPERCASE: `COMPANY_NAMES`, `COMPANY_FACE_VALUE`, `MARKET_PRICES`

**Types:**
- Struct names: Suffix `Offsets` or `Layout`: `StateLayout`, `TurnStateOffsets`, `PlayerFieldOffsets`, `CorpFieldOffsets`
- Constants: UPPERCASE with `_`: `CASH_DIVISOR`, `SHARE_DIVISOR`, `STAR_DIVISOR`, `NUM_COMPANIES`, `NUM_CORPS`

## Where to Add New Code

**New Feature (e.g., new game mechanic):**
- State storage: Add fields to appropriate struct in `core/state.pyx` (layout computation)
- Entity access: Add getters/setters in `entities/<entity>.pyx`
- Helper functions: Add nogil accessors in `helpers/<entity>.pyx` if performance-critical
- Action handling: Add action types in `actions.pyx` if user-triggered
- Phase logic: Implement in `phases/<phase>.pyx` (once phase handlers are created)

**New Component/Module:**
- If it's a game entity (Player, Market, etc.): Create `entities/<name>.pyx` and `entities/<name>.pxd`
- If it's helper functions: Create `helpers/<name>.pyx` and `helpers/<name>.pxd`
- If it's a phase handler: Create `phases/<name>.pyx` and `phases/<name>.pxd`
- Re-export in `__init__.pyx` or `__init__.py` at package level

**Utilities/Helpers:**
- Shared accessor functions: `helpers/<entity>.pyx`
- Shared constants: `core/data.pyx`
- Offset computation: `core/state.pyx` (add struct + compute function)

**Tests:**
- Test files: `tests/test_<module>.py` (e.g., `tests/test_player.py`)
- Use pytest; follow existing test structure (none present yet; establish pattern)

## Special Directories

**.planning/codebase/**
- Purpose: GSD codebase analysis documents (this document, ARCHITECTURE.md, CONCERNS.md, etc.)
- Generated: Yes (by GSD analyzer)
- Committed: Yes

**build/**
- Purpose: Cython build artifacts (generated)
- Generated: Yes (by setup.py build)
- Committed: No (.gitignore)

**__pycache__/**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No

**.venv/**
- Purpose: Python virtual environment
- Generated: Yes
- Committed: No

## File Size & Import Structure

**Large files (complexity indicators):**
- `core/state.pyx` (671 lines): GameState class + layout computation
- `entities/turn.pyx` (440+ lines): Turn state entity with all phase tracking
- `helpers/corp.pyx` (300+ lines): Corp accessor functions
- `entities/corp.pyx` (265+ lines): Corporation entity
- `entities/company.pyx` (330+ lines): Company entity
- `entities/player.pyx` (200+ lines): Player entity
- `actions.pyx` (400+ lines): Action space layout and masking

**Import pattern:**
1. Core imports: `core.state` (GameState), `core.data` (constants)
2. Entity imports: `entities.<entity>` for high-level access
3. Helper imports: `helpers.<entity>` for nogil-compatible functions (internal use only)
4. Action imports: `actions` for action decoding and masking

**Module initialization:**
- `core/__init__.py`: Imports and re-exports `GameState` and data constants
- `entities/__init__.pyx`: Imports all entity modules, creates singleton instances (`PLAYERS`, `CORPS`, etc.), re-exports
- `helpers/__init__.pxd`: Declares helper struct types for cross-module use

---

*Structure analysis: 2026-01-20*
