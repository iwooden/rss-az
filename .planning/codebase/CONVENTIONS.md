# Coding Conventions

**Analysis Date:** 2026-01-20

## Naming Patterns

**Files:**
- Cython files use `.pyx` for implementation and `.pxd` for declarations (C interface)
- Python helper files use `.py` for pure Python (e.g., `core/__init__.py`)
- File names are lowercase with underscores: `player.pyx`, `company.pyx`, `state.pyx`
- Related pairs: declaration files (`.pxd`) paired with implementation files (`.pyx`)

**Functions:**
- Snake_case for function names: `get_player_cash()`, `set_player_cash()`, `compute_layout()`
- Functions prefixed with underscore for internal/private: `_init_synergies()`, `_init_price_lookup()`, `_is_player_president()`
- Cdef functions (C-level) use lowercase: `get_player_cash(float* player, PlayerOffsets* p) noexcept nogil`
- Cpdef functions (C and Python callable) use lowercase: `cpdef int get_cash(self, GameState state)`
- Getter/setter pattern: `get_*()` and `set_*()` for state access
- Specific naming: `add_*()` for incremental updates, e.g., `add_player_cash()`, `add_cash()`

**Variables:**
- Local variables use snake_case: `player_id`, `corp_id`, `num_players`, `offset`
- Struct field names use lowercase: `cash`, `net_worth`, `owned_shares`, `is_president`
- Constants (module-level) use UPPERCASE: `NUM_COMPANIES`, `NUM_CORPS`, `CASH_DIVISOR`, `SHARE_DIVISOR`
- DEF compile-time constants use UPPERCASE: `DEF NUM_COMPANIES = 36`
- Struct instances use lowercase: `layout`, `state`, `offsets`

**Types:**
- Struct names use CamelCase: `StateLayout`, `ActionLayout`, `PlayerOffsets`, `CorpFieldOffsets`
- Class names use CamelCase: `GameState`, `Player`, `Corporation`, `Company`, `Deck`
- Enum names implicit (use ACTION_TYPE style constants in cdef enum)
- Type aliases use cimport: `from core.state cimport GameState`

## Code Style

**Formatting:**
- Line length: No strict limit enforced (files contain lines up to ~100 chars typically)
- Indentation: 4 spaces per level (consistent across `.pyx` files)
- No external linter/formatter detected (no `.eslintrc`, `.pylintrc`, `.flake8`, or `pyproject.toml` config)
- Cython pragma comments on first line of modules: `# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True`

**Linting:**
- No active linting framework configured (no pytest.ini, flake8, or black config found)
- Manual style adherence expected based on conventions observed

**Compiler Directives:**
- Always use high-performance Cython directives in `.pyx` files:
  ```cython
  # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
  ```
- These disable Python safety checks for maximum speed (appropriate for AlphaZero training context)
- In function signatures: `noexcept nogil` for C-level performance functions
- Example: `cdef int get_player_cash(float* player, PlayerOffsets* p) noexcept nogil`

## Import Organization

**Order:**
1. Cython directives (cimport cython first if needed)
2. Standard library imports (libc imports)
3. NumPy imports (cimport numpy, import numpy)
4. Local cimports from pxd files (other modules' declarations)
5. Local imports from pyx files (runtime access)
6. Constants and DEF statements

**Pattern:**
```cython
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Docstring explaining module purpose."""

cimport cython
from libc.string cimport memcpy, memset
from libc.math cimport round
cimport numpy as cnp
import numpy as np

from core.state cimport GameState, StateLayout
from core.data cimport GameConstants, CASH_DIVISOR
from entities.player cimport Player

DEF NUM_COMPANIES = 36
DEF NUM_CORPS = 8
```

**Path Organization (No aliases detected):**
- Direct imports from module path: `from core.state cimport GameState`
- No path aliases configured (no `jsconfig.json` or TypeScript equivalents)
- Relative imports avoided; absolute module paths preferred

**Barrel Files:**
- Used in entity and helper packages for re-export
- Example `entities/__init__.pyx`:
  ```python
  from entities import player as _player_module
  Player = _player_module.Player
  PLAYERS = _player_module.PLAYERS

  __all__ = ['Player', 'PLAYERS', ...]
  ```
- Pattern: Import module, re-export selected symbols at package level

## Error Handling

**Patterns:**
- No explicit Python exception handling detected (try/except not used)
- Cython `noexcept nogil` functions prevent exceptions from being raised (by design)
- Silent failures on invalid state: Functions return -1 or 0 on "not found" conditions
  - Example in `entities/player.pyx:93-94`: `get_turn_order()` returns `-1` if not found
- No validation of input parameters; assumes preconditions met
- Memory safety relied on through Cython type system rather than runtime checks

**Design:**
- High-performance code assumes valid inputs (no defensive checks)
- Error conditions implicit in return values (0 = not found, -1 = error flag)
- Appropriate for game engine (validation happens at higher Python level)

## Logging

**Framework:** None active (no logging library used)

**Patterns:**
- Debugging output via `print()` statements in setup.py only
- No logging calls in game logic (all code is performance-critical)
- Debug support via comments and documentation rather than log statements
- Comment in `setup.py:56`: `'embedsignature': True,  # Useful for debugging`

**Expected Approach:**
- Use inline comments for non-obvious logic
- Print debugging only in non-performance-critical initialization code
- Game state introspection via direct array access for analysis

## Comments

**When to Comment:**
- Clarify non-obvious algorithm logic (e.g., layout computations in `.pxd` files)
- Explain complex indexing calculations or offset computations
- Document layout structures before major sections
- Mark performance-critical decisions (e.g., bounds checking disabled)

**Style:**
- Inline comments use `#` with space: `# This is a comment`
- Section headers use separator lines:
  ```cython
  # =============================================================================
  # ACTION LAYOUT STRUCT
  # =============================================================================
  ```
- Block comments explain struct/array layout before definitions

**Docstrings:**
- Module-level docstring explains purpose (always present)
- Class docstrings explain role and stateless operation pattern
  - Example from `entities/player.pyx:17-22`:
    ```python
    """
    Entity handle for accessing player state.
    Players are instantiated once at module load with their player_id.
    Offsets are computed lazily on first access to a GameState.
    All methods take GameState as first argument for stateless operation.
    """
    ```
- Function docstrings explain inputs, outputs, and behavior
  - Example: `"""Get player's cash (integer dollars)."""`
  - One-liners for simple accessors
- No formal docstring format (no Sphinx/Google/NumPy style)

## Function Design

**Size:** Most functions are small (1-20 lines)
- Accessor functions typically 1-3 lines
- Layout computation functions ~30-60 lines
- Complex game logic functions in 50-150 line range (e.g., `decode_action()` in `actions.pyx`)

**Parameters:**
- Minimalist approach: pass only what's needed
- Cdef functions take direct pointers: `(float* player, PlayerOffsets* p)`
- Methods on entities take `GameState state` as first parameter (stateless design)
- No default parameters used

**Return Values:**
- Functions return computed values directly, never None
- Invalid/not-found cases use sentinel values (-1, 0, false)
- Cdef void functions modify state in-place via pointers
- Cpdef functions bridge C and Python: return native types that map cleanly

**Entity Methods Pattern:**
- All operations stateless: state object passed in, not stored
- Example from `entities/corp.pyx:78-80`:
  ```python
  cpdef int get_cash(self, GameState state):
      """Get corporation's cash (integer dollars)."""
      return <int>(state._data[self._cash_offset] * CASH_DIVISOR + 0.5)
  ```

## Module Design

**Exports:**
- Declaration file (`.pxd`) exports C-level types and functions for other Cython modules
- Implementation file (`.pyx`) exports Python-callable functions via `cpdef`
- Python init files (`.py`) re-export modules at package level

**Pattern from `core/__init__.py`:**
```python
from core.state import GameState
from core.data import *
```

**Barrel Files:**
- All entity types re-exported in `entities/__init__.pyx`
- All helpers available from `helpers/__init__.py` (though imports are inline where needed)
- Example usage: `from entities import Player, PLAYERS, Corporation, CORPS`

## Normalization & Constants

**Float Normalization:**
- Game values (cash, shares) stored as floats with divisors for memory efficiency
- `CASH_DIVISOR = 200.0`: $1 = 0.005 in storage
- `SHARE_DIVISOR = 7.0`: Shares normalized to [0, 1] range
- `STAR_DIVISOR = 20.0`: Star ratings normalized
- Pattern: `get` multiplies by divisor, `set` divides by divisor
  - From `entities/player.pyx:56-58`:
    ```python
    cpdef int get_cash(self, GameState state):
        """Get player's cash (integer dollars)."""
        return <int>round(state._data[self._cash_offset] * CASH_DIVISOR)
    ```

## Cython-Specific Patterns

**Cdef vs Cpdef:**
- `cdef`: C-level only, no Python overhead, used for performance-critical code
- `cpdef`: Both C and Python callable, used for public API
- Example from helpers: low-level pointer access uses `cdef`, entity methods use `cpdef`

**Noexcept Nogil:**
- `noexcept`: Function cannot raise exceptions (compile-time verified)
- `nogil`: Function releases the Python GIL (allows true parallelism)
- Used everywhere in helpers and core modules for performance
- Example: `cdef int get_player_cash(float* player, PlayerOffsets* p) noexcept nogil:`

**Type Casts:**
- Explicit casting with angle brackets: `<int>(value + 0.5)` for rounding, `<float>cash / CASH_DIVISOR`
- Used to convert between integer game values and float storage
- Rounding with `+ 0.5` before cast to truncate

---

*Convention analysis: 2026-01-20*
