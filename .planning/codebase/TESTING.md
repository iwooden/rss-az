# Testing Patterns

**Analysis Date:** 2026-01-20

## Test Framework

**Runner:**
- pytest (configured via presence of `.pytest_cache/`)
- Framework: Not explicitly configured (no `pytest.ini` or `setup.cfg` with pytest config)

**Assertion Library:**
- Not detected in codebase

**Run Commands:**
```bash
pytest tests/                          # Run all tests
pytest tests/test_invest.py -v         # Run specific test file with verbose output
python setup.py benchmark              # Run benchmark (1000 games, 3 players)
python setup.py benchmark --num-games=5000 --num-players=6  # Custom benchmark
```

**Current State:**
- Tests directory is empty: `/home/icebreaker/rss-az-cython2/tests/` contains no test files
- Pytest cache exists (`.pytest_cache/`) indicating tests have been run previously
- As of commit `315a129` ("delete tests"), test files have been removed from repo
- **Note:** Project is in "pre-testing" phase; testing infrastructure exists but test suite has been deleted

## Test File Organization

**Location:**
- Tests directory: `/home/icebreaker/rss-az-cython2/tests/`
- Expected test naming pattern: `test_*.py` (inferred from `pytest tests/test_invest.py` command in CLAUDE.md)
- **Current:** No test files present (deleted as of commit 315a129)

**Naming:**
- Convention: `test_*` prefix for test discovery
- Example from CLAUDE.md: `test_invest.py` (for investment phase tests)

## Build & Test Cycle

**Required Pre-Step:**
- Cython extensions must be compiled before running Python tests:
  ```bash
  python setup.py build_ext --inplace
  ```
- Tests depend on compiled `.so` files generated from `.pyx` sources
- Failure to build first will result in import errors

**Clean Build:**
```bash
python setup.py clean  # Remove .c, .so, .html, build/, *.egg-info
```

## Test Structure

**Expected Pattern (inferred from codebase):**

Since no test files remain, patterns are inferred from Cython codebase structure:

1. **Unit test organization** (expected):
   - One test file per core module: `test_state.py`, `test_actions.py`, `test_entities.py`
   - Group tests by entity: test Player methods, Corporation methods, etc.
   - Use GameState as fixture for all tests

2. **Test isolation:**
   - GameState instances created fresh for each test (no shared state between tests)
   - Pattern: Create new state, perform operations, assert results
   - No teardown needed (state is garbage collected)

**Setup/Teardown Pattern (expected):**

Based on CLAUDE.md and entity initialization pattern:

```python
# Expected setup pattern (not present in codebase)
from core import GameState
from entities import PLAYERS, CORPS

def test_player_cash():
    # Setup
    state = GameState(num_players=3)

    # Call initialize on all entities
    for p in PLAYERS:
        p.initialize(state)
    for c in CORPS:
        c.initialize(state)

    # Exercise
    PLAYERS[0].add_cash(state, 100)

    # Assert
    assert PLAYERS[0].get_cash(state) == 100
```

## Mocking

**Framework:** Not detected (no mock library imports in codebase)

**Patterns:**
- No mocking detected in codebase
- Expected approach: Mocking not necessary (all state is direct array manipulation)
- GameState is the test double for the entire game world

**What to Mock:**
- Nothing typically needed - GameState is already minimal (single float array)
- All game entities (Player, Corporation, Company, etc.) are stateless handles to GameState
- No external dependencies to mock (no DB, file I/O, networking)

**What NOT to Mock:**
- GameState itself (use real instances)
- Entity classes (use real instances for testing)
- Core game logic (test actual implementations)

## Fixtures and Factories

**Test Data:**

No fixture factories present in codebase. Expected pattern (inferred):

```python
# Hypothetical fixture pattern for GameState setup
@pytest.fixture
def game_state_3p():
    """Create 3-player game state with all entities initialized."""
    from core import GameState
    from entities import PLAYERS, CORPS, COMPANIES, DECK

    state = GameState(num_players=3)
    for p in PLAYERS:
        p.initialize(state)
    for c in CORPS:
        c.initialize(state)
    COMPANIES.initialize(state)
    DECK.initialize(state)

    return state

@pytest.fixture
def game_state_6p():
    """Create 6-player game state."""
    # Same pattern, different player count
    pass
```

**Location:**
- Expected: `tests/conftest.py` for shared fixtures
- Pattern: Pytest fixtures for common GameState configurations
- **Current:** No conftest.py exists (tests deleted)

## Coverage

**Requirements:** No coverage requirements detected
- No `.coveragerc` or pytest coverage config found
- No enforced minimum coverage percentage

**View Coverage (expected command):**
```bash
pytest --cov=core --cov=entities --cov=helpers tests/
pytest --cov=core --cov=entities --cov=helpers --cov-report=html tests/
```

## Test Types

**Unit Tests:**
- Scope: Individual function/method behavior
- Approach: Create GameState, call entity methods, verify state changes
- Example targets: Player cash operations, Corporation share tracking, Action decoding
- Coverage needed for: All accessor/setter methods, offset computations, state transitions

**Integration Tests:**
- Scope: Multi-entity interactions (player operations affect corporations)
- Approach: Execute game phases, verify cascading state updates
- Example: Player buys share → verify player shares updated, corp issued_shares updated, etc.
- **Current:** None present (tests deleted)

**E2E Tests:**
- Not detected in codebase
- Game engine is library (not an application) - no E2E testing needed
- AlphaZero would validate through self-play, not unit tests

**Benchmark Tests:**
- Custom command: `python setup.py benchmark`
- Measures: Games per minute (throughput)
- Configured parameters: `--num-games` (default 1000), `--num-players` (default 3)
- Purpose: Verify performance optimization (key for AlphaZero training)
- Implementation: Custom setuptools command in `setup.py` (lines 8-45)

## Common Patterns

**Async Testing:**
- Not applicable (no async code in Cython engine)

**Error Testing:**
- Cdef functions marked `noexcept` cannot raise exceptions
- No exception testing needed for performance-critical code
- Error conditions tested as invalid return values (-1, 0, false)

**State Assertions (expected pattern):**

```python
def test_player_can_buy_shares(game_state_3p):
    """Verify player share purchases update state correctly."""
    state = game_state_3p
    player = PLAYERS[0]
    corp = CORPS[0]  # Junkyard Scrappers

    # Initial state
    initial_shares = player.get_shares(state, corp_id=0)
    initial_cash = player.get_cash(state)

    # Execute purchase (hypothetical API)
    # player.buy_shares(state, corp_id=0, num_shares=5, price_per_share=10)

    # Verify
    assert player.get_shares(state, 0) == initial_shares + 5
    assert player.get_cash(state) == initial_cash - 50
```

**Stateless Design Tests (expected pattern):**

```python
def test_entity_operations_are_stateless():
    """Verify entities don't cache state between calls."""
    state1 = GameState(num_players=3)
    state2 = GameState(num_players=3)

    player = PLAYERS[0]
    player.initialize(state1)
    # No need to initialize on state2 - entity is stateless

    player.add_cash(state1, 100)
    # state2 unaffected
    assert player.get_cash(state2) != player.get_cash(state1)
```

## Cython Testing Considerations

**Build Required:**
- All `.pyx` files must be compiled to `.so` before importing in tests
- `python setup.py build_ext --inplace` required before `pytest`
- Compiler directives (`boundscheck=False`, etc.) remove Python safety checks
  - Tests should verify safety assumptions (e.g., valid indices)

**Performance Testing:**
- Benchmark command measures overall throughput, not individual functions
- For function-level profiling: Use cProfile or external tools
- Game engine designed for high throughput (AlphaZero training); tests should verify this

**Type Testing:**
- Cython provides compile-time type safety
- Runtime type validation tests not needed (caught at compile time)
- Focus on logic validation rather than type validation

---

*Testing analysis: 2026-01-20*

**Summary:** Tests have been deleted (commit 315a129). Rebuild test suite using patterns above. Core infrastructure (pytest, benchmark command) remains in place. Focus on GameState initialization, entity accessor methods, and phase transitions when rebuilding.
