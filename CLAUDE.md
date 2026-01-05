# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Cython game engine for "Rolling Stock Stars" board game, optimized for AlphaZero-style self-play training. Game state is stored as a single contiguous float32 array that can be passed directly to PyTorch without serialization overhead.

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

# Benchmark: measure games per minute (requires build first)
python setup.py benchmark                          # 1000 games, 3 players
python setup.py benchmark --num-games=5000 --num-players=6
```

## Architecture

### State Representation
Game state is a contiguous `float32` NumPy array with two sections:
- **Visible state** (~1650-2050 elements, varies by player count): Neural network input
- **Hidden state** (52 elements): Internal tracking (deck order, compact indices)

Size formula: Use `get_state_size(num_players)` and `get_visible_size(num_players)`.

### Action Space
Action count: `186 + (num_players × 20)` (246 for 3 players, 306 for 6 players)

### Core Modules
- **state.pyx**: `GameState` class, array offset calculations, state manipulation
- **actions.pyx**: Action encoding/decoding, valid action mask generation
- **driver.pyx**: `GameDriver` orchestrates action application and phase transitions
- **helpers/**: Cython utilities for player, corp, market, company, turn state access
- **phases/**: Phase implementations (invest, acquisition, closing, income, dividends, endcard, issue, ipo, wrapup)

### Game Phases (11 total)
0: INVEST, 1: BID_IN_AUCTION, 2: WRAP_UP, 3: ACQUISITION, 4: CLOSING, 5: INCOME, 6: DIVIDENDS, 7: END_CARD, 8: ISSUE_SHARES, 9: IPO, 10: GAME_OVER

### Key Constants
| Constant | Value | Used For |
|----------|-------|----------|
| CASH_DIVISOR | 200.0 | Cash, prices, net worth normalization |
| SHARE_DIVISOR | 7.0 | Share count normalization |
| STAR_DIVISOR | 20.0 | Star ratings normalization |
| INCOME_DIVISOR | 10.0 | Company income normalization |

## Testing

Tests use `StateBuilder` from `tests/test_common.py` to construct specific game states. Key test files:
- `test_random_game.py`: Full game simulation with invariant checking
- `test_<phase>.py`: Per-phase rule validation (invest, acquisition, closing, income, dividends, issue, ipo, wrapup, endcard)
- `test_actions_cython.py`: Action encoding/decoding correctness

## Documentation

- **VECTORS.md**: Complete state/action vector layout, offsets, and encoding schemes
- **RULES.md**: Official Rolling Stock Stars game rules

## Cython Development Notes

- All `.pyx` files have corresponding `.pxd` declaration files
- Compiler directives optimize for performance: `boundscheck=False, wraparound=False, cdivision=True`
- Build generates `.html` annotation files showing Python/C interaction
- After modifying `.pyx` files, rebuild with `python setup.py build_ext --inplace`

## Development Guidelines

### Code Duplication
Never duplicate code or constants that are already defined in a central file. If the same logic is needed in multiple places, add it to one of the `helpers/` modules, `state.pyx`, or another appropriate shared location. Constants should be defined once and imported where needed.

If you encounter existing duplicated code or constants, stop your current task and refactor immediately: remove the duplicates and update all usages to reference the single centralized definition. Do not proceed with other work until the duplication is eliminated.

### Test Failures
When a test fails, never make the test more permissive, skip it, or otherwise weaken assertions to make it pass. Tests exist to verify the engine implements the rules correctly. When fixing test failures:
1. Read `RULES.md` to understand the correct game behavior
2. Modify the engine code to follow those rules exactly
3. If the fix causes other tests to fail, fix those as well

The primary objective is a game engine that implements the rules with complete accuracy. Performance is a strong secondary concern but must never compromise correctness.

### Debugging

Use the built-in debug infrastructure rather than writing long inline Python scripts to debug tests. There are two ways to enable debug output:

**1. GameDriver debug mode** (for detailed action history):
```python
from driver import GameDriver

driver = GameDriver(3)
driver.enable_debug()  # Start recording

# ... run game actions ...

print(driver.dump_history())  # Print formatted action history
driver.get_history()          # Get raw history as list of dicts
driver.clear_history()        # Clear without disabling
driver.disable_debug()        # Stop recording
```

**2. Pytest --game-debug flag** (for test debugging):
```bash
# Enable debug output for all tests
pytest tests/test_invest.py -v --game-debug

# Or via environment variable
RSS_DEBUG=1 pytest tests/test_invest.py -v
```

**Adding debug output**: When you need additional debug information, add it directly to the engine or test code using `debug_print()` from `tests/conftest.py`:
```python
from conftest import debug_print

debug_print(f"Current phase: {state.phase}")  # Only prints when debug enabled
```

Do not write long standalone Python scripts to debug issues. Instead, add permanent debug instrumentation to the codebase that can be enabled via these flags.
