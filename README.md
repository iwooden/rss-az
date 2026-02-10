# Rolling Stock Stars - Cython Game Engine

High-performance Cython game engine for the "Rolling Stock Stars" board game, designed for AlphaZero-style self-play training.

## Overview

This engine implements the complete rules of Rolling Stock Stars, a financial strategy board game where players buy, sell, and manage railroad companies and their shares. The codebase is optimized for machine learning training loops:

- **Single float32 array state**: Game state stored as a contiguous numpy array (~3500-3800 floats depending on player count) that can be passed directly to PyTorch without serialization
- **Dynamic player support**: 2-6 players with state size that scales appropriately
- **Compact action space**: 186 + (num_players × 20) discrete actions
- **No-GIL execution**: Performance-critical paths release the GIL for parallel training

## Project Status

**Game engine: Complete.** All 11 game phases are implemented and tested:

| Phase | Type | Description |
|-------|------|-------------|
| INVEST | Player choice | Buy/sell shares, start company auctions |
| BID_IN_AUCTION | Player choice | Competitive bidding for companies |
| WRAP_UP | Automated | Foreign Investor purchases at face value |
| ACQUISITION | Hybrid | Corporations acquiring companies |
| CLOSING | Hybrid | Company closures (voluntary and mandatory) |
| INCOME | Automated | Revenue distribution |
| DIVIDENDS | Player choice | Dividend payout decisions |
| END_CARD | Automated | Game-end trigger |
| ISSUE_SHARES | Player choice | Corporation share issuance |
| IPO | Player choice | Company to Corporation conversion |
| GAME_OVER | Terminal | Final scoring |

**Training code: Not yet implemented.** This repo contains only the game simulation engine. AlphaZero training infrastructure will be developed separately.

## Directory Structure

```
rss-az-cython2/
├── core/           # Low-level game engine
│   ├── state.pyx   # GameState: the central float32 array
│   ├── driver.pyx  # GameDriver: action dispatch and game loop
│   ├── actions.pyx # Action space layout and decoding
│   └── data.pyx    # Static game constants (companies, corps, prices)
│
├── entities/       # Entity handles for state access
│   ├── player.pyx  # Player cash, shares, companies
│   ├── corp.pyx    # Corporation state (IPO'd companies)
│   ├── company.pyx # Company state (auction deck items)
│   ├── deck.pyx    # Company deck management
│   ├── turn.pyx    # Turn and phase tracking
│   ├── market.pyx  # Share price market spaces
│   ├── fi.pyx      # Foreign Investor entity
│   └── encoding.pyx # One-hot encoding utilities
│
├── phases/         # Game phase handlers
│   ├── invest.pyx  # Investment phase logic
│   ├── bid.pyx     # Auction bidding
│   ├── acquisition.pyx # Company acquisition offers
│   ├── closing.pyx # Company closure logic
│   ├── dividends.pyx # Dividend calculations
│   ├── income.pyx  # Income distribution
│   ├── issue.pyx   # Share issuance
│   ├── ipo.pyx     # IPO conversions
│   ├── wrap_up.pyx # Turn wrap-up (FI buying)
│   └── end_card.pyx # Game-end handling
│
├── tests/          # Test suite
│   ├── phases/     # Phase-specific tests
│   └── conftest.py # Pytest fixtures
│
├── RULES.md        # Complete game rules (authoritative)
├── VECTORS.md      # State/action vector documentation
└── RSS.pdf         # Original board game rulebook
```

## Requirements

- Python 3.12+
- Cython 3.x
- NumPy 2.x
- pytest (for testing)

## Building

```bash
# Build Cython extensions
python setup.py build_ext --inplace

# Run tests
pytest tests/

# Clean build artifacts
python setup.py clean
```

## Usage

```python
from core.state import GameState
from core.driver import GameDriver, STATUS_OK, STATUS_GAME_OVER
from core.actions import get_action_mask

# Initialize a 3-player game
state = GameState(num_players=3, seed=42)

# Get legal actions as a boolean mask
mask = get_action_mask(state)

# Apply an action (returns status code)
status = GameDriver.apply_action(state, action_idx)

if status == STATUS_GAME_OVER:
    # Game finished - compute final scores
    pass
```

## Architecture Notes

### Entity Handles Pattern

Global singleton instances provide typed access to state array regions:

```python
from entities.player import PLAYERS
from entities.corp import CORPS

# Read player 0's cash
cash = PLAYERS[0].get_cash(state)

# Read corporation 2's share price
price = CORPS[2].get_share_price(state)
```

### State Layout

The state array has two regions:
- **Visible state**: Fed to the neural network (player-rotated so active player is always first)
- **Hidden state**: Internal bookkeeping (deck order, offer buffers, canonical indices)

See `VECTORS.md` for exact field offsets.

### Normalization

All values are normalized for neural network consumption:
- Cash/prices: divided by 200
- Share counts: divided by 7
- Star ratings: divided by 20

## Documentation

- **RULES.md**: Complete game rules - the authoritative source for how the game works
- **VECTORS.md**: Detailed state and action vector layouts with byte offsets
- **RSS.pdf**: Original board game rulebook for reference

## License

This project implements the rules of Rolling Stock Stars for research purposes.
