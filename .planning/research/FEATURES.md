# Feature Landscape: INVEST & BID_IN_AUCTION Phase Actions

**Domain:** Board game AI engine - action dispatch and state mutation
**Phases:** INVEST (phase 0), BID_IN_AUCTION (phase 1)
**Researched:** 2026-01-20

## Table Stakes

Features users expect. Missing = phase implementations won't function correctly.

### INVEST Phase Actions

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| **Pass action** | Core game mechanic - players must be able to pass | Low | Turn state consecutive_passes counter |
| **Consecutive pass tracking** | Phase ends when all players pass consecutively | Low | Turn state already has consecutive_passes field |
| **Pass counter reset on non-pass action** | Required by game rules | Low | Turn state |
| **Buy Share action** | Core investment mechanic | Medium | Corp bank shares, player cash, market price lookup |
| **Price movement on buy (up)** | Rolling Stock core rule - buying moves price up | Medium | Market module, next higher available space lookup |
| **Sell Share action** | Core investment mechanic | Medium | Player shares, market price lookup |
| **Price movement on sell (down)** | Rolling Stock core rule - selling moves price down | Medium | Market module, next lower available space lookup |
| **Start Auction action** | Core acquisition mechanic | Medium | Auction state, company availability, player cash |
| **Phase transition to BID_IN_AUCTION** | Auction start must change phase | Low | Turn state phase setter |
| **Active player rotation** | Must track current player in turn order | Low | Hidden state active_player field |
| **Phase end detection** | consecutive_passes >= num_players triggers end | Low | Turn state |

### BID_IN_AUCTION Phase Actions

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| **Leave Auction action** | Players must be able to exit auction | Low | auction_passed flags per player |
| **Raise Bid action** | Core auction mechanic | Low | Auction price, player cash |
| **Bid validation** | New bid must exceed current bid | Low | Auction state |
| **Active bidder rotation** | Skip players who have left | Medium | auction_passed flags, turn order |
| **Auction resolution** | One bidder remaining = winner | Medium | auction_passed flags, company transfer |
| **Winner pays bid price** | Transfer cash from winner to bank | Low | Player cash setter |
| **Company transfer to winner** | Winner receives company | Medium | Company location tracking |
| **Return to INVEST phase** | Auction completion returns to investment | Low | Phase setter |
| **Auction starter receives next turn** | Game rule - auction starter resumes | Low | Turn order tracking |

### State Mutation Mechanics

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| **Atomic state updates** | Single action = single consistent state | Medium | All entity setters |
| **Cash balance validation** | Cannot spend more than player has | Low | Player cash getter |
| **Share count validation** | Cannot sell more shares than owned | Low | Player shares getter |
| **Market space availability tracking** | Price movements require available spaces | Low | Market module already exists |

## Differentiators

Features that improve AI training quality. Not required but valuable.

### Share Trading Advanced Mechanics

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| **Corporation bankruptcy on price 0** | Required for full game - price dropping to 0 eliminates corp | High | Corp deactivation, share invalidation, company release |
| **Change of presidency** | Share majority changes must transfer control | High | All player share counts, president flag updates |
| **Receivership on all shares sold** | Corp with no player shareholders enters receivership | Medium | Issued shares vs bank shares tracking |
| **Round-trip buy/sell limits** | Prevents price manipulation - MAX_ROUNDTRIPS=2 per corp per turn | Medium | share_buys/share_sells tracking (already in state) |
| **Round-trip tracking reset** | Clear counters at start of player's turn | Low | Player entity method |
| **Net worth recalculation** | Update after share price changes | Medium | Player net worth calculation |
| **Skip unavailable market spaces** | Price movement skips occupied spaces | Medium | Market availability scan |

### Auction Advanced Mechanics

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| **Auction starter tracking** | Needed for turn resumption after auction | Low | auction_starter field (already exists) |
| **FI auction fallback** | If no players bid, FI gets company at face value | Medium | FI entity, company transfer |
| **Multiple companies for auction** | Support N companies available (N=player count) | Low | companies_for_auction flags (already exists) |
| **Company draw on auction** | Draw new company when auction completes | Medium | Deck module |

### AI Training Quality

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| **Forced action detection** | Optimize training by detecting single valid action | Low | Action mask analysis (already exists) |
| **Action mask efficiency** | O(1) mask generation for training speed | Medium | Action layout optimization |
| **State cloning for MCTS** | Support state copying for tree search | Medium | NumPy array copy |

## Anti-Features

Features to explicitly NOT build. Common mistakes in game engine development.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Undo/redo stack** | Adds complexity, not needed for AI training | MCTS clones state before simulation |
| **Event logging/history** | Performance overhead, not needed for training | Log only during debugging/validation |
| **UI state (animations, highlights)** | Engine is headless for training | Separate UI layer if needed later |
| **Human-readable action names in hot path** | String allocation overhead | Use numeric action types internally |
| **Defensive bounds checking in nogil code** | Performance critical - validated at entry | Validate at Python/Cython boundary |
| **Dynamic allocation during game loop** | Memory pressure during training | Pre-allocate all structures |
| **Complex inheritance hierarchies** | Cython performance suffers | Flat entity accessor pattern (already established) |
| **Action replay/playback** | Not needed for self-play training | Would add complexity without training benefit |
| **Multiplayer synchronization** | Single-machine training focus | Each game instance is self-contained |
| **Save/load to disk** | In-memory state sufficient for training | Can add later if needed for analysis |

## Feature Dependencies

```
INVEST Phase:
  Pass ─────────────────────────────────────> consecutive_passes increment
  Buy Share ─────> Price movement up ────────> Market space lookup
           └────> Player cash deduction       └─> Skip unavailable spaces
           └────> Corp bank shares decrement
           └────> Player shares increment
           └────> Round-trip tracking
           └────> Presidency check
           └────> Net worth update

  Sell Share ────> Price movement down ──────> Market space lookup
           └────> Player cash addition        └─> Skip unavailable spaces
           └────> Corp bank shares increment
           └────> Player shares decrement
           └────> Round-trip tracking
           └────> Bankruptcy check (price = 0)
           └────> Receivership check
           └────> Presidency change
           └────> Net worth update

  Start Auction ─> Phase change to BID_IN_AUCTION
            └───> Auction state initialization
            └───> Auction starter set
            └───> consecutive_passes cleared

BID_IN_AUCTION Phase:
  Leave Auction ─> auction_passed flag set
            └───> Active bidder rotation
            └───> Auction resolution check

  Raise Bid ─────> Auction price update
          └────> High bidder update
          └────> Active bidder rotation

  Auction Resolution:
            └───> Winner pays bid
            └───> Company transferred
            └───> Auction state cleared
            └───> Phase return to INVEST
            └───> Turn resumes with auction starter
```

## MVP Recommendation

For MVP, prioritize:

1. **All table stakes features** - These are required for the phase to function
2. **Bankruptcy handling** - Critical edge case that can occur early in games
3. **Presidency change** - Required when shares are traded
4. **Round-trip limits** - Prevents degenerate AI behavior during training

Defer to post-MVP:

- **State cloning optimization** - Basic NumPy copy is sufficient initially
- **FI auction fallback** - Edge case, can implement when testing full game loops
- **Net worth calculation** - Already has implementation, just needs to be called at right times

## Phase Implementation Order

Based on dependencies, implement in this order:

1. **Pass action** (simplest, unlocks testing)
2. **Consecutive pass tracking** (needed for phase end)
3. **Start Auction action** (unlocks BID_IN_AUCTION testing)
4. **Leave Auction action** (simplest auction action)
5. **Raise Bid action** (completes auction bidding)
6. **Auction resolution** (completes auction flow)
7. **Buy Share action** (complex state mutation)
8. **Sell Share action** (similar to buy, adds bankruptcy risk)
9. **Price movement mechanics** (requires market space scanning)
10. **Presidency change** (requires share counting across players)
11. **Bankruptcy handling** (requires full corp state cleanup)
12. **Round-trip limits** (enforcement layer)

## Sources

Research derived from:

- [Rolling Stock Stars How to Play](https://www.boardgameblitz.com/posts/280/how-to-play-rolling-stock-stars) - Official game rules
- [Daemon18xx GitHub](https://github.com/jmahmood/Daemon18xx) - 18XX rules engine implementation patterns
- [18XX with Ambie: Dumping Companies & Hostile Takeovers](https://www.boardgameblitz.com/posts/248/18xx-with-ambie-dumping-companies-hostile-takeovers) - Presidency change mechanics
- [Simple Alpha Zero](https://suragnair.github.io/posts/alphazero.html) - AlphaZero action dispatch patterns
- [Game Programming Patterns: State](https://gameprogrammingpatterns.com/state.html) - State machine design
- [BoardGameGeek: Turn Order Pass Order](https://boardgamegeek.com/boardgamemechanic/2830/turn-order-pass-order) - Consecutive pass mechanics
- Existing codebase: `/home/icebreaker/rss-az-cython2/core/actions.pyx` - Action mask generation patterns
- Existing codebase: `/home/icebreaker/rss-az-cython2/VECTORS.md` - State and action vector specifications

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| Table stakes features | HIGH | Derived from existing codebase, game rules, and action mask implementation |
| State mutation patterns | HIGH | Existing entity accessors provide clear patterns |
| Bankruptcy/receivership | MEDIUM | Game rules clear, implementation complexity uncertain |
| Round-trip limits | HIGH | State fields already exist, just need enforcement |
| AI training features | MEDIUM | Based on general AlphaZero patterns, not game-specific verification |
