# 18xx.games Rolling Stock Stars JSON Schema

## Overview

This document describes the JSON format exported from 18xx.games for Rolling Stock Stars games. The JSON contains complete game state and action history that can be used to replay the game.

## Top-Level Structure

```json
{
    "id": 232836,                    // Game ID on 18xx.games
    "description": "Game description",
    "user": { "id": 2465, "name": "reveler" },  // Game creator
    "players": [...],               // Player list in seat order
    "min_players": 4,
    "max_players": 4,
    "title": "Rolling Stock Stars",
    "settings": {
        "seed": 849991845,          // Random seed for deck order
        "is_async": true,
        "optional_rules": []
    },
    "status": "finished",
    "turn": 14,                     // Final turn number
    "round": "Investment",          // Final phase
    "acting": [2465],               // Currently acting player(s)
    "result": {                     // Final net worth by player ID
        "1377": 206,
        "21841": 106,
        "1420": 149,
        "2465": 151
    },
    "actions": [...],               // Action history
    "loaded": true,
    "created_at": 1763859670,       // Unix timestamp
    "updated_at": 1764943572,
    "finished_at": 1764943572
}
```

## Player Object

```json
{
    "id": 21841,           // Unique player ID (numeric)
    "name": "CardboardBits" // Display name
}
```

Player seat order in `players[]` array determines initial turn order.

## Actions Array

The `actions` array contains all game actions in chronological order. Each action has a unique `id` field that increments.

### Common Action Fields

All actions share these fields:
```json
{
    "type": "action_type",      // Action type string
    "entity": ...,              // Acting entity (player ID or entity name)
    "entity_type": "player"|"company"|"corporation",
    "id": 1,                    // Sequential action ID
    "user": 21841,              // Player ID who submitted action
    "created_at": 1763909209    // Unix timestamp
}
```

### Action Types

#### 1. `bid` - Start Auction / Raise Bid

**Start auction (INVEST phase):**
```json
{
    "type": "bid",
    "price": 8,                 // Bid amount
    "entity": 21841,            // Player ID
    "company": "MHE",           // Company being auctioned
    "entity_type": "player"
}
```

**Raise bid (BID_IN_AUCTION phase):**
```json
{
    "type": "bid",
    "price": 9,                 // New bid amount
    "entity": 1377,             // Player ID
    "company": "MHE",           // Company being bid on
    "entity_type": "player"
}
```

#### 2. `pass` - Pass Action

Used in multiple contexts:
- Pass on auction (during bidding)
- End investment turn
- Pass in acquisition phase
- Pass in closing phase
- Pass IPO opportunity
- Pass issue share

**Simple pass:**
```json
{
    "type": "pass",
    "entity": 2465,
    "entity_type": "player"
}
```

**Pass with auto_actions (multiple players auto-passing):**
```json
{
    "type": "pass",
    "entity": 2465,
    "entity_type": "player",
    "auto_actions": [
        {"type": "pass", "entity": 1377, "entity_type": "player", "created_at": ...},
        {"type": "pass", "entity": 1420, "entity_type": "player", "created_at": ...}
    ]
}
```

**Company/Corp passing IPO:**
```json
{
    "type": "pass",
    "entity": "MHE",            // Company name
    "entity_type": "company"
}
```

#### 3. `buy_shares` - Buy Share from Market

```json
{
    "type": "buy_shares",
    "entity": 21841,
    "shares": ["SI_1"],         // Share identifier: CORP_INDEX
    "percent": 10,              // Always 10 for this game
    "entity_type": "player",
    "share_price": 11           // Price paid
}
```

#### 4. `sell_shares` - Sell Share to Market

```json
{
    "type": "sell_shares",
    "entity": 1377,
    "shares": ["SI_0"],         // Share being sold
    "percent": 10,
    "entity_type": "player",
    "share_price": 11           // Price received
}
```

**Corporation issuing share:**
```json
{
    "type": "sell_shares",
    "entity": "DA",             // Corporation name
    "shares": ["DA_3"],
    "percent": 10,
    "entity_type": "corporation",
    "share_price": 20
}
```

#### 5. `par` - IPO (Company converts to Corporation)

```json
{
    "type": "par",
    "entity": "BSE",            // Converting company
    "corporation": "SI",        // Target corporation
    "entity_type": "company",
    "share_price": "10,0,6"     // "price,x,y" format (price index encoded)
}
```

The `share_price` field uses format `"price,row,col"` where price is the par price.

#### 6. `dividend` - Pay Dividend

```json
{
    "kind": "variable",         // Always "variable" for RSS
    "type": "dividend",
    "amount": 3,                // Per-share dividend amount
    "entity": "SI",             // Corporation name
    "entity_type": "corporation"
}
```

#### 7. `offer` - Offer Acquisition

```json
{
    "type": "offer",
    "price": 16,                // Offered price
    "entity": 1377,             // Offering player (president)
    "company": "BY",            // Company being acquired
    "corporation": "SI",        // Acquiring corporation
    "entity_type": "player"
}
```

#### 8. `respond` - Accept/Reject Acquisition Offer

```json
{
    "type": "respond",
    "accept": "false",          // "true" or "false" (string!)
    "entity": 1377,             // Responding player
    "company": "PR",            // Company in question
    "corporation": "PR",        // Acquiring corporation
    "entity_type": "player"
}
```

#### 9. `sell_company` - Close Company (CLOSING phase)

```json
{
    "type": "sell_company",
    "price": 0,                 // Always 0 for closings
    "entity": 21841,            // Acting player
    "company": "AKE",           // Company being closed
    "entity_type": "player"
}
```

Note: The 18xx name "sell_company" corresponds to our CLOSING phase's close action.

#### 10. `undo` / `redo` - State Changes

```json
{"type": "undo", "entity": 2465, "entity_type": "player", ...}
{"type": "redo", "entity": 1420, "entity_type": "player", ...}
```

**IMPORTANT:** These actions should be filtered out during replay - they represent UI corrections, not actual game moves.

#### 11. `program_share_pass` - Auto-Pass Setting

```json
{
    "type": "program_share_pass",
    "entity": 21841,
    "indefinite": false,
    "entity_type": "player",
    "unconditional": true
}
```

This sets up auto-passing for a player. The test harness needs to track auto-pass state.

## Entity Naming Conventions

### Player IDs
- Numeric IDs: `21841`, `1377`, `1420`, `2465`
- Map to names via `players[]` array

### Company Names (36 total)
Red (stars=1): `BME`, `BSE`, `KME`, `AKE`, `BPM`, `MHE`
Orange (stars=2): `WT`, `BY`, `BD`, `HE`, `OL`, `SX`, `MS`, `PR`
Yellow (stars=3): `DSB`, `KK`, `NS`, `SBB`, `B`, `PKP`, `SNCF`, `DR`
Green (stars=4): `SZD`, `SJ`, `FS`, `RENFE`, `BR`, `BSR`, `E`
Blue (stars=5): `HH`, `HA`, `HR`, `MAD`, `FRA`, `LHR`, `CDG`

### Corporation Names (8 total)
`JS`, `S`, `OS`, `SM`, `PR`, `DA`, `VM`, `SI`

### Share Identifiers
Format: `CORP_INDEX` (e.g., `SI_1`, `DA_3`)
- Index appears to be share number within the corporation

## Deck Order Reconstruction

The deck order cannot be directly determined from the seed (different algorithm). Instead:

1. **Initial auction companies**: Listed at game start (from `.md` file: BME, BSE, MHE, BPM)
2. **Subsequent reveals**: Parse game log for "revealed from deck" messages
3. **Order**: AKE, BD, BY, MS, OL, SX, PR, DSB, PKP, KK, B, DR, SJ, RENFE, FS, BR, E, HH, MAD, FRA, LHR, CDG (from game log)

## Key Translation Notes (from DIFFERENCES.md)

1. **Auction slot translation**: 18xx lets players bid on any company; our engine uses ordered slots by face value
2. **Acquisition ordering**: 18xx allows arbitrary order; our engine presents one acquisition at a time
3. **Closing ordering**: Similar to acquisitions - one at a time in our engine
4. **Auto-actions**: 18xx auto-passes single-choice situations; verify our engine does the same
5. **Auto-pass state**: Track `program_share_pass` to handle automatic passes
