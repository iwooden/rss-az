# Cython Core Vector Documentation

This document describes the in-memory game state layout. The state is the engine's only authoritative game data; per-token features for the neural network are produced lazily by `get_token_data()` (a separate path).

---

## State Buffer

`GameState` wraps a single contiguous `int16` numpy array. All values are raw signed integers — no normalization, no one-hot encoding, no visible/hidden split. `int16` is sufficient for every game quantity (player net worth maxes around 400, share counts are single digits, `-1` sentinels fit in the negative range).

### Sizes by player count

| Players | total_size | player_stride | corp_stride | turn_size |
|---------|-----------|---------------|-------------|-----------|
| 2       | 444       | 39            | 16          | 59        |
| 3       | 483       | 39            | 16          | 59        |
| 4       | 522       | 39            | 16          | 59        |
| 5       | 561       | 39            | 16          | 59        |
| 6       | 600       | 39            | 16          | 59        |

`player_stride`, `corp_stride`, and `turn_size` are all fixed across player counts. The players section is the **only** part of the buffer whose size depends on `num_players`, so `total_size = 366 + 39 * num_players` — the constant 366 is the fixed prefix, and only the trailing players section grows.

Layout offsets are computed once at module load and exposed as Cython `cdef` structs at module scope on `core.state`:

- `LAYOUT` (`cdef StateLayout`) — top-level section offsets and strides. `total_size` is intentionally not in this struct because it depends on `num_players`; compute it inline as `LAYOUT.players_offset + LAYOUT.player_stride * num_players` at the few sites that need it.
- `TURN_OFFSETS` (`cdef TurnStateOffsets`) — relative offsets within the (fixed-size) turn block.
- `PLAYER_FIELDS` (`cdef PlayerFieldOffsets`) — relative offsets within a player block.
- `CORP_FIELDS` (`cdef CorpFieldOffsets`) — relative offsets within a corp block.

Cython code reads them directly via `from core.state cimport LAYOUT, TURN_OFFSETS, PLAYER_FIELDS, CORP_FIELDS`. Python code uses the namedtuple accessors `core.state.get_layout(num_players)`, `get_player_fields()`, `get_corp_fields()`, `get_turn_fields()` (none of the field accessors take a `num_players` argument since the fields are fixed-size).

---

## Top-level Layout

| Section | Start offset (3p) | Size | Description |
|---------|------------------:|------|-------------|
| Metadata          | 0   | 5   | active_player, num_players, phase, coo_level, turn_number |
| FI                | 5   | 2   | Foreign investor cash, income |
| Company incomes   | 7   | 36  | Per-company adjusted income (base − CoO cost) |
| Market            | 43  | 27  | Per-price availability flags |
| Corps             | 70  | 128 | Per-corp blocks: `corp_stride (16) * 8` (see [Corp block](#corp-block)) |
| Turn              | 198 | 59  | Turn-scoped state (see [Turn block](#turn-block)) |
| Deck              | 257 | 37  | `deck_top` index (1) + `deck_order` (36) |
| Company tracking  | 294 | 72  | `company_locations` (36) + `company_owner_ids` (36) |
| Players           | 366 | `player_stride * num_players` | Per-player blocks (see [Player block](#player-block)) |

Every offset above is **constant across all player counts** — the players section lives at the end of the buffer for exactly this reason. The "Start offset (3p)" column also matches every other player count up to and including the players section start.

Section start offsets are exposed via `LayoutInfo` as `fi_offset`, `company_incomes_offset`, `market_offset`, `corps_offset`, `turn_offset`, `deck_top_offset`, `deck_order_offset`, `company_locations_offset`, `company_owner_ids_offset`, and finally `players_offset`.

---

## Metadata (offsets 0–4)

| Offset | Field | Notes |
|--------|-------|-------|
| 0 | active_player | Canonical player_id |
| 1 | num_players | 2–6 |
| 2 | phase | 0–11, see [Phase enum](#phase-enum) |
| 3 | coo_level | 1–7 |
| 4 | turn_number | 1+ |

---

## Player block

Stride: **39**. Player `i` lives at `players_offset + i * 39`. Field offsets via `core.state.get_player_fields()` (`PlayerFields` namedtuple) for Python, or `from core.state cimport PLAYER_FIELDS` for Cython.

| Relative offset | Field | Size | Notes |
|----------------|-------|------|-------|
| 0  | cash            | 1 | |
| 1  | net_worth       | 1 | |
| 2  | liquidity       | 1 | Iterative share liquidation value (cash + value of shares + companies) |
| 3  | turn_order      | 1 | Position 0 = first to act |
| 4  | owned_shares    | 8 | Per-corp share count |
| 12 | is_president    | 8 | Per-corp president flag |
| 20 | round_trips     | 1 | `max(min(buys[c], sells[c]) for c in corps)` this turn |
| 21 | income          | 1 | Income from owned private companies |
| 22 | share_buys      | 8 | Per-corp buy counts (this turn) |
| 30 | share_sells     | 8 | Per-corp sell counts (this turn) |
| 38 | auction_passed  | 1 | `1` once this player has left the current auction |

All per-player tracking lives inside one player block, so `_player_ptr(i)` reaches everything for player `i` in a single pointer hop. The `auction_passed` flag previously lived in the turn block as a per-player array; moving it into the player block makes the player block fully self-contained and the turn block fixed-size.

---

## Foreign Investor (size 2)

| Relative offset | Field |
|----------------|-------|
| 0 | cash |
| 1 | income |

FI ownership of companies is tracked via `company_locations` (`LOC_FI`), not via flags on the FI block.

---

## Company adjusted incomes (size 36)

Per-company income after CoO is applied (`base_income − coo_cost(coo_level)`). Recomputed when the CoO level changes.

---

## Market availability (size 27)

One flag per market price slot at `market_offset + i`. `1` = available, `0` = claimed by a corp.

| Index → price (slot index → dollar value) |
|---|
| 0→0, 1→5, 2→6, 3→7, 4→8, 5→9, 6→10, 7→11, 8→12, 9→13, 10→14, 11→16, 12→18, 13→20, 14→22, 15→24, 16→27, 17→30, 18→33, 19→37, 20→41, 21→45, 22→50, 23→55, 24→61, 25→68, 26→75 |

---

## Corp block

Stride: **16**. Corp `c` lives at `corps_offset + c * 16`. Field offsets via `core.state.get_corp_fields()` (`CorpFields` namedtuple) for Python, or `from core.state cimport CORP_FIELDS` for Cython.

| Relative offset | Field | Notes |
|----------------|-------|-------|
| 0  | active                | `1` once floated |
| 1  | cash                  | |
| 2  | unissued_shares       | |
| 3  | issued_shares         | |
| 4  | bank_shares           | |
| 5  | income                | `raw_revenue + synergy_income − coo_cost + ability_income` |
| 6  | stars                 | Aggregate star total across owned companies |
| 7  | share_price           | Cached from `price_index` |
| 8  | acquisition_proceeds  | Pending payment, written and consumed during ACQ |
| 9  | in_receivership       | flag |
| 10 | price_index           | Market position 0–26 |
| 11 | pending_price_move    | Predicted index delta assuming $0 dividend |
| 12 | raw_revenue           | Sum of base company incomes |
| 13 | synergy_income        | |
| 14 | coo_cost              | Always ≤ 0 |
| 15 | ability_income        | |

---

## Turn block

Block size: **59**, fixed across player counts. Sub-offsets via `core.state.get_turn_fields()` (`TurnFields` namedtuple) for Python, or `from core.state cimport TURN_OFFSETS` for Cython.

| Relative offset | Field | Size | Notes |
|----------------|-------|------|-------|
| 0  | end_card_flipped     | 1  | flag |
| 1  | consecutive_passes   | 1  | INVEST pass counter; phase ends when this reaches `num_players` |
| 2  | cards_remaining      | 1  | |
| 3  | auction_price        | 1  | 0 when no auction |
| 4  | auction_company      | 1  | `company_id` or `-1` |
| 5  | auction_high_bidder  | 1  | `player_id` or `-1` |
| 6  | auction_starter      | 1  | `player_id` or `-1` |
| 7  | dividend_remaining   | 8  | Per-corp pending flag |
| 15 | issue_remaining      | 8  | Per-corp pending flag |
| 23 | ipo_remaining        | 36 | Per-company pending flag |

The per-player `auction_passed` flag used to live here and made the turn block scale with player count. It now lives in the player block, and the turn block is fully fixed-size.

---

## Deck

| Offset | Field | Size | Notes |
|--------|-------|------|-------|
| `deck_top_offset`   | deck_top   | 1  | Index of next card to draw; `-1` = empty |
| `deck_order_offset` | deck_order | 36 | Shuffled company IDs |

Companies excluded for the current player count are placed outside the active draw range and marked `LOC_EXCLUDED` in `company_locations`.

---

## Company tracking

Two parallel 36-element arrays.

### `company_locations` (size 36)

One `CompanyLocation` enum value per company.

| Value | Name          | Owner field meaning |
|-------|---------------|---------------------|
| 0 | LOC_DECK      | none (`owner_id = -1`) |
| 1 | LOC_AUCTION   | none |
| 2 | LOC_REVEALED  | none |
| 3 | LOC_PLAYER    | `player_id` |
| 4 | LOC_FI        | none |
| 5 | LOC_CORP      | `corp_id` |
| 6 | LOC_CORP_ACQ  | `corp_id` (in corp's acquisition pile) |
| 7 | LOC_REMOVED   | none — closed during play |
| 8 | LOC_EXCLUDED  | none — never dealt for this player count |

`LOC_REMOVED` and `LOC_EXCLUDED` are kept distinct so excluded companies do not leak deck composition.

### `company_owner_ids` (size 36)

Owner ID per company. `-1` when the location has no meaningful owner. Initialized to `-1` for all companies in `__cinit__` (zero-init would otherwise mark every company as owned by player 0).

---

## Phase enum

Stored as a raw integer in `phase_offset`. Defined in `core/data.pxd` (`GamePhases`).

| Value | Name |
|-------|------|
| 0  | PHASE_INVEST |
| 1  | PHASE_BID_IN_AUCTION |
| 2  | PHASE_WRAP_UP |
| 3  | PHASE_ACQUISITION |
| 4  | PHASE_CLOSING |
| 5  | PHASE_INCOME |
| 6  | PHASE_DIVIDENDS |
| 7  | PHASE_END_CARD |
| 8  | PHASE_ISSUE_SHARES |
| 9  | PHASE_IPO |
| 10 | PHASE_PAR |
| 11 | PHASE_GAME_OVER |

---

## Pass tracking

| Phase | Mechanism | Location |
|-------|-----------|----------|
| INVEST            | `consecutive_passes` counter | turn block |
| BID_IN_AUCTION    | per-player `auction_passed` flag | player block |
| CLOSING / ACQUISITION | per-offer accept/pass via offer surface | (engine, not state) |

---

## Construction surface

| Method | Behavior |
|--------|----------|
| `GameState(num_players)` | Allocate a fresh zero-initialized buffer; seed `num_players` and `company_owner_ids` to `-1`. |
| `GameState.from_array(arr, num_players)` | Allocate and copy `arr` into the new state. |
| `GameState.from_buffer(buf, num_players)` | Wrap an existing C-contiguous int16 buffer zero-copy. Buffer must already contain valid state — does **not** seed `company_owner_ids`. |
| `state.rebind(buf)` | Repoint an existing `GameState` at a different backing buffer. Used in MCTS hot paths. |
| `state.initialize_game(seed=-1)` | Set up players, FI, corps, market, deck, turn state, and active player. `seed=-1` uses current time. |

`GameState` exposes only structural primitives publicly: `_player_ptr` / `_corp_ptr` / `_turn_ptr` (cdef nogil, used by entity handles), `_num_players` (cdef int, readable from cdef code for assertions), `get_active_player` / `set_active_player`, `get_num_players`, and `initialize_game`. There are no per-instance layout-offset fields — every entity handle reads offsets directly from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` constants on `core.state`. All field-level reads and writes go through the entity handles in `entities/`.

---

## Action Vector

Action space size varies by player count:

| Players | Auction Actions | Total Actions |
|---------|-----------------|---------------|
| 3       | 45 (3 x 15)     | 183           |
| 4       | 60 (4 x 15)     | 198           |
| 5       | 75 (5 x 15)     | 213           |
| 6       | 90 (6 x 15)     | 228           |

Formula: `138 + num_players * 15`

Use `get_total_action_count(num_players)` for the exact size.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `AUCTION_CAP` | 15 | Max bid offset over face value |
| `MAX_PAR_SLOTS` | 8 | Max valid par prices per star tier (legacy) |
| `NUM_PAR_PRICES` | 14 | Total par price count (used by PAR phase) |
| `ACQ_PRICE_RANGE` | 51 | Price offsets 0-50 |
| `MAX_DIVIDEND` | 26 | Dividend amounts 0-25 |
| `NUM_CORPS` | 8 | Corporation count |
| `NUM_COMPANIES` | 36 | Company count |
| `NUM_PAR_PRICES` | 14 | Valid par price count |

### Action Layout by Phase (for 3 players)

| Phase | Actions | Count | Indices |
|-------|---------|-------|---------|
| **INVEST** | pass, auction[3x15], buy[8], sell[8] | 62 | 0-61 |
| **BID_IN_AUCTION** | leave, raise_bid[14] | 15 | 62-76 |
| **ACQUISITION** | price[51], fi_buy, pass | 53 | 77-129 |
| **CLOSING** | close, pass | 2 | 130-131 |
| **DIVIDENDS** | dividend[26] | 26 | 132-157 |
| **ISSUE_SHARES** | pass, issue | 2 | 158-159 |
| **IPO** | pass, corp[8] | 9 | 160-168 |
| **PAR** | par[14] | 14 | 169-182 |

### Detailed Action Indices (for N players)

#### INVEST Phase (0 to 16 + N*15)

| Index | Action | Decoding |
|-------|--------|----------|
| 0 | Pass | - |
| 1 to N*15 | Auction | `slot = (idx-1) // 15`, `bid_offset = (idx-1) % 15` |
| N*15+1 to N*15+8 | Buy Share | `corp_id = idx - (N*15+1)` |
| N*15+9 to N*15+16 | Sell Share | `corp_id = idx - (N*15+9)` |

**Auction slot mapping:** Slot N maps to the Nth available-for-auction company (ordered by company_id). Use `get_auction_company_for_slot(state, slot)` to resolve.

#### BID_IN_AUCTION Phase (+15 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 | Leave Auction | - |
| +1 to +14 | Raise Bid | `bid_offset = idx - base - 1` (new bid = face + offset + 1) |

#### ACQUISITION Phase (+53 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 to +50 | Price Offer | `price = low_price + (idx - base)` |
| +51 | FI Buy | Buy FI company (OS=face value, others=high price) |
| +52 | Pass | Decline acquisition |

#### CLOSING Phase (+2 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 | Close | Close current company |
| +1 | Pass | Keep current company |

#### DIVIDENDS Phase (+26 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 to +25 | Pay Dividend | `amount = idx - base` (per share) |

#### ISSUE_SHARES Phase (+2 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 | Pass | Don't issue |
| +1 | Issue | Issue one share |

#### IPO Phase (+9 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 | Pass | Don't IPO |
| +1 to +8 | Select Corp | `corp_id = idx - base - 1` |

Selecting a corp transitions to the PAR sub-phase.

#### PAR Phase (+14 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 to +13 | Select Par Price | `par_index = idx - base` |

No pass action — once a corp is selected, a par price must be chosen.
Par index maps directly to `ALL_PAR_PRICES[par_index]`. Invalid prices for the company's star tier are masked out.

### Action Types Enum

| Value | Type | Description |
|-------|------|-------------|
| 0 | ACTION_PASS | Pass/decline |
| 1 | ACTION_AUCTION | Start auction (slot, bid_offset) |
| 2 | ACTION_BUY_SHARE | Buy share (corp_id) |
| 3 | ACTION_SELL_SHARE | Sell share (corp_id) |
| 4 | ACTION_LEAVE_AUCTION | Leave auction |
| 5 | ACTION_RAISE_BID | Raise bid (bid_offset) |
| 6 | ACTION_ACQ_PRICE | Acquire at price (price_offset) |
| 7 | ACTION_ACQ_FI_BUY | FI buy (OS=face, others=high) |
| 8 | ACTION_CLOSE | Close current company |
| 9 | ACTION_DIVIDEND | Pay dividend (amount) |
| 10 | ACTION_ISSUE | Issue share |
| 11 | ACTION_IPO | IPO: select corp (corp_id) |
| 12 | ACTION_PAR | PAR: select par price (par_index) |

---

## Usage Examples

### Python Access

```python
from core.state import GameState, get_layout, get_player_fields
from core.actions import get_valid_action_mask, get_total_action_count
from entities.player import PLAYERS

# State buffer
state = GameState(num_players=3)
state.initialize_game(seed=42)
print(f"buffer length = {len(state._array)}")  # 483 for 3p

# Layout introspection
layout = get_layout(3)            # LayoutInfo namedtuple
print(layout.players_offset)      # 366 (constant across player counts)
print(layout.total_size)          # 483

pf = get_player_fields()          # PlayerFields namedtuple
print(pf.cash, pf.auction_passed) # 0 38

# Field access goes through entity handles, not raw buffer indexing
print(PLAYERS[0].get_cash(state))

# Action vector
total_actions = get_total_action_count(3)
mask = get_valid_action_mask(state)
valid_actions = mask.nonzero()[0]
```

### Cython Access

```cython
from core.state cimport GameState, LAYOUT, PLAYER_FIELDS, TURN_OFFSETS
from libc.stdint cimport int16_t

cdef GameState state = GameState(3)

# Read a player field directly via the module-level constants
cdef int16_t cash = state._player_ptr(0)[PLAYER_FIELDS.cash]

# Read a turn-block field
cdef int phase = state._data[LAYOUT.phase_offset]
cdef int auction_price = state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price]
```

In practice, every field read/write should go through an entity handle (`PLAYERS[i].get_cash(state)`, `TURN.get_auction_price(state)`, etc.) — the snippets above show what the handles do under the hood.
