# Cython Core Vector Documentation

This document describes the state vector (neural network input) and action vector (neural network output) layouts for the Cython game engine.

---

## State Vector

The game state is stored as a contiguous `float32` array organized as:
```
[VISIBLE STATE (for NN)] [HIDDEN STATE (internal only)]
```

### Size Calculation

State size varies by player count due to player-indexed arrays:

| Players | Visible Size | Hidden Size | Total Size |
|---------|--------------|-------------|------------|
| 2       | 1491         | 1184        | 2675       |
| 3       | 1577         | 1184        | 2761       |
| 4       | 1665         | 1184        | 2849       |
| 5       | 1755         | 1184        | 2939       |
| 6       | 1847         | 1184        | 3031       |

Use `get_state_size(num_players)` and `get_visible_size(num_players)` for exact values.

### Normalization Constants

| Constant | Value | Used For |
|----------|-------|----------|
| `CASH_DIVISOR` | 100.0 | Player/corp/FI cash, net worth, acquisition proceeds |
| `INCOME_DIVISOR` | 10.0 | Per-company adjusted incomes, synergy values, active company income, auction slot income |
| `PRICE_DIVISOR` | 40.0 | Company prices (face/low/high), share prices, auction price, entity incomes (player/corp/FI), buy/sell impacts |
| `SHARE_DIVISOR` | 7.0 | Share counts |
| `COMPANY_STAR_DIVISOR` | 5.0 | Per-company star ratings (1-5) |
| `CORP_STAR_DIVISOR` | 20.0 | Corporation aggregate star totals |
| `MAX_ROUNDTRIPS` | 2.0 | Round-trip limit (divisor = MAX_ROUNDTRIPS * 2 = 4.0) |
| `IMPACT_DIVISOR` | 5.0 | Dividend price index deltas |

### Context-Dependent Fields

Many fields in the visible state are only meaningful during specific game phases (e.g., auction info during BID, active corp during DIVIDENDS). When their context is inactive, these fields are **zeroed out** so the model can distinguish "no active context" from valid data.

**Convention:**
- **Binary fields** (one-hots, flags): all elements set to **0** when inactive. This is distinct from a valid one-hot (which has exactly one 1) or valid flags (which may have any combination of 1s).
- **Data scalars** (cash, prices, counts): set to **0** when inactive.

**Context-dependent fields and their relevant phases:**

| Field | Type | Relevant Phases |
|-------|------|-----------------|
| `auction_price` | scalar | BID |
| `auction_high_bidder` | one-hot | BID |
| `auction_starter` | one-hot | BID |
| `auction_passed` | flags | BID |
| `dividend_impact` | values | DIVIDENDS |
| `dividend_remaining` | flags | DIVIDENDS |
| `issue_remaining` | flags | ISSUE |
| `issue_price_impact` | scalar | ISSUE |
| `issue_cash_gain` | scalar | ISSUE |
| `ipo_remaining` | flags | IPO |
| `invest_buy_impact` | values | INVEST |
| `invest_sell_impact` | values | INVEST |
| `acq_is_fi_offer` | flag | ACQUISITION |
| `acq_synergy_values` | values | ACQUISITION |
| `active_company` | one-hot | BID, ACQ, CLOSING, IPO |
| `active_company_stars` | scalar | BID, ACQ, CLOSING, IPO |
| `active_company_low_price` | scalar | BID, ACQ, CLOSING, IPO |
| `active_company_face_value` | scalar | BID, ACQ, CLOSING, IPO |
| `active_company_high_price` | scalar | BID, ACQ, CLOSING, IPO |
| `active_company_income` | scalar | BID, ACQ, CLOSING, IPO |
| `active_corp` | one-hot | DIVIDENDS, ISSUE, ACQ, CLOSING |
| `active_corp_income` | scalar | DIVIDENDS, ISSUE, ACQ, CLOSING |
| `active_corp_stars` | scalar | DIVIDENDS, ISSUE, ACQ, CLOSING |
| `active_corp_share_price` | scalar | DIVIDENDS, ISSUE, ACQ, CLOSING |
| `active_corp_companies` | flags | DIVIDENDS, ISSUE, ACQ, CLOSING |

**Non-context-dependent fields** (always valid regardless of phase): `phase`, `coo`, all player fields, FI fields, company locations, company adjusted incomes, market availability, corporation data blocks, `turn_number`, `end_card_flipped`, `consecutive_passes`, `cards_remaining`, auction slot info.

---

## Visible State Layout

### Phase & CoO (Offset 0)

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `phase` | 11 | one-hot | Phase indices 0-10 |
| `coo` | 7 | one-hot | Cost of ownership level 1-7 |

**Phase Indices:**
| Index | Phase |
|-------|-------|
| 0 | INVEST |
| 1 | BID_IN_AUCTION |
| 2 | WRAP_UP |
| 3 | ACQUISITION |
| 4 | CLOSING |
| 5 | INCOME |
| 6 | DIVIDENDS |
| 7 | END_CARD |
| 8 | ISSUE_SHARES |
| 9 | IPO |
| 10 | GAME_OVER |

### Players (repeated `num_players` times)

Player stride = `4 + num_players + 36 + 24` = `64 + num_players`

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `cash` | 1 | normalized | / CASH_DIVISOR |
| `net_worth` | 1 | normalized | / CASH_DIVISOR |
| `turn_order` | num_players | one-hot | Position 0 = first to act |
| `owned_companies` | 36 | flags | 1 per company |
| `owned_shares` | 8 | normalized | / SHARE_DIVISOR |
| `is_president` | 8 | flags | 1 per corp |
| `round_trips` | 8 | normalized | min(buys, sells) / MAX_ROUNDTRIPS. Context-dependent: zeroed outside INVEST |
| `acquisition_proceeds` | 1 | normalized | Cash from selling companies this phase / CASH_DIVISOR |
| `income` | 1 | normalized | Total income from owned private companies / PRICE_DIVISOR |

**Player Field Offsets (within player stride):**
| Field | Offset |
|-------|--------|
| cash | 0 |
| net_worth | 1 |
| turn_order | 2 |
| owned_companies | 2 + num_players |
| owned_shares | 38 + num_players |
| is_president | 46 + num_players |
| round_trips | 54 + num_players |
| acquisition_proceeds | 62 + num_players |
| income | 63 + num_players |

### Foreign Investor

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `fi_cash` | 1 | normalized | / CASH_DIVISOR |
| `fi_income` | 1 | normalized | Total income including +5 base bonus / PRICE_DIVISOR |
| `fi_companies` | 36 | flags | Companies owned by FI |

### Company Locations

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `companies_for_auction` | 36 | flags | Available to buy |
| `companies_revealed` | 36 | flags | Drawn this turn (unavailable) |
| `companies_removed` | 36 | flags | Closed/out of game |

### Company Adjusted Incomes

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `company_incomes` | 36 | normalized | / INCOME_DIVISOR, updated when CoO changes |

These are the companies' adjusted incomes (base income minus cost of ownership). They are automatically updated whenever the CoO level changes via `set_coo_level()`.

### Market Availability

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `market_available` | 27 | flags | 1=available, 0=taken by corp |

**Market Price Table (index -> price):**
```
0->0, 1->5, 2->6, 3->7, 4->8, 5->9, 6->10, 7->11, 8->12, 9->13,
10->14, 11->16, 12->18, 13->20, 14->22, 15->24, 16->27, 17->30,
18->33, 19->37, 20->41, 21->45, 22->50, 23->55, 24->61, 25->68, 26->75
```

### Corporations (repeated 8 times)

Corp stride = `10 + 27 + 36 + 36` = `109`

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `active` | 1 | flag | Has been IPO'd |
| `cash` | 1 | normalized | / CASH_DIVISOR |
| `unissued_shares` | 1 | normalized | / SHARE_DIVISOR |
| `issued_shares` | 1 | normalized | / SHARE_DIVISOR |
| `bank_shares` | 1 | normalized | Issued but not player-owned |
| `income` | 1 | normalized | Derived from companies / PRICE_DIVISOR |
| `stars` | 1 | normalized | / CORP_STAR_DIVISOR |
| `share_price` | 1 | normalized | / PRICE_DIVISOR |
| `acquisition_proceeds` | 1 | normalized | Pending this phase |
| `in_receivership` | 1 | flag | |
| `price_index` | 27 | one-hot | Market position |
| `owned_companies` | 36 | flags | Companies owned |
| `acquisition_companies` | 36 | flags | Pending acquisition |

**Corp Field Offsets (within corp stride):**
| Field | Offset |
|-------|--------|
| active | 0 |
| cash | 1 |
| unissued_shares | 2 |
| issued_shares | 3 |
| bank_shares | 4 |
| income | 5 |
| stars | 6 |
| share_price | 7 |
| acquisition_proceeds | 8 |
| in_receivership | 9 |
| price_index | 10 |
| owned_companies | 37 |
| acquisition_companies | 73 |

### Turn State

Size varies with player count: `209 + (3 * num_players)`

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `end_card_flipped` | 1 | flag | |
| `consecutive_passes` | 1 | normalized | / num_players, INVEST phase |
| **Auction:** | | | |
| `auction_price` | 1 | normalized | / PRICE_DIVISOR. 0 when no auction |
| `auction_high_bidder` | num_players | one-hot | 0 when no auction |
| `auction_starter` | num_players | one-hot | 0 when no auction |
| `auction_passed` | num_players | flags | Player left auction. 0 when no auction |
| **Dividends:** | | | |
| `dividend_impact` | 26 | normalized | Price index delta per dividend level / IMPACT_DIVISOR. 0 for invalid levels |
| `dividend_remaining` | 8 | flags | Corps left to process |
| **Issue:** | | | |
| `issue_remaining` | 8 | flags | Corps left to process |
| `issue_price_impact` | 1 | normalized | Price index delta if issuing / IMPACT_DIVISOR. 0 outside ISSUE. |
| `issue_cash_gain` | 1 | normalized | Cash corp receives if issuing / PRICE_DIVISOR. 0 outside ISSUE. |
| **IPO:** | | | |
| `ipo_remaining` | 36 | flags | Companies left |
| **Acquisition:** | | | |
| `acq_is_fi_offer` | 1 | flag | 1=FI target |
| `acq_synergy_values` | 36 | normalized | Synergy income bonus per company / INCOME_DIVISOR, 0 if corp doesn't own |
| **Active Company:** | | | |
| `active_company` | 36 | one-hot | Company under consideration in BID, ACQ, CLOSING, IPO. 0 when inactive. |
| `active_company_stars` | 1 | normalized | / COMPANY_STAR_DIVISOR. 0 when inactive. |
| `active_company_low_price` | 1 | normalized | / PRICE_DIVISOR. 0 when inactive. |
| `active_company_face_value` | 1 | normalized | / PRICE_DIVISOR. 0 when inactive. |
| `active_company_high_price` | 1 | normalized | / PRICE_DIVISOR. 0 when inactive. |
| `active_company_income` | 1 | normalized | Adjusted income / INCOME_DIVISOR. 0 when inactive. |
| **Active Corp:** | | | |
| `active_corp` | 8 | one-hot | Corp under consideration in DIVIDENDS, ISSUE, ACQ, CLOSING (corp-owned offers only). 0 when inactive or player-owned. |
| `active_corp_income` | 1 | normalized | / PRICE_DIVISOR. 0 when inactive. |
| `active_corp_stars` | 1 | normalized | / CORP_STAR_DIVISOR. 0 when inactive. |
| `active_corp_share_price` | 1 | normalized | / PRICE_DIVISOR. 0 when inactive. |
| `active_corp_companies` | 36 | flags | Owned company flags copied from corp data block. 0 when inactive. |
| **Deck:** | | | |
| `cards_remaining` | 1 | normalized | Cards remaining in deck / NUM_COMPANIES |

### Auction Slot Info (5 × num_players)

Per slot (5 floats, ordered by auction slot index):
| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `stars` | 1 | normalized | / COMPANY_STAR_DIVISOR |
| `low_price` | 1 | normalized | / PRICE_DIVISOR |
| `face_value` | 1 | normalized | / PRICE_DIVISOR |
| `high_price` | 1 | normalized | / PRICE_DIVISOR |
| `income` | 1 | normalized | Adjusted income / INCOME_DIVISOR (reflects current CoO) |

Updated when auction row changes (init, auction resolution, WRAP_UP). Empty slots are zero-filled.

### Invest Impacts (16)

Price index delta from buying/selling each corp's share. Context-dependent: zeroed outside INVEST phase.

| Field | Size | Encoding | Notes |
|-------|------|----------|-------|
| `buy_impact` | 8 | normalized | `(new_index - current_index) / IMPACT_DIVISOR` per corp. Positive (price steps up). 0 for inactive corps |
| `sell_impact` | 8 | normalized | `(new_index - current_index) / IMPACT_DIVISOR` per corp. Negative (price steps down). 0 for inactive corps |

Shown for all active corps regardless of affordability or share ownership. Recomputed on active player change during INVEST. Buy uses next higher market space, sell uses next lower.

---

## Hidden State Layout

Hidden state starts at `visible_size` offset. Total hidden size = `1185 + 16 * num_players`.

The hidden state serves several purposes:
- **Information hiding**: Data the NN shouldn't see (deck order, active player before rotation, turn number)
- **Bookkeeping**: Offer buffers for acquisition/closing phases, per-player share buy/sell tracking
- **Performance**: Compact storage for O(1) access to one-hot values and company locations

| Field | Offset | Size | Notes |
|-------|--------|------|-------|
| `active_player` | 0 | 1 | Canonical player index |
| `num_players` | 1 | 1 | Player count |
| `deck_top` | 2 | 1 | Index of top card (-1 = empty) |
| `deck_order` | 3 | 36 | Company IDs in draw order |
| `phase` | 39 | 1 | Compact phase storage |
| `coo_level` | 40 | 1 | Compact CoO storage |
| `auction_company` | 41 | 1 | Compact auction company |
| `auction_high_bidder` | 42 | 1 | Compact high bidder |
| `auction_starter` | 43 | 1 | Compact auction starter |
| `corp_price_indices` | 44 | 8 | Compact price indices per corp |
| `offer_count` | 52 | 1 | Number of acquisition offers |
| `offer_index` | 53 | 1 | Current acquisition offer |
| `offer_buffer` | 54 | 750 | Acquisition offers (owner_type, corp_id, company_id) - 250 offers × 3 floats |
| `close_offer_count` | 804 | 1 | Number of close offers |
| `close_offer_index` | 805 | 1 | Current close offer |
| `close_offer_buffer` | 806 | 300 | Close offers (owner_type, owner_id, company_id) - 100 offers × 3 floats |
| `acq_active_corp` | 1106 | 1 | Compact storage for O(1) access |
| `acq_target_company` | 1107 | 1 | Compact storage for O(1) access |
| `closing_company` | 1108 | 1 | Compact storage for O(1) access |
| `dividend_corp` | 1109 | 1 | Compact storage for O(1) access |
| `issue_corp` | 1110 | 1 | Compact storage for O(1) access |
| `ipo_company` | 1111 | 1 | Compact storage for O(1) access |
| `turn_number` | 1112 | 1 | / 50.0 (moved from visible turn state) |
| `share_buys` | 1113 | num_players × 8 | Per-player buy counts / (MAX_ROUNDTRIPS * 2) |
| `share_sells` | 1113 + np×8 | num_players × 8 | Per-player sell counts / (MAX_ROUNDTRIPS * 2) |
| `company_locations` | varies | 36 | CompanyLocation enum per company (O(1) clearing) |
| `company_owner_ids` | varies | 36 | Owner ID per company (-1 if N/A, player_id or corp_id) |

**CompanyLocation Enum:**
| Value | Location | Notes |
|-------|----------|-------|
| 0 | LOC_DECK | In draw deck (default) |
| 1 | LOC_AUCTION | Available for auction |
| 2 | LOC_REVEALED | Drawn this turn but not auctionable |
| 3 | LOC_PLAYER | Owned by player (owner_id = player_id) |
| 4 | LOC_FI | Owned by Foreign Investor |
| 5 | LOC_CORP | Owned by corporation (owner_id = corp_id) |
| 6 | LOC_CORP_ACQ | In corporation's acquisition pile (owner_id = corp_id) |
| 7 | LOC_REMOVED | Closed/removed from game |

---

## Passing Mechanisms

Different phases use different mechanisms for tracking passes:

| Phase | Mechanism | Location | Description |
|-------|-----------|----------|-------------|
| **INVEST** | `consecutive_passes` counter | Turn State | Counts consecutive passes. Clears when any action is taken. Phase ends when counter >= num_players. |
| **BID_IN_AUCTION** | `auction_passed[num_players]` flags | Turn State | Per-player flags tracking who has left the auction. Auction resolves when only one player remains. |
| **CLOSING** | Offer-based flow | Turn State | Companies are offered one at a time via `closing_company`. Player can close or pass each offer. |

---

## Action Vector

Action space size varies by player count:

| Players | Auction Actions | Total Actions |
|---------|-----------------|---------------|
| 3       | 45 (3 x 15)     | 226           |
| 4       | 60 (4 x 15)     | 241           |
| 5       | 75 (5 x 15)     | 256           |
| 6       | 90 (6 x 15)     | 271           |

Formula: `181 + (num_players * 15)`

Use `get_total_action_count(num_players)` for the exact size.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `AUCTION_CAP` | 15 | Max bid offset over face value |
| `MAX_PAR_SLOTS` | 8 | Max valid par prices per star tier |
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
| **ACQUISITION** | price[51], fi_high, fi_face, pass | 54 | 77-130 |
| **CLOSING** | close, pass | 2 | 131-132 |
| **DIVIDENDS** | dividend[26] | 26 | 133-158 |
| **ISSUE_SHARES** | pass, issue | 2 | 159-160 |
| **IPO** | pass, ipo[8x8] | 65 | 161-225 |

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

#### ACQUISITION Phase (+54 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 to +50 | Price Offer | `price = low_price + (idx - base)` |
| +51 | FI Buy High | Buy FI company at high price |
| +52 | FI Buy Face | Buy FI company at face (OS only) |
| +53 | Pass | Decline acquisition |

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

#### IPO Phase (+65 actions)

| Index | Action | Decoding |
|-------|--------|----------|
| +0 | Pass | Don't IPO |
| +1 to +64 | IPO | `corp_id = (idx - base - 1) // 8`, `par_slot = (idx - base - 1) % 8` |

**Par slot mapping:** Slot N maps to the Nth valid par price for the company's star tier. Use `get_par_index_for_slot(star_tier, slot)` to resolve.

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
| 7 | ACTION_ACQ_FI_HIGH | FI buy at high price |
| 8 | ACTION_ACQ_FI_FACE | FI buy at face value |
| 9 | ACTION_CLOSE | Close current company |
| 10 | ACTION_DIVIDEND | Pay dividend (amount) |
| 11 | ACTION_ISSUE | Issue share |
| 12 | ACTION_IPO | IPO (corp_id, par_slot) |

---

## Usage Examples

### Python Access

```python
from cython_core.state import GameState, get_state_size, get_visible_size
from cython_core.actions import (
    get_valid_action_mask, decode_action_py,
    get_action_layout, get_total_action_count
)

# State vector
state = GameState(num_players=3)
nn_input = state.get_nn_input()  # Shape: (visible_size,)
print(f"Visible size: {state.visible_size}")

# Action vector (size depends on player count)
total_actions = get_total_action_count(3)  # 226 for 3 players
mask = get_valid_action_mask(state)  # Shape: (226,)
valid_actions = mask.nonzero()[0]

# Decode an action
phase, action_type, slot, corp_id, amount = decode_action_py(action_idx, num_players=3)

# Get layout offsets
layout = get_action_layout(num_players=3)
invest_start = layout['invest_start']
```

### Cython Access

```cython
from cython_core.state cimport GameState
from cython_core.actions cimport (
    ActionLayout, ActionInfo, decode_action, compute_action_layout,
    get_total_actions_for_players
)

cdef int num_players = 3
cdef GameState state = GameState(num_players)
cdef ActionLayout layout = compute_action_layout(num_players)
cdef ActionInfo info = decode_action(&layout, action_idx)
```
