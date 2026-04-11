# Cython Core Vector Documentation

This document describes the in-memory game state layout. The state is the engine's only authoritative game data; per-token features for the neural network are produced lazily by `get_token_data()` (a separate path, still to be written as part of the transformer refactor — see `CLAUDE.md`).

---

## State Buffer

`GameState` wraps a single contiguous `int16` numpy array. All values are raw signed integers — no normalization, no one-hot encoding, no visible/hidden split. `int16` is sufficient for every game quantity (player net worth maxes around 400, share counts are single digits, `-1` sentinels fit in the negative range).

### Sizes by player count

| Players | total_size | player_stride | corp_stride | turn_size |
|---------|-----------|---------------|-------------|-----------|
| 2       | 421       | 30            | 15          | 67        |
| 3       | 451       | 30            | 15          | 67        |
| 4       | 481       | 30            | 15          | 67        |
| 5       | 511       | 30            | 15          | 67        |
| 6       | 541       | 30            | 15          | 67        |

`player_stride`, `corp_stride`, and `turn_size` are all fixed across player counts. The players section is the **only** part of the buffer whose size depends on `num_players`, so `total_size = 361 + 30 * num_players` — the constant 361 is the fixed prefix, and only the trailing players section grows.

Layout offsets are computed once at module load and exposed as Cython `cdef` structs at module scope on `core.state`:

- `LAYOUT` (`cdef StateLayout`) — top-level section offsets only: `fi_offset`, `companies_offset`, `market_offset`, `corps_offset`, `turn_offset`, `deck_offset`, `players_offset`. **Strides are not on this struct** — each section's block size lives on its own field-offset struct as a trailing `size` slot (`PLAYER_FIELDS.size`, `CORP_FIELDS.size`, `TURN_OFFSETS.size`, `COMPANY_OFFSETS.size`, `DECK_OFFSETS.size`, `FI_OFFSETS.size`). `total_size` is intentionally not in `StateLayout` either because it depends on `num_players`; compute it inline as `LAYOUT.players_offset + PLAYER_FIELDS.size * num_players` at the few sites that need it.
- `TURN_OFFSETS` (`cdef TurnStateOffsets`) — relative offsets within the (fixed-size) turn block, plus `size`.
- `PLAYER_FIELDS` (`cdef PlayerFieldOffsets`) — relative offsets within a player block, plus `size`.
- `CORP_FIELDS` (`cdef CorpFieldOffsets`) — relative offsets within a corp block, plus `size`.
- `COMPANY_OFFSETS` (`cdef CompanyOffsets`) — relative offsets of the companies-section sub-arrays (`incomes`, `locations`, `owner_ids`), plus `size`.
- `DECK_OFFSETS` (`cdef DeckOffsets`) — relative offsets of the deck-section sub-arrays (`top`, `order`), plus `size`.
- `FI_OFFSETS` (`cdef FIOffsets`) — relative offsets within the FI section (`cash`, `income`), plus `size`.

Cython code reads them directly via `from core.state cimport LAYOUT, TURN_OFFSETS, PLAYER_FIELDS, CORP_FIELDS, COMPANY_OFFSETS, DECK_OFFSETS, FI_OFFSETS`. Python code uses the namedtuple accessors `core.state.get_layout(num_players)`, `get_player_fields()`, `get_corp_fields()`, `get_turn_fields()`, `get_company_fields()`, `get_deck_fields()`, `get_fi_fields()` (none of the field accessors take a `num_players` argument since the fields are fixed-size).

> **Direct layout access is for entity handles only.** See `CLAUDE.md` "Code Conventions". Phase handlers, MCTS code, trainer code, tests, and anything outside `entities/` should go through the handle methods (`PLAYERS[i].get_cash(state)`, etc.) rather than cimporting these constants or indexing `state._data` directly.

`get_turn_fields()` exposes only the public turn-state slots. The final two turn-block slots (`TURN_OFFSETS.player_cache_dirty` and `TURN_OFFSETS.corp_cache_dirty`) are internal dirty masks used by the lazy derived-state caches and remain Cython-only.

---

## Top-level Layout

| Section | Start offset | Size | Description |
|---------|-------------:|------|-------------|
| FI        | 0   | 2   | Foreign investor cash, income |
| Companies | 2   | 108 | Three parallel 36-slot sub-arrays: `incomes`, `locations`, `owner_ids` (see [Companies section](#companies-section)) |
| Market    | 110 | 27  | Per-price availability flags |
| Corps     | 137 | 128 | Per-corp blocks: `corp_stride (16) * 8` (see [Corp block](#corp-block)) |
| Turn      | 265 | 69  | Turn-scoped state including game-wide metadata, active corp/company selectors, plus two internal cache-dirty masks (see [Turn block](#turn-block)) |
| Deck      | 334 | 37  | `top` (1) + `order` (36) — see [Deck section](#deck-section) |
| Players   | 371 | `player_stride * num_players` | Per-player blocks (see [Player block](#player-block)) |

Every offset above is **constant across all player counts** — the players section lives at the end of the buffer for exactly this reason. The "Start offset" column is identical for every player count up to and including the players section start.

Section start offsets are exposed via `LayoutInfo` as `fi_offset`, `companies_offset`, `market_offset`, `corps_offset`, `turn_offset`, `deck_offset`, and finally `players_offset`. There is no separate top-level "metadata" section — the [turn block](#turn-block) starts with `active_player`, `active_corp`, `active_company`, then the canonical game metadata (`num_players`, `phase`, `coo_level`, `turn_number`).

---

## Player block

Stride: **30**. Player `i` lives at `players_offset + i * 30`. Field offsets via `core.state.get_player_fields()` (`PlayerFields` namedtuple) for Python, or `from core.state cimport PLAYER_FIELDS` for Cython.

| Relative offset | Field | Size | Notes |
|----------------|-------|------|-------|
| 0  | cash            | 1 | |
| 1  | net_worth       | 1 | |
| 2  | liquidity       | 1 | Iterative share liquidation value (cash + value of shares + companies) |
| 3  | turn_order      | 1 | Position 0 = first to act |
| 4  | owned_shares    | 8 | Per-corp share count |
| 12 | income          | 1 | Income from owned private companies |
| 13 | share_buys      | 8 | Per-corp buy counts (this turn) |
| 21 | share_sells     | 8 | Per-corp sell counts (this turn) |
| 29 | has_passed      | 1 | `1` once this player has passed in the current phase |

All per-player tracking lives inside one player block, so a single pointer hop reaches everything for player `i`. Presidency is tracked per-corp via `CORP_FIELDS.president_id` (see [Corp block](#corp-block)), not in the player block. Round-trip counts are derived on demand from `min(share_buys, share_sells)` per corp — no dedicated slot. The generic `has_passed` flag previously lived in the turn block as an auction-specific per-player array; moving it into the player block makes the player block fully self-contained and the turn block fixed-size.

---

## Foreign Investor (size 2)

Sub-offsets via `core.state.get_fi_fields()` (`FIFields` namedtuple) for Python, or `from core.state cimport FI_OFFSETS` for Cython.

| Relative offset | Field |
|----------------|-------|
| 0 | cash |
| 1 | income |

FI ownership of companies is tracked via `companies.locations` (`LOC_FI`), not via flags on the FI block.

---

## Companies section

Block size: **108**. Sub-offsets via `core.state.get_company_fields()` (`CompanyFields` namedtuple) for Python, or `from core.state cimport COMPANY_OFFSETS` for Cython. Each sub-array is indexed by `company_id` (0-35) regardless of current location.

| Relative offset | Sub-array       | Size | Description |
|----------------|-----------------|------|-------------|
| 0  | `incomes`    | 36 | Per-company adjusted income (`base_income − coo_cost(coo_level)`). Recomputed when the CoO level changes. |
| 36 | `locations`  | 36 | `CompanyLocation` enum — see [Company tracking](#company-tracking). |
| 72 | `owner_ids`  | 36 | Owner ID per company (`player_id`, `corp_id`, or `-1`). Seeded to `-1` in `__cinit__`. |

`locations` + `owner_ids` are the single source of truth for "who owns what". There is no per-player or per-corp `owned_companies` bitmap — entity handles scan these arrays directly.

---

## Market availability (size 27)

One flag per market price slot at `market_offset + i`. `1` = available, `0` = claimed by a corp.

State invariant: index `0` (`$0`, bankruptcy) and index `26` (`$75`, max price) are sentinel spaces and must remain available. `MARKET.set_space_available(state, idx, False)` rejects those two indices.

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
| 6  | company_stars         | Cached sum of `COMPANY_STARS` over owned + acq-zone companies |
| 7  | acquisition_proceeds  | Pending payment, written and consumed during ACQ |
| 8  | in_receivership       | flag |
| 9  | price_index           | Market position 0–26 |
| 10 | raw_revenue           | Sum of base company incomes |
| 11 | synergy_income        | |
| 12 | coo_cost              | Always ≤ 0 |
| 13 | ability_income        | |
| 14 | president_id          | `player_id` or `-1` (inactive / receivership). Initialized to `-1`. |
| 15 | passed_acq_offer      | flag — `1` if corp passed on current ACQ_OFFER |

`share_price` is **not stored** in the corp block. It is derived from `price_index` via `MARKET_PRICES[price_index]`; use `CORPS[c].get_share_price(state)` rather than expecting a backing slot.

`total_stars`, `cash_stars`, and `pending_price_move` are **not stored** either. They are all derived on demand:

- `cash_stars = floor(cash / 10)` clamped at 0.
- `total_stars = company_stars + cash_stars + (2 if corp_id == CORP_SI else 0)`.
- `pending_price_move` is predicted on demand by the corp handle when asked.

Only `company_stars` — the expensive part of the total that depends on owned-company scans — is cached in the block.

### Derived corp cache

Several corp-block fields are stored but derived from authoritative state: `income`, `company_stars`, `raw_revenue`, `synergy_income`, `coo_cost`, and `ability_income`.

- Mutations of authoritative corp state mark the corporation dirty via the internal `TURN_OFFSETS.corp_cache_dirty` bitmask (one bit per corp).
- The first read through any derived getter (`get_income`, `get_total_stars`, `get_pending_price_move`, etc.) runs one `_refresh_cache(state)` pass, recomputing the full derived bundle coherently from authoritative state.
- Cache invalidation happens through the entity handles' published mutators — don't touch the backing slots directly, because direct writes bypass the dirty-mask bookkeeping and leave the corp's derived view inconsistent.

---

## Turn block

Block size: **69**, fixed across player counts. Sub-offsets via `core.state.get_turn_fields()` (`TurnFields` namedtuple) for Python, or `from core.state cimport TURN_OFFSETS` for Cython. The turn block starts with the active player plus the generic active corp/company selectors, then the remaining game-wide metadata and phase state.

| Relative offset | Field | Size | Notes |
|----------------|-------|------|-------|
| 0  | active_player        | 1  | Canonical player_id |
| 1  | active_corp          | 1  | Active corp for ISSUE / DIVIDENDS / ACQ_OFFER context, or `-1`. **Unused during ACQUISITION** — that phase picks corps directly from the masked action space, so this slot stays at `-1`. |
| 2  | active_company       | 1  | Active company for BID / IPO / ACQ_OFFER context, or `-1`. **Unused during ACQUISITION and CLOSING** — those phases pick companies directly from the masked action space, so this slot stays at `-1`. |
| 3  | num_players          | 1  | 2–6, seeded in `__cinit__` |
| 4  | phase                | 1  | 0–11, see [Phase enum](#phase-enum) |
| 5  | coo_level            | 1  | 1–7 |
| 6  | turn_number          | 1  | 1+ |
| 7  | end_card_flipped     | 1  | flag |
| 8  | consecutive_passes   | 1  | INVEST pass counter; phase ends when this reaches `num_players` |
| 9  | cards_remaining      | 1  | |
| 10 | auction_price        | 1  | 0 when no auction |
| 11 | auction_high_bidder  | 1  | `player_id` or `-1` |
| 12 | auction_starter      | 1  | `player_id` or `-1` |
| 13 | acq_offer_corp       | 1  | `corp_id` being offered in ACQ_OFFER, or `-1` |
| 14 | acq_offer_company    | 1  | `company_id` being contested in ACQ_OFFER, or `-1` |
| 15 | dividend_remaining   | 8  | Per-corp pending flag |
| 23 | issue_remaining      | 8  | Per-corp pending flag |
| 31 | ipo_remaining        | 36 | Per-company pending flag |

Internal slots at relative offsets `67` and `68` hold the player-finance (`player_cache_dirty`) and corp-derived (`corp_cache_dirty`) dirty masks used by the lazy cache system. These offsets exist in `TURN_OFFSETS` for Cython code but are intentionally omitted from the Python `TurnFields` namedtuple.

There is no dedicated `auction_company` slot — the generic `active_company` selector at offset 2 carries that role during BID, and likewise covers the active-company context for IPO and ACQ_OFFER. It is **not** used during ACQUISITION or CLOSING: the old per-phase offer buffers are gone, and those phases now pick the target company directly from the masked action space, so both `active_corp` and `active_company` sit at `-1` throughout them. ACQ_OFFER reuses the same selectors for FI-priority resolution: `active_corp` = the preempting corp being offered the FI-owned company, `active_company` = the contested FI company, and `active_player` = that corp's president. No new per-phase fields (`closing_company` / `dividend_corp` / `issue_corp` / `ipo_company`) are needed — the generic selectors plus the per-phase remaining bitmasks already in the turn block cover every phase. The per-player `has_passed` flag used to live in the turn block as an auction-specific array; it now lives in the player block, so the turn block is fully fixed-size.

---

## Deck section

Block size: **37**. Sub-offsets via `core.state.get_deck_fields()` (`DeckFields` namedtuple) for Python, or `from core.state cimport DECK_OFFSETS` for Cython.

| Relative offset | Field | Size | Notes |
|----------------|-------|------|-------|
| 0 | `top`   | 1  | Index of next card to draw; `-1` = empty |
| 1 | `order` | 36 | Shuffled company IDs; only the `[0..top]` range is live |

Companies excluded for the current player count never appear in `order` and are marked `LOC_EXCLUDED` in the companies section's `locations` sub-array.

---

## Company tracking

The `locations` and `owner_ids` sub-arrays live inside the [companies section](#companies-section) — there is no separate top-level section for them. This table documents the enum values:

### `CompanyLocation` enum (stored in `locations`)

| Value | Name          | Owner field meaning |
|-------|---------------|---------------------|
| 0 | LOC_DECK      | none (`owner_id = -1`) |
| 1 | LOC_AUCTION   | none |
| 2 | LOC_REVEALED  | none |
| 3 | LOC_PLAYER    | `player_id` |
| 4 | LOC_FI        | none (`owner_id = -1`) |
| 5 | LOC_CORP      | `corp_id` |
| 6 | LOC_CORP_ACQ  | `corp_id` (in corp's acquisition pile) |
| 7 | LOC_REMOVED   | none — closed during play |
| 8 | LOC_EXCLUDED  | none — never dealt for this player count |

`LOC_REMOVED` and `LOC_EXCLUDED` are kept distinct so excluded companies do not leak deck composition. `LOC_DECK = 0` is the zero-init default — `__cinit__` seeds `owner_ids` to `-1` so freshly-allocated states don't read as "every company owned by player 0".

---

## Phase enum

Stored as a raw integer at `LAYOUT.turn_offset + TURN_OFFSETS.phase`. Defined in `core/data.pxd` (`GamePhases`).

| Value | Name | Notes |
|-------|------|-------|
| 0  | PHASE_INVEST         | Buy/sell shares, start auctions |
| 1  | PHASE_BID | Bidding for a company |
| 2  | PHASE_WRAP_UP        | Automated: FI buys companies at face value |
| 3  | PHASE_ACQUISITION    | Corps acquiring companies |
| 4  | PHASE_ACQ_OFFER      | FI preemption sub-phase (higher-priority corps get first shot) |
| 5  | PHASE_CLOSING        | Player-owned companies closing |
| 6  | PHASE_INCOME         | Automated: income payouts |
| 7  | PHASE_DIVIDENDS      | Dividend declaration |
| 8  | PHASE_END_CARD       | Automated: game end card trigger |
| 9  | PHASE_ISSUE_SHARES   | Corp issuing a share |
| 10 | PHASE_IPO            | Select corp charter + par price in one action |
| 11 | PHASE_GAME_OVER      | Terminal state |

**PAR is gone.** The old separate `PHASE_PAR` is merged into `PHASE_IPO`: each IPO decision picks `(corp, par_index)` in a single action.

**`PHASE_ACQ_OFFER` is the FI-priority resolution phase.** It is a first-class engine phase, not a hidden sub-state of ACQUISITION. The engine enters ACQ_OFFER whenever a player or receivership corp attempts to acquire an FI-owned company AND one or more higher-priority corps exist (OS has absolute priority at face value; remaining corps ordered by descending share price at high value). Each higher-priority corp is offered, in turn, a 2-action `{pass, buy}` choice; once all decline, control returns to the original ACQUISITION acquirer. The turn block's existing `active_corp` / `active_company` / `active_player` selectors carry the preempting corp, the contested FI company, and that corp's president for the duration.

**No per-phase offer-buffer bookkeeping.** The old ACQUISITION and CLOSING offer buffers (pre-sorted one-by-one presentation) are gone. ACQUISITION picks `(corp, company, offset)` directly from the masked action space; CLOSING picks a company directly. This means ACQUISITION has no "active" corp or company — both turn slots sit at `-1` during that phase. The old engine used careful offer ordering to implement FI priority; the new engine replaces that with explicit `PHASE_ACQ_OFFER` transitions.

---

## Pass tracking

| Phase | Mechanism | Location |
|-------|-----------|----------|
| INVEST            | `consecutive_passes` counter | turn block (slot 8) |
| BID    | per-player `has_passed` flag | player block (slot 29) |
| CLOSING / ACQUISITION / IPO / ISSUE | per-phase, handled by the decision-phase enumeration + pass action | engine, not a dedicated state slot |

---

## Construction surface

| Method | Behavior |
|--------|----------|
| `GameState(num_players)` | Allocate a fresh zero-initialized buffer; seed `turn.num_players`, `companies.owner_ids`, and `corps.president_id` to `-1`. |
| `GameState.from_array(arr, num_players)` | Allocate and copy `arr` into the new state. Accepts any 1-D `int16` array/view of the correct logical length, contiguous or not. |
| `GameState.from_buffer(buf, num_players)` | Wrap an existing writable C-contiguous `int16` buffer zero-copy. Buffer must already contain valid state — does **not** seed `companies.owner_ids`. |
| `state.rebind(buf, num_players)` | Repoint an existing `GameState` at a different writable C-contiguous buffer. Used in MCTS hot paths. |
| `state.initialize_game(num_players, seed=-1)` | Reset to a fresh game state for the requested player count, then set up players, FI, corps, market, deck, turn state, and active player. `num_players` is required; `seed=-1` uses current time. |

`GameState` exposes only structural primitives publicly: `_player_ptr` / `_corp_ptr` / `_turn_ptr` (cdef nogil, used by entity handles), `_num_players` (cdef int, readable from cdef code for assertions), `get_active_player` / `set_active_player`, `get_num_players`, and `initialize_game`. There are no per-instance layout-offset fields — every entity handle reads offsets directly from the module-level `LAYOUT` / `PLAYER_FIELDS` / `CORP_FIELDS` / `TURN_OFFSETS` / `COMPANY_OFFSETS` / `DECK_OFFSETS` / `FI_OFFSETS` constants on `core.state`. All field-level reads and writes go through the entity handles in `entities/`.

---

## Action Space

The old global dense `action_dim` vector is gone. The new action space is **per-phase and phase-local**: the same integer means different things in different phases, so callers must always carry `phase_id` alongside `action_id`. The canonical per-phase sizes live in `core/data.pxd` as the `ActionSize` `cpdef enum`; `core/actions.pxd` cimports them for its encode/decode arithmetic, and `nn/transformer.py` imports the `PHASE_ACTION_SIZES` Python list and `MAX_ACTION_SIZE` from `core.data`. Single source of truth — no cross-file sync. An import-time roundtrip assert in `core/actions.pyx` catches encode-formula drift against those sizes.

### Decision phases

The engine has 12 `GamePhases`; the model only sees the 8 decision phases where a real choice is needed. The `DecisionPhase` `cpdef enum` lives in `core/data.pxd`, as does the `ENGINE_TO_DECISION_PHASE[12]` lookup table that maps each `GamePhases` value to its `DecisionPhase` (or `-1` for automated/terminal engine phases: `WRAP_UP`, `INCOME`, `END_CARD`, `GAME_OVER`). `get_decision_phase(state)` in `core/actions.pyx` is the nogil helper that reads the engine phase from state and indexes that table.

| DecisionPhase       | ID | Action count | Layout |
|---------------------|----|-------------:|--------|
| `DPHASE_INVEST`     | 0  | 557   | 1 pass + 36×15 auction (company-indexed) + 8×2 trade (buy/sell) |
| `DPHASE_BID`        | 1  | 15    | 1 pass (= leave auction) + 14 raise offsets |
| `DPHASE_ACQUISITION`| 2  | 14977 | 1 pass + 8×36×52 corp × company × {51 price offsets + FI_BUY} |
| `DPHASE_ACQ_OFFER`  | 3  | 2     | pass + buy (FI preemption) |
| `DPHASE_CLOSING`    | 4  | 37    | 1 pass + 36 per-company close |
| `DPHASE_DIVIDENDS`  | 5  | 26    | dividend amounts 0–25 |
| `DPHASE_ISSUE`      | 6  | 2     | pass + issue |
| `DPHASE_IPO`        | 7  | 113   | 1 pass + 8×14 corp × par index (merged IPO+PAR) |

Max action id fits comfortably in `uint16`. `MAX_ACTION_SIZE = 14977` (ACQUISITION). The sparse legal-action buffer width is `MAX_LEGAL_ACTIONS = 256`; any enumeration that exceeds this is a bug and `enumerate_legal_actions` asserts on overflow.

### Encode / decode contract

`core/actions.pxd` exposes one `encode_*` helper per action kind per phase (all `cdef inline nogil`) plus a single `decode_action(phase_id, action_id) → ActionInfo` inverse. `encode_action(ActionInfo)` is the tested roundtrip. All encoders use pure arithmetic, no state reads.

Highlighted encoder layouts:

```
INVEST:
  0                                    = pass
  1 + company_id*15 + bid_offset       = start auction on company at face + (bid_offset + 0..14)
  541 + corp_id*2                      = buy share in corp
  541 + corp_id*2 + 1                  = sell share in corp

BID:
  0                                    = leave auction (pass-class action)
  1 + raise_offset                     = raise by (face + 1 + raise_offset)

ACQUISITION:
  0                                    = pass
  1 + (corp_id*36 + company_id)*52 + k = acquire (corp, company) at:
                                           k < 51  → low_price + k  (normal)
                                           k == 51 → FI_BUY (special price)

ACQ_OFFER:
  0 = pass
  1 = preempting corp takes the offered company

CLOSING:
  0             = pass
  1 + company_id = close company

DIVIDENDS:
  amount  (0..25, per share)

ISSUE:
  0 = pass
  1 = issue one share

IPO:
  0                              = pass
  1 + corp_id*14 + par_index     = start corp at par price ALL_PAR_PRICES[par_index]
```

### Sparse legal-action interface

Legal-action enumeration is sparse: `enumerate_legal_actions(state, phase_id, uint16_t* ids)` writes deterministic phase-local ids into the buffer and returns the count. Per-phase helpers live in `core/actions.pyx` as `_enumerate_*`. **Right now these helpers are all empty stubs** (tracked as rss-az-848a); the rule-level legality logic needs to be ported out of the old `actions-old.pyx` on `main` into these helpers as part of the refactor. Until that lands, `enumerate_legal_actions` returns 0 for every phase and `get_forced_action` always returns `(-1, False)`.

Python-accessible wrappers (for tests + diagnostics):

| Function | Returns |
|----------|---------|
| `get_phase_action_size(phase_id)` | Per-phase action count from `core.data.PHASE_ACTION_SIZES` |
| `decode_action_py(phase_id, action_id)` | Tuple `(phase, action_type, corp_id, company_id, amount)` |
| `enumerate_legal_actions_py(state, phase_id=-1)` | Tuple `(phase_id, uint16 ndarray of legal ids)` — empty until enumerators land |
| `get_forced_action_py(state)` | Tuple `(action_id, found)` — `(-1, False)` until enumerators land |

### ActionType enum (decoded semantics)

`decode_action` populates an `ActionInfo` struct with `phase`, `action_type`, `corp_id`, `company_id`, and `amount` (unused fields = -1).

| Value | Type | Used by | Meaning |
|-------|------|---------|---------|
| 0  | `ACTION_PASS`          | INVEST, BID, ACQUISITION, ACQ_OFFER, CLOSING, ISSUE, IPO | Universal pass/opt-out. In BID this means "leave the auction". |
| 1  | `ACTION_AUCTION`       | INVEST     | Start auction on `company_id` at face + `amount` (bid_offset) |
| 2  | `ACTION_BUY_SHARE`     | INVEST     | Buy share in `corp_id` |
| 3  | `ACTION_SELL_SHARE`    | INVEST     | Sell share in `corp_id` |
| 4  | `ACTION_RAISE`         | BID        | Raise current bid by (face + 1 + `amount`) |
| 5  | `ACTION_ACQ_PRICE`     | ACQUISITION| `corp_id` acquires `company_id` at low_price + `amount` |
| 6  | `ACTION_ACQ_FI_BUY`    | ACQUISITION| `corp_id` buys `company_id` from FI at fixed price (OS=face, others=high) |
| 7  | `ACTION_ACQ_OFFER_BUY` | ACQ_OFFER  | Preempting corp takes the offered company |
| 8  | `ACTION_CLOSE`         | CLOSING    | Close `company_id` |
| 9  | `ACTION_DIVIDEND`      | DIVIDENDS  | Pay dividend of `amount` per share |
| 10 | `ACTION_ISSUE`         | ISSUE      | Issue one share |
| 11 | `ACTION_IPO`           | IPO        | Start `corp_id` at par price `ALL_PAR_PRICES[amount]` |

---

## Usage Examples

### Python Access

```python
from core.state import GameState, get_layout, get_player_fields
from core.data import PHASE_ACTION_SIZES
from core.actions import (
    decode_action_py,
    enumerate_legal_actions_py,
    get_phase_action_size,
)
from entities.player import PLAYERS

# State buffer
state = GameState(num_players=3)
state.initialize_game(3, seed=42)
print(f"buffer length = {len(state._array)}")  # 451 for 3p

# Layout introspection
layout = get_layout(3)            # LayoutInfo namedtuple
print(layout.players_offset)      # 361 (constant across player counts)
print(layout.total_size)          # 451

pf = get_player_fields()          # PlayerFields namedtuple
print(pf.cash, pf.has_passed)     # 0 29

# Field access goes through entity handles, not raw buffer indexing
print(PLAYERS[0].get_cash(state))

# Per-phase action space
print(PHASE_ACTION_SIZES)             # [557, 15, 14977, 2, 37, 26, 2, 113]
print(get_phase_action_size(2))       # 14977 (ACQUISITION)
print(decode_action_py(0, 0))         # (0, 0, -1, -1, -1) — INVEST pass

# Sparse legal actions (currently returns empty until enumerators land)
phase_id, legal_ids = enumerate_legal_actions_py(state)
```

### Cython Access

```cython
from core.state cimport GameState, LAYOUT, PLAYER_FIELDS, TURN_OFFSETS
from libc.stdint cimport int16_t

cdef GameState state = GameState(3)

# Read a player field directly via the module-level constants
cdef int16_t cash = state._player_ptr(0)[PLAYER_FIELDS.cash]

# Read a turn-block field (phase lives inside the turn block;
# there is no stand-alone LAYOUT.phase_offset slot)
cdef int phase = state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
cdef int auction_price = state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price]
```

**The snippets above are what entity handle implementations do under the hood — they are not examples of application code.** Every phase handler, MCTS path, trainer call site, and test should go through handle methods (`PLAYERS[i].get_cash(state)`, `TURN.get_auction_price(state)`, `CORPS[c].get_share_price(state)`, etc.) rather than cimporting layout constants or indexing `state._data` directly. The direct-indexing pattern is reserved for the entity modules themselves.
