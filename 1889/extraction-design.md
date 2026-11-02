# Shikoku 1889 Feature Extraction Design

Status: design for step 7 of [`pipeline-refactor.md`](pipeline-refactor.md),
dated 2026-11-02, for review. Nothing here is implemented. Section 13 lists
the open questions with a recommendation each; the user settles them before
7.1 starts, as for [`engine-design.md`](engine-design.md) §12.

Step 7 fills the extraction half of `games/g1889.py`, whose callables raise
`NotImplementedError` and whose sizes are `NO_SIZE` (`games/g1889.py:48-49`,
`:108-117`). Step 8 builds the first model on it, `nn/transformer-1889.py`
v0, without build-distance planes. The blueprint is [`1889.md`](1889.md)
§5-9; where it and the code disagree, the code wins, and Section 12 lists
the differences.

Paths are relative to the repo, and `path:line` cites the code at 90d5825.
**Measured:** marks a number from the scripts in
`scratchpad/g1889_extract_design/` (Section 2); **Test:** marks a claim to
measure once the code exists.

**Affected by 6.10.** Engine sub-step 6.10 refactored the decision phases
after this draft (engine-design §5.2-5.3, §12 item 7): RAISE folded into
BID, sales factor into a corp and SELL_AMOUNT or EMERGENCY_SELL_AMOUNT, and
BUY_TRAIN into a source, SELECT_TRAIN, and TRAIN_PRICE. The draft is not yet
revised for it; these parts change:

- **Phase count and sizes.** 18 decision phases, not 16: GLOBAL's phase
  one-hot (Section 5.3, slots 1-16) needs 18 slots, so GLOBAL grows to 81
  and its later slots shift. The unified dimension is 320, not 421
  (Section 12, item 8), and `max_action_size` 62, not 102.
- **The price-grid row's phases** (Sections 5.1 and 5.4). PRICE is visible
  at BID, PAR, TRAIN_PRICE, and PRIVATE_PRICE; its kind slots lose RAISE.
  BID's ids are 0 (a closed auction's drop, else unused) and 1-16 (the
  points), so slots 6-22 read BID's points from id 1, and slots 23-24's
  RAISE cases are BID's in a closed auction.
- **Selections** (Sections 5.2, 5.5, 5.6). PRIVATE's `is_selected` is BID's
  private, `auction_private` in a closed auction, else `pending_private`.
  The selling corp is `pending_train_source` − `TRAIN_SOURCE_CORP` at
  SELECT_TRAIN and TRAIN_PRICE (not `/ 6`), and TRAIN's `is_selected` is
  `pending_train_type` at TRAIN_PRICE (not `% 6`). SELECT_TRAIN's source
  (the depot, the pool, or a corp) and the corp a sale is of
  (`pending_corp` at SELL_AMOUNT and EMERGENCY_SELL_AMOUNT) need selection
  slots, and the decision corp's definition needs the three new phases.
- **Per-choice features move to the sub-phases** (Sections 5.5, 5.6, 5.10,
  7.2, and 9). TRAIN 10-12's train deltas and their cost ("a solve per
  offered type and trade-in") belong to SELECT_TRAIN, over the pending
  source's trains; BUY_TRAIN's choices are sources and corps to sell.
  CORP 14-18's sale prices serve SELL_AMOUNT and EMERGENCY_SELL_AMOUNT for
  the pending corp, and STOCK's choice of a corp. Section 7.2's table needs
  rows for SELL_AMOUNT, SELECT_TRAIN, and EMERGENCY_SELL_AMOUNT, and its
  BID, STOCK, and BUY_TRAIN rows change; Section 13, item 8 refers to them.
- **Tests** (Section 10, item 2): sale prices check against applying
  SELL_AMOUNT and EMERGENCY_SELL_AMOUNT, and train deltas at SELECT_TRAIN.
- **Measurements** (Sections 2 and 9): the decision mix and the per-phase
  costs predate 6.10, whose sub-phases take shares of BUY_TRAIN's and
  STOCK's decisions. `path:line` citations of `actions.pyx` and
  `constants.pxd` predate it too.

## 1. Contract and scope

What the pipeline already fixes (pipeline-refactor §3.2 and §5,
`games/__init__.py:126-142`):

- **Shapes.** Tokens are `(num_tokens(max_players), token_dim(layout_version))`
  float32, sent to the eval server as float16 (`train/eval_server.py:13`).
  Dense relations are `(num_relations, T, T)` uint8, queries by keys. Sparse
  records are `(max_relation_edges, 4)` uint8 rows `(relation, query, key,
  value)`, zero past the count (`games/__init__.py:17-18`, `:40`).
- **Rows only.** Inputs depend only on the row, never on runtime flags, and
  batch extractors take raw rows that may mix player counts at one capacity.
- **Three tensors.** Tile-candidate features go in extra token rows, and
  token indices fit `uint8`.
- **Consumers.** The eval worker fills tokens and sparse records for one
  state (`train/eval_server.py:1437-1447`) or a batch of pool-row views
  (`:1505-1515`). `NNEvaluator` fills dense planes (`mcts/evaluator.py:374-378`,
  `:419-424`, `:498-503`). Replay rebuilds dense planes from copied rows at
  every sample (`train/replay_buffer.py:365`, `:420`), and the trainer then
  fills tokens from the same copies (`train/trainer.py:313-322`, `:392-415`).
- **How v3 reads them**, the pattern step 8 copies (pipeline-refactor §5):
  - slot 0 of every row is a 0/1 key mask (`nn/transformer-v3.py:886-897`);
  - an `is_selected` slot picks the actor's embeddings (`:1167-1196`);
  - a record with value 0 is padding (`:951`), and an edge into a hidden key
    is dropped (`:391`);
  - every policy head runs on every row, and the legal mask selects
    (`:1540-1547`);
  - a model may ignore trailing relation ids: v2 keeps the first ten
    (`nn/transformer-v2.py:832`, `:1526`).

**In scope:** the token rows, 25 dynamic relations, a helper for the static
planes, the candidate rows and other per-choice outcome features, the
adapter wiring, and the tests. **Out of scope:** build distance and the
shortest-build-path planes (Section 8), the model (step 8), and 2- and
6-player rows, which stay engine-only (Section 13, item 14).

## 2. What random play reaches

**Measured.** 900 games per policy, 300 at each of 3-5 players at capacity
5, with the self-play flags, played by engine-design §6's two random
players: uniform over the policy set (642,741 returned decisions) and the
default suite's random player (498,250). Scripts: `measure.py` (counts in
12 processes; costs in one, on 100 games per player count),
`buy_train_tail.py`, `value_ranges.py`, `rss_baseline.py`; outputs beside
them. Ranges span the two policies.

| Quantity | Median | p90 | p99 | Max |
|---|---|---|---|---|
| Hexes holding a supply tile, every decision | 10-12 | 27-30 | 36-38 | 45 |
| The same, at the game's end | 31-33 | 38-40 | 43 | 45 |
| TILE: candidates at the pending hex | 3 | 6 | 9 | 14-15 |
| TRACK: hexes with a candidate | 4 | 10 | 16 | 26-28 |
| TRACK: candidates summed over those hexes | 11-12 | 25-28 | 42-44 | 74-77 |
| TOKEN: tokenable cities | 1 | 3 | 7 | 10-11 |
| ER_BONUS: candidates on C4 | 8 | 8 | 8 | 8 |
| PORT: port lays | 10 | 10 | 10 | 10 |

- **Laid hexes by phase** (median; max): phase 2 0-2, 13-15; phase 3 7-9,
  28-31; phase 4 12-16, 37-38; phase 5 22-27, 43-45; phase D 33-36, 45.
- **Largest raw values** (`value_ranges.py`, 360 games): a corp's maximum
  revenue ¥860, a treasury ¥4,746, a player's cash ¥2,128 and value ¥2,548,
  the bank −¥5,272.
- **Decision mix:** STOCK 53-61%, TRACK 10-13%, DIVIDEND 7-10%, BUY_TRAIN
  6-8%, TILE 5-7%, TRAIN_PRICE 1-2%, TOKEN 1.2-1.6%, everything else under
  3% each.
- MAX_LAY_CANDIDATES (15) is reached: one TILE state of the biased policy
  had 15.

Random play lays track on random hexes, so self-play maps may be denser
(engine-design §6). The edge bound in Section 6.3 is therefore a worst case
by construction, not a measured maximum.

## 3. Modules and entry points

Two modules at the package root, named after RSS's, which the root modules
mirror (engine-design §1):

| Module | Holds |
|---|---|
| `g1889/token_data.pxd/.pyx` | the row map and widths (`cpdef enum`s), the prologue (Section 4), the fill kernels, `get_num_tokens`, `get_token_dim`, `get_token_widths`, `get_token_data`, `get_token_data_batch` |
| `g1889/relations.pxd/.pyx` | the relation enum and bound, one edge emitter, `get_relation_data(_batch)`, `get_relation_coord_data(_batch)`, `static_relation_planes(max_players)` |

- **Signatures** follow `core/token_data_v3.pyx:311-435` and
  `core/relations.pyx:77-359`, minus RSS's legacy `(rows, num_players,
  buffer)` call shape, which no adapter caller uses. Shape asserts as
  `core/relations.pyx:99-116`. Player counts as RSS: 3 ≤ `num_players` ≤
  `max_players` ≤ 5 (`core/relations.pyx:362-380`).
- **Adapter.** `num_tokens=get_num_tokens`, `token_dim=get_token_dim`,
  `num_relations=25`, `max_relation_edges=2048`, and the six fills, as
  `games/rss.py:134-143` does RSS's. `NO_SIZE` and `_no_extraction` go.
- **Engine additions** (nogil, in the owning modules, so extraction mutates
  scratch rows only through entity code; `AGENTS.md` "Boundaries"):

| Module | Addition | Reused by |
|---|---|---|
| `entities/hexmap` | `map_put_tile(state, hex, tile, rotation)`: supply exchange, canonical rotation, `invalidate_connectivity` (`hexmap.pyx:55`) | `MAP.lay` after its asserts (`hexmap.pyx:215-228`) |
| `entities/hexmap` | `map_put_token(state, hex, corp)`: `tokens_left` − 1, the mask bit | `MAP.place_token` (`:239-248`) |
| `entities/trains` | `trains_set_count(state, corp, type, count)` | `TRAINS.set_corp_trains` (`trains.pyx:144-149`) |
| `entities/market` | `market_moved(row, col, move, steps, int* row, int* col)`, pure | `MARKET.move_*` (`market.pyx:113-146`) |

Everything else extraction reads is already exported nogil: GRAPH
(`graph.pxd:19-21`), `route_max` and `solve_routes` (`routes/solver.pxd:29-30`),
`lay_candidates`, `er_candidates`, `can_token`, `can_buy_private`,
`can_exchange_dr`, `must_buy_train`, `in_emergency`, and `price_amount`
(`actions.pxd:78-98`), and the entity getters (`corp.pxd:21-34`,
`player.pxd:9-22`, `private.pxd:11-22`, `trains.pxd:6-16`,
`market.pxd:13-26`, `hexmap.pxd:18-28`). Fills read other raw slots by
layout offsets, as RSS's extractors do.

## 4. The prologue, the cache write-back, and the scratch state

**Prologue.** Each row's fill begins, nogil, with `graph_refresh(state, c)`
and `route_max(state, c)` for all seven corps, as RSS refreshes player
caches (`core/token_data_v3.pyx:345-348`, `:431-434`). The fills then read
the graph block, `route_available`, and `route_cache` raw. Following
engine-design §8's recommendation, extraction reads `route_max`, never
`solve_routes`, for the current state, and both fills run the prologue: the
trainer fills relations before tokens, the eval worker tokens before
relations, and the second finds the caches fresh.

**Write-back.** The refresh writes the caches into the row it reads, as the
engine's lazy getters do (`graph.pyx:190-206`, `solver.pyx:610-619`).

- **What changes.** Only the graph block, `route_available`, `route_cache`,
  and the dirty bits `conn_dirty` and `route_dirty`; each written value is
  what the engine computes from the rest of the row.
- **Where it pays.** The eval worker's leaf batches are pool-row views, so a
  leaf's solves ride into its children's copies. Replay and trainer rows
  are copies (`replay_buffer.py:355`, `:412`), and the root of a search is
  the live state, whose next decision reuses the caches.
- **Rules.** Rows passed to extraction must be writable; no feature reads a
  dirty bit; the output for a row is the same whether its caches were fresh
  or stale.

**Measured** on live rows, where the cache rides as in pool rows (from
Python, about 0.08 µs of call overhead per call):

| Prologue | Median | p90 | p99 | Max |
|---|---|---|---|---|
| GRAPH refresh × 7 + `route_max` × 7 | 0.48-0.50 µs | 2.6-3.8 µs | 19-20 µs | 0.43-0.44 ms |
| of which GRAPH | 0.32-0.33 µs | 0.6-1.2 µs | 6.6-7.0 µs | 0.11 ms |
| of which `route_max` | 0.16-0.17 µs | 1.9-2.6 µs | 13 µs | 0.33-0.35 ms |

GRAPH is fresh for every corp at 88-91% of decisions. Engine-design §8
found 93.5-94.4% of `route_max` reads cached.

**Scratch state for hypotheticals.** Outcome features (Section 7) need the
row after a lay, a token, or a train change. Each public call allocates one
scratch `GameState` of the batch's capacity before its nogil loop (a numpy
buffer and `from_buffer`; **Test:** about 1 µs). For each hypothetical, nogil code
copies the row in, applies the change through Section 3's primitives, and
calls `graph_refresh`, `route_max`, or `lay_candidates` on it. A lay marks
every corp stale (`hexmap.pyx:55-57`), so the scratch caches are recomputed.
The live row is never touched beyond its own caches. The solver aborts above
`MAX_ROUTES` (4) trains (`solver.pyx:560-562`), so a hypothetical train is
added only to a corp holding fewer than 4.

## 5. Tokens

### 5.1 Rows

| Rows | Type | Count | Width | Visible (slot 0 = 1) |
|---|---|---|---|---|
| 0 | GLOBAL | 1 | 79 | always |
| 1 | PRICE | 1 | 25 | at BID, RAISE, PAR, TRAIN_PRICE, PRIVATE_PRICE |
| 2 | DIVIDEND | 1 | 8 | at DIVIDEND |
| 3-9 | CORP | 7 | 49 | always |
| 10-16 | PRIVATE | 7 | 18 | unless not in play (UTF below 4 players) |
| 17-22 | TRAIN | 6 | 13 | always |
| 23-74 | HEX | 52 | 104 | always |
| 75-89 | CANDIDATE | 15 | 108 | row k < the candidate count, at TILE and ER_BONUS |
| 90-94 | PLAYER | `max_players` | 37 | real players; padded rows are all zero |

- **Counts.** `num_tokens(max_players) = 3 + 7 + 7 + 6 + 52 + 15 +
  max_players = 90 + max_players`, so 93, 94, and 95 at capacities 3-5.
  The largest index is 94, far below `uint8`'s 255.
- **Width.** `token_dim(1) = 108`, the CANDIDATE width; other rows are
  zero-padded, as in RSS (`core/token_data_v3.pxd:18-43`). A state is
  95 × 108 floats: 20.5 KB as float16, against RSS v3's 11.6 KB (59 × 98).
- **Order.** Players come last, so every other index is the same at every
  capacity (RSS's rule, `core/token_data_v3.pyx:16-17`). The candidate block
  is fixed-size, so it sits before the players.
- **Layout id.** 1, the id `tests/test_game_adapter.py:749-753` already
  passes; `get_token_dim` raises for any other (Section 13, item 2).
- **Identity** comes from row order. Step 8 adds learned identity embeddings
  for hexes (the blueprint's §5 plan), corps, privates, and train types.

### 5.2 Conventions

- **Slot 0** is the key mask. **Slot 1** is `is_selected`, whose meaning per
  type is in its table; at most one row per type sets it.
- **The decision corp** is the operating corp in an OR decision,
  TRAIN_PRICE, PRIVATE_PRICE, BUY_PRIVATE, and ER_BONUS included; the pending
  corp at PAR; the first crowded corp at DISCARD (`trains_first_crowded`).
  **The actor** is `TURN.active_player`, whom MCTS backs values up to
  (engine-design §11). Actor- and decision-corp features are zero when there
  is none.
- **Per-choice features use engine legality, never the policy filter**, so
  they do not depend on the runtime flags (engine-design §7). The legal mask
  carries the restriction.
- **Divisors.** ¥1000 for cash and train-scale money, ¥100 for share prices,
  private values, hex and route revenues, and token prices, ¥7000 for the
  bank; counts by their natural maximum. Values are of order 1: the largest
  in random play is a route revenue at 8.6 (¥860, Section 2), and the bank
  can reach about −3.6 (−¥25,000, engine-design §3) after it breaks. Float16
  transport keeps about three significant digits at any of these sizes.

### 5.3 GLOBAL (79)

| Slots | Feature | Source |
|---|---|---|
| 0 | key mask (1) | |
| 1-16 | decision phase one-hot; all zero at GAME_OVER | `get_decision_phase` |
| 17-19 | round: auction, stock, operating | `round` |
| 20-25 | game phase one-hot | `game_phase` |
| 26, 27, 28 | SR number /10, OR in set /3, ORs in set /3 | `turn`, `or_num`, `or_count` |
| 29, 30 | bank cash /7000, bank broken | `bank_cash`, `bank_broken` |
| 31-33 | player count one-hot (3, 4, 5) | `num_players` |
| 34, 35 | all-pass payouts /3, A's discount /20 | `auction_pass_payouts`, `auction_discount` |
| 36, 37, 38 | SR turn bought, sold then bought; emergency sale since the last purchase | `sr_bought`, `sr_sold_then_bought`, `ebuy_sold_any` |
| 39-78 | copies left / `TILE_COUNT`, per supply tile | supply block |

Train limit, tile colors, and the private-purchase window are functions of
the game phase (`PHASE_*` tables), so they are left to the one-hot.

### 5.4 PRICE (25) and DIVIDEND (8)

The price grids (engine-design §5.2) and PAR share one row. Its points are
the phase-local ids, so a step-8 head reads slot 6 + id for action id.

| Slots | PRICE feature |
|---|---|
| 1-5 | kind: BID, RAISE, PAR, TRAIN_PRICE, PRIVATE_PRICE |
| 6-22 | the yen of phase-local ids 0-16, /1000: BID and RAISE grid points and TRAIN_PRICE and PRIVATE_PRICE points from `price_amount` (`actions.pyx:406-418`); RAISE's drop and unoffered points are 0; PAR's six are 2 × `PAR_PRICE`, the President's Certificate |
| 23 | reference /1000: the private's face (BID, RAISE, PRIVATE_PRICE), the train's list price (TRAIN_PRICE), 0 (PAR) |
| 24 | payer's funds /1000: `player_available` (BID, RAISE), the actor's cash (PAR), the buying corp's cash (TRAIN_PRICE, PRIVATE_PRICE) |

| Slots | DIVIDEND feature (the operating corp's revenue `route_revenue`, r) |
|---|---|
| 1, 2 | r /100; per-share payout ⌈r/10⌉ /100 (engine-design §6.1) |
| 3, 4 | share price after a payout (right) and a withhold (left) /100 (`market_moved`) |
| 5 | treasury after a withhold /1000 |
| 6, 7 | the president's payout and the corp's payout for its pool shares /100 |

### 5.5 CORP (49)

| Slots | Feature | Source |
|---|---|---|
| 1, 2 | is_selected: the decision corp (Section 5.2); the selling corp at TRAIN_PRICE | `pending_train_source` / 6 |
| 3, 4, 5 | parred, floated, home token placed | `par_index`, `floated`, the home hex's mask |
| 6, 7 | par price, share price /100 | `corp_par_price`, `market_price` |
| 8, 9 | market row /10, column /14 | `market_row`, `market_col` |
| 10-13 | zone: normal, par, no-cert-limit, unlimited | `market_type` |
| 14-18 | share price after selling 1-5 shares /100 | `market_moved` down |
| 19 | sold out (no IPO or pool shares): moves up at the SR's end | |
| 20, 21, 22 | cash /1000, IPO and pool shares /10 | |
| 23, 24 | tokens left /3, next token price /100 | `corp_token_price` |
| 25-30, 31 | trains by type /4; over the limit | `trains`, `trains_limit` |
| 32, 33 | last revenue, route max /100 | `last_revenue`, `route_cache` |
| 34, 35, 36 | route available, must buy a train, in an emergency (decision corp) | `route_available`, `must_buy_train`, `in_emergency` |
| 37, 38 | OR position (index + 1)/7, 0 outside the order; operated this OR (index < `or_index`) | `or_order`, `or_index` (`phases/operating.pyx:28-29`) |
| 39, 40, 41 | hexes reached /52, centers reached /25, their revenue /100 | graph block, `map_revenue` |
| 42 | cities reached with a slot for it /5 | `graph_node`, `map_has_slot_for` |
| 43, 44, 45 | sold by the actor this SR; sold this SR turn; sold in this turn's emergency | `sold_mask`, `sr_sold_mask`, `ebuy_sold_mask` |
| 46, 47 | the actor's largest sale and largest emergency sale /5 | `corp_max_sale`, `corp_emergency_sale` |
| 48 | privates owned /3 | |

### 5.6 PRIVATE (18) and TRAIN (13)

| Slots | PRIVATE feature | Source |
|---|---|---|
| 1, 2 | is_selected: BID's pending, RAISE's auctioned, PRIVATE_PRICE's pending private; topmost unsold (AUCTION's buy) | `pending_private`, `auction_private`, `topmost_private` |
| 3-6 | unsold, player-owned, corp-owned, closed | `owner` |
| 7, 8, 9 | face, revenue now, price now (A's discount) /100 | `PRIVATE_VALUE`, `private_revenue`, `private_price` |
| 10, 11, 12, 13 | high bid /100, bids /5, minimum bid /100, the actor's bid /100 | `private_high_bid`, `private_bid_count`, `private_min_bid`, `bids` |
| 14, 15 | blocking its hex (TR, ER); MF's port tile used | `private_blocks_hex`, `flags` |
| 16, 17 | the decision corp may buy it; the actor may exchange it (DR) | `can_buy_private`, `can_exchange_dr` |

| Slots | TRAIN feature | Source |
|---|---|---|
| 1 | is_selected: TRAIN_PRICE's train type | `pending_train_source` % 6 |
| 2, 3, 4, 5 | list price /1000; depot copies /6 (D: 0); unlimited (D); pool copies /4 | `TRAIN_PRICE`, `depot`, `pool` |
| 6, 7 | the depot sells it now; cheapest available | `trains_depot_types`, `trains_cheapest_type` |
| 8, 9 | rusted; rusts at the next phase | `TRAIN_RUSTS_ON` against `game_phase` |
| 10, 11, 12 | Δ route max /100 if the decision corp adds one, trades one in for a D, or discards one (Section 7.2) | scratch `route_max` |

### 5.7 HEX (104) and the tile block

The **tile block** (76 slots) describes a tile as laid. HEX rows hold the
current tile's at slots 2-77 and CANDIDATE rows the candidate's at the same
slots, so step 8 can project both with one layer.

| Tile-block slots | Feature | Source |
|---|---|---|
| +0 to +39 | supply tile one-hot; all zero for a printed tile | `tile` < 40 |
| +40 | printed tile | `tile` ≥ 40 |
| +41 to +61 | path mask, absolute sides, PAIR order | `TILE_MASK[tile][rotation]` (`g1889/data/__init__.py`, "Endpoints and paths") |
| +62 to +67 | color: white, yellow, green, brown, gray, red | `TILE_COLOR` |
| +68 to +70 | center: city, town, offboard | `TILE_RC` |
| +71, +72 | revenue now /100 (an offboard's by phase); slots /3 | `map_revenue`, `TILE_SLOTS` |
| +73 to +75 | label H, K, T | `TILE_LABEL` |

| HEX slots | Feature | Source |
|---|---|---|
| 1 | is_selected: TILE's pending hex; C4 at ER_BONUS | `pending_hex`, `ER_HEX` |
| 2-77 | tile block | |
| 78 | revenue to a D train /100 | `map_revenue(..., is_d=True)` |
| 79, 80 | mountain, water | `TILE_TERRAIN` of the printed tile |
| 81, 82, 83 | tokens /3, free slots /3, home reservation open | `hex_tokens`, `map_free_slots` (`hexmap.pyx:74-79`) |
| 84 | lay cost /100, before SMR | `TILE_COST[tile]` |
| 85 | blocked by a private | `private_blocks_hex` |
| 86-91 | dangling exits: track at a side whose hex across has none facing back | `hex_paths` of both |
| 92 | upgrade targets now /15: in supply, of a phase color, exits on open sides, not `blocks_lay`; connectivity and cost aside | `UPG_*`, `supply_count` |
| 93-98, 99 | the decision corp's reached sides; its center reached | `graph_sides`, `graph_node` |
| 100 | blocked for the decision corp | `map_blocked_for` |
| 101 | the decision corp's lay candidates here /15, at TRACK | `lay_candidates` |
| 102, 103 | the decision corp may token here; Δ route max /100 if it does, at TOKEN | `can_token`; scratch `route_max` |

The blueprint's "laid this OR" is not in the row (engine-design §3,
"Not stored"), so it is dropped. The decision phase says where the
operating corp is in its turn, which is all the row keeps.

### 5.8 CANDIDATE (108)

| Slots | Feature | Source |
|---|---|---|
| 1 | an ER_BONUS candidate (0 at TILE) | |
| 2-77 | tile block of the candidate at its rotation; revenue and slots are the new tile's | `LayCandidate` |
| 78-98 | added paths: in the new mask, not the old | |
| 99 | lay cost /100 (0 for ER) | `map_lay_cost` |
| 100 | copies left after the lay / `TILE_COUNT` | |
| 101, 102, 103 | the decision corp's Δ hexes reached /52, Δ centers reached /25, Δ reached revenue /100 | scratch GRAPH |
| 104, 105 | Δ route max /100; route available after | scratch `route_max`, graph block |
| 106, 107 | other corps with tokens whose reached centers grow /6; their Δ centers reached /25 | scratch GRAPH of the six |

### 5.9 PLAYER (37)

| Slots | Feature | Source |
|---|---|---|
| 1 | is_selected: the actor | `active_player` |
| 2-6 | seats after the actor, one-hot (0 = the actor) | |
| 7 | priority holder | `priority_player` |
| 8, 9 | cash, value /1000 | `player_value` (`player.pxd:22`) |
| 10, 11 | certificates / limit; certificates /25 | `player_cert_count`, `CERT_LIMIT` |
| 12-18, 19-25, 26-32 | shares /10, sold this SR, bids /100, by corp or private | `shares`, `sold_mask`, `bids` |
| 33, 34 | passed, bankrupt | |
| 35, 36 | presidencies /7, privates owned /3 | |

### 5.10 What needs GRAPH or the solver

| Feature | Needs | When | Cost |
|---|---|---|---|
| reach relations, CORP 39-42, HEX 93-100 | graph block | every state | the prologue |
| CORP 33, 34 | `route_cache`, `route_available` | every state | the prologue |
| HEX 101 | `lay_candidates` per hex (semi-restrictive walks) | TRACK | **Test:** about TRACK's enumeration, 1.5 µs median (engine-design §5.4) |
| HEX 103 | scratch token, `route_max` | TOKEN | Section 9 |
| CANDIDATE 101-107 | scratch lay, GRAPH × 7, `route_max` × 1 | TILE, ER_BONUS | Section 9 |
| TRAIN 10-12 | scratch train change, `route_max` | BUY_TRAIN, DISCARD | Section 9 |

## 6. Relations

### 6.1 Dynamic relations (IPC)

`num_relations` = 25. Binary planes hold 1; the two count planes hold raw
counts, which the model scales (shares by 1/10, trains by 1/4), as v3 does
(`core/attention_relations.py:50-57`). A record never has value 0, and no
(relation, query, key) cell repeats, so `count_nonzero(dense)` equals the
record count, as conformance asserts (`tests/test_game_adapter.py:419`).

| Id | Relation | Query → key | Value | Source | Bound |
|---|---|---|---|---|---|
| 0, 1 | PLAYER_HOLDS, CORP_HELD_BY | player ↔ corp | 1 | `shares` > 0 | 35 each |
| 2, 3 | PLAYER_SHARES, CORP_SHARES | player ↔ corp | shares, 1-10 | `shares` | 35 each |
| 4, 5 | PRESIDES, PRESIDENT | player ↔ corp | 1 | `president` | 7 each |
| 6, 7 | PLAYER_OWNS, OWNED_BY_PLAYER | player ↔ private | 1 | `owner` | 7 per direction with 8, 9 |
| 8, 9 | CORP_OWNS, OWNED_BY_CORP | corp ↔ private | 1 | `owner` | |
| 10, 11 | BIDS_ON, BID_BY | player ↔ private | 1 | `bids` > 0 | 35 each |
| 12, 13 | CORP_TRAINS, TRAIN_HELD_BY | corp ↔ train | count, 1-4 | `trains` | 28 each |
| 14, 15 | STATION, STATION_OF | corp ↔ hex | 1 | `tokens` | 15 each |
| 16, 17 | REACHES_HEX, HEX_REACHED_BY | corp ↔ hex | 1 | graph sides ≠ 0 | 364 each |
| 18, 19 | REACHES_CENTER, CENTER_REACHED_BY | corp ↔ hex | 1 | graph node bit | 175 each |
| 20 | TRACK_LINK | hex ↔ hex | 1 | both hexes have track at their shared side | 232 |
| 21, 22 | CANDIDATE_AT, CANDIDATE_OF | candidate ↔ the laid hex | 1 | | 15 each |
| 23, 24 | CANDIDATE_EXIT, EXIT_OF | candidate ↔ each hex across its exits | 1 | `TILE_EXITS` | 90 each |

- **Reach is GRAPH's,** Ruby's `connected_hexes` and `connected_nodes`
  (`graph.pxd:1-10`). A reached hex is one where the corp's walks reach some
  side: the network's own hexes and the hexes across its dangling exits
  (`graph.pyx:84-104`). That is where a lay can start, not "on the
  network" (Section 12). Corps without a token reach nothing.
- **TRACK_LINK** is one symmetric plane, both entries set; direction comes
  from the static side planes.
- Ids 21-24 exist only at TILE and ER_BONUS, for visible candidate rows.
- Relations come in both directions, as in RSS and blueprint §6.1, so each
  side can attend to the other; TRACK_LINK is its own reverse.

### 6.2 Static relations (model buffers)

`static_relation_planes(max_players)` returns them as `(10, T, T)` uint8,
so the model does not repeat the row map. Step 8 registers them as
non-persistent buffers, as v3 does its synergy planes (`token-data-v3.md`,
"Token Order"); they never cross IPC.

| Plane | Query → key | Entries |
|---|---|---|
| HEX_SIDE_0 to HEX_SIDE_5 | hex → the hex across side s (`HEX_NEIGHBOR`) | 244 in all |
| HOME, HOME_OF | corp ↔ its home hex (`CORP_HOME`) | 7 each |
| ABILITY_HEX, ABILITY_OF | private ↔ hex: TR → K4, ER → C4 (`HEX_BLOCKER`), MF → its 4 port hexes (`MF_PORT_HEX`) | 6 each |

### 6.3 The edge bound

The bound column sums to **1,844**: 140 for shares, 14 presidency, 14
ownership, 70 bids, 56 trains, 30 stations, 728 reached hexes, 350 reached
centers, 232 track links, and 210 candidate edges. It assumes:

- at most 4 trains per corp (`MAX_ROUTES`), so 4 types;
- 15 charter tokens (`CORP_TOKENS`);
- 232 directed hex sides open on both sides;
- `MAX_LAY_CANDIDATES` candidates of at most 6 exits.

**`max_relation_edges` = 2,048**, the next power of two, as RSS rounds 248
to 256 (`core/relations.pxd:31-35`). The emitter asserts the count.

**Measured** (records per state):

| Group | Median | p99 | p99.9 | Max | Bound |
|---|---|---|---|---|---|
| reached hexes (16, 17) | 46-56 | 364-444 | 544-598 | 646-704 | 728 |
| reached centers (18, 19) | 16-20 | 150-178 | 236-256 | 282-332 | 350 |
| track links (20) | 22-26 | 96-104 | 118-120 | 130-132 | 232 |
| **total** | 174-184 | 720-836 | 1,012-1,079 | 1,182-1,300 | 1,844 |

The other groups peak far below their bounds: 44-50 records per pair of
share planes, 14 presidency, 14 ownership, 30 bids, 24 trains, 30 stations,
and 98 candidate edges. By phase, the median total climbs from 52-68 in
phase 2 to 558-720 in phase D. Reach dominates, and random play already
comes within 3% of its bound, so a measured cap would overflow on
self-play's denser maps.

### 6.4 Dense and sparse

- **One emitter.** RSS writes its dense and sparse loops twice
  (`core/relations.pyx:392-602`). Here `_emit(state, num_players, edges)`
  writes up to 2,048 `(relation, query, key, value)` records to a stack
  buffer (8 KB). Sparse copies them and zeroes the tail; dense zeroes its
  planes and scatters them. The two are equal by construction, and
  conformance checks it.
- **Sizes per state:**

| | 1889 | RSS v3 |
|---|---|---|
| Sparse records | 8 KB | 1 KB |
| Dense planes | 25 × 95² = 225,625 B | 12 × 59² = 41,772 B |

- **Training.** The trainer's dense batch is 5.4 times RSS's: 58 MB of
  uint8 at batch 256, and four times that once normalized to fp32
  (`nn/transformer-v3.py:899-906`). On the local ROCm host, fp32
  dense-relation batches above about 150 are already a known problem. Sparse
  records carry the same edges at 1/27 the bytes, and the model already
  accepts them. Section 13, item 11.

## 7. Per-choice features and step 8's readouts

### 7.1 TILE and ER_BONUS candidate rows

- **Mapping.** TILE slot k is candidate row k: `_enumerate_tile` emits ids
  0 to count − 1 over `enumerate_lay_candidates` (`actions.pyx:751-761`).
  ER_BONUS slot 1 + k is row k over `er_candidates` (`actions.pyx:1183-1213`;
  `ER_BONUS_CANDIDATE` = 1, `constants.pxd:194-197`). Extraction calls the
  same functions, so the order cannot drift.
- **Outcomes** (CANDIDATE 99-107). For each candidate, on the scratch
  state: `map_put_tile`, then `graph_refresh` and `route_max` for the
  decision corp, and `graph_refresh` for each other corp with a token.
  Deltas are against the prologue's caches. ER's lay is the seller's, but
  the decision corp is the buyer, the corp that gains or loses, so its
  deltas are the ones reported.
- **Not in v0:**
  - opponents' Δ route max: refreshing and solving all seven corps costs
    7.3-7.7 µs median per candidate, against 1.5-1.7 µs for the decision
    corp alone (Section 13, item 5);
  - the blueprint's "next train tier", a second solve per candidate with a
    hypothetical train (item 6);
  - build-distance deltas (Section 8).

**Step 8's head** (sketch): logit_k = MLP([decision corp, actor, candidate
row k, pending hex, GLOBAL]) for k < 15, with ER_BONUS's pass from a context
MLP and its candidates from the same scorer. The trunk has already let each
candidate attend to its hex and to the hexes across its exits (relations
21-24), and the masked rows past the count never reach the softmax.

### 7.2 Other decisions

| Decision | Choices (engine-design §5.2) | Features the head reads | Extra cost |
|---|---|---|---|
| AUCTION | pass, buy topmost, bid on p | PRIVATE rows (price, bids, topmost), PLAYER | none |
| BID, RAISE | grid points | PRICE 6-24, the selected private | none |
| STOCK | per corp: IPO, pool, par, sell 1-5; DR; MF | CORP rows (price, zone, sale prices 14-18, the actor's sales 43-47), PLAYER | none |
| PAR | six par prices | PRICE (2 × par), the actor's cash and certificates | none |
| TRACK | pass, hex h, buy private p, DR, MF | HEX rows (the decision corp's view 93-101), PRIVATE 16 | `lay_candidates` per hex |
| TILE | candidate k | CANDIDATE rows (7.1) | 7.1 |
| TOKEN | pass, hex h, buy private, DR | HEX 102-103, CORP 23-24 | a solve per tokenable city |
| DIVIDEND | withhold, payout, buy private, DR | DIVIDEND row | none |
| DISCARD | train type | TRAIN 12 | a solve per held type |
| BUY_TRAIN | depot t, trade-in i, pool t, corp c's t, sale (c, n), bankrupt, private, DR | TRAIN 2-11, CORP rows and relations 12-13, sale prices | a solve per offered type and trade-in |
| TRAIN_PRICE, PRIVATE_PRICE | price points | PRICE | none |
| BUY_PRIVATE | done, private p, DR | PRIVATE rows | none |
| PORT | port hex i × rotation r | the 4 port HEX rows and tile 437's mask at r, a static table | none |

- **PORT** needs no candidate rows. Its ids are fixed, `6i + r`
  (`constants.pxd:200-202`), and random play reached at most 10 of the 24.
  A port lay changes no route, and it is one decision per game.
- **Train deltas** add one train of each type the policy set offers
  (depot, pool, or another corp's), or swap a traded 4, 5, or 6 for a D,
  never past 4 trains. Adding every type instead, offered or not, reached
  6.3-6.7 ms per decision, against 0.45-0.59 ms when restricted. The
  deltas ignore what a purchase does to the phase (rusting, limits), which
  the TRAIN rows' flags show.

## 8. Build distance and shortest-build-path planes

**Recommendation: none of blueprint §7-8 in step 7.** Step 8's v0 reads
none of it, and pipeline-refactor §8 trains v0 first. Build distance then
lands as its own step (here called 7b) before the comparison.

**Why wait:**

- Its approximation study (blueprint §7.3) needs self-play maps, which may
  be denser than random play (engine-design §6).
- It would be extraction's largest cost. A 0-1-2 search over 52 × 7
  endpoints with up to 52 × 21 in-hex transitions and 232 links, for 7
  corps, at every state: **Test:** estimated 10-20 µs in C, against the
  prologue's 0.5 µs median. Its inputs change with every lay and supply
  change, so a cache would need a row block like GRAPH's.
- v0 already covers the one-lay horizon: reach relations 16-19 (where a lay
  can start, and which centers are reached) and the candidate outcomes.

**Hooks step 7 leaves:**

- **Relation ids** append after 24, so v0 keeps reading ids 0-24 and
  filters the rest, as v2 does (`nn/transformer-v2.py:832`). Buckets
  1, 2, 3+ for corp ↔ hex add at most 728 records (one bucket per pair and
  direction), for 2,572. The §8 planes, hex → hex from frontier to target
  for the decision corp, add at most 52 × 25 = 1,300 more, for 3,872: E
  becomes 4,096.
- **Step 8's model** must therefore take E from the input and the relation
  count it consumes from its own constants, not from `Game` (Section 13,
  item 10).
- **Corp summaries** (revenue within 0, 1, 2 lays) and unfloated corps
  "as if stationed at home" (blueprint §7.1) go in a new token layout, id 3
  (Section 13, item 2). GRAPH
  needs a token (`graph.pyx:167-172`), so the home case runs on a scratch
  row with a hypothetical home token.
- **Validation** as blueprint §7.3:
  - exact minimum-lay search to depth 3 on self-play and fixture states;
  - distance 0 equals GRAPH;
  - a hex reported 1 lay away has a lay that brings it to 0.

## 9. Cost budget

**Measured** per returned decision, on random-playout states (Section 2),
one core, from Python, so each call includes about 0.08 µs of overhead:

| Work | Where (share of decisions) | Median | p99 | Max |
|---|---|---|---|---|
| Prologue (Section 4) | every state | 0.48-0.50 µs | 19-20 µs | 0.43-0.44 ms |
| TILE: lay, refresh, and solve the decision corp, for all candidates | TILE (5-7%) | 5.2-5.7 µs | 44-50 µs | 0.31-0.89 ms |
| TILE: refresh the six others, per candidate | TILE | 1.6 µs | 15-19 µs | 0.10-0.19 ms |
| TOKEN: a solve per tokenable city | TOKEN (1.2-1.6%) | 1.6-1.7 µs | 27 µs | 0.14-0.20 ms |
| BUY_TRAIN: a solve per offered type and trade-in | BUY_TRAIN (6-8%) | 1.5-1.6 µs | 25-26 µs | 0.45-0.59 ms |
| DISCARD: a solve per held type | DISCARD (0.1-0.2%) | 2.4-3.0 µs | 7-9 µs | 20-27 µs |

**RSS baseline** (v3 layout and engine options, 58,014 random-playout rows,
capacity 5): `fill_tokens` 2.88 µs and `fill_relations_sparse` 1.48 µs
median per single state; in 64-row batches, per state, tokens 2.57 µs,
sparse 1.38 µs, dense 1.96 µs.

**Targets** for 7.4's profile, per state, nogil end to end as in RSS:

| | Target | Basis |
|---|---|---|
| Tokens + sparse records, prologue included: median | ≤ 10 µs | about twice RSS's 4.4 µs for 1.8 × the floats and a larger emitter |
| The same: p99 | ≤ 100 µs | the prologue's and candidates' tails |
| TILE states: median | ≤ 25 µs | base + 5.7 µs + 3.6 candidates × 1.6 µs |
| Batch, per state | ≤ single | one scratch state and one Python dispatch per batch |
| Dense planes, per state | about 10 µs (**Test:**) | RSS's 1.96 µs scaled by 5.4 × the bytes |

- **Against the engine.** A decision costs 0.84-0.96 µs median in
  `apply_action`, and a playout runs at about 390,000 decisions a second
  (engine-design §6). Extraction runs once per leaf, beside a forward pass
  over 95 tokens, which costs (95/59)² ≈ 2.6 times RSS's attention.
- **Mean.** About 10 µs per state, by the decision mix: the base fills
  (**Test:** about 7 µs), the prologue's 1.6-1.8 µs mean, TILE's extras
  (about 18 µs mean: 7.7-8.5 µs for the decision corp and 2.6-2.9 µs per
  candidate for the others) on 5-7% of states, and under 0.5 µs for the
  other phases together.
- **Tails.** The tails come from the solver on dense phase-D maps where a D
  runs beside a second long train (engine-design §8; `buy_train_tail.py`'s
  slowest are D + D, 6 + D, 5 + D, and 4 + D). No time limit is planned
  (engine-design §8's conclusion); recheck on self-play states once step 8
  trains.

## 10. Testing plan

1. **Invariants** (`tests/g1889/test_extraction.py`, on random-playout
   states at 3-5 players, capacities 3 and 5, flags on and off):
   - every value is finite and within a pinned bound per divisor family;
   - padded rows and padding slots are zero, and slot 0 follows Section 5.1;
   - each `is_selected` group has at most one 1;
   - `get_token_widths` matches what the fills write;
   - **determinism:** a row extracted twice, and its copy with every dirty
     bit set, give equal outputs;
   - **write-back:** after extraction, every slot but the caches is
     unchanged, and the caches equal a recomputation
     (`tests/g1889/phases/test_random_playouts.py:94-109`).
2. **Outcome features equal the move they describe** (end-to-end, not
   mirrors of the code). For each candidate k, apply TILE k (or ER_BONUS
   1 + k) to a copy:
   - the hex row's tile block afterwards equals candidate row k's;
   - Δ route max equals `max_route_revenue` after minus before, and the
     reach deltas equal GRAPH's.

   Likewise:
   - TOKEN's Δ against applying TOKEN 1 + h;
   - DIVIDEND's prices against applying payout or withhold;
   - PRICE amounts against `price_amount_py` for each policy id;
   - CORP sale prices against applying STOCK sales;
   - TRAIN deltas against `TRAINS.set_corp_trains` on a copy, since a real
     purchase can change the phase.
3. **Relations against entity APIs** on the same states: each edge's
   presence and value decoded against `PLAYERS`, `CORPS`, `PRIVATES`,
   `TRAINS`, `MAP`, and `GRAPH`. **Static planes** against the tables.
4. **Conformance** (`tests/test_game_adapter.py`):
   - CASES' two 1889 entries get layout 1 (`:108-109`), which un-skips
     every extraction check: scratch states reproduce the inputs, dense and
     sparse encode the same edges, and batches over mixed player counts at
     capacity 5 equal single states (`PLAYER_SETUPS`, `:116`).
   - `test_g1889_extraction_raises_until_step_7` (`:743-757`) becomes a pin
     on the sizes (95 tokens at capacity 5, width 108, 25 relations, 2,048
     edges).
5. **Oracles from 18xx.games** (`tests/g1889/replay/`, the cached extracts;
   skip without them). On `load_state` of each snapshot, decode extraction's
   output, which checks token indices and plumbing, not the engine again:
   - at every OR snapshot, relations 16-19 for every corp with a token
     against Ruby's `connected_hexes` and `connected_nodes`, as
     `test_network_oracles.py:76-98` checks GRAPH;
   - at Track snapshots, HEX 101 against Ruby's legal lays per hex
     (`ruby_lays`), and after TRACK 1 + h on a copy, the candidate rows'
     tiles and rotations against Ruby's lays at h;
   - at Token snapshots, HEX 102 against Ruby's token cities;
   - at Route snapshots, CORP 33 against AutoRouter's revenue
     (`test_route_oracles.py:83-97`).

   No other feature has a Ruby counterpart worth extracting.
6. **RSS guard.** Nothing under `core/`, `entities/`, `phases/`, or the v2
   and v3 model files changes, and `tests/test_v2_parity.py` passes with its
   fixture. The RSS conformance cases and `tests/test_pipeline_boundary.py`
   stay green, and only `games/` imports `g1889`. The engine additions keep
   the 1889 suite and the differential replays green, since `MAP.lay`,
   `MAP.place_token`, `TRAINS.set_corp_trains`, and `MARKET.move_*` call
   them.
7. **Profile** (7.4, `scratchpad/`, not the suite): Section 9's targets on
   the playout states, plus the trainer's sample-and-extract rate at batch
   512, dense against sparse (pipeline-refactor §9's open **Test:**).

## 11. Sub-steps

Each sub-step is one commit or a short series. Each leaves `pytest -q` and
pyright green and RSS untouched, with `tests/test_v2_parity.py` passing
on its existing fixture.

| # | Scope | Acceptance |
|---|---|---|
| 7.1 | Section 3's engine additions; `token_data` at its final sizes, with every row type but CANDIDATE, PRICE, and DIVIDEND (hidden and zero until 7.2-7.3) and no per-choice slots (CORP 14-18, HEX 101-103, TRAIN 10-12: zero); the prologue and write-back; `relations` at 25 ids with 0-20 emitted, and the static planes; the adapter fields | conformance's extraction checks pass for both 1889 cases; Section 10, items 1, 3, and 6; the engine suite and differential replays unchanged |
| 7.2 | CANDIDATE rows at TILE and ER_BONUS with their outcomes; relations 21-24; the scratch state | item 2 for TILE and ER_BONUS; the edge-bound assert never fires in a soak like engine-design §10.4's |
| 7.3 | PRICE and DIVIDEND rows; HEX 101-103; TRAIN 10-12; CORP sale prices | item 2 for the rest |
| 7.4 | the oracle tests; the profile, recorded in this document; `token-data-1889.md` as the contract step 8 reads; status lines here and in pipeline-refactor | oracles equal at every snapshot; Section 9's targets met, or the gap recorded with its cause |
| 7b | build distance and the §8 planes, after step 8's first run | Section 8's validation; its own design note first |

## 12. Where the blueprint is wrong or underspecified

1. **Token count** (§1, §5: "about 78"). 95 at capacity 5: the blueprint's
   types plus 15 candidate rows and the PRICE and DIVIDEND rows.
2. **"Laid this OR"** (§5, hex features) is not in the row (engine-design
   §3, "Not stored"); dropped.
3. **"Ability-used flags"** (§5, privates): only MF's exists
   (`state.pxd:90-93`). DR closes when exchanged, and ER's bonus is
   ER_BONUS itself.
4. **"Operated" flags** (§5, corps) are derived from `or_order` and
   `or_index`. The **train limit** (§5, global) is a function of the phase.
5. **Build distance 0, "on network"** (§6.3). The v0 reach planes are
   GRAPH's: they include the hexes across dangling exits
   (`graph.pyx:84-104`), so they mean "a lay can start here".
6. **Edge budget** (§6.4). Without build distance, v0 already bounds at
   1,844 (2,048 rounded), and reach accounts for 1,078 of it. Build distance
   adds 728, and the §8 planes up to 1,300.
7. **Dense size** (§6.4: "R ≈ 25, N ≈ 80, about 160 KB"). 225 KB at
   N = 95; Section 6.4 weighs it.
8. **Tile slots** (§9: "52 + 240"). Engine-design §5.4 settled a compact
   15-slot list: 52 + 15, with a unified dimension of 421.
9. **"Next train tier"** (§9) and any hypothetical train must stay within 4
   trains: the solver aborts above `MAX_ROUTES` (`solver.pyx:560-562`). They
   must also use offered types, or solves reach 6.7 ms.
10. **Unfloated corps "as if stationed"** (§7.1). GRAPH and the solver need
    a token, so this needs a scratch row with a hypothetical home token; it
    goes to 7b.
11. **Underspecified, settled here:**
    - the price grids, PAR, DIVIDEND, and stock-sale outcomes (Sections
      5.4-5.5);
    - PORT and ER_BONUS (Section 7);
    - the actor and decision-corp features (Section 5.2);
    - who owns the static planes (Section 6.2);
    - the layout id;
    - the cache write-back and the scratch state (Section 4);
    - relation ids that later steps can extend (Section 8).

## 13. Decisions

Each item gives the options and a recommendation; the user accepts or
amends each.

1. **Row set and order.** (a) Section 5.1: three shared rows (GLOBAL,
   PRICE, DIVIDEND), then corps, privates, trains, hexes, 15 candidates,
   and players last; 95 at capacity 5. (b) One phase row per decision, as
   RSS's seven, about 100. (c) No phase rows: price points and dividend
   outcomes on GLOBAL and the corp. *Recommend (a):* each row has one
   meaning per kind, as RSS's shared PAR/IPO row, and (c) would make one
   projection read phase-dependent slots.
2. **Layout ids.** *Recommend 1* for v0, which the existing test already
   passes, with `get_token_dim` raising for any other id. Step 8's module
   sets `INPUT_LAYOUT_VERSION = 1`. A module that forgets gets the loader's
   default, 2 (`nn/__init__.py:192`), which names no 1889 layout, so it
   fails loudly. Later 1889 layouts skip 2 to keep that true: build
   distance's corp summaries take id 3.
3. **Reach for all corps, both directions** (relations 16-19), the bulk of
   the edges.
   - (a) All seven corps, both directions.
   - (b) Corp → hex only: the bound drops by 539, to 1,305.
   - (c) The decision corp's view only, on HEX rows.

   *Recommend (a):* track is public (blueprint §7.1), the reverse lets a hex
   gather who can build there, and records are cheap (8 KB at 2,048).
4. **`max_relation_edges`.** (a) 2,048, above the 1,844 worst case, with an
   assert. (b) A measured cap (p99.9 1,079), with records dropped past it.
   *Recommend (a):* random play already comes within 3% of the reach bound,
   and dropping edges silently changes the input.
5. **Candidate outcomes in v0.**
   - (a) The decision corp's reach and route deltas, plus the other corps'
     reach growth: about 11 µs median per TILE state.
   - (b) Also the others' Δ route max: about 26 µs median, p99 about
     0.2 ms.
   - (c) The tile block only.

   *Recommend (a).* Revisit (b) if v0 lays track that helps opponents.
6. **"Next train tier" deltas on candidates** (blueprint §9). *Recommend
   deferring* to 7b or later: they double TILE's solves and need the
   4-train guard.
7. **Outcome summaries at TRACK** (the best candidate per hex).
   *Recommend no:* there are 11-12 candidates in all at the median and up
   to 77, a TILE child shows them anyway, and search backs its value up to
   TRACK. Give TRACK the candidate count (HEX 101) only.
8. **The other per-choice features** (Section 7.2: TOKEN, train deltas
   restricted to offered types, PRICE amounts, DIVIDEND outcomes, sale
   prices). *Recommend all*; they cost 1.5-3 µs median where they apply,
   and nothing elsewhere.
9. **PORT.** *Recommend* a pointer over the 4 port hex rows × a static
   rotation table in the model, not candidate rows.
10. **What step 8's model must accept**, so step 7's ids and sizes can grow:
    - E from the input's shape, not a constant;
    - relation ids past those it consumes, filtered as v2 does;
    - the static planes from `static_relation_planes`;
    - `is_selected` and the decision corp as in Section 5.2.

    *Recommend adopting* these as step 8's requirements.
11. **Relation transport in training.** (a) Dense planes, as now: 225 KB
    per sample. (b) Sparse records for 1889: replay and trainer call
    `fill_relations_sparse_batch`, 8 KB per sample. *Recommend (b)* in step
    8, chosen per game so RSS's training is unchanged, unless 7.4's trainer
    profile shows dense keeping up and fp32 dense batches fit locally. Step 7
    provides both and proves them equal.
12. **Build distance timing.** (a) In step 7. (b) 7b, after step 8's first
    run, with Section 8's hooks. *Recommend (b).*
13. **Module names.** *Recommend* `g1889/token_data` and `g1889/relations`,
    with the engine additions in their owning entity modules (Section 3), and
    `token-data-1889.md` as the contract (7.4).
14. **Player counts.** *Recommend* 3-5 players in extraction, as RSS's
    asserts: the pipeline trains 3-5, and the one-hots (GLOBAL 31-33,
    PLAYER 2-6) are sized for it. 2- and 6-player rows stay engine-only.
15. **Divisors.** *Recommend* Section 5.2's families. Step 8 sets the
    relation scales: shares by 1/10, trains by 1/4.
