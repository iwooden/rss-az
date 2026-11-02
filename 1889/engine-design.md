# Shikoku 1889 Engine Design

Status: design for step 6 of [`pipeline-refactor.md`](pipeline-refactor.md),
dated 2026-11-01; the user accepted every recommendation in Section 12 the
same day. Nothing is implemented yet. Implementation agents work
through the sub-steps in Section 11, and each sub-step ends with its
acceptance criteria green.

The rules are [`1889-rules.md`](1889-rules.md). Where they are silent or
disagree with 18xx.games, the engine follows 18xx.games (rules §14,
decision 1). The blueprint is [`1889.md`](1889.md), and the adapter contract
is pipeline-refactor §3.

The RSS boundaries in `AGENTS.md` carry over:

- a raw `int16` row;
- entity APIs for mutations;
- legality in `actions`;
- handlers that assume legal actions;
- sparse phase-local ids.

No Cython is shared with RSS (pipeline-refactor §6.1).

RSS paths are relative to the repo, and Ruby paths are relative to
`submodules/18xx/lib/engine/`. Section 12 records the user's decisions, and
**Test:** marks a claim to measure.

## 1. Package layout

```
g1889/
  constants.pxd/.pyx  rounds, engine steps, decision phases, action sizes, owner codes
  state.pxd/.pyx      layout structs, module-level offsets, GameState
  actions.pxd/.pyx    ActionInfo, encode/decode, legality, enumerators, lay candidates
  driver.pxd/.pyx     GameDriver: validate, dispatch, auto-chain, route override
  data/               tables.json, generated tables.pxd/.pyx, generate.py, dump_1889.rb
  entities/           bank, player, corp, private, market, trains, phase, hexmap, graph, turn
  phases/             auction, stock, operating, track, token, route, dividend,
                      buy_train, discard, buy_private, abilities
  routes/             solver.pxd/.pyx
```

- **The RSS split, kept where it pays.** Entity handles own the row
  (`entities/corp.pxd:112`). Phase handlers are
  `cdef void apply_*(GameState, ActionInfo*) noexcept` and assume legal
  actions (`phases/par.pxd:12`).
- **The four root modules** play the role of RSS `core/`. At the package
  root they keep one `core` in the repo and short cimports
  (`from g1889.state cimport GameState`). The solver gets `routes/` because
  it is its own sub-step and step 7's extraction will call it.
- **`setup.py`.** `EXTENSION_DIRS` gains `'g1889'` (`setup.py:17`).
  `packages` gains `g1889` and its four subpackages (`setup.py:245`). Each
  subpackage gets an `__init__.pxd`. `clean` deletes only `.c`, `.so`, and
  `.html` files, so the generated `.pyx` and `.pxd` files survive it.

## 2. Static data codegen

**Two stages.** Stage 1 is unchanged: the Ruby dump becomes `tables.json`
(`generate.py:85`, `:209`). The new stage 2, `render_cython`, turns
`tables.json` into `tables.pxd` and `tables.pyx` and derives the lookup
tables below. `python -m g1889.data.generate` runs both, and `--cython-only` runs
stage 2 without Ruby. The output is checked in, and the build never runs the
generator. `tables.json` stays as the reviewable artifact and stage 2's
input. Python tests and tools keep using `load_tables()`, but the engine
never opens the JSON.

**RSS pattern.** `core/data.pxd:153-171` declares the C arrays and
`core/data.pyx:157-300` initializes them, so modules write
`from g1889.data.tables cimport HEX_NEIGHBOR, TILE_MASK`. Dimensions and rule
constants are a `cpdef enum`. Python mirrors are set through `globals()`
(`core/data.pyx:285`). Ragged tables are stored in CSR form.

| Group | Names (C type, shape) |
|---|---|
| Dimensions | `NUM_HEXES` 52, `NUM_RCS` 25, `NUM_SUPPLY_TILES` 40, `NUM_TILES` 92, `NUM_CORPS` 7, `NUM_PRIVATES` 7, `NUM_TRAIN_TYPES` 6, `NUM_GAME_PHASES` 6, `MARKET_ROWS` 11, `MARKET_COLS` 15, `NUM_PAR_PRICES` 6, `NUM_PAIRS` 21, `NODE` 6, `NUM_PORTS` 4, `MAX_CITY_SLOTS` 3, `MAX_LAY_CANDIDATES` 15, `MAX_ROUTE_HEXES` 104 |
| Rules | `BANK_CASH`, `MIN_BID_INCREMENT`, `MARKET_SHARE_LIMIT`, `MAX_OWNERSHIP_PERCENT`, `FLOAT_PERCENT` |
| Hexes | `HEX_NEIGHBOR` int8[52][6], `HEX_EXIT_MASK` uint8[52], `HEX_RC`, `HEX_HOME_CORP`, `HEX_BLOCKER` int8[52] |
| Revenue centers | `RC_HEX`, `RC_TYPE`, `RC_PORT` int8[25]; `RC_REVENUE` int16[25][6]; `RC_D_REVENUE` int16[25] |
| Tiles | `TILE_COLOR`, `TILE_LABEL`, `TILE_RC`, `TILE_SLOTS`, `TILE_TERRAIN`, `TILE_BLOCKS_LAY` uint8[92]; `TILE_COUNT`, `TILE_REVENUE`, `TILE_COST` int16[92]; `TILE_MASK` uint32[92][6]; derived `TILE_EXITS` and `TILE_CANON_ROT` (the lowest rotation with an equal mask) uint8[92][6] |
| Upgrades | CSR over `tile·6 + rotation`, canonical targets only: `UPG_START` int16[553]; `UPG_TILE`, `UPG_ROT` uint8[~1,200] |
| Corps, privates | `CORP_HOME`, `CORP_TOKENS` int8[7]; `CORP_TOKEN_PRICE` int16[7][3]; `PRIVATE_VALUE`, `PRIVATE_REVENUE` int16[7]; `MF_PORT_HEX` int8[4]; `ER_TILE` int8[6]; `UTF_LATE_REVENUE`, `SMR_DISCOUNT` |
| Players | `START_CASH`, `CERT_LIMIT` int16[7]; `PRIVATES_IN_PLAY` uint8[7], indexed by player count |
| Trains | `TRAIN_DISTANCE`, `TRAIN_PRICE`, `TRAIN_COUNT`, `TRAIN_RUSTS_ON`, `TRAIN_AVAILABLE_ON` int16[6]; `TRAIN_TRADE_IN` int16[6][6] |
| Phases, market | `PHASE_ON_TRAIN`, `PHASE_TRAIN_LIMIT`, `PHASE_TILE_COLORS`, `PHASE_ORS`, `PHASE_CAN_BUY_PRIVATES` int8[6]; `MARKET_PRICE` int16[11][15] (0 = no cell), `MARKET_TYPE` uint8[11][15], `MARKET_ROW_LEN`; `PAR_ROW`, `PAR_COL`, `PAR_PRICE`; `PAIR_A`, `PAIR_B` uint8[21] |

`MAX_LAY_CANDIDATES` is the largest number of distinct-mask lay targets any
hex state allows once exit masks are applied. It is 15 today: the empty plain
and city hexes reach it, and symmetric duplicates would make it 18. The
generator computes it, and `actions` asserts it.

**Test:** extend `tests/g1889/test_data_vs_ruby.py:36` so a fresh Ruby dump
reproduces all three files byte for byte. A Ruby-free test regenerates the
Cython from the checked-in JSON, and a third compares the compiled mirrors
with `load_tables()`.

## 3. State layout

The state is one `int16` row. `compute_*_offsets() noexcept nogil` builds
the layout structs, which are module constants (`core/state.pyx:114`,
`core/state.pxd:180-186`). Players come last, so only the total width varies
with player capacity (`core/state.pyx:171`).

**Padding.** `MAX_PLAYERS` is 6. Rows are padded to a capacity of 2-6, and
padded strides are zero. `num_players` lives in the global block, and
`get_storage_player_capacity` inverts the width (`core/state.pyx:456`). The
pipeline trains with 3-5 players at capacity 5. The privates in play follow
the actual player count, so a 3-player row marks UTF removed. Action ids
never depend on the player count.

**`GameState`** mirrors RSS (`core/state.pxd:190-207`,
`core/state.pyx:715-859`): `_data`, `_array`, `max_players`, `from_array`,
`from_buffer`, `rebind`, the Section 7 flags, and `initialize_game(num_players,
seed, max_players, priority_player=-1)`, where −1 lets the seed pick the first
player and replay passes 0. Canonical order is the 18xx JSON's seat order;
Ruby rotates `@players` (`game/base.rb:3159`), and the engine stores
`priority_player` instead.

**`int16` range.** Every value fits in `int16`. The tightest is `bank_cash`
after the bank breaks: at most one OR set follows, roughly 3 × 7 corps ×
¥1,200, which is about −¥25,000. `BANK` asserts the bound. If a game ever
exceeds it, the value moves to two slots behind the same API.

| Block | Size | Fields |
|---|---|---|
| Global | 40 | in the next table |
| Corporations | 7 × 19 | `par_index` (−1 unparred), `floated`, `cash`, `ipo_shares`, `pool_shares`, `president`, `market_row`, `market_col`, `market_stamp`, `tokens_left`, `trains[6]`, `last_revenue`, `route_cache`, `route_available` |
| Privates | 7 × 2 | `owner` (−3 not in play, −2 closed, −1 unsold, 0-5 player, 8+c corp), `flags` (MF tile used) |
| Hexes | 52 × 3 | `tile` (0-39 supply, 40+h printed), `rotation` (canonical), `tokens` (7-bit corp mask) |
| Tile supply | 40 | copies left |
| Trains | 12 | `depot[6]` (D: −1, unlimited), `pool[6]` |
| Graph cache | 7 × 52 | per (corp, hex): bits 0-5 sides reached, bit 6 node reached |
| Players (last) | 18 × capacity | `cash`, `shares[7]` (10% units; the President's Certificate counts 2), `sold_mask` (sold this SR), `bids[7]`, `passed`, `bankrupt` |

| Global group | Fields |
|---|---|
| Game | `num_players`, `round`, `step`, `return_step`, `game_phase`, `turn` (SR number), `or_num`, `or_count`, `active_player`, `priority_player` |
| Bank | `bank_cash`, `bank_broken` (sticky), `end_reason` |
| Auction | `auction_turn`, `auction_private`, `auction_discount`, `auction_pass_payouts` |
| Stock round | `sr_bought`, `sr_sold_mask`, `sr_sold_then_bought` |
| Operating round | `operating_corp`, `or_index`, `or_order[7]`, `route_revenue` |
| Pending choice | `pending_hex`, `pending_corp`, `pending_private`, `pending_train_source`, `pending_seller` |
| Emergency, caches | `ebuy_sold_mask`, `ebuy_sold_any`, `market_stamp`, `conn_dirty`, `route_dirty` |

**Total: 759 + 18 × capacity int16.** That is 813, 831, 849, and 867 at
capacities 3, 4, 5, and 6. RSS v3 uses 755 at capacity 5.

- **Tokens are a corp mask, not per-slot entries.** A hex holds at most one
  city, and slot identity has no rule effect. Free slots are
  `TILE_SLOTS` − tokens, less 1 more while the home corp's token is unplaced;
  its mask bit doubles as "home token placed". The 18xx mapper ignores
  `slot`.
- **Market stacking.** `market_stamp` increments on every move, and a lower
  stamp sits higher in the stack. Price order uses Ruby's key: −price,
  −column, row, then stack index (`corporation.rb:102`).
- **Not stored:** this turn's lay and token use (the step implies them) and
  which corps have operated (`or_index`). Trains are counted by type, so an
  18xx id such as `3-0` maps to (owner, type). The graph block can pack two
  hexes per slot if replay memory matters.

## 4. Entities

Entities are stateless singleton handles, like those in `entities/corp.pyx`.
Each offers nogil `cdef` primitives for hot paths and `cpdef` methods for
handlers and tests (`entities/corp.pxd:31-56`).

| Entity | Owns | API |
|---|---|---|
| `BANK` | `bank_cash`, `bank_broken` | `transfer(state, src, dst, amount)` is the only code that changes cash. The bank breaks permanently when a payment would take it below zero (`bank.rb:26-39`) |
| `PLAYERS[i]` | player block | `cash`, `shares(c)`, `cert_count` (privates count, zone shares don't), `can_gain(c)` (60% cap unless orange zone; cert limit), `net_worth`, `bid`, `committed`, `clear_bids`, `passed`, `sold_mask`, `bankrupt` |
| `CORPS[c]` | corp block | `par`, `float_if_ready` (pays 10 × par at 5 IO shares left), `buy_share(player, source)`, `sell_shares(player, n)` with presidency recompute (as `entities/corp.pxd:178`), cached `route_revenue` and `route_available` |
| `PRIVATES[p]` | private block | `owner`, `revenue` (UTF pays ¥50 from phase 5), `sell_to_corp`, `close`, `blocks_hex(h)` (TR blocks K4 and ER blocks C4 while a player owns them), `exchange_dr`, the MF flag |
| `MARKET` | market fields | 2D moves with edge rules (rules §6.7), zone tests, `order_key`, `sort_by_price(mask)` |
| `TRAINS` | trains block, corp `trains` | `depot_types` (the lowest rank, plus D from phase 6), buys from the depot, pool, or a corp, `trade_in_for_d`, `discard`, `rust`, `limit`, `crowded`, `cheapest_price` |
| `PHASE` | `game_phase` | `advance` on the first train of a rank: rusting, private closing, UTF revenue, the new limit, D availability |
| `MAP` | hexes, supply | `lay(h, tile, rot)` (returns the old tile to supply and stores the canonical rotation), `place_token`, `free_slots`, `blocked_for(h, c)`, `revenue(h, phase, is_d)` |
| `GRAPH` | graph block | `refresh(c)` (nogil), `sides_reached(c, h)`, `node_reached(c, h)`; it also fills `route_available` |
| `TURN` | global block | round and step transitions, counters, players, OR order, pending fields |

**Dirty caches** follow `entities/corp.pyx:81-108`: one bit per corp in the
global block, a lazy refresh in the getter (`:389`), and a nogil refresh that
extraction can run in a prologue (as `core/token_data_v3.pyx:345-347`). They
live in the row, so MCTS pool copies keep them across `rebind`. Net worth,
cert counts, and crowding are cheap enough to compute on demand.

| Cache | Dirty bit | Invalidated by | Refresh |
|---|---|---|---|
| connectivity, `route_available` | `conn_dirty` | `MAP.lay` and `MAP.place_token`, for all corps | search from the corp's token nodes |
| `route_cache` | `route_dirty` | the same, plus the corp's train changes, rusting, and phase changes | the route solver (Section 8) |

**Connectivity** follows `Graph#compute` (`graph.rb:193`). A tokened hex
reaches all its passable sides. A walk follows paths, never reverses at a
shared edge, passes towns, and may enter but not pass an offboard or a city
blocked for the corp (`part/city.rb:37`). `route_available` is Ruby's
`route_train_purchase`: some token's walk reaches two or more revenue
centers. The engine searches directed path states, a superset of Ruby's walk
(Ruby marks each path visited once per source, so it can drop a junction
branch first reached from the wrong end). A superset only makes more lays
legal, so replay stays safe. **Test:** compare with `connected_hexes` and
`connected_nodes` on fixture and random states; on a mismatch, replicate
Ruby's walk order and drop the path sort at `generate.py:42`.

## 5. Decision phases and actions

### 5.1 Conventions

- **Encoding.** `ActionInfo` (phase, type, corp, private, hex, train_type,
  count, amount) and `decode_action`/`encode_action` mirror
  `core/actions.pxd:111-173`, with import-time size asserts
  (`core/actions.pyx:1250-1268`).
- **Three id sets.** *Enumerated engine-legal* ids come from
  `enumerate_legal_actions`, nogil, in a fixed order (`core/actions.pyx:1054`).
  *Policy* ids are those minus what the state's flags restrict (Section 7),
  from `enumerate_policy_actions` (as `core/actions.pyx:1133`); search expands
  only these. *Exact amounts* (size + ¥amount in the four amount phases) are
  legal by range check, never enumerated, and outside the policy layout, so
  replay can express any human price. Their ids are at or above the phase's
  size, so code that maps ids to unified slots (the LUT, history, analysis)
  must skip them rather than index with them.
- **Validation.** `is_legal_action(state, id)` scans the enumerated set (at
  most 102 ids, as `core/driver.pyx:322-331`) or checks the amount range.
- **Permanent policy filters.** The policy never includes exact amounts, or
  a bid on the topmost private, which buying it at its price dominates.
- **Rotations.** Rotations with equal masks are one candidate
  (`TILE_CANON_ROT`). The 18xx mapper canonicalizes, since Ruby accepts any
  of them.
- **Interjection slots.** "Buy private p" (in phases 3-4), the DR exchange,
  and the MF port tile appear only in the active player's top-level phases
  (Section 12, item 4).

### 5.2 Phases

| # | Phase | Size | Phase-local ids | Actor |
|---|---|---|---|---|
| 0 | AUCTION | 9 | 0 pass · 1 buy topmost at its price · 2+p bid on private p | turn player |
| 1 | BID | 16 | k: minimum + ¥5k; exact 16 + amount | same |
| 2 | RAISE | 17 | 0 drop · 1+k: high bid + ¥5 + ¥5k; exact 17 + amount | lowest bidder |
| 3 | STOCK | 59 | 0 pass or done · 1+c IPO buy · 8+c pool buy · 15+c par · 22+5c+(n−1) sell n · 57 DR · 58 MF | turn player |
| 4 | PAR | 6 | par index (¥65-100) | same |
| 5 | TRACK | 62 | 0 pass · 1+h hex · 53+p buy private · 60 DR · 61 MF | president |
| 6 | TILE | 15 | candidate index at the chosen hex | president |
| 7 | TOKEN | 61 | 0 pass · 1+h hex · 53+p buy private · 60 DR | president |
| 8 | DIVIDEND | 10 | 0 withhold · 1 payout · 2+p buy private · 9 DR | president |
| 9 | DISCARD | 6 | train type | crowded corp's president |
| 10 | BUY_TRAIN | 102 | 0 pass · 1+t depot · 7+i trade in a 4/5/6 for a D · 10+t pool · 16+6c+t from corp c · 58+5c+(n−1) emergency sale · 93 bankrupt · 94+p buy private · 101 DR | president |
| 11 | TRAIN_PRICE | 5 | price points; exact 5 + price | president |
| 12 | BUY_PRIVATE | 9 | 0 done · 1+p · 8 DR | president |
| 13 | PRIVATE_PRICE | 4 | price points; exact 4 + price | president |
| 14 | ER_BONUS | 16 | 0 pass · 1+k candidate on C4 | ER's seller |
| 15 | PORT | 24 | 6i + r: port hex i, rotation r | MF's owner |

**Unified dimension: 421**; `max_action_size` 102 (RSS: 255 and 53). Pass
slots: AUCTION, RAISE, STOCK, TRACK, TOKEN, BUY_TRAIN, BUY_PRIVATE, ER_BONUS.
Metric names follow the phase names (`div`, `train`, `private` for DIVIDEND,
BUY_TRAIN, BUY_PRIVATE). Price points form the policy grid; a point outside
the legal range is not offered, and one equal to a lower point is dropped.

| Grid | Legal range | Points |
|---|---|---|
| BID | high bid (or face) + ¥5, up to cash minus the player's other bids | the minimum, then ¥5 steps |
| TRAIN_PRICE | ¥1 to the buyer's cash (`step/train.rb:247`) | ¥1, list price, ⌈cash/3⌉, ⌈2·cash/3⌉, cash |
| PRIVATE_PRICE | ⌈face/2⌉ to min(2 × face, cash) (`company.rb:29-30`) | low, face, 1½ × face, high |

### 5.3 Factored decisions

| Selection | Stored in | Sub-phase | Then |
|---|---|---|---|
| AUCTION 2+p | `pending_private` | BID | the auction continues |
| STOCK 15+c | `pending_corp` | PAR | the same SR turn |
| STOCK 58, TRACK 61 | `return_step` | PORT | `return_step` |
| TRACK 1+h | `pending_hex` | TILE | TOKEN |
| any buy-private slot | `pending_private`, `return_step` | PRIVATE_PRICE | ER_BONUS if p is ER (actor `pending_seller`), then `return_step` |
| BUY_TRAIN 16+6c+t | `pending_train_source` | TRAIN_PRICE | BUY_TRAIN |
| a depot buy that changes the phase | `return_step` | DISCARD for each crowded corp, in Ruby's order | BUY_TRAIN |

Sub-phases have no pass, and a selection is legal only if its sub-phase has
a legal completion, so there are no dead ends or cancel loops. ER_BONUS always
follows a sale of ER, with pass legal, because Ruby always asks
(`step/track_lay_when_company_sold.rb:48`). Nesting is one level deep, so one
`return_step` suffices.

### 5.4 Tile lays (blueprint §9 open questions)

**Recommendation:** factor a lay into a hex choice, then a candidate, and
index the candidates with a compact list.

- **Factored, not joint.** Joint (hex, tile, rotation) needs 12,480 dense
  slots or a per-state list of up to 780; factoring needs 52 + 15, and TILE
  auto-chains when the hex has one candidate.
- **A compact list, not a 40 × 6 grid.** The candidates are the hex's
  canonical engine-legal targets, sorted by (supply tile, rotation). TILE
  then needs 15 slots and ER_BONUS 16, instead of 240 and 241, so the unified
  dimension is 421 rather than 871. Replay stores a `uint8` mask and a
  `float32` target for each unified slot (`train/replay_buffer.py:103-109`),
  so one sample, including its row, takes about 3.8 KB instead of 6.1 KB.
- **Scoring.** Slot i has no fixed meaning, so the model scores candidates
  from candidate token rows, the plan already made for outcome features
  (blueprint §9, pipeline-refactor §5). `enumerate_lay_candidates(state, hex,
  buf)` (nogil) gives step 7 the same list.

### 5.5 Every fixture move is expressible

| 18xx action | Engine |
|---|---|
| `bid` | the topmost private at its price: AUCTION 1. Another private: AUCTION 2+p, then BID. In a closed auction: RAISE. BID and RAISE take a grid point or an exact amount |
| `pass` | slot 0 of the current phase; ER_BONUS 0 for entity ER |
| `par` (`"70,4,3"`) | STOCK 15+c, then the PAR index for ¥70 |
| `buy_shares` | STOCK 1+c or 8+c, chosen by the share's owner (extractor annotation). Entity DR maps to the DR slot |
| `sell_shares` | a STOCK or BUY_TRAIN sale of (c, percent/10); the engine performs any presidency swap |
| `lay_tile` | by a corp: TRACK 1+h, then TILE (tile, canonical rotation), ignoring the copy suffix in names like `57-1`. By MF: the MF slot, then PORT. By ER: ER_BONUS 1+k |
| `place_token` | TOKEN 1+h; the extractor supplies the hex, and `slot` is ignored |
| `run_routes` | nothing, or `DRIVER.override_route_revenue` if the revenue differs |
| `dividend`, `discard_train`, `bankrupt` | DIVIDEND 0 or 1; DISCARD; BUY_TRAIN 93 |
| `buy_train` | BUY_TRAIN by source and type (`exchange` maps to 7+i); TRAIN_PRICE follows for a buy from a corp |
| `buy_company` | the current phase's buy-private slot, then PRIVATE_PRICE |
| other actions | `undo`/`redo` are resolved by the extractor; `program_*` and `message` are skipped; `auto_actions` apply after their host action; `end_game` stops the replay |

Some fixture prices need exact ids: the corp-to-corp train prices ¥2, ¥100,
¥150, ¥165, ¥360, ¥370, and ¥450. Every private sale is at 2 × face, and
every bid lands on the grid.

## 6. Driver and auto-chaining

`g1889/driver.pyx` copies `core/driver.pyx`: `apply_action` validates,
dispatches, and auto-chains (`:297-338`); history records `(row copy, decision
phase, id)` before each mutation and `(row, −1, step)` for automated steps
(`:147`, `:197`); a guard stops at 1,000 iterations (`:90`); status codes are
0/1/2 (`core/driver.pxd:35-41`), with no step mode.

**One deliberate difference: a decision is forced when its policy set has one
action**, where RSS counts engine-legal actions (`core/driver.pyx:113-122`).
Replay runs with every flag off, where the two sets coincide. In self-play, a
decision whose alternatives are all restricted is applied instead of becoming
a one-child search node. Amount ranges with two or more values usually
yield two or more policy points. The exception (6.3): after a human's
off-grid bid, a BID range narrower than ¥5 has one grid point, and forcing
it would hide the exact amounts above it. So since 6.4, a decision with
exact amounts whose one policy point is also its only engine-legal grid point
is forced only when its legal range has one value. When a flag drops legal
grid points (the auction pass cap), the one remaining policy point is still
forced: only self-play sets flags, and it never plays exact amounts.

### 6.1 Rounds and steps

`STEP_TO_DECISION_PHASE` maps each engine step to its decision phase, or to
−1 for automated steps (as `core/data.pxd:79` and `core/actions.pyx:352`).

| Steps | Kind | Notes |
|---|---|---|
| `AUCTION`, `BID`, `RAISE` | decisions | See the auction notes below |
| `STOCK`, `PAR`, `PORT` | decisions | A pass without acting sets `passed`; "done" after acting clears it (`step/buy_sell_par_shares.rb:202`). A must-sell turn offers only sales, and any legal sale counts, as in Ruby (`:106`). Buying and then selling the same corp in one turn is legal. Priority follows Ruby's `priority_deal_player`: the player after the last to buy or sell. A player with nothing to do gets only slot 0, which the forced rule applies, as Ruby skips them |
| `SR_END` | automated | Priority goes to the player after the last passer. Sold-out corps move up in price order (`round/stock.rb:70`). `or_count` is fixed |
| `OR_START`, `TURN_START` | automated | Pays private revenue in Ruby's order (`game/base.rb:1649`), places home tokens, and orders the floated corps (`game/base.rb:986`) |
| `TRACK`, `TILE`, `TOKEN` | decisions | One lay and one token per turn |
| `ROUTE` | automated | Section 8 |
| `DIVIDEND`, `BUY_TRAIN`, `TRAIN_PRICE`, `DISCARD`, `BUY_PRIVATE`, `PRIVATE_PRICE`, `ER_BONUS` | decisions | Open Market shares pay their dividend to the corp (`step/dividend.rb:151-181`) |
| `TURN_END`, `OR_END` | automated | Emergency sales re-sort the corps yet to run (`round/operating.rb:97`). After a set's last OR: game over if the bank has broken, otherwise an SR. A corp floated mid-OR by DR runs from the next OR |
| `GAME_OVER` | terminal | |

**Auction notes.** After a sale, the handler resolves the next topmost
private in place: one bid wins it, and several go to RAISE, where the lowest
bidder acts next (`step/waterfall_auction.rb:48-54`; the rulebook says
clockwise). A bid clears only the bidder's `passed` flag. When every flag is
set, `all_passed!` runs (`:137-163`): an unsold A drops ¥5 and goes free at ¥0
to the player after the last passer; otherwise owned privates pay out. When
the last private sells, priority goes to `auction_turn`.

Floats, presidency changes, price moves, phase effects (rusting, including
pool trains; private closing, except a player-owned UTF; the new limit; D
availability), bank breaks, and bankruptcy all happen automatically inside
handlers. Pass-only steps and withholds at zero revenue resolve through the
forced rule.

### 6.2 Forced purchase, emergency money, bankruptcy, game end

**Must buy.** A corp with no trains must buy one when a train is available
in the depot or pool and `route_available` is set (`game/base.rb:1404`).
BUY_TRAIN then has no pass.

**Short treasury.** If the treasury can't cover the cheapest train, these
options are legal (`step/train.rb:156-185`, `EBUY_FROM_OTHERS = :never`):

- **The cheapest depot or pool train,** with the president paying the
  shortfall. A depot buy still changes the phase.
- **Another corp's train,** for ¥1 up to the corp's cash, while the corp has
  cash and no shares have been sold (`:180`).
- **Sales of the president's shares** (`step/emergency_money.rb:16-50`): no
  presidency may change anywhere, the pool cap applies, each corp is sold at
  most once (`ebuy_sold_mask`), and a sale is legal only while the next
  smaller bundle would still fall short. Ruby has no "sell down to 60%"
  exception.
- **Bankruptcy** (slot 93), when everything above together falls short
  (`game/base.rb:1989`). The largest sellable bundle in each corp is sold,
  and the president's cash goes to the bank (`step/bankrupt.rb:24-44`). The
  rulebook sends the cash to the corp instead; the scores come out the same.

**Game end.** Bankruptcy ends the game immediately. A broken bank ends it
after the current OR set, whether it broke during an SR or an OR (`:full_or`,
`game/base.rb:3042`).

**Searched decisions** are returned states with two or more policy actions.
**Hot paths** run nogil as in RSS (`core/actions.pyx:352`, `:1054`,
`core/token_data_v3.pyx:345`): `get_decision_phase`, `_enumerate_*`,
`is_legal_action`, `enumerate_lay_candidates`, `GRAPH.refresh`, the solver,
and extraction prologues. Handlers hold the GIL. **Test:** profile at 6.9.

## 7. Runtime flags

Like `acq_same_president` (`core/state.pxd:196-207`), these flags are
`GameState` attributes outside the row; `scratch_state` copies them
(`games/rss.py:72`), and extraction never reads them. Unlike `acq_same_president`, which RSS applies inside legality
(`core/actions.pyx:705`), they filter only the policy set, as the ACQ cap
does (pipeline-refactor §4.1). The driver therefore accepts every 18xx.games
move. Replay turns all three off. In live play the real game state also runs
with them off, because the policy-based forced rule (Section 6) would
otherwise auto-apply a move for a human whose alternatives are only
policy-restricted. Search roots copied from it turn them on, as in self-play.

| Flag | Self-play | Policy effect |
|---|---|---|
| `private_sale_same_president` | True | a corp buys only privates its own president owns |
| `train_sale_same_president` | True | corp-to-corp train sales only between corps with the same president |
| `auction_pass_payout_cap` | 1 (0 = off) | after this many all-pass payouts, a player who can buy the topmost private may neither pass nor bid in a way that leaves them unable to buy it |

- **Same-president flags.** These settle blueprint §4.4 and rules §12.11.
  18xx.games enforces no consent (`depot.rb:118-126`,
  `step/buy_company.rb:22`), so an unrestricted policy could take a rival's
  private at ½ face, or a rival's train for ¥1. A same-president deal only
  moves money between one player's pockets, and the policy can price that
  itself. An `ACQ_OFFER`-style accept phase would serve a move that strong
  players rarely make.
- **Pass cap.** It ends rules §12.9's endless all-pass auction; players who
  can't afford the topmost private may still pass. Capping passes alone
  isn't enough (6.3): in random 2-3 player auctions, players outbid
  themselves on later privates, never afford the topmost, and the payouts
  never stop. So once capped, the policy also drops bids that would leave
  the bidder unable to buy the topmost, which makes the auction terminate. **Config:** a
  `TrainingConfig` "1889 engine options" block, read by `games/g1889.py` and
  rejected unless `game == "1889"` (pipeline-refactor §3.4).

## 8. Route solver boundary

```cython
# g1889/routes/solver.pxd
cdef struct RouteSet:
    int revenue
    int num_routes                 # at most 4, the largest train limit
    int train_type[4]
    int route_revenue[4]
    int num_hexes[4]
    int8_t hexes[4][MAX_ROUTE_HEXES]  # route order (18xx "connections"); 2 × 52, since
                                      # a route can cross a hex twice on a crossing tile
cdef int solve_routes(GameState state, int corp, RouteSet* out) noexcept nogil  # out may be NULL
cpdef int max_route_revenue(GameState state, int corp)   # cached in route_cache
cpdef list best_routes_py(GameState state, int corp)      # tests, analysis, live play later
```

**Rules** (rules §8.6; `route.rb`, `game/base.rb:1481-1525`). A route has 2
stops up to the train's distance (D unlimited), counts every revenue center
on it, includes a city with one of the corp's tokens, visits each revenue
center at most once, and may end at but never pass through an offboard or a
city blocked for the corp. Across all of a corp's routes, each (hex, side) endpoint is used at most
once (`:1481`), so trains share no track but may share stops. Cities and
towns pay their tile value; an empty printed city pays ¥0. Offboards pay by
phase, and D trains score the D value in phase D.

**Placement.** The automated `ROUTE` step solves the operating corp, stores
`route_revenue` (0 when the corp has no trains or no route), and caches the
result in `route_cache` behind `route_dirty`. Extraction may solve other
corps in its nogil prologue.

**Replay.** `DRIVER.override_route_revenue(state, revenue)` is legal only at
the operating corp's DIVIDEND decision, for any multiple of ¥10 from 0 up.
It re-runs the auto-chain, since zero revenue forces the dividend. The
pipeline never calls it. It exists for human routes, which are usually
below the maximum (314 actions 250, 361, 435, 446; 962 action 356), but not
always: Ruby pays logged revenue unchecked, and 314 action 367 pays ¥400
where the legal maximum is ¥370 (IR's route passes through I2, which SR and
AR fill). So there is no upper bound.

**Algorithm** (its own sub-step), suggested shape: search from the token
nodes over precomputed node-to-node segments, giving each train's candidate
routes as 312-bit half-edge sets; then branch and bound over trains in
descending distance, keeping the sets disjoint.

**Test:** the solver is exact against brute force and against Ruby's
`AutoRouter` (`auto_router.rb`) at every fixture Route step. Profile it
early, because step 7's candidate features depend on its cost (blueprint
§4.3).

## 9. Final scores and terminal detection

`is_terminal(state)` checks `round == ROUND_GAME_OVER`. `final_scores(state)`
returns one int per canonical player as Ruby's `Player#value`
(`player.rb:28-30`): cash, plus shares × price in parred corps (the
President's Certificate counts 2; an unparred corp's shares count ¥0), plus
the face value of each open private owned. Bankrupt players are scored the
same way, at any state; values are the pipeline's job (blueprint §10
**Open:**). The tests compare against Ruby's `result` at the end of `314` and `962`, and
against Ruby's player values where the `hs_*` fixtures stop at `end_game`.

## 10. Testing plan (`tests/g1889/`)

1. **Static data,** as in Section 2.
2. **Replay tooling** (`tests/g1889/replay/`), test-only until live play
   needs it:
   - **`extract_states.rb`** resolves undo and redo with Ruby's own
     `Game::Base.filtered_actions` (`game/base.rb:789`), processes each
     surviving action with its `auto_actions`, and writes per action: the
     action; pre-action annotations (the `buy_shares` source, `place_token`
     hex, train owner, and `run_routes` revenue); and a post-action snapshot
     (players, corps, privates, tiles, tokens, supply, depot and pool, phase,
     round, turn, OR, priority, bank, result). Extracts are cached as
     `_extract.json` (`tests/games_18xx/replay_harness.py:76`); tests skip
     without Ruby.
   - **Oracles,** added by the sub-steps that need them: legal lays at Track
     steps (`upgradeable_tiles` with `legal_tile_rotations`, blockers, cost),
     token cities at Token steps, `must_buy_train?`, `connected_hexes`, and
     `AutoRouter` revenue.
   - **`snapshot.py`** provides `compare(state, snap)` and
     `load_state(snap)`. `load_state` builds the state through entity APIs,
     so oracle tests don't depend on replaying up to that point.
   - **`action_map.py`** implements Section 5.5.
   - **Sync rule.** A Ruby action that matches a forced action in the driver
     history since the last sync is consumed rather than applied. States are
     compared whenever no auto-applied history is pending.
   - **Fixture findings** from the extractor (6.2), for the sub-steps that
     reach them:
     - **Discard order (6.7).** Ruby accepts a discard from any crowded
       corp: 314 action 339 discards UR's train while SR is current. Order
       doesn't change the outcome, so the mapper reorders such discards
       instead of the engine widening DISCARD.
     - **Port tile (6.8).** 962 action 39 lays MF's port tile at the start
       of TR's turn, whose president owns MF, and TR then lays its own tile.
       The PORT lay doesn't use up the corp's lay.
     - **Bankruptcy (6.8).** A bankrupt player keeps shares that can't be
       sold: in 314 player 639 ends with KO 40%, TR 20%, and KU 20% as
       president, at ¥0 cash.
     - **Programmed actions.** In the `hs_*` fixtures, programmed buys ride
       as `auto_actions` on another player's pass, alongside
       `program_disable` and `program_share_pass` entries that the mapper
       skips.
     - **Ids.** Corp TR and private TR share a symbol. Ruby keeps one D
       train in the depot and clones it per sale; the engine stores −1.
3. **Differential replay** (`test_replay.py`): all four fixtures, compared
   after every action, including the final results. **Test:** can Ruby's
   `check_track_restrictions!` (`step/tracker.rb:326`, skipped while loading)
   run cheaply per candidate on a cloned game? If not, the lay oracle
   compares the set before that filter, and unit tests cover the
   semi-restrictive rule.
4. **Random playouts** (`test_random_game.py`, like
   `tests/test_random_game.py`): many seeds at 3-5 players, plus smoke tests
   at 2 and 6, with flags on and off. After every step and history
   intermediate, check pipeline-refactor §6.2's invariants: players + corps +
   bank = ¥7,000; 10 shares per corp; one owner per private; no negative
   supply, and `tokens_left` + placed tokens = the charter count; train
   limits after discards; a legal action in every non-terminal state; an end
   within the step cap. Also check that every returned decision has two or
   more policy actions, that the policy set is non-empty whenever the engine
   set is, that recomputed caches equal the stored ones, and that a rebound
   row copy gives the same legal and policy sets.
5. **Per-phase unit tests** (`tests/g1889/phases/`), in the style of
   `tests/phases/` and `.claude/commands/phase-test.md`, run through the
   driver. A `conftest.py` provides `game_state`, `apply_and_verify`, and
   `find_legal_action`, as in `tests/phases/conftest.py:165` and `:263`. Each
   phase has a `setup_<phase>_py` entry point, as in `phases/ipo.pyx:148`.
6. **Route solver:** brute force and `AutoRouter` (Section 8).

## 11. Sub-steps

Each sub-step is one commit or a short series. Each leaves `pytest -q` and
pyright green, and leaves RSS untouched: `tests/test_v2_parity.py` still
passes with its existing fixture. "Green through X" means the replay matches
up to X; the harness reports the first unsupported action.

| # | Scope | Acceptance |
|---|---|---|
| 6.1 | Stage-2 codegen; package skeleton and `__init__.pxd` files; `setup.py`; `constants` | regeneration is byte-identical with and without Ruby; the mirrors equal the JSON; clean build |
| 6.2 | `GameState` and the layout; all entities except `GRAPH` and the solver; an actions skeleton (decode/encode for every phase, dispatch, `is_legal_action`); the driver; the extractor, `compare`/`load_state`, a mapper skeleton, and the sync harness | the initial states of all four fixtures, and Ruby's `new_game` at 2-6 players, equal Ruby's; padded rows rebind across capacities; every id round-trips through encode and decode |
| 6.3 | Auction: AUCTION, BID, and RAISE; resolution chains; all-pass rules; priority; the pass cap | all four fixtures green through the auction; unit tests; random auctions terminate |
| 6.4 | Stock round: buys, pars, block sales with presidency swaps, must-sell, limits and zones, no buy-back, sell/buy order, floats, `SR_END` | all four fixtures green through SR 1 (`hs_*` through their first OR needs a Track pass and a withhold that moves the price, so it moved to 6.5); every later SR of every fixture replays from `load_state`; unit tests |
| 6.5 | OR flow; TRACK and TILE (terrain, SMR, blockers, colors, supply, the semi-restrictive rule); TOKEN; `GRAPH`; zero-revenue DIVIDEND; a pass-only BUY_TRAIN | `hs_*` fully green; the replay harness's temporary OR_START workaround (`_before_or_start`) removed; lay and token sets equal the Ruby oracles at every Track and Token snapshot of `314` and `962`; connectivity matches, or the gap is measured; unit tests |
| 6.6 | Route solver, the route cache, `ROUTE`, `override_route_revenue` | matches `AutoRouter` at every Route snapshot; at least every human revenue; agrees with brute force; profile recorded |
| 6.7 | Dividends; BUY_TRAIN from the depot, the pool, and other corps, plus the D trade-in; TRAIN_PRICE; train limits; must-buy funded from the treasury; `PHASE`; DISCARD | `314` and `962` green through the first private buy or emergency; random playouts reach phase D |
| 6.8 | Buy-private windows, PRIVATE_PRICE, ER_BONUS, PORT, and the DR exchange; emergency money; bankruptcy; game end; `final_scores` | all four fixtures green end to end, including the final results |
| 6.9 | Full random playouts; flags; profiling; the engine half of `games/g1889.py`; remove the driver's development stop (6.3) | seeds terminate within the cap with the invariants intact; the driver no longer catches `NotImplementedError`; adapter conformance passes the engine checks for 1889; the profile is recorded here |

**The `Game` adapter.** `games/g1889.py` is first wired in at 6.9, once a
full random game runs (pipeline-refactor §6.1). It fills every engine field
(`policy`, `state_size`, `storage_player_capacity`, `num_players_from_row`,
`new_game`, `scratch_state`, `apply_action`, `active_player`, `is_terminal`,
`decision_phase`, `enumerate_policy_actions`, `final_scores`). Its extraction
fields raise `NotImplementedError` until step 7, so the conformance test
(pipeline-refactor §6.3) runs only its engine checks for 1889. `"1889"` joins
`_GAME_MODULES` (`games/__init__.py:40`) at the same time.

## 12. Decisions

The user accepted every recommendation below (2026-11-01).

1. **Decided:** consent for sales across players. *Adopted:* the
   same-president policy flags (Section 7). The engine accepts every sale.
2. **Decided:** the price grids (Section 5.2). *Adopted:* adopt them as
   given, then check which prices self-play picks.
3. **Decided:** the auction pass cap. *Adopted:* policy-only, defaulting
   to 1 payout round.
4. **Decided:** out-of-turn abilities. In Ruby, DR's owner can exchange at any
   moment of an SR or OR, because the Exchange step is part of both rounds
   (`game/base.rb:3201`, `game/g_1889/game.rb:172-185`). MF's owner can lay
   the port tile at any moment of an SR, or at the start of any corp's turn
   (`game/base.rb:3389`). No fixture does either. *Adopted:* support
   in-turn use only, make replay fail loudly on out-of-turn use, and add
   `DRIVER.apply_out_of_turn` along with live play.
5. **Decided:** a forced-decision rule based on the policy set (Section 6).
   *Adopted:* adopt it.
6. **Decided:** buy-private slots at every top-level OR phase in phases 3-4.
   *Adopted:* keep them all, since the engine needs them anyway.
   Restrict them to the turn's start and end only if search breadth suffers.

Settled by following 18xx.games, and recorded in `1889-rules.md` §13-14:

- **Rules §14, open item 2.** An unparred corp's share cannot be sold
  (`game/base.rb:1252`).
- **Rules §14, open item 3, and §12.8.** DR and MF timing is as in item 4.
  Neither use counts as an SR action or resets the passes, because
  `step/exchange.rb:23` never touches `current_actions`.
- **Closed auctions:** the lowest bidder acts next. **Bankruptcy:** shares
  are sold and cash goes to the bank. **Emergency sales:** no 60% exception.
- **Private purchases.** Corps may buy privates at any step of their OR turn
  in phases 3-4. The non-blocking BuyCompany step precedes Track
  (`game/g_1889/game.rb:173-185`, `round/base.rb:134`), and a fixture probe
  saw purchases during Ruby's BuyTrain step.
- **Discards.** Crowded corps discard in corporation-list order
  (`game/base.rb:2231`), which has no effect on the outcome.
