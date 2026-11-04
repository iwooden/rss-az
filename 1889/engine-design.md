# Shikoku 1889 Engine Design

Status: design for step 6 of [`pipeline-refactor.md`](pipeline-refactor.md),
dated 2026-11-01; the user accepted every recommendation in Section 12 the
same day. Every sub-step, 6.1 through 6.10, is implemented, with "As built"
notes where the code departs from or refines this design: the engine plays
complete games, and `games/g1889.py` wires its engine half into the adapter
(Section 11). 6.10 refactored the decision phases (Section 12, item 7).
A code review of `g1889/` paused after three runs (Section 13). Step 7 added
extraction ([`extraction-design.md`](extraction-design.md)), so Section 11's
raising extraction callables describe the adapter as step 6 left it.

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
  phases/             auction, stock (with PAR and SELL_AMOUNT), operating, track, token,
                      route, dividend, buy_train (with SELECT_TRAIN, TRAIN_PRICE,
                      EMERGENCY_SELL_AMOUNT, and DISCARD), privates
  routes/             solver.pxd/.pyx
```

- **The RSS split, kept where it pays.** Entity handles own the row
  (`entities/corp.pxd:112`). Phase handlers are
  `cdef void apply_*(GameState, ActionInfo*) noexcept` and assume legal
  actions (`phases/par.pxd:12`).
- **One OR module for now.** 6.5 keeps the OR's flow and its turn's steps
  through BUY_TRAIN in `phases/operating.pyx`, a few lines each; the
  sub-steps that grow ROUTE, DIVIDEND, BUY_TRAIN, and the rest split them
  into the modules above as they do. 6.7 moves DIVIDEND to
  `phases/dividend.pyx` and BUY_TRAIN, TRAIN_PRICE, and DISCARD, which share
  the purchase's phase change, to `phases/buy_train.pyx`; ROUTE stays in
  `operating.pyx`. 6.10 adds SELL_AMOUNT to `phases/stock.pyx`, and
  SELECT_TRAIN and EMERGENCY_SELL_AMOUNT to `phases/buy_train.pyx`. 6.8a puts
  BUY_PRIVATE_PRE, BUY_PRIVATE_POST, PRIVATE_PRICE, ER_BONUS, PORT, and the
  privates' slots in `phases/privates.pyx`; the driver dispatches those slots
  by action type.
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
| Dimensions | `NUM_HEXES` 52, `NUM_RCS` 25, `NUM_SUPPLY_TILES` 40, `NUM_TILES` 92, `NUM_CORPS` 7, `NUM_PRIVATES` 7, `NUM_TRAIN_TYPES` 6, `NUM_GAME_PHASES` 6, `MARKET_ROWS` 11, `MARKET_COLS` 15, `NUM_PAR_PRICES` 6, `NUM_PAIRS` 21, `NODE` 6, `NUM_PORTS` 4, `MAX_CITY_SLOTS` 3, `MAX_LAY_CANDIDATES` 15, `MAX_ROUTE_HEXES` 104, `MAX_TILE_PATHS` 6 |
| Rules | `BANK_CASH`, `MIN_BID_INCREMENT`, `MARKET_SHARE_LIMIT`, `MAX_OWNERSHIP_PERCENT`, `FLOAT_PERCENT` |
| Hexes | `HEX_NEIGHBOR` int8[52][6], `HEX_EXIT_MASK` uint8[52], `HEX_RC`, `HEX_HOME_CORP`, `HEX_BLOCKER` int8[52]; derived `HEX_WALK_ORDER` int8[52] (Ruby's `Game#hexes` order) |
| Revenue centers | `RC_HEX`, `RC_TYPE`, `RC_PORT` int8[25]; `RC_REVENUE` int16[25][6]; `RC_D_REVENUE` int16[25] |
| Tiles | `TILE_COLOR`, `TILE_LABEL`, `TILE_RC`, `TILE_SLOTS`, `TILE_TERRAIN`, `TILE_BLOCKS_LAY` uint8[92]; `TILE_COUNT`, `TILE_REVENUE`, `TILE_COST` int16[92]; `TILE_MASK` uint32[92][6]; derived `TILE_EXITS` and `TILE_CANON_ROT` (the lowest rotation with an equal mask) uint8[92][6], `TILE_PATH_COUNT` uint8[92], and `TILE_PATHS` uint8[92][6][6] (each path's pair index by rotation, in Ruby's `Tile#paths` order) |
| Upgrades | CSR over `tile·6 + rotation`, canonical targets only: `UPG_START` int16[553]; `UPG_TILE`, `UPG_ROT` uint8[~1,200] |
| Action slots | derived (Section 12, item 8): `TRACK_HEXES` int8[45], `TOKEN_HEXES` int8[15], `PORT_LAY_HEX`, `PORT_LAY_ROT` int8[10], `DISCARD_TYPES` int8[3], `NUM_ER_CANDIDATES`; each value's slot or −1: `HEX_TRACK_SLOT`, `HEX_TOKEN_SLOT` int8[52], `PORT_LAY_SLOT` int8[4][6], `TRAIN_DISCARD_SLOT` int8[6] |
| Corps, privates | `CORP_HOME`, `CORP_TOKENS` int8[7]; `CORP_TOKEN_PRICE` int16[7][3]; `PRIVATE_VALUE`, `PRIVATE_REVENUE` int16[7]; `MF_PORT_HEX` int8[4]; `ER_TILE` int8[6]; `UTF_LATE_REVENUE`, `SMR_DISCOUNT` |
| Players | `START_CASH`, `CERT_LIMIT` int16[7]; `PRIVATES_IN_PLAY` uint8[7], indexed by player count |
| Trains | `TRAIN_DISTANCE`, `TRAIN_PRICE`, `TRAIN_COUNT`, `TRAIN_RUSTS_ON`, `TRAIN_AVAILABLE_ON` int16[6]; `TRAIN_TRADE_IN` int16[6][6] |
| Phases, market | `PHASE_ON_TRAIN`, `PHASE_TRAIN_LIMIT`, `PHASE_TILE_COLORS`, `PHASE_ORS`, `PHASE_CAN_BUY_PRIVATES` int8[6]; `MARKET_PRICE` int16[11][15] (0 = no cell), `MARKET_TYPE` uint8[11][15], `MARKET_ROW_LEN`; `PAR_ROW`, `PAR_COL`, `PAR_PRICE`; `PAIR_A`, `PAIR_B` uint8[21]; derived `ENDPOINT_PAIRS` uint32[7] (the pairs at each endpoint) |

`MAX_LAY_CANDIDATES` is the largest number of distinct-mask lay targets any
hex state allows once exit masks are applied. It is 15 today: the empty plain
and city hexes reach it, and symmetric duplicates would make it 18. The
generator computes it, and `actions` aborts if a scan would exceed it.

`tables.json` keeps each tile's paths in Ruby's order and each hex's
`ruby_order`, because the graph walk replays Ruby's visiting order
(Section 4); the masks ignore both.

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
| Global | 41 | in the next table |
| Corporations | 7 × 19 | `par_index` (−1 unparred), `floated`, `cash`, `ipo_shares`, `pool_shares`, `president`, `market_row`, `market_col`, `market_stamp`, `tokens_left`, `trains[6]`, `last_revenue`, `route_cache`, `route_available` |
| Privates | 7 × 2 | `owner` (−3 not in play, −2 closed, −1 unsold, 0-5 player, 8+c corp), `flags` (MF tile used) |
| Hexes | 52 × 3 | `tile` (0-39 supply, 40+h printed), `rotation` (canonical), `tokens` (7-bit corp mask) |
| Tile supply | 40 | copies left |
| Trains | 12 | `depot[6]` (D: −1, unlimited), `pool[6]` |
| Graph cache | 7 × 52 | per (corp, hex): bits 0-5 sides reached (Ruby's `connected_hexes`), bit 6 node reached (`connected_nodes`) |
| Players (last) | 18 × capacity | `cash`, `shares[7]` (10% units; the President's Certificate counts 2), `sold_mask` (sold this SR), `bids[7]`, `passed`, `bankrupt` |

| Global group | Fields |
|---|---|
| Game | `num_players`, `round`, `step`, `return_step`, `game_phase`, `turn` (SR number), `or_num`, `or_count`, `active_player`, `priority_player` |
| Bank | `bank_cash`, `bank_broken` (sticky), `end_reason` |
| Auction | `auction_turn`, `auction_private`, `auction_discount`, `auction_pass_payouts` |
| Stock round | `sr_bought`, `sr_sold_mask`, `sr_sold_then_bought` |
| Operating round | `operating_corp`, `or_index`, `or_order[7]`, `route_revenue` |
| Pending choice | `pending_hex` (TILE), `pending_corp` (PAR, SELL_AMOUNT, EMERGENCY_SELL_AMOUNT), `pending_private` (BID, PRIVATE_PRICE), `pending_train_source` (SELECT_TRAIN and TRAIN_PRICE: the depot, the pool, or corp c), `pending_train_type` (TRAIN_PRICE), `pending_seller` (ER_BONUS) |
| Emergency, caches | `ebuy_sold_mask`, `ebuy_sold_any`, `market_stamp`, `conn_dirty`, `route_dirty` |

**Total: 760 + 18 × capacity int16.** That is 814, 832, 850, and 868 at
capacities 3, 4, 5, and 6. RSS v3 uses 755 at capacity 5. 6.10 added
`pending_train_type`; until then `pending_train_source` held TRAIN_PRICE's
train as 6c + t.

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
| `CORPS[c]` | corp block | `par`, `float_if_ready` (pays 10 × par at 5 IO shares left), `buy_share(player, source)`, `exchange_share(player)` (DR's free IPO share, parred or not), `sell_shares(player, n)` with presidency recompute (as `entities/corp.pxd:178`), `token_price` (the next charter token's), cached `route_revenue`; `route_available` is GRAPH's |
| `PRIVATES[p]` | private block | `owner`, `revenue` (UTF pays ¥50 from phase 5), `sell_to_corp(c, price)` (paid to the seller), `close`, `blocks_hex(h)` (TR blocks K4 and ER blocks C4 while a player owns them), `exchange` (DR: the IR share, then DR closes), `flags` (`MF_PORT_USED`) |
| `MARKET` | market fields | 2D moves with edge rules (rules §6.7), zone tests, `order_key`, `sort_by_price(mask)` |
| `TRAINS` | trains block, corp `trains` | `depot_types` (the lowest rank, plus D from phase 6), `buy_from_depot`/`buy_from_pool` (list price, to the bank), `buy_from_corp` (a price, to the seller), `trade_in_for_d` (the old train to the pool, then ¥800 for a D), `discard` (to the pool), `rust` (corps', pool's, and depot's), `limit`, `crowded`, `first_crowded`, `cheapest_price` (depot or pool) |
| `PHASE` | `game_phase` | `on_train_bought` (`Phase#buying_train!`: a train of the next phase's rank starts it), `advance` (Ruby's order: the phase, then rusting, then the first 5's private closing) |
| `MAP` | hexes, supply | `lay(h, tile, rot)` (returns the old tile to supply, stores the canonical rotation, keeps tokens, asserts the upgrade table), `lay_track(c, h, tile, rot)` (pays `lay_cost` first), `place_token(h, c)` (`tokens_left` − 1), `buy_token(h, c)` (pays the next token's price first: ¥0 for the home token), `free_slots`, `has_slot_for(h, c)`, `blocked_for(h, c)`, `revenue(h, phase, is_d)`, `lay_cost(h, c)` (terrain or upgrade cost, less SMR's ¥80 on pure mountain) |
| `GRAPH` | graph block | `refresh(c)` (nogil), `sides_reached(c, h)`, `node_reached(c, h)`, `route_available(c)`, `walks_new_track(c, h, tile, rot)` (the semi-restrictive walk, below) |
| `TURN` | global block | round and step transitions, counters, players, OR order, pending fields |

**Dirty caches** follow `entities/corp.pyx:81-108`: one bit per corp in the
global block, a lazy refresh in the getter (`:389`), and a nogil refresh that
extraction can run in a prologue (as `core/token_data_v3.pyx:345-347`). They
live in the row, so MCTS pool copies keep them across `rebind`. Net worth,
cert counts, and crowding are cheap enough to compute on demand.

| Cache | Dirty bit | Invalidated by | Refresh |
|---|---|---|---|
| connectivity, `route_available` | `conn_dirty` | `MAP.lay` and `MAP.place_token`, for all corps | search from the corp's token nodes |
| `route_cache` | `route_dirty` | the same, plus `TRAINS.set_corp_trains` for that corp (buys, discards, rusting) and `PHASE.set_current` for all | `route_max`, the route solver (Section 8) |

**Connectivity** is `Graph#compute` (`graph.rb:193`), replayed step for step
(`entities/graph.pyx`). A tokened hex reaches all its open sides. Each token
walks in Ruby's hex order, with the other token cities marked visited. A walk
follows paths in Ruby's tile order, never reverses at an edge (it takes only
the paths across it), passes towns, and may enter but not pass an offboard or
a city blocked for the corp (`part/city.rb:37`). `route_available` is Ruby's
`route_train_purchase`: one token's walk meets paths, unseen by earlier
walks, that touch two or more revenue centers.

**Order matters.** Ruby marks each path and node visited for the rest of a
token's walk (`part/path.rb`, `part/node.rb`), so a path in a loop is walked
from whichever end the walk meets first, and the loop is followed one way
round. The marks backtrack only past a "converging" exit, a side two paths of
the exited tile share (`Tile#converging_exit?`), and the walk also refuses a
path whose edge is on its stack (the edge counter). The first design, a search
over directed path states, reached more than Ruby on both fixtures: straights
in loops entered from both ends, and lays Ruby rejects under the
semi-restrictive rule (314 record 252, H3; 962 records 270-437). Replaying
Ruby's walk closes the gap; it costs a median 0.5 µs per refresh (2.8 µs at
most on the fixtures).

**Measured:** connectivity, token cities, and lay sets equal Ruby's at every
OR snapshot of all four fixtures, for every corp with a token, and GRAPH and
the semi-restrictive verdicts equal Ruby's on 150 random grown networks
(`tests/g1889/replay/test_network_oracles.py`). Those networks are denser in
junctions than real games (phase D colors, ample cash), and there the
converging walks backtrack more: a refresh takes a median 1.8 µs, a p99 of
34 µs, and 78 µs at most. **Measured (6.9):** on random-playout states with
the self-play flags (Section 6's profile), a refresh of each corp with a
token takes a median 0.3 µs, a p99 of 2.5-2.8 µs, and 91 µs at most.

**Ruby quirk.** The game's own graph keeps `route_train_purchase`
once it is true: `Graph#clear` keeps such routes, so `must_buy_train?` can
read a stale true after the corp's route is cut. The extractor's oracles use
a fresh graph, as GRAPH does. **Measured (6.5):** the `must_buy` oracle
answers `must_buy_train?` both ways at every BuyTrain snapshot. At all 115
(22 must-buys), the game's graph had the operator's route cached and agreed
with the fresh one: no stale value. The engine follows the fresh graph.

**Trains and phases (6.7).** A phase's tile colors, train limit, offboard
values, private-purchase window, and ORs per set (from the next SR's end,
`TURN.or_count`) are table lookups by `game_phase`, as are UTF's ¥50 for a
player owner (`private_revenue`) and D's availability (`depot_types`).
`PHASE.advance` applies the rest in Ruby's order (`Phase#buying_train!`):
`next!`, then `rust_trains!` (2s at phase 4, 3s at 6, 4s at D, on corps, in
the pool, and in the depot), then the first 5's `close_companies`, which
closes every private but a player-owned UTF (`event_close_companies!`: its
`close` ability, `on_phase: 'never'`, binds only for a player owner, so a
corp-owned UTF closes). Ruby runs `rust_trains!` on every purchase, but a
rank's trains are gone after its first, so the engine rusts at the phase
change only. Every purchase calls `PHASE.on_train_bought`, as Ruby calls
`buying_train!`; only a depot train or a D trade-in can start a phase.
Closing a private changes no cache: blocks enter only lay legality. PHASE
cimports TRAINS and PRIVATES, so TRAINS reads the phase slot raw.

**Privates (6.8a).** A sale to a corp moves the owner code and pays the
seller from the treasury (`BuyCompany#pay`); everything else follows the
owner code: the seller's certificates (`num_certs` counts privates), TR's
and ER's blocks (`owner_type: player`), SMR's discount (`MAP.lay_cost`),
and UTF's revenue (¥30 to a corp; its ¥50 and its survival at phase 5 bind
only for a player owner). No cache changes: blocks enter only lay legality.
DR's exchange gives its owner an IR IPO share free and closes DR
(`step/exchange.rb`); before IR is parred the share has no price and makes
nobody president. MF's port tile and ER's bonus tile go down through
`MAP.lay`, so GRAPH and the route caches go stale; both are free, as C4 and
the port hexes have no terrain (`phases/privates.pyx` asserts it).

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
  only these. *Exact amounts* (size + ¥amount in the three amount phases,
  BID, TRAIN_PRICE, and PRIVATE_PRICE) are legal by range check, never
  enumerated, and outside the policy layout, so replay can express any human
  price. Their ids are at or above the phase's
  size, so code that maps ids to unified slots (the LUT, history, analysis)
  must skip them rather than index with them.
- **Validation.** `is_legal_action(state, id)` scans the enumerated set (at
  most 46 ids, as `core/driver.pyx:322-331`) or checks the amount range.
- **Guards (2026-11-02).** As in RSS, input and invariant guards are
  `assert`s, which `setup.py --release` strips (`CYTHON_WITHOUT_ASSERTIONS`):
  a release build's callers must pass valid input, and anything else is
  undefined behavior. An unreachable path is `assert False, "<message>"`
  followed by a harmless return for release builds, never the `except` value
  of an `except -1` signature (`core/driver.pyx:253-254`). Legality is no
  guard: `is_legal_action`, and the driver's `STATUS_INVALID` for an illegal
  action, are the engine's contract with callers. Buffer overflows `abort()`
  in every build, as in `core/actions.pyx:1096-1104`.
- **Permanent policy filters.** The policy never includes exact amounts, or
  a bid on the topmost private, which buying it at its price dominates.
  Private 0 is the topmost while it is unsold, so a bid on it has no slot
  at all (Section 12, item 8).
- **Rotations.** Rotations with equal masks are one candidate
  (`TILE_CANON_ROT`). The 18xx mapper canonicalizes, since Ruby accepts any
  of them.
- **Interjection slots.** "Buy private p" (in phases 3-4), the DR exchange,
  and the MF port tile appear only at STOCK, BUY_PRIVATE_PRE, and
  BUY_PRIVATE_POST (Section 12, items 4 and 6), never in a sub-phase
  (Section 5.3). PRE opens the corp's turn when any slot besides "done" is
  engine-legal, and otherwise the turn goes straight to TRACK. It repeats
  until "done": PRIVATE_PRICE, ER_BONUS, and PORT return to it, and the
  exchange stays there. Where the policy filter leaves only "done"
  (`private_sale_same_president`), the forced rule applies it (Section 6), so
  it costs no search. POST is Ruby's blocking BuyCompany after BuyTrain, in
  phases 3-4: "done", each private the corp may buy, and with one to buy, DR;
  with nothing to buy, "done" is forced, as Ruby skips the step
  (`skip_steps`). STOCK offers the DR and MF slots on every
  turn of their owner while they are usable, a turn with nothing else to do
  included (Section 12, item 4, revised 2026-11-02). There `[pass, DR]` is a
  decision, where Ruby skips the turn or ends it, and replay supplies the
  pass (Section 10.2): a turn passes by itself only when the pass is its one
  legal action. The privates' slots are buy-private slots of the
  operating corp (a player's private, for ⌈face/2⌉ within the treasury); the
  DR slot for the active player while they own DR and IR's IPO has a share
  for them within 60%; and the MF slot (STOCK and PRE) while they own MF,
  unused, and PORT has a lay. Ruby's BuyCompany, Exchange, and SpecialTrack
  steps are non-blocking and join whichever step Ruby stops at
  (`Round::Base#actions_for`); a human's move at another decision replays
  through `DRIVER.override_private_slot` (Section 5.5).
- **Accepted limitation.** The model cannot sell a private into a corp after
  this turn's withhold and before the same turn's first 5-train, since that
  train closes the privates and POST never opens. PRE covers the sale when
  the cash was there at the turn's start.

### 5.2 Phases

| # | Phase | Size | Phase-local ids | Actor |
|---|---|---|---|---|
| 0 | AUCTION | 8 | 0 pass · 1 buy topmost at its price · 2+(p−1) bid on private p, p ≥ 1 | turn player |
| 1 | BID | 17 | 0 drop (a closed auction only) · 1+k: the minimum + ¥5k; exact 17 + amount | turn player; in a closed auction, the lowest bidder |
| 2 | STOCK | 31 | 0 pass or done · 1+c IPO buy · 8+c pool buy · 15+c par · 22+c sell corp c · 29 DR · 30 MF | turn player |
| 3 | PAR | 6 | par index (¥65-100) | same |
| 4 | SELL_AMOUNT | 5 | n − 1: sell n shares of the pending corp | same |
| 5 | BUY_PRIVATE_PRE | 10 | 0 done · 1+p buy private · 8 DR · 9 MF | president |
| 6 | TRACK | 46 | 0 pass · 1+i hex `TRACK_HEXES[i]` | president |
| 7 | TILE | 15 | candidate index at the chosen hex | president |
| 8 | TOKEN | 16 | 0 pass · 1+i city `TOKEN_HEXES[i]` | president |
| 9 | DIVIDEND | 2 | 0 withhold · 1 payout | president |
| 10 | DISCARD | 3 | i: train type `DISCARD_TYPES[i]` (3, 4, 5) | crowded corp's president |
| 11 | BUY_TRAIN | 18 | 0 pass · 1 the depot · 2 the pool · 3+c corp c's trains · 10+c emergency sale of corp c · 17 bankrupt | president |
| 12 | SELECT_TRAIN | 9 | t: a type-t train from the pending source · 6+i: from the depot, a D for a traded 4, 5, or 6 | president |
| 13 | TRAIN_PRICE | 5 | price points; exact 5 + price | president |
| 14 | EMERGENCY_SELL_AMOUNT | 5 | n − 1: the president sells n shares of the pending corp | president |
| 15 | BUY_PRIVATE_POST | 9 | 0 done · 1+p · 8 DR | president |
| 16 | PRIVATE_PRICE | 4 | price points; exact 4 + price | president |
| 17 | ER_BONUS | 9 | 0 pass · 1+k candidate on C4 | ER's seller |
| 18 | PORT | 10 | i: port hex `PORT_LAY_HEX[i]`, rotation `PORT_LAY_ROT[i]` | MF's owner |

**Unified dimension: 228**; `max_action_size` 46, TRACK's (RSS: 255 and 53).
These slots hold only what can ever be legal (Section 12, item 8), through
the tables Section 2 derives; token rows are unaffected. Pass slots, whose slot 0 decodes to a pass ahead of the phase's
other choices: AUCTION, BID, STOCK, BUY_PRIVATE_PRE, TRACK, TOKEN, BUY_TRAIN,
BUY_PRIVATE_POST, ER_BONUS. Metric names follow the phase names (`div`,
`train`, `private_pre`, `private_post` for DIVIDEND, BUY_TRAIN,
BUY_PRIVATE_PRE, BUY_PRIVATE_POST; `sell_amount`, `select_train`, and
`emergency_sell_amount` for the 6.10 sub-phases). Price points form the
policy grid; a point outside the legal range is not offered, and one equal
to an earlier point is dropped (TRAIN_PRICE at ¥300 for a 4 offers ¥1, ¥300,
¥100, and ¥200).

**As built (6.10).** Through 6.9 there were 16 phases, sized
9/16/17/59/6/62/15/61/10/6/102/5/9/4/16/24, with a unified dimension of 421
and a `max_action_size` of 102: BID (16 points, no drop) beside RAISE (17:
the lowest bidder's drop, then the high bid + ¥5 + ¥5k), STOCK's sales as
22+5c+(n−1), and BUY_TRAIN's trains as 1+t (depot), 7+i (trade-in), 10+t
(pool), and 16+6c+t (corp c), with its emergency sales at 58+5c+(n−1) and
bankruptcy at 93. Section 12, item 7 factored them (Section 5.3).

| Grid | Legal range | Points |
|---|---|---|
| BID | high bid (or face) + ¥5, up to cash minus the player's other bids; a closed auction's included | the minimum, then ¥5 steps |
| TRAIN_PRICE | ¥1 to the buyer's cash (`step/train.rb:247`) | ¥1, list price, ⌈cash/3⌉, ⌈2·cash/3⌉, cash |
| PRIVATE_PRICE | ⌈face/2⌉ to min(2 × face, cash) (`company.rb:29-30`) | low, face, 1½ × face, high |

### 5.3 Factored decisions

| Selection | Stored in | Sub-phase | Then |
|---|---|---|---|
| AUCTION 2+(p−1) | `pending_private` | BID | the auction continues |
| two or more bids on the topmost private (a closed auction) | `auction_private` | BID, for its lowest bidder | the next lowest bidder's BID, until one bid is left |
| STOCK 15+c | `pending_corp` | PAR | the same SR turn |
| STOCK 22+c | `pending_corp` | SELL_AMOUNT | the same SR turn |
| STOCK 30, BUY_PRIVATE_PRE 9 | `return_step` | PORT | `return_step` |
| TRACK 1+i | `pending_hex` | TILE | TOKEN |
| any buy-private slot | `pending_private`, `return_step` | PRIVATE_PRICE | ER_BONUS if p is ER (actor `pending_seller`), then `return_step` |
| BUY_TRAIN 1+s (s: 0 the depot, 1 the pool, 2+c corp c) | `pending_train_source` (s) | SELECT_TRAIN | BUY_TRAIN; for corp c's train, TRAIN_PRICE (`pending_train_type`), then BUY_TRAIN |
| BUY_TRAIN 10+c | `pending_corp` | EMERGENCY_SELL_AMOUNT | BUY_TRAIN |
| a purchase that changes the phase | `return_step` | DISCARD for each crowded corp, in Ruby's order (corp order), its president acting | BUY_TRAIN |

Sub-phases have no pass, and a selection is legal only if its sub-phase has
a legal completion, so there are no dead ends or cancel loops. BID's drop
is no cancel: it is legal only in a closed auction, which no selection
opens. ER_BONUS always follows a sale of ER, with pass legal, because Ruby
always asks (`step/track_lay_when_company_sold.rb:48`). Only PORT,
PRIVATE_PRICE, ER_BONUS, and DISCARD return to a step they do not fix, and
they nest one level deep, so one `return_step` suffices.

**As built (6.10).** Section 12, item 7's factoring:

- **BID** serves both kinds of bid (`_bid_private`): the selected private,
  or in a closed auction `auction_private` with the lowest bidder acting,
  whose minimum is the high bid + ¥5 (`private_min_bid`). Its grid starts at
  slot 1 in both, and slot 0, the drop, is legal only in a closed auction.
- **Sales.** STOCK offers "sell corp c" when the player may sell a block of
  c, and SELL_AMOUNT each block from one share to `_sale_limit`. BUY_TRAIN
  offers "sell corp c" in an emergency when the president may sell a block
  of c, and EMERGENCY_SELL_AMOUNT each block that emergency legality admits
  (Section 6.2, `_emergency_sale_limit`): each corp once a turn, no
  presidency change, within the pool's room, and only while the next
  smaller block falls short of the need.
- **Trains.** BUY_TRAIN offers a source when SELECT_TRAIN would have a
  train for it (`_source_trains`): the depot's types the treasury affords
  with room, and its D for a traded 4, 5, or 6 with ¥800, room or not; the
  pool's types with room; another corp's trains with room and ¥1. In an
  emergency the depot or the pool offers only the cheapest train
  (`min_depot_train`), and another corp its trains with ¥1 until a share
  is sold, room aside. The same function lists SELECT_TRAIN's ids, so the
  two cannot disagree.
- **The forced rule** (Section 6) applies at every step of a factored
  choice: a source or a corp that is the only choice, a source with one
  train, a corp with one legal block. The privates' slots join no sub-phase
  (Section 5.1).
- **Policy filters.** `train_sale_same_president` drops BUY_TRAIN's corp
  sources (Section 7); the pass cap's BID filter keeps the points from slot
  1 that leave the bidder able to buy the topmost, and leaves a closed
  auction alone, as it left RAISE (its private is the topmost).

**As built (6.8a).** PRIVATE_PRICE pays the seller and clears
`pending_private`; after ER, `pending_seller` holds the seller, who is the
active player (1889's `active_players`) until ER_BONUS resumes the step with
the corp's president. ER_BONUS's candidates are the upgrade table's targets
of C4 among ER's green city tiles (`ER_TILE`), in supply, with exits on open
sides, sorted like TILE's: Ruby's SpecialTrack checks no route
(`reachable: false`) and no semi-restrictive rule (`check_track_restrictions!`
returns for a non-operator), and its `upgrades_to?` skips the label and city
checks for a private (`special`), which no ER tile needs. On the printed C4
there are eight. PORT's ids are `NUM_SIDES·i + rotation` for each port hex
whose tile the port tile upgrades (an empty port town) and each canonical
rotation whose exits stay on open sides; the port tile itself `blocks_lay`,
so no corp lays it. A DR exchange applies in place; a port tile or a
purchase resumes `return_step`. Neither DR nor MF counts as an SR action
(`step/exchange.rb:23`; SpecialTrack tracks none), so the turn and the
passes go on as before.

### 5.4 Tile lays (blueprint §9 open questions)

**Recommendation:** factor a lay into a hex choice, then a candidate, and
index the candidates with a compact list.

- **Factored, not joint.** Joint (hex, tile, rotation) needs 12,480 dense
  slots or a per-state list of up to 780; factoring needs 45 + 15, and TILE
  auto-chains when the hex has one candidate.
- **A compact list, not a 40 × 6 grid.** The candidates are the hex's
  canonical engine-legal targets, sorted by (supply tile, rotation). TILE
  then needs 15 slots and ER_BONUS 9, instead of 240 and 241, so the unified
  dimension is 228 rather than 685. Replay stores a `uint8` mask and a
  `float32` target for each unified slot (`train/replay_buffer.py:103-109`),
  so one sample, including its row, takes about 3.3 KB instead of 5.5 KB.
- **Scoring.** Slot i has no fixed meaning, so the model scores candidates
  from candidate token rows, the plan already made for outcome features
  (blueprint §9, pipeline-refactor §5). `enumerate_lay_candidates(state, hex,
  buf)` (nogil) gives step 7 the same list.

**Legality** (`actions.pyx`, as 6.5 built it). `lay_candidates(state, corp,
hex, buf)` (nogil; `enumerate_lay_candidates` for the operating corp) keeps
an upgrade-table target when Ruby's Track step would offer it:

- the hex is reached (`sides_reached` ≠ 0) and no player-owned TR or ER
  blocks it (`PRIVATES.blocks_hex`);
- the corp can afford `MAP.lay_cost`;
- the tile is in supply, of a phase color, and not `blocks_lay`;
- its exits stay on open sides and meet a reached side
  (`legal_tile_rotation?`);
- the semi-restrictive rule holds (`check_track_restrictions!`): after the
  lay, the corp's walks meet a path of the new tile that is new track, or
  that touches a revenue center whose value changed.

Most lays settle that rule without a walk:

- A lay where there is no track passes.
- A city upgrade changes the city's value, and every path of a city tile
  touches the city. So it passes once its exits meet a reached side: the
  walks run as before the lay until they first enter the hex at a side the
  new tile has track on, and there they walk every path at that side.
- New track that ends only at sides no walk can enter (the hex across has no
  track facing them) fails.

Any other lay, a plain-track upgrade, is decided by `GRAPH.walks_new_track`:
Ruby's walk with the tile in place, stopped at the first new path. TRACK is
pass plus each hex with a candidate (the scan stops at the first), TILE
indexes the pending hex's candidates, and TOKEN is pass plus each city with a
slot for the corp, reached (`node_reached`), and an affordable next token.
**Measured:** a full TRACK enumeration takes a median 1.5 µs on fixture Track
states, 29 µs at most (late 962, with many plain-track upgrades to walk).

### 5.5 Every fixture move is expressible

| 18xx action | Engine |
|---|---|
| `bid` | the topmost private at its price: AUCTION 1. Another private: AUCTION 2+(p−1), then BID. Private 0 above its price: `DRIVER.override_topmost_bid`, then BID. In a closed auction: BID. BID takes a grid point or an exact amount |
| `pass` | slot 0 of the current phase; ER_BONUS 0 for entity ER |
| `par` (`"70,4,3"`) | STOCK 15+c, then the PAR index for ¥70 |
| `buy_shares` | STOCK 1+c or 8+c, chosen by the share's owner (extractor annotation). Entity DR is a private's move (below), on its owner's turn only |
| `sell_shares` | of n = percent/10 shares of c: in an SR, STOCK 22+c, then SELL_AMOUNT n − 1; in an emergency, BUY_TRAIN 10+c, then EMERGENCY_SELL_AMOUNT n − 1. The engine performs any presidency swap |
| `lay_tile` | by a corp: TRACK's slot for the hex, then TILE (tile, canonical rotation), ignoring the copy suffix in names like `57-1`. By MF: a private's move (below), then PORT, on its owner's turn only. By ER: ER_BONUS 1+k |
| `place_token` | TOKEN's slot for the hex; the extractor supplies the hex, and `slot` is ignored |
| `run_routes` | nothing, or `DRIVER.override_route_revenue` if the revenue differs |
| `dividend`, `discard_train`, `bankrupt` | DIVIDEND 0 or 1; DISCARD's slot for the type, reordered to the engine's corp order (Section 10.2); BUY_TRAIN 17 |
| `buy_train` | BUY_TRAIN's source (1 the depot, 2 the pool, 3+c the seller), then SELECT_TRAIN's type t (`exchange` maps to the depot and 6+i) at the list price, or ¥800 for a trade-in; TRAIN_PRICE follows for a buy from a corp, at the grid point its price lands on, or else the exact id |
| `buy_company` | a private's move (below), then PRIVATE_PRICE at the grid point its price lands on, or else the exact id |
| other actions | `undo`/`redo` are resolved by the extractor; `program_*` and `message` are skipped; `auto_actions` apply after their host action; `end_game` stops the replay |

**A private's move** (`buy_company`, DR's exchange, MF's port tile) is STOCK's
slot in an SR. In an OR it is BUY_PRIVATE_PRE's slot while the engine is at
PRE, BUY_PRIVATE_POST's at Ruby's blocking BuyCompany, and at Ruby's other
stops (Track, Token, Dividend, BuyTrain) `DRIVER.override_private_slot(state,
id)` with PRE's id. The override is replay-only and in no legal mask or policy
block, like `override_route_revenue`: at TRACK, TOKEN, DIVIDEND, or BUY_TRAIN
it enters the same chain as PRE's slot (PRIVATE_PRICE at the human's price,
ER_BONUS or PORT if due) and returns to the decision it interrupted, and
elsewhere, or for a slot PRE would not offer, it returns `STATUS_INVALID`.
Where the engine rests at PRE and Ruby does anything else, replay applies
PRE's "done" first (Section 10.2). Ruby's Route step, which the engine runs by
itself, and DiscardTrain, whose `return_step` the move would need, stay
unsupported.

A Ruby action that needs a slot Section 12, item 8 removed raises
`UnsupportedAction`, as it overturns the argument for removing it.

A factored action maps to its decisions in order; where the driver forces
one of them (a corp's one legal block, a source's one train, a source or a
corp that is BUY_TRAIN's only choice), the sync rule takes it from the
driver's history (Section 10.2).

Some fixture prices need exact ids: of the corp-to-corp train prices, ¥1
(three times), ¥2 (⌈2·cash/3⌉ at ¥2), and ¥360, ¥370, and ¥450 (the buyer's
whole treasury) land on the grid, and ¥100 (twice), ¥150, and ¥165 are
exact. Every private sale (6 in 314, 7 in 962) is at 2 × face, the top
grid point, and every bid lands on the grid.

## 6. Driver and auto-chaining

`g1889/driver.pyx` copies `core/driver.pyx`: `apply_action` validates,
dispatches, and auto-chains (`:297-338`); history records `(row copy, decision
phase, id)` before each mutation and `(row, −1, step)` for automated steps
(`:147`, `:197`); a guard stops at 1,000 iterations (`:90`); status codes are
0/1/2 (`core/driver.pxd:35-41`), with no step mode. Replay's other entry
points are `override_route_revenue` (Section 8), `override_private_slot`, and
`override_topmost_bid` (Section 5.5).

**No development stop (6.9).** From 6.3 to 6.8b, a case a later sub-step
owned raised NotImplementedError, and the auto-chain rested there. Every
decision and automated step now has its handler, so the auto-chain ends only
at a choice or the game's end; the driver catches nothing, and an unknown
phase or step fails an assertion, as in `core/driver.pyx`. Nothing in
`g1889/` or its test tooling raises NotImplementedError: the replay harness
and `apply_and_verify` fail on an engine error or a state at rest at an
automated step. The driver records history only once a handler or step
returns. `DRIVER.advance` runs the auto-chain on a state set up by hand at
an automated step.

**One deliberate difference: a decision is forced when its policy set has one
action**, where RSS counts engine-legal actions (`core/driver.pyx:113-122`).
Replay runs with every flag off, where the two sets coincide. In self-play, a
decision whose alternatives are all restricted is applied instead of becoming
a one-child search node. Each step of a factored choice (Section 5.3) is a
decision of its own, so the rule applies to each: a corp whose one legal
block SELL_AMOUNT or EMERGENCY_SELL_AMOUNT applies, a source whose one train
SELECT_TRAIN applies, and a source or a corp to sell that is BUY_TRAIN's one
choice. A closed auction's BID always offers the drop, so it is forced only
when the drop is all the bidder can afford. Amount ranges with two or more
values usually yield two or more policy points. The exception (6.3): after a human's
off-grid bid, a BID range narrower than ¥5 has one grid point, and forcing
it would hide the exact amounts above it. So since 6.4, a decision with
exact amounts whose one policy point is also its only engine-legal grid point
is forced only when its legal range has one value (`wide_amount_range`;
TRAIN_PRICE's range is ¥1 to the buyer's cash, one value at ¥1, and
PRIVATE_PRICE's ⌈face/2⌉ to the lesser of twice face and the treasury, one
value when the treasury holds just ⌈face/2⌉). When a
flag drops legal grid points (the auction pass cap), the one remaining
policy point is still forced: only self-play sets flags, and it never plays
exact amounts.

### 6.1 Rounds and steps

`STEP_TO_DECISION_PHASE` maps each engine step to its decision phase, or to
−1 for automated steps (as `core/data.pxd:79` and `core/actions.pyx:352`).

| Steps | Kind | Notes |
|---|---|---|
| `AUCTION`, `BID` | decisions | See the auction notes below |
| `STOCK`, `PAR`, `SELL_AMOUNT`, `PORT` | decisions | A pass without acting sets `passed`; "done" after acting clears it (`step/buy_sell_par_shares.rb:202`). A must-sell turn offers only sales, and any legal sale counts, as in Ruby (`:106`); STOCK picks the corp and SELL_AMOUNT the block, both forced when they have one choice. Buying and then selling the same corp in one turn is legal. Priority follows Ruby's `priority_deal_player`: the player after the last to buy or sell. Every turn also offers the player's DR exchange and MF port tile while they are usable, a turn Ruby skips or ends for having nothing else included (Section 12, item 4); PORT returns to STOCK, and neither counts as an action. A player with nothing to do and neither private to use gets only slot 0, which the forced rule applies, as Ruby skips them; so does a turn the exchange or port tile leaves with only slot 0, which Ruby ends (`Round::Stock#after_process`) |
| `SR_END` | automated | Priority goes to the player after the last passer. Sold-out corps move up in price order (`round/stock.rb:70`). `or_count` is fixed |
| `OR_START`, `TURN_START` | automated | OR_START pays private revenue (`game/base.rb:1649`; the bank only pays out, so Ruby's payment order cannot matter), places every floated corp's free home token (`HOME_TOKEN_TIMING = :operating_round`), and fixes the order: the floated corps in stock price order (`game/base.rb:986`). With none, OR_END follows. TURN_START gives the turn to the next corp's president, at BUY_PRIVATE_PRE when a private's slot is legal (Section 5.1), else at TRACK |
| `BUY_PRIVATE_PRE`, `TRACK`, `TILE`, `TOKEN` | decisions | PRE's "done" opens TRACK. One lay and one token per turn: TILE moves on to TOKEN, and TOKEN to ROUTE |
| `ROUTE` | automated | The operating corp's maximum revenue (Section 8), ¥0 without trains or a route. Ruby's Route step waits for the president's routes instead; replay pays a human's revenue with `override_route_revenue` |
| `DIVIDEND`, `BUY_TRAIN`, `SELECT_TRAIN`, `TRAIN_PRICE`, `EMERGENCY_SELL_AMOUNT`, `DISCARD`, `BUY_PRIVATE_POST`, `PRIVATE_PRICE`, `ER_BONUS` | decisions | A payout pays each share a tenth of the revenue, rounded up (`dividends_for_entity`): players by their shares, the corp for its Open Market shares (`holder_for_corporation` is the share pool under full capitalization, `step/dividend.rb:151-181`), nobody for the IPO's; the price moves right (up at a row's end). A withhold pays the treasury and moves the price left (down at column 0, `TwoDimensionalMovement`). At zero revenue only the withhold is legal, which Ruby applies by skipping the step. BUY_TRAIN buys one train at a time (a source, then SELECT_TRAIN's train, then for another corp's train TRAIN_PRICE) until the pass, which ends the turn, or in phases 3-4 opens BUY_PRIVATE_POST (Ruby's blocking BuyCompany); with nothing left to buy, the pass is forced, as Ruby passes the step (`pass_if_cannot_buy_train?`). BUY_PRIVATE_POST offers each private the corp may buy and "done", which ends the turn; with nothing to buy, "done" is forced, as Ruby skips the step. PRIVATE_PRICE and ER_BONUS return to the step the purchase came from |
| `TURN_END`, `OR_END` | automated | Emergency sales re-sort the corps yet to run (`round/operating.rb:97`). After a set's last OR: game over if the bank has broken, otherwise an SR, with the SR number up and `or_num` back to 1; Ruby reorders the players only after an SR, so the priority holder opens it. A corp floated mid-OR by DR runs from the next OR |
| `GAME_OVER` | terminal | |

**Auction notes.** After a sale, the handler resolves the next topmost
private in place: one bid wins it, and several open a closed auction at BID,
where the lowest bidder acts next (`step/waterfall_auction.rb:48-54`; the
rulebook says clockwise), raising or dropping out (slot 0). A bid clears only the bidder's `passed` flag. When every flag is
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
BUY_TRAIN then has no pass (the corp's actions are `sell_shares` and
`buy_train`, `step/buy_train.rb:20`). When the treasury covers the cheapest
depot or pool train (6.7), the corp buys as any other: any depot or pool
train it affords, or another corp's train for ¥1 up to its cash; Ruby's
president may contribute only toward the cheapest train
(`check_for_cheapest_train`, `step/train.rb:289`), which the treasury
covers.

**Emergency (as built, 6.8b).** If the treasury can't cover the cheapest
train, the president must raise the need: that train's price less the
treasury and their cash (`needed_cash`, `available_cash`,
`step/train.rb:142-150`). 1889 keeps Ruby's `EBUY_*` defaults
(`game/base.rb:275-298`) but `EBUY_PRES_SWAP = false` and
`EBUY_FROM_OTHERS = :never` (`game/g_1889/game.rb:129-130`), with
`MUST_SELL_IN_BLOCKS = true`. BUY_TRAIN offers, without a pass
(`_enumerate_buy_train` in `actions.pyx`, with `_source_trains` for the
trains and `_append_emergency` for the rest):

- **The cheapest train,** Ruby's `min_depot_train` (`depot.rb:57`), which
  `buyable_trains` lists alone (`step/train.rb:160-161`): the pool's when
  the depot does not sell its type, else the depot's. BUY_TRAIN offers that
  source, and SELECT_TRAIN that train alone, which the driver applies. It is legal while the
  treasury and the president's cash cover it, and the president pays the
  rest (`buy_train_action`, `:76-97`; `Spender#spend` refuses more than
  they hold). A depot buy still changes the phase. Ruby logs the list
  price and no contribution.
- **Another corp's train,** for ¥1 up to the treasury (`spend_minmax`,
  `:247-260`), while the treasury holds ¥1 (`:180`) and the president has
  sold nothing since the turn's last purchase (`last_share_sold_price`,
  `:171-177`; the engine's `ebuy_sold_any`): BUY_TRAIN's corp source,
  SELECT_TRAIN's type, and TRAIN_PRICE. 314 record 350: TR, holding ¥2,
  buys KO's 3 for ¥2.
- **The president's sales** (`step/emergency_money.rb:8-57`,
  `step/train.rb:129-140`), of any corp they hold, the operating corp's
  included: each corp once a turn (`MUST_SELL_IN_BLOCKS` with
  `corporations_sold`; `ebuy_sold_mask`), within the pool's room
  (`fit_in_bank?`), with no presidency changing anywhere: a president keeps
  as many shares as anyone else holds, and so never sells the President's
  Certificate (`causes_president_swap?`, `ShareBundle#can_dump?`;
  `corp_emergency_sale`). And a block sells only while the next smaller one
  would still fall short (`selling_minimum_shares?`,
  `EBUY_SELL_MORE_THAN_NEEDED = false`). BUY_TRAIN offers each corp with a
  legal block (`_emergency_sale_limit`), and EMERGENCY_SELL_AMOUNT its legal
  blocks, its own phase so that the model tells these sales from an SR's
  (Section 12, item 7). Ruby has no "sell down to 60%" exception. A sale drops the price a row per share (`SELL_MOVEMENT =
  :down_share`, `game/base.rb:1311`), and the corps yet to run this OR
  re-sort into price order (`recalculate_order`, `round/operating.rb:97`).
- **Bankruptcy** (slot 17), when the treasury, the president's cash, and
  the largest block they may still sell of each corp fall short of the
  cheapest train (`can_go_bankrupt?`, `game/base.rb:1989`, through
  `liquidity` and BuyTrain's `can_sell?`, so a corp sold this turn counts
  nothing; each corp's largest block counts in full, as every smaller one
  falls short then). Other corps' trains do not count, so both may be
  legal. Its effects (`step/bankrupt.rb:24-57`): the president sells the
  largest block they may of each corp not sold this turn, the corps yet to
  run re-sort, their cash goes to the bank (`set_cash(0, bank)`; the
  rulebook gives it to the corp, and nobody scores either way), and they
  are bankrupt (`declare_bankrupt`), which ends the game at once. The corp
  keeps its treasury. The bankrupt player keeps the shares that could not be
  sold and their privates, which score (Section 9), and Ruby's priority deal
  skips them (`priority_deal_player`, `game/base.rb:3114`), so it passes
  on. 314's bankrupt president, player 639, ends at ¥0 with KO 40% (sold
  that turn), TR 20% (its pool full), and KU 20% (the President's
  Certificate, which nobody could take).
- **Privates** stay for sale (Ruby's non-blocking BuyCompany ignores the
  must-buy, `can_buy_company?`), and DR's exchange stays open.

A decision with one choice is forced as anywhere: a purchase the president
can fund with nothing else legal, a bankruptcy with nothing left to sell,
and since 6.10 a corp to sell that is BUY_TRAIN's only choice, after which
EMERGENCY_SELL_AMOUNT waits for the block when there are two or more.
Ruby logs the purchase, the bankruptcy, and the sale, and the sync rule
consumes the forced decisions. A purchase clears
`ebuy_sold_any` (after one funded by the president the treasury is empty,
so Ruby's step passes and clears `last_share_sold_price`), and TURN_END
clears both fields.

**Where the design was wrong.** It had the cheapest depot *or* pool
train, bankruptcy when "everything above" falls short, and the largest
block of *each* corp sold at bankruptcy. Ruby offers one train, other corps'
trains don't count toward bankruptcy, and a corp sold this turn is not sold
again (314's KO). The re-sort happens at each sale and at the bankruptcy,
not at TURN_END.

**Ruby's steps take more than its UI offers.** BuyTrain gives the president
`sell_shares` at every BuyTrain step (`step/buy_train.rb:16`), and its
`can_sell?` admits a sale whenever their cash and the treasury fall short of
the cheapest train, must-buy or not (8 BuyTrain snapshots of 314 and 11 of
962). The OR's first step, Bankrupt (`game/g_1889/game.rb:174`), takes
`bankrupt` at any step of the turn on that shortfall, counting no shares
outside BuyTrain (22 OR snapshots of 314 and 81 of 962 outside an
emergency). And `buy_train_action` also accepts the pool's train of the
cheapest type beside the depot's (it checks `Depot#available`). The UI
offers emergency money only in a must-buy (the submodule's
`assets/app/view/game/buy_trains.rb:308-310`) and lists only
`buyable_trains`, and so does the engine.

**Game end** (`GAME_END_CHECK = { bankrupt: :immediate, bank: :full_or }`,
`game/base.rb:125`, which 1889 keeps; `game_end_check`, `:2988`, after every
action, `:903-915`). A bankruptcy ends the game at once (one suffices:
`BANKRUPTCY_ENDS_GAME_AFTER = :one`). A broken bank ends it after the
set's last OR (`end_now?`, `:3042`), whether it broke during an SR or an OR.
The engine sets `round = ROUND_GAME_OVER`, `step = STEP_GAME_OVER`, and
`end_reason`, and the row keeps the rest, as Ruby keeps its round.

**Searched decisions** are returned states with two or more policy actions.
**Hot paths** run nogil as in RSS (`core/actions.pyx:352`, `:1054`,
`core/token_data_v3.pyx:345`): `get_decision_phase`, `_enumerate_*`,
`is_legal_action`, `enumerate_lay_candidates`, `GRAPH.refresh`, the solver,
and extraction prologues. Handlers hold the GIL.

**Profile (6.9).** One core, from Python, on the states of 600 random games
(200 at each of 3-5 players, capacity 5) with the self-play flags, played
two ways: uniformly over the policy set, as the adapter conformance test
plays (424,235 returned decisions), and by the default suite's random
player (Section 10.4; 332,940), which reaches phase D more often. Each call
below is timed at every returned decision; the row copy, `GRAPH.refresh`,
and the solver run on a copy of the state, and the last two take the
minimum of three runs with the cache marked stale. Ranges span the two
policies. Scripts and output are in `scratchpad/g1889_69/`.

| Call | Median | p99 | Max |
|---|---|---|---|
| `enumerate_policy_actions` | 0.37-0.42 µs | 5.2-5.9 µs | 171 µs |
| `apply_action`, its auto-chain included | 0.84-0.96 µs | 11-12 µs | 304 µs |
| a row copy plus `rebind` | 0.50-0.52 µs | 3.8-4.0 µs | 160 µs |
| `GRAPH.refresh`, each corp with a token | 0.31-0.32 µs | 2.5-2.8 µs | 91 µs |
| `max_route_revenue`, each corp with trains | 1.0 µs | 5.1-5.7 µs | 279 µs |

`apply_action`'s p99 is highest at TRACK, TILE, DIVIDEND, and TRAIN_PRICE
(12-20 µs), whose auto-chains run the turn's ROUTE step or the next corp's
turn up to its TRACK. **Throughput:** through the adapter, a playout loop
(enumerate, a uniform pick, apply) runs 389,000-391,000 decisions a second,
against RSS's adapter's 292,000-301,000 with the v3 training config's engine
options and 278,000-292,000 with the defaults, on the same machine; the
loop's Python is about half of 1889's 2.6 µs a decision. A random 1889 game
takes 707 decisions to RSS's 197, so it plays in 1.8 ms to RSS's 0.7 ms.
Random play differs from self-play: it lays track on random hexes, buys
random trains, and often ends early in a bankruptcy, so self-play's maps may
be denser and its games longer. Profile self-play states again once step 8
trains.

**Re-measured (6.10),** after the decision-phase refactor (Section 12, item
7), with the same loop, seeds, and machine (`scratchpad/g1889_610/
throughput.py`: 600 games, the self-play flags, three runs each): 386,000-
402,000 decisions a second against 399,500-401,000 before, and 717 returned
decisions a game against 707, so a game plays in 1.8-1.85 ms against
1.77 ms. The new plies are SELL_AMOUNT's, SELECT_TRAIN's, and
EMERGENCY_SELL_AMOUNT's choices; with a corp rather than each block as a
STOCK slot, a uniform pick also sells less often, which offsets part of
them. The fixtures' human moves take 23 more returned decisions in 314 (16
SR blocks, 5 trains from a source with several, 2 emergency blocks) and 8 in
962 (6 and 2).

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
| `train_sale_same_president` | True | corp-to-corp train sales only between corps with the same president: BUY_TRAIN's corp sources |
| `auction_pass_payout_cap` | 1 (0 = off) | after this many all-pass payouts, a player who can buy the topmost private may neither pass nor bid in a way that leaves them unable to buy it: AUCTION's pass and bids, and BID's points on another private. A closed auction's BID is left alone: its private is the topmost |

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
  the bidder unable to buy the topmost, which makes the auction terminate.
- **Config (as built, 6.9).** `TrainingConfig`'s "1889 engine options" block
  holds the three flags under their `GameState` names, defaulting to the
  self-play column above: `private_sale_same_president` and
  `train_sale_same_president` True, `auction_pass_payout_cap` 1.
  `games/g1889.py`'s `new_game` sets them on each state, and
  `scratch_state` copies the root's. Validation rejects a non-default 1889
  option unless `game == "1889"`, and a non-default RSS option
  (`v3_behavior`, `acq_same_president`, `max_acq_price_actions`) unless
  `game == "rss"` (pipeline-refactor §3.4).

## 8. Route solver

```cython
# g1889/routes/solver.pxd
cdef struct RouteSet:
    int revenue
    int num_routes                 # at most MAX_ROUTES (4, the largest train limit): one per train that earns
    int train_type[4]
    int route_revenue[4]
    int num_hexes[4]
    int8_t hexes[4][104]           # MAX_ROUTE_HEXES: each route's hexes in order, each stop once (Ruby's
                                   # Route#ordered_hexes); a crossing tile's hex can appear twice
cdef int solve_routes(GameState state, int corp, RouteSet* out) noexcept nogil  # out may be NULL
cdef int route_max(GameState state, int corp) noexcept nogil   # cached in route_cache behind route_dirty
cpdef int max_route_revenue(GameState state, int corp)         # route_max for Python
cpdef list best_routes_py(GameState state, int corp)           # tests, analysis, live play later
```

**Rules,** as `Route#revenue` checks them (`route.rb`, `game/base.rb:1434-1625`;
rules §8.6). A route:

- has 2 stops up to the train's distance (D's 999 is no limit). Every
  revenue center costs 1 (`visit_cost`), towns included;
- counts every center it visits (`compute_stops` keeps every visit for a
  numeric distance), each once (`check_cycles!`);
- includes a city with one of the corp's tokens, anywhere on it;
- may end at, but not pass, an offboard or a city whose slots hold only
  other corps' tokens (`check_connected`, `Node#blocks?`). A home
  reservation blocks nothing;
- crosses no hex side twice. `check_overlap` keys every path end by (hex,
  side), within the route and across the corp's routes, so trains share no
  track but may share stops. Its other keys (a town off center, two nodes on
  one tile) never bind in 1889: a town path's key comes with a side key, and
  no tile joins two nodes.

Cities and towns pay their tile value, offboards pay by phase (yellow to
phase 4, brown from 5), and a D train scores an offboard's diesel value
(`route_base_revenue` gives it in any phase; D trains exist only in phase D).
An empty printed city pays ¥0, but it has no track, so no route reaches it.

**AutoRouter** (`auto_router.rb`) is exact in 1889 within its limits. Its
path walk (`Node#walk`, backtracking) lists every route that starts at one
of the graph's `connected_nodes`, its hexside bitfield is `check_overlap`'s
key, and its combination search (JavaScript, which the extractor ports) is
an exhaustive branch and bound. The limits are those start nodes, each
train's 10,000 best routes (`route_limit`), and timeouts (30 s for the path
walk, 10 s for the search). None bound at a fixture Route step, where a
train has at most 144 routes. On random maps denser than the tests use, one
train's list reached 10,000 without changing the best.

**Placement.** The automated `ROUTE` step stores `route_max` for the
operating corp in `route_revenue` (¥0 without trains or a route) and moves
on to DIVIDEND. Extraction may call `solve_routes(state, corp, NULL)` for
any corp in its nogil prologue. `route_cache` goes stale through `MAP`'s
writes (lays and tokens: every corp), `TRAINS.set_corp_trains` (that corp;
6.7's buys, discards, trade-ins, and rusting must use it), and
`PHASE.set_current` (every corp).

**Replay.** `DRIVER.override_route_revenue(state, revenue)` is legal only at
the operating corp's DIVIDEND decision, for any multiple of ¥10 from 0 up.
It re-runs the auto-chain, since zero revenue forces the dividend; history
gets the auto-chain only. The pipeline never calls it. It exists for human
routes, which are usually below the maximum (314 actions 250, 361, 435,
446; 962 action 356), but not always: Ruby pays logged revenue unchecked,
and 314 action 367 pays ¥400 where the legal maximum is ¥370 (IR's route
passes through I2, which SR and AR fill). So there is no upper bound.

**Algorithm** (`routes/solver.pyx`):

1. **Segments.** Every path of a hex with a revenue center ends at the
   center, and no other hex has one (`tests/g1889/test_routes.py`). So a
   route is a sequence of centers joined by segments: track between two
   centers through hexes without one. A walk from each center takes the
   paths of the hex across a side at that side only (no reversal at a
   junction) and crosses no side twice. Each segment keeps its half-edges
   (hex × 6 + side, 312 bits) and its hexes. A map has tens of them.
2. **Routes.** Each route is generated once, from the lowest-numbered token
   city on it (the anchor; lower token cities are off limits): a right ray
   from the anchor, then a left ray, sharing no center or half-edge, with
   the left ray's first segment numbered above the right's, which picks one
   orientation. A ray may end at a blocking center but not pass it. Up to
   the longest train's distance, the routes form a forest in pre-order: a
   route's subtree holds the routes that extend it, which keep its track
   and add stops, so revenue never falls down a subtree. Each node keeps its
   subtree's best revenue to each train within the train's distance.
3. **Trains,** longest first. What trains k on earn with some track taken is
   the better of train k running nothing and train k running a route r: r's
   revenue plus what trains k + 1 on earn with r's track taken too. The
   search walks train k's forest, the best subtree first. What the later
   trains earn at r bounds r's whole subtree, since every extension keeps
   r's track: a subtree is cut when its best revenue plus that bound cannot
   beat the best total, and the later trains' routes at r are still their
   best at a child whose new segment they miss, with no search. A call needs
   the exact value only above a floor and returns early below it. Identical
   trains are interchangeable, so the next one takes a route only from the
   first one's tree (the subtree of its first segment) on; each pair is
   still tried.

It is exact: steps 1-2 list each route Ruby accepts once, the bounds are
admissible, and nothing else is cut. The suggested shape, per-train route
lists as half-edge sets and branch and bound bounded by the sum of each
train's best single route, was built first. On dense maps, where one long
route fences off the token cities, that bound is loose and the search tries
nearly every pair: two D trains took 30 ms and three trains 0.43 s on one
map, which the forest's bound brings to 0.16 ms and 1.8 ms.

**Measured (6.6).** The maximum equals AutoRouter's at all 80 fixture Route
steps (32 in 314, 48 in 962), and the routes are AutoRouter's own at 67; at
the other 13, other routes tie, and brute force confirms them legal. It
equals AutoRouter's for 296 corps on 60 random phase-D maps (D, 5, and 6
trains, one or two each, with D + D), and brute force for every corp with
trains at every fixture OR snapshot (840 in 314, 1,331 in 962) and for 187
corps on 40 denser random maps with one to three trains of any type. The
maximum is at least every human revenue but 314 action 367's.

Time per `solve_routes` (one core, run from Python; a dense map is
`random_map` with 30-119 lays in phase D colors, denser in junctions than a
real game):

| States | n | Median | p99 | Max |
|---|---|---|---|---|
| Fixture Route steps, the operating corp | 80 | 1.5 µs | 2.8 µs | 2.9 µs |
| Fixture OR snapshots, every corp with trains | 2,171 | 1.4 µs | 3.5 µs | 6.4 µs |
| 1,000 dense maps, every corp with a token: one D | 4,482 | 3.9 µs | 103 µs | 435 µs |
| The same: D and a 5, 6, or D | 4,482 | 5.9 µs | 223 µs | 1.8 ms |
| The same: D and two of 3-6 (not reachable: phase D's limit is 2) | 4,482 | 14 µs | 1.6 ms | 18 ms |

**Flag for step 7.** States like the fixtures' cost a few microseconds,
so solving all seven corps in an extraction prologue is cheap. Dense maps
have a tail: one D train in 0.43 ms at most (listing up to about 10,000
routes), and two trains above a millisecond in 3 of 4,482 states, up to 1.8
ms, all two D trains splitting a dense network. Self-play may build such
maps early in training. Extraction should read `route_max`, whose cache
rides in the row through MCTS copies, rather than solve each time, and a
profile of self-play states at 6.9 should decide whether the tail matters.

**Measured (6.9)** on the random-play states of Section 6's profile, since
no self-play exists yet. A solve takes a median 1.0 µs and a p99 of 5.1-5.7
µs. Of 2,457,405 solves, 79 took over 100 µs (0.003%) and none over 0.3 ms:
all were D + D or 6 + D in phase D on maps of 38-42 lays, the slowest 279 µs
(D + D, 42 lays). Random play never reached the dense maps' tail (0.43 ms
for one D, 1.8 ms for two, on up to 119 lays). An extraction-like prologue
that reads `route_max` of all seven corps at every returned decision, on
the live row so the cache rides along as it would in MCTS pool rows, finds
93.5-94.4% of reads cached: 84-87% of decisions need no solve, and a
decision needs 0.39-0.46 solves on average. It takes a median 0.17 µs, a
p99 of 11-12 µs, and 0.2-0.3 ms at most (single runs, so noisy), against
3.2-3.4 µs, 19-20 µs, and 0.44 ms when every corp is solved (the minimum of
three runs). The engine's own ROUTE step finds the operating corp's cache
fresh only 0.7-1.5% of the time: the turn's lay, or an earlier turn's, has
just made it stale.

**Recommendation for step 7:** read `route_max` in the extraction prologue,
not `solve_routes`, and let it write the cache into the row it reads. The
pipeline's leaf extraction (`evaluate_leaves`) reads pool-row views, so a
leaf's solves ride into its children's copies, as the prologue above
measures. The tail then costs one solve per stale corp, and only at a miss;
nothing on this evidence calls for more (a time limit or an approximate
bound). Extraction that wants the routes themselves, not only the revenue,
must call `solve_routes` each time, at about the uncached prologue's cost.
Check the prologue's p99 and maximum on self-play states once step 8
trains, since self-play may build denser maps than random play.

**`train.operated` never binds (6.7).** Ruby runs only `runnable_trains`,
which skip a train that ran this OR (`train.operated`). It is cleared for
every floated corp's trains at each OR's start (`Round::Operating#setup`,
`round/operating.rb:27-29`), and only floated corps hold trains; Dividend
sets it for the operator's trains (`step/dividend.rb:83`). A train changes
hands only at BuyTrain and DiscardTrain, both after the buyer's Route step
(1889's steps put Route before Dividend, DiscardTrain, and BuyTrain), and a
discard goes to the pool, whence a buy again follows the buyer's Route. So a
train that moves can run for its new owner only in a later OR, by when the
flag is clear: the engine needs no state for it.

## 9. Final scores and terminal detection

`is_terminal(state)` checks `round == ROUND_GAME_OVER`. `final_scores(state)`
returns one int per canonical player as Ruby's `Player#value`
(`player.rb:28-30`): cash, plus shares × price in parred corps (the
President's Certificate counts 2; an unparred corp's shares count ¥0), plus
the face value of each open private owned. Bankrupt players are scored the
same way, at any state; values are the pipeline's job (blueprint §10
**Open:**). The tests compare against Ruby's `result` at the end of `314` and `962`, and
against Ruby's player values where the `hs_*` fixtures stop at `end_game`.

**As built (6.8b).** `is_terminal` and `final_scores(state, scores)` are
nogil in `actions.pyx`, beside `get_decision_phase`, with `is_terminal_py`
and `final_scores_py` (a list) for the 6.9 adapter, as `games/rss.py` wraps
RSS's `TURN` and `PLAYERS`; `end_reason` tells why the game ended.
`final_scores` reads `player_value` (`entities/player.pyx`,
`PLAYERS[i].value`); Ruby's `result` (`game/base.rb:730`) ranks the same
values. **Measured:** `compare` checks every player's value at every
snapshot of the four fixtures, and Ruby's `result` once the game is over:
314 ends at {90: ¥1,473, 344: ¥1,134, 639: ¥320}, the bankrupt 639's ¥320
being its unsold shares, and 962 at {545: ¥4,444, 253: ¥4,317, 1230:
¥3,880, 147: ¥3,091}. The `hs_*` fixtures' values before their `end_game`
equal its `result`.

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
     `AutoRouter` revenue. Schema 3 (6.5) records `connected_hexes`,
     `connected_nodes`, `route_train_purchase`, token cities, and lays with
     cost and the semi-restrictive verdict for every corp with a token, at
     every OR snapshot (lays at Track snapshots only), on a fresh `Graph`.
     Schema 4 (6.5) adds `must_buy` at BuyTrain snapshots, on a fresh graph
     and from the game's own cached route (Section 4). Schema 5 (6.6) moves
     AutoRouter's best (`auto_router`: revenue, routes with their hexes,
     each train's route count, the timeout) from the `run_routes`
     annotation to every Route snapshot, the state before it, on a fresh
     graph. Schema 6 (6.7) adds `trains` at BuyTrain snapshots, what the
     operator's BuyTrain step offers it on a fresh graph (its action types,
     `room?`, `buyable_trains` by source and type with `spend_minmax`'s
     price range, and `discountable_trains_for`), and `crowded` at
     DiscardTrain snapshots (`crowded_corps`, recomputed without filling its
     cache). Schema 7 (6.8a) adds `companies` at every OR snapshot with an
     operator (the non-blocking BuyCompany's action types, and
     `purchasable_companies` that `company_sellable` admits, with
     `min_price` and `max_price`), `er_bonus` while SpecialTrack blocks for
     ER just sold (ER's lays as SpecialTrack offers them, and the step Ruby
     resumes after it), and a top-level `abilities` with MF's port lays while
     its ability is usable and `Exchange#can_exchange?` for DR, whoever's
     turn it is. Schema 8 (6.8b) adds `emergency` at every OR snapshot
     with an operator: its president, `can_go_bankrupt?` (which Ruby's
     Bankrupt step checks at any step of the turn), and at BuyTrain the
     percents of the bundles BuyTrain's `can_sell?` lets the president sell,
     by corp. `extract_states.rb --graphs` answers the same graph queries, the
     semi-restrictive verdicts, and AutoRouter, for hand-built maps given a
     phase and trains, and `--probes` (6.8b) snapshots a fixture's game at an
     action after in-place changes (pool shares moved to a player, a
     player's cash set), for situations no fixture reaches. Cold extraction
     is unchanged by schemas 5-8 (3.1 s for 314, 8.0 s for 962); the
     random-map route extract takes about 8 s, and the six emergency probes
     13 s.
   - **`snapshot.py`** provides `compare(state, snap)` and
     `load_state(snap)`. `load_state` builds the state through entity APIs,
     so oracle tests don't depend on replaying up to that point; a closed
     auction loads at BID (6.10). Ruby's legal sets list a factored choice
     whole, so `compare` composes the engine's (6.10, `sub_decisions`): each
     STOCK "sell corp c" with the blocks SELL_AMOUNT then offers, and each
     BUY_TRAIN source with SELECT_TRAIN's trains and each corp to sell with
     EMERGENCY_SELL_AMOUNT's blocks (`engine_train_buys`). It enters the
     sub-phase through the driver on a copy, taking the row before a choice
     the driver forces there, so a selection whose sub-phase had no choice
     would fail rather than pass.
   - **`action_map.py`** implements Section 5.5. An OR pass maps by Ruby's
     step, not the engine's phase, so the sync rule never takes a BuyTrain
     pass for a forced TOKEN pass. A factored action maps to its decisions
     in order (6.10): `sell_shares` to STOCK's corp and SELL_AMOUNT's block
     in an SR, or to BUY_TRAIN's corp and EMERGENCY_SELL_AMOUNT's block in an
     emergency, by the engine's phase; `buy_train` to BUY_TRAIN's source,
     SELECT_TRAIN's train, and for another corp's train TRAIN_PRICE.
   - **Sync rule.** A Ruby action that matches a forced action in the driver
     history since the last sync is consumed rather than applied. States are
     compared whenever no auto-applied history is pending. Ruby waits at its
     Route step for the president's routes, and the engine's ROUTE runs by
     itself, so a Route snapshot is compared with the engine's row before
     its ROUTE step (6.6). Ruby logs a forced train purchase (a must-buy
     with one affordable train), which the sync rule consumes. A forced
     sub-decision inside one Ruby action (6.10: a block, a source's one
     train), or a forced parent with a choice left in its sub-phase (a corp
     that is BUY_TRAIN's only choice), is matched against the action's
     decisions the same way, as TILE's was. When the
     replay stops at an action, or runs out, while forced decisions are
     pending, the record before compares with the engine as it is or as it
     was before one of them (6.7): Ruby may wait at a Track step whose only
     choice is the pass, which the engine forces.
   - **Fixture findings** from the extractor (6.2), for the sub-steps that
     reach them:
     - **Discard order (6.7).** Ruby accepts a discard from any crowded
       corp: 314 action 339 discards UR's train while SR is current. Order
       doesn't change the outcome, so the mapper reorders such discards
       instead of the engine widening DISCARD: a discard by a crowded corp
       other than the one the engine's DISCARD is for maps to a held
       discard, which the harness applies when the engine reaches that corp
       (consuming it if the driver forced it), comparing no snapshot in
       between. 314 records 339-340 replay that way.
     - **Unchecked prices (6.7).** Ruby checks no depot, pool, or trade-in
       price (`check_spend` admits ¥1 to the cash for a depot train, and
       `must_pay_face_value?` binds only for other corps' trains), and its
       `buy_train_action` checks no train limit: at the limit, BuyTrain
       opens only for a trade-in (`can_buy_train?`), and the UI then offers
       only that. The engine uses list prices and enforces the limit, and the
       mapper rejects another price; every fixture purchase is at its price.
     - **Emergency buys from other corps (6.8b).** 314 record 350: TR, which
       must buy and holds ¥2, buys KO's 3 for ¥2, as Ruby allows before any
       share sale (Section 6.2); it replays as BUY_TRAIN's corp source
       (16+6c+t before 6.10), SELECT_TRAIN's 3, and TRAIN_PRICE's
       ⌈2·cash/3⌉ point.
     - **Port tile (6.8a).** 962 action 39 lays MF's port tile at the start
       of TR's turn, whose president owns MF, and TR then lays its own tile.
       The PORT lay doesn't use up the corp's lay; it replays as
       BUY_PRIVATE_PRE 9, PORT, PRE's "done", then TR's own TRACK and TILE.
     - **Private purchases (6.8a).** The 13 fixture sales are at Ruby's
       blocking BuyCompany after BuyTrain (10) or during BuyTrain (962
       records 83, 272, 273, through the override), all at twice face; ER's
       sellers pass (314 record 90) or lay tile 14 on C4 (962 record 42). Ruby's BuyCompany
       also offers a purchase during an emergency, and its check ignores the
       must-buy (`can_buy_company?`), which BUY_TRAIN keeps (6.8b).
     - **Bankruptcy (6.8b).** A bankrupt player keeps shares that can't be
       sold: in 314 player 639 ends with KO 40%, TR 20%, and KU 20% as
       president, at ¥0 cash. KO was sold that turn, TR's pool is full, and
       KU's 20% is the President's Certificate with no other holder. In
       KO's emergency (records 449-453) 639 sells UR 10%, KU 30%, IR 10%,
       and TR 10% and pays ¥518 toward a D; in TR's (456-457) it sells KO
       20% and goes bankrupt.
     - **Game ends (6.8b).** Ruby keeps the last round's fields once the
       game is over. At 314's end its round is still the OR, at BuyTrain,
       with TR operating, KO in `corporations_sold`, and the priority deal
       past the bankrupt player; at 962's the OR's last turn is over, with no
       operator. `compare` and `load_state` take a finished snapshot so, and
       Ruby's `result` is compared once the game is over (not for the `hs_*`
       fixtures' `end_game`, which the engine does not model).
     - **Track after the lay (6.5b, 6.8a).** Ruby keeps the Track step open
       after the corp's lay while its president owns MF
       (`tile_lay_abilities_should_block?`), for the port tile; 314 records
       22, 47, and 93 rest there. The corp cannot lay again, and the engine
       has moved on to TOKEN, so the replay needs an equivalence there. The
       extractor records no lays for such a snapshot; `compare` takes its
       Track for the engine's TOKEN, and `load_state` loads it at TOKEN. The
       replay compares it with the engine at TOKEN (before a forced token
       pass, if any), and maps Ruby's pass that closes Track to no decision:
       the decisions the engine forced since the lay are what that pass does.
     - **BUY_PRIVATE_PRE and the override (2026-11-04).** Ruby has no PRE, so
       `compare` takes the engine at PRE for Ruby's Track, and `load_state`
       loads a Track snapshot at TRACK. Where the engine rests at PRE and
       Ruby's next action is not one of PRE's, the harness applies "done"
       (`_leave_pre`); it is not queued as a forced decision, and the forced
       decisions after it are queued as usual. Ruby lets MF's owner lay the
       port tile at that open Track after the corp's lay
       (`owning_player_track` reads only the step), and BuyCompany and
       Exchange act there too, and at Token, Dividend, and BuyTrain; these
       take the override (Section 5.5), at the engine's TOKEN for the open
       Track. With no slot to hold it, the driver may already have forced
       that decision, as a must-buy's one source (then its train, and the
       turn's rest), where Ruby still waits. The harness then backs up to the
       row before the forced decision (`_forced_at_ruby_step`), compares the
       deferred snapshot there, and applies the override, and the driver
       forces the rest again. A live integration needs the same back-up.
     - **Divergences (6.8a).** Replay stops as unsupported for DR or MF out
       of their owner's turn (Section 12, item 4), and for a purchase at
       Ruby's Route or DiscardTrain step, which the engine runs by itself or
       has no slot at. No fixture does any of these. Ruby also ends an SR
       turn without a pass when an exchange or port tile leaves BuySellParShares nothing
       to offer (`Round::Stock#after_process`), where the engine forces the
       pass, which sets the player's `passed` flag; no fixture reaches it.
     - **SR turns Ruby skips (2026-11-02).** STOCK offers DR and MF on every
       turn of their owner (Section 12, item 4), so the engine stops at a
       turn whose only other choice is the pass (or done), where Ruby's
       BuySellParShares offers nothing, not even the pass
       (`step/buy_sell_par_shares.rb:18-30`). Ruby skips such a turn
       (`Round::Stock#start_entity`) or ends it after a purchase or sale
       (`#after_process`), and logs nothing. After each Ruby action the
       harness applies the engine's pass there (`skipped_sr_turn` in
       `action_map.py`). It is not queued as a forced decision, so the sync
       rule never takes a Ruby action for it; the forced decisions after it
       are queued as usual. The fixtures
       reach it 25 times: three skipped turns in SR 1 (314's DR owner with
       ¥25 and MF owner with ¥15, and 962's MF owner with ¥20) and 22 turns
       that end after a purchase or sale. A use of DR or MF that Ruby logs
       later, in another player's turn, is out of turn, and replay stops at
       it as unsupported.
     - **Programmed actions.** In the `hs_*` fixtures, programmed buys ride
       as `auto_actions` on another player's pass, alongside
       `program_disable` and `program_share_pass` entries that the mapper
       skips.
     - **Ids.** Corp TR and private TR share a symbol. Ruby keeps one D
       train in the depot and clones it per sale; the engine stores −1.
3. **Differential replay** (`test_differential_replay.py`): all four fixtures,
   compared after every action, including the final results. Later rounds
   replay from `load_state` of their first snapshot before the replay
   reaches them: every SR, and every corp's turn in every OR. **As of
   6.5:** the `hs_*` fixtures replay to their `end_game`, and 314 and 962 to
   the record before their first train purchase (23 and 25). The 43 OR turns
   of 314 and 56 of 962 replay until the engine rests where a later
   sub-step takes over (ROUTE for a corp with trains, a train purchase) or a
   later sub-step's action, and they cover every corp lay and token of 314
   (46) and 57 of 962's 59; the two left follow MF's port tile (962 record
   33, 6.8a). **As of 6.6:** those turns also run their routes: all 80 fixture
   Route steps (32 in 314, 48 in 962), 6 of them paying the human's revenue
   through `override_route_revenue`, and then rest at DIVIDEND (6.7).
   **As of 6.7:** 314 replays to the record before its first private
   purchase (80; record 81 is UR's `buy_company`), and 962 to the record
   before MF's port tile (32; record 33, 6.8a), which comes before its first
   private purchase (40). The OR turns replay every dividend of 314 (32) and
   47 of 962's 48, 27 of 314's 30 train purchases and 17 of 962's 21, and
   314's 2 discards. Seven turns stop at a private purchase or MF's port tile
   (6.8a), four of 314's at emergency money (records 350, 444, 449, 456), and
   962's last at the game's end after the bank breaks. Replayed alone from
   the snapshot before it, every dividend, purchase, and discard replays but
   314's three emergency purchases and 962's game-ending dividend; among
   them, 962 record 274's first 5 closes the UTF its buyer, KU, owns. At
   every BuyTrain snapshot (60 in 314, 52 in 962, 1 in each `hs_*`),
   BUY_TRAIN's legal set equals Ruby's, and the enumerator raises at exactly
   the 9 where Ruby's must-buy needs emergency money (314: TR at 349, SR at
   443, KO at 448-452, TR at 455-456); at both DiscardTrain snapshots (314)
   the crowded corps and DISCARD's choices do too
   (`replay/test_train_oracles.py`).
   **As of 6.8a:** 314 replays to the record before its first emergency
   purchase (349; record 350 is TR's must-buy short of cash, 6.8b), and 962
   to the record before its last (445; record 446's dividend ends the OR set
   after the bank broke, and OR_END's game end is 6.8b's). The OR turns now
   replay every corp lay and token (46 and 59), every private purchase (6
   of 314's, 7 of 962's), MF's port tile and both ER bonuses, every
   dividend of 314 (32) and 47 of 962's 48, 27 of 314's 30 train purchases
   and all 21 of 962's, and 314's 2 discards; no turn stops at an
   unsupported action, four of 314's at emergency money and 962's last at
   the game's end. Replayed alone, every private purchase replays too, ER's
   up to its seller's ER_BONUS. `replay/test_private_oracles.py` checks the
   schema-7 oracles at every SR and OR snapshot: the privates the operator
   may buy (197 snapshots in 314, 243 in 962), the slots at STOCK, PRE, and
   POST (305 and 274 snapshots, of which 10 and 42 offer a purchase), what
   the override accepts at TRACK, TOKEN, DIVIDEND, and BUY_TRAIN, which offer
   no slot (154 and 172, of which 9 and 74 hold a purchase), ER's lays at
   both ER bonuses, MF's port lays wherever
   Ruby's ability is usable (63 and 19) and the MF slot at its owner's
   decisions, and DR's exchange (76 and 264 snapshots; Ruby refuses 43 of
   962's) and the DR slot. Two scripted games add what no fixture does: an
   SR turn that exchanges DR for an unparred IR share and lays the port
   tile, and an exchange at the operating corp's Track that floats IR
   mid-OR, which then waits for the next OR.
   **As of 6.8b:** all four fixtures replay end to end: 314 through its
   bankruptcy (record 457), 962 through the dividend that ends the OR set in
   which the bank broke (446), each with its final state and `result`
   equal to Ruby's, and the `hs_*` fixtures to their `end_game`. Every OR
   turn (43 in 314, 56 in 962) replays to its end from its first snapshot,
   the game's end included: every corp lay and token (46 and 59), private
   purchase (6 and 7), dividend (32 and 48), train purchase (30 and 21), and
   discard (2 and 0), and 314's 5 emergency sales and its bankruptcy.
   Replayed alone from the snapshot before it, each of those actions
   replays. At every BuyTrain snapshot BUY_TRAIN's legal set equals what
   Ruby offers, emergencies included (`replay/test_train_oracles.py`): at
   314's 9 the president-funded purchase where the president's cash
   suffices (3), 35 sales, and the bankruptcy (2: TR at 455-456); the
   private oracles check what the override accepts at those 9 decisions too.
   Six in-place probes of 314's emergencies
   (`--probes`) equal Ruby too: a president beside a co-holder of three or
   four shares (keeping as many), a need of ¥0 and ¥1, and a bankruptcy
   either side of its boundary.
   **As of 6.9:** the harness checks who acts, for MCTS backs each value up
   to the active player. `compare` already checked Ruby's active players at
   every snapshot; now every decision a Ruby action maps to, the sub-phases
   no snapshot sees (BID, PAR, TILE, TRAIN_PRICE, PRIVATE_PRICE, PORT, and
   since 6.10 SELL_AMOUNT, SELECT_TRAIN, and EMERGENCY_SELL_AMOUNT)
   included, must be the engine's active player's and one of Ruby's active
   players before the action, and so must every forced decision it consumes.
   A held discard is its corp's president's, whose discard Ruby takes out of
   its order (314 record 339: UR's president, while Ruby's active player is
   SR's). Every decision phase occurs in the replays. A replay no longer
   stops quietly at an engine error or a state at rest at an automated step.
   **As of 6.10:** the factored phases (Section 5.3) replay the same games.
   All four fixtures replay end to end with Ruby's final results, and every
   later SR and every OR turn replays from its first snapshot without a
   stop, as does each dividend, purchase, discard, emergency sale, and the
   bankruptcy from the snapshot before it. Replayed from the start, 314's
   25 SR sales take 25 STOCK corps and 16 SELL_AMOUNT choices (9 forced:
   one block), its 30 train purchases 28 chosen sources (2 forced: must-buys
   with the depot alone, one of them an emergency), 5 SELECT_TRAIN choices
   (25 forced), and 8 TRAIN_PRICEs, and
   its 5 emergency sales 5 corps and 2 EMERGENCY_SELL_AMOUNT choices (3
   forced); 962 takes 14 STOCK corps and 6 SELL_AMOUNT choices (8 forced),
   and for its 21 purchases 21 sources and 2 SELECT_TRAIN choices (19
   forced); and each `hs_*` fixture bids twice in a closed auction at BID.
   With the per-turn and per-SR replays, every one of the 19 decision
   phases occurs. The legal-set oracles equal Ruby's through the composition
   at every snapshot: SELL_AMOUNT at 1,143 STOCK corps of the four fixtures
   (2,551 blocks; 496 with one block), SELECT_TRAIN at 441 BUY_TRAIN sources
   (625 trains, a trade-in among them; 257 with one train; 23 in
   emergencies), EMERGENCY_SELL_AMOUNT at 27 corps to sell (35 blocks; 21
   with one), and the four closed-auction snapshots' BID (68 choices). The
   composed BUY_TRAIN sets are those of 6.8b (366, 385, 2, and 2 choices),
   and so are the probes'.

   **Test:** can Ruby's `check_track_restrictions!` (`step/tracker.rb:326`,
   skipped while loading) run cheaply per candidate on a cloned game? **Answer
   (6.5): yes, and no clone is needed.** The extractor lays each candidate in
   place with `Hex#lay` (which moves tokens and reservations across), swaps in
   a fresh `Graph`, turns loading off, runs the check, and lays the old tile
   back, failing if the hex's snapshot changed. A probe costs about 0.15 ms,
   and the fixtures' final results are unchanged. With every corp's lays probed
   at every Track snapshot, cold extraction takes 3.1 s for 314 (1.0 s before),
   8.4 s for 962 (1.3 s), and 0.5 s for each `hs_*` fixture. The rule rejects
   1,091 of 314's probed rotations and 3,670 of 962's, so the lay oracle
   compares the set after the filter; unit tests pin the rule too.
4. **Random playouts** (`phases/test_random_playouts.py`, like
   `tests/test_random_game.py`): many seeds at 3-5 players, plus smoke tests
   at 2 and 6, with flags on and off. After every step and history
   intermediate, check pipeline-refactor §6.2's invariants: players + corps +
   bank = ¥7,000; 10 shares per corp; one owner per private; no negative
   supply, and `tokens_left` + placed tokens = the charter count; train
   limits after discards; a legal action in every non-terminal state; an end
   within the step cap. Also check that every returned decision has two or
   more policy actions, that the policy set is non-empty whenever the engine
   set is, that recomputed caches (GRAPH and the route cache) equal the
   stored ones, and that a rebound
   row copy gives the same legal and policy sets. **As of 6.5:** 20 seeds at
   each of 2-6 players, flags on and off, play from new games to the
   development stop. All 200 stop where a corp must buy its first train,
   after about nine SR and OR cycles each (random SRs float few corps, so
   many ORs are empty) and 913 corp turns in all: 579 lays, 12 tokens, and
   913 withholds. **As of 6.7:** the same 200 games also check that every
   train is in the depot, the pool, or a floated corp until its rank rusts,
   and no corp is over the limit outside DISCARD. A corp that can buy from
   the depot buys its newest offer (a D by trade-in included); every other
   choice stays uniform. Uniform BUY_TRAIN choices strand corps in
   emergencies after rusting (4 of 200 games reach phase D). With the bias,
   42 reach phase D (20 flags off, 22 on), 81 phase 6, 156 phase 5, and 199
   phase 4; 73 end when the bank breaks and its OR set ends, and 127 at a
   must-buy the treasury cannot fund, both 6.8's. They play 11,127 corp
   turns, each with its dividend (4,124 payouts), and 3,488 depot buys, 167
   pool buys, 1,418 buys from other corps, 46 trade-ins, and 417 discards.
   The test takes 19 s. **As of 6.8a:** the games also check that TR and ER
   block their hexes exactly while a player owns them, that no corp's
   private goes back to a player, and that every private but a player's
   UTF has closed after phase 4. A decision that offers a private's slot
   besides other choices takes one a tenth of the time; uniform choices
   there drain the treasuries into privates at up to twice face (13 of the
   flags-on games reach phase D). The 200 games reach phase 4 in 198,
   phase 5 in 150, 6 in 72, and D in 40 (22 flags off, 18 on), and end 76
   times at a broken bank and 124 in emergencies. Over 10,558 corp turns
   they buy 686 privates (224 at TRACK, 23 at TOKEN, 127 at DIVIDEND, 90 at
   BUY_TRAIN, 222 at BUY_PRIVATE; 107 at exact prices, flags off), and ER's
   sellers lay 120 times on C4 and pass 15; DR's owners exchange 173 times
   (155 in SRs, 18 in ORs) and MF's lay the port tile 173 times (166 from
   STOCK, 7 from TRACK). The test still takes 19 s. **As of 6.8b:** the
   games run to their end, all 200 within 1,126 decisions (the cap is
   3,000): 106 at a bankruptcy (63 flags off, 43 on) and 94 at a broken
   bank's OR set (37 and 57). They also check that the game is over exactly
   with an `end_reason` that matches it (one bankrupt player at ¥0, or a
   broken bank), that the turn's emergency sales stand only from BUY_TRAIN
   to the turn's end and `ebuy_sold_any` from a sale to the purchase, and
   that `final_scores` equals `Player#value` recomputed and adds up to the
   players' cash, the shares outside the IPO and pool at market, and their
   privates at face. Emergency choices are uniform but the depot's (the bias
   above): the emergencies take 239 buys from other corps, 71
   president-funded depot buys and 22 pool buys, 105 sales, 106
   bankruptcies, and 4 privates. Over 13,321 corp turns the games reach
   phase 4 in 198, 5 in 184, 6 in 104, and D in 78 (51 flags off, 27 on).
   The test takes 26 s. **As of 6.9:** `apply_and_verify` also checks who
   acts: the priority holder opens an SR, and only STOCK's slot 0 hands the
   turn to the next seat; MF's owner, with MF unused, acts at PORT; and
   ER_BONUS opens for the player who sold ER (the auction's and the OR's
   actors were checked already). A one-off soak (`scratchpad/g1889_69/
   soak.py`, not in the default suite) played 6,000 games from new, 500 at
   each of 3-5 players for each of four settings: flags on and off, each by
   this test's random player and by uniform choices (over the policy set,
   or the legal set flags off), with every check of this test on; flags-off
   games turn the pass cap on after three payout rounds, as here. All 6,000
   reached GAME_OVER, at most 1,684 decisions in (the cap is 3,000):

   | Setting | Decisions: median, p99, max | Bankrupt / bank | Phases 4, 5, 6, D |
   |---|---|---|---|
   | this test's player, flags on | 548, 874, 1,176 | 670 / 830 | 1,494, 1,389, 654, 502 |
   | this test's player, flags off | 562, 984, 1,125 | 851 / 649 | 1,497, 1,379, 868, 669 |
   | uniform, flags on | 727, 1,202, 1,560 | 846 / 654 | 1,471, 1,077, 437, 332 |
   | uniform, flags off | 725, 1,459, 1,684 | 976 / 524 | 1,253, 631, 112, 88 |

   Every decision phase and automated step occurs in each setting. The
   uniform players reach the later phases less often, as corps spend their
   treasuries on each other's trains and privates, and longer games come
   from more SR turns (a uniform pass is rare). The soak takes 66 s on 16
   processes.

   **As of 6.10:** the random player's choices are uniform at each step of a
   factored choice (a sale's corp, then its block; a train's source, then
   its type), and its depot bias takes the depot as BUY_TRAIN's source and
   then SELECT_TRAIN's newest offer, a D by trade-in included.
   `apply_and_verify` also checks the new pending fields: `pending_corp`
   exactly at PAR, SELL_AMOUNT, and EMERGENCY_SELL_AMOUNT (an SR block of a
   corp not yet sold this turn; an emergency's), `pending_train_source` at
   SELECT_TRAIN and TRAIN_PRICE, `pending_train_type` at TRAIN_PRICE (a
   train its seller holds), and a closed auction only at BID for its lowest
   bidder. Every decision phase occurs, and every new action kind: BID's
   drop (447 times), SR sales (8,679, each corp then its block), each
   BUY_TRAIN source (3,766 depot, 240 pool, and 2,088 corp sources),
   SELECT_TRAIN's types (5,993) and trade-ins (101), and emergency sales
   (104, each corp then its block). All 200 games end within 998 decisions:
   91 at a bankruptcy (53 flags off, 38 on) and 109 at a broken bank's OR
   set (47 and 62). They reach phase 4 in 197, 5 in 182, 6 in 87, and D in
   64 (39 flags off, 25 on), against 6.8b's 78: a uniform pick now means a
   uniform step, not a uniform joint choice, which shifts the play (a corp
   to sell is one STOCK slot, not one per block). The test takes 27 s.
   **As of 2026-11-02,** STOCK offers DR and MF on every turn of their
   owner (Section 12, item 4). The new decisions draw from the generator,
   so the games shift: 94 end at a bankruptcy (55 flags off, 39 on) and 106
   at a broken bank's OR set, all within 1,204 decisions, and they reach
   phase 4 in 197, 5 in 175, 6 in 84, and D in 56 (35 and 21).
   A one-off soak (`scratchpad/g1889_610/soak.py`, 6.9's at the same seeds
   and settings; rerun 2026-11-02 after STOCK's DR and MF change) played
   6,000 games from new, with every check of this test on. All reached
   GAME_OVER, at most 1,597 decisions in (the cap is 3,000), and every
   decision phase and automated step occurred in each setting:

   | Setting | Decisions: median, p99, max | Bankrupt / bank | Phases 4, 5, 6, D |
   |---|---|---|---|
   | this test's player, flags on | 571, 901, 1,141 | 634 / 866 | 1,497, 1,412, 631, 465 |
   | this test's player, flags off | 577, 985, 1,239 | 895 / 605 | 1,496, 1,332, 856, 606 |
   | uniform, flags on | 739, 1,174, 1,271 | 839 / 661 | 1,483, 1,083, 426, 309 |
   | uniform, flags off | 712, 1,495, 1,597 | 1,015 / 485 | 1,260, 668, 166, 122 |

   The soak takes 72 s on 16 processes.
5. **Per-phase unit tests** (`tests/g1889/phases/`), in the style of
   `tests/phases/` and `.claude/commands/phase-test.md`, run through the
   driver. A `conftest.py` provides `game_state`, `apply_and_verify`, and
   `find_legal_action`, as in `tests/phases/conftest.py:165` and `:263`. Each
   phase has a `setup_<phase>_py` entry point, as in `phases/ipo.pyx:148`.
6. **Route solver** (Section 8): `test_routes.py` pins each rule and the
   cache through entity APIs; `replay/test_route_oracles.py` compares the
   solver with AutoRouter (fixture Route snapshots and random phase-D maps)
   and with a brute-force path walk (`route_brute.py`) at every fixture OR
   snapshot and on denser random maps.

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
| 6.5 | OR flow; TRACK and TILE (terrain, SMR, blockers, colors, supply, the semi-restrictive rule); TOKEN; `GRAPH`; zero-revenue DIVIDEND; BUY_TRAIN's pass and its depot offers, with must-buy (a purchase rests at the development stop until 6.7) | `hs_*` fully green; the replay harness's temporary OR_START workaround (`_before_or_start`) removed; lay and token sets equal the Ruby oracles at every Track and Token snapshot of `314` and `962`; connectivity matches, or the gap is measured; unit tests |
| 6.6 | Route solver, the route cache, `ROUTE`, `override_route_revenue` | matches `AutoRouter` at every Route snapshot; at least every human revenue; agrees with brute force; profile recorded |
| 6.7 | Dividends; BUY_TRAIN from the depot, the pool, and other corps, plus the D trade-in; TRAIN_PRICE; train limits; must-buy funded from the treasury; `PHASE`; DISCARD | `314` and `962` green through the first private buy or emergency; random playouts reach phase D |
| 6.8a | Privates: the buy-private slots at the top-level OR decisions Ruby asks in phases 3-4, PRIVATE_PRICE, ER_BONUS, PORT, and the DR exchange, in their owners' turns; `private_sale_same_president`; extractor schema 7 | `314` and `962` green up to the first record that needs 6.8b (314 to 349, before TR's emergency; 962 to 445, before the game's end); per-turn replays no longer stop at a private's action; the schema-7 oracles equal Ruby at every snapshot; random playouts use every new action kind |
| 6.8b | Emergency money, including Ruby's emergency buys from other corps and any private purchase in an emergency; bankruptcy; game end; `final_scores` and `is_terminal`; extractor schema 8 and in-place probes | all four fixtures green end to end, including the final results; per-turn replays no longer stop at emergencies or the game's end; the schema-8 oracles and probes equal Ruby; random playouts run to the game's end |
| 6.9 | Full random playouts; flags; profiling; the engine half of `games/g1889.py`; remove the driver's development stop (6.3) | seeds terminate within the cap with the invariants intact; the driver no longer catches `NotImplementedError`; adapter conformance passes the engine checks for 1889; the profile is recorded here. **As built:** all met: the soak (Section 10.4), the driver (Section 6), the config (Section 7), the profile (Sections 4, 6, and 8), and the adapter below |
| 6.10 | Decision-phase refactor (Section 12, item 7): RAISE folds into BID; STOCK's sales factor into a corp and SELL_AMOUNT; BUY_TRAIN factors into a source, SELECT_TRAIN, and TRAIN_PRICE, with its emergency sales in EMERGENCY_SELL_AMOUNT | every fixture still replays end to end with Ruby's final results; the legal-set oracles equal Ruby at every snapshot through the new factoring; random playouts and the conformance test pass; Sections 5.2-5.5 describe the phases as built. **As built:** all met. 18 phases with a unified dimension of 320 and a `max_action_size` of 62 (Section 5.2), the factoring in Section 5.3, the driver's forced rule at each step (Section 6), the layout's `pending_train_type` (Section 3), and the replay's composed oracles and coverage (Section 10.3); a 6,000-game soak and the default playouts (Section 10.4) and the throughput (Section 6) re-measured. `extraction-design.md` designs step 7's extraction for these phases |

**The `Game` adapter (as built, 6.9).** `games/g1889.py` is wired in at
6.9, once a full random game runs (pipeline-refactor §6.1), and `"1889"`
joins `_GAME_MODULES`; importing `games` still imports no engine. It fills
every engine field:

- `policy`: a `PolicyLayout` of Section 5.2's sizes (unified dimension
  228), named as Section 5.2 names the metrics (`div`, `train`, `private_pre`,
  `private_post`, and the other phases in lower case). `phases_with_pass_slot` means what
  it means for RSS: the phases whose slot 0 passes before several other
  choices, whose pass logit the trainer compares with theirs. RSS lists
  INVEST, BID, ACQ_SELECT_CORP, CLOSING, and IPO, and leaves out its zero
  dividend (an amount, not a pass) and its two-slot ACQ_OFFER and ISSUE.
  For 1889 that is `constants.PHASES_WITH_PASS_SLOT`, the phases whose slot
  0 decodes to a pass: AUCTION, BID (the lowest bidder's drop), STOCK,
  BUY_PRIVATE_PRE, TRACK, TOKEN, BUY_TRAIN, BUY_PRIVATE_POST, and ER_BONUS.
  DIVIDEND's withhold is
  a choice, like RSS's zero dividend, and the sub-phases have no
  pass.
- `state_size`, `storage_player_capacity`, and `num_players_from_row`
  from the layout (Section 3).
- `new_game` builds a `GameState` with the config's Section 7 flags and lets
  the seed pick the first player; `scratch_state` wraps a row with
  `from_buffer` and copies the root's three flags, as `games/rss.py` does
  RSS's.
- The rest are the engine's own functions: `DRIVER.apply_action`,
  `TURN.active_player`, `is_terminal_py`, `get_decision_phase_py`,
  `enumerate_policy_actions_py`, and `final_scores_py`. The driver's status
  codes equal `games.STATUS_*`.
- **`active_player`** is the player whose choice it is, to whom MCTS backs
  values up: the turn player in the auction (in a closed auction's BID, the
  lowest bidder) and in an SR (at PAR, SELL_AMOUNT, and PORT too); the
  operating corp's president in an OR, SELECT_TRAIN, TRAIN_PRICE,
  EMERGENCY_SELL_AMOUNT, PRIVATE_PRICE, and PORT included; the crowded corp's
  president at DISCARD; and ER's seller at ER_BONUS. The replay checks it
  against Ruby at every decision of the fixtures, and the invariants at
  every state of the random playouts (Sections 10.3-10.4).

The extraction callables raise `NotImplementedError` ("1889 extraction lands
in pipeline step 7"), and `num_relations` and `max_relation_edges` are
`NO_SIZE`, −1, which any buffer or layer sized from them rejects, where 0
would pass for a game without relations. The conformance test
(pipeline-refactor §6.3) runs every engine check for 1889, with the
self-play flags and with them off, and skips the extraction checks with the
adapter's reason. Training, self-play, evaluation, and analysis reach the
engine through the adapter since pipeline step 2, and each builds its model
with `nn.create_model` before it plays. Since pipeline step 4, a 1889 config
needs a `model_path` module that defines `build_model(config, spec)`, which
arrives in step 8, and building one's input spec raises the extraction's
`NotImplementedError` until step 7.

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
   `DRIVER.apply_out_of_turn` along with live play. *Built:* DR and MF at
   their owner's STOCK turn and at BUY_PRIVATE_PRE of a corp they are
   president of, and DR at such a corp's BUY_PRIVATE_POST (item 6, Section
   5.1).
   *Revised (2026-11-02):* the user decided that STOCK offers DR and MF
   whenever they are usable, whether or not the turn has anything else, so
   a turn passes by itself only when the pass is its one legal action.
   Before, a STOCK turn offered them only where Ruby asks. But DR's exchange
   is `when: 'any'` in Ruby
   (`game/g_1889/entities.rb:91`), so a player Ruby skips for having
   nothing to buy or sell can still exchange during another player's turn;
   in-turn use only, with Ruby's skip, locked them out until their cash
   grew. The OR loses nothing that way, as a president with a use for either
   gets PRE. Replay supplies the pass for the turns Ruby skips (Section
   10.2).
5. **Decided:** a forced-decision rule based on the policy set (Section 6).
   *Adopted:* adopt it.
6. **Decided (2026-11-04):** the privates' OR slots sit at the turn's start
   and end only, BUY_PRIVATE_PRE and BUY_PRIVATE_POST (Section 5.1). The
   action space serves the model rather than perfect rules fidelity, and
   should be as small as possible; slots at TRACK, TOKEN, DIVIDEND, and
   BUY_TRAIN cluttered search. A human's move at those decisions replays
   through `DRIVER.override_private_slot` (Section 5.5), and Section 5.1's
   limitation is accepted.
7. **Decided (2026-11-02):** the decision phases, after 6.9. The user asked
   for smaller action spaces where a choice factors cleanly, in the style of
   RSS's ACQ_SELECT_CORP, ACQ_SELECT_COMPANY, and ACQ_SELECT_PRICE. Which
   factoring trains better is unknown until there is training data, so this
   is a heuristic to revisit then. *Adopted*, built in 6.10:
   - **BID absorbs RAISE.** AUCTION stays the select-or-pass step, like
     RSS's INVEST. BID's slot 0 drops out, legal only in a closed auction
     among a private's bidders (RAISE's case; RSS likewise disables the pass
     on the first bid). Slots 1+k bid the minimum + ¥5k, on the grid BID
     and RAISE already shared.
   - **Sales factor.** STOCK offers "sell corp c" and SELL_AMOUNT the count
     (1-5, only legal counts). Emergency sales at BUY_TRAIN factor the same
     way, into a separate EMERGENCY_SELL_AMOUNT. That way the model sees an
     SR sale and an emergency sale as different decisions, with their own
     legality (Section 6.2), policy slots, and metrics. The user asked for
     the two to be distinguishable, by phase or by a token flag.
   - **Tile lays stay hex, then (tile, rotation)** (Section 5.4). The model
     scores TILE's candidates from their token rows: the tile type as an
     embedding shared across hexes, its exits as laid, and its outcomes. It
     never learns fixed per-slot logits, since slot k is only the hex's k-th
     candidate. Splitting tile from rotation would fix the tile slot's
     meaning, but not the rotation's, and would add a ply. The user accepted
     this, to be revisited with data.
   - **BUY_TRAIN factors by source.** BUY_TRAIN picks pass, the depot, the
     pool, or corp c, plus its emergency sales and bankruptcy.
     SELECT_TRAIN picks the type, a D by trade-in of a 4, 5, or 6 included,
     and is forced when one is offered. TRAIN_PRICE follows for a buy from a
     corp. In self-play, `train_sale_same_president` leaves few corp sources.

   *Built (6.10):* as above (Sections 5.2-5.3); SELL_AMOUNT follows PAR and
   SELECT_TRAIN, TRAIN_PRICE, and EMERGENCY_SELL_AMOUNT follow BUY_TRAIN in the phase
   order, and `pending_corp` holds the corp a sale is of.

8. **Decided (2026-11-04):** the action slots hold only what can ever be
   legal. The user's standard for a slot random play never reached: if the
   rules, the map, or the tables show it can never be legal, remove it; on a
   good argument short of that, remove it anyway and record the argument and
   the doubt here, since a slot can come back on more evidence; with no
   argument, only "never observed", keep it. A sweep over playouts only named
   the candidates. *Adopted:* 69 slots went, taking the unified dimension from
   297 to 228. `tests/g1889/test_action_slots.py` holds each argument against
   the tables, and `test_data_vs_ruby.py` against Ruby's checks. An engine-legal
   choice with no slot fails an assertion, and replay raises
   `UnsupportedAction`, so a wrong argument cannot give a wrong id.
   - **TRACK, 7 hexes (B3, B7, F1, G14, J1, J7, L7): airtight.** Their
     printed tiles, gray or red, have no upgrade target, and no private's lay
     names them, so their tiles never change.
   - **TOKEN, 37 hexes: airtight.** A walk of the upgrade table from each
     printed tile, whatever the phase, supply, or open sides, finds no city:
     no upgrade makes one of plain track or a town, the port tile (a town) and
     ER's tiles (on C4, a city) included. The 15 left are the printed cities.
   - **ER_BONUS 9-15: airtight.** ER blocks C4 while a player owns it, no other
     private's lay names C4, the auction sells ER before any OR at every
     player count, and a corp never sells it on. So the bonus fires once, on
     the printed tile, where full supply gives the most candidates, 8.
   - **PORT, 14 hex and rotation pairs: airtight.** `port_lays` reads the state,
     but only to remove lays: the upgrade table reaches the port tile only from
     a printed tile, at rotation 0, and each pair left out has an exit off
     the board.
   - **DISCARD, types 2, 6, and D: airtight under the rules.** A corp goes over
     its limit only as the limit drops, on the first 4 and the first 5. The 2s
     rust on that 4; a 6 or a D ranks above that 5, and the depot sells by
     rank; and at the limit a corp may only trade in, which keeps its count.
     *Doubt:* Ruby's `buy_train_action` checks no room, so a hand-written
     action, though not its UI (`assets/app/view/game/buy_trains.rb:217`),
     could over-buy where a trade-in keeps BuyTrain open, then discard a 6 or
     a D. The engine rejects such a purchase at BUY_TRAIN, so replay stops
     there, as it already did.
   - **AUCTION 2, a bid on private 0: airtight for the policy.** Private 0 is
     in play at every player count and the topmost while it is unsold, so a
     bid on it is always a purchase above the price, which the policy has
     always dropped. It stays legal in Ruby: `DRIVER.override_topmost_bid` opens
     BID for a human's, and BID's amount buys it. Later privates keep their
     slots, which the policy drops while they are topmost.
   - **Kept: EMERGENCY_SELL_AMOUNT's 5 shares.** No game has shown it, but
     nothing bars it: two players at 50% of a sold-out corp may reach it, and
     the non-president may then sell all five into an empty pool.

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
  saw purchases during Ruby's BuyTrain step. The action space offers them at
  item 6's two decisions, and replay takes the rest.
- **Discards.** Crowded corps discard in corporation-list order
  (`game/base.rb:2231`), which has no effect on the outcome.

## 13. Code review (2026-11-02)

This review is the first gate before step 7 (pipeline-refactor §8). A
separate session ran `/code-review` on one area at a time against a brief,
and wrote its findings to a file. The user triaged each run, and its fixes
were committed as one verified batch. The brief and the findings are local
files in the gitignored `scratchpad/`: `g1889_review_brief.md` and
`g1889_review_findings.md`. The user paused the review after three of the
five planned runs.

| Run | Scope | Reviewed at | Findings | Fixed in |
|---|---|---|---|---|
| 1 | `actions.pyx`, `actions.pxd` | 64cae0b | 7: 6 fixed, 1 deferred | 004159c |
| 2 | `driver.pyx`, `phases/` | 004159c | 8: 7 fixed, 1 a doc note | e043afa |
| 3 | `state.pyx`, `entities/` | 004159c | 12: 8 fixed, 3 deferred, 1 rejected | e043afa |

Every finding was rated low severity. What changed:

- **Engine guards are asserts,** stripped from release builds (`AGENTS.md`,
  1bd3003). The guards in `g1889/` that raised became asserts.
- **STOCK offers DR and MF on every turn their owner takes** (Section 12,
  item 4, revised). Replay supplies the pass where Ruby skips the turn
  (Section 10.2).
- **The rest is cleanup:**
  - shared helpers in place of copies;
  - size asserts beside buffers sized by literals;
  - entity mutators that check before they write;
  - tests for untested legality edges.

  The RSS fingerprint matched after both batches. The run 2-3 batch
  changed no 1889 play: a hash of 800 seeded random games matched before
  and after.

**Deferred:**

- **1.6 and 3.9: BUY_TRAIN rescans the depot.** `_enumerate_buy_train`
  recomputes the corp's train room, the depot's types, and the emergency
  need for each source. `trains_cheapest_in_pool` and the cheapest-price
  helpers rescan the depot on every call. These are short scans, and
  Section 6.2's profile does not single out BUY_TRAIN. Compute them once
  when `_source_trains` next changes.
- **2.1: the forced rule in live play.** Live play's real state must run with
  the policy flags off, and nothing enforces that yet (pipeline-refactor §9).
- **3.1 and 3.5: the `GameState` API.** These quirks are shared with RSS:
  - `initialize_game` on a `from_buffer` row too small for the new player
    count allocates a private row, without warning, instead of reusing the
    caller's;
  - `_alloc` is the constructor's second positional parameter, so
    `GameState(4, 5)` ignores the 5;
  - `rebind` does not infer `max_players` from the row's width;
  - `_array` and `max_players` are writable from Python.

  `core/state.pyx` has the same four, and no caller trips them. Fix both
  engines together, if at all.

**Rejected:**

- **3.7: buffers sized by named constants.** Cython rejects `cpdef enum`
  constants as array dimensions ("Array dimension not integer"). The literal
  sizes and their import-time asserts stay.

**Not yet reviewed:**

- `g1889/routes/`, the route solver behind `route_max`;
- the table codegen, `g1889/data/generate.py` and `dump_1889.rb`, whose
  output `test_data_vs_ruby.py` checks against Ruby;
- `games/g1889.py`, the adapter's engine half;
- a pass over `g1889/` and `tests/g1889/` for fit with the RSS patterns and
  for test coverage.

To resume, have a session read the brief and run
`/code-review high g1889/routes g1889/data/generate.py g1889/data/dump_1889.rb games/g1889.py --max-findings all`.
Its findings go in the same file as run 4.
