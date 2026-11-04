# Shikoku 1889 Token Data Spec

This document is the token contract for 1889's layout 1, which step 8's
`nn/transformer-1889.py` reads: the rows, their slots, the relations, and
which slots each policy head reads. It matches `g1889/token_data.pyx/.pxd`
and `g1889/relations.pyx/.pxd`; where they differ, the code is right and
this document is wrong. The rationale is in
[`1889/extraction-design.md`](1889/extraction-design.md) §4-7, and
`tests/g1889/test_token_data_doc.py` pins the tables below against the code.

The `Game` adapter (`games/g1889.py`) exposes the extractors:
`num_tokens(max_players)`, `token_dim(1)`, `num_relations` (25),
`max_relation_edges` (2,048), and the six fills, which take
`layout_version=1` for tokens.

## Layout id

`get_token_dim(layout_version)` and `get_token_widths(max_players,
layout_version)` accept only 1 and raise `ValueError` for any other id. A
1889 model module declares `INPUT_LAYOUT_VERSION = 1`.

## Shapes

- **Tokens:** `(num_tokens(max_players), 108)` float32, sent to the eval
  server as float16. `TOKEN_DIM` = 108, the CANDIDATE width; every row is
  zero-padded to it, and `get_token_widths` gives each row's width.
- **Dense relations:** `(25, T, T)` uint8, queries by keys, T =
  `num_tokens(max_players)`.
- **Sparse records:** `(2048, 4)` uint8 rows `(relation, query, key,
  value)`, zero past the count. A record with value 0 is padding.
- **Players:** 3 ≤ `num_players` ≤ `max_players` ≤ 5. Batches take raw rows,
  which may mix player counts at one capacity.

Inputs depend only on the row, never on the runtime policy flags.
Extraction writes the refreshed GRAPH and route caches back into the row it
reads (extraction-design §4), so rows passed to it must be writable; the
output is the same whether those caches were fresh or stale.

## Rows

| First row | Type | Count | Width | Visible (slot 0 = 1) |
|---|---|---|---|---|
| 0 | GLOBAL | 1 | 44 | always |
| 1 | PRICE | 1 | 24 | at BID, PAR, TRAIN_PRICE, PRIVATE_PRICE |
| 2 | DIVIDEND | 1 | 8 | at DIVIDEND |
| 3 | CORP | 7 | 51 | always |
| 10 | PRIVATE | 7 | 18 | unless not in play (UTF below 4 players) |
| 17 | TRAIN | 6 | 13 | always |
| 23 | HEX | 52 | 105 | always |
| 75 | CANDIDATE | 15 | 108 | row k < the candidate count, at TILE and ER_BONUS |
| 90 | PLAYER | max_players | 38 | real players |

- `num_tokens(max_players) = 90 + max_players`: 93, 94, 95 at capacities
  3-5. Players come last, so every other index is the same at every
  capacity, and records name tokens by these indices.
- A hidden row is all zero, slot 0 included: the phase rows away from their
  decisions, candidate rows past the count, privates not in play, and the
  padded player rows `[num_players, max_players)`.
- Identity comes from row order: corps in `CorpId` order (AR, IR, SR, KO,
  TR, KU, UR), privates in `PrivateId` order (TR, MF, ER, SMR, DR, SIR, UTF),
  trains by type (2, 3, 4, 5, 6, D), hexes in `HEX_IDS` order, players in
  canonical (seat) order.

## Conventions

- **Slot 0** of every row is the key mask.
- **`is_selected`** slots pick one row of a type: CORP 1, 2, 3; PRIVATE 1, 2;
  TRAIN 1; HEX 1; PLAYER 1. At most one row of the type sets each. Slot 1 of
  GLOBAL, PRICE, DIVIDEND, and CANDIDATE is an ordinary feature.
- **The actor** is `TURN.active_player`, whom search backs values up to;
  none at GAME_OVER or an automated step.
- **The decision corp** is the pending corp at PAR, the first crowded corp
  at DISCARD (`trains_first_crowded`), and the operating corp at every other
  OR decision (BUY_PRIVATE_PRE, SELECT_TRAIN, TRAIN_PRICE,
  EMERGENCY_SELL_AMOUNT, BUY_PRIVATE_POST, PRIVATE_PRICE, ER_BONUS, and an
  OR's PORT included). The
  auction, STOCK, SELL_AMOUNT, and an SR's PORT have none; CORP 3 marks the
  corp a sale is of.
- Actor and decision-corp features are 0 when there is none.
- **Per-choice features** describe engine-legal choices, never the policy
  filter; the legal mask carries the restriction. They are 0 at other
  decisions.
- **Divisors:** ¥1000 for cash and train-scale money (treasuries, cash,
  values, list prices, the PRICE row), ¥100 for share prices, private
  values, bids, hex and route revenues, and token prices, ¥7000 for the
  bank, and counts by their natural maximum (stated per slot). Revenue
  deltas are signed.

## GLOBAL (44)

| Slots | Feature |
|---|---|
| 1-19 | decision phase one-hot, in `DecisionPhase` order (AUCTION, BID, STOCK, PAR, SELL_AMOUNT, BUY_PRIVATE_PRE, TRACK, TILE, TOKEN, DIVIDEND, DISCARD, BUY_TRAIN, SELECT_TRAIN, TRAIN_PRICE, EMERGENCY_SELL_AMOUNT, BUY_PRIVATE_POST, PRIVATE_PRICE, ER_BONUS, PORT); all zero without a decision |
| 20-22 | round: auction, stock, operating; all zero at GAME_OVER |
| 23-28 | game phase one-hot: 2, 3, 4, 5, 6, D |
| 29 | SR number ≤ 1: the first SR bars sales |
| 30-32 | the OR's number in its set, one-hot (1, 2, 3); all zero outside an OR |
| 33 | ORs in the set /3: the current set's in an OR, else the coming set's (`PHASE_ORS`) |
| 34, 35 | bank cash /7000 (negative once broken), bank broken |
| 36-38 | player count one-hot: 3, 4, 5 |
| 39, 40 | all-pass auction payouts /3, A's auction discount /20 |
| 41-43 | the train's source at SELECT_TRAIN and TRAIN_PRICE: the depot, the pool, a corp (CORP 2 names it) |

## PRICE (24)

Visible at BID, PAR, TRAIN_PRICE, and PRIVATE_PRICE. The amounts are
indexed by phase-local id, so a head reads slot 5 + id for action id.

| Slots | Feature |
|---|---|
| 1-4 | kind: BID, PAR, TRAIN_PRICE, PRIVATE_PRICE |
| 5-21 | ¥ of phase-local ids 0-16, /1000. BID: grid points 1-16 (`price_amount`); id 0, a closed auction's drop, has none. PAR: ids 0-5, the President's Certificate, 2 × `PAR_PRICE`. TRAIN_PRICE: ids 0-4; PRIVATE_PRICE: ids 0-3 (`price_amount`). Ids not legal are 0 |
| 22 | reference /1000: the private's face (BID, PRIVATE_PRICE), the train's list price (TRAIN_PRICE), 0 (PAR) |
| 23 | payer's funds /1000: the actor's `player_available` for the private (BID), the actor's cash (PAR), the decision corp's cash (TRAIN_PRICE, PRIVATE_PRICE) |

## DIVIDEND (8)

Visible at DIVIDEND, for the operating corp's revenue r (`route_revenue`).

| Slots | Feature |
|---|---|
| 1 | r /100 |
| 2 | per-share payout ⌈r/10⌉ /100 |
| 3, 4 | share price after a payout and after a withhold /100 (`market_moved` right and left) |
| 5 | treasury after a withhold /1000 |
| 6, 7 | the president's payout and the corp's payout for its pool shares /100, each ⌈shares × r/10⌉ |

## CORP (51)

| Slots | Feature |
|---|---|
| 1 | is_selected: the decision corp |
| 2 | is_selected: the train's seller, at SELECT_TRAIN and TRAIN_PRICE from a corp |
| 3 | is_selected: the corp a sale is of, at SELL_AMOUNT and EMERGENCY_SELL_AMOUNT (`pending_corp`) |
| 4, 5, 6 | parred, floated, home token placed |
| 7, 8 | par price, share price /100 |
| 9, 10 | market row /10, market column /14 |
| 11-14 | market zone: normal, par, no certificate limit, unlimited |
| 15-19 | share price after selling 1-5 shares /100 (a row down per share): every corp's at STOCK and BUY_TRAIN, the row CORP 3 selects at SELL_AMOUNT and EMERGENCY_SELL_AMOUNT |
| 20 | sold out: no IPO or pool shares |
| 21 | cash /1000 |
| 22, 23 | IPO shares /10, pool shares /10 |
| 24, 25 | tokens left /3, next token's price /100 |
| 26-31 | trains by type /4: 2, 3, 4, 5, 6, D |
| 32 | trains over the phase's limit |
| 33, 34 | last revenue /100, route max /100 (the solver's best now, `route_cache`) |
| 35 | route available: some token's walk reaches two revenue centers |
| 36 | must buy a train (every corp) |
| 37 | in an emergency (the decision corp only) |
| 38, 39 | OR position (index + 1)/7, 0 outside the OR's order; operated this OR |
| 40, 41, 42 | hexes reached /52, centers reached /25, their revenue now /100 (GRAPH) |
| 43 | reached cities with a slot for the corp /5 |
| 44 | sold by the actor this SR |
| 45 | sold this SR turn |
| 46 | sold in this turn's emergency |
| 47 | bought or parred this SR turn |
| 48 | the actor's largest sale now /5: 0 outside an SR and when the turn bars it |
| 49 | the actor's largest emergency sale /5 |
| 50 | privates owned /3 |

Slots 7-19 are 0 while the corp is unparred.

## PRIVATE (18)

| Slots | Feature |
|---|---|
| 1 | is_selected: BID's private (a closed auction's, else the pending one), PRIVATE_PRICE's pending private |
| 2 | is_selected: the topmost unsold private, which AUCTION's buy takes |
| 3-6 | owner: unsold, a player, a corp, closed |
| 7, 8, 9 | face value, revenue now (0 once closed), price now with A's discount (unsold only) /100 |
| 10, 11, 12, 13 | while unsold: high bid /100, bids /5, minimum bid /100, the actor's bid /100 |
| 14 | blocking its hex (TR on K4, ER on C4) |
| 15 | MF's port tile used |
| 16 | the decision corp meets `can_buy_private` now, the slot's legality at BUY_PRIVATE_PRE and BUY_PRIVATE_POST |
| 17 | the actor meets `can_exchange_dr` now (DR only), the slot's legality at STOCK and BUY_PRIVATE_PRE |

## TRAIN (13)

| Slots | Feature |
|---|---|
| 1 | is_selected: TRAIN_PRICE's train type |
| 2 | list price /1000 |
| 3, 4, 5 | depot copies /6 (0 for D), unlimited (D), pool copies /4 |
| 6, 7 | the depot sells it now; the cheapest train available |
| 8, 9 | rusted; rusts when the next phase starts |
| 10 | at SELECT_TRAIN: Δ route max /100 if the decision corp adds one of this type from the pending source |
| 11 | at SELECT_TRAIN: Δ route max /100 if it trades one of this type in for a D |
| 12 | at DISCARD: Δ route max /100 if it discards one of this type |

Slots 10-12 are set only for the decision's legal choices, and slot 10 only
while the corp holds fewer than 4 trains. They ignore what a purchase does
to the phase (rusting, limits).

## The tile block (76)

A tile as laid: HEX rows hold the current tile's and CANDIDATE rows the
candidate's, both at slots 2-77 (`TILE_BLOCK` = 2), so one projection reads
both. Offsets are from slot 2.

| Offsets | Feature |
|---|---|
| 0-39 | supply tile one-hot, ids 0-39; all zero for a printed tile |
| 40 | printed tile |
| 41-61 | path mask at its rotation, absolute sides, one slot per endpoint pair in `PAIRS` order: (0,1), (0,2), …, (0,6), (1,2), …, (5,6), where 6 is the center |
| 62-67 | color: white, yellow, green, brown, gray, red |
| 68-70 | center: city, town, offboard |
| 71, 72 | revenue now /100 (an offboard's by phase), city slots /3 |
| 73-75 | label H, K, T |

## HEX (105)

| Slots | Feature |
|---|---|
| 1 | is_selected: TILE's pending hex; C4 at ER_BONUS |
| 2-77 | the tile block of the current tile |
| 78 | revenue to a D train /100 |
| 79, 80 | mountain, water (the printed tile's terrain) |
| 81, 82, 83 | tokens /3, slots open to any corp /3, a home reservation still open |
| 84 | lay cost /100 of the current tile, before SMR's discount |
| 85 | blocked by a private |
| 86-91 | dangling exits by side: track at an open side whose hex across has none facing back |
| 92 | upgrade targets now /15: in supply, of a phase color, exits on open sides, not the port tile; connectivity and cost aside |
| 93 | upgrade targets in supply whatever the phase /15 |
| 94-99, 100 | the decision corp's reached sides by side; its reach of the center |
| 101 | blocked for the decision corp: every slot filled by other corps |
| 102 | at TRACK: the decision corp's lay candidates here /15 |
| 103, 104 | at TOKEN: the decision corp may token here; its Δ route max /100 if it does |

## CANDIDATE (108)

Row k is lay candidate k: TILE's id k, or ER_BONUS's id 1 + k, in the order
`enumerate_lay_candidates` (TILE, the operating corp at the pending hex) and
`er_candidates` (ER_BONUS, on C4) give them. Outcomes are the decision
corp's, the buyer at ER_BONUS, against the current state.

| Slots | Feature |
|---|---|
| 1 | an ER_BONUS candidate (0 at TILE) |
| 2-77 | the tile block of the candidate at its rotation, with the new tile's revenue and slots |
| 78-98 | paths added: in the candidate's mask and not the current tile's, `PAIRS` order |
| 99 | lay cost /100, after SMR's discount (0 at ER_BONUS: ER's seller pays) |
| 100 | the tile's copies left after the lay / its `TILE_COUNT` |
| 101, 102, 103 | Δ hexes reached /52, Δ centers reached /25, Δ reached revenue /100 |
| 104 | Δ route max /100 |
| 105 | route available after the lay |
| 106, 107 | other corps whose reached centers grow /6; their summed Δ centers reached /25 |

## PLAYER (38)

| Slots | Feature |
|---|---|
| 1 | is_selected: the actor |
| 2-6 | seats after the actor, one-hot (slot 2 is the actor); all zero without an actor |
| 7 | holds priority |
| 8, 9 | cash /1000, value /1000 |
| 10, 11 | certificates / the player count's limit, certificates /25 |
| 12-18 | shares by corp /10 |
| 19-25 | sold this SR, by corp |
| 26-32 | bids by private /100 |
| 33, 34 | passed, bankrupt |
| 35, 36 | presidencies /7, privates owned /3 |
| 37 | on the operating corp's president: an emergency sale since the turn's last purchase |

## Known unbounded slots

Values are of order 1 under self-play's flags. Under uncapped engine-legal
play, the auction can loop on passes, paying the sold privates' revenue
each round without end: GLOBAL 39 then grows without bound, and with the
players' cash (PLAYER 8-9) the bid slots grow too: PRICE 5-23 at BID,
PRIVATE 10, 12, and 13, and PLAYER 26-32. Self-play's
`auction_pass_payout_cap` (1) bounds them.

## Relations

### Dynamic relations (IPC)

`num_relations` = 25. Each pair's reverse is the next id; TRACK_LINK is its
own. No (relation, query, key) cell repeats, so the dense planes hold
exactly the records.

| Id | Relation | Query → key | Value |
|---|---|---|---|
| 0 | PLAYER_HOLDS | player → corp: holds shares | 1 |
| 1 | CORP_HELD_BY | corp → player | 1 |
| 2 | PLAYER_SHARES | player → corp | shares, 1-10 |
| 3 | CORP_SHARES | corp → player | shares, 1-10 |
| 4 | PRESIDES | player → corp: its president | 1 |
| 5 | PRESIDENT | corp → player | 1 |
| 6 | PLAYER_OWNS | player → private | 1 |
| 7 | OWNED_BY_PLAYER | private → player | 1 |
| 8 | CORP_OWNS | corp → private | 1 |
| 9 | OWNED_BY_CORP | private → corp | 1 |
| 10 | BIDS_ON | player → private: an open bid | 1 |
| 11 | BID_BY | private → player | 1 |
| 12 | CORP_TRAINS | corp → train type | trains, 1-4 |
| 13 | TRAIN_HELD_BY | train type → corp | trains, 1-4 |
| 14 | STATION | corp → hex: a token there | 1 |
| 15 | STATION_OF | hex → corp | 1 |
| 16 | REACHES_HEX | corp → hex: GRAPH reaches a side | 1 |
| 17 | HEX_REACHED_BY | hex → corp | 1 |
| 18 | REACHES_CENTER | corp → hex: GRAPH reaches its center | 1 |
| 19 | CENTER_REACHED_BY | hex → corp | 1 |
| 20 | TRACK_LINK | hex ↔ hex: track on both sides of their shared side | 1 |
| 21 | CANDIDATE_AT | candidate → its hex | 1 |
| 22 | CANDIDATE_OF | hex → candidate | 1 |
| 23 | CANDIDATE_EXIT | candidate → the hex across each of its exits | 1 |
| 24 | EXIT_OF | hex → candidate | 1 |

- **The model scales** the two count pairs: shares (2, 3) by 1/10 and
  trains (12, 13) by 1/4. Every other plane is binary.
- **Reach** (16-19) is GRAPH's, Ruby's `connected_hexes` and
  `connected_nodes`: a reached hex is one where a lay can start, the
  network's own hexes and those across its dangling exits. A corp without a
  token reaches nothing.
- **Candidates** (21-24) exist only at TILE and ER_BONUS, for the visible
  candidate rows.
- A model may ignore trailing ids; later steps append theirs after 24.

### Static relations (model buffers)

`static_relation_planes(max_players)` returns `(10, T, T)` uint8 planes that
never change. The model calls it and holds them as non-persistent buffers;
they never cross IPC, and the `Game` adapter has no field for them.

| Id | Plane | Query → key |
|---|---|---|
| 0-5 | HEX_SIDE_0 to HEX_SIDE_5 | hex → the hex across side s (`HEX_NEIGHBOR`) |
| 6 | HOME | corp → its home hex |
| 7 | HOME_OF | home hex → corp |
| 8 | ABILITY_HEX | private → hex: TR → K4, ER → C4, MF → its 4 port hexes |
| 9 | ABILITY_OF | hex → private |

### The edge bound

The records per state are bounded by **1,844**: 140 for shares, 14
presidency, 14 ownership, 70 bids, 56 trains, 30 stations, 728 reached
hexes, 350 reached centers, 232 track links, and 210 candidate edges.
`max_relation_edges` = **2,048**, and the fills assert the count. A model
takes E from its input's shape, not from a constant.

## Per-choice readouts

What each policy head reads for its decision's choices; every head also has
the trunk's context (the decision corp's and actor's rows, GLOBAL). Action
slots hold only what can ever be legal (engine-design §12 item 8), so TRACK,
TOKEN, DISCARD, and PORT reach their rows through the tables of
`g1889/data/tables.pxd`; every HEX and TRAIN row stays, and at ER_BONUS only
CANDIDATE rows 0-7 can be visible.

| Decision | Choice (phase-local id) | Row | Slots |
|---|---|---|---|
| AUCTION | buy (1); bid on p, p ≥ 1 (1 + p) | the PRIVATE row slot 2 selects; PRIVATE p | 7-13 |
| BID | point k (1-16), drop (0) | PRICE; the private slot 1 selects | 5 + id |
| STOCK | IPO, pool, par, sell corp c (1 + c, 8 + c, 15 + c, 22 + c); DR (29); MF (30) | CORP c; PRIVATE DR, MF | 7-25, 44-48; 17; 15 |
| PAR | par index i (0-5) | PRICE | 5 + id |
| SELL_AMOUNT | n shares (n − 1) | the CORP row slot 3 selects | 15 + id |
| BUY_PRIVATE_PRE | done (0); private p (1 + p); DR (8); MF (9) | PRIVATE p; PRIVATE DR, MF | 16; 17; 15 |
| TRACK | hex `TRACK_HEXES[i]` (1 + i) | that HEX row | 92-102 |
| TILE | candidate k (k) | CANDIDATE k | 2-107 |
| TOKEN | hex `TOKEN_HEXES[i]` (1 + i) | that HEX row | 103, 104 |
| DIVIDEND | withhold (0), payout (1) | DIVIDEND | 1, 4, 5 for withhold; 1-3, 6, 7 for payout |
| DISCARD | type `DISCARD_TYPES[i]` (i) | that TRAIN row | 12 |
| BUY_TRAIN | the depot (1), the pool (2), corp c (3 + c); emergency sale of corp c (10 + c); bankrupt (17) | TRAIN rows; CORP c | 2-9; 15-19, 46, 49 |
| SELECT_TRAIN | type t (t); a D for traded `TRADE_IN_TYPE[i]` (6 + i: 4, 5, 6) | TRAIN t; TRAIN `TRADE_IN_TYPE[i]` | 10; 11 |
| TRAIN_PRICE | point k (0-4) | PRICE; the TRAIN row slot 1 selects, its seller (CORP 2) | 5 + id |
| EMERGENCY_SELL_AMOUNT | n shares (n − 1) | the CORP row slot 3 selects | 15 + id |
| BUY_PRIVATE_POST | done (0); private p (1 + p); DR (8) | PRIVATE p; PRIVATE DR | 16; 17 |
| PRIVATE_PRICE | point k (0-3) | PRICE; the private slot 1 selects | 5 + id |
| ER_BONUS | pass (0); candidate k (1 + k) | CANDIDATE k | 2-107 |
| PORT | `PORT_LAY_HEX[i]` at `PORT_LAY_ROT[i]` (i) | that HEX row | the model's static table of tile 437's mask at that rotation |

BUY_TRAIN reads no outcome deltas: SELECT_TRAIN's, the next decision, show
them, as TILE's candidate rows show TRACK's.
