# Shikoku 1889 Build Distance Design

**Status, 2026-11-10: designed, not built.** Nothing here is in the engine, the
extraction, or a model. Section 9 gives what has to be seen from the v1 run
before any of it is built. This replaces
[`extraction-design.md`](extraction-design.md) §8, which planned the same
feature per corp.

**Build distance** is the number of tile lays or upgrades, legal now, after
which two revenue centers are joined by track. The model receives it for every
pair of the map's 25 revenue centers, in every phase, with summaries of what
each lay on offer would do to it.

## 1. Why

A corp lays one tile an operating round, so a route that needs three lays is
three rounds away, with every other corp's turn and the stock rounds between.
Search at 800-1,600 simulations does not reach that far, so only the value
head can carry such a plan, and today the first two lays change nothing it can
easily read: CANDIDATE 101-108 show a lay's gain in reach and route revenue,
which are zero until the last lay.

**Measured** (2026-11-10; `scratchpad/build_distance_play/`, on 87 human games
and 41 self-play games of v0 at epoch 58, of which 24 are greedy):

- **Follow-through.** After a lay that only brings a corp closer to a center it
  does not reach, does its next lay bring that center closer again? As the
  excess over a random legal lay at the same decision: humans +48 points (63%
  against 15%; 95% interval over games 43-52), v0 greedy +9 (3-14), v0 sampled
  +1 (-5 to 7). Greedy games at full simulations are no better than those at
  200.
- **This is the gap the feature aims at.** It is not the largest: v0 passes
  with a lay legal at 52% of phase 4's track decisions and 45% of phase 5's
  (humans 10% and 13%), and 24% of its phase 5 lays do nothing (humans 6%),
  with a connecting lay legal in 66% of those. Those are one-lay misses, which
  the model's inputs already show, in games that stall in phase 4 with poor
  corps. v1 was built to fix the stalling. Section 9 waits on whether it does.

**Why per pair of centers, not per corp.** Track is public. One relation
between centers serves all seven corps, and two corps of one president
building toward each other. A corp's own distance to a center follows from it
through the corp's token cities (3.3), which the model already has as STATION.

## 2. Scope

- A build-distance module in the engine (Section 3, 5).
- Five relation ids, 29-33 (4.1).
- Token layout 3: layout 2 with 6 CORP slots (4.2), 9 CANDIDATE slots (4.3),
  and 5 HEX slots (4.4). `TOKEN_DIM` becomes 118.
- `nn/transformer-1889-v2.py`: v1 with five more relation channels and the new
  slots (Section 6).

**Out of scope:** distances under the next phase's colours; a second,
independent connection between a pair (Section 10); token placement, where v0
also falls short of the humans (3.5 tokens a game against 7.4); TRACK-time
route revenue deltas.

## 3. Definition

Validated by the prototype in `scratchpad/build_distance/` (`bd.pyx`, with a
Python reference in `ref.py`), on 2,400 human positions, 400 a game phase.

### 3.1 Points, traversals, and legal lays

- **Points.** A hex has six sides and, if it is a revenue center, a center. A
  side and the facing side of the hex across are one edge point, and exist
  only where both hexes leave the side open: 116 edge points. `HEX_EXIT_MASK`
  is not symmetric at B7 and J7.
- **A traversal** is one of a hex's 21 pairs of endpoints. Its cost is the
  fewest lays on that hex after which the two are joined inside it: 0 if the
  tile there joins them, 1 if one legal lay does, 2 if it takes two successive
  lays of colours the phase allows, and otherwise unreachable. Side to side
  through a center counts when both sides have track to it and the center can
  be passed (3.2). A plain hex joins only its own paths: no reversal where two
  paths share a side, and a crossing joins nothing.
- **A legal lay** is `_lay_scan`'s (`g1889/actions.pyx`) without its
  corp-specific parts.
  - **Kept:** the upgrade table, the phase's colours, the tile in supply,
    exits inside the hex's open sides, a private's block on its hex (TR on K4,
    ER on C4 while a player owns it), and no port tile.
  - **Dropped:** the corp's reach, its cash against the lay cost, and
    privates' own lays.
- **Phase 2 is yellow only,** so a connection that needs a green tile is
  unreachable there until phase 3.

### 3.2 Tokens

A city blocks a corp when its tokens fill its slots and the corp has none
there (`tile_blocks`). Home reservations never block. Offboards are never
passed.

**Chosen: a full city stops every path.** No path passes through a city whose
slots are full, whoever asks; a path may end there. A lay on that hex is
judged by the resulting tile's slots, so an upgrade that adds a slot opens it.

- A full city X is a revenue center, so A-X and X-B are both in the relation.
  A corp with a token in X has its own A-B distance by adding them; a corp
  without one correctly sees A and B as not joined through X.
- **Measured:** 93% of the positions have a full city (every one from phase
  3). The distance exact for a corp differs from this one for 17% of (corp,
  pair). Composing through the corp's own token cities (3.3) gives the exact
  distance for 99.3%. Of the rest, 0.6 points come out below it, nearly all a
  path that would reverse at the token city, and 0.04 above.

### 3.3 A corp's distance

D_c(Y), corp c's distance to center Y, is the shortest path from any of c's
token cities to Y over the graph whose nodes are the centers and whose edges
are the build distances, with only c's token cities as intermediate nodes. It
is 0 for the centers c reaches. A corp without a token has none.

Every corp-specific number in Section 4 is this composed distance, before and
after a lay alike, so a delta compares like with like, and the model is given
nothing it could not derive from the relation and STATION.

### 3.4 The search

One search per revenue center, 25 a state.

- **States** are directed: about to enter hex h at side s (232 of them).
  Centers are sources and targets only; passing a center hex is its
  side-to-side traversal, so one tile serves both halves at its true cost.
- **Shortest simple path.** A path uses no edge point and no center twice.
  Without this, a walk goes round a loop and back along its own track, which
  the engine does not allow: the plain shortest walk gave 474 false
  connections from 192 of 36,000 city sources.
- **The algorithm** is Dial's, with three rotating first-in-first-out buckets.
  Each state keeps the points the first path to reach it used (two 64-bit
  masks of edge points, one of centers), and a step that would reuse one is
  dropped. The center matrix is made symmetric by its minimum.
- **It stops at 5 lays** (4.1). Uncapped it costs a fifth more (Section 7).

**Measured:**

- **Distance 0 is the engine's connectivity:** no mismatch against
  `GRAPH.node_reached` on 72,000 city sources, on 12,096 corp reach sets, and
  on 2.58M city sources after hypothetical lays. A separate check of 2,814
  corp reach sets on other positions found none either.
- **Upper bound:** all 13,680 witness paths replayed through `MAP.lay`
  connect their pair in exactly the claimed lays.
- **Lower bound, one lay:** over 688,368 legal lays, none connects a pair
  claimed at 2 or more.
- **Exactness:** keeping one path per state makes this a heuristic for the
  shortest simple path. Center distances were exact on all 3,185 pairs an
  exhaustive search could decide where walk and search differ, and never above
  it. 95 pairs it calls unreachable are unproven.
- **The fresh map's farthest pair** is A10-K4 at 10 lays. Human positions
  reach 13, since tiles run out and laid tiles commit their hexes.

### 3.5 Known gaps

- **A returned tile.** An upgrade puts the tile it replaces back in supply,
  which can make a second lay legal that was not. All 8,806 two-lay
  connections the search missed (of 201M lay pairs) are this. Not modelled.
- **The last copy.** Each lay is checked against supply by itself. 4-8% of
  3-lay witness paths want a copy an earlier lay on the path used.
- **Three lays on one hex** (yellow, green, brown from phase 5) count as
  unreachable. Not measured.
- **Lays are not monotone.** 38% of legal lays raise some pair's distance, by
  committing their hex or taking a tile's last copy. So a lay's effect cannot
  be looked up from the distances before it; it is recomputed (Section 5).

## 4. What the model receives

### 4.1 Relations 29-33

| Id | Relation | Query ↔ key | Value |
|---|---|---|---|
| 29 | BUILD_0 | center hex ↔ center hex: joined by track now | 1 |
| 30 | BUILD_1 | 1 lay apart | 1 |
| 31 | BUILD_2 | 2 lays apart | 1 |
| 32 | BUILD_3 | 3 lays apart | 1 |
| 33 | BUILD_4_5 | 4 or 5 lays apart | 1 |

- Each is its own reverse, as TRACK_LINK is, and a pair has at most one of
  them. A pair 6 or more lays apart, or unreachable, has no record.
- **Why buckets:** a channel's bias is its weight times the value, so one
  channel holding the distance would make the bias linear in it.
- **Why 5:** in the human positions 18% of pairs are at 0, 9% at 1, 15% at 2,
  15% at 3, 24% at 4-5, and 16% at 6 or more, with 3% unreachable. A pair
  6 lays apart is two or more of a game's phases away.
- **Records:** a mean of 483 a state and at most 600. The edge bound rises
  from 2,642 to 3,242, inside `max_relation_edges` = 4,096.
- **Emission is the reading model's.** v1 gives ids past 28 no weight, but the
  eval server sizes a launch by every record above value 0, so sending these
  to v1 would cost its run GPU time for nothing. Relation records gain a
  version, as tokens have a layout: 1 for ids 0-28, 2 with 29-33. Each
  evaluator asks for its model's.

### 4.2 CORP, layout 3 (83)

For each corp with a token on the map, by D_c (3.3); zero otherwise.

| Slots | Feature |
|---|---|
| 1-76 | as layout 2 |
| 77, 78, 79 | centers it does not reach at distance 1, 2, 3, each /10 |
| 80, 81 | revenue now of the centers at distance 1, and at 2, each /300 |
| 82 | the nearest unreached center's distance /6, 1 if none within 5 |

These give the value head each corp's prospects directly, in every phase. The
model could compose them from BUILD and STATION; the slots save it the layers.

### 4.3 CANDIDATE, layout 3 (118)

For the decision corp, the buyer at ER_BONUS, comparing D_c before the lay
with D_c after it. Slot 102 already counts the centers the lay reaches.

| Slots | Feature |
|---|---|
| 1-108 | as layout 2 |
| 109, 110 | unreached centers the lay brings to distance 1, to distance 2, each /5 |
| 111 | centers brought closer that stay at 3 or more /10 |
| 112 | revenue now of the centers brought to distance 1 /200 |
| 113 | centers within 5 that the lay puts farther or cuts off /10 |
| 114, 115 | other corps: centers brought closer to within 3, and centers within 3 put farther, summed over corps, each /20 |
| 116, 117 | center pairs of the map brought closer, and put farther, each /50 |

The candidate head reads 109-117 raw, with 99-108.

### 4.4 HEX, layout 3 (110)

At TRACK, over the decision corp's lay candidates on the hex (HEX 102 counts
them); zero elsewhere and at hexes with none.

| Slots | Feature |
|---|---|
| 1-104 | as layout 2 |
| 105 | the most centers a candidate here reaches /5 (the best of CANDIDATE 102) |
| 106, 107 | the most it brings to distance 1, to distance 2, each /5 |
| 108 | the fewest centers a candidate here puts farther /10 |
| 109 | the most centers of other corps a candidate here puts farther /20 |

`track_hex` reads 105-109 raw, with 92-102.

**This part is optional and the dearest** (Section 7). TRACK shows no outcome
today; search finds it by expanding TILE one ply down, and would find these
the same way through the candidates of 4.3. Section 9's offline test runs with
and without it.

## 5. Engine and extraction

- **A module,** `g1889/entities/build.pyx`: the traversal table, the 25
  searches, and the composition of 3.3, nogil, on a state. Ported from
  `bd.pyx`, with the engine's accessors inlined; the prototype calls them
  across modules.
- **No cache in the row.** The center matrix alone is 625 values against the
  row's 893 slots, and a state's distances cost about 36 µs (Section 7).
  Every fill recomputes them. The row and the replay buffer do not change.
- **Candidates** are tried on the scratch state `_fill_candidates` already
  lays them on: lay, table, searches, composition. The lookup from stored
  distances that the prototype tried is wrong for 40% of candidates (6%
  counting only falls), because of 3.5's last item.
- **Layout 3** fills beside layouts 1 and 2, which stay as they are.
  `tests/g1889/test_layout1_parity.py` and layout 2's tests pass unchanged.

**Test:**

- The prototype's checks, as tests on fixture states: distance 0 against
  `GRAPH.node_reached`; a witness replay for distances 1-3; no pair at 2 or
  more connected by any one legal lay.
- **Per-choice truth:** each slot of 4.3 against the state after the real lay,
  and each of 4.4 against the hex's candidates.
- **Purity:** the same tokens and records whatever the row's caches hold.
- **Cost,** as `cloud/profile_1889.py`'s extraction section measures it
  (Section 7's budget).

## 6. The model

`nn/transformer-1889-v2.py` is v1 with:

- `INPUT_LAYOUT_VERSION` 3 and relation version 2;
- 30 dynamic relation channels: v1's 25, then ids 29-33. The BUILD records
  join HEX rows, which the trunk holds;
- `corp_proj`, `candidate_proj`, and `hex_proj` wider by the new slots, and
  the raw reads of 4.3 and 4.4.

**It can start from a v1 checkpoint.** `relation_bias_mult` is zero at
initialization, and the new input columns and raw-slot weights can be set to
zero, so a v2 built from v1's weights gives v1's outputs until it trains.
**Test:** exactly that, on fixture states. This is what makes Section 9's
comparison cheap: both arms start from one checkpoint.

## 7. Cost

**Measured** on one core of a 9950X3D, default build, by the prototype:

| | µs, mean (99th percentile) |
|---|---|
| The traversal table | 4.7 (7.3) |
| 25 searches, stopped at 3 | 19.7 (41.6) |
| stopped at 5 | 31.2 (55.1) |
| stopped at 10 | 38.0 (60.2) |
| uncapped | 37.0 (53.4) |
| One candidate: copy, lay, table, 25 searches uncapped | 42.5 (60.6) |

For scale, a leaf's whole extraction in layout 2 averages 20 µs.

**CPU, estimated** per leaf, averaged over the decision mix:

| Part | Where | Added, µs |
|---|---|---|
| Relation and CORP slots (4.1, 4.2) | every state | about 36 |
| CANDIDATE slots (4.3) | TILE, 5-7% of states, about 6 candidates | about 15 |
| HEX slots (4.4) | TRACK, 9% of states, about 20 candidates | about 70 |

At 25,000 leaves a second, 120 µs is 3 cores of the 9950X's 16. Training uses
about 20% of its 32 threads today.

**GPU, estimated.** The 483 records roughly double a state's, which average
about 500 on human positions. The 5090's profile (model-design §12.7) has a
launch of 256 at 9.24 ms with 512 records a state, 9.41 at 1,024, and 10.03
at 2,048: 2-9% more, by where launches land. **Test:** the profile, with
relation version 2, before any run.

## 8. Decisions taken

Settled with the user on 2026-11-10:

1. Distances are between revenue centers, shared by all corps.
2. Only lays legal in the current phase count.
3. A full city stops every path; corps compose through their token cities.
4. No hex-to-center relation: it would add about 470 records a state at 3
   lays. HEX and CANDIDATE summaries carry a lay's effect.
5. Nothing is built or sent to the 5090 until Section 9's first gate.

## 9. Before building, and how to judge it

1. **The same comparison on v1,** at epoch 30-60: 24 or more greedy self-play
   games through `scratchpad/build_distance_play/`. If v1 still passes with a
   lay legal at half its phase 4-5 track decisions, that comes first, and this
   waits. If the passing and the useless lays are near the humans' and
   follow-through is still near chance, build it.
2. **A probe,** on the same checkpoint: can a linear readout of v1's final HEX
   rows give a center pair's bucket of 4.1? If it can, the trunk has learned
   the distances and the relation adds little; the CORP and CANDIDATE slots
   may still.
3. **Offline,** as v1-design Section 10 ran Tier C, on v1's replay buffer:
   v1 against v2 without 4.4 and v2 with it, from one checkpoint. Measures:
   TRACK and TILE policy KL and held-out value MSE. This scores how well each
   imitates v1's search, which never had the feature, so a null result is weak
   evidence, as `frontier_rel`'s was.
4. **In play,** which is the test: two arms from one v1 checkpoint, v1
   continued and v2, some tens of epochs each, then a tournament between them
   and item 1's follow-through measure on each. This is days of the 5090 and
   is the user's to schedule.

## 10. Open

- **The buckets and the stop at 5** (4.1), against stopping at 3 (338 records)
  or sending every finite pair (580).
- **Whether 4.4 is worth its cost,** which item 3 above answers.
- **Returned tiles** (3.5): a real two-lay plan the search does not see.
- **The next phase's distances:** the same search on a table built with the
  next phase's colours, to show what a train purchase opens. Deferred.
- **A second connection** between a pair that shares no track with the first,
  for corps running two trains: the fewest lays for two such paths, which is a
  two-unit minimum-cost flow on the same graph. Deferred until the first shows
  a gain.
