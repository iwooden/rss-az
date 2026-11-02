# Pipeline Refactor for Shikoku 1889

Status: steps 0, 1, and 5 and the ACQ cap commit done (2026-11-01); the
rest is plan. This expands Section 11
of the model blueprint, [`1889.md`](1889.md). The rules are in
[`1889-rules.md`](1889-rules.md).

## 1. Goal and scope

- Make MCTS, self-play, the eval server, replay, and the trainer run any game
  through one adapter object, so the 1889 engine plugs in by filling one in.
  The algorithms in those modules are already game-agnostic. The work is in
  their engine calls, game constants bound at import, and a few RSS-only
  features (Section 2).
- The work lives on `az-1889` and reaches `main` when the branch merges.
- Priorities, in order:
  1. a working 1889 engine, with tests;
  2. the generic pipeline;
  3. a 1889 model definition that trains.
- Deferred until needed: 1889 live play and anything else new in
  `utils_18xx`, and 1889 support in `train/analyze_game.py`,
  `train/tournament.py`, `train/strategy_data.py`, and `train/debug_trace.py`.
- RSS keeps working throughout. `tests/test_v2_parity.py` passes with its
  existing fixture, and the live v3 checkpoint loads and produces the same
  outputs. RSS tooling (`utils_18xx/live.py` and the four `train/` tools above)
  gets mechanical call-site updates when contracts change. It is not
  generalized.
- Purity is not a goal. All games share one adapter class, and it may carry
  game-specific members in a labeled block. There are no per-game subclasses.
- The RSS packages (`core/`, `entities/`, `phases/`) stay where they are.
  Moving them under `rss/` would rewrite the imports of about 70 Cython files
  and gain nothing. New shared code does not go in `core/`.

## 2. Where the pipeline touches RSS today

| Module | RSS coupling |
|---|---|
| `mcts/mcts_core.pyx`, `mcts/node.py` | None. |
| `mcts/search.py` | Engine calls in `run_search` root setup (`:436-466`) and expansion (`:671-694`). RSS runtime flags copied onto `StatePool` (`:457-459`, `:566-571`). ACQ root-prior filter (`:310`, called at `:503`). |
| `mcts/evaluator.py` | Single-state paths enumerate the full engine-legal set (`:453`, `:517`). Terminal values read RSS net worth (`:294-307`). The cross-president prior adapter (`:626-773`) is used only by tournament and live play. |
| `train/eval_server.py` | Imports extractors and dimensions (`:99-124`). The single-state path enumerates the full legal set (`:1446`). No rules logic. |
| `train/replay_buffer.py`, `train/trainer.py` | Dimensions, the two extractors, phase names. |
| `train/self_play.py` | The `play_game` loop is generic. RSS-only: `AcquisitionStats` (`:199`), `_StrategyTraceBuilder` (`:469`), end-of-game share and company stats, the trace path's ACQ filter (`:1070`), and `GameState` construction with RSS flags. |
| `train/config.py` | RSS engine flags (`:128-129`), the ACQ cap and its validation (`:15-37`, `:174`). |
| `nn/policy_layout.py`, `train/policy_metrics.py` | Phase sizes, names, and pass-slot phases from `core.data`, bound at import. |
| `nn/__init__.py` | Imports `core.token_data` and loads `transformer-v2.py` at import. Requires `RSSTransformerNet`, `TransformerConfig`, and a fixed list of RSS keyword arguments. |

The widest problem is import-time constants. `UNIFIED_LOGIT_DIM`,
`MAX_ACTION_SIZE`, `NUM_ATTENTION_RELATIONS`, `TOKEN_DIM`, `build_action_lut`,
`get_layout`, `get_num_tokens`, `PHASE_NAMES`, and similar appear about 150
times across 13 pipeline modules, some as module globals in hot code (`U_DIM`,
`eval_server.py:126`). Each becomes an attribute of the adapter or its policy
layout. The change is mechanical but wide.

## 3. The adapter

### 3.1 Shape

- `games/__init__.py` defines one `Game` dataclass of constants and callables,
  and `get_game(config) -> Game`, keyed by a new `config.game` (`"rss"` or
  `"1889"`).
- `games/rss.py` and `games/g1889.py` each export `make_game(config)`. The RSS
  module wraps the existing `cpdef` functions and holds every RSS import the
  pipeline used to make.
- Each process builds its own `Game` from its config. Configs pickle, so this
  is spawn-safe, and nothing game-specific is decided at import time.
- Fields hold the engine's `cpdef` functions directly where possible. Hot
  loops bind fields to locals once, so dispatch costs one attribute lookup per
  call site, not per call. `mcts_core` does not change.

```python
@dataclass(frozen=True)
class Game:
    name: str                       # stored in configs, checkpoints, replay metadata
    policy: PolicyLayout            # Section 3.3

    # Raw state: int16 rows padded to a player capacity
    state_size: Callable            # max_players -> row width
    storage_player_capacity: Callable  # row width -> max_players
    num_players_from_row: Callable  # raw row -> actual player count
    new_game: Callable              # (num_players, max_players, seed) -> state
    scratch_state: Callable         # (root_state, row) -> wrapper carrying root's runtime flags

    # Engine
    apply_action: Callable          # validate, dispatch, auto-chain; returns a status code
    active_player: Callable
    is_terminal: Callable
    decision_phase: Callable
    enumerate_policy_actions: Callable  # (state, buf) -> count; the searched action set
    final_scores: Callable          # canonical order; feeds compute_terminal_values

    # Model inputs: dense planes for trainer/NNEvaluator, sparse records for eval IPC
    num_tokens: Callable            # max_players -> token count
    token_dim: Callable             # token layout version -> width
    num_relations: int
    max_relation_edges: int
    fill_tokens: Callable
    fill_tokens_batch: Callable
    fill_relations_dense: Callable
    fill_relations_dense_batch: Callable
    fill_relations_sparse: Callable
    fill_relations_sparse_batch: Callable

    # Self-play diagnostics (Section 4.3)
    new_selfplay_stats: Callable    # num_players -> collector

    # --- RSS-only; None for other games ---
    # e.g. the strategy-trace builder used by train/strategy_data.py
```

### 3.2 Conventions every game follows

- Status codes match RSS: 0 OK, 1 GAME_OVER, 2 INVALID (3 PAUSED for step
  mode, optional).
- Action ids are phase-local and fit `uint16`. The NN sees unified slots
  through the policy layout's LUT.
- Values are per player, in canonical order, padded to the model's player
  capacity.
- Two action sets. Engine legality is what the driver validates against;
  replay and live play need all of it. The policy set is what search expands
  and what masks and targets cover. Policy ⊆ engine-legal. The adapter exposes
  only the policy set.
- States expose their raw row as `_array` and their player capacity as
  `max_players`, and `rebind(row, num_players, max_players=)` repoints a state
  at another row without allocating. The pipeline uses these directly; they
  are rules for every game's state, not adapter fields.
- Model inputs depend only on the row, never on runtime flags: batch
  extractors build flagless states from raw rows.
- Runtime flags travel with the state. RSS keeps `v3_behavior`,
  `acq_same_president`, `acq_offer_limits`, and `max_acq_price_actions` as
  attributes of the `GameState` object, outside the `int16` row. Search
  inherits them from the root state, which is how live play and
  `tournament._search_state` give each search its own rules. The pipeline
  treats them as opaque: `StatePool` keeps the root state instead of named
  flags, and `scratch_state(root, row)` returns a pool-row wrapper that carries
  them. Flags move into the row only if the model must see them, or with the
  per-player bot/human bits (Section 9).

### 3.3 Policy layout

`nn/policy_layout.py` becomes a `PolicyLayout` value built from a game's phase
sizes and names. It carries offsets, `unified_dim`, `max_action_size`,
`build_action_lut()`, `phases_with_pass_slot`, and `phase_names`. RSS values
equal today's constants, and TensorBoard metric names keep their RSS
spellings.

### 3.4 Configuration

- `TrainingConfig.game: str = "rss"`. A config or checkpoint without the field
  means `"rss"`, permanently.
- RSS engine options (`v3_behavior`, `acq_same_president`, and the ACQ cap)
  stay as top-level fields under an "RSS engine options" comment block.
  `games/rss.py` reads them, and validation rejects non-default values when
  `game != "rss"`. 1889 options get their own block.

## 4. Game-specific behavior: where it belongs

The rule: legality and the searched action set belong to the engine and reach
the pipeline through `enumerate_policy_actions`. The pipeline holds only search
and training algorithms. Prior overrides belong to tooling, and diagnostics are
game hooks.

### 4.1 ACQ price cap → engine

- The cap is already engine-side. `enumerate_policy_actions`
  (`core/actions.pyx:1133`) drops the middle `ACQ_SELECT_PRICE` offsets, and
  search expansion and self-play call it.
- It leaks into the pipeline because the evaluators' single-state paths
  (`NNEvaluator.evaluate`, `evaluate_batch`, `RemoteEvaluator.evaluate`) call
  the full engine enumerator. Root priors therefore arrive over the uncapped
  set, and `_filter_acq_price_root_priors` trims and renormalizes them
  (`search.py:503`, `self_play.py:1070`, `analyze_game.py:740`).
- Fix (decided 2026-11-01; lands before step 2): make the cap a runtime flag
  on the RSS state, next to `acq_same_president`, and have the evaluators call
  the policy enumerator. Then the filter, `MCTSConfig`'s cap field, and the
  cap argument all go away. Tournament seats with different caps set them the
  way they already set their search rules. Middle prices stay engine-legal, so
  the driver still accepts human ones in replay and live play.
  - Rejected: binding the cap into the RSS `Game` with `functools.partial`.
    Tournament seats and analysis tools would each need a `Game` built from
    the same cap as their `MCTSConfig`, and one process could no longer share
    one `Game`.
- Equivalence: both models use the legal mask only to mask logits
  (`nn/transformer-v3.py:1735`, `nn/transformer-v2.py:1567`). Softmax over
  the capped mask therefore equals the renormalized capped slice of the full
  softmax, up to float rounding. With cap 0 (v2 configs) nothing changes. With
  cap 8 (v3 config), priors can differ in the last bits, which can steer a
  search differently. Land this change as its own commit and check it with a
  tolerance, not the exact fingerprint (Section 7).

### 4.2 Cross-president offer priors → RSS tooling

- This is not a legality question. Accept and reject must stay legal, because
  18xx.games players make offers that self-play never produces.
  `CrossPresidentOfferPriorEvaluator` replaces the model's untrained priors at
  those states with 50/50, so search decides from values alone.
- Search and self-play never reference it. Only `train/tournament.py` and
  `utils_18xx/live.py` do. Move the adapter and its predicates out of
  `mcts/evaluator.py` into an RSS module (e.g. `games/rss_live.py`); nothing
  else changes.

### 4.3 Self-play stats → hook

- `AcquisitionStats`, the end-of-game share and company stats, and the
  strategy trace are RSS diagnostics inside `play_game`.
- `game.new_selfplay_stats(num_players)` returns a collector with `observe`
  (once per played decision) and `finish(state) -> dict[str, float]`.
  `train/main.py` logs whatever comes back. 1889 starts with an empty
  collector.
- The strategy trace stays RSS-only: `collect_strategy_trace` raises for other
  games.

### 4.4 The same pattern in 1889

- Self-play restrictions are engine options that change legality, like
  `acq_same_president`. Private sales across presidents are the first
  candidate (blueprint 4.4 **Open:**).
- Routes: self-play uses engine max-revenue routes, but replay must accept
  human routes (blueprint 4.3). The driver can accept an explicit route in
  replay and compute one otherwise. The pipeline never sees a route decision.

## 5. Model contract

- What the pipeline needs from a model is already duck-typed:
  - `forward(tokens, legal_mask, relations)`, with relations as dense planes
    or sparse records, returning logits over the unified slots and values
    padded to the player capacity;
  - `cfg.layout_version`;
  - `value_head`;
  - `input_output_layers()` for the Muon split;
  - optional diagnostics such as `pass_action_logit_abs`.
- Loader (`nn/__init__.py`):
  - Stop importing `core.token_data` and loading v2 at import.
  - If a model module defines `build_model(config, spec)`, use it; otherwise
    fall back to today's `RSSTransformerNet` and `TransformerConfig` path.
  - The v2 and v3 model files stay untouched.
- `ModelInputSpec` gains `game`. Its `num_tokens`, `token_dim`, and
  `policy_dim` come from the `Game`. `layout_version` stays a per-game id for
  the token layout.
- `nn/transformer-1889.py` copies the v3 trunk pieces it needs (relation bias,
  `RelationInputMixing`, blocks) rather than extracting shared modules from v3.
  Extraction risks changing the state-dict keys or numerics of the live v3
  checkpoint. Deduplicate later if it pays off.
- Inputs stay three tensors: tokens, mask, relations. For v0, tile-candidate
  outcome features (blueprint 9) go in extra token rows. A fourth input would
  touch eval-server shared memory, replay, the trainer, and every `forward`,
  so it waits until token rows prove awkward.
- `max_relation_edges` comes from the `Game`; 1889 needs more than RSS's 256
  (blueprint 6.4). Sparse records use `uint8` token indices, which works below
  256 tokens.

## 6. 1889 engine

### 6.1 What the adapter needs from it

- A `g1889/` package (blueprint Section 3), added to `EXTENSION_DIRS` in
  `setup.py`.
- A raw `int16` `GameState` padded to a player capacity, with `from_buffer`
  and `rebind` over pool rows.
- A driver that validates against engine legality, dispatches, and
  auto-chains automated steps and single-legal-action decisions, returning
  the shared status codes.
- Decision phases with phase-local sparse action ids, the engine-legal and
  policy enumerators, and phase sizes for the policy layout.
- Token and relation extraction: single states and batches of raw rows, as
  dense planes and sparse records.
- Final scores.
- Copy the RSS patterns: layout structs, entity APIs with dirty caches,
  legality in an actions module, and phase handlers that assume legal actions.
  Don't share Cython code. A common `GameState` base would tie the two builds
  together, and the layouts have nothing in common.
- Order: rules and driver first, tested without the pipeline. Feature
  extraction comes later (blueprint milestone 4). The adapter can be wired in
  once a full random game runs.
- Static data (step 5) is in `g1889/data/` (`tables.json`, conventions in
  its `__init__.py`). Findings the rules step must handle:
  - Port tile 437 matches every white town in the upgrade table, including
    the non-port J5. Enforce `blocks_lay` and Mitsubishi Ferry's hex list.
  - Symmetric tiles have rotations with identical masks, and Ruby accepts all
    of them. Dedupe lay candidates by mask.
  - Ruby's `legal_tile_rotation?` also requires a new exit on a side the
    corporation reaches. That needs game state, so the table leaves it out.
  - 18xx actions name a tile copy (`57-1`), so the action mapping picks a
    copy index.

### 6.2 Engine tests (`tests/g1889/`)

1. Static data: regenerating from the submodule leaves the checked-in tables
   unchanged, and the upgrade table matches the Ruby engine (blueprint 4.1).
2. Unit tests per round and step from constructed states, in the style of
   `tests/phases/`.
3. Random playouts, as in `tests/test_random_game.py`: many seeds at 3-5
   players, with invariants checked after every step:
   - cash is conserved, including the bank;
   - each corp's shares total 100%;
   - every private has exactly one owner;
   - token and tile supplies never go negative;
   - train limits hold after discards;
   - every non-terminal state has a legal action;
   - every game ends within a step cap.
4. Differential replay against the 18xx.games fixtures (`314.json`,
   `962.json`, and the two `hs_*` files), comparing state after every action
   (blueprint 4.5). This is the main correctness net.
   - It needs a Ruby state extractor like `utils_18xx/extract_states.rb`.
   - It also needs a mapping from 18xx action JSON to engine actions, kept
     minimal for tests. Live play later builds on both.
5. Route solver exactness against brute force (blueprint 4.3).

### 6.3 Adapter conformance test (both games)

One test, parametrized over the registered games, that talks only to the
adapter:

- random playouts to a terminal state, where every enumerated policy action
  applies without `INVALID`;
- `scratch_state` over a copied row reproduces the same legal set and tokens;
- `num_players_from_row` and `storage_player_capacity` agree with the state;
- dense and sparse relations encode the same edges;
- `final_scores` returns one score per player.

It runs for RSS from step 1 and catches 1889 contract slips as soon as the
adapter exists.

### 6.4 Boundary test

An AST import check. The generic pipeline modules (`mcts/`, the training core
in `train/`, `nn/__init__.py`, `nn/policy_layout.py`, `nn/model_contract.py`)
never import `core`, `entities`, `phases`, or `g1889`; only `games/` does. RSS
tooling and model files are exempt. Together with RSS conformance, this covers
what a toy test game would. 1889 then becomes the second real implementation.

## 7. Guarding RSS behavior

- `tests/test_v2_parity.py` passes with its existing fixture at every step.
- Before step 1, capture a self-play fingerprint:
  - a few seeded games on CPU in fp32, with tiny seeded v2 and v3 models, at
    3 and 5 players, with ACQ cap 0 and 8;
  - record states, masks, targets, and values, plus the live v3 checkpoint's
    outputs on a sample of states.

  Rerun after each step on the same machine; mechanical steps must match
  exactly. Chained search amplifies float differences across hosts, so this
  is a scratchpad before/after check, not a committed fixture. The script and
  per-commit baselines are in `scratchpad/fingerprint/` on the dev host.
- The existing tests:
  - pipeline: `tests/test_mcts.py`, `tests/test_self_play*.py`,
    `tests/test_eval_server_batching.py`, `tests/test_replay_buffer_mixed.py`,
    and the trainer tests;
  - RSS tooling: `tests/test_18xx_*` and `tests/test_tournament_config.py`.

### 7.1 Divergences from `main`

Every intended behavior or API difference from `main`, with the commit that
introduced it. Add to this list whenever a step changes RSS behavior.

- **Root priors at capped ACQ price decisions** (6713d2b). Fresh searches
  (tournament, `analyze_game.py`, the self-play trace) take root priors from a
  softmax over the capped mask instead of a renormalized slice of the full
  softmax. They differ by float rounding (at most 2e-8 in the fingerprint),
  which can flip a near-tied search. Self-play games are unchanged: their
  roots are reused, and expansion already used the capped set.
- **Evaluator and search API** (6713d2b). `evaluate` and `evaluate_batch`,
  in-process and remote, return the capped action set when the state carries
  a cap. `MCTSConfig` no longer takes `max_acq_price_actions`; set it on the
  root state.
- **Tournament** (6713d2b). A seat whose cap differs from the game state's
  searches a copy of the state. Speed only.

Known `main` bugs, kept for parity until fixed on `main`:

- **Live play ignores the ACQ price cap.** `_mcts_config_for`
  (`utils_18xx/live.py`) rebuilds `MCTSConfig` from eight fields of
  `to_mcts_config()` and drops `max_acq_price_actions`, so the live v3 bot
  searches every legal price, including the middle prices its checkpoint was
  never trained on. The branch keeps this: `prepare_live_decision_state` sets
  the state's cap to 0, and a test pins it. The fix is to set it from the
  checkpoint config there, preferably on `main` first so the branch stays
  aligned.

## 8. Steps

Pipeline track. Each step is a behavior-preserving commit or a short series:

0. Capture the fingerprint (Section 7).
1. Add `games/`, `Game`, `PolicyLayout`, and the RSS adapter wrapping the
   existing functions, plus the conformance test. No consumer changes.
2. Move consumers onto the adapter: search, evaluators, eval server, replay,
   trainer, `play_game`, and `train/main.py`. `StatePool` keeps the root state
   instead of three named flags, since search builds its scratch state through
   `scratch_state`. Update RSS tooling call sites mechanically, and add the
   boundary test. (`config.game` landed in step 1.)
3. Move game-specific behavior out of the pipeline:
   - the self-play stats hook;
   - the cross-president adapter, into an RSS module.

   The ACQ cap moved onto the state as its own commit between steps 1 and 2
   (Section 4.1).
4. Model loader: `build_model`, `ModelInputSpec.game`, and the checkpoint
   default.

Engine track, independent of the pipeline track:

5. Static data generator and its tests (blueprint milestone 1).
6. Rules and driver: auction, stock round, OR (track, tokens, routes,
   dividends, trains, privates), and game end, with random-playout and
   differential tests (milestones 2-3).
7. Feature extraction (milestone 4), `games/g1889.py`, and conformance for
   1889.

Then both tracks join:

8. `nn/transformer-1889.py` v0 without build-distance planes and a first
   training run, then the build-distance comparison (milestone 6).

Steps 0-1 are small, and they fix the contract the engine targets. Steps 2-4
block 1889 only at step 7, so the two tracks can alternate.

## 9. Open questions

- **Open:** Names and contents of the RSS-only adapter block.
- **Open:** Human moves the bot's rules forbid, in live play. Two options:
  - a per-player bot/human flag in the `int16` row, so restrictions apply only
    to bot seats. It travels with pool rows, and search would model humans
    making restricted moves, with untrained priors;
  - the driver accepts a move outside the bot's restrictions when the call
    marks it as human (`human_player=True`).

  Either fits the engine-legal/policy split (Section 3.2) for restrictions
  that only filter actions, such as the ACQ cap and same-president offers.
  Variants that change consequences, such as v3's price floors and
  trade-history lifetime, still need state flags. Neither replaces the prior
  override (Section 4.2).
- **Open:** Replay metadata records the game. Resume should skip a buffer from
  another game, as it already skips one with mismatched shapes.
- **Test:** Replay rebuilds tokens and relations every time a row is sampled.
  Once 1889 extraction includes build distance and candidate features, check
  that the trainer keeps up (step 7).
