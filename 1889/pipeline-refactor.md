# Pipeline Refactor for Shikoku 1889

Status: steps 0, 1, and 5 and the ACQ cap commit done (2026-11-01);
step 6, the 1889 engine with the engine half of `games/g1889.py`
(2026-11-02, [`engine-design.md`](engine-design.md) §11); step 2, the
consumers on the adapter (2026-11-02); step 3, the game-specific
behavior out of the pipeline (2026-11-02); and step 4, the game-aware model
loader (2026-11-02). The rest is plan.
This expands Section 11
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

## 2. Where the pipeline touches RSS

Step 2 moved search, the evaluators, the eval server, replay, the trainer,
`play_game`, and `train/main.py` onto the adapter: their engine calls, the
import-time constants (`UNIFIED_LOGIT_DIM`, `MAX_ACTION_SIZE`,
`NUM_ATTENTION_RELATIONS`, `TOKEN_DIM`, `build_action_lut`, `get_layout`,
`get_num_tokens`, `PHASE_NAMES`, and module globals such as `U_DIM`), and the
RSS runtime flags `StatePool` used to copy. Step 3 moved the RSS-only
features out: the cross-president prior adapter to `games/rss_live.py`
(Section 4.2), and the self-play stats and strategy trace behind game hooks in
`games/rss_selfplay.py` (Section 4.3). Step 4 made the model loader
game-aware (Section 5), and the boundary test's allowlist is empty
(Section 6.4). What remains is by design:

| Module | RSS coupling |
|---|---|
| `mcts/`, `train/eval_server.py`, `train/replay_buffer.py`, `train/trainer.py`, `train/policy_metrics.py`, `train/self_play.py`, `train/main.py`, `nn/model_contract.py` | None. |
| `train/config.py` | RSS engine flags and the ACQ cap. The cap's bound comes from `games/rss.py`, imported only to check a non-default cap. |
| `nn/__init__.py` | The RSS transformer fallback and RSS's default model, `transformer-v2.py`, loaded on first use and only for RSS configs. |
| `nn/policy_layout.py` | RSS's layout under its legacy module names (`NUM_PHASES`, `PHASE_OFFSETS`, `UNIFIED_LOGIT_DIM`, `PHASES_WITH_PASS_SLOT`, `build_action_lut`) for the v2 and v3 model files, resolved from `games.rss` on first use. |

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
    new_strategy_trace: Callable | None = None  # num_players -> trace builder
```

As built (step 3): the self-play hook is `new_selfplay_stats`, whose
collectors follow the `games.SelfPlayStats` protocol; a game with nothing to
report uses `games.EmptySelfPlayStats` (1889). The RSS-only block holds one
field, `new_strategy_trace`, which defaults to `None`, so other games leave it
out (Section 4.3).

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
- As built (step 6): the "1889 engine options" block holds
  `private_sale_same_president` and `train_sale_same_president` (True) and
  `auction_pass_payout_cap` (1), the self-play values of engine-design §7,
  under their `GameState` names; `games/g1889.py` reads them, and
  validation rejects non-default values when `game != "1889"`.

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
- As built (step 3): `games/rss_live.py` holds
  `CrossPresidentOfferPriorEvaluator`, `is_cross_president_acq_offer_state`,
  `is_acq_offer_outside_v3_limits`, and their helpers, unchanged except that
  the dense LUT comes from `games.rss.POLICY_LAYOUT` (equal to the legacy
  `build_action_lut()`). It sits beside the RSS adapter rather than in
  `utils_18xx/`: it wraps an evaluator over RSS engine state with no
  18xx.games dependency, and its other user, `train/tournament.py`, has
  nothing to do with 18xx.games. `games/` also holds the rest of the RSS code
  the pipeline used to (Section 4.3). Tournament, live play, and their tests
  import it from there.

### 4.3 Self-play stats → hook

- `AcquisitionStats`, the end-of-game share and company stats, and the
  strategy trace are RSS diagnostics inside `play_game`.
- `game.new_selfplay_stats(num_players)` returns a collector with `observe`
  (once per played decision) and `finish(state) -> dict[str, float]`.
  `train/main.py` logs whatever comes back. 1889 starts with an empty
  collector.
- The strategy trace stays RSS-only: `collect_strategy_trace` raises for other
  games.
- As built (step 3):
  - `finish` returns `{name: (numerator, denominator)}`, not plain floats.
    `train/main.py` sums both over each reporting group's games (all games
    and each player count) and logs `self_play_<group>/<name>` as
    numerator / max(denominator, 1). Plain floats can't reproduce RSS's
    scalars: the acquisition rates are ratios of sums over a group's
    decisions and offers, not means of per-game ratios, and the per-rank
    stats average over the games that have that rank. With pairs, a mean is
    a denominator of 1 per game, and RSS's scalar names and values are
    unchanged.
  - `GameRecord.game_stats` carries the report, replacing `acquisition` and
    the ten RSS stat fields. `net_worths` stays as the generic final
    scores (`game.final_scores`), which `main.py` ranks for its net worth
    scalars and the console panel. `play_game` binds `observe` once and calls
    it per played decision, before applying it, as it called
    `AcquisitionStats.observe`.
  - `games/rss_selfplay.py` holds RSS's collector (`RSSSelfPlayStats`:
    `AcquisitionStats` plus the end-of-game share, company, and corp stats,
    reported per game or by finishing rank), `StrategyTrace`, and its builder.
    1889 uses `games.EmptySelfPlayStats`, whose `finish` returns `{}`.
  - The trace builder is the RSS-only `Game.new_strategy_trace`, `None` for
    1889; `play_game` raises `ValueError` when asked to trace a game without
    one. The builder now decodes the chosen action and keeps the decision's
    pre-search summary itself, and `play_game` passes it the greedy leaf
    depth, so `games/` imports nothing from `mcts/`.
  - `train/strategy_data.py` reads its `final_shares_per_player` and
    `final_companies_per_player` from each game's final state
    (`games.rss_selfplay.player_holdings`), since the record no longer
    carries them.

### 4.4 The same pattern in 1889

- Self-play restrictions are policy-only runtime flags, like the ACQ cap
  (Section 4.1), not legality options like `acq_same_president`
  ([`engine-design.md`](engine-design.md) §7). `private_sale_same_president`,
  `train_sale_same_president`, and `auction_pass_payout_cap` filter only the
  policy set that `enumerate_policy_actions` returns; the driver still
  accepts every move, so replay and live play apply 18xx.games moves the
  policy would never make.
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

As built (step 4):

- Loading. Importing `nn` loads no engine and no model file. A config's
  module is its `model_path` (a module name or a file), or `transformer-v2.py`
  for an RSS config without one, loaded on first use. `nn.RSSTransformerNet`,
  `TransformerBlock`, and `TransformerConfig` are v2's, resolved on first
  access through a module `__getattr__` for tests and RSS tooling.
- Resolution. `create_model(config)` calls the module's
  `build_model(config, spec)` if it defines one, with the `TrainingConfig` and
  `get_model_input_spec(config)`. Otherwise it builds the RSS fallback,
  `RSSTransformerNet(TransformerConfig(...))`, as before. The fallback, and
  so the default, is RSS-only: for another game, `create_model` and
  `get_model_input_spec` raise `ValueError` unless `model_path` names a
  module that defines `build_model`. Other games have no default model. The
  keyword form, `create_model(num_players=...)`, builds only the fallback and
  raises `TypeError` for a module that defines `build_model`.
- `layout_version` is still the module's `INPUT_LAYOUT_VERSION`, default 2
  (RSS's v2), so a non-RSS model module sets its own.
- `ModelInputSpec.game` is the registry name, last among the fields with
  default `"rss"`, so a spec built without it is RSS's.
  `get_model_input_spec`, `NNEvaluator`'s inferred spec, and
  `SharedEvalBuffers`' default spec take it, and their sizes, from the
  `Game`. `NNEvaluator` and `SharedEvalBuffers` reject a spec for another
  game, so tournament seats from two games fail there.
- Checkpoints store the game in `config_json`; a config without it is RSS's
  (Section 3.4). The v2 parity fixture and the live v3 checkpoint have none.
  Resuming a run (`train.main._resolve_config`) raises `ValueError` when the
  `--config` JSON or a CLI override changes the checkpoint's game. The other
  loaders (`load_model_from_checkpoint` for analysis, tournaments, and live
  play, and strategy data) take their config from the checkpoint, so their
  game always matches.
- Replay `metadata.json` records `game`. Resume skips another game's buffer
  with a warning, as it skips one with mismatched shapes; metadata without
  the field is RSS's.
- `nn/policy_layout.py` resolves its legacy names from
  `games.rss.POLICY_LAYOUT` on first access and caches them, so it imports no
  engine. `build_action_lut` is the layout's bound method. With `nn` no
  longer loading v2 at import, `games.rss` → `nn.policy_layout` → `nn` ends
  there, and the step-2 cycle is gone in either import order.
- `train/config.py` imports `games.rss` only to check a non-default ACQ cap,
  so validating a 1889 config loads no RSS engine.

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
adapter exists. Since step 6 (engine-design 6.9) it runs every engine check
for 1889, with the self-play flags and with them off, and skips the
extraction checks, whose 1889 callables raise until step 7, with that
reason.

### 6.4 Boundary test

An AST import check. The generic pipeline modules (`mcts/`, the training core
in `train/`, `nn/__init__.py`, `nn/policy_layout.py`, `nn/model_contract.py`)
never import `core`, `entities`, `phases`, or `g1889`; only `games/` does. RSS
tooling and model files are exempt. Together with RSS conformance, this covers
what a toy test game would. 1889 then becomes the second real implementation.

As built (step 2): `tests/test_pipeline_boundary.py`. The training core is
every `train/` module except the four RSS tools of Section 1, and every
import counts, including function-level and type-checking ones. Its allowlist
names each remaining import by module and imported names, with the step that
removes it. Since step 4 it is empty. The test also fails on an allowlisted
import that is gone, so entries leave with their imports. A second test
imports every generic pipeline module in a fresh process, validates an RSS and
a 1889 config, and checks that no engine module and no model file loaded.

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

- **Consumer APIs take a `Game`** (step 2), as their first argument:
  `run_search(game, root_state, ...)`, `StatePool(game, capacity,
  state_size)`, `NNEvaluator(game, model, device, ...)`,
  `RemoteEvaluator(game, num_players, shared_bufs, worker_idx, ...)`,
  `BaseEvaluator(game, num_players, ...)`, `SharedEvalBuffers(game,
  num_workers, ...)`, and `ReplayBuffer(game, capacity, ...)`.
  `play_game`, `Trainer`, and `get_model_input_spec` build theirs from the
  config. `SharedEvalBuffers` keeps the game's `policy`, which pickles, not the
  `Game`. Tournament's `_play_game` and `_build_players` take the game too.
- **Search inherits runtime flags from the root state object** (step 2).
  `StatePool` keeps the last fresh search's root state instead of copies of
  its four RSS flags, and reuse searches build their scratch state from it,
  so a caller that changed the root's flags between a fresh search and its
  reuse searches would now see the change. No caller does. A reuse search on
  a pool that never ran a fresh search raises `ValueError` instead of
  searching with default flags.
- **Removed constants and wrappers** (step 2). `MCTSConfig.action_dim` and
  `TrainingConfig.action_dim` (unused; `game.policy.max_action_size`);
  `train.policy_metrics.PHASE_NAMES` (`game.policy.phase_names`); the
  `fill_token_buffer*` and `fill_relation_buffer*` wrappers in
  `mcts.evaluator` (the game's `fill_*` fields); and the `TOKEN_DIM` and
  `U_DIM` module constants of `mcts.evaluator`, `mcts.search`,
  `train.eval_server`, and `train.trainer`. `nn.policy_layout` keeps its RSS
  constants for the model files (Section 2).
- **Small signature changes** (step 2). `get_action_probabilities(root,
  temperature, max_action_size)` takes the dense width.
  `PolicyMetrics.observe` takes the phase name instead of its id.
  `average_training_metrics` finds phases from its `policy_samples_<phase>`
  keys; its output is unchanged.
- **Non-RSS configs reach their own adapter** (step 2).
  `get_model_input_spec` and the evaluators size inputs from the config's
  game, so a 1889 config raises the adapter's `NotImplementedError` (until
  step 7) instead of getting RSS's sizes. `nn.create_model` still rejects it
  first (until step 4).

- **Cross-president prior adapter moved** (step 3).
  `CrossPresidentOfferPriorEvaluator`, `is_cross_president_acq_offer_state`,
  and `is_acq_offer_outside_v3_limits` import from `games.rss_live`, not
  `mcts.evaluator`. Behavior is unchanged.
- **Self-play stats behind a game hook** (step 3). `Game` gains
  `new_selfplay_stats` and the RSS-only `new_strategy_trace`.
  - `GameRecord` drops `acquisition`, `invest_roundtrip_cap_hits`,
    `shares_per_player`, `companies_per_player`, `pres_share_values`, the
    three `nw_*_pct` lists, `avg_active_corp_price`,
    `corps_in_receivership`, and `has_max_price_corp` for
    `game_stats: {name: (numerator, denominator)}`. Per-player values are
    reported by finishing rank, as `main.py` logged them. `strategy_trace`
    is typed `Any`.
  - `AcquisitionStats`, `StrategyTrace`, and the trace builder (now
    `StrategyTraceBuilder`) move from `train.self_play` to
    `games.rss_selfplay`. `AcquisitionStats.add` and `.scalars` give way to
    `.ratios()`. The builder takes `unified_dim`, decodes the chosen action
    itself, and takes the greedy leaf depth from `play_game`.
  - `main.py`'s self-play snapshots replace their RSS keys (`acquisition`,
    `invest_roundtrip_cap_hits_per_game`, `rank_invest_roundtrip_cap_hits`,
    the `avg_*` per-rank lists, `total_shares`, `total_companies`,
    `avg_active_corp_price`, `corps_in_receivership`, and
    `pct_games_max_price_corp`) with one `game_stats` dict of finished
    scalars. TensorBoard names and values are unchanged, bit for bit.
  - `play_game(collect_strategy_trace=True)` raises `ValueError` for a game
    without a strategy trace (1889).
  - `train/strategy_data.py` reads final shares and companies from each
    record's final state; its shards are unchanged.

- **The model loader is game-aware** (step 4, Section 5).
  - `nn` no longer loads `transformer-v2.py` at import.
    `nn.RSSTransformerNet`, `TransformerBlock`, and `TransformerConfig`
    resolve on first access.
  - A model module may define `build_model(config, spec)`. `create_model`'s
    "cannot be trained, played, or analyzed yet" guard gives way to "game ...
    needs a model_path module that defines build_model(config, spec)", which
    `get_model_input_spec` raises too. The keyword form of `create_model`
    raises `TypeError` for a module that defines `build_model`.
  - `nn.policy_layout`'s legacy names resolve from `games.rss.POLICY_LAYOUT`
    with unchanged values. The module no longer re-exports `core.data`'s
    `MAX_ACTION_SIZE`, `PHASE_ACTION_SIZES`, `DecisionPhase`, and
    `GameConstants`, which nothing imported from it.
- **`ModelInputSpec.game`** (step 4), default `"rss"`. `NNEvaluator` and
  `SharedEvalBuffers` reject a spec for another game. `NNEvaluator` now
  validates a passed spec against its game before comparing it with the
  model's inferred spec, so a spec that fails both reports the specific
  field instead of "does not match the model input contract".
- **Resume keeps the checkpoint's game** (step 4). `train/main.py`'s config
  resolution moved into `_resolve_config(args, cp)`, which raises when an
  override changes the game. Its output is otherwise unchanged.
- **Replay metadata records the game** (step 4). `ReplayBuffer.save` writes
  `"game"`, and `load` skips another game's buffer with a warning. Buffers
  saved on `main` load as RSS, and `main` ignores the new key.

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
   boundary test. (`config.game` landed in step 1.) Until step 4,
   `nn.create_model` rejects a config whose game is not RSS: every consumer
   builds its model there before it plays (step 6). Done 2026-11-02; the
   remaining RSS couplings are in Section 2.
3. Move game-specific behavior out of the pipeline:
   - the self-play stats hook;
   - the cross-president adapter, into an RSS module.

   The ACQ cap moved onto the state as its own commit between steps 1 and 2
   (Section 4.1). Done 2026-11-02 (Sections 4.2-4.3): `mcts/` and the
   training core import no engine code; only step 4's `nn/policy_layout.py`
   entry remains on the boundary test's allowlist.
4. Model loader: `build_model`, `ModelInputSpec.game`, and the checkpoint
   default. Done 2026-11-02 (Section 5, As built), with replay metadata's
   game (Section 9); the boundary test's allowlist is empty.

Engine track, independent of the pipeline track:

5. Static data generator and its tests (blueprint milestone 1).
6. Rules and driver: auction, stock round, OR (track, tokens, routes,
   dividends, trains, privates), and game end, with random-playout and
   differential tests (milestones 2-3).
7. Feature extraction (milestone 4), the extraction half of
   `games/g1889.py`, and extraction conformance for 1889. The engine half and
   its conformance landed with step 6.

Then both tracks join:

8. `nn/transformer-1889.py` v0 without build-distance planes and a first
   training run, then the build-distance comparison (milestone 6).

Steps 0-1 are small, and they fix the contract the engine targets. Steps 2-4
block 1889 only at step 7, so the two tracks can alternate.

## 9. Open questions

- **Open:** Names and contents of the RSS-only adapter block. As of step 3
  it holds only `new_strategy_trace`.
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
- **Settled (step 4):** Replay metadata records the game. Resume skips
  another game's buffer with a warning, as it skips one with mismatched
  shapes, and metadata without the field is RSS's.
- **Test:** Replay rebuilds tokens and relations every time a row is sampled.
  Once 1889 extraction includes build distance and candidate features, check
  that the trainer keeps up (step 7).
