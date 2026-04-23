# AGENTS.md

Concise repo context for future Codex/Hermes sessions. Keep this file practical:
things that prevent wrong assumptions, speed up navigation, or improve the dev
loop.

## Project Snapshot

- High-performance Cython engine plus AlphaZero-style search/training stack for
  **Rolling Stock Stars**.
- **Rolling Stock Stars is not Rolling Stock.** Read `RULES.md` before changing
  rule or phase behavior.
- Engine supports 2-6 players. NN/MCTS/training are scoped to 3-5 players; only
  `train_configs/3p.json` is present currently.
- Core state is one compact contiguous `int16` array wrapped by `GameState`.
- The NN never reads `GameState` directly. The only live engine-to-NN bridge is
  `core/token_data.pyx:get_token_data(...)` / `get_token_data_batch(...)`.
- Search, eval, self-play, replay, and trainer use the sparse legal-action +
  token-buffer contract. Do not assume the old visible/hidden float32 state,
  state rotation, residual MLP, or dense per-phase mask path is current.

## Source Of Truth

When sources disagree, prefer this order:

1. `RULES.md` for game rules.
2. Current code in `core/`, `entities/`, `phases/`, `nn/`, `mcts/`, `train/`.
3. Runtime checks: clean build, imports, targeted pytest, smoke scripts.
4. `VECTORS.md` for state/action layout and `token-data.md` for token features,
   but verify against code if touching internals.
5. Obsidian cartography, if available:
   `/mnt/c/Users/Isaac/Documents/obsidian/Archivum Cogitata/Hermes/rss-az-cython2/`
   starting with `00 Index.md`, `01 Source of Truth.md`, `03 Architecture.md`,
   `04 Build, Test, and Dev Loop.md`.
6. `CLAUDE.md` as broader guidance, after checking any task-critical claim
   against code.

Known stale-prose traps:

- `core/actions.pyx` still has an old module docstring claiming legal
  enumeration is stubbed. The live sparse enumerator exists and is used by the
  driver, MCTS, evaluator, self-play, and tests.
- Old notes mentioning token width `63`, visible/hidden float32 vectors, rotated
  state, or pre-refactor MCTS/training are stale.

## Architecture Landmarks

- `core/state.pyx`: compact `int16` layout. Use
  `core.state.get_layout(num_players)` from Python for exact offsets/sizes.
- `entities/*`: stateless singleton handles over the state array. They own
  coherence, dirty-cache invalidation, ownership/location sync, and semantic
  mutations.
- `core/data.{pxd,pyx}`: static data, enums, normalization constants,
  `DecisionPhase`, `ActionSize`, `PHASE_ACTION_SIZES`, `MAX_ACTION_SIZE`.
- `core/actions.{pxd,pyx}`: phase-local action encoding/decoding and sparse
  legal enumeration into caller-provided `uint16` buffers.
- `core/driver.pyx`: stateless dispatch, legality check, auto-chaining through
  automated phases and forced actions, plus step-mode helpers.
- `core/token_data.{pxd,pyx}`: token extraction. Engine-side tokens are
  `num_players + 55` rows, `TOKEN_DIM = 93`; 4 learned pass anchors are appended
  inside the model for entity-readout pass phases, not emitted by the engine.
- `nn/transformer.py`: active model. Token-based entity readout, unified dense
  policy output over `UNIFIED_LOGIT_DIM = 255`, legal mask supplied by caller,
  canonical-order per-player values. Corp/player token identity uses learned
  `corp_id_embed` / `player_id_embed`; the leading ID one-hots remain in token
  data but are skipped by the corresponding projections. Company ownership
  fields are skipped by `company_proj` and re-injected from corp/player/FI
  owner-reference embeddings.
- `mcts/search.py`, `mcts/node.py`, `mcts/evaluator.py`, `mcts/mcts_core.pyx`:
  sparse/token search stack with batched leaf eval and subtree reuse.
- `train/eval_server.py`, `train/self_play.py`, `train/replay_buffer.py`,
  `train/trainer.py`, `train/main.py`: live training pipeline.
- `train/analyze_game.py`: checkpoint/game analysis entry point; re-check it if
  evaluator, search, or model return shapes change.

## Read Order By Task

- Rules or phase behavior: `RULES.md`, relevant `phases/*.pyx`,
  `core/driver.pyx`, relevant `tests/phases/*`.
- State/layout/invariants: `core/state.{pyx,pxd}`, relevant `entities/*.pyx`,
  `VECTORS.md`, and `core/token_data.pyx` if NN-visible state changes.
- Action-space changes: `core/data.pxd`, `core/actions.{pxd,pyx}`, relevant
  phase module, `nn/transformer.py`, MCTS/eval mask consumers.
- Token/model changes: `core/token_data.{pxd,pyx}`, `token-data.md`,
  `nn/transformer.py`, `tests/test_transformer.py`, token invariant tests.
- Search/eval changes: `mcts/*`, `train/eval_server.py`,
  `tests/test_mcts.py`, `tests/test_eval_server_batching.py`.
- Self-play/training changes: `train/self_play.py`, `train/replay_buffer.py`,
  `train/trainer.py`, `train/main.py`, `train/config.py`.
- 18xx replay compatibility: `utils_18xx/`, `tests/games_18xx/`,
  `tests/games_18xx/data/`.

## Build And Test

- Use `.venv/bin/python`, not bare `python`. Run commands from the repo root.
- For ad hoc imports, set `PYTHONPATH=/home/icebreaker/rss-az-cython2`.
- `setup.py` is the build entry point; there is no `pyproject.toml`.
- Build before importing compiled modules:

```bash
.venv/bin/python setup.py build_ext --inplace
```

- Clean when `.pxd` files, signatures, layout, or low-level Cython dependencies
  changed:

```bash
.venv/bin/python setup.py clean
.venv/bin/python setup.py build_ext --inplace
```

- Shortcut for clean rebuild:

```bash
.venv/bin/python setup.py clean_build
```

- Useful targeted checks:

```bash
.venv/bin/pytest tests/phases -q
.venv/bin/pytest tests/test_actions_roundtrip.py tests/test_actions_width.py -q
.venv/bin/pytest tests/test_driver_step_mode.py -q
.venv/bin/pytest tests/phases/test_token_data_invariants_meta.py tests/test_transformer.py -q
.venv/bin/pytest tests/test_mcts.py tests/test_self_play.py -q
.venv/bin/pytest tests/test_eval_server_batching.py tests/test_trainer_loss.py -q
```

- `pyright` is the system binary, not `.venv/bin/pyright`.

## Working Rules

- Do not hardcode layout numbers, action counts, token widths, or policy sizes;
  query helpers/constants (`get_layout`, `ActionSize`, `MAX_ACTION_SIZE`,
  `TokenDataSize`, `TokenWidth`, `PHASE_ACTION_SIZES`, `build_action_lut`).
- Do not bypass entity handles with raw state-array writes except in tightly
  scoped tests/setup or inside the owning low-level module.
- Do not cimport state layout structs into phase code just to reach fields.
  Prefer entity primitives/handles.
- Semantic mutations belong on entity/phase APIs that already manage cache and
  invalidation. Adding a new entity method is a design change; inspect existing
  surfaces first.
- Phase handlers assume legal actions. Legality lives in `core/actions.pyx` and
  the driver checks it before dispatch.
- Use sparse legal lists as the default contract. Convert to dense unified
  masks only at NN/eval boundaries via the transformer action LUT.
- For replay/training/search work, values are canonical player order; there is
  no state rotation.
- Use `scratchpad/` for temporary investigation scripts; it is gitignored.
- The repo uses `bd` / beads for issue tracking. `bd prime` is the workflow
  reference when issue state matters.
