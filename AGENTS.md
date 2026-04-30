# AGENTS.md

Concise repo context for future Codex/Hermes sessions. Keep this practical:
facts that prevent wrong assumptions, speed up navigation, or improve the dev
loop.

## Project Snapshot

- High-performance Cython engine plus AlphaZero-style search/training stack for
  **Rolling Stock Stars**.
- **Rolling Stock Stars is not Rolling Stock and not an 18xx game.** Read
  `RULES.md` before changing rules, phases, entity semantics, or model features.
- Engine state supports 2-6 players. Token extraction, relations, NN, MCTS, and
  training are scoped to 3-5 players. Only `train_configs/3p.json` exists as a
  production training config.
- `GameState` wraps one compact contiguous `int16` numpy array. Values are raw
  integers; normalization happens only while extracting model inputs.
- The NN input bridge is `core/token_data.pyx` plus `core/relations.pyx`.
  The model never reads `GameState` directly.
- The live pipeline is sparse phase-local legal actions -> dense unified legal
  masks/targets at NN/eval/training boundaries, token buffers, relation planes,
  and canonical-order per-player values.

## Source Of Truth

When sources disagree, prefer this order:

1. `RULES.md` for game rules.
2. Current code in `core/`, `entities/`, `phases/`, `nn/`, `mcts/`, `train/`.
3. Runtime checks: clean build, imports, targeted pytest, smoke scripts.
4. `VECTORS.md` for state/action layout and `token-data.md` for token features,
   verified against code when touching internals.
5. `CLAUDE.md` for broader repo guidance, after checking task-critical claims
   against code.

Some inline prose comments lag implementation. Prefer imported constants,
layout helpers, and tests over comments for dimensions, action sizes, and token
widths.

## Architecture Landmarks

- `core/state.{pyx,pxd}`: compact state layout. Python callers use
  `core.state.get_layout(num_players)` and field accessors; Cython hot paths
  cimport layout structs only where appropriate.
- `entities/*`: stateless singleton handles over the state array. They own
  coherence, dirty-cache invalidation, ownership/location sync, and semantic
  mutations.
- `core/data.{pxd,pyx}`: static data, enums, normalization constants,
  `GamePhases`, `DecisionPhase`, `ActionSize`, `PHASE_ACTION_SIZES`,
  `MAX_ACTION_SIZE`.
- `core/actions.{pxd,pyx}`: phase-local action encode/decode and sparse legal
  enumeration into caller-provided `uint16` buffers sized to `MAX_ACTION_SIZE`.
- `core/driver.pyx`: legality check, phase dispatch, automated phase chaining,
  forced-action chaining, history recording, and step-mode helpers.
- `core/token_data.{pxd,pyx}`: token extraction. Engine tokens are
  `num_players + 54` rows, `TokenDataSize.TOKEN_DIM == 95`; per-row meaningful
  widths come from `TokenWidth` / `get_token_widths(num_players)`.
- `core/relations.{pxd,pyx}` and `core/attention_relations.py`: directed
  Graphormer-style relation planes. Current constants are
  `NUM_ATTENTION_RELATIONS == 10`, sparse IPC coord shape
  `(MAX_ATTENTION_RELATION_EDGES == 256, ATTENTION_RELATION_COORD_WIDTH == 3)`.
- `nn/transformer.py`: `RSSTransformerNet`. Forward contract is
  `model(tokens, legal_mask, relations) -> (policy_logits, values)`, where
  policy logits are dense over `UNIFIED_LOGIT_DIM == 255` and values are
  canonical player order. Use `build_action_lut()` for phase-local action id
  to unified slot mapping.
- `mcts/search.py`, `mcts/node.py`, `mcts/evaluator.py`, `mcts/mcts_core.pyx`:
  sparse/token/relation MCTS stack with batched leaf eval, leaf locking, lock
  propagation, a compact `StatePool`, and subtree reuse.
- `train/eval_server.py`, `train/self_play.py`, `train/replay_buffer.py`,
  `train/trainer.py`, `train/main.py`: live training pipeline. Replay stores
  compact int16 states plus dense unified masks/targets; trainer materializes
  tokens and dense relation planes per sampled batch.
- `train/analyze_game.py`: checkpoint/game analysis entry point; re-check it if
  evaluator, search, model inputs, or model return shapes change.
- `utils_18xx/` and `tests/games_18xx/`: 18xx.games compatibility harness and
  fixtures. Useful for replay tests, not as a source of RSS rules.

## Read Order By Task

- Rules or phase behavior: `RULES.md`, relevant `phases/*.pyx`,
  `core/actions.pyx`, `core/driver.pyx`, relevant `tests/phases/*`.
- State/layout/invariants: `core/state.{pyx,pxd}`, relevant `entities/*.pyx`,
  `VECTORS.md`, and token/relation extraction if NN-visible state changes.
- Entity behavior: relevant `entities/<entity>.{pyx,pxd}` first, then callers.
- Action-space changes: `core/data.pxd`, `core/actions.{pxd,pyx}`, relevant
  phase module, `nn/transformer.py`, MCTS/eval/training mask consumers.
- Token/relation/model changes: `core/token_data.{pyx,pxd}`,
  `core/relations.{pyx,pxd}`, `token-data.md`, `nn/transformer.py`,
  `tests/test_transformer.py`, `tests/test_relations.py`, token invariant tests.
- Search/eval changes: `mcts/*`, `train/eval_server.py`,
  `tests/test_mcts.py`, `tests/test_eval_server_batching.py`,
  `tests/test_eval_bitmap.py`.
- Self-play/training changes: `train/self_play.py`, `train/replay_buffer.py`,
  `train/trainer.py`, `train/main.py`, `train/config.py`,
  `tests/test_self_play.py`, `tests/test_trainer_*`,
  `tests/test_training_step_scaling.py`.
- 18xx replay compatibility: `utils_18xx/`, `tests/games_18xx/`,
  `tests/games_18xx/data/`.

## Build And Test

- Use `.venv/bin/python`, not bare `python`. Run commands from the repo root.
- For ad hoc imports from outside the repo root, set
  `PYTHONPATH=/home/icebreaker/rss-az-cython2_worktree2`.
- `setup.py` is the build entry point; there is no `pyproject.toml`.
- Build before importing compiled modules:

```bash
.venv/bin/python setup.py build_ext --inplace
```

- Clean when `.pxd` files, Cython signatures, layout, or low-level dependencies
  changed:

```bash
.venv/bin/python setup.py clean_build
```

- Useful targeted checks:

```bash
.venv/bin/pytest tests/phases -q
.venv/bin/pytest tests/test_actions_roundtrip.py tests/test_actions_width.py -q
.venv/bin/pytest tests/test_driver_step_mode.py tests/test_random_game.py -q
.venv/bin/pytest tests/phases/test_token_data_invariants_meta.py tests/test_transformer.py tests/test_relations.py -q
.venv/bin/pytest tests/test_mcts.py tests/test_self_play.py -q
.venv/bin/pytest tests/test_eval_server_batching.py tests/test_eval_bitmap.py tests/test_eval_batch_config.py -q
.venv/bin/pytest tests/test_trainer_loss.py tests/test_trainer_optimizer.py tests/test_trainer_scheduler.py tests/test_training_step_scaling.py -q
```

- `pyright` is the system binary, not `.venv/bin/pyright`.
- Training entry point:

```bash
.venv/bin/python -m train --config train_configs/3p.json
```

## Working Rules

- Do not hardcode layout numbers, action counts, token widths, relation counts,
  or policy sizes. Query helpers/constants such as `get_layout`,
  `GameConstants`, `ActionSize`, `MAX_ACTION_SIZE`, `PHASE_ACTION_SIZES`,
  `TokenDataSize`, `TokenWidth`, `get_num_tokens`, `NUM_ATTENTION_RELATIONS`,
  and `build_action_lut`.
- Do not bypass entity handles with raw state-array writes except in tightly
  scoped tests/setup or inside the owning low-level module.
- Do not cimport state layout structs into phase code just to reach fields.
  Prefer existing entity `cdef` primitives or handle methods.
- Semantic mutations belong on entity/phase APIs that already manage cache and
  invalidation. Inspect existing entity surfaces before adding new methods.
- Phase handlers assume legal actions. Legality lives in `core/actions.pyx` and
  the driver checks it before dispatch.
- Use sparse legal lists as the default engine/search contract. Convert to
  dense unified masks only at NN/eval/trainer boundaries via the transformer
  action LUT.
- Model/eval/training values are canonical player order. There is no state
  rotation.
- Model forward calls require relation planes. Eval-server IPC ships sparse
  relation coords and materializes dense planes on-device; trainer/replay use
  `get_relation_data_batch`.
- `GameState.step_mode`, `acq_same_president`, and
  `allow_positive_income_closing` are Python-level driver/config flags, not
  state-array fields. Replay and tests may set them deliberately.
- Use `scratchpad/` for temporary investigation scripts; it is gitignored.
- The repo uses `bd` / beads for issue tracking. `bd prime` is the workflow
  reference when issue state matters.
