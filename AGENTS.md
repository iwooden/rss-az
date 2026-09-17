# AGENTS.md

## Direction

- This branch develops and trains a successor to `nn/transformer-v2.py`.
  A v2 checkpoint performed well against top-level human players over summer
  2026; use it as the baseline for incorporating lessons from those games.
- Backward compatibility is not a requirement here: old model APIs,
  checkpoints, and configs may break. ResNet and the original transformer are
  not maintenance targets. Update active consumers together when contracts
  change; shared code may still live in legacy modules.
- Current engine support is 2-6 players; model/search/training support is 3-5,
  including mixed-player transformer training.

## Game and implementation authority

- **Rolling Stock Stars is not Rolling Stock or an 18xx game.** Read
  `RULES.md` when changing game behavior or the meaning of model features.
  The 18xx.games integration is a replay/live-play interface, not a rules source.
- `RULES.md` defines intended game behavior. Code and targeted tests establish
  current implementation behavior; investigate discrepancies.
- `README.md` provides navigation and launch commands. Consult `VECTORS.md`
  and `token-data.md` as needed, checking their claims against current code.
  Read files relevant to the task; these references are not a mandatory tour.

## Boundaries worth preserving

These describe the current pipeline. Intentional redesigns should update the
affected producers, consumers, tests, and documentation together.

- `GameState` stores raw integers in a contiguous `int16` array; normalization
  belongs in model input extraction.
- Use entity/phase APIs for semantic mutations: they maintain dirty caches,
  ownership, and location consistency. Reuse or extend existing entity APIs.
  Raw writes belong in owning low-level modules or tightly scoped test setup;
  avoid importing layout structs into phases just to bypass handles.
- Query layout helpers and exported constants for dimensions and action sizes
  (e.g. `get_layout`, `TokenDataSize`, `MAX_ACTION_SIZE`, `build_action_lut`).
- Legality belongs in `core/actions.pyx`; the driver checks it before dispatch.
  Phase handlers assume legal actions. Engine/MCTS use sparse phase-local
  actions; NN/eval/trainer boundaries use dense unified masks and targets.
- Transformer inputs come from `core/token_data.pyx` and `core/relations.pyx`.
  Forward calls require relation planes; eval-server IPC transports sparse
  relation coordinates and materializes dense planes on-device.
- Transformer values and replay/self-play/evaluator values use canonical
  player order. Preserve player identity across the pipeline.
- When changing model/input/output contracts, follow the active path through
  `nn/__init__.py`, `nn/model_contract.py`, `mcts/evaluator.py`,
  `train/eval_server.py`, replay/trainer code, and `train/analyze_game.py`.

## Development loop

- Run from the repo root with `.venv/bin/python` and `.venv/bin/pytest`.
  `pyright` is a system command, not a virtualenv binary.
- Build missing or stale Cython extensions before imports/tests:
  `.venv/bin/python setup.py build_ext --inplace`.
- Changes to `.pxd`, Cython signatures, or layouts require a clean rebuild.
  `setup.py clean_build` performs one, but its current cleaner recursively
  deletes `.c`, `.cpp`, and `.so` files beyond the engine directories;
  inspect its scope before using it in a checkout with vendored code.
- Run focused tests for affected behavior using
  `.venv/bin/pytest <relevant paths> -q`; include downstream integration checks
  when changing pipeline contracts. Documentation-only changes need no build.
- Judge tests against intended behavior; legacy compatibility expectations may
  need updating. Report unrelated failures without expanding the task to fix
  them. Prefer invariant and end-to-end checks over tests that mirror code.
- Put temporary investigation scripts in gitignored `scratchpad/`. When needed,
  set `PYTHONPATH` to the current checkout root, not a fixed absolute path.
