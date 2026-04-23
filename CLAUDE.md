# CLAUDE.md

Guidance for Claude Code working in this repository.

## Project

High-performance Cython engine for "Rolling Stock Stars" — AlphaZero self-play target of thousands of games/min. Single contiguous `int16` `GameState`, nogil hot paths, entity-readout transformer NN. Per-token NN features are produced lazily by `get_token_data()`, separate from state storage.

> **"Rolling Stock Stars" is NOT "Rolling Stock"** — rules differ. `RULES.md` is authoritative.

- Engine supports 2–6 players; NN/MCTS/training scoped to 3–5p.
- Working branch: `transformer-refactor`. `main` preserves pre-refactor code as a reference for **rule intent only** — do not copy state layout, field offsets, action indices, or signatures from it.

## Authoritative docs

Don't auto-load; open when the task needs them.

| Doc | Scope |
|-----|-------|
| `RULES.md` | Game rules — before touching game logic or phase handlers |
| `VECTORS.md` | State buffer layout, action encoding, enums — before touching `core/state`, `core/actions`, entity handles |
| `token-data.md` | Per-token feature spec — paired with `core/token_data.{pyx,pxd}` |

## Directory map

```
core/        state.pyx, data.pyx, actions.pyx, driver.pyx, token_data.pyx
entities/    player, corp, company, deck, turn, market, fi — 7 stateless handles
phases/      14 phase handlers: invest, bid, acq_select_corp, acq_select_company,
             acq_select_price, acq_offer, closing, dividends, income, issue,
             ipo, par, wrap_up, end_card
nn/          transformer.py — token-based model
mcts/        search.py, node.py, evaluator.py, mcts_core.pyx
train/       main, self_play, eval_server, trainer, replay_buffer,
             analyze_game, tournament; gpu/{nvidia,amd}.py
tests/       phases/ (full phase suite), test_driver_step_mode, test_mcts,
             test_self_play; games_18xx/data/ (JSON fixtures, replay harness pending)
scratchpad/  Ad-hoc scripts (gitignored)
```

## Architecture in one breath

- **State.** `GameState` wraps one contiguous `int16` numpy array. Raw integers only — no normalization, no one-hot, no visible/hidden split. Layout offsets live as Cython `cdef` structs at module scope on `core.state` (`LAYOUT`, `PLAYER_FIELDS`, `CORP_FIELDS`, `TURN_OFFSETS`, `COMPANY_OFFSETS`, `DECK_OFFSETS`, `FI_OFFSETS`). Full layout in `VECTORS.md`.
- **Entity handles.** `PLAYERS[i]`, `CORPS[c]`, `COMPANIES[i]`, `TURN`, `FI`, `MARKET`, `DECK` are stateless singletons — one set, reused with any `GameState` at any player count. All state reads and writes go through them.
- **Actions.** Phase-local integer ids (see `VECTORS.md` Action Space). 11 decision phases (`DecisionPhase` in `core/data.pxd`); engine's 15 `GamePhases` fold to them via `ENGINE_TO_DECISION_PHASE`. Sparse legal-action enumeration via `enumerate_legal_actions(state, uint16_t* ids)`; buffers size to `MAX_ACTION_SIZE` (the tight per-phase upper bound, 53) and overflow is a loud assert.
- **Driver.** `core/driver.pyx::GameDriver` routes a phase-local `action_id` through legality check → phase handler → auto-chain (fast-forward through automated phases + forced decisions until a real multi-choice decision or `PHASE_GAME_OVER`). Optional `history` list records `(state._array.copy(), phase_id, action_id)` tuples pre-mutation.
- **Token extraction.** `core/token_data.get_token_data(state, buffer)` fills `(num_players + 55, TOKEN_DIM=85)` float32 inside one nogil block; `get_token_data_batch` amortizes Python dispatch via `GameState.rebind`. Sole engine → NN bridge. Player tokens trail so higher-player padding masks cleanly. Four learned pass anchors for entity-readout pass phases are appended inside the model after projection — not in the engine-side buffer. BID, ISSUE, and ACQ_OFFER read pass logits from their phase-info tokens.
- **Model.** `nn/transformer.py::RSSTransformerNet` — pre-RMSNorm + SwiGLU, permutation-equivariant (no positional encoding), entity-readout heads, unified-head policy dispatch via static action LUT (`build_action_lut`). Company/corp/player token identity uses learned `company_id_embed` / `corp_id_embed` / `player_id_embed` inferred from token row order; entity ID one-hots are not emitted. Company ownership fields are skipped by `company_proj` then re-injected from corp/player/FI owner-reference embeddings. `TransformerConfig` defaults: 3p, d_model=192, 10 layers, 3 heads, ~5.29M params. `forward(x, legal_mask) → (policy_logits[B, UNIFIED_LOGIT_DIM=255], values[B, N])` — dense output over unified slots (block-per-phase in `DecisionPhase` order, sized by `PHASE_ACTION_SIZES`), illegal slots masked to `-1e9`. Values read from player tokens in canonical order; **no state rotation anywhere in the pipeline**.
- **Evaluator / MCTS / training.** `mcts/evaluator.py` (`NNEvaluator`, `RemoteEvaluator`) speaks token buffers + dense `(B, UNIFIED_LOGIT_DIM)` legal masks end-to-end. `mcts/search.py` does leaf-batched PUCT with subtree reuse. Eval-server IPC carries tokens + legal masks in, softmaxed priors + canonical values out — GPU gather / softmax runs inside the server's autocast region. `train/*` runs full self-play + training loop on 3p.

## Code conventions

- **All state access goes through entity handles.** Non-negotiable. Use `PLAYERS[i].get_cash(state)`, `CORPS[c].get_share_price(state)`, `COMPANIES[i].get_location(state)`, etc. Do **not** cimport `LAYOUT` / `*_FIELDS` / `*_OFFSETS` outside `entities/` or `core/token_data.pyx`. Do **not** index `state._data` directly. Do **not** write ad-hoc field accessors in phase code. Handles own dirty-mask invalidation, cached-star bookkeeping, and ownership-location sync — reaching around them corrupts state.
- **Entity API is three-layered.** Private `cdef` storage helpers (e.g. `_location_at`) in the `.pyx` only; exported `cdef … noexcept nogil` primitives (e.g. `company_location`, `company_owned_by_corp`) in the `.pxd` for hot loops in `phases/` and `core/actions.pyx`; `cpdef` handle methods (e.g. `Company.get_location`) that delegate to the exported primitives for Python callers and slow paths. Both layers share one implementation path — don't write a parallel shortcut. Semantic mutations stay on handle methods (they own cache invalidation); never add a raw field-setter primitive outside the entity's own `.pyx`.
- **Entity method surfaces are mostly finalized.** If you want a method that doesn't exist, **stop and ask the user** before adding one. Same for changing existing semantics.
- **`core/data` is data-only.** Static arrays + enums + normalization divisors. No field-level helpers. Per-entity helpers (synergy aggregation, CoO lookup, par validity) live as private `cdef` functions in the entity that uses them.
- **Naming.** `player_id` / `corp_id` / `company_id` are indices; `PLAYERS[i]` / `CORPS[c]` / `COMPANIES[i]` are the singleton handles. `PHASE_*` = `GamePhases`, `DPHASE_*` = `DecisionPhase`, `LOC_*` = `CompanyLocation`.
- **Assert, don't fall back silently.** In Cython, validate invariants with `assert` + an f-string message — not `if bad: return 0`. Out-of-range ids, inactive entities that should be active, malformed state: crash loudly. `assert` compiles out under `python -O`. **Exception:** genuine business-logic branches ("can't afford the share, so the action is illegal") still return cleanly.
- **No method-level imports.** All imports at file scope — production, tests, scripts.

## Build

- Python: always `.venv/bin/python` (venv may not be activated).
- Pyright: system `/usr/bin/pyright`, NOT `.venv/bin/pyright`.
- Submodules: absolute paths or single-line `cd /home/icebreaker/rss-az-cython2/submodules/18xx && ...`.

```bash
.venv/bin/python setup.py clean # especially needed if you've edited any .pxd files
.venv/bin/python setup.py build_ext --inplace 2>&1 | grep -E "(warning|error)" || true
```

- **Warning-free builds required.** File a beads issue if warnings appear.
- **Clean before final verification** — incremental builds miss `.pxd` header changes.
- **Fix pyright errors before moving on**, even pre-existing. Run `pyright <file>` via Bash for the definitive result (auto-injected diagnostics can be stale).

## Testing

- `pytest tests/` — phase suite (`tests/phases/`), driver step-mode, MCTS, end-to-end self-play smoke. 1800+ tests; runs clean.
- Prefer invariant assertions (cash conservation, share counts, ownership consistency) at transitions over narrow field-level checks.
- **When a test fails, assume the implementation is broken** until proven otherwise.
- `tests/games_18xx/data/` holds 18xx.games JSON fixtures; `tests/games_18xx/replay_harness.py` replays them against the current driver (`test_replay.py` / `test_replay_harness.py`). Known rule divergences: (1) cross-president ACQ transfers, (2) directly offering positive-income company closes in CLOSING.

## Key files by task

| Task | Primary | Secondary |
|------|---------|-----------|
| Layout / field offsets | `core/state.{pyx,pxd}` | `VECTORS.md`, `entities/*.pxd` |
| Action encoding | `core/actions.{pyx,pxd}` | `core/data.pxd::ActionSize`, `VECTORS.md` |
| Static data (synergies, CoO, par) | `core/data.{pyx,pxd}` | `RULES.md` |
| Entity field access | `entities/<entity>.{pyx,pxd}` | — |
| Phase logic | `phases/<phase>.{pyx,pxd}` | entity handles, `RULES.md` (ask before adding handle methods) |
| Driver / game loop | `core/driver.{pyx,pxd}` | `phases/*.pyx` |
| Token features | `core/token_data.{pyx,pxd}` | `token-data.md` |
| Model | `nn/transformer.py` | — |
| Evaluator / MCTS | `mcts/evaluator.py`, `mcts/search.py`, `mcts/node.py`, `mcts/mcts_core.pyx` | — |
| Training loop | `train/main.py`, `train/trainer.py`, `train/self_play.py` | `train/replay_buffer.py`, `train/eval_server.py` |
| Training schedules | `train/config.py` (`TrainingConfig.get_schedule`) | `train_configs/3p.json` (production overrides) |

## Training pipeline specifics

Non-vanilla AlphaZero choices baked into this codebase. Mostly invisible from a casual read — check here before assuming standard behavior.

- **Per-player value head.** Values are `(num_players,)` tanh outputs read from player tokens, not a single scalar. PUCT Q indexes `value_sums[:, active_player_id]` using canonical player ids.
- **Leaf-locked batched MCTS.** Up to `search_batch_size` leaves per GPU call; a queued leaf overwrites its parent edge's Q with `-inf` so in-batch PUCT never re-selects it, with the lock propagating to ancestors. Visit counts are incremented at selection, not backup. Original Q is restored after the NN returns.
- **Subtree reuse with Dirichlet catch-up.** After a real move, the chosen child becomes the root; its per-action visit counts are reset to 0 so fresh Dirichlet noise has real PUCT influence, then virtual backups of each child's mean-Q replay the old visit totals back into the root. The `StatePool` is compacted in-place during reuse (no fragmentation).
- **A0GB value targets.** Value target = NN value at the tree-edge leaf reached by following max-visit children (stop when the current node's best child has 0 visits — the current node *is* the leaf; `value_sum/visit_count = V_NN`). Not game outcomes.
- **Value target annealing.** Linearly blend game-outcome → A0GB from `value_blend_start_epoch` (10) to `value_blend_end_epoch` (200). Bootstraps on reliable signal before switching to lower-variance MCTS-derived targets.
- **c_puct annealing.** Linear `c_puct_initial` (3.5) → `c_puct_final` (2.5) over `c_puct_anneal_epochs` (20).
- **Temperature annealing.** Per-game, `temp_initial` held until move `temp_anneal_start` (60), linear decay to `temp_final` by move `temp_anneal_end` (120).
- **Blended terminal rewards.** `terminal_blend` (default 0.75) mixes rank-based (evenly spaced in [-1, +1] by final placement, ties averaged) with margin-based (zero-sum net-worth deviation, scaled `n/(n-1)`) signals.
- **Unified-policy IPC.** Eval server input is `(W, B, num_tokens, token_dim)` tokens + `(W, B, UNIFIED_LOGIT_DIM)` uint8 legal masks + `(W, B)` int8 phase_ids. Output is `(W, B, UNIFIED_LOGIT_DIM)` f32 priors (already softmaxed over legal slots on GPU) + `(W, B, num_players)` f32 values. No sparse action-id buffers cross the wire; the worker builds the dense mask via `build_action_lut` from phase-local `enumerate_legal_actions` output.
- **Graceful shutdown.** `q + Enter` at the training TTY drains workers and saves checkpoint + replay buffer; `Ctrl-C` is hard exit.

## Devbox

WSL2 on Windows. AMD Ryzen 9 9950X3D (32 cores). AMD Radeon RX 9070 XT (ROCm 7.2.0). Production training is driven by `train_configs/3p.json` (128 self-play workers, 1 eval server, 800→1600 MCTS sims ramped over epochs 200–400). Only 3p is trained currently; 4p/5p configs were removed. Launch with `python -m train <config.json>`.

---

# Agent Instructions

Issue tracking: **bd** (beads). `bd prime` is the authoritative workflow reference — the `SessionStart` hook runs it automatically, so treat its output as source of truth for commands and the session-close protocol.

- File beads issues for any out-of-scope discoveries. **No insights lost.** Always include `--description` — a title alone is not enough.
- Use `bd create --parent=<id>` for related subtasks (dot-notation ids), independent issues for unrelated bugs.
- Claim work with `bd update <id> --claim` (the old `--status=in_progress` still works but `--claim` is preferred).
- Remote sync is `bd dolt pull` / `bd dolt push` (the old `bd sync` no longer exists).
- Persistent cross-session knowledge lives in `bd remember "insight"` / `bd memories <keyword>`, injected by `bd prime`. Prefer this over ad-hoc MEMORY.md files for project-wide context.
- **Ad-hoc scripts:** write to `scratchpad/` (gitignored), iterate with `Edit`, run with `PYTHONPATH=/home/icebreaker/rss-az-cython2 .venv/bin/python scratchpad/<script>.py`. Don't inline 100-line scripts in Bash.
- **Session close:** follow the protocol from `bd prime`. Minimum gate is a clean + `build_ext`. This branch is ephemeral (no upstream) — code is merged to `main` locally, not pushed.
