# MCTS Refactor Plan

> Bead: rss-az-t3d2 — `mcts: rewrite search, node, evaluator, mcts_core for token eval buffers and sparse policy`
> Touches: `mcts/{search.py, node.py, evaluator.py, mcts_core.pyx, mcts_core.pxd}`
> Downstream: rss-az-phli (`train: rewrite eval_server, self_play, main for sparse policy + token buffers`).
> Cross-cuts: `train/eval_server.py`, `train/self_play.py`, `train/replay_buffer.py`. The MCTS-side rewrite changes the evaluator/IPC contract, so eval_server has to move in lockstep — this plan covers eval_server too. self_play/replay_buffer changes are sketched here but executed under rss-az-phli.

## What changed under us

The old MCTS targeted an MLP that consumed a flat float32 visible state with active player rotated to slot 0. The new transformer (`nn/transformer.py`) consumes:

- **Input:** `(batch, num_tokens, token_dim)` float32, where `num_tokens = num_players + 54` and `token_dim = 97`. Built by `core.token_data.get_token_data(state, buffer)` from a compact int16 `GameState`. The active player is just a flag inside the player token — **no state rotation**.
- **Aux input:** `phase_ids: (batch,)` int — model dispatches per decision phase (8 phases).
- **Output:** `policy_logits: (batch, MAX_ACTION_SIZE=14977)` float32 (positions beyond a phase's action count are pre-filled with −1e9), and `values: (batch, num_players)` float32 in canonical order — **no un-rotation**.

Engine bridge:
- `core.driver.DRIVER.apply_action(state, action_id)` → `STATUS_OK | STATUS_GAME_OVER | STATUS_INVALID`. Action ids are **phase-local**.
- `core.actions.enumerate_legal_actions_py(state, uint16_buf)` → count; legal phase-local action ids written into `uint16_buf`.
- `core.actions.get_decision_phase_py(state)` → 0–7 (decision phase) or −1 if currently in an automated/terminal engine phase. The driver's auto-chain guarantees the engine sits at a real decision (≥2 legal actions) on return, so callers always see a non-negative decision phase after `apply_action`.
- `MAX_LEGAL_ACTIONS = 256`. Pad sparse buffers to this width.
- `core.token_data.get_num_tokens(num_players)` and `TokenDataSize.TOKEN_DIM = 97`.

State storage flips from float32 to int16 (compact), so the StatePool gets ~3× smaller.

**IPC contract (worker ↔ eval server).** The model output stays dense `(B, MAX_ACTION_SIZE=14977)`, but **logits never cross the process boundary**. The eval server gathers at per-leaf `action_ids[:n_legal]` and softmaxes on the GPU before scatter, so the worker-visible return is sparse `priors: (B, K_MAX=256)` float32 + canonical-order `values: (B, num_players)`. Worker-side inputs grow to include `action_ids: (B, K_MAX) uint16` and `n_legals: (B,) int16` alongside the token buffer and `phase_ids`. Per-leaf wire shape: in ≈ 23 KB (dominated by the 22 KB token buffer), out ≈ 1 KB. Shared-mem output buffer drops from ~46 MB (`workers × batch × 14977 × 4`) to ~0.8 MB, and per-round-trip IPC drops from ~63 MB → ~18 MB.

What stays the same:
- PUCT selection with FPU = parent value
- Lazy expansion (children only on first PUCT visit)
- Dirichlet root noise (with dynamic α option)
- A0GB greedy-backup value targets
- Pre-allocated `StatePool` + scratch `GameState` rebound across rows
- Subtree reuse via `prepare_reuse_root` + virtual backups during root catch-up
- Leaf-lock batch eval with parent-edge `-inf` Q-locks and lock propagation
- Per-server uint64 bitmap + per-worker `mp.Event` signalling
- Cython nogil `gather_states`/`scatter_results` memcpy helpers

## Out-of-scope for this bead

- **Candidate-scoring ACQ head** (`sparse-refactor.md` §"ACQUISITION"). The model still emits dense `(B, 14977)` logits internally; ACQUISITION pays the full width, and the eval server pays a 14977-wide gather source per leaf. The follow-up bead replaces the corp×company×offset head with a per-candidate scoring head that only scores legal pairs. `nn/transformer.py` is untouched by this rewrite.
- **Raw int16 state over IPC** (worker sends compact state, server runs `get_token_data`). Deferred because it shifts `get_token_data`'s Python prologue from 96 workers onto 2 eval servers — 48× concentration — and the current prologue isn't nogil-clean. Profile the per-leaf 22 KB token-buffer cost before optimizing this; if it's a problem, fix `core/token_data` first.
- Trainer (`train/trainer.py`) updates — handled separately; this rewrite locks down the on-disk replay format the trainer will consume but doesn't touch the loss code.
- 18xx replay tests — they live under rss-az-???? (driver replay path) and don't depend on MCTS.

**In scope (now, vs. prior draft):** sparse priors on the server→worker return path. The model output stays dense, but the eval server does a GPU `gather` + `softmax` before copy-back, so workers receive `priors: (B, K_MAX=256)` float32 instead of dense logits. Kills the 14977-wide shared-mem logits buffer and the worker-side masked-softmax step outright.

---

## Step-by-step plan

Each step is its own commit-sized chunk. Build after each.

### 1. `mcts/mcts_core.pyx` / `.pxd` — purge rotation, add token gather

**Status:** ✅ Done (rss-az-t3d2.1). Post-implementation notes:

- `setup.py` didn't previously include `mcts/` in the pyx build list (it was stripped when MCTS was parked). Added `mcts/` to both `pyx_files` and `packages=` so `build_ext --inplace` actually compiles `mcts_core.pyx`. Without this the "sanity build" in the plan doesn't exercise anything.
- Used `def` (not `cpdef`) for the new gather helpers + `expand_node_sparse`, matching the existing convention in `mcts_core.pyx`. Nothing `cimport`s from this module today; if a later step needs a cimport path we can promote to `cpdef` and add `.pxd` decls then.
- `gather_states` signature unchanged — the 3-D `(W, B, row_floats)` view pattern handles both the old `visible_size` width and the new `num_tokens * token_dim` width. Only the inner var name (`vis` → `row_floats`) and the docstring changed.
- `scatter_results` **parameter rename** (breaking for callers): `src_logits`/`dst_logits`/`logit_row_bytes` → `src_priors`/`dst_priors`/`priors_row_bytes`. Byte-view dtype-agnostic abstraction preserved; only the names reflect the new payload. Step 4 consumers in `train/eval_server.py` must use the new kwargs.
- `expand_node` (the old dense-mask form) is retained alongside `expand_node_sparse` as called out in the plan — Step 2 deletes it when `MCTSNode.expand` rewires onto the sparse path. `mcts/node.py` still imports it; removing it now would break `mcts/node.py`'s import even though `mcts_core.so` builds clean.
- Smoke coverage: `scratchpad/step1_smoke.py` (gitignored) exercises `expand_node_sparse` (full + partial-n), all four gather helpers, and `scatter_results` against hand-rolled per-worker shared-mem layouts. Keep around for rss-az-t3d2.2 to extend.
- **Expected post-step fallout:** `mcts/evaluator.py` and `mcts/search.py` import deleted names (`rotate_visible_state_into`, `masked_softmax`, `unrotate_values`); those modules fail at import-time until Steps 3/5 land. `mcts_core.so` itself builds warning-free.

**Delete:**
- `_rotate_visible_state`, `rotate_visible_state_into` (Python+cdef), and the field-offset args they take (`field0/1/2_offset`, `players_offset`, `player_stride`, `visible_size`).
- `unrotate_values` (no rotation needed; values come back canonical from the model).
- `_masked_softmax` / `masked_softmax`: **delete**. The eval server gathers + softmaxes on GPU before returning priors, so neither the worker nor `NNEvaluator` needs the Cython helper. (`NNEvaluator`, used in-process for tests, calls `torch.gather` + `torch.softmax` on the model output directly — cheap, no need for a separate Cython path.)
- `_select_child_impl` / `select_child`: keep verbatim — PUCT works against per-action `value_sums[:, active_player_id]` and `active_player_id` is now the canonical id (no rotation), which is exactly what we want.
- `_backup_node` / `backup`: keep verbatim — values are already canonical.
- `_virtual_backup_node` / `virtual_backup`: keep verbatim.
- `_increment_visits`: keep verbatim.
- Atomic primitives (`worker_publish_request`, `server_drain_bitmap`, `server_peek_bitmap`): keep verbatim.

**Modify:**
- `gather_states`: row width parameterizes from `visible_size` to `num_tokens * token_dim`. Easiest path is to pass a precomputed `row_floats` and let the existing code memcpy `row_floats * sizeof(float)` per row. The src/dst become 4-D logically (`workers × batch × num_tokens × token_dim`) but stored as 3-D `(workers, batch, row_floats)` from Cython's POV; the contiguous memcpy is identical. Update the docstring.
- `scatter_results`: row shape shrinks from `(batch, MAX_ACTION_SIZE=14977)` logits to `(batch, K_MAX=256)` sparse priors. Same contiguous memcpy, smaller stride. Update docstring: the server has already gathered + softmaxed, so the worker expands directly from the scattered `priors` row with no further transform.

**Add:**
- `cpdef expand_node_sparse(node, const uint16_t[:] action_ids, int n, const float[:] priors_legal, const float[:] default_value, int num_players)` — directly accepts the sparse legal list + aligned priors. Replaces the existing `expand_node(node, policy_priors, legal_mask, …)` which does mask scanning. Internally:
  - allocates `legal_actions = np.empty(n, int32)` and copies from `action_ids[:n]`,
  - allocates `priors`, `visit_counts`, `value_sums` like before,
  - broadcasts `default_value` into each `value_sums` row.
  - Drop `expand_node` entirely once `MCTSNode.expand` is updated.
- `cpdef gather_phase_ids(dst_int8, src_int8, worker_indices, counts, num_requests)` — rank-2 `(W, B)` int8 → rank-1 `(max_batch,)` int8. Same `prange` + memcpy skeleton as `gather_states`, 1-byte row.
- `cpdef gather_action_ids(dst_int16, src_int16, worker_indices, counts, num_requests)` — rank-3 `(W, B, K_MAX)` int16 → rank-2 `(max_batch, K_MAX)` int16. Row = `K_MAX * 2` bytes.
- `cpdef gather_n_legals(dst_int16, src_int16, worker_indices, counts, num_requests)` — rank-2 `(W, B)` int16 → rank-1 `(max_batch,)` int16. 2-byte row.

All three gather helpers are structurally identical to `gather_states` (same prange, same memcpy, different dtype/width). Factor as a fused-type template if Cython's cooperative; otherwise a trivial copy+paste for three.

**Sanity check at end of step:** `python setup.py build_ext --inplace` warns/errors only on consumers we'll fix in the next steps.

### 2. `mcts/node.py` — sparse expand + per-leaf phase context

**Status:** ✅ Done (rss-az-t3d2.2, commit `fa9171f`). Post-implementation notes:

- `MCTSNode.expand` rewired as specified — delegates straight to `expand_node_sparse`.
- `__slots__`: dropped `pending_mask`; added `pending_action_ids`, `pending_n`, `pending_phase`.
- Dense `expand_node` removed from `mcts_core.pyx` at the same time (the only caller was `MCTSNode.expand`). This is slightly ahead of where Step 1's notes parked it, but nothing imports the dense path anymore.
- `active_player_id` docstring clarified: canonical actor id, not rotated slot 0. PUCT indexing `value_sums[:, active_player_id]` is unchanged because the new transformer returns canonical-order values.
- **Expected post-step fallout:** `mcts/search.py` still references `pending_mask` and the dense expand signature; it fails at import until Step 5 lands. `mcts_core.so` + `mcts/node.py` import warning-free.

**Replace:** `MCTSNode.expand(policy_priors, legal_mask, num_players, default_value)` with `MCTSNode.expand(action_ids, n, priors, num_players, default_value)`.

`MCTSNode.__slots__` changes:
- Remove `pending_mask` (was the dense legal mask cached at child creation for use after batch eval).
- Add `pending_action_ids: np.ndarray | None` (uint16, length ≥ K_legal; only first n entries valid).
- Add `pending_n: int` (count of legal actions for this leaf).
- Add `pending_phase: int` (decision phase 0–7, populated at child creation; needed because the model takes per-leaf `phase_ids`).

`active_player_id` semantics unchanged but no longer means "rotated slot 0" — it's just the canonical id of whoever acts at this node. The `value_sums[:, active_player_id]` indexing in PUCT is identical.

`mean_value(player_id)` unchanged.

### 3. `mcts/evaluator.py` — strip rotation, add token-buffer fill

**Status:** ✅ Done (rss-az-t3d2.3). Post-implementation notes:

- `BaseEvaluator` + `NNEvaluator` rewritten against the plan. `RemoteEvaluator` stays in `train/eval_server.py` and is owned by Step 4 — Step 3 only touches `mcts/evaluator.py`.
- `fill_token_buffer` added; `_rotation_buf` replaced by `_token_buf` of shape `(n, num_tokens, token_dim)` f32 that grows on demand.
- `evaluate`, `evaluate_batch`, `evaluate_leaves` all run the **gather + softmax inside the autocast region** (bf16/fp16 when enabled) and only cast to f32 on the final assignment, matching the server-side path.
- `evaluate_leaves` takes raw int16 `state_arrays` + pre-enumerated `phase_ids` + `(n, K_MAX) uint16 action_ids_buf` + `n_legals`. It rebinds a single scratch `GameState` across rows (no per-leaf wrapper allocation) to hand each row to `get_token_data`. The `active_player_ids` argument is gone as specified.
- **Model-config validation** relaxed: old `cfg.action_dim` / `cfg.value_dim` checks were removed (the new `TransformerConfig` doesn't expose those fields — action width is set by `PHASE_ACTION_SIZES`, value width always tracks `num_players`). Kept a `cfg.num_players == num_players` sanity check.
- `evaluate_terminal` migrated from `state.get_player_net_worth(i)` to `PLAYERS[i].get_net_worth(state)` — the old GameState method is gone post-refactor, entity handles are the single access path.
- **Expected post-step fallout:** `mcts/search.py` imports `masked_softmax` + dense `evaluate_leaves(state_arrays, active_player_ids)`; broken until Step 5. `train/eval_server.py` imports `rotate_visible_state_into`; broken until Step 4. `train/self_play.py` imports `rotate_visible_state`; broken until Step 6 (under rss-az-phli). `train/{tournament, analyze_game, main}.py` and `utils_18xx/server.py` import `NNEvaluator`/`compute_terminal_values` — still exported, so they survive at import time.

This file shrinks substantially. Goal: a clean `BaseEvaluator` + `NNEvaluator` (in-process model) + `RemoteEvaluator` (shared-mem IPC) trio that all speak token buffers.

**Delete:**
- `rotate_visible_state_into` / `rotate_visible_state` (Python wrappers + Cython call) — entire concept gone.
- `unrotate_values` — values are already canonical.
- `LayoutInfo` / `get_layout` cache used to pull rotation offsets — no longer needed here. (Other modules still call `core.state.get_layout` for `total_size` when sizing the StatePool — that's fine, just not the evaluator's job.)
- `BaseEvaluator._finalize_*` helpers: drop both `unrotate_values` (values come back canonical) and `apply_mask_softmax` (priors come back already gathered + softmaxed on the server). `_finalize_*` collapses to a thin "slice out of shared mem into the return tuple" step.
- The `_DTYPE_MAP` autocast dance stays in `NNEvaluator` (eval-server-side `_eval_server_serve` is the production path; `NNEvaluator` is mostly used for unit-test single-process eval).

**Add:**
- A small helper `fill_token_buffer(state, buf2d)` that wraps `core.token_data.get_token_data(state, buf2d)`. This isolates the import-site so callers don't all need to know the Cython entry point. Rejecting num_players outside 3–5 is already enforced inside `get_token_data` via assert.
- `BaseEvaluator.__init__` keeps `num_players`, `terminal_rank_weight`. Add `self.num_tokens = get_num_tokens(num_players)` and `self.token_dim = int(TokenDataSize.TOKEN_DIM)` to size buffers. Drop `self.layout`.
- `compute_terminal_values` and `evaluate_terminal` are unchanged (`PLAYERS[i].get_net_worth(state)` still works).

**Reshape `evaluate*` methods:**

`NNEvaluator.evaluate(state) -> (sparse_priors_K, values_canonical, action_ids_K, n_legal, phase_id)`
- Compute `phase_id = get_decision_phase_py(state)`.
- Allocate (or reuse) a `(1, num_tokens, token_dim)` float32 buffer; fill with `get_token_data`.
- Enumerate legal actions into a `uint16[K_max]` buffer; record `n_legal`.
- Forward through model with `phase_ids = torch.tensor([phase_id])`; get back `logits: (1, MAX_ACTION_SIZE)` and `values: (1, num_players)`.
- Gather sparse logits at `action_ids[:n_legal]` via `torch.gather`, apply softmax → `priors_legal: (n_legal,)`. (Mirrors what the eval server does on GPU for `RemoteEvaluator`; here it runs in-process since no IPC is involved.)
- Return canonical-order values directly (no roll).

`NNEvaluator.evaluate_batch(states)` and `NNEvaluator.evaluate_leaves(state_arrays, phase_ids, action_ids_buf, n_legals)` are batch versions of the above. The signature for `evaluate_leaves` becomes:

```python
def evaluate_leaves(
    self,
    state_arrays: list[np.ndarray],   # raw int16 GameState arrays
    phase_ids: list[int],
    action_ids_buf: np.ndarray,       # (n, K_max) uint16
    n_legals: list[int],
) -> list[tuple[np.ndarray, np.ndarray]]:  # (priors_legal[:n], values_canonical)
```

Note the `active_player_ids` argument is gone.

**Reusable buffers** on `NNEvaluator`:
- `_token_buf: np.ndarray | None` — `(n, num_tokens, token_dim)` f32, grows as needed.
- Drop `_rotation_buf`.

### 4. `train/eval_server.py` — token inputs, sparse priors out, GPU-side gather

**Status:** ✅ Done (rss-az-t3d2.4, commit `46560d6`). Implementation landed as specified below. Post-implementation notes:

- `SharedEvalBuffers` constructor: `visible_size` and `action_dim` gone. `num_tokens` / `token_dim` derived from `core.token_data`; `K_MAX` from `core.actions.MAX_LEGAL_ACTIONS_PY`. `num_players` retained for validation.
- `_action_ids` stored as `int16` (torch has no `uint16`); workers view the same memory as `uint16` via numpy. Accessor naming follows the plan (`get_input_action_ids_np` etc., `get_output_priors_np` replaces `get_output_logits_np`).
- `_eval_server_serve` runs `gather` + `masked_fill(-1e9)` + `softmax` inside the autocast region, casting to f32 only on the final `gpu_priors[:B] = ...` copy. All four Cython gather helpers from Step 1 are used in parallel for the four input fields.
- `RemoteEvaluator.evaluate` writes legal action ids directly into the shared-mem input buffer via `enumerate_legal_actions_py` — no worker-side scratch round-trip. Returns the 5-tuple `(priors_legal, values, action_ids_legal, n_legal, phase_id)`.
- `evaluate_leaves` signature realigned to `(state_arrays, phase_ids, action_ids_buf, n_legals)` per plan — `active_player_ids` gone.
- **Expected post-step fallout:** `train/self_play.py` still imports the old flat-state eval path; broken until Step 6. `mcts/search.py` already speaks the new contract (Step 5 landed the same day).

`SharedEvalBuffers` schema (`K_MAX = MAX_LEGAL_ACTIONS = 256`):

```python
# Worker → Server
self._states     = torch.zeros(num_workers, batch_size, num_tokens, token_dim,
                               dtype=torch.float32).share_memory_()
self._phase_ids  = torch.zeros(num_workers, batch_size,
                               dtype=torch.int8).share_memory_()
self._action_ids = torch.zeros(num_workers, batch_size, K_MAX,
                               dtype=torch.int16).share_memory_()   # uint16 via bit-reinterp
self._n_legals   = torch.zeros(num_workers, batch_size,
                               dtype=torch.int16).share_memory_()

# Server → Worker  (sparse over legal actions — no dense logits cross IPC)
self._priors     = torch.zeros(num_workers, batch_size, K_MAX,
                               dtype=torch.float32).share_memory_()
self._values     = torch.zeros(num_workers, batch_size, num_players,
                               dtype=torch.float32).share_memory_()
self._counts     = torch.zeros(num_workers, dtype=torch.int32).share_memory_()
```

Constructor loses `visible_size` and `action_dim`; everything is derived — `num_tokens`/`token_dim` from `core.token_data`, `K_MAX` from `core.actions.MAX_LEGAL_ACTIONS`. `MAX_ACTION_SIZE` is the model's internal output width and **never appears in a shared-mem shape**. Keep `num_players` for validation.

**Dtype note:** `torch.uint16` doesn't exist, so `_action_ids` is stored as `int16`. Action ids max out at 14976 ≪ 32767, so a signed view is safe and bit-reinterpretable with a worker-side `uint16` numpy view over the same memory.

Accessors on `SharedEvalBuffers`:
- `get_input_states_np(worker_idx) → (batch, num_tokens, token_dim)` f32
- `get_input_phase_ids_np(worker_idx) → (batch,)` int8
- `get_input_action_ids_np(worker_idx) → (batch, K_MAX)` uint16 (numpy view over the int16 tensor)
- `get_input_n_legals_np(worker_idx) → (batch,)` int16
- `get_output_priors_np(worker_idx) → (batch, K_MAX)` f32  *(replaces `get_output_logits_np`)*
- `get_output_values_np(worker_idx) → (batch, num_players)` f32

`_eval_server_serve` changes:

- Pinned buffers:
  - `pin_s`, `gpu_s`: `(max_batch, num_tokens, token_dim)` f32.
  - `pin_phase_ids`, `gpu_phase_ids`: `(max_batch,)` int8.
  - `pin_action_ids`, `gpu_action_ids`: `(max_batch, K_MAX)` int16.
  - `pin_n_legals`, `gpu_n_legals`: `(max_batch,)` int16.
  - `pin_priors`, `gpu_priors`: `(max_batch, K_MAX)` f32 — **replaces** `pin_log (max_batch, MAX_ACTION_SIZE)`.
  - `pin_val`, `gpu_val`: `(max_batch, num_players)` f32.

- Gather path assembles all four inputs in parallel via the Cython helpers added in Step 1 (one row-size-agnostic memcpy per field): `gather_states`, `gather_phase_ids`, `gather_action_ids`, `gather_n_legals`.

- Forward + **GPU gather + softmax** (inline after the model call, inside the autocast region so the gather runs in the model's output dtype — see Risks):
  ```python
  logits_dense, values = model(gpu_s_batch, gpu_phase_ids_batch)
  #   logits_dense: (B, MAX_ACTION_SIZE)   values: (B, num_players)
  idx = gpu_action_ids_batch.to(torch.long)                   # (B, K_MAX)
  gathered = logits_dense.gather(1, idx)                      # (B, K_MAX)
  k_range = torch.arange(K_MAX, device=dev)
  k_mask = k_range[None, :] < gpu_n_legals_batch[:, None].to(torch.long)
  gathered = gathered.masked_fill(~k_mask, -1e9)
  gpu_priors[:B] = gathered.softmax(dim=1).to(torch.float32)  # cast before copy
  gpu_val[:B]    = values.to(torch.float32)
  ```
  Tail slots `[n_legal:K_MAX]` end up at near-zero prob (softmax of −1e9). Workers must only read `priors[:n_legal]`.

- Scatter: `scatter_results(pin_priors, pin_val, ...)` with priors row = `K_MAX * 4 = 1024` bytes (vs. old `MAX_ACTION_SIZE * 4 ≈ 60 KB`).

- The `_DTYPE_MAP` autocast dance stays — it determines `logits_dense.dtype`, which the gather inherits.

`RemoteEvaluator` (worker side):

- Shared-mem views:
  - `_in_states_np`: `(batch_size, num_tokens, token_dim)` f32.
  - `_in_phase_ids_np`: `(batch_size,)` int8.
  - `_in_action_ids_np`: `(batch_size, K_MAX)` uint16 (numpy view of the int16 shared tensor).
  - `_in_n_legals_np`: `(batch_size,)` int16.
  - `_out_priors_np`: `(batch_size, K_MAX)` f32  *(replaces `_out_logits_np`)*.
  - `_out_values_np`: `(batch_size, num_players)` f32.

- `evaluate(state)`:
  - `phase_id = get_decision_phase_py(state)`.
  - `get_token_data(state, self._in_states_np[0])`.
  - `self._in_phase_ids_np[0] = phase_id`.
  - `n = enumerate_legal_actions_py(state, self._in_action_ids_np[0])` — writes directly into shared mem.
  - `self._in_n_legals_np[0] = n`.
  - Submit / wait via the existing bitmap+event protocol.
  - Return `(self._out_priors_np[0, :n].copy(), self._out_values_np[0].copy(), self._in_action_ids_np[0, :n].copy(), n, phase_id)`. **No worker-side gather or softmax.**

- `evaluate_leaves(state_arrays, phase_ids, action_ids_buf, n_legals)`:
  - For each leaf `i`: fill `_in_states_np[i]` via `get_token_data`, set `_in_phase_ids_np[i]`, copy `_in_action_ids_np[i, :n_legals[i]] = action_ids_buf[i, :n_legals[i]]`, set `_in_n_legals_np[i] = n_legals[i]`.
  - Submit single batch via bitmap; wait.
  - Return `[(self._out_priors_np[i, :n_legals[i]].copy(), self._out_values_np[i].copy()) for i in range(len(state_arrays))]`.

- `evaluate_terminal(state)` unchanged.

### 5. `mcts/search.py` — wire up the new contract

**Status:** ✅ Done (rss-az-t3d2.5, commit `17ac1e6`). Post-implementation notes:

- `StatePool` now **owns the per-search scratch buffers** rather than having `run_search` allocate them locally: `_legal_scratch` (`K_MAX` uint16) is created in `__init__`; `_pending_action_ids_buf` / `_pending_n_buf` grow lazily via `StatePool.ensure_pending_bufs(batch_size)` on first use. The plan originally parked these as `run_search` locals — co-locating them with the pool keeps all persistent MCTS scratch in one place and matches the "allocate once, reuse" intent.
- `_pending_n_buf` stored as **int32** (plan said int64 / implicit). `evaluate_leaves` consumes it via Python list comprehension, so the width is immaterial — int32 is cheap + matches other internal counters.
- **Critical correctness fix** baked into this step: the `[n_legal:K_MAX]` tail of each row of `_pending_action_ids_buf` is zeroed before handoff to `evaluate_leaves`. `torch.gather` on the server runs across the full `K_MAX` width before `masked_fill` applies — stale action ids from a larger prior batch would otherwise blow the 14977-wide dense-logits range (CPU raises, CUDA corrupts silently). The zero fill is cheap and mandatory.
- Root eval: `TURN.get_phase(root_state)` / `TURN.get_active_player(root_state)` replace the missing `GameState.get_phase` / `.get_active_player` accessors. Root's `pending_phase` is stashed before `root.expand(action_ids, n_legal, priors, num_players, values)`.
- Leaf creation: drops `child.pending_mask = get_valid_action_mask(...)` (dense helper is gone). Writes `child.pending_phase`, `child.pending_n`, and copies `_legal_scratch[:n]` into `child.pending_action_ids`.
- `get_action_probabilities(root, temperature)` — `action_dim` argument dropped. Current implementation returns a dense `(MAX_ACTION_SIZE,)` float32 vector keyed off the module-level `MAX_ACTION_SIZE` constant. `self_play` now works directly off `root.visit_counts` (sparse), so the dense return is effectively dead — leave in place until it's either removed or migrated to a sparse `(action_ids, probs)` return under rss-az-phli.
- Smoke coverage: 3p game, 16 sims, batch 4, fresh search + `prepare_reuse_root` + 5-move sequence all ran clean end-to-end before commit. Warning-free build.

`StatePool` shape becomes `(capacity, total_int16_size)` int16:

```python
self.states = np.zeros((capacity, state_size), dtype=np.int16)
```

`from core.state import get_layout; total_size = get_layout(num_players).total_size` already returns int16 element count. Everything else (`alloc`, `alloc_from_row`, `compact`, `row`) is shape-agnostic and stays as-is.

`run_search` rewrites in three small pieces.

**Root setup:**
- `is_terminal = root_state.get_phase() == GamePhases.PHASE_GAME_OVER` — `GameState.get_phase` doesn't exist; replace with `TURN.get_phase(root_state)` (already done elsewhere).
- For non-terminal root, evaluator returns 5-tuple `(sparse_priors, values, action_ids, n_legal, phase_id)`. Stash `phase_id` and `action_ids` on the root via `expand` (Step 2's new sparse signature).
- `root.expand(action_ids, n_legal, sparse_priors, num_players, default_value=values)`.

**Selection / leaf creation:**
- After `DRIVER.apply_action(scratch_gs, action_idx)` succeeds and we're not terminal:
  - `child.pending_phase = get_decision_phase_py(scratch_gs)`.
  - `child.pending_n = enumerate_legal_actions_py(scratch_gs, scratch_uint16_buf)`.
  - `child.pending_action_ids = scratch_uint16_buf[:child.pending_n].copy()` — copy because the scratch buffer is reused across leaves.
- Drop `child.pending_mask = get_valid_action_mask(scratch_gs)` (function gone with the dense path).

**Batch eval:**
- Build `phase_ids = [node.pending_phase for _, node in pending]` and pass to `evaluator.evaluate_leaves(state_arrays, phase_ids, ...)`.
- For the action_ids arg, build a `(len(pending), K_max) uint16` packed buffer from `node.pending_action_ids` per-row + `n_legals` list. Reusable across batches.
- Returned `priors` are already sparse + softmaxed. Apply `expand(node.pending_action_ids, node.pending_n, priors, num_players, default_value=values)` directly. Drop the `_apply_mask_softmax` call.
- Clear `pending_action_ids`, `pending_n`, `pending_phase` after expansion (they're consumed).

**Helpers:**
- `_propagate_lock`, `_propagate_unlock`, `_add_dirichlet_noise`, `get_action_probabilities`, `get_greedy_leaf_value`, `_collect_subtree_nodes`, `_reset_root_for_reuse`, `prepare_reuse_root`: unchanged. They operate on sparse per-action arrays already.

**Scratch buffers (allocate once, reuse across simulations):**
- `scratch_gs` — bind to pool row 0 initially (same as today).
- `_legal_scratch = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint16)` for the per-leaf enumerate call.
- `_pending_action_ids_buf = np.empty((batch_size, MAX_LEGAL_ACTIONS), dtype=np.uint16)` and `_pending_n_buf = np.empty(batch_size, dtype=np.int32)` to hand to `evaluator.evaluate_leaves` without per-leaf allocation.

`get_action_probabilities(root, temperature)` — drop the `action_dim` argument; the function can return a dense `(MAX_ACTION_SIZE,)` for now (caller in self_play samples from it). Once self_play moves to sparse sampling (Step 7), this function can return `(action_ids, probs)` instead.

### 6. `train/self_play.py` — sparse policy targets, no rotation

**Status:** ✅ Done (rss-az-phli.1, commit `f31f1d9`). Post-implementation notes:

- `SelfPlayExample` dataclass **added alongside** the legacy `TrainingExample` rather than replacing it — `TrainingExample` still lives on in `train/replay_buffer.py` under the pre-refactor dense schema, annotated as "legacy dense-MLP schema, scheduled for replacement under rss-az-phli.2" (= Step 7 below).
- `GameRecord` schema flipped to the six padded arrays listed in the plan: `states (N, total_int16)` int16, `phase_ids (N,)` int8, `n_legals (N,)` int16, `action_ids (N, K_MAX)` uint16 zero-padded, `policy_targets (N, K_MAX)` f32 zero-padded, `value_targets (N, num_players)` f32. `legal_masks` dropped.
- Per-move loop: enumerates legal actions once via `enumerate_legal_actions_py`, uses `root.visit_counts` directly as the sparse policy target (no `get_action_probabilities` call), samples via a temperature-scaled variant with a `temp < 1e-8` greedy guard, reads `get_greedy_leaf_value(root, num_players)` for the canonical value target (no `np.roll`).
- Value blending (`value_blend_alpha`) simplified: since `compute_terminal_values` is already canonical, the blend broadcasts `terminal_values[None, :]` across the record without a per-row rotation loop.
- Preferred net-worth accessor: `PLAYERS[i].get_net_worth(state)` (the old `GameState.get_player_net_worth` method is gone post-refactor). All previously-inline imports in `self_play.py` hoisted to file scope per CLAUDE.md convention.
- **Collateral config churn:** `train/config.py` drops `TrainingConfig.visible_size` entirely (field, assignment, and legacy JSON-normalize pop). `action_dim` is now sourced directly from `core.data.MAX_ACTION_SIZE` (the removed `core.actions.get_total_action_count(num_players)` helper no longer exists). `train/replay_buffer.py` renamed its `visible_size` init arg to `state_size` (still on the legacy dense schema — Step 7).
- Downstream **stale references** in `train/{main,tournament,analyze_game}.py` and `utils_18xx/server.py` (flat-state IPC, dense policy loss, rotated value targets) are knowingly left broken — those files need full rewrites under the rss-az-phli parent and are tracked separately.

This file lives under rss-az-phli but the changes are tightly coupled to MCTS, so plan them here.

Per-move:
```python
active_player = TURN.get_active_player(state)
phase_id = get_decision_phase_py(state)
n_legal = enumerate_legal_actions_py(state, scratch)
legal_actions = scratch[:n_legal].copy()

root = run_search(state, evaluator, mcts_config, rng,
                  state_pool=state_pool, reuse_root=reuse_root)

# Sparse policy target — visit counts over legal actions only
counts = root.visit_counts.astype(np.float32)
policy_target_sparse = counts / counts.sum()    # (n_legal,)

# Action sampling (still sparse)
temp_scaled = counts ** (1.0 / temperature)
sample_probs = temp_scaled / temp_scaled.sum()
chosen_idx = rng.choice(n_legal, p=sample_probs)
action_idx = int(legal_actions[chosen_idx])

# A0GB value target (already canonical, no rotation)
value_target = get_greedy_leaf_value(root, num_players)

examples.append(SelfPlayExample(
    state=state._array.copy(),         # int16 raw state
    phase_id=phase_id,
    n_legal=n_legal,
    action_ids=legal_actions,           # uint16 (n_legal,)
    policy_target=policy_target_sparse, # float32 (n_legal,)
    value_target=value_target,          # float32 (num_players,)
))

DRIVER.apply_action(state, action_idx)
```

Drop the `rotate_visible_state(...)` and `np.roll(value_target, ...)` calls.

`GameRecord` arrays change:
- `states`: `(N, total_int16_size)` int16
- `phase_ids`: `(N,)` int8
- `n_legals`: `(N,)` int16
- `action_ids`: `(N, K_max)` uint16, padded
- `policy_targets`: `(N, K_max)` float32, padded with zeros
- `value_targets`: `(N, num_players)` float32

Drop `legal_masks` from the record.

### 7. `train/replay_buffer.py` — sparse padded schema

**Status:** ✅ Done (rss-az-phli.2). Post-implementation notes:

- `ReplayBuffer.__init__(capacity, state_size_int16, num_players, k_max=256)` — the six-array sparse schema from the plan landed verbatim: `_states (capacity, state_size_int16) int16`, `_phase_ids (capacity,) int8`, `_n_legals (capacity,) int16`, `_action_ids (capacity, k_max) uint16`, `_policy_targets (capacity, k_max) f32`, `_value_targets (capacity, num_players) f32`.
- `add_stacked(states, phase_ids, n_legals, action_ids, policy_targets, value_targets)` — six positional args mirroring the `GameRecord` fields. The ring-buffer wrap + overflow-truncate-to-last-N skeleton is preserved unchanged across all six arrays.
- `sample()` returns all six tensors. `action_ids` is returned as **int16** (torch has no `uint16`); the reinterpret is lossless because max action id 14976 ≪ 32767 and consumers can view the tensor's underlying memory as `uint16` via numpy when needed. Same decision as Step 4's shared-mem storage — kept symmetric across the whole pipeline.
- `TrainingExample` NamedTuple flipped to the sparse row contract (`state int16`, `phase_id`, `n_legal`, `action_ids uint16`, `policy_target f32`, `value_target f32`). It's not used by the hot path today (worker→trainer goes through `add_stacked` on pre-stacked arrays from `GameRecord`), but keeping it 1:1 with `SelfPlayExample` from `train/self_play.py` preserves the single-example API for tests + manual inspection.
- `save` / `load` — metadata now records `state_size` / `num_players` / `k_max` alongside `capacity` / `size` / `index`. Load rejects on any of those mismatching (not just capacity), because silently padding or truncating sparse rows across a `k_max` change would corrupt the policy target's probability mass. Warn-and-skip behavior preserved.
- **Expected post-step fallout:** `train/main.py` still calls the legacy 4-arg `ReplayBuffer(capacity, visible_size, action_dim, num_players)` constructor and the old 4-arg `add_stacked(states, legal_masks, policy_targets, value_targets)` — broken at call time. Left as-is; the full `train/main.py` rewrite is tracked under the rss-az-phli parent. Nothing else in the repo imports `ReplayBuffer`, so the blast radius is contained to that one file.
- Smoke coverage: happy-path add/sample/dtype checks, ring-buffer wrap-around, overflow (n ≥ capacity → keep last-N), save/load round-trip, shape-mismatch skip on load, and oversample `ValueError` — all green via a scratchpad script (no persistent test suite — `tests/test_replay_buffer.py` worth adding once `train/main.py` comes back online, so the end-to-end `GameRecord` → `ReplayBuffer` → `sample` → trainer path is exercised together).

Mirror the `GameRecord` schema:

```python
class ReplayBuffer:
    def __init__(self, capacity, state_size_int16, num_players, k_max=256):
        self._states         = np.zeros((capacity, state_size_int16), dtype=np.int16)
        self._phase_ids      = np.zeros(capacity, dtype=np.int8)
        self._n_legals       = np.zeros(capacity, dtype=np.int16)
        self._action_ids     = np.zeros((capacity, k_max), dtype=np.uint16)
        self._policy_targets = np.zeros((capacity, k_max), dtype=np.float32)
        self._value_targets  = np.zeros((capacity, num_players), dtype=np.float32)
```

`add_stacked` and `sample` follow the same ring-buffer pattern; just plumb the new arrays. `sample` returns:

```python
{
    "states": torch.from_numpy(self._states[indices]),         # int16
    "phase_ids": torch.from_numpy(self._phase_ids[indices]),
    "n_legals": torch.from_numpy(self._n_legals[indices]),
    "action_ids": torch.from_numpy(self._action_ids[indices]),
    "policy_targets": torch.from_numpy(self._policy_targets[indices]),
    "value_targets": torch.from_numpy(self._value_targets[indices]),
}
```

`save`/`load` write/read the new arrays to disk. The trainer is then responsible for calling `get_token_data` per sampled state at training time (small cython call, runs nogil — fine in DataLoader workers).

### 8. Smoke test before declaring victory

**Status:** 🟡 Partially done — the scratchpad-smoke intent was promoted to a **persistent test suite**. `tests/test_mcts.py` (113 tests, commit `79b4807`) covers post-refactor MCTS end-to-end: `MCTSConfig`, `MCTSNode`, `expand_node_sparse`, PUCT `select_child`, `compute_terminal_values`, `backup` / `increment_visits` / `virtual_backup`, Dirichlet noise, `StatePool`, `NNEvaluator.{evaluate, evaluate_batch, evaluate_leaves, evaluate_terminal}`, `fill_token_buffer`, `run_search` (single + batched), propagation lock/unlock, subtree reuse, and all four `gather_*` + `scatter_results` IPC primitives. Replaces the deleted `tests/test_mcts-old.py`.

The worker ↔ eval_server IPC smoke (Step 4's multi-worker round-trip with real shared memory + the bitmap+event protocol) is **not yet covered**. `RemoteEvaluator` is imported by `test_mcts.py` only at the class-surface level — no end-to-end IPC invocation test. The original plan's "4-worker / 1-server `train.main` smoke" stays as follow-up once `train/main.py` + `train/replay_buffer.py` (Step 7) are back online.

**Test-suite bug caught during this step:** `mcts/evaluator.py::NNEvaluator.evaluate_batch` was allocating `action_ids_buf` with `np.empty`, leaving garbage past `n_legals[i]`. `torch.gather` then read across the full `K_MAX` width and produced OOB indices into the dense 14977-wide logits. Fixed by switching to `np.zeros`; legal rows still read `[:n_legal]`, and the zero tail is masked out by the `k_mask` path downstream. Same correctness failure mode flagged in Step 5 for the `_pending_action_ids_buf` tail — two sites, one invariant: **any action_ids buffer that reaches `torch.gather` must have its tail zeroed (or otherwise in-range) before the gather runs.**

---

## Risks / things to watch

- **State buffer dtype.** The old StatePool was float32 because the old eval consumed float32 directly. The new pool stores int16 and `get_token_data` widens to float32 *into the eval buffer*. Make sure no leftover code interprets the pool row as float32 anywhere — it's a silent bug if it does (mismatched stride).
- **Phase context tokens are zeroed outside their phase.** That's correct by design — the trunk learns to ignore them via the phase-specific projection — but it means the same `(num_tokens, token_dim)` buffer shape works for every phase. Don't be tempted to skip writing them.
- **`get_token_data` runs a Python prologue** (forces per-player cache refresh) before its nogil body. That's a per-call GIL acquisition. With 96 workers each calling it per leaf, the GIL pressure is real. If it shows up in profiling, the fix is in `core.token_data` (push the cache refresh into nogil) — not here.
- **Input IPC width grows; output shrinks.** Per-leaf input is now `num_tokens * token_dim * 4` bytes (3p: 22 KB) + `K_MAX * 2` for action_ids (512 B) + small scalars, vs. old `visible_size * 4` (a few hundred bytes). Per-leaf output goes the other direction — sparse priors `K_MAX * 4 = 1 KB` vs. dense `MAX_ACTION_SIZE * 4 ≈ 60 KB`. Net per-round-trip (96 workers × batch 8 = 768 leaves) drops from ~63 MB → ~18 MB; shared-mem `_priors` buffer sits at ~0.8 MB where the old `_logits` buffer would have been ~46 MB. The input side is now the bottleneck — profile the worker-side `get_token_data` fill under load.
- **GPU gather/softmax latency.** The server adds one `torch.gather` + one `masked_fill` + one `softmax` per batch after the model forward. All are bandwidth-bound kernels on the already-resident logits tensor (`B × 14977 × dtype_size`). Expected negligible vs. the transformer forward; if profiling ever shows otherwise, fuse into a single custom kernel or move the gather to the CPU side after copy-back (still cheap since the copied tensor would only be `B × K_MAX`).
- **Autocast must wrap the gather.** Run `gather`/`masked_fill`/`softmax` *inside* the autocast region so they operate on the model's output dtype (bf16/fp16). Casting to f32 happens only on the final `gpu_priors[:B] = ...` copy. Running softmax outside autocast forces an up-cast of the 14977-wide logits — exactly the tensor we're trying not to materialize at f32 — and undoes most of the win.
- **`int16` vs `uint16` action ids.** `torch.uint16` doesn't exist, so shared-mem storage is `int16`. Max action id is 14976 ≪ 32767, so the signed view is lossless and bit-reinterpretable with the worker-side `uint16` numpy view. But if `K_MAX` or action-space dimensions ever grow past 32767, this reinterpretation breaks — file an issue if `MAX_ACTION_SIZE` changes.
- **Dirichlet alpha scaling.** Old code computed `alpha = numerator / len(root.priors)`. That still works against the new sparse `priors` array — `len(priors)` is the legal-action count, which is exactly what we want.
- **Subtree reuse correctness.** `_reset_root_for_reuse` rebuilds `value_sums` from `default_value`; that path doesn't touch state encoding so it should survive the dtype flip. But verify in the smoke test that virtual backups still produce sensible Q values after move 1.
- **Evaluator API churn.** Changing `evaluate_leaves` signature breaks anything that imports it. Today only `mcts/search.py` calls it. Verify with `grep evaluate_leaves` after the rewrite.

## Cleanup checklist

After the rewrite, these should be gone everywhere (`grep` to confirm):

- `rotate_visible_state`, `unrotate_values`, `rotate_visible_state_into`
- `pending_mask` (replaced by `pending_action_ids` + `pending_n`)
- `get_valid_action_mask` (function never existed in the new actions module — old MCTS used it; check it's not imported anywhere)
- `LayoutInfo`, `_get_layout_uncached` import in `mcts/evaluator.py`
- Any reference to `visible_size` outside engine internals
- `np.roll(...)` in self_play (was the un-rotation step)
- `legal_masks` array on `GameRecord` and `ReplayBuffer`
- `masked_softmax`, `_masked_softmax`, `apply_mask_softmax` — server-side GPU softmax + sparse return kill all of these.
- `_logits` shared buffer in `train/eval_server.py` (replaced by `_priors`), and `get_output_logits_np` accessor (replaced by `get_output_priors_np`). `MAX_ACTION_SIZE` still lives in `nn/transformer.py` for the dense model head but **must not appear in any `SharedEvalBuffers` shape or pinned-buffer shape**.
