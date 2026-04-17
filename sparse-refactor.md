# Sparse Policy / Replay Refactor — Historical Design Notes

> **Status (2026-04): mostly landed.** Phase-local action ids, sparse MCTS
> plumbing, sparse replay, sparse eval-server IPC, and sparse trainer loss
> have all shipped. The only unlanded item is candidate-direct scoring inside
> the ACQUISITION head, and the current thinking is that it isn't worth
> pursuing at this model scale — see the ACQUISITION section below. This
> document survives as a reference for the *why*, not as a work queue.

## Current state (pointers to authoritative code)

| Piece | State | Code |
|-------|-------|------|
| Phase-local action ids, `enumerate_legal_actions` | ✅ landed | `core/actions.{pyx,pxd}` |
| Phase dispatch in driver | ✅ landed | `core/driver.pyx` |
| Sparse node expansion (`action_ids`, `priors`) | ✅ landed | `mcts/node.py`, `mcts/mcts_core.pyx` |
| Sparse evaluator contract | ✅ landed | `mcts/evaluator.py` (returns `(priors[:n_legal], values, action_ids, n_legal, phase_id)`) |
| Sparse eval-server shm protocol | ✅ landed | `train/eval_server.py` (worker→server `action_ids (W,B,K_MAX)`, server→worker softmaxed `priors (W,B,K_MAX)`) |
| Sparse replay buffer | ✅ landed | `train/replay_buffer.py` — `TrainingExample(action_ids, policy_target)` padded to K_MAX |
| Sparse self-play export | ✅ landed | `train/self_play.py` |
| Trainer: phase-local loss, gather + softmax over legal only | ✅ landed | `train/trainer.py` (batch by phase via `phase_indices`) |
| Dense heads for small phases + per-row gather in model | ✅ landed | `nn/transformer.py::_policy_forward` |
| ACQ candidate-direct scoring inside head | ❌ not pursued | current head is per-offset low-rank bilinear; rationale below |

## Why this document exists (preserved for context)

The refactor was motivated by an earlier MLP-era assumption that policy outputs,
legal masks, replay targets, and eval-server IPC all lived on a single dense
`action_dim` vector. That assumption predated phase-local action ids and the
transformer entity-readout design.

The clearest pressure point was full-joint ACQUISITION. A direct action space of

`8 corps * 36 companies * 51 price offsets = 14,688`

is feasible **inside the model**, but forcing it through dense fixed-width
runtime interfaces was wasteful. The expensive parts were:

- dense logits over eval-server IPC
- dense legal masks
- dense replay `policy_target`
- dense replay `legal_mask`
- dense trainer-side `log_softmax` over large padded action vectors

All five are now gone: IPC, replay, and trainer loss all travel on `(K_MAX,)`
phase-local legal lists. The engine enumerates, the model scores (dense per
phase, then gathered to the legal slice inside `_policy_forward`), MCTS stores
priors only for those legal actions, replay stores the same. Training loss is
softmax over the gathered legal set, which is mathematically equivalent to
dense `-inf` masking but avoids the dense intermediate outside the model.


## Goals

- Remove the need for a global dense `action_dim` policy interface.
- Support full-joint ACQUISITION without blowing up IPC or replay.
- Use one replay format for **all** phases.
- Preserve exact AlphaZero behavior: policy is normalized over the legal action set.
- Keep MCTS operating only over legal actions.
- Prefer a simple first implementation over an aggressively optimized one,
  since epoch wallclock is dominated by self-play generation rather than SGD.


## Non-goals

- Teaching the policy head legality via invalid-action gradients.
- Learning from the full theoretical action universe in every phase.
- Designing the final mixed-player-count batching story.
- Over-optimizing trainer throughput before self-play is no longer the bottleneck.


## Core idea

The action space should be represented as:

- `phase_id`
- `num_legal`
- `action_ids[0:num_legal]`
- `policy_targets[0:num_legal]`

where `action_ids` are **phase-local** ids, not global ids.

Important distinction:

- The **action universe** for a phase may be large.
- The **legal set** for a specific state is much smaller.

MCTS, replay, and training only need the legal set.

For ACQUISITION specifically:

- The theoretical universe can be full-joint `(corp, company, price)`.
- The runtime interface only needs to score and store the legal subset for the current state.

If the practical upper bound on legal actions is around `Kmax ~= 200`, then the
problem becomes much easier. We can build the system around variable-length or
padded-to-`Kmax` legal candidate lists rather than dense 14,688-wide vectors.


## Recommended v1: padded sparse everywhere

For an initial implementation, use **padded sparse** arrays rather than a fully
ragged arena/pool design.

Recommended shape:

- `phase_id`: scalar per example
- `legal_count`: scalar per example
- `action_ids`: shape `(Kmax,)`, padded
- `policy_targets`: shape `(Kmax,)`, padded
- `value_target`: shape `(num_players,)`

Recommended initial `Kmax`:

- start with `256`
- add overflow logging
- add a safe fallback path if exceeded

Why this is the right first step:

- It removes dense replay/mask waste immediately.
- It keeps batching simple.
- It keeps trainer code straightforward.
- It is fast enough for current wallclock constraints.
- It avoids building two different pipelines for "small phases" and "ACQ".

A more complex ragged arena format can be added later if profiling shows this
is worth it.


## Action-id design

### Phase-local ids

Every phase gets its own id space. Examples:

- INVEST:
  - pass
  - auction `(company_id, bid_offset)`
  - buy share `(corp_id)`
  - sell share `(corp_id)`
- BID:
  - leave
  - raise `(offset)`
- CLOSING:
  - pass
  - close `(company_id)` — direct company selection, no offer-buffer indirection
- IPO:
  - pass
  - `(corp_id, par_index)`
- ACQ:
  - pass
  - `(corp_id, company_id, price_offset_or_fi_buy)` — the pair space includes the
    fixed-price FI purchase as the 52nd option per `(corp, company)` pair. No
    offer-buffer indirection: the engine exposes the full tuple space and the
    acting player picks directly.
- ACQ_OFFER:
  - pass
  - buy — a first-class 2-action decision phase for FI-priority resolution.
    Entered whenever a player or receivership corp attempts to acquire an
    FI-owned company AND one or more higher-priority corps exist (OS first at
    face value; remaining corps ordered by descending share price at high
    value). Each higher-priority corp is offered the chance to preempt, with
    that corp's president as the active player. The contested corp and company
    are carried in the existing `turn.active_corp` / `turn.active_company`
    slots; no new state fields are needed.

The state phase disambiguates the meaning of the local id. No global action
index is required.

### Encoding

Use compact integer packing for each phase.

For ACQ:

```text
0                                     -> pass
1 + ((corp * 36 + company) * 52 + k)  -> acquisition tuple
    where k < 51  -> low_price + k  (normal price offsets)
    where k == 51 -> ACQ_FI_BUY     (fixed-price FI purchase; OS=face, others=high)
```

This fits comfortably in `uint16`:

- `1 + 8 * 36 * 52 = 14,977`

FI purchases are folded into the same pair space rather than a separate
sub-phase — the old offer-buffer indirection is gone, so the engine exposes
every legal `(corp, company, price-or-fi-buy)` tuple as a masked action. The
separate `ACQ_OFFER` phase handles only FI *priority* resolution (offering a
higher-priority corp the chance to preempt), not the fixed-price FI buy itself.

For most other phases, `uint16` is also enough.

### Decode / encode rules

`core/actions.pyx` should become phase-local:

- `enumerate_legal_actions(state) -> (phase_id, action_ids, count)`
- `decode_action(phase_id, action_id) -> ActionInfo`
- `apply_action(state, action_id)` can interpret `action_id` relative to `state.get_phase()`

Within one node, children are keyed by local action id only, which is already sufficient.


## Model-side design

Replay/storage should be sparse for all phases.

Model implementation does **not** need to be uniform across phases.

That is the key simplification.

### Small phases

For smaller phases, keep the current style of dense per-phase heads:

- BID: 15 logits
- ACQ_OFFER: 2 logits
- ISSUE: 2 logits
- DIVIDENDS: 26 logits
- CLOSING: 37 logits
- IPO: 113 logits
- INVEST: 557 logits

During training and inference:

- model produces dense per-phase logits
- code gathers only the logits referenced by `action_ids`
- softmax/loss is computed over those gathered legal logits only

This preserves simplicity where the dense head is already small and natural.

### ACQUISITION — what actually landed, and why candidate-direct wasn't pursued

The ACQ head currently materializes the full `(B, 8, 36, 52)` logit tensor,
then `_policy_forward` gathers the legal slice and hands `(B, K_MAX)` back to
the caller. Nothing dense leaves the model.

The original design sketch proposed going further — score only the legal
`(corp, company, offset)` triples directly, never materializing the dense
tensor. That is **not** what shipped, and on current evidence isn't worth
shipping. The honest accounting:

- The dense `(B, 8, 36, 52)` transient is ~15 MB at B=256 (fp32). Nothing on a
  16 GB GPU cares about 15 MB.
- Head FLOPs are <1% of a training step — the trunk (10 layers, d=128) swamps
  the head by two orders of magnitude. Making ~1% of compute smaller buys
  nothing measurable.
- Candidate-direct scoring on GPU is *not* free: per-row gathers into
  per-offset factors break memory coalescence, warp divergence grows with
  per-row legal-count variation, and kernel-launch overhead dominates at the
  small batch sizes used in self-play eval (B=1-16). The factored dense path
  runs as two fused GEMMs + one contraction under Inductor; candidate-direct
  would very likely be neutral or slower.
- The data-pipeline wins the sparse design promised (replay storage, IPC
  bandwidth, avoiding dense `-inf` masks in the trainer) are already banked
  by the current architecture — the dense tensor exists only as a microsecond
  transient between the head's last matmul and the in-model `.gather()`.

What the head currently does: per-offset low-rank bilinear +
additive unary paths, implemented as three einsums:

```text
score(c, t, p) = (corp_h[c] @ U[p]) · (comp_h[t] @ V[p])          # bilinear, rank-r per offset
               + W_corp[p] · corp_h[c] + W_comp[p] · comp_h[t]    # unary fallback
```

with `U, V: (52, d_model, r)` and per-offset bias linears. See
`nn/transformer.py::_policy_acquisition` and `TransformerConfig.acq_rank`.
This replaced the older `cat([corp_h, comp_h, corp_h*comp_h]) → MLP → 52` head,
cutting peak activation roughly in half while giving each offset its own
rank-r interaction pattern instead of sharing a single GELU bottleneck.

If a future model scale (d_model=512, dk=256, bigger batches, or a larger
combinatorial phase) ever makes the head's dense intermediate actually
matter, revisit this section — the candidate-direct math is sketched below
for that eventuality.

**Sketched alternatives (not implemented):**

1. Low-rank trilinear scoring: `score(c, t, p) = sum_r q_c[r] * k_t[r] * e_p[r]`.
   Smallest parameter count, easy vectorization, preserves a natural
   factorization. Less expressive than the current bilinear form — each offset
   gets a rank-1 interaction instead of rank-r.
2. Candidate-feature MLP: gather `corp_token[c]`, `company_token[t]`,
   `price_embedding[p]`, fuse, small MLP → scalar logit. Most flexible, but
   worst on the gather-coalescence/divergence axes above.

### Important modeling note

This refactor does **not** train invalid actions.

That is correct.

AlphaZero-style masked training is a softmax over the legal set. Sparse legal-only
training is mathematically equivalent to dense training with invalid logits set to `-inf`.

If we ever want the model to learn legality explicitly, that should be a separate
auxiliary objective, not an accidental side effect of policy masking.


## Eval-server / worker interface

The current interface sends states to the server and returns dense logits.

That should change to legal-candidate scoring.

### Worker -> server

Each request should contain:

- state or token buffer
- `phase_id`
- `legal_count`
- `action_ids[0:legal_count]`

In the padded v1 design:

- `phase_id`: shape `(batch,)`
- `legal_count`: shape `(batch,)`
- `action_ids`: shape `(batch, Kmax)`

### Server -> worker

Return:

- `priors`: shape `(batch, Kmax)`
- `values`: shape `(batch, num_players)`

The returned `priors` should already be normalized over the legal set.

This means:

- no dense logits over IPC
- no worker-side legal masking
- no worker-side softmax
- no need to ship dense legal masks at all

### Why server-side normalization is a good fit

The worker already knows the legal list and count. The server can score those
candidates and normalize over them immediately.

That makes the server output match exactly what MCTS needs:

- candidate ids
- candidate priors
- values

This is simpler than returning raw logits and redoing sparse softmax on the worker.


## Search / MCTS changes

The good news is that the tree is already conceptually sparse.

Current nodes already store:

- legal action ids
- priors for those action ids
- visit counts for those action ids
- value sums for those action ids

What changes is how those arrays are constructed.

### Current path

- dense logits
- dense legal mask
- `expand_node()` extracts legal actions from mask

### New path

- worker enumerates legal actions up front
- evaluator returns priors aligned to that legal list
- `expand_node()` takes `action_ids` and `priors` directly

This removes:

- dense mask scans in hot paths
- dense action vectors at leaf expansion time

### Branching factor

If the legal count really stays under about 200, current PUCT selection remains fine.

No progressive widening is needed initially.

If this assumption proves false, ACQ is the first place to reconsider:

- progressive widening
- top-k preselection
- two-stage ACQ again

But that is a contingency plan, not the default.


## Replay buffer design

### Recommended v1 schema

For every training example:

- `state`
- `phase_id`
- `legal_count`
- `action_ids[Kmax]`
- `policy_targets[Kmax]`
- `value_target`

This removes `legal_mask` entirely.

The legal action list itself is the mask.

### Why sparse replay should be universal

Even though ACQ is the largest driver, sparse replay should be used for **all** phases.

Reasons:

- one replay schema
- one self-play export path
- one trainer input format
- one collate/padding story
- avoids a special ACQ-only storage exception

The model side can still be phase-specific. Replay does not need to mirror model internals.

### Storage savings

Dense ACQ replay is obviously bad.

But even for small phases, sparse replay is cleaner:

- no dense masks
- no dense zero-filled policy targets
- no dependence on the largest phase size

With padded sparse `Kmax=256`, replay remains compact even if ACQ is present.


## Self-play changes

Self-play should no longer materialize dense root policies.

Instead of:

- dense legal mask
- dense `policy_target[action_dim]`
- dense action sampling distribution

store:

- `phase_id`
- `action_ids`
- `visit_probs`

Action selection should sample from the sparse root distribution directly.

This means:

- no conversion from sparse visits to dense probabilities
- no dense root policy allocation
- cleaner data passed to replay

The MCTS root already has the information we need:

- legal action ids
- visit counts aligned with those ids


## Trainer changes

The trainer should batch by phase.

That is important for both clarity and performance.

### Small phases

For dense small heads:

- run the phase head
- gather logits by `action_ids`
- apply padding mask using `legal_count`
- compute `log_softmax` over gathered legal logits only

### ACQ

For ACQ:

- score the candidate list directly
- apply `log_softmax` over those candidate logits

### Padding strategy

Use:

- `action_ids`: `(B, Kmax)`
- `policy_targets`: `(B, Kmax)`
- `legal_count`: `(B,)`

Build a boolean padding mask:

- positions `>= legal_count` are invalid
- fill their logits with `-inf` before `log_softmax`

This gives one stable tensor shape per phase batch and works well with compilation.


## File-by-file status

### `core/actions.pyx` and `core/actions.pxd` ✅

- ~~Remove dependence on global action layout.~~ Done.
- ~~Introduce phase-local encode/decode helpers.~~ Done.
- ~~Add `enumerate_legal_actions(state)` API.~~ Done — all 8 `_enumerate_*` helpers implemented.
- ~~Ensure returned order is deterministic.~~ Done — documented per-phase ordering.

### `core/driver.pyx` ✅

- ~~Update action application to use phase-local ids.~~ Done.
- ~~Decode relative to current phase.~~ Done — full dispatch table + forced-action auto-chain.

### `mcts/evaluator.py` ✅

- ~~Replace dense `(policy_logits, legal_mask)` contract with sparse candidate scoring.~~ Done — `evaluate*()` returns `(priors[:n_legal], values, action_ids, n_legal, phase_id)`.
- Model forward still does dense-per-phase + gather internally; evaluator sees only the sparse result.

### `train/eval_server.py` ✅

- ~~Add shared buffers for `phase_ids`, `legal_counts`, `action_ids`, sparse `priors`.~~ Done.
- ~~Remove dense output-logit buffer.~~ Done — ~46 MB shared-mem buffer gone at 96 workers.
- ~~Score and normalize legal candidates on the server.~~ Done — softmax runs on-GPU inside the server's autocast region.

### `mcts/node.py` ✅

- ~~Sparse storage of action ids, priors, visits, value sums.~~ Done — `MCTSNode.expand(action_ids, n, priors, default_value)`.

### `mcts/mcts_core.pyx` ✅

- ~~Sparse expansion entrypoint.~~ Done — `_expand_node_sparse`.
- PUCT scan unchanged.

### `mcts/search.py` ✅

- ~~Stop materializing dense root probability vectors.~~ Done.
- ~~Sample directly from sparse root visits.~~ Done.
- ~~Store sparse policy targets in self-play records.~~ Done.

### `train/replay_buffer.py` ✅

- ~~Replace dense `legal_masks` and dense `policy_targets`.~~ Done — schema is `(action_ids (uint16, K_MAX), policy_targets (float32, K_MAX), phase_ids, n_legals, state, value_targets)`.

### `train/self_play.py` ✅

- ~~Export sparse targets directly from root visit counts.~~ Done — `SelfPlayExample.policy_target` is `(n_legal,)` float32.

### `train/trainer.py` ✅

- ~~Batch by phase.~~ Done — phase dispatch flows through `phase_indices` in the model.
- ~~Gather logits for small phases.~~ Done inside `_policy_forward`.
- ~~Sparse legal-only cross-entropy.~~ Done.
- ❌ **Not pursued:** candidate-score ACQ directly — see the ACQUISITION modeling note above.

### `nn/transformer.py` ✅ (with one deliberate omission)

- ~~Remove the assumption that every phase returns a common `MAX_ACTIONS`.~~ Done — `_policy_forward` dispatches per-phase and gathers to `K_MAX` inside the model.
- ~~Keep dense heads for small phases.~~ Done — all 7 small-phase heads are dense-then-gather.
- ❌ **Deliberately not added:** candidate-scoring path for ACQ. Current head is per-offset low-rank bilinear + unary paths, dense `(B, 8, 36, 52)` transient collapsed by the in-model `.gather()`. See ACQUISITION modeling note for rationale.

### Tests

- Add encode/decode roundtrip tests for phase-local ids.
- Add legal enumeration tests with deterministic order assertions.
- Add replay save/load tests for sparse examples.
- Add trainer equivalence tests:
  - dense masked loss vs sparse legal-only loss on the same toy batch


## Determinism requirements

This refactor depends on action enumeration being deterministic.

That means:

- the legal candidate list for a state must always be returned in the same order
- replay and training must use the same action-id encoding
- search diagnostics must be able to decode sparse action ids consistently

If action order changes nondeterministically between self-play and training,
targets will silently misalign.

For safety:

- define one canonical enumeration order per phase
- test it explicitly
- do not rely on incidental dict/set iteration order


## Pitfalls and things to watch

### 1. Confusing "all possible actions" with "all legal actions"

Do not store only nonzero visit targets.

Replay must contain the **entire legal action set**, including legal actions
with target probability 0. Those zero-target legal actions still belong in the
softmax denominator and still produce gradient through normalization.

### 2. Candidate overflow

If `Kmax` is too small and legal actions exceed it, silently truncating the list
would be catastrophic.

Required safeguards:

- log max legal counts per phase
- assert or fall back to a slower dynamic path on overflow
- do not clip legal actions in production training

### 3. ACQ scorer / engine mismatch

The ACQ candidate scorer must use exactly the same encoding and semantics as
the engine's legal enumerator and decoder.

This is especially important if FI fixed-price purchases are folded into the
same `(corp, company, price)` encoding.

### 4. Graph-shape explosion

Variable-length tensors can hurt `torch.compile`.

Mitigation:

- batch by phase
- pad to fixed `Kmax`
- keep one stable tensor shape per phase batch

### 5. Server/worker complexity

The eval server becomes more than a pure "state in, dense logits out" box.

That is acceptable, but keep the contract narrow:

- worker enumerates legal actions
- server scores them
- server returns priors and values

Do not let candidate generation drift into the server. The engine should remain
the single source of truth for legality.

### 6. Debugging visibility

Dense vectors are easy to print. Sparse policies are not.

Add debugging helpers early:

- decode sparse action ids to readable strings
- pretty-print top-k priors from sparse lists
- log max legal counts per phase


## Performance guidance

### Prioritize simplicity first

Epoch wallclock is dominated by self-play, not training.

That means:

- accept a somewhat slower trainer if it yields a much cleaner sparse design
- optimize replay and eval IPC first

### Use compact dtypes

- `action_ids`: `uint16`
- `legal_count`: `uint16` or `int16`
- `phase_id`: `uint8`
- `policy_targets`: `float32` initially; consider `float16` later if acceptable

### Normalize on the server

Server-side normalization avoids:

- worker-side softmax
- dense mask application
- extra CPU work on the worker

### Batch by phase

This reduces:

- head dispatch overhead
- compile graph fragmentation
- padding waste

### Keep candidate scoring vectorized

Avoid Python loops over legal actions in model code.

For ACQ:

- gather corp/company/price factors in tensors
- score all candidates in one batched operation

### Instrument everything

At minimum, log:

- max legal count per phase
- mean legal count per phase
- 95th/99th percentile legal count per phase
- eval-server batch size
- eval-server time spent in:
  - gather
  - model forward
  - candidate scoring
  - normalization
- trainer time spent in:
  - collation
  - forward
  - gather/scoring
  - loss

This refactor should be driven by measured legal counts, not guesses.


## Rollout status

### Phase 1: action ids and legal enumeration ✅

- phase-local ids — **done** (`core/actions.{pxd,pyx}`)
- deterministic `enumerate_legal_actions()` — **done** (all 8 `_enumerate_*` helpers)
- decode/apply path updated — **done** (`core/driver.pyx` dispatches to `phases/*.pyx`)

### Phase 2: sparse MCTS plumbing ✅

- sparse evaluator contract — **done** (`mcts/evaluator.py`)
- sparse node expansion — **done** (`mcts/node.py`, `_expand_node_sparse`)
- sparse root action sampling — **done** (`mcts/search.py`)

### Phase 3: sparse replay ✅

- padded sparse replay schema — **done** (`train/replay_buffer.py`)
- self-play export changes — **done** (`train/self_play.py`)
- trainer input changes — **done** (`train/trainer.py`)

### Phase 4: transformer integration ✅ / ❌

- small dense heads + sparse gather loss — **done** (`nn/transformer.py::_policy_forward`)
- ACQ candidate scorer — **deliberately not pursued**; see ACQUISITION modeling note

### Phase 5: performance tuning (ongoing, as-needed)

- `K_MAX = MAX_LEGAL_ACTIONS_PY = 256` in `core/actions`; overflow is a loud assert.
- `policy_targets` kept at float32 — cheap and simplifies the loss path.
- eval-server scoring path is profiled via the training-loop monitoring
  (`train/main.py`, tensorboard scalars).

## Design decisions taken

This is what the shipped system actually commits to:

- phase-local action ids ✅
- padded sparse `K_MAX` everywhere outside the model ✅
- dense per-phase heads + in-model `.gather()` (not candidate-direct scoring) ✅
- server returns already-softmaxed sparse priors ✅
- engine is the single source of truth for legality ✅
- sparse replay for **all** phases, not just ACQ ✅

Net effect: a coherent end-to-end sparse design that does *not* force every
phase to share the same internal model implementation, while avoiding the
gather-heavy GPU access patterns that true candidate-direct scoring would
require.
