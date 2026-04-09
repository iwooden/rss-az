# Sparse Policy / Replay Refactor

## Why this document exists

The current engine/training stack still reflects an older MLP-era assumption:
policy outputs, legal masks, replay targets, and eval-server IPC are all built
around a single dense `action_dim` vector.

That assumption is increasingly out of date:

- We already treat phases separately at the model level.
- We no longer need globally unique action indices.
- MCTS only searches over **legal** actions, not the full theoretical action universe.
- The proposed transformer architecture can represent large combinatorial phases
  like ACQUISITION cleanly, but dense policy tensors make that awkward and wasteful.

The clearest pressure point is full-joint ACQUISITION. A direct action space of

`8 corps * 36 companies * 51 price offsets = 14,688`

is completely feasible **inside the model**, but awkward when forced through
dense fixed-width runtime interfaces. The expensive part is not the model head
itself. The expensive part is:

- dense logits over eval-server IPC
- dense legal masks
- dense replay `policy_target`
- dense replay `legal_mask`
- dense trainer-side `log_softmax` over large padded action vectors

This refactor proposes a unified sparse policy interface that matches how
search already works conceptually:

- the engine enumerates legal actions
- the model scores those legal actions
- MCTS stores priors only for those legal actions
- replay stores those legal actions plus target probabilities
- training computes loss over that legal set only

This matches standard AlphaZero masking semantics. Invalid actions are excluded
from the softmax competition set. They are **not** trained implicitly by dense
masking with `-inf`.


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

### ACQUISITION

ACQ should use direct candidate scoring rather than dense `[14,689]` output.

The model should accept the legal ACQ candidate list and score those candidates directly.

Two reasonable implementations:

1. Low-rank trilinear scoring:

```text
score(c, t, p) = sum_r q_c[r] * k_t[r] * e_p[r]
```

2. Candidate-feature MLP:

- gather `corp_token[c]`
- gather `company_token[t]`
- gather `price_embedding[p]`
- concatenate / fuse
- small MLP -> scalar logit

The trilinear form is probably the better first implementation:

- fewer parameters
- easy vectorization
- preserves a natural factorization

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


## Suggested file-by-file changes

### `core/actions.pyx` and `core/actions.pxd`

- Remove dependence on global action layout.
- Introduce phase-local encode/decode helpers.
- Add `enumerate_legal_actions(state)` API.
- Ensure returned order is deterministic.

### `core/driver.pyx`

- Update action application to use phase-local ids.
- Decode relative to current phase.

### `mcts/evaluator.py`

- Replace dense `(policy_logits, legal_mask)` contract with sparse candidate scoring.
- Evaluator should return priors aligned with the legal action list.

### `train/eval_server.py`

- Add shared buffers for:
  - `phase_ids`
  - `legal_counts`
  - `action_ids`
  - sparse `priors`
- Remove dense output-logit buffer for the sparse path.
- Score and normalize legal candidates on the server.

### `mcts/node.py`

- Keep sparse storage of action ids, priors, visits, and value sums.
- No conceptual redesign needed beyond the input contract.

### `mcts/mcts_core.pyx`

- Add `expand_node_from_sparse(action_ids, priors, default_value)`.
- Keep current PUCT scan logic.

### `mcts/search.py`

- Stop materializing dense root probability vectors.
- Sample directly from sparse root visits.
- Store sparse policy targets in self-play records.

### `train/replay_buffer.py`

- Replace dense `legal_masks` and dense `policy_targets`.
- Store sparse padded arrays plus `phase_id` and `legal_count`.

### `train/self_play.py`

- Export sparse targets directly from root visit counts.
- Stop building dense root distributions.

### `train/trainer.py`

- Batch by phase.
- Gather dense logits for small phases.
- Candidate-score ACQ directly.
- Compute sparse legal-only cross-entropy.

### `nn/transformer.py`

- Remove the assumption that every phase returns a common `MAX_ACTIONS`.
- Keep dense heads for small phases if desired.
- Add candidate-scoring path for ACQ.

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


## Suggested rollout order

### Phase 1: action ids and legal enumeration

- phase-local ids
- deterministic `enumerate_legal_actions()`
- decode/apply path updated

### Phase 2: sparse MCTS plumbing

- sparse evaluator contract
- sparse node expansion
- sparse root action sampling

### Phase 3: sparse replay

- padded sparse replay schema
- self-play export changes
- trainer input changes

### Phase 4: transformer integration

- small dense heads + sparse gather loss
- ACQ candidate scorer

### Phase 5: performance tuning

- tune `Kmax`
- decide whether `float16` policy targets are acceptable
- profile eval-server scoring path
- only then consider ragged arena storage


## Recommended decisions

If we are optimizing for the best tradeoff between clarity, risk, and performance:

- use phase-local ids
- make replay sparse for **all** phases
- keep a padded sparse `Kmax` design for v1
- keep dense heads for small phases
- use direct candidate scoring for ACQ
- let the server return normalized sparse priors
- keep the engine as the single source of truth for legal action enumeration

This gives a coherent end-to-end design without forcing every phase to share
the same internal model implementation.
