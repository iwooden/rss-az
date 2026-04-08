# Transformer Model Ideas

Working notes for possible future experiments in `nn/transformer.py`.

This is not a committed roadmap. These are hypotheses to try if transformer training underperforms, fails to specialize by phase, overfits, or runs into throughput issues. The goal is to preserve ideas discussed during the refactor so they do not get lost.

## Current Baseline

Current prototype assumptions:

- 3-player only for the first real training runs.
- Entity-token transformer trunk with type-specific input projections.
- Raw token features are expected to include explicit player/corp/company identity one-hots.
- The global token is expected to include a phase one-hot.
- Unified ACQ policy head uses a corp/company pair feature and outputs 52 price-or-FI-buy logits per pair.
- `ACQ_OFFER` stays as a separate 2-action phase.
- Sparse replay / sparse evaluator / sparse training changes are planned separately in [sparse-refactor.md](/home/icebreaker/rss-az-cython2/sparse-refactor.md).

## High-Priority Experiments

### 1. Explicit Phase Embeddings

Idea:
- Add `phase_embed = nn.Embedding(NUM_PHASES, d_model)`.
- Broadcast the selected phase embedding to every token after input projection.

Sketch:

```python
tokens = self._project_tokens(x)
tokens = tokens + self.phase_embed(phase_ids)[:, None, :]
```

Why try it:
- The current plan assumes the global token carries phase one-hot features.
- That should be enough for the trunk to learn phase specialization, but it is indirect.
- A learned phase embedding broadcast to all tokens is a stronger inductive bias and may make phase specialization easier.

When to try it:
- The model appears to confuse phases.
- Shared heads like the pass head behave inconsistently across phases.
- Attention maps show weak use of the global token for phase context.

Alternatives:
- Add the phase embedding only to the global token.
- Add the phase embedding to the global token and pass token only.

### 2. Learned Identity Embeddings

Idea:
- Instead of relying only on raw one-hot identity fields, add learned embeddings for player, corp, and company IDs.

Sketch:

```python
token = token + player_id_embed[player_id]
token = token + corp_id_embed[corp_id]
token = token + company_id_embed[company_id]
```

Why try it:
- One-hot identity features are likely enough for correctness.
- Learned ID embeddings may be a cleaner and more parameter-efficient way to let the model represent asymmetric entity behavior.
- This matters most for corps because corp abilities are not symmetric.

When to try it:
- Training is stable but underpowered.
- The model struggles to differentiate corp-specific behavior.
- Input feature width becomes annoying as the token schema grows.

Notes:
- It is reasonable to start with one-hot IDs in the raw token data and add learned ID embeddings only if needed.

### 3. ACQ Head Variants

Current dense prototype:
- Build a shared `(corp, company)` feature from `[corp_h, comp_h, corp_h * comp_h]`.
- Read 52 price/FI-buy logits from that pair feature.

This is a good default dense head, but not the only reasonable choice.

#### Pair-feature head

Sketch:

```python
corp_h = Wc(corp_token)
comp_h = Wt(company_token)
pair_h = MLP([corp_h, comp_h, corp_h * comp_h])
logits = Wp(pair_h)  # 52 logits
```

Why it is appealing:
- Keeps unary information from corp and company.
- Includes an explicit interaction term.
- Still small enough to be cheap.

Possible variants:
- Use `[corp_h, comp_h]` only.
- Use `[corp_h, comp_h, corp_h - comp_h, corp_h * comp_h]`.
- Give `FI buy` its own small head instead of folding it into the 52-way output.

#### Trilinear head

Sketch:

```python
q_c = Wc(corp_token)       # (dk,)
k_t = Wt(company_token)    # (dk,)
e_p = price_embed[p]       # (dk,)
score(c, t, p) = sum_r q_c[r] * k_t[r] * e_p[r]
```

Why try it:
- Much smaller than the pair-feature head.
- Strong factorized inductive bias.
- Naturally matches future sparse candidate scoring.

Why not default to it:
- Less expressive.
- Uses one shared price representation for every corp/company pair.

When to try it:
- Dense ACQ head is still too large.
- ACQ overfits.
- Sparse candidate scoring work needs a cleaner factorized baseline.

## Capacity / Architecture Experiments

### 4. Head Count vs Head Width

Current concern:
- Very small attention head dimension can split representation space too aggressively.

Things to compare:
- `d_model=128, num_heads=2` gives `64` dims/head.
- `d_model=128, num_heads=4` gives `32` dims/head.
- `d_model=192, num_heads=3` gives `64` dims/head.

Why try it:
- The best head count for this model is not obvious.
- Token count is small and structured, so fewer wider heads may work better than many narrow heads.

What to watch:
- Eval-server throughput.
- Early policy loss and value loss.
- Whether deeper layers learn obviously phase/entity-structured attention patterns.

### 5. Wider `d_model`

Idea:
- Increase `d_model` once the architecture is basically correct.

Likely candidates:
- `128 -> 192`
- `128 -> 256`

Why try it:
- The current prototype is still comfortably below the old residual MLP parameter budget.
- Wider token representations may help multi-entity reasoning more than adding narrow attention heads.

What to watch:
- Self-play throughput is the real wall-clock bottleneck, not SGD.
- A larger transformer can easily be slower than a larger MLP at inference even if parameter counts are similar.

### 6. FFN Width and Alignment

Idea:
- Prefer cleaner FFN widths instead of awkward rounded values.

Examples:
- `ff_mult = 3.0` gives `d_ff = 384` for `d_model = 128`.
- Clean widths may map better to GPU kernels than awkward values like `342`.

Why try it:
- This is a cheap knob.
- It changes model capacity without touching the action heads.

What to watch:
- Throughput changes during eval inference.
- Whether wider FFNs help more than wider attention.

## Token-Interface Experiments

### 7. Remove Leftover Context Tokens

Question:
- Once the new heads stabilize, do we still want dedicated tokens like `PAR` and `Acq Price`, or were those mostly transitional scaffolding?

Why this matters:
- Extra tokens are not free.
- If a token has no clear semantic job, it adds bookkeeping and potential confusion.

Current view:
- A token is worth keeping if it carries real reusable context for the trunk.
- A token is not worth keeping if it only exists because the old action factoring needed it.

When to revisit:
- After the actual token feature spec is written.
- After sparse-policy plumbing is in place.

### 8. Broadcast Phase Features to All Tokens in Raw Input

Alternative to learned phase embeddings:
- Put the phase one-hot directly into every raw token feature vector, not just the global token.

Why try it:
- Very explicit.
- Makes phase context available from layer 0 without requiring attention to the global token.

Why not default to it:
- Increases raw token width.
- Feels less elegant than a learned phase embedding.
- Probably unnecessary if the global token plus attention already works.

## Performance-Focused Ideas

### 9. Replace `nn.MultiheadAttention` With `scaled_dot_product_attention` Plumbing

Why consider it:
- The current block uses `nn.MultiheadAttention`, which is fine for the prototype.
- If inference throughput becomes a problem, a custom attention path can give tighter control over kernels, shapes, and cached projections.

Why not now:
- More code complexity.
- Premature before profiling on real self-play workloads.

### 10. Torch Compile / Kernel Benchmarking

Idea:
- Benchmark the transformer under realistic eval-server settings before making architecture calls based on intuition alone.

Important context:
- This repo cares far more about self-play eval throughput than raw SGD speed.
- Small shape changes that barely matter in training can matter a lot in eval-server hot loops.

Things worth benchmarking directly:
- `num_heads=2` vs `4`
- `d_model=128` vs `192`
- `ff_mult=3.0` vs smaller widths
- phase one-hot on global token only vs learned phase embedding

## Interpretation / Debugging Ideas

### 11. Attention Inspection for Phase Use

Question:
- Does the model actually use the global token for phase information?

What to inspect:
- How often non-global tokens attend to the global token.
- Whether those attention patterns change by phase.
- Whether pass-token behavior becomes phase-specific.

Use this if:
- Training quality is weak and the suspicion is poor phase specialization.

### 12. Token Ablation Tests

Idea:
- Zero out selected token types or feature groups at eval time.

Useful ablations:
- global token
- pass token
- entity ID features
- company synergy features
- corp ability / identity features

Why try it:
- Helps separate “the model can in theory use this signal” from “the model actually uses this signal.”

## Suggested Order If Training Struggles

1. Verify the token feature spec is actually correct, especially phase and identity fields.
2. Add instrumentation before changing architecture.
3. Try explicit phase embeddings if phase confusion is suspected.
4. Try head-count / width changes (`128/2`, `192/3`, `192/4`) if capacity looks wrong.
5. Revisit the ACQ head if ACQ policy quality is the clear weak point.
6. Only then consider deeper implementation work like custom attention kernels.

## Non-Goals For Now

- Do not mix these experiments with the sparse-policy refactor unless there is a clear reason.
- Do not assume any of these are improvements without profiling and training data.
- Do not overfit the architecture to smoke-test cleanliness; the real gate is self-play training behavior.
