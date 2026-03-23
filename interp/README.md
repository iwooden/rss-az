# Interpretability Analysis Guide

How to analyze a trained checkpoint. All commands assume Cython extensions are built and the venv is available.

## Quick Start

Run the full analysis pipeline on the latest checkpoint with 10 self-play games:

```bash
# 1. Collect states (shared across analyses)
.venv/bin/python -m interp.full_ablation --num-games 10 --save-data interp/data/states.npz --no-open

# 2. Reuse collected states for remaining analyses
.venv/bin/python -m interp.norm_check --load-data interp/data/states.npz
.venv/bin/python -m interp.decision_attr --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.acig --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.arch_analysis --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.probing --load-data interp/data/states.npz
.venv/bin/python -m interp.tb_summary
```

All scripts auto-detect the latest checkpoint. Use `--checkpoint path/to/file.pt` to specify one.

---

## 1. Feature Ablation (`full_ablation.py`)

**Question:** Which input features does the model rely on, and in which phases?

**Method:** Zeros out each feature group and measures both policy KL divergence and value MSE. Higher KL/MSE = the model's output changes more = more reliant on that feature. Separate heatmaps for policy and value heads show which features each head depends on.

**Output:** Markdown table (policy KL only) + HTML heatmap with separate policy/value tables. Both written automatically.

```bash
.venv/bin/python -m interp.full_ablation --num-games 10 --save-data interp/data/states.npz
# Opens HTML heatmap in browser. Use --no-open to suppress.
# Writes: interp/data/sensitivity_epoch<N>.md and .html
```

**What to look for:**
- **Top features by Total KL** — what the model cares about most overall
- **Phase-specific spikes** — features that only matter in their relevant phase (e.g., `turn:dividend` only in DIV). These validate that the model has learned phase-appropriate reasoning
- **Surprisingly high/low features** — features near zero (like `fi:income`) are essentially unused
- **Context-dependent fields** — `turn:active_company`, `turn:auction`, etc. should spike only in their relevant phases and be near-zero elsewhere

**Key options:**
- `--num-games N` — more games = more stable estimates (default 5, use 10-20 for publication)
- `--save-data` / `--load-data` — save states for reuse across analyses
- `--output path.md` — custom output path (HTML goes alongside)

---

## 2. Decision Attribution (`decision_attr.py`)

**Question:** What features drive the model's decisions at critical moments? Where is the policy head confused?

**Method:** Identifies high-uncertainty states (top-2 actions close in probability), then uses Captum IntegratedGradients to attribute per-feature importance for each candidate action. The differential attribution (action A minus action B) shows which features tip the model toward one choice over another.

```bash
.venv/bin/python -m interp.decision_attr --load-data interp/data/states.npz
# Writes: interp/data/decisions_epoch<N>.html
# NOTE: ~0.5s per decision on GPU. Default 15 decisions ≈ 8s.
```

**What to look for:**
- **Top decisive features per phase** — do they match game intuition? (e.g., `player:cash` should matter for buy decisions, `turn:dividend` for dividend choices)
- **Irrelevant features with high attribution** — suggests spurious correlations the model has learned
- **Missing features** — if `corp:cash` never appears in acquisition decisions, the model hasn't learned that corp cash constrains acquisitions
- **Phase distribution of critical states** — which phases produce the most uncertainty? Those are where policy capacity is most needed

**Key options:**
- `--top-k N` — number of decisions to analyze (default 15)
- `--margin-threshold F` — max probability gap between top-2 actions (default 0.15)
- `--n-steps N` — IntegratedGradients integration steps (default 50, higher = more precise)

---

## 2b. Action-Conditioned IG (`acig.py`)

**Question:** What features drive each *type* of action? Where ablation groups by phase and decision attribution examines individual uncertain states, ACIG groups by the action the model actually chose — revealing the conditional logic within each phase.

**Method:** Classifies all states by their argmax action, groups into action-type buckets, samples from each bucket, then runs IntegratedGradients targeting the chosen action's logit. Signed means are preserved (positive = feature pushes toward this action). A "discriminative features" table highlights features with highest variance across action types — these are the conditional gates.

**Output:** Console summary + HTML report with per-phase heatmaps and discriminative feature tables.

```bash
.venv/bin/python -m interp.acig --load-data interp/data/states.npz
# Writes: interp/data/acig_epoch<N>.html
# ~3s on GPU with default settings (50 samples/bucket, 30 IG steps)
```

**What to look for:**
- **Action-specific feature spikes** — `player:cash` pushing toward sell but away from auction confirms the model gates on affordability
- **Discriminative features** (high std across action types) — these are the features the policy head uses to distinguish actions. If `co:adj_incomes` discriminates buy vs sell, the model is correctly reasoning about income potential
- **Surprising feature usage** — a feature with high attribution for an unexpected action type suggests learned correlations worth investigating
- **Phase-specific sub-groups** — dividends are split into low/mid/high buckets to reveal how dividend magnitude decisions differ

**Key options:**
- `--samples-per-bucket N` — states sampled per action bucket (default 50, more = stabler)
- `--n-steps N` — IG integration steps (default 30, lower than decision_attr since averaging)
- `--min-bucket-size N` — skip buckets with fewer states (default 5)

---

## 3. Architecture Analysis (`arch_analysis.py`)

**Question:** Is the model using its depth and width efficiently? Should we add/remove blocks? Are the heads sized correctly?

**Method:** Four analyses:
1. **Block contribution** — how much each residual block changes the representation (||residual|| / ||input||)
2. **Trunk block conductance** (optional) — Captum-based measurement of each trunk block's importance to policy/value heads
3. **Head-layer conductance** (optional) — Captum conductance within each head's and input preprocessing's Linear layers, showing which layer does the most work
4. **Effective rank** (SVD) — dimensionality utilization at each layer, including head layers and optionally per-sublayer within each residual block

**Output:** Console summary + HTML report with bar charts.

```bash
# Standard analysis (~3s on GPU):
.venv/bin/python -m interp.arch_analysis --load-data interp/data/states.npz

# With detailed block internals (norm/fc1/fc2 SVD per block):
.venv/bin/python -m interp.arch_analysis --load-data interp/data/states.npz --block-detail
```

**What to look for:**

*Block contribution:*
- **Flat profile** (all blocks ~same ratio) — all blocks are active, model may benefit from more depth
- **Declining profile** (later blocks near zero) — later blocks are dead weight, consider removing them
- **Single dominant block** — usually block 0, indicates the input projection is a bottleneck

*Effective rank:*
- **Rank still growing at the last block** — model wants more depth (hasn't finished computing)
- **Rank plateau** — adding more blocks past the plateau point won't help
- **Utilization %** — effective rank / layer width. <30% means the width is wasteful, >70% means well-utilized
- **Head layer utilization** — if a head's expansion layer (e.g. 384→768) has low utilization, adding depth is more useful than adding width

*Trunk block conductance:*
- **Different top-3 blocks for policy vs value** — the heads use different parts of the network, blocks may specialize
- **Conductance concentrated in early blocks** — later blocks contribute little to the heads

*Head-layer conductance:*
- **Uneven split** — if the first layer does >50% of the work, it's the bottleneck. Consider adding depth (more nonlinear transforms) rather than width
- **Policy vs value balance** — the value head should be evenly split (simple task); the policy head may be imbalanced (complex task)

**Key options:**
- `--skip-heads` — skip head-specific analyses (head conductance + head SVD)
- `--block-detail` — break down each residual block into norm/fc1/fc2 in the SVD table

**Reference results (v2, epoch 50, 6 blocks, hidden_dim=384):**
- Block contribution: flat 0.07-0.11, all blocks active
- Effective rank: 160 → 218, still growing at block 5
- Width utilization: 214/384 = 55.8%
- Conclusion: model can use more depth → increased to 10 blocks

---

## 4. Probing Classifiers (`probing.py`)

**Question:** Where in the network does the model understand various game concepts? Which blocks serve value vs policy?

**Method:** Trains linear probes (logistic regression / ridge regression) on intermediate activations at each layer to predict game-relevant quantities. If a probe at block 2 predicts as well as at block 5, the later blocks aren't contributing to that type of understanding.

**Output:** Console summary + HTML report with separate tables for trunk, policy head, and value head layers.

```bash
.venv/bin/python -m interp.probing --load-data interp/data/states.npz
# Writes: interp/data/probing_epoch<N>.md and .html
# NOTE: Slow — trains 180+ probes. Allow ~8-10 minutes for 4-5K states.

# Heads-only mode — probes only policy/value head layers (much faster):
.venv/bin/python -m interp.probing --load-data interp/data/states.npz --heads-only
```

**Probe categories:**
- `--probes sanity` — phase, game_progress (should be perfect from input layer)
- `--probes game` — winning_player, lead_margin, num_active_corps, etc. (require cross-entity reasoning)
- `--probes policy` — action_type, invest_action, model_top_action (policy behavior)
- `--probes value` — model_value_p0, model_entropy (value behavior)
- `--probes all` — everything (default)

**Nonlinear comparison** — tests whether policy info is in the trunk but nonlinearly encoded:
```bash
.venv/bin/python -m interp.probing --load-data interp/data/states.npz --nonlinear
```
Note: requires ~10K+ states for reliable MLP probes. With <3K states the MLPs overfit and results are unreliable.

**What to look for:**

*Value probes (model_value_p0):*
- **R² climbing through blocks** — value computation happens progressively, more depth helps value
- **R² flat after block N** — value computation is done by block N, blocks after that serve policy
- **High R² at trunk (>0.95)** — value is almost a linear function of the trunk; value head doesn't need to be deep

*Policy probes (action_type, invest_action, model_top_action):*
- **Accuracy improving through blocks** — trunk depth helps policy
- **Accuracy flat or declining** — trunk doesn't help policy; the policy head's nonlinear layers are doing the work. Consider a deeper/wider policy head instead of a deeper trunk
- **action_type vs model_top_action gap** — if action_type (broad category) is well-predicted but exact action isn't, the trunk knows the strategy but specifics require nonlinear computation

*Game state probes:*
- **Declining from input to trunk** — expected. The model transforms raw features into abstract representations. Raw features get harder to linearly decode
- **Already high at input** — the input preprocessing layer handles this concept easily (e.g., who's winning)

**Reference results (v2, epoch 50, 6 blocks):**
- `model_value_p0`: R² 0.89 → 0.97 (value climbs through all blocks)
- `invest_action`: 0.80 → 0.80 (flat — policy doesn't benefit from depth)
- Conclusion: trunk serves value; deeper policy head needed → expanded to 2 hidden layers

---

## 5. Tensorboard Summary (`tb_summary.py`)

**Question:** How is training progressing? What are the key training metrics?

```bash
.venv/bin/python -m interp.tb_summary
.venv/bin/python -m interp.tb_summary --max-rows 30  # more detail
```

**What to look for:**
- **Policy entropy** — should decrease over training but not collapse to near-zero (that kills MCTS exploration). Healthy range: 0.3-0.8 after 50 epochs
- **Top-1 visit fraction** — complement of entropy. If >95%, the model is too confident and MCTS can't explore alternatives
- **Value loss** — should decrease steadily. If it plateaus while policy loss still drops, the value head may need more capacity
- **Game length** — if games get longer over training, the model is learning to play more defensively (not necessarily bad)
- **Net worth spread** (1st vs 3rd) — wider spread = model has learned to differentiate player outcomes

Uses `train.tb_reader.read_tb_scalars()` which properly merges multiple event files from training restarts.

---

## 6. Normalization Check (`norm_check.py`)

**Question:** Are state vector values well-normalized? Which features exceed [-1, +1] or are too sparse to be useful?

**Method:** Computes per-feature-group statistics (min, max, zero fraction, out-of-range counts) from collected game states. Flags features with values outside [-1, +1] (divisor too small) or excessive sparsity (signal too weak for learning).

**Output:** Console summary + HTML report with sortable overview, out-of-range, and sparsity tables.

```bash
.venv/bin/python -m interp.norm_check --load-data interp/data/states.npz
.venv/bin/python -m interp.norm_check --load-data interp/data/states.npz --feature invest:buy_impact
# Writes: interp/data/norm_epoch<N>.html. Use --no-open to suppress browser.
```

**What to look for:**
- **Out-of-range features** — values >1.0 suggest the normalization divisor is too small. Mild overflow (1.0–1.5) is acceptable; >2.0 may hurt training
- **Sparse features** (>95% zero) — context-dependent features (e.g., `turn:dividend`) are expected to be sparse. Non-context features with high sparsity may indicate a weak signal
- **Feature detail** (`--feature name`) — per-sub-feature and per-phase breakdown. Useful for diagnosing why a feature is sparse or out of range

---

## Shared Utilities (`utils.py`)

All scripts share common infrastructure:

- **`load_model()`** — loads latest checkpoint (or specified path), returns (model, config, device, epoch)
- **`collect_states()`** — plays fast games via policy sampling (no MCTS) to collect diverse states. Returns an `InterpDataset`
- **`InterpDataset`** — states, legal_masks, phases, active_players. Saveable to `.npz` for reuse

**State collection tip:** Collect once with `--save-data`, then `--load-data` for all subsequent analyses. This is much faster than re-playing games for each script.

---

## Future Work

See beads issues for planned analyses:
- **Feature interaction detection** (rss-az-ospf) — pairwise ablation to find feature combinations
- **Counterfactual sensitivity** (rss-az-9dox) — sweep individual features to plot response curves
