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

**Method:** Zeros out each feature group and measures both policy KL divergence and value MSE. Higher KL/MSE = the model's output changes more = more reliant on that feature. Separate heatmaps for policy and value heads show which features each head depends on. The `phase` feature is filtered from policy KL tables since it determines head routing and dominates color scaling.

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

**Method:** Five analyses:
1. **Preprocessing contribution** — signal gain/attenuation through each input preprocessing layer (||output|| / ||input||)
2. **Block contribution** — how much each residual block changes the representation (||residual|| / ||input||)
3. **Trunk block conductance** (optional) — Captum-based measurement of each trunk block's importance to policy/value heads
4. **Head-layer conductance** (optional) — Captum conductance within each per-phase policy head's, value head's, and input preprocessing's Linear layers, showing which layer does the most work. Phase heads are filtered to only use states that route to that head.
5. **Effective rank** (SVD) — dimensionality utilization at each layer, including per-phase policy head and value head layers, and optionally per-sublayer within each residual block

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
- **Per-phase variation** — each phase head has its own conductance profile. Phases with more complex action spaces (INVEST, ACQ) may show different patterns than simpler ones (CLOSE, ISSUE)
- **Policy vs value balance** — the value head should be evenly split (simple task); individual phase heads may be imbalanced (complex action spaces)

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

**Output:** Console summary + HTML report with separate tables for trunk layers, per-phase policy head layers (each phase probed independently using only matching states), and value head layers.

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
- **Per-phase head variation** — each phase head is probed independently with its own states. Some heads (e.g., INVEST with 4 action types) may show stronger probe accuracy than simpler phases

*Game state probes:*
- **Declining from input to trunk** — expected. The model transforms raw features into abstract representations. Raw features get harder to linearly decode
- **Already high at input** — the input preprocessing layer handles this concept easily (e.g., who's winning)

**Reference results (v2, epoch 50, 6 blocks, single policy head):**
- `model_value_p0`: R² 0.89 → 0.97 (value climbs through all blocks)
- `invest_action`: 0.80 → 0.80 (flat — policy doesn't benefit from depth)
- Conclusion: trunk serves value; deeper policy head needed → expanded to per-phase heads with 3 hidden layers each

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

## 7. Layer-Specific Analysis (`layers/`)

Deep analysis of individual model components, beyond what the global arch_analysis provides.

### 7a. Input Preprocessing (`layers/preprocess.py`)

**Question:** Is the 768->512->256 compression losing policy-relevant signal?

**Method:** Three analyses:
1. **Signal attenuation** — for each feature group, measures activation delta at 768-dim, 512-dim, and 256-dim when the group is ablated. Attenuation = delta_256/delta_768 (1.0 = preserved, <1.0 = lost). Cross-referenced with policy KL.
2. **Expanded probing** — linear probes at raw input, 768-dim, 512-dim, 256-dim (post-LN), and block_0. Tracks information loss through each step.
3. **SVD projection** — effective rank at each intermediate dimension. Identifies which 768-dim singular vectors the 768->512 weight matrix preserves vs discards, and correlates with feature group importance.

**Output:** Console summary + markdown report + HTML report.

```bash
.venv/bin/python -m interp.layers.preprocess --load-data interp/data/states.npz
# Writes: interp/data/preprocess_epoch<N>.md and .html
```

### 7b. Policy Heads (`layers/policy_head.py`)

**Question:** How does each per-phase policy head organize its computation? Does each head use its depth/width efficiently?

**Method:** Three analyses applied independently to each of the 8 per-phase heads:
1. **Logit lens** — projects intermediate head representations through that head's final weight matrix to get "early logits." Shows when decisions crystallize within each head.
2. **Neuron specialization** — NeuronConductance per action type reveals which neurons serve which actions within each head. Measures functional width utilization.
3. **Layer causal necessity** — replaces each Linear+GELU pair within a head with identity, measuring the causal effect on that phase's policy.

**Output:** Console summary + markdown report + HTML report with tabbed per-phase views.

```bash
.venv/bin/python -m interp.layers.policy_head --load-data interp/data/states.npz
# Writes: interp/data/policy_head_epoch<N>.md and .html

# Skip neuron specialization (slowest analysis):
.venv/bin/python -m interp.layers.policy_head --load-data interp/data/states.npz --skip-neurons

# Analyze specific phases only:
.venv/bin/python -m interp.layers.policy_head --load-data interp/data/states.npz --phases INVEST ACQ DIV
```

### 7c. Value Head (`layers/value_head.py`)

**Question:** Does V0 add meaningful computation? Are neurons specialized by player? Where is value prediction weakest?

**Method:** Four analyses:
1. **Value lens** — projects trunk directly through V2 (skipping V0+GELU). Compares to full head output to measure V0's contribution.
2. **Per-player neuron specialization** — NeuronConductance at V0 GELU toward each player's value output. Shows whether neurons specialize by player.
3. **Phase-stratified value characteristics** — value magnitude, spread, and per-player means by phase and game progress (early/mid/late).
4. **Layer causal necessity** — bypasses V0+GELU, measures value MSE change per phase.

**Output:** Console summary + markdown report + HTML report.

```bash
.venv/bin/python -m interp.layers.value_head --load-data interp/data/states.npz
# Writes: interp/data/value_head_epoch<N>.md and .html
```

---

## 8. Phase-Specific Analysis (`phases/`)

Deep analysis of model behavior within specific game phases.

### 8a. Acquisition Phase (`phases/acquisition.py`)

**Question:** How does the model price acquisitions? Does it use OS's face-value FI ability correctly?

**Method:** Filters to ACQ phase states and analyzes:
1. **Action distribution** — pass/price/FI-buy breakdown
2. **Price offset histogram** — how the model distributes price offers (0=low, 50=max)
3. **By-tier breakdown** — pricing behavior for red/orange/yellow/green/blue companies
4. **FI offer analysis** — pass vs buy rates, broken out by OS (face-value ability) vs non-OS corps
5. **Uncertainty analysis** — where the model is most uncertain about acquisition decisions

```bash
.venv/bin/python -m interp.phases.acquisition --load-data interp/data/states.npz
# Writes: interp/data/acq_phase_epoch<N>.html
```

### 8b. Invest Phase (`phases/invest.py`)

**Question:** How does the model value companies at auction? What share trading patterns emerge?

**Method:** Reconstructs auction narratives from sequential INVEST/BID states and analyzes:
1. **Auction pricing** — per-company opening bid, bid rounds, final price vs face value
2. **Share trades** — buys/sells per corp with president vs non-president breakdown
3. **Per-turn activity** — share buys/sells per corp per game turn (uses engine turn numbers)

```bash
.venv/bin/python -m interp.phases.invest --load-data interp/data/states.npz
# Writes: interp/data/invest_phase_epoch<N>.html
```

---

## Shared Utilities (`utils.py`)

All scripts share common infrastructure:

- **`load_model()`** — loads latest checkpoint (or specified path), returns (model, config, device, epoch)
- **`collect_states()`** — plays fast games via policy sampling (no MCTS) to collect diverse states. Returns an `InterpDataset`
- **`InterpDataset`** — states, legal_masks, phases, active_players, turn_numbers, game_indices. Saveable to `.npz` for reuse (backward-compatible with old files missing turn/game data)

**State collection tip:** Collect once with `--save-data`, then `--load-data` for all subsequent analyses. This is much faster than re-playing games for each script.

---

## Captum Attribution Toolkit (v0.8.0)

Reference for available attribution methods in `captum.attr`. Currently used: `IntegratedGradients` (decision_attr, acig), `LayerConductance` (arch_analysis), `NeuronConductance` (layers/policy_head).

### Input-Level Attribution
| Method | Type | Notes |
|--------|------|-------|
| `IntegratedGradients` | Gradient | Our workhorse. Path integral from baseline to input. |
| `Saliency` | Gradient | Simple gradient magnitude. Fast but noisy. |
| `InputXGradient` | Gradient | Input * gradient. Cheap approximation to IG. |
| `DeepLift` / `DeepLiftShap` | Reference | Compares to reference activation, not just gradient. |
| `GradientShap` | Gradient+SHAP | Stochastic IG variant with Shapley guarantees. |
| `FeatureAblation` | Perturbation | Zero/replace features, measure output change. What our full_ablation does manually. |
| `FeaturePermutation` | Perturbation | Permute features across samples. |
| `Occlusion` | Perturbation | Sliding window ablation. |
| `Lime` / `KernelShap` | Surrogate | Local linear model. Expensive but model-agnostic. |
| `ShapleyValueSampling` | Game-theoretic | Approximate Shapley values. Very expensive. |
| `LRP` | Propagation | Layer-wise relevance propagation. |

### Layer-Level Attribution
| Method | Notes |
|--------|-------|
| `LayerConductance` | IG through a specific layer. Our primary layer tool. |
| `LayerActivation` | Raw activations (no attribution, just extraction). |
| `LayerGradientXActivation` | Gradient * activation at a layer. Fast approximation. |
| `LayerIntegratedGradients` | IG computed at a specific layer. |
| `LayerFeatureAblation` | Ablation at intermediate layer (not just input). |
| `LayerDeepLift` / `LayerGradientShap` | Layer variants of input methods. |
| `LayerLRP` | LRP at a specific layer. |

### Neuron-Level Attribution
| Method | Notes |
|--------|-------|
| `NeuronConductance` | Conductance for a single neuron toward an output. Key for specialization analysis. |
| `NeuronIntegratedGradients` | IG for a single neuron. |
| `NeuronGradient` / `NeuronGradientShap` | Gradient-based neuron importance. |
| `NeuronFeatureAblation` | Which input features matter for a specific neuron. |
| `NeuronDeepLift` | DeepLift for neurons. |

### Utility
| Method | Notes |
|--------|-------|
| `NoiseTunnel` | Wraps any method with SmoothGrad (average over noisy inputs). Reduces gradient noise. |
| `InternalInfluence` | Influence of intermediate activations on output. |

---

## Future Work

See beads issues for planned analyses:
- **Feature interaction detection** (rss-az-ospf) — pairwise ablation to find feature combinations
- **Counterfactual sensitivity** (rss-az-9dox) — sweep individual features to plot response curves
