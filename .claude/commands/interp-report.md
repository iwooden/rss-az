Run the full interpretability analysis pipeline on the latest checkpoint and produce a structured assessment of model behavior.

Optional argument: `$ARGUMENTS`
- `--num-games N`: Number of self-play games for data collection (default 100)
- If no arguments given, runs all analyses including probing with 100 games

## Instructions

### Phase 1: Data Collection

Collect states from self-play games using the latest checkpoint. This shared dataset is reused across all analyses.

```bash
.venv/bin/python -m interp.full_ablation --num-games <N> --save-data interp/data/states.npz --no-open
```

This runs feature ablation AND saves the state data. Note the epoch number from the output.

### Phase 2: Run Remaining Analyses

Run these in parallel (they are independent and all load from the saved data):

```bash
.venv/bin/python -m interp.norm_check --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.decision_attr --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.acig --load-data interp/data/states.npz --no-open
.venv/bin/python -m interp.arch_analysis --load-data interp/data/states.npz --block-detail --no-open
.venv/bin/python -m interp.tb_summary --max-rows 30
```

```bash
.venv/bin/python -m interp.probing --load-data interp/data/states.npz
```

### Phase 3: Collect Additional TB Metrics

Use `train.tb_reader` to get the full self-play metric picture:

```python
from train.tb_reader import read_tb_scalars
data = read_tb_scalars()
tags = [
    'self_play/nw_cash_pct_1st', 'self_play/nw_shares_pct_1st', 'self_play/nw_companies_pct_1st',
    'self_play/nw_cash_pct_3rd', 'self_play/nw_shares_pct_3rd', 'self_play/nw_companies_pct_3rd',
    'self_play/total_net_worth', 'self_play/corps_in_receivership',
    'self_play/total_companies', 'self_play/total_shares',
    'self_play/avg_active_corp_price', 'self_play/shares_1st', 'self_play/shares_3rd',
    'self_play/pres_share_value_1st', 'self_play/pres_share_value_3rd',
]
for tag in tags:
    vals = data.get(tag, [])
    # Print first, mid, and last epoch values for each
```

### Phase 4: Read All Reports

Read the generated markdown and console output from each analysis. The key files are:
- `interp/data/sensitivity_epoch<N>.md` — feature ablation (policy KL per feature per phase)
- Norm check console output — normalization issues, out-of-range features
- Decision attribution console output — what drives uncertain decisions
- ACIG console output — action-conditioned feature importance
- Architecture analysis console output — preprocessing contribution, block contribution, per-phase head conductance, SVD/effective rank
- TB summary console output — training curves, self-play stats
- `interp/data/probing_epoch<N>.md` — linear probing results (trunk, per-phase policy head, value head tables)

Read ALL output carefully before producing the analysis.

### Phase 5: Read Game Rules Context

Before writing your analysis, read `RULES.md` in the project root. Understanding game mechanics is essential for interpreting whether the model's learned feature sensitivities and strategies make game-theoretic sense. Also read `VECTORS.md` for state/action vector design context.

## Analysis to Produce

### 1. Strategic Overview

Identify the model's current play style from the TB self-play metrics. Track how the wealth composition (cash/shares/companies) and game structure (total shares, companies, corps in receivership, game length) have evolved over training. Call out any major strategic shifts (e.g., company-centric to share-centric play).

### 2. Feature Sensitivity Analysis

From the ablation report, identify:
- **Top 10 features by total policy KL** — what drives the model's decisions overall
- **Phase-specific spikes** — features that only matter in their relevant phase (validates context learning)
- **Surprisingly high features** — features with unexpectedly large influence
- **Surprisingly low features** — features the model ignores that it shouldn't (or features confirmed redundant)
- **Scalar vs one-hot preference** — compare `corp:share_price` (scalar) vs `corp:price_index` (one-hot)

### 3. Decision Quality

From decision attribution and ACIG:
- **Phase distribution of uncertainty** — which phases have the most high-uncertainty states?
- **Discriminative features per phase** — what does the model use to distinguish actions within each phase?
- **Emergent strategies** — identify any sophisticated decision logic (e.g., wealth-dependent buy/sell strategies, market-aware IPO timing)
- **Red flags** — features with high attribution for unexpected actions (spurious correlations)

### 4. Architecture Assessment

From the architecture analysis:
- **Depth utilization** — is effective rank still growing at the last block? Are all blocks contributing?
- **Width utilization** — trunk output rank vs hidden_dim
- **Head efficiency** — conductance distribution across head layers
- **Trunk specialization** — do policy and value use different blocks?

From probing:
- **Value probe trajectory** — R² through blocks (where does value get computed?)
- **Policy probe trajectory** — accuracy through blocks (does the trunk help policy?)
- **Trunk role** — does the trunk primarily serve value, policy, or both?

### 5. Normalization Issues

From the norm check:
- Features exceeding [-1, +1] — sorted by importance (cross-reference with ablation)
- Sparse features that should have signal
- Recommended divisor adjustments (only for features that matter)

### 6. Training Dynamics

From the TB summary:
- **Loss trajectory** — policy loss, value loss, total loss trends
- **Convergence status** — still improving? Plateauing? Diverging?
- **Entropy health** — policy entropy in healthy range (0.3-0.8)?
- **Value loss anomalies** — any unexplained increases or plateaus?

### 7. Data-Driven Recommendations

Prioritized, actionable suggestions organized as:
- **High priority** — changes likely to improve the model meaningfully
- **Medium priority** — worth trying but less certain impact
- **Low priority / monitoring** — things to watch or minor optimizations

For each recommendation, cite the specific data that motivates it. Avoid vague suggestions — be specific about what to change and why.

## Output Format

Use markdown with the section headers above. Be concise but data-driven — cite specific numbers, epochs, feature names, and KL/attribution values. The user is an experienced ML practitioner who understands AlphaZero and this game engine; don't explain basic concepts.

Compare results to previous interp runs if prior reports exist in `interp/data/` (e.g., `sensitivity_epoch149.md`). Call out improvements and regressions between checkpoints.
