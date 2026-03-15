Analyze the Tensorboard training logs and produce a structured assessment of how training is progressing.

## Instructions

Load all Tensorboard scalar data using the helper module at `train/tb_reader.py`. Run a Python script via Bash like:

```python
from train.tb_reader import read_tb_scalars, latest_value, sample_epochs
import json

data = read_tb_scalars("runs")
# data is dict[str, list[tuple[int, float]]] — tag -> sorted (step, value) pairs

# Print all available tags
print("Tags:", sorted(data.keys()))

# Get sampled epoch-level data (keeps first 3, last 3, evenly spaced middle)
for tag in ["epoch/total_loss_avg", "epoch/policy_loss_avg", "epoch/value_loss_avg"]:
    sampled = sample_epochs(data.get(tag, []))
    print(f"\n{tag}:")
    for step, val in sampled:
        print(f"  epoch {step}: {val:.4f}")

# Get latest value for a tag
print("Current LR:", latest_value(data, "lr"))
```

Adapt the script to extract whatever data you need for the analysis below. You may need multiple script invocations to keep output manageable.

The key scalar tags are:

**Loss curves:** `loss/total`, `loss/policy`, `loss/value` (per training step, every 100 steps); `epoch/total_loss_avg`, `epoch/policy_loss_avg`, `epoch/value_loss_avg` (per epoch); `lr` (learning rate)

**Self-play stats (per epoch):** `self_play/game_length_mean`, `self_play/duration_mean`, `self_play/examples_per_game`, `self_play/total_examples`, `self_play/net_worth_{1st,2nd,3rd}`, `self_play/net_worth_{1st,2nd,3rd}_{min,max}`

**Buffer (per epoch):** `buffer/size`, `buffer/utilization`

**Timing (per epoch):** `epoch/duration_secs`

**Profile (per epoch, only if `--profile` used):** `profile/search_*`, `profile/eval_client_*`, `profile/server_*`

## Analysis to produce

### 1. Training Overview

| Metric | Value |
|--------|-------|
| Epochs completed | N |
| Total training steps | N |
| Current learning rate | X |
| Buffer size / capacity | N / M (X%) |
| Avg epoch duration | Xm Ys |
| Avg game length | N moves |
| Avg game duration | Xs |

### 2. Loss Curves

Analyze the trajectory of policy and value losses:

- **Overall trend**: Are losses decreasing? Plateauing? Increasing (bad)?
- **Recent trend**: Compare the last 5 epochs to the preceding 5. Is learning still progressing or stalling?
- **Policy vs value balance**: Is one loss dominating? Are they decreasing at similar rates? A value loss that plateaus while policy loss keeps dropping (or vice versa) may indicate an imbalance.
- **Per-step noise**: How noisy are the per-step losses? High variance might suggest the batch size or learning rate needs adjustment.

Present the epoch-level loss values as a compact table showing the trend:

```
Epoch  Total   Policy  Value   LR
  1    X.XXX   X.XXX   X.XXX   X.Xe-X
  5    X.XXX   X.XXX   X.XXX   X.Xe-X
 10    ...
```

Show every epoch if there are fewer than 20, otherwise sample ~15-20 representative epochs (first few, evenly spaced middle, last few).

### 3. Self-Play Quality

Analyze how game quality evolves as the model improves:

- **Game length trend**: Are games getting longer (model learning to play more moves before game over)? Shorter? Stable? Early in training, random play produces short games; better play should produce longer, more strategic games — up to a point.
- **Net worth trends**: Are 1st-place net worths increasing over epochs? Is the spread between 1st and 3rd widening (model learning to differentiate skill) or narrowing?
- **Net worth ranges**: Are the min/max ranges tightening (more consistent play) or staying wide?

Present as a compact table:

```
Epoch  AvgMoves  1st$    2nd$    3rd$    Spread
  1      XX     $XXX    $XXX    $XXX     $XXX
  5      XX     $XXX    $XXX    $XXX     $XXX
 ...
```

### 4. Throughput & Efficiency

- **Games/minute**: Derive from `epoch/duration_secs` and games_per_epoch (from config or total_examples/examples_per_game).
- **Epoch time trend**: Is epoch duration increasing (expected as games get longer)?
- **Buffer fill rate**: How quickly is the buffer filling? Is it full yet?

If profile stats are present, also analyze:
- **GPU utilization**: What fraction of search time is spent waiting for GPU eval? What are the batch sizes?
- **Throughput**: States evaluated per second.
- **Bottleneck identification**: Is the system CPU-bound (low GPU utilization) or GPU-bound (high eval wait)?

### 5. Training Health Assessment

Based on the data, give a brief overall assessment:
- **Learning status**: Is the model actively learning, plateauing, or showing signs of divergence?
- **Key concerns**: Any red flags? (Loss spikes, value loss not decreasing, game lengths stuck, etc.)
- **Recommendations**: Any obvious hyperparameter adjustments suggested by the data? (LR too high/low, buffer too small, etc.)

## Output format

Use markdown with the section headers above. Be data-driven — cite specific numbers, epochs, and trends rather than vague statements. The user is an experienced ML practitioner who understands AlphaZero; don't explain basic concepts.
