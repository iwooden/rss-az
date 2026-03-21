"""Action-Conditioned Integrated Gradients (ACIG).

Groups IG attributions by the action the model chose, revealing which features
drive specific decision types within each phase.

Where ablation (full_ablation) shows "what matters per phase" and decision
attribution (decision_attr) shows "what tips uncertain decisions", ACIG shows
"what drives each action type."

Usage:
    .venv/bin/python -m interp.acig --load-data interp/data/states.npz
    .venv/bin/python -m interp.acig --load-data interp/data/states.npz --samples-per-bucket 100
    .venv/bin/python -m interp.acig --num-games 20 --save-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from core.actions import decode_action_py
from interp.decision_attr import (
    _ACTION_TYPE_NAMES,
    aggregate_to_groups,
    compute_attributions,
)
from interp.full_ablation import _PHASE_NAMES, _build_feature_groups
from interp.utils import (
    InterpDataset,
    collect_states,
    forward_batched,
    load_model,
)

# Phases with player decisions (skip automated: WRAP_UP, INCOME, END_CARD, GAME_OVER)
_DECISION_PHASES = {"INVEST", "BID", "ACQ", "CLOSE", "DIV", "ISSUE", "IPO"}

# Dividend sub-grouping boundaries
_DIV_LOW_MAX = 8
_DIV_MID_MAX = 17


def _bucket_name(action_type: int, amount: int) -> str:
    """Readable bucket name, sub-grouping dividends by amount range."""
    if action_type == 10:  # ACTION_DIVIDEND
        if amount <= _DIV_LOW_MAX:
            return "div_low"
        elif amount <= _DIV_MID_MAX:
            return "div_mid"
        else:
            return "div_high"
    return _ACTION_TYPE_NAMES.get(action_type, f"type_{action_type}")


def _classify_states(
    dataset: InterpDataset,
    model: torch.nn.Module,
    device: torch.device,
    num_players: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Classify each state by its argmax action.

    Returns:
        action_indices: (N,) int array of chosen action indices
        phase_names: per-state phase name
        bucket_names: per-state action bucket name
    """
    logits, _ = forward_batched(model, device, dataset.states, batch_size)
    masked_logits = logits.copy()
    masked_logits[dataset.legal_masks <= 0] = -1e9
    action_indices = np.argmax(masked_logits, axis=-1).astype(np.int32)

    phase_names: list[str] = []
    bucket_names: list[str] = []
    for i in range(len(dataset.states)):
        phase_id = int(dataset.phases[i])
        _, action_type, _, _, amount = decode_action_py(
            int(action_indices[i]), num_players
        )
        phase_names.append(_PHASE_NAMES.get(phase_id, str(phase_id)))
        bucket_names.append(_bucket_name(action_type, amount))

    return action_indices, phase_names, bucket_names


# Type aliases for results
BucketAttrs = dict[str, list[tuple[str, float]]]  # {bucket: [(feat, mean_attr)]}
PhaseAttrs = dict[str, BucketAttrs]  # {phase: BucketAttrs}
BucketCounts = dict[str, dict[str, int]]  # {phase: {bucket: count}}


def run_acig(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    samples_per_bucket: int = 50,
    n_steps: int = 30,
    min_bucket_size: int = 5,
    batch_size: int = 256,
) -> tuple[PhaseAttrs, BucketCounts]:
    """Run Action-Conditioned Integrated Gradients.

    Returns:
        attributions: {phase: {bucket: [(feat, mean_signed_attr), ...]}} sorted by |attr|
        bucket_counts: {phase: {bucket: total_state_count}}
    """
    print("  Classifying all states by argmax action...")
    action_indices, phase_names, bucket_names = _classify_states(
        dataset, model, device, num_players, batch_size,
    )

    # Group state indices by (phase, bucket)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in range(len(dataset.states)):
        if phase_names[i] in _DECISION_PHASES:
            buckets[(phase_names[i], bucket_names[i])].append(i)

    bucket_counts: BucketCounts = defaultdict(dict)
    print(f"  {len(buckets)} action buckets across decision phases:")
    for (phase, bucket), indices in sorted(buckets.items()):
        print(f"    {phase:>8} / {bucket:<20} {len(indices):>5} states")
        bucket_counts[phase][bucket] = len(indices)

    # Sample from each bucket
    rng = np.random.default_rng(42)
    sampled: dict[tuple[str, str], np.ndarray] = {}
    total_ig = 0
    skipped = 0
    for key, indices in sorted(buckets.items()):
        if len(indices) < min_bucket_size:
            skipped += 1
            continue
        n = min(len(indices), samples_per_bucket)
        sampled[key] = rng.choice(indices, n, replace=False)
        total_ig += n
    if skipped:
        print(f"  Skipped {skipped} buckets with < {min_bucket_size} states")
    print(f"\n  Running IG on {total_ig} sampled states (~{total_ig * 0.5:.0f}s)...")

    groups = _build_feature_groups(num_players)
    num_features = len(groups)

    # Run IG and accumulate per-bucket
    attributions: PhaseAttrs = defaultdict(dict)
    done = 0
    t0 = time.perf_counter()

    for (phase_name, bucket_name), indices in sorted(sampled.items()):
        acc = np.zeros(num_features, dtype=np.float64)

        for idx in indices:
            attr = compute_attributions(
                model, device,
                dataset.states[idx],
                dataset.legal_masks[idx],
                int(action_indices[idx]),
                n_steps=n_steps,
            )
            grouped = aggregate_to_groups(attr, groups)
            for fi, (_, val) in enumerate(grouped):
                acc[fi] += val

            done += 1
            if done % 50 == 0 or done == total_ig:
                elapsed = time.perf_counter() - t0
                rate = elapsed / done
                remaining = (total_ig - done) * rate
                print(f"    {done}/{total_ig} ({elapsed:.1f}s, ~{remaining:.0f}s remaining)")

        n = len(indices)
        avg = [(groups[fi][0], float(acc[fi] / n)) for fi in range(num_features)]
        avg.sort(key=lambda x: -abs(x[1]))
        attributions[phase_name][bucket_name] = avg

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")
    return dict(attributions), dict(bucket_counts)


def _discriminative_features(
    phase_attrs: BucketAttrs,
    top_n: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    """Find features whose attribution varies most across action types.

    Returns [(feat_name, std, {bucket: mean_attr}), ...] sorted by std desc.
    """
    feat_per_bucket: dict[str, dict[str, float]] = defaultdict(dict)
    for bucket, feats in phase_attrs.items():
        for name, val in feats:
            feat_per_bucket[name][bucket] = val

    results: list[tuple[str, float, dict[str, float]]] = []
    for name, per_bucket in feat_per_bucket.items():
        vals = list(per_bucket.values())
        if len(vals) < 2:
            continue
        std = float(np.std(vals))
        results.append((name, std, per_bucket))

    results.sort(key=lambda x: -x[1])
    return results[:top_n]


def format_console(
    attributions: PhaseAttrs,
    bucket_counts: BucketCounts,
    epoch: int,
    num_states: int,
) -> str:
    """Format console summary."""
    lines: list[str] = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  ACIG SUMMARY (epoch {epoch}, {num_states:,} states)")
    lines.append(f"{'='*70}")

    phase_order = ["INVEST", "BID", "ACQ", "CLOSE", "DIV", "ISSUE", "IPO"]
    for phase in phase_order:
        if phase not in attributions:
            continue
        phase_buckets = attributions[phase]
        counts = bucket_counts.get(phase, {})
        total = sum(counts.values())
        lines.append(f"\n  {phase} ({total:,} states):")

        for bucket in sorted(phase_buckets.keys()):
            feats = phase_buckets[bucket]
            n = counts.get(bucket, 0)
            lines.append(f"    {bucket} (n={n}):")
            for name, val in feats[:5]:
                sign = "+" if val >= 0 else ""
                lines.append(f"      {name:<30} {sign}{val:.4f}")

        # Discriminative features
        disc = _discriminative_features(phase_buckets)
        if disc:
            lines.append(
                f"\n    Discriminative (highest variance across actions):"
            )
            for name, std, per_bucket in disc[:5]:
                parts = [f"{b}={v:+.2f}" for b, v in sorted(per_bucket.items())]
                lines.append(
                    f"      {name:<30} std={std:.3f}  "
                    f"[{', '.join(parts)}]"
                )

    return "\n".join(lines)


def format_html(
    attributions: PhaseAttrs,
    bucket_counts: BucketCounts,
    epoch: int,
    num_states: int,
) -> str:
    """Generate self-contained HTML report."""
    phases_js: list[dict[str, object]] = []
    phase_order = ["INVEST", "BID", "ACQ", "CLOSE", "DIV", "ISSUE", "IPO"]

    for phase in phase_order:
        if phase not in attributions:
            continue
        phase_buckets = attributions[phase]
        counts = bucket_counts.get(phase, {})
        bucket_names = sorted(phase_buckets.keys())

        # Collect top features: union of top 15 per bucket, keep top 20 overall
        top_feats: set[str] = set()
        for bucket in bucket_names:
            for name, _ in phase_buckets[bucket][:15]:
                top_feats.add(name)

        feat_grid: dict[str, dict[str, float]] = {}
        for feat in top_feats:
            feat_grid[feat] = {}
            for bucket in bucket_names:
                for name, val in phase_buckets[bucket]:
                    if name == feat:
                        feat_grid[feat][bucket] = round(val, 4)
                        break

        sorted_feats = sorted(
            feat_grid.keys(),
            key=lambda f: max(
                (abs(v) for v in feat_grid[f].values()), default=0
            ),
            reverse=True,
        )[:20]

        disc = _discriminative_features(phase_buckets, top_n=10)
        disc_js = [
            {
                "name": name,
                "std": round(std, 4),
                "per_bucket": {b: round(v, 4) for b, v in per_bucket.items()},
            }
            for name, std, per_bucket in disc
        ]

        phases_js.append({
            "name": phase,
            "buckets": [
                {"name": b, "count": counts.get(b, 0)} for b in bucket_names
            ],
            "features": sorted_feats,
            "grid": {f: feat_grid[f] for f in sorted_feats},
            "discriminative": disc_js,
        })

    data_json = json.dumps(phases_js)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ACIG Report — Epoch {epoch}</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  background: #1a1a2e; color: #e0e0e0;
  margin: 2rem auto; max-width: 1400px; padding: 0 1rem;
}}
h1 {{ color: #f0f0f0; font-size: 1.4rem; margin-bottom: 0.3rem; }}
h2 {{
  color: #ccc; font-size: 1.1rem; margin-top: 2rem;
  border-bottom: 1px solid #333; padding-bottom: 0.3rem;
}}
h3 {{ color: #aaa; font-size: 0.95rem; margin-top: 1.5rem; }}
.meta {{ color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }}
.bucket-info {{ color: #888; font-size: 0.82rem; margin: 0.3rem 0 0.8rem 0; }}
table {{
  border-collapse: collapse; width: 100%;
  font-size: 0.82rem; margin-top: 0.5rem;
}}
th, td {{ padding: 4px 8px; border: 1px solid #2a2a4a; text-align: right; }}
th {{ background: #16213e; color: #aaa; font-weight: 600; }}
th:first-child, td:first-child {{ text-align: left; }}
td:first-child {{
  font-family: "SF Mono", "Fira Code", Consolas, monospace;
  font-size: 0.8rem;
}}
.heatmap td {{
  min-width: 80px;
  font-family: "SF Mono", "Fira Code", Consolas, monospace;
  font-size: 0.78rem;
}}
</style>
</head>
<body>
<h1>Action-Conditioned Integrated Gradients — Epoch {epoch}</h1>
<div class="meta">
  Per-action-type feature attributions via IntegratedGradients.
  {num_states:,} total states. Signed means: positive = feature pushes
  logit for this action UP.
  <br>Green = positive attribution, red = negative, intensity = magnitude.
</div>
<div id="content"></div>
<script>
const phases = {data_json};

function heatColor(val, maxAbs) {{
  if (maxAbs === 0) return "transparent";
  const intensity = Math.min(Math.abs(val) / maxAbs, 1.0);
  const alpha = intensity * 0.7;
  if (val >= 0) return "rgba(78, 204, 163, " + alpha + ")";
  return "rgba(233, 69, 96, " + alpha + ")";
}}

const content = document.getElementById("content");
for (const phase of phases) {{
  let html = "<h2>" + phase.name + "</h2>";
  html += '<div class="bucket-info">';
  html += phase.buckets.map(
    function(b) {{ return b.name + ": " + b.count + " states"; }}
  ).join(" &middot; ");
  html += "</div>";

  // Attribution heatmap
  html += "<h3>Feature Attributions by Action Type</h3>";
  const bucketNames = phase.buckets.map(function(b) {{ return b.name; }});
  html += '<table class="heatmap"><tr><th>Feature</th>';
  for (let bi = 0; bi < bucketNames.length; bi++)
    html += "<th>" + bucketNames[bi] + "</th>";
  html += "</tr>";

  // Color scale: max abs across all cells in this phase
  let maxAbs = 0;
  for (let fi = 0; fi < phase.features.length; fi++) {{
    const feat = phase.features[fi];
    const row = phase.grid[feat];
    if (!row) continue;
    for (let bi = 0; bi < bucketNames.length; bi++) {{
      const v = row[bucketNames[bi]];
      if (v !== undefined && Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
    }}
  }}

  for (let fi = 0; fi < phase.features.length; fi++) {{
    const feat = phase.features[fi];
    const row = phase.grid[feat] || {{}};
    html += "<tr><td>" + feat + "</td>";
    for (let bi = 0; bi < bucketNames.length; bi++) {{
      const v = row[bucketNames[bi]];
      if (v !== undefined) {{
        const bg = heatColor(v, maxAbs);
        const sign = v >= 0 ? "+" : "";
        html += '<td style="background:' + bg + '">' + sign + v.toFixed(4) + "</td>";
      }} else {{
        html += '<td style="color:#555">&mdash;</td>';
      }}
    }}
    html += "</tr>";
  }}
  html += "</table>";

  // Discriminative features
  if (phase.discriminative.length > 0) {{
    html += "<h3>Most Discriminative Features (highest variance across actions)</h3>";
    html += "<table><tr><th>Feature</th><th>Std</th>";
    for (let bi = 0; bi < bucketNames.length; bi++)
      html += "<th>" + bucketNames[bi] + "</th>";
    html += "</tr>";

    let discMax = 0;
    for (let di = 0; di < phase.discriminative.length; di++) {{
      const d = phase.discriminative[di];
      const vals = Object.values(d.per_bucket);
      for (let vi = 0; vi < vals.length; vi++) {{
        if (Math.abs(vals[vi]) > discMax) discMax = Math.abs(vals[vi]);
      }}
    }}

    for (let di = 0; di < phase.discriminative.length; di++) {{
      const d = phase.discriminative[di];
      html += "<tr><td>" + d.name + "</td><td>" + d.std.toFixed(4) + "</td>";
      for (let bi = 0; bi < bucketNames.length; bi++) {{
        const v = d.per_bucket[bucketNames[bi]];
        if (v !== undefined) {{
          const bg = heatColor(v, discMax);
          const sign = v >= 0 ? "+" : "";
          html += '<td style="background:' + bg + '">' + sign + v.toFixed(4) + "</td>";
        }} else {{
          html += '<td style="color:#555">&mdash;</td>';
        }}
      }}
      html += "</tr>";
    }}
    html += "</table>";
  }}

  content.innerHTML += html;
}}
</script>
</body>
</html>"""


def _open_file(path: Path) -> None:
    """Open a file with the platform's default handler."""
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(
                ["open", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except OSError:
        print(f"  Could not open browser. Open manually: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Action-Conditioned Integrated Gradients (ACIG)",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--load-data", type=str, default=None)
    parser.add_argument("--save-data", type=str, default=None)
    parser.add_argument(
        "--samples-per-bucket", type=int, default=50,
        help="Max states to sample per action bucket (default: 50)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=30,
        help="IG integration steps per state (default: 30)",
    )
    parser.add_argument(
        "--min-bucket-size", type=int, default=5,
        help="Skip action buckets with fewer states (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="HTML output path (default: interp/data/acig_epoch<N>.html)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open the HTML report in a browser",
    )
    args = parser.parse_args()

    model, config, device, epoch = load_model(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    if args.load_data:
        print(f"\nLoading data from {args.load_data}")
        dataset = InterpDataset.load(args.load_data)
        print(f"Loaded {dataset.num_states} states from {dataset.num_games} games")
    else:
        print(f"\nCollecting states from {args.num_games} games...")
        dataset = collect_states(
            model, config, device,
            num_games=args.num_games, seed=args.seed,
            checkpoint_path=str(args.checkpoint or f"latest (epoch {epoch})"),
        )
        if args.save_data:
            dataset.save(args.save_data)

    print("\nRunning ACIG analysis...")
    attributions, bucket_counts = run_acig(
        model, device, dataset, config.num_players,
        samples_per_bucket=args.samples_per_bucket,
        n_steps=args.n_steps,
        min_bucket_size=args.min_bucket_size,
        batch_size=args.batch_size,
    )

    summary = format_console(attributions, bucket_counts, epoch, dataset.num_states)
    print(summary)

    if args.output:
        html_path = Path(args.output)
    else:
        html_path = Path("interp/data") / f"acig_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = format_html(attributions, bucket_counts, epoch, dataset.num_states)
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        _open_file(html_path)


if __name__ == "__main__":
    main()
