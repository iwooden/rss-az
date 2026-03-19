"""Decision attribution on critical game states using IntegratedGradients.

Identifies high-uncertainty decisions (top-2 actions close in probability),
then attributes per-feature importance for each candidate action. Shows
which features push the model toward action A vs action B.

Usage:
    .venv/bin/python -m interp.decision_attr --load-data interp/data/states.npz
    .venv/bin/python -m interp.decision_attr --load-data interp/data/states.npz --top-k 20
    .venv/bin/python -m interp.decision_attr --num-games 20 --save-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.actions import decode_action_py
from interp.full_ablation import _PHASE_NAMES, _build_feature_groups
from interp.utils import (
    InterpDataset,
    batch_masked_softmax,
    collect_states,
    forward_batched,
    load_model,
)

# Action type names for readable output
_ACTION_TYPE_NAMES = {
    0: "pass", 1: "auction", 2: "buy_share", 3: "sell_share",
    4: "leave_bid", 5: "raise_bid", 6: "acq_price", 7: "acq_fi_high",
    8: "acq_fi_face", 9: "close", 10: "dividend", 11: "issue", 12: "ipo",
}


def _describe_action(action_idx: int, num_players: int) -> str:
    """Human-readable description of an action index."""
    _, atype, slot, corp_id, amount = decode_action_py(action_idx, num_players)
    name = _ACTION_TYPE_NAMES.get(atype, f"type_{atype}")
    parts = [name]
    if corp_id >= 0:
        parts.append(f"corp={corp_id}")
    if slot >= 0:
        parts.append(f"slot={slot}")
    if amount >= 0:
        parts.append(f"amt={amount}")
    return " ".join(parts)


def find_critical_states(
    states: np.ndarray,
    masks: np.ndarray,
    phases: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    num_players: int,
    margin_threshold: float = 0.15,
    min_entropy: float = 0.3,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    """Find states where the model is most uncertain between top-2 actions.

    Returns list of dicts with state info, sorted by decreasing uncertainty.
    """
    logits, values = forward_batched(model, device, states, batch_size)
    probs = batch_masked_softmax(logits, masks)

    # Mask out illegal actions for argmax
    masked_logits = logits.copy()
    masked_logits[masks <= 0] = -1e9

    critical: list[dict[str, Any]] = []
    for i in range(len(states)):
        sorted_idx = np.argsort(-probs[i])
        top1_idx = int(sorted_idx[0])
        top2_idx = int(sorted_idx[1])
        top1_prob = float(probs[i, top1_idx])
        top2_prob = float(probs[i, top2_idx])
        margin = top1_prob - top2_prob

        # Policy entropy
        p = probs[i]
        p_safe = p[p > 1e-10]
        entropy = float(-np.sum(p_safe * np.log(p_safe)))

        if margin < margin_threshold and entropy > min_entropy:
            critical.append({
                "idx": i,
                "phase": int(phases[i]),
                "phase_name": _PHASE_NAMES.get(int(phases[i]), str(phases[i])),
                "top1_action": top1_idx,
                "top2_action": top2_idx,
                "top1_prob": top1_prob,
                "top2_prob": top2_prob,
                "margin": margin,
                "entropy": entropy,
                "top1_desc": _describe_action(top1_idx, num_players),
                "top2_desc": _describe_action(top2_idx, num_players),
                "value_active": float(values[i, 0]),
            })

    # Sort by margin ascending (most uncertain first)
    critical.sort(key=lambda d: d["margin"])
    return critical


def compute_attributions(
    model: torch.nn.Module,
    device: torch.device,
    state: np.ndarray,
    mask: np.ndarray,
    action_idx: int,
    n_steps: int = 50,
    internal_batch_size: int = 50,
) -> np.ndarray:
    """Run IntegratedGradients for a specific action's policy logit.

    Returns per-feature attribution array, shape (visible_size,).
    """
    from captum.attr import IntegratedGradients

    model.eval()
    inp = torch.from_numpy(state).unsqueeze(0).to(device).requires_grad_(True)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        logits, _ = model(x)
        # Mask illegal actions before selecting the target logit
        m = torch.from_numpy(mask).unsqueeze(0).to(device)
        logits = logits.masked_fill(m <= 0, -1e9)
        return logits[:, action_idx]

    ig = IntegratedGradients(forward_fn)
    attr = ig.attribute(
        inp,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    assert isinstance(attr, torch.Tensor)
    return attr.squeeze(0).detach().cpu().numpy()


def aggregate_to_groups(
    attr: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
) -> list[tuple[str, float]]:
    """Sum attributions within each feature group. Returns (name, sum) pairs."""
    result = []
    for name, indices in groups:
        result.append((name, float(np.sum(attr[indices]))))
    return result


def run_decision_attribution(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    top_k: int = 15,
    margin_threshold: float = 0.15,
    min_entropy: float = 0.3,
    n_steps: int = 50,
    batch_size: int = 256,
) -> tuple[list[dict[str, Any]], int]:
    """Full pipeline: find critical states, compute attributions, aggregate.

    Returns (analysis dicts for top_k decisions, total critical count).
    """
    print("  Finding critical states...")
    critical = find_critical_states(
        dataset.states, dataset.legal_masks, dataset.phases,
        model, device, num_players,
        margin_threshold=margin_threshold, min_entropy=min_entropy,
        batch_size=batch_size,
    )
    print(f"  Found {len(critical)} critical states (margin < {margin_threshold})")

    num_critical = len(critical)

    if not critical:
        print("  No critical states found. Try increasing --margin-threshold.")
        return [], 0

    # Phase distribution of critical states
    from collections import Counter
    phase_counts = Counter(c["phase_name"] for c in critical)
    print(f"  Phase distribution: {dict(phase_counts)}")

    groups = _build_feature_groups(num_players)
    selected = critical[:top_k]

    print(f"  Computing attributions for top {len(selected)} states...")
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, info in enumerate(selected):
        idx = info["idx"]
        state = dataset.states[idx]
        mask = dataset.legal_masks[idx]

        # Attribution for top-1 action
        attr1 = compute_attributions(
            model, device, state, mask, info["top1_action"], n_steps=n_steps,
        )
        # Attribution for top-2 action
        attr2 = compute_attributions(
            model, device, state, mask, info["top2_action"], n_steps=n_steps,
        )

        # Differential attribution: what pushes toward action 1 vs action 2
        diff_attr = attr1 - attr2

        # Aggregate to feature groups
        grouped_top1 = aggregate_to_groups(attr1, groups)
        grouped_top2 = aggregate_to_groups(attr2, groups)
        grouped_diff = aggregate_to_groups(diff_attr, groups)

        # Sort diff by absolute value (most decisive features first)
        grouped_diff_sorted = sorted(grouped_diff, key=lambda x: -abs(x[1]))

        info["attr_top1"] = grouped_top1
        info["attr_top2"] = grouped_top2
        info["attr_diff"] = grouped_diff_sorted
        info["top_decisive"] = grouped_diff_sorted[:10]
        results.append(info)

        elapsed = time.perf_counter() - t0
        print(f"    {i + 1}/{len(selected)} ({elapsed:.1f}s)")

    return results, num_critical


def aggregate_patterns(
    results: list[dict[str, Any]],
) -> dict[str, list[tuple[str, float, int]]]:
    """Aggregate attribution patterns across decisions, grouped by phase.

    Returns {phase_name: [(feature, mean_abs_diff, count), ...]}.
    """
    from collections import defaultdict

    phase_features: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in results:
        phase = r["phase_name"]
        for name, val in r["attr_diff"]:
            phase_features[phase][name].append(abs(val))

    aggregated: dict[str, list[tuple[str, float, int]]] = {}
    for phase, features in phase_features.items():
        ranked = [
            (name, float(np.mean(vals)), len(vals))
            for name, vals in features.items()
        ]
        ranked.sort(key=lambda x: -x[1])
        aggregated[phase] = ranked

    return aggregated


def format_html_report(
    results: list[dict[str, Any]],
    patterns: dict[str, list[tuple[str, float, int]]],
    epoch: int,
    num_states: int,
    num_critical: int,
) -> str:
    """Generate a self-contained HTML report with decision gallery and patterns."""
    # Build JS data
    decisions_js: list[dict[str, Any]] = []
    for r in results:
        decisions_js.append({
            "phase": r["phase_name"],
            "top1": r["top1_desc"],
            "top2": r["top2_desc"],
            "top1_prob": round(r["top1_prob"], 3),
            "top2_prob": round(r["top2_prob"], 3),
            "margin": round(r["margin"], 4),
            "entropy": round(r["entropy"], 3),
            "value": round(r["value_active"], 3),
            "decisive": [
                {"name": n, "val": round(v, 4)} for n, v in r["top_decisive"]
            ],
        })

    patterns_js: dict[str, list[dict[str, Any]]] = {}
    for phase, features in patterns.items():
        patterns_js[phase] = [
            {"name": n, "mean": round(m, 4), "count": c}
            for n, m, c in features[:15]
        ]

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Decision Attribution — Epoch {epoch}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    margin: 2rem auto; max-width: 1200px; padding: 0 1rem;
  }}
  h1 {{ color: #f0f0f0; font-size: 1.4rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #ccc; font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #333; padding-bottom: 0.3rem; }}
  h3 {{ color: #aaa; font-size: 0.95rem; margin-top: 1.5rem; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .decision {{
    background: #16213e; border: 1px solid #2a2a4a; border-radius: 6px;
    padding: 1rem; margin-bottom: 1rem;
  }}
  .decision-header {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 0.5rem;
  }}
  .phase-tag {{
    background: #0f3460; color: #e0e0e0; padding: 2px 8px;
    border-radius: 3px; font-size: 0.8rem; font-weight: 600;
  }}
  .actions {{
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
    font-size: 0.85rem; margin: 0.5rem 0;
  }}
  .action-a {{ color: #4ecca3; }}
  .action-b {{ color: #e94560; }}
  .features {{ font-size: 0.82rem; }}
  .feat-row {{
    display: flex; align-items: center; margin: 2px 0;
  }}
  .feat-name {{
    width: 220px; min-width: 220px; flex-shrink: 0;
    font-family: monospace; font-size: 0.8rem; color: #aaa;
  }}
  .feat-bar-container {{
    flex: 1; min-width: 0;
  }}
  .feat-bar {{
    height: 14px; border-radius: 2px; min-width: 2px;
  }}
  .feat-val {{ font-size: 0.75rem; color: #888; margin-left: 6px; flex-shrink: 0; }}
  .bar-pos {{ background: #4ecca3; }}
  .bar-neg {{ background: #e94560; }}
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 0.82rem; margin-top: 0.5rem;
  }}
  th, td {{ padding: 4px 8px; border: 1px solid #2a2a4a; text-align: right; }}
  th {{ background: #16213e; color: #aaa; font-weight: 600; }}
  th:first-child, td:first-child {{ text-align: left; }}
  td:first-child {{ font-family: monospace; font-size: 0.8rem; }}
</style>
</head>
<body>
<h1>Decision Attribution — Epoch {epoch}</h1>
<div class="meta">
  IntegratedGradients on critical decisions (top-2 margin &lt; threshold).
  {num_critical} critical states found in {num_states:,} total states.
  Showing top {len(results)} most uncertain decisions.
  <br>Green bars = feature favors action A (top-1). Red bars = feature favors action B (top-2).
</div>

<h2>Decision Gallery</h2>
<div id="gallery"></div>

<h2>Aggregate Patterns by Phase</h2>
<div id="patterns"></div>

<script>
const decisions = {json.dumps(decisions_js)};
const patterns = {json.dumps(patterns_js)};

const gallery = document.getElementById("gallery");
for (const d of decisions) {{
  const div = document.createElement("div");
  div.className = "decision";

  // Find max |val| for bar scaling
  const maxVal = Math.max(...d.decisive.map(f => Math.abs(f.val)), 0.001);

  let featHtml = "";
  for (const f of d.decisive) {{
    const pct = Math.abs(f.val) / maxVal * 100;
    const cls = f.val >= 0 ? "bar-pos" : "bar-neg";
    featHtml += '<div class="feat-row">' +
      '<span class="feat-name">' + f.name + '</span>' +
      '<div class="feat-bar-container"><div class="feat-bar ' + cls + '" style="width:' + pct + '%"></div></div>' +
      '<span class="feat-val">' + (f.val >= 0 ? "+" : "") + f.val.toFixed(4) + '</span></div>';
  }}

  div.innerHTML = '<div class="decision-header">' +
    '<span class="phase-tag">' + d.phase + '</span>' +
    '<span style="color:#888;font-size:0.8rem">entropy=' + d.entropy +
    ' value=' + d.value + '</span></div>' +
    '<div class="actions">' +
    'A: <span class="action-a">' + d.top1 + '</span> (' + (d.top1_prob * 100).toFixed(1) + '%)' +
    ' &nbsp; B: <span class="action-b">' + d.top2 + '</span> (' + (d.top2_prob * 100).toFixed(1) + '%)' +
    '</div><div class="features">' + featHtml + '</div>';
  gallery.appendChild(div);
}}

const pDiv = document.getElementById("patterns");
for (const [phase, feats] of Object.entries(patterns)) {{
  let html = '<h3>' + phase + '</h3><table><tr><th>Feature</th><th>Mean |diff|</th><th>Count</th></tr>';
  for (const f of feats.slice(0, 10)) {{
    html += '<tr><td>' + f.name + '</td><td>' + f.mean.toFixed(4) + '</td><td>' + f.count + '</td></tr>';
  }}
  html += '</table>';
  pDiv.innerHTML += html;
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
        description="Decision attribution on critical game states (IntegratedGradients)"
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
        "--top-k", type=int, default=15,
        help="Number of critical decisions to analyze (default: 15)",
    )
    parser.add_argument(
        "--margin-threshold", type=float, default=0.15,
        help="Max probability margin between top-2 actions (default: 0.15)",
    )
    parser.add_argument(
        "--min-entropy", type=float, default=0.3,
        help="Minimum policy entropy to qualify as critical (default: 0.3)",
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="IntegratedGradients integration steps (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="HTML output path (default: interp/data/decisions_epoch<N>.html)",
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

    print(f"\nRunning decision attribution...")
    results, num_critical = run_decision_attribution(
        model, device, dataset, config.num_players,
        top_k=args.top_k,
        margin_threshold=args.margin_threshold,
        min_entropy=args.min_entropy,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )

    if not results:
        return

    patterns = aggregate_patterns(results)

    # Print summary to console
    print(f"\n{'='*70}")
    print(f"  DECISION ATTRIBUTION SUMMARY (epoch {epoch})")
    print(f"{'='*70}")
    for r in results[:5]:
        print(f"\n  [{r['phase_name']}] {r['top1_desc']} ({r['top1_prob']:.1%})"
              f" vs {r['top2_desc']} ({r['top2_prob']:.1%})")
        print(f"  Top decisive features (positive = favors action A):")
        for name, val in r["top_decisive"][:5]:
            sign = "+" if val >= 0 else ""
            print(f"    {name:<28} {sign}{val:.4f}")

    print(f"\n  AGGREGATE PATTERNS:")
    for phase, features in patterns.items():
        print(f"\n  {phase}:")
        for name, mean_abs, count in features[:5]:
            print(f"    {name:<28} mean|diff|={mean_abs:.4f} (n={count})")

    # Write HTML report
    if args.output:
        html_path = Path(args.output)
    else:
        html_path = Path("interp/data") / f"decisions_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = format_html_report(results, patterns, epoch, dataset.num_states, num_critical)
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        _open_file(html_path)


if __name__ == "__main__":
    main()
