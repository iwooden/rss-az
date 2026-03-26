"""Full state vector sensitivity analysis.

Ablates every named field in the state vector and produces a heatmap
HTML report (and optional markdown table) showing policy KL divergence
per game phase.

Usage:
    .venv/bin/python -m interp.full_ablation
    .venv/bin/python -m interp.full_ablation --load-data interp/data/states.npz
    .venv/bin/python -m interp.full_ablation --no-open  # skip launching browser
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from interp.html import html_page, open_file
from interp.utils import (
    InterpDataset,
    batch_masked_softmax,
    build_feature_groups,
    collect_states,
    forward_batched,
    kl_divergence_batch,
    load_model,
)

_PHASE_NAMES = {
    0: "INVEST",
    1: "BID",
    2: "WRAP_UP",
    3: "ACQ",
    4: "CLOSE",
    5: "INCOME",
    6: "DIV",
    7: "END_CARD",
    8: "ISSUE",
    9: "IPO",
    10: "PAR",
    11: "GAME_OVER",
}

_build_feature_groups = build_feature_groups  # backward compat alias


# Row type: (name, num_features, total_metric, {phase_id: metric})
AblationRow = tuple[str, int, float, dict[int, float]]


def run_full_ablation(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    batch_size: int = 256,
) -> tuple[list[AblationRow], list[AblationRow], list[AblationRow], list[int]]:
    """Ablate every feature group, measuring policy KL and value MSE.

    Returns (policy_rows, value_rows, combined_rows, phase_ids).
    Each row: (name, num_features, total_metric, {phase_id: metric}).
    Combined rows use policy KL (same as before, for backward compat).
    """
    groups = _build_feature_groups(num_players)
    phase_ids = sorted(set(dataset.phases))

    # Original outputs (computed once)
    print("  Computing original model outputs...")
    orig_logits, orig_values = forward_batched(model, device, dataset.states, batch_size)
    orig_pol = batch_masked_softmax(orig_logits, dataset.legal_masks)

    policy_rows: list[AblationRow] = []
    value_rows: list[AblationRow] = []
    t0 = time.perf_counter()
    for i, (name, indices) in enumerate(groups):
        ablated = dataset.states.copy()
        ablated[:, indices] = 0.0
        abl_logits, abl_values = forward_batched(model, device, ablated, batch_size)

        # Policy: KL divergence
        abl_pol = batch_masked_softmax(abl_logits, dataset.legal_masks)
        kl = kl_divergence_batch(orig_pol, abl_pol)
        total_kl = float(np.mean(kl))
        phase_kls = {pid: float(np.mean(kl[dataset.phases == pid])) for pid in phase_ids}
        policy_rows.append((name, len(indices), total_kl, phase_kls))

        # Value: MSE across all player slots
        val_mse = np.mean((orig_values - abl_values) ** 2, axis=-1)  # (N,)
        total_mse = float(np.mean(val_mse))
        phase_mses = {pid: float(np.mean(val_mse[dataset.phases == pid])) for pid in phase_ids}
        value_rows.append((name, len(indices), total_mse, phase_mses))

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(groups)} groups ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(groups)} groups in {elapsed:.1f}s")

    # Sort each by total metric descending
    policy_rows.sort(key=lambda r: -r[2])
    value_rows.sort(key=lambda r: -r[2])

    # Combined = policy rows (backward compat)
    combined_rows = list(policy_rows)
    return policy_rows, value_rows, combined_rows, phase_ids


def format_markdown_table(
    rows: list[AblationRow],
    phase_ids: list[int],
) -> str:
    """Format results as a markdown table."""
    phase_names = [_PHASE_NAMES.get(pid, str(pid)) for pid in phase_ids]

    lines: list[str] = []

    # Header
    cols = ["Feature", "#", "Total"] + phase_names
    lines.append("| " + " | ".join(cols) + " |")
    aligns = [":---", "---:", "---:"] + ["---:"] * len(phase_names)
    lines.append("| " + " | ".join(aligns) + " |")

    # Rows
    for name, size, total_kl, phase_kls in rows:
        vals = [name, str(size), f"{total_kl:.4f}"]
        for pid in phase_ids:
            vals.append(f"{phase_kls[pid]:.4f}")
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def format_html_table(
    policy_rows: list[AblationRow],
    value_rows: list[AblationRow],
    phase_ids: list[int],
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Format results as an HTML heatmap page with policy, value, and overall tables."""
    phase_names = [_PHASE_NAMES.get(pid, str(pid)) for pid in phase_ids]
    headers = ["Feature", "#", "Total"] + phase_names

    def rows_to_json(rows: list[AblationRow]) -> str:
        js: list[list[object]] = []
        for name, size, total, phase_vals in rows:
            js.append([name, size, total] + [phase_vals[pid] for pid in phase_ids])
        return json.dumps(js)

    policy_json = rows_to_json(policy_rows)
    value_json = rows_to_json(value_rows)
    headers_json = json.dumps(headers)

    body = (
        '<h2>Policy Head (KL Divergence)</h2>\n'
        '<table id="tbl-policy"></table>\n'
        '\n'
        '<h2>Value Head (MSE)</h2>\n'
        '<table id="tbl-value"></table>'
    )

    data_js = (
        f"const policyRows = {policy_json};\n"
        f"const valueRows = {value_json};\n"
        f"const headers = {headers_json};"
    )

    report_js = (
        "function buildTable(tblId, rows) {\n"
        "  let maxVal = 0;\n"
        "  for (const r of rows)\n"
        "    for (let i = 2; i < r.length; i++)\n"
        "      if (r[i] > maxVal) maxVal = r[i];\n"
        "\n"
        "  function heatColor(v) {\n"
        '    if (v === 0) return "rgba(0,0,0,0)";\n'
        "    const t = v / maxVal;\n"
        "    const h = t * 120;\n"
        "    const s = 70 + t * -5;\n"
        "    const l = 18 + t * 22;\n"
        '    return "hsl(" + h + "," + s + "%," + l + "%)";\n'
        "  }\n"
        "\n"
        "  function fmtVal(v) {\n"
        '    if (v === 0) return "0";\n'
        "    if (v < 0.0005) return v.toExponential(0);\n"
        "    return v.toFixed(4);\n"
        "  }\n"
        "\n"
        '  const tbl = document.getElementById(tblId);\n'
        "  const thead = tbl.createTHead().insertRow();\n"
        "  for (const h of headers) {\n"
        '    const th = document.createElement("th");\n'
        "    th.textContent = h;\n"
        "    thead.appendChild(th);\n"
        "  }\n"
        "  const tbody = tbl.createTBody();\n"
        "  for (const r of rows) {\n"
        "    const tr = tbody.insertRow();\n"
        "    for (let i = 0; i < r.length; i++) {\n"
        "      const td = tr.insertCell();\n"
        "      if (i <= 1) {\n"
        "        td.textContent = r[i];\n"
        "      } else {\n"
        "        td.textContent = fmtVal(r[i]);\n"
        "        td.style.backgroundColor = heatColor(r[i]);\n"
        '        if (r[i] / maxVal > 0.45) td.style.color = "#fff";\n'
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
        "\n"
        'buildTable("tbl-policy", policyRows);\n'
        'buildTable("tbl-value", valueRows);'
    )

    meta = (
        f"Feature group ablation (zero out each group, measure output change). "
        f"{num_states:,} states from {num_games} games. "
        f"Sorted by total metric (most sensitive first). "
        f"Each table has independent color scaling."
    )

    extra_css = (
        "table { table-layout: fixed; }\n"
        "th:first-child, td:first-child { width: 200px; }\n"
        "td:nth-child(2) { color: #888; width: 45px; }"
    )

    return html_page(
        f"State Vector Sensitivity \u2014 Epoch {epoch}",
        meta=meta,
        body=body,
        script=data_js + "\n\n" + report_js,
        extra_css=extra_css,
        max_width=1200,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full state vector sensitivity analysis (policy KL per feature per phase)"
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
        "--output", type=str, default=None,
        help="Markdown output path (default: interp/data/sensitivity_epoch<N>.md)",
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

    print(f"\nRunning full ablation ({dataset.num_states} states)...")
    policy_rows, value_rows, combined_rows, phase_ids = run_full_ablation(
        model, device, dataset, config.num_players, batch_size=args.batch_size,
    )

    table = format_markdown_table(combined_rows, phase_ids)

    # Always write both markdown and HTML
    md_header = (
        f"# State Vector Sensitivity (epoch {epoch})\n\n"
        f"Policy KL divergence when each feature group is zeroed.\n"
        f"Data: {dataset.num_states} states from {dataset.num_games} games.\n"
        f"Sorted by total KL (most sensitive first).\n\n"
    )

    if args.output:
        md_path = Path(args.output)
    else:
        md_path = Path("interp/data") / f"sensitivity_epoch{epoch}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_header + table + "\n")
    print(f"\nMarkdown written to {md_path}")

    html_path = md_path.with_suffix(".html")
    html_content = format_html_table(
        policy_rows, value_rows, phase_ids, epoch, dataset.num_states, dataset.num_games,
    )
    html_path.write_text(html_content)
    print(f"HTML heatmap written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
