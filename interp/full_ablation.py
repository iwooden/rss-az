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
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from core.state import get_layout
from interp.utils import (
    InterpDataset,
    batch_masked_softmax,
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
    10: "GAME_OVER",
}

NUM_COMPANIES = 36
NUM_CORPS = 8


def _build_feature_groups(num_players: int) -> list[tuple[str, np.ndarray]]:
    """Build (name, indices) for every named field in the visible state."""
    layout = get_layout(num_players)
    groups: list[tuple[str, np.ndarray]] = []

    # --- Phase & CoO ---
    groups.append(("phase", np.arange(layout.phase_offset, layout.phase_offset + layout.phase_size)))
    groups.append(("coo_level", np.arange(layout.coo_offset, layout.coo_offset + layout.coo_size)))

    # --- Player fields (aggregated across all players) ---
    _pf = [
        ("player:cash", 0, 1),
        ("player:net_worth", 1, 1),
        ("player:turn_order", 2, num_players),
        ("player:owned_companies", 2 + num_players, 36),
        ("player:owned_shares", 38 + num_players, 8),
        ("player:is_president", 46 + num_players, 8),
        ("player:round_trips", 54 + num_players, 8),
        ("player:acq_proceeds", 62 + num_players, 1),
        ("player:income", 63 + num_players, 1),
    ]
    for name, rel, size in _pf:
        idx: list[int] = []
        for p in range(num_players):
            base = layout.players_offset + p * layout.player_stride + rel
            idx.extend(range(base, base + size))
        groups.append((name, np.array(idx)))

    # --- Foreign Investor ---
    groups.append(("fi:cash", np.array([layout.fi_offset])))
    groups.append(("fi:income", np.array([layout.fi_offset + 1])))
    groups.append(("fi:companies", np.arange(layout.fi_offset + 2, layout.fi_offset + layout.fi_size)))

    # --- Company location flags ---
    groups.append(("co:for_auction", np.arange(layout.auction_companies_offset, layout.auction_companies_offset + 36)))
    groups.append(("co:revealed", np.arange(layout.revealed_companies_offset, layout.revealed_companies_offset + 36)))
    groups.append(("co:removed", np.arange(layout.removed_companies_offset, layout.removed_companies_offset + 36)))

    # --- Company adjusted incomes ---
    groups.append(("co:adj_incomes", np.arange(layout.company_incomes_offset, layout.company_incomes_offset + layout.company_incomes_size)))

    # --- Market availability ---
    groups.append(("market:available", np.arange(layout.market_offset, layout.market_offset + layout.market_size)))

    # --- Corporation fields (aggregated across all corps) ---
    _cf = [
        ("corp:active", 0, 1),
        ("corp:cash", 1, 1),
        ("corp:unissued_shares", 2, 1),
        ("corp:issued_shares", 3, 1),
        ("corp:bank_shares", 4, 1),
        ("corp:income", 5, 1),
        ("corp:stars", 6, 1),
        ("corp:share_price", 7, 1),
        ("corp:acq_proceeds", 8, 1),
        ("corp:in_receivership", 9, 1),
        ("corp:price_index", 10, 27),
        ("corp:owned_companies", 37, 36),
        ("corp:acq_companies", 73, 36),
    ]
    for name, rel, size in _cf:
        idx = []
        for c in range(NUM_CORPS):
            base = layout.corps_offset + c * layout.corp_stride + rel
            idx.extend(range(base, base + size))
        groups.append((name, np.array(idx)))

    # --- Turn state ---
    t = layout.turn_offset
    groups.append(("turn:end_card_flipped", np.array([t])))
    groups.append(("turn:consec_passes", np.array([t + 1])))

    # Auction block: price(1) + high_bidder(np) + starter(np) + passed(np)
    auction_idx: list[int] = [t + 2]  # auction_price
    auction_idx.extend(range(layout.auction_high_bidder_offset, layout.auction_high_bidder_offset + num_players))
    auction_idx.extend(range(layout.auction_starter_offset, layout.auction_starter_offset + num_players))
    auction_idx.extend(range(layout.auction_passed_offset, layout.auction_passed_offset + num_players))
    groups.append(("turn:auction", np.array(auction_idx)))

    # Remaining turn fields are sequential after auction_passed
    post = layout.auction_passed_offset + num_players
    groups.append(("turn:dividend", np.arange(post, post + 34)))  # impact(26) + remaining(8)
    post += 34
    groups.append(("turn:issue", np.arange(post, post + 10)))  # remaining(8) + price_impact(1) + cash_gain(1)
    post += 10
    groups.append(("turn:ipo_remaining", np.arange(post, post + 36)))  # remaining(36)
    post += 36
    groups.append(("turn:acq", np.arange(post, post + 37)))  # fi_offer(1) + synergy(36)
    post += 37
    groups.append(("turn:active_company", np.arange(post, post + 36)))  # one-hot(36)
    post += 36
    groups.append(("turn:active_company_info", np.arange(post, post + 5)))  # stars, low, face, high, income
    post += 5
    groups.append(("turn:active_corp", np.arange(post, post + 8)))  # one-hot(8)
    post += 8
    groups.append(("turn:active_corp_info", np.arange(post, post + 3)))  # income, stars, share_price
    post += 3
    groups.append(("turn:active_corp_companies", np.arange(post, post + 36)))  # owned company flags
    post += 36
    groups.append(("turn:cards_remaining", np.array([post])))  # cards remaining in deck
    post += 1
    assert post == layout.auction_slot_info_offset, (
        f"Turn end {post} != auction_slot_info_offset {layout.auction_slot_info_offset}"
    )

    # --- Auction slot info (5 scalars per slot: stars, low, face, high, income) ---
    s = layout.auction_slot_info_offset
    groups.append(("auction_slot_info", np.arange(s, s + layout.auction_slot_info_size)))

    # --- Invest impacts (8 buy + 8 sell, context-dependent on INVEST phase) ---
    groups.append(("invest:buy_impact", np.arange(layout.invest_impacts_offset, layout.invest_impacts_offset + NUM_CORPS)))
    groups.append(("invest:sell_impact", np.arange(layout.invest_impacts_offset + NUM_CORPS, layout.invest_impacts_offset + layout.invest_impacts_size)))

    # Verify full coverage
    all_idx = np.concatenate([g[1] for g in groups])
    assert len(all_idx) == layout.visible_size, (
        f"Coverage {len(all_idx)} != visible_size {layout.visible_size}"
    )
    assert len(set(all_idx)) == layout.visible_size, "Overlapping indices"

    return groups


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
    import json

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

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>State Vector Sensitivity — Epoch {epoch}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    margin: 2rem auto;
    max-width: 1200px;
    padding: 0 1rem;
  }}
  h1 {{ color: #f0f0f0; font-size: 1.4rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #ccc; font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #333; padding-bottom: 0.3rem; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.82rem;
    table-layout: fixed;
    margin-bottom: 2rem;
  }}
  th, td {{ padding: 5px 8px; border: 1px solid #2a2a4a; text-align: right; }}
  th {{
    background: #16213e; color: #aaa; font-weight: 600;
    position: sticky; top: 0; z-index: 1;
  }}
  th:first-child, td:first-child {{ text-align: left; width: 200px; }}
  td:first-child {{
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
    font-size: 0.8rem; color: #ccc;
  }}
  td:nth-child(2) {{ color: #888; width: 45px; }}
  tr:hover td {{ border-color: #555; }}
</style>
</head>
<body>
<h1>State Vector Sensitivity — Epoch {epoch}</h1>
<div class="meta">
  Feature group ablation (zero out each group, measure output change).
  {num_states:,} states from {num_games} games.
  Sorted by total metric (most sensitive first). Each table has independent color scaling.
</div>

<h2>Policy Head (KL Divergence)</h2>
<table id="tbl-policy"></table>

<h2>Value Head (MSE)</h2>
<table id="tbl-value"></table>

<script>
const policyRows = {policy_json};
const valueRows = {value_json};
const headers = {headers_json};

function buildTable(tblId, rows) {{
  let maxVal = 0;
  for (const r of rows)
    for (let i = 2; i < r.length; i++)
      if (r[i] > maxVal) maxVal = r[i];

  function heatColor(v) {{
    if (v === 0) return "rgba(0,0,0,0)";
    const t = v / maxVal;
    const h = t * 120;
    const s = 70 + t * -5;
    const l = 18 + t * 22;
    return "hsl(" + h + "," + s + "%," + l + "%)";
  }}

  function fmtVal(v) {{
    if (v === 0) return "0";
    if (v < 0.0005) return v.toExponential(0);
    return v.toFixed(4);
  }}

  const tbl = document.getElementById(tblId);
  const thead = tbl.createTHead().insertRow();
  for (const h of headers) {{
    const th = document.createElement("th");
    th.textContent = h;
    thead.appendChild(th);
  }}
  const tbody = tbl.createTBody();
  for (const r of rows) {{
    const tr = tbody.insertRow();
    for (let i = 0; i < r.length; i++) {{
      const td = tr.insertCell();
      if (i <= 1) {{
        td.textContent = r[i];
      }} else {{
        td.textContent = fmtVal(r[i]);
        td.style.backgroundColor = heatColor(r[i]);
        if (r[i] / maxVal > 0.45) td.style.color = "#fff";
      }}
    }}
  }}
}}

buildTable("tbl-policy", policyRows);
buildTable("tbl-value", valueRows);
</script>
</body>
</html>"""


def _open_file(path: Path) -> None:
    """Open a file with the platform's default handler."""
    system = platform.system()
    try:
        if system == "Linux":
            subprocess.Popen(  # noqa: S603
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Darwin":
            subprocess.Popen(  # noqa: S603
                ["open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            subprocess.Popen(  # noqa: S603
                ["start", "", str(path)],
                shell=True,  # noqa: S602
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except OSError:
        print(f"  Could not open browser. Open manually: {path}")


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
        _open_file(html_path)


if __name__ == "__main__":
    main()
