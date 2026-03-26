"""Normalization health check for the visible state vector.

Analyzes collected game states to report per-feature-group value ranges,
out-of-range counts, and distribution statistics. Helps identify features
where the normalization divisor is too small (values exceed [-1, +1]) or
too large (values are clustered near zero and underutilized).

Usage:
    .venv/bin/python -m interp.norm_check --load-data interp/data/states.npz
    .venv/bin/python -m interp.norm_check --num-games 20
    .venv/bin/python -m interp.norm_check --load-data interp/data/states.npz --feature invest:buy_impact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from interp.full_ablation import _PHASE_NAMES, _build_feature_groups
from interp.html import BAR_CSS, JS_FMT_PCT, JS_FMT_VAL, JS_MAKE_BAR, TAG_CSS, html_page, open_file
from interp.utils import InterpDataset, collect_states, load_model

# Feature group name -> list of phase indices where the feature is active.
# Derived from the "Context-Dependent Fields" table in VECTORS.md.
_PHASE_SPECIFIC_FEATURES: dict[str, list[int]] = {
    "turn:auction_price": [1],                       # BID
    "turn:auction_high_bidder": [1],                 # BID
    "turn:auction_starter": [1],                     # BID
    "turn:auction_passed": [1],                      # BID
    "turn:dividend_impact": [6],                     # DIV
    "turn:dividend_remaining": [6],                  # DIV
    "turn:issue_remaining": [8],                     # ISSUE
    "turn:issue_price_impact": [8],                  # ISSUE
    "turn:issue_cash_gain": [8],                     # ISSUE
    "turn:acq_is_fi_offer": [3],                     # ACQ
    "turn:acq_synergy": [3],                         # ACQ
    "turn:active_company": [1, 3, 4, 9, 10],         # BID, ACQ, CLOSE, IPO, PAR
    "turn:active_company_stars": [1, 3, 4, 9, 10],
    "turn:active_company_low_price": [1, 3, 4, 9, 10],
    "turn:active_company_face_value": [1, 3, 4, 9, 10],
    "turn:active_company_high_price": [1, 3, 4, 9, 10],
    "turn:active_company_income": [1, 3, 4, 9, 10],
    "turn:active_corp": [3, 4, 6, 8, 10],            # ACQ, CLOSE, DIV, ISSUE, PAR
    "turn:active_corp_income": [3, 4, 6, 8, 10],
    "turn:active_corp_stars": [3, 4, 6, 8, 10],
    "turn:active_corp_share_price": [3, 4, 6, 8, 10],
    "turn:active_corp_raw_revenue": [3, 4, 6, 8, 10],
    "turn:active_corp_synergy_income": [3, 4, 6, 8, 10],
    "turn:active_corp_coo_cost": [3, 4, 6, 8, 10],
    "turn:active_corp_ability_income": [3, 4, 6, 8, 10],
    "turn:active_corp_companies": [3, 4, 6, 8, 10],
    "turn:par_corp_treasury": [9, 10],               # IPO, PAR
    "turn:par_shares": [9, 10],                      # IPO, PAR
    "invest:buy_impact": [0],                        # INVEST
    "invest:sell_impact": [0],                       # INVEST
    "player:round_trips": [0],                       # INVEST
    "co:acquired": [3],                              # ACQ
    "corp:acq_proceeds": [3],                        # ACQ
}


def _group_stats(
    states: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
) -> list[dict[str, object]]:
    """Compute per-group normalization statistics."""
    rows: list[dict[str, object]] = []
    for name, indices in groups:
        vals = states[:, indices]
        flat = vals.ravel()
        n_total = flat.size
        n_nonzero = int(np.count_nonzero(flat))
        n_outside = int(np.sum((flat < -1.0) | (flat > 1.0)))
        abs_max = float(np.max(np.abs(flat))) if n_total > 0 else 0.0

        nz_vals = flat[flat != 0]
        nz_absmax = float(np.max(np.abs(nz_vals))) if nz_vals.size > 0 else 0.0
        nz_mean = float(np.mean(nz_vals)) if nz_vals.size > 0 else 0.0
        nz_std = float(np.std(nz_vals)) if nz_vals.size > 0 else 0.0

        rows.append({
            "name": name,
            "n_features": len(indices),
            "n_total": n_total,
            "n_nonzero": n_nonzero,
            "zero_frac": 1.0 - n_nonzero / n_total if n_total > 0 else 1.0,
            "n_outside": n_outside,
            "outside_frac": n_outside / n_total if n_total > 0 else 0.0,
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "abs_max": abs_max,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "nz_absmax": nz_absmax,
            "nz_mean": nz_mean,
            "nz_std": nz_std,
        })
    return rows


def _phase_specific_stats(
    states: np.ndarray,
    groups: list[tuple[str, np.ndarray]],
    phases: np.ndarray,
) -> list[dict[str, object]]:
    """Compute per-group stats filtered to only the phases where each feature is active."""
    rows: list[dict[str, object]] = []
    for name, indices in groups:
        phase_ids = _PHASE_SPECIFIC_FEATURES.get(name)
        if phase_ids is None:
            continue

        mask = np.isin(phases, phase_ids)
        if not mask.any():
            continue

        phase_states = states[mask]
        vals = phase_states[:, indices]
        flat = vals.ravel()
        n_total = flat.size
        n_nonzero = int(np.count_nonzero(flat))
        n_outside = int(np.sum((flat < -1.0) | (flat > 1.0)))
        abs_max = float(np.max(np.abs(flat))) if n_total > 0 else 0.0

        phase_labels = ", ".join(
            _PHASE_NAMES.get(p, str(p)) for p in sorted(phase_ids)
        )

        rows.append({
            "name": name,
            "n_features": len(indices),
            "n_states": int(mask.sum()),
            "n_total": n_total,
            "n_nonzero": n_nonzero,
            "zero_frac": 1.0 - n_nonzero / n_total if n_total > 0 else 1.0,
            "n_outside": n_outside,
            "outside_frac": n_outside / n_total if n_total > 0 else 0.0,
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "abs_max": abs_max,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "phases": phase_labels,
        })
    return rows


def _print_overview(rows: list[dict[str, object]], num_states: int) -> None:
    """Print high-level normalization summary."""
    total_features = sum(r["n_features"] for r in rows)  # type: ignore[arg-type]
    groups_outside = sum(1 for r in rows if r["n_outside"] > 0)  # type: ignore[arg-type]
    total_outside = sum(r["n_outside"] for r in rows)  # type: ignore[arg-type]
    total_vals = num_states * total_features
    worst = max(rows, key=lambda r: r["abs_max"])  # type: ignore[arg-type]

    print(f"\n{'='*70}")
    print(f"  NORMALIZATION OVERVIEW ({num_states:,} states, {total_features} features)")
    print(f"{'='*70}")
    print(f"  Feature groups with values outside [-1,+1]: {groups_outside} / {len(rows)}")
    print(f"  Individual values outside [-1,+1]: {total_outside:,} / {total_vals:,} ({total_outside/total_vals:.4%})")
    print(f"  Worst absolute value: {worst['abs_max']:.3f} ({worst['name']})")


def _print_table(rows: list[dict[str, object]]) -> None:
    """Print per-group table sorted by abs_max descending."""
    sorted_rows = sorted(rows, key=lambda r: -r["abs_max"])  # type: ignore[arg-type]

    print(f"\n  {'Feature':<28} {'#':>3} {'Min':>7} {'Max':>7} {'|Max|':>6} "
          f"{'Mean':>7} {'Std':>7} {'%Zero':>6} {'%Out':>6}")
    print(f"  {'-'*28} {'-'*3} {'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*6}")

    for r in sorted_rows:
        outside_str = f"{r['outside_frac']:.1%}" if r["n_outside"] > 0 else "-"  # type: ignore[arg-type]
        print(
            f"  {r['name']:<28} {r['n_features']:>3} "
            f"{r['min']:>7.3f} {r['max']:>7.3f} {r['abs_max']:>6.3f} "
            f"{r['mean']:>7.4f} {r['std']:>7.4f} "
            f"{r['zero_frac']:>6.1%} {outside_str:>6}"
        )


def _print_out_of_range(rows: list[dict[str, object]]) -> None:
    """Print details for groups with values outside [-1, +1]."""
    outside = [r for r in rows if r["n_outside"] > 0]  # type: ignore[arg-type]
    if not outside:
        print("\n  No features outside [-1, +1].")
        return

    outside.sort(key=lambda r: -r["abs_max"])  # type: ignore[arg-type]
    print(f"\n  FEATURES EXCEEDING [-1, +1]:")
    print(f"  {'Feature':<28} {'|Max|':>6} {'#Out':>8} {'%Out':>7}")
    print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*7}")
    for r in outside:
        print(
            f"  {r['name']:<28} {r['abs_max']:>6.3f} "
            f"{r['n_outside']:>8,} {r['outside_frac']:>6.2%}"
        )


def _print_sparsity(rows: list[dict[str, object]]) -> None:
    """Print groups with high zero fractions, excluding phase-specific features."""
    sparse = [
        r for r in rows
        if r["zero_frac"] > 0.90  # type: ignore[arg-type]
        and r["n_nonzero"] > 0  # type: ignore[arg-type]
        and r["name"] not in _PHASE_SPECIFIC_FEATURES
    ]
    sparse.sort(key=lambda r: -r["zero_frac"])  # type: ignore[arg-type]

    if not sparse:
        return

    print(f"\n  SPARSE FEATURES (>90% zero, excluding phase-specific):")
    print(f"  {'Feature':<28} {'%Zero':>7} {'NZ Mean':>8} {'NZ |Max|':>8}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*8}")
    for r in sparse:
        print(
            f"  {r['name']:<28} {r['zero_frac']:>6.1%} "
            f"{r['nz_mean']:>8.4f} {r['nz_absmax']:>8.4f}"
        )


def _print_phase_specific(phase_rows: list[dict[str, object]]) -> None:
    """Print phase-specific features with stats filtered to their relevant phases."""
    if not phase_rows:
        return

    sorted_rows = sorted(phase_rows, key=lambda r: -r["abs_max"])  # type: ignore[arg-type]

    print(f"\n  PHASE-SPECIFIC FEATURES (stats filtered to relevant phases):")
    print(f"  {'Feature':<28} {'Phases':<22} {'#St':>5} {'#':>3} "
          f"{'Min':>7} {'Max':>7} {'|Max|':>6} {'Mean':>7} {'Std':>7} {'%Zero':>7} {'%Out':>6}")
    print(f"  {'-'*28} {'-'*22} {'-'*5} {'-'*3} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    for r in sorted_rows:
        outside_str = f"{r['outside_frac']:.1%}" if r["n_outside"] > 0 else "-"  # type: ignore[arg-type]
        print(
            f"  {r['name']:<28} {r['phases']:<22} {r['n_states']:>5} {r['n_features']:>3} "
            f"{r['min']:>7.3f} {r['max']:>7.3f} {r['abs_max']:>6.3f} "
            f"{r['mean']:>7.4f} {r['std']:>7.4f} "
            f"{r['zero_frac']:>6.1%} {outside_str:>6}"
        )


def _print_feature_detail(
    name: str,
    states: np.ndarray,
    indices: np.ndarray,
    phases: np.ndarray,
) -> None:
    """Print detailed stats for a single feature group."""
    from interp.full_ablation import _PHASE_NAMES

    vals = states[:, indices]
    print(f"\n{'='*70}")
    print(f"  DETAIL: {name} ({len(indices)} features, indices [{indices[0]}..{indices[-1]}])")
    print(f"{'='*70}")

    # Per-sub-feature
    print(f"\n  Per-feature breakdown:")
    print(f"  {'Idx':>5} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8} {'%Zero':>6} {'%Out':>6}")
    print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
    for idx in indices:
        col = states[:, idx]
        nz = np.count_nonzero(col)
        n_out = int(np.sum((col < -1) | (col > 1)))
        zero_pct = 1.0 - nz / len(col)
        out_str = f"{n_out/len(col):.1%}" if n_out > 0 else "-"
        print(
            f"  {idx:>5} {col.min():>8.4f} {col.max():>8.4f} "
            f"{col.mean():>8.4f} {col.std():>8.4f} "
            f"{zero_pct:>5.0%} {out_str:>6}"
        )

    # Per-phase
    flat = vals.ravel()
    nz = flat[flat != 0]
    print(f"\n  Per-phase non-zero rates:")
    phase_ids = sorted(set(phases))
    for pid in phase_ids:
        mask = phases == pid
        pvals = vals[mask]
        pflat = pvals.ravel()
        pnz = np.count_nonzero(pflat)
        pname = _PHASE_NAMES.get(pid, str(pid))
        if pnz > 0:
            pnz_vals = pflat[pflat != 0]
            print(
                f"    {pname:<10} {mask.sum():>5} states, "
                f"nonzero={pnz:>6} ({pnz/pflat.size:>5.1%}), "
                f"mean(nz)={pnz_vals.mean():>8.4f}, "
                f"|max|={np.max(np.abs(pnz_vals)):>7.4f}"
            )
        else:
            print(f"    {pname:<10} {mask.sum():>5} states, nonzero=0")

    # Value distribution
    if nz.size > 0:
        print(f"\n  Non-zero value distribution:")
        for pct in [1, 5, 25, 50, 75, 95, 99]:
            print(f"    p{pct:<2}: {np.percentile(nz, pct):>8.4f}")


def _format_html_report(
    rows: list[dict[str, object]],
    phase_rows: list[dict[str, object]],
    num_states: int,
    num_games: int,
    epoch: int,
) -> str:
    """Generate a self-contained HTML report for normalization check."""
    total_features = sum(r["n_features"] for r in rows)  # type: ignore[arg-type]
    rows_json = json.dumps(rows)
    phase_rows_json = json.dumps(phase_rows)
    phase_specific_names_json = json.dumps(list(_PHASE_SPECIFIC_FEATURES.keys()))

    meta = (
        f"{num_states:,} states from {num_games} games."
        f" {total_features} visible features across {len(rows)} groups."
    )

    body = (
        '<h2>1. All Feature Groups</h2>'
        '<p style="color:#888;font-size:0.85rem">Sorted by |Max| descending.'
        ' Values outside [-1, +1] indicate the normalization divisor may be too small.</p>'
        '<table id="tbl-all"></table>'

        '<h2>2. Out-of-Range Features</h2>'
        '<p style="color:#888;font-size:0.85rem">Features with values exceeding [-1, +1].'
        ' Bar shows % of values out of range.</p>'
        '<table id="tbl-oor"></table>'

        '<h2>3. Sparse Features (&gt;90% zero)</h2>'
        '<p style="color:#888;font-size:0.85rem">Non-phase-specific features that are mostly zero.'
        ' Phase-specific features are shown in table 4.</p>'
        '<table id="tbl-sparse"></table>'

        '<h2>4. Phase-Specific Features</h2>'
        '<p style="color:#888;font-size:0.85rem">Context-dependent features, stats filtered to'
        ' only the phases where each feature is active.'
        ' Zeroed outside these phases by design.</p>'
        '<table id="tbl-phase"></table>'
    )

    data_js = (
        f"const rows = {rows_json};\n"
        f"const phaseRows = {phase_rows_json};\n"
        f"const phaseSpecificNames = new Set({phase_specific_names_json});"
    )

    report_js = (
        'function statusTag(absMax) {\n'
        '  if (absMax <= 1.0) return \'<span class="tag tag-ok">OK</span>\';\n'
        '  if (absMax <= 1.5) return \'<span class="tag tag-warn">mild</span>\';\n'
        '  return \'<span class="tag tag-bad">high</span>\';\n'
        '}\n'
        '\n'
        'function phaseTags(phases) {\n'
        '  return phases.split(\', \').map(p => \'<span class="tag-phase">\' + p + \'</span>\').join(\'\');\n'
        '}\n'
        '\n'
        '// --- All features table ---\n'
        '(function() {\n'
        '  const sorted = rows.slice().sort((a, b) => b.abs_max - a.abs_max);\n'
        '  const tbl = document.getElementById("tbl-all");\n'
        '  let html = \'<tr><th>Feature</th><th>#</th><th>Min</th><th>Max</th><th>|Max|</th>\' +\n'
        '    \'<th></th><th>Status</th><th>Mean</th><th>Std</th><th>%Zero</th><th>%Out</th></tr>\';\n'
        '  const maxAbsMax = Math.max(...sorted.map(r => r.abs_max));\n'
        '  for (const r of sorted) {\n'
        '    const oorStr = r.n_outside > 0 ? fmtPct(r.outside_frac) : \'-\';\n'
        '    const barCls = r.abs_max > 1.5 ? \'bar-red\' : r.abs_max > 1.0 ? \'bar-yellow\' : \'bar-green\';\n'
        '    html += \'<tr><td>\' + r.name + \'</td>\' +\n'
        '      \'<td>\' + r.n_features + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.min) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.max) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.abs_max) + \'</td>\' +\n'
        '      \'<td>\' + makeBar(r.abs_max, maxAbsMax, barCls) + \'</td>\' +\n'
        '      \'<td style="text-align:center">\' + statusTag(r.abs_max) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.mean, 4) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.std, 4) + \'</td>\' +\n'
        '      \'<td>\' + (r.zero_frac * 100).toFixed(1) + \'%</td>\' +\n'
        '      \'<td>\' + oorStr + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Out-of-range table ---\n'
        '(function() {\n'
        '  const oor = rows.filter(r => r.n_outside > 0).sort((a, b) => b.abs_max - a.abs_max);\n'
        '  const tbl = document.getElementById("tbl-oor");\n'
        '  if (oor.length === 0) {\n'
        '    tbl.innerHTML = \'<tr><td colspan="5" style="text-align:center;color:#4ecca3">No features outside [-1, +1]</td></tr>\';\n'
        '    return;\n'
        '  }\n'
        '  const maxPct = Math.max(...oor.map(r => r.outside_frac));\n'
        '  let html = \'<tr><th>Feature</th><th>|Max|</th><th># Out</th><th>% Out</th><th></th></tr>\';\n'
        '  for (const r of oor) {\n'
        '    html += \'<tr><td>\' + r.name + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.abs_max) + \'</td>\' +\n'
        '      \'<td>\' + r.n_outside.toLocaleString() + \'</td>\' +\n'
        '      \'<td>\' + fmtPct(r.outside_frac) + \'</td>\' +\n'
        '      \'<td>\' + makeBar(r.outside_frac, maxPct, \'bar-red\') + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Sparse features table (excluding phase-specific) ---\n'
        '(function() {\n'
        '  const sparse = rows.filter(r => r.zero_frac > 0.90 && r.n_nonzero > 0 && !phaseSpecificNames.has(r.name))\n'
        '    .sort((a, b) => b.zero_frac - a.zero_frac);\n'
        '  const tbl = document.getElementById("tbl-sparse");\n'
        '  if (sparse.length === 0) {\n'
        '    tbl.innerHTML = \'<tr><td colspan="4" style="text-align:center;color:#4ecca3">No sparse features</td></tr>\';\n'
        '    return;\n'
        '  }\n'
        '  let html = \'<tr><th>Feature</th><th>% Zero</th><th>NZ Mean</th><th>NZ |Max|</th></tr>\';\n'
        '  for (const r of sparse) {\n'
        '    html += \'<tr><td>\' + r.name + \'</td>\' +\n'
        '      \'<td>\' + (r.zero_frac * 100).toFixed(1) + \'%</td>\' +\n'
        '      \'<td>\' + fmtVal(r.nz_mean, 4) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.nz_absmax, 4) + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- Phase-specific features table ---\n'
        '(function() {\n'
        '  const sorted = phaseRows.slice().sort((a, b) => b.abs_max - a.abs_max);\n'
        '  const tbl = document.getElementById("tbl-phase");\n'
        '  if (sorted.length === 0) {\n'
        '    tbl.innerHTML = \'<tr><td colspan="12" style="text-align:center;color:#4ecca3">No phase-specific features</td></tr>\';\n'
        '    return;\n'
        '  }\n'
        '  let html = \'<tr><th>Feature</th><th>Phases</th><th># States</th><th>#</th>\' +\n'
        '    \'<th>Min</th><th>Max</th><th>|Max|</th><th></th><th>Status</th>\' +\n'
        '    \'<th>Mean</th><th>Std</th><th>%Zero</th><th>%Out</th></tr>\';\n'
        '  const maxAbsMax = Math.max(...sorted.map(r => r.abs_max));\n'
        '  for (const r of sorted) {\n'
        '    const oorStr = r.n_outside > 0 ? fmtPct(r.outside_frac) : \'-\';\n'
        '    const barCls = r.abs_max > 1.5 ? \'bar-red\' : r.abs_max > 1.0 ? \'bar-yellow\' : \'bar-green\';\n'
        '    html += \'<tr><td>\' + r.name + \'</td>\' +\n'
        '      \'<td style="text-align:left">\' + phaseTags(r.phases) + \'</td>\' +\n'
        '      \'<td>\' + r.n_states.toLocaleString() + \'</td>\' +\n'
        '      \'<td>\' + r.n_features + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.min) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.max) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.abs_max) + \'</td>\' +\n'
        '      \'<td>\' + makeBar(r.abs_max, maxAbsMax, barCls) + \'</td>\' +\n'
        '      \'<td style="text-align:center">\' + statusTag(r.abs_max) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.mean, 4) + \'</td>\' +\n'
        '      \'<td>\' + fmtVal(r.std, 4) + \'</td>\' +\n'
        '      \'<td>\' + (r.zero_frac * 100).toFixed(1) + \'%</td>\' +\n'
        '      \'<td>\' + oorStr + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();'
    )

    extra_css = BAR_CSS + "\n" + TAG_CSS + "\n" + (
        '.tag-phase {\n'
        '  display: inline-block; padding: 1px 5px; border-radius: 3px;\n'
        '  font-size: 0.7rem; font-weight: 600; margin: 1px 2px;\n'
        '  background: #1a2a3e; color: #4a9eff;\n'
        '}'
    )

    script = (
        data_js + "\n\n" + JS_MAKE_BAR + "\n" + JS_FMT_PCT + "\n" + JS_FMT_VAL
        + "\n\n" + report_js
    )

    return html_page(
        f"Normalization Check \u2014 Epoch {epoch}",
        meta=meta,
        body=body,
        script=script,
        extra_css=extra_css,
        max_width=1200,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalization health check for the visible state vector"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-data", type=str, default=None)
    parser.add_argument("--save-data", type=str, default=None)
    parser.add_argument(
        "--feature", type=str, default=None,
        help="Show detailed stats for a specific feature group (e.g. 'invest:buy_impact')",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="HTML output path (default: interp/data/norm_epoch<N>.html)",
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

    groups = _build_feature_groups(config.num_players)
    rows = _group_stats(dataset.states, groups)
    phase_rows = _phase_specific_stats(dataset.states, groups, dataset.phases)

    _print_overview(rows, dataset.num_states)
    _print_table(rows)
    _print_out_of_range(rows)
    _print_sparsity(rows)
    _print_phase_specific(phase_rows)

    if args.feature:
        match = [(n, idx) for n, idx in groups if n == args.feature]
        if not match:
            available = [n for n, _ in groups]
            print(f"\nFeature '{args.feature}' not found. Available: {', '.join(available)}")
        else:
            name, indices = match[0]
            _print_feature_detail(name, dataset.states, indices, dataset.phases)

    # --- HTML report ---
    if args.output:
        html_path = Path(args.output)
    else:
        html_path = Path("interp/data") / f"norm_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = _format_html_report(rows, phase_rows, dataset.num_states, dataset.num_games, epoch)
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
