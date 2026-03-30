"""Architecture analysis: block contribution, layer conductance, effective rank.

Three analyses to inform architecture tuning (depth and width):
1. Block contribution: how much each residual block changes the representation
2. Layer conductance: how much each block matters for policy/value (Captum)
3. Effective rank: how much of hidden_dim is actually utilized (SVD)

Usage:
    .venv/bin/python -m interp.arch_analysis
    .venv/bin/python -m interp.arch_analysis --load-data interp/data/states.npz
    .venv/bin/python -m interp.arch_analysis --skip-conductance
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp.html import BAR_CSS, JS_MAKE_BAR, html_page, open_file
from interp.utils import DECISION_PHASE_ORDER, InterpDataset, collect_states, load_model


# ---------------------------------------------------------------------------
# 1. Residual block contribution (hook-based, no Captum)
# ---------------------------------------------------------------------------

def analyze_block_contributions(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    """Measure ||output - input|| / ||input|| for each residual block."""
    model.eval()
    block_ratios: list[list[float]] = [[] for _ in range(len(model.blocks))]

    handles = []
    for i, block in enumerate(model.blocks):
        def hook(
            _module: torch.nn.Module,  # noqa: ARG001
            inp: tuple[torch.Tensor, ...],
            out: torch.Tensor,
            idx: int = i,
        ) -> None:
            x_in = inp[0].detach()
            ratio = (out - x_in).detach().norm(dim=-1) / (x_in.norm(dim=-1) + 1e-8)
            block_ratios[idx].extend(ratio.cpu().tolist())

        handles.append(block.register_forward_hook(hook))

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    results: list[dict[str, float]] = []
    for i, block in enumerate(model.blocks):
        r = np.array(block_ratios[i])
        results.append({
            "block": float(i),
            "mean": float(r.mean()),
            "std": float(r.std()),
            "p95": float(np.percentile(r, 95)),
            "fc2_weight_norm": float(block.fc2.weight.data.norm().item()),
        })
    return results


def analyze_preprocess_contributions(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    """Measure ||output|| / ||input|| for each Linear layer in input_preprocess."""
    model.eval()
    layer_ratios: list[list[float]] = []
    layer_info: list[tuple[int, int, int]] = []  # (seq_idx, in_feat, out_feat)

    handles = []
    for seq_idx, layer in enumerate(model.input_preprocess):
        if not isinstance(layer, torch.nn.Linear):
            continue
        ratios: list[float] = []
        layer_ratios.append(ratios)
        layer_info.append((seq_idx, layer.in_features, layer.out_features))

        def hook(
            _module: torch.nn.Module,  # noqa: ARG001
            inp: tuple[torch.Tensor, ...],
            out: torch.Tensor,
            _ratios: list[float] = ratios,
        ) -> None:
            x_in = inp[0].detach()
            ratio = out.detach().norm(dim=-1) / (x_in.norm(dim=-1) + 1e-8)
            _ratios.extend(ratio.cpu().tolist())

        handles.append(layer.register_forward_hook(hook))

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    results: list[dict[str, float]] = []
    for ratios, (seq_idx, in_feat, out_feat) in zip(layer_ratios, layer_info):
        r = np.array(ratios)
        results.append({
            "label": f"preprocess[{seq_idx}] {in_feat}\u2192{out_feat}",  # type: ignore[dict-item]
            "mean": float(r.mean()),
            "std": float(r.std()),
            "p95": float(np.percentile(r, 95)),
            "weight_norm": float(model.input_preprocess[seq_idx].weight.data.norm().item()),
        })
    return results


# ---------------------------------------------------------------------------
# 2. Layer conductance (Captum)
# ---------------------------------------------------------------------------

def analyze_layer_conductance(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    max_samples: int = 200,
    n_steps: int = 20,
    internal_batch_size: int | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Captum LayerConductance for each block toward policy and value heads.

    Returns {"policy": [...], "value": [...]}, each a list of per-block dicts.
    """
    from captum.attr import LayerConductance

    model.eval()

    # Subsample for speed
    if states.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(states.shape[0], max_samples, replace=False)
        states = states[idx]

    inputs = torch.from_numpy(states).to(device).requires_grad_(True)

    # Captum requires internal_batch_size >= 2 * num_samples for finite differences
    if internal_batch_size is None:
        internal_batch_size = 2 * states.shape[0]

    def policy_forward(x: torch.Tensor) -> torch.Tensor:
        p, _ = model(x)
        return p.sum(dim=-1)

    def value_forward(x: torch.Tensor) -> torch.Tensor:
        _, v = model(x)
        return v.sum(dim=-1)

    results: dict[str, list[dict[str, float]]] = {"policy": [], "value": []}

    for head_name, fwd_fn in [("policy", policy_forward), ("value", value_forward)]:
        conductances: list[float] = []
        t0 = time.perf_counter()
        for i, block in enumerate(model.blocks):
            lc = LayerConductance(fwd_fn, block)
            attr = lc.attribute(
                inputs, n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
            assert isinstance(attr, torch.Tensor)
            conductances.append(float(attr.abs().sum(dim=-1).mean().item()))
            elapsed = time.perf_counter() - t0
            print(f"    {head_name} block {i}/{len(model.blocks)} ({elapsed:.1f}s)")

        total = sum(conductances)
        for i, c in enumerate(conductances):
            results[head_name].append({
                "block": float(i),
                "conductance": c,
                "pct": c / total * 100 if total > 0 else 0.0,
            })

    return results


# ---------------------------------------------------------------------------
# 2b. Head-layer conductance (Captum)
# ---------------------------------------------------------------------------

def analyze_head_conductance(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    max_samples: int = 200,
    n_steps: int = 20,
    internal_batch_size: int | None = None,
) -> dict[str, list[dict[str, float]]]:
    """Captum LayerConductance for each Linear layer within the input
    preprocessing, policy head, and value head.

    Returns {"input_preprocess": [...], "policy": [...], "value": [...]},
    each a list of per-layer dicts.
    """
    from captum.attr import LayerConductance

    model.eval()

    if states.shape[0] > max_samples:
        idx = np.random.default_rng(0).choice(states.shape[0], max_samples, replace=False)
        states = states[idx]

    inputs = torch.from_numpy(states).to(device).requires_grad_(True)

    if internal_batch_size is None:
        internal_batch_size = 2 * states.shape[0]

    def policy_forward(x: torch.Tensor) -> torch.Tensor:
        p, _ = model(x)
        return p.sum(dim=-1)

    def value_forward(x: torch.Tensor) -> torch.Tensor:
        _, v = model(x)
        return v.sum(dim=-1)

    # Analyze input_preprocess toward both heads combined (full model output)
    def total_forward(x: torch.Tensor) -> torch.Tensor:
        p, v = model(x)
        return p.sum(dim=-1) + v.sum(dim=-1)

    results: dict[str, list[dict[str, float]]] = {"input_preprocess": []}

    modules_to_analyze: list[tuple[str, Any, Any]] = [
        ("input_preprocess", model.input_preprocess, total_forward),
    ]
    for head_idx, head in enumerate(model.phase_heads):
        key = f"phase:{DECISION_PHASE_ORDER[head_idx]}"
        results[key] = []
        modules_to_analyze.append((key, head, policy_forward))
    results["value"] = []
    modules_to_analyze.append(("value", model.value_head, value_forward))

    for section_name, module, fwd_fn in modules_to_analyze:
        # Phase heads: filter inputs to states that route to this head
        if section_name.startswith("phase:"):
            head_idx = next(
                i for i, name in enumerate(DECISION_PHASE_ORDER)
                if section_name == f"phase:{name}"
            )
            phase_mask = states[:, head_idx] > 0.5
            n_phase = int(phase_mask.sum())
            if n_phase == 0:
                print(f"    {section_name}: no matching states, skipping")
                continue
            section_inputs = torch.from_numpy(states[phase_mask]).to(device).requires_grad_(True)
            section_batch_size = 2 * n_phase
        else:
            section_inputs = inputs
            section_batch_size = internal_batch_size

        linear_layers: list[tuple[int, torch.nn.Module]] = [
            (i, layer) for i, layer in enumerate(module)
            if isinstance(layer, torch.nn.Linear)
        ]

        conductances: list[float] = []
        layer_names: list[str] = []
        t0 = time.perf_counter()

        for seq_idx, layer in linear_layers:
            in_feat = layer.in_features
            out_feat = layer.out_features
            label = f"{section_name}[{seq_idx}] {in_feat}→{out_feat}"
            layer_names.append(label)

            lc = LayerConductance(fwd_fn, layer)
            attr = lc.attribute(
                section_inputs, n_steps=n_steps,
                internal_batch_size=section_batch_size,
            )
            assert isinstance(attr, torch.Tensor)
            conductances.append(float(attr.abs().sum(dim=-1).mean().item()))
            elapsed = time.perf_counter() - t0
            print(f"    {label} ({elapsed:.1f}s)")

        total = sum(conductances)
        for i, c in enumerate(conductances):
            results[section_name].append({
                "layer_idx": float(i),
                "label": layer_names[i],  # type: ignore[dict-item]
                "conductance": c,
                "pct": c / total * 100 if total > 0 else 0.0,
            })

    return results


# ---------------------------------------------------------------------------
# 3. Effective rank (SVD)
# ---------------------------------------------------------------------------

def analyze_effective_rank(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
    include_heads: bool = False,
    include_block_detail: bool = False,
) -> list[dict[str, object]]:
    """SVD-based effective rank after input_proj, each block, trunk_norm, and optionally head/block-internal layers."""
    model.eval()

    input_name = "input_preprocess"
    input_module = model.input_preprocess

    layer_names: list[str] = []
    activations: dict[str, list[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(
            _module: torch.nn.Module,  # noqa: ARG001
            _inp: tuple[torch.Tensor, ...],  # noqa: ARG001
            out: torch.Tensor,
        ) -> None:
            activations[name].append(out.detach().cpu())
        return hook

    # Input preprocessing — per-layer breakdown if include_heads, else composite
    if include_heads and hasattr(input_module, "__iter__"):
        for seq_idx, layer in enumerate(input_module):
            if isinstance(layer, torch.nn.Linear):
                label = f"{input_name}[{seq_idx}] {layer.in_features}→{layer.out_features}"
                layer_names.append(label)
                activations[label] = []
                handles.append(layer.register_forward_hook(make_hook(label)))
    else:
        layer_names.append(input_name)
        activations[input_name] = []
        handles.append(input_module.register_forward_hook(make_hook(input_name)))

    # Trunk blocks (composite + optional per-sublayer detail)
    for i, block in enumerate(model.blocks):
        if include_block_detail:
            for sub_name, sub_module in [("norm", block.norm), ("fc1", block.fc1), ("fc2", block.fc2)]:
                label = f"block_{i}.{sub_name}"
                if isinstance(sub_module, torch.nn.Linear):
                    label += f" {sub_module.in_features}→{sub_module.out_features}"
                layer_names.append(label)
                activations[label] = []
                handles.append(sub_module.register_forward_hook(make_hook(label)))
        else:
            name = f"block_{i}"
            layer_names.append(name)
            activations[name] = []
            handles.append(block.register_forward_hook(make_hook(name)))

    layer_names.append("trunk_norm")
    activations["trunk_norm"] = []
    handles.append(model.trunk_norm.register_forward_hook(make_hook("trunk_norm")))

    # Head layers (Linear only, skip activations like GELU/Tanh)
    if include_heads:
        # Per-phase policy heads
        for head_idx, head in enumerate(model.phase_heads):
            phase_name = DECISION_PHASE_ORDER[head_idx]
            for seq_idx, layer in enumerate(head):
                if isinstance(layer, torch.nn.Linear):
                    label = f"phase:{phase_name}[{seq_idx}] {layer.in_features}→{layer.out_features}"
                    layer_names.append(label)
                    activations[label] = []
                    handles.append(layer.register_forward_hook(make_hook(label)))
        # Value head
        for seq_idx, layer in enumerate(model.value_head):
            if isinstance(layer, torch.nn.Linear):
                label = f"value[{seq_idx}] {layer.in_features}→{layer.out_features}"
                layer_names.append(label)
                activations[label] = []
                handles.append(layer.register_forward_hook(make_hook(label)))

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    results: list[dict[str, object]] = []
    for name in layer_names:
        acts = torch.cat(activations[name], dim=0)  # (N, hidden_dim)
        acts = acts - acts.mean(dim=0, keepdim=True)  # center
        _, s, _ = torch.linalg.svd(acts, full_matrices=False)

        # Entropy-based effective rank
        s_pos = s[s > 1e-10]
        p = s_pos / s_pos.sum()
        entropy = -(p * torch.log(p)).sum().item()
        eff_rank = float(np.exp(entropy))

        # Threshold-based (1% of max singular value)
        eff_rank_1pct = int((s > 0.01 * s[0]).sum().item())

        # Top-k energy (variance explained)
        total_var = float((s**2).sum().item())
        top50 = float((s[:50] ** 2).sum().item()) / total_var * 100 if total_var > 0 else 0
        top100 = float((s[:100] ** 2).sum().item()) / total_var * 100 if total_var > 0 else 0
        top200 = float((s[:200] ** 2).sum().item()) / total_var * 100 if total_var > 0 else 0

        width = int(acts.shape[1])
        results.append({
            "layer": name,
            "width": width,
            "eff_rank": eff_rank,
            "utilization": eff_rank / width * 100 if width > 0 else 0.0,
            "eff_rank_1pct": eff_rank_1pct,
            "top50_energy": top50,
            "top100_energy": top100,
            "top200_energy": top200,
        })

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_preprocess_contributions(rows: list[dict[str, float]]) -> None:
    print(f"\n  {'Layer':<30}  {'||out||/||in||':>14}  {'std':>8}  {'p95':>8}  {'||W||':>10}")
    print(f"  {'-'*30}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in rows:
        bar = chr(0x2588) * int(r["mean"] * 20)
        print(
            f"  {r['label']:<30}  {r['mean']:>14.4f}  {r['std']:>8.4f}  "
            f"{r['p95']:>8.4f}  {r['weight_norm']:>10.4f}  {bar}"
        )


def _print_contributions(rows: list[dict[str, float]]) -> None:
    print(f"\n  {'Block':>5}  {'||res||/||in||':>14}  {'std':>8}  {'p95':>8}  {'fc2 ||W||':>10}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in rows:
        b = int(r["block"])
        bar = chr(0x2588) * int(r["mean"] * 60)
        print(
            f"  {b:>5}  {r['mean']:>14.4f}  {r['std']:>8.4f}  "
            f"{r['p95']:>8.4f}  {r['fc2_weight_norm']:>10.4f}  {bar}"
        )


def _print_conductance(data: dict[str, list[dict[str, float]]]) -> None:
    print(
        f"\n  {'Block':>5}  {'Pol Cond':>10}  {'Pol %':>7}  "
        f"{'Val Cond':>10}  {'Val %':>7}"
    )
    print(f"  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*10}  {'-'*7}")
    for pi, vi in zip(data["policy"], data["value"]):
        b = int(pi["block"])
        p_bar = chr(0x2588) * int(pi["pct"] / 2)
        v_bar = chr(0x2591) * int(vi["pct"] / 2)
        print(
            f"  {b:>5}  {pi['conductance']:>10.2f}  {pi['pct']:>6.1f}%  "
            f"{vi['conductance']:>10.2f}  {vi['pct']:>6.1f}%  {p_bar}{v_bar}"
        )


def _print_head_conductance(data: dict[str, list[dict[str, float]]]) -> None:
    _titles = {"input_preprocess": "Input Preprocessing", "value": "Value Head"}
    for section_name, layers in data.items():
        if not layers:
            continue
        title = _titles.get(section_name, section_name.replace("phase:", "Phase Head: "))
        print(f"\n  {title}:")
        print(f"    {'Layer':<35}  {'Cond':>10}  {'%':>7}")
        print(f"    {'-'*35}  {'-'*10}  {'-'*7}")
        for r in layers:
            bar = chr(0x2588) * int(r["pct"] / 3)
            print(f"    {r['label']:<35}  {r['conductance']:>10.1f}  {r['pct']:>6.1f}%  {bar}")


def _print_ranks(rows: list[dict[str, object]], hidden_dim: int) -> None:
    print(f"\n  hidden_dim = {hidden_dim}")
    print(
        f"\n  {'Layer':<25}  {'Width':>5}  {'Eff.Rank':>9}  {'Util%':>6}  {'Rank(1%)':>9}  "
        f"{'Top-50':>8}  {'Top-100':>8}  {'Top-200':>8}"
    )
    print(
        f"  {'-'*25}  {'-'*5}  {'-'*9}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}"
    )
    for r in rows:
        name = str(r["layer"])
        w = r["width"]
        assert isinstance(w, int)
        er = r["eff_rank"]
        assert isinstance(er, float)
        util = r["utilization"]
        assert isinstance(util, float)
        er1 = r["eff_rank_1pct"]
        assert isinstance(er1, int)
        t50 = r["top50_energy"]
        t100 = r["top100_energy"]
        t200 = r["top200_energy"]
        assert isinstance(t50, float) and isinstance(t100, float) and isinstance(t200, float)
        print(
            f"  {name:<25}  {w:>5}  {er:>9.1f}  {util:>5.1f}%  {er1:>9}  "
            f"{t50:>7.1f}%  {t100:>7.1f}%  {t200:>7.1f}%"
        )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def format_html_report(
    contributions: list[dict[str, float]],
    preprocess: list[dict[str, float]],
    ranks: list[dict[str, object]],
    hidden_dim: int,
    epoch: int,
    num_states: int,
    conductance: dict[str, list[dict[str, float]]],
    head_conductance: dict[str, list[dict[str, float]]] | None = None,
) -> str:
    """Generate a self-contained HTML report for architecture analysis."""
    contrib_json = json.dumps(contributions)
    preprocess_json = json.dumps(preprocess)
    ranks_json = json.dumps(ranks)
    conductance_json = json.dumps(conductance) if conductance else "null"
    head_cond_json = json.dumps(head_conductance) if head_conductance else "null"

    body = """\
<h2>1. Layer Contribution</h2>

<h3 style="color:#aaa;font-size:0.95rem">Input Preprocessing \u2014 ||output|| / ||input||</h3>
<p style="color:#888;font-size:0.85rem">Signal gain/attenuation through each preprocessing layer.</p>
<table id="tbl-preprocess"></table>

<h3 style="color:#aaa;font-size:0.95rem;margin-top:1.5rem">Residual Blocks \u2014 ||residual|| / ||input||</h3>
<p style="color:#888;font-size:0.85rem">How much each block changes the representation. Higher = more active.</p>
<table id="tbl-contrib"></table>

<div id="conductance-section"></div>
<div id="head-conductance-section"></div>

<h2 id="rank-header">Effective Rank (SVD)</h2>
<p style="color:#888;font-size:0.85rem">Entropy-based effective dimensionality at each layer.</p>
<table id="tbl-rank"></table>"""

    data_js = f"""\
const contribs = {contrib_json};
const preprocess = {preprocess_json};
const ranks = {ranks_json};
const conductance = {conductance_json};
const headCond = {head_cond_json};
const hiddenDim = {hidden_dim};"""

    report_js = """\
// --- Input preprocessing table ---
(function() {
  const tbl = document.getElementById("tbl-preprocess");
  const maxMean = Math.max(...preprocess.map(r => r.mean));
  let html = '<tr><th>Layer</th><th>||out||/||in||</th><th></th><th>Std</th><th>P95</th><th>||W||</th></tr>';
  for (const r of preprocess) {
    html += '<tr><td>' + r.label + '</td>' +
      '<td>' + r.mean.toFixed(4) + '</td>' +
      '<td>' + makeBar(r.mean, maxMean, 'bar-blue') + '</td>' +
      '<td>' + r.std.toFixed(4) + '</td>' +
      '<td>' + r.p95.toFixed(4) + '</td>' +
      '<td>' + r.weight_norm.toFixed(1) + '</td></tr>';
  }
  tbl.innerHTML = html;
})();

// --- Block contribution table ---
(function() {
  const tbl = document.getElementById("tbl-contrib");
  const maxMean = Math.max(...contribs.map(r => r.mean));
  let html = '<tr><th>Block</th><th>||res||/||in||</th><th></th><th>Std</th><th>P95</th><th>fc2 ||W||</th></tr>';
  for (const r of contribs) {
    html += '<tr><td>Block ' + Math.round(r.block) + '</td>' +
      '<td>' + r.mean.toFixed(4) + '</td>' +
      '<td>' + makeBar(r.mean, maxMean, 'bar-blue') + '</td>' +
      '<td>' + r.std.toFixed(4) + '</td>' +
      '<td>' + r.p95.toFixed(4) + '</td>' +
      '<td>' + r.fc2_weight_norm.toFixed(1) + '</td></tr>';
  }
  tbl.innerHTML = html;
})();

// --- Trunk conductance table ---
if (conductance) {
  const section = document.getElementById("conductance-section");
  const maxPol = Math.max(...conductance.policy.map(r => r.conductance));
  const maxVal = Math.max(...conductance.value.map(r => r.conductance));
  let html = '<h2>2. Trunk Block Conductance</h2>' +
    '<p style="color:#888;font-size:0.85rem">Integrated-gradient conductance toward each head. Shows which blocks each head relies on.</p>' +
    '<table><tr><th>Block</th><th>Policy</th><th></th><th>Policy %</th><th>Value</th><th></th><th>Value %</th></tr>';
  for (let i = 0; i < conductance.policy.length; i++) {
    const p = conductance.policy[i];
    const v = conductance.value[i];
    html += '<tr><td>Block ' + Math.round(p.block) + '</td>' +
      '<td>' + p.conductance.toFixed(1) + '</td>' +
      '<td>' + makeBar(p.conductance, maxPol, 'bar-green') + '</td>' +
      '<td>' + p.pct.toFixed(1) + '%</td>' +
      '<td>' + v.conductance.toFixed(1) + '</td>' +
      '<td>' + makeBar(v.conductance, maxVal, 'bar-orange') + '</td>' +
      '<td>' + v.pct.toFixed(1) + '%</td></tr>';
  }
  html += '</table>';
  section.innerHTML = html;
}

// --- Head-layer conductance table ---
if (headCond) {
  const section = document.getElementById("head-conductance-section");
  let html = '<h2>Per-Layer Conductance</h2>' +
    '<p style="color:#888;font-size:0.85rem">Conductance within each module\\'s Linear layers. Shows which layer does the most work.</p>';

  const colorMap = {'input_preprocess': 'bar-blue', 'value': 'bar-orange'};
  const titleMap = {'input_preprocess': 'Input Preprocessing', 'value': 'Value Head'};

  for (const [headName, layers] of Object.entries(headCond)) {
    const maxC = Math.max(...layers.map(r => r.conductance));
    const cls = colorMap[headName] || (headName.startsWith('phase:') ? 'bar-green' : 'bar-blue');
    const title = titleMap[headName] || (headName.startsWith('phase:') ? 'Phase Head: ' + headName.slice(6) : headName);
    html += '<h3 style="color:#aaa;font-size:0.95rem;margin-top:1rem">' + title + '</h3>';
    html += '<table><tr><th>Layer</th><th>Conductance</th><th></th><th>%</th></tr>';
    for (const r of layers) {
      html += '<tr><td>' + r.label + '</td>' +
        '<td>' + r.conductance.toFixed(1) + '</td>' +
        '<td>' + makeBar(r.conductance, maxC, cls) + '</td>' +
        '<td>' + r.pct.toFixed(1) + '%</td></tr>';
    }
    html += '</table>';
  }
  section.innerHTML = html;
}

// --- Effective rank table ---
(function() {
  const tbl = document.getElementById("tbl-rank");
  let html = '<tr><th>Layer</th><th>Width</th><th>Eff. Rank</th><th></th><th>Util%</th><th>Rank(1%)</th><th>Top-50</th><th>Top-100</th><th>Top-200</th></tr>';
  for (const r of ranks) {
    html += '<tr><td>' + r.layer + '</td>' +
      '<td>' + r.width + '</td>' +
      '<td>' + r.eff_rank.toFixed(1) + '</td>' +
      '<td>' + makeBar(r.eff_rank, r.width, 'bar-blue') + '</td>' +
      '<td>' + r.utilization.toFixed(1) + '%</td>' +
      '<td>' + r.eff_rank_1pct + '</td>' +
      '<td>' + r.top50_energy.toFixed(1) + '%</td>' +
      '<td>' + r.top100_energy.toFixed(1) + '%</td>' +
      '<td>' + r.top200_energy.toFixed(1) + '%</td></tr>';
  }
  tbl.innerHTML = html;
})();

// Renumber rank header based on what's shown
document.getElementById("rank-header").textContent =
  (conductance ? "3" : "2") + ". Effective Rank (SVD)";"""

    meta = f"{num_states:,} states. hidden_dim={hidden_dim}."

    return html_page(
        f"Architecture Analysis \u2014 Epoch {epoch}",
        meta=meta,
        body=body,
        script=data_js + "\n\n" + JS_MAKE_BAR + "\n\n" + report_js,
        extra_css=BAR_CSS,
        max_width=1100,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Architecture analysis: block depth, layer conductance, effective rank"
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
        "--skip-heads", action="store_true",
        help="Skip head-layer analysis (conductance + SVD for policy/value head layers)",
    )
    parser.add_argument(
        "--block-detail", action="store_true",
        help="Show per-sublayer SVD within each residual block (norm, fc1, fc2)",
    )
    parser.add_argument("--conductance-samples", type=int, default=200)
    parser.add_argument("--conductance-steps", type=int, default=20)
    parser.add_argument(
        "--output", type=str, default=None,
        help="HTML output path (default: interp/data/arch_epoch<N>.html)",
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

    # --- 1. Layer contribution ---
    print("\n" + "=" * 65)
    print("  1. LAYER CONTRIBUTION")
    print("=" * 65)

    print("\n  Input preprocessing: ||output|| / ||input|| per layer")
    t0 = time.perf_counter()
    preprocess = analyze_preprocess_contributions(
        model, device, dataset.states, batch_size=args.batch_size,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    _print_preprocess_contributions(preprocess)

    print("\n  Residual blocks: ||residual|| / ||input|| per block (higher = more active)")
    t0 = time.perf_counter()
    contributions = analyze_block_contributions(
        model, device, dataset.states, batch_size=args.batch_size,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    _print_contributions(contributions)

    # --- 2. Trunk block conductance ---
    print("\n" + "=" * 65)
    print("  2. TRUNK BLOCK CONDUCTANCE (Captum)")
    print("=" * 65)
    print(
        f"  Integrated-gradient conductance toward policy/value sum"
        f" ({args.conductance_samples} samples, {args.conductance_steps} steps)"
    )
    t0 = time.perf_counter()
    conductance = analyze_layer_conductance(
        model, device, dataset.states,
        max_samples=args.conductance_samples,
        n_steps=args.conductance_steps,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    _print_conductance(conductance)

    # --- 2b. Head-layer conductance ---
    head_conductance = None
    if not args.skip_heads:
        print("\n" + "=" * 65)
        print("  2b. HEAD-LAYER CONDUCTANCE (Captum)")
        print("=" * 65)
        print("  Conductance within each head's Linear layers")
        t0 = time.perf_counter()
        head_conductance = analyze_head_conductance(
            model, device, dataset.states,
            max_samples=args.conductance_samples,
            n_steps=args.conductance_steps,
        )
        print(f"  ({time.perf_counter() - t0:.1f}s)")
        _print_head_conductance(head_conductance)

    # --- 3. Effective rank ---
    print("\n" + "=" * 65)
    print("  3. EFFECTIVE RANK (SVD)")
    print("=" * 65)
    print("  Activation dimensionality at each layer")
    t0 = time.perf_counter()
    ranks = analyze_effective_rank(
        model, device, dataset.states, batch_size=args.batch_size,
        include_heads=not args.skip_heads,
        include_block_detail=args.block_detail,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    hidden_dim: int = model.cfg.hidden_dim  # type: ignore[union-attr]
    _print_ranks(ranks, hidden_dim)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    # Block contribution trend
    contribs = [r["mean"] for r in contributions]
    peak_block = int(np.argmax(contribs))
    min_block = int(np.argmin(contribs))
    last3_mean = float(np.mean(contribs[-3:]))
    first3_mean = float(np.mean(contribs[:3]))
    print(f"\n  Block contribution:")
    print(f"    Most active block: {peak_block} ({contribs[peak_block]:.4f})")
    print(f"    Least active block: {min_block} ({contribs[min_block]:.4f})")
    print(f"    First 3 blocks avg: {first3_mean:.4f}")
    print(f"    Last 3 blocks avg:  {last3_mean:.4f}")
    if last3_mean < first3_mean * 0.3:
        print(f"    -> Last blocks contribute <30% of first blocks — may be prunable")
    elif last3_mean > first3_mean * 0.8:
        print(f"    -> Contribution is flat — all blocks are active, more depth may help")

    # Conductance
    pol_pcts = [r["pct"] for r in conductance["policy"]]
    val_pcts = [r["pct"] for r in conductance["value"]]
    n_blocks = len(pol_pcts)
    pol_top3 = sorted(range(n_blocks), key=lambda i: -pol_pcts[i])[:3]
    val_top3 = sorted(range(n_blocks), key=lambda i: -val_pcts[i])[:3]
    print(f"\n  Layer conductance:")
    print(f"    Policy top-3 blocks: {pol_top3} ({sum(pol_pcts[i] for i in pol_top3):.1f}% of total)")
    print(f"    Value top-3 blocks:  {val_top3} ({sum(val_pcts[i] for i in val_top3):.1f}% of total)")
    if set(pol_top3) != set(val_top3):
        print(f"    -> Heads use different blocks — blocks may specialize by task")

    # Effective rank
    trunk_rank = None
    for r in ranks:
        if r["layer"] == "trunk_norm":
            trunk_rank = r
    if trunk_rank is not None:
        er = trunk_rank["eff_rank"]
        assert isinstance(er, float)
        utilization = er / hidden_dim * 100
        print(f"\n  Width utilization:")
        print(f"    Trunk output effective rank: {er:.0f} / {hidden_dim} ({utilization:.1f}%)")
        t200 = trunk_rank["top200_energy"]
        assert isinstance(t200, float)
        print(f"    Top-200 components capture: {t200:.1f}% of variance")
        if utilization < 40:
            print(f"    -> Low utilization — hidden_dim could likely be reduced")
        elif utilization > 70:
            print(f"    -> High utilization — current width is well-matched or tight")

    # --- HTML report ---
    if args.output:
        html_path = Path(args.output)
    else:
        html_path = Path("interp/data") / f"arch_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = format_html_report(
        contributions, preprocess, ranks, hidden_dim, epoch, dataset.num_states,
        conductance=conductance,
        head_conductance=head_conductance,
    )
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
