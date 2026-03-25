"""Value head analysis: value lens, per-player neuron specialization, error stratification, causal necessity.

Four analyses to understand the value head's internal organization:

1. **Value Lens**: Project trunk output directly through the final projection
   (skipping V0+GELU) to see if the hidden layer adds meaningful computation.

2. **Per-Player Neuron Specialization**: NeuronConductance at V0's GELU output
   toward each of the 3 per-player value outputs.

3. **Phase-Stratified Value Error**: Where is value prediction weakest?
   Stratify by phase and game progress (early/mid/late).

4. **Layer Causal Necessity**: Bypass V0+GELU and measure value output change
   per phase.

Usage:
    .venv/bin/python -m interp.layers.value_head --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp.utils import (
    InterpDataset,
    collect_states,
    forward_batched,
    load_model,
)


_PHASE_NAMES = {
    0: "INVEST", 1: "BID", 3: "ACQ", 4: "CLOSE",
    6: "DIV", 8: "ISSUE", 9: "IPO", 10: "PAR",
}


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _collect_value_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Collect trunk output and V0 GELU output.

    Value head layout: [0] Linear 256->256, [1] GELU, [2] Linear 256->3, [3] Tanh.
    Returns {"trunk": ..., "V0": ...} as numpy arrays.
    """
    model.eval()
    acts: dict[str, list[torch.Tensor]] = {"trunk": [], "V0": []}
    handles = []

    def hook_trunk(
        _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        acts["trunk"].append(out.detach().cpu())

    def hook_v0(
        _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        acts["V0"].append(out.detach().cpu())

    handles.append(model.trunk_norm.register_forward_hook(hook_trunk))
    handles.append(model.value_head[1].register_forward_hook(hook_v0))  # GELU

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    return {k: torch.cat(v, dim=0).numpy() for k, v in acts.items()}


# ---------------------------------------------------------------------------
# 1. Value Lens
# ---------------------------------------------------------------------------


@dataclass
class ValueLensResult:
    """Compare trunk->V2 (direct) vs trunk->V0->V2 (full head)."""

    # Per-player MSE between direct and full-head values
    direct_vs_full_mse: list[float]  # (3,) per player
    # Per-player correlation between direct and full-head values
    direct_vs_full_corr: list[float]  # (3,)
    # Value range stats
    full_values_mean: list[float]  # (3,)
    full_values_std: list[float]  # (3,)
    direct_values_mean: list[float]  # (3,)
    direct_values_std: list[float]  # (3,)
    # Overall MSE
    overall_mse: float
    overall_corr: float


def analyze_value_lens(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
) -> ValueLensResult:
    """Project trunk output directly through V2 (skipping V0+GELU).

    Compares the "direct" value predictions to the full value head output.
    """
    print("  Collecting activations...")
    activations = _collect_value_activations(model, device, dataset.states, batch_size)
    trunk = activations["trunk"]  # (N, 256)

    # Full head output
    _, full_values = forward_batched(model, device, dataset.states, batch_size)

    # Direct projection: trunk -> V2 -> Tanh (skip V0+GELU)
    W = model.value_head[2].weight.detach().cpu().numpy()  # (3, 256)
    b = model.value_head[2].bias.detach().cpu().numpy()  # (3,)
    direct_raw = trunk @ W.T + b  # (N, 3)
    direct_values = np.tanh(direct_raw)

    # Per-player comparison
    per_player_mse = []
    per_player_corr = []
    for p in range(3):
        mse = float(np.mean((full_values[:, p] - direct_values[:, p]) ** 2))
        corr = float(np.corrcoef(full_values[:, p], direct_values[:, p])[0, 1])
        per_player_mse.append(mse)
        per_player_corr.append(corr)

    overall_mse = float(np.mean((full_values - direct_values) ** 2))
    # Flatten for overall correlation
    overall_corr = float(np.corrcoef(
        full_values.ravel(), direct_values.ravel()
    )[0, 1])

    return ValueLensResult(
        direct_vs_full_mse=per_player_mse,
        direct_vs_full_corr=per_player_corr,
        full_values_mean=[float(full_values[:, p].mean()) for p in range(3)],
        full_values_std=[float(full_values[:, p].std()) for p in range(3)],
        direct_values_mean=[float(direct_values[:, p].mean()) for p in range(3)],
        direct_values_std=[float(direct_values[:, p].std()) for p in range(3)],
        overall_mse=overall_mse,
        overall_corr=overall_corr,
    )


# ---------------------------------------------------------------------------
# 2. Per-Player Neuron Specialization
# ---------------------------------------------------------------------------


@dataclass
class ValueNeuronResult:
    """Per-player neuron specialization in V0."""

    num_neurons: int
    # (num_neurons, 3) conductance matrix
    conductance_matrix: np.ndarray
    # Per neuron: which player it primarily serves
    dominant_player: np.ndarray  # (num_neurons,)
    neurons_per_player: list[int]  # (3,) count
    dead_fraction: float
    # Specialization: max / sum per neuron
    specialization_scores: np.ndarray


def analyze_value_neuron_specialization(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    max_samples: int = 300,
    n_steps: int = 20,
) -> ValueNeuronResult:
    """NeuronConductance at V0 GELU toward each player's value output."""
    from captum.attr import NeuronConductance

    model.eval()

    states = dataset.states
    if states.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(states.shape[0], max_samples, replace=False)
        states = states[idx]

    inputs = torch.from_numpy(states).to(device).requires_grad_(True)
    target_layer = model.value_head[1]  # GELU output
    num_neurons = 256
    internal_batch_size = 2 * states.shape[0]

    conductance_matrix = np.zeros((num_neurons, 3), dtype=np.float32)

    t0 = time.perf_counter()
    for player_idx in range(3):
        def make_forward(pi: int):  # noqa: ANN202
            def forward_fn(x: torch.Tensor) -> torch.Tensor:
                _, v = model(x)
                return v[:, pi]
            return forward_fn

        fwd = make_forward(player_idx)

        for ni in range(num_neurons):
            nc = NeuronConductance(fwd, target_layer)
            attr = nc.attribute(
                inputs, neuron_selector=ni,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
            assert isinstance(attr, torch.Tensor)
            conductance_matrix[ni, player_idx] = float(attr.abs().mean().item())

        elapsed = time.perf_counter() - t0
        print(f"    Player {player_idx} ({elapsed:.1f}s)")

    dominant_player = np.argmax(conductance_matrix, axis=1)
    neurons_per_player = [int(np.sum(dominant_player == p)) for p in range(3)]

    max_per_neuron = conductance_matrix.max(axis=1)
    global_max = max_per_neuron.max()
    dead_fraction = float(np.mean(max_per_neuron < 0.05 * global_max))

    row_sums = conductance_matrix.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-10, None)
    specialization_scores = conductance_matrix.max(axis=1) / row_sums.squeeze()

    return ValueNeuronResult(
        num_neurons=num_neurons,
        conductance_matrix=conductance_matrix,
        dominant_player=dominant_player,
        neurons_per_player=neurons_per_player,
        dead_fraction=dead_fraction,
        specialization_scores=specialization_scores,
    )


# ---------------------------------------------------------------------------
# 3. Phase-Stratified Value Error
# ---------------------------------------------------------------------------


@dataclass
class ValueErrorResult:
    """Value prediction error stratified by phase and game progress."""

    # Per-phase: mean abs value, std of values
    phase_stats: dict[int, dict[str, float]]
    # Per-phase: value spread (std across 3 players)
    phase_spread: dict[int, float]
    # Game progress bins: (early, mid, late) -> value stats
    progress_stats: dict[str, dict[str, float]]
    # Overall stats
    overall_mean_abs: float
    overall_spread: float


def analyze_value_error(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
) -> ValueErrorResult:
    """Stratify value outputs by phase and game progress."""
    _, values = forward_batched(model, device, dataset.states, batch_size)
    phases = dataset.phases

    # Per-phase stats
    phase_ids = sorted(set(phases))
    phase_stats: dict[int, dict[str, float]] = {}
    phase_spread: dict[int, float] = {}

    for pid in phase_ids:
        mask = phases == pid
        v = values[mask]
        phase_stats[pid] = {
            "count": int(mask.sum()),
            "mean_p0": float(v[:, 0].mean()),
            "mean_p1": float(v[:, 1].mean()),
            "mean_p2": float(v[:, 2].mean()),
            "std_p0": float(v[:, 0].std()),
            "mean_abs": float(np.abs(v).mean()),
            "value_range": float(v.max() - v.min()),
        }
        phase_spread[pid] = float(np.std(v, axis=1).mean())

    # Game progress bins based on co:removed count (proxy for game stage)
    # More removed companies = later game
    # We use the sum of the full state to estimate progress
    # Simple approach: tertile split by state index within each game
    n = len(phases)
    thirds = np.zeros(n, dtype=int)  # 0=early, 1=mid, 2=late
    # Use the visible_size to find removed companies offset for a better proxy
    # But for simplicity, just use position within dataset (states are in game order)
    game_breaks: list[int] = [0]
    for i in range(1, n):
        # New game when phase resets to INVEST after a different phase
        if phases[i] == 0 and i > 0 and phases[i - 1] != 0:
            game_breaks.append(i)
    game_breaks.append(n)

    for gi in range(len(game_breaks) - 1):
        start, end = game_breaks[gi], game_breaks[gi + 1]
        game_len = end - start
        if game_len < 3:
            continue
        t1 = start + game_len // 3
        t2 = start + 2 * game_len // 3
        thirds[start:t1] = 0
        thirds[t1:t2] = 1
        thirds[t2:end] = 2

    progress_names = {0: "early", 1: "mid", 2: "late"}
    progress_stats: dict[str, dict[str, float]] = {}
    for stage, name in progress_names.items():
        mask = thirds == stage
        v = values[mask]
        progress_stats[name] = {
            "count": int(mask.sum()),
            "mean_p0": float(v[:, 0].mean()),
            "std_p0": float(v[:, 0].std()),
            "mean_abs": float(np.abs(v).mean()),
            "spread": float(np.std(v, axis=1).mean()),
            "confidence": float(np.abs(v).max(axis=1).mean()),
        }

    return ValueErrorResult(
        phase_stats=phase_stats,
        phase_spread=phase_spread,
        progress_stats=progress_stats,
        overall_mean_abs=float(np.abs(values).mean()),
        overall_spread=float(np.std(values, axis=1).mean()),
    )


# ---------------------------------------------------------------------------
# 4. Layer Causal Necessity
# ---------------------------------------------------------------------------


@dataclass
class ValueNecessityResult:
    """Causal effect of bypassing V0+GELU."""

    # Overall MSE when V0+GELU bypassed
    bypass_mse: float
    # Per-player MSE
    per_player_mse: list[float]
    # Per-phase MSE
    phase_mse: dict[int, float]
    # Correlation between bypassed and full values
    bypass_corr: float


def analyze_value_necessity(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
) -> ValueNecessityResult:
    """Replace V0+GELU with identity and measure value output change."""
    model.eval()

    # Original values
    _, orig_values = forward_batched(model, device, dataset.states, batch_size)

    # Bypass V0+GELU
    orig_linear = model.value_head[0]
    orig_gelu = model.value_head[1]
    model.value_head[0] = torch.nn.Identity()
    model.value_head[1] = torch.nn.Identity()

    _, bypass_values = forward_batched(model, device, dataset.states, batch_size)

    # Restore
    model.value_head[0] = orig_linear
    model.value_head[1] = orig_gelu

    # Metrics
    diff = orig_values - bypass_values
    overall_mse = float(np.mean(diff ** 2))
    per_player = [float(np.mean(diff[:, p] ** 2)) for p in range(3)]

    bypass_corr = float(np.corrcoef(
        orig_values.ravel(), bypass_values.ravel()
    )[0, 1])

    phases = dataset.phases
    phase_ids = sorted(set(phases))
    phase_mse: dict[int, float] = {}
    for pid in phase_ids:
        mask = phases == pid
        phase_mse[pid] = float(np.mean(diff[mask] ** 2))

    return ValueNecessityResult(
        bypass_mse=overall_mse,
        per_player_mse=per_player,
        phase_mse=phase_mse,
        bypass_corr=bypass_corr,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_value_lens_report(result: ValueLensResult) -> None:
    print("\n" + "=" * 90)
    print("  1. VALUE LENS (trunk -> V2 directly vs trunk -> V0 -> V2)")
    print("=" * 90)

    print(f"\n  Overall: MSE={result.overall_mse:.6f}, correlation={result.overall_corr:.4f}")

    print(f"\n  {'Player':<10s} {'MSE':>10s} {'Corr':>8s} {'Full mean':>10s} {'Full std':>10s} {'Direct mean':>12s} {'Direct std':>11s}")
    print(f"  {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 11}")
    for p in range(3):
        print(
            f"  Player {p:<3d} {result.direct_vs_full_mse[p]:>10.6f} "
            f"{result.direct_vs_full_corr[p]:>8.4f} "
            f"{result.full_values_mean[p]:>10.4f} {result.full_values_std[p]:>10.4f} "
            f"{result.direct_values_mean[p]:>12.4f} {result.direct_values_std[p]:>11.4f}"
        )


def print_neuron_spec_report(result: ValueNeuronResult) -> None:
    print("\n" + "=" * 90)
    print("  2. PER-PLAYER NEURON SPECIALIZATION")
    print("=" * 90)

    print(f"\n  Dead neurons: {result.dead_fraction:.1%}")
    print(f"  Mean specialization: {result.specialization_scores.mean():.3f}")

    print(f"\n  Neurons per player:")
    for p in range(3):
        n = result.neurons_per_player[p]
        bar = "#" * (n * 50 // result.num_neurons)
        print(f"    Player {p}: {n:>3d} / {result.num_neurons}  {bar}")

    # Top conductance neurons per player
    print(f"\n  Top-5 neurons per player:")
    for p in range(3):
        col = result.conductance_matrix[:, p]
        top5 = np.argsort(col)[-5:][::-1]
        vals = col[top5]
        print(f"    Player {p}: " + ", ".join(f"n{n}={v:.4f}" for n, v in zip(top5, vals)))


def print_error_report(result: ValueErrorResult) -> None:
    print("\n" + "=" * 90)
    print("  3. PHASE-STRATIFIED VALUE CHARACTERISTICS")
    print("=" * 90)

    print(f"\n  Overall: mean|v|={result.overall_mean_abs:.4f}, spread={result.overall_spread:.4f}")

    decision_phases = [p for p in sorted(result.phase_stats.keys()) if p in _PHASE_NAMES]
    print(f"\n  {'Phase':<8s} {'Count':>6s} {'mean|v|':>8s} {'Spread':>8s} {'V(p0)':>8s} {'V(p1)':>8s} {'V(p2)':>8s}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for pid in decision_phases:
        s = result.phase_stats[pid]
        sp = result.phase_spread[pid]
        print(
            f"  {_PHASE_NAMES[pid]:<8s} {s['count']:>6d} {s['mean_abs']:>8.4f} {sp:>8.4f} "
            f"{s['mean_p0']:>8.4f} {s['mean_p1']:>8.4f} {s['mean_p2']:>8.4f}"
        )

    print(f"\n  Game progress:")
    print(f"  {'Stage':<8s} {'Count':>6s} {'mean|v|':>8s} {'Spread':>8s} {'Confidence':>11s}")
    print(f"  {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 11}")
    for name in ["early", "mid", "late"]:
        s = result.progress_stats[name]
        print(
            f"  {name:<8s} {s['count']:>6d} {s['mean_abs']:>8.4f} {s['spread']:>8.4f} "
            f"{s['confidence']:>11.4f}"
        )


def print_necessity_report(result: ValueNecessityResult) -> None:
    print("\n" + "=" * 90)
    print("  4. LAYER CAUSAL NECESSITY (bypass V0+GELU)")
    print("=" * 90)

    print(f"\n  Bypass MSE: {result.bypass_mse:.6f}")
    print(f"  Bypass correlation: {result.bypass_corr:.4f}")
    print(f"\n  Per-player MSE:")
    for p in range(3):
        print(f"    Player {p}: {result.per_player_mse[p]:.6f}")

    decision_phases = [p for p in sorted(result.phase_mse.keys()) if p in _PHASE_NAMES]
    print(f"\n  Per-phase MSE:")
    for pid in decision_phases:
        mse = result.phase_mse[pid]
        print(f"    {_PHASE_NAMES[pid]:<8s}: {mse:.6f}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    lens: ValueLensResult,
    neuron: ValueNeuronResult,
    error: ValueErrorResult,
    necessity: ValueNecessityResult,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    lens_json = json.dumps({
        "overall_mse": lens.overall_mse,
        "overall_corr": lens.overall_corr,
        "per_player": [
            {
                "player": p,
                "mse": lens.direct_vs_full_mse[p],
                "corr": lens.direct_vs_full_corr[p],
                "full_mean": lens.full_values_mean[p],
                "full_std": lens.full_values_std[p],
                "direct_mean": lens.direct_values_mean[p],
                "direct_std": lens.direct_values_std[p],
            }
            for p in range(3)
        ],
    })

    neuron_json = json.dumps({
        "num_neurons": neuron.num_neurons,
        "dead_fraction": neuron.dead_fraction,
        "mean_spec": float(neuron.specialization_scores.mean()),
        "median_spec": float(np.median(neuron.specialization_scores)),
        "neurons_per_player": neuron.neurons_per_player,
        "top_neurons": [
            [
                {"neuron": int(n), "cond": float(neuron.conductance_matrix[n, p])}
                for n in np.argsort(neuron.conductance_matrix[:, p])[-5:][::-1]
            ]
            for p in range(3)
        ],
    })

    decision_phases = [p for p in sorted(error.phase_stats.keys()) if p in _PHASE_NAMES]
    error_json = json.dumps({
        "overall_mean_abs": error.overall_mean_abs,
        "overall_spread": error.overall_spread,
        "phases": [
            {
                "phase": _PHASE_NAMES[pid],
                "count": error.phase_stats[pid]["count"],
                "mean_abs": error.phase_stats[pid]["mean_abs"],
                "spread": error.phase_spread[pid],
                "mean_p0": error.phase_stats[pid]["mean_p0"],
                "mean_p1": error.phase_stats[pid]["mean_p1"],
                "mean_p2": error.phase_stats[pid]["mean_p2"],
            }
            for pid in decision_phases
        ],
        "progress": [
            {"stage": name, **error.progress_stats[name]}
            for name in ["early", "mid", "late"]
        ],
    })

    necessity_json = json.dumps({
        "bypass_mse": necessity.bypass_mse,
        "bypass_corr": necessity.bypass_corr,
        "per_player": necessity.per_player_mse,
        "phases": [
            {"phase": _PHASE_NAMES[pid], "mse": necessity.phase_mse[pid]}
            for pid in decision_phases
        ],
    })

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Value Head Analysis — Epoch {epoch}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #1a1a2e; color: #e0e0e0;
    margin: 2rem auto; max-width: 1100px; padding: 0 1rem;
  }}
  h1 {{ color: #f0f0f0; font-size: 1.4rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #ccc; font-size: 1.1rem; margin-top: 2rem;
        border-bottom: 1px solid #333; padding-bottom: 0.3rem; }}
  h3 {{ color: #aaa; font-size: 0.95rem; margin-top: 1rem; }}
  .meta {{ color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 0.82rem; margin-bottom: 1.5rem;
  }}
  th, td {{ padding: 5px 8px; border: 1px solid #2a2a4a; text-align: right; }}
  th {{ background: #16213e; color: #aaa; font-weight: 600; }}
  th:first-child, td:first-child {{ text-align: left; }}
  td:first-child {{
    font-family: "SF Mono", "Fira Code", Consolas, monospace;
    font-size: 0.8rem; color: #ccc;
  }}
  tr:hover td {{ border-color: #555; }}
  .bar-container {{ display: inline-block; width: 120px; vertical-align: middle; }}
  .bar {{
    display: inline-block; height: 12px; border-radius: 2px;
    vertical-align: middle;
  }}
  .bar-blue {{ background: #4a9eff; }}
  .bar-green {{ background: #4ecca3; }}
  .bar-orange {{ background: #e9a945; }}
  .stat-box {{
    display: inline-block; background: #16213e; border: 1px solid #2a2a4a;
    border-radius: 4px; padding: 8px 16px; margin: 4px 8px 4px 0;
    font-size: 0.85rem;
  }}
  .stat-label {{ color: #888; font-size: 0.75rem; }}
  .stat-value {{ color: #e0e0e0; font-size: 1.1rem; font-weight: 600; }}
</style>
</head>
<body>
<h1>Value Head Analysis — Epoch {epoch}</h1>
<div class="meta">
  {num_states:,} states from {num_games} games.
  Value head: Linear(256,256) &rarr; GELU &rarr; Linear(256,3) &rarr; Tanh.
</div>

<h2>1. Value Lens</h2>
<p style="color:#888;font-size:0.85rem">Compare trunk&rarr;V2 (skip V0+GELU) vs full value head output.</p>
<div id="lens-stats"></div>
<table id="tbl-lens"></table>

<h2>2. Per-Player Neuron Specialization</h2>
<p style="color:#888;font-size:0.85rem">NeuronConductance at V0 GELU output toward each player's value.</p>
<div id="neuron-stats"></div>
<h3>Top-5 Neurons per Player</h3>
<table id="tbl-neurons"></table>

<h2>3. Value Characteristics by Phase</h2>
<p style="color:#888;font-size:0.85rem">Value output statistics stratified by phase and game progress.</p>
<h3>By Phase</h3>
<table id="tbl-phase"></table>
<h3>By Game Progress</h3>
<table id="tbl-progress"></table>

<h2>4. Layer Causal Necessity</h2>
<p style="color:#888;font-size:0.85rem">Value MSE when V0+GELU is replaced with identity.</p>
<div id="necessity-stats"></div>
<table id="tbl-necessity"></table>

<script>
const lens = {lens_json};
const neuron = {neuron_json};
const error = {error_json};
const necessity = {necessity_json};

function makeBar(val, maxVal, cls) {{
  const pct = maxVal > 0 ? Math.min(val / maxVal * 100, 100) : 0;
  return '<span class="bar-container"><span class="bar ' + cls + '" style="width:' + pct + '%"></span></span>';
}}

// --- 1. Value Lens ---
(function() {{
  const div = document.getElementById("lens-stats");
  div.innerHTML =
    '<div class="stat-box"><div class="stat-label">Overall MSE</div><div class="stat-value">' +
    lens.overall_mse.toFixed(6) + '</div></div>' +
    '<div class="stat-box"><div class="stat-label">Correlation</div><div class="stat-value">' +
    lens.overall_corr.toFixed(4) + '</div></div>';

  const tbl = document.getElementById("tbl-lens");
  let html = '<tr><th>Player</th><th>MSE</th><th>Corr</th><th>Full Mean</th><th>Full Std</th><th>Direct Mean</th><th>Direct Std</th></tr>';
  for (const p of lens.per_player) {{
    html += '<tr><td>Player ' + p.player + '</td>' +
      '<td>' + p.mse.toFixed(6) + '</td>' +
      '<td>' + p.corr.toFixed(4) + '</td>' +
      '<td>' + p.full_mean.toFixed(4) + '</td>' +
      '<td>' + p.full_std.toFixed(4) + '</td>' +
      '<td>' + p.direct_mean.toFixed(4) + '</td>' +
      '<td>' + p.direct_std.toFixed(4) + '</td></tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 2. Neuron Specialization ---
(function() {{
  const div = document.getElementById("neuron-stats");
  div.innerHTML =
    '<div class="stat-box"><div class="stat-label">Dead Neurons</div><div class="stat-value">' +
    (neuron.dead_fraction * 100).toFixed(1) + '%</div></div>' +
    '<div class="stat-box"><div class="stat-label">Mean Spec.</div><div class="stat-value">' +
    neuron.mean_spec.toFixed(3) + '</div></div>' +
    '<div class="stat-box"><div class="stat-label">Neurons P0/P1/P2</div><div class="stat-value">' +
    neuron.neurons_per_player.join(' / ') + '</div></div>';

  const tbl = document.getElementById("tbl-neurons");
  let html = '<tr><th>Player</th><th>#1</th><th>#2</th><th>#3</th><th>#4</th><th>#5</th></tr>';
  for (let p = 0; p < 3; p++) {{
    html += '<tr><td>Player ' + p + '</td>';
    for (const n of neuron.top_neurons[p]) {{
      html += '<td>n' + n.neuron + ' (' + n.cond.toFixed(4) + ')</td>';
    }}
    html += '</tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 3. Phase stats ---
(function() {{
  const tbl = document.getElementById("tbl-phase");
  let html = '<tr><th>Phase</th><th>Count</th><th>Mean |V|</th><th>Spread</th><th>V(p0)</th><th>V(p1)</th><th>V(p2)</th></tr>';
  for (const p of error.phases) {{
    html += '<tr><td>' + p.phase + '</td>' +
      '<td>' + p.count + '</td>' +
      '<td>' + p.mean_abs.toFixed(4) + '</td>' +
      '<td>' + p.spread.toFixed(4) + '</td>' +
      '<td>' + p.mean_p0.toFixed(4) + '</td>' +
      '<td>' + p.mean_p1.toFixed(4) + '</td>' +
      '<td>' + p.mean_p2.toFixed(4) + '</td></tr>';
  }}
  tbl.innerHTML = html;

  const tbl2 = document.getElementById("tbl-progress");
  let html2 = '<tr><th>Stage</th><th>Count</th><th>Mean |V|</th><th>Spread</th><th>Confidence</th></tr>';
  for (const p of error.progress) {{
    html2 += '<tr><td>' + p.stage + '</td>' +
      '<td>' + p.count + '</td>' +
      '<td>' + p.mean_abs.toFixed(4) + '</td>' +
      '<td>' + p.spread.toFixed(4) + '</td>' +
      '<td>' + p.confidence.toFixed(4) + '</td></tr>';
  }}
  tbl2.innerHTML = html2;
}})();

// --- 4. Necessity ---
(function() {{
  const div = document.getElementById("necessity-stats");
  div.innerHTML =
    '<div class="stat-box"><div class="stat-label">Bypass MSE</div><div class="stat-value">' +
    necessity.bypass_mse.toFixed(6) + '</div></div>' +
    '<div class="stat-box"><div class="stat-label">Bypass Corr</div><div class="stat-value">' +
    necessity.bypass_corr.toFixed(4) + '</div></div>';

  const tbl = document.getElementById("tbl-necessity");
  const maxMSE = Math.max(...necessity.phases.map(p => p.mse));
  let html = '<tr><th>Phase</th><th>MSE when bypassed</th><th></th></tr>';
  for (const p of necessity.phases) {{
    html += '<tr><td>' + p.phase + '</td>' +
      '<td>' + p.mse.toFixed(6) + '</td>' +
      '<td>' + makeBar(p.mse, maxMSE, 'bar-orange') + '</td></tr>';
  }}
  html += '<tr style="border-top:2px solid #444"><td>Per-player</td><td colspan="2">' +
    necessity.per_player.map((v,i) => 'P' + i + '=' + v.toFixed(6)).join(', ') + '</td></tr>';
  tbl.innerHTML = html;
}})();
</script>
</body>
</html>"""


def _open_file(path: Path) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Value head analysis (value lens, neuron specialization, error stratification, causal necessity)"
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
        "--max-neuron-samples", type=int, default=300,
        help="Max states for neuron conductance",
    )
    parser.add_argument(
        "--neuron-steps", type=int, default=20,
        help="IG steps for neuron conductance",
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

    # --- Analysis 1: Value Lens ---
    print(f"\nRunning value lens analysis...")
    lens_result = analyze_value_lens(
        model, device, dataset, batch_size=args.batch_size,
    )
    print_value_lens_report(lens_result)

    # --- Analysis 2: Neuron Specialization ---
    print(f"\nRunning per-player neuron specialization (this is slow)...")
    neuron_result = analyze_value_neuron_specialization(
        model, device, dataset,
        max_samples=args.max_neuron_samples,
        n_steps=args.neuron_steps,
    )
    print_neuron_spec_report(neuron_result)

    # --- Analysis 3: Value Error Stratification ---
    print(f"\nRunning value error stratification...")
    error_result = analyze_value_error(
        model, device, dataset, batch_size=args.batch_size,
    )
    print_error_report(error_result)

    # --- Analysis 4: Layer Necessity ---
    print(f"\nRunning layer causal necessity...")
    necessity_result = analyze_value_necessity(
        model, device, dataset, batch_size=args.batch_size,
    )
    print_necessity_report(necessity_result)

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")

    print(f"\n  Value lens (V0 contribution):")
    print(f"    Direct vs full MSE: {lens_result.overall_mse:.6f}")
    print(f"    Correlation: {lens_result.overall_corr:.4f}")
    if lens_result.overall_corr > 0.99:
        print(f"    -> V0 adds minimal nonlinear computation (corr > 0.99)")
    else:
        print(f"    -> V0 provides meaningful transformation")

    print(f"\n  Neuron utilization:")
    print(f"    Dead: {neuron_result.dead_fraction:.1%}")
    print(f"    Player balance: {neuron_result.neurons_per_player}")

    print(f"\n  Layer necessity:")
    print(f"    Bypass MSE: {necessity_result.bypass_mse:.6f}")
    print(f"    Bypass corr: {necessity_result.bypass_corr:.4f}")

    # --- HTML report ---
    html_path = Path("interp/data") / f"value_head_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = format_html_report(
        lens_result, neuron_result, error_result, necessity_result,
        epoch=epoch, num_states=dataset.num_states, num_games=dataset.num_games,
    )
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        _open_file(html_path)


if __name__ == "__main__":
    main()
