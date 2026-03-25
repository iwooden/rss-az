"""Policy head analysis: logit lens, neuron specialization, layer causal necessity.

Three analyses to understand the policy head's internal organization:

1. **Logit Lens**: Project intermediate policy representations through the final
   weight matrix to see when decisions crystallize and whether later layers
   refine or override earlier commitments.

2. **Neuron Specialization**: NeuronConductance per action type reveals which
   neurons serve which actions. Measures functional width utilization.

3. **Layer Causal Necessity**: Replace each layer with a skip connection and
   measure policy KL per phase. Shows which layers are causally necessary for
   which phases (beyond gradient-based conductance).

Usage:
    .venv/bin/python -m interp.layers.policy_head --load-data interp/data/states.npz
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
    batch_masked_softmax,
    collect_states,
    forward_batched,
    kl_divergence_batch,
    load_model,
)


_PHASE_NAMES = {
    0: "INVEST", 1: "BID", 3: "ACQ", 4: "CLOSE",
    6: "DIV", 8: "ISSUE", 9: "IPO", 10: "PAR",
}

# Action type buckets for neuron specialization
_ACTION_TYPES = {
    0: "pass", 1: "auction", 2: "buy", 3: "sell",
    4: "leave_bid", 5: "raise_bid", 6: "acq_price", 7: "acq_fi_buy",
    8: "close", 9: "dividend", 10: "issue", 11: "ipo", 12: "par",
}


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _collect_policy_layer_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Collect activations after each GELU in the policy head, plus trunk output.

    Returns dict with keys: "trunk", "P0" (after GELU[1]), "P2" (after GELU[3]),
    "P4" (after GELU[5]).
    """
    model.eval()
    acts: dict[str, list[torch.Tensor]] = {
        "trunk": [], "P0": [], "P2": [], "P4": [],
    }
    handles = []

    # trunk_norm output
    def hook_trunk(
        _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
    ) -> None:
        acts["trunk"].append(out.detach().cpu())

    handles.append(model.trunk_norm.register_forward_hook(hook_trunk))

    # Policy head GELU outputs: indices 1, 3, 5
    for name, idx in [("P0", 1), ("P2", 3), ("P4", 5)]:
        def make_hook(n: str):  # noqa: ANN202
            def hook(
                _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
            ) -> None:
                acts[n].append(out.detach().cpu())
            return hook
        handles.append(model.policy_head[idx].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    return {k: torch.cat(v, dim=0).numpy() for k, v in acts.items()}


# ---------------------------------------------------------------------------
# 1. Logit Lens
# ---------------------------------------------------------------------------


@dataclass
class LogitLensResult:
    """Per-layer logit lens results."""

    layer_names: list[str]
    # Per-layer agreement with final policy (fraction of states where argmax matches)
    argmax_agreement: dict[str, float]
    # Per-layer KL divergence from final policy
    mean_kl: dict[str, float]
    # Per-phase argmax agreement
    phase_agreement: dict[str, dict[int, float]]
    # Per-layer top-1 probability of the final argmax action
    final_action_prob: dict[str, float]


def analyze_logit_lens(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
) -> LogitLensResult:
    """Project intermediate policy representations through the final weight matrix.

    At each policy layer, applies W_final @ h + b_final to get "early logits",
    then compares the resulting policy to the actual final policy.
    """
    print("  Collecting policy layer activations...")
    activations = _collect_policy_layer_activations(
        model, device, dataset.states, batch_size
    )

    # Get the final projection weights
    final_layer = model.policy_head[6]  # Linear 256→183
    W = final_layer.weight.detach().cpu().numpy()  # (183, 256)
    b = final_layer.bias.detach().cpu().numpy()  # (183,)

    # Get actual final policy
    print("  Computing final policy...")
    logits, _ = forward_batched(model, device, dataset.states, batch_size)
    final_pol = batch_masked_softmax(logits, dataset.legal_masks)
    final_argmax = np.argmax(logits + np.where(dataset.legal_masks > 0, 0, -1e9), axis=1)

    layer_names = ["trunk", "P0", "P2", "P4"]
    argmax_agreement: dict[str, float] = {}
    mean_kl: dict[str, float] = {}
    phase_agreement: dict[str, dict[int, float]] = {}
    final_action_prob: dict[str, float] = {}

    phases = dataset.phases
    phase_ids = sorted(set(phases))

    print("  Computing logit lens at each layer...")
    for name in layer_names:
        # Project through final weight matrix
        h = activations[name]  # (N, 256)
        early_logits = h @ W.T + b  # (N, 183)

        # Apply legal mask and softmax
        early_pol = batch_masked_softmax(early_logits, dataset.legal_masks)
        early_argmax = np.argmax(
            early_logits + np.where(dataset.legal_masks > 0, 0, -1e9), axis=1,
        )

        # Argmax agreement with final policy
        agree = float(np.mean(early_argmax == final_argmax))
        argmax_agreement[name] = agree

        # KL divergence from final policy
        kl = float(np.mean(kl_divergence_batch(final_pol, early_pol)))
        mean_kl[name] = kl

        # Per-phase agreement
        pa: dict[int, float] = {}
        for pid in phase_ids:
            mask = phases == pid
            if mask.sum() > 0:
                pa[pid] = float(np.mean(early_argmax[mask] == final_argmax[mask]))
        phase_agreement[name] = pa

        # Probability assigned to the final argmax action at this layer
        probs_for_final = early_pol[np.arange(len(final_argmax)), final_argmax]
        final_action_prob[name] = float(np.mean(probs_for_final))

    return LogitLensResult(
        layer_names=layer_names,
        argmax_agreement=argmax_agreement,
        mean_kl=mean_kl,
        phase_agreement=phase_agreement,
        final_action_prob=final_action_prob,
    )


# ---------------------------------------------------------------------------
# 2. Neuron Specialization
# ---------------------------------------------------------------------------


@dataclass
class NeuronSpecResult:
    """Neuron specialization analysis results."""

    num_neurons: int
    # (num_neurons, num_action_types) conductance matrix
    conductance_matrix: np.ndarray
    action_type_names: list[str]
    action_type_ids: list[int]
    # Per-neuron: dominant action type index
    dominant_action: np.ndarray
    # Per action type: how many neurons are dominated by it
    neurons_per_action: dict[str, int]
    # Fraction of neurons that are "dead" (max conductance < threshold)
    dead_fraction: float
    # Specialization score per neuron (max / sum, 1.0 = fully specialized)
    specialization_scores: np.ndarray


def analyze_neuron_specialization(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    max_samples: int = 300,
    n_steps: int = 20,
) -> NeuronSpecResult:
    """Compute NeuronConductance for each neuron in the final hidden layer
    toward each action type's logit sum.

    Uses the GELU output before P6 (policy_head[5]) as the target layer.
    """
    from captum.attr import NeuronConductance
    from core.actions import decode_action_py

    model.eval()

    # Subsample states for speed
    states = dataset.states
    masks = dataset.legal_masks
    if states.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(states.shape[0], max_samples, replace=False)
        states = states[idx]
        masks = masks[idx]

    # Determine which action types are present
    logits, _ = forward_batched(model, device, states, 256)
    masked_logits = logits + np.where(masks > 0, 0, -1e9)
    argmax_actions = np.argmax(masked_logits, axis=1)

    action_types = np.array([
        decode_action_py(int(a), num_players)[1] for a in argmax_actions
    ])
    unique_types = sorted(set(action_types))

    # Build action type -> logit indices mapping
    # For each action type, find all action indices that map to it
    all_action_indices: dict[int, list[int]] = {t: [] for t in unique_types}
    action_dim = logits.shape[1]
    for a_idx in range(action_dim):
        try:
            _, atype, *_ = decode_action_py(a_idx, num_players)
            if atype in all_action_indices:
                all_action_indices[atype].append(a_idx)
        except (ValueError, IndexError):
            continue

    inputs = torch.from_numpy(states).to(device).requires_grad_(True)
    target_layer = model.policy_head[5]  # GELU before final projection
    num_neurons = 256  # hidden_dim

    type_names = [_ACTION_TYPES.get(t, str(t)) for t in unique_types]
    conductance_matrix = np.zeros((num_neurons, len(unique_types)), dtype=np.float32)

    internal_batch_size = 2 * states.shape[0]

    t0 = time.perf_counter()
    for ti, atype in enumerate(unique_types):
        action_indices = all_action_indices[atype]
        if not action_indices:
            continue

        # Forward function: sum of logits for this action type
        action_idx_tensor = torch.tensor(action_indices, device=device)

        def make_forward(ait: torch.Tensor):  # noqa: ANN202
            def forward_fn(x: torch.Tensor) -> torch.Tensor:
                p, _ = model(x)
                return p[:, ait].sum(dim=-1)
            return forward_fn

        fwd = make_forward(action_idx_tensor)

        for ni in range(num_neurons):
            nc = NeuronConductance(fwd, target_layer)
            attr = nc.attribute(
                inputs, neuron_selector=ni,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
            )
            assert isinstance(attr, torch.Tensor)
            conductance_matrix[ni, ti] = float(attr.abs().mean().item())

        elapsed = time.perf_counter() - t0
        print(f"    Action type {type_names[ti]:>10s} ({ti + 1}/{len(unique_types)}, {elapsed:.1f}s)")

    # Compute derived metrics
    dominant_action = np.argmax(conductance_matrix, axis=1)

    neurons_per_action: dict[str, int] = {}
    for ti, name in enumerate(type_names):
        neurons_per_action[name] = int(np.sum(dominant_action == ti))

    # Dead neurons: max conductance across all action types < 5% of overall max
    max_per_neuron = conductance_matrix.max(axis=1)
    global_max = max_per_neuron.max()
    dead_threshold = 0.05 * global_max
    dead_fraction = float(np.mean(max_per_neuron < dead_threshold))

    # Specialization: max / sum (1.0 = only serves one action type)
    row_sums = conductance_matrix.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-10, None)
    specialization_scores = conductance_matrix.max(axis=1) / row_sums.squeeze()

    return NeuronSpecResult(
        num_neurons=num_neurons,
        conductance_matrix=conductance_matrix,
        action_type_names=type_names,
        action_type_ids=unique_types,
        dominant_action=dominant_action,
        neurons_per_action=neurons_per_action,
        dead_fraction=dead_fraction,
        specialization_scores=specialization_scores,
    )


# ---------------------------------------------------------------------------
# 3. Layer Causal Necessity
# ---------------------------------------------------------------------------


@dataclass
class LayerNecessityResult:
    """Per-layer causal necessity results."""

    layer_names: list[str]
    # Overall KL when layer is bypassed
    overall_kl: dict[str, float]
    # Per-phase KL
    phase_kl: dict[str, dict[int, float]]


def analyze_layer_necessity(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
) -> LayerNecessityResult:
    """Replace each policy hidden layer (Linear+GELU) with identity and measure
    the causal effect on policy output, stratified by phase.

    Layer pairs: (0,1)=P0+GELU, (2,3)=P2+GELU, (4,5)=P4+GELU.
    """
    model.eval()

    # Get original policy
    print("  Computing original policy...")
    orig_logits, _ = forward_batched(model, device, dataset.states, batch_size)
    orig_pol = batch_masked_softmax(orig_logits, dataset.legal_masks)

    phases = dataset.phases
    phase_ids = sorted(set(phases))

    layer_names = ["P0+GELU", "P2+GELU", "P4+GELU"]
    # Sequential indices for each Linear+GELU pair
    layer_pairs = [(0, 1), (2, 3), (4, 5)]

    overall_kl: dict[str, float] = {}
    phase_kl: dict[str, dict[int, float]] = {}

    for name, (lin_idx, gelu_idx) in zip(layer_names, layer_pairs):
        print(f"  Bypassing {name}...")

        # Save original modules
        orig_linear = model.policy_head[lin_idx]
        orig_gelu = model.policy_head[gelu_idx]

        # Replace with identity
        model.policy_head[lin_idx] = torch.nn.Identity()
        model.policy_head[gelu_idx] = torch.nn.Identity()

        # Forward pass with bypassed layer
        bypass_logits, _ = forward_batched(model, device, dataset.states, batch_size)
        bypass_pol = batch_masked_softmax(bypass_logits, dataset.legal_masks)

        # Restore
        model.policy_head[lin_idx] = orig_linear
        model.policy_head[gelu_idx] = orig_gelu

        # Compute KL
        kl = kl_divergence_batch(orig_pol, bypass_pol)
        overall_kl[name] = float(np.mean(kl))

        pk: dict[int, float] = {}
        for pid in phase_ids:
            mask = phases == pid
            if mask.sum() > 0:
                pk[pid] = float(np.mean(kl[mask]))
        phase_kl[name] = pk

    return LayerNecessityResult(
        layer_names=layer_names,
        overall_kl=overall_kl,
        phase_kl=phase_kl,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_logit_lens_report(result: LogitLensResult) -> None:
    """Print logit lens results to console."""
    print("\n" + "=" * 90)
    print("  1. LOGIT LENS (projecting intermediate representations through final W)")
    print("=" * 90)

    print(f"\n  {'Layer':<8s} {'Argmax Agree':>13s} {'Mean KL':>10s} {'Final Act Prob':>15s}")
    print(f"  {'-' * 8} {'-' * 13} {'-' * 10} {'-' * 15}")
    for name in result.layer_names:
        print(
            f"  {name:<8s} {result.argmax_agreement[name]:>12.1%} "
            f"{result.mean_kl[name]:>10.4f} "
            f"{result.final_action_prob[name]:>14.1%}"
        )

    # Per-phase agreement
    phase_ids = sorted(next(iter(result.phase_agreement.values())).keys())
    decision_phases = [p for p in phase_ids if p in _PHASE_NAMES]

    print(f"\n  Per-phase argmax agreement with final policy:")
    header = f"  {'Layer':<8s}" + "".join(
        f" {_PHASE_NAMES.get(p, str(p)):>8s}" for p in decision_phases
    )
    print(header)
    print(f"  {'-' * 8}" + "".join(f" {'-' * 8}" for _ in decision_phases))
    for name in result.layer_names:
        row = f"  {name:<8s}"
        for pid in decision_phases:
            agree = result.phase_agreement[name].get(pid, 0.0)
            row += f" {agree:>7.1%}"
        print(row)

    # Highlight phases where decisions form late
    print(f"\n  Decision crystallization (layer where agreement first exceeds 90%):")
    for pid in decision_phases:
        for name in result.layer_names:
            if result.phase_agreement[name].get(pid, 0) >= 0.9:
                print(f"    {_PHASE_NAMES[pid]:<8s}: {name}")
                break
        else:
            last = result.layer_names[-1]
            last_agree = result.phase_agreement[last].get(pid, 0)
            print(f"    {_PHASE_NAMES[pid]:<8s}: not reached (best: {last} at {last_agree:.1%})")


def print_neuron_spec_report(result: NeuronSpecResult) -> None:
    """Print neuron specialization results to console."""
    print("\n" + "=" * 90)
    print("  2. NEURON SPECIALIZATION (NeuronConductance per action type)")
    print("=" * 90)
    print(f"  Target layer: policy_head GELU before final projection (256 neurons)")

    print(f"\n  Dead neurons (max cond < 5% of global max): {result.dead_fraction:.1%}")
    print(f"  Mean specialization score (max/sum): {result.specialization_scores.mean():.3f}")
    print(f"  Median specialization score: {np.median(result.specialization_scores):.3f}")

    print(f"\n  Neurons dominated by each action type:")
    for name, count in sorted(result.neurons_per_action.items(), key=lambda x: -x[1]):
        bar = "#" * (count // 2)
        print(f"    {name:<12s}: {count:>3d} / {result.num_neurons}  {bar}")

    # Show top conductance per action type (which neurons matter most)
    print(f"\n  Top-5 neurons per action type (by conductance):")
    for ti, name in enumerate(result.action_type_names):
        col = result.conductance_matrix[:, ti]
        top5 = np.argsort(col)[-5:][::-1]
        top5_vals = col[top5]
        neurons_str = ", ".join(f"n{n}={v:.3f}" for n, v in zip(top5, top5_vals))
        print(f"    {name:<12s}: {neurons_str}")

    # Show highly specialized neurons
    highly_spec = np.where(result.specialization_scores > 0.6)[0]
    print(f"\n  Highly specialized neurons (score > 0.6): {len(highly_spec)} / {result.num_neurons}")
    if len(highly_spec) > 0 and len(highly_spec) <= 30:
        for ni in highly_spec:
            dom = result.dominant_action[ni]
            score = result.specialization_scores[ni]
            aname = result.action_type_names[dom]
            print(f"    neuron {ni:>3d}: {aname:<12s} (spec={score:.3f})")


def print_necessity_report(result: LayerNecessityResult) -> None:
    """Print layer causal necessity results to console."""
    print("\n" + "=" * 90)
    print("  3. LAYER CAUSAL NECESSITY (bypass each Linear+GELU pair)")
    print("=" * 90)

    print(f"\n  {'Layer':<12s} {'Overall KL':>11s}")
    print(f"  {'-' * 12} {'-' * 11}")
    for name in result.layer_names:
        print(f"  {name:<12s} {result.overall_kl[name]:>11.4f}")

    # Per-phase
    phase_ids = sorted(next(iter(result.phase_kl.values())).keys())
    decision_phases = [p for p in phase_ids if p in _PHASE_NAMES]

    print(f"\n  Per-phase causal effect (KL when layer bypassed):")
    header = f"  {'Layer':<12s}" + "".join(
        f" {_PHASE_NAMES.get(p, str(p)):>8s}" for p in decision_phases
    )
    print(header)
    print(f"  {'-' * 12}" + "".join(f" {'-' * 8}" for _ in decision_phases))
    for name in result.layer_names:
        row = f"  {name:<12s}"
        for pid in decision_phases:
            kl = result.phase_kl[name].get(pid, 0.0)
            row += f" {kl:>8.4f}"
        print(row)

    # Which layer matters most per phase?
    print(f"\n  Most causally necessary layer per phase:")
    for pid in decision_phases:
        best_name = ""
        best_kl = 0.0
        for name in result.layer_names:
            kl = result.phase_kl[name].get(pid, 0.0)
            if kl > best_kl:
                best_kl = kl
                best_name = name
        print(f"    {_PHASE_NAMES[pid]:<8s}: {best_name} (KL={best_kl:.4f})")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    lens_result: LogitLensResult,
    neuron_result: NeuronSpecResult,
    necessity_result: LayerNecessityResult,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a self-contained HTML report for policy head analysis."""
    # --- Serialize logit lens data ---
    phase_ids_lens = sorted(next(iter(lens_result.phase_agreement.values())).keys())
    decision_phases_lens = [int(p) for p in phase_ids_lens if p in _PHASE_NAMES]

    lens_data = {
        "layer_names": lens_result.layer_names,
        "argmax_agreement": lens_result.argmax_agreement,
        "mean_kl": lens_result.mean_kl,
        "final_action_prob": lens_result.final_action_prob,
        "phase_agreement": {
            name: {str(int(k)): float(v) for k, v in pa.items()}
            for name, pa in lens_result.phase_agreement.items()
        },
        "decision_phases": decision_phases_lens,
        "phase_names": {str(int(k)): v for k, v in _PHASE_NAMES.items()},
    }

    # --- Serialize neuron specialization data (convert numpy to lists) ---
    neuron_data = {
        "num_neurons": neuron_result.num_neurons,
        "action_type_names": neuron_result.action_type_names,
        "neurons_per_action": neuron_result.neurons_per_action,
        "dead_fraction": neuron_result.dead_fraction,
        "mean_specialization": float(neuron_result.specialization_scores.mean()),
        "median_specialization": float(np.median(neuron_result.specialization_scores)),
        "specialization_scores": neuron_result.specialization_scores.tolist(),
        "dominant_action": neuron_result.dominant_action.tolist(),
        "highly_specialized": [],
    }
    highly_spec = np.where(neuron_result.specialization_scores > 0.6)[0]
    for ni in highly_spec:
        dom = neuron_result.dominant_action[ni]
        score = neuron_result.specialization_scores[ni]
        aname = neuron_result.action_type_names[dom]
        neuron_data["highly_specialized"].append({
            "neuron": int(ni),
            "action": aname,
            "score": float(score),
        })

    # --- Serialize layer necessity data ---
    phase_ids_nec = sorted(next(iter(necessity_result.phase_kl.values())).keys())
    decision_phases_nec = [int(p) for p in phase_ids_nec if p in _PHASE_NAMES]

    necessity_data = {
        "layer_names": necessity_result.layer_names,
        "overall_kl": necessity_result.overall_kl,
        "phase_kl": {
            name: {str(int(k)): float(v) for k, v in pk.items()}
            for name, pk in necessity_result.phase_kl.items()
        },
        "decision_phases": decision_phases_nec,
        "phase_names": {str(int(k)): v for k, v in _PHASE_NAMES.items()},
    }

    lens_json = json.dumps(lens_data)
    neuron_json = json.dumps(neuron_data)
    necessity_json = json.dumps(necessity_data)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Policy Head Analysis — Epoch {epoch}</title>
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
<h1>Policy Head Analysis — Epoch {epoch}</h1>
<div class="meta">
  {num_states:,} states from {num_games} games.
</div>

<h2>1. Logit Lens</h2>
<p style="color:#888;font-size:0.85rem">Project intermediate policy representations through the final weight matrix. Shows when decisions crystallize.</p>
<h3>Summary</h3>
<table id="tbl-lens-summary"></table>
<h3>Per-Phase Argmax Agreement</h3>
<table id="tbl-lens-phase"></table>

<h2>2. Neuron Specialization</h2>
<p style="color:#888;font-size:0.85rem">NeuronConductance per action type in the final hidden layer (256 neurons).</p>
<div id="neuron-stats"></div>
<h3>Neurons per Action Type</h3>
<table id="tbl-neuron-bar"></table>
<h3>Highly Specialized Neurons (score &gt; 0.6)</h3>
<table id="tbl-neuron-spec"></table>

<h2>3. Layer Causal Necessity</h2>
<p style="color:#888;font-size:0.85rem">KL divergence when each Linear+GELU pair is replaced with identity.</p>
<h3>Overall</h3>
<table id="tbl-necessity-overall"></table>
<h3>Per-Phase KL Heatmap</h3>
<table id="tbl-necessity-phase"></table>

<script>
const lens = {lens_json};
const neuron = {neuron_json};
const necessity = {necessity_json};

function heatColor(v, maxVal) {{
  if (maxVal <= 0) return "transparent";
  const t = Math.min(v / maxVal, 1.0);
  return "hsl(" + (120 - t * 120) + ",70%," + (18 + t * 22) + "%)";
}}

function agreeColor(v) {{
  return "hsl(" + (v * 120) + ",70%," + (18 + v * 22) + "%)";
}}

function makeBar(val, maxVal, cls) {{
  const pct = maxVal > 0 ? val / maxVal * 100 : 0;
  return '<span class="bar-container"><span class="bar ' + cls + '" style="width:' + pct + '%"></span></span>';
}}

// --- 1. Logit Lens Summary ---
(function() {{
  const tbl = document.getElementById("tbl-lens-summary");
  let html = '<tr><th>Layer</th><th>Argmax Agreement</th><th>Mean KL</th><th>Final Act Prob</th></tr>';
  for (const name of lens.layer_names) {{
    const agree = lens.argmax_agreement[name];
    const kl = lens.mean_kl[name];
    const prob = lens.final_action_prob[name];
    html += '<tr><td>' + name + '</td>' +
      '<td>' + (agree * 100).toFixed(1) + '%</td>' +
      '<td>' + kl.toFixed(4) + '</td>' +
      '<td>' + (prob * 100).toFixed(1) + '%</td></tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 1b. Logit Lens Per-Phase Agreement Heatmap ---
(function() {{
  const tbl = document.getElementById("tbl-lens-phase");
  const phases = lens.decision_phases;
  let html = '<tr><th>Layer</th>';
  for (const p of phases) {{
    html += '<th>' + (lens.phase_names[String(p)] || p) + '</th>';
  }}
  html += '</tr>';
  for (const name of lens.layer_names) {{
    html += '<tr><td>' + name + '</td>';
    const pa = lens.phase_agreement[name];
    for (const p of phases) {{
      const v = pa[String(p)] || 0;
      const bg = agreeColor(v);
      html += '<td style="background:' + bg + '">' + (v * 100).toFixed(1) + '%</td>';
    }}
    html += '</tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 2. Neuron Specialization Stats ---
(function() {{
  const div = document.getElementById("neuron-stats");
  div.innerHTML =
    '<div class="stat-box"><div class="stat-label">Dead Neurons</div><div class="stat-value">' +
    (neuron.dead_fraction * 100).toFixed(1) + '%</div></div>' +
    '<div class="stat-box"><div class="stat-label">Mean Specialization</div><div class="stat-value">' +
    neuron.mean_specialization.toFixed(3) + '</div></div>' +
    '<div class="stat-box"><div class="stat-label">Median Specialization</div><div class="stat-value">' +
    neuron.median_specialization.toFixed(3) + '</div></div>' +
    '<div class="stat-box"><div class="stat-label">Highly Specialized</div><div class="stat-value">' +
    neuron.highly_specialized.length + ' / ' + neuron.num_neurons + '</div></div>';
}})();

// --- 2b. Neurons per Action Type (bar chart) ---
(function() {{
  const tbl = document.getElementById("tbl-neuron-bar");
  const entries = Object.entries(neuron.neurons_per_action).sort((a, b) => b[1] - a[1]);
  const maxCount = entries.length > 0 ? entries[0][1] : 1;
  let html = '<tr><th>Action Type</th><th>Count</th><th></th><th>Fraction</th></tr>';
  for (const [name, count] of entries) {{
    html += '<tr><td>' + name + '</td>' +
      '<td>' + count + ' / ' + neuron.num_neurons + '</td>' +
      '<td>' + makeBar(count, maxCount, 'bar-green') + '</td>' +
      '<td>' + (count / neuron.num_neurons * 100).toFixed(1) + '%</td></tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 2c. Highly Specialized Neurons ---
(function() {{
  const tbl = document.getElementById("tbl-neuron-spec");
  let html = '<tr><th>Neuron</th><th>Dominant Action</th><th>Score</th></tr>';
  if (neuron.highly_specialized.length === 0) {{
    html += '<tr><td colspan="3" style="text-align:center;color:#888">None found</td></tr>';
  }} else {{
    for (const n of neuron.highly_specialized) {{
      html += '<tr><td>n' + n.neuron + '</td>' +
        '<td>' + n.action + '</td>' +
        '<td>' + n.score.toFixed(3) + '</td></tr>';
    }}
  }}
  tbl.innerHTML = html;
}})();

// --- 3. Layer Causal Necessity Overall ---
(function() {{
  const tbl = document.getElementById("tbl-necessity-overall");
  const maxKL = Math.max(...necessity.layer_names.map(n => necessity.overall_kl[n]));
  let html = '<tr><th>Layer</th><th>Overall KL</th><th></th></tr>';
  for (const name of necessity.layer_names) {{
    const kl = necessity.overall_kl[name];
    html += '<tr><td>' + name + '</td>' +
      '<td>' + kl.toFixed(4) + '</td>' +
      '<td>' + makeBar(kl, maxKL, 'bar-orange') + '</td></tr>';
  }}
  tbl.innerHTML = html;
}})();

// --- 3b. Layer Causal Necessity Per-Phase Heatmap ---
(function() {{
  const tbl = document.getElementById("tbl-necessity-phase");
  const phases = necessity.decision_phases;
  // Find max KL across all cells for color scaling
  let maxKL = 0;
  for (const name of necessity.layer_names) {{
    const pk = necessity.phase_kl[name];
    for (const p of phases) {{
      const v = pk[String(p)] || 0;
      if (v > maxKL) maxKL = v;
    }}
  }}
  let html = '<tr><th>Layer</th>';
  for (const p of phases) {{
    html += '<th>' + (necessity.phase_names[String(p)] || p) + '</th>';
  }}
  html += '</tr>';
  for (const name of necessity.layer_names) {{
    html += '<tr><td>' + name + '</td>';
    const pk = necessity.phase_kl[name];
    for (const p of phases) {{
      const v = pk[String(p)] || 0;
      const bg = heatColor(v, maxKL);
      html += '<td style="background:' + bg + '">' + v.toFixed(4) + '</td>';
    }}
    html += '</tr>';
  }}
  tbl.innerHTML = html;
}})();
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy head analysis (logit lens, neuron specialization, causal necessity)"
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
        help="Max states for neuron conductance (slow analysis)",
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

    # --- Analysis 1: Logit Lens ---
    print(f"\nRunning logit lens analysis...")
    lens_result = analyze_logit_lens(
        model, device, dataset, batch_size=args.batch_size,
    )
    print_logit_lens_report(lens_result)

    # --- Analysis 2: Neuron Specialization ---
    print(f"\nRunning neuron specialization analysis (this is slow)...")
    neuron_result = analyze_neuron_specialization(
        model, device, dataset, config.num_players,
        max_samples=args.max_neuron_samples,
        n_steps=args.neuron_steps,
    )
    print_neuron_spec_report(neuron_result)

    # --- Analysis 3: Layer Causal Necessity ---
    print(f"\nRunning layer causal necessity analysis...")
    necessity_result = analyze_layer_necessity(
        model, device, dataset, batch_size=args.batch_size,
    )
    print_necessity_report(necessity_result)

    # --- Summary ---
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY")
    print(f"{'=' * 90}")

    # Logit lens: when do decisions form?
    trunk_agree = lens_result.argmax_agreement["trunk"]
    p4_agree = lens_result.argmax_agreement["P4"]
    print(f"\n  Decision formation:")
    print(f"    Trunk -> final agreement: {trunk_agree:.1%}")
    print(f"    P4 -> final agreement: {p4_agree:.1%}")
    print(f"    Layers P0-P4 change {(1 - trunk_agree) * 100:.1f}% of decisions")

    # Neuron specialization: is width well-used?
    print(f"\n  Width utilization:")
    print(f"    Dead neurons: {neuron_result.dead_fraction:.1%}")
    print(f"    Mean specialization: {neuron_result.specialization_scores.mean():.3f}")
    top_action = max(neuron_result.neurons_per_action.items(), key=lambda x: x[1])
    print(f"    Most neurons serve: {top_action[0]} ({top_action[1]}/{neuron_result.num_neurons})")

    # Necessity: which layer matters most?
    most_necessary = max(necessity_result.overall_kl.items(), key=lambda x: x[1])
    least_necessary = min(necessity_result.overall_kl.items(), key=lambda x: x[1])
    print(f"\n  Layer necessity:")
    print(f"    Most necessary: {most_necessary[0]} (KL={most_necessary[1]:.4f})")
    print(f"    Least necessary: {least_necessary[0]} (KL={least_necessary[1]:.4f})")

    # --- HTML report ---
    html_path = Path("interp/data") / f"policy_head_epoch{epoch}.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    html = format_html_report(
        lens_result, neuron_result, necessity_result,
        epoch=epoch,
        num_states=dataset.num_states,
        num_games=dataset.num_games,
    )
    html_path.write_text(html)
    print(f"\nHTML report written to {html_path}")

    if not args.no_open:
        _open_file(html_path)


if __name__ == "__main__":
    main()
