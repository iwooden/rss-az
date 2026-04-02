"""Per-phase policy head analysis: logit lens, neuron specialization, layer causal necessity.

Three analyses applied independently to each of the 8 per-phase policy heads:

1. **Logit Lens**: Project intermediate representations through the head's final
   weight matrix to see when decisions crystallize within that head.

2. **Neuron Specialization**: NeuronConductance per action type reveals which
   neurons serve which actions within each head. Measures functional width
   utilization and dead neuron fraction.

3. **Layer Causal Necessity**: Replace each Linear+GELU pair with a skip
   connection and measure policy KL. Shows which layers are causally necessary
   within each head.

Usage:
    .venv/bin/python -m interp.layers.policy_head --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp.html import BAR_CSS, JS_MAKE_BAR, STAT_BOX_CSS, html_page, open_file
from interp.utils import (
    DECISION_PHASE_ORDER,
    InterpDataset,
    batch_masked_softmax,
    collect_states,
    forward_batched,
    kl_divergence_batch,
    load_model,
)


# ---------------------------------------------------------------------------
# Per-phase action type names (each head only handles its own phase's actions)
# ---------------------------------------------------------------------------

_PHASE_ACTION_TYPES: dict[str, list[str]] = {
    "INVEST": ["pass", "auction", "buy", "sell"],
    "BID": ["leave", "raise_bid"],
    "ACQ": ["acq_price", "acq_fi_buy", "pass"],
    "CLOSE": ["close", "pass"],
    "DIV": ["div_zero", "div_max", "div_mid"],
    "ISSUE": ["pass", "issue"],
    "IPO": ["pass", "ipo_corp"],
    "PAR": ["par_min", "par_max", "par_mid"],
}

# Phases with mask-dependent action types (min/max/mid from per-state legal mask)
_MASK_DEPENDENT_PHASES = {"DIV", "PAR"}


def _get_action_type_indices(phase_name: str, num_players: int) -> dict[str, list[int]]:
    """Map action type names to LOCAL indices within a phase head's output.

    Returns dict mapping type name -> list of local action indices.
    """
    from core.actions import decode_action_py, get_action_layout, get_total_action_count

    layout = get_action_layout(num_players)
    phase_start_keys = [
        'invest_start', 'bid_start', 'acquisition_start', 'closing_start',
        'dividends_start', 'issue_start', 'ipo_start', 'par_start',
    ]
    phase_starts = [layout[k] for k in phase_start_keys]
    head_idx = DECISION_PHASE_ORDER.index(phase_name)
    action_dim = get_total_action_count(num_players)
    start = phase_starts[head_idx]
    end = phase_starts[head_idx + 1] if head_idx + 1 < len(phase_starts) else action_dim
    n_actions = end - start

    result: dict[str, list[int]] = {t: [] for t in _PHASE_ACTION_TYPES[phase_name]}
    type_id_to_name = {
        0: "pass", 1: "auction", 2: "buy", 3: "sell",
        4: "leave", 5: "raise_bid", 6: "acq_price", 7: "acq_fi_buy",
        8: "close", 9: "dividend", 10: "issue", 11: "ipo_corp", 12: "par_price",
    }
    for local_idx in range(n_actions):
        global_idx = start + local_idx
        _, atype, *_ = decode_action_py(global_idx, num_players)
        name = type_id_to_name.get(atype)
        if name and name in result:
            result[name].append(local_idx)

    return result


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _collect_head_activations(
    model: Any,
    device: torch.device,
    phase_states: np.ndarray,
    head_idx: int,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Collect trunk output and per-GELU activations for one phase head.

    phase_states must contain only states that route to this head.
    Returns (trunk_acts, {layer_name: acts}).
    """
    model.eval()
    head = model.phase_heads[head_idx]

    acts: dict[str, list[torch.Tensor]] = {"trunk": []}
    handles = []

    def make_hook(name: str):  # noqa: ANN202
        def hook(
            _mod: torch.nn.Module,  # noqa: ARG001
            _inp: tuple[torch.Tensor, ...],  # noqa: ARG001
            out: torch.Tensor,
        ) -> None:
            acts[name].append(out.detach().cpu())
        return hook

    handles.append(model.trunk_norm.register_forward_hook(make_hook("trunk")))

    # Hook GELU outputs within this head (indices 1, 3, 5 in Sequential)
    for label, idx in [("L0", 1), ("L1", 3), ("L2", 5)]:
        if idx < len(head):
            acts[label] = []
            handles.append(head[idx].register_forward_hook(make_hook(label)))

    with torch.inference_mode():
        for i in range(0, phase_states.shape[0], batch_size):
            j = min(i + batch_size, phase_states.shape[0])
            model(torch.from_numpy(phase_states[i:j]).to(device))

    for h in handles:
        h.remove()

    trunk = torch.cat(acts.pop("trunk"), dim=0).numpy()
    layer_acts = {k: torch.cat(v, dim=0).numpy() for k, v in acts.items()}
    return trunk, layer_acts


# ---------------------------------------------------------------------------
# 1. Logit Lens
# ---------------------------------------------------------------------------


@dataclass
class LogitLensResult:
    """Per-head logit lens results."""

    phase_name: str
    layer_names: list[str]
    # Per-layer agreement with final policy (fraction where argmax matches)
    argmax_agreement: dict[str, float]
    # Per-layer KL divergence from final policy
    mean_kl: dict[str, float]
    # Per-layer probability assigned to the final argmax action
    final_action_prob: dict[str, float]
    num_states: int


def analyze_logit_lens(
    model: Any,
    device: torch.device,
    phase_states: np.ndarray,
    phase_masks: np.ndarray,
    head_idx: int,
    batch_size: int = 256,
) -> LogitLensResult:
    """Project intermediate head representations through the final weight matrix.

    phase_states/phase_masks are pre-filtered to only this phase's states.
    """
    phase_name = DECISION_PHASE_ORDER[head_idx]
    head = model.phase_heads[head_idx]

    # Collect activations
    trunk_acts, layer_acts = _collect_head_activations(
        model, device, phase_states, head_idx, batch_size,
    )

    # Get the final projection weights (last Linear in the Sequential)
    final_linear = None
    for layer in reversed(list(head)):
        if isinstance(layer, torch.nn.Linear):
            final_linear = layer
            break
    assert final_linear is not None
    W = final_linear.weight.detach().cpu().numpy()  # (n_actions, hidden_dim)
    b = final_linear.bias.detach().cpu().numpy()  # (n_actions,)

    # Get actual final policy (local logits only)
    model.eval()
    start = model._phase_starts[head_idx]
    end = model._phase_ends[head_idx]
    full_logits, _ = forward_batched(model, device, phase_states, batch_size)
    local_logits = full_logits[:, start:end]
    local_masks = phase_masks[:, start:end]

    final_pol = batch_masked_softmax(local_logits, local_masks)
    final_argmax = np.argmax(
        local_logits + np.where(local_masks > 0, 0, -1e9), axis=1,
    )

    layer_names = ["trunk"] + sorted(layer_acts.keys())
    all_acts = {"trunk": trunk_acts, **layer_acts}

    argmax_agreement: dict[str, float] = {}
    mean_kl: dict[str, float] = {}
    final_action_prob: dict[str, float] = {}

    for name in layer_names:
        h = all_acts[name]
        early_logits = h @ W.T + b
        early_pol = batch_masked_softmax(early_logits, local_masks)
        early_argmax = np.argmax(
            early_logits + np.where(local_masks > 0, 0, -1e9), axis=1,
        )

        argmax_agreement[name] = float(np.mean(early_argmax == final_argmax))
        mean_kl[name] = float(np.mean(kl_divergence_batch(final_pol, early_pol)))
        probs_for_final = early_pol[np.arange(len(final_argmax)), final_argmax]
        final_action_prob[name] = float(np.mean(probs_for_final))

    return LogitLensResult(
        phase_name=phase_name,
        layer_names=layer_names,
        argmax_agreement=argmax_agreement,
        mean_kl=mean_kl,
        final_action_prob=final_action_prob,
        num_states=phase_states.shape[0],
    )


# ---------------------------------------------------------------------------
# 2. Neuron Specialization
# ---------------------------------------------------------------------------


@dataclass
class NeuronSpecResult:
    """Neuron specialization results for one phase head."""

    phase_name: str
    num_neurons: int
    # (num_neurons, num_action_types) conductance matrix
    conductance_matrix: np.ndarray
    action_type_names: list[str]
    # Per-neuron: dominant action type index
    dominant_action: np.ndarray
    # Per action type: how many neurons are dominated by it
    neurons_per_action: dict[str, int]
    # Fraction of dead neurons (max conductance < 5% of global max)
    dead_fraction: float
    # Specialization score per neuron (max / sum, 1.0 = fully specialized)
    specialization_scores: np.ndarray


def analyze_neuron_specialization(
    model: Any,
    device: torch.device,
    phase_states: np.ndarray,
    phase_masks: np.ndarray,
    head_idx: int,
    num_players: int,
    max_samples: int = 300,
    n_steps: int = 20,
) -> NeuronSpecResult:
    """NeuronConductance for each neuron in a phase head's final hidden layer.

    Target layer: the GELU before the final projection (index 5 in head Sequential).
    For DIV/PAR phases, action types are mask-dependent (zero/min/max/mid).
    """
    from captum.attr import NeuronConductance

    model.eval()
    phase_name = DECISION_PHASE_ORDER[head_idx]
    head = model.phase_heads[head_idx]

    # Subsample (keeping masks aligned)
    states = phase_states
    masks = phase_masks
    if states.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(states.shape[0], max_samples, replace=False)
        states = states[idx]
        masks = masks[idx]

    # Target layer: GELU before final projection (index 5)
    target_layer = head[5]
    hidden_dim = model.cfg.hidden_dim
    num_neurons = hidden_dim

    type_names = _PHASE_ACTION_TYPES[phase_name]
    start = model._phase_starts[head_idx]
    end = model._phase_ends[head_idx]
    local_masks = masks[:, start:end]  # (N, n_local_actions)
    n_samples = states.shape[0]

    conductance_matrix = np.zeros((num_neurons, len(type_names)), dtype=np.float32)
    inputs = torch.from_numpy(states).to(device).requires_grad_(True)
    internal_batch_size = 2 * n_samples

    # Direct forward path bypassing phase dispatch (Captum's zero baseline
    # has no phase one-hot and gets misrouted by the model's phase dispatch).
    def _trunk_then_head(x: torch.Tensor) -> torch.Tensor:
        h = model.input_preprocess(x)
        for block in model.blocks:
            h = block(h)
        h = model.trunk_norm(h)
        return head(h)

    # Build forward functions per action type
    if phase_name in _MASK_DEPENDENT_PHASES:
        # Compute per-state min/max legal local index from masks
        # legal_indices[i] = set of legal local action indices for state i
        local_masks_bool = local_masks > 0

        if phase_name == "DIV":
            # div_zero: local index 0 (dividend $0, always legal)
            # div_max: highest legal local index per state
            # div_mid: everything legal in between
            per_state_max = np.zeros(n_samples, dtype=np.int64)
            for i in range(n_samples):
                legal = np.where(local_masks_bool[i])[0]
                per_state_max[i] = legal[-1] if len(legal) > 0 else 0
            per_state_max_t = torch.from_numpy(per_state_max).to(device)

            def make_fwd_div_zero():  # noqa: ANN202
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    return _trunk_then_head(x)[:, 0]
                return fwd

            def make_fwd_div_max(idx_t: torch.Tensor):  # noqa: ANN202
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    logits = _trunk_then_head(x)
                    n = x.shape[0]
                    tiled = idx_t.repeat(n // idx_t.shape[0])
                    return logits[torch.arange(n, device=x.device), tiled]
                return fwd

            def make_fwd_div_mid(idx_t: torch.Tensor):  # noqa: ANN202
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    logits = _trunk_then_head(x)
                    n = x.shape[0]
                    tiled = idx_t.repeat(n // idx_t.shape[0])
                    # Sum all logits, subtract zero and max
                    return logits.sum(dim=-1) - logits[:, 0] - logits[torch.arange(n, device=x.device), tiled]
                return fwd

            forward_fns = {
                "div_zero": make_fwd_div_zero(),
                "div_max": make_fwd_div_max(per_state_max_t),
                "div_mid": make_fwd_div_mid(per_state_max_t),
            }
        else:  # PAR
            # par_min: lowest legal local index per state
            # par_max: highest legal local index per state
            # par_mid: everything legal in between
            per_state_min = np.zeros(n_samples, dtype=np.int64)
            per_state_max = np.zeros(n_samples, dtype=np.int64)
            for i in range(n_samples):
                legal = np.where(local_masks_bool[i])[0]
                if len(legal) > 0:
                    per_state_min[i] = legal[0]
                    per_state_max[i] = legal[-1]
            per_state_min_t = torch.from_numpy(per_state_min).to(device)
            per_state_max_t = torch.from_numpy(per_state_max).to(device)

            def make_fwd_par_gather(idx_t: torch.Tensor):  # noqa: ANN202
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    logits = _trunk_then_head(x)
                    n = x.shape[0]
                    tiled = idx_t.repeat(n // idx_t.shape[0])
                    return logits[torch.arange(n, device=x.device), tiled]
                return fwd

            def make_fwd_par_mid(min_t: torch.Tensor, max_t: torch.Tensor):  # noqa: ANN202
                def fwd(x: torch.Tensor) -> torch.Tensor:
                    logits = _trunk_then_head(x)
                    n = x.shape[0]
                    min_tiled = min_t.repeat(n // min_t.shape[0])
                    max_tiled = max_t.repeat(n // max_t.shape[0])
                    arange = torch.arange(n, device=x.device)
                    return logits.sum(dim=-1) - logits[arange, min_tiled] - logits[arange, max_tiled]
                return fwd

            forward_fns = {
                "par_min": make_fwd_par_gather(per_state_min_t),
                "par_max": make_fwd_par_gather(per_state_max_t),
                "par_mid": make_fwd_par_mid(per_state_min_t, per_state_max_t),
            }
    else:
        # Static action types
        type_indices = _get_action_type_indices(phase_name, num_players)
        type_names = [t for t in type_names if type_indices.get(t)]

        def make_direct_forward(local_idx: torch.Tensor):  # noqa: ANN202
            def forward_fn(x: torch.Tensor) -> torch.Tensor:
                logits = _trunk_then_head(x)
                return logits[:, local_idx].sum(dim=-1)
            return forward_fn

        forward_fns = {
            t: make_direct_forward(torch.tensor(type_indices[t], device=device))
            for t in type_names
        }

    # Resize matrix for actual type count
    conductance_matrix = np.zeros((num_neurons, len(type_names)), dtype=np.float32)

    t0 = time.perf_counter()
    for ti, atype_name in enumerate(type_names):
        fwd = forward_fns[atype_name]

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
        print(f"      {atype_name:>12s} ({ti + 1}/{len(type_names)}, {elapsed:.1f}s)")

    # Derived metrics
    dominant_action = np.argmax(conductance_matrix, axis=1)

    neurons_per_action: dict[str, int] = {}
    for ti, name in enumerate(type_names):
        neurons_per_action[name] = int(np.sum(dominant_action == ti))

    max_per_neuron = conductance_matrix.max(axis=1)
    global_max = max_per_neuron.max() if max_per_neuron.max() > 0 else 1.0
    dead_fraction = float(np.mean(max_per_neuron < 0.05 * global_max))

    row_sums = np.clip(conductance_matrix.sum(axis=1, keepdims=True), 1e-10, None)
    specialization_scores = conductance_matrix.max(axis=1) / row_sums.squeeze()

    return NeuronSpecResult(
        phase_name=phase_name,
        num_neurons=num_neurons,
        conductance_matrix=conductance_matrix,
        action_type_names=type_names,
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
    """Per-layer causal necessity results for one phase head."""

    phase_name: str
    layer_names: list[str]
    # KL when each layer pair is bypassed
    overall_kl: dict[str, float]


def analyze_layer_necessity(
    model: Any,
    device: torch.device,
    phase_states: np.ndarray,
    phase_masks: np.ndarray,
    head_idx: int,
    batch_size: int = 256,
) -> LayerNecessityResult:
    """Replace each Linear+GELU pair with identity and measure KL.

    Layer pairs within each head: (0,1)=L0+GELU, (2,3)=L1+GELU, (4,5)=L2+GELU.
    """
    model.eval()
    phase_name = DECISION_PHASE_ORDER[head_idx]
    head = model.phase_heads[head_idx]

    start = model._phase_starts[head_idx]
    end = model._phase_ends[head_idx]

    # Get original policy (local to this phase)
    orig_logits, _ = forward_batched(model, device, phase_states, batch_size)
    local_orig = orig_logits[:, start:end]
    local_masks = phase_masks[:, start:end]
    orig_pol = batch_masked_softmax(local_orig, local_masks)

    layer_names = ["L0+GELU", "L1+GELU", "L2+GELU"]
    layer_pairs = [(0, 1), (2, 3), (4, 5)]

    overall_kl: dict[str, float] = {}

    for name, (lin_idx, gelu_idx) in zip(layer_names, layer_pairs):
        if gelu_idx >= len(head):
            continue

        orig_linear = head[lin_idx]
        orig_gelu = head[gelu_idx]

        head[lin_idx] = torch.nn.Identity()
        head[gelu_idx] = torch.nn.Identity()

        bypass_logits, _ = forward_batched(model, device, phase_states, batch_size)
        local_bypass = bypass_logits[:, start:end]
        bypass_pol = batch_masked_softmax(local_bypass, local_masks)

        head[lin_idx] = orig_linear
        head[gelu_idx] = orig_gelu

        kl = kl_divergence_batch(orig_pol, bypass_pol)
        overall_kl[name] = float(np.mean(kl))

    return LayerNecessityResult(
        phase_name=phase_name,
        layer_names=[n for n in layer_names if n in overall_kl],
        overall_kl=overall_kl,
    )


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------


@dataclass
class PolicyHeadResults:
    """All per-phase results bundled together."""

    lens: dict[str, LogitLensResult] = field(default_factory=dict)
    neurons: dict[str, NeuronSpecResult] = field(default_factory=dict)
    necessity: dict[str, LayerNecessityResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_report(results: PolicyHeadResults) -> None:
    """Print all results to console."""
    # --- Logit Lens ---
    print("\n" + "=" * 90)
    print("  1. LOGIT LENS (per-phase head, projecting through final W)")
    print("=" * 90)

    for phase_name in DECISION_PHASE_ORDER:
        if phase_name not in results.lens:
            continue
        r = results.lens[phase_name]
        print(f"\n  [{phase_name}] ({r.num_states} states)")
        print(f"  {'Layer':<8s} {'Agree':>8s} {'KL':>10s} {'FinalProb':>10s}")
        print(f"  {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}")
        for name in r.layer_names:
            print(
                f"  {name:<8s} {r.argmax_agreement[name]:>7.1%} "
                f"{r.mean_kl[name]:>10.4f} "
                f"{r.final_action_prob[name]:>9.1%}"
            )

    # --- Neuron Specialization ---
    if results.neurons:
        print("\n" + "=" * 90)
        print("  2. NEURON SPECIALIZATION (NeuronConductance per action type)")
        print("=" * 90)

        for phase_name in DECISION_PHASE_ORDER:
            if phase_name not in results.neurons:
                continue
            r = results.neurons[phase_name]
            print(f"\n  [{phase_name}] ({r.num_neurons} neurons)")
            print(f"    Dead: {r.dead_fraction:.1%}  "
                  f"Mean spec: {r.specialization_scores.mean():.3f}  "
                  f"Median spec: {np.median(r.specialization_scores):.3f}")

            for name, count in sorted(r.neurons_per_action.items(), key=lambda x: -x[1]):
                bar = "#" * max(1, count * 40 // r.num_neurons)
                print(f"    {name:<12s}: {count:>3d}/{r.num_neurons}  {bar}")

    # --- Layer Necessity ---
    print("\n" + "=" * 90)
    print("  3. LAYER CAUSAL NECESSITY (bypass each Linear+GELU pair)")
    print("=" * 90)

    for phase_name in DECISION_PHASE_ORDER:
        if phase_name not in results.necessity:
            continue
        r = results.necessity[phase_name]
        print(f"\n  [{phase_name}]")
        for name in r.layer_names:
            kl = r.overall_kl[name]
            bar = "#" * max(1, int(kl * 200))
            print(f"    {name:<12s} KL={kl:.4f}  {bar}")

    # --- Cross-phase summary ---
    print("\n" + "=" * 90)
    print("  CROSS-PHASE SUMMARY")
    print("=" * 90)

    print(f"\n  {'Phase':<8s} {'Trunk→Final':>12s} {'Dead%':>7s} {'MostNecLayer':>14s} {'NecKL':>8s}")
    print(f"  {'-' * 8} {'-' * 12} {'-' * 7} {'-' * 14} {'-' * 8}")
    for phase_name in DECISION_PHASE_ORDER:
        lens_r = results.lens.get(phase_name)
        neuron_r = results.neurons.get(phase_name)
        nec_r = results.necessity.get(phase_name)

        agree = f"{lens_r.argmax_agreement['trunk']:.1%}" if lens_r else "—"
        dead = f"{neuron_r.dead_fraction:.1%}" if neuron_r else "—"

        if nec_r and nec_r.overall_kl:
            top = max(nec_r.overall_kl.items(), key=lambda x: x[1])
            nec_layer = top[0]
            nec_kl = f"{top[1]:.4f}"
        else:
            nec_layer = "—"
            nec_kl = "—"

        print(f"  {phase_name:<8s} {agree:>12s} {dead:>7s} {nec_layer:>14s} {nec_kl:>8s}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def format_markdown_report(
    results: PolicyHeadResults,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a machine-readable markdown report with all policy head data."""
    lines: list[str] = [
        f"# Policy Head Analysis (epoch {epoch})\n",
        f"{num_states:,} states from {num_games} games.",
        f"8 per-phase policy heads, each 3 hidden layers at {next(iter(results.lens.values())).layer_names[0]}→hidden_dim.\n"
        if results.lens else "",
    ]

    # --- 1. Logit Lens ---
    lines.append("## 1. Logit Lens\n")
    lines.append("Project intermediate representations through each head's final W.\n")

    for phase_name in DECISION_PHASE_ORDER:
        if phase_name not in results.lens:
            continue
        r = results.lens[phase_name]
        lines.append(f"### {phase_name} ({r.num_states} states)\n")
        lines.append("| Layer | Argmax Agree | Mean KL | Final Act Prob |")
        lines.append("| :--- | ---: | ---: | ---: |")
        for name in r.layer_names:
            lines.append(
                f"| {name} | {r.argmax_agreement[name]:.1%} "
                f"| {r.mean_kl[name]:.4f} "
                f"| {r.final_action_prob[name]:.1%} |"
            )
        lines.append("")

    # --- 2. Neuron Specialization ---
    if results.neurons:
        lines.append("## 2. Neuron Specialization\n")
        lines.append("NeuronConductance per action type in each head's final hidden layer.\n")

        for phase_name in DECISION_PHASE_ORDER:
            if phase_name not in results.neurons:
                continue
            r = results.neurons[phase_name]
            lines.append(f"### {phase_name} ({r.num_neurons} neurons)\n")
            lines.append(f"- **Dead neurons:** {r.dead_fraction:.1%}")
            lines.append(f"- **Mean specialization:** {r.specialization_scores.mean():.3f}")
            lines.append(f"- **Median specialization:** {np.median(r.specialization_scores):.3f}\n")

            lines.append("| Action Type | Neurons | Fraction |")
            lines.append("| :--- | ---: | ---: |")
            for name, count in sorted(r.neurons_per_action.items(), key=lambda x: -x[1]):
                lines.append(f"| {name} | {count}/{r.num_neurons} | {count / r.num_neurons:.1%} |")

            highly_spec = np.where(r.specialization_scores > 0.6)[0]
            lines.append(f"\nHighly specialized neurons (score > 0.6): {len(highly_spec)}/{r.num_neurons}")
            lines.append("")

    # --- 3. Layer Causal Necessity ---
    lines.append("## 3. Layer Causal Necessity\n")
    lines.append("KL divergence when each Linear+GELU pair is replaced with identity.\n")

    for phase_name in DECISION_PHASE_ORDER:
        if phase_name not in results.necessity:
            continue
        r = results.necessity[phase_name]
        lines.append(f"### {phase_name}\n")
        lines.append("| Layer | KL |")
        lines.append("| :--- | ---: |")
        for name in r.layer_names:
            lines.append(f"| {name} | {r.overall_kl[name]:.4f} |")
        lines.append("")

    # --- Cross-phase summary ---
    lines.append("## Cross-Phase Summary\n")
    lines.append("| Phase | States | Trunk->Final | Dead% | Most Nec. Layer | Nec. KL |")
    lines.append("| :--- | ---: | ---: | ---: | :--- | ---: |")

    for phase_name in DECISION_PHASE_ORDER:
        lens_r = results.lens.get(phase_name)
        neuron_r = results.neurons.get(phase_name)
        nec_r = results.necessity.get(phase_name)

        n = str(lens_r.num_states) if lens_r else "—"
        agree = f"{lens_r.argmax_agreement['trunk']:.1%}" if lens_r else "—"
        dead = f"{neuron_r.dead_fraction:.1%}" if neuron_r else "—"

        if nec_r and nec_r.overall_kl:
            top = max(nec_r.overall_kl.items(), key=lambda x: x[1])
            nec_layer = top[0]
            nec_kl = f"{top[1]:.4f}"
        else:
            nec_layer = "—"
            nec_kl = "—"

        lines.append(f"| {phase_name} | {n} | {agree} | {dead} | {nec_layer} | {nec_kl} |")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    results: PolicyHeadResults,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a self-contained HTML report."""
    phase_names = [p for p in DECISION_PHASE_ORDER if p in results.lens]

    # Serialize all data
    lens_data: dict[str, Any] = {}
    for pn in phase_names:
        r = results.lens[pn]
        lens_data[pn] = {
            "layer_names": r.layer_names,
            "argmax_agreement": r.argmax_agreement,
            "mean_kl": r.mean_kl,
            "final_action_prob": r.final_action_prob,
            "num_states": r.num_states,
        }

    neuron_data: dict[str, Any] = {}
    for pn in phase_names:
        if pn not in results.neurons:
            continue
        r = results.neurons[pn]
        highly_spec = []
        for ni in np.where(r.specialization_scores > 0.6)[0]:
            dom = r.dominant_action[ni]
            highly_spec.append({
                "neuron": int(ni),
                "action": r.action_type_names[dom],
                "score": float(r.specialization_scores[ni]),
            })
        neuron_data[pn] = {
            "num_neurons": r.num_neurons,
            "action_type_names": r.action_type_names,
            "neurons_per_action": r.neurons_per_action,
            "dead_fraction": r.dead_fraction,
            "mean_specialization": float(r.specialization_scores.mean()),
            "median_specialization": float(np.median(r.specialization_scores)),
            "highly_specialized": highly_spec,
        }

    necessity_data: dict[str, Any] = {}
    for pn in phase_names:
        if pn not in results.necessity:
            continue
        r = results.necessity[pn]
        necessity_data[pn] = {
            "layer_names": r.layer_names,
            "overall_kl": r.overall_kl,
        }

    all_data = {
        "phase_names": phase_names,
        "lens": lens_data,
        "neurons": neuron_data,
        "necessity": necessity_data,
    }
    data_json = json.dumps(all_data)

    # Build body with placeholder divs for each section
    body = (
        '<h2>1. Logit Lens</h2>\n'
        '<p style="color:#888;font-size:0.85rem">Project intermediate representations '
        "through each head's final W. Shows when decisions crystallize within each phase head.</p>\n"
        '<div id="lens-tabs" class="tab-row"></div>\n'
        '<div id="lens-content"></div>\n'
        '\n'
        '<h2>2. Neuron Specialization</h2>\n'
        '<p style="color:#888;font-size:0.85rem">NeuronConductance per action type '
        "in each head's final hidden layer.</p>\n"
        '<div id="neuron-tabs" class="tab-row"></div>\n'
        '<div id="neuron-content"></div>\n'
        '\n'
        '<h2>3. Layer Causal Necessity</h2>\n'
        '<p style="color:#888;font-size:0.85rem">KL divergence when each '
        "Linear+GELU pair is replaced with identity.</p>\n"
        '<div id="necessity-tabs" class="tab-row"></div>\n'
        '<div id="necessity-content"></div>\n'
        '\n'
        '<h2>Cross-Phase Summary</h2>\n'
        '<table id="tbl-summary"></table>'
    )

    report_js = (
        f'const D = {data_json};\n'
        '\n'
        'function heatColor(v, maxVal) {\n'
        '  if (maxVal <= 0) return "transparent";\n'
        '  const t = Math.min(v / maxVal, 1.0);\n'
        '  return "hsl(" + (120 - t * 120) + ",70%," + (18 + t * 22) + "%)";\n'
        '}\n'
        'function agreeColor(v) {\n'
        '  return "hsl(" + (v * 120) + ",70%," + (18 + v * 22) + "%)";\n'
        '}\n'
        '\n'
        '// Tab system\n'
        'function makeTabs(containerId, contentId, items, renderFn) {\n'
        '  const tabs = document.getElementById(containerId);\n'
        '  const content = document.getElementById(contentId);\n'
        '  let html = "";\n'
        '  items.forEach((name, i) => {\n'
        '    const cls = i === 0 ? "tab active" : "tab";\n'
        '    html += \'<span class="\' + cls + \'" data-idx="\' + i + \'">\' + name + "</span>";\n'
        '  });\n'
        '  tabs.innerHTML = html;\n'
        '  function show(idx) {\n'
        '    tabs.querySelectorAll(".tab").forEach((t, i) => {\n'
        '      t.className = i === idx ? "tab active" : "tab";\n'
        '    });\n'
        '    content.innerHTML = renderFn(items[idx]);\n'
        '  }\n'
        '  tabs.addEventListener("click", e => {\n'
        '    const t = e.target.closest(".tab");\n'
        '    if (t) show(parseInt(t.dataset.idx));\n'
        '  });\n'
        '  if (items.length > 0) show(0);\n'
        '}\n'
        '\n'
        '// 1. Logit Lens\n'
        'makeTabs("lens-tabs", "lens-content", D.phase_names, function(pn) {\n'
        '  const d = D.lens[pn];\n'
        '  if (!d) return "<p>No data</p>";\n'
        '  let html = \'<p style="color:#888;font-size:0.82rem">\' + d.num_states + " states</p>";\n'
        '  html += \'<table><tr><th>Layer</th><th>Argmax Agreement</th><th>Mean KL</th>'
        '<th>Final Act Prob</th></tr>\';\n'
        '  for (const name of d.layer_names) {\n'
        '    const agree = d.argmax_agreement[name];\n'
        '    const bg = agreeColor(agree);\n'
        '    html += "<tr><td>" + name + "</td>";\n'
        '    html += \'<td style="background:\' + bg + \'">\' + (agree * 100).toFixed(1) + "%</td>";\n'
        '    html += "<td>" + d.mean_kl[name].toFixed(4) + "</td>";\n'
        '    html += "<td>" + (d.final_action_prob[name] * 100).toFixed(1) + "%</td></tr>";\n'
        '  }\n'
        '  html += "</table>";\n'
        '  return html;\n'
        '});\n'
        '\n'
        '// 2. Neuron Specialization\n'
        'const neuronPhases = D.phase_names.filter(p => D.neurons[p]);\n'
        'makeTabs("neuron-tabs", "neuron-content", neuronPhases, function(pn) {\n'
        '  const d = D.neurons[pn];\n'
        '  if (!d) return "<p>No data</p>";\n'
        '  let html = \'<div style="margin-bottom:1rem">\';\n'
        '  html += \'<div class="stat-box"><div class="stat-label">Dead Neurons</div>\';\n'
        '  html += \'<div class="stat-value">\' + (d.dead_fraction * 100).toFixed(1) + "%</div></div>";\n'
        '  html += \'<div class="stat-box"><div class="stat-label">Mean Spec</div>\';\n'
        '  html += \'<div class="stat-value">\' + d.mean_specialization.toFixed(3) + "</div></div>";\n'
        '  html += \'<div class="stat-box"><div class="stat-label">Median Spec</div>\';\n'
        '  html += \'<div class="stat-value">\' + d.median_specialization.toFixed(3) + "</div></div>";\n'
        '  html += "</div>";\n'
        '  // Neurons per action type\n'
        '  const entries = Object.entries(d.neurons_per_action).sort((a, b) => b[1] - a[1]);\n'
        '  const maxC = entries.length > 0 ? entries[0][1] : 1;\n'
        '  html += \'<table><tr><th>Action Type</th><th>Count</th><th></th><th>Fraction</th></tr>\';\n'
        '  for (const [name, count] of entries) {\n'
        '    html += "<tr><td>" + name + "</td>";\n'
        '    html += "<td>" + count + "/" + d.num_neurons + "</td>";\n'
        '    html += "<td>" + makeBar(count, maxC, "bar-green") + "</td>";\n'
        '    html += "<td>" + (count / d.num_neurons * 100).toFixed(1) + "%</td></tr>";\n'
        '  }\n'
        '  html += "</table>";\n'
        '  // Highly specialized\n'
        '  if (d.highly_specialized.length > 0) {\n'
        '    html += "<h3>Highly Specialized (score &gt; 0.6)</h3>";\n'
        '    html += \'<table><tr><th>Neuron</th><th>Action</th><th>Score</th></tr>\';\n'
        '    for (const n of d.highly_specialized) {\n'
        '      html += "<tr><td>n" + n.neuron + "</td><td>" + n.action + "</td>";\n'
        '      html += "<td>" + n.score.toFixed(3) + "</td></tr>";\n'
        '    }\n'
        '    html += "</table>";\n'
        '  }\n'
        '  return html;\n'
        '});\n'
        '\n'
        '// 3. Layer Necessity\n'
        'const necPhases = D.phase_names.filter(p => D.necessity[p]);\n'
        'makeTabs("necessity-tabs", "necessity-content", necPhases, function(pn) {\n'
        '  const d = D.necessity[pn];\n'
        '  if (!d) return "<p>No data</p>";\n'
        '  const maxKL = Math.max(...d.layer_names.map(n => d.overall_kl[n]));\n'
        '  let html = \'<table><tr><th>Layer</th><th>KL</th><th></th></tr>\';\n'
        '  for (const name of d.layer_names) {\n'
        '    const kl = d.overall_kl[name];\n'
        '    html += "<tr><td>" + name + "</td>";\n'
        '    html += "<td>" + kl.toFixed(4) + "</td>";\n'
        '    html += "<td>" + makeBar(kl, maxKL, "bar-orange") + "</td></tr>";\n'
        '  }\n'
        '  html += "</table>";\n'
        '  return html;\n'
        '});\n'
        '\n'
        '// Cross-phase summary table\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-summary");\n'
        '  let html = \'<tr><th>Phase</th><th>States</th><th>Trunk→Final</th>\';\n'
        '  html += \'<th>Dead%</th><th>Most Nec. Layer</th><th>Nec. KL</th></tr>\';\n'
        '  for (const pn of D.phase_names) {\n'
        '    html += "<tr><td>" + pn + "</td>";\n'
        '    const lens = D.lens[pn];\n'
        '    html += "<td>" + (lens ? lens.num_states : "—") + "</td>";\n'
        '    const agree = lens ? lens.argmax_agreement["trunk"] : null;\n'
        '    if (agree !== null) {\n'
        '      html += \'<td style="background:\' + agreeColor(agree) + \'">\' + (agree * 100).toFixed(1) + "%</td>";\n'
        '    } else { html += "<td>—</td>"; }\n'
        '    const neur = D.neurons[pn];\n'
        '    html += "<td>" + (neur ? (neur.dead_fraction * 100).toFixed(1) + "%" : "—") + "</td>";\n'
        '    const nec = D.necessity[pn];\n'
        '    if (nec && nec.layer_names.length > 0) {\n'
        '      let bestN = nec.layer_names[0], bestKL = nec.overall_kl[nec.layer_names[0]];\n'
        '      for (const n of nec.layer_names) {\n'
        '        if (nec.overall_kl[n] > bestKL) { bestN = n; bestKL = nec.overall_kl[n]; }\n'
        '      }\n'
        '      html += "<td>" + bestN + "</td><td>" + bestKL.toFixed(4) + "</td>";\n'
        '    } else { html += "<td>—</td><td>—</td>"; }\n'
        '    html += "</tr>";\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();'
    )

    tab_css = (
        '.tab-row { margin: 0.5rem 0; }\n'
        '.tab {\n'
        '  display: inline-block; padding: 4px 12px; margin: 0 2px;\n'
        '  border: 1px solid #2a2a4a; border-radius: 3px;\n'
        '  font-size: 0.8rem; color: #888; cursor: pointer;\n'
        '}\n'
        '.tab:hover { color: #ccc; border-color: #555; }\n'
        '.tab.active { color: #e0e0e0; background: #16213e; border-color: #4a9eff; }'
    )

    extra_css = BAR_CSS + "\n" + STAT_BOX_CSS + "\n" + tab_css

    return html_page(
        f"Policy Head Analysis \u2014 Epoch {epoch}",
        meta=f"{num_states:,} states from {num_games} games. Per-phase analysis of 8 policy heads.",
        body=body,
        script=JS_MAKE_BAR + "\n\n" + report_js,
        extra_css=extra_css,
        max_width=1100,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-phase policy head analysis (logit lens, neuron specialization, causal necessity)"
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
        help="Max states per phase for neuron conductance (slow)",
    )
    parser.add_argument(
        "--neuron-steps", type=int, default=20,
        help="IG steps for neuron conductance",
    )
    parser.add_argument(
        "--skip-neurons", action="store_true",
        help="Skip neuron specialization (slowest analysis)",
    )
    parser.add_argument(
        "--phases", type=str, nargs="+", default=None,
        help="Analyze only these phases (e.g., --phases INVEST ACQ DIV)",
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

    # Determine which phases to analyze
    target_phases = args.phases or list(DECISION_PHASE_ORDER)
    target_phases = [p.upper() for p in target_phases]

    # Build phase ID -> head index mapping
    from core.data import GamePhases
    phase_id_to_name = {
        GamePhases.PHASE_INVEST: "INVEST",
        GamePhases.PHASE_BID_IN_AUCTION: "BID",
        GamePhases.PHASE_ACQUISITION: "ACQ",
        GamePhases.PHASE_CLOSING: "CLOSE",
        GamePhases.PHASE_DIVIDENDS: "DIV",
        GamePhases.PHASE_ISSUE_SHARES: "ISSUE",
        GamePhases.PHASE_IPO: "IPO",
        GamePhases.PHASE_PAR: "PAR",
    }
    sorted_phase_ids = sorted(phase_id_to_name.keys())

    results = PolicyHeadResults()

    for head_idx, phase_id in enumerate(sorted_phase_ids):
        phase_name = phase_id_to_name[phase_id]
        if phase_name not in target_phases:
            continue

        phase_mask = dataset.phases == phase_id
        n_phase = int(phase_mask.sum())
        if n_phase < 5:
            print(f"\n  Skipping {phase_name}: only {n_phase} states")
            continue

        phase_states = dataset.states[phase_mask]
        phase_legal_masks = dataset.legal_masks[phase_mask]

        print(f"\n{'=' * 60}")
        print(f"  {phase_name} ({n_phase} states, head {head_idx})")
        print(f"{'=' * 60}")

        # 1. Logit Lens
        print(f"  Logit lens...")
        results.lens[phase_name] = analyze_logit_lens(
            model, device, phase_states, phase_legal_masks,
            head_idx, batch_size=args.batch_size,
        )

        # 2. Neuron Specialization
        if not args.skip_neurons:
            print(f"  Neuron specialization...")
            results.neurons[phase_name] = analyze_neuron_specialization(
                model, device, phase_states, phase_legal_masks,
                head_idx, config.num_players,
                max_samples=args.max_neuron_samples,
                n_steps=args.neuron_steps,
            )

        # 3. Layer Necessity
        print(f"  Layer causal necessity...")
        results.necessity[phase_name] = analyze_layer_necessity(
            model, device, phase_states, phase_legal_masks,
            head_idx, batch_size=args.batch_size,
        )

    # Print report
    print_report(results)

    # Markdown report
    out_dir = Path("interp/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"policy_head_epoch{epoch}.md"
    md = format_markdown_report(
        results, epoch=epoch,
        num_states=dataset.num_states,
        num_games=dataset.num_games,
    )
    md_path.write_text(md)
    print(f"\nMarkdown report written to {md_path}")

    # HTML report
    html_path = out_dir / f"policy_head_epoch{epoch}.html"
    html = format_html_report(
        results, epoch=epoch,
        num_states=dataset.num_states,
        num_games=dataset.num_games,
    )
    html_path.write_text(html)
    print(f"HTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
