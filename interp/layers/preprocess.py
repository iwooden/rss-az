"""Input preprocessing analysis: signal attenuation, expanded probing, SVD projection.

Three analyses to determine whether the 768->512->256 compression loses
policy-relevant signal:

1. **Signal Attenuation**: For each feature group, measure how much of its signal
   survives each compression step (768->512->256). Cross-reference with ablation KL
   to find features that matter AND lose signal.

2. **Expanded Probing**: Probe at raw input, 768-dim, 512-dim, 256-dim (post-LN),
   and block_0 to track information through preprocessing.

3. **SVD Projection Analysis**: Effective rank and energy at each intermediate
   dimension. Identify which 768-dim singular vectors the learned 768->512 weight
   matrix preserves vs discards, and correlate with feature groups.

Usage:
    .venv/bin/python -m interp.layers.preprocess --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from interp.html import BAR_CSS, html_page, open_file
from interp.utils import (
    InterpDataset,
    batch_masked_softmax,
    build_feature_groups,
    collect_states,
    forward_batched,
    kl_divergence_batch,
    load_model,
)


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _collect_preprocessing_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect activations at each preprocessing stage.

    Preprocessing layout: [0] Linear(→768), [1] GELU, [2] Linear(768→512),
    [3] GELU, [4] Linear(512→256), [5] LayerNorm(256).

    Returns (h768, h512, h256_pre_ln, h256_post_ln) as numpy arrays.
    h768 = post-GELU[1], h512 = post-GELU[3],
    h256_pre_ln = post-Linear[4], h256_post_ln = post-LayerNorm[5].
    """
    model.eval()
    keys = ["768", "512", "256_pre", "256_post"]
    acts: dict[str, list[torch.Tensor]] = {k: [] for k in keys}

    def make_hook(key: str):  # noqa: ANN202
        def hook(
            _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            acts[key].append(out.detach().cpu())
        return hook

    preprocess = model.input_preprocess
    handles = [
        preprocess[1].register_forward_hook(make_hook("768")),      # post-GELU (768-dim)
        preprocess[3].register_forward_hook(make_hook("512")),      # post-GELU (512-dim)
        preprocess[4].register_forward_hook(make_hook("256_pre")),  # post-Linear (256-dim, pre-LN)
        preprocess[5].register_forward_hook(make_hook("256_post")), # post-LayerNorm (256-dim)
    ]

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    return tuple(torch.cat(acts[k], dim=0).numpy() for k in keys)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. Feature group signal attenuation
# ---------------------------------------------------------------------------


@dataclass
class AttenuationRow:
    """Signal attenuation result for one feature group."""

    name: str
    num_features: int
    policy_kl: float
    delta_768: float
    delta_512: float
    delta_256: float       # pre-LN (consistent with 768/512 post-activation)
    delta_256_post_ln: float  # post-LN (what the trunk sees)
    attenuation: float     # delta_256 / delta_768 (pre-LN, consistent comparison)
    lost_signal: float     # policy_kl * (1 - attenuation)


def analyze_signal_attenuation(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    batch_size: int = 256,
) -> list[AttenuationRow]:
    """Measure how much each feature group's signal is preserved vs lost
    through the 768->512->256 compression.

    For each feature group:
    - Zero it out, forward pass, measure activation delta at 768, 512, and 256 dims
    - Compute attenuation = delta_256 / delta_768
    - Cross-reference with policy KL from full forward pass
    """
    groups = build_feature_groups(num_players)

    # Get original activations and policy
    print("  Computing original activations...")
    h768_orig, h512_orig, h256_orig, h256_ln_orig = _collect_preprocessing_activations(
        model, device, dataset.states, batch_size
    )
    orig_logits, _ = forward_batched(model, device, dataset.states, batch_size)
    orig_pol = batch_masked_softmax(orig_logits, dataset.legal_masks)

    # Norms for relative delta (all pre-LN for consistent comparison)
    norm_768 = np.linalg.norm(h768_orig, axis=1, keepdims=True).mean()
    norm_512 = np.linalg.norm(h512_orig, axis=1, keepdims=True).mean()
    norm_256 = np.linalg.norm(h256_orig, axis=1, keepdims=True).mean()
    norm_256_ln = np.linalg.norm(h256_ln_orig, axis=1, keepdims=True).mean()

    rows: list[AttenuationRow] = []
    t0 = time.perf_counter()

    for i, (name, indices) in enumerate(groups):
        ablated = dataset.states.copy()
        ablated[:, indices] = 0.0

        # Get ablated activations
        h768_abl, h512_abl, h256_abl, h256_ln_abl = _collect_preprocessing_activations(
            model, device, ablated, batch_size
        )

        # Signal deltas (relative to mean norm)
        # 768/512/256 are all post-activation/post-linear (pre-LN) for consistent comparison
        d768 = float(np.linalg.norm(h768_orig - h768_abl, axis=1).mean() / norm_768)
        d512 = float(np.linalg.norm(h512_orig - h512_abl, axis=1).mean() / norm_512)
        d256 = float(np.linalg.norm(h256_orig - h256_abl, axis=1).mean() / norm_256)
        d256_ln = float(np.linalg.norm(h256_ln_orig - h256_ln_abl, axis=1).mean() / norm_256_ln)

        # Policy KL
        abl_logits, _ = forward_batched(model, device, ablated, batch_size)
        abl_pol = batch_masked_softmax(abl_logits, dataset.legal_masks)
        kl = float(np.mean(kl_divergence_batch(orig_pol, abl_pol)))

        attn = d256 / d768 if d768 > 1e-8 else 1.0
        lost = kl * (1.0 - attn) if attn < 1.0 else 0.0

        rows.append(AttenuationRow(
            name=name,
            num_features=len(indices),
            policy_kl=kl,
            delta_768=d768,
            delta_512=d512,
            delta_256=d256,
            delta_256_post_ln=d256_ln,
            attenuation=attn,
            lost_signal=lost,
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(groups)} groups ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(groups)} groups in {elapsed:.1f}s")

    rows.sort(key=lambda r: -r.lost_signal)
    return rows


# ---------------------------------------------------------------------------
# 2. Expanded probing through preprocessing layers
# ---------------------------------------------------------------------------


def collect_preprocessing_layer_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Collect activations at raw input, 768-dim, 512-dim, 256-dim, and block_0.

    Returns dict with keys: "raw_input", "preprocess_768", "preprocess_512",
    "preprocess_256", "block_0".
    """
    model.eval()
    acts: dict[str, list[torch.Tensor]] = {"768": [], "512": [], "256": [], "b0": []}

    def make_hook(key: str):  # noqa: ANN202
        def hook(
            _mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor
        ) -> None:
            acts[key].append(out.detach().cpu())
        return hook

    preprocess = model.input_preprocess
    handles = [
        preprocess[1].register_forward_hook(make_hook("768")),  # post-GELU (768-dim)
        preprocess[3].register_forward_hook(make_hook("512")),  # post-GELU (512-dim)
        preprocess[5].register_forward_hook(make_hook("256")),  # post-LayerNorm (256-dim)
        model.blocks[0].register_forward_hook(make_hook("b0")),
    ]

    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    return {
        "raw_input": states,
        "preprocess_768": torch.cat(acts["768"], dim=0).numpy(),
        "preprocess_512": torch.cat(acts["512"], dim=0).numpy(),
        "preprocess_256": torch.cat(acts["256"], dim=0).numpy(),
        "block_0": torch.cat(acts["b0"], dim=0).numpy(),
    }


@dataclass
class PreprocessProbeResult:
    """Probe result across preprocessing layers."""

    probe_name: str
    task_type: str
    metric_name: str
    raw_input: float
    preprocess_768: float
    preprocess_512: float
    preprocess_256: float
    block_0: float
    delta_compression: float  # preprocess_768 - preprocess_256 (overall)


def run_preprocessing_probes(
    model: Any,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    batch_size: int = 256,
) -> list[PreprocessProbeResult]:
    """Train linear probes at each preprocessing stage.

    Probes: model_value_p0, invest_action, acq_action, nw_rank,
    active_leading, winning_player, phase, game_progress.
    """
    from interp.probing import (
        ProbeResult,
        _extract_game_targets,
        _extract_model_targets,
        train_probes_gpu,
    )

    print("  Collecting preprocessing activations...")
    activations = collect_preprocessing_layer_activations(
        model, device, dataset.states, batch_size
    )

    print("  Extracting probe targets...")
    targets = _extract_game_targets(dataset.states, num_players)
    targets.update(_extract_model_targets(
        model, device, dataset.states, dataset.legal_masks, dataset.phases,
        num_players, batch_size,
    ))

    # Focus on the most informative probes
    enabled = {
        "phase", "game_progress", "winning_player", "active_leading",
        "lead_margin", "nw_rank", "model_value_p0", "model_entropy",
        "invest_action", "acq_action", "bid_action", "ipo_action",
        "issue_action", "dividend_level", "par_price_level",
        "value_spread", "action_type",
    }

    print("  Training probes...")
    results: list[ProbeResult] = train_probes_gpu(
        activations, targets, device, enabled_probes=enabled,
    )

    # Reshape into per-probe results
    layer_names = list(activations.keys())
    probe_map: dict[str, dict[str, float]] = {}
    probe_meta: dict[str, tuple[str, str]] = {}

    for r in results:
        if r.probe_name not in probe_map:
            probe_map[r.probe_name] = {}
            probe_meta[r.probe_name] = (r.task_type, r.metric_name)
        probe_map[r.probe_name][r.layer_name] = r.metric

    output: list[PreprocessProbeResult] = []
    for probe_name, layer_scores in sorted(probe_map.items()):
        task_type, metric_name = probe_meta[probe_name]
        raw = layer_scores.get("raw_input", 0.0)
        h768 = layer_scores.get("preprocess_768", 0.0)
        h512 = layer_scores.get("preprocess_512", 0.0)
        h256 = layer_scores.get("preprocess_256", 0.0)
        b0 = layer_scores.get("block_0", 0.0)
        output.append(PreprocessProbeResult(
            probe_name=probe_name,
            task_type=task_type,
            metric_name=metric_name,
            raw_input=raw,
            preprocess_768=h768,
            preprocess_512=h512,
            preprocess_256=h256,
            block_0=b0,
            delta_compression=h768 - h256,
        ))

    output.sort(key=lambda r: -abs(r.delta_compression))
    return output


# ---------------------------------------------------------------------------
# 3. SVD projection analysis
# ---------------------------------------------------------------------------


@dataclass
class SVDProjectionResult:
    """How well the 768->512 weight matrix preserves each SVD component."""

    eff_rank_768: float
    eff_rank_512: float
    eff_rank_256: float       # pre-LN
    eff_rank_256_ln: float    # post-LN (what the trunk sees)
    top_50_energy_768: float
    top_50_energy_512: float
    top_50_energy_256: float  # pre-LN
    top_50_energy_256_ln: float  # post-LN
    # Per singular vector: how much of its energy is preserved by 768->512 step
    preservation_by_sv: np.ndarray  # (768,) — preservation ratio per SV
    # Per feature group: fraction of signal in discarded subspace
    feature_group_discarded: list[tuple[str, float, float]]  # (name, frac_discarded, policy_kl)


def _svd_stats(h: np.ndarray, svd_idx: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute effective rank, top-50 energy, singular values, and right SVs."""
    centered = h - h.mean(axis=0, keepdims=True)
    sub = centered[svd_idx]
    _, S, Vt = np.linalg.svd(sub, full_matrices=False)
    s_norm = S / S.sum()
    s_safe = s_norm[s_norm > 1e-10]
    eff_rank = float(np.exp(-np.sum(s_safe * np.log(s_safe))))
    total_energy = float((S ** 2).sum())
    top_50 = float((S[:50] ** 2).sum() / total_energy) if total_energy > 0 else 0.0
    return eff_rank, top_50, S, Vt


def analyze_svd_projection(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    num_players: int,
    policy_kl_map: dict[str, float] | None = None,
    batch_size: int = 256,
) -> SVDProjectionResult:
    """Analyze which 768-dim components the 768->512 weight matrix preserves.

    1. SVD of 768/512/256-dim activations -> effective rank at each stage
    2. For each 768-dim singular vector, measure how well W_512 preserves it
    3. Per feature group, measure what fraction of signal lands in discarded dims
    """
    print("  Collecting preprocessing activations...")
    h768, h512, h256_pre, h256_post = _collect_preprocessing_activations(
        model, device, states, batch_size,
    )

    # Subsample for SVD if dataset is large
    rng = np.random.default_rng(42)
    n = h768.shape[0]
    svd_idx = rng.choice(n, min(5000, n), replace=False) if n > 5000 else np.arange(n)

    print("  Computing SVD at each intermediate dimension...")
    eff_rank_768, top_50_768, S768, Vt768 = _svd_stats(h768, svd_idx)
    eff_rank_512, top_50_512, _, _ = _svd_stats(h512, svd_idx)
    eff_rank_256, top_50_256, _, _ = _svd_stats(h256_pre, svd_idx)
    eff_rank_256_ln, top_50_256_ln, _, _ = _svd_stats(h256_post, svd_idx)

    # How well does the 768->512 weight matrix preserve each SV?
    W = model.input_preprocess[2].weight.detach().cpu().numpy()  # (512, 768)
    preservation = np.zeros(min(S768.shape[0], Vt768.shape[0]))
    for i in range(preservation.shape[0]):
        preservation[i] = float(np.linalg.norm(W @ Vt768[i]))
    max_pres = preservation.max()
    if max_pres > 0:
        preservation = preservation / max_pres

    # Per feature group: what fraction of signal lands in the poorly-preserved subspace?
    threshold = np.median(preservation[:min(365, len(preservation))])
    discarded_mask = preservation < threshold

    groups = build_feature_groups(num_players)
    W_first = model.input_preprocess[0].weight.detach().cpu().numpy()  # (768, input_dim)

    print("  Computing per-feature-group projection into discarded subspace...")
    group_discarded: list[tuple[str, float, float]] = []
    for name, indices in groups:
        delta_input = np.zeros(states.shape[1], dtype=np.float32)
        delta_input[indices] = np.abs(states[:, indices]).mean(axis=0) + 1e-8

        delta_768 = W_first @ delta_input
        if np.linalg.norm(delta_768) < 1e-8:
            group_discarded.append((name, 0.0, policy_kl_map.get(name, 0.0) if policy_kl_map else 0.0))
            continue

        coeffs = Vt768 @ delta_768
        energy_total = float((coeffs ** 2).sum())
        energy_discarded = float((coeffs[discarded_mask[:len(coeffs)]] ** 2).sum())
        frac_discarded = energy_discarded / energy_total if energy_total > 0 else 0.0

        kl = policy_kl_map.get(name, 0.0) if policy_kl_map else 0.0
        group_discarded.append((name, frac_discarded, kl))

    group_discarded.sort(key=lambda x: -(x[1] * x[2]))

    return SVDProjectionResult(
        eff_rank_768=eff_rank_768,
        eff_rank_512=eff_rank_512,
        eff_rank_256=eff_rank_256,
        eff_rank_256_ln=eff_rank_256_ln,
        top_50_energy_768=top_50_768,
        top_50_energy_512=top_50_512,
        top_50_energy_256=top_50_256,
        top_50_energy_256_ln=top_50_256_ln,
        preservation_by_sv=preservation,
        feature_group_discarded=group_discarded,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def print_attenuation_report(rows: list[AttenuationRow]) -> None:
    """Print signal attenuation results to console."""
    print("\n" + "=" * 100)
    print("  1. FEATURE GROUP SIGNAL ATTENUATION (768 -> 512 -> 256)")
    print("=" * 100)
    print("  Attenuation = delta_256 / delta_768 (1.0 = fully preserved, <1.0 = signal lost)")
    print("  Lost Signal = policy_KL * (1 - attenuation)")
    print()
    print(f"  {'Feature':<30s} {'#':>3s} {'Pol KL':>8s} {'d768':>8s} {'d512':>8s} {'d256':>8s} {'d256LN':>8s} {'Atten':>7s} {'Lost':>8s}")
    print(f"  {'-' * 30} {'-' * 3} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 8}")

    for r in rows[:30]:  # Top 30 by lost signal
        atten_str = f"{r.attenuation:.3f}"
        if r.attenuation < 0.8:
            atten_str = f"{r.attenuation:.3f} !"
        print(
            f"  {r.name:<30s} {r.num_features:>3d} "
            f"{r.policy_kl:>8.4f} {r.delta_768:>8.4f} {r.delta_512:>8.4f} {r.delta_256:>8.4f} "
            f"{r.delta_256_post_ln:>8.4f} {atten_str:>7s} {r.lost_signal:>8.4f}"
        )

    # Summary stats
    attenuations = [r.attenuation for r in rows if r.delta_768 > 0.001]
    if attenuations:
        print(f"\n  Attenuation stats (groups with measurable signal):")
        print(f"    Mean: {np.mean(attenuations):.3f}  Median: {np.median(attenuations):.3f}")
        print(f"    Min: {np.min(attenuations):.3f}  Max: {np.max(attenuations):.3f}")
        below_80 = [r for r in rows if r.attenuation < 0.8 and r.delta_768 > 0.001]
        print(f"    Groups with attenuation < 0.8: {len(below_80)}")
        total_lost = sum(r.lost_signal for r in rows)
        print(f"    Total lost signal (sum of KL * (1-att)): {total_lost:.4f}")


def print_probing_report(results: list[PreprocessProbeResult]) -> None:
    """Print preprocessing probing results to console."""
    print("\n" + "=" * 100)
    print("  2. LINEAR PROBING THROUGH PREPROCESSING")
    print("=" * 100)
    print("  Tracks information through: raw_input -> 768 -> 512 -> 256 (post-LN) -> block_0")
    print()
    print(
        f"  {'Probe':<24s} {'Type':>4s} {'Raw':>8s} {'768':>8s} {'512':>8s} {'256':>8s} "
        f"{'B0':>8s} {'d(comp)':>8s}"
    )
    print(
        f"  {'-' * 24} {'-' * 4} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} "
        f"{'-' * 8} {'-' * 8}"
    )

    for r in results:
        delta_str = f"{r.delta_compression:+.4f}"
        if r.delta_compression > 0.01:
            delta_str += " !"  # lost in compression
        print(
            f"  {r.probe_name:<24s} {r.metric_name:>4s} "
            f"{r.raw_input:>8.4f} {r.preprocess_768:>8.4f} {r.preprocess_512:>8.4f} "
            f"{r.preprocess_256:>8.4f} {r.block_0:>8.4f} {delta_str:>8s}"
        )

    # Highlight compression losses
    losses = [(r.probe_name, r.delta_compression) for r in results if r.delta_compression > 0.005]
    if losses:
        print(f"\n  Probes with notable compression loss (768 > 256):")
        for name, delta in sorted(losses, key=lambda x: -x[1]):
            print(f"    {name}: {delta:+.4f}")


def print_svd_report(result: SVDProjectionResult) -> None:
    """Print SVD projection analysis to console."""
    print("\n" + "=" * 100)
    print("  3. SVD PROJECTION ANALYSIS")
    print("=" * 100)

    for label, dim, eff, top50 in [
        ("768-dim (post-GELU)",   768, result.eff_rank_768,    result.top_50_energy_768),
        ("512-dim (post-GELU)",   512, result.eff_rank_512,    result.top_50_energy_512),
        ("256-dim (pre-LN)",      256, result.eff_rank_256,    result.top_50_energy_256),
        ("256-dim (post-LN)",     256, result.eff_rank_256_ln, result.top_50_energy_256_ln),
    ]:
        print(f"\n  {label}:")
        print(f"    Effective rank: {eff:.1f} / {dim} ({eff / dim * 100:.1f}%)")
        print(f"    Top-50 energy: {top50 * 100:.1f}%")

    p = result.preservation_by_sv
    print(f"\n  768->512 weight matrix preservation of 768-dim singular vectors:")
    print(f"    SVs 1-10:   mean={p[:10].mean():.3f}")
    print(f"    SVs 11-50:  mean={p[10:50].mean():.3f}")
    print(f"    SVs 51-100: mean={p[50:100].mean():.3f}")
    print(f"    SVs 101-200: mean={p[100:200].mean():.3f}")
    print(f"    SVs 201-365: mean={p[200:365].mean():.3f}")
    print(f"    SVs 366+:   mean={p[365:].mean():.3f}")

    well_preserved = np.sum(p[:365] > 0.5)
    poorly_preserved = np.sum(p[:365] < 0.3)
    print(f"\n    Well-preserved (>0.5): {well_preserved} / 365 effective dims")
    print(f"    Poorly-preserved (<0.3): {poorly_preserved} / 365 effective dims")

    print(f"\n  Feature groups with most signal in discarded subspace:")
    print(f"  {'Feature':<30s} {'%Discarded':>10s} {'Policy KL':>10s} {'Weighted':>10s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")
    for name, frac, kl in result.feature_group_discarded[:20]:
        if frac < 0.001 and kl < 0.001:
            continue
        print(f"  {name:<30s} {frac * 100:>9.1f}% {kl:>10.4f} {frac * kl:>10.4f}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def format_markdown_report(
    attenuation_rows: list[AttenuationRow],
    probe_results: list[PreprocessProbeResult],
    svd_result: SVDProjectionResult,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a machine-readable markdown report for preprocessing analysis."""
    lines: list[str] = [
        f"# Preprocessing Analysis (epoch {epoch})\n",
        f"{num_states:,} states from {num_games} games.",
        f"Preprocessing: Linear(input->768) -> GELU -> Linear(768->512) -> GELU -> Linear(512->256) -> LayerNorm.\n",
    ]

    # --- 1. Signal Attenuation ---
    lines.append("## 1. Signal Attenuation (768 -> 512 -> 256)\n")
    lines.append("Attenuation = delta_256 / delta_768 (1.0 = fully preserved). Sorted by lost signal.\n")

    lines.append("| Feature | # | Policy KL | d768 | d512 | d256 | d256(LN) | Atten | Lost |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in attenuation_rows:
        lines.append(
            f"| {r.name} | {r.num_features} | {r.policy_kl:.4f} "
            f"| {r.delta_768:.4f} | {r.delta_512:.4f} | {r.delta_256:.4f} "
            f"| {r.delta_256_post_ln:.4f} | {r.attenuation:.3f} | {r.lost_signal:.4f} |"
        )

    attenuations = [r.attenuation for r in attenuation_rows if r.delta_768 > 0.001]
    if attenuations:
        lines.append("")
        lines.append(f"- **Mean attenuation:** {np.mean(attenuations):.3f}")
        lines.append(f"- **Median attenuation:** {np.median(attenuations):.3f}")
        lines.append(f"- **Total lost signal:** {sum(r.lost_signal for r in attenuation_rows):.4f}")

    # --- 2. Probing ---
    lines.append("")
    lines.append("## 2. Linear Probing Through Preprocessing\n")

    lines.append("| Probe | Type | Raw | 768 | 512 | 256 | B0 | d(comp) |")
    lines.append("| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in probe_results:
        sign = "+" if r.delta_compression >= 0 else ""
        lines.append(
            f"| {r.probe_name} | {r.metric_name} | {r.raw_input:.4f} "
            f"| {r.preprocess_768:.4f} | {r.preprocess_512:.4f} "
            f"| {r.preprocess_256:.4f} | {r.block_0:.4f} | {sign}{r.delta_compression:.4f} |"
        )

    # --- 3. SVD ---
    lines.append("")
    lines.append("## 3. SVD Projection Analysis\n")

    lines.append("| Dimension | Effective Rank | Utilization | Top-50 Energy |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for label, dim, eff, top50 in [
        ("768 (post-GELU)", 768, svd_result.eff_rank_768,    svd_result.top_50_energy_768),
        ("512 (post-GELU)", 512, svd_result.eff_rank_512,    svd_result.top_50_energy_512),
        ("256 (pre-LN)",    256, svd_result.eff_rank_256,    svd_result.top_50_energy_256),
        ("256 (post-LN)",   256, svd_result.eff_rank_256_ln, svd_result.top_50_energy_256_ln),
    ]:
        lines.append(f"| {label} | {eff:.1f} / {dim} | {eff / dim * 100:.1f}% | {top50 * 100:.1f}% |")

    p = svd_result.preservation_by_sv
    lines.append("")
    lines.append("### 768->512 Weight Matrix SV Preservation\n")
    lines.append("| SV Range | Mean Preservation |")
    lines.append("| :--- | ---: |")
    for label, sl in [
        ("1-10", slice(0, 10)), ("11-50", slice(10, 50)), ("51-100", slice(50, 100)),
        ("101-200", slice(100, 200)), ("201-365", slice(200, 365)), ("366+", slice(365, None)),
    ]:
        lines.append(f"| SVs {label} | {p[sl].mean():.3f} |")

    lines.append("")
    lines.append("### Feature Groups in Discarded Subspace\n")
    lines.append("| Feature | % Discarded | Policy KL | Weighted |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for name, frac, kl in svd_result.feature_group_discarded:
        if frac < 0.001 and kl < 0.001:
            continue
        lines.append(f"| {name} | {frac * 100:.1f}% | {kl:.4f} | {frac * kl:.4f} |")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def format_html_report(
    attenuation_rows: list[AttenuationRow],
    probe_results: list[PreprocessProbeResult],
    svd_result: SVDProjectionResult,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a self-contained HTML report for preprocessing analysis."""
    attenuation_json = json.dumps([
        {
            "name": r.name,
            "num_features": r.num_features,
            "policy_kl": r.policy_kl,
            "delta_768": r.delta_768,
            "delta_512": r.delta_512,
            "delta_256": r.delta_256,
            "delta_256_ln": r.delta_256_post_ln,
            "attenuation": r.attenuation,
            "lost_signal": r.lost_signal,
        }
        for r in attenuation_rows
    ])

    probe_json = json.dumps([
        {
            "probe_name": r.probe_name,
            "task_type": r.task_type,
            "metric_name": r.metric_name,
            "raw_input": r.raw_input,
            "preprocess_768": r.preprocess_768,
            "preprocess_512": r.preprocess_512,
            "preprocess_256": r.preprocess_256,
            "block_0": r.block_0,
            "delta_compression": r.delta_compression,
        }
        for r in probe_results
    ])

    # SVD: preservation by SV range
    p = svd_result.preservation_by_sv
    sv_ranges = [
        {"range": "SVs 1-10", "mean": float(p[:10].mean())},
        {"range": "SVs 11-50", "mean": float(p[10:50].mean())},
        {"range": "SVs 51-100", "mean": float(p[50:100].mean())},
        {"range": "SVs 101-200", "mean": float(p[100:200].mean())},
        {"range": "SVs 201-365", "mean": float(p[200:365].mean())},
        {"range": "SVs 366+", "mean": float(p[365:].mean())},
    ]
    svd_summary_json = json.dumps({
        "eff_rank_768": svd_result.eff_rank_768,
        "eff_rank_512": svd_result.eff_rank_512,
        "eff_rank_256": svd_result.eff_rank_256,
        "eff_rank_256_ln": svd_result.eff_rank_256_ln,
        "top_50_energy_768": svd_result.top_50_energy_768,
        "top_50_energy_512": svd_result.top_50_energy_512,
        "top_50_energy_256": svd_result.top_50_energy_256,
        "top_50_energy_256_ln": svd_result.top_50_energy_256_ln,
        "sv_ranges": sv_ranges,
    })

    svd_groups_json = json.dumps([
        {"name": name, "frac_discarded": frac, "policy_kl": kl}
        for name, frac, kl in svd_result.feature_group_discarded
    ])

    body = (
        '<h2>1. Signal Attenuation (768 &rarr; 512 &rarr; 256)</h2>\n'
        '<p style="color:#888;font-size:0.85rem">\n'
        '  Attenuation = delta_256 / delta_768 (1.0 = fully preserved). Lost Signal = policy_KL * (1 - attenuation). Sorted by lost signal descending.\n'
        '</p>\n'
        '<table id="tbl-attenuation"></table>\n'
        '\n'
        '<h2>2. Linear Probing Through Preprocessing</h2>\n'
        '<p style="color:#888;font-size:0.85rem">\n'
        '  Information tracked through raw_input &rarr; 768 &rarr; 512 &rarr; 256 (post-LN) &rarr; block_0. Cells highlighted where delta(compression) &gt; 0.01.\n'
        '</p>\n'
        '<table id="tbl-probing"></table>\n'
        '\n'
        '<h2>3. SVD Projection Analysis</h2>\n'
        '<p style="color:#888;font-size:0.85rem">\n'
        '  Effective rank at each preprocessing stage. 768&rarr;512 weight matrix SV preservation.\n'
        '</p>\n'
        '<div id="svd-summary"></div>\n'
        '<h3 style="color:#aaa;font-size:0.95rem;margin-top:1.5rem">Preservation by Singular Vector Range</h3>\n'
        '<table id="tbl-svd-ranges"></table>\n'
        '<h3 style="color:#aaa;font-size:0.95rem;margin-top:1.5rem">Feature Groups in Discarded Subspace</h3>\n'
        '<table id="tbl-svd-groups"></table>'
    )

    data_js = (
        f"const attenuation = {attenuation_json};\n"
        f"const probes = {probe_json};\n"
        f"const svdSummary = {svd_summary_json};\n"
        f"const svdGroups = {svd_groups_json};"
    )

    report_js = (
        'function heatColor(val) {\n'
        '  // 0.0 = deep red, 0.5 = yellow, 1.0 = green (HSL)\n'
        '  const h = Math.max(0, Math.min(1, val)) * 120;\n'
        "  return 'hsl(' + h + ', 70%, 25%)';\n"
        '}\n'
        '\n'
        '// --- 1. Signal Attenuation table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-attenuation");\n'
        "  let html = '<tr><th>Feature</th><th>#</th><th>Policy KL</th><th>d768</th><th>d512</th><th>d256</th><th>d256(LN)</th><th>Attenuation</th><th>Lost Signal</th></tr>';\n"
        '  for (const r of attenuation) {\n'
        '    const bg = heatColor(r.attenuation);\n'
        '    const barW = Math.max(0, Math.min(100, r.attenuation * 100));\n'
        "    html += '<tr><td>' + r.name + '</td>' +\n"
        "      '<td>' + r.num_features + '</td>' +\n"
        "      '<td>' + r.policy_kl.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.delta_768.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.delta_512.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.delta_256.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.delta_256_ln.toFixed(4) + '</td>' +\n"
        '      \'<td style="text-align:left"><span class="bar-container" style="width:80px;background:#111;border-radius:2px">\' +\n'
        "        '<span class=\"bar\" style=\"width:' + barW + '%;background:' + bg + '\"></span></span> ' +\n"
        "        r.attenuation.toFixed(3) + '</td>' +\n"
        "      '<td>' + r.lost_signal.toFixed(4) + '</td></tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- 2. Probing table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-probing");\n'
        "  let html = '<tr><th>Probe</th><th>Type</th><th>Raw</th><th>768</th><th>512</th><th>256</th><th>B0</th><th>d(comp)</th></tr>';\n"
        '  for (const r of probes) {\n'
        "    const cls = r.delta_compression > 0.01 ? ' class=\"highlight\"' : '';\n"
        "    html += '<tr><td>' + r.probe_name + '</td>' +\n"
        "      '<td>' + r.metric_name + '</td>' +\n"
        "      '<td>' + r.raw_input.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.preprocess_768.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.preprocess_512.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.preprocess_256.toFixed(4) + '</td>' +\n"
        "      '<td>' + r.block_0.toFixed(4) + '</td>' +\n"
        "      '<td' + cls + '>' + (r.delta_compression >= 0 ? '+' : '') + r.delta_compression.toFixed(4) + '</td></tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- 3. SVD summary ---\n'
        '(function() {\n'
        '  const s = svdSummary;\n'
        '  const div = document.getElementById("svd-summary");\n'
        '  div.innerHTML =\n'
        '    \'<table style="width:auto">\' +\n'
        '    \'<tr><th style="text-align:left">Metric</th><th>768-dim</th><th>512-dim</th><th>256-dim (pre-LN)</th><th>256-dim (post-LN)</th></tr>\' +\n'
        "    '<tr><td>Effective rank</td><td>' + s.eff_rank_768.toFixed(1) + '</td><td>' + s.eff_rank_512.toFixed(1) + '</td><td>' + s.eff_rank_256.toFixed(1) + '</td><td>' + s.eff_rank_256_ln.toFixed(1) + '</td></tr>' +\n"
        "    '<tr><td>Top-50 energy</td><td>' + (s.top_50_energy_768 * 100).toFixed(1) + '%</td><td>' + (s.top_50_energy_512 * 100).toFixed(1) + '%</td><td>' + (s.top_50_energy_256 * 100).toFixed(1) + '%</td><td>' + (s.top_50_energy_256_ln * 100).toFixed(1) + '%</td></tr>' +\n"
        "    '</table>';\n"
        '})();\n'
        '\n'
        '// --- 3a. SV range preservation table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-svd-ranges");\n'
        '  const ranges = svdSummary.sv_ranges;\n'
        "  let html = '<tr><th>SV Range</th><th>Mean Preservation</th><th></th></tr>';\n"
        '  for (const r of ranges) {\n'
        '    const barW = Math.max(0, Math.min(100, r.mean * 100));\n'
        '    const bg = heatColor(r.mean);\n'
        "    html += '<tr><td>' + r.range + '</td>' +\n"
        "      '<td>' + r.mean.toFixed(3) + '</td>' +\n"
        '      \'<td style="text-align:left"><span class="bar-container" style="background:#111;border-radius:2px">\' +\n'
        "        '<span class=\"bar\" style=\"width:' + barW + '%;background:' + bg + '\"></span></span></td></tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();\n'
        '\n'
        '// --- 3b. Feature group discarded subspace table ---\n'
        '(function() {\n'
        '  const tbl = document.getElementById("tbl-svd-groups");\n'
        "  let html = '<tr><th>Feature</th><th>% Discarded</th><th>Policy KL</th><th>Weighted</th></tr>';\n"
        '  for (const r of svdGroups) {\n'
        '    if (r.frac_discarded < 0.001 && r.policy_kl < 0.001) continue;\n'
        '    const weighted = r.frac_discarded * r.policy_kl;\n'
        "    html += '<tr><td>' + r.name + '</td>' +\n"
        "      '<td>' + (r.frac_discarded * 100).toFixed(1) + '%</td>' +\n"
        "      '<td>' + r.policy_kl.toFixed(4) + '</td>' +\n"
        "      '<td>' + weighted.toFixed(4) + '</td></tr>';\n"
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '})();'
    )

    extra_css = BAR_CSS + "\n" + ".highlight { background: rgba(233, 169, 69, 0.15); }"

    return html_page(
        f"Preprocessing Analysis \u2014 Epoch {epoch}",
        meta=f"{num_states:,} states from {num_games} games.",
        body=body,
        script=data_js + "\n\n" + report_js,
        extra_css=extra_css,
        max_width=1100,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Input preprocessing analysis (signal attenuation, probing, SVD)"
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

    # --- Analysis 1: Signal Attenuation ---
    print(f"\nRunning signal attenuation analysis ({dataset.num_states} states)...")
    attenuation_rows = analyze_signal_attenuation(
        model, device, dataset, config.num_players, batch_size=args.batch_size,
    )
    print_attenuation_report(attenuation_rows)

    # Build policy KL map for cross-referencing in SVD analysis
    policy_kl_map = {r.name: r.policy_kl for r in attenuation_rows}

    # --- Analysis 2: Expanded Probing ---
    print(f"\nRunning preprocessing probes...")
    probe_results = run_preprocessing_probes(
        model, device, dataset, config.num_players, batch_size=args.batch_size,
    )
    print_probing_report(probe_results)

    # --- Analysis 3: SVD Projection ---
    print(f"\nRunning SVD projection analysis...")
    svd_result = analyze_svd_projection(
        model, device, dataset.states, config.num_players,
        policy_kl_map=policy_kl_map, batch_size=args.batch_size,
    )
    print_svd_report(svd_result)

    print(f"\n{'=' * 100}")
    print(f"  SUMMARY")
    print(f"{'=' * 100}")

    # Key findings
    total_lost = sum(r.lost_signal for r in attenuation_rows)
    worst_atten = sorted(
        [r for r in attenuation_rows if r.policy_kl > 0.01],
        key=lambda r: r.attenuation,
    )[:5]

    print(f"\n  Total lost signal (sum of KL * (1-att)): {total_lost:.4f}")
    if worst_atten:
        print(f"\n  Most attenuated features (high KL, low preservation):")
        for r in worst_atten:
            print(f"    {r.name}: attenuation={r.attenuation:.3f}, KL={r.policy_kl:.4f}")

    compression_losses = [r for r in probe_results if r.delta_compression > 0.005]
    if compression_losses:
        print(f"\n  Probes losing information in 768->512->256 compression:")
        for r in sorted(compression_losses, key=lambda x: -x.delta_compression):
            print(f"    {r.probe_name}: 768={r.preprocess_768:.4f} -> 256={r.preprocess_256:.4f} (d={r.delta_compression:+.4f})")
    else:
        print(f"\n  No probes show significant information loss in 768->512->256 compression.")

    # --- Write markdown ---
    md_path = Path("interp/data") / f"preprocess_epoch{epoch}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_markdown_report(
        attenuation_rows, probe_results, svd_result,
        epoch=epoch, num_states=dataset.num_states, num_games=dataset.num_games,
    ))
    print(f"\nMarkdown written to {md_path}")

    # --- Write HTML ---
    html_path = md_path.with_suffix(".html")
    html = format_html_report(
        attenuation_rows, probe_results, svd_result,
        epoch=epoch, num_states=dataset.num_states, num_games=dataset.num_games,
    )
    html_path.write_text(html)
    print(f"HTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
