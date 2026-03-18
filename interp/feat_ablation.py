"""Feature ablation: measure model sensitivity to auction slot info.

Zeroes out the auction slot info region (5 floats per auction slot: stars,
low_price, face_value, high_price, income) and measures how the model's
policy and value outputs change. With --breakdown, ablates each sub-feature
(stars, prices, income) independently to identify which signals drive
model decisions.

Usage:
    .venv/bin/python -m interp.feat_ablation
    .venv/bin/python -m interp.feat_ablation --breakdown
    .venv/bin/python -m interp.feat_ablation --num-games 100 --save-data interp/data/states.npz
    .venv/bin/python -m interp.feat_ablation --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from core.state import get_layout
from interp.utils import InterpDataset, collect_states, load_model

# Phase name mapping for readable output
_PHASE_NAMES = {
    0: "INVEST",
    1: "BID",
    2: "WRAP_UP",
    3: "ACQ",
    4: "CLOSING",
    5: "INCOME",
    6: "DIVIDENDS",
    7: "END_CARD",
    8: "ISSUE",
    9: "IPO",
    10: "GAME_OVER",
}


def _batch_masked_softmax(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Apply legal action mask and softmax to batched logits via torch."""
    t = torch.from_numpy(logits)
    t = t.masked_fill(torch.from_numpy(masks) <= 0, -1e9)
    return torch.softmax(t, dim=-1).numpy()


def _kl_divergence_batch(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-sample KL(P || Q), shape (N,)."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)


def _forward_batched(
    model: torch.nn.Module,
    device: torch.device,
    states: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model forward on states in batches. Returns (logits, values)."""
    logits_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    model.eval()
    with torch.no_grad():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            x = torch.from_numpy(states[i:j]).to(device)
            with torch.autocast(
                device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                lo, va = model(x)
            logits_list.append(lo.float().cpu().numpy())
            values_list.append(va.float().cpu().numpy())

    return np.concatenate(logits_list), np.concatenate(values_list)


def _compute_metrics(
    orig_logits: np.ndarray,
    orig_values: np.ndarray,
    abl_logits: np.ndarray,
    abl_values: np.ndarray,
    masks: np.ndarray,
    phases: np.ndarray,
) -> dict[str, object]:
    """Compute all ablation metrics from original vs ablated model outputs."""
    orig_pol = _batch_masked_softmax(orig_logits, masks)
    abl_pol = _batch_masked_softmax(abl_logits, masks)

    kl_per_sample = _kl_divergence_batch(orig_pol, abl_pol)

    masked_orig = orig_logits.copy()
    masked_abl = abl_logits.copy()
    masked_orig[masks <= 0] = -1e9
    masked_abl[masks <= 0] = -1e9
    top1_match = np.argmax(masked_orig, axis=-1) == np.argmax(masked_abl, axis=-1)

    val_diff = orig_values - abl_values

    results: dict[str, object] = {
        "policy_kl": float(np.mean(kl_per_sample)),
        "policy_kl_median": float(np.median(kl_per_sample)),
        "policy_kl_p95": float(np.percentile(kl_per_sample, 95)),
        "top1_agreement": float(np.mean(top1_match)),
        "value_mse": float(np.mean(val_diff**2)),
        "value_mae": float(np.mean(np.abs(val_diff))),
        "value_max_diff": float(np.max(np.abs(val_diff))),
        "value_correlation": float(
            np.corrcoef(orig_values.ravel(), abl_values.ravel())[0, 1]
        ),
        "per_player_value_mse": [
            float(np.mean(val_diff[:, p] ** 2))
            for p in range(orig_values.shape[1])
        ],
    }

    phase_breakdown: dict[str, dict[str, float]] = {}
    for phase_id in sorted(set(phases)):
        idx = phases == phase_id
        count = int(np.sum(idx))
        if count == 0:
            continue
        name = _PHASE_NAMES.get(phase_id, str(phase_id))
        phase_breakdown[name] = {
            "count": count,
            "policy_kl": float(np.mean(kl_per_sample[idx])),
            "top1_agreement": float(np.mean(top1_match[idx])),
            "value_mse": float(np.mean(val_diff[idx] ** 2)),
        }
    results["per_phase"] = phase_breakdown
    return results


def _build_auction_slot_subfeature_indices(offset: int, num_slots: int) -> dict[str, np.ndarray]:
    """Build index arrays for each auction slot sub-feature.

    Per slot (5 floats): stars(1), low_price(1), face_value(1), high_price(1), income(1)
    """
    groups: dict[str, list[int]] = {
        "stars": [],
        "low_price": [],
        "face_value": [],
        "high_price": [],
        "income": [],
    }
    for s in range(num_slots):
        base = offset + s * 5
        groups["stars"].append(base + 0)
        groups["low_price"].append(base + 1)
        groups["face_value"].append(base + 2)
        groups["high_price"].append(base + 3)
        groups["income"].append(base + 4)

    return {k: np.array(v) for k, v in groups.items()}


def run_ablation(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
    ablate_indices: np.ndarray | None = None,
    orig_cache: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, object], tuple[np.ndarray, np.ndarray]]:
    """Zero out specified features and measure output changes.

    Args:
        ablate_indices: Feature indices to zero. If None, zeros auction slot info region.
        orig_cache: Pre-computed (orig_logits, orig_values) to avoid redundant
            forward passes when running multiple ablations on the same data.

    Returns:
        (results_dict, (orig_logits, orig_values)) — the latter can be passed
        as orig_cache to subsequent calls.
    """
    layout = get_layout(3)
    states = dataset.states
    masks = dataset.legal_masks

    if ablate_indices is None:
        ablate_indices = np.arange(
            layout.auction_slot_info_offset,
            layout.auction_slot_info_offset + layout.auction_slot_info_size,
        )

    # Original forward pass (compute once, reuse)
    if orig_cache is not None:
        orig_logits, orig_values = orig_cache
    else:
        orig_logits, orig_values = _forward_batched(model, device, states, batch_size)

    # Ablated forward pass
    ablated = states.copy()
    ablated[:, ablate_indices] = 0.0
    abl_logits, abl_values = _forward_batched(model, device, ablated, batch_size)

    metrics = _compute_metrics(
        orig_logits, orig_values, abl_logits, abl_values, masks, dataset.phases
    )
    return metrics, (orig_logits, orig_values)


def run_auction_slot_breakdown(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    batch_size: int = 256,
    orig_cache: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, dict[str, object]]:
    """Ablate each auction slot sub-feature group independently.

    Returns dict mapping group name to its ablation metrics.
    """
    layout = get_layout(3)
    groups = _build_auction_slot_subfeature_indices(
        layout.auction_slot_info_offset, layout.num_players
    )

    # Also add a combined "all" group
    groups["all"] = np.arange(
        layout.auction_slot_info_offset,
        layout.auction_slot_info_offset + layout.auction_slot_info_size,
    )

    # Compute original outputs once
    if orig_cache is None:
        orig_cache = _forward_batched(model, device, dataset.states, batch_size)

    results: dict[str, dict[str, object]] = {}
    for name, indices in groups.items():
        metrics, _ = run_ablation(
            model, device, dataset, batch_size,
            ablate_indices=indices, orig_cache=orig_cache,
        )
        metrics["num_features"] = len(indices)
        results[name] = metrics

    return results


def _print_results(
    results: dict[str, object],
    label: str,
    num_features: int,
    visible_size: int,
) -> None:
    """Pretty-print ablation results for one feature group."""
    print()
    print("=" * 65)
    print(f"  {label}")
    print("=" * 65)
    print(f"  Features zeroed:  {num_features} / {visible_size} "
          f"({num_features / visible_size:.1%} of input)")
    print()

    print("  Policy impact:")
    print(f"    KL(orig || ablated):  {results['policy_kl']:.6f}  "
          f"(median {results['policy_kl_median']:.6f}, p95 {results['policy_kl_p95']:.6f})")
    print(f"    Top-1 agreement:      {results['top1_agreement']:.1%}")
    print()

    print("  Value impact:")
    print(f"    MSE:           {results['value_mse']:.6f}")
    print(f"    MAE:           {results['value_mae']:.6f}")
    print(f"    Max |diff|:    {results['value_max_diff']:.6f}")
    print(f"    Correlation:   {results['value_correlation']:.6f}")
    per_player = results["per_player_value_mse"]
    assert isinstance(per_player, list)
    for p, mse in enumerate(per_player):
        print(f"    Player {p} MSE:  {mse:.6f}")
    print()

    per_phase = results["per_phase"]
    assert isinstance(per_phase, dict)
    if per_phase:
        print("  Per-phase breakdown:")
        print(f"    {'Phase':<12} {'Count':>6} {'KL':>10} {'Top-1':>8} {'Val MSE':>10}")
        print(f"    {'-'*12} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
        for name, info in per_phase.items():
            assert isinstance(info, dict)
            print(
                f"    {name:<12} {info['count']:>6} "
                f"{info['policy_kl']:>10.6f} {info['top1_agreement']:>7.1%} "
                f"{info['value_mse']:>10.6f}"
            )
    print()


def _print_breakdown_summary(
    breakdown: dict[str, dict[str, object]],
    visible_size: int,
) -> None:
    """Print a comparison table across all sub-feature groups."""
    print()
    print("=" * 75)
    print("  AUCTION SLOT SUB-FEATURE COMPARISON")
    print("=" * 75)
    print(
        f"  {'Group':<14} {'# Feats':>8} {'% Input':>8} "
        f"{'KL':>10} {'Top-1':>8} {'Val MSE':>10}"
    )
    print(
        f"  {'-'*14} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*10}"
    )

    # Print individual fields first, then composites
    field_order = ["stars", "low_price", "face_value", "high_price",
                   "income", "all"]
    for name in field_order:
        if name not in breakdown:
            continue
        r = breakdown[name]
        n = r["num_features"]
        assert isinstance(n, int)
        pct = n / visible_size
        print(
            f"  {name:<14} {n:>8} {pct:>7.1%} "
            f"{r['policy_kl']:>10.6f} {r['top1_agreement']:>7.1%} "
            f"{r['value_mse']:>10.6f}"
        )
    print()

    # Highlight the dominant contributor
    field_names = [n for n in field_order if n in breakdown and n != "all_scalars"]
    max_kl_name = max(field_names, key=lambda n: breakdown[n]["policy_kl"])  # type: ignore[arg-type]
    max_kl = breakdown[max_kl_name]
    print(f"  Highest policy impact: {max_kl_name} (KL={max_kl['policy_kl']:.6f})")

    max_val_name = max(field_names, key=lambda n: breakdown[n]["value_mse"])  # type: ignore[arg-type]
    max_val = breakdown[max_val_name]
    print(f"  Highest value impact:  {max_val_name} (MSE={max_val['value_mse']:.6f})")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature ablation: measure model sensitivity to static company data"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint path (default: latest)"
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--num-games", type=int, default=50,
        help="Games to play for state collection (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--load-data", type=str, default=None,
        help="Load pre-collected states from .npz instead of playing games",
    )
    parser.add_argument(
        "--save-data", type=str, default=None,
        help="Save collected states to .npz for reuse",
    )
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Also ablate each auction slot sub-feature (stars, prices, income) independently",
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
        print(f"\nCollecting states from {args.num_games} games (policy sampling, no MCTS)...")
        dataset = collect_states(
            model,
            config,
            device,
            num_games=args.num_games,
            seed=args.seed,
            checkpoint_path=str(args.checkpoint or f"latest (epoch {epoch})"),
        )
        if args.save_data:
            dataset.save(args.save_data)

    # Phase distribution summary
    phase_counts = Counter(dataset.phases.tolist())
    print(f"\nPhase distribution ({dataset.num_states} total states):")
    for phase_id in sorted(phase_counts):
        name = _PHASE_NAMES.get(phase_id, str(phase_id))
        count = phase_counts[phase_id]
        print(f"  {name:<12} {count:>6} ({count / dataset.num_states:>5.1%})")

    layout = get_layout(config.num_players)

    # Overall auction slot info ablation
    print("\nRunning auction slot info ablation...")
    print(
        f"  Zeroing indices [{layout.auction_slot_info_offset}:"
        f"{layout.auction_slot_info_offset + layout.auction_slot_info_size}] "
        f"({layout.auction_slot_info_size} floats, "
        f"{layout.auction_slot_info_size / layout.visible_size:.1%} of input)"
    )

    results, orig_cache = run_ablation(
        model, device, dataset, batch_size=args.batch_size,
    )
    _print_results(
        results, "AUCTION SLOT INFO ABLATION (ALL)",
        layout.auction_slot_info_size, layout.visible_size,
    )

    # Sub-feature breakdown
    if args.breakdown:
        print("\nRunning per-sub-feature ablation...")
        breakdown = run_auction_slot_breakdown(
            model, device, dataset,
            batch_size=args.batch_size, orig_cache=orig_cache,
        )
        for name, sub_results in breakdown.items():
            n = sub_results["num_features"]
            assert isinstance(n, int)
            _print_results(
                sub_results, f"ABLATION: {name.upper()}",
                n, layout.visible_size,
            )
        _print_breakdown_summary(breakdown, layout.visible_size)


if __name__ == "__main__":
    main()
