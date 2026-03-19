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

import numpy as np

from interp.full_ablation import _build_feature_groups
from interp.utils import InterpDataset, collect_states, load_model


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
            f"{r['zero_frac']:>5.0%} {outside_str:>6}"
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
    """Print groups with high zero fractions (potential underutilization)."""
    sparse = [r for r in rows if r["zero_frac"] > 0.90 and r["n_nonzero"] > 0]  # type: ignore[arg-type]
    sparse.sort(key=lambda r: -r["zero_frac"])  # type: ignore[arg-type]

    if not sparse:
        return

    print(f"\n  SPARSE FEATURES (>90% zero):")
    print(f"  {'Feature':<28} {'%Zero':>6} {'NZ Mean':>8} {'NZ |Max|':>8}")
    print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*8}")
    for r in sparse:
        print(
            f"  {r['name']:<28} {r['zero_frac']:>5.0%} "
            f"{r['nz_mean']:>8.4f} {r['nz_absmax']:>8.4f}"
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

    _print_overview(rows, dataset.num_states)
    _print_table(rows)
    _print_out_of_range(rows)
    _print_sparsity(rows)

    if args.feature:
        match = [(n, idx) for n, idx in groups if n == args.feature]
        if not match:
            available = [n for n, _ in groups]
            print(f"\nFeature '{args.feature}' not found. Available: {', '.join(available)}")
        else:
            name, indices = match[0]
            _print_feature_detail(name, dataset.states, indices, dataset.phases)

    print()


if __name__ == "__main__":
    main()
