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
import time
from typing import Any

import numpy as np
import torch

from interp.utils import InterpDataset, collect_states, load_model


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


# ---------------------------------------------------------------------------
# 2. Layer conductance (Captum)
# ---------------------------------------------------------------------------

def analyze_layer_conductance(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    max_samples: int = 200,
    n_steps: int = 20,
    internal_batch_size: int = 64,
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
            print(f"    {head_name} block {i}/9 ({elapsed:.1f}s)")

        total = sum(conductances)
        for i, c in enumerate(conductances):
            results[head_name].append({
                "block": float(i),
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
) -> list[dict[str, object]]:
    """SVD-based effective rank after input_proj, each block, and trunk_norm."""
    model.eval()

    # Collection points — detect v1 (input_proj) vs v2 (input_preprocess)
    input_name = "input_preprocess" if hasattr(model, "input_preprocess") else "input_proj"
    input_module = getattr(model, input_name)
    layer_names = [input_name] + [f"block_{i}" for i in range(len(model.blocks))] + ["trunk_norm"]
    activations: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}

    handles = []

    def make_hook(name: str):
        def hook(
            _module: torch.nn.Module,  # noqa: ARG001
            _inp: tuple[torch.Tensor, ...],  # noqa: ARG001
            out: torch.Tensor,
        ) -> None:
            activations[name].append(out.detach().cpu())
        return hook

    handles.append(input_module.register_forward_hook(make_hook(input_name)))
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(f"block_{i}")))
    handles.append(model.trunk_norm.register_forward_hook(make_hook("trunk_norm")))

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

        results.append({
            "layer": name,
            "eff_rank": eff_rank,
            "eff_rank_1pct": eff_rank_1pct,
            "top50_energy": top50,
            "top100_energy": top100,
            "top200_energy": top200,
        })

    return results


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

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


def _print_ranks(rows: list[dict[str, object]], hidden_dim: int) -> None:
    print(f"\n  hidden_dim = {hidden_dim}")
    print(
        f"\n  {'Layer':<12}  {'Eff.Rank':>9}  {'Rank(1%)':>9}  "
        f"{'Top-50':>8}  {'Top-100':>8}  {'Top-200':>8}"
    )
    print(
        f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}"
    )
    for r in rows:
        name = str(r["layer"])
        er = r["eff_rank"]
        assert isinstance(er, float)
        er1 = r["eff_rank_1pct"]
        assert isinstance(er1, int)
        t50 = r["top50_energy"]
        t100 = r["top100_energy"]
        t200 = r["top200_energy"]
        assert isinstance(t50, float) and isinstance(t100, float) and isinstance(t200, float)
        print(
            f"  {name:<12}  {er:>9.1f}  {er1:>9}  "
            f"{t50:>7.1f}%  {t100:>7.1f}%  {t200:>7.1f}%"
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
        "--skip-conductance", action="store_true",
        help="Skip Captum layer conductance (the slowest analysis)",
    )
    parser.add_argument("--conductance-samples", type=int, default=200)
    parser.add_argument("--conductance-steps", type=int, default=20)
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

    # --- 1. Block contribution ---
    print("\n" + "=" * 65)
    print("  1. RESIDUAL BLOCK CONTRIBUTION")
    print("=" * 65)
    print("  ||output - input|| / ||input|| per block (higher = more active)")
    t0 = time.perf_counter()
    contributions = analyze_block_contributions(
        model, device, dataset.states, batch_size=args.batch_size,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    _print_contributions(contributions)

    # --- 2. Layer conductance ---
    conductance = None
    if not args.skip_conductance:
        print("\n" + "=" * 65)
        print("  2. LAYER CONDUCTANCE (Captum)")
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

    # --- 3. Effective rank ---
    print("\n" + "=" * 65)
    print("  3. EFFECTIVE RANK (SVD)")
    print("=" * 65)
    print("  Activation dimensionality at each layer")
    t0 = time.perf_counter()
    ranks = analyze_effective_rank(
        model, device, dataset.states, batch_size=args.batch_size,
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
    if conductance is not None:
        pol_pcts = [r["pct"] for r in conductance["policy"]]
        val_pcts = [r["pct"] for r in conductance["value"]]
        pol_top3 = sorted(range(10), key=lambda i: -pol_pcts[i])[:3]
        val_top3 = sorted(range(10), key=lambda i: -val_pcts[i])[:3]
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

    print()


if __name__ == "__main__":
    main()
