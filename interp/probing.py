"""Probing classifiers: where does game knowledge crystallize in the network?

Trains linear probes on intermediate activations at each layer to predict
game-relevant quantities. Compares probe accuracy across layers to reveal
which blocks contribute to which types of understanding.

Probe categories:
- **sanity** (directly in input): phase, game_progress
- **game** (require cross-entity reasoning): winning_player, lead_margin, etc.
- **policy** (model behavior): action_type, invest_action, model_top_action
- **value** (model behavior): model_value_p0, model_entropy
- **nonlinear** (MLP vs linear at trunk): tests if policy info is nonlinearly encoded

Usage:
    .venv/bin/python -m interp.probing
    .venv/bin/python -m interp.probing --load-data interp/data/states.npz
    .venv/bin/python -m interp.probing --probes policy,value  # subset
    .venv/bin/python -m interp.probing --probes all            # everything (default)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from core.actions import decode_action_py
from core.state import get_layout
from interp.utils import batch_masked_softmax, forward_batched
from interp.utils import InterpDataset, collect_states, load_model

CASH_DIVISOR = 100.0
SHARE_DIVISOR = 7.0

# Action type names for readable output
_ACTION_TYPE_NAMES = {
    0: "pass", 1: "auction", 2: "buy", 3: "sell",
    4: "leave_bid", 5: "raise_bid", 6: "acq_price", 7: "acq_fi_high",
    8: "acq_fi_face", 9: "close", 10: "dividend", 11: "issue", 12: "ipo",
}

# Probe categories for --probes filtering
_PROBE_CATEGORIES: dict[str, list[str]] = {
    "sanity": ["phase", "game_progress"],
    "game": [
        "winning_player", "active_leading", "lead_margin", "nw_rank",
        "num_active_corps", "total_shares", "corps_invested", "companies_owned",
    ],
    "policy": ["action_type", "invest_action", "model_top_action"],
    "value": ["model_value_p0", "model_entropy"],
}


# ---------------------------------------------------------------------------
# Probe target extraction
# ---------------------------------------------------------------------------


def _extract_game_targets(
    states: np.ndarray, num_players: int,
) -> dict[str, tuple[np.ndarray, str]]:
    """Extract probe targets from rotated state arrays."""
    layout = get_layout(num_players)
    n = states.shape[0]
    targets: dict[str, tuple[np.ndarray, str]] = {}

    # Player net worths (denormalized)
    net_worths = np.zeros((n, num_players), dtype=np.float32)
    for p in range(num_players):
        off = layout.players_offset + p * layout.player_stride + 1
        net_worths[:, p] = states[:, off] * CASH_DIVISOR

    # Phase (sanity)
    phase_oh = states[:, layout.phase_offset : layout.phase_offset + layout.phase_size]
    targets["phase"] = (np.argmax(phase_oh, axis=1), "classification")

    # Game progress (sanity)
    targets["game_progress"] = (states[:, layout.turn_offset].copy(), "regression")

    # Winning player
    targets["winning_player"] = (np.argmax(net_worths, axis=1), "classification")

    # Active player leading?
    active_ahead = (
        net_worths[:, 0] >= np.max(net_worths[:, 1:], axis=1)
    ).astype(np.int32)
    targets["active_leading"] = (active_ahead, "classification")

    # Lead margin
    max_opp = np.max(net_worths[:, 1:], axis=1)
    targets["lead_margin"] = ((net_worths[:, 0] - max_opp) / CASH_DIVISOR, "regression")

    # Net worth rank (0=first)
    rank = np.sum(
        net_worths[:, 1:] > net_worths[:, 0:1], axis=1,
    ).astype(np.float32)
    targets["nw_rank"] = (rank, "regression")

    # Number of active corps
    num_active = np.zeros(n, dtype=np.float32)
    for c in range(8):
        off = layout.corps_offset + c * layout.corp_stride
        num_active += (states[:, off] > 0.5).astype(np.float32)
    targets["num_active_corps"] = (num_active, "regression")

    # Active player total shares
    shares_off = layout.players_offset + 39 + num_players
    shares = states[:, shares_off : shares_off + 8] * SHARE_DIVISOR
    targets["total_shares"] = (np.sum(shares, axis=1), "regression")

    # Corps invested in
    targets["corps_invested"] = (
        np.sum(shares > 0.5, axis=1).astype(np.float32), "regression",
    )

    # Companies owned
    co_off = layout.players_offset + 3 + num_players
    companies = states[:, co_off : co_off + 36]
    targets["companies_owned"] = (
        np.sum(companies > 0.5, axis=1).astype(np.float32), "regression",
    )

    return targets


def _extract_model_targets(
    model: torch.nn.Module,
    device: torch.device,
    states: np.ndarray,
    masks: np.ndarray,
    phases: np.ndarray,
    num_players: int,
    batch_size: int = 256,
) -> dict[str, tuple[np.ndarray, str]]:
    """Extract model outputs and action-type probes."""
    logits, values = forward_batched(model, device, states, batch_size)

    targets: dict[str, tuple[np.ndarray, str]] = {
        "model_value_p0": (values[:, 0].copy(), "regression"),
    }

    # Top-1 action
    masked = logits.copy()
    masked[masks <= 0] = -1e9
    top_actions = np.argmax(masked, axis=1).astype(np.int32)
    targets["model_top_action"] = (top_actions, "classification")

    # Policy entropy
    probs = batch_masked_softmax(logits, masks)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0)), axis=-1)
    targets["model_entropy"] = (entropy, "regression")

    # Action type (broad category: pass/auction/buy/sell/bid/etc.)
    action_types = np.array([
        decode_action_py(int(a), num_players)[1] for a in top_actions
    ], dtype=np.int32)
    targets["action_type"] = (action_types, "classification")

    # INVEST-phase action type only (pass=0, auction=1, buy=2, sell=3)
    invest_mask = phases == 0
    if np.sum(invest_mask) >= 50:
        invest_types = action_types[invest_mask]
        # Store with index mask for subsetting activations later
        targets["invest_action"] = (invest_types, "classification")
        targets["_invest_mask"] = (invest_mask, "_index_mask")

    return targets


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def collect_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Collect activations at each layer via hooks."""
    model.eval()

    input_name = "input_preprocess" if hasattr(model, "input_preprocess") else "input_proj"
    input_module = getattr(model, input_name)

    layer_names = (
        [input_name]
        + [f"block_{i}" for i in range(len(model.blocks))]
        + ["trunk_norm"]
    )
    activations: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}
    handles = []

    def make_hook(name: str):  # noqa: ANN202
        def hook(
            _mod: torch.nn.Module,  # noqa: ARG001
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

    return {name: torch.cat(acts, dim=0).numpy() for name, acts in activations.items()}


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result from one probe at one layer."""

    probe_name: str
    layer_name: str
    task_type: str
    metric: float
    metric_name: str


def _fit_one_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    seed: int,
    nonlinear: bool = False,
) -> tuple[float, str]:
    """Fit a single probe and return (metric, metric_name)."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    if task_type == "classification":
        if nonlinear:
            clf = MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=500,
                random_state=seed, early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(x_train_s, y_train)
        return float(accuracy_score(y_test, clf.predict(x_test_s))), "acc"
    else:
        if nonlinear:
            reg = MLPRegressor(
                hidden_layer_sizes=(128,), max_iter=500,
                random_state=seed, early_stopping=True,
                validation_fraction=0.15,
            )
        else:
            reg = Ridge(alpha=1.0)
        reg.fit(x_train_s, y_train)
        return float(r2_score(y_test, reg.predict(x_test_s))), "R²"


def train_probes(
    activations: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, str]],
    enabled_probes: set[str] | None = None,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> list[ProbeResult]:
    """Train linear probes for all targets at all layers."""
    rng = np.random.default_rng(seed)
    n = next(iter(activations.values())).shape[0]

    indices = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Handle invest_action subsetting: target is only invest-phase states,
    # but activations are full-size. We need to subset activations too.
    invest_mask = None
    if "_invest_mask" in targets:
        invest_mask = targets["_invest_mask"][0].astype(bool)

    results: list[ProbeResult] = []
    layer_names = list(activations.keys())

    for probe_name, (target, task_type) in targets.items():
        if task_type.startswith("_"):
            continue
        if enabled_probes is not None and probe_name not in enabled_probes:
            continue

        # For invest_action, work within the invest-phase subset
        if probe_name == "invest_action" and invest_mask is not None:
            invest_idx = np.where(invest_mask)[0]
            n_invest = len(invest_idx)
            sub_indices = rng.permutation(n_invest)
            sub_split = int(n_invest * (1 - test_fraction))
            probe_train_local = sub_indices[:sub_split]
            probe_test_local = sub_indices[sub_split:]
            # invest_idx maps local → global for activation indexing
            probe_train_global = invest_idx[probe_train_local]
            probe_test_global = invest_idx[probe_test_local]
        else:
            probe_train_local = train_idx
            probe_test_local = test_idx
            probe_train_global = train_idx
            probe_test_global = test_idx

        y_train = target[probe_train_local]
        y_test = target[probe_test_local]

        if task_type == "classification" and len(np.unique(y_train)) < 2:
            continue
        if task_type == "regression" and np.std(y_train) < 1e-8:
            continue

        for layer_name in layer_names:
            x_train = activations[layer_name][probe_train_global]
            x_test = activations[layer_name][probe_test_global]
            metric, metric_name = _fit_one_probe(
                x_train, y_train, x_test, y_test, task_type, seed,
            )
            results.append(ProbeResult(
                probe_name=probe_name,
                layer_name=layer_name,
                task_type=task_type,
                metric=metric,
                metric_name=metric_name,
            ))

    return results


def train_nonlinear_comparison(
    activations: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, str]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> list[tuple[str, str, float, float]]:
    """Compare linear vs MLP probes at trunk_norm for key targets.

    Returns list of (probe_name, metric_name, linear_metric, mlp_metric).
    """
    rng = np.random.default_rng(seed)
    trunk_key = list(activations.keys())[-1]  # trunk_norm
    acts = activations[trunk_key]
    n = acts.shape[0]

    indices = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    train_idx = indices[:split]
    test_idx = indices[split:]

    # Key targets for comparison
    compare_probes = [
        "model_value_p0", "model_top_action", "model_entropy",
        "action_type", "winning_player", "lead_margin",
    ]

    invest_mask = None
    if "_invest_mask" in targets:
        invest_mask = targets["_invest_mask"][0].astype(bool)

    # Include invest_action if available
    if "invest_action" in targets:
        compare_probes.append("invest_action")

    results: list[tuple[str, str, float, float]] = []

    for probe_name in compare_probes:
        if probe_name not in targets:
            continue
        target, task_type = targets[probe_name]
        if task_type.startswith("_"):
            continue

        if probe_name == "invest_action" and invest_mask is not None:
            invest_idx = np.where(invest_mask)[0]
            n_invest = len(invest_idx)
            sub_indices = rng.permutation(n_invest)
            sub_split = int(n_invest * (1 - test_fraction))
            local_train = sub_indices[:sub_split]
            local_test = sub_indices[sub_split:]
            global_train = invest_idx[local_train]
            global_test = invest_idx[local_test]
        else:
            local_train = train_idx
            local_test = test_idx
            global_train = train_idx
            global_test = test_idx

        y_train = target[local_train]
        y_test = target[local_test]

        if task_type == "classification" and len(np.unique(y_train)) < 2:
            continue
        if task_type == "regression" and np.std(y_train) < 1e-8:
            continue

        x_train = acts[global_train]
        x_test = acts[global_test]

        linear_metric, metric_name = _fit_one_probe(
            x_train, y_train, x_test, y_test, task_type, seed, nonlinear=False,
        )
        mlp_metric, _ = _fit_one_probe(
            x_train, y_train, x_test, y_test, task_type, seed, nonlinear=True,
        )
        results.append((probe_name, metric_name, linear_metric, mlp_metric))

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _short_layer_name(name: str) -> str:
    if name.startswith("block_"):
        return f"B{name[6:]}"
    if name == "trunk_norm":
        return "trunk"
    if name in ("input_preprocess", "input_proj"):
        return "input"
    return name[:6]


def format_results_table(
    results: list[ProbeResult],
    layer_names: list[str],
) -> str:
    """Format probe results as a readable table."""
    probes: dict[str, dict[str, float]] = {}
    probe_meta: dict[str, str] = {}

    for r in results:
        if r.probe_name not in probes:
            probes[r.probe_name] = {}
            probe_meta[r.probe_name] = r.metric_name
        probes[r.probe_name][r.layer_name] = r.metric

    col_w = 8
    probe_w = 22

    header = f"{'Probe':<{probe_w}} {'Type':>4}"
    for ln in layer_names:
        header += f" {_short_layer_name(ln):>{col_w}}"
    header += f" {'Δ(deep)':>{col_w}}"

    lines: list[str] = [header, "-" * len(header)]

    probe_order = sorted(probes.keys(), key=lambda p: (
        0 if probe_meta[p] == "acc" else 1,
        -probes[p].get(layer_names[-1], 0),
    ))

    for pname in probe_order:
        metrics = probes[pname]
        mtype = probe_meta[pname]
        row = f"{pname:<{probe_w}} {mtype:>4}"
        vals = [metrics.get(ln, float("nan")) for ln in layer_names]
        for v in vals:
            row += f" {v:>{col_w}.4f}"
        delta = vals[-1] - vals[0]
        sign = "+" if delta >= 0 else ""
        row += f" {sign}{delta:>{col_w - 1}.4f}"
        lines.append(row)

    return "\n".join(lines)


def format_nonlinear_table(
    comparisons: list[tuple[str, str, float, float]],
) -> str:
    """Format linear vs MLP comparison."""
    lines: list[str] = []
    header = f"{'Probe':<22} {'Type':>4} {'Linear':>8} {'MLP':>8} {'Δ':>8} {'Gain':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for probe_name, metric_name, linear, mlp in comparisons:
        delta = mlp - linear
        # Gain as percentage of remaining headroom
        headroom = 1.0 - linear if linear < 1.0 else 1.0
        gain_pct = (delta / headroom * 100) if headroom > 0.01 else 0.0
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"{probe_name:<22} {metric_name:>4} {linear:>8.4f} {mlp:>8.4f}"
            f" {sign}{delta:>7.4f} {gain_pct:>7.1f}%"
        )

    return "\n".join(lines)


def format_markdown(
    results: list[ProbeResult],
    layer_names: list[str],
    comparisons: list[tuple[str, str, float, float]] | None,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a full markdown report."""
    probes: dict[str, dict[str, float]] = {}
    probe_meta: dict[str, str] = {}
    for r in results:
        if r.probe_name not in probes:
            probes[r.probe_name] = {}
            probe_meta[r.probe_name] = r.metric_name
        probes[r.probe_name][r.layer_name] = r.metric

    short_names = [_short_layer_name(ln) for ln in layer_names]

    lines = [
        f"# Probing Classifier Results (epoch {epoch})\n",
        f"{num_states} states from {num_games} games, train/test 80/20, linear probes.\n",
    ]

    # Main table
    cols = ["Probe", "Type"] + short_names + ["Δ(deep)"]
    lines.append("| " + " | ".join(cols) + " |")
    aligns = [":---", ":---:"] + ["---:"] * (len(short_names) + 1)
    lines.append("| " + " | ".join(aligns) + " |")

    probe_order = sorted(probes.keys(), key=lambda p: (
        0 if probe_meta[p] == "acc" else 1,
        -probes[p].get(layer_names[-1], 0),
    ))

    for pname in probe_order:
        metrics = probes[pname]
        mtype = probe_meta[pname]
        vals = [metrics.get(ln, float("nan")) for ln in layer_names]
        delta = vals[-1] - vals[0]
        sign = "+" if delta >= 0 else ""
        cells = [pname, mtype] + [f"{v:.4f}" for v in vals] + [f"{sign}{delta:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")

    # Nonlinear comparison table
    if comparisons:
        lines.append("")
        lines.append("## Linear vs MLP at trunk_norm\n")
        lines.append("| Probe | Type | Linear | MLP | Δ | Gain |")
        lines.append("| :--- | :---: | ---: | ---: | ---: | ---: |")
        for probe_name, metric_name, linear, mlp in comparisons:
            delta = mlp - linear
            headroom = 1.0 - linear if linear < 1.0 else 1.0
            gain_pct = (delta / headroom * 100) if headroom > 0.01 else 0.0
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {probe_name} | {metric_name} | {linear:.4f} | {mlp:.4f}"
                f" | {sign}{delta:.4f} | {gain_pct:.1f}% |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_probes(spec: str) -> set[str] | None:
    """Parse --probes argument into a set of probe names, or None for all."""
    if spec == "all":
        return None
    names: set[str] = set()
    for token in spec.split(","):
        token = token.strip()
        if token in _PROBE_CATEGORIES:
            names.update(_PROBE_CATEGORIES[token])
        else:
            names.add(token)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probing classifiers: where does game knowledge crystallize?"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--load-data", type=str, default=None)
    parser.add_argument("--save-data", type=str, default=None)
    parser.add_argument(
        "--probes", type=str, default="all",
        help="Comma-separated probe names or categories: sanity,game,policy,value,all",
    )
    parser.add_argument(
        "--nonlinear", action="store_true",
        help="Compare linear vs MLP probes at trunk_norm",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Markdown output path (default: interp/data/probing_epoch<N>.md)",
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

    enabled_probes = _resolve_probes(args.probes)

    # --- Extract targets ---
    print("\nExtracting probe targets...")
    targets = _extract_game_targets(dataset.states, config.num_players)

    print("  Computing model outputs for self-prediction probes...")
    model_targets = _extract_model_targets(
        model, device, dataset.states, dataset.legal_masks,
        dataset.phases, config.num_players, args.batch_size,
    )
    targets.update(model_targets)

    for name, (target, task_type) in targets.items():
        if task_type.startswith("_"):
            continue
        if enabled_probes is not None and name not in enabled_probes:
            continue
        if task_type == "classification":
            classes, counts = np.unique(target, return_counts=True)
            majority = float(counts.max()) / len(target)
            print(f"  {name}: {task_type}, {len(classes)} classes, majority={majority:.1%}")
        else:
            print(f"  {name}: {task_type}, mean={np.mean(target):.3f}, std={np.std(target):.3f}")

    # --- Collect activations ---
    print("\nCollecting activations...")
    t0 = time.perf_counter()
    activations = collect_activations(model, device, dataset.states, args.batch_size)
    layer_names = list(activations.keys())
    print(f"  {len(layer_names)} layers in {time.perf_counter() - t0:.1f}s")

    # --- Train linear probes ---
    active_count = sum(
        1 for name, (_, tt) in targets.items()
        if not tt.startswith("_") and (enabled_probes is None or name in enabled_probes)
    )
    print(f"\nTraining {active_count * len(layer_names)} linear probes...")
    t0 = time.perf_counter()
    results = train_probes(activations, targets, enabled_probes, seed=args.seed)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Nonlinear comparison ---
    comparisons: list[tuple[str, str, float, float]] | None = None
    if args.nonlinear:
        print("\nTraining nonlinear (MLP) comparison probes at trunk_norm...")
        t0 = time.perf_counter()
        comparisons = train_nonlinear_comparison(activations, targets, seed=args.seed)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Print results ---
    print(f"\n{'=' * 100}")
    print(f"  PROBING CLASSIFIER RESULTS (epoch {epoch})")
    print(f"  {dataset.num_states} states from {dataset.num_games} games, "
          f"train/test 80/20, linear probes")
    print(f"{'=' * 100}\n")
    print(format_results_table(results, layer_names))

    if comparisons:
        print(f"\n{'=' * 70}")
        print("  LINEAR vs MLP (128-unit hidden layer) AT TRUNK_NORM")
        print(f"{'=' * 70}\n")
        print(format_nonlinear_table(comparisons))

    print()

    # --- Write markdown ---
    md_path = Path(args.output) if args.output else Path("interp/data") / f"probing_epoch{epoch}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_markdown(
        results, layer_names, comparisons, epoch, dataset.num_states, dataset.num_games,
    ))
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
