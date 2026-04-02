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
import json
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
from core.data import GameConstants, GamePhases, PY_CASH_DIVISOR, PY_NET_WORTH_DIVISOR, PY_SHARE_DIVISOR
from core.state import get_corp_fields, get_layout, get_player_fields
from interp.html import html_page, open_file
from interp.utils import DECISION_PHASE_ORDER, PHASE_NAMES, batch_masked_softmax, forward_batched
from interp.utils import InterpDataset, collect_states, load_model

# Action type names for readable output
_ACTION_TYPE_NAMES = {
    0: "pass", 1: "auction", 2: "buy", 3: "sell",
    4: "leave_bid", 5: "raise_bid", 6: "acq_price", 7: "acq_fi_buy",
    8: "close", 9: "dividend", 10: "issue", 11: "ipo", 12: "par",
}

# Probe categories for --probes filtering
_PROBE_CATEGORIES: dict[str, list[str]] = {
    "sanity": ["phase", "game_progress"],
    "game": [
        "winning_player", "active_leading", "lead_margin", "nw_rank",
        "num_active_corps", "total_shares", "corps_invested", "companies_owned",
    ],
    "policy": [
        "action_type", "invest_action", "bid_action", "acq_action",
        "ipo_action", "issue_action", "dividend_level", "par_price_level",
        "close_action", "model_top_action", "policy_margin", "policy_concentration",
    ],
    "value": ["model_value_p0", "model_entropy", "value_spread"],
}


# ---------------------------------------------------------------------------
# Probe target extraction
# ---------------------------------------------------------------------------


def _extract_game_targets(
    states: np.ndarray, num_players: int,
) -> dict[str, tuple[np.ndarray, str]]:
    """Extract probe targets from rotated state arrays."""
    layout = get_layout(num_players)
    pf = get_player_fields(num_players)
    cf = get_corp_fields()
    NK = GameConstants.NUM_CORPS
    NC = GameConstants.NUM_COMPANIES
    n = states.shape[0]
    targets: dict[str, tuple[np.ndarray, str]] = {}

    # Player net worths (denormalized)
    net_worths = np.zeros((n, num_players), dtype=np.float32)
    for p in range(num_players):
        off = layout.players_offset + p * layout.player_stride + pf.net_worth
        net_worths[:, p] = states[:, off] * PY_NET_WORTH_DIVISOR

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
    targets["lead_margin"] = ((net_worths[:, 0] - max_opp) / PY_CASH_DIVISOR, "regression")

    # Net worth rank (0=first)
    rank = np.sum(
        net_worths[:, 1:] > net_worths[:, 0:1], axis=1,
    ).astype(np.float32)
    targets["nw_rank"] = (rank, "regression")

    # Number of active corps
    num_active = np.zeros(n, dtype=np.float32)
    for c in range(NK):
        off = layout.corps_offset + c * layout.corp_stride + cf.active
        num_active += (states[:, off] > 0.5).astype(np.float32)
    targets["num_active_corps"] = (num_active, "regression")

    # Active player total shares
    shares_off = layout.players_offset + pf.owned_shares
    shares = states[:, shares_off : shares_off + NK] * PY_SHARE_DIVISOR
    targets["total_shares"] = (np.sum(shares, axis=1), "regression")

    # Corps invested in
    targets["corps_invested"] = (
        np.sum(shares > 0.5, axis=1).astype(np.float32), "regression",
    )

    # Companies owned
    co_off = layout.players_offset + pf.owned_companies
    companies = states[:, co_off : co_off + NC]
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

    # Phase-specific action probes
    # Convention: target stored as (subset_array, "classification"/"regression"),
    # mask stored as (_<probe_name>_mask, "_index_mask") for activation subsetting.

    # INVEST: pass=0, auction=1, buy=2, sell=3
    invest_mask = phases == GamePhases.PHASE_INVEST
    if np.sum(invest_mask) >= 50:
        invest_types = action_types[invest_mask]
        targets["invest_action"] = (invest_types, "classification")
        targets["_invest_action_mask"] = (invest_mask, "_index_mask")

    # BID: leave=0, raise=1
    bid_mask = phases == GamePhases.PHASE_BID_IN_AUCTION
    if np.sum(bid_mask) >= 50:
        bid_types = action_types[bid_mask]
        # leave_bid=4 → 0, raise_bid=5 → 1
        bid_actions = (bid_types == 5).astype(np.int32)
        targets["bid_action"] = (bid_actions, "classification")
        targets["_bid_action_mask"] = (bid_mask, "_index_mask")

    # ACQ: acq_price=0, acq_fi_buy=1, pass=2
    acq_mask = phases == GamePhases.PHASE_ACQUISITION
    if np.sum(acq_mask) >= 50:
        acq_types = action_types[acq_mask]
        acq_actions = np.where(acq_types == 6, 0,
                      np.where(acq_types == 7, 1, 2)).astype(np.int32)
        targets["acq_action"] = (acq_actions, "classification")
        targets["_acq_action_mask"] = (acq_mask, "_index_mask")

    # IPO: pass=0, ipo=1
    ipo_mask = phases == GamePhases.PHASE_IPO
    if np.sum(ipo_mask) >= 50:
        ipo_types = action_types[ipo_mask]
        ipo_actions = (ipo_types == 11).astype(np.int32)
        targets["ipo_action"] = (ipo_actions, "classification")
        targets["_ipo_action_mask"] = (ipo_mask, "_index_mask")

    # ISSUE: pass=0, issue=1
    issue_mask = phases == GamePhases.PHASE_ISSUE_SHARES
    if np.sum(issue_mask) >= 50:
        issue_types = action_types[issue_mask]
        issue_actions = (issue_types == 10).astype(np.int32)
        targets["issue_action"] = (issue_actions, "classification")
        targets["_issue_action_mask"] = (issue_mask, "_index_mask")

    # DIVIDENDS: chosen dividend level (regression, normalized 0-1)
    div_mask = phases == GamePhases.PHASE_DIVIDENDS
    if np.sum(div_mask) >= 50:
        div_action_indices = top_actions[div_mask]
        div_amounts = np.array([
            decode_action_py(int(a), num_players)[4]
            for a in div_action_indices
        ], dtype=np.float32) / (GameConstants.MAX_DIVIDEND - 1)
        targets["dividend_level"] = (div_amounts, "regression")
        targets["_dividend_level_mask"] = (div_mask, "_index_mask")

    # PAR: chosen par price index (regression, normalized 0-1)
    par_mask = phases == GamePhases.PHASE_PAR
    if np.sum(par_mask) >= 50:
        par_action_indices = top_actions[par_mask]
        par_levels = np.array([
            decode_action_py(int(a), num_players)[2]  # par index is in 'slot' field
            for a in par_action_indices
        ], dtype=np.float32) / (GameConstants.NUM_PAR_PRICES - 1)
        targets["par_price_level"] = (par_levels, "regression")
        targets["_par_price_level_mask"] = (par_mask, "_index_mask")

    # CLOSING: close=1, pass=0
    close_mask = phases == GamePhases.PHASE_CLOSING
    if np.sum(close_mask) >= 50:
        close_types = action_types[close_mask]
        close_actions = (close_types == 8).astype(np.int32)
        targets["close_action"] = (close_actions, "classification")
        targets["_close_action_mask"] = (close_mask, "_index_mask")

    # Value spread: std of the 3 per-player value outputs
    value_spread = np.std(values, axis=1)
    targets["value_spread"] = (value_spread.astype(np.float32), "regression")

    # Policy decisiveness probes (all states, no phase mask)
    # policy_margin: top-1 minus top-2 probability
    sorted_probs = np.sort(probs, axis=-1)
    policy_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    targets["policy_margin"] = (policy_margin.astype(np.float32), "regression")

    # policy_concentration: KL(policy || uniform over legal actions)
    # KL = log(num_legal) - entropy
    num_legal = np.sum(masks > 0, axis=-1).astype(np.float32)
    kl = np.log(np.clip(num_legal, 1, None)) - entropy
    targets["policy_concentration"] = (kl.astype(np.float32), "regression")

    return targets


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def collect_activations(
    model: Any,
    device: torch.device,
    states: np.ndarray,
    batch_size: int = 256,
    include_heads: bool = False,
) -> dict[str, np.ndarray]:
    """Collect activations at each layer via hooks.

    If include_heads is True, also hooks into each Linear layer within
    the per-phase policy heads and value_head Sequential modules.
    """
    model.eval()

    # Input preprocessing: per-Linear-layer breakdown
    layer_names: list[str] = []
    for i, layer in enumerate(model.input_preprocess):
        if isinstance(layer, torch.nn.Linear):
            layer_names.append(f"preprocess[{i}] {layer.in_features}\u2192{layer.out_features}")

    # Trunk blocks + final norm
    layer_names += [f"block_{i}" for i in range(len(model.blocks))]
    layer_names.append("trunk_norm")

    if include_heads:
        for i, layer in enumerate(model.value_head):
            if isinstance(layer, torch.nn.Linear):
                layer_names.append(f"value_{i}")

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

    for i, layer in enumerate(model.input_preprocess):
        if isinstance(layer, torch.nn.Linear):
            label = f"preprocess[{i}] {layer.in_features}\u2192{layer.out_features}"
            handles.append(layer.register_forward_hook(make_hook(label)))
    for i, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(make_hook(f"block_{i}")))
    handles.append(model.trunk_norm.register_forward_hook(make_hook("trunk_norm")))

    if include_heads:
        for i, layer in enumerate(model.value_head):
            if isinstance(layer, torch.nn.Linear):
                handles.append(layer.register_forward_hook(make_hook(f"value_{i}")))

    with torch.inference_mode():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            model(torch.from_numpy(states[i:j]).to(device))

    for h in handles:
        h.remove()

    return {name: torch.cat(acts, dim=0).numpy() for name, acts in activations.items()}


def collect_phase_activations(
    model: Any,
    device: torch.device,
    phase_states: np.ndarray,
    head_idx: int,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Collect activations from a specific phase head's Linear layers.

    phase_states must contain only states matching this phase head.
    Since all states route to the same head, the model's internal phase
    dispatch preserves their order.
    """
    model.eval()
    phase_name = DECISION_PHASE_ORDER[head_idx]
    head = model.phase_heads[head_idx]

    layer_names: list[str] = []
    activations: dict[str, list[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):  # noqa: ANN202
        def hook(
            _mod: torch.nn.Module,  # noqa: ARG001
            _inp: tuple[torch.Tensor, ...],  # noqa: ARG001
            out: torch.Tensor,
        ) -> None:
            activations[name].append(out.detach().cpu())
        return hook

    for i, layer in enumerate(head):
        if isinstance(layer, torch.nn.Linear):
            name = f"phase:{phase_name}[{i}]"
            layer_names.append(name)
            activations[name] = []
            handles.append(layer.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        for i in range(0, phase_states.shape[0], batch_size):
            j = min(i + batch_size, phase_states.shape[0])
            model(torch.from_numpy(phase_states[i:j]).to(device))

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


# ---------------------------------------------------------------------------
# GPU-accelerated probe training
# ---------------------------------------------------------------------------


def _standardize_gpu(
    X_train: torch.Tensor, X_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize features using train statistics (matches sklearn StandardScaler)."""
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0, correction=0)
    # Zero out near-constant features instead of dividing by tiny std
    # (wider preprocessing layers can have dead/constant activation dims)
    alive = std > 1e-6
    safe_std = std.clamp(min=1.0)  # avoid div-by-zero; masked out anyway
    X_tr = (X_train - mean) / safe_std * alive
    X_te = (X_test - mean) / safe_std * alive
    return X_tr, X_te


@torch.inference_mode()
def _fit_ridge_gpu(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    alpha: float = 1.0,
) -> float:
    """Closed-form Ridge regression on GPU. Returns R²."""
    X_tr, X_te = _standardize_gpu(X_train, X_test)
    # Center target — intercept = y_mean (since X is zero-mean after standardization)
    y_mean = y_train.mean()
    y_c = y_train - y_mean
    # Tikhonov: augment [X; √α·I] and [y; 0] then solve via lstsq
    # lstsq handles rank-deficient matrices (dead activation dims) gracefully
    d = X_tr.shape[1]
    X_aug = torch.cat([X_tr, (alpha ** 0.5) * torch.eye(d, device=X_tr.device)], dim=0)
    y_aug = torch.cat([y_c, torch.zeros(d, device=y_c.device)])
    w = torch.linalg.lstsq(X_aug, y_aug).solution
    y_pred = X_te @ w + y_mean
    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    return (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()


def _fit_logreg_gpu(
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    C: float = 1.0,
) -> float:
    """L2-regularized logistic regression via L-BFGS on GPU. Returns accuracy.

    Matches sklearn's objective: sum(CE) + 1/(2C) * ||W||².
    Since cross_entropy(reduction='mean') = sum(CE)/n, we scale the L2
    term by 1/n to get the equivalent: mean(CE) + 1/(2Cn) * ||W||².
    """
    d = X_train.shape[1]
    n = X_train.shape[0]
    device = X_train.device

    # Remap labels to dense 0..K-1 (sklearn does this internally)
    classes = y_train.unique(sorted=True)
    n_classes = classes.shape[0]
    if classes[-1] >= n_classes:
        remap = torch.zeros(int(classes[-1].item()) + 1, dtype=torch.long, device=device)
        for new, old in enumerate(classes):
            remap[old] = new
        y_train = remap[y_train]
        y_test = remap[y_test]

    X_tr, X_te = _standardize_gpu(X_train, X_test)

    W = torch.zeros(d, n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b], max_iter=100, history_size=10, line_search_fn="strong_wolfe",
    )
    reg = 1.0 / (2.0 * C * n)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = X_tr @ W + b
        loss = torch.nn.functional.cross_entropy(logits, y_train) + reg * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)

    with torch.inference_mode():
        logits = X_te @ W + b
        return float((logits.argmax(dim=1) == y_test).float().mean().item())


def train_probes_gpu(
    activations: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, str]],
    device: torch.device,
    enabled_probes: set[str] | None = None,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> list[ProbeResult]:
    """GPU-accelerated linear probes. Drop-in replacement for train_probes."""
    rng = np.random.default_rng(seed)
    n = next(iter(activations.values())).shape[0]

    indices = rng.permutation(n)
    split = int(n * (1 - test_fraction))
    train_idx, test_idx = indices[:split], indices[split:]

    # Collect phase masks
    phase_masks: dict[str, np.ndarray] = {}
    for key, (val, tt) in targets.items():
        if tt == "_index_mask":
            phase_masks[key[1:-5]] = val.astype(bool)

    # Pre-compute splits and GPU targets for each probe (avoids redundant
    # work in the per-layer loop and ensures consistent splits across layers).
    probe_info: list[tuple[str, str, str, torch.Tensor, torch.Tensor,
                           np.ndarray, np.ndarray]] = []

    for probe_name, (target, task_type) in targets.items():
        if task_type.startswith("_"):
            continue
        if enabled_probes is not None and probe_name not in enabled_probes:
            continue

        if probe_name in phase_masks:
            sub_idx = np.where(phase_masks[probe_name])[0]
            sub_perm = rng.permutation(len(sub_idx))
            sub_split = int(len(sub_idx) * (1 - test_fraction))
            local_train, local_test = sub_perm[:sub_split], sub_perm[sub_split:]
            global_train = sub_idx[local_train]
            global_test = sub_idx[local_test]
        else:
            local_train, local_test = train_idx, test_idx
            global_train, global_test = train_idx, test_idx

        y_train_np, y_test_np = target[local_train], target[local_test]

        if task_type == "classification" and len(np.unique(y_train_np)) < 2:
            continue
        if task_type == "regression" and np.std(y_train_np) < 1e-8:
            continue

        if task_type == "classification":
            y_train_gpu = torch.from_numpy(y_train_np.astype(np.int64)).to(device)
            y_test_gpu = torch.from_numpy(y_test_np.astype(np.int64)).to(device)
            metric_name = "acc"
        else:
            y_train_gpu = torch.from_numpy(y_train_np.astype(np.float32)).to(device)
            y_test_gpu = torch.from_numpy(y_test_np.astype(np.float32)).to(device)
            metric_name = "R²"

        probe_info.append((
            probe_name, task_type, metric_name,
            y_train_gpu, y_test_gpu, global_train, global_test,
        ))

    results: list[ProbeResult] = []
    layer_names = list(activations.keys())

    for layer_name in layer_names:
        acts_gpu = torch.from_numpy(activations[layer_name]).to(device)

        for (probe_name, task_type, metric_name,
             y_train_gpu, y_test_gpu, global_train, global_test) in probe_info:

            X_train = acts_gpu[global_train]
            X_test = acts_gpu[global_test]

            if task_type == "classification":
                metric = _fit_logreg_gpu(X_train, y_train_gpu, X_test, y_test_gpu)
            else:
                metric = _fit_ridge_gpu(X_train, y_train_gpu, X_test, y_test_gpu)

            results.append(ProbeResult(
                probe_name=probe_name,
                layer_name=layer_name,
                task_type=task_type,
                metric=metric,
                metric_name=metric_name,
            ))

        del acts_gpu

    return results


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

    # Collect phase masks for phase-specific probes.
    # Convention: _<probe_name>_mask → used to subset activations for that probe.
    phase_masks: dict[str, np.ndarray] = {}
    for key, (val, tt) in targets.items():
        if tt == "_index_mask":
            # _foo_mask → foo
            probe_name_for_mask = key[1:-5]  # strip leading _ and trailing _mask
            phase_masks[probe_name_for_mask] = val.astype(bool)

    results: list[ProbeResult] = []
    layer_names = list(activations.keys())

    for probe_name, (target, task_type) in targets.items():
        if task_type.startswith("_"):
            continue
        if enabled_probes is not None and probe_name not in enabled_probes:
            continue

        # Phase-specific probes: target is a subset, activations are full-size
        if probe_name in phase_masks:
            sub_idx = np.where(phase_masks[probe_name])[0]
            n_sub = len(sub_idx)
            sub_perm = rng.permutation(n_sub)
            sub_split = int(n_sub * (1 - test_fraction))
            probe_train_local = sub_perm[:sub_split]
            probe_test_local = sub_perm[sub_split:]
            probe_train_global = sub_idx[probe_train_local]
            probe_test_global = sub_idx[probe_test_local]
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
        "policy_margin", "policy_concentration",
    ]

    # Collect phase masks
    phase_masks: dict[str, np.ndarray] = {}
    for key, (val, tt) in targets.items():
        if tt == "_index_mask":
            probe_name_for_mask = key[1:-5]
            phase_masks[probe_name_for_mask] = val.astype(bool)

    # Include phase-specific action probes if available
    for name in ["invest_action", "bid_action", "acq_action",
                 "ipo_action", "issue_action", "dividend_level"]:
        if name in targets:
            compare_probes.append(name)

    results: list[tuple[str, str, float, float]] = []

    for probe_name in compare_probes:
        if probe_name not in targets:
            continue
        target, task_type = targets[probe_name]
        if task_type.startswith("_"):
            continue

        if probe_name in phase_masks:
            sub_idx = np.where(phase_masks[probe_name])[0]
            n_sub = len(sub_idx)
            sub_perm = rng.permutation(n_sub)
            sub_split = int(n_sub * (1 - test_fraction))
            local_train = sub_perm[:sub_split]
            local_test = sub_perm[sub_split:]
            global_train = sub_idx[local_train]
            global_test = sub_idx[local_test]
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
    if name.startswith("preprocess["):
        # "preprocess[0] 1109→768" → "P0 1109→768"
        bracket_end = name.index("]")
        idx = name[len("preprocess["):bracket_end]
        suffix = name[bracket_end + 1:]  # " 1109→768"
        return f"P{idx}{suffix}"
    if name.startswith("block_"):
        return f"B{name[6:]}"
    if name == "trunk_norm":
        return "trunk"
    if name.startswith("phase:"):
        # "phase:INVEST[0]" → "INVEST[0]"
        return name[6:]
    if name.startswith("value_"):
        return f"V{name[6:]}"
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
    phase_probe_data: list[dict[str, Any]],
    comparisons: list[tuple[str, str, float, float]] | None,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a full markdown report."""
    trunk_layers, _, value_layers = _split_layer_groups(layer_names)

    lines = [
        f"# Probing Classifier Results (epoch {epoch})\n",
        f"{num_states} states from {num_games} games, train/test 80/20, linear probes.\n",
    ]

    def _add_table(
        probe_results: list[ProbeResult], layers: list[str],
    ) -> None:
        probes: dict[str, dict[str, float]] = {}
        probe_meta: dict[str, str] = {}
        for r in probe_results:
            if r.layer_name not in layers:
                continue
            if r.probe_name not in probes:
                probes[r.probe_name] = {}
                probe_meta[r.probe_name] = r.metric_name
            probes[r.probe_name][r.layer_name] = r.metric

        short = [_short_layer_name(ln) for ln in layers]
        cols = ["Probe", "Type"] + short + ["\u0394(deep)"]
        lines.append("| " + " | ".join(cols) + " |")
        aligns = [":---", ":---:"] + ["---:"] * (len(short) + 1)
        lines.append("| " + " | ".join(aligns) + " |")

        probe_order = sorted(probes.keys(), key=lambda p: (
            0 if probe_meta[p] == "acc" else 1,
            -probes[p].get(layers[-1], 0),
        ))
        for pname in probe_order:
            metrics = probes[pname]
            mtype = probe_meta[pname]
            vals = [metrics.get(ln, float("nan")) for ln in layers]
            delta = vals[-1] - vals[0]
            sign = "+" if delta >= 0 else ""
            cells = [pname, mtype] + [f"{v:.4f}" for v in vals] + [f"{sign}{delta:.4f}"]
            lines.append("| " + " | ".join(cells) + " |")

    # Trunk table
    if trunk_layers:
        lines.append("## Trunk\n")
        _add_table(results, trunk_layers)

    # Per-phase tables
    for pdata in phase_probe_data:
        lines.append("")
        lines.append(f"## {pdata['name']} Head ({pdata['n_states']} states)\n")
        _add_table(pdata["results"], pdata["layer_names"])

    # Value head table
    if value_layers:
        lines.append("")
        lines.append("## Value Head\n")
        _add_table(results, value_layers)

    # Nonlinear comparison table
    if comparisons:
        lines.append("")
        lines.append("## Linear vs MLP at trunk_norm\n")
        lines.append("| Probe | Type | Linear | MLP | \u0394 | Gain |")
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
# HTML report
# ---------------------------------------------------------------------------


def _split_layer_groups(
    layer_names: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split layer names into trunk, policy-head, and value-head groups."""
    trunk = [n for n in layer_names if not n.startswith(("phase:", "value_"))]
    policy = [n for n in layer_names if n.startswith("phase:")]
    value = [n for n in layer_names if n.startswith("value_")]
    return trunk, policy, value


def _results_to_json(
    results: list[ProbeResult],
    layer_names: list[str],
) -> list[dict[str, Any]]:
    """Convert probe results to JSON-serializable rows for a given set of layers."""
    probes: dict[str, dict[str, float]] = {}
    probe_meta: dict[str, str] = {}
    for r in results:
        if r.layer_name not in layer_names:
            continue
        if r.probe_name not in probes:
            probes[r.probe_name] = {}
            probe_meta[r.probe_name] = r.metric_name
        probes[r.probe_name][r.layer_name] = r.metric

    # Sort: accuracy first, then by descending trunk/last-layer metric
    probe_order = sorted(probes.keys(), key=lambda p: (
        0 if probe_meta[p] == "acc" else 1,
        -probes[p].get(layer_names[-1], 0),
    ))

    rows = []
    for pname in probe_order:
        metrics = probes[pname]
        vals = [metrics.get(ln, None) for ln in layer_names]
        present = [v for v in vals if v is not None]
        delta = present[-1] - present[0] if len(present) >= 2 else 0.0
        rows.append({
            "probe": pname,
            "type": probe_meta[pname],
            "values": vals,
            "delta": delta,
        })
    return rows


PROBING_CSS = """\
th:first-child, td:first-child { width: 180px; }
.delta-pos { color: #4ecca3; }
.delta-neg { color: #e94560; }
.delta-flat { color: #888; }
.tag {
  display: inline-block; padding: 1px 5px; border-radius: 3px;
  font-size: 0.75rem; font-weight: 600;
}
.tag-acc { background: #1a2a3a; color: #4a9eff; }
.tag-r2 { background: #1a3a2a; color: #4ecca3; }"""


def _format_html_report(
    results: list[ProbeResult],
    layer_names: list[str],
    phase_probe_data: list[dict[str, Any]],
    comparisons: list[tuple[str, str, float, float]] | None,
    epoch: int,
    num_states: int,
    num_games: int,
) -> str:
    """Generate a self-contained HTML report for probing results."""
    trunk_layers, _policy_layers, value_layers = _split_layer_groups(layer_names)

    trunk_rows = _results_to_json(results, trunk_layers)
    trunk_headers = [_short_layer_name(n) for n in trunk_layers]

    value_rows = _results_to_json(results, value_layers) if value_layers else []
    value_headers = [_short_layer_name(n) for n in value_layers]

    comp_json = json.dumps([
        {"probe": p, "type": t, "linear": l, "mlp": m}
        for p, t, l, m in comparisons
    ]) if comparisons else "null"

    phase_json = json.dumps([
        {"name": pd["name"], "nStates": pd["n_states"],
         "headers": pd["headers"], "rows": pd["rows"]}
        for pd in phase_probe_data
    ])

    meta = f"{num_states:,} states from {num_games} games. Train/test 80/20, linear probes."

    body = (
        '<h2>1. Trunk Probes</h2>\n'
        '<p style="color:#888;font-size:0.85rem">Linear probe accuracy/R&sup2; at each trunk layer. &Delta;(deep) = trunk &minus; input.</p>\n'
        '<table id="tbl-trunk"></table>\n'
        '\n'
        '<div id="policy-section"></div>\n'
        '<div id="value-section"></div>\n'
        '<div id="comp-section"></div>'
    )

    data_js = (
        f"const trunkRows = {json.dumps(trunk_rows)};\n"
        f"const trunkHeaders = {json.dumps(trunk_headers)};\n"
        f"const phaseData = {phase_json};\n"
        f"const valueRows = {json.dumps(value_rows)};\n"
        f"const valueHeaders = {json.dumps(value_headers)};\n"
        f"const comparisons = {comp_json};"
    )

    report_js = (
        'function heatColor(val, isAcc) {\n'
        '  if (val === null) return "transparent";\n'
        '  const base = isAcc ? 0.5 : 0.0;\n'
        '  const t = Math.max(0, Math.min(1, (val - base) / (1.0 - base)));\n'
        '  const l = 18 + t * 22;\n'
        '  return "hsl(150, 50%," + l + "%)";\n'
        '}\n'
        '\n'
        'function deltaSpan(d) {\n'
        '  const cls = d > 0.005 ? "delta-pos" : d < -0.005 ? "delta-neg" : "delta-flat";\n'
        '  const sign = d >= 0 ? "+" : "";\n'
        '  return \'<span class="\' + cls + \'">\' + sign + d.toFixed(4) + \'</span>\';\n'
        '}\n'
        '\n'
        'function buildTable(tblId, rows, headers) {\n'
        '  const tbl = document.getElementById(tblId);\n'
        '  if (!tbl || rows.length === 0) return;\n'
        '  const isAcc = (t) => t === "acc";\n'
        '  let html = \'<tr><th>Probe</th><th>Type</th>\';\n'
        '  for (const h of headers) html += \'<th>\' + h + \'</th>\';\n'
        '  html += \'<th>&Delta;</th></tr>\';\n'
        '  for (const r of rows) {\n'
        '    const tag = isAcc(r.type) ? \'<span class="tag tag-acc">acc</span>\' : \'<span class="tag tag-r2">R&sup2;</span>\';\n'
        '    html += \'<tr><td>\' + r.probe + \'</td><td style="text-align:center">\' + tag + \'</td>\';\n'
        '    for (const v of r.values) {\n'
        '      if (v === null) {\n'
        '        html += \'<td>-</td>\';\n'
        '      } else {\n'
        '        html += \'<td style="background:\' + heatColor(v, isAcc(r.type)) + \'">\' + v.toFixed(4) + \'</td>\';\n'
        '      }\n'
        '    }\n'
        '    html += \'<td>\' + deltaSpan(r.delta) + \'</td></tr>\';\n'
        '  }\n'
        '  tbl.innerHTML = html;\n'
        '}\n'
        '\n'
        'buildTable("tbl-trunk", trunkRows, trunkHeaders);\n'
        '\n'
        'if (phaseData.length > 0) {\n'
        '  const sec = document.getElementById("policy-section");\n'
        '  let html = \'<h2>2. Policy Head Probes</h2>\' +\n'
        '    \'<p style="color:#888;font-size:0.85rem">Per-phase linear probes using only matching game states. &Delta; = last layer &minus; first layer.</p>\';\n'
        '  for (const pd of phaseData) {\n'
        '    const tblId = "tbl-phase-" + pd.name;\n'
        '    html += \'<h3 style="color:#aaa;font-size:0.95rem;margin-top:1rem">\' + pd.name +\n'
        '      \' <span style="color:#666;font-size:0.8rem">(\' + pd.nStates.toLocaleString() + \' states)</span></h3>\';\n'
        '    html += \'<table id="\' + tblId + \'"></table>\';\n'
        '  }\n'
        '  sec.innerHTML = html;\n'
        '  for (const pd of phaseData) {\n'
        '    buildTable("tbl-phase-" + pd.name, pd.rows, pd.headers);\n'
        '  }\n'
        '}\n'
        '\n'
        'if (valueRows.length > 0) {\n'
        '  const n = phaseData.length > 0 ? 3 : 2;\n'
        '  const sec = document.getElementById("value-section");\n'
        '  sec.innerHTML = \'<h2>\' + n + \'. Value Head Probes</h2>\' +\n'
        '    \'<p style="color:#888;font-size:0.85rem">Linear probes at each hidden layer within the value head.</p>\' +\n'
        '    \'<table id="tbl-value"></table>\';\n'
        '  buildTable("tbl-value", valueRows, valueHeaders);\n'
        '}\n'
        '\n'
        'if (comparisons) {\n'
        '  const n = (phaseData.length > 0 ? 3 : 2) + (valueRows.length > 0 ? 1 : 0);\n'
        '  const sec = document.getElementById("comp-section");\n'
        '  let html = \'<h2>\' + n + \'. Linear vs MLP at trunk_norm</h2>\' +\n'
        '    \'<p style="color:#888;font-size:0.85rem">Compares linear probe to 128-unit MLP. Large gains indicate nonlinearly encoded information.</p>\' +\n'
        '    \'<table><tr><th>Probe</th><th>Type</th><th>Linear</th><th>MLP</th><th>&Delta;</th><th>Gain</th></tr>\';\n'
        '  for (const c of comparisons) {\n'
        '    const d = c.mlp - c.linear;\n'
        '    const headroom = c.linear < 1.0 ? 1.0 - c.linear : 1.0;\n'
        '    const gain = headroom > 0.01 ? (d / headroom * 100).toFixed(1) + \'%\' : \'-\';\n'
        '    const tag = c.type === "acc" ? \'<span class="tag tag-acc">acc</span>\' : \'<span class="tag tag-r2">R&sup2;</span>\';\n'
        '    html += \'<tr><td>\' + c.probe + \'</td><td style="text-align:center">\' + tag + \'</td>\' +\n'
        '      \'<td>\' + c.linear.toFixed(4) + \'</td><td>\' + c.mlp.toFixed(4) + \'</td>\' +\n'
        '      \'<td>\' + deltaSpan(d) + \'</td><td>\' + gain + \'</td></tr>\';\n'
        '  }\n'
        '  html += \'</table>\';\n'
        '  sec.innerHTML = html;\n'
        '}'
    )

    return html_page(
        f"Probing Classifiers \u2014 Epoch {epoch}",
        meta=meta,
        body=body,
        script=data_js + "\n\n" + report_js,
        extra_css=PROBING_CSS,
        max_width=1400,
    )



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
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open the HTML report in a browser",
    )
    parser.add_argument(
        "--heads-only", action="store_true",
        help="Only probe policy/value head layers (skip trunk, much faster)",
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

    # --- Collect trunk + value head activations ---
    print("\nCollecting trunk + value head activations...")
    t0 = time.perf_counter()
    activations = collect_activations(
        model, device, dataset.states, args.batch_size, include_heads=True,
    )
    all_layer_names = list(activations.keys())
    print(f"  {len(all_layer_names)} layers in {time.perf_counter() - t0:.1f}s")

    # If --heads-only, only probe trunk_norm + value head (skip trunk blocks)
    if args.heads_only:
        probe_layers = {n: activations[n] for n in all_layer_names
                        if n.startswith("value_") or n == "trunk_norm"}
    else:
        probe_layers = activations

    probe_layer_names = list(probe_layers.keys())

    # --- Train trunk + value probes ---
    active_count = sum(
        1 for name, (_, tt) in targets.items()
        if not tt.startswith("_") and (enabled_probes is None or name in enabled_probes)
    )
    print(f"\nTraining {active_count * len(probe_layer_names)} trunk/value linear probes...")
    t0 = time.perf_counter()
    if device.type == "cuda":
        results = train_probes_gpu(
            probe_layers, targets, device, enabled_probes, seed=args.seed,
        )
    else:
        results = train_probes(probe_layers, targets, enabled_probes, seed=args.seed)
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Per-phase policy head probes ---
    sorted_phase_ids = sorted(PHASE_NAMES.keys())
    phase_probe_data: list[dict[str, Any]] = []

    print("\nPer-phase policy head probes:")
    for head_idx, phase_name in enumerate(DECISION_PHASE_ORDER):
        phase_id = sorted_phase_ids[head_idx]
        phase_mask = dataset.phases == phase_id
        n_phase = int(phase_mask.sum())
        if n_phase < 100:
            print(f"  {phase_name}: {n_phase} states (too few, skipping)")
            continue

        phase_states = dataset.states[phase_mask]
        phase_legal_masks = dataset.legal_masks[phase_mask]
        phase_phases = dataset.phases[phase_mask]

        # Extract targets for this phase subset
        phase_targets = _extract_game_targets(phase_states, config.num_players)
        phase_model_targets = _extract_model_targets(
            model, device, phase_states, phase_legal_masks,
            phase_phases, config.num_players, args.batch_size,
        )
        phase_targets.update(phase_model_targets)

        # Collect phase head activations
        t0 = time.perf_counter()
        phase_acts = collect_phase_activations(
            model, device, phase_states, head_idx, args.batch_size,
        )
        phase_layer_names = list(phase_acts.keys())

        # Train probes
        if device.type == "cuda":
            phase_results = train_probes_gpu(
                phase_acts, phase_targets, device, enabled_probes, seed=args.seed,
            )
        else:
            phase_results = train_probes(
                phase_acts, phase_targets, enabled_probes, seed=args.seed,
            )
        elapsed = time.perf_counter() - t0
        print(f"  {phase_name}: {n_phase} states, {len(phase_layer_names)} layers ({elapsed:.1f}s)")

        phase_probe_data.append({
            "name": phase_name,
            "n_states": n_phase,
            "headers": [_short_layer_name(n) for n in phase_layer_names],
            "rows": _results_to_json(phase_results, phase_layer_names),
            "results": phase_results,
            "layer_names": phase_layer_names,
        })

    # --- Nonlinear comparison ---
    comparisons: list[tuple[str, str, float, float]] | None = None
    if args.nonlinear:
        print("\nTraining nonlinear (MLP) comparison probes at trunk_norm...")
        t0 = time.perf_counter()
        comparisons = train_nonlinear_comparison(probe_layers, targets, seed=args.seed)
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # --- Print results ---
    trunk_layers, _, value_head_layers = _split_layer_groups(probe_layer_names)

    print(f"\n{'=' * 100}")
    print(f"  PROBING CLASSIFIER RESULTS (epoch {epoch})")
    print(f"  {dataset.num_states} states from {dataset.num_games} games, "
          f"train/test 80/20, linear probes")
    print(f"{'=' * 100}\n")

    if trunk_layers:
        print(format_results_table(results, trunk_layers))

    for pdata in phase_probe_data:
        print(f"\n{'=' * 70}")
        print(f"  {pdata['name']} HEAD ({pdata['n_states']} states)")
        print(f"{'=' * 70}\n")
        print(format_results_table(pdata["results"], pdata["layer_names"]))

    if value_head_layers:
        print(f"\n{'=' * 70}")
        print("  VALUE HEAD LAYERS")
        print(f"{'=' * 70}\n")
        print(format_results_table(results, value_head_layers))

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
        results, probe_layer_names, phase_probe_data,
        comparisons, epoch, dataset.num_states, dataset.num_games,
    ))
    print(f"Markdown written to {md_path}")

    # --- Write HTML ---
    html_path = md_path.with_suffix(".html")
    html = _format_html_report(
        results, probe_layer_names, phase_probe_data, comparisons, epoch,
        dataset.num_states, dataset.num_games,
    )
    html_path.write_text(html)
    print(f"HTML report written to {html_path}")

    if not args.no_open:
        open_file(html_path)


if __name__ == "__main__":
    main()
