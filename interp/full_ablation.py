"""Full state vector sensitivity analysis.

Ablates every named field in the state vector and produces a markdown
table showing policy KL divergence per game phase.

Usage:
    .venv/bin/python -m interp.full_ablation
    .venv/bin/python -m interp.full_ablation --load-data interp/data/states.npz
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from core.state import get_layout
from interp.feat_ablation import (
    _batch_masked_softmax,
    _forward_batched,
    _kl_divergence_batch,
)
from interp.utils import InterpDataset, collect_states, load_model

_PHASE_NAMES = {
    0: "INVEST",
    1: "BID",
    2: "WRAP_UP",
    3: "ACQ",
    4: "CLOSE",
    5: "INCOME",
    6: "DIV",
    7: "END_CARD",
    8: "ISSUE",
    9: "IPO",
    10: "GAME_OVER",
}

NUM_COMPANIES = 36
NUM_CORPS = 8


def _build_feature_groups(num_players: int) -> list[tuple[str, np.ndarray]]:
    """Build (name, indices) for every named field in the visible state."""
    layout = get_layout(num_players)
    groups: list[tuple[str, np.ndarray]] = []

    # --- Phase & CoO ---
    groups.append(("phase", np.arange(layout.phase_offset, layout.phase_offset + layout.phase_size)))
    groups.append(("coo_level", np.arange(layout.coo_offset, layout.coo_offset + layout.coo_size)))

    # --- Player fields (aggregated across all players) ---
    _pf = [
        ("player:cash", 0, 1),
        ("player:net_worth", 1, 1),
        ("player:turn_order", 2, num_players),
        ("player:auction_bidder", 2 + num_players, 1),
        ("player:owned_companies", 3 + num_players, 36),
        ("player:owned_shares", 39 + num_players, 8),
        ("player:is_president", 47 + num_players, 8),
        ("player:share_buys", 55 + num_players, 8),
        ("player:share_sells", 63 + num_players, 8),
        ("player:acq_proceeds", 71 + num_players, 1),
        ("player:income", 72 + num_players, 1),
    ]
    for name, rel, size in _pf:
        idx: list[int] = []
        for p in range(num_players):
            base = layout.players_offset + p * layout.player_stride + rel
            idx.extend(range(base, base + size))
        groups.append((name, np.array(idx)))

    # --- Foreign Investor ---
    groups.append(("fi:cash", np.array([layout.fi_offset])))
    groups.append(("fi:income", np.array([layout.fi_offset + 1])))
    groups.append(("fi:companies", np.arange(layout.fi_offset + 2, layout.fi_offset + layout.fi_size)))

    # --- Company location flags ---
    groups.append(("co:for_auction", np.arange(layout.auction_companies_offset, layout.auction_companies_offset + 36)))
    groups.append(("co:revealed", np.arange(layout.revealed_companies_offset, layout.revealed_companies_offset + 36)))
    groups.append(("co:removed", np.arange(layout.removed_companies_offset, layout.removed_companies_offset + 36)))

    # --- Company adjusted incomes ---
    groups.append(("co:adj_incomes", np.arange(layout.company_incomes_offset, layout.company_incomes_offset + layout.company_incomes_size)))

    # --- Market availability ---
    groups.append(("market:available", np.arange(layout.market_offset, layout.market_offset + layout.market_size)))

    # --- Corporation fields (aggregated across all corps) ---
    _cf = [
        ("corp:active", 0, 1),
        ("corp:cash", 1, 1),
        ("corp:unissued_shares", 2, 1),
        ("corp:issued_shares", 3, 1),
        ("corp:bank_shares", 4, 1),
        ("corp:income", 5, 1),
        ("corp:stars", 6, 1),
        ("corp:share_price", 7, 1),
        ("corp:acq_proceeds", 8, 1),
        ("corp:in_receivership", 9, 1),
        ("corp:price_index", 10, 27),
        ("corp:owned_companies", 37, 36),
        ("corp:acq_companies", 73, 36),
    ]
    for name, rel, size in _cf:
        idx = []
        for c in range(NUM_CORPS):
            base = layout.corps_offset + c * layout.corp_stride + rel
            idx.extend(range(base, base + size))
        groups.append((name, np.array(idx)))

    # --- Turn state ---
    t = layout.turn_offset
    groups.append(("turn:turn_number", np.array([t])))
    groups.append(("turn:end_card_flipped", np.array([t + 1])))
    groups.append(("turn:consec_passes", np.array([t + 2])))

    # Auction block: price(1) + high_bidder(np) + starter(np) + passed(np)
    auction_idx: list[int] = [t + 3]  # auction_price
    auction_idx.extend(range(layout.auction_high_bidder_offset, layout.auction_high_bidder_offset + num_players))
    auction_idx.extend(range(layout.auction_starter_offset, layout.auction_starter_offset + num_players))
    auction_idx.extend(range(layout.auction_passed_offset, layout.auction_passed_offset + num_players))
    groups.append(("turn:auction", np.array(auction_idx)))

    # Remaining turn fields are sequential after auction_passed
    post = layout.auction_passed_offset + num_players
    groups.append(("turn:dividend", np.arange(post, post + 34)))  # impact(26) + remaining(8)
    post += 34
    groups.append(("turn:issue", np.arange(post, post + 8)))  # remaining(8)
    post += 8
    groups.append(("turn:ipo_remaining", np.arange(post, post + 36)))  # remaining(36)
    post += 36
    groups.append(("turn:acq", np.arange(post, post + 37)))  # fi_offer(1) + synergy(36)
    post += 37
    groups.append(("turn:active_company", np.arange(post, post + 36)))  # one-hot(36)
    post += 36
    groups.append(("turn:active_company_info", np.arange(post, post + 5)))  # stars, low, face, high, income
    post += 5
    groups.append(("turn:active_corp", np.arange(post, post + 8)))  # one-hot(8)
    post += 8
    groups.append(("turn:active_corp_info", np.arange(post, post + 3)))  # income, stars, share_price
    post += 3
    groups.append(("turn:active_corp_companies", np.arange(post, post + 36)))  # owned company flags
    post += 36
    groups.append(("turn:cards_remaining", np.array([post])))  # cards remaining in deck
    post += 1
    assert post == layout.auction_slot_info_offset, (
        f"Turn end {post} != auction_slot_info_offset {layout.auction_slot_info_offset}"
    )

    # --- Auction slot info (5 scalars per slot: stars, low, face, high, income) ---
    s = layout.auction_slot_info_offset
    groups.append(("auction_slot_info", np.arange(s, s + layout.auction_slot_info_size)))

    # Verify full coverage
    all_idx = np.concatenate([g[1] for g in groups])
    assert len(all_idx) == layout.visible_size, (
        f"Coverage {len(all_idx)} != visible_size {layout.visible_size}"
    )
    assert len(set(all_idx)) == layout.visible_size, "Overlapping indices"

    return groups


def run_full_ablation(
    model: torch.nn.Module,
    device: torch.device,
    dataset: InterpDataset,
    num_players: int,
    batch_size: int = 256,
) -> tuple[list[tuple[str, int, float, dict[int, float]]], list[int]]:
    """Ablate every feature group, return rows and phase IDs.

    Each row: (name, num_features, total_kl, {phase_id: kl}).
    """
    groups = _build_feature_groups(num_players)
    phase_ids = sorted(set(dataset.phases))

    # Original outputs (computed once)
    print("  Computing original model outputs...")
    orig_logits, _ = _forward_batched(model, device, dataset.states, batch_size)
    orig_pol = _batch_masked_softmax(orig_logits, dataset.legal_masks)

    rows: list[tuple[str, int, float, dict[int, float]]] = []
    t0 = time.perf_counter()
    for i, (name, indices) in enumerate(groups):
        ablated = dataset.states.copy()
        ablated[:, indices] = 0.0
        abl_logits, _ = _forward_batched(model, device, ablated, batch_size)
        abl_pol = _batch_masked_softmax(abl_logits, dataset.legal_masks)
        kl = _kl_divergence_batch(orig_pol, abl_pol)

        total_kl = float(np.mean(kl))
        phase_kls = {pid: float(np.mean(kl[dataset.phases == pid])) for pid in phase_ids}
        rows.append((name, len(indices), total_kl, phase_kls))

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i + 1}/{len(groups)} groups ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(groups)} groups in {elapsed:.1f}s")

    # Sort by total KL descending
    rows.sort(key=lambda r: -r[2])
    return rows, phase_ids


def format_markdown_table(
    rows: list[tuple[str, int, float, dict[int, float]]],
    phase_ids: list[int],
) -> str:
    """Format results as a markdown table."""
    phase_names = [_PHASE_NAMES.get(pid, str(pid)) for pid in phase_ids]

    lines: list[str] = []

    # Header
    cols = ["Feature", "#", "Total"] + phase_names
    lines.append("| " + " | ".join(cols) + " |")
    aligns = [":---", "---:", "---:"] + ["---:"] * len(phase_names)
    lines.append("| " + " | ".join(aligns) + " |")

    # Rows
    for name, size, total_kl, phase_kls in rows:
        vals = [name, str(size), f"{total_kl:.4f}"]
        for pid in phase_ids:
            vals.append(f"{phase_kls[pid]:.4f}")
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full state vector sensitivity analysis (policy KL per feature per phase)"
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
        "--output", type=str, default=None,
        help="Write markdown table to file (default: stdout)",
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

    print(f"\nRunning full ablation ({dataset.num_states} states)...")
    rows, phase_ids = run_full_ablation(
        model, device, dataset, config.num_players, batch_size=args.batch_size,
    )

    table = format_markdown_table(rows, phase_ids)

    if args.output:
        with open(args.output, "w") as f:
            f.write(f"# State Vector Sensitivity (epoch {epoch})\n\n")
            f.write(f"Policy KL divergence when each feature group is zeroed.\n")
            f.write(f"Data: {dataset.num_states} states from {dataset.num_games} games.\n")
            f.write(f"Sorted by total KL (most sensitive first).\n\n")
            f.write(table)
            f.write("\n")
        print(f"\nTable written to {args.output}")
    else:
        print(f"\n{table}")


if __name__ == "__main__":
    main()
