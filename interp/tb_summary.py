"""Tensorboard training summary: extract key metrics from event files.

Prints a concise summary of training progress including loss curves,
policy entropy, self-play statistics, and schedule parameters.

Usage:
    .venv/bin/python -m interp.tb_summary
    .venv/bin/python -m interp.tb_summary --run-dir runs
    .venv/bin/python -m interp.tb_summary --max-rows 30
"""

from __future__ import annotations

import argparse

from train.tb_reader import read_tb_scalars, sample_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Tensorboard training summary")
    parser.add_argument("--run-dir", type=str, default="runs")
    parser.add_argument("--max-rows", type=int, default=15, help="Max epochs to show per table")
    args = parser.parse_args()

    data = read_tb_scalars(args.run_dir)
    if not data:
        print(f"No event data found in {args.run_dir}")
        return

    all_tags = sorted(data.keys())

    # Group tags by category
    categories = {
        "Epoch Loss": ["epoch/policy_loss_avg", "epoch/value_loss_avg", "epoch/total_loss_avg"],
        "Self-Play": [
            "self_play/policy_entropy_mean", "self_play/top1_visit_fraction",
            "self_play/game_length_mean",
        ],
        "Net Worth": [
            "self_play/net_worth_1st", "self_play/net_worth_2nd", "self_play/net_worth_3rd",
        ],
        "Schedule": ["schedule/c_puct", "schedule/value_blend_alpha", "schedule/subtree_reuse"],
        "Performance": [
            "self_play/duration_mean", "profile/server_throughput",
            "profile/server_batch_avg",
        ],
        "Buffer": ["buffer/size", "buffer/utilization"],
    }

    for category, tags in categories.items():
        present_tags = [t for t in tags if t in all_tags]
        if not present_tags:
            continue

        print(f"\n{'=' * 70}")
        print(f"  {category}")
        print(f"{'=' * 70}")

        # Collect all epochs across present tags, then sample
        all_epochs: set[int] = set()
        tag_data: dict[str, dict[int, float]] = {}
        for tag in present_tags:
            series = data[tag]
            tag_data[tag] = dict(series)
            all_epochs.update(step for step, _ in series)

        # Sample epochs for display
        epoch_list = sorted(all_epochs)
        sampled = sample_epochs([(e, 0) for e in epoch_list], args.max_rows)
        show_epochs = [e for e, _ in sampled]

        # Header
        short_names = [t.split("/")[-1] for t in present_tags]
        col_w = max(14, max(len(s) for s in short_names) + 2)
        header = f"  {'epoch':>6}"
        for sn in short_names:
            header += f" {sn:>{col_w}}"
        print(header)
        print(f"  {'-----':>6}" + f" {'-' * col_w}" * len(present_tags))

        for epoch in show_epochs:
            row = f"  {epoch:>6}"
            has_data = False
            for tag in present_tags:
                val = tag_data[tag].get(epoch)
                if val is not None:
                    row += f" {val:>{col_w}.4f}"
                    has_data = True
                else:
                    row += f" {'':>{col_w}}"
            if has_data:
                print(row)

    # Summary: first and last values for key metrics
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY (first -> last)")
    print(f"{'=' * 70}")

    summary_tags = [
        ("Policy loss", "epoch/policy_loss_avg"),
        ("Value loss", "epoch/value_loss_avg"),
        ("Policy entropy", "self_play/policy_entropy_mean"),
        ("Top-1 visit frac", "self_play/top1_visit_fraction"),
        ("Game length", "self_play/game_length_mean"),
        ("1st place NW", "self_play/net_worth_1st"),
        ("3rd place NW", "self_play/net_worth_3rd"),
        ("GPU throughput", "profile/server_throughput"),
    ]

    for label, tag in summary_tags:
        series = data.get(tag)
        if not series or len(series) < 2:
            continue
        first = series[0][1]
        last = series[-1][1]
        delta = last - first
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {first:>10.4f} -> {last:>10.4f}  ({sign}{delta:.4f})")

    # List unshown tags
    shown: set[str] = set()
    for tags in categories.values():
        shown.update(tags)
    unshown = [t for t in all_tags if t not in shown]
    if unshown:
        print(f"\n  Other available tags: {', '.join(unshown)}")

    print()


if __name__ == "__main__":
    main()
