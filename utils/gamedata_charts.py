#!/usr/bin/env python3
"""Chart generation for collected strategy-data runs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rss-az-cython2-matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.gamedata_analysis import (  # noqa: E402
    CompanyAuctionOutcomeSummary,
    CompanyBidDeltaSummary,
    CorpIpoAvailablePickSummary,
    CorpIpoHeadToHeadSummary,
    CorpIpoPositionPickSummary,
    CorpIpoTurnShareSummary,
    EarlyMaxPriceEndSummary,
    InitialAuctionPositionSummary,
    NetWorthBreakdownSummary,
    OpeningValueSummary,
    RankedNetWorthBreakdownSummary,
    StrategyDataset,
    TurnAverageSummary,
    TurnMeanSummary,
    TurnOneAuctionCompanyPresenceSummary,
    TurnOneAuctionDeckPresenceSummary,
    TurnOneAuctionPoolPremiumSummary,
    TurnOneIpoParPriceSummary,
    TurnOneOpeningSummary,
    WinRateSummary,
)


POSITION_COLORS = [
    "#1b6ca8",
    "#5c8f22",
    "#c07a18",
    "#8f5aa3",
    "#b44747",
]
POSITIVE_COLOR = "#247a52"
NEGATIVE_COLOR = "#b84747"
FAIR_LINE_COLOR = "#333333"
COMPANY_STAR_COLORS = {
    1: "#c43c39",  # red
    2: "#de7c22",  # orange
    3: "#c49a17",  # yellow/gold, dark enough for labels
    4: "#3c8f4f",  # green
    5: "#2d6fb7",  # blue
}
COMPANY_STAR_LABELS = {
    1: "Red",
    2: "Orange",
    3: "Yellow",
    4: "Green",
    5: "Blue",
}
CORP_COLORS = {
    "JS": "#9c4f12",  # darker/burnt orange
    "S": "#79c7ee",   # light blue
    "OS": "#f2cf3a",  # yellow
    "SM": "#3fa34d",  # green
    "PR": "#111111",  # black
    "DA": "#c62828",  # red
    "VM": "#ff7a00",  # bright orange
    "SI": "#7b3f98",  # purple
}
NET_WORTH_COLORS = {
    "Cash": "#2f7d4f",
    "Companies": "#d69a2d",
    "Shares": "#356bb3",
}
AUCTION_OUTCOME_COLORS = {
    "Acquired by Corp": "#3c7ea6",
    "IPO Seed": "#3fa34d",
    "Closed": "#b45c3a",
    "Held to End": "#d1a33b",
}


def _parse_player_counts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise ValueError("--player-counts must contain at least one integer")
    return counts


def _as_axes_array(axes: object) -> np.ndarray:
    return np.asarray(axes, dtype=object).reshape(-1)


def _position_labels(num_players: int) -> list[str]:
    return [f"P{i}" for i in range(1, num_players + 1)]


def _bar_colors(num_players: int) -> list[str]:
    return [POSITION_COLORS[i % len(POSITION_COLORS)] for i in range(num_players)]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars(
    ax: plt.Axes,
    bars: object,
    values: np.ndarray,
    *,
    fmt: str,
    zero_based: bool,
) -> None:
    for bar, value in zip(bars, values):
        height = float(bar.get_height())
        if zero_based:
            y = height
            va = "bottom"
            offset = 3
        elif height >= 0:
            y = height
            va = "bottom"
            offset = 3
        else:
            y = height
            va = "top"
            offset = -4
        ax.annotate(
            fmt.format(float(value)),
            xy=(bar.get_x() + bar.get_width() / 2.0, y),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=8,
            color="#222222",
        )


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_opening_nn_values(
    summaries: dict[int, OpeningValueSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel bar chart of opening NN values by player position."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.2 * len(counts), 3.6))
    axes_arr = _as_axes_array(axes)

    all_values = np.concatenate([summaries[count].mean_values for count in counts])
    limit = max(0.02, float(np.max(np.abs(all_values))) * 1.25)

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        values = summary.mean_values.astype(float)
        positions = np.arange(num_players)
        bars = ax.bar(
            positions,
            values,
            color=_bar_colors(num_players),
            edgecolor="#ffffff",
            linewidth=0.8,
        )
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_title(f"{num_players} Players")
        ax.set_xticks(positions)
        ax.set_xticklabels(_position_labels(num_players))
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Mean NN value" if ax is axes_arr[0] else "")
        ax.set_xlabel("Player position")
        _style_axis(ax)
        _annotate_bars(ax, bars, values, fmt="{:+.3f}", zero_based=False)

    fig.suptitle("Opening Move: NN Value Estimate by Position", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_actual_win_rates(
    summaries: dict[int, WinRateSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel bar chart of actual final-net-worth win rates."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.2 * len(counts), 3.6))
    axes_arr = _as_axes_array(axes)
    max_rate = max(
        float(np.max(summary.fractional_win_rates))
        for summary in summaries.values()
    )
    y_max = max_rate * 100.0 * 1.25

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        values = summary.fractional_win_rates.astype(float) * 100.0
        positions = np.arange(num_players)
        fair = 100.0 / num_players
        bars = ax.bar(
            positions,
            values,
            color=_bar_colors(num_players),
            edgecolor="#ffffff",
            linewidth=0.8,
        )
        ax.axhline(
            fair,
            color=FAIR_LINE_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="Fair share",
        )
        ax.set_title(f"{num_players} Players")
        ax.set_xticks(positions)
        ax.set_xticklabels(_position_labels(num_players))
        ax.set_ylim(0.0, y_max)
        ax.set_ylabel("Win rate (%)" if ax is axes_arr[0] else "")
        ax.set_xlabel("Player position")
        _style_axis(ax)
        _annotate_bars(ax, bars, values, fmt="{:.1f}%", zero_based=True)

    axes_arr[-1].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Actual Win Rate by Position", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_win_rate_deltas(
    summaries: dict[int, WinRateSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel chart of win-rate delta from fair share."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.2 * len(counts), 3.6))
    axes_arr = _as_axes_array(axes)

    deltas_by_count = {
        num_players: (
            summaries[num_players].fractional_win_rates.astype(float)
            - (1.0 / num_players)
        )
        * 100.0
        for num_players in counts
    }
    limit = max(
        1.0,
        max(float(np.max(np.abs(values))) for values in deltas_by_count.values()) * 1.25,
    )

    for ax, num_players in zip(axes_arr, counts):
        values = deltas_by_count[num_players]
        positions = np.arange(num_players)
        colors = [POSITIVE_COLOR if value >= 0 else NEGATIVE_COLOR for value in values]
        bars = ax.bar(
            positions,
            values,
            color=colors,
            edgecolor="#ffffff",
            linewidth=0.8,
        )
        ax.axhline(0.0, color="#222222", linewidth=1.0)
        ax.set_title(f"{num_players} Players")
        ax.set_xticks(positions)
        ax.set_xticklabels(_position_labels(num_players))
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Delta from fair (%)" if ax is axes_arr[0] else "")
        ax.set_xlabel("Player position")
        _style_axis(ax)
        _annotate_bars(ax, bars, values, fmt="{:+.1f}%", zero_based=False)

    fig.suptitle("Actual Win Rate Minus Fair Share", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_auction_bid_delta_by_company(
    summary: CompanyBidDeltaSummary,
    output_path: str | Path,
    *,
    title_prefix: str = "Average Auction Price",
    title: str | None = None,
    ylabel: str = "Winning auction price - face value",
) -> Path:
    """Write a wide company chart of bid delta over face value."""
    values = summary.mean_deltas.astype(float)
    positions = np.arange(len(summary.company_names))
    colors = [
        COMPANY_STAR_COLORS.get(int(stars), "#888888")
        for stars in summary.company_stars
    ]
    finite_values = values[np.isfinite(values)]
    y_max = max(1.0, float(np.max(finite_values)) * 1.18) if finite_values.size else 1.0

    fig, ax = plt.subplots(figsize=(18.0, 5.2))
    bars = ax.bar(
        positions,
        np.nan_to_num(values, nan=0.0),
        color=colors,
        edgecolor="#ffffff",
        linewidth=0.55,
    )
    zero_count = summary.counts == 0
    for bar, missing in zip(bars, zero_count):
        if missing:
            bar.set_hatch("//")
            bar.set_alpha(0.35)

    if title is None:
        title = f"{title_prefix} Delta Over Face Value"
    ax.set_title(f"{summary.num_players} Players: {title}")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Company")
    ax.set_xticks(positions)
    ax.set_xticklabels(summary.company_names, rotation=65, ha="right", fontsize=8)
    for tick, stars in zip(ax.get_xticklabels(), summary.company_stars):
        tick.set_color(COMPANY_STAR_COLORS.get(int(stars), "#555555"))
        tick.set_fontweight("bold")
    ax.set_ylim(0.0, y_max)
    ax.margins(x=0.005)
    _style_axis(ax)

    legend_items = [
        Patch(
            facecolor=COMPANY_STAR_COLORS[tier],
            edgecolor="none",
            label=COMPANY_STAR_LABELS[tier],
        )
        for tier in sorted(COMPANY_STAR_COLORS)
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        ncols=5,
        frameon=False,
        fontsize=9,
    )
    observed = int(np.count_nonzero(summary.counts))
    ax.text(
        1.0,
        -0.28,
        (
            f"Averaged over observed bids; "
            f"{observed}/{len(summary.company_names)} companies observed."
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_auction_bid_delta_charts(
    summaries: dict[int, CompanyBidDeltaSummary],
    output_dir: str | Path,
    *,
    filename_prefix: str = "auction_bid_delta",
    title_prefix: str = "Average Auction Price",
    title: str | None = None,
    ylabel: str = "Winning auction price - face value",
) -> list[Path]:
    """Write one company bid-delta chart per player count."""
    output = Path(output_dir)
    return [
        plot_auction_bid_delta_by_company(
            summary,
            output / f"{filename_prefix}_{num_players}p.png",
            title_prefix=title_prefix,
            title=title,
            ylabel=ylabel,
        )
        for num_players, summary in sorted(summaries.items())
    ]


def plot_auctioned_company_outcomes(
    summary: CompanyAuctionOutcomeSummary,
    output_path: str | Path,
) -> Path:
    """Write a company-level stacked area chart of auction-won outcomes."""
    x = np.arange(len(summary.company_names))
    colors = [
        AUCTION_OUTCOME_COLORS.get(name, "#777777")
        for name in summary.outcome_names
    ]

    fig, ax = plt.subplots(figsize=(18.0, 5.8))
    if x.size:
        ax.stackplot(
            x,
            summary.percentages,
            labels=summary.outcome_names,
            colors=colors,
            alpha=0.92,
            linewidth=0.35,
            edgecolor="#ffffff",
        )
    ax.set_title(
        f"{summary.num_players} Players: Outcomes for Companies Won at Auction"
    )
    ax.set_ylabel("Outcome share (%)")
    ax.set_xlabel("Company")
    ax.set_xticks(x)
    ax.set_xticklabels(summary.company_names, rotation=65, ha="right", fontsize=8)
    for tick, stars in zip(ax.get_xticklabels(), summary.company_stars):
        tick.set_color(COMPANY_STAR_COLORS.get(int(stars), "#555555"))
        tick.set_fontweight("bold")
    ax.set_xlim(float(x.min()), float(x.max())) if x.size else None
    ax.set_ylim(0.0, 100.0)
    ax.margins(x=0.005)
    _style_axis(ax)
    ax.legend(
        loc="upper left",
        ncols=len(summary.outcome_names),
        frameon=False,
        fontsize=9,
    )
    ax.text(
        1.0,
        -0.28,
        (
            "Denominator is companies won by players at auction; "
            f"total observed outcomes: {int(summary.totals.sum())}."
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_auctioned_company_outcome_charts(
    summaries: dict[int, CompanyAuctionOutcomeSummary],
    output_dir: str | Path,
) -> list[Path]:
    """Write one auction-won outcome chart per player count."""
    output = Path(output_dir)
    return [
        plot_auctioned_company_outcomes(
            summary,
            output / f"auctioned_company_outcomes_{num_players}p.png",
        )
        for num_players, summary in sorted(summaries.items())
    ]


def plot_turn_one_opening_summary(
    summaries: dict[int, TurnOneOpeningSummary],
    output_path: str | Path,
) -> Path:
    """Write a 2x3 chart of Turn 1 auctions and first IPO corp choices."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(2, len(counts), figsize=(5.0 * len(counts), 7.6))
    axes_arr = np.asarray(axes, dtype=object)

    all_deltas = np.concatenate(
        [
            summary.auction_mean_deltas[np.isfinite(summary.auction_mean_deltas)]
            for summary in summaries.values()
            if np.any(np.isfinite(summary.auction_mean_deltas))
        ]
    )
    auction_y_max = (
        max(1.0, float(np.max(all_deltas)) * 1.22)
        if all_deltas.size
        else 1.0
    )
    first_pick_y_max = max(
        5.0,
        max(
            (
                float(np.max(summary.first_ipo_percentages))
                for summary in summaries.values()
                if summary.first_ipo_percentages.size
            ),
            default=0.0,
        )
        * 1.22,
    )

    for col, num_players in enumerate(counts):
        summary = summaries[num_players]

        auction_ax = axes_arr[0, col]
        company_positions = np.arange(len(summary.company_names))
        auction_values = np.nan_to_num(summary.auction_mean_deltas, nan=0.0)
        auction_bars = auction_ax.bar(
            company_positions,
            auction_values,
            width=0.72,
            color=[
                COMPANY_STAR_COLORS.get(int(stars), "#888888")
                for stars in summary.company_stars
            ],
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        for bar, missing in zip(auction_bars, summary.auction_counts == 0):
            if missing:
                bar.set_hatch("//")
                bar.set_alpha(0.35)
        auction_ax.set_title(f"{num_players} Players")
        auction_ax.set_xlabel("Red company")
        auction_ax.set_ylabel("Auction price - face value" if col == 0 else "")
        auction_ax.set_xticks(company_positions)
        auction_ax.set_xticklabels(
            summary.company_names,
            rotation=45,
            ha="right",
            fontsize=9,
        )
        for tick, stars in zip(auction_ax.get_xticklabels(), summary.company_stars):
            tick.set_color(COMPANY_STAR_COLORS.get(int(stars), "#555555"))
            tick.set_fontweight("bold")
        auction_ax.set_ylim(0.0, auction_y_max)
        _style_axis(auction_ax)
        _annotate_bars(
            auction_ax,
            auction_bars,
            auction_values,
            fmt="{:.2f}",
            zero_based=True,
        )

        corp_ax = axes_arr[1, col]
        corp_positions = np.arange(len(summary.corp_names))
        corp_bars = corp_ax.bar(
            corp_positions,
            summary.first_ipo_percentages,
            width=0.72,
            color=[CORP_COLORS.get(name, "#777777") for name in summary.corp_names],
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        corp_ax.set_xlabel("First floated corp")
        corp_ax.set_ylabel("First pick share (%)" if col == 0 else "")
        corp_ax.set_xticks(corp_positions)
        corp_ax.set_xticklabels(summary.corp_names, rotation=45, ha="right", fontsize=9)
        corp_ax.set_ylim(0.0, first_pick_y_max)
        _style_axis(corp_ax)
        _annotate_bars(
            corp_ax,
            corp_bars,
            summary.first_ipo_percentages,
            fmt="{:.1f}%",
            zero_based=True,
        )
        corp_ax.text(
            1.0,
            -0.28,
            f"First IPO observed in {summary.first_ipo_games}/{summary.num_games} games.",
            transform=corp_ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )

    axes_arr[0, 0].set_title(f"{counts[0]} Players\nTurn 1 Auction Deltas")
    for col in range(1, len(counts)):
        axes_arr[0, col].set_title(
            f"{counts[col]} Players\nTurn 1 Auction Deltas"
        )
    axes_arr[1, 0].set_title("First Corp to IPO")
    for col in range(1, len(counts)):
        axes_arr[1, col].set_title("First Corp to IPO")

    fig.suptitle("Turn 1 Auction Prices and First IPO Corp", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_corp_ipo_turn_share(
    summary: CorpIpoTurnShareSummary,
    output_path: str | Path,
) -> Path:
    """Write a stacked area chart of per-turn IPO share by corp."""
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    colors = [CORP_COLORS.get(name, "#777777") for name in summary.corp_names]

    if summary.turns.size:
        ax.stackplot(
            summary.turns,
            summary.percentages.T,
            labels=summary.corp_names,
            colors=colors,
            alpha=0.92,
            linewidth=0.35,
            edgecolor="#ffffff",
        )
        ax.set_xlim(float(summary.turns.min()), float(summary.turns.max()))
        ax.set_xticks(summary.turns)
    else:
        ax.text(
            0.5,
            0.5,
            "No IPO events",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    ax.set_ylim(0.0, 100.0)
    ax.set_title(f"{summary.num_players} Players: IPO Share by Corp and Turn")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Share of IPOs on turn (%)")
    _style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=8,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_corp_ipo_turn_share_charts(
    summaries: dict[int, CorpIpoTurnShareSummary],
    output_dir: str | Path,
) -> list[Path]:
    """Write one corp IPO turn-share chart per player count."""
    output = Path(output_dir)
    return [
        plot_corp_ipo_turn_share(
            summary,
            output / f"corp_ipo_turn_share_{num_players}p.png",
        )
        for num_players, summary in sorted(summaries.items())
    ]


def plot_corp_ipo_positional_pick_bubbles(
    summaries: dict[int, CorpIpoPositionPickSummary],
    output_path: str | Path,
) -> Path:
    """Write a bubble matrix of corp pick share by turn and IPO slot."""
    counts = sorted(summaries)
    if not counts:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No player counts", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, Path(output_path))

    first_summary = summaries[counts[0]]
    corp_names = first_summary.corp_names
    num_corps = len(corp_names)
    max_picks = max(
        (int(summary.pick_positions.shape[0]) for summary in summaries.values()),
        default=0,
    )
    max_turn = max(
        (int(summary.turns.max()) for summary in summaries.values()
         if summary.turns.size),
        default=0,
    )
    all_turns = np.arange(1, max_turn + 1, dtype=np.int64)
    y_positions = np.arange(num_corps - 1, -1, -1, dtype=np.float64)

    fig, axes = plt.subplots(
        max_picks,
        len(counts),
        figsize=(5.8 * len(counts), 3.35 * max_picks),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row in range(max_picks):
        pick_label = _rank_labels(np.asarray([row + 1], dtype=np.int64))[0]
        for col, num_players in enumerate(counts):
            ax = axes[row, col]
            summary = summaries[num_players]
            turn_index = {
                int(turn): idx
                for idx, turn in enumerate(summary.turns)
            }

            if row < summary.percentages.shape[0]:
                if row < summary.totals.shape[0] and int(summary.totals[row].sum()) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No observed picks",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#777777",
                    )
                for corp_id, corp_name in enumerate(corp_names):
                    xs: list[int] = []
                    ys: list[float] = []
                    sizes: list[float] = []
                    for turn in all_turns:
                        idx = turn_index.get(int(turn))
                        if idx is None:
                            continue
                        pct = float(summary.percentages[row, idx, corp_id])
                        if pct <= 0.0:
                            continue
                        xs.append(int(turn))
                        ys.append(float(y_positions[corp_id]))
                        sizes.append(18.0 + pct * 8.0)
                    if xs:
                        ax.scatter(
                            xs,
                            ys,
                            s=sizes,
                            c=CORP_COLORS.get(corp_name, "#777777"),
                            alpha=0.90,
                            edgecolors="#ffffff",
                            linewidths=0.55,
                        )

            for turn in all_turns:
                total = 0
                idx = turn_index.get(int(turn))
                if idx is not None and row < summary.totals.shape[0]:
                    total = int(summary.totals[row, idx])
                if total:
                    ax.text(
                        float(turn),
                        -0.92,
                        str(total),
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color="#666666",
                    )
            if all_turns.size:
                ax.text(
                    float(all_turns[0]) - 0.62,
                    -0.92,
                    "n",
                    ha="right",
                    va="center",
                    fontsize=6.0,
                    color="#555555",
                    fontweight="bold",
                )

            if row == 0:
                ax.set_title(f"{num_players} Players")
            if col == 0:
                ax.set_ylabel(f"{pick_label} pick\nCorp")
            ax.set_xlabel("Turn", labelpad=2)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(corp_names)
            for tick, corp_name in zip(ax.get_yticklabels(), corp_names):
                tick.set_color(CORP_COLORS.get(corp_name, "#555555"))
                tick.set_fontweight("bold")
            if all_turns.size:
                ax.set_xlim(float(all_turns.min()) - 0.5, float(all_turns.max()) + 0.5)
                ax.set_xticks(all_turns)
                ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylim(-1.35, float(num_corps) - 0.35)
            ax.grid(axis="both", color="#e1e1e1", linewidth=0.75, alpha=0.85)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    size_handles = [
        plt.scatter(
            [],
            [],
            s=18.0 + pct * 8.0,
            c="#777777",
            alpha=0.85,
            edgecolors="#ffffff",
            linewidths=0.55,
            label=f"{pct}%",
        )
        for pct in (25, 50, 100)
    ]
    fig.legend(
        handles=size_handles,
        title="Pick share",
        loc="lower center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "Corp IPO Positional Pick Percentage by Turn",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.085,
        "Small gray numbers show the number of observed picks in each turn/slot bucket.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.16, hspace=0.38)
    return _save_figure(fig, Path(output_path))


def plot_corp_ipo_available_pick_bubbles(
    summaries: dict[int, CorpIpoAvailablePickSummary],
    output_path: str | Path,
) -> Path:
    """Write a bubble matrix of pick probability with at least two available corps."""
    counts = sorted(summaries)
    if not counts:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No player counts", ha="center", va="center")
        ax.axis("off")
        return _save_figure(fig, Path(output_path))

    first_summary = summaries[counts[0]]
    corp_names = first_summary.corp_names
    num_corps = len(corp_names)
    max_picks = max(
        (int(summary.pick_positions.shape[0]) for summary in summaries.values()),
        default=0,
    )
    max_turn = max(
        (int(summary.turns.max()) for summary in summaries.values()
         if summary.turns.size),
        default=0,
    )
    all_turns = np.arange(1, max_turn + 1, dtype=np.int64)
    y_positions = np.arange(num_corps - 1, -1, -1, dtype=np.float64)

    fig, axes = plt.subplots(
        max_picks,
        len(counts),
        figsize=(5.8 * len(counts), 3.35 * max_picks),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row in range(max_picks):
        pick_label = _rank_labels(np.asarray([row + 1], dtype=np.int64))[0]
        for col, num_players in enumerate(counts):
            ax = axes[row, col]
            summary = summaries[num_players]
            turn_index = {
                int(turn): idx
                for idx, turn in enumerate(summary.turns)
            }

            if row < summary.percentages.shape[0]:
                if (
                    row < summary.available_counts.shape[0]
                    and int(summary.available_counts[row].sum()) == 0
                ):
                    ax.text(
                        0.5,
                        0.5,
                        "No available picks",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#777777",
                    )
                for corp_id, corp_name in enumerate(corp_names):
                    xs: list[int] = []
                    ys: list[float] = []
                    sizes: list[float] = []
                    for turn in all_turns:
                        idx = turn_index.get(int(turn))
                        if idx is None:
                            continue
                        available = int(summary.available_counts[row, idx, corp_id])
                        if available <= 0:
                            continue
                        picked = int(summary.counts[row, idx, corp_id])
                        pct = float(picked * 100.0 / available)
                        if pct <= 0.0:
                            continue
                        xs.append(int(turn))
                        ys.append(float(y_positions[corp_id]))
                        sizes.append(18.0 + pct * 8.0)
                    if xs:
                        ax.scatter(
                            xs,
                            ys,
                            s=sizes,
                            c=CORP_COLORS.get(corp_name, "#777777"),
                            alpha=0.90,
                            edgecolors="#ffffff",
                            linewidths=0.55,
                        )

            if row == 0:
                ax.set_title(f"{num_players} Players")
            if col == 0:
                ax.set_ylabel(f"{pick_label} pick\nCorp")
            ax.set_xlabel("Turn", labelpad=2)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(corp_names)
            for tick, corp_name in zip(ax.get_yticklabels(), corp_names):
                tick.set_color(CORP_COLORS.get(corp_name, "#555555"))
                tick.set_fontweight("bold")
            if all_turns.size:
                ax.set_xlim(float(all_turns.min()) - 0.5, float(all_turns.max()) + 0.5)
                ax.set_xticks(all_turns)
                ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylim(-0.35, float(num_corps) - 0.35)
            ax.grid(axis="both", color="#e1e1e1", linewidth=0.75, alpha=0.85)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    size_handles = [
        plt.scatter(
            [],
            [],
            s=18.0 + pct * 8.0,
            c="#777777",
            alpha=0.85,
            edgecolors="#ffffff",
            linewidths=0.55,
            label=f"{pct}%",
        )
        for pct in (25, 50, 100)
    ]
    fig.legend(
        handles=size_handles,
        title="Pick probability",
        loc="lower center",
        ncols=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "Corp IPO Pick Probability When Available with Alternatives by Turn",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.085,
        (
            "Bubble size is P(corp picked | at least two corps inactive, including this corp); "
            "columns do not sum to 100%."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.16, hspace=0.38)
    return _save_figure(fig, Path(output_path))


def plot_corp_ipo_head_to_head_heatmaps(
    summary: CorpIpoHeadToHeadSummary,
    output_path: str | Path,
) -> Path:
    """Write 27 directional corp-vs-corp IPO heatmaps for one player count."""
    corp_names = summary.corp_names
    num_corps = len(corp_names)
    pick_count = int(summary.pick_positions.shape[0])
    turns = summary.turns[:9]
    turn_grid_cols = 3
    turn_grid_rows = 3
    rows = turn_grid_rows
    cols = pick_count * turn_grid_cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.85 * cols, 2.75 * rows),
        squeeze=False,
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    colorbar_mappable.set_array([])
    # Draw cells directly so subplot scaling cannot resample colors across rows.
    cell_gap = 0.018

    for pick_index, pick_position in enumerate(summary.pick_positions):
        pick_label = _rank_labels(np.asarray([pick_position], dtype=np.int64))[0]
        for turn_offset, turn in enumerate(turns):
            grid_row = int(turn_offset // turn_grid_cols)
            grid_col = (pick_index * turn_grid_cols) + int(turn_offset % turn_grid_cols)
            ax = axes[grid_row, grid_col]
            turn_index = int(turn) - 1
            values = summary.percentages[pick_index, turn_index].copy()
            values[summary.totals[pick_index, turn_index] == 0] = np.nan
            np.fill_diagonal(values, np.nan)
            ax.set_facecolor("#ffffff")
            for row in range(num_corps):
                for col in range(num_corps):
                    value = values[row, col]
                    facecolor = "#eeeeee" if np.isnan(value) else cmap(norm(float(value)))
                    ax.add_patch(
                        Rectangle(
                            (col - 0.5 + cell_gap, row - 0.5 + cell_gap),
                            1.0 - (2.0 * cell_gap),
                            1.0 - (2.0 * cell_gap),
                            facecolor=facecolor,
                            edgecolor="none",
                            antialiased=False,
                        )
                    )

            ax.set_title(f"Turn {int(turn)}", fontsize=9, pad=4)
            ax.set_xticks(np.arange(num_corps))
            ax.set_yticks(np.arange(num_corps))
            ax.set_xticklabels(corp_names, rotation=45, ha="right", fontsize=6)
            for tick, corp_name in zip(ax.get_xticklabels(), corp_names):
                tick.set_color(CORP_COLORS.get(corp_name, "#555555"))
                tick.set_fontweight("bold")
            ax.set_yticklabels(corp_names, fontsize=6)
            for tick, corp_name in zip(ax.get_yticklabels(), corp_names):
                tick.set_color(CORP_COLORS.get(corp_name, "#555555"))
                tick.set_fontweight("bold")

            ax.set_xlim(-0.5, num_corps - 0.5)
            ax.set_ylim(num_corps - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", length=0, pad=1)
            for spine in ax.spines.values():
                spine.set_visible(False)

        for turn_offset in range(len(turns), turn_grid_cols * turn_grid_rows):
            grid_row = int(turn_offset // turn_grid_cols)
            grid_col = (pick_index * turn_grid_cols) + int(turn_offset % turn_grid_cols)
            axes[grid_row, grid_col].axis("off")

    fig.suptitle(
        f"{summary.num_players} Players: IPO Head-to-Head Pick Percentages",
        fontsize=15,
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Rows are the picked corp, columns are the comparison corp. "
            "Each cell is P(row picked | row and column both inactive, picked corp is one of the pair)."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.045,
        right=0.99,
        top=0.82,
        bottom=0.19,
        hspace=0.36,
        wspace=0.08,
    )
    grid_top = axes[0, 0].get_position().y1
    grid_bottom = axes[-1, 0].get_position().y0
    for pick_index, pick_position in enumerate(summary.pick_positions):
        pick_label = _rank_labels(np.asarray([pick_position], dtype=np.int64))[0]
        start_col = pick_index * turn_grid_cols
        end_col = start_col + turn_grid_cols - 1
        section_left = axes[0, start_col].get_position().x0
        section_right = axes[0, end_col].get_position().x1
        fig.text(
            (section_left + section_right) * 0.5,
            grid_top + 0.045,
            f"{pick_label} Pick",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    for pick_index in range(1, pick_count):
        previous_col = (pick_index * turn_grid_cols) - 1
        next_col = pick_index * turn_grid_cols
        x_mid = (
            axes[0, previous_col].get_position().x1
            + axes[0, next_col].get_position().x0
        ) * 0.5
        fig.add_artist(
            Rectangle(
                (x_mid - 0.0015, grid_bottom),
                0.003,
                grid_top - grid_bottom,
                transform=fig.transFigure,
                color="#b8b8b8",
                linewidth=0,
            )
        )
    if pick_count >= 2:
        middle_start_col = turn_grid_cols
        middle_end_col = (2 * turn_grid_cols) - 1
        colorbar_left = axes[-1, middle_start_col].get_position().x0
        colorbar_right = axes[-1, middle_end_col].get_position().x1
    else:
        colorbar_left = axes[-1, 0].get_position().x0
        colorbar_right = axes[-1, -1].get_position().x1
    colorbar_ax = fig.add_axes(
        (colorbar_left, 0.085, colorbar_right - colorbar_left, 0.025)
    )
    colorbar = fig.colorbar(
        colorbar_mappable,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("P(row corp picked over column corp)", fontsize=8)
    colorbar.set_ticks([0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=8)
    return _save_figure(fig, Path(output_path))


def plot_corp_ipo_head_to_head_heatmap_charts(
    summaries: dict[int, CorpIpoHeadToHeadSummary],
    output_dir: str | Path,
) -> list[Path]:
    """Write one IPO head-to-head heatmap chart per player count."""
    output = Path(output_dir)
    return [
        plot_corp_ipo_head_to_head_heatmaps(
            summary,
            output / f"corp_ipo_head_to_head_{num_players}p.png",
        )
        for num_players, summary in sorted(summaries.items())
    ]


def plot_floated_corps_by_turn(
    summaries: dict[int, TurnAverageSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel bar chart of average floated corps per game by turn."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.5 * len(counts), 3.8))
    axes_arr = _as_axes_array(axes)
    max_mean = max(
        (float(np.max(summary.mean_counts)) for summary in summaries.values()
         if summary.mean_counts.size),
        default=1.0,
    )
    y_max = max(0.25, max_mean * 1.22)

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        bars = ax.bar(
            summary.turns,
            summary.mean_counts,
            width=0.78,
            color="#3c7ea6",
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Turn number")
        ax.set_ylabel("Avg floated corps / game" if ax is axes_arr[0] else "")
        ax.set_ylim(0.0, y_max)
        if summary.turns.size:
            ax.set_xticks(summary.turns)
        _style_axis(ax)
        if summary.turns.size <= 14:
            _annotate_bars(
                ax,
                bars,
                summary.mean_counts,
                fmt="{:.2f}",
                zero_based=True,
            )

    fig.suptitle("Average Number of Corps Floated per Game by Turn", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_average_par_price_by_turn(
    summaries: dict[int, TurnMeanSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel bar chart of average IPO par price by turn."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.5 * len(counts), 3.8))
    axes_arr = _as_axes_array(axes)
    all_values = np.concatenate(
        [summary.mean_values for summary in summaries.values()
         if summary.mean_values.size]
    )
    y_max = max(1.0, float(np.nanmax(all_values)) * 1.15) if all_values.size else 1.0

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        bars = ax.bar(
            summary.turns,
            summary.mean_values,
            width=0.78,
            color="#7a5aa6",
            edgecolor="#ffffff",
            linewidth=0.7,
        )
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Turn number")
        ax.set_ylabel("Average par price" if ax is axes_arr[0] else "")
        ax.set_ylim(0.0, y_max)
        if summary.turns.size:
            ax.set_xticks(summary.turns)
        _style_axis(ax)
        if summary.turns.size <= 14:
            _annotate_bars(
                ax,
                bars,
                summary.mean_values,
                fmt="{:.1f}",
                zero_based=True,
            )

    fig.suptitle("Average IPO Par Price by Turn", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_turn_one_ipo_par_price_distribution(
    summaries: dict[int, TurnOneIpoParPriceSummary],
    output_path: str | Path,
) -> Path:
    """Write T1 IPO par-price percentage lines by pick order."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(
        1,
        len(counts),
        figsize=(5.0 * len(counts), 4.2),
        sharey=True,
    )
    axes_arr = _as_axes_array(axes)
    price_chunks = [
        summary.par_prices
        for summary in summaries.values()
        if summary.par_prices.size
    ]
    percentage_chunks = [
        summary.percentages[np.isfinite(summary.percentages)]
        for summary in summaries.values()
        if summary.percentages.size
    ]
    all_prices = (
        np.concatenate(price_chunks)
        if price_chunks
        else np.empty(0, dtype=np.int64)
    )
    all_percentages = (
        np.concatenate(percentage_chunks)
        if percentage_chunks
        else np.empty(0, dtype=np.float64)
    )
    y_max = (
        max(1.0, float(np.max(all_percentages)) * 1.16)
        if all_percentages.size
        else 1.0
    )

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        if summary.par_prices.size == 0 or summary.pick_positions.size == 0:
            ax.text(
                0.5,
                0.5,
                "No Turn 1 IPOs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#777777",
            )
        else:
            for line_index, pick_position in enumerate(summary.pick_positions):
                label = _rank_labels(np.asarray([pick_position], dtype=np.int64))[0]
                ax.plot(
                    summary.par_prices,
                    summary.percentages[line_index],
                    marker="o",
                    markersize=4.0,
                    linewidth=1.7,
                    color=POSITION_COLORS[line_index % len(POSITION_COLORS)],
                    label=f"{label} pick",
                )

        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("IPO par price")
        ax.set_ylabel("Share of T1 IPOs at par price (%)" if ax is axes_arr[0] else "")
        ax.set_ylim(0.0, y_max)
        if all_prices.size:
            unique_prices = np.unique(all_prices)
            ax.set_xticks(unique_prices)
            x_min = float(np.min(unique_prices))
            x_max = float(np.max(unique_prices))
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
        _style_axis(ax)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("Turn 1 IPO Par Price Distribution by Pick Order", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_net_worth_breakdown_by_turn(
    summary: NetWorthBreakdownSummary,
    output_path: str | Path,
) -> Path:
    """Write a stacked area chart of average player net-worth components."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    labels = list(NET_WORTH_COLORS)
    values = [summary.cash, summary.companies, summary.shares]

    if summary.turns.size:
        ax.stackplot(
            summary.turns,
            values,
            labels=labels,
            colors=[NET_WORTH_COLORS[label] for label in labels],
            alpha=0.92,
            linewidth=0.45,
            edgecolor="#ffffff",
        )
        if summary.turns.size == 1:
            ax.set_xlim(float(summary.turns[0]) - 0.5, float(summary.turns[0]) + 0.5)
        else:
            ax.set_xlim(float(summary.turns.min()), float(summary.turns.max()))
        ax.set_xticks(summary.turns)
    else:
        ax.text(
            0.5,
            0.5,
            "No INVEST rows",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
    ax.set_title(f"{summary.num_players} Players: Net Worth at Start of INVEST")
    ax.set_xlabel("Turn number")
    ax.set_ylabel("Average player value")
    ax.set_ylim(bottom=0.0)
    _style_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    if summary.observed_games.size:
        ax.text(
            1.0,
            -0.26,
            (
                "Averaged across players in games with a first INVEST row "
                f"for that turn; observed games range "
                f"{int(summary.observed_games.min())}-"
                f"{int(summary.observed_games.max())}."
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_net_worth_breakdown_charts(
    summaries: dict[int, NetWorthBreakdownSummary],
    output_dir: str | Path,
) -> list[Path]:
    """Write one start-of-INVEST net-worth breakdown chart per player count."""
    output = Path(output_dir)
    return [
        plot_net_worth_breakdown_by_turn(
            summary,
            output / f"net_worth_breakdown_by_turn_{num_players}p.png",
        )
        for num_players, summary in sorted(summaries.items())
    ]


def _rank_labels(ranks: np.ndarray) -> list[str]:
    labels: list[str] = []
    for rank in ranks:
        rank_int = int(rank)
        suffix = "th"
        if rank_int % 100 not in (11, 12, 13):
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(rank_int % 10, "th")
        labels.append(f"{rank_int}{suffix}")
    return labels


def plot_initial_auction_position_deltas(
    summaries: dict[int, InitialAuctionPositionSummary],
    output_path: str | Path,
) -> Path:
    """Write 3 panels of Turn 1 initial-offering auction deltas by FV rank."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.8 * len(counts), 4.1))
    axes_arr = _as_axes_array(axes)
    all_values = np.concatenate(
        [
            summary.mean_deltas[np.isfinite(summary.mean_deltas)]
            for summary in summaries.values()
            if np.any(np.isfinite(summary.mean_deltas))
        ]
    )
    y_max = max(1.0, float(np.max(all_values)) * 1.22) if all_values.size else 1.0

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        positions = np.arange(summary.position_ranks.shape[0])
        values = np.nan_to_num(summary.mean_deltas, nan=0.0)
        bars = ax.bar(
            positions,
            values,
            width=0.68,
            color="#c43c39",
            edgecolor="#ffffff",
            linewidth=0.8,
        )
        for bar, missing in zip(bars, summary.counts == 0):
            if missing:
                bar.set_hatch("//")
                bar.set_alpha(0.35)
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Initial offering FV rank")
        ax.set_ylabel("Auction price - face value" if ax is axes_arr[0] else "")
        ax.set_xticks(positions)
        ax.set_xticklabels(_rank_labels(summary.position_ranks))
        ax.set_ylim(0.0, y_max)
        _style_axis(ax)
        _annotate_bars(ax, bars, values, fmt="{:.2f}", zero_based=True)
        ax.text(
            1.0,
            -0.24,
            (
                f"Counted Turn 1 auctions only; "
                f"{int(summary.counts.sum())} position observations."
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#555555",
        )

    fig.suptitle("Turn 1 Auction Delta by Initial Offering Face-Value Rank", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_turn_one_auction_pool_premiums(
    summaries: dict[int, TurnOneAuctionPoolPremiumSummary],
    output_path: str | Path,
) -> Path:
    """Write a line chart of T1 premium totals by initial auction-pool FV."""
    counts = sorted(summaries)
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    all_x = np.concatenate(
        [
            summary.face_value_sums
            for summary in summaries.values()
            if summary.face_value_sums.size
        ]
    )
    all_y = np.concatenate(
        [
            summary.mean_premium_sums[np.isfinite(summary.mean_premium_sums)]
            for summary in summaries.values()
            if np.any(np.isfinite(summary.mean_premium_sums))
        ]
    )

    for color_index, num_players in enumerate(counts):
        summary = summaries[num_players]
        if summary.face_value_sums.size == 0:
            continue
        ax.plot(
            summary.face_value_sums,
            summary.mean_premium_sums,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=POSITION_COLORS[color_index % len(POSITION_COLORS)],
            label=f"{num_players} Players",
        )

    ax.set_title("Turn 1 Auction Premium by Initial Auction Pool Face Value")
    ax.set_xlabel("Initial auction pool total face value")
    ax.set_ylabel("Average total auction premium")
    if all_x.size:
        x_min = float(np.min(all_x))
        x_max = float(np.max(all_x))
        x_pad = max(1.0, (x_max - x_min) * 0.04)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if all_y.size:
        ax.set_ylim(0.0, max(1.0, float(np.max(all_y)) * 1.14))
    _style_axis(ax)
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        1.0,
        -0.18,
        (
            "Each point averages per-game Turn 1 auction premiums for companies "
            "that started in the setup auction pool."
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_turn_one_auction_company_presence_effects(
    summaries: dict[int, TurnOneAuctionCompanyPresenceSummary],
    output_path: str | Path,
) -> Path:
    """Write included/excluded red-company effects on T1 auction premium."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(2, len(counts), figsize=(5.2 * len(counts), 8.0))
    axes_arr = np.asarray(axes, dtype=object).reshape(2, len(counts))
    all_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for summary in summaries.values()
            for values in (summary.included_deltas, summary.excluded_deltas)
            if np.any(np.isfinite(values))
        ]
    )
    y_abs = max(1.0, float(np.max(np.abs(all_values))) * 1.18) if all_values.size else 1.0

    for col, num_players in enumerate(counts):
        summary = summaries[num_players]
        x = np.arange(len(summary.company_names))
        for row, (label, values, observed) in enumerate(
            (
                ("Included in Initial Pool", summary.included_deltas, summary.included_counts),
                ("Excluded from Initial Pool", summary.excluded_deltas, summary.excluded_counts),
            )
        ):
            ax = axes_arr[row, col]
            plot_values = np.nan_to_num(values, nan=0.0)
            colors = [
                POSITIVE_COLOR if value >= 0.0 else NEGATIVE_COLOR
                for value in plot_values
            ]
            bars = ax.bar(
                x,
                plot_values,
                width=0.72,
                color=colors,
                edgecolor="#ffffff",
                linewidth=0.7,
            )
            for bar, missing in zip(bars, observed == 0):
                if missing:
                    bar.set_hatch("//")
                    bar.set_alpha(0.35)
            ax.axhline(0.0, color=FAIR_LINE_COLOR, linewidth=0.9)
            ax.set_title(
                f"{num_players} Players"
                if row == 0
                else f"Baseline: {summary.baseline_mean_premium:.2f}",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel(f"{label}\nDelta from overall avg premium")
            ax.set_xticks(x)
            ax.set_xticklabels(summary.company_names, rotation=55, ha="right", fontsize=8)
            for tick in ax.get_xticklabels():
                tick.set_color(COMPANY_STAR_COLORS[1])
                tick.set_fontweight("bold")
            ax.set_ylim(-y_abs, y_abs)
            count_y = -y_abs * 0.91
            for company_x, count in zip(x, observed):
                ax.text(
                    float(company_x),
                    count_y,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#666666",
                )
            _style_axis(ax)

    fig.suptitle(
        "Turn 1 Auction Premium Effect by Red Company Initial Pool Presence",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Bars show conditional average per-game Turn 1 auction premium minus "
            "the overall average for that player count."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.12, hspace=0.34)
    return _save_figure(fig, Path(output_path))


def plot_turn_one_auction_deck_presence_effects(
    summaries: dict[int, TurnOneAuctionDeckPresenceSummary],
    output_path: str | Path,
) -> Path:
    """Write red-deck inclusion/exclusion effects on T1 auction premium."""
    counts = [
        num_players
        for num_players, summary in sorted(summaries.items())
        if summary.company_ids.size
        and np.any(summary.included_counts > 0)
        and np.any(summary.excluded_counts > 0)
    ]
    if not counts:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.text(
            0.5,
            0.5,
            "No player counts with red deck inclusion/exclusion variation",
            ha="center",
            va="center",
        )
        ax.axis("off")
        return _save_figure(fig, Path(output_path))

    fig, axes = plt.subplots(2, len(counts), figsize=(5.2 * len(counts), 8.0))
    axes_arr = np.asarray(axes, dtype=object).reshape(2, len(counts))
    all_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for num_players in counts
            for values in (
                summaries[num_players].included_deltas,
                summaries[num_players].excluded_deltas,
            )
            if np.any(np.isfinite(values))
        ]
    )
    y_abs = max(1.0, float(np.max(np.abs(all_values))) * 1.18) if all_values.size else 1.0

    for col, num_players in enumerate(counts):
        summary = summaries[num_players]
        x = np.arange(len(summary.company_names))
        for row, (label, values, observed) in enumerate(
            (
                ("Included in Deck", summary.included_deltas, summary.included_counts),
                ("Excluded from Deck", summary.excluded_deltas, summary.excluded_counts),
            )
        ):
            ax = axes_arr[row, col]
            plot_values = np.nan_to_num(values, nan=0.0)
            colors = [
                POSITIVE_COLOR if value >= 0.0 else NEGATIVE_COLOR
                for value in plot_values
            ]
            bars = ax.bar(
                x,
                plot_values,
                width=0.72,
                color=colors,
                edgecolor="#ffffff",
                linewidth=0.7,
            )
            for bar, missing in zip(bars, observed == 0):
                if missing:
                    bar.set_hatch("//")
                    bar.set_alpha(0.35)
            ax.axhline(0.0, color=FAIR_LINE_COLOR, linewidth=0.9)
            ax.set_title(
                f"{num_players} Players"
                if row == 0
                else f"Baseline: {summary.baseline_mean_premium:.2f}",
                fontsize=10,
            )
            if col == 0:
                ax.set_ylabel(f"{label}\nDelta from overall avg premium")
            ax.set_xticks(x)
            ax.set_xticklabels(summary.company_names, rotation=55, ha="right", fontsize=8)
            for tick in ax.get_xticklabels():
                tick.set_color(COMPANY_STAR_COLORS[1])
                tick.set_fontweight("bold")
            ax.set_ylim(-y_abs, y_abs)
            count_y = -y_abs * 0.91
            for company_x, count in zip(x, observed):
                ax.text(
                    float(company_x),
                    count_y,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#666666",
                )
            _style_axis(ax)

    fig.suptitle(
        "Turn 1 Auction Premium Effect by Red Company Deck Presence",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.025,
        (
            "MHE is omitted because the highest face-value red company is always in the deck; "
            "player counts with all red companies always included are omitted."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.12, hspace=0.34)
    return _save_figure(fig, Path(output_path))


def plot_final_rank_net_worth_breakdown(
    summaries: dict[int, RankedNetWorthBreakdownSummary],
    output_path: str | Path,
) -> Path:
    """Write a 3-panel stacked area chart of endgame components by rank."""
    counts = sorted(summaries)
    fig, axes = plt.subplots(1, len(counts), figsize=(4.6 * len(counts), 4.4))
    axes_arr = _as_axes_array(axes)
    labels = list(NET_WORTH_COLORS)
    y_max = max(
        (
            float(np.max(summary.cash + summary.companies + summary.shares))
            for summary in summaries.values()
            if summary.ranks.size
        ),
        default=1.0,
    ) * 1.12

    for ax, num_players in zip(axes_arr, counts):
        summary = summaries[num_players]
        values = [summary.cash, summary.companies, summary.shares]
        if summary.ranks.size:
            ax.stackplot(
                summary.ranks,
                values,
                labels=labels,
                colors=[NET_WORTH_COLORS[label] for label in labels],
                alpha=0.92,
                linewidth=0.45,
                edgecolor="#ffffff",
            )
            if summary.ranks.size == 1:
                ax.set_xlim(
                    float(summary.ranks[0]) - 0.5,
                    float(summary.ranks[0]) + 0.5,
                )
            else:
                ax.set_xlim(float(summary.ranks.min()), float(summary.ranks.max()))
            ax.set_xticks(summary.ranks)
            ax.set_xticklabels(_rank_labels(summary.ranks))
        else:
            ax.text(
                0.5,
                0.5,
                "No final states",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
        ax.set_title(f"{num_players} Players")
        ax.set_xlabel("Finish position")
        ax.set_ylabel("Average player value" if ax is axes_arr[0] else "")
        ax.set_ylim(0.0, y_max)
        _style_axis(ax)
        if summary.tie_games:
            ax.text(
                0.98,
                0.04,
                f"{summary.tie_games} tied games",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#555555",
            )

    axes_arr[-1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncols=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Endgame Net Worth Breakdown by Finish Position", fontsize=14)
    fig.tight_layout()
    return _save_figure(fig, Path(output_path))


def plot_early_max_price_endings(
    summaries: dict[int, EarlyMaxPriceEndSummary],
    output_path: str | Path,
) -> Path:
    """Write a bar chart plus table for max-price early game endings."""
    counts = sorted(summaries)
    labels = [f"{count}p" for count in counts]
    pct_values = np.asarray(
        [summaries[count].early_pct for count in counts],
        dtype=np.float64,
    )
    y_max = max(5.0, float(np.max(pct_values)) * 1.25) if pct_values.size else 5.0

    fig = plt.figure(figsize=(10.8, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.2], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])

    bars = ax.bar(
        np.arange(len(counts)),
        pct_values,
        width=0.62,
        color="#b45c3a",
        edgecolor="#ffffff",
        linewidth=0.8,
    )
    ax.set_title("Early Endings from $75 Share Price")
    ax.set_xlabel("Player count")
    ax.set_ylabel("Games ending early (%)")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, y_max)
    _style_axis(ax)
    for bar, count, pct in zip(bars, counts, pct_values):
        summary = summaries[count]
        ax.annotate(
            f"{pct:.1f}%\n{summary.early_games}/{summary.num_games}",
            xy=(bar.get_x() + bar.get_width() / 2.0, float(bar.get_height())),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#222222",
        )

    table_ax.axis("off")
    table_rows = []
    for count in counts:
        summary = summaries[count]
        avg_turn = (
            f"{summary.average_turn_count:.2f}"
            if np.isfinite(summary.average_turn_count)
            else "-"
        )
        table_rows.append(
            [
                f"{count}p",
                avg_turn,
                f"{summary.invest_early_games}",
                f"{summary.dividends_early_games}",
            ]
        )
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["Players", "Avg Turn", "INV", "DIV"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        if row == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold", color="#222222")
        else:
            cell.set_facecolor("#ffffff")
    table_ax.set_title("Early Games", pad=12)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.16)
    return _save_figure(fig, Path(output_path))


def generate_default_charts(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> list[Path]:
    """Generate the core opening-position charts for one strategy-data run."""
    dataset = StrategyDataset(run_dir)
    opening = dataset.opening_nn_value_summary(player_counts=player_counts)
    wins = dataset.final_net_worth_win_rate_summary(player_counts=player_counts)
    auction_final = dataset.auction_bid_delta_summary(player_counts=player_counts)
    auction_opening = dataset.opening_bid_delta_summary(player_counts=player_counts)
    auction_spread = dataset.auction_price_spread_summary(player_counts=player_counts)
    ipo_turn_share = dataset.corp_ipo_turn_share_summary(player_counts=player_counts)
    ipo_positional_picks = dataset.corp_ipo_positional_pick_summary(
        player_counts=player_counts,
        max_picks=4,
    )
    ipo_available_picks = dataset.corp_ipo_available_pick_summary(
        player_counts=player_counts,
        max_picks=4,
    )
    ipo_head_to_head = dataset.corp_ipo_head_to_head_summary(
        player_counts=player_counts,
        max_picks=3,
        max_turn=9,
    )
    floated_by_turn = dataset.floated_corps_by_turn_summary(player_counts=player_counts)
    par_price_by_turn = dataset.average_par_price_by_turn_summary(
        player_counts=player_counts
    )
    turn_one_ipo_par_prices = dataset.turn_one_ipo_par_price_summary(
        player_counts=player_counts,
        max_picks=4,
    )
    net_worth_breakdown = dataset.net_worth_breakdown_by_turn_summary(
        player_counts=player_counts
    )
    final_rank_breakdown = dataset.final_rank_net_worth_breakdown_summary(
        player_counts=player_counts
    )
    early_max_price = dataset.early_max_price_end_summary(
        player_counts=player_counts
    )
    auctioned_outcomes = dataset.auctioned_company_outcome_summary(
        player_counts=player_counts
    )
    turn_one_opening = dataset.turn_one_opening_summary(player_counts=player_counts)
    initial_auction_positions = dataset.initial_auction_position_summary(
        player_counts=player_counts
    )
    turn_one_pool_premiums = dataset.turn_one_auction_pool_premium_summary(
        player_counts=player_counts
    )
    turn_one_presence_effects = dataset.turn_one_auction_company_presence_summary(
        player_counts=player_counts
    )
    turn_one_deck_presence_effects = dataset.turn_one_auction_deck_presence_summary(
        player_counts=player_counts
    )
    output = Path(output_dir)
    written = [
        plot_opening_nn_values(opening, output / "opening_nn_values.png"),
        plot_actual_win_rates(wins, output / "actual_win_rates.png"),
        plot_win_rate_deltas(wins, output / "win_rate_deltas.png"),
    ]
    written.extend(
        plot_auction_bid_delta_charts(
            auction_final,
            output,
            filename_prefix="auction_final_price_delta",
            title_prefix="Final Auction Price",
            ylabel="Final auction price - face value",
        )
    )
    written.extend(
        plot_auction_bid_delta_charts(
            auction_opening,
            output,
            filename_prefix="opening_bid_delta",
            title_prefix="Opening Bid",
            ylabel="Opening bid - face value",
        )
    )
    written.extend(
        plot_auction_bid_delta_charts(
            auction_spread,
            output,
            filename_prefix="auction_price_spread",
            title="Final Auction Price Minus Opening Bid",
            ylabel="Final auction price - opening bid",
        )
    )
    written.extend(plot_corp_ipo_turn_share_charts(ipo_turn_share, output))
    written.append(
        plot_corp_ipo_positional_pick_bubbles(
            ipo_positional_picks,
            output / "corp_ipo_positional_pick_bubbles.png",
        )
    )
    written.append(
        plot_corp_ipo_available_pick_bubbles(
            ipo_available_picks,
            output / "corp_ipo_available_pick_bubbles.png",
        )
    )
    written.extend(plot_corp_ipo_head_to_head_heatmap_charts(ipo_head_to_head, output))
    written.append(
        plot_floated_corps_by_turn(
            floated_by_turn,
            output / "floated_corps_by_turn.png",
        )
    )
    written.append(
        plot_average_par_price_by_turn(
            par_price_by_turn,
            output / "average_par_price_by_turn.png",
        )
    )
    written.append(
        plot_turn_one_ipo_par_price_distribution(
            turn_one_ipo_par_prices,
            output / "turn1_ipo_par_price_distribution.png",
        )
    )
    written.extend(plot_net_worth_breakdown_charts(net_worth_breakdown, output))
    written.append(
        plot_final_rank_net_worth_breakdown(
            final_rank_breakdown,
            output / "final_rank_net_worth_breakdown.png",
        )
    )
    written.append(
        plot_early_max_price_endings(
            early_max_price,
            output / "early_max_price_endings.png",
        )
    )
    written.extend(plot_auctioned_company_outcome_charts(auctioned_outcomes, output))
    written.append(
        plot_turn_one_opening_summary(
            turn_one_opening,
            output / "turn1_auction_and_first_ipo.png",
        )
    )
    written.append(
        plot_initial_auction_position_deltas(
            initial_auction_positions,
            output / "turn1_initial_auction_position_deltas.png",
        )
    )
    written.append(
        plot_turn_one_auction_pool_premiums(
            turn_one_pool_premiums,
            output / "turn1_auction_pool_premiums.png",
        )
    )
    written.append(
        plot_turn_one_auction_company_presence_effects(
            turn_one_presence_effects,
            output / "turn1_auction_company_presence_effects.png",
        )
    )
    written.append(
        plot_turn_one_auction_deck_presence_effects(
            turn_one_deck_presence_effects,
            output / "turn1_auction_deck_presence_effects.png",
        )
    )
    return written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate charts for collected RSS strategy-data shards"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="strategy_data/run_001",
        help="Directory containing metadata.json and strategy shards",
    )
    parser.add_argument(
        "--out",
        default="/tmp/gamedata_charts",
        help="Output directory for chart images",
    )
    parser.add_argument(
        "--player-counts",
        help="Comma-separated player counts to include, e.g. 3,5",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    written = generate_default_charts(
        args.run_dir,
        args.out,
        player_counts=_parse_player_counts(args.player_counts),
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
