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
from matplotlib.patches import Patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.gamedata_analysis import (  # noqa: E402
    CompanyAuctionOutcomeSummary,
    CompanyBidDeltaSummary,
    CorpIpoTurnShareSummary,
    EarlyMaxPriceEndSummary,
    InitialAuctionPositionSummary,
    NetWorthBreakdownSummary,
    OpeningValueSummary,
    RankedNetWorthBreakdownSummary,
    StrategyDataset,
    TurnAverageSummary,
    TurnMeanSummary,
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
    floated_by_turn = dataset.floated_corps_by_turn_summary(player_counts=player_counts)
    par_price_by_turn = dataset.average_par_price_by_turn_summary(
        player_counts=player_counts
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
