#!/usr/bin/env python3
"""Helpers for analyzing collected strategy-data shards.

The strategy-data collector writes one ``metadata.json`` plus compressed
``strategy_<N>p_shard_<K>.npz`` files. This module keeps common analysis code
centralized and avoids loading whole runs when a question only needs a few
arrays.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.data import DecisionPhase, GameConstants
from core.state import (
    get_company_fields,
    get_corp_fields,
    get_layout,
    get_player_fields,
    get_turn_fields,
)
from entities.company import CompanyLocation
from entities.market import MARKET


SHARD_RE = re.compile(r"strategy_(?P<num_players>\d+)p_shard_(?P<idx>\d+)\.npz$")
BID_PHASE_ID = int(DecisionPhase.DPHASE_BID)
INVEST_PHASE_ID = int(DecisionPhase.DPHASE_INVEST)
DIVIDENDS_PHASE_ID = int(DecisionPhase.DPHASE_DIVIDENDS)
LOC_PLAYER_INT = int(CompanyLocation.LOC_PLAYER)


@dataclass(frozen=True)
class StrategyShard:
    """One strategy-data shard on disk."""

    path: Path
    num_players: int
    shard_index: int


@dataclass(frozen=True)
class OpeningValueSummary:
    """NN value estimates on each game's first decision row."""

    num_players: int
    num_games: int
    mean_values: np.ndarray
    std_values: np.ndarray
    min_values: np.ndarray
    max_values: np.ndarray


@dataclass(frozen=True)
class WinRateSummary:
    """Final-net-worth win rates by canonical player position."""

    num_players: int
    num_games: int
    fractional_win_rates: np.ndarray
    outright_win_rates: np.ndarray
    tied_first_rates: np.ndarray
    tie_games: int


@dataclass(frozen=True)
class CompanyBidDeltaSummary:
    """Auction acquisition price delta over face value by company."""

    num_players: int
    num_games: int
    company_names: list[str]
    company_stars: np.ndarray
    face_values: np.ndarray
    mean_deltas: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class CorpIpoTurnShareSummary:
    """Per-turn IPO attribution by corporation."""

    num_players: int
    num_games: int
    corp_names: list[str]
    turns: np.ndarray
    counts: np.ndarray
    percentages: np.ndarray


@dataclass(frozen=True)
class TurnAverageSummary:
    """Average per-game event counts by turn."""

    num_players: int
    num_games: int
    turns: np.ndarray
    mean_counts: np.ndarray
    total_counts: np.ndarray


@dataclass(frozen=True)
class TurnMeanSummary:
    """Mean event value by turn."""

    num_players: int
    num_games: int
    turns: np.ndarray
    mean_values: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class NetWorthBreakdownSummary:
    """Average per-player net-worth component values by turn."""

    num_players: int
    turns: np.ndarray
    cash: np.ndarray
    companies: np.ndarray
    shares: np.ndarray
    observed_games: np.ndarray


@dataclass(frozen=True)
class RankedNetWorthBreakdownSummary:
    """Average endgame net-worth components by finish position."""

    num_players: int
    num_games: int
    ranks: np.ndarray
    cash: np.ndarray
    companies: np.ndarray
    shares: np.ndarray
    tie_games: int


@dataclass(frozen=True)
class EarlyMaxPriceEndSummary:
    """Games ending early because an active corp reached the $75 market space."""

    num_players: int
    num_games: int
    early_games: int
    early_pct: float
    average_turn_count: float
    invest_early_games: int
    dividends_early_games: int


class StrategyDataset:
    """Reader for a strategy-data run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.metadata_path = self.run_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"missing metadata file: {self.metadata_path}")
        with self.metadata_path.open() as f:
            self.metadata: dict[str, object] = json.load(f)
        self._shards = _discover_shards(self.run_dir, self.metadata)

    @property
    def player_counts(self) -> list[int]:
        return [int(v) for v in self.metadata.get("player_counts", [])]

    @property
    def shards(self) -> tuple[StrategyShard, ...]:
        return self._shards

    def iter_shards(
        self,
        *,
        num_players: int | None = None,
    ) -> Iterator[StrategyShard]:
        for shard in self._shards:
            if num_players is None or shard.num_players == num_players:
                yield shard

    def count_games_by_player_count(self) -> dict[int, int]:
        """Return completed game counts using per-shard game metadata."""
        counts = {num_players: 0 for num_players in self.player_counts}
        for shard in self.iter_shards():
            with np.load(shard.path, allow_pickle=False) as data:
                counts[shard.num_players] = (
                    counts.get(shard.num_players, 0)
                    + int(data["game_num_examples"].shape[0])
                )
        return counts

    def collect_opening_rows(
        self,
        field: str,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Collect one array's opening rows, grouped by player count.

        The opening move is the first row for each game, identified by
        ``game_start_offsets``. ``field`` must be a per-decision array with a
        leading axis aligned to the move rows, such as ``nn_values``,
        ``mcts_policy_pct``, or ``legal_masks``.
        """
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}
        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                if field not in data:
                    raise KeyError(f"{field!r} is not present in {shard.path}")
                starts = data["game_start_offsets"]
                values = data[field]
                if starts.size == 0:
                    continue
                by_count.setdefault(shard.num_players, []).append(values[starts])

        return {
            num_players: np.concatenate(chunks, axis=0)
            for num_players, chunks in sorted(by_count.items())
            if chunks
        }

    def collect_game_rows(
        self,
        field: str,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Collect one per-game array, grouped by player count.

        ``field`` must have a leading axis aligned to games, such as
        ``final_net_worths`` or ``game_durations_sec``.
        """
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}
        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                if field not in data:
                    raise KeyError(f"{field!r} is not present in {shard.path}")
                by_count.setdefault(shard.num_players, []).append(data[field])

        return {
            num_players: np.concatenate(chunks, axis=0)
            for num_players, chunks in sorted(by_count.items())
            if chunks
        }

    def opening_nn_value_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, OpeningValueSummary]:
        """Summarize NN value estimates on each game's opening move.

        Values are returned in canonical player order.
        """
        opening_values = self.collect_opening_rows(
            "nn_values",
            player_counts=player_counts,
        )

        summaries: dict[int, OpeningValueSummary] = {}
        for num_players, raw_values in opening_values.items():
            values = raw_values.astype(np.float64, copy=False)
            summaries[num_players] = OpeningValueSummary(
                num_players=num_players,
                num_games=int(values.shape[0]),
                mean_values=values.mean(axis=0),
                std_values=values.std(axis=0),
                min_values=values.min(axis=0),
                max_values=values.max(axis=0),
            )
        return summaries

    def final_net_worth_win_rate_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, WinRateSummary]:
        """Summarize actual winners by final net worth.

        ``fractional_win_rates`` splits tied wins evenly among tied players.
        ``outright_win_rates`` only counts sole winners. ``tied_first_rates``
        counts every player who tied for first, so it can sum above 1.0.
        """
        final_net_worths = self.collect_game_rows(
            "final_net_worths",
            player_counts=player_counts,
        )
        summaries: dict[int, WinRateSummary] = {}
        for num_players, raw_values in final_net_worths.items():
            values = raw_values.astype(np.float64, copy=False)
            max_values = values.max(axis=1, keepdims=True)
            winners = values == max_values
            winners_per_game = winners.sum(axis=1)
            fractional = (winners / winners_per_game[:, None]).sum(axis=0)
            outright = (winners & (winners_per_game[:, None] == 1)).sum(axis=0)
            tied_first = winners.sum(axis=0)
            num_games = int(values.shape[0])
            summaries[num_players] = WinRateSummary(
                num_players=num_players,
                num_games=num_games,
                fractional_win_rates=fractional / num_games,
                outright_win_rates=outright.astype(np.float64) / num_games,
                tied_first_rates=tied_first.astype(np.float64) / num_games,
                tie_games=int(np.count_nonzero(winners_per_game > 1)),
            )
        return summaries

    def auction_bid_delta_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, CompanyBidDeltaSummary]:
        """Summarize auction winning-price deltas over company face value.

        Deltas are ``auction_events.price - company.face_value``. Means are
        averaged over observed auction wins for each company; companies with no
        auction event in a player-count subset have ``NaN`` means and count 0.
        """
        company_names = [str(v) for v in self.metadata["company_names"]]  # type: ignore[index]
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        company_stars = np.asarray(company_static["stars"], dtype=np.int16)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        num_companies = len(company_names)
        sums = {
            count: np.zeros(num_companies, dtype=np.float64)
            for count in requested
        }
        counts = {
            count: np.zeros(num_companies, dtype=np.int64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["auction_events"]
                if events.shape[0] == 0:
                    continue
                company_ids = events[:, 3].astype(np.int64)
                prices = events[:, 5].astype(np.float64)
                deltas = prices - face_values[company_ids]
                np.add.at(sums[shard.num_players], company_ids, deltas)
                np.add.at(counts[shard.num_players], company_ids, 1)

        summaries: dict[int, CompanyBidDeltaSummary] = {}
        for num_players in sorted(requested):
            observed = counts[num_players]
            mean_deltas = np.full(num_companies, np.nan, dtype=np.float64)
            np.divide(
                sums[num_players],
                observed,
                out=mean_deltas,
                where=observed > 0,
            )
            summaries[num_players] = CompanyBidDeltaSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                company_names=company_names,
                company_stars=company_stars.copy(),
                face_values=face_values.copy(),
                mean_deltas=mean_deltas,
                counts=observed.copy(),
            )
        return summaries

    def opening_bid_delta_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, CompanyBidDeltaSummary]:
        """Summarize the first BID action's delta over face value by company.

        Opening bids are derived from the first recorded BID row in each
        auction group. If ``auction_high_bidders`` is -1 on that row, the
        selected BID action is the opening bid and ``action_amount`` is its
        face-value offset. If the high bidder is already set, the opening bid
        was forced by the driver before the next recorded decision; in that
        case ``auction_prices - face_value`` recovers the opening delta.
        """
        company_names = [str(v) for v in self.metadata["company_names"]]  # type: ignore[index]
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        company_stars = np.asarray(company_static["stars"], dtype=np.int16)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        num_companies = len(company_names)
        sums = {
            count: np.zeros(num_companies, dtype=np.float64)
            for count in requested
        }
        counts = {
            count: np.zeros(num_companies, dtype=np.int64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                company_ids, deltas, _ = _opening_bid_delta_rows(
                    data, face_values, num_companies
                )
                if company_ids.size == 0:
                    continue
                np.add.at(sums[shard.num_players], company_ids, deltas)
                np.add.at(counts[shard.num_players], company_ids, 1)

        summaries: dict[int, CompanyBidDeltaSummary] = {}
        for num_players in sorted(requested):
            observed = counts[num_players]
            mean_deltas = np.full(num_companies, np.nan, dtype=np.float64)
            np.divide(
                sums[num_players],
                observed,
                out=mean_deltas,
                where=observed > 0,
            )
            summaries[num_players] = CompanyBidDeltaSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                company_names=company_names,
                company_stars=company_stars.copy(),
                face_values=face_values.copy(),
                mean_deltas=mean_deltas,
                counts=observed.copy(),
            )
        return summaries

    def auction_price_spread_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, CompanyBidDeltaSummary]:
        """Summarize final auction price minus opening bid by company.

        Each final auction event is paired with that auction's opening bid
        using ``(game_id, company_id)``. The per-company mean is then computed
        over paired auction spreads.
        """
        company_names = [str(v) for v in self.metadata["company_names"]]  # type: ignore[index]
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        company_stars = np.asarray(company_static["stars"], dtype=np.int16)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        num_companies = len(company_names)
        sums = {
            count: np.zeros(num_companies, dtype=np.float64)
            for count in requested
        }
        counts = {
            count: np.zeros(num_companies, dtype=np.int64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                opening_company_ids, opening_deltas, opening_keys = (
                    _opening_bid_delta_rows(data, face_values, num_companies)
                )
                if opening_keys.size == 0:
                    continue
                opening_by_key = {
                    int(key): float(delta)
                    for key, delta in zip(opening_keys, opening_deltas)
                }

                events = data["auction_events"]
                if events.shape[0] == 0:
                    continue
                company_ids = events[:, 3].astype(np.int64)
                final_deltas = (
                    events[:, 5].astype(np.float64) - face_values[company_ids]
                )
                event_keys = (
                    events[:, 0].astype(np.int64) * num_companies + company_ids
                )
                spreads: list[float] = []
                paired_company_ids: list[int] = []
                for company_id, final_delta, key in zip(
                    company_ids, final_deltas, event_keys
                ):
                    opening_delta = opening_by_key.get(int(key))
                    if opening_delta is None:
                        raise ValueError(
                            f"missing opening bid for game/company key {int(key)} "
                            f"in {shard.path}"
                        )
                    paired_company_ids.append(int(company_id))
                    spreads.append(float(final_delta) - opening_delta)
                np.add.at(
                    sums[shard.num_players],
                    np.asarray(paired_company_ids, dtype=np.int64),
                    np.asarray(spreads, dtype=np.float64),
                )
                np.add.at(
                    counts[shard.num_players],
                    np.asarray(paired_company_ids, dtype=np.int64),
                    1,
                )

        summaries: dict[int, CompanyBidDeltaSummary] = {}
        for num_players in sorted(requested):
            observed = counts[num_players]
            mean_deltas = np.full(num_companies, np.nan, dtype=np.float64)
            np.divide(
                sums[num_players],
                observed,
                out=mean_deltas,
                where=observed > 0,
            )
            summaries[num_players] = CompanyBidDeltaSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                company_names=company_names,
                company_stars=company_stars.copy(),
                face_values=face_values.copy(),
                mean_deltas=mean_deltas,
                counts=observed.copy(),
            )
        return summaries

    def corp_ipo_turn_share_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, CorpIpoTurnShareSummary]:
        """Summarize each corp's percentage of IPOs by game turn."""
        corp_names = [str(v) for v in self.metadata["corp_names"]]  # type: ignore[index]
        num_corps = len(corp_names)
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        turns_by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}
        corps_by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue
                turns_by_count[shard.num_players].append(events[:, 2].astype(np.int64))
                corps_by_count[shard.num_players].append(events[:, 4].astype(np.int64))

        summaries: dict[int, CorpIpoTurnShareSummary] = {}
        for num_players in sorted(requested):
            turn_chunks = turns_by_count[num_players]
            if not turn_chunks:
                summaries[num_players] = CorpIpoTurnShareSummary(
                    num_players=num_players,
                    num_games=game_counts[num_players],
                    corp_names=corp_names,
                    turns=np.empty(0, dtype=np.int64),
                    counts=np.empty((0, num_corps), dtype=np.int64),
                    percentages=np.empty((0, num_corps), dtype=np.float64),
                )
                continue

            event_turns = np.concatenate(turn_chunks)
            event_corps = np.concatenate(corps_by_count[num_players])
            turns = np.arange(int(event_turns.min()), int(event_turns.max()) + 1)
            counts = np.zeros((turns.shape[0], num_corps), dtype=np.int64)
            np.add.at(counts, (event_turns - turns[0], event_corps), 1)
            totals = counts.sum(axis=1, keepdims=True)
            percentages = np.zeros_like(counts, dtype=np.float64)
            np.divide(
                counts * 100.0,
                totals,
                out=percentages,
                where=totals > 0,
            )
            observed = totals[:, 0] > 0
            summaries[num_players] = CorpIpoTurnShareSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                corp_names=corp_names,
                turns=turns[observed],
                counts=counts[observed],
                percentages=percentages[observed],
            )
        return summaries

    def floated_corps_by_turn_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, TurnAverageSummary]:
        """Average number of corp IPOs/floats per game by turn."""
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        turns_by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue
                turns_by_count[shard.num_players].append(events[:, 2].astype(np.int64))

        summaries: dict[int, TurnAverageSummary] = {}
        for num_players in sorted(requested):
            turn_chunks = turns_by_count[num_players]
            if not turn_chunks:
                summaries[num_players] = TurnAverageSummary(
                    num_players=num_players,
                    num_games=game_counts[num_players],
                    turns=np.empty(0, dtype=np.int64),
                    mean_counts=np.empty(0, dtype=np.float64),
                    total_counts=np.empty(0, dtype=np.int64),
                )
                continue

            event_turns = np.concatenate(turn_chunks)
            turns = np.arange(int(event_turns.min()), int(event_turns.max()) + 1)
            total_counts = np.bincount(
                event_turns - turns[0],
                minlength=turns.shape[0],
            ).astype(np.int64)
            mean_counts = total_counts.astype(np.float64) / max(
                game_counts[num_players], 1
            )
            summaries[num_players] = TurnAverageSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                turns=turns,
                mean_counts=mean_counts,
                total_counts=total_counts,
            )
        return summaries

    def average_par_price_by_turn_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, TurnMeanSummary]:
        """Average IPO par price by game turn."""
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        turns_by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}
        prices_by_count: dict[int, list[np.ndarray]] = {count: [] for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue
                turns_by_count[shard.num_players].append(events[:, 2].astype(np.int64))
                prices_by_count[shard.num_players].append(events[:, 7].astype(np.float64))

        summaries: dict[int, TurnMeanSummary] = {}
        for num_players in sorted(requested):
            turn_chunks = turns_by_count[num_players]
            if not turn_chunks:
                summaries[num_players] = TurnMeanSummary(
                    num_players=num_players,
                    num_games=game_counts[num_players],
                    turns=np.empty(0, dtype=np.int64),
                    mean_values=np.empty(0, dtype=np.float64),
                    counts=np.empty(0, dtype=np.int64),
                )
                continue

            event_turns = np.concatenate(turn_chunks)
            prices = np.concatenate(prices_by_count[num_players])
            turns = np.arange(int(event_turns.min()), int(event_turns.max()) + 1)
            counts = np.bincount(
                event_turns - turns[0],
                minlength=turns.shape[0],
            ).astype(np.int64)
            sums = np.bincount(
                event_turns - turns[0],
                weights=prices,
                minlength=turns.shape[0],
            ).astype(np.float64)
            mean_values = np.full(turns.shape[0], np.nan, dtype=np.float64)
            np.divide(sums, counts, out=mean_values, where=counts > 0)
            observed = counts > 0
            summaries[num_players] = TurnMeanSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                turns=turns[observed],
                mean_values=mean_values[observed],
                counts=counts[observed],
            )
        return summaries

    def net_worth_breakdown_by_turn_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, NetWorthBreakdownSummary]:
        """Average per-player net-worth components at each turn's first INVEST row."""
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        cash_sums = {count: {} for count in requested}
        company_sums = {count: {} for count in requested}
        share_sums = {count: {} for count in requested}
        player_counts_by_turn = {count: {} for count in requested}
        game_counts_by_turn = {count: {} for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                rows = _first_decision_rows_by_game_turn(
                    data,
                    phase_id=INVEST_PHASE_ID,
                )
                if rows.size == 0:
                    continue

                turns = data["turn_numbers"][rows].astype(np.int64)
                cash_by_player = data["player_cash"][rows].astype(np.float64)
                share_by_player = (
                    data["player_shares"][rows].astype(np.float64)
                    * data["corp_prices"][rows, None, :].astype(np.float64)
                ).sum(axis=2)
                company_by_player = np.zeros(
                    (rows.shape[0], num_players),
                    dtype=np.float64,
                )
                owners = data["company_owners"][rows]
                locations = data["company_locations"][rows]
                for player_id in range(num_players):
                    owned = (locations == LOC_PLAYER_INT) & (owners == player_id)
                    company_by_player[:, player_id] = (
                        owned.astype(np.float64) * face_values[None, :]
                    ).sum(axis=1)

                for turn in np.unique(turns):
                    mask = turns == turn
                    turn_int = int(turn)
                    observed_games = int(np.count_nonzero(mask))
                    observed_players = observed_games * num_players
                    cash_sums[num_players][turn_int] = (
                        cash_sums[num_players].get(turn_int, 0.0)
                        + float(cash_by_player[mask].sum())
                    )
                    company_sums[num_players][turn_int] = (
                        company_sums[num_players].get(turn_int, 0.0)
                        + float(company_by_player[mask].sum())
                    )
                    share_sums[num_players][turn_int] = (
                        share_sums[num_players].get(turn_int, 0.0)
                        + float(share_by_player[mask].sum())
                    )
                    player_counts_by_turn[num_players][turn_int] = (
                        player_counts_by_turn[num_players].get(turn_int, 0)
                        + observed_players
                    )
                    game_counts_by_turn[num_players][turn_int] = (
                        game_counts_by_turn[num_players].get(turn_int, 0)
                        + observed_games
                    )

        summaries: dict[int, NetWorthBreakdownSummary] = {}
        for num_players in sorted(requested):
            turns = np.asarray(
                sorted(player_counts_by_turn[num_players]),
                dtype=np.int64,
            )
            denominators = np.asarray(
                [player_counts_by_turn[num_players][int(turn)] for turn in turns],
                dtype=np.float64,
            )
            observed_games = np.asarray(
                [game_counts_by_turn[num_players][int(turn)] for turn in turns],
                dtype=np.int64,
            )
            if turns.size == 0:
                empty = np.empty(0, dtype=np.float64)
                summaries[num_players] = NetWorthBreakdownSummary(
                    num_players=num_players,
                    turns=turns,
                    cash=empty,
                    companies=empty,
                    shares=empty,
                    observed_games=observed_games,
                )
                continue
            summaries[num_players] = NetWorthBreakdownSummary(
                num_players=num_players,
                turns=turns,
                cash=np.asarray(
                    [cash_sums[num_players][int(turn)] for turn in turns],
                    dtype=np.float64,
                ) / denominators,
                companies=np.asarray(
                    [company_sums[num_players][int(turn)] for turn in turns],
                    dtype=np.float64,
                ) / denominators,
                shares=np.asarray(
                    [share_sums[num_players][int(turn)] for turn in turns],
                    dtype=np.float64,
                ) / denominators,
                observed_games=observed_games,
            )
        return summaries

    def final_rank_net_worth_breakdown_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, RankedNetWorthBreakdownSummary]:
        """Average terminal net-worth components by ordinal finish position."""
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        market_prices = _market_prices()
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        cash_sums = {
            count: np.zeros(count, dtype=np.float64)
            for count in requested
        }
        company_sums = {
            count: np.zeros(count, dtype=np.float64)
            for count in requested
        }
        share_sums = {
            count: np.zeros(count, dtype=np.float64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}
        tie_games = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                final_states = data["final_states"]
                if final_states.shape[0] == 0:
                    continue
                cash, companies, shares = _net_worth_components_from_states(
                    final_states,
                    num_players=num_players,
                    face_values=face_values,
                    market_prices=market_prices,
                )
                final_net_worths = data["final_net_worths"].astype(
                    np.float64,
                    copy=False,
                )
                order = np.argsort(-final_net_worths, axis=1, kind="stable")
                ranked_cash = np.take_along_axis(cash, order, axis=1)
                ranked_companies = np.take_along_axis(companies, order, axis=1)
                ranked_shares = np.take_along_axis(shares, order, axis=1)
                cash_sums[num_players] += ranked_cash.sum(axis=0)
                company_sums[num_players] += ranked_companies.sum(axis=0)
                share_sums[num_players] += ranked_shares.sum(axis=0)
                game_counts[num_players] += int(final_states.shape[0])
                tie_games[num_players] += int(
                    np.count_nonzero(
                        np.any(
                            np.diff(np.sort(final_net_worths, axis=1), axis=1)
                            == 0.0,
                            axis=1,
                        )
                    )
                )

        summaries: dict[int, RankedNetWorthBreakdownSummary] = {}
        for num_players in sorted(requested):
            denominator = max(game_counts[num_players], 1)
            summaries[num_players] = RankedNetWorthBreakdownSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                ranks=np.arange(1, num_players + 1, dtype=np.int64),
                cash=cash_sums[num_players] / denominator,
                companies=company_sums[num_players] / denominator,
                shares=share_sums[num_players] / denominator,
                tie_games=tie_games[num_players],
            )
        return summaries

    def early_max_price_end_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, EarlyMaxPriceEndSummary]:
        """Summarize games that ended early due to an active corp reaching $75.

        INVEST endings are counted when the final state has an active corp at
        the top market index and the last recorded decision was INVEST.
        DIVIDENDS endings are counted under the same max-price condition when
        the last recorded decision was DIVIDENDS, excluding games whose final
        state already has the end-card flag set.
        """
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        early_counts = {count: 0 for count in requested}
        turn_sums = {count: 0.0 for count in requested}
        invest_counts = {count: 0 for count in requested}
        dividends_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                game_lengths = data["game_num_examples"].astype(np.int64)
                valid_games = game_lengths > 0
                if not np.any(valid_games):
                    continue

                starts = data["game_start_offsets"].astype(np.int64)
                last_rows = starts[valid_games] + game_lengths[valid_games] - 1
                final_states = data["final_states"][valid_games]
                last_decision_phases = data["phase_ids"][last_rows].astype(np.int64)
                active_max = _states_have_active_max_price(final_states, num_players)
                end_card_flipped = _states_end_card_flipped(final_states, num_players)
                turn_numbers = _states_turn_numbers(final_states, num_players)

                invest_early = active_max & (last_decision_phases == INVEST_PHASE_ID)
                dividends_early = (
                    active_max
                    & (last_decision_phases == DIVIDENDS_PHASE_ID)
                    & ~end_card_flipped
                )
                early = invest_early | dividends_early

                game_counts[num_players] += int(valid_games.sum())
                early_counts[num_players] += int(early.sum())
                turn_sums[num_players] += float(turn_numbers[early].sum())
                invest_counts[num_players] += int(invest_early.sum())
                dividends_counts[num_players] += int(dividends_early.sum())

        summaries: dict[int, EarlyMaxPriceEndSummary] = {}
        for num_players in sorted(requested):
            total_games = game_counts[num_players]
            early_games = early_counts[num_players]
            summaries[num_players] = EarlyMaxPriceEndSummary(
                num_players=num_players,
                num_games=total_games,
                early_games=early_games,
                early_pct=(100.0 * early_games / total_games) if total_games else 0.0,
                average_turn_count=(
                    turn_sums[num_players] / early_games
                    if early_games
                    else float("nan")
                ),
                invest_early_games=invest_counts[num_players],
                dividends_early_games=dividends_counts[num_players],
            )
        return summaries


def _opening_bid_delta_rows(
    data: object,
    face_values: np.ndarray,
    num_companies: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return opening-bid company ids, deltas over face, and pair keys."""
    phase_ids = data["phase_ids"]
    active_companies = data["active_companies"]
    bid_rows = (phase_ids == BID_PHASE_ID) & (active_companies >= 0)
    if not np.any(bid_rows):
        empty_i64 = np.empty(0, dtype=np.int64)
        empty_f64 = np.empty(0, dtype=np.float64)
        return empty_i64, empty_f64, empty_i64

    prev_bid_same_company = np.zeros_like(bid_rows, dtype=bool)
    prev_bid_same_company[1:] = (
        bid_rows[:-1]
        & bid_rows[1:]
        & (data["game_ids"][:-1] == data["game_ids"][1:])
        & (active_companies[:-1] == active_companies[1:])
    )
    group_starts = bid_rows & ~prev_bid_same_company
    company_ids = active_companies[group_starts].astype(np.int64)
    high_bidders = data["auction_high_bidders"][group_starts]
    explicit_first_bid = high_bidders < 0
    deltas = np.empty(company_ids.shape[0], dtype=np.float64)
    deltas[explicit_first_bid] = data["action_amounts"][group_starts][
        explicit_first_bid
    ].astype(np.float64)
    forced_first_bid = ~explicit_first_bid
    deltas[forced_first_bid] = (
        data["auction_prices"][group_starts][forced_first_bid].astype(np.float64)
        - face_values[company_ids[forced_first_bid]]
    )
    keys = data["game_ids"][group_starts].astype(np.int64) * num_companies + company_ids
    if np.unique(keys).size != keys.size:
        raise ValueError("duplicate opening bid keys in shard")
    return company_ids, deltas, keys


def _first_decision_rows_by_game_turn(
    data: object,
    *,
    phase_id: int,
) -> np.ndarray:
    """Return the first row for each game/turn within one decision phase."""
    phase_rows = data["phase_ids"] == phase_id
    if not np.any(phase_rows):
        return np.empty(0, dtype=np.int64)

    indices = np.flatnonzero(phase_rows)
    keys = (
        data["game_ids"][indices].astype(np.int64)
        * (int(data["turn_numbers"].max()) + 1)
        + data["turn_numbers"][indices].astype(np.int64)
    )
    _, first_positions = np.unique(keys, return_index=True)
    return indices[np.sort(first_positions)]


def _market_prices() -> np.ndarray:
    """Return share prices indexed by market index."""
    return np.asarray(
        [
            MARKET.get_price_at_index(index)
            for index in range(int(GameConstants.NUM_MARKET_SPACES))
        ],
        dtype=np.float64,
    )


def _net_worth_components_from_states(
    states: np.ndarray,
    *,
    num_players: int,
    face_values: np.ndarray,
    market_prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``cash, company_face_value, share_value`` for final state rows."""
    layout = get_layout(num_players)
    player_fields = get_player_fields()
    corp_fields = get_corp_fields()
    company_fields = get_company_fields()
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)

    player_bases = (
        layout.players_offset
        + np.arange(num_players, dtype=np.int64) * layout.player_size
    )
    cash_slots = player_bases + player_fields.cash
    cash = states[:, cash_slots].astype(np.float64)

    share_slots = (
        player_bases[:, None]
        + player_fields.owned_shares
        + np.arange(num_corps, dtype=np.int64)[None, :]
    )
    player_shares = states[:, share_slots].astype(np.float64)

    corp_bases = (
        layout.corps_offset
        + np.arange(num_corps, dtype=np.int64) * layout.corp_size
    )
    active = states[:, corp_bases + corp_fields.active] != 0
    price_indices = states[:, corp_bases + corp_fields.price_index].astype(np.int64)
    share_prices = market_prices[price_indices]
    share_prices = np.where(active, share_prices, 0.0)
    shares = (player_shares * share_prices[:, None, :]).sum(axis=2)

    company_ids = np.arange(num_companies, dtype=np.int64)
    location_slots = (
        layout.companies_offset
        + company_fields.locations
        + company_ids
    )
    owner_slots = (
        layout.companies_offset
        + company_fields.owner_ids
        + company_ids
    )
    locations = states[:, location_slots]
    owners = states[:, owner_slots]
    companies = np.zeros((states.shape[0], num_players), dtype=np.float64)
    for player_id in range(num_players):
        owned = (locations == LOC_PLAYER_INT) & (owners == player_id)
        companies[:, player_id] = (
            owned.astype(np.float64) * face_values[None, :]
        ).sum(axis=1)

    return cash, companies, shares


def _states_have_active_max_price(
    states: np.ndarray,
    num_players: int,
) -> np.ndarray:
    """Return a mask for final states with any active corp at the top price."""
    layout = get_layout(num_players)
    corp_fields = get_corp_fields()
    num_corps = int(GameConstants.NUM_CORPS)
    max_price_index = int(GameConstants.NUM_MARKET_SPACES) - 1
    corp_bases = (
        layout.corps_offset
        + np.arange(num_corps, dtype=np.int64) * layout.corp_size
    )
    active = states[:, corp_bases + corp_fields.active] != 0
    price_indices = states[:, corp_bases + corp_fields.price_index]
    return np.any(active & (price_indices == max_price_index), axis=1)


def _states_end_card_flipped(
    states: np.ndarray,
    num_players: int,
) -> np.ndarray:
    """Return final-state end-card flags."""
    layout = get_layout(num_players)
    turn_fields = get_turn_fields()
    return states[:, layout.turn_offset + turn_fields.end_card_flipped] != 0


def _states_turn_numbers(
    states: np.ndarray,
    num_players: int,
) -> np.ndarray:
    """Return final-state turn numbers."""
    layout = get_layout(num_players)
    turn_fields = get_turn_fields()
    return states[:, layout.turn_offset + turn_fields.turn_number].astype(np.float64)


def _discover_shards(
    run_dir: Path,
    metadata: dict[str, object],
) -> tuple[StrategyShard, ...]:
    metadata_files = metadata.get("files")
    paths = (
        [run_dir / str(name) for name in metadata_files]
        if isinstance(metadata_files, list)
        else list(run_dir.glob("strategy_*p_shard_*.npz"))
    )
    shards: list[StrategyShard] = []
    for path in paths:
        match = SHARD_RE.fullmatch(path.name)
        if match is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"metadata references missing shard: {path}")
        shards.append(
            StrategyShard(
                path=path,
                num_players=int(match.group("num_players")),
                shard_index=int(match.group("idx")),
            )
        )
    shards.sort(key=lambda shard: (shard.num_players, shard.shard_index))
    return tuple(shards)


def opening_nn_value_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, OpeningValueSummary]:
    """Convenience wrapper for ``StrategyDataset(...).opening_nn_value_summary``."""
    return StrategyDataset(run_dir).opening_nn_value_summary(player_counts=player_counts)


def final_net_worth_win_rate_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, WinRateSummary]:
    """Convenience wrapper for final-net-worth win-rate summaries."""
    return StrategyDataset(run_dir).final_net_worth_win_rate_summary(
        player_counts=player_counts
    )


def auction_bid_delta_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, CompanyBidDeltaSummary]:
    """Convenience wrapper for auction bid-delta summaries."""
    return StrategyDataset(run_dir).auction_bid_delta_summary(
        player_counts=player_counts
    )


def opening_bid_delta_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, CompanyBidDeltaSummary]:
    """Convenience wrapper for opening bid-delta summaries."""
    return StrategyDataset(run_dir).opening_bid_delta_summary(
        player_counts=player_counts
    )


def auction_price_spread_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, CompanyBidDeltaSummary]:
    """Convenience wrapper for final-minus-opening auction price spreads."""
    return StrategyDataset(run_dir).auction_price_spread_summary(
        player_counts=player_counts
    )


def corp_ipo_turn_share_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, CorpIpoTurnShareSummary]:
    """Convenience wrapper for per-turn corp IPO percentage summaries."""
    return StrategyDataset(run_dir).corp_ipo_turn_share_summary(
        player_counts=player_counts
    )


def floated_corps_by_turn_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, TurnAverageSummary]:
    """Convenience wrapper for average floated corps by turn."""
    return StrategyDataset(run_dir).floated_corps_by_turn_summary(
        player_counts=player_counts
    )


def average_par_price_by_turn_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, TurnMeanSummary]:
    """Convenience wrapper for average IPO par price by turn."""
    return StrategyDataset(run_dir).average_par_price_by_turn_summary(
        player_counts=player_counts
    )


def net_worth_breakdown_by_turn_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, NetWorthBreakdownSummary]:
    """Convenience wrapper for start-of-INVEST net-worth component summaries."""
    return StrategyDataset(run_dir).net_worth_breakdown_by_turn_summary(
        player_counts=player_counts
    )


def final_rank_net_worth_breakdown_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, RankedNetWorthBreakdownSummary]:
    """Convenience wrapper for endgame net-worth components by finish rank."""
    return StrategyDataset(run_dir).final_rank_net_worth_breakdown_summary(
        player_counts=player_counts
    )


def early_max_price_end_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, EarlyMaxPriceEndSummary]:
    """Convenience wrapper for early max-share-price game endings."""
    return StrategyDataset(run_dir).early_max_price_end_summary(
        player_counts=player_counts
    )


def format_opening_value_summary(
    summaries: dict[int, OpeningValueSummary],
) -> str:
    """Render opening-value summaries as a compact text table."""
    max_players = max(
        (summary.mean_values.shape[0] for summary in summaries.values()),
        default=0,
    )
    lines = [
        "players games " + " ".join(f"P{i}" for i in range(1, max_players + 1)),
    ]
    for num_players, summary in sorted(summaries.items()):
        values = " ".join(
            f"{float(value): .6f}" for value in summary.mean_values
        )
        lines.append(f"{num_players}p {summary.num_games:5d} {values}")
    return "\n".join(lines)


def format_opening_value_vs_win_rate(
    opening: dict[int, OpeningValueSummary],
    wins: dict[int, WinRateSummary],
) -> str:
    """Render opening NN values next to actual final-net-worth win rates."""
    lines = [
        "players pos opening_nn_value win_pct fair_pct win_minus_fair_pct",
    ]
    for num_players in sorted(set(opening) & set(wins)):
        value_summary = opening[num_players]
        win_summary = wins[num_players]
        fair_rate = 1.0 / num_players
        for player_idx in range(num_players):
            win_rate = float(win_summary.fractional_win_rates[player_idx])
            lines.append(
                f"{num_players}p P{player_idx + 1} "
                f"{float(value_summary.mean_values[player_idx]): .6f} "
                f"{100.0 * win_rate:6.2f} "
                f"{100.0 * fair_rate:6.2f} "
                f"{100.0 * (win_rate - fair_rate):+7.2f}"
            )
        if win_summary.tie_games:
            lines.append(
                f"{num_players}p ties {win_summary.tie_games}/{win_summary.num_games}"
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze collected RSS strategy-data shards"
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="strategy_data/run_001",
        help="Directory containing metadata.json and strategy shards",
    )
    parser.add_argument(
        "--player-counts",
        help="Comma-separated player counts to include, e.g. 3,5",
    )
    return parser


def _parse_player_counts(value: str | None) -> list[int] | None:
    if value is None:
        return None
    counts = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not counts:
        raise ValueError("--player-counts must contain at least one integer")
    return counts


def main() -> None:
    args = _build_parser().parse_args()
    dataset = StrategyDataset(args.run_dir)
    counts = _parse_player_counts(args.player_counts)
    opening = dataset.opening_nn_value_summary(player_counts=counts)
    wins = dataset.final_net_worth_win_rate_summary(player_counts=counts)
    print(format_opening_value_vs_win_rate(opening, wins))


if __name__ == "__main__":
    main()
