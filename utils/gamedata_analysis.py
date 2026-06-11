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

from core.data import DecisionPhase, GameConstants, GamePhases
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
IPO_PHASE_ID = int(DecisionPhase.DPHASE_IPO)
PAR_PHASE_ID = int(DecisionPhase.DPHASE_PAR)
ENGINE_IPO_PHASE_ID = int(GamePhases.PHASE_IPO)
ENGINE_PAR_PHASE_ID = int(GamePhases.PHASE_PAR)
LOC_PLAYER_INT = int(CompanyLocation.LOC_PLAYER)
LOC_CORP_INT = int(CompanyLocation.LOC_CORP)
LOC_CORP_ACQ_INT = int(CompanyLocation.LOC_CORP_ACQ)
LOC_REMOVED_INT = int(CompanyLocation.LOC_REMOVED)
AUCTION_OUTCOME_ACQUIRED = 0
AUCTION_OUTCOME_IPO = 1
AUCTION_OUTCOME_CLOSED = 2
AUCTION_OUTCOME_HELD = 3
AUCTION_OUTCOME_NAMES = (
    "Acquired by Corp",
    "IPO Seed",
    "Closed",
    "Held to End",
)


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
class CorpIpoPositionPickSummary:
    """Per-turn corp pick percentages by ordinal IPO position."""

    num_players: int
    num_games: int
    corp_names: list[str]
    turns: np.ndarray
    pick_positions: np.ndarray
    counts: np.ndarray
    totals: np.ndarray
    percentages: np.ndarray


@dataclass(frozen=True)
class CorpIpoAvailablePickSummary:
    """Per-turn IPO pick probability conditional on corp inactivity."""

    num_players: int
    num_games: int
    corp_names: list[str]
    turns: np.ndarray
    pick_positions: np.ndarray
    counts: np.ndarray
    available_counts: np.ndarray
    percentages: np.ndarray


@dataclass(frozen=True)
class CorpIpoHeadToHeadSummary:
    """Pairwise corp IPO pick percentages by turn and ordinal pick position."""

    num_players: int
    num_games: int
    corp_names: list[str]
    turns: np.ndarray
    pick_positions: np.ndarray
    counts: np.ndarray
    totals: np.ndarray
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


@dataclass(frozen=True)
class CompanyAuctionOutcomeSummary:
    """Outcomes for companies after they are won by a player at auction."""

    num_players: int
    num_games: int
    company_names: list[str]
    company_stars: np.ndarray
    outcome_names: tuple[str, ...]
    counts: np.ndarray
    percentages: np.ndarray
    totals: np.ndarray


@dataclass(frozen=True)
class TurnOneOpeningSummary:
    """Turn 1 auction deltas and first floated corporation percentages."""

    num_players: int
    num_games: int
    company_names: list[str]
    company_ids: np.ndarray
    company_stars: np.ndarray
    face_values: np.ndarray
    auction_mean_deltas: np.ndarray
    auction_counts: np.ndarray
    corp_names: list[str]
    first_ipo_counts: np.ndarray
    first_ipo_percentages: np.ndarray
    first_ipo_games: int


@dataclass(frozen=True)
class InitialAuctionPositionSummary:
    """Auction deltas by face-value rank within the initial auction offering."""

    num_players: int
    num_games: int
    offering_games: int
    complete_games: int
    position_ranks: np.ndarray
    mean_deltas: np.ndarray
    counts: np.ndarray


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

    def corp_ipo_positional_pick_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
        max_picks: int = 4,
    ) -> dict[int, CorpIpoPositionPickSummary]:
        """Summarize corp IPO percentages by turn and ordinal pick slot.

        For each ``(game, turn)``, IPO events are ordered by move number and
        counted as the 1st, 2nd, 3rd, ... pick on that turn. Percentages are
        normalized independently within each ``(turn, pick slot)`` bucket.
        """
        if max_picks <= 0:
            raise ValueError("max_picks must be positive")

        corp_names = [str(v) for v in self.metadata["corp_names"]]  # type: ignore[index]
        num_corps = len(corp_names)
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        events_by_count: dict[int, list[np.ndarray]] = {
            count: [] for count in requested
        }

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue
                # Columns: game_id, move_number, turn_number, player_id, corp_id, ...
                selected = events[:, [0, 1, 2, 4]].astype(np.int64, copy=False)
                events_by_count[shard.num_players].append(
                    np.column_stack(
                        (
                            np.full(selected.shape[0], shard.shard_index, dtype=np.int64),
                            selected,
                        )
                    )
                )

        summaries: dict[int, CorpIpoPositionPickSummary] = {}
        for num_players in sorted(requested):
            event_chunks = events_by_count[num_players]
            pick_positions = np.arange(1, max_picks + 1, dtype=np.int64)
            if not event_chunks:
                summaries[num_players] = CorpIpoPositionPickSummary(
                    num_players=num_players,
                    num_games=game_counts[num_players],
                    corp_names=corp_names,
                    turns=np.empty(0, dtype=np.int64),
                    pick_positions=pick_positions,
                    counts=np.empty((max_picks, 0, num_corps), dtype=np.int64),
                    totals=np.empty((max_picks, 0), dtype=np.int64),
                    percentages=np.empty((max_picks, 0, num_corps), dtype=np.float64),
                )
                continue

            events = np.concatenate(event_chunks, axis=0)
            order = np.lexsort((events[:, 2], events[:, 3], events[:, 1], events[:, 0]))
            events = events[order]
            max_turn = int(events[:, 3].max())
            turns = np.arange(1, max_turn + 1, dtype=np.int64)
            counts = np.zeros((max_picks, turns.shape[0], num_corps), dtype=np.int64)

            current_shard = -1
            current_game = -1
            current_turn = -1
            pick_index = 0
            for shard_index, game_id, _move_number, turn_number, corp_id in events:
                shard_index_int = int(shard_index)
                game_id_int = int(game_id)
                turn_int = int(turn_number)
                if (
                    shard_index_int != current_shard
                    or game_id_int != current_game
                    or turn_int != current_turn
                ):
                    current_shard = shard_index_int
                    current_game = game_id_int
                    current_turn = turn_int
                    pick_index = 0
                if 1 <= turn_int <= max_turn and pick_index < max_picks:
                    counts[pick_index, turn_int - 1, int(corp_id)] += 1
                pick_index += 1

            totals = counts.sum(axis=2)
            percentages = np.zeros_like(counts, dtype=np.float64)
            np.divide(
                counts * 100.0,
                totals[:, :, None],
                out=percentages,
                where=totals[:, :, None] > 0,
            )
            summaries[num_players] = CorpIpoPositionPickSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                corp_names=corp_names,
                turns=turns,
                pick_positions=pick_positions,
                counts=counts,
                totals=totals,
                percentages=percentages,
            )
        return summaries

    def corp_ipo_available_pick_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
        max_picks: int = 4,
    ) -> dict[int, CorpIpoAvailablePickSummary]:
        """Summarize IPO pick probability conditional on corp inactivity.

        For each IPO event with at least two currently inactive corps, all
        inactive corps contribute to that ``(turn, pick slot, corp)``
        denominator. The picked corp contributes to the numerator for the same
        bucket. Forced single-corp IPO choices are excluded.
        """
        if max_picks <= 0:
            raise ValueError("max_picks must be positive")

        corp_names = [str(v) for v in self.metadata["corp_names"]]  # type: ignore[index]
        num_corps = len(corp_names)
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        events_by_count: dict[int, list[np.ndarray]] = {
            count: [] for count in requested
        }

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue
                # Columns: game_id, move_number, turn_number, player_id, corp_id, ...
                selected = events[:, [0, 1, 2, 4]].astype(np.int64, copy=False)
                active_before = np.zeros((events.shape[0], num_corps), dtype=np.int64)
                starts = data["game_start_offsets"].astype(np.int64)
                lengths = data["game_num_examples"].astype(np.int64)
                game_ids = data["game_ids_per_game"].astype(np.int64)
                for game_id, start, length in zip(game_ids, starts, lengths):
                    event_indices = np.flatnonzero(events[:, 0] == int(game_id))
                    if event_indices.shape[0] == 0 or int(length) <= 0:
                        continue
                    row_start = int(start)
                    row_end = row_start + int(length)
                    move_to_row = {
                        int(move_number): row_index
                        for row_index, move_number in enumerate(
                            data["move_numbers"][row_start:row_end]
                        )
                    }
                    active_history = data["corp_active"][row_start:row_end]
                    for event_index in event_indices:
                        row_index = move_to_row.get(int(events[event_index, 1]))
                        if row_index is None:
                            continue
                        active_before[event_index] = active_history[row_index].astype(
                            np.int64,
                            copy=False,
                        )
                events_by_count[shard.num_players].append(
                    np.column_stack(
                        (
                            np.full(selected.shape[0], shard.shard_index, dtype=np.int64),
                            selected,
                            active_before,
                        )
                    )
                )

        summaries: dict[int, CorpIpoAvailablePickSummary] = {}
        for num_players in sorted(requested):
            event_chunks = events_by_count[num_players]
            pick_positions = np.arange(1, max_picks + 1, dtype=np.int64)
            if not event_chunks:
                summaries[num_players] = CorpIpoAvailablePickSummary(
                    num_players=num_players,
                    num_games=game_counts[num_players],
                    corp_names=corp_names,
                    turns=np.empty(0, dtype=np.int64),
                    pick_positions=pick_positions,
                    counts=np.empty((max_picks, 0, num_corps), dtype=np.int64),
                    available_counts=np.empty(
                        (max_picks, 0, num_corps),
                        dtype=np.int64,
                    ),
                    percentages=np.empty((max_picks, 0, num_corps), dtype=np.float64),
                )
                continue

            events = np.concatenate(event_chunks, axis=0)
            order = np.lexsort((events[:, 2], events[:, 3], events[:, 1], events[:, 0]))
            events = events[order]
            max_turn = int(events[:, 3].max())
            turns = np.arange(1, max_turn + 1, dtype=np.int64)
            counts = np.zeros((max_picks, turns.shape[0], num_corps), dtype=np.int64)
            available_counts = np.zeros_like(counts)

            current_shard = -1
            current_game = -1
            current_turn = -1
            pick_index = 0
            for event in events:
                shard_index = event[0]
                game_id = event[1]
                turn_number = event[3]
                corp_id = event[4]
                shard_index_int = int(shard_index)
                game_id_int = int(game_id)
                turn_int = int(turn_number)
                corp_id_int = int(corp_id)
                if shard_index_int != current_shard or game_id_int != current_game:
                    current_shard = shard_index_int
                    current_game = game_id_int
                    current_turn = -1
                    pick_index = 0
                if turn_int != current_turn:
                    current_turn = turn_int
                    pick_index = 0

                if 1 <= turn_int <= max_turn and pick_index < max_picks:
                    inactive = ~event[5:5 + num_corps].astype(bool, copy=False)
                    if int(inactive.sum()) >= 2:
                        available_counts[pick_index, turn_int - 1, inactive] += 1
                        if 0 <= corp_id_int < num_corps and inactive[corp_id_int]:
                            counts[pick_index, turn_int - 1, corp_id_int] += 1

                pick_index += 1

            percentages = np.zeros_like(counts, dtype=np.float64)
            np.divide(
                counts * 100.0,
                available_counts,
                out=percentages,
                where=available_counts > 0,
            )
            summaries[num_players] = CorpIpoAvailablePickSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                corp_names=corp_names,
                turns=turns,
                pick_positions=pick_positions,
                counts=counts,
                available_counts=available_counts,
                percentages=percentages,
            )
        return summaries

    def corp_ipo_head_to_head_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
        max_picks: int = 3,
        max_turn: int = 9,
    ) -> dict[int, CorpIpoHeadToHeadSummary]:
        """Summarize directional head-to-head IPO choices by corp pair.

        Cell ``A, B`` is ``P(A picked | A and B both inactive, picked corp is
        A or B)`` for a given player count, turn, and pick position.
        """
        if max_picks <= 0:
            raise ValueError("max_picks must be positive")
        if max_turn <= 0:
            raise ValueError("max_turn must be positive")

        corp_names = [str(v) for v in self.metadata["corp_names"]]  # type: ignore[index]
        num_corps = len(corp_names)
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        game_counts = {count: 0 for count in requested}
        events_by_count: dict[int, list[np.ndarray]] = {
            count: [] for count in requested
        }

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[shard.num_players] += int(data["game_num_examples"].shape[0])
                events = data["ipo_events"]
                if events.shape[0] == 0:
                    continue

                selected = events[:, [0, 1, 2, 4]].astype(np.int64, copy=False)
                active_before = np.zeros((events.shape[0], num_corps), dtype=np.int64)
                starts = data["game_start_offsets"].astype(np.int64)
                lengths = data["game_num_examples"].astype(np.int64)
                game_ids = data["game_ids_per_game"].astype(np.int64)
                for game_id, start, length in zip(game_ids, starts, lengths):
                    event_indices = np.flatnonzero(events[:, 0] == int(game_id))
                    if event_indices.shape[0] == 0 or int(length) <= 0:
                        continue
                    row_start = int(start)
                    row_end = row_start + int(length)
                    move_to_row = {
                        int(move_number): row_index
                        for row_index, move_number in enumerate(
                            data["move_numbers"][row_start:row_end]
                        )
                    }
                    active_history = data["corp_active"][row_start:row_end]
                    for event_index in event_indices:
                        row_index = move_to_row.get(int(events[event_index, 1]))
                        if row_index is None:
                            continue
                        active_before[event_index] = active_history[row_index].astype(
                            np.int64,
                            copy=False,
                        )
                events_by_count[shard.num_players].append(
                    np.column_stack(
                        (
                            np.full(selected.shape[0], shard.shard_index, dtype=np.int64),
                            selected,
                            active_before,
                        )
                    )
                )

        summaries: dict[int, CorpIpoHeadToHeadSummary] = {}
        turns = np.arange(1, max_turn + 1, dtype=np.int64)
        pick_positions = np.arange(1, max_picks + 1, dtype=np.int64)
        for num_players in sorted(requested):
            counts = np.zeros(
                (max_picks, max_turn, num_corps, num_corps),
                dtype=np.int64,
            )
            totals = np.zeros_like(counts)
            event_chunks = events_by_count[num_players]
            if event_chunks:
                events = np.concatenate(event_chunks, axis=0)
                order = np.lexsort((events[:, 2], events[:, 3], events[:, 1], events[:, 0]))
                events = events[order]

                current_shard = -1
                current_game = -1
                current_turn = -1
                pick_index = 0
                for event in events:
                    shard_index = event[0]
                    game_id = event[1]
                    turn_number = event[3]
                    corp_id = event[4]
                    shard_index_int = int(shard_index)
                    game_id_int = int(game_id)
                    turn_int = int(turn_number)
                    corp_id_int = int(corp_id)
                    if shard_index_int != current_shard or game_id_int != current_game:
                        current_shard = shard_index_int
                        current_game = game_id_int
                        current_turn = -1
                        pick_index = 0
                    if turn_int != current_turn:
                        current_turn = turn_int
                        pick_index = 0

                    if (
                        1 <= turn_int <= max_turn
                        and pick_index < max_picks
                        and 0 <= corp_id_int < num_corps
                    ):
                        inactive = event[5:5 + num_corps].astype(bool, copy=False) == 0
                        if inactive[corp_id_int]:
                            opponents = np.flatnonzero(inactive)
                            opponents = opponents[opponents != corp_id_int]
                            if opponents.size:
                                pick_slot = pick_index
                                turn_slot = turn_int - 1
                                counts[pick_slot, turn_slot, corp_id_int, opponents] += 1
                                totals[pick_slot, turn_slot, corp_id_int, opponents] += 1
                                totals[pick_slot, turn_slot, opponents, corp_id_int] += 1

                    pick_index += 1

            percentages = np.full_like(counts, np.nan, dtype=np.float64)
            np.divide(
                counts * 100.0,
                totals,
                out=percentages,
                where=totals > 0,
            )
            for corp_id in range(num_corps):
                percentages[:, :, corp_id, corp_id] = np.nan

            summaries[num_players] = CorpIpoHeadToHeadSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                corp_names=corp_names,
                turns=turns.copy(),
                pick_positions=pick_positions.copy(),
                counts=counts,
                totals=totals,
                percentages=percentages,
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

    def auctioned_company_outcome_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, CompanyAuctionOutcomeSummary]:
        """Summarize outcomes for companies won by players at auction.

        The denominator is strictly ``auction_events``: companies that moved
        from auction to a player. Each such company is followed until its
        first exit from that player's possession:

        - player -> corp during IPO/PAR: IPO seed
        - player -> corp outside IPO/PAR: acquired by corp
        - player -> removed: closed while personally held
        - still player-owned at terminal state: held to end
        """
        company_names = [str(v) for v in self.metadata["company_names"]]  # type: ignore[index]
        company_static = self.metadata["company_static"]  # type: ignore[index]
        company_stars = np.asarray(company_static["stars"], dtype=np.int16)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        num_companies = len(company_names)
        outcome_count = len(AUCTION_OUTCOME_NAMES)
        counts = {
            count: np.zeros((outcome_count, num_companies), dtype=np.int64)
            for count in requested
        }
        totals = {
            count: np.zeros(num_companies, dtype=np.int64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[num_players] += int(data["game_num_examples"].shape[0])
                shard_counts, shard_totals = _auctioned_company_outcomes_for_shard(
                    data,
                    num_players=num_players,
                    num_companies=num_companies,
                )
                counts[num_players] += shard_counts
                totals[num_players] += shard_totals

        summaries: dict[int, CompanyAuctionOutcomeSummary] = {}
        for num_players in sorted(requested):
            percentages = np.zeros_like(counts[num_players], dtype=np.float64)
            np.divide(
                counts[num_players] * 100.0,
                totals[num_players][None, :],
                out=percentages,
                where=totals[num_players][None, :] > 0,
            )
            summaries[num_players] = CompanyAuctionOutcomeSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                company_names=company_names,
                company_stars=company_stars.copy(),
                outcome_names=AUCTION_OUTCOME_NAMES,
                counts=counts[num_players].copy(),
                percentages=percentages,
                totals=totals[num_players].copy(),
            )
        return summaries

    def turn_one_opening_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, TurnOneOpeningSummary]:
        """Summarize Turn 1 red-company auctions and first corp IPO choices."""
        all_company_names = [str(v) for v in self.metadata["company_names"]]  # type: ignore[index]
        corp_names = [str(v) for v in self.metadata["corp_names"]]  # type: ignore[index]
        company_static = self.metadata["company_static"]  # type: ignore[index]
        company_stars_all = np.asarray(company_static["stars"], dtype=np.int16)  # type: ignore[index]
        face_values_all = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        red_company_ids = np.flatnonzero(company_stars_all == 1).astype(np.int64)
        red_index_by_company = {
            int(company_id): idx
            for idx, company_id in enumerate(red_company_ids)
        }
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        num_red = int(red_company_ids.shape[0])
        num_corps = len(corp_names)
        auction_sums = {
            count: np.zeros(num_red, dtype=np.float64)
            for count in requested
        }
        auction_counts = {
            count: np.zeros(num_red, dtype=np.int64)
            for count in requested
        }
        first_ipo_counts = {
            count: np.zeros(num_corps, dtype=np.int64)
            for count in requested
        }
        first_ipo_games = {count: 0 for count in requested}
        game_counts = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[num_players] += int(data["game_num_examples"].shape[0])

                events = data["auction_events"]
                if events.shape[0] > 0:
                    turn_one = events[events[:, 2] == 1]
                    for event in turn_one:
                        company_id = int(event[3])
                        red_idx = red_index_by_company.get(company_id)
                        if red_idx is None:
                            continue
                        auction_sums[num_players][red_idx] += (
                            float(event[5]) - face_values_all[company_id]
                        )
                        auction_counts[num_players][red_idx] += 1

                first_corps = _first_ipo_corps_for_shard(
                    data,
                    num_players=num_players,
                    num_corps=num_corps,
                )
                if first_corps.size:
                    np.add.at(first_ipo_counts[num_players], first_corps, 1)
                    first_ipo_games[num_players] += int(first_corps.size)

        summaries: dict[int, TurnOneOpeningSummary] = {}
        for num_players in sorted(requested):
            observed = auction_counts[num_players]
            mean_deltas = np.full(num_red, np.nan, dtype=np.float64)
            np.divide(
                auction_sums[num_players],
                observed,
                out=mean_deltas,
                where=observed > 0,
            )
            first_pct = np.zeros(num_corps, dtype=np.float64)
            denominator = first_ipo_games[num_players]
            if denominator:
                first_pct = first_ipo_counts[num_players].astype(np.float64) * (
                    100.0 / denominator
                )
            summaries[num_players] = TurnOneOpeningSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                company_names=[all_company_names[int(i)] for i in red_company_ids],
                company_ids=red_company_ids.copy(),
                company_stars=company_stars_all[red_company_ids].copy(),
                face_values=face_values_all[red_company_ids].copy(),
                auction_mean_deltas=mean_deltas,
                auction_counts=observed.copy(),
                corp_names=corp_names,
                first_ipo_counts=first_ipo_counts[num_players].copy(),
                first_ipo_percentages=first_pct,
                first_ipo_games=first_ipo_games[num_players],
            )
        return summaries

    def initial_auction_position_summary(
        self,
        *,
        player_counts: Sequence[int] | None = None,
    ) -> dict[int, InitialAuctionPositionSummary]:
        """Summarize setup-offering auction deltas by face-value rank."""
        company_static = self.metadata["company_static"]  # type: ignore[index]
        face_values = np.asarray(company_static["face_value"], dtype=np.float64)  # type: ignore[index]
        requested = (
            set(self.player_counts)
            if player_counts is None
            else {int(v) for v in player_counts}
        )
        sums = {
            count: np.zeros(count, dtype=np.float64)
            for count in requested
        }
        counts = {
            count: np.zeros(count, dtype=np.int64)
            for count in requested
        }
        game_counts = {count: 0 for count in requested}
        offering_games = {count: 0 for count in requested}
        complete_games = {count: 0 for count in requested}

        for shard in self.iter_shards():
            if shard.num_players not in requested:
                continue
            num_players = shard.num_players
            with np.load(shard.path, allow_pickle=False) as data:
                game_counts[num_players] += int(data["game_num_examples"].shape[0])
                shard_sums, shard_counts, shard_offerings, shard_complete = (
                    _initial_auction_position_deltas_for_shard(
                        data,
                        num_players=num_players,
                        face_values=face_values,
                    )
                )
                sums[num_players] += shard_sums
                counts[num_players] += shard_counts
                offering_games[num_players] += shard_offerings
                complete_games[num_players] += shard_complete

        summaries: dict[int, InitialAuctionPositionSummary] = {}
        for num_players in sorted(requested):
            mean_deltas = np.full(num_players, np.nan, dtype=np.float64)
            np.divide(
                sums[num_players],
                counts[num_players],
                out=mean_deltas,
                where=counts[num_players] > 0,
            )
            summaries[num_players] = InitialAuctionPositionSummary(
                num_players=num_players,
                num_games=game_counts[num_players],
                offering_games=offering_games[num_players],
                complete_games=complete_games[num_players],
                position_ranks=np.arange(num_players, 0, -1, dtype=np.int64),
                mean_deltas=mean_deltas,
                counts=counts[num_players].copy(),
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


def _auctioned_company_outcomes_for_shard(
    data: object,
    *,
    num_players: int,
    num_companies: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify auction-won company outcomes for one shard."""
    outcome_count = len(AUCTION_OUTCOME_NAMES)
    counts = np.zeros((outcome_count, num_companies), dtype=np.int64)
    totals = np.zeros(num_companies, dtype=np.int64)
    auction_events = data["auction_events"]
    if auction_events.shape[0] == 0:
        return counts, totals

    layout = get_layout(num_players)
    company_fields = get_company_fields()
    company_ids = np.arange(num_companies, dtype=np.int64)
    final_locations = data["final_states"][
        :,
        layout.companies_offset + company_fields.locations + company_ids,
    ]
    final_owners = data["final_states"][
        :,
        layout.companies_offset + company_fields.owner_ids + company_ids,
    ]

    game_ids_per_game = data["game_ids_per_game"].astype(np.int64)
    starts = data["game_start_offsets"].astype(np.int64)
    lengths = data["game_num_examples"].astype(np.int64)

    for game_index, game_id in enumerate(game_ids_per_game):
        game_auctions = auction_events[auction_events[:, 0] == game_id]
        if game_auctions.shape[0] == 0 or lengths[game_index] <= 0:
            continue

        start = int(starts[game_index])
        end = start + int(lengths[game_index])
        move_numbers = data["move_numbers"][start:end]
        phase_ids = data["phase_ids"][start:end]
        engine_phase_ids = data["engine_phase_ids"][start:end]
        locations = data["company_locations"][start:end]
        owners = data["company_owners"][start:end]

        for event in game_auctions:
            auction_move = int(event[1])
            company_id = int(event[3])
            winner = int(event[4])
            outcome = _classify_auctioned_company_outcome(
                move_numbers=move_numbers,
                phase_ids=phase_ids,
                engine_phase_ids=engine_phase_ids,
                company_locations=locations[:, company_id],
                company_owners=owners[:, company_id],
                final_location=int(final_locations[game_index, company_id]),
                final_owner=int(final_owners[game_index, company_id]),
                auction_move=auction_move,
                winner=winner,
            )
            counts[outcome, company_id] += 1
            totals[company_id] += 1

    return counts, totals


def _classify_auctioned_company_outcome(
    *,
    move_numbers: np.ndarray,
    phase_ids: np.ndarray,
    engine_phase_ids: np.ndarray,
    company_locations: np.ndarray,
    company_owners: np.ndarray,
    final_location: int,
    final_owner: int,
    auction_move: int,
    winner: int,
) -> int:
    """Classify one auction-won company by first exit from player ownership."""
    held_after_auction = (
        (move_numbers > auction_move)
        & (company_locations == LOC_PLAYER_INT)
        & (company_owners == winner)
    )
    if np.any(held_after_auction):
        first_held = int(np.flatnonzero(held_after_auction)[0])
        for idx in range(first_held + 1, move_numbers.shape[0]):
            if (
                int(company_locations[idx]) == LOC_PLAYER_INT
                and int(company_owners[idx]) == winner
            ):
                continue
            return _auction_exit_outcome(
                int(company_locations[idx]),
                int(phase_ids[idx - 1]),
                int(engine_phase_ids[idx - 1]),
            )

    if final_location == LOC_PLAYER_INT and final_owner == winner:
        return AUCTION_OUTCOME_HELD
    if move_numbers.shape[0] == 0:
        raise ValueError("cannot classify auctioned company in empty game trace")
    return _auction_exit_outcome(
        final_location,
        int(phase_ids[-1]),
        int(engine_phase_ids[-1]),
    )


def _auction_exit_outcome(
    location: int,
    phase_id: int,
    engine_phase_id: int,
) -> int:
    """Map a location exit from player ownership to an outcome bucket."""
    if location in (LOC_CORP_INT, LOC_CORP_ACQ_INT):
        if phase_id in (IPO_PHASE_ID, PAR_PHASE_ID) or engine_phase_id in (
            ENGINE_IPO_PHASE_ID,
            ENGINE_PAR_PHASE_ID,
        ):
            return AUCTION_OUTCOME_IPO
        return AUCTION_OUTCOME_ACQUIRED
    if location == LOC_REMOVED_INT:
        return AUCTION_OUTCOME_CLOSED
    raise ValueError(
        "unexpected auction-won company exit: "
        f"location={location}, phase_id={phase_id}, engine_phase_id={engine_phase_id}"
    )


def _first_ipo_corps_for_shard(
    data: object,
    *,
    num_players: int,
    num_corps: int,
) -> np.ndarray:
    """Return first corp activated on Turn 1 in each game in one shard."""
    layout = get_layout(num_players)
    corp_fields = get_corp_fields()
    corp_bases = (
        layout.corps_offset
        + np.arange(num_corps, dtype=np.int64) * layout.corp_size
    )
    final_active = (
        data["final_states"][:, corp_bases + corp_fields.active] != 0
    )
    first_corps: list[int] = []

    starts = data["game_start_offsets"].astype(np.int64)
    lengths = data["game_num_examples"].astype(np.int64)
    for game_index, (start_raw, length_raw) in enumerate(zip(starts, lengths)):
        length = int(length_raw)
        if length <= 0:
            continue
        start = int(start_raw)
        end = start + length
        active_history = data["corp_active"][start:end].astype(bool, copy=False)
        combined = np.vstack((active_history, final_active[game_index]))
        transitions = (~combined[:-1]) & combined[1:]
        if not np.any(transitions):
            continue
        first_row = int(np.flatnonzero(transitions.any(axis=1))[0])
        transition_turn = int(data["turn_numbers"][start + min(first_row, length - 1)])
        if transition_turn != 1:
            continue
        corp_id = int(np.flatnonzero(transitions[first_row])[0])
        first_corps.append(corp_id)

    return np.asarray(first_corps, dtype=np.int64)


def _initial_auction_position_deltas_for_shard(
    data: object,
    *,
    num_players: int,
    face_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return sums/counts for initial-auction positions in one shard."""
    sums = np.zeros(num_players, dtype=np.float64)
    counts = np.zeros(num_players, dtype=np.int64)
    offering_games = 0
    complete_games = 0

    auction_events = data["auction_events"]
    auction_prices = {
        (int(event[0]), int(event[3])): float(event[5])
        for event in auction_events
        if int(event[2]) == 1
    }

    starts = data["game_start_offsets"].astype(np.int64)
    lengths = data["game_num_examples"].astype(np.int64)
    game_ids = data["game_ids_per_game"].astype(np.int64)
    for game_id, start, length in zip(game_ids, starts, lengths):
        if int(length) <= 0:
            continue
        initial_locations = data["company_locations"][int(start)]
        company_ids = np.flatnonzero(initial_locations == int(CompanyLocation.LOC_AUCTION))
        if company_ids.shape[0] != num_players:
            continue
        offering_games += 1

        sorted_company_ids = company_ids[
            np.argsort(face_values[company_ids], kind="stable")
        ]
        complete = True
        for position, company_id in enumerate(sorted_company_ids):
            price = auction_prices.get((int(game_id), int(company_id)))
            if price is None:
                complete = False
                continue
            sums[position] += price - face_values[int(company_id)]
            counts[position] += 1
        if complete:
            complete_games += 1

    return sums, counts, offering_games, complete_games


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


def corp_ipo_positional_pick_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
    max_picks: int = 4,
) -> dict[int, CorpIpoPositionPickSummary]:
    """Convenience wrapper for per-turn IPO pick-position summaries."""
    return StrategyDataset(run_dir).corp_ipo_positional_pick_summary(
        player_counts=player_counts,
        max_picks=max_picks,
    )


def corp_ipo_available_pick_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
    max_picks: int = 4,
) -> dict[int, CorpIpoAvailablePickSummary]:
    """Convenience wrapper for availability-conditional IPO pick summaries."""
    return StrategyDataset(run_dir).corp_ipo_available_pick_summary(
        player_counts=player_counts,
        max_picks=max_picks,
    )


def corp_ipo_head_to_head_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
    max_picks: int = 3,
    max_turn: int = 9,
) -> dict[int, CorpIpoHeadToHeadSummary]:
    """Convenience wrapper for pairwise IPO head-to-head summaries."""
    return StrategyDataset(run_dir).corp_ipo_head_to_head_summary(
        player_counts=player_counts,
        max_picks=max_picks,
        max_turn=max_turn,
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


def auctioned_company_outcome_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, CompanyAuctionOutcomeSummary]:
    """Convenience wrapper for auction-won company outcome summaries."""
    return StrategyDataset(run_dir).auctioned_company_outcome_summary(
        player_counts=player_counts
    )


def turn_one_opening_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, TurnOneOpeningSummary]:
    """Convenience wrapper for Turn 1 auction and first-IPO summaries."""
    return StrategyDataset(run_dir).turn_one_opening_summary(
        player_counts=player_counts
    )


def initial_auction_position_summary(
    run_dir: str | Path,
    *,
    player_counts: Sequence[int] | None = None,
) -> dict[int, InitialAuctionPositionSummary]:
    """Convenience wrapper for initial-offering Turn 1 auction positions."""
    return StrategyDataset(run_dir).initial_auction_position_summary(
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
