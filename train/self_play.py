"""Self-play game generation via MCTS.

Post-refactor contract: dense unified-slot policy targets + legal masks,
raw canonical int16 game state stored on examples, and canonical value
targets. The trainer consumes
``(state, phase_id, legal_mask, policy_target, value_target)`` and
materializes model inputs from the raw state at training time: transformer
runs token/relation extraction, while ResNet runs dense active-relative
vector extraction and rotates canonical value targets at the loss boundary.
Policy cross-entropy is computed over the full unified-logit slot space
(illegal slots are already zero in ``policy_target`` and masked to -1e9
inside the model).
"""

from __future__ import annotations

import queue
import signal
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY,
    ACTION_ACQ_PRICE_PY,
    ACTION_ACQ_SELECT_COMPANY_PY,
    ACTION_CLOSE_PY,
    ACTION_DIVIDEND_PY,
    ACTION_BUY_SHARE_PY,
    ACTION_ISSUE_PY,
    ACTION_PAR_PY,
    ACTION_RAISE_PY,
    ACTION_SELL_SHARE_PY,
    decode_action_py,
    enumerate_legal_actions_py,
    get_decision_phase_py,
)
from core.data import ALL_PAR_PRICES, CorpIndices, GameConstants, MAX_ACTION_SIZE
from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState, get_layout
from entities.company import (
    COMPANIES,
    CompanyLocation,
)
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import compute_terminal_values
from mcts.search import (
    StatePool,
    get_greedy_leaf_depth,
    get_greedy_leaf_value,
    prepare_reuse_root,
    run_search,
    scale_visit_counts_by_temperature,
)
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut
from train.config import EpochConfig, TrainingConfig
from train.eval_server import RemoteEvaluator
from train.profile_stats import EvalClientStats, GameProfileData, SearchStats


U_DIM = int(UNIFIED_LOGIT_DIM)
LOC_AUCTION_INT = int(CompanyLocation.LOC_AUCTION)
LOC_CORP_INT = int(CompanyLocation.LOC_CORP)
LOC_CORP_ACQ_INT = int(CompanyLocation.LOC_CORP_ACQ)
LOC_FI_INT = int(CompanyLocation.LOC_FI)
LOC_PLAYER_INT = int(CompanyLocation.LOC_PLAYER)
LOC_REMOVED_INT = int(CompanyLocation.LOC_REMOVED)


@dataclass
class SelfPlayExample:
    """Single training example from self-play.

    Raw compact int16 game state — the trainer derives model-family-specific
    input buffers from this row at training time.
    ``legal_mask`` and ``policy_target`` are dense over the unified-logit
    slot space; slots outside the current phase's legal set are zero in
    both. ``phase_id`` is carried purely for per-phase TB bucketing.
    ``value_target`` is always canonical player order.
    """

    state: np.ndarray  # (total_int16_size,), int16 — raw compact state
    phase_id: int  # decision phase id 0-10 (TB reporting only)
    legal_mask: np.ndarray  # (UNIFIED_LOGIT_DIM,), uint8 — 1 = legal slot
    policy_target: np.ndarray  # (UNIFIED_LOGIT_DIM,), float32 — MCTS visit probs
    value_target: np.ndarray  # (num_players,), float32 — canonical A0GB


@dataclass
class StrategyTrace:
    """Optional per-decision trace payload for offline strategy analysis.

    Arrays are aligned row-for-row with ``GameRecord.states``. Policy arrays
    are percentages in dense unified-slot space, not raw probabilities.
    Event arrays use move-relative rows; the shard writer prepends game IDs.
    """

    nn_policy_pct: np.ndarray  # (num_examples, UNIFIED_LOGIT_DIM), float32
    nn_values: np.ndarray  # (num_examples, num_players), float32
    mcts_visit_counts: np.ndarray  # (num_examples, UNIFIED_LOGIT_DIM), int32
    a0gb_values: np.ndarray  # (num_examples, num_players), float32
    mcts_root_values: np.ndarray  # (num_examples, num_players), float32
    selected_action_ids: np.ndarray  # (num_examples,), uint16
    selected_unified_slots: np.ndarray  # (num_examples,), int16
    action_types: np.ndarray  # (num_examples,), int16
    action_corps: np.ndarray  # (num_examples,), int16
    action_companies: np.ndarray  # (num_examples,), int16
    action_amounts: np.ndarray  # (num_examples,), int16
    engine_phase_ids: np.ndarray  # (num_examples,), int8
    active_players: np.ndarray  # (num_examples,), int8
    active_corps: np.ndarray  # (num_examples,), int8
    active_companies: np.ndarray  # (num_examples,), int8
    turn_numbers: np.ndarray  # (num_examples,), int16
    coo_levels: np.ndarray  # (num_examples,), int8
    cards_remaining: np.ndarray  # (num_examples,), int8
    auction_prices: np.ndarray  # (num_examples,), int16
    auction_high_bidders: np.ndarray  # (num_examples,), int8
    auction_starters: np.ndarray  # (num_examples,), int8
    acq_offer_prices: np.ndarray  # (num_examples,), int16
    acq_offer_corps: np.ndarray  # (num_examples,), int8
    target_temperatures: np.ndarray  # (num_examples,), float32
    sample_temperatures: np.ndarray  # (num_examples,), float32
    greedy_leaf_depths: np.ndarray  # (num_examples,), int16
    root_visit_counts: np.ndarray  # (num_examples,), int32
    player_cash: np.ndarray  # (num_examples, num_players), int16
    player_net_worth: np.ndarray  # (num_examples, num_players), int16
    player_liquidity: np.ndarray  # (num_examples, num_players), int16
    player_income: np.ndarray  # (num_examples, num_players), int16
    player_shares: np.ndarray  # (num_examples, num_players, NUM_CORPS), int8
    corp_active: np.ndarray  # (num_examples, NUM_CORPS), int8
    corp_prices: np.ndarray  # (num_examples, NUM_CORPS), int16
    corp_cash: np.ndarray  # (num_examples, NUM_CORPS), int16
    corp_income: np.ndarray  # (num_examples, NUM_CORPS), int16
    corp_presidents: np.ndarray  # (num_examples, NUM_CORPS), int8
    corp_issued_shares: np.ndarray  # (num_examples, NUM_CORPS), int8
    corp_bank_shares: np.ndarray  # (num_examples, NUM_CORPS), int8
    corp_unissued_shares: np.ndarray  # (num_examples, NUM_CORPS), int8
    corp_receivership: np.ndarray  # (num_examples, NUM_CORPS), int8
    company_locations: np.ndarray  # (num_examples, NUM_COMPANIES), int8
    company_owners: np.ndarray  # (num_examples, NUM_COMPANIES), int8
    company_adjusted_income: np.ndarray  # (num_examples, NUM_COMPANIES), int16
    auction_events: np.ndarray  # (n, 9), int32
    ipo_events: np.ndarray  # (n, 11), int32
    acquisition_events: np.ndarray  # (n, 12), int32
    share_trade_events: np.ndarray  # (n, 12), int32
    dividend_events: np.ndarray  # (n, 9), int32
    issue_events: np.ndarray  # (n, 12), int32
    close_events: np.ndarray  # (n, 8), int32


@dataclass
class GameRecord:
    """Results from a single self-play game.

    Training data is pre-stacked into contiguous arrays (5 arrays instead
    of N×5 small arrays) so that pickling through mp.Queue is a fast
    memcpy rather than per-object serialization. ``legal_masks`` and
    ``policy_targets`` are dense over ``UNIFIED_LOGIT_DIM`` unified
    slots; illegal slots carry 0 in both.
    """

    states: np.ndarray  # (num_examples, total_int16_size), int16
    phase_ids: np.ndarray  # (num_examples,), int8 — TB reporting only
    legal_masks: np.ndarray  # (num_examples, UNIFIED_LOGIT_DIM), uint8
    policy_targets: np.ndarray  # (num_examples, UNIFIED_LOGIT_DIM), float32
    value_targets: np.ndarray  # (num_examples, num_players), float32
    num_players: int  # Actual player count for this game
    num_examples: int  # Number of training examples
    total_moves: int  # Decision points (MCTS searches)
    net_worths: list[int]  # Final net worth per player (canonical order)
    shares_per_player: list[int]  # Total shares held per player (canonical order)
    companies_per_player: list[int]  # Companies owned per player (canonical order)
    pres_share_values: list[float]  # Value of shares in corps where player is president
    nw_cash_pct: list[float]  # % of net worth from cash per player
    nw_companies_pct: list[float]  # % of net worth from owned company face values
    nw_shares_pct: list[float]  # % of net worth from owned shares (count * price)
    avg_active_corp_price: float  # Average share price of active corps
    corps_in_receivership: int  # Number of corps in receivership
    has_max_price_corp: bool  # Whether any active corp finished at max share price (75)
    duration_secs: float  # Wall-clock time
    policy_entropy_mean: float = 0.0  # Legacy alias: policy target entropy
    top1_visit_fraction: float = 0.0  # Legacy alias: sample top-1 fraction
    policy_target_entropy_mean: float = 0.0  # Mean target entropy (nats)
    policy_target_top1_fraction: float = 0.0  # Mean target top-1 mass
    sample_policy_entropy_mean: float = 0.0  # Mean action-sampling entropy (nats)
    sample_top1_action_fraction: float = 0.0  # Mean action-sampling top-1 mass
    profile: GameProfileData | None = None
    game_id: int = -1
    game_seed: int = -1
    rng_seed: int = -1
    final_state: np.ndarray | None = None
    strategy_trace: StrategyTrace | None = None


def _compute_linear_temperature(
    move_count: int,
    initial: float,
    anneal_start: int,
    anneal_end: int,
    final: float,
) -> float:
    """Compute a move-indexed linear temperature schedule."""
    if move_count <= anneal_start:
        return initial
    if move_count >= anneal_end:
        return final
    # Linear interpolation
    span = anneal_end - anneal_start
    t = (move_count - anneal_start) / span
    return initial + t * (final - initial)


def _compute_temperature(
    move_count: int,
    config: TrainingConfig,
    num_players: int,
) -> float:
    """Compute action-sampling temperature for the current move."""
    anneal_start, anneal_end = config.temp_anneal_window(num_players)
    return _compute_linear_temperature(
        move_count,
        config.temp_initial,
        anneal_start,
        anneal_end,
        config.temp_final,
    )


def _compute_policy_target_temperature(
    move_count: int,
    config: TrainingConfig,
    num_players: int,
) -> float:
    """Compute policy-target temperature for the current move."""
    anneal_start, anneal_end = config.policy_target_temp_anneal_window(
        num_players
    )
    return _compute_linear_temperature(
        move_count,
        config.policy_target_temp_initial,
        anneal_start,
        anneal_end,
        config.policy_target_temp_final,
    )


def build_epoch_player_count_schedule(
    config: TrainingConfig,
    games_per_epoch: int | None = None,
) -> list[int]:
    """Build the deterministic per-epoch player-count assignment schedule.

    Mixed-count epochs allocate an even quota to each player count, put the
    remainder on the lowest count, then emit assignments round-robin from
    non-depleted quotas.
    """
    total_games = (
        config.games_per_epoch
        if games_per_epoch is None
        else int(games_per_epoch)
    )
    if total_games < 0:
        raise ValueError(f"games_per_epoch must be >= 0, got {total_games}")

    player_counts = list(config.iter_player_counts())
    if len(player_counts) == 1:
        return [player_counts[0]] * total_games

    base_quota, remainder = divmod(total_games, len(player_counts))
    quotas = {num_players: base_quota for num_players in player_counts}
    quotas[player_counts[0]] += remainder

    schedule: list[int] = []
    while len(schedule) < total_games:
        for num_players in player_counts:
            remaining = quotas[num_players]
            if remaining <= 0:
                continue
            schedule.append(num_players)
            quotas[num_players] = remaining - 1
            if len(schedule) == total_games:
                break
    return schedule


def _validate_assigned_num_players(
    config: TrainingConfig,
    num_players: int,
) -> int:
    """Validate a main-process player-count assignment."""
    num_players = int(num_players)
    if not (
        config.effective_min_players
        <= num_players
        <= config.effective_max_players
    ):
        raise ValueError(
            f"assigned num_players must be in "
            f"[{config.effective_min_players}, {config.effective_max_players}], "
            f"got {num_players}"
        )
    return num_players


def _resolve_game_num_players(
    config: TrainingConfig,
    num_players: int | None,
) -> int:
    """Resolve the actual player count for a game task."""
    if num_players is None:
        if config.is_mixed_player_training:
            raise ValueError(
                "mixed player-count self-play requires an explicit "
                "num_players assignment"
            )
        return config.num_players
    return _validate_assigned_num_players(config, num_players)


def _strategy_empty_events(width: int) -> np.ndarray:
    return np.empty((0, width), dtype=np.int32)


def _strategy_rows(rows: list[list[int]], width: int) -> np.ndarray:
    if not rows:
        return _strategy_empty_events(width)
    return np.asarray(rows, dtype=np.int32).reshape((-1, width))


class _StrategyTraceBuilder:
    """Accumulate per-move analysis traces for one self-play game."""

    _NUM_CORPS = int(GameConstants.NUM_CORPS)
    _NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)
    _OS_CORP_ID = int(CorpIndices.CORP_OS)

    def __init__(self, num_players: int) -> None:
        self.num_players = int(num_players)

        self.nn_policy_pct: list[np.ndarray] = []
        self.nn_values: list[np.ndarray] = []
        self.mcts_visit_counts: list[np.ndarray] = []
        self.a0gb_values: list[np.ndarray] = []
        self.mcts_root_values: list[np.ndarray] = []
        self.selected_action_ids: list[int] = []
        self.selected_unified_slots: list[int] = []
        self.action_types: list[int] = []
        self.action_corps: list[int] = []
        self.action_companies: list[int] = []
        self.action_amounts: list[int] = []

        self.engine_phase_ids: list[int] = []
        self.active_players: list[int] = []
        self.active_corps: list[int] = []
        self.active_companies: list[int] = []
        self.turn_numbers: list[int] = []
        self.coo_levels: list[int] = []
        self.cards_remaining: list[int] = []
        self.auction_prices: list[int] = []
        self.auction_high_bidders: list[int] = []
        self.auction_starters: list[int] = []
        self.acq_offer_prices: list[int] = []
        self.acq_offer_corps: list[int] = []
        self.target_temperatures: list[float] = []
        self.sample_temperatures: list[float] = []
        self.greedy_leaf_depths: list[int] = []
        self.root_visit_counts: list[int] = []

        self.player_cash: list[np.ndarray] = []
        self.player_net_worth: list[np.ndarray] = []
        self.player_liquidity: list[np.ndarray] = []
        self.player_income: list[np.ndarray] = []
        self.player_shares: list[np.ndarray] = []
        self.corp_active: list[np.ndarray] = []
        self.corp_prices: list[np.ndarray] = []
        self.corp_cash: list[np.ndarray] = []
        self.corp_income: list[np.ndarray] = []
        self.corp_presidents: list[np.ndarray] = []
        self.corp_issued_shares: list[np.ndarray] = []
        self.corp_bank_shares: list[np.ndarray] = []
        self.corp_unissued_shares: list[np.ndarray] = []
        self.corp_receivership: list[np.ndarray] = []
        self.company_locations: list[np.ndarray] = []
        self.company_owners: list[np.ndarray] = []
        self.company_adjusted_income: list[np.ndarray] = []

        self.auction_events: list[list[int]] = []
        self.ipo_events: list[list[int]] = []
        self.acquisition_events: list[list[int]] = []
        self.share_trade_events: list[list[int]] = []
        self.dividend_events: list[list[int]] = []
        self.issue_events: list[list[int]] = []
        self.close_events: list[list[int]] = []

    def capture_summary(self, state: GameState) -> dict[str, object]:
        num_players = self.num_players
        num_corps = self._NUM_CORPS
        num_companies = self._NUM_COMPANIES

        summary: dict[str, object] = {
            "engine_phase": int(TURN.get_phase(state)),
            "active_player": int(TURN.get_active_player(state)),
            "active_corp": int(TURN.get_active_corp(state)),
            "active_company": int(TURN.get_active_company(state)),
            "turn_number": int(TURN.get_turn_number(state)),
            "coo_level": int(TURN.get_coo_level(state)),
            "cards_remaining": int(TURN.get_cards_remaining(state)),
            "auction_price": int(TURN.get_auction_price(state)),
            "auction_high_bidder": int(TURN.get_auction_high_bidder(state)),
            "auction_starter": int(TURN.get_auction_starter(state)),
            "acq_offer_price": int(TURN.get_acq_offer_price(state)),
            "acq_offer_corp": int(TURN.get_acq_offer_corp(state)),
            "player_cash": np.asarray(
                [PLAYERS[p].get_cash(state) for p in range(num_players)],
                dtype=np.int16,
            ),
            "player_net_worth": np.asarray(
                [PLAYERS[p].get_net_worth(state) for p in range(num_players)],
                dtype=np.int16,
            ),
            "player_liquidity": np.asarray(
                [PLAYERS[p].get_liquidity(state) for p in range(num_players)],
                dtype=np.int16,
            ),
            "player_income": np.asarray(
                [PLAYERS[p].get_income(state) for p in range(num_players)],
                dtype=np.int16,
            ),
            "player_shares": np.asarray(
                [
                    [PLAYERS[p].get_shares(state, c) for c in range(num_corps)]
                    for p in range(num_players)
                ],
                dtype=np.int8,
            ),
            "corp_active": np.asarray(
                [int(CORPS[c].is_active(state)) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "corp_prices": np.asarray(
                [CORPS[c].get_share_price(state) for c in range(num_corps)],
                dtype=np.int16,
            ),
            "corp_cash": np.asarray(
                [CORPS[c].get_cash(state) for c in range(num_corps)],
                dtype=np.int16,
            ),
            "corp_income": np.asarray(
                [CORPS[c].get_income(state) for c in range(num_corps)],
                dtype=np.int16,
            ),
            "corp_presidents": np.asarray(
                [CORPS[c].get_president_id(state) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "corp_issued_shares": np.asarray(
                [CORPS[c].get_issued_shares(state) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "corp_bank_shares": np.asarray(
                [CORPS[c].get_bank_shares(state) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "corp_unissued_shares": np.asarray(
                [CORPS[c].get_unissued_shares(state) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "corp_receivership": np.asarray(
                [int(CORPS[c].is_in_receivership(state)) for c in range(num_corps)],
                dtype=np.int8,
            ),
            "company_locations": np.asarray(
                [COMPANIES[c].get_location(state) for c in range(num_companies)],
                dtype=np.int8,
            ),
            "company_owners": np.asarray(
                [COMPANIES[c].get_owner_id(state) for c in range(num_companies)],
                dtype=np.int8,
            ),
            "company_adjusted_income": np.asarray(
                [COMPANIES[c].get_adjusted_income(state) for c in range(num_companies)],
                dtype=np.int16,
            ),
        }
        return summary

    def append_pre_search(
        self,
        state: GameState,
        slots: np.ndarray,
        nn_policy_sparse: np.ndarray,
        nn_values: np.ndarray,
    ) -> dict[str, object]:
        summary = self.capture_summary(state)

        dense_nn = np.zeros(U_DIM, dtype=np.float32)
        dense_nn[slots] = nn_policy_sparse.astype(np.float32) * 100.0
        self.nn_policy_pct.append(dense_nn)
        self.nn_values.append(nn_values.astype(np.float32, copy=True))

        self.engine_phase_ids.append(int(summary["engine_phase"]))
        self.active_players.append(int(summary["active_player"]))
        self.active_corps.append(int(summary["active_corp"]))
        self.active_companies.append(int(summary["active_company"]))
        self.turn_numbers.append(int(summary["turn_number"]))
        self.coo_levels.append(int(summary["coo_level"]))
        self.cards_remaining.append(int(summary["cards_remaining"]))
        self.auction_prices.append(int(summary["auction_price"]))
        self.auction_high_bidders.append(int(summary["auction_high_bidder"]))
        self.auction_starters.append(int(summary["auction_starter"]))
        self.acq_offer_prices.append(int(summary["acq_offer_price"]))
        self.acq_offer_corps.append(int(summary["acq_offer_corp"]))

        self.player_cash.append(summary["player_cash"])  # type: ignore[arg-type]
        self.player_net_worth.append(summary["player_net_worth"])  # type: ignore[arg-type]
        self.player_liquidity.append(summary["player_liquidity"])  # type: ignore[arg-type]
        self.player_income.append(summary["player_income"])  # type: ignore[arg-type]
        self.player_shares.append(summary["player_shares"])  # type: ignore[arg-type]
        self.corp_active.append(summary["corp_active"])  # type: ignore[arg-type]
        self.corp_prices.append(summary["corp_prices"])  # type: ignore[arg-type]
        self.corp_cash.append(summary["corp_cash"])  # type: ignore[arg-type]
        self.corp_income.append(summary["corp_income"])  # type: ignore[arg-type]
        self.corp_presidents.append(summary["corp_presidents"])  # type: ignore[arg-type]
        self.corp_issued_shares.append(summary["corp_issued_shares"])  # type: ignore[arg-type]
        self.corp_bank_shares.append(summary["corp_bank_shares"])  # type: ignore[arg-type]
        self.corp_unissued_shares.append(summary["corp_unissued_shares"])  # type: ignore[arg-type]
        self.corp_receivership.append(summary["corp_receivership"])  # type: ignore[arg-type]
        self.company_locations.append(summary["company_locations"])  # type: ignore[arg-type]
        self.company_owners.append(summary["company_owners"])  # type: ignore[arg-type]
        self.company_adjusted_income.append(summary["company_adjusted_income"])  # type: ignore[arg-type]
        return summary

    def append_search_result(
        self,
        *,
        root: Any,
        slots: np.ndarray,
        a0gb_value: np.ndarray,
        action_id: int,
        selected_slot: int,
        action_info: Any,
        target_temperature: float,
        sample_temperature: float,
    ) -> None:
        assert root.visit_counts is not None
        dense_counts = np.zeros(U_DIM, dtype=np.int32)
        dense_counts[slots] = root.visit_counts.astype(np.int32)
        self.mcts_visit_counts.append(dense_counts)
        self.a0gb_values.append(a0gb_value.astype(np.float32, copy=True))

        if root.visit_count > 0:
            root_values = root.value_sum / root.visit_count
        else:
            root_values = np.zeros(self.num_players, dtype=np.float32)
        self.mcts_root_values.append(root_values.astype(np.float32, copy=True))

        self.selected_action_ids.append(int(action_id))
        self.selected_unified_slots.append(int(selected_slot))
        self.action_types.append(int(action_info.action_type))
        self.action_corps.append(int(action_info.corp_id))
        self.action_companies.append(int(action_info.company_id))
        self.action_amounts.append(int(action_info.amount))
        self.target_temperatures.append(float(target_temperature))
        self.sample_temperatures.append(float(sample_temperature))
        self.greedy_leaf_depths.append(int(get_greedy_leaf_depth(root)))
        self.root_visit_counts.append(int(root.visit_count))

    def append_action_events(
        self,
        pre: dict[str, object],
        post_state: GameState,
        *,
        move_number: int,
        action_id: int,
        action_info: Any,
    ) -> None:
        post = self.capture_summary(post_state)
        action_type = int(action_info.action_type)
        turn_number = int(pre["turn_number"])
        active_player = int(pre["active_player"])
        active_corp = int(pre["active_corp"])
        active_company = int(pre["active_company"])

        pre_company_locations = pre["company_locations"]  # type: ignore[assignment]
        pre_company_owners = pre["company_owners"]  # type: ignore[assignment]
        post_company_locations = post["company_locations"]  # type: ignore[assignment]
        post_company_owners = post["company_owners"]  # type: ignore[assignment]
        pre_corp_prices = pre["corp_prices"]  # type: ignore[assignment]
        post_corp_prices = post["corp_prices"]  # type: ignore[assignment]
        pre_corp_cash = pre["corp_cash"]  # type: ignore[assignment]
        post_corp_cash = post["corp_cash"]  # type: ignore[assignment]
        pre_corp_active = pre["corp_active"]  # type: ignore[assignment]
        post_corp_active = post["corp_active"]  # type: ignore[assignment]
        pre_corp_bank = pre["corp_bank_shares"]  # type: ignore[assignment]
        post_corp_bank = post["corp_bank_shares"]  # type: ignore[assignment]
        pre_corp_issued = pre["corp_issued_shares"]  # type: ignore[assignment]
        post_corp_issued = post["corp_issued_shares"]  # type: ignore[assignment]
        pre_player_cash = pre["player_cash"]  # type: ignore[assignment]
        post_player_cash = post["player_cash"]  # type: ignore[assignment]
        pre_player_shares = pre["player_shares"]  # type: ignore[assignment]
        post_player_shares = post["player_shares"]  # type: ignore[assignment]

        if 0 <= active_company < self._NUM_COMPANIES:
            pre_loc = int(pre_company_locations[active_company])
            post_loc = int(post_company_locations[active_company])
            post_owner = int(post_company_owners[active_company])
            if pre_loc == LOC_AUCTION_INT and post_loc == LOC_PLAYER_INT:
                if action_type == int(ACTION_RAISE_PY):
                    auction_price = (
                        COMPANIES[active_company].get_face_value()
                        + int(action_info.amount)
                    )
                    winner = active_player
                else:
                    auction_price = int(pre["auction_price"])
                    winner = post_owner
                self.auction_events.append([
                    move_number,
                    turn_number,
                    active_company,
                    winner,
                    auction_price,
                    int(pre["auction_starter"]),
                    int(pre["auction_high_bidder"]),
                    action_type,
                    action_id,
                ])

        if action_type == int(ACTION_PAR_PY) and 0 <= active_corp < self._NUM_CORPS:
            par_index = int(action_info.amount)
            company_id = active_company
            if (
                0 <= company_id < self._NUM_COMPANIES
                and int(pre_corp_active[active_corp]) == 0
                and int(post_corp_active[active_corp]) == 1
            ):
                par_price = int(ALL_PAR_PRICES[par_index])
                float_shares, _market_idx, player_payment, corp_cash, issued = (
                    CORPS[active_corp].simulate_float(company_id, par_index)
                )
                self.ipo_events.append([
                    move_number,
                    turn_number,
                    active_player,
                    active_corp,
                    company_id,
                    par_index,
                    par_price,
                    int(float_shares),
                    int(player_payment),
                    int(corp_cash),
                    int(issued),
                ])

        if action_type in (
            int(ACTION_ACQ_PRICE_PY),
            int(ACTION_ACQ_OFFER_ACCEPT_PY),
            int(ACTION_ACQ_SELECT_COMPANY_PY),
        ):
            for company_id in range(self._NUM_COMPANIES):
                pre_loc = int(pre_company_locations[company_id])
                pre_owner = int(pre_company_owners[company_id])
                post_loc = int(post_company_locations[company_id])
                post_owner = int(post_company_owners[company_id])
                if post_loc not in (LOC_CORP_INT, LOC_CORP_ACQ_INT):
                    continue
                if pre_loc == post_loc and pre_owner == post_owner:
                    continue
                price = -1
                buyer_corp = post_owner
                if (
                    action_type == int(ACTION_ACQ_PRICE_PY)
                    and company_id == active_company
                ):
                    price = COMPANIES[company_id].get_low_price() + int(action_info.amount)
                    buyer_corp = active_corp
                elif (
                    action_type == int(ACTION_ACQ_OFFER_ACCEPT_PY)
                    and company_id == active_company
                ):
                    price = int(pre["acq_offer_price"])
                    buyer_corp = active_corp
                elif (
                    action_type == int(ACTION_ACQ_SELECT_COMPANY_PY)
                    and company_id == int(action_info.company_id)
                    and active_corp >= 0
                ):
                    buyer_corp = active_corp
                    if pre_loc == LOC_FI_INT:
                        if buyer_corp == self._OS_CORP_ID:
                            price = COMPANIES[company_id].get_face_value()
                        else:
                            price = COMPANIES[company_id].get_high_price()
                if price < 0:
                    continue
                self.acquisition_events.append([
                    move_number,
                    turn_number,
                    active_player,
                    buyer_corp,
                    company_id,
                    price,
                    pre_loc,
                    pre_owner,
                    post_loc,
                    post_owner,
                    action_type,
                    action_id,
                ])

        if (
            action_type in (int(ACTION_BUY_SHARE_PY), int(ACTION_SELL_SHARE_PY))
            and 0 <= int(action_info.corp_id) < self._NUM_CORPS
        ):
            corp_id = int(action_info.corp_id)
            self.share_trade_events.append([
                move_number,
                turn_number,
                active_player,
                corp_id,
                action_type,
                int(pre_player_shares[active_player, corp_id]),
                int(post_player_shares[active_player, corp_id]),
                int(pre_player_cash[active_player]),
                int(post_player_cash[active_player]),
                int(pre_corp_prices[corp_id]),
                int(post_corp_prices[corp_id]),
                action_id,
            ])

        if action_type == int(ACTION_DIVIDEND_PY) and 0 <= active_corp < self._NUM_CORPS:
            self.dividend_events.append([
                move_number,
                turn_number,
                active_player,
                active_corp,
                int(action_info.amount),
                int(pre_corp_cash[active_corp]),
                int(post_corp_cash[active_corp]),
                int(pre_corp_prices[active_corp]),
                int(post_corp_prices[active_corp]),
            ])

        if action_type == int(ACTION_ISSUE_PY) and 0 <= active_corp < self._NUM_CORPS:
            self.issue_events.append([
                move_number,
                turn_number,
                active_player,
                active_corp,
                int(pre_corp_bank[active_corp]),
                int(post_corp_bank[active_corp]),
                int(pre_corp_issued[active_corp]),
                int(post_corp_issued[active_corp]),
                int(pre_corp_cash[active_corp]),
                int(post_corp_cash[active_corp]),
                int(pre_corp_prices[active_corp]),
                int(post_corp_prices[active_corp]),
            ])

        if action_type == int(ACTION_CLOSE_PY):
            for company_id in range(self._NUM_COMPANIES):
                if int(post_company_locations[company_id]) != LOC_REMOVED_INT:
                    continue
                if int(pre_company_locations[company_id]) == LOC_REMOVED_INT:
                    continue
                self.close_events.append([
                    move_number,
                    turn_number,
                    active_player,
                    company_id,
                    int(pre_company_locations[company_id]),
                    int(pre_company_owners[company_id]),
                    action_type,
                    action_id,
                ])

    def finalize(self) -> StrategyTrace:
        return StrategyTrace(
            nn_policy_pct=np.stack(self.nn_policy_pct).astype(np.float32),
            nn_values=np.stack(self.nn_values).astype(np.float32),
            mcts_visit_counts=np.stack(self.mcts_visit_counts).astype(np.int32),
            a0gb_values=np.stack(self.a0gb_values).astype(np.float32),
            mcts_root_values=np.stack(self.mcts_root_values).astype(np.float32),
            selected_action_ids=np.asarray(self.selected_action_ids, dtype=np.uint16),
            selected_unified_slots=np.asarray(self.selected_unified_slots, dtype=np.int16),
            action_types=np.asarray(self.action_types, dtype=np.int16),
            action_corps=np.asarray(self.action_corps, dtype=np.int16),
            action_companies=np.asarray(self.action_companies, dtype=np.int16),
            action_amounts=np.asarray(self.action_amounts, dtype=np.int16),
            engine_phase_ids=np.asarray(self.engine_phase_ids, dtype=np.int8),
            active_players=np.asarray(self.active_players, dtype=np.int8),
            active_corps=np.asarray(self.active_corps, dtype=np.int8),
            active_companies=np.asarray(self.active_companies, dtype=np.int8),
            turn_numbers=np.asarray(self.turn_numbers, dtype=np.int16),
            coo_levels=np.asarray(self.coo_levels, dtype=np.int8),
            cards_remaining=np.asarray(self.cards_remaining, dtype=np.int8),
            auction_prices=np.asarray(self.auction_prices, dtype=np.int16),
            auction_high_bidders=np.asarray(self.auction_high_bidders, dtype=np.int8),
            auction_starters=np.asarray(self.auction_starters, dtype=np.int8),
            acq_offer_prices=np.asarray(self.acq_offer_prices, dtype=np.int16),
            acq_offer_corps=np.asarray(self.acq_offer_corps, dtype=np.int8),
            target_temperatures=np.asarray(self.target_temperatures, dtype=np.float32),
            sample_temperatures=np.asarray(self.sample_temperatures, dtype=np.float32),
            greedy_leaf_depths=np.asarray(self.greedy_leaf_depths, dtype=np.int16),
            root_visit_counts=np.asarray(self.root_visit_counts, dtype=np.int32),
            player_cash=np.stack(self.player_cash).astype(np.int16),
            player_net_worth=np.stack(self.player_net_worth).astype(np.int16),
            player_liquidity=np.stack(self.player_liquidity).astype(np.int16),
            player_income=np.stack(self.player_income).astype(np.int16),
            player_shares=np.stack(self.player_shares).astype(np.int8),
            corp_active=np.stack(self.corp_active).astype(np.int8),
            corp_prices=np.stack(self.corp_prices).astype(np.int16),
            corp_cash=np.stack(self.corp_cash).astype(np.int16),
            corp_income=np.stack(self.corp_income).astype(np.int16),
            corp_presidents=np.stack(self.corp_presidents).astype(np.int8),
            corp_issued_shares=np.stack(self.corp_issued_shares).astype(np.int8),
            corp_bank_shares=np.stack(self.corp_bank_shares).astype(np.int8),
            corp_unissued_shares=np.stack(self.corp_unissued_shares).astype(np.int8),
            corp_receivership=np.stack(self.corp_receivership).astype(np.int8),
            company_locations=np.stack(self.company_locations).astype(np.int8),
            company_owners=np.stack(self.company_owners).astype(np.int8),
            company_adjusted_income=np.stack(self.company_adjusted_income).astype(np.int16),
            auction_events=_strategy_rows(self.auction_events, 9),
            ipo_events=_strategy_rows(self.ipo_events, 11),
            acquisition_events=_strategy_rows(self.acquisition_events, 12),
            share_trade_events=_strategy_rows(self.share_trade_events, 12),
            dividend_events=_strategy_rows(self.dividend_events, 9),
            issue_events=_strategy_rows(self.issue_events, 12),
            close_events=_strategy_rows(self.close_events, 8),
        )


def play_game(
    evaluator: Any,
    config: TrainingConfig,
    game_seed: int,
    rng: np.random.Generator,
    state_pool: StatePool | None = None,
    epoch_config: EpochConfig | None = None,
    num_players: int | None = None,
    collect_strategy_trace: bool = False,
    game_id: int = -1,
    rng_seed: int = -1,
) -> GameRecord:
    """Play one self-play game, returning training examples.

    Args:
        evaluator: NNEvaluator or RemoteEvaluator for leaf evaluation.
        state_pool: Optional pre-allocated StatePool for MCTS node states.
            Reused across searches within the game and across games.
        epoch_config: Per-epoch dynamic parameters (c_puct, value blend).
            If None, uses config defaults (pure A0GB, c_puct_final).
        num_players: Main-process assigned actual player count. Required for
            mixed player-count training.
    """
    t0 = time.perf_counter()

    num_players = _resolve_game_num_players(config, num_players)
    max_players = config.effective_max_players
    state = GameState(num_players, max_players=max_players)
    state.initialize_game(num_players, seed=game_seed, max_players=max_players)

    total_int16_size = get_layout(max_players).total_size

    # Ensure state pool exists for subtree reuse across searches
    if state_pool is None:
        state_pool = StatePool(2 * (config.max_simulations + 1), total_int16_size)
    elif state_pool.states.shape[1] != total_int16_size:
        raise ValueError(
            f"state_pool row width {state_pool.states.shape[1]} does not match "
            f"configured max-player state width {total_int16_size}"
        )

    # Use epoch-specific overrides if provided
    c_puct_override = epoch_config.c_puct if epoch_config is not None else None
    sims_override = (epoch_config.num_simulations if epoch_config is not None
                     and epoch_config.num_simulations > 0 else None)
    mcts_config = config.to_mcts_config(
        c_puct_override=c_puct_override,
        num_simulations_override=sims_override,
        num_players=num_players,
    )

    # Profile stats (None when --profile not set → zero overhead)
    search_stats: SearchStats | None = None
    if config.profile:
        search_stats = SearchStats()
        if hasattr(evaluator, "reset_profile_stats"):
            evaluator.reset_profile_stats()

    examples: list[SelfPlayExample] = []
    target_entropy_sum = 0.0
    target_top1_sum = 0.0
    sample_entropy_sum = 0.0
    sample_top1_sum = 0.0
    move_count = 0
    reuse_root: Any = None

    # Scratch buffer for enumerating legal actions at each decision point.
    # Copied-out per move so the buffer is free to be reused.
    legal_scratch = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
    # Static LUT mapping (phase_id, phase-local action id) → unified slot.
    # Used once per decision to scatter the sparse visit distribution and
    # legal set into dense (UNIFIED_LOGIT_DIM,) rows for the trainer.
    action_lut_np = build_action_lut().numpy()
    trace_builder = (
        _StrategyTraceBuilder(num_players) if collect_strategy_trace else None
    )

    while True:
        phase_id = get_decision_phase_py(state)
        n_legal = enumerate_legal_actions_py(state, legal_scratch)
        legal_actions = legal_scratch[:n_legal].copy()
        slots = action_lut_np[phase_id, legal_actions]

        trace_pre_summary: dict[str, object] | None = None
        if trace_builder is not None:
            nn_priors, nn_values, nn_actions, nn_n_legal, nn_phase_id = (
                evaluator.evaluate(state)
            )
            if (
                nn_phase_id != phase_id
                or nn_n_legal != n_legal
                or not np.array_equal(nn_actions, legal_actions)
            ):
                raise AssertionError(
                    "trace root eval legal actions diverged from self-play "
                    f"enumeration: phase {phase_id}/{nn_phase_id}, "
                    f"n {n_legal}/{nn_n_legal}"
                )
            trace_pre_summary = trace_builder.append_pre_search(
                state, slots, nn_priors, nn_values,
            )

        # MCTS search (reuses subtree from previous move when available).
        # The root's own enumerate inside run_search is deterministic against
        # the same state, so root.visit_counts aligns with legal_actions.
        root = run_search(
            state, evaluator, mcts_config, rng,
            state_pool=state_pool, reuse_root=reuse_root,
            profile=search_stats,
        )

        # Sparse policy target: temperature-shaped visit-count proportions
        # over legal actions. Setting policy_target_temp_* to a constant 1.0
        # recovers raw visit-count targets.
        assert root.visit_counts is not None
        counts = root.visit_counts.astype(np.float32)
        counts_sum = float(counts.sum())
        assert counts_sum > 0.0, "run_search produced zero total visits"
        target_temperature = _compute_policy_target_temperature(
            move_count, config, num_players,
        )
        policy_target_sparse = scale_visit_counts_by_temperature(
            counts, target_temperature,
        )

        # A0GB value target — already canonical (no np.roll).
        value_target = get_greedy_leaf_value(root, num_players)

        # Temperature-scaled sampling distribution over the same sparse list.
        temperature = _compute_temperature(move_count, config, num_players)
        sample_probs = scale_visit_counts_by_temperature(counts, temperature)

        # Stats: keep target and action-sampling distributions separate.
        target_nonzero = policy_target_sparse[policy_target_sparse > 0]
        sample_nonzero = sample_probs[sample_probs > 0]
        target_entropy_sum += float(
            -np.sum(target_nonzero * np.log(target_nonzero))
        )
        target_top1_sum += float(np.max(policy_target_sparse))
        sample_entropy_sum += float(
            -np.sum(sample_nonzero * np.log(sample_nonzero))
        )
        sample_top1_sum += float(np.max(sample_probs))

        # Scatter sparse visit probs + legal set into dense unified-slot
        # rows. The same LUT the model uses to collapse its dense forward
        # back to phase-local ids gives us the inverse mapping here.
        dense_legal_mask = np.zeros(U_DIM, dtype=np.uint8)
        dense_legal_mask[slots] = 1
        dense_policy_target = np.zeros(U_DIM, dtype=np.float32)
        dense_policy_target[slots] = policy_target_sparse

        examples.append(
            SelfPlayExample(
                state=state._array.copy(),
                phase_id=phase_id,
                legal_mask=dense_legal_mask,
                policy_target=dense_policy_target,
                value_target=value_target,
            )
        )

        # Sample and apply action.
        chosen_idx = int(rng.choice(n_legal, p=sample_probs))
        action_idx = int(legal_actions[chosen_idx])
        if trace_builder is not None:
            selected_slot = int(action_lut_np[phase_id, action_idx])
            action_info = decode_action_py(phase_id, action_idx)
            trace_builder.append_search_result(
                root=root,
                slots=slots,
                a0gb_value=value_target,
                action_id=action_idx,
                selected_slot=selected_slot,
                action_info=action_info,
                target_temperature=target_temperature,
                sample_temperature=temperature,
            )
            status = DRIVER.apply_action(state, action_idx)
            assert trace_pre_summary is not None
            trace_builder.append_action_events(
                trace_pre_summary,
                state,
                move_number=move_count,
                action_id=action_idx,
                action_info=action_info,
            )
        else:
            status = DRIVER.apply_action(state, action_idx)
        move_count += 1

        if status == STATUS_GAME_OVER_PY:
            break

        # Extract chosen child's subtree for reuse in next search
        reuse_root = prepare_reuse_root(root, action_idx, state_pool)

    # End-of-game stats via entity handles (GameState no longer exposes them).
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)

    net_worths = [
        PLAYERS[i].get_net_worth(state) for i in range(num_players)
    ]
    shares_per_player = [
        sum(PLAYERS[i].get_shares(state, c) for c in range(num_corps))
        for i in range(num_players)
    ]
    companies_per_player = [
        sum(1 for c in range(num_companies) if PLAYERS[i].owns_company(state, c))
        for i in range(num_players)
    ]

    # Per-player value of shares in corps where they are president
    pres_share_values: list[float] = []
    for i in range(num_players):
        val = 0.0
        for c in range(num_corps):
            if PLAYERS[i].is_president_of(state, c):
                val += PLAYERS[i].get_shares(state, c) * CORPS[c].get_share_price(state)
        pres_share_values.append(val)

    # Net worth component breakdown (% of total)
    nw_cash_pct: list[float] = []
    nw_companies_pct: list[float] = []
    nw_shares_pct: list[float] = []
    for i in range(num_players):
        nw = net_worths[i]
        cash = PLAYERS[i].get_cash(state)
        company_value = sum(
            COMPANIES[c].get_face_value()
            for c in range(num_companies)
            if PLAYERS[i].owns_company(state, c)
        )
        share_value = sum(
            PLAYERS[i].get_shares(state, c) * CORPS[c].get_share_price(state)
            for c in range(num_corps)
        )
        if nw > 0:
            nw_cash_pct.append(cash / nw)
            nw_companies_pct.append(company_value / nw)
            nw_shares_pct.append(share_value / nw)
        else:
            nw_cash_pct.append(0.0)
            nw_companies_pct.append(0.0)
            nw_shares_pct.append(0.0)

    # Corp-level stats
    active_prices: list[int] = []
    corps_in_receivership = 0
    for c in range(num_corps):
        if CORPS[c].is_active(state):
            active_prices.append(CORPS[c].get_share_price(state))
            if CORPS[c].is_in_receivership(state):
                corps_in_receivership += 1
    avg_active_corp_price = sum(active_prices) / len(active_prices) if active_prices else 0.0
    has_max_price_corp = any(p == 75 for p in active_prices)

    # Pre-stack training data into contiguous arrays (5 large arrays instead
    # of N×5 small ones) so pickle through mp.Queue is a fast memcpy. Rows
    # are dense over UNIFIED_LOGIT_DIM; illegal slots are zero in both the
    # mask and the policy target.
    n_examples = len(examples)
    stacked_states = np.empty((n_examples, total_int16_size), dtype=np.int16)
    stacked_phase_ids = np.empty(n_examples, dtype=np.int8)
    stacked_legal_masks = np.empty((n_examples, U_DIM), dtype=np.uint8)
    stacked_policy_targets = np.empty((n_examples, U_DIM), dtype=np.float32)
    stacked_value_targets = np.empty((n_examples, num_players), dtype=np.float32)
    for i, ex in enumerate(examples):
        stacked_states[i] = ex.state
        stacked_phase_ids[i] = ex.phase_id
        stacked_legal_masks[i] = ex.legal_mask
        stacked_policy_targets[i] = ex.policy_target
        stacked_value_targets[i] = ex.value_target

    # Blend A0GB value targets with canonical game outcome if configured.
    # No rotation — compute_terminal_values already returns canonical order.
    blend_alpha = epoch_config.value_blend_alpha if epoch_config is not None else 1.0
    if blend_alpha < 1.0:
        rank_weight = getattr(evaluator, "terminal_rank_weight", 0.5)
        terminal_values = compute_terminal_values(
            net_worths, num_players, rank_weight
        )
        stacked_value_targets = (
            blend_alpha * stacked_value_targets
            + (1.0 - blend_alpha) * terminal_values[None, :]
        ).astype(np.float32)

    game_profile: GameProfileData | None = None
    if config.profile and search_stats is not None:
        eval_client: EvalClientStats | None = None
        if hasattr(evaluator, "get_profile_stats"):
            eval_client = evaluator.get_profile_stats()
        game_profile = GameProfileData(
            search=search_stats,
            eval_client=eval_client,
            game_duration=time.perf_counter() - t0,
        )

    return GameRecord(
        states=stacked_states,
        phase_ids=stacked_phase_ids,
        legal_masks=stacked_legal_masks,
        policy_targets=stacked_policy_targets,
        value_targets=stacked_value_targets,
        num_players=num_players,
        num_examples=n_examples,
        total_moves=move_count,
        net_worths=net_worths,
        shares_per_player=shares_per_player,
        companies_per_player=companies_per_player,
        pres_share_values=pres_share_values,
        nw_cash_pct=nw_cash_pct,
        nw_companies_pct=nw_companies_pct,
        nw_shares_pct=nw_shares_pct,
        avg_active_corp_price=avg_active_corp_price,
        corps_in_receivership=corps_in_receivership,
        has_max_price_corp=has_max_price_corp,
        duration_secs=time.perf_counter() - t0,
        policy_entropy_mean=target_entropy_sum / max(move_count, 1),
        top1_visit_fraction=sample_top1_sum / max(move_count, 1),
        policy_target_entropy_mean=target_entropy_sum / max(move_count, 1),
        policy_target_top1_fraction=target_top1_sum / max(move_count, 1),
        sample_policy_entropy_mean=sample_entropy_sum / max(move_count, 1),
        sample_top1_action_fraction=sample_top1_sum / max(move_count, 1),
        profile=game_profile,
        game_id=game_id,
        game_seed=game_seed,
        rng_seed=rng_seed,
        final_state=(state._array.copy() if trace_builder is not None else None),
        strategy_trace=(
            trace_builder.finalize() if trace_builder is not None else None
        ),
    )


def self_play_worker(
    task_queue: Any,
    result_queue: Any,
    config: TrainingConfig,
    shared_bufs: Any,
    worker_idx: int,
) -> None:
    """Worker process: play games using remote NN evaluation.

    Loops until a None sentinel is received on the task queue
    or the connection breaks (shutdown).
    """
    # Main drives shutdown via None sentinel + eval-server stop_event; Ctrl-C
    # SIGINT delivered to the process group would otherwise interrupt the
    # RemoteEvaluator's Condition.wait() mid lock-reacquire and bubble up as
    # AssertionError from RLock.__exit__.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)  # Prevent OpenMP oversubscription with many workers

    evaluator = RemoteEvaluator(
        config.effective_max_players, shared_bufs, worker_idx,
        profile=config.profile,
        terminal_rank_weight=config.terminal_blend,
    )

    total_size = get_layout(config.effective_max_players).total_size
    state_pool = StatePool(2 * (config.max_simulations + 1), total_size)

    try:
        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if task is None:
                break
            task_len = len(task)
            if task_len == 3:
                game_seed, rng_seed, epoch_config = task
                num_players = None
            elif task_len == 4:
                game_seed, rng_seed, num_players, epoch_config = task
            else:
                raise ValueError(
                    f"self-play task must have 3 or 4 fields, got {task_len}"
                )
            rng = np.random.default_rng(rng_seed)
            record = play_game(
                evaluator, config, game_seed, rng,
                state_pool=state_pool, epoch_config=epoch_config,
                num_players=num_players, rng_seed=rng_seed,
            )
            result_queue.put(record)
    except (KeyboardInterrupt, EOFError, BrokenPipeError, OSError):
        pass
