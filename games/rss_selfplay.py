"""RSS self-play diagnostics (1889/pipeline-refactor.md §4.3).

The stats collector behind ``Game.new_selfplay_stats``, and the strategy trace
behind the RSS-only ``Game.new_strategy_trace``, which ``train/strategy_data.py``
collects through ``play_game(collect_strategy_trace=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np

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
)
from core.data import ALL_PAR_PRICES, CorpIndices, GameConstants, GamePhases
from core.state import GameState
from entities.company import (
    COMPANIES,
    CompanyLocation,
)
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN


LOC_AUCTION_INT = int(CompanyLocation.LOC_AUCTION)
LOC_CORP_INT = int(CompanyLocation.LOC_CORP)
LOC_CORP_ACQ_INT = int(CompanyLocation.LOC_CORP_ACQ)
LOC_FI_INT = int(CompanyLocation.LOC_FI)
LOC_PLAYER_INT = int(CompanyLocation.LOC_PLAYER)
LOC_REMOVED_INT = int(CompanyLocation.LOC_REMOVED)

# Finishing ranks, in train/main.py's TensorBoard spellings.
RANK_LABELS = ("1st", "2nd", "3rd", "4th", "5th", "6th")


# ---------------------------------------------------------------------------
# Self-play stats
# ---------------------------------------------------------------------------

# Seller accept-prior bands splitting rejections the network expected from
# rejections that surprised it.
EXPECTED_REJECTION_ACCEPT_PRIOR = 0.2
SURPRISE_REJECTION_ACCEPT_PRIOR = 0.5


@dataclass
class AcquisitionStats:
    """Real self-play decisions only; search simulations and FI offers are excluded
    from negotiation counts. Phases count turns with an acquisition decision.

    Offers also record the seller's unnoised root prior on accept, which the
    proposer's search saw at the same state, and whether every searched
    proposer decision leading to the offer was search's most-visited action.
    """

    phases: int = 0
    decisions: int = 0
    offers: int = 0
    rejections: int = 0
    cap_hits: int = 0
    accept_prior_sum: float = 0.0
    expected_rejections: int = 0
    surprise_rejections: int = 0
    top_choice_offers: int = 0
    top_choice_rejections: int = 0
    _last_turn: int = -1
    _top_choice: bool = True

    def observe(
        self,
        state: GameState,
        legal_actions: np.ndarray,
        chosen_idx: int,
        priors: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        """Record a played decision; arrays align with ``legal_actions``."""
        phase = TURN.get_phase(state)
        if phase not in (
            GamePhases.PHASE_ACQ_SELECT_CORP, GamePhases.PHASE_ACQ_SELECT_COMPANY,
            GamePhases.PHASE_ACQ_SELECT_PRICE, GamePhases.PHASE_ACQ_OFFER,
        ):
            return
        self.decisions += 1
        turn = TURN.get_turn_number(state)
        if turn != self._last_turn:
            self.phases += 1
            self._last_turn = turn
        top_choice = bool(visit_counts[chosen_idx] >= visit_counts.max())
        # Each proposal starts at SELECT_CORP; company and price steps narrow it.
        if phase == GamePhases.PHASE_ACQ_SELECT_CORP:
            self._top_choice = top_choice
            return
        if phase != GamePhases.PHASE_ACQ_OFFER:
            self._top_choice = self._top_choice and top_choice
            return
        if COMPANIES[TURN.get_active_company(state)].get_location(state) == CompanyLocation.LOC_FI:
            return
        # ACQ_OFFER pass/reject is 0; accept is 1.
        accept_prior = float(priors[np.flatnonzero(legal_actions == 1)[0]])
        self.offers += 1
        self.accept_prior_sum += accept_prior
        self.top_choice_offers += self._top_choice
        if legal_actions[chosen_idx] == 0:
            self.rejections += 1
            self.expected_rejections += accept_prior < EXPECTED_REJECTION_ACCEPT_PRIOR
            self.surprise_rejections += accept_prior > SURPRISE_REJECTION_ACCEPT_PRIOR
            self.top_choice_rejections += self._top_choice
            proposer = CORPS[TURN.get_active_corp(state)].get_president_id(state)
            if PLAYERS[proposer].get_acq_rejections(state) == int(GameConstants.ACQ_REJECTION_CAP) - 1:
                self.cap_hits += 1

    def ratios(self) -> dict[str, tuple[float, float]]:
        """``{scalar: (numerator, denominator)}``; rates over the group's events."""
        return {
            "acq_decisions_per_phase": (self.decisions, self.phases),
            "acq_offers_per_phase": (self.offers, self.phases),
            "acq_offer_acceptance_rate": (self.offers - self.rejections, self.offers),
            "acq_cap_hits_per_phase": (self.cap_hits, self.phases),
            "acq_offer_accept_prior_mean": (self.accept_prior_sum, self.offers),
            "acq_rejections_expected_fraction": (self.expected_rejections, self.rejections),
            "acq_rejections_surprise_fraction": (self.surprise_rejections, self.rejections),
            "acq_offers_top_choice_fraction": (self.top_choice_offers, self.offers),
            "acq_top_choice_acceptance_rate": (
                self.top_choice_offers - self.top_choice_rejections, self.top_choice_offers,
            ),
        }


def player_holdings(state: GameState, num_players: int) -> tuple[list[int], list[int]]:
    """Shares and companies each player holds, in canonical order."""
    num_corps = int(GameConstants.NUM_CORPS)
    num_companies = int(GameConstants.NUM_COMPANIES)
    shares_per_player = [
        sum(PLAYERS[i].get_shares(state, c) for c in range(num_corps))
        for i in range(num_players)
    ]
    companies_per_player = [
        sum(1 for c in range(num_companies) if PLAYERS[i].owns_company(state, c))
        for i in range(num_players)
    ]
    return shares_per_player, companies_per_player


class RSSSelfPlayStats:
    """One self-play game's acquisition, share, and company stats.

    ``finish`` reports per-game values as means over games (denominator 1),
    per-player values by finishing rank (mean over the games with that rank),
    and acquisition stats as rates over the group's decisions and offers.
    """

    def __init__(self, num_players: int) -> None:
        self.num_players = num_players
        self.acquisition = AcquisitionStats()

    def observe(
        self,
        state: GameState,
        legal_actions: np.ndarray,
        chosen_idx: int,
        priors: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        self.acquisition.observe(state, legal_actions, chosen_idx, priors, visit_counts)

    def finish(self, state: GameState) -> dict[str, tuple[float, float]]:
        # End-of-game stats via entity handles (GameState no longer exposes them).
        num_players = self.num_players
        num_corps = int(GameConstants.NUM_CORPS)
        num_companies = int(GameConstants.NUM_COMPANIES)

        net_worths = [
            PLAYERS[i].get_net_worth(state) for i in range(num_players)
        ]
        invest_roundtrip_cap_hits = [
            PLAYERS[i].get_invest_roundtrip_cap_hits(state) for i in range(num_players)
        ]
        shares_per_player, companies_per_player = player_holdings(state, num_players)

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

        stats = self.acquisition.ratios()
        stats.update({
            "invest_roundtrip_cap_hits_per_game": (sum(invest_roundtrip_cap_hits), 1),
            "total_shares": (float(sum(shares_per_player)), 1),
            "total_companies": (float(sum(companies_per_player)), 1),
            "avg_active_corp_price": (float(avg_active_corp_price), 1),
            "corps_in_receivership": (float(corps_in_receivership), 1),
            "pct_games_max_price_corp": (float(has_max_price_corp), 1),
        })
        ranked = sorted(range(num_players), key=lambda p: net_worths[p], reverse=True)
        for rank, player_id in enumerate(ranked):
            label = RANK_LABELS[rank]
            stats[f"invest_roundtrip_cap_hits_{label}"] = (invest_roundtrip_cap_hits[player_id], 1)
            stats[f"shares_{label}"] = (float(shares_per_player[player_id]), 1)
            stats[f"companies_{label}"] = (float(companies_per_player[player_id]), 1)
            stats[f"pres_share_value_{label}"] = (float(pres_share_values[player_id]), 1)
            stats[f"nw_cash_pct_{label}"] = (float(nw_cash_pct[player_id]), 1)
            stats[f"nw_companies_pct_{label}"] = (float(nw_companies_pct[player_id]), 1)
            stats[f"nw_shares_pct_{label}"] = (float(nw_shares_pct[player_id]), 1)
        return stats


# ---------------------------------------------------------------------------
# Strategy trace
# ---------------------------------------------------------------------------


class _StateSummary(TypedDict):
    """Scalar turn fields and entity arrays captured for a strategy trace."""

    engine_phase: int
    active_player: int
    active_corp: int
    active_company: int
    turn_number: int
    coo_level: int
    cards_remaining: int
    auction_price: int
    auction_high_bidder: int
    auction_starter: int
    acq_offer_price: int
    acq_offer_corp: int
    player_cash: np.ndarray
    player_net_worth: np.ndarray
    player_liquidity: np.ndarray
    player_income: np.ndarray
    player_shares: np.ndarray
    corp_active: np.ndarray
    corp_prices: np.ndarray
    corp_cash: np.ndarray
    corp_income: np.ndarray
    corp_presidents: np.ndarray
    corp_issued_shares: np.ndarray
    corp_bank_shares: np.ndarray
    corp_unissued_shares: np.ndarray
    corp_receivership: np.ndarray
    company_locations: np.ndarray
    company_owners: np.ndarray
    company_adjusted_income: np.ndarray


@dataclass
class StrategyTrace:
    """Optional per-decision trace payload for offline strategy analysis.

    Arrays are aligned row-for-row with ``GameRecord.states``. Policy arrays
    are percentages in dense unified-slot space, not raw probabilities.
    Event arrays use move-relative rows; the shard writer prepends game IDs.
    """

    nn_policy_pct: np.ndarray  # (num_examples, unified_dim), float32
    nn_values: np.ndarray  # (num_examples, num_players), float32
    mcts_visit_counts: np.ndarray  # (num_examples, unified_dim), int32
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


def _strategy_empty_events(width: int) -> np.ndarray:
    return np.empty((0, width), dtype=np.int32)


def _strategy_rows(rows: list[list[int]], width: int) -> np.ndarray:
    if not rows:
        return _strategy_empty_events(width)
    return np.asarray(rows, dtype=np.int32).reshape((-1, width))


class StrategyTraceBuilder:
    """Accumulate per-move analysis traces for one self-play game.

    Per decision, ``play_game`` calls ``append_pre_search`` before searching,
    ``append_search_result`` once it has chosen an action, and
    ``append_action_events`` after applying it; ``finalize`` builds the trace.
    """

    _NUM_CORPS = int(GameConstants.NUM_CORPS)
    _NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)
    _OS_CORP_ID = int(CorpIndices.CORP_OS)

    def __init__(self, num_players: int, unified_dim: int) -> None:
        self.num_players = int(num_players)
        self.unified_dim = int(unified_dim)
        # The decision in progress: its pre-search summary and chosen action.
        self._pre: _StateSummary | None = None
        self._action: tuple[int, Any] | None = None

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

    def capture_summary(self, state: GameState) -> _StateSummary:
        num_players = self.num_players
        num_corps = self._NUM_CORPS
        num_companies = self._NUM_COMPANIES

        summary: _StateSummary = {
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
    ) -> None:
        summary = self.capture_summary(state)
        self._pre = summary

        dense_nn = np.zeros(self.unified_dim, dtype=np.float32)
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

        self.player_cash.append(summary["player_cash"])
        self.player_net_worth.append(summary["player_net_worth"])
        self.player_liquidity.append(summary["player_liquidity"])
        self.player_income.append(summary["player_income"])
        self.player_shares.append(summary["player_shares"])
        self.corp_active.append(summary["corp_active"])
        self.corp_prices.append(summary["corp_prices"])
        self.corp_cash.append(summary["corp_cash"])
        self.corp_income.append(summary["corp_income"])
        self.corp_presidents.append(summary["corp_presidents"])
        self.corp_issued_shares.append(summary["corp_issued_shares"])
        self.corp_bank_shares.append(summary["corp_bank_shares"])
        self.corp_unissued_shares.append(summary["corp_unissued_shares"])
        self.corp_receivership.append(summary["corp_receivership"])
        self.company_locations.append(summary["company_locations"])
        self.company_owners.append(summary["company_owners"])
        self.company_adjusted_income.append(summary["company_adjusted_income"])

    def append_search_result(
        self,
        *,
        root: Any,
        slots: np.ndarray,
        a0gb_value: np.ndarray,
        greedy_leaf_depth: int,
        phase_id: int,
        action_id: int,
        selected_slot: int,
        target_temperature: float,
        sample_temperature: float,
    ) -> None:
        assert root.visit_counts is not None
        action_info = decode_action_py(phase_id, action_id)
        self._action = (action_id, action_info)
        dense_counts = np.zeros(self.unified_dim, dtype=np.int32)
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
        self.greedy_leaf_depths.append(int(greedy_leaf_depth))
        self.root_visit_counts.append(int(root.visit_count))

    def append_action_events(
        self,
        post_state: GameState,
        *,
        move_number: int,
    ) -> None:
        pre, action = self._pre, self._action
        assert pre is not None and action is not None
        action_id, action_info = action
        self._pre = self._action = None
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
