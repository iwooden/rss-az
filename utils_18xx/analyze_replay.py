"""Analyze an 18xx.games Rolling Stock Stars replay with a policy/value model.

Usage:
    .venv/bin/python -m utils_18xx.analyze_replay GAME_JSON CHECKPOINT [options]

    .venv/bin/python -m utils_18xx.analyze_replay tests/games_18xx/data/224967.json new
    .venv/bin/python -m utils_18xx.analyze_replay game.json latest --checkpoint-dir checkpoints
    .venv/bin/python -m utils_18xx.analyze_replay game.json checkpoints/checkpoint_epoch_0045.pt --output replay.html
    .venv/bin/python -m utils_18xx.analyze_replay game.json new --format markdown --output replay.md
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_PASS_PY as ACTION_PASS,
    get_decision_phase_py,
)
from core.data import GamePhases
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.state import GameState
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, compute_terminal_values
from nn import create_model, get_model_input_spec
from train.analyze_game import (
    _apply_player_names,
    _format_checkpoint_path,
    _format_nn_eval,
)
from train.checkpoint import find_latest_checkpoint, load_model_from_checkpoint
from train.config import TrainingConfig
from train.debug_trace import (
    PHASE_NAMES,
    format_action,
    format_phase_context,
    format_state_full,
)
from utils_18xx.action_parser import (
    find_legal_action,
    filter_actions,
    flatten_auto_actions,
    get_legal_actions,
    map_action,
    map_bid_action,
    map_par_action,
)
from utils_18xx.game_session import ACQ_PHASES, PHASE_CLO, GameSession
from utils_18xx.replay_state import (
    initialize_replay_state,
    is_representable_acquisition_offer,
    settle_to_player_choice,
)

from core.data import COMPANY_NAME_TO_ID, CORP_NAME_TO_ID
from entities.company import COMPANIES


@dataclass
class DecisionEval:
    """One model evaluation before applying a replayed engine action."""

    step: int
    action_id: int
    source_action: dict[str, Any]
    note: str
    turn: int
    active_player: int
    engine_phase: int
    phase_id: int
    phase_context: str
    state_array: np.ndarray
    priors: np.ndarray
    values: np.ndarray
    legal_action_ids: np.ndarray
    recorded_action_id: int
    recorded_action_text: str
    recorded_prior: float | None
    recorded_rank: int | None


@dataclass
class ReplaySummary:
    """Result of replay instrumentation before Markdown rendering."""

    game_data: dict[str, Any]
    initial_record: dict[str, Any]
    final_state: GameState
    decisions: list[DecisionEval]
    replay_notes: list[str]


@dataclass(frozen=True)
class ActionGroup:
    """Recorded engine decisions grouped under one 18xx action."""

    action_id: int
    action: dict[str, Any] | None
    decisions: list[DecisionEval]


class ReplayAnalyzerSession(GameSession):
    """GameSession variant that records NN evals before replayed decisions."""

    def __init__(
        self,
        evaluator: NNEvaluator,
        *,
        num_players: int,
        max_players: int,
    ) -> None:
        super().__init__(num_players=num_players, max_players=max_players)
        self.evaluator = evaluator
        self.decisions: list[DecisionEval] = []
        self.replay_notes: list[str] = []

    def sync_with_analysis(self, game_data: dict[str, Any]) -> ReplaySummary:
        """Replay the game and return final state plus recorded evals."""
        self.decisions = []
        self.replay_notes = []

        gid = game_data.get("id", "")
        initial_record = None
        if gid != self.game_id:
            initial_record = self._init_game_metadata(game_data)
        if initial_record is None:
            initial_record = self._run_extractor(game_data)
            self.committed_ids = set(initial_record.get("committed_action_ids", []))
        else:
            self.committed_ids = set(initial_record.get("committed_action_ids", []))

        if "player_order" in initial_record:
            self._player_ids = list(initial_record["player_order"])

        state = initialize_replay_state(
            self.num_players,
            self._deck_order,
            self._offering,
            max_players=self.max_players,
            cost_level=initial_record.get("cost_level"),
        )
        self.state = state

        raw_actions = game_data.get("actions", [])
        actions = filter_actions(raw_actions, self.committed_ids or None)
        actions = flatten_auto_actions(actions)

        idx = 0
        while idx < len(actions):
            action = actions[idx]
            self._align_to_action_logged(state, action)

            phase = TURN.get_phase(state)
            if phase == GamePhases.PHASE_GAME_OVER:
                break

            if phase in ACQ_PHASES:
                next_type = action.get("type")
                if next_type in ("offer", "respond", "pass"):
                    idx = self._sync_acq_round(state, actions, idx)
                else:
                    self._drain_offer_phases_logged(
                        state,
                        action,
                        note="drain acquisition/closing before next 18xx action",
                    )
                continue

            if phase == PHASE_CLO:
                next_type = action.get("type")
                if next_type in ("sell_company", "close", "pass"):
                    idx = self._sync_clo_round(state, actions, idx)
                else:
                    self._drain_offer_phases_logged(
                        state,
                        action,
                        note="drain closing before next 18xx action",
                    )
                continue

            engine_action = map_action(state, action, phase, self.layout)
            if engine_action is None:
                idx += 1
                continue

            cash_snapshot = self._share_price_cash_snapshot(state, action)
            share_owner_snapshot = self._share_owner_snapshot(game_data, state, action)
            result = self._apply_logged_engine_action(
                state,
                int(engine_action),
                action,
                note="18xx mapped action",
            )
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid action {engine_action} at action stream index {idx}, "
                    f"phase={phase}, 18xx_type={action.get('type')}"
                )
            self._apply_share_price_cash_adjustment(state, action, cash_snapshot)
            self._apply_share_owner_adjustment(state, action, share_owner_snapshot)
            result = self._apply_split_action_followup_logged(state, action, phase)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid split follow-up for action stream index {idx}, "
                    f"phase={phase}, 18xx_type={action.get('type')}"
                )

            idx += 1

        if self._should_drain_trailing_offer_phases(game_data, state):
            self._drain_offer_phases_logged(
                state,
                None,
                note="trailing acquisition/closing drain",
            )

        return ReplaySummary(
            game_data=game_data,
            initial_record=initial_record,
            final_state=state,
            decisions=list(self.decisions),
            replay_notes=list(self.replay_notes),
        )

    def _record_decision(
        self,
        state: GameState,
        action_idx: int,
        source_action: dict[str, Any] | None,
        *,
        note: str,
    ) -> None:
        phase_id = get_decision_phase_py(state)
        if phase_id < 0:
            self.replay_notes.append(
                f"Skipped model eval in non-decision phase {TURN.get_phase(state)}"
            )
            return

        priors, values, action_ids, _n_legal, eval_phase_id = self.evaluator.evaluate(state)
        action_ids_i = action_ids.astype(np.int64, copy=False)
        matches = np.flatnonzero(action_ids_i == int(action_idx))
        if len(matches):
            prior_idx = int(matches[0])
            recorded_prior = float(priors[prior_idx])
            sorted_indices = np.argsort(-priors)
            recorded_rank = int(np.flatnonzero(sorted_indices == prior_idx)[0] + 1)
        else:
            recorded_prior = None
            recorded_rank = None

        source = dict(source_action or {})
        action_id = _source_action_id(source)
        decision = DecisionEval(
            step=len(self.decisions),
            action_id=action_id,
            source_action=source,
            note=note,
            turn=int(TURN.get_turn_number(state)),
            active_player=int(TURN.get_active_player(state)),
            engine_phase=int(TURN.get_phase(state)),
            phase_id=int(eval_phase_id),
            phase_context=format_phase_context(state),
            state_array=state._array.copy(),
            priors=priors,
            values=values,
            legal_action_ids=action_ids.copy(),
            recorded_action_id=int(action_idx),
            recorded_action_text=format_action(eval_phase_id, int(action_idx), state),
            recorded_prior=recorded_prior,
            recorded_rank=recorded_rank,
        )
        self.decisions.append(decision)

    def _apply_logged_engine_action(
        self,
        state: GameState,
        action_idx: int,
        source_action: dict[str, Any] | None,
        *,
        note: str,
    ) -> int:
        self._record_decision(state, action_idx, source_action, note=note)
        return DRIVER.apply_action(state, action_idx)

    def _align_to_action_logged(
        self,
        state: GameState,
        action: dict[str, Any],
    ) -> bool:
        """Apply omitted forced choices until ``action`` becomes mappable."""
        advanced = False
        while True:
            settle_to_player_choice(state)
            if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
                return advanced

            try:
                engine_action = map_action(state, action, TURN.get_phase(state), self.layout)
            except (ValueError, KeyError, IndexError):
                engine_action = None

            if engine_action is not None:
                return advanced

            forced_action = _single_legal_action(state)
            if forced_action is None:
                return advanced

            result = self._apply_logged_engine_action(
                state,
                forced_action,
                action,
                note="forced replay alignment before 18xx action",
            )
            if result == STATUS_INVALID:
                raise RuntimeError(f"Invalid forced replay action {forced_action}")
            advanced = True
            if result in (STATUS_GAME_OVER, STATUS_PAUSED):
                return advanced

    def _apply_split_action_followup_logged(
        self,
        state: GameState,
        action: dict[str, Any],
        initial_phase: int,
    ) -> int:
        """Apply the second engine step for one-action 18xx split actions."""
        atype = action.get("type")
        phase = TURN.get_phase(state)

        if (
            initial_phase == GamePhases.PHASE_INVEST
            and atype == "bid"
            and phase == GamePhases.PHASE_BID
        ):
            if self._bid_followup_already_applied(state, action):
                return STATUS_OK
            return self._apply_logged_engine_action(
                state,
                map_bid_action(state, action, self.layout),
                action,
                note="18xx bid follow-up",
            )

        if (
            initial_phase == GamePhases.PHASE_IPO
            and atype == "par"
            and phase == GamePhases.PHASE_PAR
        ):
            return self._apply_logged_engine_action(
                state,
                map_par_action(state, action, self.layout),
                action,
                note="18xx par-price follow-up",
            )

        return STATUS_OK

    def _begin_acq_offer(
        self,
        state: GameState,
        offer: dict,
        deferred_transfers: list[tuple[int, int, int]],
    ) -> bool:
        """Replay an offer up to ACQ_OFFER, recording each engine choice."""
        buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
        company_id = COMPANY_NAME_TO_ID[offer["company"]]
        price = int(offer["price"])

        if not is_representable_acquisition_offer(state, buyer_corp_id, company_id):
            deferred_transfers.append((buyer_corp_id, company_id, price))
            return False

        state_snapshot = state._array.copy()
        deferred_len = len(deferred_transfers)
        decision_len = len(self.decisions)

        def rollback_speculative_offer() -> None:
            state._array[:] = state_snapshot
            del deferred_transfers[deferred_len:]
            del self.decisions[decision_len:]

        for _ in range(50):
            if COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id):
                return False
            if COMPANIES[company_id].is_in_corp_acquisition(state, buyer_corp_id):
                return False

            phase = TURN.get_phase(state)
            if phase == GamePhases.PHASE_ACQ_OFFER:
                return True
            if phase not in ACQ_PHASES:
                return False

            try:
                action_idx = map_action(state, offer, phase, self.layout)
            except (ValueError, KeyError, IndexError):
                rollback_speculative_offer()
                return False
            if action_idx is None:
                rollback_speculative_offer()
                return False

            result = self._apply_logged_engine_action(
                state,
                int(action_idx),
                offer,
                note="18xx acquisition offer setup",
            )
            if result == STATUS_INVALID:
                rollback_speculative_offer()
                raise RuntimeError(
                    "Invalid ACQ offer replay action while opening offer "
                    f"phase={phase} corporation={offer.get('corporation')} "
                    f"company={offer.get('company')}"
                )

        rollback_speculative_offer()
        raise RuntimeError("Exceeded ACQ offer opening iteration limit")

    def _resolve_acq_offer(
        self,
        state: GameState,
        offer: dict,
        *,
        accepted: bool,
        deferred_transfers: list[tuple[int, int, int]],
    ) -> None:
        """Resolve a single ACQ offer, recording represented engine choices."""
        buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
        company_id = COMPANY_NAME_TO_ID[offer["company"]]
        price = int(offer["price"])

        if not is_representable_acquisition_offer(state, buyer_corp_id, company_id):
            if accepted:
                deferred_transfers.append((buyer_corp_id, company_id, price))
            return

        matched = self._replay_acquisition_offer_logged(
            state,
            offer,
            buyer_corp_id,
            company_id,
            price,
            accept=accepted,
        )
        if accepted and not matched:
            deferred_transfers.append((buyer_corp_id, company_id, price))

    def _replay_acquisition_offer_logged(
        self,
        state: GameState,
        source_action: dict[str, Any],
        buyer_corp_id: int,
        company_id: int,
        price: int,
        *,
        accept: bool,
        max_iterations: int = 200,
    ) -> bool:
        """Logged variant of replay_state.replay_acquisition_offer."""
        offer_action = {
            "type": "offer",
            "corporation": source_action["corporation"],
            "company": source_action["company"],
            "price": price,
            "id": source_action.get("id", -1),
            "entity": source_action.get("entity"),
            "entity_type": source_action.get("entity_type"),
        }

        for _ in range(max_iterations):
            if COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id):
                return True
            if COMPANIES[company_id].is_in_corp_acquisition(state, buyer_corp_id):
                return True

            settle_to_player_choice(state)
            phase = TURN.get_phase(state)

            if phase in (
                GamePhases.PHASE_ACQ_SELECT_CORP,
                GamePhases.PHASE_ACQ_SELECT_COMPANY,
                GamePhases.PHASE_ACQ_SELECT_PRICE,
            ):
                try:
                    action_id = map_action(state, offer_action, phase, self.layout)
                except (ValueError, KeyError, IndexError):
                    try:
                        action_id = find_legal_action(state, action_type=ACTION_PASS)
                    except ValueError:
                        return False
                    result = self._apply_logged_engine_action(
                        state,
                        int(action_id),
                        source_action,
                        note="18xx acquisition offer pass-through",
                    )
                    if result == STATUS_INVALID:
                        return False
                    continue

                if action_id is None:
                    return False

                result = self._apply_logged_engine_action(
                    state,
                    int(action_id),
                    source_action,
                    note="18xx acquisition offer",
                )
                if result == STATUS_INVALID:
                    return False
                if COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id):
                    return True
                if COMPANIES[company_id].is_in_corp_acquisition(state, buyer_corp_id):
                    return True
                if TURN.get_phase(state) not in ACQ_PHASES:
                    return True
                continue

            if phase == GamePhases.PHASE_ACQ_OFFER:
                try:
                    action_id = find_legal_action(
                        state,
                        action_type=ACTION_ACQ_OFFER_ACCEPT if accept else ACTION_PASS,
                    )
                except ValueError:
                    return False
                result = self._apply_logged_engine_action(
                    state,
                    int(action_id),
                    source_action,
                    note="18xx acquisition offer response",
                )
                return result != STATUS_INVALID

            return False

        raise RuntimeError("Exceeded ACQ replay iteration limit")

    def _apply_acq_pass(self, state: GameState, action: dict | None = None) -> None:
        if (
            action is not None
            and TURN.get_phase(state) == GamePhases.PHASE_ACQ_SELECT_CORP
        ):
            player_idx = self._player_index_for_action_entity(action)
            if player_idx is not None:
                TURN.set_active_player(state, player_idx)
                TURN.clear_acquisition_context(state)

        try:
            action_idx = find_legal_action(state, action_type=ACTION_PASS)
        except ValueError:
            return

        result = self._apply_logged_engine_action(
            state,
            int(action_idx),
            action,
            note="18xx acquisition pass",
        )
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid ACQ pass replay action "
                f"phase={TURN.get_phase(state)}"
            )

    def _apply_acq_offer_response(self, state: GameState, action: dict) -> None:
        accepted = str(action.get("accept", "")).lower() == "true"
        try:
            action_idx = find_legal_action(
                state,
                action_type=(
                    ACTION_ACQ_OFFER_ACCEPT
                    if accepted
                    else ACTION_PASS
                ),
            )
        except ValueError as exc:
            raise RuntimeError(
                "Failed to replay ACQ offer response "
                f"corporation={action.get('corporation')} "
                f"company={action.get('company')} accept={action.get('accept')}"
            ) from exc

        result = self._apply_logged_engine_action(
            state,
            int(action_idx),
            action,
            note="18xx acquisition response",
        )
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid ACQ offer response replay action "
                f"phase={TURN.get_phase(state)}"
            )

    def _apply_clo_pass(self, state: GameState, action: dict | None = None) -> None:
        player_idx = (
            self._player_index_for_action_entity(action)
            if action is not None
            else None
        )
        if player_idx is not None:
            TURN.set_active_player(state, player_idx)

        try:
            action_idx = find_legal_action(state, action_type=ACTION_PASS)
        except ValueError:
            return

        result = self._apply_logged_engine_action(
            state,
            int(action_idx),
            action,
            note="18xx closing pass",
        )
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid CLO pass replay action "
                f"phase={TURN.get_phase(state)}"
            )

    def _apply_clo_close(self, state: GameState, action: dict) -> bool:
        player_idx = self._player_index_for_action_entity(action)
        if player_idx is not None:
            TURN.set_active_player(state, player_idx)

        company_id = COMPANY_NAME_TO_ID[action["company"]]
        try:
            action_idx = find_legal_action(
                state,
                action_type=ACTION_CLOSE,
                company_id=company_id,
            )
        except ValueError:
            return False

        result = self._apply_logged_engine_action(
            state,
            int(action_idx),
            action,
            note="18xx closing action",
        )
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid CLO close replay action "
                f"company={action.get('company')}"
            )
        return True

    def _drain_offer_phases_logged(
        self,
        state: GameState,
        source_action: dict[str, Any] | None,
        *,
        note: str,
        max_iterations: int = 500,
    ) -> None:
        for _ in range(max_iterations):
            settle_to_player_choice(state)
            phase = TURN.get_phase(state)

            if phase not in (*ACQ_PHASES, PHASE_CLO):
                return

            try:
                pass_id = find_legal_action(state, action_type=ACTION_PASS)
            except ValueError:
                forced_action = _single_legal_action(state)
                if forced_action is None:
                    return
                result = self._apply_logged_engine_action(
                    state,
                    forced_action,
                    source_action,
                    note=note,
                )
            else:
                result = self._apply_logged_engine_action(
                    state,
                    int(pass_id),
                    source_action,
                    note=note,
                )

            if result == STATUS_INVALID:
                raise RuntimeError(f"Invalid replay drain action in phase={phase}")

        raise RuntimeError("Exceeded ACQ/CLO drain iteration limit")


def _single_legal_action(state: GameState) -> int | None:
    actions = get_legal_actions(state)
    if len(actions) == 1:
        return int(actions[0][0])
    return None


def _source_action_id(action: dict[str, Any]) -> int:
    try:
        action_id = int(action.get("id", -1))
    except (TypeError, ValueError):
        action_id = -1
    if action_id >= 0:
        return action_id
    try:
        return int(action.get("_auto_parent_id", -1))
    except (TypeError, ValueError):
        return -1


def _player_names_from_game_data(
    game_data: dict[str, Any],
    initial_record: dict[str, Any],
) -> list[str]:
    by_id = {str(player.get("id")): str(player.get("name")) for player in game_data.get("players", [])}
    player_order = initial_record.get("player_order")
    if player_order is None:
        player_order = [player.get("id") for player in initial_record.get("players", [])]
    return [by_id.get(str(player_id), f"P{idx}") for idx, player_id in enumerate(player_order)]


def _format_18xx_action(action: dict[str, Any] | None, action_id: int) -> str:
    if action is None:
        return f"implicit engine action {action_id}"

    atype = action.get("type", "?")
    entity = action.get("entity")
    entity_type = action.get("entity_type")
    parts = [f"type={atype}"]
    if entity is not None:
        parts.append(f"{entity_type or 'entity'}={entity}")
    for key in ("corporation", "company", "shares", "price", "amount", "accept"):
        if key in action:
            parts.append(f"{key}={action[key]}")
    if "_auto_parent_type" in action:
        parts.append(f"auto_parent={action.get('_auto_parent_type')}")
    return f"18xx action {action_id}: " + ", ".join(parts)


def _action_lookup(summary: ReplaySummary) -> dict[int, dict[str, Any]]:
    committed = set(summary.initial_record.get("committed_action_ids", [])) or None
    return {
        _source_action_id(action): action
        for action in flatten_auto_actions(
            filter_actions(summary.game_data.get("actions", []), committed)
        )
        if _source_action_id(action) >= 0
    }


def _group_decisions(summary: ReplaySummary) -> list[ActionGroup]:
    action_lookup = _action_lookup(summary)
    grouped: list[ActionGroup] = []
    for decision in summary.decisions:
        if not grouped or grouped[-1].action_id != decision.action_id:
            action = action_lookup.get(decision.action_id, decision.source_action)
            grouped.append(ActionGroup(decision.action_id, action, [decision]))
        else:
            grouped[-1].decisions.append(decision)
    return grouped


def _policy_entropy(priors: np.ndarray) -> tuple[float, float]:
    positive = priors[priors > 0]
    if len(positive) == 0:
        return 0.0, 0.0
    entropy = float(-(positive * np.log(positive)).sum())
    normalizer = float(np.log(len(priors))) if len(priors) > 1 else 0.0
    normalized = entropy / normalizer if normalizer > 0 else 0.0
    return entropy, normalized


def _phase_name_for_group(group: ActionGroup) -> str:
    action = group.action or {}
    atype = action.get("type")
    if atype == "dividend":
        return "DIVIDENDS"
    if atype == "sell_shares" and action.get("entity_type") == "corporation":
        return "ISSUE_SHARES"
    if atype == "par":
        return "IPO/PAR"
    if atype in {"offer", "respond"}:
        return "ACQUISITION"
    if atype in {"sell_company", "close"}:
        return "CLOSING"
    if not group.decisions:
        return "UNKNOWN"
    return PHASE_NAMES.get(group.decisions[0].engine_phase, str(group.decisions[0].engine_phase))


def _turn_phase_label(group: ActionGroup) -> str:
    if not group.decisions:
        return f"#{group.action_id}"
    return f"T{group.decisions[0].turn:02d} {_phase_name_for_group(group)} #{group.action_id}"


def _action_actor(
    group: ActionGroup,
    player_names: list[str],
) -> str:
    action = group.action or {}
    entity = action.get("entity")

    if group.decisions:
        active_player = group.decisions[0].active_player
        if 0 <= active_player < len(player_names):
            return player_names[active_player]

    if action.get("corporation") is not None:
        return str(action["corporation"])
    if entity is not None:
        return str(entity)
    return "-"


def _action_actor_from_initial(
    group: ActionGroup,
    player_names: list[str],
    initial_record: dict[str, Any],
) -> str:
    action = group.action or {}
    player_order = initial_record.get("player_order")
    if player_order is None:
        player_order = [player.get("id") for player in initial_record.get("players", [])]

    for key in ("user", "entity"):
        if key == "entity" and action.get("entity_type") != "player":
            continue
        if action.get(key) is None:
            continue
        target = str(action.get(key))
        for idx, player_id in enumerate(player_order):
            if str(player_id) == target and idx < len(player_names):
                return player_names[idx]

    if action.get("entity_type") == "corporation" and action.get("entity") is not None:
        return str(action.get("entity"))

    return _action_actor(group, player_names)


def _compact_action_description(group: ActionGroup) -> str:
    action = group.action or {}
    atype = action.get("type")
    if atype == "bid":
        price = action.get("price")
        company = action.get("company")
        return f"BID ${price}{f' on {company}' if company else ''}"
    if atype == "pass":
        return "PASS"
    if atype == "par":
        corp = action.get("corporation")
        company = action.get("company")
        last = group.decisions[-1].recorded_action_text if group.decisions else ""
        return last or f"PAR {company or ''} -> {corp or ''}".strip()
    if atype == "offer":
        corp = action.get("corporation", "?")
        company = action.get("company", "?")
        price = action.get("price", "?")
        return f"OFFER {corp} -> {company} @ ${price}"
    if atype == "respond":
        answer = "ACCEPT" if str(action.get("accept", "")).lower() == "true" else "DECLINE"
        return f"{answer} OFFER"
    if atype == "dividend":
        return f"DIVIDEND ${action.get('amount', '?')}"
    if atype == "sell_shares":
        if action.get("entity_type") == "corporation":
            return "ISSUE SHARES"
        shares = action.get("shares") or []
        return f"SELL {shares[0] if shares else 'SHARE'}"
    if atype == "buy_shares":
        shares = action.get("shares") or []
        if shares:
            return f"BUY {shares[0]}"
        if action.get("corporation"):
            return f"BUY {action['corporation']} SHARE"
        return "BUY SHARE"
    if atype in {"sell_company", "close"}:
        return f"CLOSE {action.get('company') or action.get('entity') or ''}".strip()
    if group.decisions:
        if len(group.decisions) == 1:
            return group.decisions[0].recorded_action_text
        return " -> ".join(decision.recorded_action_text for decision in group.decisions)
    return str(atype or "ACTION")


def _group_average_values(group: ActionGroup, num_players: int) -> list[float]:
    if not group.decisions:
        return [0.0 for _ in range(num_players)]
    values = np.stack([decision.values[:num_players] for decision in group.decisions])
    return [float(v) for v in values.mean(axis=0)]


def _fmt_value(value: float) -> str:
    return f"{value:+.3f}"


def _html(text: object) -> str:
    return html_lib.escape(str(text), quote=True)


def _plain_phase_context(text: str) -> str:
    return text.replace("**", "")


def _format_recorded_prior(decision: DecisionEval) -> str:
    if decision.recorded_prior is None or decision.recorded_rank is None:
        return "not present in legal list"
    return (
        f"{decision.recorded_prior:6.1%}, "
        f"rank {decision.recorded_rank}/{len(decision.legal_action_ids)}"
    )


def _append_decision_lines(
    lines: list[str],
    decision: DecisionEval,
    *,
    num_players: int,
    max_players: int,
    top_n: int,
) -> None:
    phase_name = PHASE_NAMES.get(decision.engine_phase, str(decision.engine_phase))
    decision_state = GameState.from_array(
        decision.state_array.copy(),
        num_players,
        max_players=max_players,
    )
    lines.append(
        f"#### Engine Step {decision.step}: P{decision.active_player} "
        f"[{phase_name}]"
    )
    lines.append("")
    lines.append(f"  Turn: {decision.turn} | Replay note: {decision.note}")
    if decision.phase_context:
        lines.append(f"  {decision.phase_context}")
    lines.append("")
    lines.extend(
        _format_nn_eval(
            decision.priors,
            decision.values,
            decision.legal_action_ids,
            decision.phase_id,
            num_players,
            decision_state,
            top_n,
        )
    )
    lines.append("")
    lines.append(
        f"  **Recorded engine action: {decision.recorded_action_text}** "
        f"(NN prior {_format_recorded_prior(decision)})"
    )
    lines.append("")


def format_replay_markdown(
    summary: ReplaySummary,
    *,
    checkpoint_path: str | Path,
    config: TrainingConfig,
    top_n: int,
    verbose: bool,
) -> str:
    game_data = summary.game_data
    initial = summary.initial_record
    num_players = len(game_data.get("players", []))
    player_names = _player_names_from_game_data(game_data, initial)
    initial_state = initialize_replay_state(
        num_players,
        initial.get("deck_order", []),
        initial.get("initial_offering", []),
        max_players=config.effective_max_players,
        cost_level=initial.get("cost_level"),
    )
    lines: list[str] = [
        f"# 18xx Replay Analysis: game {game_data.get('id', '?')}",
        f"# Checkpoint: {_format_checkpoint_path(checkpoint_path)}",
        f"# Players: {num_players} | Model capacity: {config.effective_max_players}",
        (
            f"# Initial offering: {', '.join(initial.get('initial_offering', []))} | "
            f"remaining deck: {len(initial.get('deck_order', []))}"
        ),
        "",
        format_state_full(initial_state),
        "",
    ]
    lines.append("---")
    lines.append("")

    for group in _group_decisions(summary):
        lines.append(f"### {_format_18xx_action(group.action, group.action_id)}")
        lines.append("")
        if len(group.decisions) > 1:
            lines.append(
                "_Grouped compound 18xx action: multiple engine decisions are shown below._"
            )
            lines.append("")
        for decision in group.decisions:
            _append_decision_lines(
                lines,
                decision,
                num_players=num_players,
                max_players=config.effective_max_players,
                top_n=top_n,
            )

            if verbose:
                state_line = (
                    f"  Engine phase id={decision.engine_phase}, "
                    f"decision phase id={decision.phase_id}, "
                    f"legal actions={len(decision.legal_action_ids)}"
                )
                lines.append(state_line)
                lines.append("")

    if summary.replay_notes:
        lines.append("---")
        lines.append("")
        lines.append("## Replay Notes")
        lines.append("")
        for note in summary.replay_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Final State")
    lines.append("")
    lines.append(format_state_full(summary.final_state))
    lines.append("")

    if TURN.get_phase(summary.final_state) == GamePhases.PHASE_GAME_OVER:
        net_worths = [PLAYERS[pid].get_net_worth(summary.final_state) for pid in range(num_players)]
        terminal_values = compute_terminal_values(
            net_worths,
            num_players,
            config.terminal_blend,
        )
        tv_parts = [f"P{i}={terminal_values[i]:+.3f}" for i in range(num_players)]
        winner = max(range(num_players), key=lambda i: net_worths[i])
        lines.append(f"Winner: P{winner} (${net_worths[winner]})")
        lines.append(f"Terminal values (blend={config.terminal_blend}): {', '.join(tv_parts)}")

    return _apply_player_names("\n".join(lines), player_names)


PLAYER_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
]


def _value_swing_rows(
    groups: list[ActionGroup],
    *,
    player_names: list[str],
    initial_record: dict[str, Any],
    num_players: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, group in enumerate(groups[:-1]):
        if not group.decisions:
            continue
        next_group = groups[idx + 1]
        if not next_group.decisions:
            continue
        before = group.decisions[0].values[:num_players].astype(np.float64)
        after = next_group.decisions[0].values[:num_players].astype(np.float64)
        diff = after - before
        swing = float(np.max(np.abs(diff))) if len(diff) else 0.0
        rows.append({
            "action_id": group.action_id,
            "turn_phase": _turn_phase_label(group),
            "actor": _action_actor_from_initial(group, player_names, initial_record),
            "description": _compact_action_description(group),
            "swing": swing,
            "before": [float(v) for v in before],
            "after": [float(v) for v in after],
            "diff": [float(v) for v in diff],
        })
    return sorted(rows, key=lambda row: row["swing"], reverse=True)[:10]


def _uncertainty_rows(
    groups: list[ActionGroup],
    *,
    player_names: list[str],
    initial_record: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        if not group.decisions:
            continue
        entropies = [_policy_entropy(decision.priors) for decision in group.decisions]
        raw_entropy = float(np.mean([item[0] for item in entropies]))
        norm_entropy = float(np.mean([item[1] for item in entropies]))
        legal_count = float(np.mean([len(decision.legal_action_ids) for decision in group.decisions]))
        rows.append({
            "action_id": group.action_id,
            "turn_phase": _turn_phase_label(group),
            "actor": _action_actor_from_initial(group, player_names, initial_record),
            "description": _compact_action_description(group),
            "entropy": raw_entropy,
            "normalized_entropy": norm_entropy,
            "legal_actions": legal_count,
            "engine_steps": len(group.decisions),
        })
    return sorted(
        rows,
        key=lambda row: (row["normalized_entropy"], row["entropy"]),
        reverse=True,
    )[:10]


def _chart_points(
    groups: list[ActionGroup],
    *,
    num_players: int,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for group in groups:
        if group.action_id < 0 or not group.decisions:
            continue
        points.append({
            "x": int(group.action_id),
            "label": _turn_phase_label(group),
            "description": _compact_action_description(group),
            "values": _group_average_values(group, num_players),
        })
    return points


def _render_value_swing_table(
    rows: list[dict[str, Any]],
    player_names: list[str],
) -> str:
    if not rows:
        return "<p class=\"muted\">No value swings available.</p>"

    header = [
        "<tr>",
        "<th>18xx Action</th>",
        "<th>Actor</th>",
        "<th>Action</th>",
        "<th>Swing</th>",
    ]
    for name in player_names:
        safe = _html(name)
        header.extend([
            f"<th>{safe} before</th>",
            f"<th>{safe} after</th>",
            f"<th>{safe} diff</th>",
        ])
    header.append("</tr>")

    body: list[str] = []
    for row in rows:
        body.extend([
            "<tr>",
            f"<td>{_html(row['turn_phase'])}</td>",
            f"<td>{_html(row['actor'])}</td>",
            f"<td>{_html(row['description'])}</td>",
            f"<td>{row['swing']:.3f}</td>",
        ])
        for before, after, diff in zip(row["before"], row["after"], row["diff"]):
            cls = "pos" if diff > 0 else "neg" if diff < 0 else ""
            body.extend([
                f"<td>{_fmt_value(before)}</td>",
                f"<td>{_fmt_value(after)}</td>",
                f"<td class=\"{cls}\">{_fmt_value(diff)}</td>",
            ])
        body.append("</tr>")

    return "<table>" + "".join(header) + "".join(body) + "</table>"


def _render_uncertainty_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">No policy entropy rows available.</p>"

    lines = [
        "<table>",
        "<tr>",
        "<th>18xx Action</th>",
        "<th>Actor</th>",
        "<th>Action</th>",
        "<th>Entropy</th>",
        "<th>Normalized</th>",
        "<th>Legal Actions</th>",
        "<th>Engine Steps</th>",
        "</tr>",
    ]
    for row in rows:
        lines.extend([
            "<tr>",
            f"<td>{_html(row['turn_phase'])}</td>",
            f"<td>{_html(row['actor'])}</td>",
            f"<td>{_html(row['description'])}</td>",
            f"<td>{row['entropy']:.3f}</td>",
            f"<td>{row['normalized_entropy']:.3f}</td>",
            f"<td>{row['legal_actions']:.1f}</td>",
            f"<td>{row['engine_steps']}</td>",
            "</tr>",
        ])
    lines.append("</table>")
    return "".join(lines)


def _render_action_details_html(
    groups: list[ActionGroup],
    *,
    num_players: int,
    max_players: int,
    top_n: int,
) -> str:
    chunks: list[str] = []
    for group in groups:
        title = f"{_turn_phase_label(group)} - {_compact_action_description(group)}"
        chunks.append(f"<details><summary>{_html(title)}</summary>")
        if len(group.decisions) > 1:
            chunks.append(
                "<p class=\"muted\">Compound 18xx action: multiple engine decisions are grouped here.</p>"
            )
        for decision in group.decisions:
            phase_name = PHASE_NAMES.get(decision.engine_phase, str(decision.engine_phase))
            decision_state = GameState.from_array(
                decision.state_array.copy(),
                num_players,
                max_players=max_players,
            )
            eval_lines = _format_nn_eval(
                decision.priors,
                decision.values,
                decision.legal_action_ids,
                decision.phase_id,
                num_players,
                decision_state,
                top_n,
            )
            chunks.extend([
                "<div class=\"engine-step\">",
                (
                    f"<h4>Engine Step {decision.step}: "
                    f"P{decision.active_player} [{_html(phase_name)}]</h4>"
                ),
                f"<p>Turn {decision.turn} | {_html(decision.note)}</p>",
            ])
            if decision.phase_context:
                chunks.append(f"<p>{_html(_plain_phase_context(decision.phase_context))}</p>")
            chunks.append(f"<pre>{_html(chr(10).join(eval_lines))}</pre>")
            chunks.append(
                "<p><strong>Recorded engine action:</strong> "
                f"{_html(decision.recorded_action_text)} "
                f"<span class=\"muted\">(NN prior {_html(_format_recorded_prior(decision))})</span></p>"
            )
            chunks.append("</div>")
        chunks.append("</details>")
    return "".join(chunks)


def format_replay_html(
    summary: ReplaySummary,
    *,
    checkpoint_path: str | Path,
    config: TrainingConfig,
    top_n: int,
    verbose: bool,
) -> str:
    del verbose
    game_data = summary.game_data
    initial = summary.initial_record
    num_players = len(game_data.get("players", []))
    player_names = _player_names_from_game_data(game_data, initial)
    groups = _group_decisions(summary)
    points = _chart_points(groups, num_players=num_players)
    swing_rows = _value_swing_rows(
        groups,
        player_names=player_names,
        initial_record=initial,
        num_players=num_players,
    )
    uncertainty_rows = _uncertainty_rows(
        groups,
        player_names=player_names,
        initial_record=initial,
    )

    chart_payload = {
        "players": player_names,
        "colors": PLAYER_COLORS[:num_players],
        "points": points,
    }
    chart_json = json.dumps(chart_payload, separators=(",", ":"))
    checkpoint_label = _format_checkpoint_path(checkpoint_path)
    title = f"18xx Replay Analysis: game {game_data.get('id', '?')}"
    initial_offering = ", ".join(initial.get("initial_offering", []))
    detail_html = _render_action_details_html(
        groups,
        num_players=num_players,
        max_players=config.effective_max_players,
        top_n=top_n,
    )
    replay_notes = ""
    if summary.replay_notes:
        items = "".join(f"<li>{_html(note)}</li>" for note in summary.replay_notes)
        replay_notes = f"<section class=\"panel\"><h2>Replay Notes</h2><ul>{items}</ul></section>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html(title)}</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --text: #1f2937;
  --muted: #6b7280;
  --border: #d9dee7;
  --grid: #e6eaf0;
  --header: #eef2f7;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
header {{
  padding: 20px 24px 12px;
  border-bottom: 1px solid var(--border);
  background: var(--panel);
}}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
h3 {{ margin: 0 0 10px; font-size: 16px; }}
h4 {{ margin: 0 0 6px; font-size: 14px; }}
main {{ padding: 16px 24px 28px; }}
.meta {{ color: var(--muted); display: flex; flex-wrap: wrap; gap: 12px; }}
.panel {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}}
.chart-wrap {{ width: 100%; height: 380px; }}
#valueChart {{ display: block; width: 100%; height: 100%; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 6px; color: var(--muted); }}
.swatch {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
.tables {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border: 1px solid var(--border); padding: 7px 8px; text-align: right; vertical-align: top; }}
th {{ background: var(--header); font-weight: 600; color: #374151; }}
td:first-child, th:first-child,
td:nth-child(2), th:nth-child(2),
td:nth-child(3), th:nth-child(3) {{ text-align: left; }}
.muted {{ color: var(--muted); }}
.pos {{ color: #15803d; }}
.neg {{ color: #b91c1c; }}
details {{
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--panel);
  margin-bottom: 10px;
}}
summary {{ cursor: pointer; padding: 10px 12px; font-weight: 600; }}
.engine-step {{ border-top: 1px solid var(--border); padding: 12px; }}
pre {{
  white-space: pre-wrap;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
  overflow-x: auto;
}}
@media (max-width: 900px) {{
  main {{ padding: 12px; }}
  .chart-wrap {{ height: 300px; }}
  table {{ font-size: 12px; }}
  th, td {{ padding: 6px; }}
}}
</style>
</head>
<body>
<header>
  <h1>{_html(title)}</h1>
  <div class="meta">
    <span>Checkpoint: {_html(checkpoint_label)}</span>
    <span>Players: {num_players}</span>
    <span>Model capacity: {config.effective_max_players}</span>
    <span>Initial offering: {_html(initial_offering)}</span>
    <span>Remaining deck: {len(initial.get('deck_order', []))}</span>
  </div>
</header>
<main>
  <section class="panel">
    <h2>Model Value Estimates</h2>
    <div class="chart-wrap"><canvas id="valueChart"></canvas></div>
    <div id="legend" class="legend"></div>
    <p class="muted">X-axis is 18xx action number. Compound actions use the average value estimate across their engine decisions.</p>
  </section>

  <section class="panel">
    <h2>Top 10 Value Swings</h2>
    <p class="muted">Ranked by the largest absolute per-player value change from just before the grouped 18xx action to the next model estimate.</p>
    {_render_value_swing_table(swing_rows, player_names)}
  </section>

  <section class="panel">
    <h2>Top 10 Policy Entropy Moves</h2>
    <p class="muted">Ranked by average normalized policy entropy across grouped engine decisions.</p>
    {_render_uncertainty_table(uncertainty_rows)}
  </section>

  {replay_notes}

  <section>
    <h2>Action Details</h2>
    {detail_html}
  </section>
</main>
<script>
const chartData = {chart_json};

function drawValueChart() {{
  const canvas = document.getElementById("valueChart");
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const points = chartData.points || [];
  const players = chartData.players || [];
  const colors = chartData.colors || [];
  const margin = {{ left: 54, right: 18, top: 18, bottom: 44 }};
  const plotW = Math.max(1, w - margin.left - margin.right);
  const plotH = Math.max(1, h - margin.top - margin.bottom);
  const yMin = -1;
  const yMax = 1;
  const xs = points.map(p => p.x);
  const rawXMin = xs.length ? Math.min(...xs) : 0;
  const rawXMax = xs.length ? Math.max(...xs) : 1;
  const xMin = rawXMin === rawXMax ? rawXMin - 0.5 : rawXMin;
  const xMax = rawXMin === rawXMax ? rawXMax + 0.5 : rawXMax;
  const xSpan = Math.max(1, xMax - xMin);
  const xScale = x => margin.left + ((x - xMin) / xSpan) * plotW;
  const yScale = y => margin.top + ((yMax - y) / (yMax - yMin)) * plotH;

  ctx.font = "12px system-ui, sans-serif";
  ctx.strokeStyle = "#d9dee7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(margin.left, margin.top, plotW, plotH);
  ctx.stroke();

  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [-1, -0.5, 0, 0.5, 1].forEach(tick => {{
    const y = yScale(tick);
    ctx.strokeStyle = tick === 0 ? "#c7ceda" : "#e6eaf0";
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(tick.toFixed(1), margin.left - 8, y);
  }});

  const xTicks = [xMin, xMin + xSpan * 0.25, xMin + xSpan * 0.5, xMin + xSpan * 0.75, xMax];
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  xTicks.forEach(raw => {{
    const tick = Math.round(raw);
    const x = xScale(tick);
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
    ctx.stroke();
    ctx.fillStyle = "#6b7280";
    ctx.fillText(String(tick), x, margin.top + plotH + 8);
  }});
  ctx.fillText("18xx action number", margin.left + plotW / 2, h - 16);
  ctx.save();
  ctx.translate(16, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("value", 0, 0);
  ctx.restore();

  players.forEach((player, playerIdx) => {{
    if (!points.length) return;
    ctx.strokeStyle = colors[playerIdx] || "#111827";
    ctx.fillStyle = colors[playerIdx] || "#111827";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, idx) => {{
      const x = xScale(point.x);
      const y = yScale(point.values[playerIdx]);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.stroke();
    points.forEach(point => {{
      const x = xScale(point.x);
      const y = yScale(point.values[playerIdx]);
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }});
  }});
}}

function drawLegend() {{
  const legend = document.getElementById("legend");
  legend.innerHTML = "";
  (chartData.players || []).forEach((player, idx) => {{
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = (chartData.colors || [])[idx] || "#111827";
    const label = document.createElement("span");
    label.textContent = player;
    item.appendChild(swatch);
    item.appendChild(label);
    legend.appendChild(item);
  }});
}}

drawLegend();
drawValueChart();
window.addEventListener("resize", drawValueChart);
</script>
</body>
</html>
"""


def analyze_replay(
    game_json_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    config: TrainingConfig,
    *,
    checkpoint_path: str | Path,
    top_n: int = 10,
    verbose: bool = False,
    output_format: str = "html",
) -> str:
    game_path = Path(game_json_path)
    game_data = json.loads(game_path.read_text())
    num_players = len(game_data.get("players", []))
    if not (config.effective_min_players <= num_players <= config.effective_max_players):
        raise ValueError(
            "replay player count must be within checkpoint player range "
            f"{config.effective_min_players}-{config.effective_max_players}, "
            f"got {num_players}"
        )

    evaluator = NNEvaluator(
        model,
        device,
        num_players=config.effective_max_players,
        terminal_rank_weight=config.terminal_blend,
        eval_dtype=config.eval_dtype,
        input_spec=get_model_input_spec(config),
    )
    session = ReplayAnalyzerSession(
        evaluator,
        num_players=num_players,
        max_players=config.effective_max_players,
    )
    summary = session.sync_with_analysis(game_data)
    if output_format == "markdown":
        return format_replay_markdown(
            summary,
            checkpoint_path=checkpoint_path,
            config=config,
            top_n=top_n,
            verbose=verbose,
        )
    if output_format == "html":
        return format_replay_html(
            summary,
            checkpoint_path=checkpoint_path,
            config=config,
            top_n=top_n,
            verbose=verbose,
        )
    raise ValueError(f"Unsupported output format: {output_format!r}")


def _load_model_for_cli(
    checkpoint: str,
    checkpoint_dir: str,
    device: torch.device,
    *,
    num_players: int,
) -> tuple[torch.nn.Module, TrainingConfig, str | Path]:
    if checkpoint == "new":
        if not 3 <= num_players <= 5:
            raise ValueError(f"'new' model requires 3-5 players, got {num_players}")
        config = TrainingConfig(num_players=num_players)
        model = create_model(config).to(device)
        model.eval()
        return model, config, "new"

    if checkpoint == "latest":
        cp_path = find_latest_checkpoint(Path(checkpoint_dir))
        if cp_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    else:
        cp_path = Path(checkpoint)

    model, config, _cp = load_model_from_checkpoint(cp_path, device)
    model.eval()
    return model, config, cp_path


def _resolve_output_format(format_arg: str, output_path: str | None) -> str:
    if format_arg != "auto":
        return format_arg
    if output_path is not None:
        suffix = Path(output_path).suffix.lower()
        if suffix in {".md", ".markdown"}:
            return "markdown"
    return "html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze an 18xx.games replay with one NN forward pass per engine decision"
    )
    parser.add_argument("game_json", type=str, help="Path to an 18xx.games JSON replay")
    parser.add_argument(
        "checkpoint",
        type=str,
        help='Path to checkpoint, "latest" for --checkpoint-dir, or "new"',
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=10, help="Top N legal actions to show")
    parser.add_argument("--verbose", action="store_true", help="Include extra engine metadata")
    parser.add_argument(
        "--format",
        choices=("auto", "html", "markdown"),
        default="auto",
        help="Report format. auto writes Markdown for .md/.markdown outputs and HTML otherwise.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output report file")
    args = parser.parse_args()

    game_data = json.loads(Path(args.game_json).read_text())
    num_players = len(game_data.get("players", []))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, config, checkpoint_path = _load_model_for_cli(
        args.checkpoint,
        args.checkpoint_dir,
        device,
        num_players=num_players,
    )
    output_format = _resolve_output_format(args.format, args.output)
    result = analyze_replay(
        args.game_json,
        model,
        device,
        config,
        checkpoint_path=checkpoint_path,
        top_n=args.top_n,
        verbose=args.verbose,
        output_format=output_format,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result + "\n")
        print(f"Replay {output_format} analysis written to {output_path}")
    else:
        print(result)


if __name__ == "__main__":
    main()
