"""Webhook-driven live play against humans on 18xx.games.

Receives turn notifications via webhook, fetches game state from the
18xx.games API, runs MCTS search, and posts the selected move back.

Usage:
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime --simulations 800
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime --no-compile
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime --model-output
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from core.actions import ACTION_PASS_PY as ACTION_PASS
from core.data import (
    COMPANY_NAME_TO_ID,
    COMPANY_NAMES,
    CORP_NAME_TO_ID,
    CORP_NAMES,
    GamePhases,
)
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
)
from core.state import get_layout
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from mcts.search import (
    StatePool,
    get_greedy_leaf_depth,
    get_greedy_leaf_value,
    prepare_reuse_root,
    run_search,
)
from nn import get_model_input_spec
from train.analyze_game import (
    _apply_player_names as _apply_analyze_player_names,
    _format_mcts_visits as _format_analyze_mcts_visits,
    _format_nn_eval as _format_analyze_nn_eval,
)
from train.checkpoint import find_latest_checkpoint, load_model_from_checkpoint
from train.config import MCTSConfig
from train.debug_trace import PHASE_NAMES, format_action, format_phase_context
from train.profile_stats import SearchStats

from .action_mapper import engine_action_to_18xx
from .api_client import ApiClient, PermanentError, TransientError
from .action_parser import get_legal_actions
from .auto_actions import attach_expected_auto_actions
from .game_session import GameSession, format_state_mismatches
from .share_ledger import (
    build_share_ownership as _build_share_ownership,
    resolve_buyable_share as _resolve_buyable_share,
    resolve_issuable_share as _resolve_issuable_share,
    resolve_sellable_share as _resolve_sellable_share,
)

logger = logging.getLogger(__name__)

AUTOMATED_PHASES = (
    GamePhases.PHASE_WRAP_UP,
    GamePhases.PHASE_INCOME,
    GamePhases.PHASE_END_CARD,
)
ACQ_PHASES = (
    GamePhases.PHASE_ACQ_SELECT_CORP,
    GamePhases.PHASE_ACQ_SELECT_COMPANY,
    GamePhases.PHASE_ACQ_SELECT_PRICE,
    GamePhases.PHASE_ACQ_OFFER,
)
UNORDERED_PASS_PHASES = (
    GamePhases.PHASE_ACQ_SELECT_CORP,
    GamePhases.PHASE_CLOSING,
)


def prepare_live_decision_state(state) -> None:
    """Set model-side rule flags after 18xx replay synchronization."""
    # Replay uses acq_same_president=False so historical 18xx cross-president
    # offers can be consumed. The live model/search policy should still only
    # consider same-president acquisition offers.
    state.acq_same_president = True
    state.allow_positive_income_closing = False


def _resolve_live_search_batch_size(config, override: int | None) -> int:
    """Return CLI override or checkpoint search batch size."""
    return int(override) if override is not None else int(config.search_batch_size)


def _resolve_live_eval_dtype(config, override: str | None) -> str | None:
    """Return CLI eval dtype override or checkpoint eval dtype."""
    if override is None:
        return config.eval_dtype
    if override == "float32":
        return None
    return override


def _player_names_from_game_data(game_data: dict, num_players: int) -> list[str]:
    """Return canonical-order 18xx player display names."""
    names: list[str] = []
    players = game_data.get("players", [])
    for idx in range(num_players):
        raw_name = ""
        if idx < len(players):
            raw_name = str(players[idx].get("name") or "")
        name = raw_name.replace("\n", " ").replace("\r", " ").strip()
        names.append(name or f"P{idx}")
    return names


def _format_live_model_output(
    *,
    game_data: dict,
    state,
    priors: np.ndarray,
    values: np.ndarray,
    action_ids: np.ndarray,
    phase_id: int,
    num_players: int,
    root,
    action_idx: int,
    mcts_config: MCTSConfig | None = None,
    search_stats: SearchStats | None = None,
    elapsed_secs: float | None = None,
    top_n: int = 10,
) -> str:
    """Format one live decision using the analyzer's report style."""
    engine_phase = TURN.get_phase(state)
    phase_name = PHASE_NAMES.get(engine_phase, str(engine_phase))
    turn_number = int(TURN.get_turn_number(state))
    active_player = TURN.get_active_player(state)
    player_names = _player_names_from_game_data(game_data, num_players)
    phase_ctx = format_phase_context(state)

    noised_priors: dict[int, float] | None = None
    if (
        root is not None
        and mcts_config is not None
        and mcts_config.dirichlet_epsilon > 0
        and root.priors is not None
    ):
        assert root.legal_actions is not None
        noised_priors = {
            int(root.legal_actions[i]): float(root.priors[i])
            for i in range(len(root.legal_actions))
        }

    search_text = (
        f"{elapsed_secs:.1f}s"
        if elapsed_secs is not None
        else "skipped (single legal action)"
    )
    lines = [
        f"### Live Decision: P{active_player} [{phase_name}]",
        "",
        f"  Game: {game_data.get('id', '?')} | Turn: {turn_number} | "
        f"Search: {search_text}",
    ]
    if phase_ctx:
        lines.append(f"  {phase_ctx}")
    lines.append("")
    lines.extend(
        _format_analyze_nn_eval(
            priors,
            values,
            action_ids,
            phase_id,
            num_players,
            state,
            top_n,
            noised_priors=noised_priors,
        )
    )
    lines.append("")
    if root is None:
        lines.append("  MCTS: skipped (single legal action)")
        lines.append("  A0GB Value: skipped (single legal action)")
    else:
        lines.extend(_format_analyze_mcts_visits(root, phase_id, state, top_n))

        a0gb = get_greedy_leaf_value(root, num_players)
        a0gb_depth = get_greedy_leaf_depth(root)
        a0gb_parts = [f"P{i}={a0gb[i]:+.3f}" for i in range(num_players)]
        vb = search_stats.virtual_backups if search_stats is not None else 0
        lines.append(
            f"  A0GB Value: {', '.join(a0gb_parts)} "
            f"(depth: {a0gb_depth}{f', vbackups: {vb}' if vb > 0 else ''})"
        )
    lines.append("")
    lines.append(f"  **Action: {format_action(phase_id, action_idx, state)}**")

    return _apply_analyze_player_names("\n".join(lines), player_names)

# ---------------------------------------------------------------------------
# Webhook parsing
# ---------------------------------------------------------------------------

_GAME_URL_RE = re.compile(r"/game/(\d+)")
_WEBHOOK_USER_RE = re.compile(r"<@([^>]+)>")


def parse_webhook_text(text: str) -> tuple[str | None, str | None]:
    """Extract (game_id, webhook_user_id) from webhook notification text.

    Returns (None, None) if the text can't be parsed.
    """
    game_id: str | None = None
    webhook_user_id: str | None = None

    m = _GAME_URL_RE.search(text)
    if m:
        game_id = m.group(1)

    m = _WEBHOOK_USER_RE.search(text)
    if m:
        webhook_user_id = m.group(1)

    return game_id, webhook_user_id


def is_turn_webhook_text(text: str) -> bool:
    """Return whether a webhook body is a turn notification."""
    return "your turn" in text.lower()


def parse_poke_game_id(path: str) -> str | None:
    """Extract a game id from /poke/<id> or /poke?game_id=<id>."""
    parsed = urlparse(path)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]

    if len(path_parts) == 2 and path_parts[0] == "poke":
        return path_parts[1]

    if len(path_parts) == 1 and path_parts[0] == "poke":
        query = parse_qs(parsed.query)
        for key in ("game_id", "game"):
            values = query.get(key)
            if values and values[0]:
                return values[0]

    return None


def _same_id(left, right) -> bool:
    return str(left) == str(right)


def _is_18xx_acquisition_round(game_data: dict) -> bool:
    return "acquisition" in str(game_data.get("round", "")).lower()


def _is_18xx_closing_round(game_data: dict) -> bool:
    round_name = str(game_data.get("round", "")).lower()
    return "closing" in round_name or "close" in round_name


def _is_18xx_dividend_round(game_data: dict) -> bool:
    round_name = str(game_data.get("round", "")).lower()
    return "dividend" in round_name or round_name == "div"


def _bot_is_acting(game_data: dict, bot_user_id) -> bool:
    acting = game_data.get("acting", [])
    return bool(acting) and any(_same_id(bot_user_id, actor) for actor in acting)


def _acquisition_compatibility_action(
    game_data: dict,
    session: GameSession,
    state,
    bot_user_id,
    engine_player_idx: int,
) -> dict | None:
    """Return an explicit 18xx ACQ action when RSS has already advanced.

    18xx keeps Acquisition as an unordered blocking step because cross-
    president and FI right-of-refusal choices are legal there.  The RSS model
    intentionally searches only same-president acquisitions, so the replayed
    engine state can already be on the next RSS decision while 18xx still
    needs an explicit reject/pass from the currently acting bot.
    """
    if bot_user_id is None or not _is_18xx_acquisition_round(game_data):
        return None

    if not _bot_is_acting(game_data, bot_user_id):
        return None

    pending_offer = session.pending_offer_for_user_id(bot_user_id)
    if pending_offer is not None:
        corporation = pending_offer.get("corporation")
        company = pending_offer.get("company")
        if not corporation or not company:
            return None
        return {
            "type": "respond",
            "entity": bot_user_id,
            "entity_type": "player",
            "corporation": corporation,
            "company": company,
            "accept": "false",
        }

    phase = TURN.get_phase(state)
    active = TURN.get_active_player(state)
    if phase not in ACQ_PHASES or active != engine_player_idx:
        return {
            "type": "pass",
            "entity": bot_user_id,
            "entity_type": "player",
        }

    return None


def _player_has_only_pass(state, player_idx: int) -> bool:
    """Return whether ``player_idx`` has only PASS in the current phase."""
    previous_active = TURN.get_active_player(state)
    TURN.set_active_player(state, player_idx)
    try:
        legal_actions = get_legal_actions(state)
    finally:
        TURN.set_active_player(state, previous_active)
    return (
        len(legal_actions) == 1
        and legal_actions[0][1].action_type == ACTION_PASS
    )


def _closing_compatibility_action(
    game_data: dict,
    state,
    bot_user_id,
    engine_player_idx: int,
) -> dict | None:
    """Return an explicit 18xx Closing pass when RSS has no model action."""
    if bot_user_id is None or not _is_18xx_closing_round(game_data):
        return None
    if not _bot_is_acting(game_data, bot_user_id):
        return None

    phase = TURN.get_phase(state)
    if phase != GamePhases.PHASE_CLOSING:
        return {
            "type": "pass",
            "entity": bot_user_id,
            "entity_type": "player",
        }

    active = TURN.get_active_player(state)
    if active != engine_player_idx and _player_has_only_pass(state, engine_player_idx):
        return {
            "type": "pass",
            "entity": bot_user_id,
            "entity_type": "player",
        }

    return None


def _dividend_compatibility_action(
    game_data: dict,
    session: GameSession,
    state,
    bot_user_id,
    engine_player_idx: int,
) -> dict | None:
    """Return an explicit 18xx dividend when RSS auto-applied the only choice."""
    if bot_user_id is None or not _is_18xx_dividend_round(game_data):
        return None
    if not _bot_is_acting(game_data, bot_user_id):
        return None

    active_corp_name = session._last_extract_record.get("active_corp")
    if not active_corp_name:
        return None

    corp_id = CORP_NAME_TO_ID.get(active_corp_name)
    if corp_id is None or not CORPS[corp_id].is_active(state):
        return None

    if CORPS[corp_id].get_president_id(state) != engine_player_idx:
        return None

    phase = TURN.get_phase(state)
    if phase == GamePhases.PHASE_DIVIDENDS and TURN.get_active_corp(state) == corp_id:
        return None

    if TURN.is_dividend_remaining(state, corp_id):
        return None

    return {
        "type": "dividend",
        "entity": active_corp_name,
        "entity_type": "corporation",
        "kind": "variable",
        "amount": 0,
    }


def _acting_engine_player_indices(
    game_data: dict,
    session: GameSession,
) -> set[int]:
    """Return engine player indices currently listed as acting by 18xx."""
    indices: set[int] = set()
    for actor in game_data.get("acting", []):
        try:
            indices.add(session.player_index_for_user_id(actor))
        except ValueError:
            continue
    return indices


def _align_unordered_round_to_18xx_actor(
    game_data: dict,
    session: GameSession,
    state,
    *,
    bot_player_indices: set[int] | None = None,
) -> int:
    """Apply local non-bot passes until RSS active matches 18xx acting.

    18xx Acquisition and Closing decisions are unordered: any eligible player
    can act. RSS has an ordered active-player pointer for those same decisions.
    When 18xx says a bot is acting but RSS is waiting on an earlier non-bot
    player, consume a local replay-only pass before the active-player check.
    """
    acting_indices = _acting_engine_player_indices(game_data, session)
    if not acting_indices:
        return 0
    bot_player_indices = set(bot_player_indices or ())

    applied = 0
    max_passes = max(1, TURN.get_num_players(state) + 2)
    while applied < max_passes:
        phase = TURN.get_phase(state)
        if phase not in UNORDERED_PASS_PHASES:
            break

        active = TURN.get_active_player(state)
        if active in acting_indices:
            break
        if active in bot_player_indices:
            logger.info(
                "Not injecting unordered pass for bot player: "
                f"phase={phase}, active=P{active}, "
                f"acting={sorted(acting_indices)}"
            )
            break

        legal_actions = get_legal_actions(state)
        pass_action = next(
            (action_id for action_id, info in legal_actions if info.action_type == ACTION_PASS),
            None,
        )
        if pass_action is None:
            logger.info(
                "Cannot locally align unordered round: "
                f"phase={phase}, active=P{active}, "
                f"acting={sorted(acting_indices)}, "
                f"legal_count={len(legal_actions)}"
            )
            break

        previous_step_mode = state.step_mode
        state.step_mode = True
        try:
            status = DRIVER.apply_action(state, pass_action)
        finally:
            state.step_mode = previous_step_mode

        if status == STATUS_INVALID:
            raise RuntimeError(
                f"Invalid local unordered pass alignment in phase {phase}"
            )
        applied += 1
        if status == STATUS_GAME_OVER:
            break

    if applied >= max_passes:
        logger.warning(
            "Stopped unordered round alignment after max local passes: "
            f"acting={sorted(acting_indices)}"
        )

    return applied


def _retarget_closing_active_player_to_bot(
    game_data: dict,
    state,
    bot_user_id,
    engine_player_idx: int,
) -> bool:
    """Point RSS's ordered Closing turn at this 18xx acting bot."""
    if bot_user_id is None or not _is_18xx_closing_round(game_data):
        return False
    if TURN.get_phase(state) != GamePhases.PHASE_CLOSING:
        return False
    if TURN.get_active_player(state) == engine_player_idx:
        return False
    if not _bot_is_acting(game_data, bot_user_id):
        return False

    TURN.set_active_player(state, engine_player_idx)
    return True


def _should_continue_after_postable_action(
    pre_action_phase: int,
    state,
    bot_player_idx: int,
) -> bool:
    """Return whether live planning should keep selecting before posting."""
    current_phase = TURN.get_phase(state)
    if current_phase == GamePhases.PHASE_ACQ_OFFER:
        return False
    same_round = (
        pre_action_phase in ACQ_PHASES
        and current_phase in ACQ_PHASES
    ) or (
        pre_action_phase == GamePhases.PHASE_CLOSING
        and current_phase == GamePhases.PHASE_CLOSING
    )
    return same_round and TURN.get_active_player(state) == bot_player_idx


# ---------------------------------------------------------------------------
# Action format translation
# ---------------------------------------------------------------------------


def intent_to_api_action(
    intent: dict,
    state,
    game_data: dict,
    bot_player_idx: int,
    committed_ids: set,
    bot_user_id=None,
) -> dict:
    """Convert an engine intent dict to the 18xx.games API action format.

    Maps entity/entity_type and adds fields the API expects.
    """
    if bot_user_id is None:
        player_ids = [p["id"] for p in game_data["players"]]
        bot_user_id = player_ids[bot_player_idx]
    phase = TURN.get_phase(state)
    itype = intent["type"]

    if itype == "pass":
        # Entity depends on which phase we're in.
        if phase == GamePhases.PHASE_DIVIDENDS:
            corp_id = TURN.get_active_corp(state)
            return {
                "type": "pass",
                "entity": CORP_NAMES[corp_id],
                "entity_type": "corporation",
            }
        if phase == GamePhases.PHASE_ISSUE_SHARES:
            corp_id = TURN.get_active_corp(state)
            return {
                "type": "pass",
                "entity": CORP_NAMES[corp_id],
                "entity_type": "corporation",
            }
        if phase == GamePhases.PHASE_IPO:
            # IPO pass entity is the company being considered
            company_id = TURN.get_active_company(state)
            return {
                "type": "pass",
                "entity": COMPANY_NAMES[company_id],
                "entity_type": "company",
            }
        # Player is acting (INVEST, BID, CLO)
        return {
            "type": "pass",
            "entity": bot_user_id,
            "entity_type": "player",
        }

    if itype == "bid":
        return {
            "type": "bid",
            "entity": bot_user_id,
            "entity_type": "player",
            "company": intent["company"],
            "price": intent["price"],
        }

    if itype == "buy_shares":
        corp_name = intent["corporation"]
        corp_id = CORP_NAME_TO_ID[corp_name]
        share_id = _resolve_buyable_share(
            game_data,
            corp_name,
            committed_ids,
            market_share_count=CORPS[corp_id].get_bank_shares(state),
            treasury_share_count=CORPS[corp_id].get_unissued_shares(state),
        )
        share_price = _get_corp_share_price(state, corp_name)
        return {
            "type": "buy_shares",
            "entity": bot_user_id,
            "entity_type": "player",
            "shares": [share_id],
            "percent": 10,
            "share_price": share_price,
        }

    if itype == "sell_shares":
        corp_name = intent["corporation"]
        share_id = _resolve_sellable_share(
            game_data, corp_name, bot_user_id, committed_ids,
        )
        share_price = _get_corp_share_price(state, corp_name)
        return {
            "type": "sell_shares",
            "entity": bot_user_id,
            "entity_type": "player",
            "shares": [share_id],
            "percent": 10,
            "share_price": share_price,
        }

    if itype == "par":
        return {
            "type": "par",
            "entity": intent["company"],
            "entity_type": "company",
            "corporation": intent["corporation"],
            "share_price": intent["share_price"],
        }

    if itype == "dividend":
        corp_id = TURN.get_active_corp(state)
        return {
            "type": "dividend",
            "entity": CORP_NAMES[corp_id],
            "entity_type": "corporation",
            "kind": "variable",
            "amount": intent["amount"],
        }

    if itype == "issue":
        corp_id = TURN.get_active_corp(state)
        corp_name = CORP_NAMES[corp_id]
        share_id = _resolve_issuable_share(
            game_data,
            corp_name,
            committed_ids,
            market_share_count=CORPS[corp_id].get_bank_shares(state),
            treasury_share_count=CORPS[corp_id].get_unissued_shares(state),
        )
        share_price = _get_corp_share_price(state, corp_name)
        return {
            "type": "sell_shares",
            "entity": corp_name,
            "entity_type": "corporation",
            "shares": [share_id],
            "percent": 10,
            "share_price": share_price,
        }

    if itype == "offer":
        # ACQ offers: entity is the president (player), not the corp
        return {
            "type": "offer",
            "entity": bot_user_id,
            "entity_type": "player",
            "corporation": intent["corporation"],
            "company": intent["company"],
            "price": intent["price"],
        }

    if itype == "respond":
        return {
            "type": "respond",
            "entity": bot_user_id,
            "entity_type": "player",
            "corporation": intent["corporation"],
            "company": intent["company"],
            "accept": intent["accept"],
        }

    if itype == "close":
        return {
            "type": "sell_company",
            "entity": bot_user_id,
            "entity_type": "player",
            "company": intent["company"],
            "price": 0,
        }

    if itype in {"select_corp", "select_company", "ipo_select", "par_price"}:
        raise ValueError(f"Internal intent {itype!r} cannot be posted directly")

    raise ValueError(f"Unknown intent type: {itype}")


class _LiveActionComposer:
    """Compose phase-local engine choices into postable 18xx actions."""

    def __init__(
        self,
        game_data: dict,
        bot_player_idx: int,
        committed_ids: set,
        bot_user_id=None,
    ) -> None:
        self.game_data = game_data
        self.bot_player_idx = bot_player_idx
        self.committed_ids = committed_ids
        self._bot_user_id = bot_user_id
        self.actions: list[dict] = []
        self._pending_bid: dict | None = None
        self._pending_par: dict | None = None
        self._pending_acq: dict | None = None

    @property
    def bot_user_id(self):
        if self._bot_user_id is not None:
            return self._bot_user_id
        return self.game_data["players"][self.bot_player_idx]["id"]

    def add_step(self, phase: int, intent: dict, state=None) -> None:
        """Add one pre-apply engine intent to the current 18xx action batch."""
        itype = intent["type"]

        if self._pending_bid is not None:
            if phase == GamePhases.PHASE_BID and itype == "bid":
                self._append_bid(intent)
                self._pending_bid = None
                return
            raise ValueError(f"Expected BID price after INVEST auction, got {intent}")

        if self._pending_par is not None:
            if phase == GamePhases.PHASE_PAR and itype == "par_price":
                self._append_par(intent)
                self._pending_par = None
                return
            raise ValueError(f"Expected PAR price after IPO selection, got {intent}")

        if self._pending_acq is not None:
            if itype == "select_company":
                self._pending_acq["company"] = intent["company"]
                self._pending_acq.setdefault("corporation", intent.get("corporation"))
                return
            if itype == "offer":
                self._append_offer(intent)
                self._pending_acq = None
                return
            raise ValueError(f"Expected ACQ company/price after corp selection, got {intent}")

        if phase == GamePhases.PHASE_INVEST and itype == "bid":
            self._pending_bid = {"company": intent["company"]}
            return

        if itype == "ipo_select":
            self._pending_par = {
                "company": intent["company"],
                "corporation": intent["corporation"],
            }
            return

        if itype == "select_corp":
            self._pending_acq = {"corporation": intent["corporation"]}
            return

        if itype == "select_company":
            self._pending_acq = {
                "company": intent["company"],
                "corporation": intent.get("corporation"),
            }
            return

        if itype == "offer":
            self._append_offer(intent)
            return

        if itype == "par_price":
            raise ValueError(f"PAR price without IPO selection: {intent}")

        if state is None:
            raise ValueError(f"Direct intent {intent} requires pre-action state")
        self.actions.append(
            intent_to_api_action(
                intent,
                state,
                self._game_data_with_planned_actions(),
                self.bot_player_idx,
                self.committed_ids,
                bot_user_id=self.bot_user_id,
            )
        )

    def finish(self) -> list[dict]:
        if self._pending_bid is not None:
            raise ValueError(f"Incomplete INVEST/BID action: {self._pending_bid}")
        if self._pending_par is not None:
            raise ValueError(f"Incomplete IPO/PAR action: {self._pending_par}")
        if self._pending_acq is not None:
            raise ValueError(f"Incomplete ACQ action: {self._pending_acq}")
        return list(self.actions)

    def _game_data_with_planned_actions(self) -> dict:
        if not self.actions:
            return self.game_data
        planned = []
        for action in self.actions:
            planned_action = dict(action)
            planned_action.setdefault("user", self.bot_user_id)
            planned.append(planned_action)
        game_data = dict(self.game_data)
        game_data["actions"] = list(self.game_data.get("actions", [])) + planned
        return game_data

    def _append_bid(self, intent: dict) -> None:
        company = intent.get("company") or self._pending_bid["company"]
        self.actions.append({
            "type": "bid",
            "entity": self.bot_user_id,
            "entity_type": "player",
            "company": company,
            "price": intent["price"],
        })

    def _append_par(self, intent: dict) -> None:
        assert self._pending_par is not None
        self.actions.append({
            "type": "par",
            "entity": self._pending_par["company"],
            "entity_type": "company",
            "corporation": self._pending_par["corporation"],
            "share_price": intent["share_price"],
        })

    def _append_offer(self, intent: dict) -> None:
        pending = self._pending_acq or {}
        corporation = intent.get("corporation") or pending.get("corporation")
        company = intent.get("company") or pending.get("company")
        if corporation is None or company is None:
            raise ValueError(f"Incomplete ACQ offer intent: {intent}")
        self.actions.append({
            "type": "offer",
            "entity": self.bot_user_id,
            "entity_type": "player",
            "corporation": corporation,
            "company": company,
            "price": intent["price"],
        })


def _get_corp_share_price(state, corp_name: str) -> int:
    """Get a corporation's current share price from engine state."""
    from core.data import CORP_NAME_TO_ID

    corp_id = CORP_NAME_TO_ID[corp_name]
    return CORPS[corp_id].get_share_price(state)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Lazy-load search engines from the runtime model config.

    ``models.json`` accepts exact player-count keys (``"3"``), range keys
    (``"3-5"``), or a fallback key (``"default"`` / ``"*"``). A single
    mixed-player checkpoint can therefore serve every game from 3 to 5
    players.
    """

    def __init__(
        self,
        models_config: dict[str, str],
        device: torch.device,
        num_simulations: int,
        search_batch_size: int | None,
        checkpoint_dir: Path | None = None,
        compile_model: bool = False,
        model_output: bool = False,
        eval_dtype: str | None = None,
    ):
        self._config = models_config
        self._device = device
        self._num_simulations = num_simulations
        self._search_batch_size = search_batch_size
        self._checkpoint_dir = checkpoint_dir
        self._compile_model = compile_model
        self._model_output = model_output
        self._eval_dtype = eval_dtype
        self._engines: dict[Path, _SearchEngine] = {}

    def get_engine(self, num_players: int) -> _SearchEngine:
        """Get or create a search engine for the given player count."""
        cp_path = self._resolve_checkpoint_path(num_players)
        engine = self._engines.get(cp_path)
        if engine is None:
            logger.info(f"Loading model for live play: {cp_path}")
            engine = _SearchEngine(
                cp_path,
                self._device,
                num_simulations=self._num_simulations,
                search_batch_size=self._search_batch_size,
                compile_model=self._compile_model,
                model_output=self._model_output,
                eval_dtype=self._eval_dtype,
            )
            self._engines[cp_path] = engine
        engine.validate_player_count(num_players)
        return engine

    def _resolve_checkpoint_path(self, num_players: int) -> Path:
        cp_str = self._checkpoint_spec_for(num_players)
        if cp_str == "latest":
            if self._checkpoint_dir is None:
                raise FileNotFoundError(
                    "'latest' checkpoint requested but no checkpoint_dir is set"
                )
            found = find_latest_checkpoint(self._checkpoint_dir)
            if found is None:
                raise FileNotFoundError(f"No checkpoint in {self._checkpoint_dir}")
            return found
        return Path(cp_str)

    def _checkpoint_spec_for(self, num_players: int) -> str:
        exact = self._config.get(str(num_players))
        if exact is not None:
            return exact

        for key, value in self._config.items():
            if "-" not in key:
                continue
            lo_s, hi_s = key.split("-", 1)
            try:
                lo = int(lo_s)
                hi = int(hi_s)
            except ValueError:
                continue
            if lo <= num_players <= hi:
                return value

        for key in ("default", "*"):
            value = self._config.get(key)
            if value is not None:
                return value

        if len(self._config) == 1:
            return next(iter(self._config.values()))

        raise ValueError(
            f"No model configured for {num_players} players. "
            "Use an exact key like \"4\" or a range key like \"3-5\"."
        )


class _SearchEngine:
    """Wraps one checkpoint and runs MCTS for supported player counts."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        num_simulations: int,
        search_batch_size: int | None,
        compile_model: bool = False,
        model_output: bool = False,
        eval_dtype: str | None = None,
    ):
        self.device = device
        self.num_simulations = num_simulations
        self.model_output = model_output

        model, self.config, cp = load_model_from_checkpoint(checkpoint_path, device)
        self.search_batch_size = _resolve_live_search_batch_size(
            self.config,
            search_batch_size,
        )
        self.eval_dtype = _resolve_live_eval_dtype(self.config, eval_dtype)
        model.eval()
        if compile_model and device.type == "cuda":
            from train.gpu import detect_gpu

            gpu = detect_gpu(device.type)
            compile_kwargs = gpu.get_compile_kwargs(
                for_training=False,
                eval_batch_shape_mode="dynamic",
            )
            logger.info(
                f"Compiling live model with torch.compile: {compile_kwargs}"
            )
            model = torch.compile(model, **compile_kwargs)  # type: ignore[assignment]
            model.eval()
        elif compile_model:
            logger.info(
                f"Skipping torch.compile for non-CUDA live device: {device}"
            )
        else:
            logger.info("torch.compile disabled for live model")

        self.min_players = self.config.effective_min_players
        self.max_players = self.config.effective_max_players
        input_spec = get_model_input_spec(self.config)

        self._evaluator = NNEvaluator(
            model,
            device,
            num_players=self.max_players,
            terminal_rank_weight=self.config.terminal_blend,
            eval_dtype=self.eval_dtype,
            input_spec=input_spec,
        )

        layout = get_layout(self.max_players)
        self._state_pool = StatePool(
            2 * (num_simulations + 1), layout.total_size,
        )
        self._rng = np.random.default_rng()
        self._sessions: dict[str, GameSession] = {}

        epoch = cp.get("epoch", "?")
        logger.info(
            f"Loaded model epoch {epoch}, supports "
            f"{self.min_players}-{self.max_players} players, "
            f"{num_simulations} sims/move, "
            f"search batch={self.search_batch_size}, "
            f"eval dtype={self.eval_dtype or 'float32'}"
        )

    def validate_player_count(self, num_players: int) -> None:
        if not self.min_players <= num_players <= self.max_players:
            raise ValueError(
                f"Checkpoint supports {self.min_players}-{self.max_players} "
                f"players, got {num_players}"
            )

    def _session_for(self, game_data: dict) -> GameSession:
        gid = str(game_data.get("id", ""))
        num_players = len(game_data.get("players", []))
        session = self._sessions.get(gid)
        if session is None or session.num_players != num_players:
            session = GameSession(num_players, max_players=self.max_players)
            self._sessions[gid] = session
        return session

    def _mcts_config_for(self, num_players: int) -> MCTSConfig:
        base = self.config.to_mcts_config(
            num_simulations_override=self.num_simulations,
            num_players=num_players,
        )
        return MCTSConfig(
            num_simulations=base.num_simulations,
            c_puct=base.c_puct,
            dirichlet_alpha=base.dirichlet_alpha,
            dirichlet_epsilon=base.dirichlet_epsilon,
            dirichlet_dynamic=base.dirichlet_dynamic,
            dirichlet_alpha_numerator=base.dirichlet_alpha_numerator,
            num_players=num_players,
            search_batch_size=self.search_batch_size,
        )

    def process_turn(
        self,
        game_data: dict,
        bot_player_idx: int,
        bot_user_id=None,
        bot_user_ids: set | None = None,
    ) -> list[dict]:
        """Sync state, run MCTS, return API-format actions to post."""
        num_players = len(game_data.get("players", []))
        self.validate_player_count(num_players)
        session = self._session_for(game_data)
        state = session.sync(game_data)
        prepare_live_decision_state(state)

        if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            logger.info("Game is over")
            return []

        engine_player_idx = bot_player_idx
        if bot_user_id is not None:
            engine_player_idx = session.player_index_for_user_id(bot_user_id)
        bot_player_indices = set()
        for user_id in bot_user_ids or ():
            try:
                bot_player_indices.add(session.player_index_for_user_id(user_id))
            except ValueError:
                continue
        if bot_user_id is not None:
            bot_player_indices.add(engine_player_idx)

        aligned_passes = _align_unordered_round_to_18xx_actor(
            game_data,
            session,
            state,
            bot_player_indices=bot_player_indices,
        )
        retargeted_closing = _retarget_closing_active_player_to_bot(
            game_data,
            state,
            bot_user_id,
            engine_player_idx,
        )
        active = TURN.get_active_player(state)
        if aligned_passes:
            logger.info(
                "Applied local unordered pass alignment: "
                f"passes={aligned_passes}, active=P{active}"
            )
        if retargeted_closing:
            logger.info(
                "Retargeted unordered Closing decision to acting bot: "
                f"active=P{active}"
            )
        logger.info(
            "Player mapping: "
            f"bot_user_id={bot_user_id}, api_idx={bot_player_idx}, "
            f"engine_idx={engine_player_idx}, active=P{active}"
        )

        compatibility_action = _acquisition_compatibility_action(
            game_data,
            session,
            state,
            bot_user_id,
            engine_player_idx,
        )
        if compatibility_action is None:
            compatibility_action = _closing_compatibility_action(
                game_data,
                state,
                bot_user_id,
                engine_player_idx,
            )
        if compatibility_action is None:
            compatibility_action = _dividend_compatibility_action(
                game_data,
                session,
                state,
                bot_user_id,
                engine_player_idx,
            )
        if compatibility_action is not None:
            logger.info(
                "18xx compatibility action: "
                f"{compatibility_action}"
            )
            return [compatibility_action]

        if active != engine_player_idx:
            logger.info(
                f"Not our turn yet: active=P{active}, bot=P{engine_player_idx}"
            )
            return []

        mismatches = session.validate_against_18xx(
            game_data,
            state,
            context=f"game={game_data.get('id', '?')}",
        )
        if mismatches:
            logger.error(
                "18xx/RSS replay mismatch before MCTS; refusing to move:\n%s",
                format_state_mismatches(mismatches),
            )
            return []

        previous_step_mode = state.step_mode
        state.step_mode = True
        try:
            return self._plan_live_actions(
                state, game_data, engine_player_idx, num_players,
                session.committed_ids, bot_user_id=bot_user_id,
            )
        finally:
            state.step_mode = previous_step_mode

    def _plan_live_actions(
        self,
        state,
        game_data: dict,
        bot_player_idx: int,
        num_players: int,
        committed_ids: set,
        bot_user_id=None,
    ) -> list[dict]:
        """Choose and apply engine steps until the bot no longer acts."""
        composer = _LiveActionComposer(
            game_data, bot_player_idx, committed_ids, bot_user_id=bot_user_id,
        )
        reuse_root = None
        max_steps = 100

        for _ in range(max_steps):
            phase = TURN.get_phase(state)
            if phase == GamePhases.PHASE_GAME_OVER:
                break

            if phase in AUTOMATED_PHASES:
                status = DRIVER.advance_phase(state)
                if status == STATUS_INVALID:
                    raise RuntimeError(f"Invalid automated advance in phase {phase}")
                if status == STATUS_GAME_OVER:
                    break
                continue

            active = TURN.get_active_player(state)
            if active != bot_player_idx:
                break

            legal_actions = get_legal_actions(state)
            if not legal_actions:
                raise RuntimeError(f"No legal actions in live phase {phase}")

            model_eval = None
            if self.model_output:
                model_eval = self._evaluator.evaluate(state)

            if len(legal_actions) == 1:
                action_idx = legal_actions[0][0]
                if self.model_output:
                    assert model_eval is not None
                    priors, values, action_ids_arr, _, phase_id = model_eval
                    print(
                        _format_live_model_output(
                            game_data=game_data,
                            state=state,
                            priors=priors,
                            values=values,
                            action_ids=action_ids_arr,
                            phase_id=int(phase_id),
                            num_players=num_players,
                            root=None,
                            action_idx=action_idx,
                            elapsed_secs=None,
                        ),
                        flush=True,
                    )
            else:
                search_reuse = (
                    reuse_root
                    if self._reuse_root_matches_state(reuse_root, state)
                    else None
                )
                if reuse_root is not None and search_reuse is None:
                    reuse_root = None
                action_idx, root, elapsed, search_stats = self._search(
                    state, num_players, reuse_root=search_reuse,
                )
                if self.model_output:
                    assert model_eval is not None
                    assert search_stats is not None
                    priors, values, action_ids_arr, _, phase_id = model_eval
                    print(
                        _format_live_model_output(
                            game_data=game_data,
                            state=state,
                            priors=priors,
                            values=values,
                            action_ids=action_ids_arr,
                            phase_id=int(phase_id),
                            num_players=num_players,
                            root=root,
                            action_idx=action_idx,
                            mcts_config=self._mcts_config_for(num_players),
                            search_stats=search_stats,
                            elapsed_secs=elapsed,
                        ),
                        flush=True,
                    )
                reuse_root = prepare_reuse_root(root, action_idx, self._state_pool)

            intent = engine_action_to_18xx(
                action_idx, state, self.max_players,
            )
            action_count_before = len(composer.actions)
            composer.add_step(phase, intent, state)

            status = DRIVER.apply_action(state, action_idx)
            if status == STATUS_INVALID:
                raise RuntimeError(f"Invalid engine action {action_idx} in phase {phase}")
            if status == STATUS_GAME_OVER:
                break
            if len(composer.actions) > action_count_before:
                if _should_continue_after_postable_action(
                    phase,
                    state,
                    bot_player_idx,
                ):
                    continue
                break

        else:
            raise RuntimeError("Exceeded live action step limit")

        return composer.finish()

    def _reuse_root_matches_state(self, reuse_root, state) -> bool:
        if reuse_root is None or reuse_root.state_idx < 0:
            return False
        return np.array_equal(
            self._state_pool.row(reuse_root.state_idx),
            state._array,
        )

    def _search(self, state, num_players: int, reuse_root=None):
        """Run MCTS and return the best action index."""
        search_stats = SearchStats() if self.model_output else None
        t0 = time.monotonic()
        root = run_search(
            None if reuse_root is not None else state,
            self._evaluator,
            self._mcts_config_for(num_players),
            self._rng,
            state_pool=self._state_pool,
            reuse_root=reuse_root,
            profile=search_stats,
        )
        elapsed = time.monotonic() - t0

        assert root.legal_actions is not None and root.visit_counts is not None
        best_idx = int(np.argmax(root.visit_counts))
        action_idx = int(root.legal_actions[best_idx])
        top_visits = int(root.visit_counts[best_idx])
        total_visits = int(root.visit_counts.sum())

        logger.info(
            f"MCTS: action={action_idx}, visits={top_visits}/{total_visits}, "
            f"time={elapsed:.1f}s"
        )
        return action_idx, root, elapsed, search_stats


# ---------------------------------------------------------------------------
# Auth config
# ---------------------------------------------------------------------------


def load_auth(runtime_dir: Path) -> dict[str, dict]:
    """Load auth.json: {bot_name: {token: str, ...}}."""
    path = runtime_dir / "auth.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Auth config not found: {path}\n"
            f"Create it with format: "
            f'{{"rss-az-1": {{"token": "hex_session_token"}}, ...}}'
        )
    with open(path) as f:
        return json.load(f)


def load_models_config(runtime_dir: Path) -> dict[str, str]:
    """Load models.json: exact count/range/fallback keys to checkpoint paths."""
    path = runtime_dir / "models.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Models config not found: {path}\n"
            f"Create it with format: "
            f'{{"3-5": "checkpoints/checkpoint_epoch_0050.pt"}}'
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Webhook HTTP handler
# ---------------------------------------------------------------------------


class WebhookHandler(BaseHTTPRequestHandler):
    """Receives webhook notifications and enqueues turn events."""

    work_queue: queue.Queue  # set by LiveService before serving
    auth: dict[str, dict]  # set by LiveService

    def do_GET(self):
        logger.info(f"Incoming GET: {self.path}")
        if self._is_poke_path():
            self._handle_poke()
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        logger.info(f"Incoming POST: {self.path}")

        if self._is_poke_path():
            self._handle_poke()
            return

        # Extract bot name from URL path: /webhook/<bot_name>
        path_parts = self.path.strip("/").split("/")
        if len(path_parts) != 2 or path_parts[0] != "webhook":
            logger.warning(f"Rejected path: {self.path}")
            self.send_response(404)
            self.end_headers()
            return

        bot_name = path_parts[1]
        if bot_name not in self.auth:
            logger.warning(f"Unknown bot name in webhook: {bot_name}")
            self.send_response(404)
            self.end_headers()
            return

        # Parse body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.warning(
                f"Webhook JSON parse failed: bot={bot_name}, body={body[:500]!r}"
            )
            self.send_response(400)
            self.end_headers()
            return

        # Extract game_id from webhook text
        text = data.get("text", "")
        if not text and "content" in data:
            text = data.get("content", "")
        logger.info(
            "Webhook payload: "
            f"bot={bot_name}, keys={sorted(data.keys())}, "
            f"text={str(text)[:500]!r}"
        )

        # Only respond to turn notifications.
        if not is_turn_webhook_text(str(text)):
            logger.info(
                f"Ignoring webhook for bot={bot_name}: "
                f"missing turn marker"
            )
            self.send_response(200)
            self.end_headers()
            return

        game_id, _ = parse_webhook_text(text)

        if not game_id:
            logger.warning(
                f"Ignoring webhook for bot={bot_name}: "
                f"could not parse game id from text={text[:500]!r}"
            )
            self.send_response(200)
            self.end_headers()
            return

        logger.info(f"Webhook: bot={bot_name}, game={game_id}")
        self.work_queue.put((bot_name, game_id))

        self.send_response(200)
        self.end_headers()

    def _is_poke_path(self) -> bool:
        parsed = urlparse(self.path)
        return parsed.path.strip("/").split("/", 1)[0] == "poke"

    def _handle_poke(self):
        game_id = parse_poke_game_id(self.path)
        if game_id is None:
            game_id = self._read_json_body_game_id()

        if not game_id:
            self._send_json(
                400,
                {"error": "missing_game_id", "usage": "/poke/<game_id>"},
            )
            return

        bot_names = list(self.auth.keys())
        for bot_name in bot_names:
            self.work_queue.put((bot_name, game_id))

        logger.info(
            f"Manual poke: game={game_id}, queued bots={', '.join(bot_names)}"
        )
        self._send_json(
            202,
            {"status": "queued", "game_id": game_id, "bots": bot_names},
        )

    def _read_json_body_game_id(self) -> str | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            return None

        body = self.rfile.read(content_length).decode()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None

        game_id = data.get("game_id") or data.get("game")
        return str(game_id) if game_id else None

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        """Suppress default HTTP logging — we use our own logger."""
        pass


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class MoveWorker(threading.Thread):
    """Dequeues turn notifications and processes them."""

    def __init__(
        self,
        work_queue: queue.Queue,
        api: ApiClient,
        auth: dict[str, dict],
        registry: ModelRegistry,
    ):
        super().__init__(daemon=True, name="move-worker")
        self._queue = work_queue
        self._api = api
        self._auth = auth
        self._registry = registry

    def run(self):
        logger.info("Move worker started")
        while True:
            try:
                bot_name, game_id = self._queue.get()
                self._process(bot_name, game_id)
            except Exception:
                logger.exception("Unhandled error in move worker")
            finally:
                self._queue.task_done()

    def _process(self, bot_name: str, game_id: str):
        token = self._auth[bot_name]["token"]

        # Fetch game data
        try:
            game_data = self._api.fetch_game(game_id, token)
        except TransientError:
            logger.error(
                f"[{bot_name}] Failed to fetch game {game_id} after retries"
            )
            return
        except PermanentError as e:
            logger.error(f"[{bot_name}] Cannot fetch game {game_id}: {e}")
            return

        # Determine player count and our player index
        players = game_data.get("players", [])
        num_players = len(players)

        # Find which player slot is ours
        bot_player_idx = None
        bot_user_id = self._auth[bot_name].get("user_id")

        if bot_user_id is not None:
            for i, p in enumerate(players):
                if _same_id(p["id"], bot_user_id):
                    bot_player_idx = i
                    break
        else:
            # Try matching by name
            for i, p in enumerate(players):
                if p.get("name") == bot_name:
                    bot_player_idx = i
                    bot_user_id = p.get("id")
                    break

        if bot_player_idx is None:
            logger.error(
                f"[{bot_name}] Not found in game {game_id} players: "
                f"{[p.get('name') for p in players]}"
            )
            return

        # Check if it's actually our turn (acting contains user_ids)
        acting = game_data.get("acting", [])
        if acting and not any(_same_id(bot_user_id, actor) for actor in acting):
            logger.info(f"[{bot_name}] Not in acting list for game {game_id}")
            return

        # Get the right model for this player count
        try:
            engine = self._registry.get_engine(num_players)
        except (ValueError, FileNotFoundError) as e:
            logger.error(f"[{bot_name}] {e}")
            return

        bot_user_ids = {
            auth_info.get("user_id")
            for auth_info in self._auth.values()
            if auth_info.get("user_id") is not None
        }

        # Loop: play moves until it's no longer our turn.
        # A single webhook may require multiple actions (e.g. sequential
        # IPO decisions, ACQ offers, or Closing choices).
        max_consecutive = 50  # safety limit
        for _ in range(max_consecutive):
            try:
                api_actions = engine.process_turn(
                    game_data,
                    bot_player_idx,
                    bot_user_id=bot_user_id,
                    bot_user_ids=bot_user_ids,
                )
            except Exception:
                logger.exception(
                    f"[{bot_name}] Error processing turn in game {game_id}"
                )
                return

            if not api_actions:
                return

            # Post each action (IPO+PAR is a single compound action)
            for post_idx, action in enumerate(api_actions):
                try:
                    action = attach_expected_auto_actions(game_data, action)
                except Exception:
                    logger.exception(
                        f"[{bot_name}] Failed to compute auto_actions "
                        f"for game {game_id}: {action}"
                    )
                    return

                logger.info(
                    f"[{bot_name}] Posting to game {game_id}: {action}"
                )
                try:
                    self._api.post_action(game_id, action, token)
                except PermanentError as e:
                    logger.error(
                        f"[{bot_name}] Action rejected by server: {e}\n"
                        f"  Action: {json.dumps(action)}"
                    )
                    return
                except TransientError:
                    logger.error(
                        f"[{bot_name}] Failed to post action after retries"
                    )
                    return

                if post_idx < len(api_actions) - 1:
                    try:
                        game_data = self._api.fetch_game(game_id, token)
                    except (TransientError, PermanentError) as e:
                        logger.error(
                            f"[{bot_name}] Failed to fetch game {game_id} "
                            f"after posting action: {e}"
                        )
                        return

            # Wait for the server to process the action before re-fetching
            time.sleep(5)

            # Re-fetch game data and check if we're still acting
            try:
                game_data = self._api.fetch_game(game_id, token)
            except (TransientError, PermanentError) as e:
                logger.error(
                    f"[{bot_name}] Failed to re-fetch game {game_id}: {e}"
                )
                return

            acting = game_data.get("acting", [])
            if not any(_same_id(bot_user_id, actor) for actor in acting):
                logger.info(
                    f"[{bot_name}] Turn complete in game {game_id}"
                )
                return


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class LiveService:
    """Webhook server + move worker."""

    def __init__(
        self,
        api: ApiClient,
        auth: dict[str, dict],
        registry: ModelRegistry,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self._api = api
        self._auth = auth
        self._registry = registry
        self._host = host
        self._port = port
        self._work_queue: queue.Queue = queue.Queue()

    def start(self):
        """Start webhook server and worker thread."""
        # Configure handler class attributes
        WebhookHandler.work_queue = self._work_queue
        WebhookHandler.auth = self._auth

        # Start worker
        worker = MoveWorker(
            self._work_queue, self._api, self._auth, self._registry,
        )
        worker.start()

        # Start HTTP server (blocks)
        server = HTTPServer((self._host, self._port), WebhookHandler)
        logger.info(
            f"Webhook server listening on {self._host}:{self._port}"
        )
        logger.info(
            f"Bot accounts: {', '.join(self._auth.keys())}"
        )
        logger.info(
            "Set webhook URLs to: "
            f"http://<host>:{self._port}/webhook/<bot_name>"
        )
        logger.info(
            "Manual poke URL: "
            f"http://<host>:{self._port}/poke/<game_id>"
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down")
            server.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Live play webhook server for 18xx.games",
    )
    parser.add_argument(
        "--runtime-dir",
        type=str,
        default="runtime",
        help="Directory with auth.json, models.json (default: runtime)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:9292",
        help="18xx.games base URL (default: http://localhost:9292)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
        help="MCTS simulations per move (default: 400)",
    )
    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=None,
        help="MCTS search batch size override (default: checkpoint config)",
    )
    parser.add_argument(
        "--eval-dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default=None,
        help="Eval inference dtype override (default: checkpoint config)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for 'latest' checkpoint resolution",
    )
    parser.add_argument(
        "--model-output",
        action="store_true",
        help=(
            "Print analyzer-style model values, priors, MCTS visits, and "
            "A0GB values for each live decision"
        ),
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="compile_model",
        action="store_true",
        default=False,
        help="Opt into torch.compile for live inference",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="compile_model",
        action="store_false",
        help="Disable torch.compile for live inference (default)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runtime_dir = Path(args.runtime_dir)
    auth = load_auth(runtime_dir)
    models_config = load_models_config(runtime_dir)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            logger.debug("Torch interop thread pool was already initialized")
        logger.info("Limited PyTorch CPU worker threads for live CUDA inference")

    registry = ModelRegistry(
        models_config,
        device,
        num_simulations=args.simulations,
        search_batch_size=args.search_batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
        compile_model=args.compile_model,
        model_output=args.model_output,
        eval_dtype=args.eval_dtype,
    )

    api = ApiClient(args.base_url)

    service = LiveService(
        api=api,
        auth=auth,
        registry=registry,
        host=args.host,
        port=args.port,
    )
    service.start()


if __name__ == "__main__":
    main()
