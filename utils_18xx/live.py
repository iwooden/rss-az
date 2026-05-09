"""Webhook-driven live play against humans on 18xx.games.

Receives turn notifications via webhook, fetches game state from the
18xx.games API, runs MCTS search, and posts the selected move back.

Usage:
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime
    .venv/bin/python -m utils_18xx.live --runtime-dir runtime --simulations 800
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

import numpy as np
import torch

from core.data import (
    COMPANY_NAMES,
    CORP_NAMES,
    GamePhases,
)
from core.state import get_layout
from entities.corp import CORPS
from entities.market import MARKET
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, run_search
from nn import get_model_input_spec
from train.checkpoint import find_latest_checkpoint, load_model_from_checkpoint
from train.config import MCTSConfig

from .action_mapper import engine_action_to_18xx
from .api_client import ApiClient, PermanentError, TransientError
from .game_session import GameSession

logger = logging.getLogger(__name__)

ACQ_PHASES = (
    GamePhases.PHASE_ACQ_SELECT_CORP,
    GamePhases.PHASE_ACQ_SELECT_COMPANY,
    GamePhases.PHASE_ACQ_SELECT_PRICE,
    GamePhases.PHASE_ACQ_OFFER,
)

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


def _same_id(left, right) -> bool:
    return str(left) == str(right)


# ---------------------------------------------------------------------------
# Action format translation
# ---------------------------------------------------------------------------


def intent_to_api_action(
    intent: dict,
    state,
    game_data: dict,
    bot_player_idx: int,
    committed_ids: set,
) -> dict:
    """Convert an engine intent dict to the 18xx.games API action format.

    Maps entity/entity_type and adds fields the API expects.
    """
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
        share_id = _resolve_buyable_share(game_data, corp_name, committed_ids)
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
        share_id = _resolve_issuable_share(game_data, corp_name, committed_ids)
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


def _get_corp_share_price(state, corp_name: str) -> int:
    """Get a corporation's current share price from engine state."""
    from core.data import CORP_NAME_TO_ID

    corp_id = CORP_NAME_TO_ID[corp_name]
    return CORPS[corp_id].get_share_price(state)


# ---------------------------------------------------------------------------
# Share ID resolution
# ---------------------------------------------------------------------------

# In 18xx.games, shares are identified as "{CORP}_{index}".
# Share 0 is the president's share (20%), shares 1-N are regular (10%).
# We track ownership by replaying the action history.


def _build_share_ownership(
    game_data: dict,
    committed_ids: set,
) -> tuple[dict[str, list[int]], dict[int, dict[str, list[int]]]]:
    """Build share ownership maps from game action history.

    Args:
        game_data: Full game JSON from the 18xx API.
        committed_ids: Set of committed action IDs (from Ruby extractor).

    Returns:
        (bank_shares, player_shares) where:
        - bank_shares: {corp_name: [share_indices in bank]}
        - player_shares: {user_id: {corp_name: [share_indices held]}}
    """
    # All 10 shares per corp start in the bank.
    bank: dict[str, list[int]] = {}
    players: dict[int, dict[str, list[int]]] = {}

    for p in game_data.get("players", []):
        players[p["id"]] = {}

    for action in game_data.get("actions", []):
        action_id = action.get("id")
        if action_id is not None and action_id not in committed_ids:
            continue
        atype = action.get("type")

        if atype == "par":
            # President buys share 0
            corp = action["corporation"]
            if corp not in bank:
                bank[corp] = list(range(10))
            # Find who triggered this — look at the user field
            user_id = action.get("user")
            if user_id is not None:
                if 0 in bank[corp]:
                    bank[corp].remove(0)
                    players.setdefault(user_id, {}).setdefault(
                        corp, [],
                    ).append(0)

        elif atype == "buy_shares":
            shares = action.get("shares", [])
            user_id = action.get("entity")
            if not isinstance(user_id, int):
                continue
            for share_ref in shares:
                corp, idx = _parse_share_id(share_ref)
                if corp and idx is not None:
                    if corp in bank and idx in bank[corp]:
                        bank[corp].remove(idx)
                    players.setdefault(user_id, {}).setdefault(
                        corp, [],
                    ).append(idx)

        elif atype == "sell_shares":
            shares = action.get("shares", [])
            entity = action.get("entity")
            entity_type = action.get("entity_type")

            if entity_type == "player" and isinstance(entity, int):
                # Player selling to bank
                for share_ref in shares:
                    corp, idx = _parse_share_id(share_ref)
                    if corp and idx is not None:
                        p_shares = players.get(entity, {}).get(corp, [])
                        if idx in p_shares:
                            p_shares.remove(idx)
                        bank.setdefault(corp, []).append(idx)
            elif entity_type == "corporation" and isinstance(entity, str):
                # Corp issuing shares (selling from treasury)
                for share_ref in shares:
                    corp, idx = _parse_share_id(share_ref)
                    if corp and idx is not None:
                        bank.setdefault(corp, []).append(idx)

    return bank, players


def _parse_share_id(share_ref: str) -> tuple[str | None, int | None]:
    """Parse 'IC_2' into ('IC', 2)."""
    parts = share_ref.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None, None


def _resolve_buyable_share(
    game_data: dict, corp_name: str, committed_ids: set,
) -> str:
    """Find the lowest-numbered share available in the bank pool."""
    bank, _ = _build_share_ownership(game_data, committed_ids)
    available = sorted(bank.get(corp_name, []))
    if not available:
        raise ValueError(f"No shares of {corp_name} available in bank")
    # Skip president's share (index 0) — it's bought via PAR, not buy_shares
    non_president = [i for i in available if i > 0]
    idx = non_president[0] if non_president else available[0]
    return f"{corp_name}_{idx}"


def _resolve_sellable_share(
    game_data: dict, corp_name: str, user_id: int, committed_ids: set,
) -> str:
    """Find a share the player can sell (highest non-president share)."""
    _, player_shares = _build_share_ownership(game_data, committed_ids)
    held = sorted(
        player_shares.get(user_id, {}).get(corp_name, []), reverse=True,
    )
    if not held:
        raise ValueError(
            f"User {user_id} holds no shares of {corp_name}"
        )
    # Prefer selling non-president shares
    non_president = [i for i in held if i > 0]
    idx = non_president[0] if non_president else held[0]
    return f"{corp_name}_{idx}"


def _resolve_issuable_share(
    game_data: dict, corp_name: str, committed_ids: set,
) -> str:
    """Find the next share a corp can issue (from treasury/unissued)."""
    bank, player_shares = _build_share_ownership(game_data, committed_ids)
    # Issued shares are those held by bank pool or players.
    # Unissued shares are those not yet in circulation.
    in_circulation: set[int] = set()
    for idx in bank.get(corp_name, []):
        in_circulation.add(idx)
    for user_shares in player_shares.values():
        for idx in user_shares.get(corp_name, []):
            in_circulation.add(idx)
    # Find lowest unissued share (skip 0 = president)
    for i in range(1, 10):
        if i not in in_circulation:
            return f"{corp_name}_{i}"
    raise ValueError(f"No unissued shares of {corp_name}")


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
        search_batch_size: int,
        checkpoint_dir: Path | None = None,
    ):
        self._config = models_config
        self._device = device
        self._num_simulations = num_simulations
        self._search_batch_size = search_batch_size
        self._checkpoint_dir = checkpoint_dir
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
        search_batch_size: int,
    ):
        self.device = device
        self.num_simulations = num_simulations
        self.search_batch_size = search_batch_size

        model, self.config, cp = load_model_from_checkpoint(checkpoint_path, device)
        model.eval()

        self.min_players = self.config.effective_min_players
        self.max_players = self.config.effective_max_players
        input_spec = get_model_input_spec(self.config)

        self._evaluator = NNEvaluator(
            model,
            device,
            num_players=self.max_players,
            terminal_rank_weight=self.config.terminal_blend,
            eval_dtype=self.config.eval_dtype,
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
            f"{num_simulations} sims/move"
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
    ) -> list[dict]:
        """Sync state, run MCTS, return API-format actions to post."""
        num_players = len(game_data.get("players", []))
        self.validate_player_count(num_players)
        session = self._session_for(game_data)
        state = session.sync(game_data)
        phase = TURN.get_phase(state)

        if phase == GamePhases.PHASE_GAME_OVER:
            logger.info("Game is over")
            return []

        active = state.get_active_player()
        if active != bot_player_idx:
            logger.info(
                f"Not our turn yet: active=P{active}, bot=P{bot_player_idx}"
            )
            return []

        actions: list[dict] = []

        if phase == GamePhases.PHASE_IPO:
            actions = self._handle_ipo_par(
                session, state, game_data, bot_player_idx,
            )
        elif phase in ACQ_PHASES:
            actions = self._handle_acquisition(
                session, state, game_data, bot_player_idx,
            )
        else:
            action_idx = self._search(state, num_players)
            intent = engine_action_to_18xx(
                action_idx, state, self.max_players,
            )
            # Translate BEFORE apply — apply can auto-advance the phase
            api_action = intent_to_api_action(
                intent, state, game_data, bot_player_idx,
                session.committed_ids,
            )
            session.apply_engine_action(action_idx)
            actions = [api_action]

        return actions

    def _handle_ipo_par(
        self, session: GameSession, state, game_data: dict, bot_player_idx: int,
    ) -> list[dict]:
        """IPO + PAR: two sequential searches combined into one par action."""
        num_players = len(game_data.get("players", []))
        action_idx = self._search(state, num_players)
        intent = engine_action_to_18xx(
            action_idx, state, self.max_players,
        )

        if intent["type"] == "pass":
            api_action = intent_to_api_action(
                intent, state, game_data, bot_player_idx,
                session.committed_ids,
            )
            session.apply_engine_action(action_idx)
            return [api_action]

        # IPO selected a corp — get the company before apply
        ipo_company_id = TURN.get_active_company(state)
        ipo_company = COMPANY_NAMES[ipo_company_id]
        ipo_corp = intent["corporation"]

        session.apply_engine_action(action_idx)

        if TURN.get_phase(state) != GamePhases.PHASE_PAR:
            # PAR was auto-applied (only one valid price)
            # We still need to tell the API about it
            logger.info(f"PAR auto-applied for {ipo_corp}")
            # Get the par price from the corp's share price
            from core.data import CORP_NAME_TO_ID
            corp_id = CORP_NAME_TO_ID[ipo_corp]
            par_price = CORPS[corp_id].get_share_price(state)
            col = MARKET.get_index_for_price(par_price)
            return [{
                "type": "par",
                "entity": ipo_company,
                "entity_type": "company",
                "corporation": ipo_corp,
                "share_price": f"{par_price},0,{col}",
            }]

        # Search for PAR price
        par_action = self._search(state, num_players)
        par_intent = engine_action_to_18xx(
            par_action, state, self.max_players,
        )
        session.apply_engine_action(par_action)

        return [{
            "type": "par",
            "entity": ipo_company,
            "entity_type": "company",
            "corporation": ipo_corp,
            "share_price": par_intent["share_price"],
        }]

    def _handle_acquisition(
        self,
        session: GameSession,
        state,
        game_data: dict,
        bot_player_idx: int,
    ) -> list[dict]:
        """Run split ACQ decisions until one postable 18xx action is chosen."""
        num_players = len(game_data.get("players", []))

        for _ in range(8):
            phase = TURN.get_phase(state)
            if phase not in ACQ_PHASES:
                return []

            action_idx = self._search(state, num_players)
            intent = engine_action_to_18xx(
                action_idx, state, self.max_players,
            )

            if intent["type"] in {"select_corp", "select_company"}:
                session.apply_engine_action(action_idx)
                continue

            api_action = intent_to_api_action(
                intent, state, game_data, bot_player_idx,
                session.committed_ids,
            )
            session.apply_engine_action(action_idx)
            return [api_action]

        raise RuntimeError("Exceeded split acquisition decision limit")

    def _search(self, state, num_players: int) -> int:
        """Run MCTS and return the best action index."""
        t0 = time.monotonic()
        root = run_search(
            state, self._evaluator, self._mcts_config_for(num_players), self._rng,
            state_pool=self._state_pool,
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
        return action_idx


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

    def do_POST(self):
        logger.info(f"Incoming POST: {self.path}")

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
            self.send_response(400)
            self.end_headers()
            return

        # Extract game_id from webhook text
        text = data.get("text", "")

        # Only respond to "Your Turn" notifications
        if "Your Turn" not in text:
            logger.debug(f"Ignoring non-turn webhook: {text[:80]}")
            self.send_response(200)
            self.end_headers()
            return

        game_id, _ = parse_webhook_text(text)

        if not game_id:
            logger.warning(f"Could not parse game_id from webhook: {text}")
            self.send_response(200)
            self.end_headers()
            return

        logger.info(f"Webhook: bot={bot_name}, game={game_id}")
        self.work_queue.put((bot_name, game_id))

        self.send_response(200)
        self.end_headers()

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

        # Loop: play moves until it's no longer our turn.
        # A single webhook may require multiple actions (e.g. sequential
        # IPO decisions, or ACQ offers for multiple corps).
        max_consecutive = 50  # safety limit
        for _ in range(max_consecutive):
            try:
                api_actions = engine.process_turn(game_data, bot_player_idx)
            except Exception:
                logger.exception(
                    f"[{bot_name}] Error processing turn in game {game_id}"
                )
                return

            if not api_actions:
                return

            # Post each action (IPO+PAR is a single compound action)
            for action in api_actions:
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
        default=1,
        help="MCTS search batch size (default: 1)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for 'latest' checkpoint resolution",
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

    registry = ModelRegistry(
        models_config,
        device,
        num_simulations=args.simulations,
        search_batch_size=args.search_batch_size,
        checkpoint_dir=Path(args.checkpoint_dir),
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
