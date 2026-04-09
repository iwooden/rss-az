"""HTTP API server for human-vs-AI Rolling Stock Stars play.

Loads a trained checkpoint and serves AI moves via MCTS search.
The 18xx.games frontend sends game_data, and this server returns
intent dicts that the frontend translates into Engine::Action objects.

Usage:
    .venv/bin/python -m utils_18xx.server latest
    .venv/bin/python -m utils_18xx.server latest --simulations 400
    .venv/bin/python -m utils_18xx.server checkpoints/checkpoint_epoch_0010.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from core.data import (
    COMPANY_NAME_TO_ID,
    COMPANY_NAMES,
    CORP_NAME_TO_ID,
    CORP_NAMES,
    GamePhases,
)
from core.state import get_layout
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, run_search
from nn import create_model
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import MCTSConfig, TrainingConfig

from .action_mapper import engine_action_to_18xx
from .game_session import GameSession

logger = logging.getLogger(__name__)

# Map 18xx frontend round names to the engine phases they correspond to.
# When the frontend round doesn't match our engine phase, we send passes.
_ROUND_TO_PHASES: dict[str, set[int]] = {
    "Investment": {GamePhases.PHASE_INVEST, GamePhases.PHASE_BID},
    "Acquisition": {GamePhases.PHASE_ACQUISITION},
    "Closing": {GamePhases.PHASE_CLOSING},
    "Dividends": {GamePhases.PHASE_DIVIDENDS},
    "Issue Shares": {GamePhases.PHASE_ISSUE_SHARES},
    "IPO": {GamePhases.PHASE_IPO, GamePhases.PHASE_PAR},
}


class AIServer:
    """Manages model, evaluator, and game sessions for AI play."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        num_simulations: int = 400,
        search_batch_size: int = 1,
        c_puct: float | None = None,
        dirichlet_epsilon: float | None = None,
        dirichlet_dynamic: bool | None = None,
        terminal_blend: float | None = None,
        num_players: int = 3,
    ):
        self.device = device
        self.num_simulations = num_simulations
        self.search_batch_size = search_batch_size
        self.num_players = num_players
        self.checkpoint_path = checkpoint_path
        self._checkpoint_dir = checkpoint_path.parent

        # Override settings (None = use checkpoint defaults)
        self._c_puct = c_puct
        self._dirichlet_epsilon = dirichlet_epsilon
        self._dirichlet_dynamic = dirichlet_dynamic
        self._terminal_blend = terminal_blend

        # Load model
        self._load_checkpoint(checkpoint_path)

        # Game sessions keyed by game_id
        self._sessions: dict[str, GameSession] = {}

    def _load_checkpoint(self, cp_path: Path) -> None:
        """Load model from checkpoint."""
        logger.info(f"Loading checkpoint: {cp_path}")
        cp = load_checkpoint(cp_path, self.device)
        self._config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

        model = create_model(
            self._config.model_path,
            input_dim=self._config.visible_size,
            action_dim=self._config.action_dim,
            value_dim=self._config.num_players,
        ).to(self.device)
        model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
        model.eval()

        terminal_rank_weight = (
            self._terminal_blend
            if self._terminal_blend is not None
            else self._config.terminal_blend
        )

        self._model = model
        self._evaluator = NNEvaluator(
            model, self.device, num_players=self.num_players,
            terminal_rank_weight=terminal_rank_weight,
        )

        # Build MCTS config
        base_mcts = self._config.to_mcts_config(c_puct_override=self._c_puct)
        self._mcts_config = MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=base_mcts.c_puct,
            dirichlet_alpha=base_mcts.dirichlet_alpha,
            dirichlet_epsilon=(
                self._dirichlet_epsilon
                if self._dirichlet_epsilon is not None
                else base_mcts.dirichlet_epsilon
            ),
            dirichlet_dynamic=(
                self._dirichlet_dynamic
                if self._dirichlet_dynamic is not None
                else base_mcts.dirichlet_dynamic
            ),
            dirichlet_alpha_numerator=base_mcts.dirichlet_alpha_numerator,
            num_players=self.num_players,
            search_batch_size=self.search_batch_size,
        )

        layout = get_layout(self.num_players)
        self._state_pool = StatePool(
            2 * (self.num_simulations + 1), layout.total_size,
        )
        self._rng = np.random.default_rng()

        self._cp_path = cp_path
        self._cp_mtime = cp_path.stat().st_mtime
        epoch = cp.get("epoch", "?")
        logger.info(
            f"Model from epoch {epoch}, device={self.device}, "
            f"{self.num_simulations} simulations/move"
        )

    def _maybe_reload_checkpoint(self) -> None:
        """Reload if a newer checkpoint exists."""
        try:
            latest = find_latest_checkpoint(self._checkpoint_dir)
            if latest and latest != self._cp_path:
                self._load_checkpoint(latest)
            elif latest and latest.stat().st_mtime > self._cp_mtime:
                self._load_checkpoint(latest)
        except Exception as e:
            logger.warning(f"Checkpoint reload check failed: {e}")

    def get_session(self, game_data: dict) -> GameSession:
        """Get or create a GameSession for the given game."""
        gid = game_data.get("id", "")
        if gid not in self._sessions:
            self._sessions[gid] = GameSession(self.num_players)
        return self._sessions[gid]

    def get_ai_move(
        self,
        game_data: dict,
        state_checksum: dict | None = None,
        preemption: dict | None = None,
    ) -> dict:
        """Process a game state and return the AI's move(s).

        Returns a dict with:
            actions: list of intent dicts to apply
            search_info: diagnostic info about the search
        """
        self._maybe_reload_checkpoint()

        session = self.get_session(game_data)
        state = session.sync(game_data)

        if state_checksum:
            self._compare_checksum(state, state_checksum, session.num_players)

        if session.is_game_over():
            return {"actions": [], "error": "Game is over"}

        # FI right-of-first-refusal: use MCTS to decide preemption.
        if preemption:
            return self._run_preemption_query(session, state, preemption)

        # Determine which players are AI
        settings = game_data.get("settings") or {}
        human_idx = settings.get("human_player_index", 0)
        # -1 means spectator mode (no human player)
        spectator = human_idx == -1

        # Detect phase mismatch between the 18xx frontend and our engine.
        # State divergence from ACQ/CLO handling can cause the engines to
        # be in different phases.  When this happens, send passes so the
        # frontend can advance to catch up.
        frontend_round = game_data.get("round", "")
        acting = game_data.get("acting") or []
        ai_acting_ids = [pid for pid in acting if pid != human_idx]

        if frontend_round and ai_acting_ids:
            phase = TURN.get_phase(state)
            expected_phases = _ROUND_TO_PHASES.get(frontend_round)
            if expected_phases is not None and phase not in expected_phases:
                logger.info(
                    f"Frontend in {frontend_round} but engine at phase "
                    f"{phase} — returning passes for AI players {ai_acting_ids}"
                )
                pass_actions = [{"type": "pass"} for _ in ai_acting_ids]
                return {"actions": pass_actions, "search_info": {"auto_pass": True}}

        active = state.get_active_player()

        if not spectator and active == human_idx:
            return {"actions": [], "info": "Human's turn"}

        phase = TURN.get_phase(state)
        actions: list[dict] = []
        search_info: dict = {}

        from tests.debug_trace import format_action, format_state_compact
        logger.info(
            f"AI move request: phase={phase}, active=P{active}, "
            f"human=P{human_idx}, state={format_state_compact(state)}, "
            f"n_actions={len(game_data.get('actions', []))}"
        )

        if phase in (GamePhases.PHASE_ACQUISITION, GamePhases.PHASE_CLOSING):
            # Process ONE offer at a time so the frontend can resync
            # between each.  This avoids state divergence from the 18xx
            # engine's receivership step and right-of-first-refusal.
            actions, search_info = self._run_single_offer(session, state)
        elif phase == GamePhases.PHASE_IPO:
            # IPO might need PAR follow-up
            actions, search_info = self._run_ipo_par(session, state)
        else:
            # Single action phases
            action_idx, info = self._search_and_pick(state)
            intent = engine_action_to_18xx(action_idx, state, self.num_players)
            logger.info(
                f"AI chose: {format_action(action_idx, self.num_players, state)} "
                f"-> intent={intent}"
            )
            session.apply_engine_action(action_idx)
            actions = [intent]
            search_info = info

        return {"actions": actions, "search_info": search_info}

    def _search_and_pick(self, state) -> tuple[int, dict]:
        """Run MCTS search and return (best_action_idx, search_info)."""
        t0 = time.monotonic()
        root = run_search(
            state, self._evaluator, self._mcts_config, self._rng,
            state_pool=self._state_pool,
        )
        elapsed = time.monotonic() - t0

        assert root.legal_actions is not None and root.visit_counts is not None
        best_idx_in_legal = int(np.argmax(root.visit_counts))
        action_idx = int(root.legal_actions[best_idx_in_legal])

        total_visits = int(root.visit_counts.sum())
        top_visits = int(root.visit_counts[best_idx_in_legal])

        info = {
            "action_idx": action_idx,
            "total_visits": total_visits,
            "top_visits": top_visits,
            "confidence": top_visits / max(total_visits, 1),
            "search_time_ms": round(elapsed * 1000),
        }

        return action_idx, info

    def _run_ipo_par(self, session, state) -> tuple[list[dict], dict]:
        """Handle IPO + optional PAR compound action."""
        action_idx, info = self._search_and_pick(state)
        intent = engine_action_to_18xx(action_idx, state, self.num_players)

        if intent["type"] == "pass":
            session.apply_engine_action(action_idx)
            return [intent], info

        # IPO selected a corp — apply it, then search for PAR
        ipo_corp = intent["corporation"]
        # Get the IPO company before applying (it gets cleared after)
        ipo_company_id = TURN.get_ipo_company(state)
        from core.data import COMPANY_NAMES
        ipo_company = COMPANY_NAMES[ipo_company_id]

        session.apply_engine_action(action_idx)

        # Check if PAR was auto-applied (only one valid par price)
        if TURN.get_phase(state) != GamePhases.PHASE_PAR:
            # PAR was forced — just return the IPO intent, frontend
            # will see the combined result
            return [{"type": "par", "corporation": ipo_corp,
                     "company": ipo_company, "auto_par": True}], info

        # Search for PAR price
        par_action, par_info = self._search_and_pick(state)
        par_intent = engine_action_to_18xx(par_action, state, self.num_players)
        session.apply_engine_action(par_action)

        combined = {
            "type": "par",
            "corporation": ipo_corp,
            "company": ipo_company,
            "share_price": par_intent["share_price"],
        }

        return [combined], {**info, "par_search": par_info}

    def _compare_checksum(
        self, state, checksum: dict, num_players: int,
    ) -> None:
        """Compare frontend state checksum against our engine state."""
        mismatches: list[str] = []

        for i, p_info in enumerate(checksum.get("players", [])):
            if i >= num_players:
                break
            engine_cash = PLAYERS[i].get_cash(state)
            engine_nw = PLAYERS[i].get_net_worth(state)
            fe_cash = p_info.get("cash", 0)
            fe_nw = p_info.get("value", 0)
            if engine_cash != fe_cash:
                mismatches.append(
                    f"P{i} cash: engine={engine_cash} frontend={fe_cash}"
                )
            if engine_nw != fe_nw:
                mismatches.append(
                    f"P{i} net_worth: engine={engine_nw} frontend={fe_nw}"
                )
            # Compare owned companies
            engine_cos: list[str] = sorted(
                COMPANY_NAMES[cid]
                for cid in range(36)
                if COMPANIES[cid].get_location(state)
                == CompanyLocation.LOC_PLAYER
                and COMPANIES[cid].get_owner_id(state) == i
            )
            fe_cos = sorted(p_info.get("companies", []))
            if engine_cos != fe_cos:
                mismatches.append(
                    f"P{i} companies: engine={engine_cos} frontend={fe_cos}"
                )

        for c_info in checksum.get("corps", []):
            corp_name = c_info.get("name", "")
            corp_id = next(
                (j for j in range(8) if CORP_NAMES[j] == corp_name), -1
            )
            if corp_id < 0:
                continue
            corp = CORPS[corp_id]
            if not corp.is_active(state):
                mismatches.append(f"{corp_name}: floated in frontend but not engine")
                continue
            engine_cash = corp.get_cash(state)
            engine_sp = corp.get_share_price(state)
            fe_cash = c_info.get("cash", 0)
            fe_sp = c_info.get("share_price", 0)
            if engine_cash != fe_cash:
                mismatches.append(
                    f"{corp_name} cash: engine={engine_cash} frontend={fe_cash}"
                )
            if engine_sp != fe_sp:
                mismatches.append(
                    f"{corp_name} share_price: engine={engine_sp} frontend={fe_sp}"
                )
            engine_cos = sorted(
                COMPANY_NAMES[cid]
                for cid in range(36)
                if COMPANIES[cid].get_location(state)
                in (CompanyLocation.LOC_CORP, CompanyLocation.LOC_CORP_ACQ)
                and COMPANIES[cid].get_owner_id(state) == corp_id
            )
            fe_cos = sorted(c_info.get("companies", []))
            if engine_cos != fe_cos:
                mismatches.append(
                    f"{corp_name} companies: engine={engine_cos} frontend={fe_cos}"
                )

        if mismatches:
            logger.warning(
                "STATE CHECKSUM MISMATCH:\n  " + "\n  ".join(mismatches)
            )
        else:
            logger.debug("State checksum OK")

    def _run_single_offer(
        self, session, state,
    ) -> tuple[list[dict], dict]:
        """Process ONE ACQ/CLO offer and return it.

        Returning a single action per call lets the frontend apply it,
        resync state (including 18xx receivership auto-buys), and then
        ask for the next action.  This prevents state divergence from
        the 18xx engine's two-step ACQ round.
        """
        action_idx, info = self._search_and_pick(state)
        intent = engine_action_to_18xx(action_idx, state, self.num_players)
        session.apply_engine_action(action_idx)
        return [intent], info

    def _run_preemption_query(
        self, session: GameSession, state, preemption: dict,
    ) -> dict:
        """Decide whether a corp should preempt an FI purchase via MCTS.

        Walks the engine's ACQ offer buffer (passing non-target offers)
        until the preempting corp's offer for the target company appears,
        then runs MCTS to decide buy vs pass.
        """
        preempting_corp_name = preemption["preempting_corp"]
        company_name = preemption["company"]
        original_corp_name = preemption["corporation"]

        preempting_corp_id = CORP_NAME_TO_ID[preempting_corp_name]
        company_id = COMPANY_NAME_TO_ID[company_name]

        phase = TURN.get_phase(state)
        if phase != GamePhases.PHASE_ACQUISITION:
            logger.warning(
                f"Preemption query but engine not in ACQ (phase={phase})"
            )
            return self._preemption_decline(original_corp_name, company_name)

        layout = session.layout
        for _ in range(200):
            if TURN.get_phase(state) != GamePhases.PHASE_ACQUISITION:
                break

            acq_corp = TURN.get_acq_active_corp(state)
            if acq_corp < 0:
                break  # Offer buffer exhausted

            acq_company = TURN.get_acq_target_company(state)

            if acq_corp == preempting_corp_id and acq_company == company_id:
                # Target offer found — let MCTS decide buy vs pass
                action_idx, info = self._search_and_pick(state)
                intent = engine_action_to_18xx(
                    action_idx, state, self.num_players,
                )
                session.apply_engine_action(action_idx)

                accepted = intent["type"] == "offer"
                logger.info(
                    f"Preemption: {preempting_corp_name} "
                    f"{'buys' if accepted else 'passes on'} "
                    f"{company_name} from FI"
                )
                return {
                    "actions": [{
                        "type": "respond",
                        "corporation": original_corp_name,
                        "company": company_name,
                        "accept": str(accepted).lower(),
                    }],
                    "search_info": {**info, "preemption": "resolved"},
                }

            # Not the target — pass to advance through the buffer
            session.apply_engine_action(layout.acq_pass)

        logger.warning(
            f"Preemption target not found: "
            f"{preempting_corp_name} + {company_name}"
        )
        return self._preemption_decline(original_corp_name, company_name)

    @staticmethod
    def _preemption_decline(corp_name: str, company_name: str) -> dict:
        """Return a decline response for a preemption that couldn't be resolved."""
        return {
            "actions": [{
                "type": "respond",
                "corporation": corp_name,
                "company": company_name,
                "accept": "false",
            }],
            "search_info": {"preemption": "target_not_found"},
        }


def create_app(server: AIServer) -> Flask:
    """Create Flask app with the AI move endpoint."""
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/ai-move", methods=["POST"])
    def ai_move():
        data = request.get_json()
        if not data or "game_data" not in data:
            return jsonify({"error": "Missing game_data"}), 400

        try:
            result = server.get_ai_move(
                data["game_data"],
                state_checksum=data.get("state_checksum"),
                preemption=data.get("preemption"),
            )
            return jsonify(result)
        except Exception as e:
            logger.exception("Error processing AI move")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI server for human-vs-AI Rolling Stock Stars play"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help='Path to checkpoint file, or "latest"',
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--search-batch-size", type=int, default=1)
    parser.add_argument(
        "--c-puct", type=float, default=None,
        help="Override c_puct (default from checkpoint)",
    )
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--terminal-blend", type=float, default=None,
        help="Rank vs margin weight for terminal rewards (default from checkpoint)",
    )

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--no-dirichlet-noise", dest="dirichlet_epsilon",
        action="store_const", const=0.0,
        help="Disable Dirichlet noise at root",
    )
    noise_group.add_argument(
        "--dirichlet-epsilon", type=float, default=None,
        help="Dirichlet noise epsilon (default from checkpoint)",
    )

    dyn_group = parser.add_mutually_exclusive_group()
    dyn_group.add_argument(
        "--dynamic-dirichlet", dest="dirichlet_dynamic",
        action="store_true", default=None,
        help="Use dynamic alpha = numerator / n_legal_actions",
    )
    dyn_group.add_argument(
        "--no-dynamic-dirichlet", dest="dirichlet_dynamic",
        action="store_false",
        help="Use static alpha",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    if args.checkpoint == "latest":
        cp_path = find_latest_checkpoint(Path(args.checkpoint_dir))
        if cp_path is None:
            print(f"No checkpoint found in {args.checkpoint_dir}")
            return
    else:
        cp_path = Path(args.checkpoint)

    server = AIServer(
        checkpoint_path=cp_path,
        device=device,
        num_simulations=args.simulations,
        search_batch_size=args.search_batch_size,
        c_puct=args.c_puct,
        dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_dynamic=args.dirichlet_dynamic,
        terminal_blend=args.terminal_blend,
    )

    app = create_app(server)
    logger.info(f"Starting AI server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
