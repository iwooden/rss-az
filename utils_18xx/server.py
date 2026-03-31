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

from core.data import GamePhases
from core.state import get_layout
from entities.turn import TURN
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, run_search
from nn import create_model
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import MCTSConfig, TrainingConfig

from .action_mapper import engine_action_to_18xx
from .game_session import GameSession

logger = logging.getLogger(__name__)


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

    def get_ai_move(self, game_data: dict) -> dict:
        """Process a game state and return the AI's move(s).

        Returns a dict with:
            actions: list of intent dicts to apply
            search_info: diagnostic info about the search
        """
        self._maybe_reload_checkpoint()

        session = self.get_session(game_data)
        state = session.sync(game_data)

        if session.is_game_over():
            return {"actions": [], "error": "Game is over"}

        # Determine which players are AI
        settings = game_data.get("settings") or {}
        human_idx = settings.get("human_player_index", 0)

        # Check if the 18xx frontend is in ACQ/CLOSING but our engine
        # has auto-advanced past it. Our engine doesn't support cross-
        # president acquisitions and auto-processes these phases, but
        # the 18xx frontend blocks waiting for player pass actions.
        frontend_round = game_data.get("round", "")
        acting = game_data.get("acting") or []
        ai_acting_ids = [pid for pid in acting if pid != human_idx]

        if frontend_round in ("Acquisition", "Closing") and ai_acting_ids:
            phase = TURN.get_phase(state)
            if phase not in (
                GamePhases.PHASE_ACQUISITION,
                GamePhases.PHASE_CLOSING,
            ):
                # Engine already advanced past ACQ/CLO — return passes
                # for all AI players so the frontend can catch up.
                logger.info(
                    f"Frontend in {frontend_round} but engine at phase "
                    f"{phase} — returning passes for AI players {ai_acting_ids}"
                )
                pass_actions = [{"type": "pass"} for _ in ai_acting_ids]
                return {"actions": pass_actions, "search_info": {"auto_pass": True}}

        active = state.get_active_player()

        if active == human_idx:
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
            # Run through all AI offers in this phase
            actions, search_info = self._run_phase_sequence(session, state, human_idx)
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

    def _run_phase_sequence(
        self, session, state, human_idx: int,
    ) -> tuple[list[dict], dict]:
        """Run through ACQ/CLOSING phase, collecting all AI actions."""
        actions: list[dict] = []
        total_info: dict = {"steps": []}
        start_phase = TURN.get_phase(state)

        while (
            TURN.get_phase(state) == start_phase
            and not session.is_game_over()
            and state.get_active_player() != human_idx
        ):
            action_idx, info = self._search_and_pick(state)
            intent = engine_action_to_18xx(action_idx, state, self.num_players)
            session.apply_engine_action(action_idx)
            actions.append(intent)
            total_info["steps"].append(info)

        return actions, total_info


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
            result = server.get_ai_move(data["game_data"])
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
