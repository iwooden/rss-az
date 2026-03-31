"""Game session: synchronize 18xx game_data with a Cython GameState.

Replays the full action history from scratch on each sync call.
Uses the shared action mapping from utils_18xx.action_parser.
The Ruby extractor runs once per game to get deck order; the Cython
replay of hundreds of actions takes only milliseconds.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.state import GameState
from entities.turn import TURN

from .action_parser import (
    ActionLayout,
    filter_actions,
    flatten_auto_actions,
    map_action,
    override_deck_and_offering,
    PHASE_GAME_OVER,
    PHASE_PAR,
)

EXTRACTOR_PATH = Path(__file__).parent / "extract_states.rb"


class GameSession:
    """Maintains a Cython GameState synchronized with an 18xx hotseat game.

    On each sync() call, replays the full action history from scratch.
    This avoids state drift between the engine and the frontend — the
    Cython engine replays hundreds of actions in <10ms.

    The Ruby extractor runs once per game_id to get the deck order and
    initial offering, then is cached for subsequent replays.
    """

    def __init__(self, num_players: int = 3):
        self.num_players = num_players
        self.layout = ActionLayout(num_players)
        self.state: GameState | None = None
        self.game_id: str | None = None
        self._player_id_to_index: dict[int, int] = {}
        # Cached from the Ruby extractor (per game_id)
        self._deck_order: list[str] = []
        self._offering: list[str] = []

    def sync(self, game_data: dict) -> GameState:
        """Replay the full game from scratch and return the current state.

        Always creates a fresh GameState and replays all actions. This
        ensures the engine state exactly matches the frontend regardless
        of what happened in previous sync/AI-move cycles.
        """
        gid = game_data.get("id", "")
        if gid != self.game_id:
            self._init_game_metadata(game_data)

        # Fresh state every time
        state = GameState(self.num_players)
        state.initialize_game(seed=42)
        self.state = state
        override_deck_and_offering(state, self._deck_order, self._offering)

        # Process actions using the same logic as the replay harness
        raw_actions = game_data.get("actions", [])
        actions = filter_actions(raw_actions)
        actions = flatten_auto_actions(actions)

        idx = 0
        while idx < len(actions):
            phase = TURN.get_phase(state)
            if phase == PHASE_GAME_OVER:
                break

            action = actions[idx]
            engine_action = map_action(state, action, phase, self.layout)

            if engine_action is None:
                idx += 1
                continue

            action_list = engine_action if isinstance(engine_action, list) else [engine_action]
            for i, ea in enumerate(action_list):
                if i > 0 and TURN.get_phase(state) != PHASE_PAR:
                    break
                result = DRIVER.apply_action(state, ea)
                if result == STATUS_INVALID:
                    raise RuntimeError(
                        f"Invalid action {ea} at action stream index {idx}, "
                        f"phase={phase}, 18xx_type={action.get('type')}"
                    )

            idx += 1

        return state

    def get_active_player(self) -> int:
        assert self.state is not None
        return self.state.get_active_player()

    def get_phase(self) -> int:
        assert self.state is not None
        return TURN.get_phase(self.state)

    def is_game_over(self) -> bool:
        assert self.state is not None
        return TURN.get_phase(self.state) == PHASE_GAME_OVER

    def apply_engine_action(self, action_idx: int) -> list[tuple[int, int]]:
        """Apply an engine action directly (for AI moves).

        Returns the history list of (phase, action) tuples including
        any auto-applied forced actions.
        """
        assert self.state is not None
        history: list[tuple[int, int]] = []
        result = DRIVER.apply_action(self.state, action_idx, history=history)
        if result == STATUS_INVALID:
            raise RuntimeError(f"Invalid engine action {action_idx}")
        return history

    def _init_game_metadata(self, game_data: dict) -> None:
        """Extract and cache deck order / offering via Ruby extractor."""
        self.game_id = game_data.get("id", "")
        num_players = len(game_data.get("players", []))
        if num_players != self.num_players:
            raise ValueError(
                f"Player count mismatch: session={self.num_players}, "
                f"game={num_players}"
            )

        initial = self._extract_initial_state(game_data)

        self._player_id_to_index = {}
        for idx, pid in enumerate(initial["player_order"]):
            self._player_id_to_index[pid] = idx

        self._deck_order = initial["deck_order"]
        self._offering = initial["initial_offering"]

    def _extract_initial_state(self, game_data: dict) -> dict:
        """Run Ruby extractor subprocess to get initial deck/offering state."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(game_data, f)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["ruby", str(EXTRACTOR_PATH), tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"Ruby extractor failed (rc={result.returncode}):\n{result.stderr}"
            )

        records = json.loads(result.stdout)
        initial = records[0]
        if initial.get("action_id") != 0:
            raise RuntimeError("Expected initial record with action_id=0")

        return initial

