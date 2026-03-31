"""Game session: synchronize 18xx game_data with a Cython GameState.

Maintains a persistent GameState that tracks the 18xx frontend's hotseat
game. On first call, runs the Ruby extractor to get deck order, then
replays all actions. On subsequent calls, replays only new actions.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from pathlib import Path

from core.data import (
    COMPANY_NAME_TO_ID,
    GamePhases,
)
from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from entities.turn import TURN

LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED

_ap = importlib.import_module("tests.18xx_games.action_parser")
ActionLayout = _ap.ActionLayout
AutoPassTracker = _ap.AutoPassTracker
filter_actions = _ap.filter_actions
flatten_auto_actions = _ap.flatten_auto_actions
map_invest_action = _ap.map_invest_action
map_bid_action = _ap.map_bid_action
map_ipo_action = _ap.map_ipo_action
map_dividend_action = _ap.map_dividend_action
map_issue_action = _ap.map_issue_action
entity_to_player_index = _ap.entity_to_player_index

PHASE_INVEST = GamePhases.PHASE_INVEST
PHASE_BID = GamePhases.PHASE_BID_IN_AUCTION
PHASE_WRAP_UP = GamePhases.PHASE_WRAP_UP
PHASE_ACQ = GamePhases.PHASE_ACQUISITION
PHASE_CLOSING = GamePhases.PHASE_CLOSING
PHASE_INCOME = GamePhases.PHASE_INCOME
PHASE_DIVIDENDS = GamePhases.PHASE_DIVIDENDS
PHASE_END_CARD = GamePhases.PHASE_END_CARD
PHASE_ISSUE = GamePhases.PHASE_ISSUE_SHARES
PHASE_IPO = GamePhases.PHASE_IPO
PHASE_PAR = GamePhases.PHASE_PAR
PHASE_GAME_OVER = GamePhases.PHASE_GAME_OVER

REPO_ROOT = Path(__file__).parent.parent
EXTRACTOR_PATH = REPO_ROOT / "tests" / "18xx_games" / "extract_states.rb"


class GameSession:
    """Maintains a Cython GameState synchronized with an 18xx hotseat game."""

    def __init__(self, num_players: int = 3):
        self.num_players = num_players
        self.layout = ActionLayout(num_players)
        self.state: GameState | None = None
        self.game_id: str | None = None
        self.replayed_action_count: int = 0
        self._player_id_to_index: dict[int, int] = {}

    def sync(self, game_data: dict) -> GameState:
        """Synchronize engine state with the frontend's game_data.

        On first call (or game_id change), initializes from scratch using
        the Ruby extractor for deck order. On subsequent calls, replays
        only new actions incrementally.

        Returns the current GameState after all actions have been applied.
        """
        gid = game_data.get("id", "")
        if self.state is None or gid != self.game_id:
            self._initialize(game_data)

        assert self.state is not None

        # Process new actions
        raw_actions = game_data.get("actions", [])
        actions = filter_actions(raw_actions)
        actions = flatten_auto_actions(actions)

        # Apply only actions we haven't seen yet
        idx = self.replayed_action_count
        while idx < len(actions):
            phase = TURN.get_phase(self.state)
            if phase == PHASE_GAME_OVER:
                break

            action = actions[idx]
            engine_action = self._map_action(action, phase)

            if engine_action is None:
                idx += 1
                continue

            action_list = engine_action if isinstance(engine_action, list) else [engine_action]
            for i, ea in enumerate(action_list):
                if i > 0 and TURN.get_phase(self.state) != PHASE_PAR:
                    break
                result = DRIVER.apply_action(self.state, ea)
                if result == STATUS_INVALID:
                    raise RuntimeError(
                        f"Invalid action {ea} at action stream index {idx}, "
                        f"phase={phase}, 18xx_type={action.get('type')}"
                    )

            idx += 1

        self.replayed_action_count = idx
        return self.state

    def get_active_player(self) -> int:
        """Get the engine's active player index (0-based)."""
        assert self.state is not None
        return self.state.get_active_player()

    def get_phase(self) -> int:
        """Get the engine's current phase."""
        assert self.state is not None
        return TURN.get_phase(self.state)

    def is_game_over(self) -> bool:
        """Check if the game is over."""
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

    def _initialize(self, game_data: dict) -> None:
        """Initialize engine state from game_data using Ruby extractor."""
        self.game_id = game_data.get("id", "")
        num_players = len(game_data.get("players", []))
        if num_players != self.num_players:
            raise ValueError(
                f"Player count mismatch: session={self.num_players}, "
                f"game={num_players}"
            )

        # Run Ruby extractor to get initial state (deck order, offering)
        initial = self._extract_initial_state(game_data)

        # Build player ID → engine index mapping
        self._player_id_to_index = {}
        for idx, pid in enumerate(initial["player_order"]):
            self._player_id_to_index[pid] = idx

        # Create and initialize engine state
        state = GameState(self.num_players)
        state.initialize_game(seed=42)  # seed irrelevant, we override
        self.state = state

        # Override deck and offering to match the 18xx game
        deck_order_names = initial["deck_order"]
        offering_names = initial["initial_offering"]
        self._override_deck_and_offering(deck_order_names, offering_names)

        self.replayed_action_count = 0

    def _extract_initial_state(self, game_data: dict) -> dict:
        """Run Ruby extractor subprocess to get initial deck/offering state.

        Writes game_data to a temp file, runs extract_states.rb, and
        returns the initial record (action_id=0).
        """
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

    def _override_deck_and_offering(
        self, deck_order_names: list[str], offering_names: list[str]
    ) -> None:
        """Override deck/offering to match the 18xx game's initial state."""
        assert self.state is not None
        state = self.state

        # Reset companies that init put into auction/revealed
        for cid in range(36):
            loc = COMPANIES[cid].get_location(state)
            if loc == LOC_AUCTION:
                state.set_company_for_auction(cid, False)
                COMPANIES[cid].exclude_from_game(state)
            elif loc == LOC_REVEALED:
                COMPANIES[cid].exclude_from_game(state)

        # Build full deck (offering on top, remaining below)
        # Ruby deck_order is top-to-bottom; our set_order is bottom-to-top
        remaining_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(deck_order_names)]
        offering_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(offering_names)]
        full_deck = remaining_ids + offering_ids
        DECK.set_order(state, full_deck)

        # Draw offering cards and move them to auction
        for _ in range(len(offering_names)):
            cid = DECK.draw(state)
            COMPANIES[cid].move_to_auction(state)

    def _map_action(self, action: dict, phase: int) -> int | list[int] | None:
        """Map a single 18xx action to engine action index(es).

        Returns None for actions that don't need engine application
        (auto-applied by engine, or belong to a phase already passed).
        """
        assert self.state is not None
        atype = action.get("type", "")
        entity_type = action.get("entity_type", "")

        if phase == PHASE_INVEST:
            if atype in ("bid", "buy_shares", "sell_shares", "pass"):
                if entity_type != "player":
                    return None
                return map_invest_action(self.state, action, self.layout)
            return None

        if phase == PHASE_BID:
            if atype in ("bid", "pass"):
                return map_bid_action(action, self.layout)
            return None

        if phase == PHASE_IPO:
            if atype in ("par", "pass"):
                if entity_type != "company":
                    return None
                return map_ipo_action(action, self.layout)
            return None

        if phase == PHASE_DIVIDENDS:
            if atype == "dividend":
                if entity_type != "corporation":
                    return None
                return map_dividend_action(action, self.layout)
            return None

        if phase == PHASE_ISSUE:
            if atype in ("sell_shares", "pass"):
                if entity_type != "corporation":
                    return None
                return map_issue_action(action, self.layout)
            return None

        # Automated phases — no player actions needed
        if phase in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_END_CARD,
                     PHASE_ACQ, PHASE_CLOSING):
            return None

        return None
