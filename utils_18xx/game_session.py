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

from core.data import (
    COMPANY_NAME_TO_ID,
    COMPANY_NAMES,
    CORP_NAME_TO_ID,
    CORP_NAMES,
    GamePhases,
)
from core.driver import (
    DRIVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.state import GameState
from entities.turn import TURN

from .action_parser import (
    ActionLayout,
    filter_actions,
    flatten_auto_actions,
    map_action,
    PHASE_GAME_OVER,
)
from .replay_state import (
    align_to_action,
    apply_action_sequence,
    apply_external_acquisition_transfer,
    apply_external_close,
    drain_offer_phases,
    initialize_replay_state,
    is_closing_transition_pending,
    is_representable_acquisition_offer,
    replay_acquisition_offer,
)

PHASE_ACQ = GamePhases.PHASE_ACQUISITION
PHASE_CLO = GamePhases.PHASE_CLOSING

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

        # Fresh state every time.
        state = initialize_replay_state(
            self.num_players,
            self._deck_order,
            self._offering,
            pause_before_acq_transition=True,
            pause_before_closing_transition=True,
        )
        self.state = state

        # Process actions using the same logic as the replay harness
        raw_actions = game_data.get("actions", [])
        actions = filter_actions(raw_actions)
        actions = flatten_auto_actions(actions)

        idx = 0
        while idx < len(actions):
            action = actions[idx]
            align_to_action(state, action, self.layout)

            phase = TURN.get_phase(state)
            if phase == PHASE_GAME_OVER:
                break

            if phase == PHASE_ACQ:
                idx = self._sync_acq_round(state, actions, idx)
                continue
            if phase == PHASE_CLO:
                idx = self._sync_clo_round(state, actions, idx)
                continue

            engine_action = map_action(state, action, phase, self.layout)
            if engine_action is None:
                idx += 1
                continue

            result = apply_action_sequence(state, engine_action)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid action {engine_action} at action stream index {idx}, "
                    f"phase={phase}, 18xx_type={action.get('type')}"
                )

            idx += 1

        # Drain any remaining ACQ/CLO offers that weren't in the stream
        # (our engine may have more offers than the 18xx frontend saw).
        drain_offer_phases(state, self.layout)

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

    def apply_engine_action(self, action_idx: int) -> list[tuple[object, int]]:
        """Apply an engine action directly (for AI moves).

        Returns the driver history list of ``(state_copy, action_or_sentinel)``
        entries, including any auto-applied forced actions.

        STATUS_PAUSED is a valid outcome when the action auto-chains into
        an ACQ/CLO transition with pause flags set.  The caller (server)
        discards this state on the next sync, so the pause is benign.
        """
        assert self.state is not None
        history: list[tuple[object, int]] = []
        result = DRIVER.apply_action(self.state, action_idx, history=history)
        if result == STATUS_INVALID:
            raise RuntimeError(f"Invalid engine action {action_idx}")
        return history

    # -----------------------------------------------------------------
    # ACQ / CLO replay helpers
    # -----------------------------------------------------------------
    def _sync_acq_round(
        self,
        state: GameState,
        actions: list[dict],
        idx: int,
    ) -> int:
        """Replay one ACQ round from raw offer/respond/pass actions."""
        pending_offer: dict | None = None
        deferred_transfers: list[tuple[int, int, int]] = []

        while idx < len(actions) and TURN.get_phase(state) == PHASE_ACQ:
            action = actions[idx]
            atype = action.get("type")

            if atype == "offer":
                if pending_offer is not None:
                    self._resolve_acq_offer(
                        state,
                        pending_offer,
                        accepted=True,
                        deferred_transfers=deferred_transfers,
                    )
                pending_offer = action
                idx += 1
                continue

            if atype == "respond":
                if pending_offer is not None:
                    accepted = str(action.get("accept", "")).lower() == "true"
                    self._resolve_acq_offer(
                        state,
                        pending_offer,
                        accepted=accepted,
                        deferred_transfers=deferred_transfers,
                    )
                    pending_offer = None
                idx += 1
                continue

            if atype == "pass":
                idx += 1
                continue

            break

        if pending_offer is not None and TURN.get_phase(state) == PHASE_ACQ:
            self._resolve_acq_offer(
                state,
                pending_offer,
                accepted=True,
                deferred_transfers=deferred_transfers,
            )

        if TURN.get_phase(state) == PHASE_ACQ and DRIVER.is_non_player_phase(state):
            for buyer_corp_id, company_id, price in deferred_transfers:
                if not apply_external_acquisition_transfer(
                    state,
                    buyer_corp_id,
                    company_id,
                    price,
                ):
                    raise RuntimeError(
                        "Failed to patch deferred ACQ transfer for "
                        f"{CORP_NAMES[buyer_corp_id]} -> {COMPANY_NAMES[company_id]}"
                    )
            DRIVER.advance_phase(state)

        return idx

    def _resolve_acq_offer(
        self,
        state: GameState,
        offer: dict,
        *,
        accepted: bool,
        deferred_transfers: list[tuple[int, int, int]],
    ) -> None:
        """Resolve a single ACQ offer from the live 18xx action stream."""
        buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
        company_id = COMPANY_NAME_TO_ID[offer["company"]]
        price = int(offer["price"])

        if not accepted:
            if is_representable_acquisition_offer(state, buyer_corp_id, company_id):
                replay_acquisition_offer(
                    state,
                    self.layout,
                    buyer_corp_id,
                    company_id,
                    price,
                    accept=False,
                )
            return

        if is_representable_acquisition_offer(state, buyer_corp_id, company_id):
            matched = replay_acquisition_offer(
                state,
                self.layout,
                buyer_corp_id,
                company_id,
                price,
                accept=True,
            )
            if not matched:
                # Engine buffer exhausted (paused) — fall through to defer.
                deferred_transfers.append((buyer_corp_id, company_id, price))
            return

        deferred_transfers.append((buyer_corp_id, company_id, price))

    def _sync_clo_round(
        self,
        state: GameState,
        actions: list[dict],
        idx: int,
    ) -> int:
        """Replay one CLO round from raw close/pass actions."""
        closed_companies: set[int] = set()

        while idx < len(actions) and TURN.get_phase(state) == PHASE_CLO:
            action = actions[idx]
            atype = action.get("type")

            if atype == "pass":
                idx += 1
                continue

            if atype in ("sell_company", "close"):
                closed_companies.add(COMPANY_NAME_TO_ID[action["company"]])
                idx += 1
                continue

            break

        max_iterations = 200
        for _ in range(max_iterations):
            if TURN.get_phase(state) != PHASE_CLO:
                break

            closing_company = TURN.get_closing_company(state)
            if closing_company < 0:
                if DRIVER.is_non_player_phase(state) and not is_closing_transition_pending(state):
                    DRIVER.advance_phase(state)
                    continue
                break

            action_idx = (
                self.layout.close_action
                if closing_company in closed_companies
                else self.layout.close_pass
            )
            result = DRIVER.apply_action(state, action_idx)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid CLO replay action for company={COMPANY_NAMES[closing_company]}"
                )
            if closing_company in closed_companies and result != STATUS_INVALID:
                closed_companies.discard(closing_company)
            if result == STATUS_PAUSED:
                break

        if TURN.get_phase(state) == PHASE_CLO and is_closing_transition_pending(state):
            for company_id in sorted(closed_companies):
                if not apply_external_close(state, company_id):
                    raise RuntimeError(
                        "Failed to patch deferred CLO close for "
                        f"{COMPANY_NAMES[company_id]}"
                    )
            closed_companies.clear()
            DRIVER.advance_phase(state)

        if closed_companies:
            raise RuntimeError(
                "CLO close actions never resolved in engine buffer: "
                + ", ".join(COMPANY_NAMES[company_id] for company_id in sorted(closed_companies))
            )

        return idx

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
