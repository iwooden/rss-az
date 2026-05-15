"""Game session: synchronize 18xx game_data with a Cython GameState.

Replays the full action history from scratch on each sync call.
Uses the shared action mapping from utils_18xx.action_parser.
The Ruby extractor supplies deck order and committed action metadata; the first
sync for a game reuses one extractor result, and later syncs refresh metadata
for undo/redo handling. The Cython replay of hundreds of actions takes only
milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_PASS_PY as ACTION_PASS,
)
from core.driver import (
    DRIVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

from .action_parser import (
    ActionLayout,
    find_legal_action,
    filter_actions,
    flatten_auto_actions,
    map_action,
    map_bid_action,
    map_par_action,
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
from .share_ledger import parse_share_id, share_owner_before_action

ACQ_PHASES = (
    GamePhases.PHASE_ACQ_SELECT_CORP,
    GamePhases.PHASE_ACQ_SELECT_COMPANY,
    GamePhases.PHASE_ACQ_SELECT_PRICE,
    GamePhases.PHASE_ACQ_OFFER,
)
PHASE_CLO = GamePhases.PHASE_CLOSING

EXTRACTOR_PATH = Path(__file__).parent / "extract_states.rb"


@dataclass(frozen=True)
class StateMismatch:
    """One field-level mismatch between 18xx and the reconstructed state."""

    action_id: int
    phase: str
    field: str
    expected: object
    actual: object
    context: str = ""

    def __str__(self) -> str:
        msg = (
            f"Action {self.action_id} [{self.phase}] {self.field}: "
            f"expected={self.expected}, actual={self.actual}"
        )
        if self.context:
            msg += f" ({self.context})"
        return msg


def format_state_mismatches(
    mismatches: list[StateMismatch],
    *,
    limit: int = 25,
) -> str:
    """Format live replay mismatches for compact logs."""
    lines = [str(mismatch) for mismatch in mismatches[:limit]]
    if len(mismatches) > limit:
        lines.append(f"... and {len(mismatches) - limit} more")
    return "\n".join(lines)


class GameSession:
    """Maintains a Cython GameState synchronized with an 18xx hotseat game.

    On each sync() call, replays the full action history from scratch.
    This avoids state drift between the engine and the frontend — the
    Cython engine replays hundreds of actions in <10ms.

    The Ruby extractor runs on first sync for a game_id to get deck/order,
    initial offering, and committed action metadata. Later syncs reuse the
    deck metadata and refresh committed actions to handle undo/redo.
    """

    def __init__(self, num_players: int = 3, max_players: int | None = None):
        self.num_players = num_players
        self.max_players = max_players or num_players
        self.layout = ActionLayout(num_players)
        self.state: GameState | None = None
        self.game_id: str | None = None
        # Cached from the Ruby extractor (per game_id)
        self._deck_order: list[str] = []
        self._offering: list[str] = []
        # Refreshed on every sync via Ruby extractor
        self.committed_ids: set = set()
        self._player_ids: list = []
        self._last_extract_record: dict = {}
        self._extract_records_by_action_id: dict[int, dict] = {}

    def sync(self, game_data: dict) -> GameState:
        """Replay the full game from scratch and return the current state.

        Always creates a fresh GameState and replays all actions. This
        ensures the engine state exactly matches the frontend regardless
        of what happened in previous sync/AI-move cycles.
        """
        gid = game_data.get("id", "")
        initial_extract_record = None
        if gid != self.game_id:
            initial_extract_record = self._init_game_metadata(game_data)

        # Fresh state every time.
        state = initialize_replay_state(
            self.num_players,
            self._deck_order,
            self._offering,
            max_players=self.max_players,
            pause_before_acq_transition=True,
            pause_before_closing_transition=True,
        )
        self.state = state

        # Process actions — use Ruby extractor to resolve undo/redo.
        raw_actions = game_data.get("actions", [])
        if initial_extract_record is not None:
            self.committed_ids = set(
                initial_extract_record.get("committed_action_ids", [])
            )
        else:
            self.committed_ids = self._extract_committed_ids(game_data)
        actions = filter_actions(raw_actions, self.committed_ids)
        actions = flatten_auto_actions(actions)

        idx = 0
        while idx < len(actions):
            action = actions[idx]
            align_to_action(state, action, self.layout)

            phase = TURN.get_phase(state)
            if phase == PHASE_GAME_OVER:
                break

            if phase in ACQ_PHASES:
                # Only enter ACQ handling if the next action is actually
                # an ACQ action.  Otherwise drain through ACQ/CLO so the
                # engine catches up to the action stream's phase.
                next_type = action.get("type")
                if next_type in ("offer", "respond", "pass"):
                    idx = self._sync_acq_round(state, actions, idx)
                else:
                    drain_offer_phases(state, self.layout)
                continue
            if phase == PHASE_CLO:
                next_type = action.get("type")
                if next_type in ("sell_company", "close", "pass"):
                    idx = self._sync_clo_round(state, actions, idx)
                else:
                    drain_offer_phases(state, self.layout)
                continue

            engine_action = map_action(state, action, phase, self.layout)
            if engine_action is None:
                idx += 1
                continue

            cash_snapshot = self._share_price_cash_snapshot(state, action)
            share_owner_snapshot = self._share_owner_snapshot(game_data, state, action)
            result = apply_action_sequence(state, engine_action)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid action {engine_action} at action stream index {idx}, "
                    f"phase={phase}, 18xx_type={action.get('type')}"
                )
            self._apply_share_price_cash_adjustment(state, action, cash_snapshot)
            self._apply_share_owner_adjustment(state, action, share_owner_snapshot)
            result = self._apply_split_action_followup(state, action, phase)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid split follow-up for action stream index {idx}, "
                    f"phase={phase}, 18xx_type={action.get('type')}"
                )

            idx += 1

        # Drain trailing ACQ/CLO only when 18xx has already advanced past that
        # round.  During live sync the action stream is intentionally partial:
        # if 18xx is still waiting in Acquisition/Closing, passing through the
        # remaining engine choices would reconstruct a future state.
        if self._should_drain_trailing_offer_phases(game_data, state):
            drain_offer_phases(state, self.layout)

        return state

    def get_active_player(self) -> int:
        assert self.state is not None
        return TURN.get_active_player(self.state)

    def player_index_for_user_id(self, user_id) -> int:
        """Return the Cython player index for a 18xx user id."""
        target = str(user_id)
        for idx, player_id in enumerate(self._player_ids):
            if str(player_id) == target:
                return idx
        raise ValueError(f"User id {user_id!r} is not in game player order")

    def get_phase(self) -> int:
        assert self.state is not None
        return TURN.get_phase(self.state)

    def is_game_over(self) -> bool:
        assert self.state is not None
        return TURN.get_phase(self.state) == PHASE_GAME_OVER

    def pending_offer_for_user_id(self, user_id) -> dict | None:
        """Return the current 18xx ACQ offer this user must answer, if any."""
        target = str(user_id)
        for offer in self._last_extract_record.get("offers", []):
            if str(offer.get("responder_id")) == target:
                return offer
        return None

    def validate_against_18xx(
        self,
        game_data: dict,
        state: GameState,
        *,
        context: str = "live",
    ) -> list[StateMismatch]:
        """Compare reconstructed RSS state against the latest 18xx snapshot."""
        ref = self._last_extract_record
        if not ref:
            return []

        mismatches: list[StateMismatch] = []
        action_id = int(ref.get("action_id", -1))
        phase = TURN.get_phase(state)
        phase_name = self._phase_name(phase)
        round_name = str(game_data.get("round", ""))

        expected_round = ref.get("current_round") or round_name
        expected_stage = self._round_stage_key(str(expected_round))
        actual_stage = self._phase_stage_key(phase)
        if expected_stage >= 0 and actual_stage >= 0 and expected_stage != actual_stage:
            mismatches.append(
                StateMismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="round",
                    expected=round_name,
                    actual=phase_name,
                    context=context,
                )
            )

        if expected_stage == 2:
            self._compare_unordered_active_player(
                game_data,
                state,
                action_id,
                phase_name,
                context,
                mismatches,
            )
        elif expected_stage in (0, 5):
            ref_active_player = ref.get("active_player")
            if ref_active_player is not None:
                try:
                    expected_idx = self.player_index_for_user_id(ref_active_player)
                except ValueError:
                    expected_idx = None
                actual_idx = TURN.get_active_player(state)
                if expected_idx is not None and actual_idx != expected_idx:
                    mismatches.append(
                        StateMismatch(
                            action_id=action_id,
                            phase=phase_name,
                            field="active_player",
                            expected=expected_idx,
                            actual=actual_idx,
                            context=context,
                        )
                    )

        if expected_stage in (3, 4):
            ref_active_corp = ref.get("active_corp")
            if ref_active_corp is not None:
                expected_corp_id = CORP_NAME_TO_ID.get(ref_active_corp)
                actual_corp_id = TURN.get_active_corp(state)
                if expected_corp_id is not None and actual_corp_id != expected_corp_id:
                    mismatches.append(
                        StateMismatch(
                            action_id=action_id,
                            phase=phase_name,
                            field="active_corp",
                            expected=ref_active_corp,
                            actual=(
                                CORP_NAMES[actual_corp_id]
                                if 0 <= actual_corp_id < len(CORP_NAMES)
                                else actual_corp_id
                            ),
                            context=context,
                        )
                    )

        self._compare_players(state, ref, action_id, phase_name, context, mismatches)
        self._compare_corps(state, ref, action_id, phase_name, context, mismatches)
        self._compare_foreign_investor(
            state,
            ref,
            action_id,
            phase_name,
            context,
            mismatches,
        )
        self._compare_offering(state, ref, action_id, phase_name, context, mismatches)

        ref_coo = ref.get("cost_level")
        if ref_coo is not None and TURN.get_coo_level(state) != ref_coo:
            mismatches.append(
                StateMismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="cost_level",
                    expected=ref_coo,
                    actual=TURN.get_coo_level(state),
                    context=context,
                )
            )

        return mismatches

    def _compare_unordered_active_player(
        self,
        game_data: dict,
        state: GameState,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        expected_indices: set[int] = set()
        for actor in game_data.get("acting", []):
            try:
                expected_indices.add(self.player_index_for_user_id(actor))
            except ValueError:
                continue

        if not expected_indices:
            return

        actual_idx = TURN.get_active_player(state)
        if actual_idx in expected_indices:
            return

        mismatches.append(
            StateMismatch(
                action_id=action_id,
                phase=phase_name,
                field="active_player",
                expected=sorted(expected_indices),
                actual=actual_idx,
                context=context,
            )
        )

    def _compare_players(
        self,
        state: GameState,
        ref: dict,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        for ref_player in ref.get("players", []):
            player_name = ref_player["name"]
            try:
                pidx = self.player_index_for_user_id(ref_player["id"])
            except ValueError:
                continue

            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"player[{player_name}].cash",
                ref_player["cash"],
                PLAYERS[pidx].get_cash(state),
                context,
            )
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"player[{player_name}].value",
                ref_player["value"],
                PLAYERS[pidx].get_net_worth(state),
                context,
            )

            our_companies = sorted(
                COMPANY_NAMES[cid]
                for cid in range(len(COMPANY_NAMES))
                if COMPANIES[cid].is_owned_by_player(state, pidx)
            )
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"player[{player_name}].companies",
                sorted(ref_player.get("companies", [])),
                our_companies,
                context,
            )

            our_shares = {}
            for corp_id in range(len(CORP_NAMES)):
                shares = PLAYERS[pidx].get_shares(state, corp_id)
                if shares > 0:
                    our_shares[CORP_NAMES[corp_id]] = shares
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"player[{player_name}].shares",
                ref_player.get("shares", {}),
                our_shares,
                context,
            )

    def _compare_corps(
        self,
        state: GameState,
        ref: dict,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        for ref_corp in ref.get("corporations", []):
            corp_name = ref_corp["name"]
            corp_id = CORP_NAME_TO_ID.get(corp_name)
            if corp_id is None:
                continue

            ref_floated = bool(ref_corp["floated"])
            our_active = CORPS[corp_id].is_active(state)
            if ref_floated != our_active:
                mismatches.append(
                    StateMismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].active",
                        expected=ref_floated,
                        actual=our_active,
                        context=context,
                    )
                )
                if not ref_floated:
                    continue
            if not ref_floated:
                continue

            ref_price = ref_corp.get("price")
            if ref_price is not None:
                self._append_if_different(
                    mismatches,
                    action_id,
                    phase_name,
                    f"corp[{corp_name}].price",
                    ref_price,
                    CORPS[corp_id].get_share_price(state),
                    context,
                )
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"corp[{corp_name}].cash",
                ref_corp["cash"],
                self._corp_cash_for_18xx_compare(state, corp_id),
                context,
            )

            our_companies = sorted(
                COMPANY_NAMES[cid]
                for cid in range(len(COMPANY_NAMES))
                if COMPANIES[cid].is_owned_by_corp(state, corp_id)
                or COMPANIES[cid].is_in_corp_acquisition(state, corp_id)
            )
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"corp[{corp_name}].companies",
                sorted(ref_corp.get("companies", [])),
                our_companies,
                context,
            )
            self._append_if_different(
                mismatches,
                action_id,
                phase_name,
                f"corp[{corp_name}].shares_in_market",
                ref_corp.get("shares_in_market", 0),
                CORPS[corp_id].get_bank_shares(state),
                context,
            )
            self._compare_corp_president(
                state,
                ref,
                ref_corp,
                corp_id,
                corp_name,
                action_id,
                phase_name,
                context,
                mismatches,
            )

    def _compare_corp_president(
        self,
        state: GameState,
        ref: dict,
        ref_corp: dict,
        corp_id: int,
        corp_name: str,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        expected_idx = self._ref_corp_president_index(ref, ref_corp)
        if expected_idx is None:
            return

        actual_idx = CORPS[corp_id].get_president_id(state)
        if actual_idx == expected_idx:
            return

        mismatches.append(
            StateMismatch(
                action_id=action_id,
                phase=phase_name,
                field=f"corp[{corp_name}].president",
                expected=expected_idx,
                actual=actual_idx,
                context=context,
            )
        )

    def _ref_corp_president_index(self, ref: dict, ref_corp: dict) -> int | None:
        """Return expected president index, -1 for receivership, None if unknown."""
        if "president_id" in ref_corp:
            president_id = ref_corp.get("president_id")
            if president_id is None:
                return -1
            try:
                return self.player_index_for_user_id(president_id)
            except ValueError:
                return None

        president_name = ref_corp.get("president")
        if president_name is None:
            return None
        if not president_name:
            return -1

        for ref_player in ref.get("players", []):
            if ref_player.get("name") != president_name:
                continue
            try:
                return self.player_index_for_user_id(ref_player.get("id"))
            except ValueError:
                return None
        return None

    def _compare_foreign_investor(
        self,
        state: GameState,
        ref: dict,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        ref_fi = ref.get("foreign_investor", {})
        self._append_if_different(
            mismatches,
            action_id,
            phase_name,
            "fi.cash",
            ref_fi.get("cash", 0),
            FI.get_cash(state),
            context,
        )

        our_companies = sorted(
            COMPANY_NAMES[cid]
            for cid in range(len(COMPANY_NAMES))
            if COMPANIES[cid].is_owned_by_fi(state)
        )
        self._append_if_different(
            mismatches,
            action_id,
            phase_name,
            "fi.companies",
            sorted(ref_fi.get("companies", [])),
            our_companies,
            context,
        )

    def _compare_offering(
        self,
        state: GameState,
        ref: dict,
        action_id: int,
        phase_name: str,
        context: str,
        mismatches: list[StateMismatch],
    ) -> None:
        offering_locations = (
            int(CompanyLocation.LOC_AUCTION),
            int(CompanyLocation.LOC_REVEALED),
        )
        our_offering = sorted(
            COMPANY_NAMES[cid]
            for cid in range(len(COMPANY_NAMES))
            if COMPANIES[cid].get_location(state) in offering_locations
        )
        self._append_if_different(
            mismatches,
            action_id,
            phase_name,
            "offering",
            sorted(ref.get("offering", [])),
            our_offering,
            context,
        )

    @staticmethod
    def _append_if_different(
        mismatches: list[StateMismatch],
        action_id: int,
        phase_name: str,
        field: str,
        expected,
        actual,
        context: str,
    ) -> None:
        if expected == actual:
            return
        mismatches.append(
            StateMismatch(
                action_id=action_id,
                phase=phase_name,
                field=field,
                expected=expected,
                actual=actual,
                context=context,
            )
        )

    @staticmethod
    def _corp_cash_for_18xx_compare(state: GameState, corp_id: int) -> int:
        cash = CORPS[corp_id].get_cash(state)
        if TURN.get_phase(state) in ACQ_PHASES:
            cash += CORPS[corp_id].get_acquisition_proceeds(state)
        return cash

    @staticmethod
    def _phase_name(phase: int) -> str:
        try:
            return GamePhases(phase).name
        except ValueError:
            return f"UNKNOWN({phase})"

    @staticmethod
    def _phase_stage_key(phase: int) -> int:
        if phase in (GamePhases.PHASE_INVEST, GamePhases.PHASE_BID):
            return 0
        if phase in ACQ_PHASES:
            return 1
        if phase == PHASE_CLO:
            return 2
        if phase == GamePhases.PHASE_DIVIDENDS:
            return 3
        if phase == GamePhases.PHASE_ISSUE_SHARES:
            return 4
        if phase in (GamePhases.PHASE_IPO, GamePhases.PHASE_PAR):
            return 5
        return -1

    @staticmethod
    def _round_stage_key(round_name: str) -> int:
        lower = round_name.lower()
        upper = round_name.upper()
        if upper == "INV" or "investment" in lower:
            return 0
        if upper == "ACQ" or "acquisition" in lower:
            return 1
        if upper == "CLO" or "closing" in lower or "close" in lower:
            return 2
        if upper == "DIV" or "dividend" in lower:
            return 3
        if upper == "ISS" or "issue" in lower:
            return 4
        if upper == "IPO" or "ipo" in lower:
            return 5
        return -1

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

    def _apply_split_action_followup(
        self,
        state: GameState,
        action: dict,
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
            return apply_action_sequence(state, map_bid_action(state, action, self.layout))

        if (
            initial_phase == GamePhases.PHASE_IPO
            and atype == "par"
            and phase == GamePhases.PHASE_PAR
        ):
            return apply_action_sequence(state, map_par_action(state, action, self.layout))

        return STATUS_OK

    def _bid_followup_already_applied(
        self,
        state: GameState,
        action: dict,
    ) -> bool:
        """Return whether a forced opening BID was auto-applied by the driver."""
        try:
            company_id = COMPANY_NAME_TO_ID[action["company"]]
            price = int(action["price"])
            bidder_idx = self.player_index_for_user_id(action.get("entity"))
        except (KeyError, TypeError, ValueError):
            return False

        return (
            TURN.get_active_company(state) == company_id
            and TURN.get_auction_price(state) == price
            and TURN.get_auction_high_bidder(state) == bidder_idx
        )

    def _share_price_cash_snapshot(
        self,
        state: GameState,
        action: dict,
    ) -> tuple[str, int, int] | None:
        """Capture cash for RSS share actions with explicit 18xx prices."""
        atype = action.get("type")
        if atype not in {"buy_shares", "sell_shares"} or "share_price" not in action:
            return None

        entity_type = action.get("entity_type")
        entity = action.get("entity")
        if entity_type == "player":
            try:
                player_idx = self.player_index_for_user_id(entity)
            except ValueError:
                return None
            return ("player", player_idx, PLAYERS[player_idx].get_cash(state))

        if atype == "sell_shares" and entity_type == "corporation":
            corp_id = CORP_NAME_TO_ID.get(entity)
            if corp_id is None:
                return None
            return ("corporation", corp_id, CORPS[corp_id].get_cash(state))

        return None

    def _apply_share_price_cash_adjustment(
        self,
        state: GameState,
        action: dict,
        snapshot: tuple[str, int, int] | None,
    ) -> None:
        """Adjust replay cash to 18xx's explicit transaction share price."""
        if snapshot is None:
            return

        try:
            share_price = int(str(action["share_price"]).split(",", 1)[0])
        except (KeyError, TypeError, ValueError):
            return

        holder_type, holder_id, before_cash = snapshot
        atype = action.get("type")
        desired_delta = -share_price if atype == "buy_shares" else share_price

        if holder_type == "player":
            holder = PLAYERS[holder_id]
        else:
            holder = CORPS[holder_id]

        actual_delta = holder.get_cash(state) - before_cash
        adjustment = desired_delta - actual_delta
        if adjustment:
            holder.add_cash(state, adjustment)

    def _share_owner_snapshot(
        self,
        game_data: dict,
        state: GameState,
        action: dict,
    ) -> tuple[int, int, int, int] | None:
        """Capture when 18xx share ID ownership differs from action entity."""
        if (
            action.get("type") != "sell_shares"
            or action.get("entity_type") != "player"
            or not action.get("shares")
            or action.get("id") is None
        ):
            return None

        share_ref = action["shares"][0]
        corp_name, _ = parse_share_id(share_ref)
        if corp_name is None:
            return None

        owner_user_id = share_owner_before_action(
            game_data,
            self.committed_ids,
            share_ref,
            int(action["id"]),
        )
        if owner_user_id is None or str(owner_user_id) == str(action.get("entity")):
            return None

        try:
            action_player_idx = self.player_index_for_user_id(action.get("entity"))
            owner_player_idx = self.player_index_for_user_id(owner_user_id)
            share_price = int(str(action["share_price"]).split(",", 1)[0])
        except (KeyError, TypeError, ValueError):
            return None

        return (
            CORP_NAME_TO_ID[corp_name],
            action_player_idx,
            owner_player_idx,
            share_price,
        )

    def _apply_share_owner_adjustment(
        self,
        state: GameState,
        action: dict,
        snapshot: tuple[int, int, int, int] | None,
    ) -> None:
        """Patch replay for 18xx presidency-share ID swaps.

        18xx share IDs are non-fungible. Presidency changes swap the ``_0``
        share with a non-president share, and the server resolves a sell by
        share ID owner even if the action entity is stale. RSS only tracks share
        counts, so replay applies the action for turn advancement then patches
        the economic owner to match the 18xx share ID ledger.
        """
        if snapshot is None:
            return

        corp_id, action_player_idx, owner_player_idx, share_price = snapshot
        action_player = PLAYERS[action_player_idx]
        owner_player = PLAYERS[owner_player_idx]

        action_player.set_shares(
            state,
            corp_id,
            action_player.get_shares(state, corp_id) + 1,
        )
        action_player.add_cash(state, -share_price)
        owner_player.set_shares(
            state,
            corp_id,
            owner_player.get_shares(state, corp_id) - 1,
        )
        owner_player.add_cash(state, share_price)

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

        while idx < len(actions) and TURN.get_phase(state) in ACQ_PHASES:
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
                self._retarget_acq_active_player_to_action_entity(state, action)
                if self._offer_resolves_immediately(
                    state,
                    action,
                    self._has_future_response_to_offer(actions, idx + 1, action),
                ):
                    self._resolve_acq_offer(
                        state,
                        action,
                        accepted=True,
                        deferred_transfers=deferred_transfers,
                    )
                    pending_offer = None
                else:
                    pending_offer = (
                        action
                        if self._begin_acq_offer(state, action, deferred_transfers)
                        else None
                    )
                idx += 1
                continue

            if atype == "respond":
                if pending_offer is not None:
                    if self._is_response_to_offer(action, pending_offer):
                        self._apply_acq_offer_response(state, action)
                        if TURN.get_phase(state) != GamePhases.PHASE_ACQ_OFFER:
                            pending_offer = None
                elif self._is_current_acq_offer_response(state, action):
                    self._apply_acq_offer_response(state, action)
                idx += 1
                continue

            if atype == "pass":
                if pending_offer is not None and TURN.get_phase(state) == GamePhases.PHASE_ACQ_OFFER:
                    pending_after_pass = self._extractor_offer_pending_after_action(
                        action,
                        pending_offer,
                    )
                    if pending_after_pass is True:
                        idx += 1
                        continue
                    if pending_after_pass is False:
                        self._cancel_pending_acq_offer(state)
                        pending_offer = None
                    else:
                        idx += 1
                        continue
                self._apply_acq_pass(state, action)
                idx += 1
                continue

            break

        phase = TURN.get_phase(state)
        if phase not in ACQ_PHASES:
            while idx < len(actions) and (
                self._is_acq_auto_pass(actions[idx])
                or (
                    phase != PHASE_CLO
                    and self._is_acq_redundant_pass_parent(actions[idx])
                )
            ):
                idx += 1

        if (
            pending_offer is not None
            and TURN.get_phase(state) in ACQ_PHASES
            and TURN.get_phase(state) != GamePhases.PHASE_ACQ_OFFER
        ):
            self._resolve_acq_offer(
                state,
                pending_offer,
                accepted=True,
                deferred_transfers=deferred_transfers,
            )

        if TURN.get_phase(state) in ACQ_PHASES and DRIVER.is_non_player_phase(state):
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

    def _apply_acq_pass(self, state: GameState, action: dict | None = None) -> None:
        """Apply a recorded 18xx ACQ pass when the engine has a pass choice."""
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

        result = DRIVER.apply_action(state, action_idx)
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid ACQ pass replay action "
                f"phase={TURN.get_phase(state)}"
            )

    def _begin_acq_offer(
        self,
        state: GameState,
        offer: dict,
        deferred_transfers: list[tuple[int, int, int]],
    ) -> bool:
        """Replay an offer up to ACQ_OFFER without answering it."""
        buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
        company_id = COMPANY_NAME_TO_ID[offer["company"]]
        price = int(offer["price"])

        if not is_representable_acquisition_offer(state, buyer_corp_id, company_id):
            deferred_transfers.append((buyer_corp_id, company_id, price))
            return False

        state_snapshot = state._array.copy()
        deferred_len = len(deferred_transfers)

        def rollback_speculative_offer() -> None:
            state._array[:] = state_snapshot
            del deferred_transfers[deferred_len:]

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

            result = apply_action_sequence(state, action_idx)
            if result == STATUS_INVALID:
                rollback_speculative_offer()
                raise RuntimeError(
                    "Invalid ACQ offer replay action while opening offer "
                    f"phase={phase} corporation={offer.get('corporation')} "
                    f"company={offer.get('company')}"
                )

        rollback_speculative_offer()
        raise RuntimeError("Exceeded ACQ offer opening iteration limit")

    def _retarget_acq_active_player_to_action_entity(
        self,
        state: GameState,
        action: dict,
    ) -> None:
        """Align unordered 18xx ACQ offers with RSS's player-ordered surface."""
        if TURN.get_phase(state) != GamePhases.PHASE_ACQ_SELECT_CORP:
            return
        player_idx = self._player_index_for_action_entity(action)
        if player_idx is None:
            return
        TURN.set_active_player(state, player_idx)
        TURN.clear_acquisition_context(state)

    def _apply_acq_offer_response(self, state: GameState, action: dict) -> None:
        """Apply one recorded 18xx ACQ_OFFER response."""
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

        result = DRIVER.apply_action(state, action_idx)
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid ACQ offer response replay action "
                f"phase={TURN.get_phase(state)}"
            )

    @staticmethod
    def _cancel_pending_acq_offer(state: GameState) -> None:
        """Drop a live 18xx offer that cleared without an RSS transfer."""
        TURN.clear_acquisition_context(state)
        TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))

    def _player_index_for_action_entity(self, action: dict) -> int | None:
        """Return engine player index for an 18xx player-entity action."""
        if action.get("entity_type") != "player":
            return None
        try:
            return self.player_index_for_user_id(action.get("entity"))
        except ValueError:
            return None

    @staticmethod
    def _is_acq_auto_pass(action: dict) -> bool:
        """Return whether ``action`` is an auto pass emitted by an ACQ parent."""
        return (
            action.get("type") == "pass"
            and action.get("entity_type") == "player"
            and action.get("_auto_parent_type") in {"offer", "respond", "pass"}
        )

    @staticmethod
    def _is_acq_redundant_pass_parent(action: dict) -> bool:
        """Return whether ``action`` is an explicit pass already auto-chained."""
        return (
            action.get("type") == "pass"
            and action.get("entity_type") == "player"
            and any(
                auto.get("type") == "pass"
                and auto.get("entity_type") == "player"
                for auto in action.get("auto_actions", [])
            )
        )

    def _offer_resolves_immediately(
        self,
        state: GameState,
        offer: dict,
        has_future_response: bool = False,
    ) -> bool:
        """Return whether 18xx processes this ACQ offer without a response."""
        buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
        company_id = COMPANY_NAME_TO_ID[offer["company"]]
        buyer_president = CORPS[buyer_corp_id].get_president_id(state)
        owner_loc = COMPANIES[company_id].get_location(state)
        owner_id = COMPANIES[company_id].get_owner_id(state)

        if owner_loc == int(CompanyLocation.LOC_FI):
            return not (
                has_future_response
                or self._extractor_has_pending_offer(offer)
            )

        if owner_loc == int(CompanyLocation.LOC_PLAYER):
            return owner_id == buyer_president

        if owner_loc == int(CompanyLocation.LOC_CORP):
            return CORPS[owner_id].get_president_id(state) == buyer_president

        return False

    @staticmethod
    def _has_future_response_to_offer(
        actions: list[dict],
        start_idx: int,
        offer: dict,
    ) -> bool:
        for probe in range(start_idx, len(actions)):
            action = actions[probe]
            if action.get("type") == "offer":
                return False
            if GameSession._is_response_to_offer(action, offer):
                return True
        return False

    def _extractor_has_pending_offer(self, offer: dict) -> bool:
        if self._extractor_offer_pending_after_action(offer, offer) is True:
            return True
        return self._record_has_pending_offer(self._last_extract_record, offer)

    def _extractor_offer_pending_after_action(
        self,
        action: dict,
        offer: dict,
    ) -> bool | None:
        try:
            action_id = int(action["id"])
        except (KeyError, TypeError, ValueError):
            return None
        record = self._extract_records_by_action_id.get(action_id)
        if record is None:
            return None
        return self._record_has_pending_offer(
            record,
            offer,
        )

    @staticmethod
    def _record_has_pending_offer(record: dict, offer: dict) -> bool:
        for pending in record.get("offers", []):
            if (
                pending.get("corporation") == offer.get("corporation")
                and pending.get("company") == offer.get("company")
            ):
                return True
        return False

    @staticmethod
    def _is_response_to_offer(action: dict | None, offer: dict) -> bool:
        if action is None or action.get("type") != "respond":
            return False
        return (
            action.get("corporation") == offer.get("corporation")
            and action.get("company") == offer.get("company")
        )

    @staticmethod
    def _is_current_acq_offer_response(state: GameState, action: dict) -> bool:
        """Return whether ``action`` answers the engine's open ACQ_OFFER."""
        if (
            action.get("type") != "respond"
            or TURN.get_phase(state) != GamePhases.PHASE_ACQ_OFFER
        ):
            return False

        original_corp_id = TURN.get_acq_offer_corp(state)
        company_id = TURN.get_active_company(state)
        if original_corp_id < 0 or company_id < 0:
            return False

        return (
            action.get("corporation") == CORP_NAMES[original_corp_id]
            and action.get("company") == COMPANY_NAMES[company_id]
        )

    def _should_drain_trailing_offer_phases(
        self,
        game_data: dict,
        state: GameState,
    ) -> bool:
        """Return whether sync should pass through leftover ACQ/CLO choices."""
        phase = TURN.get_phase(state)
        if phase not in ACQ_PHASES and phase != PHASE_CLO:
            return False

        round_name = str(game_data.get("round", "")).lower()
        if phase in ACQ_PHASES and "acquisition" in round_name:
            return False
        if phase == PHASE_CLO and ("closing" in round_name or "close" in round_name):
            return False

        return True

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
        while idx < len(actions) and TURN.get_phase(state) == PHASE_CLO:
            action = actions[idx]
            atype = action.get("type")

            if atype == "pass":
                self._apply_clo_pass(state, action)
                idx += 1
                continue

            if atype in ("sell_company", "close"):
                if not self._apply_clo_close(state, action):
                    company_id = COMPANY_NAME_TO_ID[action["company"]]
                    if not apply_external_close(state, company_id):
                        raise RuntimeError(
                            "Failed to replay CLO close for "
                            f"{COMPANY_NAMES[company_id]}"
                        )
                idx += 1
                continue

            break

        if TURN.get_phase(state) != PHASE_CLO:
            while idx < len(actions) and self._is_clo_auto_pass(actions[idx]):
                idx += 1

        if TURN.get_phase(state) == PHASE_CLO and is_closing_transition_pending(state):
            DRIVER.advance_phase(state)

        return idx

    def _apply_clo_pass(self, state: GameState, action: dict | None = None) -> None:
        """Apply a recorded 18xx Closing pass for its player entity."""
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

        result = DRIVER.apply_action(state, action_idx)
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid CLO pass replay action "
                f"phase={TURN.get_phase(state)}"
            )

    def _apply_clo_close(self, state: GameState, action: dict) -> bool:
        """Apply a recorded 18xx Closing close for its player entity."""
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

        result = DRIVER.apply_action(state, action_idx)
        if result == STATUS_INVALID:
            raise RuntimeError(
                "Invalid CLO close replay action "
                f"company={COMPANY_NAMES[company_id]}"
            )
        return True

    @staticmethod
    def _is_clo_auto_pass(action: dict) -> bool:
        """Return whether ``action`` is an auto pass emitted by a CLO parent."""
        return (
            action.get("type") == "pass"
            and action.get("entity_type") == "player"
            and action.get("_auto_parent_type") in {"sell_company", "close", "pass"}
        )

    def _init_game_metadata(self, game_data: dict) -> dict:
        """Extract and cache deck order / offering via Ruby extractor."""
        self.game_id = game_data.get("id", "")
        num_players = len(game_data.get("players", []))
        if num_players != self.num_players:
            raise ValueError(
                f"Player count mismatch: session={self.num_players}, "
                f"game={num_players}"
            )

        initial = self._run_extractor(game_data)

        self._deck_order = initial["deck_order"]
        self._offering = initial["initial_offering"]
        self._player_ids = [player["id"] for player in initial["players"]]
        return initial

    def _run_extractor(self, game_data: dict) -> dict:
        """Run Ruby extractor subprocess; return the initial record."""
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
        self._last_extract_record = records[-1] if records else {}
        self._extract_records_by_action_id = {
            int(record["action_id"]): record
            for record in records
            if record.get("action_id") is not None
        }
        initial = records[0]
        if initial.get("action_id") != 0:
            raise RuntimeError("Expected initial record with action_id=0")

        return initial

    def _extract_committed_ids(self, game_data: dict) -> set:
        """Get committed action IDs from the Ruby extractor.

        Runs the extractor on subsequent syncs to correctly handle undo/redo
        in the live action stream.  The ~200ms cost is negligible next to
        MCTS search time.
        """
        initial = self._run_extractor(game_data)
        return set(initial.get("committed_action_ids", []))
