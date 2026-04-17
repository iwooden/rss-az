"""Replay harness for validating the current engine against 18xx.games extracts."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.data import COMPANY_NAMES, COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, CORP_NAMES
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.deck import DECK
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN
from utils_18xx.action_parser import (
    ActionLayout,
    PHASE_ACQ,
    PHASE_ACQ_OFFER,
    PHASE_BID,
    PHASE_CLOSING,
    PHASE_DIVIDENDS,
    PHASE_END_CARD,
    PHASE_GAME_OVER,
    PHASE_INCOME,
    PHASE_INVEST,
    PHASE_IPO,
    PHASE_ISSUE,
    PHASE_WRAP_UP,
    filter_actions,
    flatten_auto_actions,
    map_action,
)
from utils_18xx.replay_state import (
    apply_external_acquisition_transfer,
    drain_offer_phases,
    initialize_replay_state,
    settle_to_player_choice,
)

LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED


@dataclass
class Mismatch:
    action_id: int
    phase: str
    field: str
    expected: object
    actual: object
    context: str = ""

    def __str__(self):
        s = f"Action {self.action_id} [{self.phase}] {self.field}: expected={self.expected}, actual={self.actual}"
        if self.context:
            s += f" ({self.context})"
        return s


def ensure_extracts(data_dir: str) -> None:
    """Generate any missing or stale ``_extract.json`` files."""
    data_path = Path(data_dir)
    repo_root = Path(__file__).resolve().parents[2]
    extractor = repo_root / "utils_18xx" / "extract_states.rb"
    engine_root = repo_root / "submodules" / "18xx" / "lib" / "engine"

    dependency_mtime_ns = extractor.stat().st_mtime_ns
    for source in engine_root.rglob("*.rb"):
        dependency_mtime_ns = max(dependency_mtime_ns, source.stat().st_mtime_ns)

    for game_path in sorted(data_path.glob("*.json")):
        if game_path.name.endswith("_extract.json"):
            continue
        extract_path = game_path.with_name(f"{game_path.stem}_extract.json")
        if extract_path.exists() and extract_path.stat().st_mtime_ns < dependency_mtime_ns:
            extract_path.unlink()

    result = subprocess.run(
        ["ruby", str(extractor), str(data_path)],
        capture_output=True,
        text=True,
        timeout=1800,
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError("State extractor failed:\n" + result.stderr)


def load_ref_states(game_json_path: str) -> list[dict]:
    extract_path = Path(str(game_json_path).replace(".json", "_extract.json"))
    if not extract_path.exists():
        raise FileNotFoundError(
            f"Extract file not found: {extract_path}\n"
            f"Run: ruby utils_18xx/extract_states.rb {Path(game_json_path).parent}"
        )
    return json.loads(extract_path.read_text())


@dataclass
class ReplayHarness:
    game_json_path: str
    ref_states: list = field(default_factory=list)
    verbose: bool = False
    mismatches: list = field(default_factory=list)

    def run(self) -> list[Mismatch]:
        self.mismatches = []

        game_data = json.loads(Path(self.game_json_path).read_text())
        ref_states = self.ref_states
        ref_by_action = {snapshot["action_id"]: snapshot for snapshot in ref_states}
        initial = ref_by_action[0]

        num_players = len(game_data["players"])
        layout = ActionLayout(num_players)
        self._player_id_to_index = {
            player_id: idx for idx, player_id in enumerate(initial["player_order"])
        }
        self._engine_index_to_player_id = list(initial["player_order"])

        state = initialize_replay_state(
            num_players,
            initial["deck_order"],
            initial["initial_offering"],
            cost_level=initial.get("cost_level"),
        )

        self._compare_state(state, initial, "initial")
        self._last_ref = initial

        committed_ids = set(initial.get("committed_action_ids", []))
        actions = flatten_auto_actions(
            filter_actions(game_data.get("actions", []), committed_ids or None)
        )

        action_idx = 0
        while action_idx < len(actions):
            settle_to_player_choice(state)
            phase = TURN.get_phase(state)
            if phase == PHASE_GAME_OVER:
                break

            action_idx = self._skip_already_satisfied_forced_actions(state, actions, action_idx, ref_by_action)
            if action_idx >= len(actions):
                break

            current_action = actions[action_idx]
            current_action_id = current_action.get("id", -1)
            if (
                current_action_id >= 0
                and current_action.get("type") == "pass"
                and (ref := ref_by_action.get(current_action_id)) is not None
                and self._phase_stage_key(phase) > self._round_stage_key(ref.get("round", ""))
                and self._matches_ref(state, ref, f"before action {current_action_id} (late pass already satisfied)")
            ):
                if self.verbose:
                    print(f"skip late already-satisfied pass {current_action_id}")
                self._last_ref = ref
                action_idx += 1
                continue

            if phase in (PHASE_ACQ, PHASE_ACQ_OFFER):
                action_idx = self._run_acquisition_round(state, actions, action_idx, layout, ref_by_action)
                continue

            if phase == PHASE_CLOSING:
                action_idx = self._run_closing_round(state, actions, action_idx, layout, ref_by_action)
                continue

            action_idx = self._replay_simple_action(state, actions, action_idx, layout, ref_by_action)

        final_ref = ref_states[-1] if ref_states else None
        self._finalize_replay_state(state, layout, final_ref)

        return self.mismatches

    def _replay_simple_action(self, state, actions, idx, layout, ref_by_action) -> int:
        action = actions[idx]
        action_id = action.get("id", -1)
        has_ref = action_id >= 0 and action_id in ref_by_action

        current_dividend_choice = (
            action.get("type") == "dividend"
            and TURN.get_phase(state) == PHASE_DIVIDENDS
            and TURN.get_active_corp(state) == CORP_NAME_TO_ID.get(action.get("entity"))
        )

        if self._should_skip_dividend_surface_auto_pass(state, actions, idx):
            if self.verbose:
                print(f"skip dividend-surface auto-pass at idx {idx}: {action.get('entity')}")
            return idx + 1

        if (
            has_ref
            and action.get("type") == "dividend"
            and not current_dividend_choice
            and self._matches_ref(state, ref_by_action[action_id], f"before action {action_id} (already satisfied)")
        ):
            if self.verbose:
                print(f"skip already-satisfied action {action_id}: {action.get('type')}")
            self._last_ref = ref_by_action[action_id]
            return idx + 1

        if has_ref and self._last_ref is not None and not self._should_skip_pre_action_compare(state, action):
            self._compare_state(state, self._last_ref, f"before action {action_id}")

        if has_ref and ref_by_action[action_id].get("forced"):
            if self.verbose:
                print(f"skip forced action {action_id}: {action.get('type')}")
            self._last_ref = ref_by_action[action_id]
            return idx + 1

        self._map_and_apply_action(
            state,
            action,
            layout,
            action_id=action_id,
            phase_name=self._get_phase_name(state),
            mapping_expected="mapped action",
            mapping_context=f"18xx_type={action.get('type')} entity={action.get('entity')}",
            invalid_expected="non-invalid status",
            invalid_context=f"18xx_type={action.get('type')} entity={action.get('entity')}",
        )

        if has_ref:
            self._last_ref = ref_by_action[action_id]
        elif action_id < 0 and action.get("type") == "pass":
            self._last_ref = None

        return idx + 1

    def _should_skip_pre_action_compare(self, state, action) -> bool:
        if not isinstance(self._last_ref, dict):
            return False
        if action.get("type") != "dividend":
            return False
        if TURN.get_phase(state) != PHASE_DIVIDENDS:
            return False
        return self._last_ref.get("round") != "DIV"

    def _should_skip_dividend_surface_auto_pass(self, state, actions, idx: int) -> bool:
        action = actions[idx]
        if action.get("id", -1) >= 0 or action.get("type") != "pass":
            return False
        if action.get("entity_type") != "player":
            return False
        if TURN.get_phase(state) != PHASE_DIVIDENDS:
            return False

        active_corp = TURN.get_active_corp(state)
        if active_corp < 0:
            return False

        next_action = self._next_real_action(actions, idx + 1)
        if next_action is None or next_action.get("type") != "dividend":
            return False

        return CORP_NAME_TO_ID.get(next_action.get("entity")) == active_corp

    @staticmethod
    def _next_real_action(actions, start_idx: int):
        for probe in range(start_idx, len(actions)):
            action = actions[probe]
            if action.get("id", -1) >= 0:
                return action
        return None

    def _run_acquisition_round(self, state, actions, idx, layout, ref_by_action) -> int:
        start_idx, end_idx, round_end_ref = self._find_round_segment(actions, idx, "ACQ", ref_by_action)
        if round_end_ref is None:
            return self._drain_silent_phase(
                state,
                actions,
                idx,
                layout,
                valid_phases=(PHASE_ACQ, PHASE_ACQ_OFFER),
                phase_name="ACQ",
            )

        first_action_id = self._first_real_action_id(actions, start_idx, end_idx)
        if first_action_id >= 0 and self._last_ref is not None:
            self._compare_state(state, self._last_ref, f"before action {first_action_id}")

        round_actions = actions[start_idx:end_idx]
        remaining_actions = list(round_actions)
        explicit_responses = self._collect_acquisition_responses(round_actions)
        trailing_auto_passes = [
            action for action in round_actions
            if action.get("id", -1) < 0 and action.get("type") == "pass"
        ]
        pending_offer = None
        max_turns = max(1, len(round_actions) * 3 + len(self._engine_index_to_player_id) * 2 + 4)

        for _ in range(max_turns):
            settle_to_player_choice(state)
            phase = TURN.get_phase(state)
            if phase not in (PHASE_ACQ, PHASE_ACQ_OFFER):
                break

            if phase == PHASE_ACQ_OFFER:
                if pending_offer is None:
                    live_response_idx = self._find_live_acquisition_response(state, remaining_actions)
                    if live_response_idx is not None:
                        response_action = remaining_actions.pop(live_response_idx)
                        accept = self._response_accepts(response_action)
                        response_action_id = response_action.get("id", first_action_id)
                        self._apply_offer_response(
                            state,
                            accept=accept,
                            action_id=response_action_id,
                            layout=layout,
                        )
                        continue

                    decline_pass_idx = self._find_decline_pass_for_pending_offer(
                        remaining_actions,
                        self._engine_index_to_player_id[TURN.get_active_player(state)],
                    )
                    if decline_pass_idx is not None:
                        decline_pass = remaining_actions.pop(decline_pass_idx)
                        self._apply_offer_response(
                            state,
                            accept=False,
                            action_id=decline_pass.get("id", first_action_id),
                            layout=layout,
                        )
                        continue

                    pending_offer_idx = self._infer_pending_acquisition_offer(state, remaining_actions)
                    if pending_offer_idx is None:
                        self.mismatches.append(
                            Mismatch(
                                action_id=first_action_id,
                                phase=self._get_phase_name(state),
                                field="phase_flow",
                                expected="known ACQ offer context",
                                actual="unexpected ACQ_OFFER without pending offer",
                            )
                        )
                        break
                    pending_offer = remaining_actions.pop(pending_offer_idx)

                pending_offer_id = pending_offer.get("id", -1)
                response_action = explicit_responses.pop(pending_offer_id, None)
                if response_action is not None:
                    remaining_actions = self._remove_action_by_id(
                        remaining_actions,
                        response_action.get("id", -1),
                    )
                    accept = self._response_accepts(response_action)
                    response_action_id = response_action.get("id", pending_offer_id)
                else:
                    decline_pass_idx = self._find_decline_pass_for_pending_offer(
                        remaining_actions,
                        self._engine_index_to_player_id[TURN.get_active_player(state)],
                    )
                    if decline_pass_idx is not None:
                        decline_pass = remaining_actions.pop(decline_pass_idx)
                        accept = False
                        response_action_id = decline_pass.get("id", pending_offer_id)
                    else:
                        accept = False
                        response_action_id = pending_offer_id

                self._apply_offer_response(
                    state,
                    accept=accept,
                    action_id=response_action_id,
                    layout=layout,
                )
                pending_offer = None
                continue

            external_offer_idx = self._find_externalizable_acquisition_offer(
                state,
                remaining_actions,
                explicit_responses,
                layout,
                ref_by_action,
            )
            if external_offer_idx is not None:
                offer = remaining_actions.pop(external_offer_idx)
                response_action = explicit_responses.get(offer.get("id", -1))
                if response_action is not None:
                    remaining_actions = self._remove_action_by_id(
                        remaining_actions,
                        response_action.get("id", -1),
                    )
                self._apply_external_acquisition_offer(state, offer, explicit_responses)
                pending_offer = None
                continue

            offer_idx = self._find_first_mappable_acquisition_offer(state, remaining_actions, layout)
            if offer_idx is not None:
                offer = remaining_actions.pop(offer_idx)
                pending_offer = offer
                engine_action = self._try_map_action(state, offer, layout)
                if engine_action is None:
                    response_action = explicit_responses.get(offer.get("id", -1))
                    if response_action is not None and not self._response_accepts(response_action):
                        remaining_actions = [
                            action
                            for action in remaining_actions
                            if action.get("id", -1) != response_action.get("id", -1)
                        ]
                        explicit_responses.pop(offer.get("id", -1), None)
                        pending_offer = None
                        continue

                    self._apply_scripted_action(state, offer, layout)
                    pending_offer = None
                    continue

                result = DRIVER.apply_action(state, engine_action)
                if result == STATUS_INVALID:
                    self.mismatches.append(
                        Mismatch(
                            action_id=offer.get("id", -1),
                            phase=self._get_phase_name(state),
                            field="action_validity",
                            expected="non-invalid status",
                            actual="STATUS_INVALID",
                            context=f"engine_action={engine_action} 18xx_type={offer.get('type')}",
                        )
                    )
                    pending_offer = None
                continue

            active_player = TURN.get_active_player(state)
            if active_player < 0:
                continue

            engine_player_id = self._engine_index_to_player_id[active_player]
            pass_idx = self._find_matching_acquisition_pass(remaining_actions, engine_player_id)
            if pass_idx is not None:
                external_offer_idx = self._find_externalizable_offer_before_pass(
                    state,
                    remaining_actions,
                    pass_idx,
                    explicit_responses,
                    layout,
                    ref_by_action,
                )
                if external_offer_idx is not None:
                    offer = remaining_actions.pop(external_offer_idx)
                    response_action = explicit_responses.get(offer.get("id", -1))
                    if response_action is not None:
                        remaining_actions = [
                            action
                            for action in remaining_actions
                            if action.get("id", -1) != response_action.get("id", -1)
                        ]
                    self._apply_external_acquisition_offer(state, offer, explicit_responses)
                    pending_offer = None
                    continue

                if pending_offer is None:
                    external_offer_idx = self._find_externalizable_offer_after_pass(
                        state,
                        remaining_actions,
                        pass_idx,
                        explicit_responses,
                        layout,
                        ref_by_action,
                    )
                    if external_offer_idx is not None:
                        offer = remaining_actions.pop(external_offer_idx)
                        response_action = explicit_responses.get(offer.get("id", -1))
                        if response_action is not None:
                            remaining_actions = [
                                action
                                for action in remaining_actions
                                if action.get("id", -1) != response_action.get("id", -1)
                            ]
                        self._apply_external_acquisition_offer(state, offer, explicit_responses)
                        pending_offer = None
                        continue

                pass_action = remaining_actions.pop(pass_idx)
                self._apply_phase_pass(
                    state,
                    layout,
                    action_id=pass_action.get("id", first_action_id),
                    phase_name="ACQ",
                )
                continue

            self._apply_phase_pass(state, layout, action_id=first_action_id, phase_name="ACQ")

        for _ in trailing_auto_passes:
            settle_to_player_choice(state)
            phase_name = self._get_phase_name(state)
            if TURN.get_phase(state) not in (PHASE_ACQ, PHASE_CLOSING):
                break
            if TURN.get_phase(state) == PHASE_CLOSING:
                if self._find_mappable_closing_action_for_active_player(state, actions, end_idx, layout) is not None:
                    break
            self._apply_phase_pass(state, layout, action_id=first_action_id, phase_name=phase_name)

        self._last_ref = round_end_ref
        return end_idx

    def _run_closing_round(self, state, actions, idx, layout, ref_by_action) -> int:
        probe_idx = idx
        while probe_idx < len(actions):
            leading = actions[probe_idx]
            if leading.get("id", -1) >= 0 or leading.get("type") != "pass":
                break
            probe_idx += 1

        start_idx, end_idx, round_end_ref = self._find_round_segment(actions, probe_idx, "CLO", ref_by_action)
        if round_end_ref is None:
            return self._drain_silent_phase(
                state,
                actions,
                idx,
                layout,
                valid_phases=(PHASE_CLOSING,),
                phase_name="CLOSING",
            )

        first_action_id = self._first_real_action_id(actions, start_idx, end_idx)
        leading_auto_passes = actions[idx:start_idx]
        for _ in leading_auto_passes:
            settle_to_player_choice(state)
            if TURN.get_phase(state) != PHASE_CLOSING:
                break
            self._apply_phase_pass(state, layout, action_id=first_action_id, phase_name="CLOSING")

        if first_action_id >= 0 and self._last_ref is not None:
            self._compare_state(state, self._last_ref, f"before action {first_action_id}")

        round_actions = actions[start_idx:end_idx]
        player_scripts = self._collect_closing_scripts(round_actions)
        trailing_auto_passes = [
            action for action in round_actions
            if action.get("id", -1) < 0 and action.get("type") == "pass"
        ]
        max_turns = max(1, len(self._engine_index_to_player_id) * 2 + 4)

        for _ in range(max_turns):
            settle_to_player_choice(state)
            if TURN.get_phase(state) != PHASE_CLOSING:
                break

            engine_player_id = self._engine_index_to_player_id[TURN.get_active_player(state)]
            for action in player_scripts.pop(engine_player_id, []):
                self._apply_scripted_action(state, action, layout)
                settle_to_player_choice(state)
                if TURN.get_phase(state) != PHASE_CLOSING:
                    break
                if self._stop_round_if_already_satisfied(
                    state,
                    round_end_ref,
                    f"closing round already satisfied after action {action.get('id', first_action_id)}",
                ):
                    return end_idx

            settle_to_player_choice(state)
            if TURN.get_phase(state) != PHASE_CLOSING:
                continue
            if self._stop_round_if_already_satisfied(
                state,
                round_end_ref,
                f"closing round already satisfied before auto-pass {first_action_id}",
            ):
                return end_idx

            active_player = TURN.get_active_player(state)
            if active_player < 0:
                continue
            if self._engine_index_to_player_id[active_player] != engine_player_id:
                continue

            self._apply_phase_pass(state, layout, action_id=first_action_id, phase_name="CLOSING")

        for _ in trailing_auto_passes:
            settle_to_player_choice(state)
            if TURN.get_phase(state) != PHASE_CLOSING:
                break
            self._apply_phase_pass(state, layout, action_id=first_action_id, phase_name="CLOSING")

        self._last_ref = round_end_ref
        return end_idx

    def _find_round_segment(self, actions, idx, round_name: str, ref_by_action):
        start_idx = idx
        if idx >= len(actions):
            return idx, idx, None

        action_id = actions[idx].get("id", -1)
        ref = ref_by_action.get(action_id)
        if ref is None or ref.get("round") != round_name:
            return idx, idx, None

        round_end_ref = ref_by_action.get(ref.get("round_end_action_id"), ref)
        if round_end_ref is None:
            return idx, idx, None

        round_end_action_id = round_end_ref.get("action_id", -1)
        end_idx = start_idx
        while end_idx < len(actions):
            action = actions[end_idx]
            action_id = action.get("id", -1)
            end_idx += 1
            if action_id == round_end_action_id:
                while end_idx < len(actions):
                    trailing = actions[end_idx]
                    if trailing.get("id", -1) >= 0 or trailing.get("type") != "pass":
                        break
                    end_idx += 1
                break

        return start_idx, end_idx, round_end_ref

    def _drain_silent_phase(self, state, actions, idx: int, layout, *, valid_phases: tuple[int, ...], phase_name: str) -> int:
        max_turns = max(1, len(self._engine_index_to_player_id) * 2 + 4)
        action_id = self._last_ref.get("action_id", -1) if isinstance(self._last_ref, dict) else -1
        cursor = idx

        while cursor < len(actions):
            action = actions[cursor]
            if action.get("id", -1) >= 0 or action.get("type") != "pass":
                break
            if TURN.get_phase(state) == PHASE_CLOSING:
                if self._find_mappable_closing_action_for_active_player(state, actions, cursor + 1, layout) is not None:
                    return cursor + 1
            settle_to_player_choice(state)
            if TURN.get_phase(state) not in valid_phases:
                return cursor
            if TURN.get_phase(state) == PHASE_ACQ_OFFER:
                self._apply_offer_response(state, accept=False, action_id=action_id, layout=layout)
            else:
                self._apply_phase_pass(state, layout, action_id=action_id, phase_name=phase_name)
            cursor += 1

        for _ in range(max_turns):
            settle_to_player_choice(state)
            if TURN.get_phase(state) not in valid_phases:
                return cursor
            if TURN.get_phase(state) == PHASE_ACQ_OFFER:
                self._apply_offer_response(state, accept=False, action_id=action_id, layout=layout)
            else:
                self._apply_phase_pass(state, layout, action_id=action_id, phase_name=phase_name)

        return cursor

    @staticmethod
    def _first_real_action_id(actions, start_idx: int, end_idx: int) -> int:
        for probe in range(start_idx, end_idx):
            action_id = actions[probe].get("id", -1)
            if action_id >= 0:
                return action_id
        return -1

    def _skip_already_satisfied_forced_actions(self, state, actions, idx: int, ref_by_action) -> int:
        cursor = idx
        furthest_match_idx = None
        furthest_match_ref = None

        while cursor < len(actions):
            action = actions[cursor]
            action_id = action.get("id", -1)
            ref = ref_by_action.get(action_id)
            if action_id < 0 or ref is None or not ref.get("forced"):
                break
            if self._matches_ref(state, ref, f"before action {action_id} (forced already satisfied)"):
                furthest_match_idx = cursor
                furthest_match_ref = ref
            cursor += 1

        if furthest_match_idx is None:
            return idx

        if self.verbose:
            print(
                f"skip already-satisfied forced actions "
                f"{actions[idx].get('id', -1)}-{actions[furthest_match_idx].get('id', -1)}"
            )
        self._last_ref = furthest_match_ref
        return furthest_match_idx + 1

    @staticmethod
    def _collect_acquisition_responses(round_actions):
        responses = {}
        unresolved_offers = []

        for action in round_actions:
            atype = action.get("type")
            if atype == "offer":
                unresolved_offers.append(action)
                continue

            if atype != "respond":
                continue

            normalized_response = ReplayHarness._normalize_acquisition_response(action)

            corp = normalized_response.get("corporation")
            company = normalized_response.get("company")
            matched_offer = None
            for offer in reversed(unresolved_offers):
                if offer.get("corporation") != corp:
                    continue
                if company is not None and offer.get("company") != company:
                    continue
                matched_offer = offer
                unresolved_offers.remove(offer)
                break

            if matched_offer is not None:
                matched_offer_id = matched_offer.get("id", -1)
                if matched_offer_id >= 0:
                    responses[matched_offer_id] = normalized_response

        return responses

    @staticmethod
    def _response_accepts(action) -> bool:
        accept = action.get("accept")
        if isinstance(accept, str):
            return accept.lower() == "true"
        return bool(accept)

    @staticmethod
    def _normalize_acquisition_response(action):
        normalized = dict(action)
        normalized["accept"] = ReplayHarness._response_accepts(action)
        return normalized

    @staticmethod
    def _remove_action_by_id(actions, action_id: int):
        if action_id < 0:
            return list(actions)
        return [action for action in actions if action.get("id", -1) != action_id]

    @staticmethod
    def _try_map_action(state, action, layout):
        try:
            return map_action(state, action, TURN.get_phase(state), layout)
        except (ValueError, KeyError, IndexError):
            return None

    def _find_first_mappable_acquisition_offer(self, state, remaining_actions, layout):
        for idx, action in enumerate(remaining_actions):
            if action.get("type") != "offer":
                continue
            if self._try_map_action(state, action, layout) is not None:
                return idx
        return None

    def _find_ref_company_owner(self, ref, company_name: str):
        for corp in ref.get("corporations", []):
            if company_name in corp.get("companies", []):
                return "corp", corp.get("name")
        for player in ref.get("players", []):
            if company_name in player.get("companies", []):
                return "player", player.get("name")
        if company_name in ref.get("foreign_investor", {}).get("companies", []):
            return "fi", None
        if company_name in ref.get("offering", []):
            return "offering", None
        return None, None

    def _is_self_player_corp_offer(self, state, action) -> bool:
        if action.get("type") != "offer":
            return False

        company_name = action.get("company")
        corp_name = action.get("corporation")
        entity = action.get("entity")
        if company_name is None or corp_name is None or entity is None:
            return False

        try:
            company_id = COMPANY_NAME_TO_ID[company_name]
            buyer_corp_id = CORP_NAME_TO_ID[corp_name]
        except KeyError:
            return False

        buyer_president_idx = CORPS[buyer_corp_id].get_president_id(state)
        if buyer_president_idx < 0:
            return False
        buyer_president_player_id = self._engine_index_to_player_id[buyer_president_idx]
        if buyer_president_player_id != entity:
            return False

        location = COMPANIES[company_id].get_location(state)
        owner_idx = COMPANIES[company_id].get_owner_id(state)
        if location == int(CompanyLocation.LOC_PLAYER):
            if owner_idx < 0:
                return False
            owner_player_id = self._engine_index_to_player_id[owner_idx]
            return owner_player_id == entity

        if location in (int(CompanyLocation.LOC_CORP), int(CompanyLocation.LOC_CORP_ACQ)):
            if owner_idx < 0:
                return False
            seller_president_idx = CORPS[owner_idx].get_president_id(state)
            if seller_president_idx < 0:
                return False
            seller_president_player_id = self._engine_index_to_player_id[seller_president_idx]
            return seller_president_player_id == entity

        return False

    def _external_offer_matches_ref_buyer(self, offer, response_action, ref_by_action) -> bool:
        terminal_action_id = offer.get("id", -1)
        if response_action is not None:
            terminal_action_id = response_action.get("id", terminal_action_id)
        ref = ref_by_action.get(terminal_action_id)
        if ref is None:
            return False

        owner_kind, owner_name = self._find_ref_company_owner(ref, offer.get("company"))
        return owner_kind == "corp" and owner_name == offer.get("corporation")

    def _find_externalizable_acquisition_offer(self, state, remaining_actions, explicit_responses, layout, ref_by_action):
        active_player = TURN.get_active_player(state)
        if active_player < 0:
            return None

        engine_player_id = self._engine_index_to_player_id[active_player]
        first_mappable_idx = self._find_first_mappable_acquisition_offer(state, remaining_actions, layout)
        if first_mappable_idx is None:
            return None

        for idx, action in enumerate(remaining_actions[:first_mappable_idx]):
            if action.get("type") != "offer":
                continue
            action_id = action.get("id", -1)
            response_action = explicit_responses.get(action_id)
            if response_action is not None:
                continue
            if not self._is_self_player_corp_offer(state, action):
                continue
            if not self._external_offer_matches_ref_buyer(action, None, ref_by_action):
                continue
            if self._try_map_action(state, action, layout) is not None:
                continue
            return idx

        blocked_offer_idx = None
        for idx, action in enumerate(remaining_actions[:first_mappable_idx]):
            if action.get("type") != "offer":
                continue
            if action.get("entity") != engine_player_id:
                continue

            action_id = action.get("id", -1)
            response_action = explicit_responses.get(action_id)
            if response_action is not None and not self._response_accepts(response_action):
                continue
            if self._try_map_action(state, action, layout) is None:
                blocked_offer_idx = idx
                break

        if blocked_offer_idx is None:
            return None

        for idx, action in enumerate(remaining_actions[:blocked_offer_idx]):
            if action.get("type") != "offer":
                continue
            if action.get("entity") == engine_player_id:
                continue

            action_id = action.get("id", -1)
            response_action = explicit_responses.get(action_id)
            if response_action is not None and not self._response_accepts(response_action):
                continue
            if not self._external_offer_matches_ref_buyer(action, response_action, ref_by_action):
                continue
            if self._try_map_action(state, action, layout) is not None:
                continue
            return idx

        return None

    def _apply_external_acquisition_offer(self, state, offer, explicit_responses) -> bool:
        action_id = offer.get("id", -1)
        response_action = explicit_responses.pop(action_id, None)
        if response_action is not None and not self._response_accepts(response_action):
            return True

        try:
            buyer_corp_id = CORP_NAME_TO_ID[offer["corporation"]]
            company_id = COMPANY_NAME_TO_ID[offer["company"]]
            price = int(offer["price"])
        except (KeyError, TypeError, ValueError) as exc:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=self._get_phase_name(state),
                    field="action_mapping",
                    expected="externalizable acquisition offer",
                    actual=str(exc),
                    context=f"18xx_type={offer.get('type')} entity={offer.get('entity')}",
                )
            )
            return False

        if apply_external_acquisition_transfer(state, buyer_corp_id, company_id, price):
            return True

        self.mismatches.append(
            Mismatch(
                action_id=action_id,
                phase=self._get_phase_name(state),
                field="action_validity",
                expected="external acquisition transfer applied",
                actual="False",
                context=(
                    f"buyer={offer.get('corporation')} company={offer.get('company')} "
                    f"price={offer.get('price')} entity={offer.get('entity')}"
                ),
            )
        )
        return False

    def _find_externalizable_offer_before_pass(self, state, remaining_actions, pass_idx, explicit_responses, layout, ref_by_action):
        for idx, action in enumerate(remaining_actions[:pass_idx]):
            if action.get("type") != "offer":
                continue

            action_id = action.get("id", -1)
            response_action = explicit_responses.get(action_id)
            if response_action is not None:
                continue
            if not self._is_self_player_corp_offer(state, action):
                continue
            if not self._external_offer_matches_ref_buyer(action, None, ref_by_action):
                continue
            if self._try_map_action(state, action, layout) is not None:
                continue

            earlier_accepted_offer_pending = False
            for earlier in remaining_actions[:idx]:
                if earlier.get("type") != "offer":
                    continue
                earlier_response = explicit_responses.get(earlier.get("id", -1))
                if earlier_response is not None and self._response_accepts(earlier_response):
                    earlier_accepted_offer_pending = True
                    break
            if earlier_accepted_offer_pending:
                continue

            return idx

        return None

    def _find_externalizable_offer_after_pass(self, state, remaining_actions, pass_idx, explicit_responses, layout, ref_by_action):
        for earlier in remaining_actions[:pass_idx]:
            if earlier.get("type") != "offer":
                continue
            earlier_response = explicit_responses.get(earlier.get("id", -1))
            if earlier_response is not None and self._response_accepts(earlier_response):
                return None

        for idx, action in enumerate(remaining_actions[pass_idx + 1:], start=pass_idx + 1):
            if action.get("type") != "offer":
                continue

            action_id = action.get("id", -1)
            response_action = explicit_responses.get(action_id)
            if response_action is not None:
                continue
            if not self._is_self_player_corp_offer(state, action):
                continue
            if not self._external_offer_matches_ref_buyer(action, None, ref_by_action):
                continue
            if self._try_map_action(state, action, layout) is not None:
                continue
            return idx

        return None

    def _find_matching_acquisition_pass(self, remaining_actions, engine_player_id):
        for idx, action in enumerate(remaining_actions):
            if action.get("type") != "pass":
                continue
            if action.get("id", -1) < 0:
                continue
            if action.get("entity") != engine_player_id:
                continue
            return idx
        return None

    def _find_decline_pass_for_pending_offer(self, remaining_actions, engine_player_id):
        pass_idx = self._find_matching_acquisition_pass(remaining_actions, engine_player_id)
        if pass_idx is None:
            return None

        for action in remaining_actions[:pass_idx]:
            if action.get("type") != "offer":
                continue
            if action.get("entity") != engine_player_id:
                continue
            return None

        return pass_idx

    def _find_live_acquisition_response(self, state, remaining_actions):
        offer_corp_id = TURN.get_acq_offer_corp(state)
        active_company_id = TURN.get_active_company(state)
        if offer_corp_id < 0:
            return None

        offer_corp_name = CORP_NAMES[offer_corp_id]
        active_company_name = COMPANY_NAMES[active_company_id] if active_company_id >= 0 else None

        for idx, action in enumerate(remaining_actions):
            if action.get("type") != "respond":
                continue
            if action.get("corporation") != offer_corp_name:
                continue
            action_company = action.get("company")
            if action_company is not None and action_company != active_company_name:
                continue
            return idx
        return None

    def _infer_pending_acquisition_offer(self, state, remaining_actions):
        active_player = TURN.get_active_player(state)
        if active_player < 0:
            return None
        engine_player_id = self._engine_index_to_player_id[active_player]
        for idx, action in enumerate(remaining_actions):
            if action.get("type") != "offer":
                continue
            if action.get("entity") != engine_player_id:
                continue
            return idx
        return None

    def _collect_closing_scripts(self, round_actions):
        scripts = {}
        for action in round_actions:
            if action.get("type") not in {"sell_company", "close"}:
                continue
            scripts.setdefault(action.get("entity"), []).append(action)
        return scripts

    def _find_mappable_closing_action_for_active_player(self, state, actions, start_idx: int, layout):
        if TURN.get_phase(state) != PHASE_CLOSING:
            return None
        active_player = TURN.get_active_player(state)
        if active_player < 0:
            return None
        engine_player_id = self._engine_index_to_player_id[active_player]
        for probe in range(start_idx, len(actions)):
            candidate = actions[probe]
            if candidate.get("id", -1) < 0:
                continue
            if candidate.get("type") not in {"sell_company", "close"}:
                continue
            if candidate.get("entity") != engine_player_id:
                continue
            try:
                mapped_candidate = map_action(state, candidate, TURN.get_phase(state), layout)
            except (ValueError, KeyError, IndexError):
                mapped_candidate = None
            if mapped_candidate is not None:
                return probe
        return None

    def _apply_declined_offer_if_mappable(self, state, offer, *, action_id: int, layout) -> bool:
        # 18xx.games ACQ offers can be interleaved in an order that our strict
        # sequential engine cannot always represent once a later accepted offer
        # has already consumed the target company. A declined offer leaves no
        # durable state change, so if it no longer maps cleanly we skip it
        # instead of treating it as a replay mismatch.
        try:
            engine_action = map_action(state, offer, TURN.get_phase(state), layout)
        except (ValueError, KeyError, IndexError):
            return False
        if engine_action is None:
            return False
        result = DRIVER.apply_action(state, engine_action)
        if result == STATUS_INVALID:
            return False
        settle_to_player_choice(state)
        if TURN.get_phase(state) == PHASE_ACQ_OFFER:
            self._apply_offer_response(state, accept=False, action_id=action_id, layout=layout)
        return True

    def _apply_offer_response(self, state, *, accept: bool, action_id: int, layout) -> None:
        action = {"type": "respond", "accept": accept}
        self._map_and_apply_action(
            state,
            action,
            layout,
            action_id=action_id,
            phase_name=self._get_phase_name(state),
            mapping_expected="mapped response action",
            mapping_context=f"accept={accept}",
            invalid_expected="non-invalid response status",
            invalid_context=f"accept={accept}",
        )

    def _apply_scripted_action(self, state, action, layout) -> None:
        action_id = action.get("id", -1)
        self._map_and_apply_action(
            state,
            action,
            layout,
            action_id=action_id,
            phase_name=self._get_phase_name(state),
            mapping_expected="mapped action",
            mapping_context=f"18xx_type={action.get('type')} entity={action.get('entity')}",
            invalid_expected="non-invalid status",
            invalid_context=f"18xx_type={action.get('type')} entity={action.get('entity')}",
        )

    def _apply_phase_pass(self, state, layout, *, action_id: int, phase_name: str) -> None:
        self._map_and_apply_action(
            state,
            {"type": "pass"},
            layout,
            action_id=action_id,
            phase_name=phase_name,
            mapping_expected="pass action",
            mapping_context="phase adapter pass",
            invalid_expected="non-invalid status",
            invalid_context="phase adapter pass",
        )

    def _map_and_apply_action(
        self,
        state,
        action,
        layout,
        *,
        action_id: int,
        phase_name: str,
        mapping_expected: str,
        mapping_context: str,
        invalid_expected: str,
        invalid_context: str,
    ) -> bool:
        try:
            engine_action = map_action(state, action, TURN.get_phase(state), layout)
        except (ValueError, KeyError, IndexError) as exc:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="action_mapping",
                    expected=mapping_expected,
                    actual=str(exc),
                    context=mapping_context,
                )
            )
            return False

        if engine_action is None:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="action_mapping",
                    expected=mapping_expected,
                    actual="None",
                    context=mapping_context,
                )
            )
            return False

        result = DRIVER.apply_action(state, engine_action)
        if result == STATUS_INVALID:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="action_validity",
                    expected=invalid_expected,
                    actual="STATUS_INVALID",
                    context=invalid_context,
                )
            )
            return False

        return True

    def _stop_round_if_already_satisfied(self, state, round_end_ref, context: str) -> bool:
        if round_end_ref is None:
            return False
        if not self._matches_ref(state, round_end_ref, context):
            return False
        self._last_ref = round_end_ref
        return True

    def _finalize_replay_state(self, state, layout, final_ref) -> None:
        if TURN.get_phase(state) != PHASE_GAME_OVER:
            settle_to_player_choice(state)
            final_already_satisfied = (
                final_ref is not None
                and self._matches_ref(state, final_ref, "final already satisfied")
            )
            if (not final_already_satisfied) and TURN.get_phase(state) in (PHASE_ACQ, PHASE_ACQ_OFFER, PHASE_CLOSING):
                drain_offer_phases(state, layout)
                settle_to_player_choice(state)

        if final_ref is not None:
            self._compare_state(state, final_ref, "final")

    def _get_phase_name(self, state) -> str:
        phase = TURN.get_phase(state)
        names = {
            PHASE_INVEST: "INVEST",
            PHASE_BID: "BID",
            PHASE_WRAP_UP: "WRAP_UP",
            PHASE_ACQ: "ACQ",
            PHASE_ACQ_OFFER: "ACQ_OFFER",
            PHASE_CLOSING: "CLOSING",
            PHASE_INCOME: "INCOME",
            PHASE_DIVIDENDS: "DIVIDENDS",
            PHASE_END_CARD: "END_CARD",
            PHASE_ISSUE: "ISSUE",
            PHASE_IPO: "IPO",
            PHASE_GAME_OVER: "GAME_OVER",
        }
        return names.get(phase, f"UNKNOWN({phase})")

    @staticmethod
    def _phase_stage_key(phase: int) -> int:
        if phase in (PHASE_INVEST, PHASE_BID):
            return 0
        if phase in (PHASE_ACQ, PHASE_ACQ_OFFER):
            return 1
        if phase == PHASE_CLOSING:
            return 2
        if phase == PHASE_DIVIDENDS:
            return 3
        if phase == PHASE_ISSUE:
            return 4
        if phase == PHASE_IPO:
            return 5
        return -1

    @staticmethod
    def _round_stage_key(round_name: str) -> int:
        return {
            "INV": 0,
            "ACQ": 1,
            "CLO": 2,
            "DIV": 3,
            "ISS": 4,
            "IPO": 5,
        }.get(round_name, -1)

    def _matches_ref(self, state, ref: dict, context: str) -> bool:
        before = len(self.mismatches)
        self._compare_state(state, ref, context)
        if len(self.mismatches) == before:
            return True
        del self.mismatches[before:]
        return False

    def _compare_state(self, state, ref: dict, context: str):
        action_id = ref.get("action_id", -1)
        phase_name = self._get_phase_name(state)
        phase = TURN.get_phase(state)
        ref_round = ref.get("round", "")

        phases_aligned = (
            (phase == PHASE_INVEST and ref_round == "INV")
            or (phase == PHASE_BID and ref_round == "INV")
            or (phase == PHASE_IPO and ref_round == "IPO")
            or (phase == PHASE_DIVIDENDS and ref_round == "DIV")
            or (phase == PHASE_ISSUE and ref_round == "ISS")
        )

        if phases_aligned:
            ref_active_player = ref.get("active_player")
            ref_action_type = ref.get("action_type", "")
            skip_active_player = ref_action_type == "end_game"
            if ref_active_player is not None and phase in (PHASE_INVEST, PHASE_BID, PHASE_IPO) and not skip_active_player:
                our_active = TURN.get_active_player(state)
                expected_idx = self._player_id_to_index.get(ref_active_player)
                if expected_idx is not None and our_active != expected_idx:
                    self.mismatches.append(
                        Mismatch(
                            action_id=action_id,
                            phase=phase_name,
                            field="active_player",
                            expected=expected_idx,
                            actual=our_active,
                            context=context,
                        )
                    )

            ref_active_corp = ref.get("active_corp")
            if ref_active_corp is not None and phase in (PHASE_DIVIDENDS, PHASE_ISSUE):
                ref_corp_id = CORP_NAME_TO_ID.get(ref_active_corp)
                our_corp_id = TURN.get_active_corp(state)
                if ref_corp_id is not None and our_corp_id != ref_corp_id:
                    self.mismatches.append(
                        Mismatch(
                            action_id=action_id,
                            phase=phase_name,
                            field="active_corp",
                            expected=ref_active_corp,
                            actual=CORP_NAMES[our_corp_id] if 0 <= our_corp_id < 8 else our_corp_id,
                            context=context,
                        )
                    )

        for ref_player in ref.get("players", []):
            player_name = ref_player["name"]
            player_id_18xx = ref_player["id"]

            try:
                pidx = self._find_player_index(player_id_18xx)
            except ValueError:
                continue

            our_cash = PLAYERS[pidx].get_cash(state)
            ref_cash = ref_player["cash"]
            if our_cash != ref_cash:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"player[{player_name}].cash",
                        expected=ref_cash,
                        actual=our_cash,
                        context=context,
                    )
                )

            our_value = PLAYERS[pidx].get_net_worth(state)
            ref_value = ref_player["value"]
            if our_value != ref_value:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"player[{player_name}].value",
                        expected=ref_value,
                        actual=our_value,
                        context=context,
                    )
                )

            our_companies = sorted(
                COMPANY_NAMES[cid]
                for cid in range(36)
                if COMPANIES[cid].is_owned_by_player(state, pidx)
            )
            ref_companies = sorted(ref_player.get("companies", []))
            if our_companies != ref_companies:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"player[{player_name}].companies",
                        expected=ref_companies,
                        actual=our_companies,
                        context=context,
                    )
                )

            our_shares = {}
            for corp_id in range(8):
                shares = PLAYERS[pidx].get_shares(state, corp_id)
                if shares > 0:
                    our_shares[CORP_NAMES[corp_id]] = shares
            ref_shares = ref_player.get("shares", {})
            if our_shares != ref_shares:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"player[{player_name}].shares",
                        expected=ref_shares,
                        actual=our_shares,
                        context=context,
                    )
                )

        for ref_corp in ref.get("corporations", []):
            corp_name = ref_corp["name"]
            corp_id = CORP_NAME_TO_ID.get(corp_name)
            if corp_id is None:
                continue

            ref_floated = ref_corp["floated"]
            our_active = CORPS[corp_id].is_active(state)

            if ref_floated and not our_active:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].active",
                        expected=True,
                        actual=False,
                        context=context,
                    )
                )
                continue

            if not ref_floated:
                if our_active:
                    self.mismatches.append(
                        Mismatch(
                            action_id=action_id,
                            phase=phase_name,
                            field=f"corp[{corp_name}].active",
                            expected=False,
                            actual=True,
                            context=context,
                        )
                    )
                continue

            our_price = CORPS[corp_id].get_share_price(state)
            ref_price = ref_corp["price"]
            if ref_price is not None and our_price != ref_price:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].price",
                        expected=ref_price,
                        actual=our_price,
                        context=context,
                    )
                )

            our_corp_cash = CORPS[corp_id].get_cash(state)
            ref_corp_cash = ref_corp["cash"]
            if our_corp_cash != ref_corp_cash:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].cash",
                        expected=ref_corp_cash,
                        actual=our_corp_cash,
                        context=context,
                    )
                )

            our_corp_companies = sorted(
                COMPANY_NAMES[cid]
                for cid in range(36)
                if COMPANIES[cid].is_owned_by_corp(state, corp_id)
                or COMPANIES[cid].is_in_corp_acquisition(state, corp_id)
            )
            ref_corp_companies = sorted(ref_corp.get("companies", []))
            if our_corp_companies != ref_corp_companies:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].companies",
                        expected=ref_corp_companies,
                        actual=our_corp_companies,
                        context=context,
                    )
                )

            our_market_shares = CORPS[corp_id].get_bank_shares(state)
            ref_market_shares = ref_corp.get("shares_in_market", 0)
            if our_market_shares != ref_market_shares:
                self.mismatches.append(
                    Mismatch(
                        action_id=action_id,
                        phase=phase_name,
                        field=f"corp[{corp_name}].shares_in_market",
                        expected=ref_market_shares,
                        actual=our_market_shares,
                        context=context,
                    )
                )

        ref_fi = ref.get("foreign_investor", {})
        our_fi_cash = FI.get_cash(state)
        ref_fi_cash = ref_fi.get("cash", 0)
        if our_fi_cash != ref_fi_cash:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="fi.cash",
                    expected=ref_fi_cash,
                    actual=our_fi_cash,
                    context=context,
                )
            )

        our_fi_companies = sorted(
            COMPANY_NAMES[cid]
            for cid in range(36)
            if COMPANIES[cid].is_owned_by_fi(state)
        )
        ref_fi_companies = sorted(ref_fi.get("companies", []))
        if our_fi_companies != ref_fi_companies:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="fi.companies",
                    expected=ref_fi_companies,
                    actual=our_fi_companies,
                    context=context,
                )
            )

        our_offering = sorted(
            COMPANY_NAMES[cid]
            for cid in range(36)
            if COMPANIES[cid].get_location(state) in (LOC_AUCTION, LOC_REVEALED)
        )
        ref_offering = sorted(ref.get("offering", []))
        if our_offering != ref_offering:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="offering",
                    expected=ref_offering,
                    actual=our_offering,
                    context=context,
                )
            )

        our_deck_size = DECK.get_remaining_count(state)
        ref_deck_size = ref.get("deck_size", 0)
        if our_deck_size != ref_deck_size:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="deck_size",
                    expected=ref_deck_size,
                    actual=our_deck_size,
                    context=context,
                )
            )

        our_coo = TURN.get_coo_level(state)
        ref_coo = ref.get("cost_level", 0)
        if our_coo != ref_coo:
            self.mismatches.append(
                Mismatch(
                    action_id=action_id,
                    phase=phase_name,
                    field="cost_level",
                    expected=ref_coo,
                    actual=our_coo,
                    context=context,
                )
            )

    def _find_player_index(self, player_id_18xx: int) -> int:
        idx = self._player_id_to_index.get(player_id_18xx)
        if idx is not None:
            return idx
        raise ValueError(f"Player {player_id_18xx} not found in player_order")


def format_mismatches(mismatches: list[Mismatch]) -> str:
    lines = [str(m) for m in mismatches[:50]]
    if len(mismatches) > 50:
        lines.append(f"... and {len(mismatches) - 50} more")
    return "\n".join(lines)
