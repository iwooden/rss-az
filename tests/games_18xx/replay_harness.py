"""Replay harness: replays 18xx game JSON through our Cython engine.

Loads reference state snapshots (from the Ruby extractor), initializes our
engine with the 18xx deck order, and replays actions — comparing state at
phase boundaries to catch discrepancies.

Usage:
    from tests.games_18xx.replay_harness import ReplayHarness

    harness = ReplayHarness(
        "tests/games_18xx/data/224885.json",
        "tests/games_18xx/data/224885_states.json",
    )
    mismatches = harness.run()
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.state import GameState
from core.driver import (
    DRIVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.data import (
    COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, COMPANY_NAMES, CORP_NAMES,
    get_company_low_price,
)
from core.actions import get_valid_action_mask
from entities.deck import DECK
from entities.turn import TURN
from entities.company import COMPANIES, CompanyLocation
LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.fi import FI

from utils_18xx.action_parser import (
    ActionLayout,
    filter_actions,
    flatten_auto_actions,
    map_action,
    override_deck_and_offering,
    PHASE_INVEST,
    PHASE_BID,
    PHASE_WRAP_UP,
    PHASE_ACQ,
    PHASE_CLOSING,
    PHASE_INCOME,
    PHASE_DIVIDENDS,
    PHASE_END_CARD,
    PHASE_ISSUE,
    PHASE_IPO,
    PHASE_PAR,
    PHASE_GAME_OVER,
)
from utils_18xx.replay_state import (
    apply_external_acquisition_transfer,
    apply_external_close,
    is_closing_transition_pending,
)


@dataclass
class Mismatch:
    """A single state mismatch between our engine and the 18xx reference."""
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
    """Generate any missing or stale ``_extract.json`` files.

    Extracts are refreshed when they predate the Ruby extractor or any Ruby
    source file under the vendored 18xx.games engine tree. This keeps the
    checked-in fixtures aligned with extractor and engine changes without
    forcing a full refresh on every test run.
    """
    import subprocess

    base_dir = Path(__file__).parent
    data_path = Path(data_dir)
    repo_root = base_dir.parent.parent
    extractor = repo_root / "utils_18xx" / "extract_states.rb"
    engine_root = repo_root / "submodules" / "18xx" / "lib" / "engine"

    dependency_mtime_ns = extractor.stat().st_mtime_ns
    for source in engine_root.rglob("*.rb"):
        dependency_mtime_ns = max(dependency_mtime_ns, source.stat().st_mtime_ns)

    # Delete stale extracts so Ruby's batch mode sees them as missing.
    for game_path in sorted(data_path.glob("*.json")):
        if game_path.name.endswith("_extract.json"):
            continue
        extract_path = game_path.with_name(f"{game_path.stem}_extract.json")
        if (
            extract_path.exists()
            and extract_path.stat().st_mtime_ns < dependency_mtime_ns
        ):
            extract_path.unlink()

    # Single Ruby process handles all missing/stale extracts at once.
    try:
        result = subprocess.run(
            ["ruby", str(extractor), str(data_path)],
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Ruby not found. Install Ruby to run 18xx replay tests."
        )

    if result.returncode != 0:
        raise RuntimeError(
            "State extractor failed:\n" + result.stderr
        )


def load_ref_states(game_json_path: str) -> list[dict]:
    """Load pre-extracted reference states from the corresponding _extract.json file."""
    extract_path = game_json_path.replace(".json", "_extract.json")
    if not Path(extract_path).exists():
        raise FileNotFoundError(
            f"Extract file not found: {extract_path}\n"
            f"Run: ruby utils_18xx/extract_states.rb tests/games_18xx/data/"
        )
    return json.loads(Path(extract_path).read_text())


@dataclass
class ReplayHarness:
    """Orchestrates replay of an 18xx game through our Cython engine."""

    game_json_path: str
    ref_states: list = field(default_factory=list)
    verbose: bool = False
    mismatches: list = field(default_factory=list)
    _round_by_action_id: dict[int, dict] = field(default_factory=dict, repr=False)

    def run(self) -> list[Mismatch]:
        """Run the full replay. Returns list of mismatches (empty = success)."""
        self.mismatches = []

        # Load data
        game_data = json.loads(Path(self.game_json_path).read_text())
        ref_states = self.ref_states

        num_players = len(game_data['players'])
        layout = ActionLayout(num_players)

        # Build reference state lookup: action_id -> snapshot
        ref_by_action = {s['action_id']: s for s in ref_states}

        # Get initial record (action_id=0) for deck order and player mapping
        initial = ref_by_action[0]
        deck_order_names = initial['deck_order']
        offering_names = initial['initial_offering']

        # Build static player ID -> engine index mapping from initial player_order.
        self._player_id_to_index = {}
        for idx, pid in enumerate(initial['player_order']):
            self._player_id_to_index[pid] = idx

        # Initialize our engine and override deck/offering to match 18xx game.
        state = GameState(num_players)
        state.initialize_game(seed=42)
        state.pause_before_acq_transition = True
        state.pause_before_closing_transition = True
        override_deck_and_offering(state, deck_order_names, offering_names)

        round_by_seq = {round_info['seq']: round_info for round_info in initial.get('rounds', [])}
        self._round_by_action_id = {
            snapshot['action_id']: round_by_seq[snapshot['round_seq']]
            for snapshot in ref_states
            if snapshot.get('round_seq') in round_by_seq
        }

        # Verify initial state matches
        self._compare_state(state, initial, "initial")
        self._last_ref = initial

        # Filter and flatten actions. The extractor embeds committed_action_ids
        # in the initial record so we can cleanly drop undone actions.
        raw_actions = game_data.get('actions', [])
        committed_ids = set(initial.get('committed_action_ids', []))
        actions = filter_actions(raw_actions, committed_ids or None)
        actions = flatten_auto_actions(actions)

        # Replay loop
        action_idx_in_stream = 0
        while action_idx_in_stream < len(actions):
            phase = TURN.get_phase(state)

            if phase == PHASE_GAME_OVER:
                break

            if self.verbose and action_idx_in_stream < len(actions):
                next_action = actions[action_idx_in_stream]
                next_id = next_action.get('id', -1)
                next_type = next_action.get('type', '')
                phase_name = self._get_phase_name(state)
                print(f"  [idx={action_idx_in_stream} aid={next_id}] engine_phase={phase_name} action_type={next_type}")

            if phase == PHASE_ACQ:
                action_idx_in_stream = self._run_acquisition_adapter(
                    state, actions, action_idx_in_stream,
                    layout, ref_by_action,
                )
            elif phase == PHASE_CLOSING:
                action_idx_in_stream = self._run_closing_adapter(
                    state, actions, action_idx_in_stream,
                    layout, ref_by_action,
                )
            else:
                action_idx_in_stream = self._replay_simple_action(
                    state, actions, action_idx_in_stream,
                    layout, ref_by_action,
                )

        # Final comparison at game end. The engine may auto-apply forced
        # actions and reach GAME_OVER with some 18xx actions left unconsumed.
        final_ref = self._last_ref
        for remaining_idx in range(action_idx_in_stream, len(actions)):
            remaining_id = actions[remaining_idx].get('id', -1)
            if remaining_id >= 0 and remaining_id in ref_by_action:
                final_ref = ref_by_action[remaining_id]
        if final_ref is not None:
            self._compare_state(state, final_ref, "final")

        return self.mismatches

    def _get_phase_name(self, state) -> str:
        """Get human-readable phase name."""
        phase = TURN.get_phase(state)
        names = {
            PHASE_INVEST: "INVEST", PHASE_BID: "BID", PHASE_WRAP_UP: "WRAP_UP",
            PHASE_ACQ: "ACQ", PHASE_CLOSING: "CLOSING", PHASE_INCOME: "INCOME",
            PHASE_DIVIDENDS: "DIVIDENDS", PHASE_END_CARD: "END_CARD",
            PHASE_ISSUE: "ISSUE", PHASE_IPO: "IPO", PHASE_PAR: "PAR",
            PHASE_GAME_OVER: "GAME_OVER",
        }
        return names.get(phase, f"UNKNOWN({phase})")

    def _replay_simple_action(self, state, actions, idx, layout, ref_by_action):
        """Replay a single action for simple phases.

        Compares engine state BEFORE applying each action against the reference
        snapshot from the previous action. This avoids phase-alignment issues
        caused by the engine auto-advancing through forced phases.
        """
        if idx >= len(actions):
            return idx

        return self._replay_simple_action_core(state, actions, idx, layout, ref_by_action)

    def _replay_simple_action_core(self, state, actions, idx, layout, ref_by_action):
        """Replay a single non-ACQ/CLO action without placeholder-pass wrapping."""
        action = actions[idx]
        action_id = action.get('id', -1)
        has_ref = action_id >= 0 and action_id in ref_by_action

        # Skip forced actions already auto-applied by the driver.
        if has_ref and ref_by_action[action_id].get('forced'):
            if self.verbose:
                print(f"  Skipping forced action {action_id} ({action.get('type')} for {action.get('entity')})")
            self._last_ref = ref_by_action[action_id]
            return idx + 1

        # Skip ACQ/CLO-round actions when the engine already advanced past
        # those phases. Comparing against the ACQ/CLO reference at that point
        # would produce false mismatches.
        if has_ref:
            ref_round = ref_by_action[action_id].get('round', '')
            phase = TURN.get_phase(state)
            if ref_round in ('ACQ', 'CLO') and phase not in (PHASE_ACQ, PHASE_CLOSING):
                if self.verbose:
                    print(
                        f"  Skipping {ref_round}-round action {action_id} "
                        f"(engine already in {self._get_phase_name(state)})"
                    )
                self._last_ref = None
                return idx + 1

        # Auto-advance when undone actions in the 18xx stream contained the
        # needed phase-transition passes and the committed stream no longer does.
        if has_ref:
            ref_round = ref_by_action[action_id].get('round', '')
            phase = TURN.get_phase(state)
            if self._should_auto_advance(phase, ref_round):
                self._auto_advance_to_round(state, ref_round, layout, action_id)

        # Compare BEFORE applying: our state should match the last reference
        if has_ref and self._last_ref is not None:
            self._compare_state(state, self._last_ref, f"before action {action_id}")

        try:
            engine_action = map_action(state, action, TURN.get_phase(state), layout)
        except (ValueError, KeyError, IndexError) as e:
            self.mismatches.append(Mismatch(
                action_id=action_id,
                phase=self._get_phase_name(state),
                field="action_mapping",
                expected="valid mapping",
                actual=str(e),
                context=f"18xx_type={action.get('type')}, entity={action.get('entity')}",
            ))
            if has_ref:
                self._last_ref = ref_by_action[action_id]
            return idx + 1

        if engine_action is None:
            ref_round = ref_by_action.get(action_id, {}).get('round', '') if has_ref else ''
            if ref_round in ('ACQ', 'CLO'):
                if self.verbose:
                    print(f"  Skipping {ref_round}-round action {action_id} (engine already advanced)")
                self._last_ref = None
                return idx + 1
            if ref_round in ('DIV', 'ISS', 'IPO') and TURN.get_phase(state) in (PHASE_INVEST, PHASE_BID):
                if self.verbose:
                    print(
                        f"  Skipping {ref_round}-round action {action_id} "
                        f"(engine already in {self._get_phase_name(state)})"
                    )
                self._last_ref = None
                return idx + 1
            if self.verbose:
                print(f"  Skipping unmappable action {action_id}: {action.get('type')}")
            if has_ref:
                self._last_ref = ref_by_action[action_id]
            return idx + 1

        # Normalize to list for uniform handling (IPO 'par' returns 2 actions)
        action_list = engine_action if isinstance(engine_action, list) else [engine_action]

        for i, ea in enumerate(action_list):
            # For multi-action sequences (IPO corp + PAR price), the driver may
            # auto-apply the second action if only one par price is valid.
            if i > 0 and TURN.get_phase(state) != PHASE_PAR:
                break
            result = DRIVER.apply_action(state, ea)
            if result == STATUS_INVALID:
                self.mismatches.append(Mismatch(
                    action_id=action_id,
                    phase=self._get_phase_name(state),
                    field="action_validity",
                    expected="STATUS_OK",
                    actual="STATUS_INVALID",
                    context=f"engine_action={ea}, 18xx_type={action.get('type')}",
                ))
                break
            if result == STATUS_PAUSED:
                break

        # Update reference for next comparison
        if has_ref:
            self._last_ref = ref_by_action[action_id]
        elif action_id < 0 and action.get('type') == 'pass':
            # Auto-action pass (from program_share_pass/program_close_pass):
            # clear _last_ref to avoid false mismatches on the next real action.
            self._last_ref = None

        return idx + 1

    def _run_acquisition_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through acquisition phase, matching 18xx offer/respond actions
        against our engine's offer buffer and pausing at ACQ exhaustion so
        deferred transfers can be patched before CLO begins.

        Returns new index into actions stream (past all acquisition actions).
        """
        round_info, round_start_idx = self._find_upcoming_round(actions, idx, 'ACQ', ref_by_action)
        end_idx = self._get_round_end_idx(actions, round_start_idx, round_info, ref_by_action) if round_info else idx

        acq_outcomes: dict[str, tuple[str, int]] = {}
        acq_outcome_details: dict[str, dict] = {}
        cross_president_outcomes: list[dict] = []
        outcomes = self._get_round_summary(round_info).get('acq_outcomes', [])
        for outcome in outcomes:
            if outcome.get('cross_president', False):
                cross_president_outcomes.append(outcome)
                continue
            acq_outcomes[outcome['company']] = (outcome['buyer'], outcome['price'])
            acq_outcome_details[outcome['company']] = outcome

        # Pre-apply same-president corp-to-corp transfers. Fixed engine offer
        # ordering can otherwise diverge once earlier buys/closures change the
        # seller's remaining-company count.
        for company_name in list(acq_outcomes.keys()):
            outcome = acq_outcome_details.get(company_name)
            if outcome is None or outcome.get('seller_type') != 'corp':
                continue
            company_id = COMPANY_NAME_TO_ID.get(company_name)
            buyer_corp_id = CORP_NAME_TO_ID.get(outcome['buyer'])
            seller_corp_id = CORP_NAME_TO_ID.get(outcome.get('seller', ''))
            if company_id is None or buyer_corp_id is None or seller_corp_id is None:
                continue
            price = outcome['price']
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            current = CORPS[seller_corp_id].get_acquisition_proceeds(state)
            CORPS[seller_corp_id].set_acquisition_proceeds(state, current + price)
            del acq_outcomes[company_name]
            if self.verbose:
                print(
                    f"  ACQ adapter: pre-applied corp-corp transfer "
                    f"{outcome['seller']}->{outcome['buyer']} for {company_name} at {price}"
                )

        if self.verbose and acq_outcomes:
            print(f"  ACQ adapter: outcomes={acq_outcomes}")

        # Walk our engine's offer buffer until we hit the paused ACQ boundary.
        max_iterations = 200
        iterations = 0
        while TURN.get_phase(state) == PHASE_ACQ and iterations < max_iterations:
            iterations += 1

            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                break
            if legal_count == 1:
                result = None
                for i, v in enumerate(mask):
                    if v > 0.5:
                        result = DRIVER.apply_action(state, i)
                        if result == STATUS_INVALID:
                            self.mismatches.append(Mismatch(
                                action_id=round_info['compare_after_action_id'] if round_info else -1,
                                phase=self._get_phase_name(state),
                                field="action_validity",
                                expected="STATUS_OK",
                                actual="STATUS_INVALID",
                                context=f"forced_action={i}",
                            ))
                        break
                if result == STATUS_PAUSED:
                    break
                continue

            acq_corp_id = TURN.get_acq_active_corp(state)
            acq_company_id = TURN.get_acq_target_company(state)

            if acq_corp_id < 0 or acq_company_id < 0:
                result = DRIVER.apply_action(state, layout.acq_pass)
                if result == STATUS_PAUSED:
                    break
                continue

            corp_name = CORP_NAMES[acq_corp_id]
            company_name = COMPANY_NAMES[acq_company_id]
            is_fi = TURN.is_acq_fi_offer(state)

            if company_name in acq_outcomes and acq_outcomes[company_name][0] == corp_name:
                if self.verbose:
                    print(f"  ACQ offer: {corp_name} -> {company_name} [ACCEPT]")

                if is_fi:
                    engine_action = layout.acq_fi_buy
                else:
                    price = acq_outcomes[company_name][1]
                    low_price = get_company_low_price(acq_company_id)
                    price_offset = price - low_price
                    engine_action = layout.acq_price_base + price_offset

                result = DRIVER.apply_action(state, engine_action)
                if result == STATUS_INVALID:
                    if self.verbose:
                        print(f"  ACQ: Invalid action for {corp_name}->{company_name}, passing")
                    result = DRIVER.apply_action(state, layout.acq_pass)
                else:
                    del acq_outcomes[company_name]
            else:
                if self.verbose:
                    print(f"  ACQ offer: {corp_name} -> {company_name} [PASS]")
                result = DRIVER.apply_action(state, layout.acq_pass)
            if result == STATUS_PAUSED:
                break

        if TURN.get_phase(state) == PHASE_ACQ and DRIVER.is_non_player_phase(state):
            for company_name in list(acq_outcomes.keys()):
                outcome = acq_outcome_details.get(company_name)
                if outcome is not None:
                    self._pre_apply_acq_outcome(state, outcome, "ACQ adapter")
                    del acq_outcomes[company_name]
            for outcome in cross_president_outcomes:
                self._pre_apply_acq_outcome(state, outcome, "ACQ adapter")
            DRIVER.advance_phase(state)
        elif acq_outcomes:
            for company_name, (buyer, price) in sorted(acq_outcomes.items()):
                self.mismatches.append(Mismatch(
                    action_id=round_info['compare_after_action_id'] if round_info else -1,
                    phase=self._get_phase_name(state),
                    field="acq_outcome",
                    expected=f"{buyer}@{price}",
                    actual="offer never matched",
                    context=company_name,
                ))

        self._last_ref = None
        return end_idx

    def _run_closing_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through closing phase, matching 18xx sell_company actions
        against our engine's offer buffer.

        Returns new index into actions stream.
        """
        round_info, round_start_idx = self._find_upcoming_round(actions, idx, 'CLO', ref_by_action)
        end_idx = self._get_round_end_idx(actions, round_start_idx, round_info, ref_by_action) if round_info else idx
        closed_companies = set(self._get_round_summary(round_info).get('closed_companies', []))

        max_iterations = 200
        iterations = 0
        while TURN.get_phase(state) == PHASE_CLOSING and iterations < max_iterations:
            iterations += 1

            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                if DRIVER.is_non_player_phase(state) and is_closing_transition_pending(state):
                    break
                if DRIVER.is_non_player_phase(state):
                    DRIVER.advance_phase(state)
                    continue
                break
            if legal_count == 1:
                result = None
                for i, v in enumerate(mask):
                    if v > 0.5:
                        result = DRIVER.apply_action(state, i)
                        if result == STATUS_INVALID:
                            self.mismatches.append(Mismatch(
                                action_id=round_info['compare_after_action_id'] if round_info else -1,
                                phase=self._get_phase_name(state),
                                field="action_validity",
                                expected="STATUS_OK",
                                actual="STATUS_INVALID",
                                context=f"forced_action={i}",
                            ))
                        break
                if result == STATUS_PAUSED:
                    break
                continue

            closing_company_id = TURN.get_closing_company(state)
            if closing_company_id < 0:
                result = DRIVER.apply_action(state, layout.close_pass)
                if result == STATUS_PAUSED:
                    break
                continue

            company_name = COMPANY_NAMES[closing_company_id]
            if company_name in closed_companies:
                result = DRIVER.apply_action(state, layout.close_action)
                if result == STATUS_INVALID:
                    result = DRIVER.apply_action(state, layout.close_pass)
                else:
                    closed_companies.discard(company_name)
            else:
                result = DRIVER.apply_action(state, layout.close_pass)
            if result == STATUS_PAUSED:
                break

        if TURN.get_phase(state) == PHASE_CLOSING and is_closing_transition_pending(state):
            for company_name in sorted(list(closed_companies)):
                company_id = COMPANY_NAME_TO_ID.get(company_name)
                if company_id is None:
                    continue
                if not apply_external_close(state, company_id):
                    self.mismatches.append(Mismatch(
                        action_id=round_info['compare_after_action_id'] if round_info else -1,
                        phase=self._get_phase_name(state),
                        field="closing_outcome",
                        expected="close",
                        actual="patch failed",
                        context=company_name,
                    ))
                    continue
                closed_companies.discard(company_name)
            DRIVER.advance_phase(state)

        for company_name in sorted(closed_companies):
            self.mismatches.append(Mismatch(
                action_id=round_info['compare_after_action_id'] if round_info else -1,
                phase=self._get_phase_name(state),
                field="closing_outcome",
                expected="close",
                actual="offer never matched",
                context=company_name,
            ))

        if TURN.get_phase(state) != PHASE_CLOSING:
            self._settle_to_player_choice(state)
        self._last_ref = None
        return end_idx

    def _pre_apply_acq_outcome(self, state, outcome: dict, context: str) -> bool:
        """Pre-apply an ACQ outcome at the paused ACQ boundary."""
        company_name = outcome.get('company', '')
        company_id = COMPANY_NAME_TO_ID.get(company_name)
        buyer_corp_id = CORP_NAME_TO_ID.get(outcome.get('buyer', ''))
        if company_id is None or buyer_corp_id is None:
            return False

        price = outcome.get('price', 0)
        seller_type = outcome.get('seller_type', '?')
        if not apply_external_acquisition_transfer(state, buyer_corp_id, company_id, price):
            return False

        if seller_type == 'player':
            seller_idx = self._player_id_to_index.get(outcome.get('seller_id'))
            seller_label = f"player[{seller_idx}]" if seller_idx is not None else "player[?]"
        elif seller_type == 'fi':
            seller_label = 'FI'
        else:
            seller_label = outcome.get('seller', '?')

        if self.verbose:
            print(
                f"  {context}: pre-applied {seller_type} transfer "
                f"{seller_label}->{outcome.get('buyer', '?')} for {company_name} at {price}"
            )
        return True

    # Maps 18xx round names to the engine phases they correspond to.
    _ROUND_TO_PHASES = {
        'INV': {PHASE_INVEST, PHASE_BID},
        'ACQ': {PHASE_ACQ},
        'CLO': {PHASE_CLOSING},
        'DIV': {PHASE_DIVIDENDS},
        'ISS': {PHASE_ISSUE},
        'IPO': {PHASE_IPO, PHASE_PAR},
    }

    def _should_auto_advance(self, phase: int, ref_round: str) -> bool:
        """Check if the engine needs to auto-advance to reach the action's round."""
        if not ref_round:
            return False
        expected_phases = self._ROUND_TO_PHASES.get(ref_round)
        if expected_phases is None:
            return False
        if phase in expected_phases:
            return False
        return phase in (PHASE_IPO, PHASE_PAR, PHASE_ISSUE, PHASE_DIVIDENDS)

    def _auto_advance_to_round(self, state, ref_round: str, layout, action_id: int):
        """Auto-advance the engine by applying passes until it reaches ref_round."""
        expected_phases = self._ROUND_TO_PHASES.get(ref_round, set())
        max_iterations = 100
        for _ in range(max_iterations):
            phase = TURN.get_phase(state)
            if phase in expected_phases or phase == PHASE_GAME_OVER:
                break

            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                break
            if legal_count == 1:
                for i, v in enumerate(mask):
                    if v > 0.5:
                        if self.verbose:
                            print(f"  Auto-advance: forced action in {self._get_phase_name(state)}")
                        DRIVER.apply_action(state, i)
                        break
            else:
                pass_action = self._find_pass_action(phase, layout)
                if pass_action is not None:
                    if self.verbose:
                        print(
                            f"  Auto-advance: pass in {self._get_phase_name(state)} "
                            f"(targeting {ref_round}, aid={action_id})"
                        )
                    DRIVER.apply_action(state, pass_action)
                else:
                    break

        self._last_ref = None

    @staticmethod
    def _find_pass_action(phase: int, layout) -> int | None:
        """Return the pass action index for the given phase, or None."""
        if phase == PHASE_IPO:
            return layout.ipo_pass
        if phase == PHASE_ISSUE:
            return layout.issue_pass
        if phase == PHASE_DIVIDENDS:
            return layout.dividend_base
        return None

    @staticmethod
    def _settle_to_player_choice(state) -> None:
        """Advance deterministic phases and forced actions until a real choice."""
        max_iterations = 200
        for _ in range(max_iterations):
            while DRIVER.is_non_player_phase(state):
                DRIVER.advance_phase(state)

            mask = get_valid_action_mask(state)
            legal_actions = [i for i, v in enumerate(mask) if v > 0.5]
            if len(legal_actions) != 1:
                return

            DRIVER.apply_action(state, legal_actions[0])

        raise RuntimeError("Exceeded settle-to-choice iteration limit")

    def _find_upcoming_round(self, actions, start_idx: int, round_name: str, ref_by_action):
        """Find the next committed round of the given type at or after start_idx."""
        for i in range(start_idx, len(actions)):
            action_id = actions[i].get('id', -1)
            ref = ref_by_action.get(action_id)
            if ref is None:
                continue
            ref_round = ref.get('round', '')
            if ref_round == round_name:
                return self._round_by_action_id.get(action_id), i
            if round_name in ('ACQ', 'CLO') and ref_round not in ('ACQ', 'CLO'):
                break
        return None, start_idx

    def _get_round_end_idx(self, actions, start_idx: int, round_info: dict | None, ref_by_action) -> int:
        """Return the first stream index after the given committed round."""
        if round_info is None:
            return start_idx

        round_seq = round_info['seq']
        idx = start_idx
        while idx < len(actions):
            action_id = actions[idx].get('id', -1)
            ref = ref_by_action.get(action_id)
            if ref is not None and ref.get('round_seq') != round_seq:
                break
            idx += 1
        return idx

    @staticmethod
    def _get_round_summary(round_info: dict | None) -> dict:
        """Return the extractor-provided summary for a committed round."""
        if round_info is None:
            return {}
        return round_info.get('summary') or {}

    def _compare_state(self, state, ref: dict, context: str):
        """Compare our engine state against a reference snapshot."""
        action_id = ref.get('action_id', -1)
        phase_name = self._get_phase_name(state)
        phase = TURN.get_phase(state)

        # Compare active player / active corp when phases are aligned.
        ref_round = ref.get('round', '')
        phases_aligned = (
            (phase == PHASE_INVEST and ref_round == 'INV')
            or (phase == PHASE_BID and ref_round == 'INV')
            or (phase == PHASE_IPO and ref_round == 'IPO')
            or (phase == PHASE_DIVIDENDS and ref_round == 'DIV')
            or (phase == PHASE_ISSUE and ref_round == 'ISS')
        )

        if phases_aligned:
            ref_active_player = ref.get('active_player')
            ref_action_type = ref.get('action_type', '')
            skip_active_player = (
                (phase == PHASE_IPO and ref_action_type == 'par')
                or ref_action_type == 'end_game'
            )
            if ref_active_player is not None and phase in (PHASE_INVEST, PHASE_BID, PHASE_IPO) and not skip_active_player:
                our_active = state.get_active_player()
                expected_idx = self._player_id_to_index.get(ref_active_player)
                if expected_idx is not None and our_active != expected_idx:
                    self.mismatches.append(Mismatch(
                        action_id=action_id, phase=phase_name,
                        field="active_player",
                        expected=expected_idx, actual=our_active, context=context,
                    ))

            ref_active_corp = ref.get('active_corp')
            if ref_active_corp is not None and phase in (PHASE_DIVIDENDS, PHASE_ISSUE):
                ref_corp_id = CORP_NAME_TO_ID.get(ref_active_corp)
                if ref_corp_id is not None:
                    if phase == PHASE_DIVIDENDS:
                        our_corp_id = TURN.get_dividend_corp(state)
                    else:
                        our_corp_id = TURN.get_issue_corp(state)
                    if our_corp_id != ref_corp_id:
                        self.mismatches.append(Mismatch(
                            action_id=action_id, phase=phase_name,
                            field="active_corp",
                            expected=ref_active_corp, actual=CORP_NAMES[our_corp_id] if 0 <= our_corp_id < 8 else our_corp_id,
                            context=context,
                        ))

        # Compare players
        for ref_player in ref.get('players', []):
            player_name = ref_player['name']
            player_id_18xx = ref_player['id']

            try:
                pidx = self._find_player_index(player_id_18xx)
            except ValueError:
                continue

            # Cash
            our_cash = PLAYERS[pidx].get_cash(state)
            ref_cash = ref_player['cash']
            if our_cash != ref_cash:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"player[{player_name}].cash",
                    expected=ref_cash, actual=our_cash, context=context,
                ))

            # Net worth
            our_value = PLAYERS[pidx].get_net_worth(state)
            ref_value = ref_player['value']
            if our_value != ref_value:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"player[{player_name}].value",
                    expected=ref_value, actual=our_value, context=context,
                ))

            # Companies owned
            our_companies = sorted([
                COMPANY_NAMES[cid] for cid in range(36)
                if COMPANIES[cid].is_owned_by_player(state, pidx)
            ])
            ref_companies = sorted(ref_player.get('companies', []))
            if our_companies != ref_companies:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"player[{player_name}].companies",
                    expected=ref_companies, actual=our_companies, context=context,
                ))

            # Shares
            our_shares = {}
            for corp_id in range(8):
                n = PLAYERS[pidx].get_shares(state, corp_id)
                if n > 0:
                    our_shares[CORP_NAMES[corp_id]] = n
            ref_shares = ref_player.get('shares', {})
            if our_shares != ref_shares:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"player[{player_name}].shares",
                    expected=ref_shares, actual=our_shares, context=context,
                ))

        # Compare corporations
        for ref_corp in ref.get('corporations', []):
            corp_name = ref_corp['name']
            corp_id = CORP_NAME_TO_ID.get(corp_name)
            if corp_id is None:
                continue

            ref_floated = ref_corp['floated']
            our_active = CORPS[corp_id].is_active(state)

            if ref_floated and not our_active:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"corp[{corp_name}].active",
                    expected=True, actual=False, context=context,
                ))
                continue

            if not ref_floated:
                if our_active:
                    self.mismatches.append(Mismatch(
                        action_id=action_id, phase=phase_name,
                        field=f"corp[{corp_name}].active",
                        expected=False, actual=True, context=context,
                    ))
                continue

            # Corp is active in both — compare details
            our_price = CORPS[corp_id].get_share_price(state)
            ref_price = ref_corp['price']
            if ref_price is not None and our_price != ref_price:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"corp[{corp_name}].price",
                    expected=ref_price, actual=our_price, context=context,
                ))

            our_corp_cash = CORPS[corp_id].get_cash(state)
            ref_corp_cash = ref_corp['cash']
            if our_corp_cash != ref_corp_cash:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"corp[{corp_name}].cash",
                    expected=ref_corp_cash, actual=our_corp_cash, context=context,
                ))

            our_corp_companies = sorted([
                COMPANY_NAMES[cid] for cid in range(36)
                if COMPANIES[cid].is_owned_by_corp(state, corp_id)
            ])
            ref_corp_companies = sorted(ref_corp.get('companies', []))
            if our_corp_companies != ref_corp_companies:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"corp[{corp_name}].companies",
                    expected=ref_corp_companies, actual=our_corp_companies,
                    context=context,
                ))

            our_market_shares = CORPS[corp_id].get_bank_shares(state)
            ref_market_shares = ref_corp.get('shares_in_market', 0)
            if our_market_shares != ref_market_shares:
                self.mismatches.append(Mismatch(
                    action_id=action_id, phase=phase_name,
                    field=f"corp[{corp_name}].shares_in_market",
                    expected=ref_market_shares, actual=our_market_shares,
                    context=context,
                ))

        # Compare FI
        ref_fi = ref.get('foreign_investor', {})
        our_fi_cash = FI.get_cash(state)
        ref_fi_cash = ref_fi.get('cash', 0)
        if our_fi_cash != ref_fi_cash:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="fi.cash",
                expected=ref_fi_cash, actual=our_fi_cash, context=context,
            ))

        our_fi_companies = sorted([
            COMPANY_NAMES[cid] for cid in range(36)
            if COMPANIES[cid].is_owned_by_fi(state)
        ])
        ref_fi_companies = sorted(ref_fi.get('companies', []))
        if our_fi_companies != ref_fi_companies:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="fi.companies",
                expected=ref_fi_companies, actual=our_fi_companies,
                context=context,
            ))

        # Compare offering
        our_offering = sorted([
            COMPANY_NAMES[cid] for cid in range(36)
            if COMPANIES[cid].get_location(state) in (LOC_AUCTION, LOC_REVEALED)
        ])
        ref_offering = sorted(ref.get('offering', []))
        if our_offering != ref_offering:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="offering",
                expected=ref_offering, actual=our_offering, context=context,
            ))

        # Compare deck size
        our_deck_size = DECK.get_remaining_count(state)
        ref_deck_size = ref.get('deck_size', 0)
        if our_deck_size != ref_deck_size:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="deck_size",
                expected=ref_deck_size, actual=our_deck_size, context=context,
            ))

        # Compare cost level (Ruby extractor normalizes to our numbering)
        our_coo = TURN.get_coo_level(state)
        ref_coo = ref.get('cost_level', 0)
        if our_coo != ref_coo:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="cost_level",
                expected=ref_coo, actual=our_coo, context=context,
            ))

    def _find_player_index(self, player_id_18xx: int) -> int:
        """Find our 0-based player index from an 18xx player ID."""
        idx = self._player_id_to_index.get(player_id_18xx)
        if idx is not None:
            return idx
        raise ValueError(f"Player {player_id_18xx} not found in player_order")


def format_mismatches(mismatches: list[Mismatch]) -> str:
    """Format mismatches for display."""
    lines = []
    for m in mismatches[:50]:  # Cap at 50 to avoid overwhelming output
        lines.append(str(m))
    if len(mismatches) > 50:
        lines.append(f"... and {len(mismatches) - 50} more")
    return "\n".join(lines)
