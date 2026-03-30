"""Replay harness: replays 18xx game JSON through our Cython engine.

Loads reference state snapshots (from the Ruby extractor), initializes our
engine with the 18xx deck order, and replays actions — comparing state at
phase boundaries to catch discrepancies.

Usage:
    from tests.18xx_games.replay_harness import ReplayHarness

    harness = ReplayHarness(
        "tests/18xx_games/data/224885.json",
        "tests/18xx_games/data/224885_states.json",
    )
    mismatches = harness.run()
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.state import GameState
from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.data import (
    COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, COMPANY_NAMES, CORP_NAMES,
    GamePhases, get_company_low_price, get_company_income,
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

import importlib
_ap = importlib.import_module("tests.18xx_games.action_parser")
ActionLayout = _ap.ActionLayout
AutoPassTracker = _ap.AutoPassTracker
filter_actions = _ap.filter_actions
flatten_auto_actions = _ap.flatten_auto_actions
entity_to_player_index = _ap.entity_to_player_index
find_auction_slot = _ap.find_auction_slot
map_invest_action = _ap.map_invest_action
map_bid_action = _ap.map_bid_action
map_ipo_action = _ap.map_ipo_action
map_dividend_action = _ap.map_dividend_action
map_issue_action = _ap.map_issue_action

# Phase constants
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
    extractor = base_dir / "extract_states.rb"
    engine_root = base_dir.parent.parent / "submodules" / "18xx" / "lib" / "engine"

    dependency_mtime_ns = extractor.stat().st_mtime_ns
    for source in engine_root.rglob("*.rb"):
        dependency_mtime_ns = max(dependency_mtime_ns, source.stat().st_mtime_ns)

    pending: list[tuple[Path, Path]] = []
    for game_path in sorted(data_path.glob("*.json")):
        if game_path.name.endswith("_extract.json"):
            continue
        extract_path = game_path.with_name(f"{game_path.stem}_extract.json")
        if (
            not extract_path.exists()
            or extract_path.stat().st_mtime_ns < dependency_mtime_ns
        ):
            pending.append((game_path, extract_path))

    if not pending:
        return

    try:
        failures = []
        for game_path, extract_path in pending:
            result = subprocess.run(
                ["ruby", str(extractor), str(game_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                failures.append(
                    f"{game_path.name} (exit {result.returncode}):\n{result.stderr}"
                )
                continue
            extract_path.write_text(result.stdout, encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            "Ruby not found. Install Ruby to run 18xx replay tests."
        )

    if failures:
        raise RuntimeError(
            "State extractor failed for one or more games:\n"
            + "\n\n".join(failures)
        )


def load_ref_states(game_json_path: str) -> list[dict]:
    """Load pre-extracted reference states from the corresponding _extract.json file."""
    extract_path = game_json_path.replace(".json", "_extract.json")
    if not Path(extract_path).exists():
        raise FileNotFoundError(
            f"Extract file not found: {extract_path}\n"
            f"Run: ruby tests/18xx_games/extract_states.rb tests/18xx_games/data/"
        )
    return json.loads(Path(extract_path).read_text())


@dataclass
class ReplayHarness:
    """Orchestrates replay of an 18xx game through our Cython engine."""

    game_json_path: str
    ref_states: list = field(default_factory=list)
    verbose: bool = False
    mismatches: list = field(default_factory=list)
    # ACQ/CLO action IDs already handled by the look-ahead. Used to avoid
    # re-running the same round pre-apply logic on later actions in that round.
    _lookahead_handled_round_actions: set[int] = field(default_factory=set, repr=False)
    # Round-local set of ACQ company names pre-applied by the look-ahead.
    # Keyed by the first ACQ snapshot's action_id for that round.
    _lookahead_preapplied_companies_by_round: dict[int, set[str]] = field(default_factory=dict, repr=False)
    # Pending player-cash credits from look-ahead cross-president transfers.
    # Ownership must be patched before the driver auto-cascades through ACQ/CLO,
    # but crediting the seller immediately can change the current INVEST round's
    # end-of-round logic. Apply the cash only after the triggering action settles.
    _pending_player_cash_by_player: dict[int, int] = field(default_factory=dict, repr=False)
    # When we skip an ACQ/CLO block because the engine already jumped forward,
    # the next flattened auto-pass can be a bookkeeping placeholder rather than
    # a real INVEST decision. Preserve turn order across that one pass so it
    # does not trigger a spurious WRAP_UP reorder.
    _preserve_order_on_next_auto_invest_pass: bool = field(default=False, repr=False)

    def run(self) -> list[Mismatch]:
        """Run the full replay. Returns list of mismatches (empty = success)."""
        self.mismatches = []

        # Load data
        game_data = json.loads(Path(self.game_json_path).read_text())
        ref_states = self.ref_states

        num_players = len(game_data['players'])
        players_json = game_data['players']
        layout = ActionLayout(num_players)

        # Build reference state lookup: action_id -> snapshot
        ref_by_action = {s['action_id']: s for s in ref_states}

        # Get initial record (action_id=0) for deck order and player mapping
        initial = ref_by_action[0]
        deck_order_names = initial['deck_order']
        offering_names = initial['initial_offering']

        # Build static player ID → engine index mapping from initial player_order.
        # The 18xx reference rotates player order between rounds, but our engine
        # uses fixed indices. player_order[0] = first player = our index 0.
        self._player_id_to_index = {}
        for idx, pid in enumerate(initial['player_order']):
            self._player_id_to_index[pid] = idx

        # Initialize our engine and override deck/offering to match 18xx game
        state = GameState(num_players)
        state.initialize_game(seed=42)  # seed doesn't matter, we override below
        self._override_deck_and_offering(state, deck_order_names, offering_names)

        # Verify initial state matches
        self._compare_state(state, initial, "initial")
        self._last_ref = initial

        # Pre-process actions
        raw_actions = game_data.get('actions', [])

        # Build auto-pass tracker from ALL actions (including program_* ones)
        auto_pass = AutoPassTracker([p['id'] for p in players_json])
        for a in raw_actions:
            atype = a.get('type', '')
            if atype.startswith('program_'):
                auto_pass.process_action(a)

        # Filter and flatten.  The extractor embeds committed_action_ids in the
        # initial record so we can cleanly drop undone actions without
        # reimplementing undo/redo logic in Python.
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

            if phase in (PHASE_ACQ,):
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

        # Final comparison at game end.
        # When the engine auto-applied forced actions (e.g. dividends for corps
        # with cash < issued_shares) and reached GAME_OVER, some 18xx actions
        # remain unconsumed.  Scan them for the latest reference snapshot so
        # the final comparison uses the true end-of-game state.
        final_ref = self._last_ref
        for remaining_idx in range(action_idx_in_stream, len(actions)):
            remaining_id = actions[remaining_idx].get('id', -1)
            if remaining_id >= 0 and remaining_id in ref_by_action:
                final_ref = ref_by_action[remaining_id]
        if final_ref is not None:
            self._compare_state(state, final_ref, "final")

        return self.mismatches

    def _override_deck_and_offering(self, state, deck_order_names, offering_names):
        """Override the deck and offering to match the 18xx game's initial state.

        After initialize_game(), the engine has a valid game state but with the
        wrong deck order and offering (from the random seed). We patch the state
        to match the 18xx reference by:
        1. Clearing stale auction/revealed company locations from the seed init
        2. Setting the correct deck order
        3. Drawing and auctioning the correct offering companies
        """
        # 1. Reset companies that init put into auction or revealed
        #    back to excluded (hidden-only, no visible flag leak)
        for cid in range(36):
            loc = COMPANIES[cid].get_location(state)
            if loc == LOC_AUCTION:
                state.set_company_for_auction(cid, False)
                COMPANIES[cid].exclude_from_game(state)
            elif loc == LOC_REVEALED:
                COMPANIES[cid].exclude_from_game(state)

        # 2. Build full deck (offering on top, remaining below) and set it.
        #    Ruby deck_order is top-to-bottom; our set_order is bottom-to-top.
        remaining_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(deck_order_names)]
        offering_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(offering_names)]
        full_deck = remaining_ids + offering_ids
        DECK.set_order(state, full_deck)

        # 3. Draw offering cards and move them to auction (same pattern as
        #    initialize_game: draw() marks revealed, move_to_auction() fixes it)
        for _ in range(len(offering_names)):
            cid = DECK.draw(state)
            COMPANIES[cid].move_to_auction(state)

        # CoO level is set correctly by draw() — if color-boundary cards
        # (e.g. MHE for red) are in the offering, CoO is bumped appropriately.

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
        """Replay a single action for simple phases (INVEST, BID, DIVIDENDS, ISSUE, IPO).

        Compares engine state BEFORE applying each action against the reference
        snapshot from the previous action. This avoids phase-alignment issues
        caused by the engine auto-advancing through forced phases.

        Returns the new index into the actions stream.
        """
        if idx >= len(actions):
            return idx

        action = actions[idx]
        action_id = action.get('id', -1)
        phase = TURN.get_phase(state)

        if (
            self._preserve_order_on_next_auto_invest_pass
            and action_id < 0
            and action.get('type') == 'pass'
            and phase == PHASE_INVEST
        ):
            saved_order = [PLAYERS[i].get_turn_order(state) for i in range(state.get_num_players())]
            self._preserve_order_on_next_auto_invest_pass = False

            next_idx = self._replay_simple_action_core(state, actions, idx, layout, ref_by_action)

            for player_id, position in enumerate(saved_order):
                PLAYERS[player_id].set_turn_order(state, position)

            return next_idx

        before_mismatches = len(self.mismatches)
        next_idx = self._replay_simple_action_core(state, actions, idx, layout, ref_by_action)

        if action_id >= 0 and action_id in ref_by_action:
            ref_round = ref_by_action[action_id].get('round', '')
            if (
                ref_round in ('ACQ', 'CLO')
                and phase == PHASE_INVEST
                and next_idx == idx + 1
                and self._last_ref is None
                and len(self.mismatches) == before_mismatches
            ):
                self._preserve_order_on_next_auto_invest_pass = True

        return next_idx

    def _replay_simple_action_core(self, state, actions, idx, layout, ref_by_action):
        """Replay a single non-ACQ/CLO action without placeholder-pass wrapping."""
        action = actions[idx]
        action_id = action.get('id', -1)
        has_ref = action_id >= 0 and action_id in ref_by_action

        # Skip forced dividend actions — the Ruby extractor tags dividends as
        # forced when corp cash < issued shares or corp is in receivership.
        # Our engine auto-applies these (only valid dividend is 0).
        if has_ref and ref_by_action[action_id].get('forced'):
            if self.verbose:
                print(f"  Skipping forced action {action_id} ({action.get('type')} for {action.get('entity')})")
            self._last_ref = ref_by_action[action_id]
            return idx + 1

        # Skip ACQ/CLO-round actions when the engine already advanced past
        # those phases (e.g. no corps active, or all CLO offers were for
        # non-negative-income companies that our engine doesn't offer).
        # Must check BEFORE comparison: our state includes INCOME effects
        # that the ACQ/CLO-round reference snapshots don't.
        if has_ref:
            ref_round = ref_by_action[action_id].get('round', '')
            phase = TURN.get_phase(state)
            if ref_round in ('ACQ', 'CLO') and phase not in (PHASE_ACQ, PHASE_CLOSING):
                if self.verbose:
                    print(f"  Skipping {ref_round}-round action {action_id} (engine already in {self._get_phase_name(state)})")
                self._last_ref = None
                return idx + 1

        # Auto-advance when the engine is stuck in a phase that doesn't
        # match the action's round.  This happens when undone actions in the
        # 18xx stream contained needed phase-transition passes (e.g. IPO
        # passes that were undone/redone and aren't in the committed set).
        if has_ref:
            ref_round = ref_by_action[action_id].get('round', '')
            phase = TURN.get_phase(state)
            if self._should_auto_advance(phase, ref_round):
                self._auto_advance_to_round(state, ref_round, layout, action_id)

        # Compare BEFORE applying: our state should match the last reference
        if has_ref and self._last_ref is not None:
            self._compare_state(state, self._last_ref, f"before action {action_id}")

        try:
            engine_action = self._map_action(state, action, TURN.get_phase(state), layout)
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

        # engine_action is either None, a single int, or a list of ints
        if engine_action is None:
            # Check if this unmappable action belongs to ACQ/CLO rounds that the
            # engine already auto-processed. If so, skip the ref update — using
            # an ACQ/CLO-round ref for comparison against post-INCOME state would
            # produce false mismatches.
            ref_round = ref_by_action.get(action_id, {}).get('round', '') if has_ref else ''
            if ref_round in ('ACQ', 'CLO'):
                if self.verbose:
                    print(f"  Skipping {ref_round}-round action {action_id} (engine already advanced)")
                self._last_ref = None
                return idx + 1
            if ref_round in ('DIV', 'ISS', 'IPO') and TURN.get_phase(state) in (PHASE_INVEST, PHASE_BID):
                if self.verbose:
                    print(f"  Skipping {ref_round}-round action {action_id} (engine already in {self._get_phase_name(state)})")
                self._last_ref = None
                return idx + 1
            if self.verbose:
                print(f"  Skipping unmappable action {action_id}: {action.get('type')}")
            if has_ref:
                self._last_ref = ref_by_action[action_id]
            return idx + 1

        # Normalize to list for uniform handling (IPO 'par' returns 2 actions)
        action_list = engine_action if isinstance(engine_action, list) else [engine_action]

        # Look-ahead: if the NEXT action in the stream is ACQ-round, this
        # action may trigger auto-advance through ACQ→CLO→INCOME.  Pre-apply
        # cross-president transfers and non-negative CLO closes NOW so
        # companies are in the correct ownership for INCOME calculations.
        # Uses acquisition_proceeds (flushed by the engine's ACQ merge step
        # which runs even when ACQ has no player offers).
        self._maybe_pre_apply_upcoming_acq(state, actions, idx, ref_by_action)

        for i, ea in enumerate(action_list):
            # For multi-action sequences (IPO corp + PAR price), the driver may
            # auto-apply the second action if only one par price is valid. Skip
            # remaining actions if the engine already advanced past the expected phase.
            if i > 0 and TURN.get_phase(state) != PHASE_PAR:
                break  # Driver auto-applied the forced PAR action
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

        # Look-ahead player-seller transfers defer the cash credit until the
        # triggering action has finished auto-advancing through the engine.
        self._apply_pending_player_cash(state)

        # Update reference for next comparison
        if has_ref:
            self._last_ref = ref_by_action[action_id]
        elif action_id < 0 and action.get('type') == 'pass':
            # Auto-action pass (from program_share_pass/program_close_pass):
            # the filtered program_* action's ref is not in the stream, so
            # _last_ref would be stale. Clear it to avoid false active_player
            # mismatches on the next real action.
            self._last_ref = None

        return idx + 1

    def _map_action(self, state, action, phase, layout):
        """Map a single 18xx action to our engine action index.

        Uses entity_type from the 18xx action to detect phase mismatches:
        - 'player': INVEST/BID actions
        - 'company': IPO actions
        - 'corporation': DIVIDENDS/ISSUE actions

        When the engine auto-advances past forced phases, the current phase
        may not match the action's intended phase. We skip actions that
        belong to a phase the engine has already processed.
        """
        atype = action.get('type', '')
        entity_type = action.get('entity_type', '')

        if phase == PHASE_INVEST:
            if atype in ('bid', 'buy_shares', 'sell_shares', 'pass'):
                if entity_type != 'player':
                    return None  # Not an INVEST action; engine auto-advanced past
                return map_invest_action(state, action, layout)
            return None

        if phase == PHASE_BID:
            if atype in ('bid', 'pass'):
                return map_bid_action(action, layout)
            return None

        if phase == PHASE_IPO:
            if atype in ('par', 'pass'):
                if entity_type != 'company':
                    return None  # Not an IPO action; engine auto-advanced past
                return map_ipo_action(action, layout)
            return None

        if phase == PHASE_DIVIDENDS:
            if atype == 'dividend':
                if entity_type != 'corporation':
                    return None  # Not a DIVIDENDS action
                return map_dividend_action(action, layout)
            return None

        if phase == PHASE_ISSUE:
            if atype in ('sell_shares', 'pass'):
                if entity_type != 'corporation':
                    return None  # Not an ISSUE action
                return map_issue_action(action, layout)
            return None

        # Automated phases — no player actions needed
        if phase in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_END_CARD):
            return None

        return None

    def _run_acquisition_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through acquisition phase, matching 18xx offer/respond actions
        against our engine's offer buffer.

        The 18xx action log records offers as made by the original proposer,
        but FI purchases may be preempted by higher-priority corps (especially
        OS which always has highest priority and pays face value). Our engine
        handles preemption implicitly by offering in priority order.

        We determine the actual outcome by diffing the reference state before
        and after the ACQ round, then match our engine's offers against that.

        Returns new index into actions stream (past all acquisition actions).
        """
        # Collect all 18xx acquisition actions until phase changes
        _, end_idx = self._collect_phase_actions(actions, idx, 'ACQ', ref_by_action)

        # Read ACQ outcomes from the Ruby extractor's annotation on the first
        # ACQ snapshot for this round. We may enter the round late (for
        # example on an auto-action pass), so scan backwards to recover the
        # round's first annotated snapshot instead of relying on start_idx.
        acq_outcomes: dict[str, tuple[str, int]] = {}
        acq_outcome_details: dict[str, dict] = {}  # Full outcome dicts for pre-apply fallback
        first_acq_ref, _, _ = self._get_first_round_snapshot(actions, idx, 'ACQ', ref_by_action)
        preapplied_companies = set()
        if first_acq_ref is not None:
            round_key = first_acq_ref['action_id']
            preapplied_companies = self._lookahead_preapplied_companies_by_round.get(round_key, set())
            outcomes = first_acq_ref.get('acq_outcomes') or []
            for outcome in outcomes:
                if outcome['company'] in preapplied_companies:
                    continue
                if not outcome.get('cross_president', False):
                    acq_outcomes[outcome['company']] = (outcome['buyer'], outcome['price'])
                    acq_outcome_details[outcome['company']] = outcome
            # Pre-apply cross-president transfers (our engine excludes these
            # from its action space — RULES.md constraint #1).
            for outcome in outcomes:
                if not outcome.get('cross_president', False):
                    continue
                if outcome['company'] in preapplied_companies:
                    continue
                self._pre_apply_acq_outcome(state, outcome, "ACQ adapter")

        # Pre-apply same-president corp-to-corp transfers.  Our engine's
        # fixed offer ordering (buyer share price DESC) can interact badly
        # with CLO pre-applies that reduce a corp's company count.  By
        # pre-applying these transfers first, the engine's buffer walk will
        # silently skip them (company already moved), and CLO pre-applies
        # below won't break the last-company validation.  Cash flows use
        # acquisition_proceeds which the engine finalizes at ACQ exit.
        for company_name in list(acq_outcomes.keys()):
            o = acq_outcome_details.get(company_name)
            if o is None or o.get('seller_type') != 'corp':
                continue
            company_id = COMPANY_NAME_TO_ID.get(company_name)
            buyer_corp_id = CORP_NAME_TO_ID.get(o['buyer'])
            seller_corp_id = CORP_NAME_TO_ID.get(o.get('seller', ''))
            if company_id is None or buyer_corp_id is None or seller_corp_id is None:
                continue
            price = o['price']
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            current = CORPS[seller_corp_id].get_acquisition_proceeds(state)
            CORPS[seller_corp_id].set_acquisition_proceeds(state, current + price)
            del acq_outcomes[company_name]
            if self.verbose:
                print(f"  ACQ adapter: pre-applied corp-corp transfer "
                      f"{o['seller']}->{o['buyer']} for {company_name} at {price}")

        if self.verbose and acq_outcomes:
            print(f"  ACQ adapter: outcomes={acq_outcomes}")

        # Pre-apply CLO closings for non-negative-income companies.
        # The driver auto-advances through CLOSING and INCOME after ACQ when
        # CLOSING has no player choices (no negative-income companies).  If the
        # CLO round contains sell_company actions for zero/positive-income
        # companies, the CLO adapter would never get a chance to pre-apply them.
        # We must apply them NOW, before any ACQ action triggers auto-advance.
        # The Ruby extractor tags sell_company snapshots with adjusted_income.
        for i in range(idx, len(actions)):
            action = actions[i]
            action_id = action.get('id', -1)
            if action_id in ref_by_action:
                ref = ref_by_action[action_id]
                round_name = ref.get('round', '')
                if round_name == 'CLO' and action.get('type') == 'sell_company':
                    if ref.get('adjusted_income', -1) >= 0:
                        company_name = action.get('company', '')
                        company_id = COMPANY_NAME_TO_ID.get(company_name)
                        if company_id is not None:
                            self._pre_apply_close(state, company_id, company_name, "ACQ adapter")
                elif round_name not in ('ACQ', 'CLO'):
                    break

        # Walk our engine's offer buffer
        max_iterations = 200
        iterations = 0
        while TURN.get_phase(state) == PHASE_ACQ and iterations < max_iterations:
            iterations += 1

            # Check if there's a forced action (auto-pass for receivership, etc.)
            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                break
            if legal_count == 1:
                # Forced action — apply it
                for i, v in enumerate(mask):
                    if v > 0.5:
                        DRIVER.apply_action(state, i)
                        break
                continue

            # We have a choice — check if this offer matches an outcome
            acq_corp_id = TURN.get_acq_active_corp(state)
            acq_company_id = TURN.get_acq_target_company(state)

            if acq_corp_id < 0 or acq_company_id < 0:
                DRIVER.apply_action(state, layout.acq_pass)
                continue

            corp_name = CORP_NAMES[acq_corp_id]
            company_name = COMPANY_NAMES[acq_company_id]
            is_fi = TURN.is_acq_fi_offer(state)

            # Check if this company was acquired by this corp in the reference
            if company_name in acq_outcomes and acq_outcomes[company_name][0] == corp_name:
                if self.verbose:
                    print(f"  ACQ offer: {corp_name} -> {company_name} [ACCEPT]")

                if is_fi:
                    # FI offer: single fi_buy action (OS pays face, others pay high)
                    engine_action = layout.acq_fi_buy
                else:
                    # Non-FI: use the recorded price
                    price = acq_outcomes[company_name][1]
                    low_price = get_company_low_price(acq_company_id)
                    price_offset = price - low_price
                    engine_action = layout.acq_price_base + price_offset

                result = DRIVER.apply_action(state, engine_action)
                if result == STATUS_INVALID:
                    if self.verbose:
                        print(f"  ACQ: Invalid action for {corp_name}->{company_name}, passing")
                    DRIVER.apply_action(state, layout.acq_pass)
                del acq_outcomes[company_name]
            else:
                if self.verbose:
                    print(f"  ACQ offer: {corp_name} -> {company_name} [PASS]")
                DRIVER.apply_action(state, layout.acq_pass)

        # Clear _last_ref: our engine auto-advanced through CLOSING+INCOME
        # after ACQ, but the Ruby ref for the last ACQ action doesn't include
        # those automated phases. Skip comparison at the next action boundary.
        self._last_ref = None

        return end_idx

    def _run_closing_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through closing phase, matching 18xx sell_company actions
        against our engine's offer buffer.

        Returns new index into actions stream.
        """
        # Collect closing actions — scan past any ACQ-round actions that
        # precede the CLO-round actions in the stream (the 18xx action stream
        # has ACQ actions first, then CLO actions, but our engine may have
        # already auto-advanced past ACQ).
        close_actions = []
        end_idx = idx
        for i in range(idx, len(actions)):
            action = actions[i]
            action_id = action.get('id', -1)
            if action_id in ref_by_action:
                ref = ref_by_action[action_id]
                round_name = ref.get('round', '')
                if round_name == 'CLO':
                    close_actions.append(action)
                elif round_name != 'ACQ':
                    # Hit a non-ACQ/non-CLO round — done collecting
                    end_idx = i
                    break
            end_idx = i + 1

        # Build close set: company names that were closed
        closed_companies = set()
        for a in close_actions:
            if a.get('type') == 'sell_company':
                closed_companies.add(a.get('company', ''))

        # Pre-apply closings for positive-income companies that our engine
        # won't offer (intentional scope constraint — see RULES.md CLO-14).
        # The Ruby extractor tags sell_company snapshots with adjusted_income.
        for a in close_actions:
            if a.get('type') != 'sell_company':
                continue
            action_id = a.get('id', -1)
            if action_id < 0 or action_id not in ref_by_action:
                continue
            ref = ref_by_action[action_id]
            if ref.get('adjusted_income', -1) >= 0:
                company_name = a.get('company', '')
                company_id = COMPANY_NAME_TO_ID.get(company_name)
                if company_id is not None:
                    self._pre_apply_close(state, company_id, company_name, "CLO adapter")
                    closed_companies.discard(company_name)

        # Walk our engine's offer buffer for remaining (negative-income) closings
        max_iterations = 200
        iterations = 0
        while TURN.get_phase(state) == PHASE_CLOSING and iterations < max_iterations:
            iterations += 1

            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                break
            if legal_count == 1:
                # Forced action
                for i, v in enumerate(mask):
                    if v > 0.5:
                        DRIVER.apply_action(state, i)
                        break
                continue

            # We have a choice: close or pass
            closing_company_id = TURN.get_closing_company(state)
            if closing_company_id < 0:
                DRIVER.apply_action(state, layout.close_pass)
                continue

            company_name = COMPANY_NAMES[closing_company_id]
            if company_name in closed_companies:
                result = DRIVER.apply_action(state, layout.close_action)
                if result == STATUS_INVALID:
                    DRIVER.apply_action(state, layout.close_pass)
                else:
                    closed_companies.discard(company_name)
            else:
                DRIVER.apply_action(state, layout.close_pass)

        # Clear _last_ref: our engine auto-advanced through INCOME after
        # CLOSING, but the Ruby ref doesn't include those automated phases.
        self._last_ref = None

        return end_idx

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
        """Check if the engine needs to auto-advance to reach the action's round.

        Returns True when the engine is stuck in a phase that doesn't match
        the action's expected round, and auto-advancing by applying passes
        would be appropriate (e.g. IPO passes when engine is in IPO but
        action is INV-round).
        """
        if not ref_round:
            return False
        expected_phases = self._ROUND_TO_PHASES.get(ref_round)
        if expected_phases is None:
            return False
        if phase in expected_phases:
            return False
        # Only auto-advance from phases where applying pass is safe
        if phase in (PHASE_IPO, PHASE_PAR, PHASE_ISSUE, PHASE_DIVIDENDS):
            return True
        return False

    def _auto_advance_to_round(self, state, ref_round: str, layout, action_id: int):
        """Auto-advance the engine by applying passes until it reaches the
        phase matching ref_round.

        Used when undone actions in the 18xx stream contained needed phase-
        transition passes that aren't in the committed action set.
        """
        expected_phases = self._ROUND_TO_PHASES.get(ref_round, set())
        max_iterations = 100
        for _ in range(max_iterations):
            phase = TURN.get_phase(state)
            if phase in expected_phases or phase == PHASE_GAME_OVER:
                break

            # Find the pass action for the current phase
            mask = get_valid_action_mask(state)
            legal_count = sum(1 for v in mask if v > 0.5)

            if legal_count == 0:
                break
            if legal_count == 1:
                # Single legal action — apply it (forced)
                for i, v in enumerate(mask):
                    if v > 0.5:
                        if self.verbose:
                            print(f"  Auto-advance: forced action in {self._get_phase_name(state)}")
                        DRIVER.apply_action(state, i)
                        break
            else:
                # Multiple legal actions — apply pass
                pass_action = self._find_pass_action(phase, layout)
                if pass_action is not None:
                    if self.verbose:
                        print(f"  Auto-advance: pass in {self._get_phase_name(state)} "
                              f"(targeting {ref_round}, aid={action_id})")
                    DRIVER.apply_action(state, pass_action)
                else:
                    break  # Can't auto-advance

        # Clear _last_ref since we auto-advanced through phases
        self._last_ref = None

    @staticmethod
    def _find_pass_action(phase: int, layout) -> int | None:
        """Return the pass action index for the given phase, or None."""
        if phase == PHASE_IPO:
            return layout.ipo_pass
        if phase == PHASE_ISSUE:
            return layout.issue_pass
        if phase == PHASE_DIVIDENDS:
            return layout.dividend_base  # dividend 0
        return None

    def _pre_apply_close(self, state, company_id: int, company_name: str, context: str):
        """Pre-apply a company close that our engine won't offer.

        Handles JS scrapping bonus (corp_id 0 = Junkyard Scrappers).
        Used when the 18xx game closes a non-negative-income company
        that our engine's closing scope constraint excludes.
        """
        for corp_id in range(8):
            if CORPS[corp_id].is_active(state) and COMPANIES[company_id].is_owned_by_corp(state, corp_id):
                if corp_id == 0:  # JS = Junkyard Scrappers
                    CORPS[corp_id].add_cash(state, get_company_income(company_id) * 2)
                break
        COMPANIES[company_id].remove_from_game(state)
        if self.verbose:
            print(f"  {context}: pre-applied non-negative-income close for {company_name}")

    def _find_round_bounds(self, actions, anchor_idx, round_name: str, ref_by_action):
        """Return the contiguous action-index window for a round around anchor_idx."""
        start = anchor_idx
        while start > 0:
            prev_id = actions[start - 1].get('id', -1)
            prev_ref = ref_by_action.get(prev_id)
            if prev_ref is not None and prev_ref.get('round', '') != round_name:
                break
            start -= 1

        end = anchor_idx
        while end < len(actions):
            action_id = actions[end].get('id', -1)
            ref = ref_by_action.get(action_id)
            if ref is not None and ref.get('round', '') != round_name:
                break
            end += 1

        return start, end

    def _get_first_round_snapshot(self, actions, anchor_idx, round_name: str, ref_by_action):
        """Return the first reference snapshot for the nearest round block.

        The anchor may already be inside the target round, immediately before
        it (look-ahead), or after it (late ACQ/CLO entry on a later auto-pass).
        Prefer the nearest matching round at or before the anchor, then fall
        back to the nearest one after it.
        """
        candidate_idx = None

        for i in range(anchor_idx, -1, -1):
            action_id = actions[i].get('id', -1)
            ref = ref_by_action.get(action_id)
            if ref is not None and ref.get('round', '') == round_name:
                candidate_idx = i
                break

        if candidate_idx is None:
            for i in range(anchor_idx + 1, len(actions)):
                action_id = actions[i].get('id', -1)
                ref = ref_by_action.get(action_id)
                if ref is not None and ref.get('round', '') == round_name:
                    candidate_idx = i
                    break

        if candidate_idx is None:
            return None, anchor_idx, anchor_idx

        start, end = self._find_round_bounds(actions, candidate_idx, round_name, ref_by_action)
        for i in range(start, end):
            action_id = actions[i].get('id', -1)
            ref = ref_by_action.get(action_id)
            if ref is not None and ref.get('round', '') == round_name:
                return ref, start, end
        return None, start, end

    def _get_round_preapplied_companies(self, round_key: int) -> set[str]:
        """Get or create the look-ahead pre-applied company set for a round."""
        return self._lookahead_preapplied_companies_by_round.setdefault(round_key, set())

    def _pre_apply_acq_outcome(
        self,
        state,
        outcome: dict,
        context: str,
        *,
        round_key: int | None = None,
        defer_player_cash: bool = False,
    ) -> bool:
        """Pre-apply an ACQ outcome directly to state."""
        company_name = outcome.get('company', '')
        preapplied_companies = None
        if round_key is not None:
            preapplied_companies = self._get_round_preapplied_companies(round_key)
            if company_name in preapplied_companies:
                return False

        company_id = COMPANY_NAME_TO_ID.get(company_name)
        buyer_corp_id = CORP_NAME_TO_ID.get(outcome.get('buyer', ''))
        if company_id is None or buyer_corp_id is None:
            return False

        price = outcome.get('price', 0)
        seller_type = outcome.get('seller_type')

        if seller_type == 'corp':
            seller_corp_id = CORP_NAME_TO_ID.get(outcome.get('seller', ''))
            if seller_corp_id is None:
                return False
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            current = CORPS[seller_corp_id].get_acquisition_proceeds(state)
            CORPS[seller_corp_id].set_acquisition_proceeds(state, current + price)
            seller_label = outcome.get('seller', '?')
        elif seller_type == 'player':
            seller_idx = self._player_id_to_index.get(outcome.get('seller_id'))
            if seller_idx is None:
                return False
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            if defer_player_cash:
                current = self._pending_player_cash_by_player.get(seller_idx, 0)
                self._pending_player_cash_by_player[seller_idx] = current + price
            else:
                PLAYERS[seller_idx].add_cash(state, price)
            seller_label = f"player[{seller_idx}]"
        elif seller_type == 'fi':
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            FI.add_cash(state, price)
            seller_label = "FI"
        else:
            return False

        if preapplied_companies is not None:
            preapplied_companies.add(company_name)

        if self.verbose:
            print(
                f"  {context}: pre-applied {seller_type} transfer "
                f"{seller_label}->{outcome.get('buyer', '?')} for {company_name} at {price}"
            )
        return True

    def _maybe_pre_apply_upcoming_acq(self, state, actions, idx, ref_by_action):
        """Look ahead for an upcoming ACQ round and pre-apply transfers.

        When the engine is about to auto-advance through ACQ (e.g. the last
        INVEST pass triggers WRAP_UP→ACQ→CLO→INCOME), acquisition transfers
        and non-negative-income CLO closes must be applied BEFORE the cascade
        so INCOME sees correct company ownership.

        Only fires when the next action in the stream is ACQ-round and
        hasn't been pre-applied yet.  Uses acquisition_proceeds (not direct
        cash) because the engine's ACQ merge step always runs, even when
        ACQ has no player offers.
        """
        if idx + 1 >= len(actions):
            return
        next_action = actions[idx + 1]
        next_id = next_action.get('id', -1)
        if next_id < 0 or next_id not in ref_by_action:
            return
        next_round = ref_by_action[next_id].get('round', '')
        if next_round not in ('ACQ', 'CLO'):
            return
        # Don't re-apply if we already handled this round
        if next_id in self._lookahead_handled_round_actions:
            return

        # Mark all ACQ+CLO action IDs in this round as handled
        for i in range(idx + 1, len(actions)):
            a = actions[i]
            aid = a.get('id', -1)
            if aid >= 0 and aid in ref_by_action:
                r = ref_by_action[aid].get('round', '')
                if r in ('ACQ', 'CLO'):
                    self._lookahead_handled_round_actions.add(aid)
                else:
                    break

        # Find acq_outcomes from the first ACQ snapshot and pre-apply only the
        # transfers our engine cannot represent in its ACQ action space.
        #
        # Same-president offers must still be left for the ACQ adapter when the
        # engine actually enters ACQ. Pre-applying them here can empty the
        # offer buffer, causing the driver to blow past ACQ/CLO and skip the
        # real DIV/ISS/IPO rounds entirely.
        first_acq_ref, _, _ = self._get_first_round_snapshot(actions, idx + 1, 'ACQ', ref_by_action)
        if first_acq_ref is not None:
            round_key = first_acq_ref['action_id']
            for outcome in first_acq_ref.get('acq_outcomes') or []:
                if not outcome.get('cross_president', False):
                    continue
                self._pre_apply_acq_outcome(
                    state,
                    outcome,
                    "Look-ahead",
                    round_key=round_key,
                    defer_player_cash=True,
                )

        # Pre-apply CLO non-negative-income closes
        for i in range(idx + 1, len(actions)):
            a = actions[i]
            aid = a.get('id', -1)
            if aid < 0 or aid not in ref_by_action:
                continue
            ref = ref_by_action[aid]
            round_name = ref.get('round', '')
            if round_name == 'CLO' and a.get('type') == 'sell_company':
                if ref.get('adjusted_income', -1) >= 0:
                    company_name = a.get('company', '')
                    company_id = COMPANY_NAME_TO_ID.get(company_name)
                    if company_id is not None:
                        self._pre_apply_close(state, company_id, company_name, "Look-ahead CLO")
            elif round_name not in ('ACQ', 'CLO'):
                break

    def _collect_phase_actions(self, actions, start_idx, round_name, ref_by_action):
        """Collect all actions belonging to the same 18xx round.

        Scans forward from start_idx, collecting actions until we find one
        whose reference snapshot shows a different round.

        Returns (collected_actions, end_index).
        """
        collected = []
        idx = start_idx
        while idx < len(actions):
            action = actions[idx]
            action_id = action.get('id', -1)

            # Check if this action's reference snapshot is in a different round
            if action_id in ref_by_action:
                ref = ref_by_action[action_id]
                if ref['round'] != round_name:
                    break

            collected.append(action)
            idx += 1

        # If we collected nothing but there are actions left, advance by 1
        if not collected and idx < len(actions):
            idx += 1

        return collected, idx

    def _compare_state(self, state, ref: dict, context: str):
        """Compare our engine state against a reference snapshot."""
        action_id = ref.get('action_id', -1)
        phase_name = self._get_phase_name(state)
        phase = TURN.get_phase(state)

        # Compare active player / active corp when phases are aligned.
        # The ref snapshot may be from a different phase than our engine
        # (e.g. ref is IPO but engine auto-advanced to INVEST), so only
        # compare when the ref round corresponds to our engine's phase.
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
            # After an IPO par, the acting player may no longer afford to par
            # their remaining companies.  Our driver auto-applies those forced
            # passes within a single apply_action call, advancing past them.
            # The 18xx engine records each pass as a separate action (often
            # visible as gaps in action IDs, e.g. 429 → 431).  The reference
            # snapshot at the par still shows the *next* company's owner,
            # while our engine has already advanced further.
            ref_action_type = ref.get('action_type', '')
            skip_active_player = (
                (phase == PHASE_IPO and ref_action_type == 'par')
                or ref_action_type == 'end_game'  # Platform-level game end (concession)
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

            # Find our player index
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

        # Compare offering — include both available (LOC_AUCTION) and
        # unavailable/revealed (LOC_REVEALED) companies.  The 18xx reference
        # counts both in its "offering" (available + drawn-but-vertical).
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
        """Find our 0-based player index from an 18xx player ID.

        Uses the static player_order mapping established at game start.
        The 18xx reference rotates players between rounds, but our engine
        uses fixed indices throughout.
        """
        idx = self._player_id_to_index.get(player_id_18xx)
        if idx is not None:
            return idx
        raise ValueError(f"Player {player_id_18xx} not found in player_order")

    def _apply_pending_player_cash(self, state):
        """Apply deferred player-cash credits from look-ahead ACQ transfers."""
        if not self._pending_player_cash_by_player:
            return
        pending = self._pending_player_cash_by_player
        self._pending_player_cash_by_player = {}
        for player_id, amount in pending.items():
            PLAYERS[player_id].add_cash(state, amount)
        if self.verbose:
            print(f"  Applied deferred player cash: {pending}")


def format_mismatches(mismatches: list[Mismatch]) -> str:
    """Format mismatches for display."""
    lines = []
    for m in mismatches[:50]:  # Cap at 50 to avoid overwhelming output
        lines.append(str(m))
    if len(mismatches) > 50:
        lines.append(f"... and {len(mismatches) - 50} more")
    return "\n".join(lines)
