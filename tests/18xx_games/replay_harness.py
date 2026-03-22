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
    GamePhases, get_company_low_price,
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


def extract_ref_states(game_json_path: str) -> list[dict]:
    """Run the Ruby state extractor and return parsed reference states.

    Executes extract_states.rb as a subprocess, capturing its JSON output
    directly into memory without writing to disk.

    Args:
        game_json_path: Path to the 18xx game JSON file.

    Returns:
        List of reference state snapshot dicts.

    Raises:
        RuntimeError: If ruby is not available or the extractor fails.
    """
    import subprocess

    extractor = str(Path(__file__).parent / "extract_states.rb")
    try:
        result = subprocess.run(
            ["ruby", extractor, game_json_path],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Ruby not found. Install Ruby to run 18xx replay tests."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"State extractor failed (exit {result.returncode}):\n{result.stderr}"
        )

    return json.loads(result.stdout)


@dataclass
class ReplayHarness:
    """Orchestrates replay of an 18xx game through our Cython engine."""

    game_json_path: str
    ref_states: list = field(default_factory=list)
    verbose: bool = False
    mismatches: list = field(default_factory=list)

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

        # Filter and flatten
        actions = filter_actions(raw_actions)
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

        # Final comparison at game end
        if self._last_ref is not None:
            self._compare_state(state, self._last_ref, "final")

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

        # 4. Restore CoO level to 1 (draw may have bumped it if a color-boundary
        #    card happened to be in the offering)
        TURN.set_coo_level(state, 1)

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
        has_ref = action_id >= 0 and action_id in ref_by_action

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
            if self.verbose:
                print(f"  Skipping unmappable action {action_id}: {action.get('type')}")
            if has_ref:
                self._last_ref = ref_by_action[action_id]
            return idx + 1

        # Normalize to list for uniform handling (IPO 'par' returns 2 actions)
        action_list = engine_action if isinstance(engine_action, list) else [engine_action]

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

        # Update reference for next comparison
        if has_ref:
            self._last_ref = ref_by_action[action_id]

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
        acq_actions, end_idx = self._collect_phase_actions(actions, idx, 'ACQ', ref_by_action)

        # Build actual outcomes from reference state diff.
        # Find the first ACQ ref and the first post-ACQ ref to diff.
        acq_outcomes = self._build_acq_outcomes(acq_actions, ref_by_action)

        if self.verbose and acq_outcomes:
            print(f"  ACQ adapter: outcomes={acq_outcomes}")

        # Pre-apply cross-president corp-to-corp transfers that our engine
        # intentionally excludes from its action space (RULES.md constraint #1).
        # Must happen BEFORE the while loop because the driver auto-advances
        # through CLOSING and INCOME after ACQ — patching after would be too late.
        for company_name, (buyer_name, price) in list(acq_outcomes.items()):
            company_id = COMPANY_NAME_TO_ID.get(company_name)
            buyer_corp_id = CORP_NAME_TO_ID.get(buyer_name)
            if company_id is None or buyer_corp_id is None:
                continue

            # Check if company is owned by a corp with a different president
            seller_corp_id = None
            for cid in range(8):
                if CORPS[cid].is_active(state) and COMPANIES[company_id].is_owned_by_corp(state, cid):
                    seller_corp_id = cid
                    break

            if seller_corp_id is None:
                continue  # Not a corp-to-corp transfer
            if CORPS[seller_corp_id].get_president_id(state) == CORPS[buyer_corp_id].get_president_id(state):
                continue  # Same president — engine handles this

            # Direct state patch: transfer company and cash
            COMPANIES[company_id].transfer_to_corp(state, buyer_corp_id)
            CORPS[buyer_corp_id].add_cash(state, -price)
            CORPS[seller_corp_id].add_cash(state, price)
            del acq_outcomes[company_name]
            if self.verbose:
                print(f"  ACQ adapter: pre-applied cross-president transfer "
                      f"{CORP_NAMES[seller_corp_id]}->{buyer_name} "
                      f"for {company_name} at price {price}")

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

    def _build_acq_outcomes(self, acq_actions, ref_by_action):
        """Build actual acquisition outcomes from reference state diffs.

        Compares corp company ownership BEFORE the ACQ round vs AFTER to
        determine which companies moved and to whom. The "before" snapshot
        must be from a non-ACQ round (typically INV) because the 18xx ref
        includes automated phase results (WRAP_UP, FI buying from offering)
        in the first ACQ-round snapshot.

        For non-FI purchases, gets the price from the raw action data.

        Returns dict: company_name -> (acquiring_corp_name, price)
        """
        # Find the first and last ACQ reference snapshots
        first_acq_aid = None
        last_acq_aid = None
        for a in acq_actions:
            aid = a.get('id', -1)
            if aid >= 0 and aid in ref_by_action:
                ref = ref_by_action[aid]
                if ref.get('round') == 'ACQ':
                    if first_acq_aid is None:
                        first_acq_aid = aid
                    last_acq_aid = aid

        if first_acq_aid is None:
            return {}

        # Find the last NON-ACQ reference state before the first ACQ action.
        # We look back from first_acq_aid to find a ref with round != 'ACQ'.
        # This captures changes from WRAP_UP (FI buying from offering) that
        # the 18xx ref folds into the first ACQ snapshot.
        before_ref = None
        for aid_candidate in range(first_acq_aid - 1, -1, -1):
            if aid_candidate in ref_by_action:
                ref = ref_by_action[aid_candidate]
                if ref.get('round') != 'ACQ':
                    before_ref = ref
                    break

        after_ref = ref_by_action.get(last_acq_aid)
        if before_ref is None or after_ref is None:
            return {}

        # Build company -> corp_owner maps for before and after
        def get_corp_companies(ref):
            result = {}
            for corp in ref.get('corporations', []):
                for comp in corp.get('companies', []):
                    result[comp] = corp['name']
            return result

        before_corps = get_corp_companies(before_ref)
        after_corps = get_corp_companies(after_ref)

        # Build price map from raw actions: company_name -> price
        # Use the last offer price for each company (handles re-offers)
        action_prices = {}
        for a in acq_actions:
            if a.get('type') == 'offer':
                company_name = a.get('company', '')
                price = int(a.get('price', 0))
                action_prices[company_name] = price

        # Find companies that moved TO a corp (new acquisitions)
        outcomes = {}
        for company_name, new_owner in after_corps.items():
            old_owner = before_corps.get(company_name)
            if old_owner != new_owner:
                # This company was acquired by new_owner
                price = action_prices.get(company_name, 0)
                outcomes[company_name] = (new_owner, price)

        return outcomes

    def _run_closing_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through closing phase, matching 18xx sell_company actions
        against our engine's offer buffer.

        Returns new index into actions stream.
        """
        # Collect closing actions
        close_actions, end_idx = self._collect_phase_actions(actions, idx, 'CLO', ref_by_action)

        # Build close set: company names that were closed
        closed_companies = set()
        for a in close_actions:
            if a.get('type') == 'sell_company':
                closed_companies.add(a.get('company', ''))

        # Walk our engine's offer buffer
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
            skip_active_player = (phase == PHASE_IPO and ref_action_type == 'par')
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

        # Compare cost level
        # The Ruby Stars engine uses levels 1-5, 7, 8 (skipping 6):
        #   Ruby 7 = END_CARD_FRONT (deck empty, 7● side)
        #   Ruby 8 = END_CARD_BACK (end card flipped, 10● side)
        # Our engine uses contiguous 1-7:
        #   Our 6 = game end card (7● side)
        #   Our 7 = game end card flipped (10● side)
        our_coo = TURN.get_coo_level(state)
        ref_coo = ref.get('cost_level', 0)
        if ref_coo == 7:
            ref_coo = 6  # Ruby END_CARD_FRONT → our level 6
        elif ref_coo == 8:
            ref_coo = 7  # Ruby END_CARD_BACK → our level 7
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


def format_mismatches(mismatches: list[Mismatch]) -> str:
    """Format mismatches for display."""
    lines = []
    for m in mismatches[:50]:  # Cap at 50 to avoid overwhelming output
        lines.append(str(m))
    if len(mismatches) > 50:
        lines.append(f"... and {len(mismatches) - 50} more")
    return "\n".join(lines)
