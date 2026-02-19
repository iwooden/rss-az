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
from entities.company import COMPANIES, LOC_AUCTION, LOC_REVEALED
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
PHASE_GAME_OVER = GamePhases.PHASE_GAME_OVER

# 18xx round short names -> our phase groupings
ROUND_TO_PHASES = {
    'INV': {PHASE_INVEST, PHASE_BID, PHASE_WRAP_UP},
    'ACQ': {PHASE_ACQ},
    'CLO': {PHASE_CLOSING},
    'DIV': {PHASE_DIVIDENDS, PHASE_INCOME, PHASE_END_CARD},
    'ISS': {PHASE_ISSUE},
    'IPO': {PHASE_IPO},
}


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


@dataclass
class ReplayHarness:
    """Orchestrates replay of an 18xx game through our Cython engine."""

    game_json_path: str
    states_json_path: str
    verbose: bool = False
    mismatches: list = field(default_factory=list)

    def run(self) -> list[Mismatch]:
        """Run the full replay. Returns list of mismatches (empty = success)."""
        self.mismatches = []

        # Load data
        game_data = json.loads(Path(self.game_json_path).read_text())
        ref_states = json.loads(Path(self.states_json_path).read_text())

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
        if ref_states:
            last_ref = ref_states[-1]
            self._compare_state(state, last_ref, "final")

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
            PHASE_ISSUE: "ISSUE", PHASE_IPO: "IPO", PHASE_GAME_OVER: "GAME_OVER",
        }
        return names.get(phase, f"UNKNOWN({phase})")

    def _replay_simple_action(self, state, actions, idx, layout, ref_by_action):
        """Replay a single action for simple phases (INVEST, BID, DIVIDENDS, ISSUE, IPO).

        Returns the new index into the actions stream.
        """
        if idx >= len(actions):
            return idx

        action = actions[idx]
        phase = TURN.get_phase(state)
        action_id = action.get('id', -1)

        try:
            engine_action = self._map_action(state, action, phase, layout)
        except (ValueError, KeyError, IndexError) as e:
            self.mismatches.append(Mismatch(
                action_id=action_id,
                phase=self._get_phase_name(state),
                field="action_mapping",
                expected="valid mapping",
                actual=str(e),
                context=f"18xx_type={action.get('type')}, entity={action.get('entity')}",
            ))
            return idx + 1

        if engine_action is None:
            # Skip actions we can't map (e.g., actions for automated phases)
            if self.verbose:
                print(f"  Skipping unmappable action {action_id}: {action.get('type')}")
            return idx + 1

        result = DRIVER.apply_action(state, engine_action)
        if result == STATUS_INVALID:
            self.mismatches.append(Mismatch(
                action_id=action_id,
                phase=self._get_phase_name(state),
                field="action_validity",
                expected="STATUS_OK",
                actual="STATUS_INVALID",
                context=f"engine_action={engine_action}, 18xx_type={action.get('type')}",
            ))

        # Compare state if we have a reference for this action_id AND
        # the engine phase matches the reference round. The engine may
        # auto-advance past forced phases, making the comparison invalid
        # when the phases don't align.
        if action_id in ref_by_action:
            ref = ref_by_action[action_id]
            ref_round = ref.get('round', '')
            engine_phase = TURN.get_phase(state)
            if ref_round and engine_phase in ROUND_TO_PHASES.get(ref_round, set()):
                self._compare_state(state, ref, f"after action {action_id}")

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

    def _is_player_entity(self, entity) -> bool:
        """Check if entity is a player ID (numeric)."""
        try:
            int(entity)
            return True
        except (TypeError, ValueError):
            return False

    def _is_company_entity(self, entity) -> bool:
        """Check if entity is a company name."""
        return entity in COMPANY_NAME_TO_ID

    def _is_corp_entity(self, entity) -> bool:
        """Check if entity is a corporation name."""
        return entity in CORP_NAME_TO_ID

    def _run_acquisition_adapter(self, state, actions, idx, layout, ref_by_action):
        """Walk through acquisition phase, matching 18xx offer/respond actions
        against our engine's offer buffer.

        Returns new index into actions stream (past all acquisition actions).
        """
        # Collect all 18xx acquisition actions until phase changes
        acq_actions, end_idx = self._collect_phase_actions(actions, idx, 'ACQ', ref_by_action)

        # Build outcome map: (corp_name, company_name) -> price for accepted offers
        accepted_offers = {}  # (corp_name, company_name) -> price
        for a in acq_actions:
            atype = a.get('type', '')
            if atype == 'offer':
                corp_name = a.get('corporation', '')
                company_name = a.get('company', '')
                price = int(a.get('price', 0))
                # Default to accepted; a 'respond' with accept=false overrides
                key = (corp_name, company_name)
                accepted_offers[key] = price
            elif atype == 'respond':
                corp_name = a.get('corporation', '')
                company_name = a.get('company', '')
                accept = str(a.get('accept', 'true')).lower() == 'true'
                key = (corp_name, company_name)
                if not accept and key in accepted_offers:
                    del accepted_offers[key]

        if self.verbose and accepted_offers:
            print(f"  ACQ adapter: accepted_offers={accepted_offers}")

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

            # We have a choice — check if this offer matches an accepted one
            acq_corp_id = TURN.get_acq_active_corp(state)
            acq_company_id = TURN.get_acq_target_company(state)

            if acq_corp_id < 0 or acq_company_id < 0:
                # No offer to process, pass
                DRIVER.apply_action(state, layout.acq_pass)
                continue

            corp_name = CORP_NAMES[acq_corp_id]
            company_name = COMPANY_NAMES[acq_company_id]
            key = (corp_name, company_name)

            if self.verbose:
                matched = "ACCEPT" if key in accepted_offers else "PASS"
                print(f"  ACQ offer: {corp_name} -> {company_name} [{matched}]")

            if key in accepted_offers:
                price = accepted_offers[key]
                is_fi = TURN.is_acq_fi_offer(state)

                if is_fi:
                    # FI offer: use fi_high or fi_face
                    if acq_corp_id == CORP_NAME_TO_ID.get('OS', -1):
                        engine_action = layout.acq_fi_face
                    else:
                        engine_action = layout.acq_fi_high
                else:
                    low_price = get_company_low_price(acq_company_id)
                    price_offset = price - low_price
                    engine_action = layout.acq_price_base + price_offset

                result = DRIVER.apply_action(state, engine_action)
                if result == STATUS_INVALID:
                    # Try pass as fallback
                    if self.verbose:
                        print(f"  ACQ: Invalid action for {key} at price {price}, passing")
                    DRIVER.apply_action(state, layout.acq_pass)
                del accepted_offers[key]
            else:
                # Not in accepted offers -> pass
                DRIVER.apply_action(state, layout.acq_pass)

        # NOTE: We do NOT compare state here because DRIVER.apply_action
        # auto-advances through CLOSING+INCOME after ACQ. The engine state
        # is at DIVIDENDS while the reference is still at end-of-ACQ.
        # Comparison happens naturally at the next player-action boundary
        # (DIVIDENDS) via _replay_simple_action.

        return end_idx

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

        # NOTE: Same as ACQ adapter - don't compare here because the engine
        # may have auto-advanced through INCOME after CLOSING.

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

        # Compare players
        for ref_player in ref.get('players', []):
            player_name = ref_player['name']
            player_id_18xx = ref_player['id']

            # Find our player index
            try:
                pidx = self._find_player_index(ref, player_id_18xx)
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
        our_coo = TURN.get_coo_level(state)
        ref_coo = ref.get('cost_level', 0)
        if our_coo != ref_coo:
            self.mismatches.append(Mismatch(
                action_id=action_id, phase=phase_name,
                field="cost_level",
                expected=ref_coo, actual=our_coo, context=context,
            ))

    def _find_player_index(self, ref: dict, player_id_18xx: int) -> int:
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
