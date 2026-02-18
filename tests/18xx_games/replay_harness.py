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
from entities.company import COMPANIES, LOC_AUCTION
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

        # Get initial record (action_id=0) for deck order
        initial = ref_by_action[0]
        deck_order_names = initial['deck_order']
        offering_names = initial['initial_offering']

        # Initialize our engine and override deck to match 18xx game
        state = GameState(num_players)
        state.initialize_game(seed=42)  # seed doesn't matter, we override deck

        # Build full deck: offering on top, remaining deck below
        # deck_order from Ruby is top-to-bottom of remaining deck
        # offering from Ruby is the initial offering
        # Our set_order takes bottom-to-top
        # So: [remaining deck reversed] + [offering reversed] = bottom-to-top
        remaining_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(deck_order_names)]
        offering_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(offering_names)]
        full_deck = remaining_ids + offering_ids  # bottom-to-top
        DECK.set_order(state, full_deck)

        # Draw offering cards from the deck
        for _ in range(num_players):
            DECK.draw(state)

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

    def _setup_initial_offering(self, state, offering_names, deck_order_names):
        """Set up the initial offering to match the 18xx game.

        After initialize_game, the engine has already drawn some companies.
        We need to make sure the offering matches the 18xx initial offering.
        The deck order was already set. We need to draw cards from the deck
        to populate the offering.
        """
        # The deck was set with set_order. Now we need to verify the offering
        # matches. The engine's initialize_game already drew cards, but from
        # the wrong deck. With set_order called after init, the deck is correct
        # but the offering may be wrong.
        #
        # Strategy: Check what's currently in auction. If it matches, great.
        # If not, we need to fix it.
        current_offering = []
        for cid in range(36):
            if COMPANIES[cid].get_location(state) == LOC_AUCTION:
                current_offering.append(COMPANY_NAMES[cid])

        expected_offering = list(offering_names)
        if sorted(current_offering) == sorted(expected_offering):
            return  # Already correct

        if self.verbose:
            print(f"Offering mismatch: current={current_offering}, expected={expected_offering}")
            print("Reinitializing game with correct deck order...")

        # Full re-init approach: initialize again, then override deck
        # This is cleaner than trying to patch individual company locations
        state.initialize_game(seed=42)

        # Build the full deck (offering + remaining deck) bottom-to-top
        # The offering is drawn from the top of the deck
        full_order = list(deck_order_names) + list(reversed(offering_names))
        full_order_ids = [COMPANY_NAME_TO_ID[n] for n in reversed(full_order)]
        DECK.set_order(state, full_order_ids)

        # Now draw num_players cards to populate the offering
        num_players = state.get_num_players()
        for _ in range(num_players):
            DECK.draw(state)

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

        engine_action = self._map_action(state, action, phase, layout)
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

        # Compare state if we have a reference for this action_id
        if action_id in ref_by_action:
            ref = ref_by_action[action_id]
            self._compare_state(state, ref, f"after action {action_id}")

        return idx + 1

    def _map_action(self, state, action, phase, layout):
        """Map a single 18xx action to our engine action index."""
        atype = action.get('type', '')

        if phase == PHASE_INVEST:
            if atype in ('bid', 'buy_shares', 'sell_shares', 'pass'):
                return map_invest_action(state, action, layout)
            return None

        if phase == PHASE_BID:
            if atype in ('bid', 'pass'):
                return map_bid_action(action, layout)
            return None

        if phase == PHASE_IPO:
            if atype in ('par', 'pass'):
                return map_ipo_action(action, layout)
            return None

        if phase == PHASE_DIVIDENDS:
            if atype == 'dividend':
                return map_dividend_action(action, layout)
            return None

        if phase == PHASE_ISSUE:
            if atype in ('sell_shares', 'pass'):
                return map_issue_action(action, layout)
            return None

        # Automated phases — no player actions needed
        if phase in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_END_CARD):
            return None

        return None

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

        # Compare state after acquisition phase
        if end_idx > idx and end_idx - 1 < len(actions):
            last_acq_action = acq_actions[-1] if acq_actions else None
            if last_acq_action:
                last_id = last_acq_action.get('id', -1)
                if last_id in ref_by_action:
                    self._compare_state(state, ref_by_action[last_id],
                                        f"after ACQ phase (action {last_id})")

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

        # Compare state after closing phase
        if close_actions:
            last_close_action = close_actions[-1]
            last_id = last_close_action.get('id', -1)
            if last_id in ref_by_action:
                self._compare_state(state, ref_by_action[last_id],
                                    f"after CLO phase (action {last_id})")

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

        # Compare offering
        our_offering = sorted([
            COMPANY_NAMES[cid] for cid in range(36)
            if COMPANIES[cid].get_location(state) == LOC_AUCTION
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

        Uses the player_order from the initial state record or falls back
        to position in the players array.
        """
        # The players in ref are in the 18xx play order, which should match
        # our player indices 0, 1, 2, ...
        for idx, p in enumerate(ref.get('players', [])):
            if p['id'] == player_id_18xx:
                return idx
        raise ValueError(f"Player {player_id_18xx} not found in reference")


def format_mismatches(mismatches: list[Mismatch]) -> str:
    """Format mismatches for display."""
    lines = []
    for m in mismatches[:50]:  # Cap at 50 to avoid overwhelming output
        lines.append(str(m))
    if len(mismatches) > 50:
        lines.append(f"... and {len(mismatches) - 50} more")
    return "\n".join(lines)
