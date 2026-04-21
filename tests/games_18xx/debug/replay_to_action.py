#!/usr/bin/env python3
"""Replay a game up to a specific action and dump engine state.

Usage:
    python tests/games_18xx/debug/replay_to_action.py <game_id> <stop_before_action_id>
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.state import GameState
from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.data import (
    COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, COMPANY_NAMES, CORP_NAMES,
    GamePhases, get_company_face_value,
)
from core.actions import get_valid_action_mask, get_action_layout
from entities.deck import DECK
from entities.turn import TURN
from entities.company import COMPANIES, CompanyLocation
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.fi import FI

LOC_AUCTION = CompanyLocation.LOC_AUCTION
LOC_REVEALED = CompanyLocation.LOC_REVEALED

from tests.games_18xx.replay_harness import ReplayHarness, load_ref_states
from utils_18xx.action_parser import (
    ActionLayout,
    AutoPassTracker,
    filter_actions,
    flatten_auto_actions,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    game_id = sys.argv[1]
    stop_aid = int(sys.argv[2])

    game_json = os.path.join(DATA_DIR, f'{game_id}.json')
    ref_states = load_ref_states(game_json)

    harness = ReplayHarness(
        game_json_path=game_json,
        ref_states=ref_states,
        verbose=True,
    )

    # Monkey-patch to stop before a specific action
    orig_replay = harness._replay_simple_action

    class StopHere(Exception):
        pass

    def patched_replay(state, actions, idx, layout, ref_by_action):
        if idx < len(actions):
            action = actions[idx]
            action_id = action.get('id', -1)
            if action_id == stop_aid:
                raise StopHere()
        return orig_replay(state, actions, idx, layout, ref_by_action)

    harness._replay_simple_action = patched_replay

    game_data = json.loads(open(game_json).read())
    num_players = len(game_data['players'])

    try:
        harness.run()
    except StopHere:
        pass

    # Access the harness's internal state
    # The state is the first positional arg of the last patched_replay call
    # Actually, we need to get state from harness. Let me just re-run more carefully.

    # Re-run from scratch but capture state
    state = None

    class StateCapture(ReplayHarness):
        def run(self):
            self.mismatches = []
            game_data = json.loads(open(self.game_json_path).read())
            ref_states = self.ref_states
            num_players = len(game_data['players'])
            players_json = game_data['players']
            layout = ActionLayout(num_players)

            ref_by_action = {s['action_id']: s for s in ref_states}
            initial = ref_by_action[0]
            deck_order_names = initial['deck_order']
            offering_names = initial['initial_offering']

            self._player_id_to_index = {}
            for idx, pid in enumerate(initial['player_order']):
                self._player_id_to_index[pid] = idx

            state = GameState(num_players)
            state.initialize_game(seed=42)
            state.allow_positive_income_closing = True
            self._override_deck_and_offering(state, deck_order_names, offering_names)
            self._last_ref = initial

            raw_actions = game_data.get('actions', [])
            auto_pass = AutoPassTracker([p['id'] for p in players_json])
            for a in raw_actions:
                if a.get('type', '').startswith('program_'):
                    auto_pass.process_action(a)

            committed_ids = set(initial.get('committed_action_ids', []))
            actions = filter_actions(raw_actions, committed_ids or None)
            actions = flatten_auto_actions(actions)

            idx = 0
            while idx < len(actions):
                phase = TURN.get_phase(state)
                if phase == GamePhases.PHASE_GAME_OVER:
                    break

                action = actions[idx]
                action_id = action.get('id', -1)

                if action_id == stop_aid:
                    # Dump state here
                    self._dump_state(state, layout, num_players, action)
                    return self.mismatches

                if phase == GamePhases.PHASE_ACQ_SELECT_CORP:
                    idx = self._run_acquisition_adapter(state, actions, idx, layout, ref_by_action)
                elif phase == GamePhases.PHASE_CLOSING:
                    idx = self._run_closing_adapter(state, actions, idx, layout, ref_by_action)
                else:
                    idx = self._replay_simple_action(state, actions, idx, layout, ref_by_action)

            return self.mismatches

        def _dump_state(self, state, layout, num_players, next_action):
            phase = TURN.get_phase(state)
            phase_names = {
                0: "INVEST", 1: "BID", 2: "WRAP_UP", 3: "ACQ", 4: "CLOSING",
                5: "INCOME", 6: "DIVIDENDS", 7: "END_CARD", 8: "ISSUE",
                9: "IPO", 10: "PAR", 11: "GAME_OVER",
            }
            print(f"\n{'='*60}")
            print(f"Engine state BEFORE action {next_action.get('id')}")
            print(f"  Phase: {phase_names.get(phase, phase)}")
            print(f"  Active player: {state.get_active_player()}")
            print(f"  _last_ref: {self._last_ref.get('action_id') if self._last_ref else None}")

            # Players
            for i in range(num_players):
                cash = PLAYERS[i].get_cash(state)
                nw = PLAYERS[i].get_net_worth(state)
                companies = [COMPANY_NAMES[c] for c in range(36)
                             if COMPANIES[c].is_owned_by_player(state, i)]
                shares = {}
                for cid in range(8):
                    n = PLAYERS[i].get_shares(state, cid)
                    if n > 0:
                        shares[CORP_NAMES[cid]] = n
                print(f"  Player {i}: cash={cash} nw={nw} companies={companies} shares={shares}")

            # Corps
            for cid in range(8):
                if CORPS[cid].is_active(state):
                    price = CORPS[cid].get_share_price(state)
                    cash = CORPS[cid].get_cash(state)
                    bank = CORPS[cid].get_bank_shares(state)
                    companies = [COMPANY_NAMES[c] for c in range(36)
                                 if COMPANIES[c].is_owned_by_corp(state, cid)]
                    print(f"  Corp {CORP_NAMES[cid]}: price={price} cash={cash} bank_shares={bank} companies={companies}")

            # FI
            fi_cash = FI.get_cash(state)
            fi_companies = [COMPANY_NAMES[c] for c in range(36)
                            if COMPANIES[c].is_owned_by_fi(state)]
            print(f"  FI: cash={fi_cash} companies={fi_companies}")

            # Offering
            offering = []
            for cid in range(36):
                loc = COMPANIES[cid].get_location(state)
                if loc == LOC_AUCTION:
                    offering.append(f"{COMPANY_NAMES[cid]}({cid})")
                elif loc == LOC_REVEALED:
                    offering.append(f"{COMPANY_NAMES[cid]}({cid})*")
            print(f"  Offering: {offering}")
            print(f"  Deck: {DECK.get_remaining_count(state)}")
            print(f"  CoO: {TURN.get_coo_level(state)}")

            # Action mask
            mask = get_valid_action_mask(state)
            legal = [(i, v) for i, v in enumerate(mask) if v > 0.5]
            print(f"  Legal actions ({len(legal)}):")
            for idx, v in legal[:20]:
                print(f"    action {idx}")
            if len(legal) > 20:
                print(f"    ... and {len(legal)-20} more")

            # Check the specific action
            aid = next_action.get('id')
            atype = next_action.get('type')
            if atype == 'bid':
                company_name = next_action.get('company')
                price = next_action.get('price')
                print(f"\n  Next action: bid on {company_name} at {price}")
                cid_company = COMPANY_NAME_TO_ID.get(company_name)
                if cid_company is not None:
                    loc = COMPANIES[cid_company].get_location(state)
                    print(f"  Company {company_name} location: {loc}")
                    face = get_company_face_value(cid_company)
                    print(f"  Face value: {face}")

                    # Find auction slot
                    slot = 0
                    for c in range(36):
                        cloc = COMPANIES[c].get_location(state)
                        if cloc == LOC_AUCTION:
                            if c == cid_company:
                                print(f"  Found in auction slot {slot}")
                                break
                            slot += 1

                    engine_action = layout.auction_base + slot * 15 + (int(price) - face)
                    print(f"  Computed engine_action: {engine_action}")
                    if engine_action < len(mask):
                        print(f"  Action mask[{engine_action}]: {mask[engine_action]}")

    sc = StateCapture(game_json_path=game_json, ref_states=ref_states, verbose=False)
    sc.run()


if __name__ == '__main__':
    main()
