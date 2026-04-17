#!/usr/bin/env python3
"""Replay a committed 18xx fixture up to a target action id and inspect engine state.

Usage:
    PYTHONPATH=. .venv/bin/python tests/games_18xx/debug/inspect_replay_point.py 202494 78
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import COMPANY_NAMES, CORP_NAMES
from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN
from tests.games_18xx.replay_harness import load_ref_states
from utils_18xx.action_parser import ActionLayout, filter_actions, flatten_auto_actions, get_legal_actions, map_action
from utils_18xx.replay_state import initialize_replay_state, settle_to_player_choice

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def usage() -> None:
    print(__doc__.strip())
    raise SystemExit(2)


def describe_state(game_data: dict, state) -> None:
    print(f"phase={TURN.get_phase(state)} active_player={TURN.get_active_player(state)} active_corp={TURN.get_active_corp(state)}")
    print("players:")
    for idx, player in enumerate(game_data["players"]):
        companies = [
            COMPANY_NAMES[cid]
            for cid in range(len(COMPANY_NAMES))
            if COMPANIES[cid].is_owned_by_player(state, idx)
        ]
        shares = {
            CORP_NAMES[cid]: PLAYERS[idx].get_shares(state, cid)
            for cid in range(len(CORP_NAMES))
            if PLAYERS[idx].get_shares(state, cid)
        }
        print(
            f"  p{idx} name={player['name']} id={player['id']} cash={PLAYERS[idx].get_cash(state)} "
            f"passed={PLAYERS[idx].has_passed(state)} companies={companies} shares={shares}"
        )
    print("corps:")
    for cid, name in enumerate(CORP_NAMES):
        if not CORPS[cid].is_active(state):
            continue
        companies = [
            COMPANY_NAMES[x]
            for x in range(len(COMPANY_NAMES))
            if COMPANIES[x].is_owned_by_corp(state, cid)
        ]
        print(
            f"  {name}: pres={CORPS[cid].get_president_id(state)} recv={CORPS[cid].is_in_receivership(state)} "
            f"cash={CORPS[cid].get_cash(state)} price={CORPS[cid].get_share_price(state)} companies={companies}"
        )
    fi_companies = [
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_fi(state)
    ]
    print(f"fi: cash={FI.get_cash(state)} companies={fi_companies}")
    print("legal_actions:")
    for action_id, info in get_legal_actions(state):
        corp_name = CORP_NAMES[info.corp_id] if 0 <= info.corp_id < len(CORP_NAMES) else None
        company_name = COMPANY_NAMES[info.company_id] if 0 <= info.company_id < len(COMPANY_NAMES) else None
        print(
            f"  aid={action_id} type={info.action_type} corp={corp_name} company={company_name} amount={info.amount}"
        )


def main() -> None:
    if len(sys.argv) != 3:
        usage()

    game_id = int(sys.argv[1])
    stop_action_id = int(sys.argv[2])

    game_json_path = DATA_DIR / f"{game_id}.json"
    if not game_json_path.exists():
        raise SystemExit(f"missing fixture: {game_json_path}")

    game_data = json.loads(game_json_path.read_text())
    ref_states = load_ref_states(str(game_json_path))
    ref_by_action = {snapshot["action_id"]: snapshot for snapshot in ref_states}
    initial = ref_by_action[0]

    state = initialize_replay_state(
        len(game_data["players"]),
        initial["deck_order"],
        initial["initial_offering"],
        cost_level=initial.get("cost_level"),
    )
    layout = ActionLayout(len(game_data["players"]))

    committed_ids = set(initial.get("committed_action_ids", []))
    actions = flatten_auto_actions(filter_actions(game_data.get("actions", []), committed_ids or None))

    for action in actions:
        action_id = action.get("id", -1)
        settle_to_player_choice(state)
        if action_id == stop_action_id:
            print(f"stopped before action_id={stop_action_id}")
            print(f"next_action={json.dumps(action, sort_keys=True)}")
            describe_state(game_data, state)
            return

        try:
            engine_action = map_action(state, action, TURN.get_phase(state), layout)
        except Exception as exc:  # pragma: no cover - debug path
            print(f"mapping failed at action_id={action_id}: {exc}")
            print(f"raw_action={json.dumps(action, sort_keys=True)}")
            describe_state(game_data, state)
            return

        if engine_action is None:
            print(f"mapping returned None at action_id={action_id}")
            print(f"raw_action={json.dumps(action, sort_keys=True)}")
            describe_state(game_data, state)
            return

        result = DRIVER.apply_action(state, engine_action)
        if result == STATUS_INVALID:
            print(f"STATUS_INVALID at action_id={action_id} engine_action={engine_action}")
            print(f"raw_action={json.dumps(action, sort_keys=True)}")
            describe_state(game_data, state)
            return

    print(f"action_id={stop_action_id} not reached")


if __name__ == "__main__":
    main()
