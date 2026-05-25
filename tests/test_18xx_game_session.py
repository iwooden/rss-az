import json
import subprocess
from pathlib import Path

from core.data import COMPANY_NAME_TO_ID, COMPANY_NAMES, CORP_NAMES, GamePhases
from core.driver import STATUS_INVALID_PY as STATUS_INVALID
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.fi import FI
from entities.market import MARKET
from entities.player import PLAYERS
from entities.turn import TURN
from phases.acq_select_corp import setup_acquisition_phase_py
from phases.ipo import setup_ipo_phase_py
from tests.phases.helpers.finance import set_player_cashs
from tests.phases.helpers.ownership import (
    give_company_to_corp,
    give_company_to_fi,
    give_company_to_player,
)
from tests.phases.conftest import draw_to_player, float_corp_for_test
import utils_18xx.game_session as game_session_module
from utils_18xx.action_parser import map_action
from utils_18xx.game_session import EXTRACTOR_PATH, GameSession
from utils_18xx.replay_state import apply_action_sequence


def _new_state() -> GameState:
    state = GameState(3)
    state.initialize_game(3, seed=42)
    return state


def test_session_replays_18xx_bid_as_invest_and_bid_steps():
    players = [
        {"id": 4, "name": "rss-az-3"},
        {"id": 3, "name": "rss-az-2"},
        {"id": 2, "name": "rss-az-1"},
    ]
    company = "MHE"
    company_id = COMPANY_NAME_TO_ID[company]
    price = COMPANIES[company_id].get_face_value()
    initial = {
        "action_id": 0,
        "deck_order": [],
        "initial_offering": [company],
        "committed_action_ids": [1],
        "players": players,
    }
    game_data = {
        "id": "unit",
        "players": players,
        "actions": [{
            "id": 1,
            "type": "bid",
            "entity": players[0]["id"],
            "entity_type": "player",
            "company": company,
            "price": price,
        }],
    }

    session = GameSession(3)
    session._run_extractor = lambda data: initial

    state = session.sync(game_data)

    assert TURN.get_phase(state) == int(GamePhases.PHASE_BID)
    assert TURN.get_active_player(state) == 1
    assert session.player_index_for_user_id(players[1]["id"]) == 1


def test_new_game_sync_uses_initial_extractor_records_for_committed_ids():
    players = [
        {"id": 4, "name": "rss-az-3"},
        {"id": 3, "name": "rss-az-2"},
        {"id": 2, "name": "rss-az-1"},
    ]
    initial = {
        "action_id": 0,
        "deck_order": [],
        "initial_offering": ["MHE"],
        "committed_action_ids": [],
        "players": players,
    }
    calls = []

    session = GameSession(3)

    def fake_run_extractor(data):
        calls.append(data)
        return initial

    session._run_extractor = fake_run_extractor

    session.sync({"id": "unit", "players": players, "actions": []})

    assert len(calls) == 1
    assert session.committed_ids == set()


def test_extractor_undo_after_message_ignores_chat(tmp_path):
    game_path = Path(__file__).parent / "games_18xx" / "data" / "223139.json"
    game_data = json.loads(game_path.read_text())
    game_data["actions"].extend([
        {
            "type": "message",
            "entity": 15352,
            "entity_type": "player",
            "id": 2,
            "message": "chat should not consume undo",
            "created_at": 1757521849,
        },
        {
            "type": "undo",
            "entity": 15352,
            "entity_type": "player",
            "id": 3,
            "created_at": 1757521850,
        },
    ])
    tmp_game = tmp_path / "game.json"
    tmp_game.write_text(json.dumps(game_data))

    result = subprocess.run(
        ["ruby", str(EXTRACTOR_PATH), str(tmp_game)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    assert [record["action_id"] for record in records] == [0]
    assert records[0]["committed_action_ids"] == []


def test_split_followup_replays_18xx_par_price_after_ipo_selection():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    company_id = 14
    par_price = 20
    give_company_to_player(state, company_id, 0)
    set_player_cashs(state, {0: 100})
    setup_ipo_phase_py(state)

    session = GameSession(3)
    action = {
        "type": "par",
        "entity": COMPANIES[company_id].name,
        "entity_type": "company",
        "corporation": CORP_NAMES[0],
        "share_price": f"{par_price},0,{MARKET.get_index_for_price(par_price)}",
    }

    first_action = map_action(state, action, TURN.get_phase(state), session.layout)
    assert apply_action_sequence(state, first_action) != STATUS_INVALID
    assert TURN.get_phase(state) == int(GamePhases.PHASE_PAR)

    result = session._apply_split_action_followup(
        state,
        action,
        GamePhases.PHASE_IPO,
    )

    assert result != STATUS_INVALID
    assert TURN.get_phase(state) != int(GamePhases.PHASE_PAR)


def test_split_bid_followup_accepts_forced_opening_bid_auto_apply():
    state = _new_state()
    company_id = COMPANY_NAME_TO_ID["MS"]
    price = COMPANIES[company_id].get_face_value()
    COMPANIES[company_id].move_to_auction(state)
    TURN.set_phase(state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(state, 1)
    PLAYERS[1].set_cash(state, price)

    session = GameSession(3)
    session._player_ids = [2, 3, 4]
    action = {
        "type": "bid",
        "entity": 3,
        "entity_type": "player",
        "company": "MS",
        "price": price,
    }

    first_action = map_action(state, action, TURN.get_phase(state), session.layout)
    assert apply_action_sequence(state, first_action) != STATUS_INVALID
    assert TURN.get_phase(state) == int(GamePhases.PHASE_BID)
    assert TURN.get_auction_price(state) == price
    assert TURN.get_auction_high_bidder(state) == 1

    result = session._apply_split_action_followup(
        state,
        action,
        GamePhases.PHASE_INVEST,
    )

    assert result != STATUS_INVALID
    assert TURN.get_auction_price(state) == price
    assert TURN.get_auction_high_bidder(state) == 1


def test_share_owner_snapshot_uses_extractor_cash_recipient(monkeypatch):
    state = _new_state()
    session = GameSession(3)
    session._player_ids = [10, 20, 30]
    session._extract_records_by_action_id = {
        4: {
            "action_id": 4,
            "players": [
                {"id": 10, "cash": 5},
                {"id": 20, "cash": 7},
                {"id": 30, "cash": 9},
            ],
        },
        5: {
            "action_id": 5,
            "players": [
                {"id": 10, "cash": 5},
                {"id": 20, "cash": 15},
                {"id": 30, "cash": 9},
            ],
        },
    }
    monkeypatch.setattr(
        game_session_module,
        "share_owner_before_action",
        lambda game_data, committed_ids, share_ref, action_id: 30,
    )

    snapshot = session._share_owner_snapshot(
        {"actions": []},
        state,
        {
            "id": 5,
            "type": "sell_shares",
            "entity": 20,
            "entity_type": "player",
            "shares": ["SI_0"],
            "share_price": 8,
        },
    )

    assert snapshot is None


def test_share_owner_snapshot_keeps_adjustment_for_extractor_owner(monkeypatch):
    state = _new_state()
    session = GameSession(3)
    session._player_ids = [10, 20, 30]
    session._extract_records_by_action_id = {
        4: {
            "action_id": 4,
            "players": [
                {"id": 10, "cash": 5},
                {"id": 20, "cash": 7},
                {"id": 30, "cash": 9},
            ],
        },
        5: {
            "action_id": 5,
            "players": [
                {"id": 10, "cash": 5},
                {"id": 20, "cash": 7},
                {"id": 30, "cash": 17},
            ],
        },
    }
    monkeypatch.setattr(
        game_session_module,
        "share_owner_before_action",
        lambda game_data, committed_ids, share_ref, action_id: 30,
    )

    snapshot = session._share_owner_snapshot(
        {"actions": []},
        state,
        {
            "id": 5,
            "type": "sell_shares",
            "entity": 20,
            "entity_type": "player",
            "shares": ["SI_0"],
            "share_price": 8,
        },
    )

    assert snapshot == (CORP_NAMES.index("SI"), 1, 2, 8)


def test_session_consumes_closing_auto_passes_before_ipo():
    players = [
        {"id": 4, "name": "rss-az-3"},
        {"id": 3, "name": "rss-az-2"},
        {"id": 2, "name": "rss-az-1"},
    ]
    initial = {
        "action_id": 0,
        "deck_order": [
            "BME",
            "PR",
            "OL",
            "MS",
            "BD",
            "PKP",
            "DR",
            "SNCF",
            "NS",
            "SZD",
            "FS",
            "E",
            "SJ",
            "LHR",
            "CDG",
            "HA",
            "FRA",
        ],
        "initial_offering": ["MHE", "BPM", "AKE"],
        "committed_action_ids": list(range(1, 12)),
        "players": players,
    }
    actions = [
        {"id": 1, "type": "bid", "entity": 4, "entity_type": "player", "company": "MHE", "price": 8},
        {"id": 2, "type": "bid", "entity": 3, "entity_type": "player", "company": "MHE", "price": 12},
        {"id": 3, "type": "pass", "entity": 2, "entity_type": "player"},
        {"id": 4, "type": "pass", "entity": 4, "entity_type": "player"},
        {"id": 5, "type": "bid", "entity": 3, "entity_type": "player", "company": "BPM", "price": 10},
        {"id": 6, "type": "pass", "entity": 2, "entity_type": "player"},
        {"id": 7, "type": "bid", "entity": 4, "entity_type": "player", "company": "BPM", "price": 11},
        {"id": 8, "type": "pass", "entity": 3, "entity_type": "player"},
        {"id": 9, "type": "bid", "entity": 2, "entity_type": "player", "company": "AKE", "price": 10},
        {"id": 10, "type": "pass", "entity": 4, "entity_type": "player"},
        {
            "id": 11,
            "type": "pass",
            "entity": 3,
            "entity_type": "player",
            "auto_actions": [
                {"type": "pass", "entity": 2, "entity_type": "player"},
                {"type": "pass", "entity": 4, "entity_type": "player"},
                {"type": "pass", "entity": 3, "entity_type": "player"},
            ],
        },
    ]
    game_data = {"id": "unit", "players": players, "actions": actions}

    session = GameSession(3)
    session._run_extractor = lambda data: initial

    state = session.sync(game_data)

    assert TURN.get_phase(state) == int(GamePhases.PHASE_IPO)
    assert TURN.get_active_player(state) == 1
    assert session.player_index_for_user_id(3) == 1


def test_session_preserves_live_auction_after_share_price_cash_replay():
    players = [
        {"id": 4, "name": "rss-az-3"},
        {"id": 3, "name": "rss-az-2"},
        {"id": 2, "name": "rss-az-1"},
    ]
    initial = {
        "action_id": 0,
        "deck_order": [
            "BME", "PR", "OL", "MS", "BD", "PKP", "DR", "SNCF", "NS",
            "SZD", "FS", "E", "SJ", "LHR", "CDG", "HA", "FRA",
        ],
        "initial_offering": ["MHE", "BPM", "AKE"],
        "committed_action_ids": list(range(1, 35)),
        "players": players,
    }
    actions = [
        {"id": 1, "type": "bid", "entity": 4, "entity_type": "player", "company": "MHE", "price": 8, "user": 4},
        {"id": 2, "type": "bid", "entity": 3, "entity_type": "player", "company": "MHE", "price": 12, "user": 3},
        {"id": 3, "type": "pass", "entity": 2, "entity_type": "player", "user": 2},
        {"id": 4, "type": "pass", "entity": 4, "entity_type": "player", "user": 4},
        {"id": 5, "type": "bid", "entity": 3, "entity_type": "player", "company": "BPM", "price": 10, "user": 3},
        {"id": 6, "type": "pass", "entity": 2, "entity_type": "player", "user": 2},
        {"id": 7, "type": "bid", "entity": 4, "entity_type": "player", "company": "BPM", "price": 11, "user": 4},
        {"id": 8, "type": "pass", "entity": 3, "entity_type": "player", "user": 3},
        {"id": 9, "type": "bid", "entity": 2, "entity_type": "player", "company": "AKE", "price": 10, "user": 2},
        {"id": 10, "type": "pass", "entity": 4, "entity_type": "player", "user": 4},
        {
            "id": 11,
            "type": "pass",
            "entity": 3,
            "entity_type": "player",
            "user": 3,
            "auto_actions": [
                {"type": "pass", "entity": 2, "entity_type": "player"},
                {"type": "pass", "entity": 4, "entity_type": "player"},
                {"type": "pass", "entity": 3, "entity_type": "player"},
            ],
        },
        {"id": 12, "type": "par", "entity": "MHE", "entity_type": "company", "corporation": "SI", "share_price": "10,0,6", "user": 3},
        {"id": 13, "type": "par", "entity": "BPM", "entity_type": "company", "corporation": "DA", "share_price": "11,0,7", "user": 4},
        {"id": 14, "type": "pass", "entity": "AKE", "entity_type": "company", "user": 2},
        {"id": 15, "type": "bid", "entity": 2, "entity_type": "player", "company": "BME", "price": 4, "user": 2},
        {"id": 16, "type": "pass", "entity": 4, "entity_type": "player", "user": 4},
        {"id": 17, "type": "pass", "entity": 3, "entity_type": "player", "user": 3},
        {"id": 18, "type": "buy_shares", "entity": 4, "entity_type": "player", "shares": ["DA_1"], "share_price": 11, "user": 4},
        {"id": 19, "type": "bid", "entity": 3, "entity_type": "player", "company": "OL", "price": 16, "user": 3},
        {"id": 20, "type": "bid", "entity": 2, "entity_type": "player", "company": "OL", "price": 17, "user": 2},
        {"id": 21, "type": "pass", "entity": 3, "entity_type": "player", "user": 3},
        {"id": 22, "type": "pass", "entity": 4, "entity_type": "player", "user": 4},
        {"id": 23, "type": "buy_shares", "entity": 3, "entity_type": "player", "shares": ["SI_1"], "share_price": 10, "user": 3},
        {"id": 24, "type": "pass", "entity": 4, "entity_type": "player", "user": 4},
        {"id": 25, "type": "pass", "entity": 3, "entity_type": "player", "user": 3},
        {"id": 26, "type": "pass", "entity": 3, "entity_type": "player", "user": 3},
        {
            "id": 27,
            "type": "pass",
            "entity": 4,
            "entity_type": "player",
            "user": 4,
            "auto_actions": [
                {"type": "pass", "entity": 2, "entity_type": "player"},
                {"type": "pass", "entity": 3, "entity_type": "player"},
                {"type": "pass", "entity": 4, "entity_type": "player"},
            ],
        },
        {"id": 28, "type": "dividend", "entity": "DA", "entity_type": "corporation", "amount": 4, "user": 4},
        {"id": 29, "type": "dividend", "entity": "SI", "entity_type": "corporation", "amount": 3, "user": 3},
        {"id": 30, "type": "pass", "entity": "SI", "entity_type": "corporation", "user": 3},
        {"id": 31, "type": "sell_shares", "entity": "DA", "entity_type": "corporation", "shares": ["DA_2"], "share_price": 12, "user": 4},
        {"id": 32, "type": "pass", "entity": "OL", "entity_type": "company", "user": 2},
        {"id": 33, "type": "par", "entity": "AKE", "entity_type": "company", "corporation": "PR", "share_price": "12,0,8", "user": 2},
        {"id": 34, "type": "bid", "entity": 3, "entity_type": "player", "company": "BD", "price": 13, "user": 3},
    ]

    session = GameSession(3)
    session._run_extractor = lambda data: initial

    state = session.sync({"id": "unit", "round": "Investment", "players": players, "actions": actions})

    assert TURN.get_phase(state) == int(GamePhases.PHASE_BID)
    assert TURN.get_active_player(state) == 0
    assert session.player_index_for_user_id(4) == 0


def test_acq_sync_applies_recorded_pass():
    state = _new_state()
    draw_to_player(state, 0)
    draw_to_player(state, 1)
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    float_corp_for_test(state, corp_id=1, player_id=1, par_index=12)
    CORPS[0].set_cash(state, 100)
    CORPS[1].set_cash(state, 100)
    setup_acquisition_phase_py(state)

    first_active = TURN.get_active_player(state)

    session = GameSession(3)
    next_idx = session._sync_acq_round(
        state,
        [{"type": "pass", "entity": "rss-az-2", "entity_type": "player"}],
        0,
    )

    assert next_idx == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_SELECT_CORP)
    assert TURN.get_active_player(state) != first_active


def test_acq_sync_applies_pass_for_recorded_entity():
    state = _new_state()
    for player_id, corp_id in enumerate((0, 1, 2)):
        float_corp_for_test(state, corp_id=corp_id, player_id=player_id, par_index=10)
        CORPS[corp_id].set_cash(state, 100)
    setup_acquisition_phase_py(state)
    state.step_mode = True

    first_active = TURN.get_active_player(state)
    target_player = (first_active + 1) % 3

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    next_idx = session._sync_acq_round(
        state,
        [{
            "type": "pass",
            "entity": session._player_ids[target_player],
            "entity_type": "player",
        }],
        0,
    )

    assert next_idx == 1
    assert PLAYERS[target_player].has_passed(state)
    assert not PLAYERS[first_active].has_passed(state)


def test_acq_sync_retargets_unordered_offer_to_recorded_entity():
    state = _new_state()
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 0)
    TURN.set_active_corp(state, 3)

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    seen_active = []

    def fake_immediate(current_state, offer, has_future_response=False):
        del offer, has_future_response
        seen_active.append(TURN.get_active_player(current_state))
        return False

    def fake_begin(current_state, offer, deferred_transfers):
        del current_state, offer, deferred_transfers
        return False

    session._offer_resolves_immediately = fake_immediate
    session._begin_acq_offer = fake_begin

    next_idx = session._sync_acq_round(
        state,
        [{
            "type": "offer",
            "entity": 3,
            "entity_type": "player",
            "corporation": "DA",
            "company": "BPM",
            "price": 4,
        }],
        0,
    )

    assert next_idx == 1
    assert seen_active == [1]
    assert TURN.get_active_player(state) == 1
    assert TURN.get_active_corp(state) == -1


def test_acq_sync_discards_trailing_auto_pass_after_closing_transition():
    state = _new_state()
    for player_id, corp_id in enumerate((0, 1, 2)):
        float_corp_for_test(state, corp_id=corp_id, player_id=player_id, par_index=10)
        CORPS[corp_id].set_cash(state, 100)
    setup_acquisition_phase_py(state)
    state.step_mode = True

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    actions = [
        {"type": "pass", "entity": 4, "entity_type": "player"},
        {
            "type": "pass",
            "entity": 3,
            "entity_type": "player",
            "_auto_parent_type": "pass",
        },
        {
            "type": "pass",
            "entity": 2,
            "entity_type": "player",
            "_auto_parent_type": "pass",
        },
        {
            "type": "pass",
            "entity": 4,
            "entity_type": "player",
            "_auto_parent_type": "pass",
        },
    ]

    next_idx = session._sync_acq_round(state, actions, 0)

    assert next_idx == len(actions)
    assert TURN.get_phase(state) != int(GamePhases.PHASE_DIVIDENDS)


def test_acq_sync_stops_before_recorded_closing_pass():
    state = _new_state()
    for player_id, corp_id in enumerate((0, 1, 2)):
        float_corp_for_test(state, corp_id=corp_id, player_id=player_id, par_index=10)
        CORPS[corp_id].set_cash(state, 100)
    setup_acquisition_phase_py(state)

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    session._extract_records_by_action_id = {
        10: {"action_id": 10, "round": "CLO", "current_round": "CLO"},
    }
    actions = [{"id": 10, "type": "pass", "entity": 4, "entity_type": "player"}]

    next_idx = session._sync_acq_round(state, actions, 0)

    assert next_idx == 0
    assert TURN.get_phase(state) in (
        int(GamePhases.PHASE_ACQ_SELECT_CORP),
        int(GamePhases.PHASE_ACQ_SELECT_COMPANY),
        int(GamePhases.PHASE_ACQ_SELECT_PRICE),
        int(GamePhases.PHASE_ACQ_OFFER),
    )

    session._drain_acq_phases(state)
    assert TURN.get_phase(state) not in (
        int(GamePhases.PHASE_ACQ_SELECT_CORP),
        int(GamePhases.PHASE_ACQ_SELECT_COMPANY),
        int(GamePhases.PHASE_ACQ_SELECT_PRICE),
        int(GamePhases.PHASE_ACQ_OFFER),
    )


def test_acq_sync_applies_same_president_offer_before_later_passes():
    state = _new_state()
    company_id = COMPANY_NAME_TO_ID["OL"]
    corp_id = CORP_NAMES.index("PR")
    give_company_to_player(state, company_id, 0)
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    CORPS[corp_id].set_cash(state, 100)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, 0)

    session = GameSession(3)
    next_idx = session._sync_acq_round(
        state,
        [
            {
                "type": "offer",
                "entity": "rss-az-1",
                "entity_type": "player",
                "corporation": "PR",
                "company": "OL",
                "price": COMPANIES[company_id].get_low_price(),
            },
            {"type": "pass", "entity": "rss-az-1", "entity_type": "player"},
        ],
        0,
    )

    assert next_idx >= 1
    assert COMPANIES[company_id].is_owned_by_corp(state, corp_id)


def test_acq_sync_rolls_back_unmappable_offer_prefix_before_next_offer():
    state = GameState(4, acq_same_president=False)
    state.initialize_game(4, seed=42)

    da_id = CORP_NAMES.index("DA")
    pr_id = CORP_NAMES.index("PR")
    bse_id = COMPANY_NAME_TO_ID["BSE"]
    by_id = COMPANY_NAME_TO_ID["BY"]
    ms_id = COMPANY_NAME_TO_ID["MS"]

    float_corp_for_test(
        state,
        corp_id=da_id,
        company_id=COMPANY_NAME_TO_ID["WT"],
        player_id=3,
        par_index=10,
    )
    float_corp_for_test(
        state,
        corp_id=pr_id,
        company_id=COMPANY_NAME_TO_ID["AKE"],
        player_id=0,
        par_index=10,
    )
    give_company_to_player(state, bse_id, 3)
    give_company_to_player(state, by_id, 1)
    give_company_to_player(state, ms_id, 0)
    CORPS[da_id].set_cash(state, COMPANIES[bse_id].get_high_price())
    CORPS[pr_id].set_cash(state, COMPANIES[ms_id].get_high_price())
    setup_acquisition_phase_py(state)

    session = GameSession(4)
    session._player_ids = [101, 202, 303, 404]
    next_idx = session._sync_acq_round(
        state,
        [
            {
                "id": 201,
                "type": "offer",
                "entity": 404,
                "entity_type": "player",
                "corporation": "DA",
                "company": "BY",
                "price": COMPANIES[by_id].get_high_price(),
            },
            {
                "id": 202,
                "type": "offer",
                "entity": 101,
                "entity_type": "player",
                "corporation": "PR",
                "company": "MS",
                "price": COMPANIES[ms_id].get_high_price(),
            },
        ],
        0,
    )

    assert next_idx == 2
    assert not COMPANIES[by_id].is_in_corp_acquisition(state, da_id)
    assert (
        COMPANIES[ms_id].is_owned_by_corp(state, pr_id)
        or COMPANIES[ms_id].is_in_corp_acquisition(state, pr_id)
    )
    assert TURN.get_phase(state) != int(GamePhases.PHASE_ACQ_SELECT_COMPANY)
    assert TURN.get_active_corp(state) != da_id


def test_acq_sync_resolves_fi_offer_before_following_passes():
    state = _new_state()
    corp_id = CORP_NAMES.index("OS")
    company_id = COMPANY_NAME_TO_ID["B"]
    float_corp_for_test(
        state,
        corp_id=corp_id,
        company_id=COMPANY_NAME_TO_ID["MHE"],
        player_id=2,
        par_index=10,
    )
    give_company_to_fi(state, company_id)
    CORPS[corp_id].set_cash(state, 50)
    FI.set_cash(state, 10)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, 2)

    session = GameSession(3)
    session._player_ids = [2, 3, 4]
    next_idx = session._sync_acq_round(
        state,
        [
            {
                "id": 88,
                "type": "offer",
                "entity": 4,
                "entity_type": "player",
                "corporation": "OS",
                "company": "B",
                "price": COMPANIES[company_id].get_face_value(),
            },
            {
                "id": 89,
                "type": "pass",
                "entity": 4,
                "entity_type": "player",
                "auto_actions": [
                    {"type": "pass", "entity": 2, "entity_type": "player"},
                    {"type": "pass", "entity": 3, "entity_type": "player"},
                    {"type": "pass", "entity": 4, "entity_type": "player"},
                ],
            },
            {
                "type": "pass",
                "entity": 2,
                "entity_type": "player",
                "_auto_parent_type": "pass",
                "_auto_parent_id": 89,
            },
            {
                "type": "pass",
                "entity": 3,
                "entity_type": "player",
                "_auto_parent_type": "pass",
                "_auto_parent_id": 89,
            },
            {
                "type": "pass",
                "entity": 4,
                "entity_type": "player",
                "_auto_parent_type": "pass",
                "_auto_parent_id": 89,
            },
        ],
        0,
    )

    assert next_idx == 5
    assert (
        COMPANIES[company_id].is_owned_by_corp(state, corp_id)
        or COMPANIES[company_id].is_in_corp_acquisition(state, corp_id)
    )
    assert CORPS[corp_id].get_cash(state) < 50
    assert FI.get_cash(state) > 10


def test_fi_offer_waits_for_explicit_response():
    state = _new_state()
    corp_id = CORP_NAMES.index("OS")
    company_id = COMPANY_NAME_TO_ID["B"]
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    give_company_to_fi(state, company_id)

    session = GameSession(3)
    offer = {
        "id": 10,
        "type": "offer",
        "corporation": "OS",
        "company": "B",
        "price": COMPANIES[company_id].get_face_value(),
    }
    response = {
        "id": 11,
        "type": "respond",
        "corporation": "OS",
        "company": "B",
        "accept": "true",
    }

    assert not session._offer_resolves_immediately(
        state,
        offer,
        has_future_response=session._has_future_response_to_offer([response], 0, offer),
    )


def test_acq_sync_replays_declined_fi_preemption_before_original_buy():
    state = GameState(4, acq_same_president=False)
    state.initialize_game(4, seed=42)

    sm_id = CORP_NAMES.index("SM")
    os_id = CORP_NAMES.index("OS")
    sbb_id = COMPANY_NAME_TO_ID["SBB"]
    float_corp_for_test(
        state,
        corp_id=sm_id,
        company_id=COMPANY_NAME_TO_ID["MHE"],
        player_id=1,
        par_index=10,
    )
    float_corp_for_test(
        state,
        corp_id=os_id,
        company_id=COMPANY_NAME_TO_ID["BY"],
        player_id=3,
        par_index=12,
    )
    give_company_to_fi(state, sbb_id)
    CORPS[sm_id].set_cash(state, 100)
    CORPS[os_id].set_cash(state, 100)
    FI.set_cash(state, 1)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, 1)

    session = GameSession(4)
    session._player_ids = [4, 2, 5, 3]
    next_idx = session._sync_acq_round(
        state,
        [
            {
                "id": 130,
                "type": "offer",
                "entity": 2,
                "entity_type": "player",
                "corporation": "SM",
                "company": "SBB",
                "price": COMPANIES[sbb_id].get_high_price(),
            },
            {"id": 131, "type": "pass", "entity": 4, "entity_type": "player"},
            {
                "id": 132,
                "type": "respond",
                "entity": 3,
                "entity_type": "player",
                "corporation": "SM",
                "company": "SBB",
                "accept": "false",
            },
        ],
        0,
    )

    assert next_idx == 3
    assert (
        COMPANIES[sbb_id].is_owned_by_corp(state, sm_id)
        or COMPANIES[sbb_id].is_in_corp_acquisition(state, sm_id)
    )
    assert not COMPANIES[sbb_id].is_owned_by_corp(state, os_id)
    assert CORPS[os_id].get_cash(state) == 100
    assert CORPS[sm_id].get_cash(state) == 100 - COMPANIES[sbb_id].get_high_price()
    assert FI.get_cash(state) == 1 + COMPANIES[sbb_id].get_high_price()


def test_acq_sync_replays_implicit_fi_preemption_response():
    state = GameState(4, acq_same_president=False)
    state.initialize_game(4, seed=42)

    sm_id = CORP_NAMES.index("SM")
    os_id = CORP_NAMES.index("OS")
    sbb_id = COMPANY_NAME_TO_ID["SBB"]
    float_corp_for_test(
        state,
        corp_id=sm_id,
        company_id=COMPANY_NAME_TO_ID["MHE"],
        player_id=1,
        par_index=10,
    )
    float_corp_for_test(
        state,
        corp_id=os_id,
        company_id=COMPANY_NAME_TO_ID["BY"],
        player_id=3,
        par_index=12,
    )
    give_company_to_fi(state, sbb_id)
    CORPS[sm_id].set_cash(state, 100)
    CORPS[os_id].set_cash(state, 100)
    FI.set_cash(state, 1)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, 1)

    session = GameSession(4)
    session._player_ids = [4, 2, 5, 3]
    offer = {
        "id": 130,
        "type": "offer",
        "entity": 2,
        "entity_type": "player",
        "corporation": "SM",
        "company": "SBB",
        "price": COMPANIES[sbb_id].get_high_price(),
    }
    assert session._begin_acq_offer(state, offer, [])
    assert TURN.get_phase(state) == int(GamePhases.PHASE_ACQ_OFFER)

    next_idx = session._sync_acq_round(
        state,
        [{
            "id": 132,
            "type": "respond",
            "entity": 3,
            "entity_type": "player",
            "corporation": "SM",
            "company": "SBB",
            "accept": "false",
        }],
        0,
    )

    assert next_idx == 1
    assert (
        COMPANIES[sbb_id].is_owned_by_corp(state, sm_id)
        or COMPANIES[sbb_id].is_in_corp_acquisition(state, sm_id)
    )
    assert not COMPANIES[sbb_id].is_owned_by_corp(state, os_id)
    assert CORPS[sm_id].get_cash(state) == 100 - COMPANIES[sbb_id].get_high_price()
    assert FI.get_cash(state) == 1 + COMPANIES[sbb_id].get_high_price()


def test_acq_replay_stops_after_same_president_offer_executes():
    state = GameState(3, acq_same_president=False)
    state.initialize_game(3, seed=42)

    da_id = CORP_NAMES.index("DA")
    pr_id = CORP_NAMES.index("PR")
    bd_id = COMPANY_NAME_TO_ID["BD"]
    ake_id = COMPANY_NAME_TO_ID["AKE"]
    ol_id = COMPANY_NAME_TO_ID["OL"]

    give_company_to_player(state, bd_id, 0)
    float_corp_for_test(state, corp_id=da_id, player_id=0, par_index=10)
    float_corp_for_test(
        state, corp_id=pr_id, company_id=ake_id, player_id=2, par_index=10
    )
    give_company_to_corp(state, ol_id, pr_id)
    CORPS[da_id].set_cash(state, 25)
    setup_acquisition_phase_py(state)
    TURN.set_active_player(state, 0)

    session = GameSession(3)
    session._sync_acq_round(
        state,
        [{
            "type": "offer",
            "entity": "rss-az-3",
            "entity_type": "player",
            "corporation": "DA",
            "company": "BD",
            "price": COMPANIES[bd_id].get_low_price(),
        }],
        0,
    )

    assert (
        COMPANIES[bd_id].is_owned_by_corp(state, da_id)
        or COMPANIES[bd_id].is_in_corp_acquisition(state, da_id)
    )
    assert TURN.get_active_company(state) != ake_id


def test_clo_sync_applies_recorded_pass_entity_only():
    state = _new_state()
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 2)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    session = GameSession(3)
    session._player_ids = [4, 3, 2]

    next_idx = session._sync_clo_round(
        state,
        [{"type": "pass", "entity": 4, "entity_type": "player"}],
        0,
    )

    assert next_idx == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert PLAYERS[0].has_passed(state)
    assert not PLAYERS[2].has_passed(state)
    assert TURN.get_active_player(state) == 2


def test_clo_sync_respects_out_of_order_pass_entity():
    state = _new_state()
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 2)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    session = GameSession(3)
    session._player_ids = [4, 3, 2]

    next_idx = session._sync_clo_round(
        state,
        [{"type": "pass", "entity": 2, "entity_type": "player"}],
        0,
    )

    assert next_idx == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert not PLAYERS[0].has_passed(state)
    assert PLAYERS[2].has_passed(state)
    assert TURN.get_active_player(state) == 0


def test_session_does_not_drain_live_acquisition_round():
    state = _new_state()
    float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
    CORPS[0].set_cash(state, 100)
    setup_acquisition_phase_py(state)

    session = GameSession(3)

    assert not session._should_drain_trailing_offer_phases(
        {"round": "Acquisition"},
        state,
    )


def test_session_does_not_drain_when_extractor_round_is_closing():
    state = _new_state()
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    session = GameSession(3)
    session._last_extract_record = {"current_round": "CLO"}

    assert not session._should_drain_trailing_offer_phases(
        {"round": "Acquisition"},
        state,
    )


def test_live_state_validation_reports_corp_price_mismatch():
    state = _new_state()
    corp_id = CORP_NAMES.index("PR")
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    TURN.set_active_player(state, 0)
    TURN.set_active_corp(state, corp_id)

    actual_companies = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_corp(state, corp_id)
        or COMPANIES[cid].is_in_corp_acquisition(state, corp_id)
    )
    actual_offering = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].get_location(state)
        in (int(CompanyLocation.LOC_AUCTION), int(CompanyLocation.LOC_REVEALED))
    )
    fi_companies = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_fi(state)
    )

    session = GameSession(3)
    session._player_ids = [101, 202, 303]
    session._last_extract_record = {
        "action_id": 99,
        "active_corp": "PR",
        "players": [],
        "corporations": [{
            "name": "PR",
            "floated": True,
            "price": CORPS[corp_id].get_share_price(state) + 2,
            "cash": CORPS[corp_id].get_cash(state),
            "companies": actual_companies,
            "shares_in_market": CORPS[corp_id].get_bank_shares(state),
        }],
        "foreign_investor": {
            "cash": FI.get_cash(state),
            "companies": fi_companies,
        },
        "offering": actual_offering,
        "cost_level": TURN.get_coo_level(state),
    }

    mismatches = session.validate_against_18xx(
        {"round": "Issue Shares"},
        state,
        context="unit",
    )

    assert any(
        mismatch.field == "corp[PR].price"
        and mismatch.expected == CORPS[corp_id].get_share_price(state) + 2
        and mismatch.actual == CORPS[corp_id].get_share_price(state)
        for mismatch in mismatches
    )


def test_live_state_validation_reports_corp_president_mismatch():
    state = _new_state()
    corp_id = CORP_NAMES.index("PR")
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    TURN.set_active_player(state, 0)
    TURN.set_active_corp(state, corp_id)

    actual_companies = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_corp(state, corp_id)
        or COMPANIES[cid].is_in_corp_acquisition(state, corp_id)
    )
    actual_offering = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].get_location(state)
        in (int(CompanyLocation.LOC_AUCTION), int(CompanyLocation.LOC_REVEALED))
    )
    fi_companies = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_fi(state)
    )

    session = GameSession(3)
    session._player_ids = [101, 202, 303]
    session._last_extract_record = {
        "action_id": 99,
        "active_corp": "PR",
        "players": [],
        "corporations": [{
            "name": "PR",
            "floated": True,
            "price": CORPS[corp_id].get_share_price(state),
            "cash": CORPS[corp_id].get_cash(state),
            "companies": actual_companies,
            "shares_in_market": CORPS[corp_id].get_bank_shares(state),
            "president_id": 202,
        }],
        "foreign_investor": {
            "cash": FI.get_cash(state),
            "companies": fi_companies,
        },
        "offering": actual_offering,
        "cost_level": TURN.get_coo_level(state),
    }

    mismatches = session.validate_against_18xx(
        {"round": "Issue Shares"},
        state,
        context="unit",
    )

    assert any(
        mismatch.field == "corp[PR].president"
        and mismatch.expected == 1
        and mismatch.actual == 0
        for mismatch in mismatches
    )


def test_live_state_validation_uses_acting_set_for_closing():
    state = _new_state()
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    actual_offering = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].get_location(state)
        in (int(CompanyLocation.LOC_AUCTION), int(CompanyLocation.LOC_REVEALED))
    )
    fi_companies = sorted(
        COMPANY_NAMES[cid]
        for cid in range(len(COMPANY_NAMES))
        if COMPANIES[cid].is_owned_by_fi(state)
    )

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    session._last_extract_record = {
        "action_id": 137,
        "active_player": 2,
        "players": [],
        "corporations": [],
        "foreign_investor": {
            "cash": FI.get_cash(state),
            "companies": fi_companies,
        },
        "offering": actual_offering,
        "cost_level": TURN.get_coo_level(state),
    }

    mismatches = session.validate_against_18xx(
        {"round": "Closing", "acting": [4]},
        state,
        context="unit",
    )

    assert not any(mismatch.field == "active_player" for mismatch in mismatches)

    mismatches = session.validate_against_18xx(
        {"round": "Closing", "acting": [3]},
        state,
        context="unit",
    )

    assert any(
        mismatch.field == "active_player"
        and mismatch.expected == [1]
        and mismatch.actual == 0
        for mismatch in mismatches
    )


def test_live_state_validation_prefers_extractor_actor_for_closing():
    state = _new_state()
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 1)

    session = GameSession(3)
    session._player_ids = [4, 3, 2]
    session._last_extract_record = {
        "action_id": 137,
        "current_round": "CLO",
        "active_player": 3,
        "players": [],
        "corporations": [],
        "foreign_investor": {},
        "offering": [],
    }

    mismatches = session.validate_against_18xx(
        {"round": "Acquisition", "acting": [4]},
        state,
        context="unit",
    )

    assert not any(mismatch.field == "active_player" for mismatch in mismatches)
