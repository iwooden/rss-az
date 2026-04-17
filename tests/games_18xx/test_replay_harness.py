from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.games_18xx import replay_harness
from tests.games_18xx.replay_harness import ReplayHarness, load_ref_states

DATA_DIR = Path(__file__).with_name("data")


def make_harness() -> ReplayHarness:
    harness = ReplayHarness(game_json_path="unused", ref_states=[])
    harness._engine_index_to_player_id = [101]
    harness._last_ref = {"action_id": 0}
    harness.mismatches = []
    return harness


def test_stop_round_if_already_satisfied_sets_last_ref_and_returns_true(monkeypatch):
    harness = make_harness()
    round_end_ref = {"action_id": 145}

    monkeypatch.setattr(harness, "_matches_ref", lambda state, ref, context: True)

    assert harness._stop_round_if_already_satisfied(object(), round_end_ref, "round complete") is True
    assert harness._last_ref is round_end_ref


def test_finalize_replay_state_skips_drain_when_final_ref_already_satisfied(monkeypatch):
    harness = make_harness()
    state = object()
    layout = object()
    final_ref = {"action_id": 145}
    events: list[object] = []

    monkeypatch.setattr(replay_harness, "settle_to_player_choice", lambda state: events.append("settle"))
    monkeypatch.setattr(replay_harness, "drain_offer_phases", lambda state, layout: events.append("drain"))
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda state: replay_harness.PHASE_CLOSING),
    )
    monkeypatch.setattr(harness, "_matches_ref", lambda state, ref, context: True)
    monkeypatch.setattr(harness, "_compare_state", lambda state, ref, context: events.append(("compare", context)))

    harness._finalize_replay_state(state, layout, final_ref)

    assert "drain" not in events
    assert ("compare", "final") in events


def test_collect_acquisition_responses_matches_specific_offer_without_mutating_input():
    round_actions = [
        {"id": 10, "type": "offer", "corporation": "OS", "company": "KK", "price": 20},
        {"id": 11, "type": "offer", "corporation": "OS", "company": "SJ", "price": 21},
        {"id": 12, "type": "respond", "corporation": "OS", "company": "KK", "accept": "true"},
        {"id": 13, "type": "respond", "corporation": "OS", "accept": "false"},
    ]

    responses = ReplayHarness._collect_acquisition_responses(round_actions)

    assert responses[10]["id"] == 12
    assert responses[10]["accept"] is True
    assert responses[11]["id"] == 13
    assert responses[11]["accept"] is False
    assert round_actions[2]["accept"] == "true"
    assert round_actions[3]["accept"] == "false"


@pytest.mark.parametrize(
    ("map_behavior", "driver_result"),
    [
        (ValueError("boom"), None),
        (None, None),
        (1234, replay_harness.STATUS_INVALID),
    ],
    ids=["map-error", "map-none", "status-invalid"],
)
def test_apply_declined_offer_if_mappable_skips_nondurable_failure_paths(monkeypatch, map_behavior, driver_result):
    harness = make_harness()
    state = object()
    offer = {"id": 77, "type": "offer", "entity": 101, "corporation": "OS", "company": "KK", "price": 20}

    if isinstance(map_behavior, Exception):
        def fake_map_action(state, action, phase, layout):
            raise map_behavior
    else:
        def fake_map_action(state, action, phase, layout):
            return map_behavior

    monkeypatch.setattr(replay_harness, "map_action", fake_map_action)
    monkeypatch.setattr(replay_harness, "TURN", SimpleNamespace(get_phase=lambda state: replay_harness.PHASE_ACQ))
    monkeypatch.setattr(replay_harness, "settle_to_player_choice", lambda state: None)
    monkeypatch.setattr(replay_harness, "DRIVER", SimpleNamespace(apply_action=lambda state, action: driver_result))
    monkeypatch.setattr(
        harness,
        "_apply_offer_response",
        lambda *args, **kwargs: pytest.fail("declined offer helper should not emit a response for skipped paths"),
    )

    handled = harness._apply_declined_offer_if_mappable(state, offer, action_id=offer["id"], layout=object())

    assert handled is False
    assert harness.mismatches == []


def test_find_externalizable_offer_before_pass_skips_later_self_sale_when_earlier_offer_has_accept_response(monkeypatch):
    harness = make_harness()
    state = object()
    remaining_actions = [
        {"id": 174, "type": "offer", "entity": 459, "corporation": "S", "company": "KK", "price": 28},
        {"id": 175, "type": "respond", "entity": 391, "corporation": "S", "company": "KK", "accept": True},
        {"id": 180, "type": "offer", "entity": 459, "corporation": "S", "company": "FS", "price": 28},
        {"id": 182, "type": "pass", "entity": 16231},
    ]
    explicit_responses = {174: {"id": 175, "accept": True, "corporation": "S", "company": "KK"}}

    monkeypatch.setattr(harness, "_is_self_player_corp_offer", lambda _state, action: action.get("id") == 180)
    monkeypatch.setattr(harness, "_external_offer_matches_ref_buyer", lambda *args, **kwargs: True)
    monkeypatch.setattr(harness, "_try_map_action", lambda *args, **kwargs: None)

    candidate = harness._find_externalizable_offer_before_pass(
        state,
        remaining_actions,
        pass_idx=3,
        explicit_responses=explicit_responses,
        layout=object(),
        ref_by_action={},
    )

    assert candidate is None


def test_find_externalizable_offer_after_pass_allows_late_self_sale_without_response(monkeypatch):
    harness = make_harness()
    state = object()
    remaining_actions = [
        {"id": 288, "type": "pass", "entity": 4205},
        {"id": 289, "type": "offer", "entity": 20554, "corporation": "S", "company": "RENFE", "price": 44},
        {"id": 291, "type": "respond", "entity": 21630, "corporation": "S", "company": "RENFE", "accept": True},
        {"id": 292, "type": "offer", "entity": 21630, "corporation": "DA", "company": "SZD", "price": 15},
    ]
    explicit_responses = {289: {"id": 291, "accept": True, "corporation": "S", "company": "RENFE"}}

    monkeypatch.setattr(harness, "_is_self_player_corp_offer", lambda _state, action: action.get("id") == 292)
    monkeypatch.setattr(harness, "_external_offer_matches_ref_buyer", lambda *args, **kwargs: True)
    monkeypatch.setattr(harness, "_try_map_action", lambda *args, **kwargs: None)

    candidate = harness._find_externalizable_offer_after_pass(
        state,
        remaining_actions,
        pass_idx=0,
        explicit_responses=explicit_responses,
        layout=object(),
        ref_by_action={},
    )

    assert candidate == 3


def test_find_externalizable_offer_after_pass_skips_late_self_sale_when_earlier_offer_has_accept_response(monkeypatch):
    harness = make_harness()
    state = object()
    remaining_actions = [
        {"id": 337, "type": "offer", "entity": 391, "corporation": "S", "company": "CDG", "price": 80},
        {"id": 338, "type": "offer", "entity": 391, "corporation": "SM", "company": "BR", "price": 45},
        {"id": 339, "type": "pass", "entity": 12439},
        {"id": 340, "type": "respond", "entity": 10327, "corporation": "S", "company": "CDG", "accept": True},
        {"id": 341, "type": "pass", "entity": 10327},
        {"id": 348, "type": "offer", "entity": 391, "corporation": "SM", "company": "SZD", "price": 29},
    ]
    explicit_responses = {337: {"id": 340, "accept": True, "corporation": "S", "company": "CDG"}}

    monkeypatch.setattr(harness, "_is_self_player_corp_offer", lambda _state, action: action.get("id") == 348)
    monkeypatch.setattr(harness, "_external_offer_matches_ref_buyer", lambda *args, **kwargs: True)
    monkeypatch.setattr(harness, "_try_map_action", lambda *args, **kwargs: None)

    candidate = harness._find_externalizable_offer_after_pass(
        state,
        remaining_actions,
        pass_idx=4,
        explicit_responses=explicit_responses,
        layout=object(),
        ref_by_action={},
    )

    assert candidate is None


def test_find_decline_pass_for_pending_offer_skips_pass_when_same_player_offer_still_pending():
    harness = make_harness()
    remaining_actions = [
        {"id": 109, "type": "offer", "entity": 391, "corporation": "SM", "company": "MS", "price": 9},
        {"id": 110, "type": "offer", "entity": 391, "corporation": "SM", "company": "HE", "price": 7},
        {"id": 113, "type": "offer", "entity": 21630, "corporation": "OS", "company": "B", "price": 24},
        {"id": 127, "type": "pass", "entity": 21630},
    ]

    assert harness._find_decline_pass_for_pending_offer(remaining_actions, 21630) is None


def test_find_decline_pass_for_pending_offer_accepts_clean_matching_pass():
    harness = make_harness()
    remaining_actions = [
        {"id": 127, "type": "pass", "entity": 21630},
        {"id": 130, "type": "offer", "entity": 391, "corporation": "SM", "company": "HE", "price": 7},
    ]

    assert harness._find_decline_pass_for_pending_offer(remaining_actions, 21630) == 0


def test_is_self_player_corp_offer_accepts_same_president_corp_sale(monkeypatch):
    harness = make_harness()
    state = object()
    action = {"type": "offer", "entity": 101, "corporation": "SM", "company": "HE"}

    monkeypatch.setattr(replay_harness, "COMPANY_NAME_TO_ID", {"HE": 0})
    monkeypatch.setattr(replay_harness, "CORP_NAME_TO_ID", {"SM": 0})
    monkeypatch.setattr(
        replay_harness,
        "COMPANIES",
        [SimpleNamespace(get_location=lambda state: replay_harness.CompanyLocation.LOC_CORP, get_owner_id=lambda state: 1)],
    )
    monkeypatch.setattr(
        replay_harness,
        "CORPS",
        [
            SimpleNamespace(get_president_id=lambda state: 0),
            SimpleNamespace(get_president_id=lambda state: 0),
        ],
    )
    harness._engine_index_to_player_id = [101]

    assert harness._is_self_player_corp_offer(state, action) is True


def test_replay_regression_213897():
    game_json = DATA_DIR / "213897.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_214336():
    game_json = DATA_DIR / "214336.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_should_skip_pre_action_compare_for_first_dividend_after_closing(monkeypatch):
    harness = make_harness()
    harness._last_ref = {"action_id": 289, "round": "CLO"}
    action = {"id": 291, "type": "dividend", "entity": "SI"}

    monkeypatch.setattr(replay_harness, "TURN", SimpleNamespace(get_phase=lambda state: replay_harness.PHASE_DIVIDENDS))

    assert harness._should_skip_pre_action_compare(object(), action) is True


def test_should_not_skip_pre_action_compare_for_in_round_dividend(monkeypatch):
    harness = make_harness()
    harness._last_ref = {"action_id": 291, "round": "DIV"}
    action = {"id": 292, "type": "dividend", "entity": "DA"}

    monkeypatch.setattr(replay_harness, "TURN", SimpleNamespace(get_phase=lambda state: replay_harness.PHASE_DIVIDENDS))

    assert harness._should_skip_pre_action_compare(object(), action) is False


def test_should_skip_dividend_surface_auto_pass_when_next_real_action_is_current_corp_dividend(monkeypatch):
    harness = make_harness()
    actions = [
        {"id": -1, "type": "pass", "entity_type": "player", "entity": 12750},
        {"id": 214, "type": "dividend", "entity": "DA"},
    ]

    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(
            get_phase=lambda state: replay_harness.PHASE_DIVIDENDS,
            get_active_corp=lambda state: replay_harness.CORP_NAME_TO_ID["DA"],
        ),
    )

    assert harness._should_skip_dividend_surface_auto_pass(object(), actions, 0) is True


def test_should_not_skip_dividend_surface_auto_pass_for_different_corp(monkeypatch):
    harness = make_harness()
    actions = [
        {"id": -1, "type": "pass", "entity_type": "player", "entity": 12750},
        {"id": 214, "type": "dividend", "entity": "OS"},
    ]

    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(
            get_phase=lambda state: replay_harness.PHASE_DIVIDENDS,
            get_active_corp=lambda state: replay_harness.CORP_NAME_TO_ID["DA"],
        ),
    )

    assert harness._should_skip_dividend_surface_auto_pass(object(), actions, 0) is False


def test_replay_regression_217545():
    game_json = DATA_DIR / "217545.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_219702():
    game_json = DATA_DIR / "219702.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_224477():
    game_json = DATA_DIR / "224477.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_226209():
    game_json = DATA_DIR / "226209.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_237823():
    game_json = DATA_DIR / "237823.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_218343():
    game_json = DATA_DIR / "218343.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []
