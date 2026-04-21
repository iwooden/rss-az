from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.games_18xx import replay_harness
from tests.games_18xx.replay_harness import ReplayHarness, load_ref_states
from utils_18xx import replay_state

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
    monkeypatch.setattr(replay_harness, "TURN", SimpleNamespace(get_phase=lambda state: replay_harness.PHASE_ACQ_SELECT_CORP))
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


def test_apply_composite_simple_action_records_mapping_mismatch_when_later_phase_stops_mapping(monkeypatch):
    harness = make_harness()
    state = SimpleNamespace(phase=replay_harness.PHASE_INVEST)
    action = {"id": 10, "type": "bid", "entity": 101}

    def fake_map_action(current_state, current_action, phase, layout):
        if phase == replay_harness.PHASE_INVEST:
            return 7
        if phase == replay_harness.PHASE_BID:
            return None
        raise AssertionError(f"unexpected phase {phase}")

    def fake_apply_action(current_state, action_id):
        assert action_id == 7
        current_state.phase = replay_harness.PHASE_BID
        return 0

    monkeypatch.setattr(replay_harness, "map_action", fake_map_action)
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(replay_harness, "DRIVER", SimpleNamespace(apply_action=fake_apply_action))
    monkeypatch.setattr(replay_harness, "settle_to_player_choice", lambda current_state: None)

    harness._apply_composite_simple_action(
        state,
        action,
        layout=object(),
        action_id=action["id"],
        phase_group=(replay_harness.PHASE_INVEST, replay_harness.PHASE_BID),
    )

    assert len(harness.mismatches) == 1
    mismatch = harness.mismatches[0]
    assert mismatch.action_id == 10
    assert mismatch.phase == "BID"
    assert mismatch.field == "action_mapping"
    assert mismatch.actual == "None"


def test_apply_composite_simple_action_accepts_bid_step_already_auto_consumed(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [101, 202]
    state = SimpleNamespace(
        phase=replay_harness.PHASE_INVEST,
        auction_price=0,
        auction_high_bidder=-1,
        active_player=0,
    )
    action = {"id": 11, "type": "bid", "entity": 101, "price": 23}

    def fake_map_action(current_state, current_action, phase, layout):
        if phase == replay_harness.PHASE_INVEST:
            return 8
        if phase == replay_harness.PHASE_BID:
            return None
        raise AssertionError(f"unexpected phase {phase}")

    def fake_apply_action(current_state, action_id):
        assert action_id == 8
        current_state.phase = replay_harness.PHASE_BID
        current_state.auction_price = 23
        current_state.auction_high_bidder = 0
        current_state.active_player = 1
        return 0

    monkeypatch.setattr(replay_harness, "map_action", fake_map_action)
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(
            get_phase=lambda current_state: current_state.phase,
            get_active_player=lambda current_state: current_state.active_player,
            get_auction_price=lambda current_state: current_state.auction_price,
            get_auction_high_bidder=lambda current_state: current_state.auction_high_bidder,
        ),
    )
    monkeypatch.setattr(replay_harness, "DRIVER", SimpleNamespace(apply_action=fake_apply_action))
    monkeypatch.setattr(replay_harness, "settle_to_player_choice", lambda current_state: None)

    harness._apply_composite_simple_action(
        state,
        action,
        layout=object(),
        action_id=action["id"],
        phase_group=(replay_harness.PHASE_INVEST, replay_harness.PHASE_BID),
    )

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


def test_offer_has_live_path_rejects_offer_that_only_maps_corp_select_prefix(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [391, 12363, 21630]
    state = SimpleNamespace(_array=SimpleNamespace(copy=lambda: [1, 2, 3]), phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    probe = SimpleNamespace(phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    offer = {"id": 108, "type": "offer", "entity": 391, "corporation": "SM", "company": "B", "price": 32}

    monkeypatch.setattr(
        replay_harness,
        "GameState",
        SimpleNamespace(from_array=lambda array, num_players: probe),
        raising=False,
    )
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(harness, "_offer_already_on_buyer_acq_pile", lambda current_state, action: False)
    monkeypatch.setattr(
        harness,
        "_try_map_action",
        lambda current_state, action, layout: 4
        if current_state.phase == replay_harness.PHASE_ACQ_SELECT_CORP
        else None,
    )
    monkeypatch.setattr(
        replay_harness,
        "DRIVER",
        SimpleNamespace(
            apply_action=lambda current_state, action_id: setattr(
                current_state,
                "phase",
                replay_harness.PHASE_ACQ_SELECT_COMPANY,
            )
            or 0
        ),
    )

    assert harness._offer_has_live_path(state, offer, object()) is False


def test_offer_has_live_path_accepts_offer_already_on_buyer_acq_pile(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [391, 12363, 21630]
    state = SimpleNamespace(_array=SimpleNamespace(copy=lambda: [1, 2, 3]), phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    probe = SimpleNamespace(phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    offer = {"id": 59, "type": "offer", "entity": 18048, "corporation": "DA", "company": "HE", "price": 18}

    monkeypatch.setattr(replay_harness, "COMPANY_NAME_TO_ID", {"HE": 0})
    monkeypatch.setattr(replay_harness, "CORP_NAME_TO_ID", {"DA": 3})
    monkeypatch.setattr(
        replay_harness,
        "COMPANIES",
        [SimpleNamespace(get_location=lambda current_state: replay_harness.CompanyLocation.LOC_CORP_ACQ, get_owner_id=lambda current_state: 3)],
    )
    monkeypatch.setattr(
        replay_harness,
        "GameState",
        SimpleNamespace(from_array=lambda array, num_players: probe),
        raising=False,
    )
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(
        harness,
        "_try_map_action",
        lambda current_state, action, layout: 6 if current_state is state else None,
    )

    assert harness._offer_has_live_path(state, offer, object()) is True


def test_offer_has_live_path_accepts_live_company_select_mapping_without_clone(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [391, 12363, 21630]
    state = SimpleNamespace(_array=SimpleNamespace(copy=lambda: [1, 2, 3]), phase=replay_harness.PHASE_ACQ_SELECT_COMPANY)
    probe = SimpleNamespace(phase=replay_harness.PHASE_ACQ_SELECT_COMPANY)
    offer = {"id": 291, "type": "offer", "entity": 17937, "corporation": "VM", "company": "CDG", "price": 40}

    monkeypatch.setattr(
        replay_harness,
        "GameState",
        SimpleNamespace(from_array=lambda array, num_players: probe),
        raising=False,
    )
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(harness, "_offer_already_on_buyer_acq_pile", lambda current_state, action: False)
    monkeypatch.setattr(
        harness,
        "_try_map_action",
        lambda current_state, action, layout: 35 if current_state is state else None,
    )

    assert harness._offer_has_live_path(state, offer, object()) is True


def test_offer_has_live_path_accepts_offer_when_full_acq_path_maps(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [391, 12363, 21630]
    state = SimpleNamespace(_array=SimpleNamespace(copy=lambda: [1, 2, 3]), phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    probe = SimpleNamespace(phase=replay_harness.PHASE_ACQ_SELECT_CORP)
    offer = {"id": 112, "type": "offer", "entity": 12363, "corporation": "VM", "company": "DSB", "price": 26}

    monkeypatch.setattr(
        replay_harness,
        "GameState",
        SimpleNamespace(from_array=lambda array, num_players: probe),
        raising=False,
    )
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(harness, "_offer_already_on_buyer_acq_pile", lambda current_state, action: False)

    def fake_try_map_action(current_state, action, layout):
        phase = current_state.phase
        if phase == replay_harness.PHASE_ACQ_SELECT_CORP:
            return 7
        if phase == replay_harness.PHASE_ACQ_SELECT_COMPANY:
            return 14
        if phase == replay_harness.PHASE_ACQ_SELECT_PRICE:
            return 16
        return None

    def fake_apply_action(current_state, action_id):
        if current_state.phase == replay_harness.PHASE_ACQ_SELECT_CORP:
            current_state.phase = replay_harness.PHASE_ACQ_SELECT_COMPANY
        elif current_state.phase == replay_harness.PHASE_ACQ_SELECT_COMPANY:
            current_state.phase = replay_harness.PHASE_ACQ_SELECT_PRICE
        else:
            current_state.phase = replay_harness.PHASE_ACQ_SELECT_CORP
        return 0

    monkeypatch.setattr(harness, "_try_map_action", fake_try_map_action)
    monkeypatch.setattr(replay_harness, "DRIVER", SimpleNamespace(apply_action=fake_apply_action))

    assert harness._offer_has_live_path(state, offer, object()) is True


def test_clone_offer_probe_state_preserves_replay_flags(monkeypatch):
    harness = make_harness()
    harness._engine_index_to_player_id = [391, 12363, 21630]
    state = SimpleNamespace(
        _array=SimpleNamespace(copy=lambda: [1, 2, 3]),
        step_mode=True,
        acq_same_president=False,
        allow_positive_income_closing=True,
    )
    probe = SimpleNamespace(
        step_mode=False,
        acq_same_president=True,
        allow_positive_income_closing=False,
    )

    monkeypatch.setattr(
        replay_harness,
        "GameState",
        SimpleNamespace(from_array=lambda array, num_players: probe),
        raising=False,
    )

    cloned = harness._clone_offer_probe_state(state)

    assert cloned is probe
    assert cloned.step_mode is True
    assert cloned.acq_same_president is False
    assert cloned.allow_positive_income_closing is True


def test_initialize_replay_state_enables_positive_income_closing(monkeypatch):
    state = SimpleNamespace(
        allow_positive_income_closing=False,
        step_mode=None,
    )

    class FakeGameState:
        def __init__(self, num_players, acq_same_president=True):
            assert num_players == 3
            assert acq_same_president is False
            self._state = state

        def initialize_game(self, num_players, seed):
            assert num_players == 3
            assert seed == 42

        def __setattr__(self, name, value):
            if name == "_state":
                object.__setattr__(self, name, value)
            else:
                setattr(self._state, name, value)

        def __getattr__(self, name):
            return getattr(self._state, name)

    monkeypatch.setattr(replay_state, "GameState", FakeGameState)
    monkeypatch.setattr(replay_state, "override_deck_and_offering", lambda *args, **kwargs: None)

    initialized = replay_state.initialize_replay_state(3, ["BPM"], ["BPM"], step_mode=True)

    assert initialized.allow_positive_income_closing is True
    assert initialized.step_mode is True


def test_replay_acquisition_offer_passes_until_offer_maps(monkeypatch):
    state = SimpleNamespace(phase=replay_state.PHASE_ACQ_SELECT_CORP, allow_offer=False)
    phases_seen = []

    monkeypatch.setattr(
        replay_state,
        "TURN",
        SimpleNamespace(get_phase=lambda current_state: current_state.phase),
    )
    monkeypatch.setattr(
        replay_state,
        "CORPS",
        [SimpleNamespace(name="OS")],
    )
    monkeypatch.setattr(
        replay_state,
        "COMPANIES",
        [SimpleNamespace(name="KK")],
    )
    monkeypatch.setattr(replay_state, "settle_to_player_choice", lambda current_state: None)

    def fake_map_action(current_state, action, phase, layout):
        phases_seen.append((phase, current_state.allow_offer))
        if not current_state.allow_offer:
            raise ValueError("not current buyer turn yet")
        assert action == {"type": "offer", "corporation": "OS", "company": "KK", "price": 20}
        return 17

    def fake_find_legal_action(current_state, *, action_type, **kwargs):
        if action_type == replay_state.ACTION_PASS:
            return 3
        if action_type == replay_state.ACTION_ACQ_OFFER_ACCEPT:
            return 9
        raise AssertionError(f"unexpected action_type {action_type}")

    def fake_apply_action(current_state, action_id):
        if action_id == 3:
            current_state.allow_offer = True
            return replay_state.STATUS_OK
        if action_id == 17:
            current_state.phase = replay_state.PHASE_ACQ_OFFER
            return replay_state.STATUS_OK
        if action_id == 9:
            current_state.phase = replay_state.PHASE_CLOSING
            return replay_state.STATUS_OK
        raise AssertionError(f"unexpected action_id {action_id}")

    monkeypatch.setattr(replay_state, "map_action", fake_map_action)
    monkeypatch.setattr(replay_state, "find_legal_action", fake_find_legal_action)
    monkeypatch.setattr(replay_state, "DRIVER", SimpleNamespace(apply_action=fake_apply_action))

    assert replay_state.replay_acquisition_offer(
        state,
        layout=object(),
        buyer_corp_id=0,
        company_id=0,
        price=20,
        accept=True,
    ) is True
    assert phases_seen[0] == (replay_state.PHASE_ACQ_SELECT_CORP, False)
    assert phases_seen[1] == (replay_state.PHASE_ACQ_SELECT_CORP, True)


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


def test_is_implicit_fi_offer_accepts_buyer_president_fi_purchase(monkeypatch):
    harness = make_harness()
    state = object()
    action = {"type": "offer", "entity": 101, "corporation": "DA", "company": "HE"}

    monkeypatch.setattr(replay_harness, "COMPANY_NAME_TO_ID", {"HE": 0})
    monkeypatch.setattr(replay_harness, "CORP_NAME_TO_ID", {"DA": 0})
    monkeypatch.setattr(
        replay_harness,
        "COMPANIES",
        [SimpleNamespace(get_location=lambda state: replay_harness.CompanyLocation.LOC_FI, get_owner_id=lambda state: -1)],
    )
    monkeypatch.setattr(
        replay_harness,
        "CORPS",
        [SimpleNamespace(get_president_id=lambda state: 0)],
    )
    harness._engine_index_to_player_id = [101]

    assert harness._is_implicit_fi_offer(state, action) is True


def test_find_externalizable_offer_before_pass_allows_implicit_fi_offer(monkeypatch):
    harness = make_harness()
    state = object()
    remaining_actions = [
        {"id": 59, "type": "offer", "entity": 18048, "corporation": "DA", "company": "HE", "price": 18},
        {"id": 60, "type": "pass", "entity": 18048},
    ]

    monkeypatch.setattr(harness, "_is_self_player_corp_offer", lambda _state, action: False)
    monkeypatch.setattr(harness, "_is_implicit_fi_offer", lambda _state, action: action.get("id") == 59)
    monkeypatch.setattr(harness, "_external_offer_matches_ref_buyer", lambda *args, **kwargs: True)
    monkeypatch.setattr(harness, "_try_map_action", lambda *args, **kwargs: None)

    candidate = harness._find_externalizable_offer_before_pass(
        state,
        remaining_actions,
        pass_idx=1,
        explicit_responses={},
        layout=object(),
        ref_by_action={},
    )

    assert candidate == 0


def test_finish_unmappable_pending_offer_removes_stale_response_from_remaining_actions(monkeypatch):
    harness = make_harness()
    state = object()
    pending_offer = {"id": 289, "type": "offer", "corporation": "S", "company": "RENFE", "price": 44}
    response_action = {"id": 291, "type": "respond", "corporation": "S", "company": "RENFE", "accept": True}
    remaining_actions = [response_action, {"id": 300, "type": "pass", "entity": 101}]
    explicit_responses = {289: dict(response_action)}

    monkeypatch.setattr(harness, "_external_offer_matches_ref_buyer", lambda *args, **kwargs: True)

    def fake_apply_external_acquisition_offer(current_state, offer, current_responses):
        current_responses.pop(offer["id"], None)
        return True

    monkeypatch.setattr(harness, "_apply_external_acquisition_offer", fake_apply_external_acquisition_offer)
    monkeypatch.setattr(
        replay_harness,
        "TURN",
        SimpleNamespace(
            clear_acquisition_context=lambda current_state: None,
            set_phase=lambda current_state, phase: None,
        ),
    )

    handled = harness._finish_unmappable_pending_offer(
        state,
        pending_offer,
        explicit_responses,
        ref_by_action={},
        remaining_actions=remaining_actions,
    )

    assert handled is True
    assert explicit_responses == {}
    assert [action["id"] for action in remaining_actions] == [300]


def test_replay_regression_204324():
    game_json = DATA_DIR / "204324.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_205830():
    game_json = DATA_DIR / "205830.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


def test_replay_regression_222206():
    game_json = DATA_DIR / "222206.json"
    ref_states = load_ref_states(str(game_json))
    harness = ReplayHarness(
        game_json_path=str(game_json),
        ref_states=ref_states,
        verbose=False,
    )

    assert harness.run() == []


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
