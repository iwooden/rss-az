import pytest

from core.data import COMPANY_NAME_TO_ID, CORP_NAME_TO_ID, GamePhases
from core.state import GameState
from entities.corp import CORPS
from entities.turn import TURN
from tests.phases.conftest import float_corp_for_test
from tests.phases.helpers.ownership import give_company_to_player
from utils_18xx.live import (
    _LiveActionComposer,
    _acquisition_compatibility_action,
    _align_unordered_round_to_18xx_actor,
    _build_share_ownership,
    _closing_compatibility_action,
    _dividend_compatibility_action,
    _resolve_buyable_share,
    _resolve_issuable_share,
    _resolve_sellable_share,
    _retarget_closing_active_player_to_bot,
    _should_continue_after_postable_action,
    prepare_live_decision_state,
)


def _game_data():
    return {
        "players": [
            {"id": 101, "name": "bot"},
            {"id": 202, "name": "other"},
        ],
        "actions": [],
    }


class _FakeSession:
    def __init__(self, offer=None, player_ids=None, active_corp=None):
        self.offer = offer
        self.player_ids = player_ids or [101, 202]
        self._last_extract_record = {}
        if active_corp is not None:
            self._last_extract_record["active_corp"] = active_corp

    def pending_offer_for_user_id(self, user_id):
        if self.offer and str(self.offer["responder_id"]) == str(user_id):
            return self.offer
        return None

    def player_index_for_user_id(self, user_id):
        for idx, player_id in enumerate(self.player_ids):
            if str(player_id) == str(user_id):
                return idx
        raise ValueError(user_id)


def test_composer_combines_invest_select_and_bid_price():
    composer = _LiveActionComposer(_game_data(), bot_player_idx=0, committed_ids=set())

    composer.add_step(
        GamePhases.PHASE_INVEST,
        {"type": "bid", "company": "MHE", "price": 5},
    )
    composer.add_step(
        GamePhases.PHASE_BID,
        {"type": "bid", "company": "MHE", "price": 8},
    )

    assert composer.finish() == [{
        "type": "bid",
        "entity": 101,
        "entity_type": "player",
        "company": "MHE",
        "price": 8,
    }]


def test_composer_combines_ipo_and_par():
    composer = _LiveActionComposer(_game_data(), bot_player_idx=0, committed_ids=set())

    composer.add_step(
        GamePhases.PHASE_IPO,
        {"type": "ipo_select", "company": "MHE", "corporation": "PR"},
    )
    composer.add_step(
        GamePhases.PHASE_PAR,
        {"type": "par_price", "share_price": "12,0,3", "par_price": 12},
    )

    assert composer.finish() == [{
        "type": "par",
        "entity": "MHE",
        "entity_type": "company",
        "corporation": "PR",
        "share_price": "12,0,3",
    }]


def test_composer_combines_acquisition_selection_and_price():
    composer = _LiveActionComposer(_game_data(), bot_player_idx=0, committed_ids=set())

    composer.add_step(
        GamePhases.PHASE_ACQ_SELECT_CORP,
        {"type": "select_corp", "corporation": "PR"},
    )
    composer.add_step(
        GamePhases.PHASE_ACQ_SELECT_COMPANY,
        {"type": "select_company", "corporation": "PR", "company": "MHE"},
    )
    composer.add_step(
        GamePhases.PHASE_ACQ_SELECT_PRICE,
        {"type": "offer", "corporation": "PR", "company": "MHE", "price": 12},
    )

    assert composer.finish() == [{
        "type": "offer",
        "entity": 101,
        "entity_type": "player",
        "corporation": "PR",
        "company": "MHE",
        "price": 12,
    }]


def test_composer_allows_fi_offer_without_price_phase():
    composer = _LiveActionComposer(_game_data(), bot_player_idx=0, committed_ids=set())

    composer.add_step(
        GamePhases.PHASE_ACQ_SELECT_CORP,
        {"type": "select_corp", "corporation": "OS"},
    )
    composer.add_step(
        GamePhases.PHASE_ACQ_SELECT_COMPANY,
        {"type": "offer", "corporation": "OS", "company": "MHE", "price": 5},
    )

    assert composer.finish() == [{
        "type": "offer",
        "entity": 101,
        "entity_type": "player",
        "corporation": "OS",
        "company": "MHE",
        "price": 5,
    }]


def test_acquisition_compatibility_rejects_pending_18xx_offer():
    state = GameState(2)
    action = _acquisition_compatibility_action(
        {"round": "Acquisition", "acting": [202]},
        _FakeSession({
            "responder_id": 202,
            "corporation": "PR",
            "company": "OL",
        }),
        state,
        bot_user_id=202,
        engine_player_idx=1,
    )

    assert action == {
        "type": "respond",
        "entity": 202,
        "entity_type": "player",
        "corporation": "PR",
        "company": "OL",
        "accept": "false",
    }


def test_acquisition_compatibility_passes_when_rss_phase_has_advanced():
    state = GameState(2)
    action = _acquisition_compatibility_action(
        {"round": "Acquisition", "acting": [101]},
        _FakeSession(),
        state,
        bot_user_id=101,
        engine_player_idx=0,
    )

    assert action == {
        "type": "pass",
        "entity": 101,
        "entity_type": "player",
    }


def test_composer_rejects_incomplete_split_action():
    composer = _LiveActionComposer(_game_data(), bot_player_idx=0, committed_ids=set())
    composer.add_step(
        GamePhases.PHASE_IPO,
        {"type": "ipo_select", "company": "MHE", "corporation": "PR"},
    )

    with pytest.raises(ValueError, match="Incomplete IPO/PAR"):
        composer.finish()


def test_composer_uses_user_id_override_for_engine_player_index():
    composer = _LiveActionComposer(
        _game_data(),
        bot_player_idx=0,
        committed_ids=set(),
        bot_user_id=202,
    )

    composer.add_step(
        GamePhases.PHASE_INVEST,
        {"type": "bid", "company": "MHE", "price": 5},
    )
    composer.add_step(
        GamePhases.PHASE_BID,
        {"type": "bid", "company": "MHE", "price": 8},
    )

    assert composer.finish()[0]["entity"] == 202


def test_prepare_live_decision_state_restores_model_acquisition_rule():
    state = GameState(3, acq_same_president=False)
    state.initialize_game(3, seed=42)
    state.allow_positive_income_closing = True

    prepare_live_decision_state(state)

    assert state.acq_same_president is True
    assert state.allow_positive_income_closing is False


def test_unordered_round_alignment_consumes_nonacting_closing_pass():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 1)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    applied = _align_unordered_round_to_18xx_actor(
        {"acting": [202]},
        _FakeSession(player_ids=[101, 202, 303]),
        state,
        bot_player_indices={1},
    )

    assert applied == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(state) == 1


def test_unordered_round_alignment_does_not_pass_bot_player():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 1)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    applied = _align_unordered_round_to_18xx_actor(
        {"acting": [202]},
        _FakeSession(player_ids=[101, 202, 303]),
        state,
        bot_player_indices={0, 1},
    )

    assert applied == 0
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(state) == 0


def test_closing_compatibility_passes_when_model_phase_advanced():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_INCOME))
    TURN.set_active_player(state, 0)

    action = _closing_compatibility_action(
        {"round": "Closing", "acting": [303]},
        state,
        bot_user_id=303,
        engine_player_idx=2,
    )

    assert action == {
        "type": "pass",
        "entity": 303,
        "entity_type": "player",
    }


def test_live_planning_continues_same_player_acquisition_batch():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 2)

    assert _should_continue_after_postable_action(
        GamePhases.PHASE_ACQ_SELECT_PRICE,
        state,
        bot_player_idx=2,
    )

    TURN.set_active_player(state, 0)
    assert not _should_continue_after_postable_action(
        GamePhases.PHASE_ACQ_SELECT_PRICE,
        state,
        bot_player_idx=2,
    )


def test_live_planning_continues_same_player_closing_batch():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 1)

    assert _should_continue_after_postable_action(
        GamePhases.PHASE_CLOSING,
        state,
        bot_player_idx=1,
    )

    TURN.set_active_player(state, 2)
    assert not _should_continue_after_postable_action(
        GamePhases.PHASE_CLOSING,
        state,
        bot_player_idx=1,
    )


def test_closing_retarget_points_ordered_turn_at_acting_bot():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    changed = _retarget_closing_active_player_to_bot(
        {"round": "Closing", "acting": [101, 303]},
        state,
        bot_user_id=303,
        engine_player_idx=2,
    )

    assert changed
    assert TURN.get_active_player(state) == 2


def test_closing_retarget_ignores_nonacting_bot():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    changed = _retarget_closing_active_player_to_bot(
        {"round": "Closing", "acting": [101]},
        state,
        bot_user_id=303,
        engine_player_idx=2,
    )

    assert not changed
    assert TURN.get_active_player(state) == 0


def test_dividend_compatibility_posts_already_satisfied_single_choice():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["DA"]
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    CORPS[corp_id].set_cash(state, 0)
    TURN.set_dividend_remaining(state, corp_id, False)
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    TURN.set_active_player(state, 1)

    action = _dividend_compatibility_action(
        {"round": "Dividends", "acting": [101]},
        _FakeSession(active_corp="DA"),
        state,
        bot_user_id=101,
        engine_player_idx=0,
    )

    assert action == {
        "type": "dividend",
        "entity": "DA",
        "entity_type": "corporation",
        "kind": "variable",
        "amount": 0,
    }


def test_dividend_compatibility_ignores_normal_active_dividend_choice():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["DA"]
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    TURN.set_phase(state, int(GamePhases.PHASE_DIVIDENDS))
    TURN.set_active_player(state, 0)
    TURN.set_active_corp(state, corp_id)
    TURN.set_dividend_remaining(state, corp_id, True)

    action = _dividend_compatibility_action(
        {"round": "Dividends", "acting": [101]},
        _FakeSession(active_corp="DA"),
        state,
        bot_user_id=101,
        engine_player_idx=0,
    )

    assert action is None


def test_share_ledger_resolves_issued_and_treasury_share_ids():
    game_data = {
        "players": [{"id": 4, "name": "rss-az-3"}],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "BPM",
                "entity_type": "company",
                "corporation": "DA",
                "share_price": "11,0,7",
                "user": 4,
            },
        ],
    }

    assert _resolve_buyable_share(game_data, "DA", {1}) == "DA_1"
    assert _resolve_issuable_share(game_data, "DA", {1}) == "DA_2"

    game_data["actions"].append(
        {
            "id": 2,
            "type": "buy_shares",
            "entity": 4,
            "entity_type": "player",
            "shares": ["DA_1"],
            "share_price": 11,
            "user": 4,
        }
    )

    assert _resolve_issuable_share(game_data, "DA", {1, 2}) == "DA_2"

    game_data["actions"].append(
        {
            "id": 3,
            "type": "sell_shares",
            "entity": "DA",
            "entity_type": "corporation",
            "shares": ["DA_2"],
            "share_price": 10,
            "user": 4,
        }
    )

    assert _resolve_buyable_share(game_data, "DA", {1, 2, 3}) == "DA_2"
    assert _resolve_issuable_share(game_data, "DA", {1, 2, 3}) == "DA_3"


def test_share_ledger_tracks_presidency_share_swaps():
    game_data = {
        "players": [
            {"id": 3, "name": "seller"},
            {"id": 2, "name": "buyer"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "MS",
                "entity_type": "company",
                "corporation": "SM",
                "share_price": "14,0,10",
                "user": 3,
            },
            {
                "id": 2,
                "type": "sell_shares",
                "entity": "SM",
                "entity_type": "corporation",
                "shares": ["SM_4"],
                "share_price": 12,
                "user": 3,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["SM_1"],
                "share_price": 12,
                "user": 3,
            },
            {
                "id": 4,
                "type": "buy_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SM_1"],
                "share_price": 10,
                "user": 2,
            },
            {
                "id": 5,
                "type": "buy_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SM_2"],
                "share_price": 12,
                "user": 2,
            },
        ],
    }

    _, owners = _build_share_ownership(game_data, {1, 2, 3, 4, 5})

    assert owners["2"]["SM"] == [0, 2]
    assert owners["3"]["SM"] == [1]
    assert _resolve_sellable_share(game_data, "SM", 3, {1, 2, 3, 4, 5}) == "SM_1"


def test_share_ledger_does_not_reacquire_sold_president_share_on_tie():
    game_data = {
        "players": [
            {"id": 3, "name": "seller"},
            {"id": 2, "name": "buyer"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "MS",
                "entity_type": "company",
                "corporation": "SM",
                "share_price": "14,0,10",
                "user": 3,
            },
            {
                "id": 2,
                "type": "sell_shares",
                "entity": "SM",
                "entity_type": "corporation",
                "shares": ["SM_4"],
                "share_price": 12,
                "user": 3,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["SM_1"],
                "share_price": 12,
                "user": 3,
            },
            {
                "id": 4,
                "type": "buy_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SM_1"],
                "share_price": 10,
                "user": 2,
            },
            {
                "id": 5,
                "type": "buy_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SM_2"],
                "share_price": 12,
                "user": 2,
            },
            {
                "id": 6,
                "type": "sell_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["SM_0"],
                "share_price": 14,
                "user": 3,
            },
            {
                "id": 7,
                "type": "buy_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["SM_0"],
                "share_price": 12,
                "user": 3,
            },
        ],
    }

    pool, owners = _build_share_ownership(game_data, set(range(1, 8)))

    assert pool["SM"] == [3, 4]
    assert owners["2"]["SM"] == [2]
    assert owners["3"]["SM"] == [0, 1]
    assert _resolve_sellable_share(game_data, "SM", 2, set(range(1, 8))) == "SM_2"


def test_share_ledger_repar_buys_market_shares_before_treasury():
    game_data = {
        "players": [
            {"id": 4, "name": "rss-az-3"},
            {"id": 3, "name": "rss-az-2"},
            {"id": 2, "name": "rss-az-1"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "AKE",
                "entity_type": "company",
                "corporation": "PR",
                "share_price": "12,0,8",
                "user": 2,
            },
            {
                "id": 2,
                "type": "buy_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["PR_1"],
                "share_price": 12,
                "user": 3,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": 3,
                "entity_type": "player",
                "shares": ["PR_1"],
                "share_price": 14,
                "user": 3,
            },
            {
                "id": 4,
                "type": "sell_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["PR_0"],
                "share_price": 10,
                "user": 2,
            },
            {
                "id": 5,
                "type": "par",
                "entity": "E",
                "entity_type": "company",
                "corporation": "PR",
                "share_price": "24,0,15",
                "user": 4,
            },
        ],
    }

    pool, owners = _build_share_ownership(game_data, set(range(1, 6)))

    assert owners["4"]["PR"] == [0, 1]
    assert pool["PR"] == [2, 3]
    assert _resolve_sellable_share(game_data, "PR", 4, set(range(1, 6))) == "PR_1"
