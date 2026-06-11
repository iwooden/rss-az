import numpy as np
import pytest

import utils_18xx.live as live_module
from core.driver import DRIVER, STATUS_INVALID_PY as STATUS_INVALID
from core.data import (
    COMPANY_NAME_TO_ID,
    COMPANY_NAMES,
    CORP_NAME_TO_ID,
    CORP_NAMES,
    DecisionPhase,
    GamePhases,
)
from core.state import GameState
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.market import MARKET
from entities.player import PLAYERS
from entities.turn import TURN
from tests.phases.conftest import float_corp_for_test
from tests.phases.helpers.ownership import give_company_to_fi, give_company_to_player
from utils_18xx.live import (
    _CrossPresidentAcqOfferPriorEvaluator,
    _LiveActionComposer,
    _SearchEngine,
    _acquisition_compatibility_action,
    _align_unordered_round_to_18xx_actor,
    _apply_expected_post_auto_actions,
    _apply_live_planned_action,
    _auto_advanced_validation_state,
    _build_share_ownership,
    _closing_compatibility_action,
    _dividend_compatibility_action,
    _filter_compatibility_mismatches,
    _planned_post_validation_game_data,
    _resolve_buyable_share,
    _resolve_issuable_share,
    _resolve_sellable_share,
    _retarget_acquisition_active_player_to_bot,
    _retarget_closing_active_player_to_bot,
    _should_continue_after_postable_action,
    prepare_live_decision_state,
)
from utils_18xx.game_session import GameSession, StateMismatch
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut


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


class _FakeProcessTurnSession(_FakeSession):
    def __init__(self, state, offer=None, player_ids=None):
        super().__init__(offer=offer, player_ids=player_ids)
        self.state = state
        self.committed_ids = set()

    def sync(self, game_data):
        return self.state

    def validate_against_18xx(self, game_data, state, *, context="live"):
        return []


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


def _invest_action_fixture(corp_name: str, par_index: int = 10):
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID[corp_name]
    company_id = COMPANY_NAME_TO_ID["MHE"]
    float_corp_for_test(
        state,
        corp_id=corp_id,
        company_id=company_id,
        player_id=0,
        par_index=par_index,
    )
    TURN.set_phase(state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(state, 0)

    par_price = MARKET.get_price_at_index(par_index)
    game_data = {
        "players": [{"id": 101, "name": "bot"}],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": COMPANY_NAMES[company_id],
                "entity_type": "company",
                "corporation": corp_name,
                "share_price": f"{par_price},0,{par_index}",
                "user": 101,
            },
        ],
    }
    return state, corp_id, game_data


def test_composer_prices_invest_buy_at_next_higher_share_price():
    state, corp_id, game_data = _invest_action_fixture("DA")
    current_price = CORPS[corp_id].get_share_price(state)
    next_index = MARKET.find_next_higher_space(
        state,
        CORPS[corp_id].get_price_index(state),
    )
    expected_price = MARKET.get_price_at_index(next_index)

    composer = _LiveActionComposer(game_data, bot_player_idx=0, committed_ids={1})
    composer.add_step(
        GamePhases.PHASE_INVEST,
        {"type": "buy_shares", "corporation": "DA"},
        state,
    )

    action = composer.finish()[0]
    assert action["type"] == "buy_shares"
    assert action["entity"] == 101
    assert action["share_price"] == expected_price
    assert action["share_price"] != current_price


def test_composer_prices_invest_sell_at_next_lower_share_price():
    state, corp_id, game_data = _invest_action_fixture("DA")
    current_price = CORPS[corp_id].get_share_price(state)
    next_index = MARKET.find_next_lower_space(
        state,
        CORPS[corp_id].get_price_index(state),
    )
    expected_price = MARKET.get_price_at_index(next_index)

    composer = _LiveActionComposer(game_data, bot_player_idx=0, committed_ids={1})
    composer.add_step(
        GamePhases.PHASE_INVEST,
        {"type": "sell_shares", "corporation": "DA"},
        state,
    )

    action = composer.finish()[0]
    assert action["type"] == "sell_shares"
    assert action["entity"] == 101
    assert action["share_price"] == expected_price
    assert action["share_price"] != current_price


def _issue_action_fixture(corp_name: str, par_index: int = 13):
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID[corp_name]
    company_id = COMPANY_NAME_TO_ID["MHE"]
    float_corp_for_test(
        state,
        corp_id=corp_id,
        company_id=company_id,
        player_id=0,
        par_index=par_index,
    )
    TURN.set_phase(state, int(GamePhases.PHASE_ISSUE_SHARES))
    TURN.set_active_corp(state, corp_id)

    par_price = MARKET.get_price_at_index(par_index)
    game_data = {
        "players": [{"id": 101, "name": "bot"}],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": COMPANY_NAMES[company_id],
                "entity_type": "company",
                "corporation": corp_name,
                "share_price": f"{par_price},0,{par_index}",
                "user": 101,
            },
        ],
    }
    return state, corp_id, game_data


def test_composer_prices_issue_at_next_lower_share_price():
    state, corp_id, game_data = _issue_action_fixture("SI")
    current_price = CORPS[corp_id].get_share_price(state)
    next_index = MARKET.find_next_lower_space(
        state,
        CORPS[corp_id].get_price_index(state),
    )
    expected_price = MARKET.get_price_at_index(next_index)

    composer = _LiveActionComposer(game_data, bot_player_idx=0, committed_ids={1})
    composer.add_step(GamePhases.PHASE_ISSUE_SHARES, {"type": "issue"}, state)

    action = composer.finish()[0]
    assert action["type"] == "sell_shares"
    assert action["entity"] == "SI"
    assert action["share_price"] == expected_price
    assert action["share_price"] != current_price


def test_composer_prices_stock_masters_issue_at_current_share_price():
    state, corp_id, game_data = _issue_action_fixture("SM")
    current_price = CORPS[corp_id].get_share_price(state)

    composer = _LiveActionComposer(game_data, bot_player_idx=0, committed_ids={1})
    composer.add_step(GamePhases.PHASE_ISSUE_SHARES, {"type": "issue"}, state)

    action = composer.finish()[0]
    assert action["type"] == "sell_shares"
    assert action["entity"] == "SM"
    assert action["share_price"] == current_price


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


def test_acquisition_compatibility_allows_represented_cross_president_offer_when_enabled():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["PR"]
    company_id = COMPANY_NAME_TO_ID["OL"]
    price = COMPANIES[company_id].get_low_price()
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    CORPS[corp_id].set_cash(state, 100)
    give_company_to_player(state, company_id, 1)
    TURN.enter_acq_offer(state, corp_id, company_id, price, corp_id, 1)

    action = _acquisition_compatibility_action(
        {"round": "Acquisition", "acting": [202]},
        _FakeSession({
            "responder_id": 202,
            "corporation": "PR",
            "company": "OL",
            "price": price,
        }),
        state,
        bot_user_id=202,
        engine_player_idx=1,
        allow_cross_president_offers=True,
    )

    assert action is None


def test_acquisition_compatibility_still_rejects_fi_offer_when_cross_pres_enabled():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    original_corp_id = CORP_NAME_TO_ID["SM"]
    offered_corp_id = CORP_NAME_TO_ID["OS"]
    company_id = COMPANY_NAME_TO_ID["B"]
    price = COMPANIES[company_id].get_face_value()
    float_corp_for_test(state, corp_id=original_corp_id, player_id=0, par_index=10)
    float_corp_for_test(state, corp_id=offered_corp_id, player_id=1, par_index=12)
    give_company_to_fi(state, company_id)
    TURN.enter_acq_offer(
        state,
        offered_corp_id,
        company_id,
        price,
        original_corp_id,
        1,
    )

    action = _acquisition_compatibility_action(
        {"round": "Acquisition", "acting": [202]},
        _FakeSession({
            "responder_id": 202,
            "corporation": "SM",
            "company": "B",
            "price": price,
        }),
        state,
        bot_user_id=202,
        engine_player_idx=1,
        allow_cross_president_offers=True,
    )

    assert action == {
        "type": "respond",
        "entity": 202,
        "entity_type": "player",
        "corporation": "SM",
        "company": "B",
        "accept": "false",
    }


class _BiasedAcqOfferEvaluator:
    def __init__(self):
        self.action_lut = build_action_lut().numpy()

    def evaluate(self, state):
        del state
        return (
            np.array([0.9, 0.1], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0, 1], dtype=np.uint16),
            2,
            int(DecisionPhase.DPHASE_ACQ_OFFER),
        )

    def evaluate_leaves(self, state_arrays, legal_mask):
        del legal_mask
        priors = np.zeros(
            (len(state_arrays), int(UNIFIED_LOGIT_DIM)),
            dtype=np.float32,
        )
        slots = self.action_lut[
            int(DecisionPhase.DPHASE_ACQ_OFFER),
            np.array([0, 1], dtype=np.intp),
        ]
        priors[:, slots[0]] = 0.9
        priors[:, slots[1]] = 0.1
        values = np.zeros((len(state_arrays), 3), dtype=np.float32)
        return priors, values

    def evaluate_terminal(self, state):
        del state
        return np.zeros(3, dtype=np.float32)


def _cross_president_offer_state_for_prior_test():
    state = GameState(3, acq_same_president=False)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["PR"]
    target_company = COMPANY_NAME_TO_ID["OL"]
    float_company = COMPANY_NAME_TO_ID["BME"]
    float_corp_for_test(
        state,
        corp_id=corp_id,
        company_id=float_company,
        player_id=0,
        par_index=10,
    )
    give_company_to_player(state, target_company, 1)
    TURN.enter_acq_offer(
        state,
        corp_id,
        target_company,
        COMPANIES[target_company].get_low_price(),
        corp_id,
        1,
    )
    return state


def _fi_preemption_offer_state_for_prior_test():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["PR"]
    target_company = COMPANY_NAME_TO_ID["OL"]
    float_company = COMPANY_NAME_TO_ID["BME"]
    float_corp_for_test(
        state,
        corp_id=corp_id,
        company_id=float_company,
        player_id=0,
        par_index=10,
    )
    give_company_to_fi(state, target_company)
    TURN.enter_acq_offer(
        state,
        corp_id,
        target_company,
        COMPANIES[target_company].get_face_value(),
        corp_id,
        0,
    )
    return state


def test_cross_president_acq_offer_prior_adapter_equalizes_response_priors():
    state = _cross_president_offer_state_for_prior_test()
    adapter = _CrossPresidentAcqOfferPriorEvaluator(
        _BiasedAcqOfferEvaluator(),
        num_players=3,
        max_players=3,
    )

    sparse_priors, _values, action_ids, n_legal, _phase_id = adapter.evaluate(
        state,
    )
    assert action_ids[:n_legal].tolist() == [0, 1]
    np.testing.assert_allclose(sparse_priors, [0.5, 0.5])

    dense_priors, _values = adapter.evaluate_leaves(
        [state._array],
        np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8),
    )
    slots = build_action_lut().numpy()[
        int(DecisionPhase.DPHASE_ACQ_OFFER),
        np.array([0, 1], dtype=np.intp),
    ]
    np.testing.assert_allclose(dense_priors[0, slots], [0.5, 0.5])
    assert dense_priors[0].sum() == pytest.approx(1.0)


def test_cross_president_acq_offer_prior_adapter_leaves_fi_priors_unchanged():
    state = _fi_preemption_offer_state_for_prior_test()
    adapter = _CrossPresidentAcqOfferPriorEvaluator(
        _BiasedAcqOfferEvaluator(),
        num_players=3,
        max_players=3,
    )

    sparse_priors, _values, _action_ids, _n_legal, _phase_id = adapter.evaluate(
        state,
    )
    np.testing.assert_allclose(sparse_priors, [0.9, 0.1])

    dense_priors, _values = adapter.evaluate_leaves(
        [state._array],
        np.zeros((1, int(UNIFIED_LOGIT_DIM)), dtype=np.uint8),
    )
    slots = build_action_lut().numpy()[
        int(DecisionPhase.DPHASE_ACQ_OFFER),
        np.array([0, 1], dtype=np.intp),
    ]
    np.testing.assert_allclose(dense_priors[0, slots], [0.9, 0.1])


def test_search_engine_threads_cross_president_flag_to_acq_compatibility():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    engine = _SearchEngine.__new__(_SearchEngine)
    engine.allow_cross_president_offers = True
    engine.validate_player_count = lambda num_players: None
    engine._session_for = lambda game_data: _FakeProcessTurnSession(
        state,
        offer={
            "responder_id": 202,
            "corporation": "PR",
            "company": "OL",
        },
        player_ids=[101, 202, 303],
    )

    actions = engine.process_turn(
        {
            "id": 1,
            "round": "Acquisition",
            "acting": [202],
            "players": [
                {"id": 101, "name": "p1"},
                {"id": 202, "name": "p2"},
                {"id": 303, "name": "p3"},
            ],
        },
        bot_player_idx=1,
        bot_user_id=202,
        bot_user_ids={202},
    )

    assert actions == [{
        "type": "respond",
        "entity": 202,
        "entity_type": "player",
        "corporation": "PR",
        "company": "OL",
        "accept": "false",
    }]


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


def test_acquisition_compatibility_does_not_pass_live_select_corp_mismatch():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 0)

    action = _acquisition_compatibility_action(
        {"round": "Acquisition", "acting": [202]},
        _FakeSession(player_ids=[101, 202, 303]),
        state,
        bot_user_id=202,
        engine_player_idx=1,
    )

    assert action is None


def test_acquisition_retarget_points_ordered_turn_at_acting_bot():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 0)

    changed = _retarget_acquisition_active_player_to_bot(
        {"round": "Acquisition", "acting": [202]},
        state,
        bot_user_id=202,
        engine_player_idx=1,
    )

    assert changed
    assert TURN.get_active_player(state) == 1


def test_search_engine_retargets_acquisition_before_compatibility_pass():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    state.acq_same_president = True
    corp_id = CORP_NAME_TO_ID["SI"]
    company_id = COMPANY_NAME_TO_ID["B"]
    float_corp_for_test(
        state,
        corp_id=CORP_NAME_TO_ID["PR"],
        player_id=0,
        par_index=10,
    )
    float_corp_for_test(state, corp_id=corp_id, player_id=1, par_index=12)
    CORPS[corp_id].set_cash(state, 100)
    give_company_to_fi(state, company_id)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 0)

    engine = _SearchEngine.__new__(_SearchEngine)
    engine.allow_cross_president_offers = False
    engine.validate_player_count = lambda num_players: None
    engine._session_for = lambda game_data: _FakeProcessTurnSession(
        state,
        player_ids=[101, 202, 303],
    )

    def plan_live_actions(
        planned_state,
        game_data,
        bot_player_idx,
        num_players,
        committed_ids,
        bot_user_id=None,
    ):
        assert TURN.get_active_player(planned_state) == 1
        return [{"type": "planned-acquisition"}]

    engine._plan_live_actions = plan_live_actions

    actions = engine.process_turn(
        {
            "id": 1,
            "round": "Acquisition",
            "acting": [202],
            "players": [
                {"id": 101, "name": "p1"},
                {"id": 202, "name": "p2"},
                {"id": 303, "name": "p3"},
            ],
        },
        bot_player_idx=1,
        bot_user_id=202,
        bot_user_ids={101, 202},
    )

    assert actions == [{"type": "planned-acquisition"}]


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


def test_live_closing_pass_uses_18xx_progression_and_restores_model_gate():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    negative_company = COMPANY_NAME_TO_ID["BME"]
    positive_company = COMPANY_NAME_TO_ID["OL"]
    give_company_to_player(state, negative_company, 0)
    give_company_to_player(state, positive_company, 1)
    COMPANIES[negative_company].set_adjusted_income(state, -1)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)
    state.allow_positive_income_closing = False
    state.step_mode = True

    pass_action = next(
        action_id
        for action_id, info in live_module.get_legal_actions(state)
        if info.action_type == live_module.ACTION_PASS
    )

    status = _apply_live_planned_action(
        state,
        GamePhases.PHASE_CLOSING,
        pass_action,
        {"type": "pass"},
    )

    assert status != STATUS_INVALID
    assert state.allow_positive_income_closing is False
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(state) == 1

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=3,
        max_players=3,
    )

    assert validation_state.allow_positive_income_closing is True
    assert TURN.get_phase(validation_state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(validation_state) == 1


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


def test_unordered_round_alignment_prefers_live_acting_over_extractor_actor():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 1)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    session = _FakeSession(player_ids=[101, 202, 303])
    session._last_extract_record = {
        "current_round": "CLO",
        "active_player": 303,
    }

    applied = _align_unordered_round_to_18xx_actor(
        {"acting": [202]},
        session,
        state,
        bot_player_indices={1},
    )

    assert applied == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(state) == 1


def test_unordered_round_alignment_falls_back_to_extractor_closing_actor():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    state.allow_positive_income_closing = True
    give_company_to_player(state, COMPANY_NAME_TO_ID["BME"], 0)
    give_company_to_player(state, COMPANY_NAME_TO_ID["OL"], 1)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)

    session = _FakeSession(player_ids=[101, 202, 303])
    session._last_extract_record = {
        "current_round": "CLO",
        "active_player": 202,
    }

    applied = _align_unordered_round_to_18xx_actor(
        {"acting": []},
        session,
        state,
        bot_player_indices={1},
    )

    assert applied == 1
    assert TURN.get_phase(state) == int(GamePhases.PHASE_CLOSING)
    assert TURN.get_active_player(state) == 1


def test_unordered_active_validation_prefers_live_acting_over_extractor_actor():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 1)

    session = GameSession(3)
    session._player_ids = [101, 202, 303]
    mismatches = []

    session._compare_unordered_active_player(
        {"acting": [202]},
        {"current_round": "CLO", "active_player": 303},
        state,
        99,
        "PHASE_CLOSING",
        "unit",
        mismatches,
    )

    assert mismatches == []


def test_planned_post_validation_preserves_unordered_closing_actors():
    session = _FakeSession(player_ids=[101, 202, 303])
    session._last_extract_record = {
        "current_round": "CLO",
        "active_player": 303,
    }

    game_data = _planned_post_validation_game_data(
        {"round": "Closing", "acting": [101, 202]},
        session,
    )

    assert game_data["acting"] == [101, 202, 303]


def test_planned_post_validation_removes_planned_closing_pass_actor():
    session = _FakeSession(player_ids=[101, 202, 303])
    session._last_extract_record = {
        "current_round": "CLO",
        "active_player": 303,
    }

    game_data = _planned_post_validation_game_data(
        {
            "round": "Closing",
            "acting": [101, 202, 303],
            "actions": [{
                "type": "pass",
                "entity": 202,
                "entity_type": "player",
            }],
        },
        session,
    )

    assert game_data["acting"] == [101, 303]


def test_planned_post_validation_narrows_ordered_round_actor():
    session = _FakeSession(player_ids=[101, 202, 303])
    session._last_extract_record = {
        "current_round": "INV",
        "active_player": 303,
    }

    game_data = _planned_post_validation_game_data(
        {"round": "Investment", "acting": [101, 202]},
        session,
    )

    assert game_data["acting"] == [303]


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


def test_live_planning_stops_on_acq_offer():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_OFFER))
    TURN.set_active_player(state, 2)

    assert not _should_continue_after_postable_action(
        GamePhases.PHASE_ACQ_SELECT_COMPANY,
        state,
        bot_player_idx=2,
    )


def test_live_planning_stops_after_acq_offer_response():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_ACQ_SELECT_CORP))
    TURN.set_active_player(state, 2)

    assert not _should_continue_after_postable_action(
        GamePhases.PHASE_ACQ_OFFER,
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


def _single_dividend_engine_state():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    corp_id = CORP_NAME_TO_ID["DA"]
    float_corp_for_test(state, corp_id=corp_id, player_id=0, par_index=10)
    CORPS[corp_id].set_cash(state, 0)
    TURN.set_phase(state, int(GamePhases.PHASE_DIVIDENDS))
    TURN.set_active_player(state, 0)
    TURN.set_active_corp(state, corp_id)
    TURN.set_dividend_remaining(state, corp_id, True)
    return state


def _single_dividend_engine():
    engine = _SearchEngine.__new__(_SearchEngine)
    engine.model_output = False
    engine.max_players = 3
    return engine


def test_live_planning_validates_post_action_state(monkeypatch):
    state = _single_dividend_engine_state()
    engine = _single_dividend_engine()
    calls = []

    def fake_validate(game_data, predicted_state, planned_actions, **kwargs):
        calls.append((game_data, predicted_state, planned_actions, kwargs))
        return []

    monkeypatch.setattr(
        live_module,
        "_validate_planned_post_state",
        fake_validate,
    )

    actions = engine._plan_live_actions(
        state,
        {
            "id": 1,
            "round": "Dividends",
            "acting": [101],
            "players": [
                {"id": 101, "name": "bot"},
                {"id": 202, "name": "p2"},
                {"id": 303, "name": "p3"},
            ],
        },
        bot_player_idx=0,
        num_players=3,
        committed_ids=set(),
        bot_user_id=101,
    )

    assert actions == [{
        "type": "dividend",
        "entity": "DA",
        "entity_type": "corporation",
        "kind": "variable",
        "amount": 0,
    }]
    assert calls
    assert calls[0][2] == actions
    assert calls[0][3] == {"num_players": 3, "max_players": 3}


def test_live_planning_skips_post_action_state_validation_in_closing(monkeypatch):
    state = GameState(3)
    state.initialize_game(3, seed=42)
    TURN.set_phase(state, int(GamePhases.PHASE_CLOSING))
    TURN.set_active_player(state, 0)
    state.step_mode = True
    assert len(live_module.get_legal_actions(state)) == 1

    engine = _SearchEngine.__new__(_SearchEngine)
    engine.model_output = False
    engine.max_players = 3

    def fake_validate(game_data, predicted_state, planned_actions, **kwargs):
        del game_data, predicted_state, planned_actions, kwargs
        raise AssertionError("CLOSING post-action validation should be skipped")

    monkeypatch.setattr(
        live_module,
        "_validate_planned_post_state",
        fake_validate,
    )

    actions = engine._plan_live_actions(
        state,
        {
            "id": 1,
            "round": "Closing",
            "acting": [101],
            "players": [
                {"id": 101, "name": "bot"},
                {"id": 202, "name": "p2"},
                {"id": 303, "name": "p3"},
            ],
        },
        bot_player_idx=0,
        num_players=3,
        committed_ids=set(),
        bot_user_id=101,
    )

    assert actions == [{
        "type": "pass",
        "entity": 101,
        "entity_type": "player",
    }]


def test_live_planning_refuses_post_action_state_mismatch(monkeypatch):
    state = _single_dividend_engine_state()
    engine = _single_dividend_engine()

    def fake_validate(game_data, predicted_state, planned_actions, **kwargs):
        del game_data, predicted_state, planned_actions, kwargs
        return [
            StateMismatch(
                action_id=999,
                phase="PHASE_DIVIDENDS",
                field="corp[DA].price",
                expected=12,
                actual=14,
                context="unit",
            )
        ]

    monkeypatch.setattr(
        live_module,
        "_validate_planned_post_state",
        fake_validate,
    )

    actions = engine._plan_live_actions(
        state,
        {
            "id": 1,
            "round": "Dividends",
            "acting": [101],
            "players": [
                {"id": 101, "name": "bot"},
                {"id": 202, "name": "p2"},
                {"id": 303, "name": "p3"},
            ],
        },
        bot_player_idx=0,
        num_players=3,
        committed_ids=set(),
        bot_user_id=101,
    )

    assert actions == []


def test_post_validation_state_advances_step_mode_automated_pause():
    state = _single_dividend_engine_state()
    state.step_mode = True

    status = DRIVER.apply_action(state, 0)

    assert status != STATUS_INVALID
    assert TURN.get_phase(state) == int(GamePhases.PHASE_END_CARD)
    assert TURN.get_active_corp(state) == -1

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=3,
        max_players=3,
    )

    assert TURN.get_phase(validation_state) == int(GamePhases.PHASE_ISSUE_SHARES)
    assert TURN.get_active_corp(validation_state) == CORP_NAME_TO_ID["DA"]
    assert TURN.get_phase(state) == int(GamePhases.PHASE_END_CARD)
    assert TURN.get_active_corp(state) == -1


def test_post_validation_state_advances_forced_ipo_pass_with_unfloated_corps():
    state = GameState(4, max_players=5)
    state.initialize_game(4, seed=42, max_players=5)
    company_id = COMPANY_NAME_TO_ID["HA"]
    give_company_to_player(state, company_id, 2)
    PLAYERS[2].set_cash(state, 0)
    TURN.set_phase(state, int(GamePhases.PHASE_IPO))
    TURN.set_active_player(state, 2)
    TURN.set_active_company(state, company_id)
    TURN.set_ipo_remaining(state, company_id, True)
    state.step_mode = True

    assert len(live_module.get_legal_actions(state)) == 1

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=4,
        max_players=5,
    )

    assert TURN.get_phase(validation_state) != int(GamePhases.PHASE_IPO)
    assert TURN.get_active_company(validation_state) == -1


def test_post_validation_state_advances_market_blocked_ipo_to_next_company():
    state = GameState(4, max_players=5)
    state.initialize_game(4, seed=42, max_players=5)
    blocked_company_id = COMPANY_NAME_TO_ID["HR"]
    next_company_id = COMPANY_NAME_TO_ID["DSB"]
    give_company_to_player(state, blocked_company_id, 2)
    give_company_to_player(state, next_company_id, 3)
    PLAYERS[2].set_cash(state, 100)
    PLAYERS[3].set_cash(state, 100)
    for price in (30, 33, 37):
        MARKET.set_space_available(state, MARKET.get_index_for_price(price), False)
    TURN.set_phase(state, int(GamePhases.PHASE_IPO))
    TURN.set_active_player(state, 2)
    TURN.set_active_company(state, blocked_company_id)
    TURN.set_ipo_remaining(state, blocked_company_id, True)
    TURN.set_ipo_remaining(state, next_company_id, True)
    state.step_mode = True

    legal_actions = live_module.get_legal_actions(state)
    assert len(legal_actions) == 1
    assert legal_actions[0][1].action_type == live_module.ACTION_PASS

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=4,
        max_players=5,
    )

    assert TURN.get_phase(validation_state) == int(GamePhases.PHASE_IPO)
    assert TURN.get_active_player(validation_state) == 3
    assert TURN.get_active_company(validation_state) == next_company_id


def test_post_validation_state_advances_unavailable_ipo_passes():
    state = GameState(4, max_players=5)
    state.initialize_game(4, seed=42, max_players=5)
    company_id = COMPANY_NAME_TO_ID["HA"]
    give_company_to_player(state, company_id, 2)
    for corp_id in range(len(CORP_NAMES)):
        CORPS[corp_id].set_active(state, True)
    TURN.set_phase(state, int(GamePhases.PHASE_IPO))
    TURN.set_active_player(state, 2)
    TURN.set_active_company(state, company_id)
    TURN.set_ipo_remaining(state, company_id, True)
    state.step_mode = True

    legal_actions = live_module.get_legal_actions(state)
    assert len(legal_actions) == 1
    assert legal_actions[0][1].action_type == live_module.ACTION_PASS

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=4,
        max_players=5,
    )

    assert TURN.get_phase(validation_state) != int(GamePhases.PHASE_IPO)
    assert TURN.get_active_company(validation_state) == -1


def test_post_validation_state_advances_internal_forced_bid_leaves():
    state = GameState(5, max_players=5)
    state.initialize_game(5, seed=42, max_players=5)
    for player_idx in range(5):
        PLAYERS[player_idx].set_turn_order(state, player_idx)
        PLAYERS[player_idx].set_cash(state, 0)

    company_id = COMPANY_NAME_TO_ID["HA"]
    give_company_to_player(state, company_id, 1)
    COMPANIES[company_id].move_to_auction(state)
    PLAYERS[1].set_cash(state, 61)
    TURN.set_phase(state, int(GamePhases.PHASE_BID))
    TURN.set_active_player(state, 2)
    TURN.set_active_company(state, company_id)
    TURN.set_auction_price(state, 46)
    TURN.set_auction_high_bidder(state, 1)
    TURN.set_auction_starter(state, 1)
    TURN.clear_passed_flags(state)
    state.step_mode = True

    legal_actions = live_module.get_legal_actions(state)
    assert len(legal_actions) == 1
    assert legal_actions[0][1].action_type == live_module.ACTION_PASS

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=5,
        max_players=5,
    )

    assert TURN.get_phase(validation_state) == int(GamePhases.PHASE_INVEST)
    assert COMPANIES[company_id].is_owned_by_player(validation_state, 1)
    assert PLAYERS[1].get_cash(validation_state) == 15
    assert TURN.get_active_company(validation_state) == -1


def test_post_validation_state_advances_internal_forced_invest_pass():
    state = GameState(4, max_players=6)
    state.initialize_game(4, seed=42, max_players=6)
    for player_idx in range(4):
        PLAYERS[player_idx].set_cash(state, 0)
    TURN.set_phase(state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(state, 0)
    TURN.set_consecutive_passes(state, 3)
    state.step_mode = True

    legal_actions = live_module.get_legal_actions(state)
    assert len(legal_actions) == 1
    assert legal_actions[0][1].action_type == live_module.ACTION_PASS

    validation_state = _auto_advanced_validation_state(
        state,
        num_players=4,
        max_players=6,
    )

    assert TURN.get_phase(validation_state) != int(GamePhases.PHASE_INVEST)


def test_post_validation_applies_expected_program_share_auto_pass():
    state = GameState(5, max_players=5)
    state.initialize_game(5, seed=42, max_players=5)
    for player_idx in range(5):
        PLAYERS[player_idx].set_turn_order(state, player_idx)
    TURN.set_phase(state, int(GamePhases.PHASE_INVEST))
    TURN.set_active_player(state, 4)
    TURN.set_consecutive_passes(state, 2)

    synthetic_game_data = {
        "actions": [{
            "type": "pass",
            "entity": 202,
            "entity_type": "player",
            "auto_actions": [{
                "type": "pass",
                "entity": 505,
                "entity_type": "player",
            }],
        }],
    }
    session = _FakeSession(player_ids=[101, 202, 303, 404, 505])

    applied = _apply_expected_post_auto_actions(
        state,
        synthetic_game_data,
        original_action_count=0,
        session=session,
    )

    assert applied
    assert TURN.get_phase(state) == int(GamePhases.PHASE_INVEST)
    assert TURN.get_active_player(state) == 0
    assert TURN.get_consecutive_passes(state) == 3


def test_compatibility_mismatch_filter_keeps_economic_mismatches():
    round_mismatch = StateMismatch(
        action_id=1,
        phase="PHASE_CLOSING",
        field="round",
        expected="Closing",
        actual="PHASE_INCOME",
    )
    active_mismatch = StateMismatch(
        action_id=1,
        phase="PHASE_CLOSING",
        field="active_player",
        expected=[1],
        actual=0,
    )
    cash_mismatch = StateMismatch(
        action_id=1,
        phase="PHASE_CLOSING",
        field="corp[PR].cash",
        expected=20,
        actual=10,
    )

    assert _filter_compatibility_mismatches(
        [round_mismatch, active_mismatch, cash_mismatch],
        {"type": "pass"},
    ) == [cash_mismatch]


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


def test_share_ledger_repar_leaves_unissued_shares_in_treasury():
    game_data = {
        "players": [
            {"id": 4, "name": "rss-az-3"},
            {"id": 2, "name": "rss-az-1"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "MHE",
                "entity_type": "company",
                "corporation": "SI",
                "share_price": "14,0,8",
                "user": 2,
            },
            {
                "id": 2,
                "type": "sell_shares",
                "entity": "SI",
                "entity_type": "corporation",
                "shares": ["SI_2"],
                "share_price": 16,
                "user": 2,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": "SI",
                "entity_type": "corporation",
                "shares": ["SI_3"],
                "share_price": 16,
                "user": 2,
            },
            {
                "id": 4,
                "type": "sell_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SI_0"],
                "share_price": 10,
                "user": 2,
            },
            {
                "id": 5,
                "type": "par",
                "entity": "BR",
                "entity_type": "company",
                "corporation": "SI",
                "share_price": "37,0,20",
                "user": 4,
            },
        ],
    }

    pool, owners = _build_share_ownership(game_data, set(range(1, 6)))

    assert owners["4"]["SI"] == [0]
    assert pool["SI"] == [1]
    assert _resolve_issuable_share(game_data, "SI", set(range(1, 6))) == "SI_2"


def test_share_ledger_reconciles_hidden_receivership_issue():
    game_data = {
        "players": [
            {"id": 4, "name": "rss-az-3"},
            {"id": 2, "name": "rss-az-1"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "BPM",
                "entity_type": "company",
                "corporation": "SM",
                "share_price": "12,0,8",
                "user": 4,
            },
            {
                "id": 2,
                "type": "sell_shares",
                "entity": "SM",
                "entity_type": "corporation",
                "shares": ["SM_2"],
                "share_price": 12,
                "user": 4,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": "SM",
                "entity_type": "corporation",
                "shares": ["SM_3"],
                "share_price": 10,
                "user": 4,
            },
            {
                "id": 4,
                "type": "sell_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["SM_0"],
                "share_price": 10,
                "user": 4,
            },
            {
                "id": 5,
                "type": "buy_shares",
                "entity": 2,
                "entity_type": "player",
                "shares": ["SM_0"],
                "share_price": 8,
                "user": 2,
            },
        ],
    }

    pool, owners = _build_share_ownership(game_data, set(range(1, 6)))

    assert pool["SM"] == [1, 2, 3]
    assert owners["2"]["SM"] == [0]
    assert _resolve_issuable_share(
        game_data,
        "SM",
        set(range(1, 6)),
        market_share_count=4,
        treasury_share_count=1,
    ) == "SM_5"


def test_share_ledger_reconciles_hidden_market_share_for_buy():
    game_data = {
        "players": [
            {"id": 5, "name": "rss-az-4"},
            {"id": 4, "name": "rss-az-3"},
        ],
        "actions": [
            {
                "id": 1,
                "type": "par",
                "entity": "WT",
                "entity_type": "company",
                "corporation": "VM",
                "share_price": "16,0,11",
                "user": 4,
            },
            {
                "id": 2,
                "type": "sell_shares",
                "entity": "VM",
                "entity_type": "corporation",
                "shares": ["VM_2"],
                "share_price": 16,
                "user": 4,
            },
            {
                "id": 3,
                "type": "sell_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["VM_0"],
                "share_price": 20,
                "user": 4,
            },
            {
                "id": 4,
                "type": "buy_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["VM_0"],
                "share_price": 13,
                "user": 4,
            },
            {
                "id": 5,
                "type": "buy_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["VM_1"],
                "share_price": 14,
                "user": 4,
            },
            {
                "id": 6,
                "type": "sell_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["VM_1"],
                "share_price": 13,
                "user": 4,
            },
            {
                "id": 7,
                "type": "buy_shares",
                "entity": 5,
                "entity_type": "player",
                "shares": ["VM_1"],
                "share_price": 10,
                "user": 5,
            },
            {
                "id": 8,
                "type": "buy_shares",
                "entity": 4,
                "entity_type": "player",
                "shares": ["VM_2"],
                "share_price": 11,
                "user": 4,
            },
        ],
    }

    pool, owners = _build_share_ownership(game_data, set(range(1, 9)))

    assert pool.get("VM", []) == []
    assert owners["5"]["VM"] == [1]
    assert owners["4"]["VM"] == [0, 2]
    assert _resolve_buyable_share(
        game_data,
        "VM",
        set(range(1, 9)),
        market_share_count=1,
        treasury_share_count=0,
    ) == "VM_3"


def test_share_ledger_buy_prefers_non_president_market_share():
    game_data = {
        "players": [{"id": 4, "name": "rss-az-3"}],
        "actions": [],
    }

    assert _resolve_buyable_share(
        game_data,
        "VM",
        set(),
        market_share_count=4,
        treasury_share_count=1,
    ) == "VM_1"
