"""Direct tests for the CLOSING phase."""

import pytest

from core.actions import (
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_PASS_PY as ACTION_PASS,
)
from core.data import GamePhases
from core.driver import DRIVER, STATUS_PAUSED_PY as STATUS_PAUSED
from entities.player import PLAYERS
from entities.turn import TURN
from entities.company import COMPANIES
from entities.corp import CORPS
from phases.closing import setup_closing_phase_py

from tests.phases.conftest import (
    assert_invariants,
    find_all_legal_actions_with_info,
    find_legal_action,
    float_corp_for_test,
    setup_receivership_corp,
)
from tests.phases.helpers.ownership import (
    give_company_to_corp,
    give_company_to_fi,
    give_company_to_player,
)


PHASE_CLOSING = int(GamePhases.PHASE_CLOSING)
PHASE_INCOME = int(GamePhases.PHASE_INCOME)


class TestAutoCloseSetup:
    def test_setup_closing_auto_closes_fi_negative_income_company(self, game_state):
        TURN.set_coo_level(game_state, 7)
        give_company_to_fi(game_state, 0)

        setup_closing_phase_py(game_state)

        assert COMPANIES[0].is_removed(game_state)
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert_invariants(game_state, "after FI auto-close during CLOSING setup")

    def test_setup_closing_auto_closes_receivership_red_company_at_threshold(self, game_state):
        TURN.set_coo_level(game_state, 5)
        setup_receivership_corp(game_state, corp_id=1, company_ids=[0, 14])

        setup_closing_phase_py(game_state)

        assert COMPANIES[0].is_removed(game_state)
        assert not COMPANIES[14].is_removed(game_state)
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert_invariants(game_state, "after receivership auto-close during CLOSING setup")

    def test_setup_closing_protects_highest_face_value_company_in_receivership(self, game_state):
        TURN.set_coo_level(game_state, 7)
        setup_receivership_corp(game_state, corp_id=1, company_ids=[0, 1, 2, 3])

        setup_closing_phase_py(game_state)

        assert not COMPANIES[3].is_removed(game_state)
        assert COMPANIES[0].is_removed(game_state)
        assert COMPANIES[1].is_removed(game_state)
        assert COMPANIES[2].is_removed(game_state)
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert_invariants(game_state, "after highest-face protection during CLOSING setup")


class TestMandatoryClose:
    def test_passing_on_voluntary_close_triggers_mandatory_close_until_income_plus_cash_non_negative(self, game_state):
        TURN.set_coo_level(game_state, 7)
        give_company_to_player(game_state, 0, 0)
        give_company_to_player(game_state, 8, 0)
        PLAYERS[0].set_cash(game_state, 10)

        setup_closing_phase_py(game_state)

        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 0

        game_state.step_mode = True
        pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        status = DRIVER.apply_action(game_state, pass_id)

        assert status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert COMPANIES[0].is_removed(game_state)
        assert not COMPANIES[8].is_removed(game_state)
        assert PLAYERS[0].get_income(game_state) + PLAYERS[0].get_cash(game_state) >= 0
        assert_invariants(game_state, "after mandatory close pass resolution")


class TestVoluntaryCloseLegality:
    def test_president_can_close_corp_subsidiaries_while_corp_has_multiple_companies(self, game_state):
        float_corp_for_test(game_state, corp_id=1, company_id=0, player_id=0)
        give_company_to_corp(game_state, 3, 1)

        setup_closing_phase_py(game_state)

        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 0
        close_actions = find_all_legal_actions_with_info(game_state, action_type=ACTION_CLOSE)
        assert [info.company_id for _, info in close_actions] == [0, 3]
        assert_invariants(game_state, "after enumerating closable corp subsidiaries")

    def test_after_closing_one_corp_subsidiary_last_company_is_protected(self, game_state):
        float_corp_for_test(game_state, corp_id=1, company_id=0, player_id=0)
        give_company_to_corp(game_state, 3, 1)

        setup_closing_phase_py(game_state)

        game_state.step_mode = True
        close_id = find_legal_action(game_state, action_type=ACTION_CLOSE, company_id=0)
        status = DRIVER.apply_action(game_state, close_id)

        assert status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 0
        assert COMPANIES[0].is_removed(game_state)
        assert not COMPANIES[3].is_removed(game_state)
        assert find_all_legal_actions_with_info(game_state, action_type=ACTION_CLOSE) == []
        assert find_legal_action(game_state, action_type=ACTION_PASS) == 0
        assert_invariants(game_state, "after protecting final corp company from voluntary close")


class TestJunkyardScrappersBonus:
    def test_js_receivership_auto_close_adds_double_income_scrapping_bonus(self, game_state):
        setup_receivership_corp(game_state, corp_id=0, company_ids=[0, 14])
        CORPS[0].set_cash(game_state, 5)
        TURN.set_coo_level(game_state, 5)

        setup_closing_phase_py(game_state)

        assert COMPANIES[0].is_removed(game_state)
        assert CORPS[0].get_cash(game_state) == 7
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert_invariants(game_state, "after JS receivership scrapping bonus")

    def test_js_president_close_adds_double_income_scrapping_bonus(self, game_state):
        float_corp_for_test(game_state, corp_id=0, company_id=0, player_id=0)
        give_company_to_corp(game_state, 3, 0)
        CORPS[0].set_cash(game_state, 5)

        setup_closing_phase_py(game_state)

        game_state.step_mode = True
        close_id = find_legal_action(game_state, action_type=ACTION_CLOSE, company_id=0)
        status = DRIVER.apply_action(game_state, close_id)

        assert status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 0
        assert COMPANIES[0].is_removed(game_state)
        assert CORPS[0].get_cash(game_state) == 7
        assert_invariants(game_state, "after JS president close scrapping bonus")


class TestPlayerOrder:
    def test_setup_closing_starts_with_lowest_player_id_who_can_close(self, game_state):
        give_company_to_player(game_state, 0, 1)
        if TURN.get_num_players(game_state) >= 3:
            give_company_to_player(game_state, 1, 2)

        setup_closing_phase_py(game_state)

        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 1
        assert_invariants(game_state, "after selecting first closer by player ID")

    def test_player_keeps_turn_until_passing_even_after_multiple_closes(self, game_state):
        trailing_player = 1 if TURN.get_num_players(game_state) == 2 else 2
        give_company_to_player(game_state, 0, 0)
        give_company_to_player(game_state, 1, 0)
        give_company_to_player(game_state, 2, trailing_player)

        setup_closing_phase_py(game_state)

        game_state.step_mode = True
        close_id = find_legal_action(game_state, action_type=ACTION_CLOSE, company_id=0)
        status = DRIVER.apply_action(game_state, close_id)

        assert status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert TURN.get_active_player(game_state) == 0
        remaining_close_actions = find_all_legal_actions_with_info(game_state, action_type=ACTION_CLOSE)
        assert [info.company_id for _, info in remaining_close_actions] == [1]
        assert_invariants(game_state, "after retaining turn for multiple closes")

    def test_pass_advances_by_player_id_and_all_passes_end_closing(self, game_state):
        give_company_to_player(game_state, 0, 0)
        give_company_to_player(game_state, 1, 1)

        setup_closing_phase_py(game_state)

        game_state.step_mode = True
        first_pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        first_status = DRIVER.apply_action(game_state, first_pass_id)

        assert first_status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert PLAYERS[0].has_passed(game_state)
        assert TURN.get_active_player(game_state) == 1

        second_pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        second_status = DRIVER.apply_action(game_state, second_pass_id)

        assert second_status == STATUS_PAUSED
        assert PLAYERS[1].has_passed(game_state)
        assert TURN.get_phase(game_state) == PHASE_INCOME
        assert_invariants(game_state, "after all closers pass in player-ID order")

    @pytest.mark.parametrize("game_state", [3, 4, 5, 6], indirect=True)
    def test_pass_skips_players_without_closable_companies(self, game_state):
        give_company_to_player(game_state, 0, 0)
        give_company_to_player(game_state, 1, 2)

        setup_closing_phase_py(game_state)

        game_state.step_mode = True
        first_pass_id = find_legal_action(game_state, action_type=ACTION_PASS)
        first_status = DRIVER.apply_action(game_state, first_pass_id)

        assert first_status == STATUS_PAUSED
        assert TURN.get_phase(game_state) == PHASE_CLOSING
        assert PLAYERS[0].has_passed(game_state)
        assert TURN.get_active_player(game_state) == 2
        assert_invariants(game_state, "after skipping players without closable companies")
