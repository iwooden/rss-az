"""Direct tests for the END_CARD phase."""

from core.data import GameConstants, GamePhases
from entities.turn import TURN
from entities.corp import CORPS
from phases.end_card import apply_end_card_py

from tests.phases.conftest import assert_invariants, float_corp_for_test
from tests.phases.helpers.ownership import give_company_to_player


PHASE_END_CARD = int(GamePhases.PHASE_END_CARD)
PHASE_GAME_OVER = int(GamePhases.PHASE_GAME_OVER)
PHASE_ISSUE_SHARES = int(GamePhases.PHASE_ISSUE_SHARES)


def _enter_end_card(state):
    TURN.set_phase(state, PHASE_END_CARD)


def _remove_all_unowned_companies(state, owner_player=0, keep_company_ids=()):
    keep_company_ids = set(keep_company_ids)
    for company_id in range(int(GameConstants.NUM_COMPANIES)):
        if company_id in keep_company_ids:
            continue
        give_company_to_player(state, company_id, owner_player)


class TestEndCardDirectCoverage:
    def test_corp_at_max_price_triggers_game_over(self, game_state):
        float_corp_for_test(game_state, corp_id=0, company_id=0, player_id=0)
        CORPS[0].set_price_index(game_state, int(GameConstants.NUM_MARKET_SPACES) - 1)
        _enter_end_card(game_state)

        apply_end_card_py(game_state)

        assert TURN.get_phase(game_state) == PHASE_GAME_OVER
        assert not TURN.is_end_card_flipped(game_state)

    def test_preflipped_end_card_triggers_game_over(self, game_state):
        TURN.set_end_card_flipped(game_state, True)
        _enter_end_card(game_state)

        apply_end_card_py(game_state)

        assert TURN.get_phase(game_state) == PHASE_GAME_OVER
        assert_invariants(game_state, "after preflipped END_CARD game over")

    def test_no_unowned_companies_flips_end_card_and_enters_issue(self, game_state):
        float_corp_for_test(game_state, corp_id=0, company_id=0, player_id=0)
        _remove_all_unowned_companies(game_state, keep_company_ids={0})
        _enter_end_card(game_state)

        apply_end_card_py(game_state)

        assert TURN.is_end_card_flipped(game_state)
        assert TURN.get_coo_level(game_state) == int(GameConstants.COO_LEVEL_END_CARD_FLIPPED)
        assert TURN.get_phase(game_state) == PHASE_ISSUE_SHARES
        assert TURN.get_active_corp(game_state) == 0
        assert TURN.is_issue_remaining(game_state, 0)
        assert_invariants(game_state, "after END_CARD flip with no unowned companies")

    def test_normal_end_card_transitions_to_issue_without_flipping(self, game_state):
        float_corp_for_test(game_state, corp_id=0, company_id=0, player_id=0)
        _enter_end_card(game_state)
        coo_before = TURN.get_coo_level(game_state)

        apply_end_card_py(game_state)

        assert not TURN.is_end_card_flipped(game_state)
        assert TURN.get_coo_level(game_state) == coo_before
        assert TURN.get_phase(game_state) == PHASE_ISSUE_SHARES
        assert TURN.get_active_corp(game_state) == 0
        assert TURN.is_issue_remaining(game_state, 0)
        assert_invariants(game_state, "after normal END_CARD transition to ISSUE")
