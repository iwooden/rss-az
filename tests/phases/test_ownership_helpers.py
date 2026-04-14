from entities.turn import TURN
from entities.company import COMPANIES, CompanyLocation

from tests.phases.conftest import draw_to_player, float_corp_for_test
from tests.phases.helpers.ownership import (
    give_company_to_player,
    give_company_to_corp,
    give_company_to_fi,
)


CO_5S = 35  # always in the live deck across player counts


class TestOwnershipHelpers:
    def test_give_company_to_player_pulls_specific_company_from_deck(self, game_state):
        cards_before = TURN.get_cards_remaining(game_state)

        give_company_to_player(game_state, CO_5S, player_id=1)

        assert COMPANIES[CO_5S].get_location(game_state) == int(CompanyLocation.LOC_PLAYER)
        assert COMPANIES[CO_5S].get_owner_id(game_state) == 1
        assert TURN.get_cards_remaining(game_state) == cards_before - 1

    def test_give_company_to_corp_rehomes_existing_owned_company(self, game_state):
        float_corp_for_test(game_state, corp_id=0, player_id=0, par_index=10)
        company_id = draw_to_player(game_state, player_id=1)

        give_company_to_corp(game_state, company_id, corp_id=0)

        assert COMPANIES[company_id].get_location(game_state) == int(CompanyLocation.LOC_CORP)
        assert COMPANIES[company_id].get_owner_id(game_state) == 0

    def test_give_company_to_fi_rehomes_existing_owned_company(self, game_state):
        company_id = draw_to_player(game_state, player_id=1)

        give_company_to_fi(game_state, company_id)

        assert COMPANIES[company_id].get_location(game_state) == int(CompanyLocation.LOC_FI)
        assert COMPANIES[company_id].get_owner_id(game_state) == -1
