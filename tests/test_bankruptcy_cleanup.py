from core.state import GameState
from entities import COMPANIES, CORPS, PLAYERS


def _float_corp(state, corp_id=0, player_id=0, company_id=0, par_index=15, float_shares=1):
    COMPANIES[company_id].transfer_to_player(state, player_id)
    CORPS[corp_id].float_corp(state, player_id, company_id, par_index, float_shares)


def test_bankruptcy_removes_acquisition_pile_companies():
    state = GameState(3)
    state.initialize_game(3, seed=42)
    _float_corp(state)

    corp = CORPS[0]
    COMPANIES[10].transfer_to_corp_acquisition(state, 0)

    assert COMPANIES[10].is_in_corp_acquisition(state, 0)

    corp.go_bankrupt(state)

    assert COMPANIES[0].is_removed(state)
    assert COMPANIES[10].is_removed(state)
    assert COMPANIES[10].get_owner_id(state) == -1
    assert not corp.is_active(state)
    assert PLAYERS[0].get_shares(state, 0) == 0
