from core.state import GameState
from entities import COMPANIES, FI, PLAYERS


def test_initialize_game_seeds_adjusted_company_incomes():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    company_id = 5  # Red company; CoO level 1 leaves base income unchanged.
    expected_income = COMPANIES[company_id].get_base_income()

    assert COMPANIES[company_id].get_adjusted_income(state) == expected_income

    COMPANIES[company_id].transfer_to_player(state, 0)
    assert PLAYERS[0].get_income(state) == expected_income


def test_initialize_game_seeds_foreign_investor_income_base():
    state = GameState(3)
    state.initialize_game(3, seed=42)

    assert FI.get_income(state) == 5
