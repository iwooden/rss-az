from entities.player import PLAYERS
from entities.corp import CORPS


def set_player_cashs(state, cash_by_player: dict[int, int]) -> None:
    for player_id, cash in cash_by_player.items():
        PLAYERS[player_id].set_cash(state, cash)



def set_corp_cashs(state, cash_by_corp: dict[int, int]) -> None:
    for corp_id, cash in cash_by_corp.items():
        CORPS[corp_id].set_cash(state, cash)



def prime_corp_income_for_test(state, corp_id: int, income: int) -> None:
    CORPS[corp_id].refresh_cache(state)
    CORPS[corp_id].set_income(state, income)
