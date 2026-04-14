from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from core.data import GameConstants


def _location_value(location) -> int:
    return int(location)


def give_company_to_player(state, company_id: int, player_id: int) -> None:
    company = COMPANIES[company_id]
    if company.get_location(state) == int(CompanyLocation.LOC_DECK):
        DECK.set_company_location(state, company_id, int(CompanyLocation.LOC_PLAYER), player_id)
        return
    company.transfer_to_player(state, player_id)



def give_company_to_corp(state, company_id: int, corp_id: int) -> None:
    company = COMPANIES[company_id]
    if company.get_location(state) == int(CompanyLocation.LOC_DECK):
        DECK.set_company_location(state, company_id, int(CompanyLocation.LOC_CORP), corp_id)
        return
    company.transfer_to_corp(state, corp_id)



def give_company_to_fi(state, company_id: int) -> None:
    company = COMPANIES[company_id]
    if company.get_location(state) == int(CompanyLocation.LOC_DECK):
        DECK.set_company_location(state, company_id, int(CompanyLocation.LOC_FI))
        return
    company.transfer_to_fi(state)



def ids_at_location(state, location) -> list[int]:
    location = _location_value(location)
    return [
        cid
        for cid in range(int(GameConstants.NUM_COMPANIES))
        if COMPANIES[cid].get_location(state) == location
    ]



def count_at_location(state, location) -> int:
    return len(ids_at_location(state, location))
