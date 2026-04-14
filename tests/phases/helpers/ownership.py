from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK


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
