from core.data import CorpIndices
from core.state import GameState, get_corp_fields
from entities.company import COMPANIES
from entities.corp import CORPS, calculate_price_move
from entities.deck import DECK
from entities.turn import TURN


def get_required_stars(price, issued_shares):
    if price < 1 or issued_shares < 2 or issued_shares > 7:
        return 0
    return int(issued_shares * price / 10.0 + 0.5)


def assert_corp_cache_fresh(state, msg=""):
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if not corp.is_active(state):
            assert corp.get_income(state) == 0, f"{msg}\nCorp {corp_id} inactive income stale"
            assert corp.get_stars(state) == 0, f"{msg}\nCorp {corp_id} inactive stars stale"
            assert corp.get_pending_price_move(state) == 0, (
                f"{msg}\nCorp {corp_id} inactive pending move stale"
            )
            assert corp.get_raw_revenue(state) == 0, (
                f"{msg}\nCorp {corp_id} inactive raw revenue stale"
            )
            assert corp.get_synergy_income(state) == 0, (
                f"{msg}\nCorp {corp_id} inactive synergy income stale"
            )
            assert corp.get_coo_cost(state) == 0, (
                f"{msg}\nCorp {corp_id} inactive CoO cost stale"
            )
            assert corp.get_ability_income(state) == 0, (
                f"{msg}\nCorp {corp_id} inactive ability income stale"
            )
            continue

        expected_company_stars = sum(
            COMPANIES[company_id].get_stars()
            for company_id in range(36)
            if corp.owns_company(state, company_id) or corp.has_acquisition_company(state, company_id)
        )
        expected_cash_stars = corp.get_cash(state) // 10 if corp.get_cash(state) > 0 else 0
        expected_total_stars = expected_company_stars + expected_cash_stars
        if corp_id == CorpIndices.CORP_SI:
            expected_total_stars += 2

        assert corp.get_company_stars(state) == expected_company_stars, (
            f"{msg}\nCorp {corp_id} company stars stale"
        )
        assert corp.get_cash_stars(state) == expected_cash_stars, (
            f"{msg}\nCorp {corp_id} cash stars stale"
        )
        assert corp.get_stars(state) == expected_total_stars, (
            f"{msg}\nCorp {corp_id} total stars stale"
        )

        expected_pending_move = calculate_price_move(
            expected_total_stars,
            get_required_stars(corp.get_price_index(state), corp.get_issued_shares(state)),
        )
        assert corp.get_pending_price_move(state) == expected_pending_move, (
            f"{msg}\nCorp {corp_id} pending move stale"
        )

        raw_revenue = corp.get_raw_revenue(state)
        synergy_income = corp.get_synergy_income(state)
        coo_cost = corp.get_coo_cost(state)
        ability_income = corp.get_ability_income(state)
        income = corp.get_income(state)

        assert raw_revenue + synergy_income + coo_cost + ability_income == income, (
            f"{msg}\nCorp {corp_id} income breakdown stale"
        )
        assert corp.calculate_income(state) == income, (
            f"{msg}\nCorp {corp_id} calculated income mismatch"
        )


class TestCorpCacheFreshness:
    def test_company_transfer_self_heals_corp_cache(self):
        state = GameState(num_players=3)
        state.initialize_game(3, seed=42)

        first_company = DECK.draw(state)
        CORPS[0].float_corp(state, 0, first_company, 10, 2)

        extra_company = DECK.draw(state)
        COMPANIES[extra_company].transfer_to_corp(state, 0)

        assert_corp_cache_fresh(state, "After direct company transfer to corp")

    def test_cash_change_self_heals_corp_cache(self):
        state = GameState(num_players=3)
        state.initialize_game(3, seed=42)

        first_company = DECK.draw(state)
        CORPS[0].float_corp(state, 0, first_company, 10, 2)
        CORPS[0].add_cash(state, 13)

        assert_corp_cache_fresh(state, "After direct corp cash mutation")

    def test_share_price_change_self_heals_corp_cache(self):
        state = GameState(num_players=3)
        state.initialize_game(3, seed=42)

        first_company = DECK.draw(state)
        CORPS[0].float_corp(state, 0, first_company, 10, 2)
        CORPS[0].set_price_index(state, 11)

        assert_corp_cache_fresh(state, "After direct corp price mutation")

    def test_share_price_is_derived_from_price_index(self):
        state = GameState(num_players=3)
        state.initialize_game(3, seed=42)

        first_company = DECK.draw(state)
        CORPS[0].float_corp(state, 0, first_company, 10, 2)

        assert not hasattr(get_corp_fields(), "share_price")
        assert CORPS[0].get_price_index(state) == 10
        assert CORPS[0].get_share_price(state) == 14
        CORPS[0].set_price_index(state, 11)
        assert CORPS[0].get_share_price(state) == 16

    def test_coo_change_self_heals_all_corp_caches(self):
        state = GameState(num_players=3)
        state.initialize_game(3, seed=42)

        first_company = DECK.draw(state)
        second_company = DECK.draw(state)
        CORPS[0].float_corp(state, 0, first_company, 10, 2)
        CORPS[1].float_corp(state, 1, second_company, 9, 2)
        assert_corp_cache_fresh(state, "Before CoO change")

        TURN.set_coo_level(state, 2)

        assert_corp_cache_fresh(state, "After CoO change")
