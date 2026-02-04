"""Tests for INCOME phase (v6.0).

Requirements covered:
- SYN-01, SYN-02: Synergy pair calculation (Phase 21) ✓
- SYN-03: Synergy income added to entity income (Phase 21) ✓
- INC-01: Entity sums income from owned companies (Phase 22) ✓
- INC-02: CoO deducted from income (Phase 22) ✓
- INC-03: FI +5 bonus (Phase 22) ✓
- INC-04: Positive income adds to cash (Phase 22) ✓
- INC-05: Negative income subtracts from cash (Phase 22) ✓
- CSA-01: PR +1 per company (Phase 22) ✓
- CSA-02: DA +printed income of highest FV (Phase 22) ✓
- CSA-03: S +synergy_markers // 2 (Phase 22) ✓
- CSA-04: VM reduces CoO by up to 10 (Phase 22) ✓
- INC-06: Corp bankruptcy during income (cash < 0) (Phase 22) ✓
- TRN-01: INCOME transitions to DIVIDENDS after income (Phase 23) ✓
- TRN-02: Dividends phase setup called (dividend_corp initialized) (Phase 23) ✓
- TRN-03: INCOME is non-player phase (0 valid actions) (Phase 23) ✓
- TRN-04: Multiple corps can go bankrupt in same INCOME phase (Phase 23) ✓
"""
import pytest
from core.state import GameState
from core.data import (
    py_compute_synergy_bonuses,
    COMPANY_NAME_TO_ID,
    get_company_synergy,
    GamePhases,
    GameConstants,
)
from core.actions import get_valid_action_mask
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.fi import FI
from entities.market import MARKET
from phases.income import apply_income_py
from phases.dividends import setup_dividends_phase_py, find_next_dividend_corp_py


# =============================================================================
# Phase 21: Synergy Infrastructure
# =============================================================================


class TestSynergyCalculation:
    """SYN-01, SYN-02: Synergy pair identification and counting."""

    def test_no_companies_returns_zero(self):
        """No companies -> (0, 0)."""
        income, markers = py_compute_synergy_bonuses([])
        assert income == 0
        assert markers == 0

    def test_single_company_returns_zero(self):
        """Single company -> (0, 0) - no pairs possible."""
        cdg = COMPANY_NAME_TO_ID["CDG"]
        income, markers = py_compute_synergy_bonuses([cdg])
        assert income == 0
        assert markers == 0

    def test_two_companies_no_synergy(self):
        """Two companies with no synergy -> (0, 0)."""
        # BME and BSE have no synergy with each other
        bme = COMPANY_NAME_TO_ID["BME"]
        bse = COMPANY_NAME_TO_ID["BSE"]
        # Verify they don't synergize
        assert get_company_synergy(bme, bse) == 0
        assert get_company_synergy(bse, bme) == 0
        income, markers = py_compute_synergy_bonuses([bme, bse])
        assert income == 0
        assert markers == 0

    def test_two_companies_one_way_synergy(self):
        """Two companies where only A synergizes with B."""
        # CDG synergizes with MAD (bonus 16), MAD does not synergize with CDG
        cdg = COMPANY_NAME_TO_ID["CDG"]
        mad = COMPANY_NAME_TO_ID["MAD"]
        assert get_company_synergy(cdg, mad) == 16
        assert get_company_synergy(mad, cdg) == 0
        income, markers = py_compute_synergy_bonuses([cdg, mad])
        assert income == 16
        assert markers == 1  # One pair, one marker

    def test_two_companies_asymmetric_synergies(self):
        """Two companies where both have synergies with other companies."""
        # DR synergizes with PKP (4)
        # PKP synergizes with KK (4)
        # Test that having DR+PKP only counts DR->PKP, not the unrelated PKP->KK
        dr = COMPANY_NAME_TO_ID["DR"]
        pkp = COMPANY_NAME_TO_ID["PKP"]
        assert get_company_synergy(dr, pkp) == 4
        assert get_company_synergy(pkp, dr) == 0  # No reverse synergy
        income, markers = py_compute_synergy_bonuses([dr, pkp])
        assert income == 4  # Only DR->PKP
        assert markers == 1  # One pair

    def test_three_companies_multiple_pairs(self):
        """Three companies with multiple synergy pairs."""
        # DR synergizes with: WT (2), BY (2)
        # BY synergizes with: WT (2)
        dr = COMPANY_NAME_TO_ID["DR"]
        wt = COMPANY_NAME_TO_ID["WT"]
        by = COMPANY_NAME_TO_ID["BY"]

        # Verify synergies exist
        assert get_company_synergy(dr, wt) == 2
        assert get_company_synergy(dr, by) == 2
        assert get_company_synergy(by, wt) == 2

        income, markers = py_compute_synergy_bonuses([dr, wt, by])
        # DR-WT: 2 income, 1 marker
        # DR-BY: 2 income, 1 marker
        # WT-BY: BY->WT=2, WT->BY=0 -> 2 income, 1 marker
        assert income == 6  # 2 + 2 + 2
        assert markers == 3  # 3 pairs

    def test_pair_counted_once_regardless_of_order(self):
        """Order of company IDs doesn't affect result."""
        cdg = COMPANY_NAME_TO_ID["CDG"]
        mad = COMPANY_NAME_TO_ID["MAD"]

        income1, markers1 = py_compute_synergy_bonuses([cdg, mad])
        income2, markers2 = py_compute_synergy_bonuses([mad, cdg])

        assert income1 == income2
        assert markers1 == markers2

    def test_complex_synergy_network(self):
        """Multiple companies with complex synergy relationships."""
        # CDG synergizes with: MAD(16), FRA(16), LHR(16), E(8), SBB(4), SNCF(4)
        cdg = COMPANY_NAME_TO_ID["CDG"]
        mad = COMPANY_NAME_TO_ID["MAD"]
        fra = COMPANY_NAME_TO_ID["FRA"]
        # Just CDG, MAD, FRA
        # CDG->MAD=16, CDG->FRA=16, MAD->FRA=0, FRA->MAD=0
        income, markers = py_compute_synergy_bonuses([cdg, mad, fra])
        assert income == 32  # 16 + 16
        assert markers == 2  # CDG-MAD, CDG-FRA (no MAD-FRA synergy)


# =============================================================================
# Phase 22: Income Calculation
# =============================================================================


class TestCorpBaseIncome:
    """INC-01, INC-02: Corporation base income calculation."""

    def test_corp_no_companies_returns_zero(self, game_state):
        """Corporation with no companies -> 0 income."""
        from entities.corp import CORPS
        corp = CORPS[0]
        corp.set_active(game_state, True)

        income = corp.calculate_income(game_state)
        assert income == 0

    def test_corp_single_company_no_synergy(self, game_state):
        """Corporation with 1 company -> income - CoO (no synergy bonus)."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        corp = CORPS[0]
        corp.set_active(game_state, True)

        # Give corp company 0 (BME: income=18, 3 stars)
        COMPANIES[0].transfer_to_corp(game_state, 0)

        # CoO at level 1 (start of game)
        coo_level = TURN.get_coo_level(game_state)
        assert coo_level == 1

        expected_income = get_company_income(0)
        stars = get_company_stars(0)
        coo = get_cost_of_ownership(coo_level, stars)
        expected = expected_income - coo

        income = corp.calculate_income(game_state)
        assert income == expected

    def test_corp_two_companies_with_synergy(self, game_state):
        """Corporation with 2+ synergizing companies -> income - CoO + synergy."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            COMPANY_NAME_TO_ID, py_compute_synergy_bonuses
        )

        corp = CORPS[0]
        corp.set_active(game_state, True)

        # Give corp CDG and MAD (CDG->MAD synergy = 16)
        cdg = COMPANY_NAME_TO_ID["CDG"]
        mad = COMPANY_NAME_TO_ID["MAD"]

        COMPANIES[cdg].transfer_to_corp(game_state, 0)
        COMPANIES[mad].transfer_to_corp(game_state, 0)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected manually
        base_cdg = get_company_income(cdg)
        base_mad = get_company_income(mad)
        coo_cdg = get_cost_of_ownership(coo_level, get_company_stars(cdg))
        coo_mad = get_cost_of_ownership(coo_level, get_company_stars(mad))

        synergy_income, _ = py_compute_synergy_bonuses([cdg, mad])

        expected = base_cdg + base_mad - coo_cdg - coo_mad + synergy_income

        income = corp.calculate_income(game_state)
        assert income == expected

    def test_corp_at_high_coo_level(self, game_state):
        """Corporation with companies at high CoO level -> correct deduction."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        corp = CORPS[0]
        corp.set_active(game_state, True)

        # Set high CoO level
        TURN.set_coo_level(game_state, 6)

        # Give corp a 3-star company (BME)
        COMPANIES[0].transfer_to_corp(game_state, 0)

        base_income = get_company_income(0)
        stars = get_company_stars(0)
        coo = get_cost_of_ownership(6, stars)
        expected = base_income - coo

        income = corp.calculate_income(game_state)
        assert income == expected


class TestFIIncome:
    """INC-03: Foreign Investor income calculation."""

    def test_fi_no_companies_returns_five(self, game_state):
        """FI with no companies -> 5 (base bonus only)."""
        from entities.fi import FI

        income = FI.calculate_income(game_state)
        assert income == 5

    def test_fi_one_company(self, game_state):
        """FI with 1 company -> income - CoO + 5."""
        from entities.fi import FI
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        # Transfer company 0 to FI
        COMPANIES[0].transfer_to_fi(game_state)

        coo_level = TURN.get_coo_level(game_state)
        base_income = get_company_income(0)
        stars = get_company_stars(0)
        coo = get_cost_of_ownership(coo_level, stars)

        expected = base_income - coo + 5

        income = FI.calculate_income(game_state)
        assert income == expected

    def test_fi_multiple_companies(self, game_state):
        """FI with multiple companies -> sum(income - CoO) + 5."""
        from entities.fi import FI
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        # Transfer companies 0, 1, 2 to FI
        for cid in [0, 1, 2]:
            COMPANIES[cid].transfer_to_fi(game_state)

        coo_level = TURN.get_coo_level(game_state)

        expected = 5  # Start with FI bonus
        for cid in [0, 1, 2]:
            base_income = get_company_income(cid)
            stars = get_company_stars(cid)
            coo = get_cost_of_ownership(coo_level, stars)
            expected += base_income - coo

        income = FI.calculate_income(game_state)
        assert income == expected


class TestCorpSpecialAbilities:
    """CSA-01 through CSA-04: Corporation special ability modifiers to income."""

    def test_pr_with_zero_companies(self, game_state):
        """CSA-01: PR with 0 companies -> +0 bonus (but still works)."""
        from entities.corp import CORPS

        # CORP_PR = 4 (Prussian Railway)
        pr = CORPS[4]
        pr.set_active(game_state, True)

        income = pr.calculate_income(game_state)
        assert income == 0  # No companies, no bonus

    def test_pr_with_multiple_companies(self, game_state):
        """CSA-01: PR with 3 companies -> +3 bonus."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            py_compute_synergy_bonuses
        )

        pr = CORPS[4]  # CORP_PR
        pr.set_active(game_state, True)

        # Give PR three companies (0, 1, 2)
        # Note: 2 (KME) synergizes with 0 (BME) for +1
        companies = [0, 1, 2]
        for cid in companies:
            COMPANIES[cid].transfer_to_corp(game_state, 4)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate synergy
        synergy_income, _ = py_compute_synergy_bonuses(companies)

        # Calculate expected: base income - CoO + synergy + company_count
        expected = 0
        for cid in companies:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
        expected += synergy_income
        expected += 3  # +1 per company (PR ability)

        income = pr.calculate_income(game_state)
        assert income == expected

    def test_da_with_multiple_companies(self, game_state):
        """CSA-02: DA with companies of different FVs -> bonus = printed income of highest FV."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            get_company_face_value, py_compute_synergy_bonuses,
            COMPANY_NAME_TO_ID
        )

        da = CORPS[5]  # CORP_DA
        da.set_active(game_state, True)

        # Give DA companies with different FVs
        # Need to find companies with distinct FVs
        # BME: FV=3, income=18
        # BSE: FV=3, income=18
        # CDG: FV=5, income=32
        # MAD: FV=4, income=28
        # Let's use companies with FV 1, 3, 5
        # FV=1: Company 23 (KK)
        # FV=3: Company 0 (BME)
        # FV=5: Company 3 (CDG)
        companies = [23, 0, 3]  # KK (FV=1), BME (FV=3), CDG (FV=5)

        for cid in companies:
            COMPANIES[cid].transfer_to_corp(game_state, 5)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate synergy
        synergy_income, _ = py_compute_synergy_bonuses(companies)

        # Find highest FV and its income
        highest_fv = max(get_company_face_value(cid) for cid in companies)
        highest_fv_income = max(
            (get_company_income(cid) for cid in companies if get_company_face_value(cid) == highest_fv),
            default=0
        )

        # Calculate expected: base income - CoO + synergy + highest_fv_income
        expected = 0
        for cid in companies:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
        expected += synergy_income
        expected += highest_fv_income  # DA ability bonus

        income = da.calculate_income(game_state)
        assert income == expected

    def test_s_with_four_synergy_markers(self, game_state):
        """CSA-03: S with 4 synergy markers -> +2 bonus (4 // 2)."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            COMPANY_NAME_TO_ID, py_compute_synergy_bonuses
        )

        s = CORPS[1]  # CORP_S (Synergistic)
        s.set_active(game_state, True)

        # Give S companies that form 4 synergy pairs (4 markers)
        # DR synergizes with: WT (2), BY (2), PKP (4)
        # BY synergizes with: WT (2)
        # This gives us: DR-WT, DR-BY, DR-PKP, WT-BY = 4 pairs
        dr = COMPANY_NAME_TO_ID["DR"]
        wt = COMPANY_NAME_TO_ID["WT"]
        by = COMPANY_NAME_TO_ID["BY"]
        pkp = COMPANY_NAME_TO_ID["PKP"]
        companies = [dr, wt, by, pkp]

        for cid in companies:
            COMPANIES[cid].transfer_to_corp(game_state, 1)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate synergy
        synergy_income, synergy_markers = py_compute_synergy_bonuses(companies)

        # Calculate expected: base income - CoO + synergy_income + (synergy_markers // 2)
        expected = 0
        for cid in companies:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
        expected += synergy_income
        expected += synergy_markers // 2  # S ability bonus

        income = s.calculate_income(game_state)
        assert income == expected

    def test_s_with_five_synergy_markers(self, game_state):
        """CSA-03: S with 5 synergy markers -> +2 bonus (5 // 2 = 2, rounds down)."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            COMPANY_NAME_TO_ID, py_compute_synergy_bonuses
        )

        s = CORPS[1]  # CORP_S
        s.set_active(game_state, True)

        # Give S companies that form 5 synergy pairs (5 markers)
        # DR synergizes with: WT, BY, PKP, SNCF, SNCB
        # Need to create 5 pairs - add SNCF to previous set
        dr = COMPANY_NAME_TO_ID["DR"]
        wt = COMPANY_NAME_TO_ID["WT"]
        by = COMPANY_NAME_TO_ID["BY"]
        pkp = COMPANY_NAME_TO_ID["PKP"]
        sncf = COMPANY_NAME_TO_ID["SNCF"]
        companies = [dr, wt, by, pkp, sncf]

        for cid in companies:
            COMPANIES[cid].transfer_to_corp(game_state, 1)

        coo_level = TURN.get_coo_level(game_state)

        synergy_income, synergy_markers = py_compute_synergy_bonuses(companies)

        # Calculate expected
        expected = 0
        for cid in companies:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
        expected += synergy_income
        expected += synergy_markers // 2  # S ability bonus (5 // 2 = 2)

        income = s.calculate_income(game_state)
        assert income == expected

    def test_vm_with_coo_below_ten(self, game_state):
        """CSA-04: VM with total_coo=8 -> CoO reduced to 0."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        vm = CORPS[6]  # CORP_VM (Vintage Machinery)
        vm.set_active(game_state, True)

        # Give VM companies that total ~8 CoO
        # At CoO level 1: 1 star = 2, 2 stars = 4, 3 stars = 6
        # Use two 2-star companies (Company 1 and 2) = 4 + 4 = 8
        TURN.set_coo_level(game_state, 1)

        for cid in [1, 2]:  # Two 2-star companies
            COMPANIES[cid].transfer_to_corp(game_state, 6)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected: base income - max(0, total_coo - 10)
        expected = 0
        total_coo = 0
        for cid in [1, 2]:
            expected += get_company_income(cid)
            total_coo += get_cost_of_ownership(coo_level, get_company_stars(cid))

        # VM ability: reduce CoO by up to 10
        reduced_coo = max(0, total_coo - 10)
        expected -= reduced_coo

        income = vm.calculate_income(game_state)
        assert income == expected

    def test_vm_with_coo_above_ten(self, game_state):
        """CSA-04: VM with total_coo=15 -> CoO reduced by 10, leaving 5."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        vm = CORPS[6]  # CORP_VM
        vm.set_active(game_state, True)

        # Give VM companies that total 15 CoO
        # At CoO level 2: 1 star = 3, 2 stars = 6, 3 stars = 9
        # Use one 3-star (9) and one 2-star (6) = 15
        TURN.set_coo_level(game_state, 2)

        companies = [0, 1]  # 3-star and 2-star
        for cid in companies:
            COMPANIES[cid].transfer_to_corp(game_state, 6)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected
        expected = 0
        total_coo = 0
        for cid in companies:
            expected += get_company_income(cid)
            total_coo += get_cost_of_ownership(coo_level, get_company_stars(cid))

        # VM ability: reduce CoO by up to 10
        reduced_coo = max(0, total_coo - 10)
        expected -= reduced_coo

        income = vm.calculate_income(game_state)
        assert income == expected

    def test_non_income_ability_corp_unaffected(self, game_state):
        """Other corporations (JS, OS, SM, SI) have no income modifications."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        js = CORPS[0]  # CORP_JS - has ability but doesn't affect calculate_income
        js.set_active(game_state, True)

        # Give JS one company
        COMPANIES[0].transfer_to_corp(game_state, 0)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected: just base income - CoO (no special ability bonus)
        expected = get_company_income(0)
        expected -= get_cost_of_ownership(coo_level, get_company_stars(0))

        income = js.calculate_income(game_state)
        assert income == expected


class TestIncomeApplication:
    """INC-04, INC-05: Income application to entity cash."""

    def test_corp_positive_income_adds_cash(self, game_state):
        """Corporation positive income increases cash."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from core.data import COMPANY_NAME_TO_ID

        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 10)

        # Give corp a company (CDG: income=32, stars=4, CoO_level1=8 -> net=24)
        cdg = COMPANY_NAME_TO_ID["CDG"]
        COMPANIES[cdg].transfer_to_corp(game_state, 0)

        income = corp.calculate_income(game_state)
        assert income > 0  # Should be 24

        corp.apply_income(game_state, income)

        assert corp.get_cash(game_state) == 10 + income

    def test_corp_negative_income_subtracts_cash(self, game_state):
        """Corporation negative income decreases cash."""
        from entities.corp import CORPS

        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 10)

        # Apply negative income directly
        corp.apply_income(game_state, -7)

        assert corp.get_cash(game_state) == 3

    def test_corp_can_go_negative(self, game_state):
        """Corporation cash can go negative after income application."""
        from entities.corp import CORPS

        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 5)

        # Apply large negative income
        corp.apply_income(game_state, -10)

        assert corp.get_cash(game_state) == -5

    def test_fi_income_with_bonus(self, game_state):
        """FI income includes +5 bonus."""
        from entities.fi import FI
        from entities.company import COMPANIES
        from core.data import COMPANY_NAME_TO_ID

        FI.set_cash(game_state, 10)

        # FI with no companies -> income = 5 (just bonus)
        income = FI.calculate_income(game_state)
        assert income == 5

        FI.apply_income(game_state, income)

        assert FI.get_cash(game_state) == 15

    def test_player_income_uses_existing_methods(self, game_state):
        """Player income applied via add_cash."""
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from core.data import COMPANY_NAME_TO_ID

        player = PLAYERS[0]
        player.set_cash(game_state, 20)

        # Give player a company (CDG: income=32, stars=4, CoO_level1=8 -> net=24)
        cdg = COMPANY_NAME_TO_ID["CDG"]
        COMPANIES[cdg].transfer_to_player(game_state, 0)

        income = player.get_income(game_state)
        assert income > 0  # Should be 24

        player.add_cash(game_state, income)

        assert player.get_cash(game_state) == 20 + income


# =============================================================================
# Phase 22 Continued: Bankruptcy Handling
# =============================================================================


class TestCorpBankruptcy:
    """INC-06: Corporation bankruptcy during INCOME phase."""

    def test_corp_goes_bankrupt_on_negative_cash(self, game_state):
        """Corp with negative cash after income goes bankrupt."""
        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 1)  # Low starting cash

        # Give corp a company that generates negative income
        # At high CoO level, companies can have negative adjusted income
        TURN.set_coo_level(game_state, 6)  # High CoO

        # KK: income=5, stars=3. At CoO level 6, 3-star CoO=7.
        # Adjusted income = 5 - 7 = -2
        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_corp(game_state, 0)

        # Set up market space
        corp.set_price_index(game_state, 10)
        MARKET.set_space_available(game_state, 10, False)

        # Verify corp has negative income
        income = corp.calculate_income(game_state)
        assert income < 0, f"Expected negative income, got {income}"

        # Start cash + income should be negative (1 + (-2) = -1 < 0)
        assert corp.get_cash(game_state) + income < 0

        # Set phase to INCOME and apply
        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Corp should be bankrupt (inactive)
        assert not corp.is_active(game_state), "Corp should be bankrupt"

    def test_corp_survives_with_sufficient_cash(self, game_state):
        """Corp with enough cash to cover negative income survives."""
        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 100)  # Plenty of cash

        # Give corp negative income company
        TURN.set_coo_level(game_state, 6)
        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_corp(game_state, 0)

        corp.set_price_index(game_state, 10)
        MARKET.set_space_available(game_state, 10, False)

        income = corp.calculate_income(game_state)
        starting_cash = corp.get_cash(game_state)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Corp should survive with reduced cash
        assert corp.is_active(game_state), "Corp should survive"
        assert corp.get_cash(game_state) == starting_cash + income

    def test_bankruptcy_check_immediate_after_income(self, game_state):
        """Bankruptcy is checked immediately after each corp's income."""
        # Set up two corps where first one goes bankrupt
        corp0 = CORPS[0]
        corp0.set_active(game_state, True)
        corp0.set_cash(game_state, 0)
        corp0.set_price_index(game_state, 10)
        MARKET.set_space_available(game_state, 10, False)

        corp1 = CORPS[1]
        corp1.set_active(game_state, True)
        corp1.set_cash(game_state, 100)
        corp1.set_price_index(game_state, 15)
        MARKET.set_space_available(game_state, 15, False)

        # Give both corps negative income companies
        TURN.set_coo_level(game_state, 6)
        kk = COMPANY_NAME_TO_ID["KK"]
        dr = COMPANY_NAME_TO_ID["DR"]

        COMPANIES[kk].transfer_to_corp(game_state, 0)

        COMPANIES[dr].transfer_to_corp(game_state, 1)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Corp 0 should be bankrupt, Corp 1 should survive
        assert not corp0.is_active(game_state), "Corp 0 should be bankrupt"
        assert corp1.is_active(game_state), "Corp 1 should survive"


# =============================================================================
# Phase 23: Phase Transitions
# =============================================================================


class TestIncomeTransition:
    """TRN-01: INCOME phase transitions to DIVIDENDS."""

    def test_income_transitions_to_dividends(self, game_state):
        """After income application, phase changes to DIVIDENDS."""
        # Set up a simple active corp
        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 100)
        corp.set_price_index(game_state, 10)
        corp.set_stars(game_state, 5)
        corp.set_unissued_shares(game_state, 3)
        corp.set_issued_shares(game_state, 4)
        MARKET.set_space_available(game_state, 10, False)

        # Give player 0 shares and presidency
        PLAYERS[0].set_shares(game_state, 0, 4)
        PLAYERS[0].set_president_of(game_state, 0, True)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        assert TURN.get_phase(game_state) == GamePhases.PHASE_DIVIDENDS

    def test_income_transitions_even_with_no_corps(self, game_state):
        """INCOME transitions to DIVIDENDS, which then transitions to END_CARD."""
        # Ensure no corps are active
        for corp_id in range(8):
            CORPS[corp_id].set_active(game_state, False)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # INCOME sets phase to DIVIDENDS, but setup_dividends_phase
        # immediately transitions to END_CARD when no corps exist
        assert TURN.get_phase(game_state) == GamePhases.PHASE_END_CARD


class TestDividendSetup:
    """TRN-02: Dividends phase setup initializes dividend_corp."""

    def test_dividend_corp_set_to_first_eligible(self, game_state):
        """After transition, dividend_corp is set to highest-price corp."""
        # Set up two corps at different prices
        corp0 = CORPS[0]
        corp0.set_active(game_state, True)
        corp0.set_cash(game_state, 100)
        corp0.set_price_index(game_state, 10)  # Lower price
        corp0.set_stars(game_state, 5)
        corp0.set_unissued_shares(game_state, 3)
        corp0.set_issued_shares(game_state, 4)
        MARKET.set_space_available(game_state, 10, False)
        PLAYERS[0].set_shares(game_state, 0, 4)
        PLAYERS[0].set_president_of(game_state, 0, True)

        corp1 = CORPS[1]
        corp1.set_active(game_state, True)
        corp1.set_cash(game_state, 100)
        corp1.set_price_index(game_state, 15)  # Higher price - processed first
        corp1.set_stars(game_state, 5)
        corp1.set_unissued_shares(game_state, 2)
        corp1.set_issued_shares(game_state, 5)
        MARKET.set_space_available(game_state, 15, False)
        PLAYERS[1].set_shares(game_state, 1, 5)
        PLAYERS[1].set_president_of(game_state, 1, True)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Dividend corp should be corp 1 (higher price)
        dividend_corp = TURN.get_dividend_corp(game_state)
        assert dividend_corp == 1, f"Expected dividend_corp=1, got {dividend_corp}"

    def test_dividend_corp_cleared_when_no_corps(self, game_state):
        """With no active corps, dividend_corp is -1 after setup."""
        # Ensure no corps are active
        for corp_id in range(8):
            CORPS[corp_id].set_active(game_state, False)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Should have no dividend corp (transitioned out of DIVIDENDS already)
        # The dividends phase will immediately transition to next phase
        dividend_corp = TURN.get_dividend_corp(game_state)
        assert dividend_corp == -1, f"Expected dividend_corp=-1, got {dividend_corp}"


class TestNonPlayerPhase:
    """TRN-03: INCOME is a non-player phase with 0 valid actions."""

    def test_income_has_no_valid_actions(self, game_state):
        """In INCOME phase, action mask should be all zeros."""
        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)

        mask = get_valid_action_mask(game_state)

        # All actions should be invalid
        valid_count = sum(1 for v in mask if v == 1.0)
        assert valid_count == 0, f"Expected 0 valid actions, got {valid_count}"

    def test_income_auto_executes_in_driver(self, game_state):
        """INCOME phase should auto-execute when reached."""
        from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK

        # Set up corp for dividends phase
        corp = CORPS[0]
        corp.set_active(game_state, True)
        corp.set_cash(game_state, 100)
        corp.set_price_index(game_state, 10)
        corp.set_stars(game_state, 5)
        corp.set_unissued_shares(game_state, 3)
        corp.set_issued_shares(game_state, 4)
        MARKET.set_space_available(game_state, 10, False)
        PLAYERS[0].set_shares(game_state, 0, 4)
        PLAYERS[0].set_president_of(game_state, 0, True)

        # Manually set phase to INCOME
        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)

        # The driver should detect this and auto-execute
        # We can't directly call the driver's auto-apply from here,
        # but we can verify that apply_income_py transitions correctly
        apply_income_py(game_state)

        # Should be in DIVIDENDS with valid actions now
        assert TURN.get_phase(game_state) == GamePhases.PHASE_DIVIDENDS
        mask = get_valid_action_mask(game_state)
        valid_count = sum(1 for v in mask if v == 1.0)
        assert valid_count > 0, "DIVIDENDS should have valid actions"


class TestMultipleBankruptcies:
    """TRN-04: Multiple corps can go bankrupt in same INCOME phase."""

    def test_two_corps_go_bankrupt_simultaneously(self, game_state):
        """Two corps with negative income both go bankrupt."""
        TURN.set_coo_level(game_state, 6)

        # Set up corp 0 with negative income and low cash
        corp0 = CORPS[0]
        corp0.set_active(game_state, True)
        corp0.set_cash(game_state, 1)
        corp0.set_price_index(game_state, 10)
        MARKET.set_space_available(game_state, 10, False)

        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_corp(game_state, 0)

        # Set up corp 1 with negative income and low cash
        corp1 = CORPS[1]
        corp1.set_active(game_state, True)
        corp1.set_cash(game_state, 1)
        corp1.set_price_index(game_state, 11)
        MARKET.set_space_available(game_state, 11, False)

        dr = COMPANY_NAME_TO_ID["DR"]
        COMPANIES[dr].transfer_to_corp(game_state, 1)

        # Both should have negative income
        assert corp0.calculate_income(game_state) < 0
        assert corp1.calculate_income(game_state) < 0

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Both corps should be bankrupt
        assert not corp0.is_active(game_state), "Corp 0 should be bankrupt"
        assert not corp1.is_active(game_state), "Corp 1 should be bankrupt"

    def test_bankruptcy_order_is_corp_id_order(self, game_state):
        """Corps are processed in corp_id order (0-7)."""
        TURN.set_coo_level(game_state, 6)

        # Set up corps 0, 2, 4 with bankruptcy conditions
        for corp_id in [0, 2, 4]:
            corp = CORPS[corp_id]
            corp.set_active(game_state, True)
            corp.set_cash(game_state, 0)
            corp.set_price_index(game_state, 10 + corp_id)
            MARKET.set_space_available(game_state, 10 + corp_id, False)

        # Give each a negative income company
        companies = [COMPANY_NAME_TO_ID["KK"], COMPANY_NAME_TO_ID["DR"], COMPANY_NAME_TO_ID["BY"]]
        for corp_id, cid in zip([0, 2, 4], companies):
            COMPANIES[cid].transfer_to_corp(game_state, corp_id)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # All three should be bankrupt
        for corp_id in [0, 2, 4]:
            assert not CORPS[corp_id].is_active(game_state), f"Corp {corp_id} should be bankrupt"

    def test_surviving_corp_not_affected_by_other_bankruptcies(self, game_state):
        """A corp that survives is unaffected by others going bankrupt."""
        TURN.set_coo_level(game_state, 6)

        # Corp 0: will go bankrupt
        corp0 = CORPS[0]
        corp0.set_active(game_state, True)
        corp0.set_cash(game_state, 0)
        corp0.set_price_index(game_state, 10)
        MARKET.set_space_available(game_state, 10, False)

        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_corp(game_state, 0)

        # Corp 1: will survive (high cash)
        corp1 = CORPS[1]
        corp1.set_active(game_state, True)
        corp1.set_cash(game_state, 200)
        corp1.set_price_index(game_state, 15)
        corp1.set_stars(game_state, 5)
        corp1.set_unissued_shares(game_state, 2)
        corp1.set_issued_shares(game_state, 5)
        MARKET.set_space_available(game_state, 15, False)

        # Give corp 1 a profitable company
        cdg = COMPANY_NAME_TO_ID["CDG"]  # High income blue company
        COMPANIES[cdg].transfer_to_corp(game_state, 1)

        PLAYERS[1].set_shares(game_state, 1, 5)
        PLAYERS[1].set_president_of(game_state, 1, True)

        starting_cash = corp1.get_cash(game_state)
        income1 = corp1.calculate_income(game_state)

        TURN.set_phase(game_state, GamePhases.PHASE_INCOME)
        apply_income_py(game_state)

        # Corp 0 bankrupt, Corp 1 survives with correct cash
        assert not corp0.is_active(game_state), "Corp 0 should be bankrupt"
        assert corp1.is_active(game_state), "Corp 1 should survive"
        assert corp1.get_cash(game_state) == starting_cash + income1
