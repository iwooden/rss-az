"""Tests for INCOME phase (v6.0).

Requirements covered:
- SYN-01, SYN-02: Synergy pair calculation (Phase 21)
- INC-01 through INC-06: Income calculation (Phase 22)
- CSA-01 through CSA-04: Corporation special abilities (Phase 22)
- TRN-01 through TRN-04: Phase transitions (Phase 23)
"""
import pytest
from core.data import (
    py_compute_synergy_bonuses,
    COMPANY_NAME_TO_ID,
    get_company_synergy,
)


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
        corp.set_owns_company(game_state, 0, True)

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
        corp.set_owns_company(game_state, cdg, True)
        COMPANIES[mad].transfer_to_corp(game_state, 0)
        corp.set_owns_company(game_state, mad, True)

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
        corp.set_owns_company(game_state, 0, True)

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
        FI.set_owns_company(game_state, 0, True)

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
            FI.set_owns_company(game_state, cid, True)

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
        from core.data import get_company_income, get_company_stars, get_cost_of_ownership

        pr = CORPS[4]  # CORP_PR
        pr.set_active(game_state, True)

        # Give PR three companies (0, 1, 2)
        for cid in [0, 1, 2]:
            COMPANIES[cid].transfer_to_corp(game_state, 4)
            pr.set_owns_company(game_state, cid, True)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected: base income - CoO + company_count
        expected = 0
        for cid in [0, 1, 2]:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
        expected += 3  # +1 per company

        income = pr.calculate_income(game_state)
        assert income == expected

    def test_da_with_multiple_companies(self, game_state):
        """CSA-02: DA with companies of different FVs -> bonus = printed income of highest FV."""
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.turn import TURN
        from core.data import (
            get_company_income, get_company_stars, get_cost_of_ownership,
            COMPANY_NAME_TO_ID, COMPANY_FACE_VALUE
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
            da.set_owns_company(game_state, cid, True)

        coo_level = TURN.get_coo_level(game_state)

        # Find highest FV and its income
        highest_fv = max(COMPANY_FACE_VALUE[cid] for cid in companies)
        highest_fv_income = max(
            (get_company_income(cid) for cid in companies if COMPANY_FACE_VALUE[cid] == highest_fv),
            default=0
        )

        # Calculate expected: base income - CoO + highest_fv_income
        expected = 0
        for cid in companies:
            expected += get_company_income(cid)
            expected -= get_cost_of_ownership(coo_level, get_company_stars(cid))
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
            s.set_owns_company(game_state, cid, True)

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
            s.set_owns_company(game_state, cid, True)

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
            vm.set_owns_company(game_state, cid, True)

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
            vm.set_owns_company(game_state, cid, True)

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
        js.set_owns_company(game_state, 0, True)

        coo_level = TURN.get_coo_level(game_state)

        # Calculate expected: just base income - CoO (no special ability bonus)
        expected = get_company_income(0)
        expected -= get_cost_of_ownership(coo_level, get_company_stars(0))

        income = js.calculate_income(game_state)
        assert income == expected


# =============================================================================
# Phase 23: Phase Integration
# =============================================================================

# TODO: Add tests for INC-06, TRN-01 through TRN-04
