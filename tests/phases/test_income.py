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

# TODO: Add tests for INC-01 through INC-05, SYN-03, CSA-01 through CSA-04


# =============================================================================
# Phase 23: Phase Integration
# =============================================================================

# TODO: Add tests for INC-06, TRN-01 through TRN-04
