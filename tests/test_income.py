"""Tests for Income phase."""

import pytest
from state import GameState
from phases.income import IncomePhase
from data import (
    py_get_company_income, py_get_company_stars, py_get_cost_of_ownership,
    py_get_company_synergy, COMPANY_NAME_TO_ID
)

from tests.test_common import (
    StateBuilder, PHASE_INCOME, PHASE_DIVIDENDS,
    CORP_JS, CORP_S, CORP_OS, CORP_SM, CORP_PR, CORP_DA, CORP_VM, CORP_SI
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def state():
    """Create a basic 3-player game state in INCOME phase."""
    s = GameState(3)
    s.phase = PHASE_INCOME
    s.coo_level = 1
    s.active_player = 0
    return s


@pytest.fixture
def handler():
    """Get income phase handler for 3 players."""
    return IncomePhase(3)


@pytest.fixture
def builder(state):
    """Create a StateBuilder for test setup."""
    return StateBuilder(state)


# =============================================================================
# FI INCOME TESTS
# =============================================================================

class TestFIIncome:
    """Test foreign investor income calculation."""

    def test_fi_gets_base_income(self, state, handler, builder):
        """FI always gets $5 base income."""
        builder.set_fi_cash(0)

        income = handler.calculate_fi_income(state)
        assert income == 5

    def test_fi_adds_company_income(self, state, handler, builder):
        """FI adds adjusted income from companies."""
        builder.set_fi_cash(0)
        builder.set_fi_owns_company(0, True)  # BME (income=1)

        income = handler.calculate_fi_income(state)
        # $5 base + $1 from BME
        assert income == 6

    def test_fi_income_applies_coo(self, state, handler, builder):
        """FI income applies cost of ownership."""
        state.coo_level = 5  # Red CoO = 4
        builder.set_fi_cash(0)
        builder.set_fi_owns_company(0, True)  # BME (income=1, CoO=4)

        income = handler.calculate_fi_income(state)
        # $5 base + (1 - 4) = $5 - 3 = $2
        assert income == 2


# =============================================================================
# PLAYER INCOME TESTS
# =============================================================================

class TestPlayerIncome:
    """Test player income calculation."""

    def test_player_income_from_company(self, state, handler, builder):
        """Player gets income from private companies."""
        builder.set_player_cash(0, 0)
        builder.set_player_owns_company(0, 0, True)  # BME (income=1)

        income = handler.calculate_player_income(state, 0)
        assert income == 1

    def test_player_income_applies_coo(self, state, handler, builder):
        """Player income applies cost of ownership."""
        state.coo_level = 5
        builder.set_player_cash(0, 10)
        builder.set_player_owns_company(0, 0, True)  # BME (income=1, CoO=4)

        income = handler.calculate_player_income(state, 0)
        # 1 - 4 = -3
        assert income == -3

    def test_player_no_companies_no_income(self, state, handler, builder):
        """Player with no companies gets no income."""
        builder.set_player_cash(0, 50)

        income = handler.calculate_player_income(state, 0)
        assert income == 0


# =============================================================================
# CORP INCOME TESTS
# =============================================================================

class TestCorpIncome:
    """Test corporation income calculation."""

    def test_corp_basic_income(self, state, handler, builder):
        """Corp gets income from companies."""
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (income=1)
        builder.set_player_president(0, CORP_JS, True)

        income = handler.calculate_corp_income(state, CORP_JS)
        assert income == 1

    def test_corp_income_applies_coo(self, state, handler, builder):
        """Corp income applies cost of ownership."""
        state.coo_level = 5
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (income=1, CoO=4)
        builder.set_player_president(0, CORP_JS, True)

        income = handler.calculate_corp_income(state, CORP_JS)
        # 1 - 4 = -3
        assert income == -3


class TestCorpSpecialAbilities:
    """Test corporation special abilities."""

    def test_pr_gets_bonus_per_company(self, state, handler, builder):
        """Prussian Railway gets +1 per company owned."""
        builder.set_corp_active(CORP_PR, True)
        builder.set_corp_cash(CORP_PR, 0)
        builder.set_corp_owns_company(CORP_PR, 0, True)  # BME
        builder.set_corp_owns_company(CORP_PR, 1, True)  # BSE
        builder.set_player_president(0, CORP_PR, True)

        income = handler.calculate_corp_income(state, CORP_PR)
        # BME (1) + BSE (1) + PR bonus (2 companies) = 1 + 1 + 2 = 4
        assert income == 4

    def test_da_gets_max_income_bonus(self, state, handler, builder):
        """Doppler AG gets +max printed income bonus."""
        builder.set_corp_active(CORP_DA, True)
        builder.set_corp_cash(CORP_DA, 0)
        # Company 0 = BME (income=1), Company 6 = WT (income=3)
        builder.set_corp_owns_company(CORP_DA, 0, True)  # BME (income=1)
        builder.set_corp_owns_company(CORP_DA, 6, True)  # WT (income=3)
        builder.set_player_president(0, CORP_DA, True)

        income = handler.calculate_corp_income(state, CORP_DA)
        # BME (1) + WT (3) + DA bonus (max=3) = 1 + 3 + 3 = 7
        assert income == 7

    def test_vm_reduces_coo(self, state, handler, builder):
        """Vintage Machinery reduces CoO by up to 10."""
        state.coo_level = 5  # Red CoO = 4
        builder.set_corp_active(CORP_VM, True)
        builder.set_corp_cash(CORP_VM, 0)
        builder.set_corp_owns_company(CORP_VM, 0, True)  # BME (income=1, CoO=4)
        builder.set_player_president(0, CORP_VM, True)

        income = handler.calculate_corp_income(state, CORP_VM)
        # BME adjusted (1-4=-3) + VM bonus (min(10, 4)=4) = -3 + 4 = 1
        assert income == 1

    def test_vm_capped_at_10(self, state, handler, builder):
        """VM bonus is capped at 10."""
        state.coo_level = 7  # Very high CoO
        builder.set_corp_active(CORP_VM, True)
        builder.set_corp_cash(CORP_VM, 0)
        # Give multiple companies with high CoO
        builder.set_corp_owns_company(CORP_VM, 0, True)  # BME (CoO=10 at level 7)
        builder.set_corp_owns_company(CORP_VM, 1, True)  # BSE (CoO=10)
        builder.set_player_president(0, CORP_VM, True)

        income = handler.calculate_corp_income(state, CORP_VM)
        # BME (1-10=-9) + BSE (1-10=-9) + VM bonus (min(10, 20)=10) = -18 + 10 = -8
        assert income == -8


class TestCorpSynergies:
    """Test corporation synergy calculations."""

    def test_synergy_adds_income(self, state, handler, builder):
        """Synergies between companies add income."""
        # BPM synergizes with BSE for +1 (directional, BPM->BSE only)
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_owns_company(CORP_JS, 1, True)  # BSE
        builder.set_corp_owns_company(CORP_JS, 4, True)  # BPM
        builder.set_player_president(0, CORP_JS, True)

        income = handler.calculate_corp_income(state, CORP_JS)
        # BSE (1) + BPM (2) + synergy bonus (BPM->BSE = 1) = 1 + 2 + 1 = 4
        assert income == 4

    def test_synergistic_corp_bonus(self, state, handler, builder):
        """Synergistic corp gets +1 per 2 synergy markers."""
        # BPM synergizes with BSE - that's 1 synergy connection
        builder.set_corp_active(CORP_S, True)
        builder.set_corp_cash(CORP_S, 0)
        builder.set_corp_owns_company(CORP_S, 1, True)  # BSE
        builder.set_corp_owns_company(CORP_S, 4, True)  # BPM
        builder.set_player_president(0, CORP_S, True)

        synergy_count = handler.calculate_corp_synergies(state, CORP_S)
        # Should be 1 synergy marker (BPM-BSE pair, directional)
        assert synergy_count == 1

        income = handler.calculate_corp_income(state, CORP_S)
        # BSE (1) + BPM (2) + synergy (BPM->BSE = 1) + S bonus (1//2=0) = 4
        assert income == 4


# =============================================================================
# BANKRUPTCY TESTS
# =============================================================================

class TestCorpBankruptcy:
    """Test corporation bankruptcy handling."""

    def test_negative_income_bankrupts_corp(self, state, handler, builder):
        """Corp with negative cash after income goes bankrupt."""
        state.coo_level = 5
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 0)
        builder.set_corp_owns_company(CORP_JS, 0, True)  # BME (adjusted=-3)
        builder.set_player_president(0, CORP_JS, True)
        builder.set_corp_price_index(CORP_JS, 10)

        handler.handle_income_phase(state)

        # Corp should be inactive (bankrupt)
        assert not builder.is_corp_active(CORP_JS)

    def test_positive_cash_survives(self, state, handler, builder):
        """Corp with enough cash survives negative income."""
        state.coo_level = 5
        builder.set_corp_active(CORP_JS, True)
        builder.set_corp_cash(CORP_JS, 10)  # Enough to absorb -3
        builder.set_corp_owns_company(CORP_JS, 0, True)
        builder.set_player_president(0, CORP_JS, True)

        handler.handle_income_phase(state)

        # Corp should still be active
        assert builder.is_corp_active(CORP_JS)
        # Cash should be 10 - 3 = 7
        assert builder.get_corp_cash(CORP_JS) == 7


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransition:
    """Test transition to Dividends phase."""

    def test_transitions_to_dividends(self, state, handler, builder):
        """Income phase transitions to Dividends."""
        handler.handle_income_phase(state)
        assert state.phase == PHASE_DIVIDENDS

    def test_applies_fi_income(self, state, handler, builder):
        """FI income is applied during phase."""
        builder.set_fi_cash(0)
        builder.set_fi_owns_company(0, True)  # BME

        handler.handle_income_phase(state)

        # FI should have $5 base + $1 from BME = $6
        assert builder.get_fi_cash() == 6

    def test_applies_player_income(self, state, handler, builder):
        """Player income is applied during phase."""
        builder.set_player_cash(0, 10)
        builder.set_player_owns_company(0, 0, True)  # BME

        handler.handle_income_phase(state)

        # Player should have $10 + $1 = $11
        assert builder.get_player_cash(0) == 11
