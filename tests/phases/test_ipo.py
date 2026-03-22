"""Tests for IPO phase (Phase 9)."""
import pytest
from core.state import get_layout
from core.data import (
    GamePhases, GameConstants,
    get_company_face_value, get_company_stars,
    get_par_price, get_par_index_for_slot, get_market_index,
    get_corp_share_count, PY_COMPANY_PRICE_DIVISOR,
)
from core.actions import get_valid_action_mask, get_action_layout
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.market import MARKET
from phases.ipo import (
    setup_ipo_phase_py,
    apply_ipo_action_py,
    apply_ipo_pass_py,
)
from tests.phases.conftest import assert_invariants


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ipo_state(game_state):
    """
    3-player game state set up at PHASE_IPO.

    By default:
    - No player-owned companies (phase will auto-transition to INVEST)
    """
    TURN.set_phase(game_state, GamePhases.PHASE_IPO)
    return game_state


@pytest.fixture
def ipo_state_with_company(game_state):
    """
    3-player game with one player-owned company ready for IPO.

    Company 14 (first yellow, FV=20, stars=3):
    - Owned by player 0
    - Valid par prices: 16, 18, 20, 22, 24, 27 (slots 0-5)
    - Player 0 has 100 cash

    All corps are inactive (available for IPO).
    """
    state = game_state

    # Transfer company 14 (FV=20, stars=3) to player 0
    COMPANIES[14].transfer_to_player(state, 0)

    # Player 0 has plenty of cash
    PLAYERS[0].set_cash(state, 100)

    # Set up IPO phase
    TURN.set_phase(state, GamePhases.PHASE_IPO)
    setup_ipo_phase_py(state)
    assert_invariants(state, "After setup_ipo_phase fixture")

    return state


@pytest.fixture
def ipo_state_multiple_companies(game_state):
    """
    3-player game with multiple player-owned companies for IPO processing order tests.

    Companies:
    - Company 30 (FV=45, stars=5, blue) - player 0
    - Company 22 (FV=30, stars=4, green) - player 1
    - Company 14 (FV=20, stars=3, yellow) - player 0

    Processing order should be: 30 (FV=45), 22 (FV=30), 14 (FV=20)
    """
    state = game_state

    # Transfer companies to players
    COMPANIES[30].transfer_to_player(state, 0)
    COMPANIES[22].transfer_to_player(state, 1)
    COMPANIES[14].transfer_to_player(state, 0)

    # Give players cash
    PLAYERS[0].set_cash(state, 200)
    PLAYERS[1].set_cash(state, 200)

    return state


# =============================================================================
# Basic IPO Mechanics
# =============================================================================


class TestBasicIPOMechanics:
    """Basic IPO mechanics (share distribution, cash flow, company transfer)."""

    def test_ipo_transfers_company_to_corp(self, ipo_state_with_company):
        """IPO transfers company from player to corporation."""
        state = ipo_state_with_company
        company = COMPANIES[14]

        # Verify company is player-owned before
        assert company.get_location(state) == CompanyLocation.LOC_PLAYER
        assert company.get_owner_id(state) == 0

        # Execute IPO: corp 0, par slot 0 (par price 16 for star=3)
        result = apply_ipo_action_py(state, 0, 0)
        assert result == 0
        assert_invariants(state, "After IPO transfers company to corp")

        # Company now belongs to corp
        assert company.get_location(state) == CompanyLocation.LOC_CORP
        assert company.get_owner_id(state) == 0  # corp 0

    def test_ipo_activates_corporation(self, ipo_state_with_company):
        """IPO activates the selected corporation."""
        state = ipo_state_with_company
        corp = CORPS[0]

        assert not corp.is_active(state)

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO activates corporation")

        assert corp.is_active(state)

    def test_ipo_sets_corp_price_index(self, ipo_state_with_company):
        """IPO sets corporation's price index based on par price."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Par slot 2 for star=3 (yellow) = par price 20, market index = 7
        par_index = get_par_index_for_slot(3, 2)
        par_price = get_par_price(par_index)
        market_index = get_market_index(par_price)

        apply_ipo_action_py(state, 0, 2)
        assert_invariants(state, "After IPO sets corp price index")

        assert corp.get_price_index(state) == market_index
        assert corp.get_share_price(state) == par_price

    def test_ipo_claims_market_space(self, ipo_state_with_company):
        """IPO claims the selected market space."""
        state = ipo_state_with_company

        par_index = get_par_index_for_slot(3, 0)  # First valid par for star=3
        par_price = get_par_price(par_index)
        market_index = get_market_index(par_price)

        assert MARKET.is_space_available(state, market_index)

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO claims market space")

        assert not MARKET.is_space_available(state, market_index)

    def test_ipo_sets_corp_stars(self, ipo_state_with_company):
        """IPO sets full owned stars (companies + cash/10 + SI bonus)."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14 has 3 stars (yellow), par slot 0 -> par=16, FV=20 > par
        # 2 shares each -> corp cash = (2*16-20) + (2*16) = 44
        # Owned stars = 3 (company) + 44//10 (cash) = 7
        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO sets corp stars")

        company_stars = get_company_stars(14)  # 3
        corp_cash = corp.get_cash(state)       # 44
        expected = company_stars + corp_cash // 10
        assert corp.get_stars(state) == expected


# =============================================================================
# Share Distribution Rules
# =============================================================================


class TestShareDistribution:
    """Share distribution rules (FV > par vs FV <= par)."""

    def test_fv_greater_than_par_gives_2_shares_each(self, ipo_state_with_company):
        """When FV > par, player and bank each get 2 shares."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14: FV=20, par slot 0 -> par_price=16 (FV > par)
        par_index = get_par_index_for_slot(3, 0)
        par_price = get_par_price(par_index)
        assert get_company_face_value(14) > par_price

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO FV > par")

        # Player gets 2 shares
        assert PLAYERS[0].get_shares(state, 0) == 2
        # issued_shares = player_shares + bank_shares = 2 + 2 = 4
        assert corp.get_issued_shares(state) == 4
        assert corp.get_bank_shares(state) == 2
        # Unissued = total - player - bank
        total = get_corp_share_count(0)
        assert corp.get_unissued_shares(state) == total - 4

    def test_fv_equal_to_par_gives_1_share_each(self, ipo_state_with_company):
        """When FV == par, player and bank each get 1 share."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14: FV=20, par slot 2 -> par_price=20 (FV == par)
        par_index = get_par_index_for_slot(3, 2)
        par_price = get_par_price(par_index)
        assert get_company_face_value(14) == par_price

        apply_ipo_action_py(state, 0, 2)
        assert_invariants(state, "After IPO FV == par")

        assert PLAYERS[0].get_shares(state, 0) == 1
        # issued_shares = player_shares + bank_shares = 1 + 1 = 2
        assert corp.get_issued_shares(state) == 2
        assert corp.get_bank_shares(state) == 1
        total = get_corp_share_count(0)
        assert corp.get_unissued_shares(state) == total - 2

    def test_fv_less_than_par_gives_1_share_each(self, ipo_state_with_company):
        """When FV < par, player and bank each get 1 share."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14: FV=20, par slot 5 -> par_price=27 (FV < par)
        par_index = get_par_index_for_slot(3, 5)
        par_price = get_par_price(par_index)
        assert get_company_face_value(14) < par_price

        apply_ipo_action_py(state, 0, 5)
        assert_invariants(state, "After IPO FV < par")

        assert PLAYERS[0].get_shares(state, 0) == 1
        # issued_shares = player_shares + bank_shares = 1 + 1 = 2
        assert corp.get_issued_shares(state) == 2
        assert corp.get_bank_shares(state) == 1


class TestPaymentCalculation:
    """Payment calculations."""

    def test_player_payment_formula(self, ipo_state_with_company):
        """Player pays (shares * par) - face_value."""
        state = ipo_state_with_company

        initial_cash = PLAYERS[0].get_cash(state)
        face_value = get_company_face_value(14)  # 20

        # Par slot 0 -> par_price=16, FV > par -> 2 shares
        par_index = get_par_index_for_slot(3, 0)
        par_price = get_par_price(par_index)
        player_shares = 2  # FV > par

        expected_payment = (player_shares * par_price) - face_value

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO player payment")

        assert PLAYERS[0].get_cash(state) == initial_cash - expected_payment

    def test_corp_receives_both_payments(self, ipo_state_with_company):
        """Corp receives player payment + bank payment."""
        state = ipo_state_with_company
        corp = CORPS[0]

        face_value = get_company_face_value(14)  # 20

        # Par slot 0 -> par_price=16, FV > par -> 2 shares each
        par_index = get_par_index_for_slot(3, 0)
        par_price = get_par_price(par_index)
        player_shares = 2
        bank_shares = 2

        player_payment = (player_shares * par_price) - face_value
        bank_payment = bank_shares * par_price
        expected_corp_cash = player_payment + bank_payment

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After IPO corp receives payment")

        assert corp.get_cash(state) == expected_corp_cash


# =============================================================================
# Processing Order
# =============================================================================


class TestProcessingOrder:
    """Processing order (descending face value)."""

    def test_highest_fv_processed_first(self, ipo_state_multiple_companies):
        """Highest face value company is processed first."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)
        assert_invariants(state, "After setup_ipo_phase highest FV")

        # Company 30 has highest FV (45)
        ipo_company = TURN.get_ipo_company(state)
        assert ipo_company == 30

    def test_processing_order_descending_fv(self, ipo_state_multiple_companies):
        """Companies processed in descending face value order."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)
        assert_invariants(state, "After setup_ipo_phase descending FV")

        # First: company 30 (FV=45)
        assert TURN.get_ipo_company(state) == 30
        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass on company 30")

        # Second: company 22 (FV=30)
        assert TURN.get_ipo_company(state) == 22
        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass on company 22")

        # Third: company 14 (FV=20)
        assert TURN.get_ipo_company(state) == 14
        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass on company 14")

        # Phase should end - transitions to INVEST for new turn
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST


# =============================================================================
# Corp and Market Space Availability
# =============================================================================


class TestCorpAvailability:
    """Corp charter availability (can't use active corp)."""

    def test_cannot_ipo_to_active_corp(self, ipo_state_with_company):
        """Cannot select a corporation that's already active."""
        state = ipo_state_with_company

        # Float corp 0 using a different company so company 14 remains available
        COMPANIES[0].transfer_to_player(state, 1)  # Different player to avoid conflict
        CORPS[0].float_corp(state, 1, 0, 10, 1)

        # Try to IPO to corp 0 - should fail (corp already active)
        result = apply_ipo_action_py(state, 0, 0)
        assert result == 1  # Invalid

    def test_mask_excludes_active_corps(self, ipo_state_with_company):
        """Action mask excludes IPO actions for active corps."""
        state = ipo_state_with_company

        # Float corp 0 using a different company
        COMPANIES[0].transfer_to_player(state, 1)
        CORPS[0].float_corp(state, 1, 0, 10, 1)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Corp 0 actions should be masked out
        for slot in range(8):  # MAX_PAR_SLOTS
            action_idx = layout['ipo_base'] + 0 * 8 + slot
            assert mask[action_idx] == 0.0


class TestMarketSpaceAvailability:
    """Market space availability (can't use occupied space)."""

    def test_mask_excludes_occupied_spaces(self, ipo_state_with_company):
        """Action mask excludes IPO actions for occupied market spaces."""
        state = ipo_state_with_company

        # Occupy the market space for par slot 0 (par=16 for star=3)
        par_index = get_par_index_for_slot(3, 0)
        par_price = get_par_price(par_index)
        market_index = get_market_index(par_price)
        MARKET.set_space_available(state, market_index, False)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Par slot 0 actions should be masked out for all corps
        for corp_id in range(int(GameConstants.NUM_CORPS)):
            action_idx = layout['ipo_base'] + corp_id * 8 + 0
            assert mask[action_idx] == 0.0


# =============================================================================
# Cost Validation
# =============================================================================


class TestCostValidation:
    """Cost validation (player must afford payment)."""

    def test_mask_excludes_unaffordable_options(self, ipo_state_with_company):
        """Action mask excludes IPO actions player can't afford."""
        state = ipo_state_with_company

        # Give player very little cash
        PLAYERS[0].set_cash(state, 5)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Company 14: FV=20
        # For each valid par price, check if player can afford
        # Par slot 0: par=16, FV > par, 2 shares, cost = 2*16 - 20 = 12 (can't afford)
        # Par slot 2: par=20, FV == par, 1 share, cost = 1*20 - 20 = 0 (can afford)

        # Slot 0 should be masked (cost > 5)
        assert mask[layout['ipo_base'] + 0 * 8 + 0] == 0.0

        # Slot 2 should be valid (cost = 0)
        assert mask[layout['ipo_base'] + 0 * 8 + 2] == 1.0


# =============================================================================
# Pass Action
# =============================================================================


class TestPassAction:
    """Pass action (skip IPO for this company)."""

    def test_pass_advances_to_next_company(self, ipo_state_multiple_companies):
        """Pass action advances to next company in processing order."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)
        assert_invariants(state, "After setup_ipo_phase pass advances")

        assert TURN.get_ipo_company(state) == 30

        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass advances to next company")

        assert TURN.get_ipo_company(state) == 22

    def test_pass_clears_remaining_flag(self, ipo_state_with_company):
        """Pass action clears ipo_remaining flag for that company."""
        state = ipo_state_with_company

        assert TURN.is_ipo_remaining(state, 14)

        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass clears remaining flag")

        assert not TURN.is_ipo_remaining(state, 14)

    def test_pass_does_not_activate_corp(self, ipo_state_with_company):
        """Pass action does not activate any corporation."""
        state = ipo_state_with_company

        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass does not activate corp")

        for corp_id in range(int(GameConstants.NUM_CORPS)):
            assert not CORPS[corp_id].is_active(state)


# =============================================================================
# Phase Transitions
# =============================================================================


class TestPhaseTransitions:
    """Phase transitions (ISSUE_SHARES -> IPO -> INVEST)."""

    def test_empty_ipo_transitions_to_invest(self, ipo_state):
        """IPO with no player-owned companies transitions to INVEST."""
        state = ipo_state

        setup_ipo_phase_py(state)
        assert_invariants(state, "After setup_ipo_phase empty transitions to invest")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_all_passed_transitions_to_invest(self, ipo_state_with_company):
        """After passing on all companies, transitions to INVEST."""
        state = ipo_state_with_company

        apply_ipo_pass_py(state)
        assert_invariants(state, "After all passed transitions to invest")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_all_ipo_transitions_to_invest(self, ipo_state_with_company):
        """After IPO-ing all companies, transitions to INVEST."""
        state = ipo_state_with_company

        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After all IPO transitions to invest")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST


# =============================================================================
# Action Mask Validation
# =============================================================================


class TestActionMask:
    """Action mask validation."""

    def test_pass_always_valid(self, ipo_state_with_company):
        """Pass action is always valid in IPO phase."""
        state = ipo_state_with_company

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['ipo_pass']] == 1.0

    def test_invalid_par_prices_not_in_mask(self, ipo_state_with_company):
        """Invalid par prices for company's color are not in mask."""
        state = ipo_state_with_company

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Company 14 is star=3, valid slots are 0-5 (6 valid par prices)
        # Slot 6 and 7 should be invalid for all corps
        for corp_id in range(int(GameConstants.NUM_CORPS)):
            # Check slots beyond valid range
            for slot in range(6, 8):
                action_idx = layout['ipo_base'] + corp_id * 8 + slot
                assert mask[action_idx] == 0.0, (
                    f"Invalid slot {slot} for corp {corp_id} should be masked"
                )


# =============================================================================
# Active Player Setting
# =============================================================================


class TestActivePlayer:
    """Active player setting (company owner becomes active)."""

    def test_company_owner_becomes_active(self, ipo_state_multiple_companies):
        """The owner of the current IPO company is the active player."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)
        assert_invariants(state, "After setup_ipo_phase active player")

        # Company 30 owned by player 0
        assert TURN.get_ipo_company(state) == 30
        assert state.get_active_player() == 0

        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass on company 30 active player")

        # Company 22 owned by player 1
        assert TURN.get_ipo_company(state) == 22
        assert state.get_active_player() == 1

        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass on company 22 active player")

        # Company 14 owned by player 0
        assert TURN.get_ipo_company(state) == 14
        assert state.get_active_player() == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIPOIntegration:
    """Integration tests for IPO phase."""

    def test_full_ipo_flow(self, ipo_state_with_company):
        """Complete IPO flow maintains game invariants."""
        state = ipo_state_with_company

        assert_invariants(state, "Before IPO")

        # Execute IPO
        apply_ipo_action_py(state, 0, 0)

        assert_invariants(state, "After IPO")

        # Verify end state
        assert CORPS[0].is_active(state)
        assert PLAYERS[0].is_president_of(state, 0)
        assert COMPANIES[14].get_location(state) == CompanyLocation.LOC_CORP

    def test_mixed_ipo_and_pass(self, ipo_state_multiple_companies):
        """Mixed IPO and pass actions work correctly."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)

        assert_invariants(state, "Before any action")

        # IPO company 30 (player 0's blue company)
        # Use corp 0, par slot 0 for star=5 (par=30)
        apply_ipo_action_py(state, 0, 0)
        assert_invariants(state, "After first IPO")
        assert CORPS[0].is_active(state)

        # Pass on company 22 (player 1)
        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass")

        # IPO company 14 (player 0's yellow company)
        # Use corp 1, par slot 0 for star=3 (par=16)
        apply_ipo_action_py(state, 1, 0)
        assert_invariants(state, "After second IPO")
        assert CORPS[1].is_active(state)

        # Should be in INVEST now (new turn)
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST


# =============================================================================
# ACTIVE COMPANY TESTS
# =============================================================================

class TestActiveCompanyIPO:
    """Test active company block during IPO phase."""

    def test_active_company_set_during_ipo(self, ipo_state_with_company):
        """Active company block matches the company being considered for IPO."""
        state = ipo_state_with_company
        company_id = TURN.get_ipo_company(state)
        assert company_id >= 0

        layout = get_layout(3)
        expected_fv = get_company_face_value(company_id) / PY_COMPANY_PRICE_DIVISOR
        assert abs(state._array[layout.active_company_face_value_offset] - expected_fv) < 1e-6
        assert state._array[layout.active_company_stars_offset] > 0.0  # stars > 0

    def test_active_company_cleared_after_ipo_phase_ends(self, ipo_state_with_company):
        """Active company block is zeroed after IPO phase transitions to INVEST."""
        state = ipo_state_with_company

        # Pass on the IPO offer
        apply_ipo_pass_py(state)

        # Should be in INVEST now
        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

        layout = get_layout(3)
        for offset_name in ('active_company_stars_offset', 'active_company_low_price_offset',
                            'active_company_face_value_offset', 'active_company_high_price_offset',
                            'active_company_income_offset'):
            assert state._array[getattr(layout, offset_name)] == 0.0, (
                f"{offset_name} should be 0 after IPO phase ends"
            )

    def test_active_company_updates_between_ipo_companies(self, ipo_state_multiple_companies):
        """Active company block updates when advancing to the next IPO company."""
        state = ipo_state_multiple_companies
        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)

        layout = get_layout(3)

        first_company = TURN.get_ipo_company(state)
        assert first_company >= 0
        first_fv = get_company_face_value(first_company) / PY_COMPANY_PRICE_DIVISOR
        assert abs(state._array[layout.active_company_face_value_offset] - first_fv) < 1e-6

        # IPO the first company (corp 0, par slot 0)
        apply_ipo_action_py(state, 0, 0)

        second_company = TURN.get_ipo_company(state)
        if second_company >= 0:
            second_fv = get_company_face_value(second_company) / PY_COMPANY_PRICE_DIVISOR
            assert abs(state._array[layout.active_company_face_value_offset] - second_fv) < 1e-6
            assert second_fv != first_fv or second_company != first_company
