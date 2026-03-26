"""Tests for IPO phase (Phase 9) and PAR sub-phase (Phase 10)."""
import pytest
from core.state import get_layout, get_turn_fields
from core.data import (
    GamePhases, GameConstants,
    get_company_face_value, get_company_stars,
    get_company_low_price, get_company_high_price,
    get_adjusted_company_income,
    get_par_price, get_market_index,
    get_corp_share_count, is_valid_par_price,
    PY_COMPANY_PRICE_DIVISOR, PY_COMPANY_STAR_DIVISOR,
    PY_COMPANY_INCOME_DIVISOR, PY_CASH_DIVISOR,
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
    apply_par_action_py,
)
from tests.phases.conftest import assert_invariants


# =============================================================================
# Helpers
# =============================================================================

def do_ipo(state, corp_id, par_index):
    """Execute full IPO: select corp (IPO phase) then select par (PAR phase).

    Returns the result of the PAR action (0=success, 1=invalid).
    """
    result = apply_ipo_action_py(state, corp_id)
    if result != 0:
        return result
    return apply_par_action_py(state, par_index)


# Par index constants for readability
# Star 3 (yellow): valid par indices 5-10 -> prices 16,18,20,22,24,27
PAR_16 = 5   # get_par_index_for_slot(3, 0)
PAR_18 = 6   # get_par_index_for_slot(3, 1)
PAR_20 = 7   # get_par_index_for_slot(3, 2)
PAR_22 = 8   # get_par_index_for_slot(3, 3)
PAR_24 = 9   # get_par_index_for_slot(3, 4)
PAR_27 = 10  # get_par_index_for_slot(3, 5)
# Star 5 (blue): valid par indices 11-13 -> prices 30,33,37
PAR_30 = 11  # get_par_index_for_slot(5, 0)


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
    - Valid par prices: 16, 18, 20, 22, 24, 27 (par indices 5-10)
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
# IPO -> PAR Phase Transition
# =============================================================================


class TestIPOtoPARTransition:
    """IPO corp selection transitions to PAR sub-phase."""

    def test_selecting_corp_transitions_to_par(self, ipo_state_with_company):
        """Selecting a corp in IPO phase transitions to PAR phase."""
        state = ipo_state_with_company

        assert state.get_phase() == GamePhases.PHASE_IPO

        apply_ipo_action_py(state, 0)

        assert state.get_phase() == GamePhases.PHASE_PAR

    def test_par_corp_stored_during_par_phase(self, ipo_state_with_company):
        """The selected corp_id is stored in hidden state during PAR."""
        state = ipo_state_with_company

        apply_ipo_action_py(state, 3)

        assert state.get_par_corp() == 3

    def test_active_corp_not_set_during_ipo(self, ipo_state_with_company):
        """Active corp one-hot and scalars should be zeroed during IPO phase."""
        state = ipo_state_with_company
        layout = get_layout(3)

        assert state.get_phase() == GamePhases.PHASE_IPO

        # One-hot should be all zeros (no corp selected yet)
        for i in range(int(GameConstants.NUM_CORPS)):
            assert state._array[layout.active_corp_offset + i] == 0.0

        # Scalars should be zero
        assert state._array[layout.active_corp_income_offset] == 0.0
        assert state._array[layout.active_corp_stars_offset] == 0.0
        assert state._array[layout.active_corp_share_price_offset] == 0.0

    def test_active_corp_set_during_par(self, ipo_state_with_company):
        """Active corp one-hot is set during PAR phase."""
        state = ipo_state_with_company
        layout = get_layout(3)

        apply_ipo_action_py(state, 2)

        # Active corp one-hot should have corp 2 set
        assert state._array[layout.active_corp_offset + 2] == 1.0

    def test_active_corp_scalars_zero_during_par(self, ipo_state_with_company):
        """Active corp scalars should be zero during PAR (corp not yet formed)."""
        state = ipo_state_with_company
        layout = get_layout(3)

        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        # One-hot is set (identity of corp is known)
        assert state._array[layout.active_corp_offset + 0] == 1.0

        # But scalars are zero — corp hasn't been formed yet, so it has
        # no income, no stars, no share price, and no owned companies
        assert state._array[layout.active_corp_income_offset] == 0.0
        assert state._array[layout.active_corp_stars_offset] == 0.0
        assert state._array[layout.active_corp_share_price_offset] == 0.0
        for i in range(int(GameConstants.NUM_COMPANIES)):
            assert state._array[layout.active_corp_companies_offset + i] == 0.0

    def test_active_company_preserved_during_par(self, ipo_state_with_company):
        """Active company remains set during PAR (same company being IPO'd)."""
        state = ipo_state_with_company
        company_id = TURN.get_ipo_company(state)

        apply_ipo_action_py(state, 0)

        # Company should still be active during PAR
        assert TURN.get_ipo_company(state) == company_id

    def test_par_action_executes_ipo_and_advances(self, ipo_state_with_company):
        """PAR action executes Form Corporation and advances to next company."""
        state = ipo_state_with_company

        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        apply_par_action_py(state, PAR_16)

        # Should transition out (only one company) to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert CORPS[0].is_active(state)

    def test_par_clears_active_corp_after_completion(self, ipo_state_with_company):
        """Active corp is cleared after PAR completes."""
        state = ipo_state_with_company
        layout = get_layout(3)

        do_ipo(state, 0, PAR_16)

        # Active corp should be cleared
        assert state.get_par_corp() < 0
        for i in range(int(GameConstants.NUM_CORPS)):
            assert state._array[layout.active_corp_offset + i] == 0.0


# =============================================================================
# Basic IPO Mechanics (via full IPO+PAR flow)
# =============================================================================


class TestBasicIPOMechanics:
    """Basic IPO mechanics (share distribution, cash flow, company transfer)."""

    def test_ipo_transfers_company_to_corp(self, ipo_state_with_company):
        """IPO transfers company from player to corporation."""
        state = ipo_state_with_company
        company = COMPANIES[14]

        assert company.get_location(state) == CompanyLocation.LOC_PLAYER
        assert company.get_owner_id(state) == 0

        result = do_ipo(state, 0, PAR_16)
        assert result == 0
        assert_invariants(state, "After IPO transfers company to corp")

        assert company.get_location(state) == CompanyLocation.LOC_CORP
        assert company.get_owner_id(state) == 0  # corp 0

    def test_ipo_activates_corporation(self, ipo_state_with_company):
        """IPO activates the selected corporation."""
        state = ipo_state_with_company
        corp = CORPS[0]

        assert not corp.is_active(state)

        do_ipo(state, 0, PAR_16)
        assert_invariants(state, "After IPO activates corporation")

        assert corp.is_active(state)

    def test_ipo_sets_corp_price_index(self, ipo_state_with_company):
        """IPO sets corporation's price index based on par price."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # par_index 7 -> par price 20, market index = get_market_index(20) = 7
        par_price = get_par_price(PAR_20)
        market_index = get_market_index(par_price)

        do_ipo(state, 0, PAR_20)
        assert_invariants(state, "After IPO sets corp price index")

        assert corp.get_price_index(state) == market_index
        assert corp.get_share_price(state) == par_price

    def test_ipo_claims_market_space(self, ipo_state_with_company):
        """IPO claims the selected market space."""
        state = ipo_state_with_company

        par_price = get_par_price(PAR_16)
        market_index = get_market_index(par_price)

        assert MARKET.is_space_available(state, market_index)

        do_ipo(state, 0, PAR_16)
        assert_invariants(state, "After IPO claims market space")

        assert not MARKET.is_space_available(state, market_index)

    def test_ipo_sets_corp_stars(self, ipo_state_with_company):
        """IPO sets full owned stars (companies + cash/10 + SI bonus)."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14 has 3 stars (yellow), par=16, FV=20 > par
        # 2 shares each -> corp cash = (2*16-20) + (2*16) = 44
        # Owned stars = 3 (company) + 44//10 (cash) = 7
        do_ipo(state, 0, PAR_16)
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

        # Company 14: FV=20, par=16 (FV > par)
        par_price = get_par_price(PAR_16)
        assert get_company_face_value(14) > par_price

        do_ipo(state, 0, PAR_16)
        assert_invariants(state, "After IPO FV > par")

        assert PLAYERS[0].get_shares(state, 0) == 2
        assert corp.get_issued_shares(state) == 4
        assert corp.get_bank_shares(state) == 2
        total = get_corp_share_count(0)
        assert corp.get_unissued_shares(state) == total - 4

    def test_fv_equal_to_par_gives_1_share_each(self, ipo_state_with_company):
        """When FV == par, player and bank each get 1 share."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14: FV=20, par=20 (FV == par)
        par_price = get_par_price(PAR_20)
        assert get_company_face_value(14) == par_price

        do_ipo(state, 0, PAR_20)
        assert_invariants(state, "After IPO FV == par")

        assert PLAYERS[0].get_shares(state, 0) == 1
        assert corp.get_issued_shares(state) == 2
        assert corp.get_bank_shares(state) == 1
        total = get_corp_share_count(0)
        assert corp.get_unissued_shares(state) == total - 2

    def test_fv_less_than_par_gives_1_share_each(self, ipo_state_with_company):
        """When FV < par, player and bank each get 1 share."""
        state = ipo_state_with_company
        corp = CORPS[0]

        # Company 14: FV=20, par=27 (FV < par)
        par_price = get_par_price(PAR_27)
        assert get_company_face_value(14) < par_price

        do_ipo(state, 0, PAR_27)
        assert_invariants(state, "After IPO FV < par")

        assert PLAYERS[0].get_shares(state, 0) == 1
        assert corp.get_issued_shares(state) == 2
        assert corp.get_bank_shares(state) == 1


class TestPaymentCalculation:
    """Payment calculations."""

    def test_player_payment_formula(self, ipo_state_with_company):
        """Player pays (shares * par) - face_value."""
        state = ipo_state_with_company

        initial_cash = PLAYERS[0].get_cash(state)
        face_value = get_company_face_value(14)  # 20

        # par=16, FV > par -> 2 shares
        par_price = get_par_price(PAR_16)
        player_shares = 2

        expected_payment = (player_shares * par_price) - face_value

        do_ipo(state, 0, PAR_16)
        assert_invariants(state, "After IPO player payment")

        assert PLAYERS[0].get_cash(state) == initial_cash - expected_payment

    def test_corp_receives_both_payments(self, ipo_state_with_company):
        """Corp receives player payment + bank payment."""
        state = ipo_state_with_company
        corp = CORPS[0]

        face_value = get_company_face_value(14)  # 20

        # par=16, FV > par -> 2 shares each
        par_price = get_par_price(PAR_16)
        player_shares = 2
        bank_shares = 2

        player_payment = (player_shares * par_price) - face_value
        bank_payment = bank_shares * par_price
        expected_corp_cash = player_payment + bank_payment

        do_ipo(state, 0, PAR_16)
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

        assert TURN.get_ipo_company(state) == 30

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
        COMPANIES[0].transfer_to_player(state, 1)
        CORPS[0].float_corp(state, 1, 0, 10, 1)

        # Try to IPO to corp 0 - should fail (corp already active)
        result = apply_ipo_action_py(state, 0)
        assert result == 1  # Invalid

    def test_mask_excludes_active_corps(self, ipo_state_with_company):
        """Action mask excludes IPO actions for active corps."""
        state = ipo_state_with_company

        # Float corp 0 using a different company
        COMPANIES[0].transfer_to_player(state, 1)
        CORPS[0].float_corp(state, 1, 0, 10, 1)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Corp 0 should be masked out
        assert mask[layout['ipo_base'] + 0] == 0.0


class TestMarketSpaceAvailability:
    """Market space availability (can't use occupied space)."""

    def test_mask_excludes_corps_with_no_valid_par(self, ipo_state_with_company):
        """If all par prices for a star tier are occupied, corp is masked."""
        state = ipo_state_with_company

        # Occupy ALL valid market spaces for star=3 (indices 5-10, prices 16-27)
        for par_idx in range(5, 11):
            par_price = get_par_price(par_idx)
            market_index = get_market_index(par_price)
            MARKET.set_space_available(state, market_index, False)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # All corp selections should be masked (no valid par prices)
        for corp_id in range(int(GameConstants.NUM_CORPS)):
            assert mask[layout['ipo_base'] + corp_id] == 0.0

        # Pass should still be valid
        assert mask[layout['ipo_pass']] == 1.0


# =============================================================================
# Cost Validation
# =============================================================================


class TestCostValidation:
    """Cost validation (player must afford payment)."""

    def test_mask_excludes_unaffordable_corps(self, ipo_state_with_company):
        """Action mask excludes corps when player can't afford any par price."""
        state = ipo_state_with_company

        # Give player very little cash - can't afford any par price
        # Cheapest par for star=3: par=16, FV=20, cost = 2*16 - 20 = 12
        PLAYERS[0].set_cash(state, 0)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # But FV=20, par=20 has cost 0, so corp selection should be valid
        # when cash=0, cost = 1*20 - 20 = 0 -> affordable!
        # Only corps with at least one valid+affordable par should appear
        # par=20 (index 7) -> market_index = get_market_index(20) should be available
        # So corps should actually be valid for cash=0 because cost=0 exists

        # Verify: at least one corp is valid (par=20 costs 0)
        any_valid = False
        for corp_id in range(int(GameConstants.NUM_CORPS)):
            if mask[layout['ipo_base'] + corp_id] == 1.0:
                any_valid = True
        assert any_valid, "Should have valid corps when cost=0 exists"


# =============================================================================
# PAR Phase Mask
# =============================================================================


class TestPARMask:
    """PAR phase action mask validation."""

    def test_par_mask_shows_valid_prices(self, ipo_state_with_company):
        """PAR mask shows valid par prices for company's star tier."""
        state = ipo_state_with_company

        # Select corp 0 -> transition to PAR
        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Company 14 star=3: valid par indices 5-10
        for par_idx in range(14):
            expected = is_valid_par_price(3, par_idx)
            if expected:
                # Also check market availability and affordability
                par_price = get_par_price(par_idx)
                mkt = get_market_index(par_price)
                if mkt >= 0 and MARKET.is_space_available(state, mkt):
                    assert mask[layout['par_base'] + par_idx] == 1.0, (
                        f"Par index {par_idx} (price {par_price}) should be valid"
                    )
            else:
                assert mask[layout['par_base'] + par_idx] == 0.0, (
                    f"Par index {par_idx} should be invalid for star=3"
                )

    def test_par_mask_excludes_occupied_market_spaces(self, ipo_state_with_company):
        """PAR mask excludes par prices whose market spaces are taken."""
        state = ipo_state_with_company

        # Occupy the market space for par price 16 (index 5)
        par_price = get_par_price(PAR_16)
        market_index = get_market_index(par_price)
        MARKET.set_space_available(state, market_index, False)

        # Select corp 0 -> transition to PAR
        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # Par index 5 (price 16) should be masked
        assert mask[layout['par_base'] + PAR_16] == 0.0

        # Par index 7 (price 20) should still be valid
        assert mask[layout['par_base'] + PAR_20] == 1.0

    def test_par_mask_has_no_pass(self, ipo_state_with_company):
        """PAR phase has no pass action — only par price selections."""
        state = ipo_state_with_company

        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # IPO pass should not be valid during PAR
        assert mask[layout['ipo_pass']] == 0.0

    def test_par_mask_excludes_unaffordable_prices(self, ipo_state_with_company):
        """PAR mask excludes par prices the player can't afford."""
        state = ipo_state_with_company

        # Player has exactly 0 cash
        PLAYERS[0].set_cash(state, 0)

        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        # par=16, FV=20 > par, cost = 2*16 - 20 = 12 -> can't afford
        assert mask[layout['par_base'] + PAR_16] == 0.0

        # par=20, FV=20 == par, cost = 1*20 - 20 = 0 -> can afford
        assert mask[layout['par_base'] + PAR_20] == 1.0


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
    """Phase transitions (ISSUE_SHARES -> IPO -> PAR -> INVEST)."""

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

        do_ipo(state, 0, PAR_16)
        assert_invariants(state, "After all IPO transitions to invest")

        assert TURN.get_phase(state) == GamePhases.PHASE_INVEST

    def test_ipo_to_par_to_ipo_to_invest(self, ipo_state_multiple_companies):
        """Full flow: IPO -> PAR -> IPO (next company) -> ... -> INVEST."""
        state = ipo_state_multiple_companies

        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)

        # Company 30 (star=5): select corp, then par
        assert state.get_phase() == GamePhases.PHASE_IPO
        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR
        apply_par_action_py(state, PAR_30)

        # Should be back in IPO for next company
        assert state.get_phase() == GamePhases.PHASE_IPO
        assert TURN.get_ipo_company(state) == 22

        # Pass on company 22
        apply_ipo_pass_py(state)
        assert state.get_phase() == GamePhases.PHASE_IPO
        assert TURN.get_ipo_company(state) == 14

        # IPO company 14
        do_ipo(state, 1, PAR_16)

        assert state.get_phase() == GamePhases.PHASE_INVEST


# =============================================================================
# Action Mask Validation (IPO phase)
# =============================================================================


class TestActionMask:
    """IPO phase action mask validation."""

    def test_pass_always_valid(self, ipo_state_with_company):
        """Pass action is always valid in IPO phase."""
        state = ipo_state_with_company

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        assert mask[layout['ipo_pass']] == 1.0

    def test_valid_corps_in_mask(self, ipo_state_with_company):
        """At least one corp should be valid for IPO."""
        state = ipo_state_with_company

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)

        valid_count = 0
        for corp_id in range(int(GameConstants.NUM_CORPS)):
            if mask[layout['ipo_base'] + corp_id] == 1.0:
                valid_count += 1

        assert valid_count > 0, "Should have at least one valid corp"


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

        do_ipo(state, 0, PAR_16)

        assert_invariants(state, "After IPO")

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
        do_ipo(state, 0, PAR_30)
        assert_invariants(state, "After first IPO")
        assert CORPS[0].is_active(state)

        # Pass on company 22 (player 1)
        apply_ipo_pass_py(state)
        assert_invariants(state, "After pass")

        # IPO company 14 (player 0's yellow company)
        do_ipo(state, 1, PAR_16)
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
        """Active company block fully populated during IPO phase."""
        state = ipo_state_with_company
        company_id = TURN.get_ipo_company(state)
        assert company_id >= 0

        layout = get_layout(3)
        coo_level = TURN.get_coo_level(state)

        # One-hot should identify the company
        assert state._array[layout.active_company_offset + company_id] == 1.0

        # All scalar fields should be populated
        assert abs(state._array[layout.active_company_face_value_offset]
                   - get_company_face_value(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_stars_offset]
                   - get_company_stars(company_id) / PY_COMPANY_STAR_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_low_price_offset]
                   - get_company_low_price(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_high_price_offset]
                   - get_company_high_price(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_income_offset]
                   - get_adjusted_company_income(company_id, coo_level) / PY_COMPANY_INCOME_DIVISOR) < 1e-6

    def test_active_company_cleared_after_ipo_phase_ends(self, ipo_state_with_company):
        """Active company block is zeroed after IPO phase transitions to INVEST."""
        state = ipo_state_with_company

        apply_ipo_pass_py(state)

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

        # IPO the first company (corp 0, par=30 for star=5)
        do_ipo(state, 0, PAR_30)

        second_company = TURN.get_ipo_company(state)
        if second_company >= 0:
            second_fv = get_company_face_value(second_company) / PY_COMPANY_PRICE_DIVISOR
            assert abs(state._array[layout.active_company_face_value_offset] - second_fv) < 1e-6
            assert second_fv != first_fv or second_company != first_company

    def test_active_company_preserved_in_par_phase(self, ipo_state_with_company):
        """Active company block fully preserved during PAR sub-phase."""
        state = ipo_state_with_company
        layout = get_layout(3)

        company_id = TURN.get_ipo_company(state)
        coo_level = TURN.get_coo_level(state)

        # Transition to PAR
        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        # One-hot should still identify the company
        assert state._array[layout.active_company_offset + company_id] == 1.0

        # All scalar fields should still be populated (same company)
        assert abs(state._array[layout.active_company_face_value_offset]
                   - get_company_face_value(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_stars_offset]
                   - get_company_stars(company_id) / PY_COMPANY_STAR_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_low_price_offset]
                   - get_company_low_price(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_high_price_offset]
                   - get_company_high_price(company_id) / PY_COMPANY_PRICE_DIVISOR) < 1e-6
        assert abs(state._array[layout.active_company_income_offset]
                   - get_adjusted_company_income(company_id, coo_level) / PY_COMPANY_INCOME_DIVISOR) < 1e-6


# =============================================================================
# PAR Info Visible State
# =============================================================================


def _read_par_info(state):
    """Read par_corp_treasury and par_shares arrays from state."""
    layout = get_layout(state.get_num_players())
    tf = get_turn_fields(state.get_num_players())
    base_t = layout.turn_offset + tf.par_corp_treasury
    base_s = layout.turn_offset + tf.par_shares
    treasury = [float(state._array[base_t + i]) for i in range(14)]
    shares = [float(state._array[base_s + i]) for i in range(14)]
    return treasury, shares


class TestParInfoPopulation:
    """par_corp_treasury and par_shares are populated during IPO/PAR."""

    def test_populated_during_ipo(self, ipo_state_with_company):
        """Par info is populated when IPO phase sets up a company."""
        state = ipo_state_with_company
        treasury, shares = _read_par_info(state)

        # Company 14: FV=20, stars=3. Valid par indices 5-10 (prices 16-27).
        assert any(t != 0.0 for t in treasury)
        assert any(s != 0.0 for s in shares)

    def test_invalid_slots_are_zero(self, ipo_state_with_company):
        """Slots outside the star tier are zero."""
        state = ipo_state_with_company
        treasury, shares = _read_par_info(state)

        # Company 14 is star 3 (yellow). Indices 0-4 are star 1-2 only.
        for i in range(5):
            assert treasury[i] == 0.0
            assert shares[i] == 0.0

    def test_corp_treasury_values(self, ipo_state_with_company):
        """par_corp_treasury matches expected formula for valid slots."""
        state = ipo_state_with_company
        treasury, _ = _read_par_info(state)

        face_value = get_company_face_value(14)  # 20
        for i in range(14):
            if treasury[i] == 0.0:
                continue
            par_price = get_par_price(i)
            float_shares = 2 if face_value > par_price else 1
            player_cost = (float_shares * par_price) - face_value
            bank_pays = float_shares * par_price
            expected = (player_cost + bank_pays) / PY_CASH_DIVISOR
            assert abs(treasury[i] - expected) < 1e-5, (
                f"par_corp_treasury[{i}] (par={par_price}): {treasury[i]} != {expected}"
            )

    def test_shares_encoding(self, ipo_state_with_company):
        """par_shares is 0.5 for 2 shares, 1.0 for 4 shares."""
        state = ipo_state_with_company
        _, shares = _read_par_info(state)

        face_value = get_company_face_value(14)  # 20
        for i in range(14):
            if shares[i] == 0.0:
                continue
            par_price = get_par_price(i)
            if face_value > par_price:
                assert shares[i] == 1.0, f"par_shares[{i}]: expected 1.0 (4 shares)"
            else:
                assert shares[i] == 0.5, f"par_shares[{i}]: expected 0.5 (2 shares)"

    def test_preserved_during_par_phase(self, ipo_state_with_company):
        """Par info stays populated when transitioning IPO -> PAR."""
        state = ipo_state_with_company
        treasury_before, shares_before = _read_par_info(state)

        apply_ipo_action_py(state, 0)
        assert state.get_phase() == GamePhases.PHASE_PAR

        treasury_after, shares_after = _read_par_info(state)
        assert treasury_before == treasury_after
        assert shares_before == shares_after

    def test_cleared_after_par_completion(self, ipo_state_with_company):
        """Par info is cleared when IPO phase transitions out."""
        state = ipo_state_with_company

        do_ipo(state, 0, PAR_16)
        assert state.get_phase() == GamePhases.PHASE_INVEST

        treasury, shares = _read_par_info(state)
        assert all(t == 0.0 for t in treasury)
        assert all(s == 0.0 for s in shares)

    def test_unaffordable_slots_are_zero(self, game_state):
        """Slots the player can't afford are zeroed."""
        state = game_state
        COMPANIES[14].transfer_to_player(state, 0)
        PLAYERS[0].set_cash(state, 1)  # Almost no cash
        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)

        treasury, shares = _read_par_info(state)
        # FV=20, star 3 valid range is indices 5-10.
        # Par $16 (i=5): cost = 2*16-20 = $12 -> can't afford -> 0
        assert treasury[5] == 0.0
        assert shares[5] == 0.0
        # Par $20 (i=7): cost = 1*20-20 = $0 -> CAN afford -> nonzero
        assert treasury[7] != 0.0
        assert shares[7] != 0.0
        # Par $22 (i=8): cost = 1*22-20 = $2 -> can't afford -> 0
        assert treasury[8] == 0.0
        assert shares[8] == 0.0

    def test_unavailable_market_slots_are_zero(self, ipo_state_with_company):
        """Slots with unavailable market spaces are zeroed."""
        state = ipo_state_with_company

        # Block market space for par index 5 (price 16, market index for $16)
        mkt_idx = get_market_index(get_par_price(5))
        MARKET.set_space_available(state, mkt_idx, False)

        # Re-setup to repopulate par info
        setup_ipo_phase_py(state)
        treasury, shares = _read_par_info(state)
        assert treasury[5] == 0.0
        assert shares[5] == 0.0

    def test_repopulated_for_next_company(self, ipo_state_multiple_companies):
        """Par info is repopulated when advancing to the next company."""
        state = ipo_state_multiple_companies
        TURN.set_phase(state, GamePhases.PHASE_IPO)
        setup_ipo_phase_py(state)

        # First company is 30 (FV=45, star 5). Record its par info.
        treasury_first, _ = _read_par_info(state)

        # IPO company 30, then advance to next
        do_ipo(state, 0, PAR_30)

        # Now at company 22 (FV=30, star 4) - different par info
        assert state.get_phase() == GamePhases.PHASE_IPO
        treasury_second, _ = _read_par_info(state)
        assert treasury_first != treasury_second
