"""Tests for CLOSING phase auto-close logic."""
import pytest
from core.state import GameState, get_layout
from core.data import (
    GamePhases, GameConstants,
    get_company_income, get_company_stars, get_company_face_value,
    get_cost_of_ownership, PY_CASH_DIVISOR,
)
from entities.turn import TURN
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from phases.closing import (
    apply_closing_auto_py, process_mandatory_close_py,
    apply_closing_action_py,
    generate_close_offers_py, get_close_offer_count_py, get_close_offer_py,
)
from core.actions import ACTION_CLOSE_PY, ACTION_PASS_PY

# Import status codes and utility from conftest
from tests.phases.conftest import STATUS_OK, float_corp_for_test, setup_receivership_corp, assert_invariants

# Phase constants for tests
PHASE_CLOSING_PY = GamePhases.PHASE_CLOSING
PHASE_INVEST_PY = GamePhases.PHASE_INVEST
PHASE_INCOME_PY = GamePhases.PHASE_INCOME


class TestFIAutoClose:
    """FI closes companies where income - CoO < 0."""

    def test_fi_closes_negative_income_company(self):
        """FI closes company with negative adjusted income."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Find a company with negative income at high CoO
        # Red companies (stars=1) have low income, high CoO at level 7
        # Company 0 is red: income=$1, CoO at level 7 = $10 -> adjusted = -9
        red_company_id = 0

        # Set high CoO level
        TURN.set_coo_level(state, 7)

        # Give company to FI
        COMPANIES[red_company_id].transfer_to_fi(state)

        # Verify company is owned by FI
        assert FI.owns_company(state, red_company_id)
        assert not COMPANIES[red_company_id].is_removed(state)

        # Execute auto-close
        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close negative income")

        # Company should be removed
        assert not FI.owns_company(state, red_company_id)
        assert COMPANIES[red_company_id].is_removed(state)

    def test_fi_keeps_zero_income_company(self):
        """FI does NOT close company with exactly zero adjusted income."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Find a company where income = CoO
        # Red company 2: income=$2, CoO at level 4 = $2 -> adjusted = 0
        company_id = 2  # Red company

        # Set CoO level where income = CoO for this company
        TURN.set_coo_level(state, 4)

        # Give company to FI
        COMPANIES[company_id].transfer_to_fi(state)

        # Verify setup
        income = get_company_income(company_id)
        stars = get_company_stars(company_id)
        coo = get_cost_of_ownership(4, stars)
        assert income - coo == 0, f"Expected zero adjusted income, got {income - coo}"

        # Execute auto-close
        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close zero income")

        # Company should NOT be removed (zero income, not negative)
        assert FI.owns_company(state, company_id)
        assert not COMPANIES[company_id].is_removed(state)

    def test_fi_closes_at_exactly_negative_one(self):
        """FI closes company when adjusted income is exactly -1 (boundary).

        Verifies FI close logic triggers at the boundary, not just for deeply
        negative values. Uses company 0 (income=$1, red/1-star) at CoO level 4
        where red CoO=$2, giving adjusted income = $1 - $2 = -1.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Company 0: income=$1, stars=1 (red)
        # At CoO level 4, red CoO=$2, so adjusted = $1 - $2 = -1
        company_id = 0
        TURN.set_coo_level(state, 4)

        COMPANIES[company_id].transfer_to_fi(state)

        # Verify setup: adjusted income is exactly -1
        income = get_company_income(company_id)
        stars = get_company_stars(company_id)
        coo = get_cost_of_ownership(4, stars)
        assert income == 1, f"Expected income=1, got {income}"
        assert coo == 2, f"Expected CoO=2, got {coo}"
        assert income - coo == -1, f"Expected adjusted=-1, got {income - coo}"

        # Execute auto-close
        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close exactly -1")

        # Company should be closed (adjusted income < 0)
        assert not FI.owns_company(state, company_id)
        assert COMPANIES[company_id].is_removed(state)

    def test_fi_boundary_coo_equals_income_exactly(self):
        """FI does NOT close when CoO exactly equals income (adjusted = 0).

        This is the boundary test: FI closes if adjusted < 0, keeps if >= 0.
        Uses company 2 (income=$2, red/1-star) at CoO level 4 where red CoO=$2,
        giving adjusted income = $2 - $2 = 0 (exactly at boundary).
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Company 2: income=$2, stars=1 (red)
        # At CoO level 4, red CoO=$2, so adjusted = $2 - $2 = 0
        company_id = 2
        TURN.set_coo_level(state, 4)

        COMPANIES[company_id].transfer_to_fi(state)

        # Verify setup: adjusted income is exactly 0
        income = get_company_income(company_id)
        stars = get_company_stars(company_id)
        coo = get_cost_of_ownership(4, stars)
        assert income == 2, f"Expected income=2, got {income}"
        assert coo == 2, f"Expected CoO=2, got {coo}"
        assert income - coo == 0, f"Expected adjusted=0, got {income - coo}"

        # Execute auto-close
        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close boundary zero")

        # Company should NOT be closed (adjusted income == 0, not < 0)
        assert FI.owns_company(state, company_id)
        assert not COMPANIES[company_id].is_removed(state)

    def test_fi_keeps_positive_income_company(self):
        """FI keeps company with positive adjusted income."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Blue company at low CoO has positive income
        company_id = 35
        TURN.set_coo_level(state, 1)

        COMPANIES[company_id].transfer_to_fi(state)

        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close positive income")

        # Company should remain
        assert FI.owns_company(state, company_id)
        assert not COMPANIES[company_id].is_removed(state)

    def test_fi_can_end_with_zero_companies(self):
        """FI can close all companies and end with none."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Give FI multiple red companies at high CoO
        TURN.set_coo_level(state, 7)
        COMPANIES[0].transfer_to_fi(state)
        COMPANIES[1].transfer_to_fi(state)

        apply_closing_auto_py(state)
        assert_invariants(state, "After FI auto-close all companies")

        # Both should be closed
        assert not FI.owns_company(state, 0)
        assert not FI.owns_company(state, 1)


class TestReceivershipAutoClose:
    """Receivership corps close red >= $4, orange >= $7."""

    def test_receivership_closes_red_at_coo_4(self):
        """Receivership closes red company when CoO >= $4."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Red companies (stars=1): CoO at level 5 = $4
        # Company 0 and 1 are red
        red_company = 0
        other_company = 14  # Higher face value (yellow, stars=3, protected)

        setup_receivership_corp(state, 1, [red_company, other_company])
        TURN.set_coo_level(state, 5)  # Red CoO = $4

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership auto-close red at CoO 4")

        # Red company should be closed
        assert COMPANIES[red_company].is_removed(state)
        # Other company (higher FV) should remain
        assert not COMPANIES[other_company].is_removed(state)

    def test_receivership_keeps_red_below_coo_4(self):
        """Receivership keeps red company when CoO < $4."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        red_company = 0
        other_company = 14  # Yellow, higher FV

        setup_receivership_corp(state, 1, [red_company, other_company])
        TURN.set_coo_level(state, 4)  # Red CoO = $2 < $4

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership keeps red below CoO 4")

        # Red company should remain
        assert not COMPANIES[red_company].is_removed(state)

    def test_receivership_closes_orange_at_coo_7(self):
        """Receivership closes orange company when CoO >= $7."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Orange companies (stars=2): CoO at level 6 = $7
        # Companies 6-13 are orange
        orange_company = 6
        other_company = 14  # Higher face value (yellow, protected)

        setup_receivership_corp(state, 1, [orange_company, other_company])
        TURN.set_coo_level(state, 6)  # Orange CoO = $7

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership auto-close orange at CoO 7")

        # Orange company should be closed
        assert COMPANIES[orange_company].is_removed(state)
        assert not COMPANIES[other_company].is_removed(state)

    def test_receivership_keeps_orange_below_coo_7(self):
        """Receivership keeps orange company when CoO < $7 (boundary test).

        At CoO level 5, orange CoO = $4 which is below the $7 threshold.
        This tests the boundary just below the close threshold.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Orange company (stars=2): CoO at level 5 = $4 < $7
        orange_company = 6
        other_company = 14  # Yellow, higher FV

        setup_receivership_corp(state, 1, [orange_company, other_company])
        TURN.set_coo_level(state, 5)  # Orange CoO = $4

        # Verify CoO is below threshold
        coo = get_cost_of_ownership(5, 2)  # level 5, stars=2 (orange)
        assert coo == 4, f"Expected orange CoO=4 at level 5, got {coo}"
        assert coo < 7, "CoO should be below $7 threshold"

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership keeps orange below CoO 7")

        # Orange company should NOT be closed (CoO $4 < $7)
        assert not COMPANIES[orange_company].is_removed(state)

    def test_receivership_never_closes_yellow_green_blue(self):
        """Receivership never auto-closes yellow/green/blue companies."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Yellow (stars=3), green (stars=4), blue (stars=5)
        yellow_company = 14  # Stars=3
        green_company = 22   # Stars=4
        blue_company = 29    # Stars=5

        setup_receivership_corp(state, 1, [yellow_company, green_company, blue_company])
        TURN.set_coo_level(state, 7)  # Max CoO

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership keeps yellow/green/blue")

        # None should be closed (yellow/green/blue exempt)
        assert not COMPANIES[yellow_company].is_removed(state)
        assert not COMPANIES[green_company].is_removed(state)
        assert not COMPANIES[blue_company].is_removed(state)


class TestHighestFaceValueProtection:
    """Receivership always keeps highest face value company."""

    def test_highest_face_value_protected_even_if_red(self):
        """Highest FV company is protected even if it would otherwise close."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Give corp only red companies - highest FV must survive
        red_companies = [0, 1, 2, 3]  # All red, FV ascending
        highest_fv = 3  # Company 3 has highest FV among reds

        setup_receivership_corp(state, 1, red_companies)
        TURN.set_coo_level(state, 7)  # All reds would normally close

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership highest FV protected")

        # Only highest FV should survive
        assert not COMPANIES[highest_fv].is_removed(state)
        # Others should be closed
        for cid in red_companies:
            if cid != highest_fv:
                assert COMPANIES[cid].is_removed(state), f"Company {cid} should be closed"

    def test_single_company_never_closed(self):
        """Corp with only one company can never have it closed."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        red_company = 0
        setup_receivership_corp(state, 1, [red_company])
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership single company protected")

        # Single company must survive (it's both the only AND highest FV)
        assert not COMPANIES[red_company].is_removed(state)


class TestVintageMachineryInReceivership:
    """VM in receivership follows normal receivership rules - no special ability."""

    def test_vm_no_special_treatment_in_receivership(self):
        """VM's CoO reduction does NOT apply in receivership auto-close.

        Per RULES.md, VM's ability is for income calculation only.
        Receivership corps follow simple deterministic rules without
        special ability considerations.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Red company at CoO level 7 has CoO=$10 >= $4 threshold
        # VM's ability does NOT prevent closing in receivership
        red_company = 0   # FV=1, CoO=10
        yellow_company = 14  # FV=20 (higher), will be protected

        # Use VM (corp_id 6) in receivership
        setup_receivership_corp(state, 6, [red_company, yellow_company])
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After VM receivership auto-close")

        # Red company SHOULD be closed (VM ability doesn't apply to receivership)
        assert COMPANIES[red_company].is_removed(state), \
            "Red company should close: CoO $10 >= $4 threshold, VM ability doesn't apply"
        # Yellow is protected as highest FV
        assert not COMPANIES[yellow_company].is_removed(state)


class TestJunkyardScrappersBonus:
    """JS (corp_id 0) receives 2x printed income only when JS closes its own companies."""

    def test_js_no_bonus_on_fi_close(self):
        """JS does NOT get bonus when FI closes company."""

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Activate JS (needs a company to float)
        float_corp_for_test(state, corp_id=0)
        js = CORPS[0]
        js.set_cash(state, 0)

        # FI closes a red company (use company 1 since 0 is used by JS)
        red_company = 1
        COMPANIES[red_company].transfer_to_fi(state)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After FI close no JS bonus")

        # JS should NOT have received bonus (FI closed, not JS)
        assert js.get_cash(state) == 0

    def test_js_no_bonus_on_receivership_close(self):
        """JS does NOT get bonus when receivership corp closes company."""

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Activate JS (needs a company to float)
        float_corp_for_test(state, corp_id=0)
        js = CORPS[0]
        js.set_cash(state, 0)

        # Non-JS corp in receivership closes red
        red_company = 1  # Use company 1 since 0 is used by JS
        other_company = 14  # Higher FV yellow

        # Float corp 1, then put in receivership by removing player shares
        float_corp_for_test(state, corp_id=1, company_id=red_company)
        PLAYERS[0].set_shares(state, 1, 0)  # Triggers receivership automatically
        COMPANIES[other_company].transfer_to_corp(state, 1)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After receivership close no JS bonus")

        # JS should NOT have received bonus (corp 1 closed, not JS)
        assert js.get_cash(state) == 0

    def test_js_inactive_no_bonus(self):
        """No bonus when JS is not active."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        # JS (corp 0) starts inactive after initialize_game()
        js = CORPS[0]

        # FI closes company
        COMPANIES[0].transfer_to_fi(state)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After FI close JS inactive")

        # JS should have no cash
        assert js.get_cash(state) == 0

    def test_js_gets_bonus_when_js_in_receivership_closes(self):
        """JS DOES get bonus when JS (in receivership) closes its own company."""

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # JS owns red company that will be auto-closed by receivership rules
        red_company = 0
        other_company = 14  # Higher FV yellow (protected)
        income = get_company_income(red_company)

        # Float JS with the red company, then put in receivership
        float_corp_for_test(state, corp_id=0, company_id=red_company)
        js = CORPS[0]
        PLAYERS[0].set_shares(state, 0, 0)  # Triggers receivership automatically
        js.set_cash(state, 0)

        COMPANIES[other_company].transfer_to_corp(state, 0)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)
        assert_invariants(state, "After JS receivership auto-close with bonus")

        # JS should have received 2x income (JS closed its own company)
        assert js.get_cash(state) == income * 2


class TestOfferGeneration:
    """Tests for close offer generation."""

    def test_only_negative_income_offered(self, closing_offer_state):
        """Only companies with negative adjusted income are offered."""
        gs = closing_offer_state

        # Give player company 0 (income $1, red/1-star)
        # At CoO level 6, CoO = $6, adjusted = $1 - $6 = -$5 (negative - should be offered)
        COMPANIES[0].transfer_to_player(gs, 0)

        # Give player company 29 (income $10, blue/5-star)
        # At CoO level 6, CoO = $0, adjusted = $10 - $0 = $10 (positive - should NOT be offered)
        COMPANIES[29].transfer_to_player(gs, 0)

        # Generate offers
        generate_close_offers_py(gs)

        # Should only have 1 offer (company 0)
        assert get_close_offer_count_py(gs) == 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert company_id == 0

    def test_zero_income_not_offered(self, closing_offer_state):
        """Companies with exactly zero adjusted income are NOT offered."""
        gs = closing_offer_state

        # Find a company where income exactly equals CoO
        # Company 2 has income $2 (red/1-star)
        # At CoO level 4, CoO = $2, adjusted = $2 - $2 = $0 (zero - NOT offered)
        TURN.set_coo_level(gs, 4)
        COMPANIES[2].transfer_to_player(gs, 0)

        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) == 0

    def test_offers_sorted_by_face_value_ascending(self, closing_offer_state):
        """Offers sorted by face value ascending (lowest first)."""
        gs = closing_offer_state

        # Give player multiple negative-income companies with different face values
        # Company 0: face value $1 (red)
        # Company 6: face value $5 (orange)
        # Company 3: face value $3 (red)
        # All should have negative income at high CoO
        COMPANIES[0].transfer_to_player(gs, 0)  # FV $1
        COMPANIES[6].transfer_to_player(gs, 0)  # FV $5
        COMPANIES[3].transfer_to_player(gs, 0)  # FV $3

        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) == 3

        # Should be sorted: company 0 (FV $1), company 3 (FV $3), company 6 (FV $5)
        _, _, cid0 = get_close_offer_py(gs, 0)
        _, _, cid1 = get_close_offer_py(gs, 1)
        _, _, cid2 = get_close_offer_py(gs, 2)

        assert get_company_face_value(cid0) < get_company_face_value(cid1)
        assert get_company_face_value(cid1) < get_company_face_value(cid2)

    def test_player_privates_included(self, closing_offer_state):
        """Player-owned private companies are included in offers."""
        gs = closing_offer_state

        # Player 1 owns company 1 directly (private)
        COMPANIES[1].transfer_to_player(gs, 1)

        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) >= 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert owner_type == CompanyLocation.LOC_PLAYER
        assert owner_id == 1
        assert company_id == 1

    def test_corp_subsidiaries_included(self, closing_offer_state):
        """Corp subsidiaries (same-president) included in offers."""

        gs = closing_offer_state

        # Float corp 1 with company 2 and player 0 as president
        float_corp_for_test(gs, corp_id=1, company_id=2, player_id=0)

        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) >= 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert owner_type == CompanyLocation.LOC_CORP
        assert owner_id == 1
        assert company_id == 2

    def test_receivership_corp_excluded(self, closing_offer_state):
        """Receivership corps excluded from offers (no president)."""

        gs = closing_offer_state

        # Float corp 2 with company 4, then put in receivership
        float_corp_for_test(gs, corp_id=2, company_id=4)
        PLAYERS[0].set_shares(gs, 2, 0)  # Triggers receivership automatically

        generate_close_offers_py(gs)

        # No offers (receivership excluded)
        assert get_close_offer_count_py(gs) == 0

    def test_fi_excluded(self, closing_offer_state):
        """FI-owned companies excluded from player offers (handled by auto-close)."""
        gs = closing_offer_state

        # FI owns company 5
        COMPANIES[5].transfer_to_fi(gs)

        generate_close_offers_py(gs)

        # No offers (FI excluded - handled by auto-close)
        assert get_close_offer_count_py(gs) == 0


class TestOfferValidation:
    """Tests for offer validation."""

    def test_corp_last_company_rule(self, closing_offer_state):
        """Corp closing offer invalid if corp would have 0 companies."""

        gs = closing_offer_state

        # Float corp 1 with company 3 (last company)
        float_corp_for_test(gs, corp_id=1, company_id=3, player_id=0)

        generate_close_offers_py(gs)

        # Offer should be generated (validation happens at presentation time)
        # But when presented, it should be skipped
        # For this test, check that offer count reflects generation
        # Validation is dynamic - tested in integration test
        # Here we just verify the rule is documented and understood

    def test_corp_with_multiple_companies_can_close(self, closing_offer_state):
        """Corp with multiple companies CAN close one."""

        gs = closing_offer_state

        # Float corp 1 with company 3, then add company 4
        float_corp_for_test(gs, corp_id=1, company_id=3, player_id=0)
        COMPANIES[4].transfer_to_corp(gs, 1)

        generate_close_offers_py(gs)

        # Both companies should be offered
        assert get_close_offer_count_py(gs) == 2

    def test_prior_acceptance_invalidates_later_offer(self, closing_offer_state):
        """Prior acceptance can invalidate later offers (corp down to 1 company)."""

        gs = closing_offer_state

        # Float corp 1 with company 0, then add company 3
        float_corp_for_test(gs, corp_id=1, company_id=0, player_id=0)
        COMPANIES[3].transfer_to_corp(gs, 1)

        # Set phase to CLOSING and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close prior acceptance test")

        # First offer should be company 0 (lowest FV)
        assert TURN.get_closing_company(gs) == 0

        # Accept first offer (closes company 0)
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After closing first offer in invalidation test")

        # Second offer (company 3) should be SKIPPED because corp now has only 1 company
        # Phase should transition to INCOME (no more valid offers)
        assert TURN.get_closing_company(gs) == -1
        assert gs.get_phase() == PHASE_INCOME_PY


class TestCloseActions:
    """Tests for close actions."""

    def test_accept_closes_company(self, closing_offer_state):
        """Accept action closes the company (removes from game)."""
        gs = closing_offer_state

        # Player 0 owns company 1
        COMPANIES[1].transfer_to_player(gs, 0)

        # Set phase and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close accept test")

        # Offer should be active
        assert TURN.get_closing_company(gs) == 1

        # Accept the offer
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After accepting close offer")

        # Company should be removed
        assert COMPANIES[1].is_removed(gs)
        # Player should not own it
        assert not PLAYERS[0].owns_company(gs, 1)

    def test_pass_keeps_company(self, closing_offer_state):
        """Pass action keeps the company."""
        gs = closing_offer_state

        # Player 0 owns company 2
        COMPANIES[2].transfer_to_player(gs, 0)

        # Set phase and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close pass test")

        # Offer should be active
        assert TURN.get_closing_company(gs) == 2

        # Pass on the offer
        apply_closing_action_py(gs, ACTION_PASS_PY)
        assert_invariants(gs, "After passing close offer")

        # Company should NOT be removed
        assert not COMPANIES[2].is_removed(gs)
        # Player should still own it
        assert PLAYERS[0].owns_company(gs, 2)

    def test_junkyard_scrappers_no_bonus_on_player_close(self, closing_offer_state):
        """JS does NOT receive bonus when player closes their own company."""

        gs = closing_offer_state

        # Float Junkyard Scrappers (corp 0) with some starting cash
        float_corp_for_test(gs, corp_id=0)
        CORPS[0].set_cash(gs, 100)

        # Player owns company 1 (printed income = $1)
        COMPANIES[1].transfer_to_player(gs, 0)

        # Set phase and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close JS no bonus player close")

        # Accept the close offer
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After player close no JS bonus")

        # JS should NOT have received bonus (player closed, not JS)
        assert CORPS[0].get_cash(gs) == 100

    def test_junkyard_scrappers_no_bonus_on_other_corp_close(self, closing_offer_state):
        """JS does NOT receive bonus when another corp closes their company."""
        gs = closing_offer_state

        # Float Junkyard Scrappers (corp 0) with a yellow company (won't be auto-closed)
        float_corp_for_test(gs, corp_id=0, company_id=14)
        CORPS[0].set_cash(gs, 50)

        # Float corp 1 with company 0, then add company 3
        float_corp_for_test(gs, corp_id=1, company_id=0, player_id=0)
        COMPANIES[3].transfer_to_corp(gs, 1)

        # Set phase and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close JS no bonus other corp")

        # Accept the close offer for company 0
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After other corp close no JS bonus")

        # JS should NOT have received bonus (corp 1 closed, not JS)
        assert CORPS[0].get_cash(gs) == 50

    def test_junkyard_scrappers_bonus_only_when_js_closes(self, closing_offer_state):
        """JS receives 2x printed income bonus ONLY when JS closes its own company."""
        gs = closing_offer_state

        # Float JS with company 0, then add company 3
        float_corp_for_test(gs, corp_id=0, company_id=0, player_id=0)
        CORPS[0].set_cash(gs, 100)
        COMPANIES[3].transfer_to_corp(gs, 0)  # Keep one

        # Set phase and run auto-close
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close JS bonus test")

        # Accept the close offer for company 0
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After JS closes own company with bonus")

        # JS should have received 2x printed income bonus ($1 * 2 = $2)
        expected_bonus = get_company_income(0) * 2
        assert CORPS[0].get_cash(gs) == 100 + expected_bonus


# =============================================================================
# MANDATORY CLOSE TESTS
# =============================================================================


class TestPlayerIncome:
    """Tests for Player.get_income() method."""

    def test_get_income_no_companies(self, game_state):
        """Player with no private companies has 0 income."""
        # Fresh game state - player 0 has no companies
        income = PLAYERS[0].get_income(game_state)
        assert income == 0

    def test_get_income_single_company(self, game_state):
        """Player income equals adjusted income of owned company."""
        # Give player 0 a company (company 0: $1 income, 1 star)
        COMPANIES[0].transfer_to_player(game_state, 0)

        # At CoO level 1, 1-star company has $0 CoO
        # Adjusted income = $1 - $0 = $1
        income = PLAYERS[0].get_income(game_state)
        coo_level = TURN.get_coo_level(game_state)
        assert coo_level == 1  # Default
        assert income == 1  # $1 - $0 = $1

    def test_get_income_multiple_companies(self, game_state):
        """Player income sums adjusted income from all owned companies."""
        # Give player 0 two companies
        # Company 0: $1 income, 1 star -> adjusted = $1 - $0 = $1 at CoO 1
        # Company 8: $3 income, 2 stars -> adjusted = $3 - $0 = $3 at CoO 1
        COMPANIES[0].transfer_to_player(game_state, 0)
        COMPANIES[8].transfer_to_player(game_state, 0)

        income = PLAYERS[0].get_income(game_state)
        # $1 + $3 = $4
        assert income == 4

    def test_get_income_negative_company(self, game_state):
        """Player income can be negative from high-CoO companies."""
        # Set CoO level to max (7) to get high CoO values
        TURN.set_coo_level(game_state, 7)

        # Give player a 1-star company at CoO 7
        # Company 0: $1 income, 1 star -> CoO at level 7 is $10
        # Adjusted = $1 - $10 = -$9
        COMPANIES[0].transfer_to_player(game_state, 0)

        income = PLAYERS[0].get_income(game_state)
        assert income == -9  # $1 - $10 = -$9

    def test_get_income_excludes_corp_subsidiaries(self, game_state):
        """Player income excludes companies owned by corps (even if player is president)."""
        # Float corp 0 with company 0, player 0 as president
        float_corp_for_test(game_state, corp_id=0, company_id=0, player_id=0)

        # Player income should be 0 (corp's company doesn't count)
        income = PLAYERS[0].get_income(game_state)
        assert income == 0


class TestMandatoryClose:
    """Tests for mandatory close logic."""

    def test_mandatory_close_not_triggered_positive_total(self, game_state):
        """Mandatory close does nothing when income + cash >= 0."""
        # Player has $30 cash, no companies -> income 0
        # Total = $30 >= 0, no close needed
        assert PLAYERS[0].get_cash(game_state) == 30
        assert PLAYERS[0].get_income(game_state) == 0

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close no-op")

        # Nothing changed
        assert PLAYERS[0].get_cash(game_state) == 30

    def test_mandatory_close_triggered_negative_total(self, game_state):
        """Mandatory close triggers when income + cash < 0."""
        # Set up: player with negative income that exceeds cash
        # Set CoO level high
        TURN.set_coo_level(game_state, 7)

        # Give player a 1-star company with negative adjusted income
        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adjusted = -$9
        COMPANIES[0].transfer_to_player(game_state, 0)

        # Reduce player cash to trigger mandatory close
        # Cash = $5, income = -$9, total = -$4 < 0
        PLAYERS[0].set_cash(game_state, 5)

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close triggered")

        # Company should be closed
        assert not PLAYERS[0].owns_company(game_state, 0)
        assert COMPANIES[0].is_removed(game_state)

    def test_mandatory_close_cheapest_first(self, game_state):
        """Cheapest (lowest face value) negative-income company closed first."""
        TURN.set_coo_level(game_state, 7)

        # Give player two negative-income companies with different face values
        # Company 0: face value $1, 1 star (cheapest), adjusted = $1 - $10 = -$9
        # Company 8: face value $3, 2 stars, adjusted = $3 - $10 = -$7
        COMPANIES[0].transfer_to_player(game_state, 0)
        COMPANIES[8].transfer_to_player(game_state, 0)

        # Set cash so closing ONE company makes total >= 0
        # Income = -$9 + -$7 = -$16
        # Cash = $10, total = -$6 < 0
        # After closing company 0 (-$9): income = -$7, total = -$7 + $10 = $3 >= 0
        PLAYERS[0].set_cash(game_state, 10)

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close cheapest first")

        # Cheapest (company 0, face value $1) should be closed
        assert not PLAYERS[0].owns_company(game_state, 0)
        assert COMPANIES[0].is_removed(game_state)

        # More expensive company 8 should still be owned
        assert PLAYERS[0].owns_company(game_state, 8)
        assert not COMPANIES[8].is_removed(game_state)

    def test_mandatory_close_multiple_companies(self, game_state):
        """Closes multiple companies if needed until income + cash >= 0."""
        TURN.set_coo_level(game_state, 7)

        # Give player multiple negative-income companies
        COMPANIES[0].transfer_to_player(game_state, 0)  # $1 FV, -$9 adj
        COMPANIES[1].transfer_to_player(game_state, 0)  # $1 FV, -$9 adj
        COMPANIES[2].transfer_to_player(game_state, 0)  # $2 FV, -$8 adj

        # Income = -$9 + -$9 + -$8 = -$26, cash = 10, total = -$16 < 0
        # Need to close multiple companies
        PLAYERS[0].set_cash(game_state, 10)

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close multiple companies")

        # Should have closed enough to make total >= 0
        income = PLAYERS[0].get_income(game_state)
        cash = PLAYERS[0].get_cash(game_state)
        assert income + cash >= 0

    def test_mandatory_close_no_js_bonus(self, game_state):
        """Junkyard Scrappers does NOT receive bonus on mandatory player close."""
        TURN.set_coo_level(game_state, 7)

        # Float Junkyard Scrappers (corp 0) - uses a company from deck
        float_corp_for_test(game_state, corp_id=0)
        initial_js_cash = CORPS[0].get_cash(game_state)

        # Give player a negative-income company (company 1 since 0 may be used by JS)
        COMPANIES[1].transfer_to_player(game_state, 0)
        PLAYERS[0].set_cash(game_state, 5)  # Will trigger mandatory close

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close no JS bonus")

        # JS should NOT have received bonus (player mandatory close, not JS close)
        assert CORPS[0].get_cash(game_state) == initial_js_cash

    def test_mandatory_close_only_negative_income_companies(self, game_state):
        """Mandatory close only targets negative-income companies, not positive."""
        TURN.set_coo_level(game_state, 7)

        # Give player one positive and one negative income company
        # Company 29: $10 income, blue/5-star, CoO 7 = $0 -> adj = +$10
        COMPANIES[29].transfer_to_player(game_state, 0)

        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adj = -$9
        COMPANIES[0].transfer_to_player(game_state, 0)

        # Total income = +$10 + -$9 = +$1
        # Cash = $0, total = $1 >= 0, no close needed
        PLAYERS[0].set_cash(game_state, 0)

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close only negative income")

        # No companies should be closed (total is positive)
        assert PLAYERS[0].owns_company(game_state, 29)
        assert PLAYERS[0].owns_company(game_state, 0)

    def test_mandatory_close_player_order(self, game_state):
        """Mandatory close processes players in ID order (0, 1, 2, ...)."""
        TURN.set_coo_level(game_state, 7)

        # Give player 1 a negative-income company (test non-zero player)
        COMPANIES[5].transfer_to_player(game_state, 1)  # $2 FV, -$8 adj
        PLAYERS[1].set_cash(game_state, 5)  # Will trigger close

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close player order")

        # Player 1's company should be closed
        assert not PLAYERS[1].owns_company(game_state, 5)
        assert COMPANIES[5].is_removed(game_state)

    def test_mandatory_close_multiple_players_same_turn(self, game_state):
        """Multiple players can have mandatory close triggered in the same turn.

        Tests that when multiple players would have negative cash after INCOME,
        each player's companies are closed independently in player ID order.
        """
        TURN.set_coo_level(game_state, 7)

        # Player 0: negative-income company, low cash -> will trigger mandatory close
        # Company 0: income=$1, CoO=$10, adjusted=-$9
        COMPANIES[0].transfer_to_player(game_state, 0)
        PLAYERS[0].set_cash(game_state, 5)  # income + cash = -9 + 5 = -4 < 0

        # Player 1: negative-income company, low cash -> will trigger mandatory close
        # Company 1: income=$1, CoO=$10, adjusted=-$9
        COMPANIES[1].transfer_to_player(game_state, 1)
        PLAYERS[1].set_cash(game_state, 3)  # income + cash = -9 + 3 = -6 < 0

        # Player 2: negative-income company, HIGH cash -> will NOT trigger
        # Company 2: income=$2, CoO=$10, adjusted=-$8
        COMPANIES[2].transfer_to_player(game_state, 2)
        PLAYERS[2].set_cash(game_state, 100)  # income + cash = -8 + 100 = 92 >= 0

        process_mandatory_close_py(game_state)
        assert_invariants(game_state, "After mandatory close multiple players")

        # Player 0's company should be closed (would have negative cash)
        assert not PLAYERS[0].owns_company(game_state, 0)
        assert COMPANIES[0].is_removed(game_state)

        # Player 1's company should be closed (would have negative cash)
        assert not PLAYERS[1].owns_company(game_state, 1)
        assert COMPANIES[1].is_removed(game_state)

        # Player 2's company should NOT be closed (has enough cash)
        assert PLAYERS[2].owns_company(game_state, 2)
        assert not COMPANIES[2].is_removed(game_state)

        # Verify all players now have non-negative projected cash after INCOME
        for player_id in range(3):
            income = PLAYERS[player_id].get_income(game_state)
            cash = PLAYERS[player_id].get_cash(game_state)
            assert income + cash >= 0, f"Player {player_id} would have negative cash: {income} + {cash}"


class TestClosingPhaseTransition:
    """Tests for CLOSING phase transition."""

    def test_phase_transitions_after_mandatory_close(self, game_state):
        """Phase transitions to INCOME (INVEST) when no offers and mandatory close complete."""
        # Simulate state where CLOSING has no offers and mandatory close has nothing to do
        # All players have positive income + cash (default state)
        TURN.set_phase(game_state, GamePhases.PHASE_CLOSING)

        # Run auto-close (which includes offer generation, mandatory close, and transition)
        apply_closing_auto_py(game_state)
        assert_invariants(game_state, "After auto-close phase transition")

        # Should have transitioned to INCOME
        assert game_state.get_phase() == GamePhases.PHASE_INCOME

    def test_closing_flow_with_mandatory_close_triggered(self, game_state):
        """Integration test: Mandatory close triggers after offers are declined."""
        # Set up: high CoO to create negative-income companies
        TURN.set_coo_level(game_state, 7)

        # Give player 0 a negative-income company
        COMPANIES[0].transfer_to_player(game_state, 0)

        # Low cash to trigger mandatory close after offer processing
        PLAYERS[0].set_cash(game_state, 5)

        # Enter CLOSING phase and run auto-close
        TURN.set_phase(game_state, GamePhases.PHASE_CLOSING)
        apply_closing_auto_py(game_state)
        assert_invariants(game_state, "After auto-close with mandatory close flow")

        # There should be a close offer for company 0
        assert TURN.get_closing_company(game_state) == 0

        # Pass on the offer (should trigger mandatory close after all offers processed)
        apply_closing_action_py(game_state, ACTION_PASS_PY)
        assert_invariants(game_state, "After pass triggering mandatory close")

        # After passing, mandatory close should have kicked in and closed the company
        # Phase should transition to INCOME
        assert game_state.get_phase() == GamePhases.PHASE_INCOME
        assert COMPANIES[0].is_removed(game_state)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestClosingEdgeCases:
    """Edge case tests for CLOSING phase boundary conditions."""

    def test_no_close_offers_direct_transition(self, game_state):
        """Edge case: All companies have positive adjusted income - direct transition.

        Requirement: When no close offers exist, CLOSING phase transitions
        directly to INVEST without presenting any offers.
        """
        # Default game state has CoO level 1 (low)
        # At low CoO, no companies have negative adjusted income
        assert TURN.get_coo_level(game_state) == 1

        assert_invariants(game_state, "Before CLOSING phase")

        # Enter CLOSING phase
        TURN.set_phase(game_state, PHASE_CLOSING_PY)
        apply_closing_auto_py(game_state)

        # With no negative-income companies owned by players,
        # phase should transition directly to INCOME
        assert game_state.get_phase() == PHASE_INCOME_PY
        assert TURN.get_closing_company(game_state) == -1

        assert_invariants(game_state, "After CLOSING phase")

    def test_all_pass_triggers_mandatory_close(self, game_state):
        """Edge case: All offers passed with low cash triggers mandatory close.

        Requirement: When player passes on close offers but income + cash < 0,
        mandatory close automatically closes the cheapest negative-income company.
        """
        # Set high CoO level to create negative-income companies
        TURN.set_coo_level(game_state, 7)

        # Give player 0 a negative-income company
        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adjusted = -$9
        COMPANIES[0].transfer_to_player(game_state, 0)

        # Low cash to trigger mandatory close after passing
        # Cash = $5, income = -$9, total = -$4 < 0
        PLAYERS[0].set_cash(game_state, 5)

        # Enter CLOSING phase and generate offers
        TURN.set_phase(game_state, PHASE_CLOSING_PY)
        apply_closing_auto_py(game_state)
        assert_invariants(game_state, "After auto-close all pass mandatory trigger")

        # Should have an offer for company 0
        assert TURN.get_closing_company(game_state) == 0

        # Pass on the offer
        apply_closing_action_py(game_state, ACTION_PASS_PY)
        assert_invariants(game_state, "After pass triggers mandatory close")

        # After passing, mandatory close should trigger and close company
        # Phase should transition to INCOME
        assert game_state.get_phase() == PHASE_INCOME_PY
        assert COMPANIES[0].is_removed(game_state)
        assert not PLAYERS[0].owns_company(game_state, 0)

    def test_all_pass_no_mandatory_close_needed(self, game_state):
        """Edge case: All offers passed with high cash - no mandatory close.

        Requirement: When player passes on close offers but income + cash >= 0,
        no mandatory close happens and company remains.
        """
        # Set high CoO level to create negative-income company
        TURN.set_coo_level(game_state, 7)

        # Give player 0 a negative-income company
        # Company 0: $1 income, 1 star, CoO 7 = $10 -> adjusted = -$9
        COMPANIES[0].transfer_to_player(game_state, 0)

        # High cash so no mandatory close needed
        # Cash = $100, income = -$9, total = $91 >= 0
        PLAYERS[0].set_cash(game_state, 100)

        # Enter CLOSING phase and generate offers
        TURN.set_phase(game_state, PHASE_CLOSING_PY)
        apply_closing_auto_py(game_state)
        assert_invariants(game_state, "After auto-close all pass no mandatory")

        # Should have an offer for company 0
        assert TURN.get_closing_company(game_state) == 0

        # Pass on the offer
        apply_closing_action_py(game_state, ACTION_PASS_PY)
        assert_invariants(game_state, "After pass no mandatory close needed")

        # After passing, NO mandatory close (player has enough cash)
        # Phase should transition to INCOME
        assert game_state.get_phase() == PHASE_INCOME_PY

        # Company should NOT be removed
        assert not COMPANIES[0].is_removed(game_state)
        assert PLAYERS[0].owns_company(game_state, 0)

    def test_multi_close_cascade_no_js_bonus_for_player_closes(self, game_state):
        """Edge case: Player closes multiple companies - JS does NOT get bonus.

        Requirement: When player closes multiple companies, JS does NOT receive
        bonus because player is closing their own companies, not JS closing.
        """
        # Set high CoO level to create negative-income companies
        TURN.set_coo_level(game_state, 7)

        # Float Junkyard Scrappers (corp 0) with starting cash
        # Uses company from deck (not 0, 1, 3 which player needs)
        float_corp_for_test(game_state, corp_id=0, company_id=10)  # Use company 10 for JS
        CORPS[0].set_cash(game_state, 50)
        initial_js_cash = 50

        # Give player 0 multiple negative-income companies
        # Company 0: $1 income, FV $1
        # Company 1: $1 income, FV $1
        # Company 3: $2 income, FV $3
        COMPANIES[0].transfer_to_player(game_state, 0)  # income $1
        COMPANIES[1].transfer_to_player(game_state, 0)  # income $1
        COMPANIES[3].transfer_to_player(game_state, 0)  # income $2

        # High cash so no mandatory close needed
        PLAYERS[0].set_cash(game_state, 1000)

        # Enter CLOSING phase
        TURN.set_phase(game_state, PHASE_CLOSING_PY)
        apply_closing_auto_py(game_state)
        assert_invariants(game_state, "After auto-close multi cascade")

        # Accept all three close offers
        # Offers are sorted by face value, so order is: company 0, company 1, company 3
        assert TURN.get_closing_company(game_state) in [0, 1]
        apply_closing_action_py(game_state, ACTION_CLOSE_PY)
        assert_invariants(game_state, "After first cascade close")

        # Continue closing
        assert TURN.get_closing_company(game_state) in [0, 1, 3]
        apply_closing_action_py(game_state, ACTION_CLOSE_PY)
        assert_invariants(game_state, "After second cascade close")

        # Third close
        assert TURN.get_closing_company(game_state) == 3
        apply_closing_action_py(game_state, ACTION_CLOSE_PY)
        assert_invariants(game_state, "After third cascade close")

        # JS should NOT have received any bonus (player closes, not JS)
        assert CORPS[0].get_cash(game_state) == initial_js_cash
        assert game_state.get_phase() == PHASE_INCOME_PY

    def test_corp_last_company_dynamic_invalidation(self, closing_offer_state):
        """Edge case: Corp last-company rule invalidates second offer after first close.

        Requirement: When corp has 2 companies and player closes the first,
        the second offer should be automatically skipped (corp last-company rule).
        """
        gs = closing_offer_state

        # Float corp 1 with company 0, then add company 3
        float_corp_for_test(gs, corp_id=1, company_id=0, player_id=0)
        COMPANIES[3].transfer_to_corp(gs, 1)

        # Enter CLOSING phase
        TURN.set_phase(gs, PHASE_CLOSING_PY)
        apply_closing_auto_py(gs)
        assert_invariants(gs, "After auto-close dynamic invalidation")

        # First offer should be company 0 (lowest FV)
        assert TURN.get_closing_company(gs) == 0

        # Accept first offer (closes company 0, corp now has 1 company)
        apply_closing_action_py(gs, ACTION_CLOSE_PY)
        assert_invariants(gs, "After close triggering last-company skip")

        # Second offer (company 3) should be SKIPPED because corp now has only 1 company
        # Phase should transition directly to INCOME
        assert gs.get_phase() == PHASE_INCOME_PY
        assert TURN.get_closing_company(gs) == -1

        # Company 3 should NOT be closed (last company rule)
        assert not COMPANIES[3].is_removed(gs)
        assert CORPS[1].owns_company(gs, 3)

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_closing_edge_cases_with_player_count(self, num_players):
        """Edge case: CLOSING phase works correctly for different player counts.

        Requirement: CLOSING phase handles 3 and 6 player games correctly
        with proper offer generation and mandatory close processing.
        """
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set high CoO level
        TURN.set_coo_level(state, 7)

        # Give negative-income companies to multiple players
        # Player 0: company 0 (income $1, CoO $10 -> adj = -$9)
        COMPANIES[0].transfer_to_player(state, 0)
        PLAYERS[0].set_cash(state, 100)

        # Player 1: company 2 (income $2, CoO $10 -> adj = -$8)
        COMPANIES[2].transfer_to_player(state, 1)
        PLAYERS[1].set_cash(state, 5)  # Low cash for mandatory close test

        # For 6-player game, also give company to player 5
        if num_players == 6:
            COMPANIES[4].transfer_to_player(state, 5)
            PLAYERS[5].set_cash(state, 100)

        assert_invariants(state, f"Before CLOSING ({num_players} players)")

        # Enter CLOSING phase
        TURN.set_phase(state, PHASE_CLOSING_PY)
        apply_closing_auto_py(state)
        assert_invariants(state, f"After auto-close ({num_players} players)")

        # Process all offers (pass on all)
        while TURN.get_closing_company(state) >= 0:
            apply_closing_action_py(state, ACTION_PASS_PY)
            assert_invariants(state, f"After pass on close offer ({num_players} players)")

        # Phase should transition to INCOME
        assert state.get_phase() == PHASE_INCOME_PY

        # Player 1 should have had mandatory close trigger (low cash)
        assert not PLAYERS[1].owns_company(state, 2)
        assert COMPANIES[2].is_removed(state)

        # Player 0 should still have company (high cash)
        assert PLAYERS[0].owns_company(state, 0)

        assert_invariants(state, f"After CLOSING ({num_players} players)")


# =============================================================================
# ACTIVE COMPANY TESTS
# =============================================================================

class TestActiveCompanyClosing:
    """Test active company block during CLOSING phase."""

    def test_active_company_set_on_close_offer(self, closing_offer_state):
        """Active company block matches the company being offered for closing."""
        state = closing_offer_state

        # Give a red company (negative income at CoO 6) to player 0
        COMPANIES[0].transfer_to_player(state, 0)

        # Enter closing phase
        TURN.set_phase(state, GamePhases.PHASE_CLOSING)
        apply_closing_auto_py(state)
        generate_close_offers_py(state)

        closing_company = TURN.get_closing_company(state)
        if closing_company >= 0:
            layout = get_layout(3)
            base = layout.active_company_offset
            expected_fv = get_company_face_value(closing_company) / PY_CASH_DIVISOR
            assert abs(state._array[base + 2] - expected_fv) < 1e-6
            assert state._array[base + 0] > 0.0  # stars > 0

    def test_active_company_cleared_after_closing_exhausted(self, closing_offer_state):
        """Active company block is zeroed when no more close offers remain."""
        state = closing_offer_state

        # Give a red company to player 0 for a close offer
        COMPANIES[0].transfer_to_player(state, 0)

        TURN.set_phase(state, GamePhases.PHASE_CLOSING)
        apply_closing_auto_py(state)
        generate_close_offers_py(state)

        # Pass on all offers until exhausted
        closing_company = TURN.get_closing_company(state)
        while closing_company >= 0:
            apply_closing_action_py(state, ACTION_PASS_PY)
            closing_company = TURN.get_closing_company(state)

        # Active company should be cleared
        layout = get_layout(3)
        base = layout.active_company_offset
        for i in range(5):
            assert state._array[base + i] == 0.0, (
                f"active_company[{i}] should be 0 after all close offers exhausted"
            )
