"""Tests for CLOSING phase auto-close logic (CLO-01 through CLO-04)."""
import pytest
from core.state import GameState
from core.data import (
    GamePhases, GameConstants,
    get_company_income, get_company_stars, get_company_face_value,
    get_cost_of_ownership
)
from entities.turn import TURN
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from phases.closing import apply_closing_auto_py
from core.actions import ACTION_CLOSE_PY, ACTION_PASS_PY

# Import status codes from conftest
from tests.phases.conftest import STATUS_OK

# Phase constants for tests
PHASE_CLOSING_PY = GamePhases.PHASE_CLOSING
PHASE_INVEST_PY = GamePhases.PHASE_INVEST


class TestFIAutoClose:
    """CLO-01: FI closes companies where income - CoO < 0."""

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

        # Company should NOT be removed (zero income, not negative)
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

        # Both should be closed
        assert not FI.owns_company(state, 0)
        assert not FI.owns_company(state, 1)


class TestReceivershipAutoClose:
    """CLO-02, CLO-03: Receivership corps close red >= $4, orange >= $7."""

    def _setup_receivership_corp(self, state, corp_id, company_ids):
        """Helper to set up receivership corp with companies."""
        corp = CORPS[corp_id]
        corp.set_active(state, True)
        corp.set_in_receivership(state, True)
        for cid in company_ids:
            COMPANIES[cid].transfer_to_corp(state, corp_id)

    def test_receivership_closes_red_at_coo_4(self):
        """CLO-02: Receivership closes red company when CoO >= $4."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Red companies (stars=1): CoO at level 5 = $4
        # Company 0 and 1 are red
        red_company = 0
        other_company = 14  # Higher face value (yellow, stars=3, protected)

        self._setup_receivership_corp(state, 1, [red_company, other_company])
        TURN.set_coo_level(state, 5)  # Red CoO = $4

        apply_closing_auto_py(state)

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

        self._setup_receivership_corp(state, 1, [red_company, other_company])
        TURN.set_coo_level(state, 4)  # Red CoO = $2 < $4

        apply_closing_auto_py(state)

        # Red company should remain
        assert not COMPANIES[red_company].is_removed(state)

    def test_receivership_closes_orange_at_coo_7(self):
        """CLO-03: Receivership closes orange company when CoO >= $7."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Orange companies (stars=2): CoO at level 6 = $7
        # Companies 6-13 are orange
        orange_company = 6
        other_company = 14  # Higher face value (yellow, protected)

        self._setup_receivership_corp(state, 1, [orange_company, other_company])
        TURN.set_coo_level(state, 6)  # Orange CoO = $7

        apply_closing_auto_py(state)

        # Orange company should be closed
        assert COMPANIES[orange_company].is_removed(state)
        assert not COMPANIES[other_company].is_removed(state)

    def test_receivership_never_closes_yellow_green_blue(self):
        """Receivership never auto-closes yellow/green/blue companies."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Yellow (stars=3), green (stars=4), blue (stars=5)
        yellow_company = 14  # Stars=3
        green_company = 22   # Stars=4
        blue_company = 29    # Stars=5

        self._setup_receivership_corp(state, 1, [yellow_company, green_company, blue_company])
        TURN.set_coo_level(state, 7)  # Max CoO

        apply_closing_auto_py(state)

        # None should be closed (yellow/green/blue exempt)
        assert not COMPANIES[yellow_company].is_removed(state)
        assert not COMPANIES[green_company].is_removed(state)
        assert not COMPANIES[blue_company].is_removed(state)


class TestHighestFaceValueProtection:
    """CLO-04: Receivership always keeps highest face value company."""

    def _setup_receivership_corp(self, state, corp_id, company_ids):
        """Helper to set up receivership corp with companies."""
        corp = CORPS[corp_id]
        corp.set_active(state, True)
        corp.set_in_receivership(state, True)
        for cid in company_ids:
            COMPANIES[cid].transfer_to_corp(state, corp_id)

    def test_highest_face_value_protected_even_if_red(self):
        """Highest FV company is protected even if it would otherwise close."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Give corp only red companies - highest FV must survive
        red_companies = [0, 1, 2, 3]  # All red, FV ascending
        highest_fv = 3  # Company 3 has highest FV among reds

        self._setup_receivership_corp(state, 1, red_companies)
        TURN.set_coo_level(state, 7)  # All reds would normally close

        apply_closing_auto_py(state)

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
        self._setup_receivership_corp(state, 1, [red_company])
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)

        # Single company must survive (it's both the only AND highest FV)
        assert not COMPANIES[red_company].is_removed(state)


class TestVintageMachineryReduction:
    """VM (corp_id 6) reduces CoO by up to $10."""

    def _setup_receivership_corp(self, state, corp_id, company_ids):
        """Helper to set up receivership corp with companies."""
        corp = CORPS[corp_id]
        corp.set_active(state, True)
        corp.set_in_receivership(state, True)
        for cid in company_ids:
            COMPANIES[cid].transfer_to_corp(state, corp_id)

    def test_vm_reduction_prevents_close(self):
        """VM's CoO reduction can prevent company from closing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Red company at CoO 7 ($10) would normally close (>= $4)
        # But VM reduces by 10, so effective CoO = $0 < $4
        red_company = 0
        other_company = 14  # Higher FV yellow (protected regardless)

        # Use VM (corp_id 6)
        self._setup_receivership_corp(state, 6, [red_company, other_company])
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)

        # Red company should NOT be closed (VM reduction makes CoO $0)
        assert not COMPANIES[red_company].is_removed(state)


class TestJunkyardScrappersBonus:
    """JS (corp_id 0) receives 2x printed income when closing."""

    def test_js_receives_bonus_on_fi_close(self):
        """JS gets 2x income bonus when FI closes company."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Activate JS
        js = CORPS[0]
        js.set_active(state, True)
        js.set_cash(state, 0)

        # FI closes a red company
        red_company = 0
        income = get_company_income(red_company)
        COMPANIES[red_company].transfer_to_fi(state)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)

        # JS should have received 2x income
        assert js.get_cash(state) == income * 2

    def test_js_receives_bonus_on_receivership_close(self):
        """JS gets 2x income bonus when receivership corp closes company."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Activate JS
        js = CORPS[0]
        js.set_active(state, True)
        js.set_cash(state, 0)

        # Non-JS corp in receivership closes red
        red_company = 0
        other_company = 14  # Higher FV yellow
        income = get_company_income(red_company)

        corp = CORPS[1]
        corp.set_active(state, True)
        corp.set_in_receivership(state, True)
        COMPANIES[red_company].transfer_to_corp(state, 1)
        COMPANIES[other_company].transfer_to_corp(state, 1)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)

        # JS should have received 2x income
        assert js.get_cash(state) == income * 2

    def test_js_inactive_no_bonus(self):
        """No bonus when JS is not active."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # JS not active
        js = CORPS[0]
        js.set_active(state, False)

        # FI closes company
        COMPANIES[0].transfer_to_fi(state)
        TURN.set_coo_level(state, 7)

        apply_closing_auto_py(state)

        # JS should have no cash
        assert js.get_cash(state) == 0


class TestOfferGeneration:
    """Tests for close offer generation (CLO-05 through CLO-08)."""

    def test_only_negative_income_offered(self, closing_offer_state):
        """CLO-05: Only companies with negative adjusted income are offered."""
        gs = closing_offer_state

        # Give player company 0 (income $1, red/1-star)
        # At CoO level 6, CoO = $6, adjusted = $1 - $6 = -$5 (negative - should be offered)
        PLAYERS[0].set_owns_company(gs, 0, True)

        # Give player company 29 (income $10, blue/5-star)
        # At CoO level 6, CoO = $0, adjusted = $10 - $0 = $10 (positive - should NOT be offered)
        PLAYERS[0].set_owns_company(gs, 29, True)

        # Generate offers
        from phases.closing import generate_close_offers_py, get_close_offer_count_py, get_close_offer_py
        generate_close_offers_py(gs)

        # Should only have 1 offer (company 0)
        assert get_close_offer_count_py(gs) == 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert company_id == 0

    def test_zero_income_not_offered(self, closing_offer_state):
        """CLO-05: Companies with exactly zero adjusted income are NOT offered."""
        gs = closing_offer_state

        # Find a company where income exactly equals CoO
        # Company 2 has income $2 (red/1-star)
        # At CoO level 4, CoO = $2, adjusted = $2 - $2 = $0 (zero - NOT offered)
        TURN.set_coo_level(gs, 4)
        PLAYERS[0].set_owns_company(gs, 2, True)

        from phases.closing import generate_close_offers_py, get_close_offer_count_py
        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) == 0

    def test_offers_sorted_by_face_value_ascending(self, closing_offer_state):
        """CLO-06: Offers sorted by face value ascending (lowest first)."""
        gs = closing_offer_state

        # Give player multiple negative-income companies with different face values
        # Company 0: face value $1 (red)
        # Company 6: face value $5 (orange)
        # Company 3: face value $3 (red)
        # All should have negative income at high CoO
        PLAYERS[0].set_owns_company(gs, 0, True)  # FV $1
        PLAYERS[0].set_owns_company(gs, 6, True)  # FV $5
        PLAYERS[0].set_owns_company(gs, 3, True)  # FV $3

        from phases.closing import generate_close_offers_py, get_close_offer_count_py, get_close_offer_py
        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) == 3

        # Should be sorted: company 0 (FV $1), company 3 (FV $3), company 6 (FV $5)
        _, _, cid0 = get_close_offer_py(gs, 0)
        _, _, cid1 = get_close_offer_py(gs, 1)
        _, _, cid2 = get_close_offer_py(gs, 2)

        assert get_company_face_value(cid0) < get_company_face_value(cid1)
        assert get_company_face_value(cid1) < get_company_face_value(cid2)

    def test_player_privates_included(self, closing_offer_state):
        """CLO-07: Player-owned private companies are included in offers."""
        gs = closing_offer_state

        # Player 1 owns company 1 directly (private)
        PLAYERS[1].set_owns_company(gs, 1, True)

        from phases.closing import generate_close_offers_py, get_close_offer_count_py, get_close_offer_py
        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) >= 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert owner_type == 0  # OWNER_PLAYER
        assert owner_id == 1
        assert company_id == 1

    def test_corp_subsidiaries_included(self, closing_offer_state):
        """CLO-08: Corp subsidiaries (same-president) included in offers."""
        gs = closing_offer_state

        # Activate corp 1 and make player 0 president
        CORPS[1].set_active(gs, True)
        CORPS[1].set_in_receivership(gs, False)
        PLAYERS[0].set_president_of(gs, 1, True)
        PLAYERS[0].set_shares(gs, 1, 3)  # 3 shares to be president

        # Corp owns company 2
        CORPS[1].set_owns_company(gs, 2, True)

        from phases.closing import generate_close_offers_py, get_close_offer_count_py, get_close_offer_py
        generate_close_offers_py(gs)

        assert get_close_offer_count_py(gs) >= 1
        owner_type, owner_id, company_id = get_close_offer_py(gs, 0)
        assert owner_type == 1  # OWNER_CORP
        assert owner_id == 1
        assert company_id == 2

    def test_receivership_corp_excluded(self, closing_offer_state):
        """Receivership corps excluded from offers (no president)."""
        gs = closing_offer_state

        # Activate corp 2 in receivership
        CORPS[2].set_active(gs, True)
        CORPS[2].set_in_receivership(gs, True)

        # Corp owns company 4
        CORPS[2].set_owns_company(gs, 4, True)

        from phases.closing import generate_close_offers_py, get_close_offer_count_py
        generate_close_offers_py(gs)

        # No offers (receivership excluded)
        assert get_close_offer_count_py(gs) == 0

    def test_fi_excluded(self, closing_offer_state):
        """FI-owned companies excluded from player offers (handled by auto-close)."""
        gs = closing_offer_state

        # FI owns company 5
        FI.set_owns_company(gs, 5, True)

        from phases.closing import generate_close_offers_py, get_close_offer_count_py
        generate_close_offers_py(gs)

        # No offers (FI excluded - handled by auto-close)
        assert get_close_offer_count_py(gs) == 0
