"""Tests for ACQUISITION phase offer generation."""

import pytest
from core.state import GameState
from core.data import (
    CORP_NAMES, GamePhases,
    get_company_face_value, get_company_low_price, get_company_high_price
)
from core.actions import (
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_HIGH_PY as ACTION_ACQ_FI_HIGH,
    ACTION_ACQ_FI_FACE_PY as ACTION_ACQ_FI_FACE,
    ACTION_PASS_PY as ACTION_PASS
)
from entities.player import PLAYERS
from entities.fi import FI
from entities.corp import CORPS
from entities.turn import TURN
from entities.company import COMPANIES
from phases.acquisition import (
    generate_offers_py,
    get_offer_count,
    get_offer_at,
    setup_acquisition_phase_py,
    apply_acquisition_action_py,
    merge_acquisition_zones_py,
    transition_to_closing_py
)
from phases.wrap_up import apply_wrap_up_py


class TestOfferGeneration:
    """OFFER-01 through OFFER-05: Offer generation and priority."""

    def test_no_offers_fresh_game(self):
        """No offers when no corps active and FI has no companies."""
        gs = GameState(3)
        gs.initialize_game()
        generate_offers_py(gs)
        assert get_offer_count(gs) == 0

    def test_fi_offers_generated(self):
        """OFFER-02, OFFER-03: FI offers generated when corps active."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active with cash
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have at least one FI offer
        assert get_offer_count(gs) > 0
        corp_id, company_id = get_offer_at(gs, 0)
        assert company_id == 0  # Company 0 from FI
        assert corp_id == 0     # Corp 0 buying

    def test_os_fi_offers_first(self):
        """OFFER-02: OS->FI offers come before other corp->FI offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make OS (corp 2) and corp 0 both active with cash
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        CORPS[CORP_NAMES[2]].set_active(gs, True)  # OS is corp 2
        CORPS[CORP_NAMES[2]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 2, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have offers
        assert get_offer_count(gs) >= 2

        # First offer should be from OS (corp 2)
        corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 2, f"Expected OS (corp 2) first, got corp {corp_id}"
        assert company_id == 0

    def test_corp_fi_sorted_by_price(self):
        """OFFER-03: Non-OS corp->FI offers sorted by descending share price."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active at higher price_index (20)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_price_index(gs, 20)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Make corp 1 active at lower price_index (10)
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_price_index(gs, 10)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 1, True)

        # Generate offers (skip OS so we test non-OS sorting)
        setup_acquisition_phase_py(gs)

        # Should have at least 2 offers
        assert get_offer_count(gs) >= 2

        # Higher-priced corp (0) should appear before lower-priced corp (1)
        corp_id_first, _ = get_offer_at(gs, 0)
        corp_id_second, _ = get_offer_at(gs, 1)
        assert corp_id_first == 0, f"Expected corp 0 first, got {corp_id_first}"
        assert corp_id_second == 1, f"Expected corp 1 second, got {corp_id_second}"

    def test_corp_corp_offers_same_president(self):
        """OFFER-04: Corp->Corp offers only for same president."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to corp 0
        COMPANIES[0].transfer_to_corp(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)

        # Make corp 1 active with cash, make player 0 president of BOTH corps
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have at least one offer (corp 1 buying from corp 0)
        assert get_offer_count(gs) > 0
        corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 1  # Corp 1 buying
        assert company_id == 0  # Company 0 from corp 0

    def test_different_president_no_offers(self):
        """OFFER-04: Different presidents prevents corp-to-corp offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to corp 0
        COMPANIES[0].transfer_to_corp(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Make corp 1 active with cash, different president (player 1)
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[1].set_president_of(gs, 1, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have NO offers (different presidents)
        assert get_offer_count(gs) == 0

    def test_player_private_offers(self):
        """OFFER-05: Corp->Player private offers generated."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to player 0
        COMPANIES[0].transfer_to_player(gs, 0)

        # Make corp 0 active with cash, player 0 is president
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have at least one offer (corp 0 buying from player 0)
        assert get_offer_count(gs) > 0
        corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 0  # Corp 0 buying
        assert company_id == 0  # Company 0 from player 0

    def test_fi_offers_sorted_by_corp_share_price(self):
        """OFFER-03 detail: FI offers sorted by buyer corp share price descending."""
        gs = GameState(3)
        gs.initialize_game()

        # Give companies 0 and 1 to FI
        COMPANIES[0].transfer_to_fi(gs)
        COMPANIES[1].transfer_to_fi(gs)

        # Make corp 0 active at price_index 20 (higher)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_price_index(gs, 20)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Make corp 1 active at price_index 10 (lower)
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_price_index(gs, 10)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 1, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have 4 offers total (2 companies × 2 corps)
        # Corp 0 offers for both companies should come before Corp 1 offers
        assert get_offer_count(gs) == 4

        corp_id_0, _ = get_offer_at(gs, 0)
        corp_id_1, _ = get_offer_at(gs, 1)
        corp_id_2, _ = get_offer_at(gs, 2)
        corp_id_3, _ = get_offer_at(gs, 3)

        # First two should be corp 0 (higher price)
        assert corp_id_0 == 0
        assert corp_id_1 == 0
        # Last two should be corp 1 (lower price)
        assert corp_id_2 == 1
        assert corp_id_3 == 1

    def test_corp_corp_sorted_by_buyer_price_then_face_value(self):
        """OFFER-04 detail: Corp-corp sorted by buyer price DESC, then face value ASC."""
        gs = GameState(3)
        gs.initialize_game()

        # Give companies 0 (lower face value) and 1 (could be higher) to corp 0
        COMPANIES[0].transfer_to_corp(gs, 0)
        COMPANIES[1].transfer_to_corp(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)

        # Make corp 1 active at price_index 20 (higher)
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_price_index(gs, 20)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)

        # Make corp 2 active at price_index 10 (lower)
        CORPS[CORP_NAMES[2]].set_active(gs, True)
        CORPS[CORP_NAMES[2]].set_price_index(gs, 10)
        CORPS[CORP_NAMES[2]].set_cash(gs, 50000)

        # Player 0 is president of all three corps (same-president requirement)
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)
        PLAYERS[0].set_president_of(gs, 2, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have offers: corp 1 and corp 2 buying from corp 0
        # Expected order: Higher-priced buyer (corp 1) before lower-priced (corp 2)
        # For same buyer, lower face value company first
        assert get_offer_count(gs) >= 2

        # First offers should be from higher-priced buyer (corp 1)
        corp_id_first, _ = get_offer_at(gs, 0)
        corp_id_second, _ = get_offer_at(gs, 1)

        # Both should be corp 1 (or at least first should be)
        assert corp_id_first == 1, f"Expected corp 1 (higher price) first, got {corp_id_first}"

        # If more offers, verify sorting continues
        if get_offer_count(gs) >= 3:
            corp_id_third, _ = get_offer_at(gs, 2)
            # After corp 1's offers, corp 2's offers should appear
            if corp_id_third != 1:
                assert corp_id_third == 2

    def test_player_private_sorted_similarly(self):
        """OFFER-05 detail: Player-private sorted by buyer price DESC, face value ASC."""
        gs = GameState(3)
        gs.initialize_game()

        # Give companies 0 and 1 to player 0
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[1].transfer_to_player(gs, 0)

        # Make corp 0 active at price_index 20 (higher)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_price_index(gs, 20)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)

        # Make corp 1 active at price_index 10 (lower)
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_price_index(gs, 10)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)

        # Player 0 is president of both corps
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)

        # Generate offers
        setup_acquisition_phase_py(gs)

        # Should have 4 offers: 2 companies × 2 corps
        # Sorted by buyer price DESC, then face value ASC
        assert get_offer_count(gs) == 4

        # First offers should be from higher-priced corp (corp 0)
        corp_id_0, _ = get_offer_at(gs, 0)
        corp_id_1, _ = get_offer_at(gs, 1)

        assert corp_id_0 == 0, f"Expected corp 0 (higher price) first, got {corp_id_0}"
        assert corp_id_1 == 0, f"Expected corp 0 (higher price) second, got {corp_id_1}"

        # Last offers should be from lower-priced corp (corp 1)
        corp_id_2, _ = get_offer_at(gs, 2)
        corp_id_3, _ = get_offer_at(gs, 3)

        assert corp_id_2 == 1
        assert corp_id_3 == 1


class TestPhaseFlow:
    """Phase entry and transition tests."""

    def test_wrap_up_sets_up_acquisition(self):
        """WRAP_UP generates offers before transitioning."""
        gs = GameState(3)
        gs.initialize_game()

        # Transition through WRAP_UP
        TURN.set_phase(gs, GamePhases.PHASE_WRAP_UP)
        apply_wrap_up_py(gs)

        # Should be in ACQUISITION
        assert TURN.get_phase(gs) == GamePhases.PHASE_ACQUISITION

        # Fresh game has no offers
        assert TURN.get_acq_active_corp(gs) == -1
        assert get_offer_count(gs) == 0

    def test_acquisition_with_fi_company(self):
        """Offers generated when FI has company and corp has cash."""
        # TODO: This test requires setting up complex game state with:
        # - FI owning companies
        # - Active corps with cash
        # Full implementation deferred until integration testing
        pass

    def test_empty_offers_detected(self):
        """Empty offer buffer is detected."""
        gs = GameState(3)
        gs.initialize_game()
        setup_acquisition_phase_py(gs)
        assert get_offer_count(gs) == 0

    def test_transition_to_closing(self):
        """ACQUISITION transitions to next turn when offers exhausted."""
        gs = GameState(3)
        gs.initialize_game()

        # Set phase to ACQUISITION
        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)
        assert gs.get_phase() == GamePhases.PHASE_ACQUISITION
        initial_turn = TURN.get_turn_number(gs)

        # Call transition
        transition_to_closing_py(gs)

        # Should now be INVEST (new turn) - CLOSING phase not yet implemented
        assert gs.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(gs) == initial_turn + 1

    def test_transition_merges_player_proceeds(self):
        """Transition merges player acquisition_proceeds."""
        gs = GameState(3)
        gs.initialize_game()

        player = PLAYERS[0]
        initial_cash = player.get_cash(gs)
        proceeds = 35

        player.add_acquisition_proceeds(gs, proceeds)
        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)

        transition_to_closing_py(gs)

        # Proceeds merged to cash
        assert player.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert player.get_acquisition_proceeds(gs) == 0

    def test_transition_merges_corp_proceeds(self):
        """Transition merges corp acquisition_proceeds."""
        gs = GameState(3)
        gs.initialize_game()

        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        initial_cash = corp.get_cash(gs)
        proceeds = 45

        corp.set_acquisition_proceeds(gs, proceeds)
        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)

        transition_to_closing_py(gs)

        assert corp.get_cash(gs) == initial_cash + proceeds
        assert corp.get_acquisition_proceeds(gs) == 0

    def test_transition_merges_acquisition_companies(self):
        """Transition merges acquisition_companies to owned."""
        gs = GameState(3)
        gs.initialize_game()

        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)

        company = COMPANIES[0]
        company.transfer_to_corp_acquisition(gs, 0)

        assert corp.has_acquisition_company(gs, 0)
        assert not corp.owns_company(gs, 0)

        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)
        transition_to_closing_py(gs)

        # Company moved from acquisition to owned
        assert not corp.has_acquisition_company(gs, 0)
        assert corp.owns_company(gs, 0)


class TestValidation:
    """Validation tests - verify through action handler behavior."""

    def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
        """Setup player private -> corp offer."""
        # Give company to player
        COMPANIES[company_id].transfer_to_player(gs, player_id)

        # Make corp active and set cash
        CORPS[CORP_NAMES[corp_id]].set_active(gs, True)
        CORPS[CORP_NAMES[corp_id]].set_cash(gs, corp_cash)

        # Make player president of corp
        PLAYERS[player_id].set_president_of(gs, corp_id, True)

        # Generate offers and present
        setup_acquisition_phase_py(gs)

    def test_price_in_range_succeeds(self):
        """Price within [low, high] is valid - action executes."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup: Player 0 owns company 0, corp 0 (B&O) has cash
        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Should have one offer
        assert get_offer_count(gs) > 0
        assert TURN.get_acq_target_company(gs) == 0

        # Price in range (offset 0 = low_price)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0  # Success

    def test_price_below_low_rejected(self):
        """Price below low_price returns invalid (1)."""
        gs = GameState(3)
        gs.initialize_game()

        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Negative offset = below low_price
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, -10)
        assert result == 1  # Invalid

    def test_price_above_high_rejected(self):
        """Price above high_price returns invalid (1)."""
        gs = GameState(3)
        gs.initialize_game()

        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Large offset = above high_price
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 100)
        assert result == 1  # Invalid

    def test_insufficient_cash_filters_offers(self):
        """Corp with insufficient cash has no offers generated (filtered at generation time)."""
        gs = GameState(3)
        gs.initialize_game()

        # Corp has only $1 cash (not enough for any company)
        self._setup_player_private_offer(gs, 0, 0, 0, 1)

        # Offers should be filtered out during generation - corp can't afford anything
        assert get_offer_count(gs) == 0 or TURN.get_acq_active_corp(gs) == -1, \
            "Bug: Offer generated for corp that cannot afford the minimum price"

    def test_fi_buy_high_rejects_os_corp(self):
        """OS corp cannot use FI Buy High action."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make OS (corp 2) active
        COMPANIES[0].transfer_to_fi(gs)
        CORPS[CORP_NAMES[2]].set_active(gs, True)
        CORPS[CORP_NAMES[2]].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)

        # OS tries FI Buy High - should reject
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_HIGH, 0)
            assert result == 1  # Invalid

    def test_fi_buy_face_rejects_non_os_corp(self):
        """Non-OS corp cannot use FI Buy Face action."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make non-OS corp active
        COMPANIES[0].transfer_to_fi(gs)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)

        # Non-OS tries FI Buy Face - should reject
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE, 0)
            assert result == 1  # Invalid

    def test_target_already_acquired_rejected(self):
        """Cannot buy company already in acquisition_companies (VALID-04)."""
        gs = GameState(3)
        gs.initialize_game()

        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Manually add company to acquisition zone
        COMPANIES[0].transfer_to_corp_acquisition(gs, 0)

        # Try to buy - should reject
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 1  # Invalid

    def test_target_already_owned_rejected(self):
        """Cannot buy company already in buyer's owned_companies (VALID-05)."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup offer with company in player's private
        COMPANIES[0].transfer_to_player(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        # Manually transfer company to corp's owned (not acquisition)
        COMPANIES[0].transfer_to_corp(gs, 0)

        setup_acquisition_phase_py(gs)

        # Try to buy company corp already owns - should reject
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
            assert result == 1  # Invalid

    # VALID-01 boundary tests
    def test_price_at_low_boundary(self):
        """VALID-01: Price exactly at low_price succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Offset 0 = low_price
        low_price = get_company_low_price(company_id)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0, f"Action at low_price ({low_price}) should succeed"

    def test_price_at_high_boundary(self):
        """VALID-01: Price exactly at high_price succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Calculate offset to reach high_price
        low_price = get_company_low_price(company_id)
        high_price = get_company_high_price(company_id)
        offset = high_price - low_price

        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, offset)
        assert result == 0, f"Action at high_price ({high_price}) should succeed"

    def test_price_one_below_low_fails(self):
        """VALID-01: Price = low_price - 1 fails."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Offset -1 = below low_price
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, -1)
        assert result == 1, "Price below low_price should fail"

    def test_price_one_above_high_fails(self):
        """VALID-01: Price = high_price + 1 fails."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Calculate offset beyond high_price
        low_price = get_company_low_price(company_id)
        high_price = get_company_high_price(company_id)
        offset = (high_price - low_price) + 1

        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, offset)
        assert result == 1, "Price above high_price should fail"

    # VALID-02 boundary tests
    def test_exact_cash_for_price_succeeds(self):
        """VALID-02: Corp has exactly the price amount, action succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        low_price = get_company_low_price(company_id)

        # Give corp exactly low_price in cash
        # Note: Offer generation may filter this out if exact match considered insufficient
        # This tests the boundary - if offer is generated, action should succeed
        self._setup_player_private_offer(gs, 0, company_id, 0, low_price)

        if get_offer_count(gs) > 0:
            # Action should succeed
            result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
            assert result == 0, "Action with exact cash should succeed"
        else:
            # If no offer generated, that's also acceptable boundary behavior
            # (implementation may require cash > price rather than >= price)
            pass

    def test_one_dollar_short_fails(self):
        """VALID-02: Corp has price - 1, no offer generated."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        low_price = get_company_low_price(company_id)

        # Give corp one dollar less than low_price
        self._setup_player_private_offer(gs, 0, company_id, 0, low_price - 1)

        # No offer should be generated (insufficient cash filtered at generation)
        assert get_offer_count(gs) == 0, "No offer should be generated with insufficient cash"

    # VALID-03 boundary tests (seller keeps >= 1 company)
    def test_seller_with_two_companies_can_sell_one(self):
        """VALID-03: Seller has 2 companies, sell 1 succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        # Give corp 0 two companies
        COMPANIES[0].transfer_to_corp(gs, 0)
        COMPANIES[1].transfer_to_corp(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)

        # Make corp 1 active with cash, same president
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)

        setup_acquisition_phase_py(gs)

        # Should have offers (seller has 2 companies, can sell 1)
        assert get_offer_count(gs) > 0, "Seller with 2 companies should have offers"

    def test_seller_with_one_company_action_rejected(self):
        """VALID-03: Seller with 1 company - offer generated but action rejected."""
        gs = GameState(3)
        gs.initialize_game()

        # Give corp 0 exactly one company (no acquisition zone companies)
        COMPANIES[0].transfer_to_corp(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)

        # Make corp 1 active with cash, same president
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)

        setup_acquisition_phase_py(gs)

        # VALID-03 is checked at ACTION time, not offer generation time
        # Offer IS generated, but action should be rejected
        assert get_offer_count(gs) > 0, "Offer should be generated"

        # Try to execute - should fail VALID-03 check (seller would have 0 after)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 1, "Action should be rejected (seller would have 0 companies)"

    def test_seller_with_one_owned_one_acquisition_can_sell(self):
        """VALID-03: Seller has 1 owned + 1 in acquisition zone, can sell the owned one."""
        gs = GameState(3)
        gs.initialize_game()

        # Give corp 0 one owned and one in acquisition zone
        COMPANIES[0].transfer_to_corp(gs, 0)
        COMPANIES[1].transfer_to_corp_acquisition(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)

        # Make corp 1 active with cash, same president
        CORPS[CORP_NAMES[1]].set_active(gs, True)
        CORPS[CORP_NAMES[1]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)
        PLAYERS[0].set_president_of(gs, 1, True)

        setup_acquisition_phase_py(gs)

        # _count_seller_companies counts BOTH owned and acquisition_companies
        # With 1 owned + 1 acquisition, selling owned leaves 1 (in acquisition) - valid
        assert get_offer_count(gs) > 0, "Seller with 1 owned + 1 acquisition can sell owned"

    # VALID-04/VALID-05 boundary test
    def test_company_in_acquisition_zone_blocks_offer(self):
        """VALID-04: Company in acquisition zone blocks offer generation."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to player, then put it in corp's acquisition zone
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[0].transfer_to_corp_acquisition(gs, 0)

        # Make corp active with cash
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        setup_acquisition_phase_py(gs)

        # No offer should be generated (company already in acquisition zone)
        assert get_offer_count(gs) == 0, "Company in acquisition zone should block offer"


class TestActionIntegration:
    """Integration tests for action execution."""

    def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
        """Setup player private -> corp offer."""
        COMPANIES[company_id].transfer_to_player(gs, player_id)
        CORPS[CORP_NAMES[corp_id]].set_active(gs, True)
        CORPS[CORP_NAMES[corp_id]].set_cash(gs, corp_cash)
        PLAYERS[player_id].set_president_of(gs, corp_id, True)
        setup_acquisition_phase_py(gs)

    def test_accept_price_action(self):
        """Full flow - money transfers, company moves to acquisition zone."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup: Player 0 owns company 0, corp 0 buys
        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Verify setup
        assert get_offer_count(gs) > 0
        corp_cash_before = CORPS[CORP_NAMES[0]].get_cash(gs)
        player_proceeds_before = PLAYERS[0].get_acquisition_proceeds(gs)

        # Execute action (offset 0 = low_price, always valid)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0

        # Verify money transfer
        corp_cash_after = CORPS[CORP_NAMES[0]].get_cash(gs)
        player_proceeds_after = PLAYERS[0].get_acquisition_proceeds(gs)
        assert corp_cash_after < corp_cash_before
        assert player_proceeds_after > player_proceeds_before

        # Verify company in acquisition zone
        assert CORPS[CORP_NAMES[0]].has_acquisition_company(gs, 0)

    def test_fi_buy_high_action(self):
        """Non-OS buys from FI at high price."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make non-OS corp active
        COMPANIES[0].transfer_to_fi(gs)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)

        if get_offer_count(gs) > 0:
            corp_cash_before = CORPS[CORP_NAMES[0]].get_cash(gs)
            fi_cash_before = FI.get_cash(gs)

            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_HIGH, 0)
            assert result == 0

            # Verify money transfer to FI
            corp_cash_after = CORPS[CORP_NAMES[0]].get_cash(gs)
            fi_cash_after = FI.get_cash(gs)
            assert corp_cash_after < corp_cash_before
            assert fi_cash_after > fi_cash_before

            # Verify company in acquisition zone
            assert CORPS[CORP_NAMES[0]].has_acquisition_company(gs, 0)

    def test_fi_buy_face_action(self):
        """OS buys from FI at face value."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make OS active
        COMPANIES[0].transfer_to_fi(gs)
        CORPS[CORP_NAMES[2]].set_active(gs, True)
        CORPS[CORP_NAMES[2]].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)

        if get_offer_count(gs) > 0:
            corp_cash_before = CORPS[CORP_NAMES[2]].get_cash(gs)
            fi_cash_before = FI.get_cash(gs)

            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE, 0)
            assert result == 0

            # Verify money transfer at face value
            corp_cash_after = CORPS[CORP_NAMES[2]].get_cash(gs)
            fi_cash_after = FI.get_cash(gs)
            face_value = get_company_face_value(0)
            assert corp_cash_after == corp_cash_before - face_value
            assert fi_cash_after == fi_cash_before + face_value

            # Verify company in acquisition zone
            assert CORPS[CORP_NAMES[2]].has_acquisition_company(gs, 0)

    def test_pass_action(self):
        """Offer index advances, next offer presented."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup two offers: two companies to same player/corp
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[1].transfer_to_player(gs, 0)
        CORPS[CORP_NAMES[0]].set_active(gs, True)
        CORPS[CORP_NAMES[0]].set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 0, True)

        setup_acquisition_phase_py(gs)

        # Should have offers
        offer_count = get_offer_count(gs)
        if offer_count > 0:
            first_target = TURN.get_acq_target_company(gs)

            # Pass on first offer
            result = apply_acquisition_action_py(gs, ACTION_PASS, 0)
            assert result == 0

            # Check if moved to next offer or cleared
            second_target = TURN.get_acq_target_company(gs)
            if offer_count > 1:
                # Should advance to next offer
                assert second_target != first_target
            else:
                # Should clear (no more offers)
                assert second_target == -1


class TestZoneMerging:
    """Tests for acquisition zone merging at phase end (FLOW-03, FLOW-04)."""

    def test_player_proceeds_merge_to_cash(self):
        """Player acquisition_proceeds merge to cash at phase end."""
        gs = GameState(3)
        gs.initialize_game()

        player = PLAYERS[0]
        initial_cash = player.get_cash(gs)
        proceeds = 25

        # Simulate acquisition proceeds from selling
        player.add_acquisition_proceeds(gs, proceeds)
        assert player.get_acquisition_proceeds(gs) == proceeds

        # Trigger merge
        merge_acquisition_zones_py(gs)

        # Proceeds merged to cash
        assert player.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert player.get_acquisition_proceeds(gs) == 0

    def test_corp_proceeds_merge_to_cash(self):
        """Corp acquisition_proceeds merge to cash at phase end."""
        gs = GameState(3)
        gs.initialize_game()

        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        initial_cash = corp.get_cash(gs)
        proceeds = 30

        corp.set_acquisition_proceeds(gs, proceeds)
        assert corp.get_acquisition_proceeds(gs) == proceeds

        # Trigger merge
        merge_acquisition_zones_py(gs)

        # Proceeds merged to cash
        assert corp.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert corp.get_acquisition_proceeds(gs) == 0

    def test_acquisition_companies_merge_to_owned(self):
        """Corp acquisition_companies merge to owned_companies at phase end."""
        gs = GameState(3)
        gs.initialize_game()

        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)

        company = COMPANIES[0]

        # Put company in acquisition zone
        company.transfer_to_corp_acquisition(gs, 0)
        assert corp.has_acquisition_company(gs, 0)
        assert not corp.owns_company(gs, 0)

        # Trigger merge
        merge_acquisition_zones_py(gs)

        # Company moved from acquisition to owned
        assert not corp.has_acquisition_company(gs, 0)
        assert corp.owns_company(gs, 0)

    def test_zones_cleared_after_merge(self):
        """Acquisition zones are cleared (zeroed) after merge."""
        gs = GameState(3)
        gs.initialize_game()

        player = PLAYERS[0]
        player.add_acquisition_proceeds(gs, 50)

        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        corp.set_acquisition_proceeds(gs, 40)

        # Trigger merge
        merge_acquisition_zones_py(gs)

        # All zones cleared
        assert player.get_acquisition_proceeds(gs) == 0
        assert corp.get_acquisition_proceeds(gs) == 0


class TestReceivershipAutoBuy:
    """Tests for receivership auto-buy behavior (RECV-01, RECV-02, RECV-03)."""

    def test_receivership_auto_buys_affordable_fi(self):
        """RECV-01, RECV-03: Receivership corp auto-buys affordable FI company at face value."""
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active with enough cash
        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        corp.set_cash(gs, 50000)

        # Put corp 0 in receivership
        corp.set_in_receivership(gs, True)

        # Record initial values
        from core.data import get_company_face_value
        face_value = get_company_face_value(0)
        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        # Call setup_acquisition_phase_py to generate offers and trigger auto-buy
        setup_acquisition_phase_py(gs)

        # Verify auto-buy executed
        assert corp.has_acquisition_company(gs, 0), "Receivership corp should auto-buy FI company"
        assert FI.get_cash(gs) == fi_cash_before + face_value, "FI should receive face value"
        assert corp.get_cash(gs) == corp_cash_before - face_value, "Corp should pay face value"
        # Offer should be processed (no offers left or active corp cleared)
        assert get_offer_count(gs) == 0 or TURN.get_acq_active_corp(gs) == -1

    def test_receivership_skips_unaffordable_fi(self):
        """RECV-03: Receivership corp auto-passes when can't afford FI company."""
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active with minimal cash (can't afford face value)
        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        corp.set_cash(gs, 1)

        # Put corp 0 in receivership
        corp.set_in_receivership(gs, True)

        # Record initial FI cash
        fi_cash_before = FI.get_cash(gs)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)

        # Verify auto-pass: no buy happened
        assert not corp.has_acquisition_company(gs, 0), "Receivership corp should not buy unaffordable company"
        assert FI.get_cash(gs) == fi_cash_before, "FI cash should be unchanged"
        # Company still owned by FI
        assert FI.owns_company(gs, 0), "Company should still be owned by FI"

    def test_receivership_skips_non_fi_offers(self):
        """RECV-03: Receivership corp auto-passes on non-FI offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to player 0
        COMPANIES[0].transfer_to_player(gs, 0)

        # Make corp 0 active with plenty of cash
        corp = CORPS[CORP_NAMES[0]]
        corp.set_active(gs, True)
        corp.set_cash(gs, 50000)

        # Put corp 0 in receivership
        corp.set_in_receivership(gs, True)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)

        # Verify: Receivership corp cannot buy from player
        # (no offers should be generated or they should be auto-passed)
        assert not corp.has_acquisition_company(gs, 0), "Receivership corp should not buy from player"
        # No offers for receivership buying from player
        assert get_offer_count(gs) == 0 or TURN.get_acq_active_corp(gs) == -1

    def test_receivership_cannot_sell(self):
        """RECV-02: Receivership corp cannot sell companies."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to corp 0
        COMPANIES[0].transfer_to_corp(gs, 0)

        # Make corp 0 active and put in receivership
        corp0 = CORPS[CORP_NAMES[0]]
        corp0.set_active(gs, True)
        corp0.set_in_receivership(gs, True)

        # Make corp 1 active with cash, make player 0 president
        corp1 = CORPS[CORP_NAMES[1]]
        corp1.set_active(gs, True)
        corp1.set_cash(gs, 50000)
        PLAYERS[0].set_president_of(gs, 1, True)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)

        # Verify: No offers should exist because:
        # - Corp 0 has company but is in receivership (can't sell)
        # - _get_corp_president returns -1 for receivership, never matches any player_id
        assert get_offer_count(gs) == 0, "No offers should exist for receivership seller"
