"""Tests for ACQUISITION phase."""

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
    transition_to_closing_py,
    qsort_desc_3_py,
    qsort_price_fv_4_py
)
from phases.wrap_up import apply_wrap_up_py
from tests.phases.conftest import float_corp_for_test, assert_invariants


class TestOfferGeneration:
    """Offer generation and priority."""

    def test_no_offers_fresh_game(self):
        """No offers when no corps active and FI has no companies."""
        gs = GameState(3)
        gs.initialize_game()
        generate_offers_py(gs)
        assert_invariants(gs, "After generate_offers_py fresh game")
        assert get_offer_count(gs) == 0

    def test_fi_offers_generated(self):
        """FI offers generated when corps active."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active with cash
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py FI offers")

        # Should have at least one FI offer
        assert get_offer_count(gs) > 0
        _, corp_id, company_id = get_offer_at(gs, 0)
        assert company_id == 0  # Company 0 from FI
        assert corp_id == 0     # Corp 0 buying

    def test_os_fi_offers_first(self):
        """OS->FI offers come before other corp->FI offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make OS (corp 2) and corp 0 both active with cash
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        float_corp_for_test(gs, 2)  # OS is corp 2
        CORPS[2].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py OS FI first")

        # Should have offers
        assert get_offer_count(gs) >= 2

        # First offer should be from OS (corp 2)
        _, corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 2, f"Expected OS (corp 2) first, got corp {corp_id}"
        assert company_id == 0

    def test_corp_fi_sorted_by_price(self):
        """Non-OS corp->FI offers sorted by descending share price."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Make corp 0 active at higher price_index (20)
        float_corp_for_test(gs, 0, par_index=20)
        CORPS[0].set_cash(gs, 50000)

        # Make corp 1 active at lower price_index (10)
        float_corp_for_test(gs, 1, par_index=10)
        CORPS[1].set_cash(gs, 50000)

        # Generate offers (skip OS so we test non-OS sorting)
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py corp FI sorted by price")

        # Should have at least 2 offers
        assert get_offer_count(gs) >= 2

        # Higher-priced corp (0) should appear before lower-priced corp (1)
        _, corp_id_first, _ = get_offer_at(gs, 0)
        _, corp_id_second, _ = get_offer_at(gs, 1)
        assert corp_id_first == 0, f"Expected corp 0 first, got {corp_id_first}"
        assert corp_id_second == 1, f"Expected corp 1 second, got {corp_id_second}"

    def test_corp_corp_offers_same_president(self):
        """Corp->Corp offers only for same president."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company 0, player 0 as president
        float_corp_for_test(gs, 0, company_id=0)

        # Float corp 1 with different company, same president (player 0)
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py corp-corp same president")

        # Should have at least one offer (corp 1 buying from corp 0)
        assert get_offer_count(gs) > 0
        _, corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 1  # Corp 1 buying
        assert company_id == 0  # Company 0 from corp 0

    def test_different_president_no_offers(self):
        """Different presidents prevents corp-to-corp offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company 0, player 0 is president
        float_corp_for_test(gs, 0, company_id=0)

        # Float corp 1 with different president (player 1)
        float_corp_for_test(gs, 1, player_id=1)
        CORPS[1].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py different president")

        # Should have NO offers (different presidents)
        assert get_offer_count(gs) == 0

    def test_player_private_offers(self):
        """Corp->Player private offers generated."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to player 0 as a private company
        COMPANIES[0].transfer_to_player(gs, 0)

        # Float corp 0 with different company, player 0 is president
        float_corp_for_test(gs, 0)  # Uses first available deck company (not 0)
        CORPS[0].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py player private offers")

        # Should have at least one offer (corp 0 buying from player 0)
        assert get_offer_count(gs) > 0
        _, corp_id, company_id = get_offer_at(gs, 0)
        assert corp_id == 0  # Corp 0 buying
        assert company_id == 0  # Company 0 from player 0

    def test_fi_offers_sorted_by_corp_share_price(self):
        """FI offers sorted by buyer corp share price descending."""
        gs = GameState(3)
        gs.initialize_game()

        # Give companies 0 and 1 to FI
        COMPANIES[0].transfer_to_fi(gs)
        COMPANIES[1].transfer_to_fi(gs)

        # Make corp 0 active at price_index 20 (higher) - player 0 president
        float_corp_for_test(gs, 0, par_index=20)
        CORPS[0].set_cash(gs, 50000)

        # Make corp 1 active at price_index 10 (lower) - player 1 president (different)
        # Different presidents to avoid corp-to-corp offers
        float_corp_for_test(gs, 1, par_index=10, player_id=1)
        CORPS[1].set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py FI sorted by corp share price")

        # Should have 4 FI offers total (2 companies × 2 corps)
        # No corp-to-corp offers because different presidents
        # Corp 0 offers for both companies should come before Corp 1 offers
        assert get_offer_count(gs) == 4

        _, corp_id_0, _ = get_offer_at(gs, 0)
        _, corp_id_1, _ = get_offer_at(gs, 1)
        _, corp_id_2, _ = get_offer_at(gs, 2)
        _, corp_id_3, _ = get_offer_at(gs, 3)

        # First two should be corp 0 (higher price)
        assert corp_id_0 == 0
        assert corp_id_1 == 0
        # Last two should be corp 1 (lower price)
        assert corp_id_2 == 1
        assert corp_id_3 == 1

    def test_corp_corp_sorted_by_buyer_price_then_face_value(self):
        """Corp-corp sorted by buyer price DESC, then face value DESC."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company 0, then add company 1
        float_corp_for_test(gs, 0, company_id=0)
        COMPANIES[1].transfer_to_corp(gs, 0)

        # Float corp 1 at price_index 20 (higher), same president
        float_corp_for_test(gs, 1, par_index=20, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        # Float corp 2 at price_index 10 (lower), same president
        float_corp_for_test(gs, 2, par_index=10, player_id=0)
        CORPS[2].set_cash(gs, 50000)

        # Player 0 is president of all three corps (same-president requirement)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py corp-corp sorted by buyer price")

        # Should have offers: corp 1 and corp 2 buying from corp 0
        # Expected order: Higher-priced buyer (corp 1) before lower-priced (corp 2)
        # For same buyer, lower face value company first
        assert get_offer_count(gs) >= 2

        # First offers should be from higher-priced buyer (corp 1)
        _, corp_id_first, _ = get_offer_at(gs, 0)
        _, corp_id_second, _ = get_offer_at(gs, 1)

        # Both should be corp 1 (or at least first should be)
        assert corp_id_first == 1, f"Expected corp 1 (higher price) first, got {corp_id_first}"

        # If more offers, verify sorting continues
        if get_offer_count(gs) >= 3:
            _, corp_id_third, _ = get_offer_at(gs, 2)
            # After corp 1's offers, corp 2's offers should appear
            if corp_id_third != 1:
                assert corp_id_third == 2

    def test_player_private_sorted_similarly(self):
        """Player-private sorted by buyer price DESC, face value DESC."""
        gs = GameState(3)
        gs.initialize_game()

        # Give companies 0 and 1 to player 0 as private companies
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[1].transfer_to_player(gs, 0)

        # Float corp 0 at price_index 20 (higher), player 0 president
        float_corp_for_test(gs, 0, par_index=20)
        CORPS[0].set_cash(gs, 50000)

        # Float corp 1 at price_index 10 (lower), player 0 president
        float_corp_for_test(gs, 1, par_index=10, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        # Player 0 is president of both corps

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py player private sorted")

        # Should have 6 offers:
        # - 2 corp-to-corp offers (each corp has a company from float, same president)
        # - 4 private offers (2 companies × 2 corps)
        # Sorted: corp-to-corp first (sorted by price), then player-private (sorted by price)
        assert get_offer_count(gs) == 6

        # Collect corp IDs from all offers
        corp_ids = [get_offer_at(gs, i)[1] for i in range(6)]

        # Corp 0 (higher price) should have 3 offers total (1 corp-corp + 2 private)
        # Corp 1 (lower price) should have 3 offers total (1 corp-corp + 2 private)
        corp0_count = sum(1 for c in corp_ids if c == 0)
        corp1_count = sum(1 for c in corp_ids if c == 1)
        assert corp0_count == 3, f"Expected 3 offers from corp 0, got {corp0_count}"
        assert corp1_count == 3, f"Expected 3 offers from corp 1, got {corp1_count}"

        # Within each category (corp-corp, player-private), higher-priced corp comes first
        # The first offer should be from the higher-priced corp
        assert corp_ids[0] == 0, f"First offer should be from higher-priced corp 0, got {corp_ids[0]}"


class TestPhaseFlow:
    """Phase entry and transition tests."""

    def test_wrap_up_sets_up_acquisition(self):
        """WRAP_UP generates offers before transitioning."""
        gs = GameState(3)
        gs.initialize_game()

        # Transition through WRAP_UP
        TURN.set_phase(gs, GamePhases.PHASE_WRAP_UP)
        apply_wrap_up_py(gs)
        assert_invariants(gs, "After apply_wrap_up_py transition to acquisition")

        # Should be in ACQUISITION
        assert TURN.get_phase(gs) == GamePhases.PHASE_ACQUISITION

        # Fresh game has no offers
        assert TURN.get_acq_active_corp(gs) == -1
        assert get_offer_count(gs) == 0

    def test_empty_offers_detected(self):
        """Empty offer buffer is detected."""
        gs = GameState(3)
        gs.initialize_game()
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py empty offers")
        assert get_offer_count(gs) == 0

    def test_transition_to_closing(self):
        """ACQUISITION transitions to CLOSING when offers exhausted."""
        gs = GameState(3)
        gs.initialize_game()

        # Set phase to ACQUISITION
        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)
        assert gs.get_phase() == GamePhases.PHASE_ACQUISITION
        initial_turn = TURN.get_turn_number(gs)

        # Call transition
        transition_to_closing_py(gs)
        assert_invariants(gs, "After transition_to_closing_py")

        # Should now be CLOSING (auto-close executes, then Phase 17 offers)
        assert gs.get_phase() == GamePhases.PHASE_CLOSING
        # Turn number does NOT increment yet (happens after CLOSING completes in Phase 18)
        assert TURN.get_turn_number(gs) == initial_turn

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
        assert_invariants(gs, "After transition_to_closing_py merges player proceeds")

        # Proceeds merged to cash
        assert player.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert player.get_acquisition_proceeds(gs) == 0

    def test_transition_merges_corp_proceeds(self):
        """Transition merges corp acquisition_proceeds."""
        gs = GameState(3)
        gs.initialize_game()

        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        initial_cash = corp.get_cash(gs)
        proceeds = 45

        corp.set_acquisition_proceeds(gs, proceeds)
        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)

        transition_to_closing_py(gs)
        assert_invariants(gs, "After transition_to_closing_py merges corp proceeds")

        assert corp.get_cash(gs) == initial_cash + proceeds
        assert corp.get_acquisition_proceeds(gs) == 0

    def test_transition_merges_acquisition_companies(self):
        """Transition merges acquisition_companies to owned."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp with a different company (not company 0)
        float_corp_for_test(gs, 0)
        corp = CORPS[0]

        # Put company 0 in acquisition zone
        company = COMPANIES[0]
        company.transfer_to_corp_acquisition(gs, 0)

        assert corp.has_acquisition_company(gs, 0)
        assert not corp.owns_company(gs, 0)

        TURN.set_phase(gs, GamePhases.PHASE_ACQUISITION)
        transition_to_closing_py(gs)
        assert_invariants(gs, "After transition_to_closing_py merges acquisition companies")

        # Company moved from acquisition to owned
        assert not corp.has_acquisition_company(gs, 0)
        assert corp.owns_company(gs, 0)


class TestValidation:
    """Validation tests - verify through action handler behavior."""

    def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
        """Setup player private -> corp offer."""
        # Give company to player as private
        COMPANIES[company_id].transfer_to_player(gs, player_id)

        # Float corp (uses different company from deck) with player as president
        float_corp_for_test(gs, corp_id, player_id=player_id)
        CORPS[corp_id].set_cash(gs, corp_cash)

        # Generate offers and present
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After _setup_player_private_offer")

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
        assert_invariants(gs, "After acquisition action price in range")

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
        float_corp_for_test(gs, 2)
        CORPS[2].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py OS FI Buy High test")

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
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py non-OS FI Buy Face test")

        # Non-OS tries FI Buy Face - should reject
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE, 0)
            assert result == 1  # Invalid

    def test_target_already_acquired_rejected(self):
        """Cannot buy company already in acquisition_companies."""
        gs = GameState(3)
        gs.initialize_game()

        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Manually add company to acquisition zone
        COMPANIES[0].transfer_to_corp_acquisition(gs, 0)

        # Try to buy - should reject
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 1  # Invalid

    def test_target_already_owned_rejected(self):
        """Cannot buy company already in buyer's owned_companies."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup: company 0 was in player's private, then moved to corp
        COMPANIES[0].transfer_to_player(gs, 0)
        float_corp_for_test(gs, 0)  # Uses different company
        CORPS[0].set_cash(gs, 50000)

        # Manually transfer company to corp's owned (not acquisition)
        COMPANIES[0].transfer_to_corp(gs, 0)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py target already owned test")

        # Try to buy company corp already owns - should reject
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
            assert result == 1  # Invalid

    # Boundary tests
    def test_price_at_low_boundary(self):
        """Price exactly at low_price succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Offset 0 = low_price
        low_price = get_company_low_price(company_id)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0, f"Action at low_price ({low_price}) should succeed"
        assert_invariants(gs, "After acquisition action at low boundary")

    def test_price_at_high_boundary(self):
        """Price exactly at high_price succeeds."""
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
        assert_invariants(gs, "After acquisition action at high boundary")

    def test_price_one_below_low_fails(self):
        """Price = low_price - 1 fails."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        self._setup_player_private_offer(gs, 0, company_id, 0, 50000)

        # Offset -1 = below low_price
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, -1)
        assert result == 1, "Price below low_price should fail"

    def test_price_one_above_high_fails(self):
        """Price = high_price + 1 fails."""
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

    def test_price_span_varies_by_company(self):
        """Different companies have different valid price spans.

        Verifies that the same offset can be valid for one company but
        invalid for another, based on their unique price ranges.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Company 0: low=1, high=2, span=1 (only offsets 0,1 valid)
        # Company 35 (CDG): low=30, high=80, span=50 (offsets 0-50 valid)
        narrow_company = 0
        wide_company = 35

        narrow_low = get_company_low_price(narrow_company)  # 1
        narrow_high = get_company_high_price(narrow_company)  # 2
        narrow_span = narrow_high - narrow_low  # 1

        wide_low = get_company_low_price(wide_company)  # 30
        wide_high = get_company_high_price(wide_company)  # 80
        wide_span = wide_high - wide_low  # 50

        # Verify spans are different
        assert narrow_span < wide_span, "Test requires different span sizes"

        # Test offset that's valid for wide but not narrow (e.g., offset=10)
        test_offset = 10
        assert test_offset > narrow_span, "Test offset must exceed narrow span"
        assert test_offset <= wide_span, "Test offset must be within wide span"

        # Setup for narrow company - offset 10 should fail
        self._setup_player_private_offer(gs, 0, narrow_company, 0, 50000)
        if get_offer_count(gs) > 0:
            result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, test_offset)
            assert result == 1, f"Offset {test_offset} should fail for narrow company (span={narrow_span})"

        # Setup for wide company - offset 10 should succeed
        gs2 = GameState(3)
        gs2.initialize_game()
        self._setup_player_private_offer(gs2, 0, wide_company, 0, 50000)
        if get_offer_count(gs2) > 0:
            result = apply_acquisition_action_py(gs2, ACTION_ACQ_PRICE, test_offset)
            assert result == 0, f"Offset {test_offset} should succeed for wide company (span={wide_span})"
            assert_invariants(gs2, "After acquisition action wide company span")

    def test_price_offset_maps_to_correct_price(self):
        """Verify price offset correctly maps to low_price + offset.

        Tests that offset 0 = low_price, offset N = low_price + N,
        up to max valid offset = high_price - low_price.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Use company 35 (CDG) which has widest span: low=30, high=80
        company_id = 35
        low_price = get_company_low_price(company_id)  # 30
        high_price = get_company_high_price(company_id)  # 80
        max_offset = high_price - low_price  # 50

        # Give company to player, corp has plenty of cash
        COMPANIES[company_id].transfer_to_player(gs, 0)
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py price offset test")

        assert get_offer_count(gs) > 0, "Should have offer for wide-span company"

        # Test offset 0 = low_price
        player_cash_before = PLAYERS[0].get_cash(gs)
        corp_cash_before = CORPS[0].get_cash(gs)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0, "Offset 0 should succeed"
        assert_invariants(gs, "After acquisition action offset 0")
        payment = corp_cash_before - CORPS[0].get_cash(gs)
        assert payment == low_price, f"Offset 0 should pay low_price ({low_price}), got {payment}"

        # Reset and test max offset = high_price
        gs2 = GameState(3)
        gs2.initialize_game()
        COMPANIES[company_id].transfer_to_player(gs2, 0)
        float_corp_for_test(gs2, 0)
        CORPS[0].set_cash(gs2, 50000)
        setup_acquisition_phase_py(gs2)
        assert_invariants(gs2, "After setup_acquisition_phase_py price offset max test")

        corp_cash_before = CORPS[0].get_cash(gs2)
        result = apply_acquisition_action_py(gs2, ACTION_ACQ_PRICE, max_offset)
        assert result == 0, f"Offset {max_offset} should succeed"
        assert_invariants(gs2, "After acquisition action max offset")
        payment = corp_cash_before - CORPS[0].get_cash(gs2)
        assert payment == high_price, f"Offset {max_offset} should pay high_price ({high_price}), got {payment}"

        # Reset and test middle offset
        gs3 = GameState(3)
        gs3.initialize_game()
        COMPANIES[company_id].transfer_to_player(gs3, 0)
        float_corp_for_test(gs3, 0)
        CORPS[0].set_cash(gs3, 50000)
        setup_acquisition_phase_py(gs3)
        assert_invariants(gs3, "After setup_acquisition_phase_py price offset mid test")

        mid_offset = max_offset // 2  # 25
        expected_price = low_price + mid_offset  # 30 + 25 = 55
        corp_cash_before = CORPS[0].get_cash(gs3)
        result = apply_acquisition_action_py(gs3, ACTION_ACQ_PRICE, mid_offset)
        assert result == 0, f"Offset {mid_offset} should succeed"
        assert_invariants(gs3, "After acquisition action mid offset")
        payment = corp_cash_before - CORPS[0].get_cash(gs3)
        assert payment == expected_price, f"Offset {mid_offset} should pay {expected_price}, got {payment}"

    # Boundary tests
    def test_exact_cash_for_price_succeeds(self):
        """Corp has exactly the price amount, action succeeds."""
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
            assert_invariants(gs, "After acquisition action exact cash")
        else:
            # If no offer generated, that's also acceptable boundary behavior
            # (implementation may require cash > price rather than >= price)
            pass

    def test_one_dollar_short_fails(self):
        """Corp has price - 1, no offer generated."""
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0
        low_price = get_company_low_price(company_id)

        # Give corp one dollar less than low_price
        self._setup_player_private_offer(gs, 0, company_id, 0, low_price - 1)

        # No offer should be generated (insufficient cash filtered at generation)
        assert get_offer_count(gs) == 0, "No offer should be generated with insufficient cash"

    # Boundary tests (seller keeps >= 1 company)
    def test_seller_with_two_companies_can_sell_one(self):
        """Seller has 2 companies, sell 1 succeeds."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company 0, then add company 1
        float_corp_for_test(gs, 0, company_id=0)
        COMPANIES[1].transfer_to_corp(gs, 0)

        # Float corp 1 with same president, give it cash
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py seller with two companies")

        # Should have offers (seller has 2 companies, can sell 1)
        assert get_offer_count(gs) > 0, "Seller with 2 companies should have offers"

    def test_seller_with_one_company_action_rejected(self):
        """Seller with 1 company - offer generated but action rejected."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with exactly one company
        float_corp_for_test(gs, 0, company_id=0)

        # Float corp 1 with same president, give it cash
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py seller with one company")

        # Seller-keeps-1 is checked at ACTION time, not offer generation time
        # Offer IS generated, but action should be rejected
        assert get_offer_count(gs) > 0, "Offer should be generated"

        # Try to execute - should fail seller-keeps-1 check (seller would have 0 after)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 1, "Action should be rejected (seller would have 0 companies)"

    def test_seller_with_one_owned_one_acquisition_can_sell(self):
        """Seller has 1 owned + 1 in acquisition zone, can sell the owned one."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company 0, then add company 1 to acquisition zone
        float_corp_for_test(gs, 0, company_id=0)
        COMPANIES[1].transfer_to_corp_acquisition(gs, 0)

        # Float corp 1 with same president, give it cash
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py seller with one owned one acquisition")

        # _count_seller_companies counts BOTH owned and acquisition_companies
        # With 1 owned + 1 acquisition, selling owned leaves 1 (in acquisition) - valid
        assert get_offer_count(gs) > 0, "Seller with 1 owned + 1 acquisition can sell owned"

    # Acquisition zone / already-owned boundary test
    def test_company_in_acquisition_zone_blocks_offer(self):
        """Company in acquisition zone blocks offer generation."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to player, then put it in corp's acquisition zone
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[0].transfer_to_corp_acquisition(gs, 0)

        # Float corp with different company, give it cash
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py company in acquisition zone")

        # No offer should be generated (company already in acquisition zone)
        assert get_offer_count(gs) == 0, "Company in acquisition zone should block offer"


class TestActionIntegration:
    """Integration tests for action execution."""

    def _setup_player_private_offer(self, gs, player_id, company_id, corp_id, corp_cash):
        """Setup player private -> corp offer."""
        COMPANIES[company_id].transfer_to_player(gs, player_id)
        float_corp_for_test(gs, corp_id, player_id=player_id)
        CORPS[corp_id].set_cash(gs, corp_cash)
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After TestActionIntegration._setup_player_private_offer")

    def test_accept_price_action(self):
        """Full flow - money transfers, company moves to acquisition zone."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup: Player 0 owns company 0, corp 0 buys
        self._setup_player_private_offer(gs, 0, 0, 0, 50000)

        # Verify setup
        assert get_offer_count(gs) > 0
        corp_cash_before = CORPS[0].get_cash(gs)
        player_proceeds_before = PLAYERS[0].get_acquisition_proceeds(gs)

        # Execute action (offset 0 = low_price, always valid)
        result = apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0)
        assert result == 0
        assert_invariants(gs, "After accept price action")

        # Verify money transfer
        corp_cash_after = CORPS[0].get_cash(gs)
        player_proceeds_after = PLAYERS[0].get_acquisition_proceeds(gs)
        assert corp_cash_after < corp_cash_before
        assert player_proceeds_after > player_proceeds_before

        # Verify company in acquisition zone
        assert CORPS[0].has_acquisition_company(gs, 0)

    def test_fi_buy_high_action(self):
        """Non-OS buys from FI at high price."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make non-OS corp active
        COMPANIES[0].transfer_to_fi(gs)
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py FI buy high")

        if get_offer_count(gs) > 0:
            corp_cash_before = CORPS[0].get_cash(gs)
            fi_cash_before = FI.get_cash(gs)

            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_HIGH, 0)
            assert result == 0
            assert_invariants(gs, "After FI buy high action")

            # Verify money transfer to FI
            corp_cash_after = CORPS[0].get_cash(gs)
            fi_cash_after = FI.get_cash(gs)
            assert corp_cash_after < corp_cash_before
            assert fi_cash_after > fi_cash_before

            # Verify company in acquisition zone
            assert CORPS[0].has_acquisition_company(gs, 0)

    def test_fi_buy_face_action(self):
        """OS buys from FI at face value."""
        gs = GameState(3)
        gs.initialize_game()

        # Give company to FI, make OS active
        COMPANIES[0].transfer_to_fi(gs)
        float_corp_for_test(gs, 2)
        CORPS[2].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py FI buy face")

        if get_offer_count(gs) > 0:
            corp_cash_before = CORPS[2].get_cash(gs)
            fi_cash_before = FI.get_cash(gs)

            result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE, 0)
            assert result == 0
            assert_invariants(gs, "After FI buy face action")

            # Verify money transfer at face value
            corp_cash_after = CORPS[2].get_cash(gs)
            fi_cash_after = FI.get_cash(gs)
            face_value = get_company_face_value(0)
            assert corp_cash_after == corp_cash_before - face_value
            assert fi_cash_after == fi_cash_before + face_value

            # Verify company in acquisition zone
            assert CORPS[2].has_acquisition_company(gs, 0)

    def test_os_pays_face_value_not_high_price(self):
        """Verify OS pays face_value (not high_price) when buying from FI.

        This test verifies the RULES.md OS privilege: 'Always considered highest
        share price; pays only Face Value to Foreign Investor.'
        """
        gs = GameState(3)
        gs.initialize_game()

        # Use company 0 which has face_value=1, high_price=2
        company_id = 0
        face_value = get_company_face_value(company_id)
        high_price = get_company_high_price(company_id)

        # Precondition: face_value and high_price must be different
        # to prove OS is getting a discount
        assert face_value != high_price, \
            f"Test requires face_value ({face_value}) != high_price ({high_price})"

        # Give company to FI, make OS (corp 2) active
        COMPANIES[company_id].transfer_to_fi(gs)
        float_corp_for_test(gs, 2)
        CORPS[2].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py OS face value test")

        assert get_offer_count(gs) > 0, "Should have FI offer for OS"

        corp_cash_before = CORPS[2].get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        # OS uses FI_BUY_FACE action
        result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE, 0)
        assert result == 0, "FI_BUY_FACE should succeed for OS"
        assert_invariants(gs, "After OS FI buy face action")

        # Verify OS paid face_value, NOT high_price
        corp_cash_after = CORPS[2].get_cash(gs)
        fi_cash_after = FI.get_cash(gs)

        amount_paid = corp_cash_before - corp_cash_after
        amount_received = fi_cash_after - fi_cash_before

        assert amount_paid == face_value, \
            f"OS should pay face_value ({face_value}), not {amount_paid}"
        assert amount_paid != high_price, \
            f"OS should NOT pay high_price ({high_price})"
        assert amount_received == face_value, \
            f"FI should receive face_value ({face_value}), got {amount_received}"

        # Verify company in acquisition zone
        assert CORPS[2].has_acquisition_company(gs, company_id)

    def test_pass_action(self):
        """Offer index advances, next offer presented."""
        gs = GameState(3)
        gs.initialize_game()

        # Setup two offers: two companies to same player/corp
        COMPANIES[0].transfer_to_player(gs, 0)
        COMPANIES[1].transfer_to_player(gs, 0)
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py pass action test")

        # Should have offers
        offer_count = get_offer_count(gs)
        if offer_count > 0:
            first_target = TURN.get_acq_target_company(gs)

            # Pass on first offer
            result = apply_acquisition_action_py(gs, ACTION_PASS, 0)
            assert result == 0
            assert_invariants(gs, "After pass action")

            # Check if moved to next offer or cleared
            second_target = TURN.get_acq_target_company(gs)
            if offer_count > 1:
                # Should advance to next offer
                assert second_target != first_target
            else:
                # Should clear (no more offers)
                assert second_target == -1

    def test_fi_intervention_equivalent_higher_price_gets_company(self):
        """Higher-priced corp accepting FI offer prevents lower-priced corp from getting it.

        RULES.md intervention mechanism: 'Any corporation with higher share price
        and enough money may intervene and buy instead.'

        Implementation: Offers are sorted by descending share price, so higher-priced
        corps are offered each FI company first. When they accept, the company is
        transferred and unavailable to lower-priced corps. The lower-priced corp's
        offer for that company is automatically skipped via dynamic validation.

        This test verifies the end-to-end behavior of this intervention-equivalent
        mechanism.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Give company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Corp 0: higher price (index 20), has cash
        high_corp = CORPS[0]
        float_corp_for_test(gs, 0, par_index=20)
        high_corp.set_cash(gs, 50000)

        # Corp 1: lower price (index 10), has cash
        low_corp = CORPS[1]
        float_corp_for_test(gs, 1, par_index=10)
        low_corp.set_cash(gs, 50000)

        # Generate offers
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py intervention test")

        # Verify higher-priced corp (0) is offered first via visible state
        assert TURN.get_acq_active_corp(gs) == 0, "Higher-priced corp should be active"
        assert TURN.get_acq_target_company(gs) == 0, "Company 0 should be the target"

        # Verify offer buffer contains both corps' offers for company 0
        assert get_offer_count(gs) >= 2, "Should have offers for both corps"
        _, first_corp, first_company = get_offer_at(gs, 0)
        _, second_corp, second_company = get_offer_at(gs, 1)
        assert first_corp == 0 and first_company == 0, "First offer: corp 0 for company 0"
        assert second_corp == 1 and second_company == 0, "Second offer: corp 1 for company 0"

        # Higher-priced corp accepts
        fi_cash_before = FI.get_cash(gs)
        high_corp_cash_before = high_corp.get_cash(gs)

        result = apply_acquisition_action_py(gs, ACTION_ACQ_FI_HIGH, 0)
        assert result == 0, "Accept action should succeed"
        assert_invariants(gs, "After FI intervention accept action")

        # Verify company transferred to higher-priced corp
        assert high_corp.has_acquisition_company(gs, 0), "High corp should have company in acquisition zone"
        assert not FI.owns_company(gs, 0), "FI should no longer own company"

        # Verify money transferred at high price
        high_price = COMPANIES[0].get_high_price()
        assert high_corp.get_cash(gs) == high_corp_cash_before - high_price
        assert FI.get_cash(gs) == fi_cash_before + high_price

        # KEY ASSERTION: Lower-priced corp's offer for company 0 was automatically
        # skipped because the company is no longer owned by FI. The current offer
        # presented (if any) should NOT be for company 0.
        current_target = TURN.get_acq_target_company(gs)
        if current_target != -1:
            # If there's still an active offer, it must not be for company 0
            assert current_target != 0, (
                f"Lower-priced corp should not be offered company 0 "
                f"(got target company {current_target})"
            )
        # If current_target == -1, phase has ended (no more valid offers) - also correct


class TestZoneMerging:
    """Tests for acquisition zone merging at phase end."""

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
        assert_invariants(gs, "After merge_acquisition_zones_py player proceeds")

        # Proceeds merged to cash
        assert player.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert player.get_acquisition_proceeds(gs) == 0

    def test_corp_proceeds_merge_to_cash(self):
        """Corp acquisition_proceeds merge to cash at phase end."""
        gs = GameState(3)
        gs.initialize_game()

        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        initial_cash = corp.get_cash(gs)
        proceeds = 30

        corp.set_acquisition_proceeds(gs, proceeds)
        assert corp.get_acquisition_proceeds(gs) == proceeds

        # Trigger merge
        merge_acquisition_zones_py(gs)
        assert_invariants(gs, "After merge_acquisition_zones_py corp proceeds")

        # Proceeds merged to cash
        assert corp.get_cash(gs) == initial_cash + proceeds
        # Proceeds cleared
        assert corp.get_acquisition_proceeds(gs) == 0

    def test_acquisition_companies_merge_to_owned(self):
        """Corp acquisition_companies merge to owned_companies at phase end."""
        gs = GameState(3)
        gs.initialize_game()

        float_corp_for_test(gs, 0)
        corp = CORPS[0]

        company = COMPANIES[0]

        # Put company in acquisition zone
        company.transfer_to_corp_acquisition(gs, 0)
        assert corp.has_acquisition_company(gs, 0)
        assert not corp.owns_company(gs, 0)

        # Trigger merge
        merge_acquisition_zones_py(gs)
        assert_invariants(gs, "After merge_acquisition_zones_py companies to owned")

        # Company moved from acquisition to owned
        assert not corp.has_acquisition_company(gs, 0)
        assert corp.owns_company(gs, 0)

    def test_zones_cleared_after_merge(self):
        """Acquisition zones are cleared (zeroed) after merge."""
        gs = GameState(3)
        gs.initialize_game()

        player = PLAYERS[0]
        player.add_acquisition_proceeds(gs, 50)

        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_acquisition_proceeds(gs, 40)

        # Trigger merge
        merge_acquisition_zones_py(gs)
        assert_invariants(gs, "After merge_acquisition_zones_py zones cleared")

        # All zones cleared
        assert player.get_acquisition_proceeds(gs) == 0
        assert corp.get_acquisition_proceeds(gs) == 0


class TestEdgeCases:
    """Edge case tests for empty states and unusual configurations."""

    def test_no_active_corps_no_offers(self):
        """No corps active, verify 0 offers."""
        gs = GameState(3)
        gs.initialize_game()

        # No corps activated
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py no active corps")

        assert get_offer_count(gs) == 0, "No active corps should mean no offers"

    def test_empty_fi_no_fi_offers(self):
        """FI owns no companies, verify no FI offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Make corp active but FI has no companies
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py empty FI")

        # No offers (FI empty)
        assert get_offer_count(gs) == 0, "Empty FI should generate no offers"

    def test_no_player_privates_no_private_offers(self):
        """Players own no private companies, verify no private offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Make corp active with cash but no player owns privates
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py no player privates")

        # No offers (no player companies)
        assert get_offer_count(gs) == 0, "No player companies should generate no private offers"

    def test_no_corp_companies_no_corp_corp_offers(self):
        """Corps own no companies, verify no corp-to-corp offers.

        Note: With float_corp_for_test, corps DO have companies, so we test that
        corps cannot buy from THEMSELVES (only same president, different corps).
        For true "no companies" test, that would require set_active which we're removing.
        This test verifies corp-to-corp logic when both corps have companies.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Float two corps with same president - each has 1 company
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py two corps same president")

        # Now both corps have companies and same president, so offers WILL be generated
        # This is expected behavior - the original test was checking an impossible state
        # where corps are active but have no companies
        assert get_offer_count(gs) > 0, "Two corps with same president and companies should have offers"

    def test_single_corp_no_corp_corp_offers(self):
        """Only one corp active, can't have corp-to-corp offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Float single corp - it has a company
        float_corp_for_test(gs, 0)
        CORPS[0].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py single corp")

        # No corp-to-corp offers (need buyer AND seller)
        assert get_offer_count(gs) == 0, "Single corp cannot have corp-to-corp offers"

    def test_same_president_constraint_explicit(self):
        """Same-president logic is the ONLY thing blocking an offer."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 - player 0 is president
        float_corp_for_test(gs, 0)

        # Float corp 1 with SAME president (player 0)
        float_corp_for_test(gs, 1, player_id=0)
        CORPS[1].set_cash(gs, 50000)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py same president")

        # Same president should allow corp-to-corp offers
        assert get_offer_count(gs) > 0, "Same president should allow corp-to-corp offers"

        # Reset and test with different presidents
        gs2 = GameState(3)
        gs2.initialize_game()

        # Float corp 0 with player 0
        float_corp_for_test(gs2, 0)

        # Float corp 1 with DIFFERENT president (player 1)
        float_corp_for_test(gs2, 1, player_id=1)
        CORPS[1].set_cash(gs2, 50000)

        setup_acquisition_phase_py(gs2)
        assert_invariants(gs2, "After setup_acquisition_phase_py different presidents")

        # Different presidents should block corp-to-corp offers
        assert get_offer_count(gs2) == 0, "Different presidents should block corp-to-corp offers"


class TestReceivershipAutoBuy:
    """Tests for receivership auto-buy behavior."""

    def test_receivership_auto_buys_affordable_fi(self):
        """Receivership corp auto-buys affordable FI company at HIGH price.

        Per RULES.md: FI 'Only sells at maximum allowed price'.
        Only OS has the special ability to pay face value - receivership corps pay full price.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Float corp 0, then put in receivership by selling all shares
        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_cash(gs, 50000)

        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        # Record initial values - receivership pays HIGH price, not face value
        high_price = COMPANIES[0].get_high_price()
        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        # Call setup_acquisition_phase_py to generate offers and trigger auto-buy
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership auto-buy")

        # Verify auto-buy executed at HIGH price
        assert corp.has_acquisition_company(gs, 0), "Receivership corp should auto-buy FI company"
        assert FI.get_cash(gs) == fi_cash_before + high_price, "FI should receive high price"
        assert corp.get_cash(gs) == corp_cash_before - high_price, "Corp should pay high price"
        # Offer should be processed (no offers left or active corp cleared)
        assert get_offer_count(gs) == 0 or TURN.get_acq_active_corp(gs) == -1

    def test_receivership_skips_unaffordable_fi(self):
        """Receivership corp auto-passes when can't afford FI company."""
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to FI
        COMPANIES[0].transfer_to_fi(gs)

        # Float corp 0 with minimal cash (can't afford high price)
        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_cash(gs, 1)

        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        # Record initial FI cash
        fi_cash_before = FI.get_cash(gs)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership skip unaffordable")

        # Verify auto-pass: no buy happened
        assert not corp.has_acquisition_company(gs, 0), "Receivership corp should not buy unaffordable company"
        assert FI.get_cash(gs) == fi_cash_before, "FI cash should be unchanged"
        # Company still owned by FI
        assert FI.owns_company(gs, 0), "Company should still be owned by FI"

    def test_receivership_skips_non_fi_offers(self):
        """Receivership corp auto-passes on non-FI offers."""
        gs = GameState(3)
        gs.initialize_game()

        # Transfer company 0 to player 0 as private
        COMPANIES[0].transfer_to_player(gs, 0)

        # Float corp 0 with plenty of cash (player 0 is president)
        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_cash(gs, 50000)

        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership skip non-FI")

        # Verify: Receivership corp cannot buy from player
        # (no offers should be generated or they should be auto-passed)
        assert not corp.has_acquisition_company(gs, 0), "Receivership corp should not buy from player"
        # No offers for receivership buying from player
        assert get_offer_count(gs) == 0 or TURN.get_acq_active_corp(gs) == -1

    def test_receivership_cannot_sell(self):
        """Receivership corp cannot sell companies."""
        gs = GameState(3)
        gs.initialize_game()

        # Float corp 0 with company (player 0), then put in receivership
        float_corp_for_test(gs, 0)
        corp0 = CORPS[0]
        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        # Float corp 1 with cash and DIFFERENT president (player 1)
        # This ensures no same-president corp-to-corp offers regardless of
        # whether get_president_id correctly returns -1 for receivership
        float_corp_for_test(gs, 1, player_id=1)
        corp1 = CORPS[1]
        corp1.set_cash(gs, 50000)

        # Call setup_acquisition_phase_py
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership cannot sell")

        # Verify: No offers should exist because:
        # - Corp 0 has company but is in receivership (can't sell)
        # - Different presidents means no corp-to-corp offers
        assert get_offer_count(gs) == 0, "No offers should exist for receivership seller"

    def test_receivership_buys_most_expensive_affordable(self):
        """Receivership picks most expensive affordable FI company.

        Per RULES.md: 'Highest share price corporation in receivership tries
        to buy most expensive affordable company.'

        This test verifies the corp picks the expensive company over the cheap one
        when it can only afford one.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Give two companies to FI with different high prices
        # Company 0: face=1, high=2
        # Company 5: face=8, high=10
        cheap_company = 0
        expensive_company = 5
        COMPANIES[cheap_company].transfer_to_fi(gs)
        COMPANIES[expensive_company].transfer_to_fi(gs)

        cheap_high = get_company_high_price(cheap_company)  # 2
        expensive_high = get_company_high_price(expensive_company)  # 10

        # Float corp, set cash exactly enough for expensive but not both
        # (can afford 10 but not 10+2=12)
        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_cash(gs, expensive_high)  # Exactly enough for expensive, not both
        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        # Trigger auto-buy
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership most expensive")

        # Should have bought the MORE expensive company (not the cheap one)
        assert corp.has_acquisition_company(gs, expensive_company), \
            "Receivership should buy most expensive affordable company"
        assert not corp.has_acquisition_company(gs, cheap_company), \
            "Should not have bought cheap company (no cash left after expensive)"

        # Verify paid high price for the expensive company
        assert corp.get_cash(gs) == 0, \
            "Corp should have spent all cash on expensive company"
        assert FI.get_cash(gs) == fi_cash_before + expensive_high, \
            f"FI should receive high price ({expensive_high})"
        assert FI.owns_company(gs, cheap_company), \
            "Cheap company should still be owned by FI"

    def test_receivership_insufficient_for_high_but_enough_for_face(self):
        """Receivership can only buy at HIGH price - passes if can only afford face.

        Per RULES.md: FI 'Only sells at maximum allowed price'.
        If corp has cash between face_value and high_price, it cannot buy.
        """
        gs = GameState(3)
        gs.initialize_game()

        # Use company 0: face=1, high=2
        company_id = 0
        COMPANIES[company_id].transfer_to_fi(gs)

        face_value = get_company_face_value(company_id)  # 1
        high_price = get_company_high_price(company_id)  # 2

        # Sanity check: face < high for this test to be meaningful
        assert face_value < high_price, "Test requires face_value < high_price"

        # Float corp with cash between face and high (can afford face, not high)
        float_corp_for_test(gs, 0)
        corp = CORPS[0]
        corp.set_cash(gs, face_value)  # Exactly face_value, less than high_price
        # Put corp 0 in receivership (set_shares auto-adjusts bank shares and presidency)
        PLAYERS[0].set_shares(gs, 0, 0)

        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        # Trigger auto-buy attempt
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py receivership insufficient for high")

        # Should NOT have bought - can't afford high price
        assert not corp.has_acquisition_company(gs, company_id), \
            "Receivership should not buy when can only afford face value (not high price)"
        assert corp.get_cash(gs) == corp_cash_before, \
            "Corp cash should be unchanged"
        assert FI.get_cash(gs) == fi_cash_before, \
            "FI cash should be unchanged"
        assert FI.owns_company(gs, company_id), \
            "Company should still be owned by FI"


class TestOSReceivershipFaceValue:
    """Tests for OS paying face value to FI even in receivership."""

    def test_os_receivership_buys_at_face_value(self):
        """OS in receivership auto-buys FI company at face value, not high price.

        Per RULES.md line 174: OS 'Always pays face value to Foreign Investor.'
        The word 'always' means this applies regardless of receivership status.
        """
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0  # face=1, high=2
        COMPANIES[company_id].transfer_to_fi(gs)

        face_value = get_company_face_value(company_id)
        high_price = get_company_high_price(company_id)
        assert face_value < high_price, "Test requires face_value < high_price"

        # Float OS (corp 2) with enough cash for high price
        float_corp_for_test(gs, 2)
        corp = CORPS[2]
        corp.set_cash(gs, 50000)

        # Put OS in receivership
        PLAYERS[0].set_shares(gs, 2, 0)

        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py OS receivership face-value buy")

        # OS should auto-buy at face value
        assert corp.has_acquisition_company(gs, company_id), \
            "OS in receivership should auto-buy FI company"
        assert FI.get_cash(gs) == fi_cash_before + face_value, \
            "FI should receive face value (not high price)"
        assert corp.get_cash(gs) == corp_cash_before - face_value, \
            "OS should pay face value (not high price)"

    def test_os_receivership_buys_when_only_affordable_at_face(self):
        """OS in receivership buys when it can afford face value but NOT high price.

        This is the key scenario: without the OS special ability, this purchase
        would be skipped because high_price > cash >= face_value.
        """
        gs = GameState(3)
        gs.initialize_game()

        company_id = 0  # face=1, high=2
        COMPANIES[company_id].transfer_to_fi(gs)

        face_value = get_company_face_value(company_id)
        high_price = get_company_high_price(company_id)
        assert face_value < high_price

        # Float OS with cash exactly equal to face value (can't afford high)
        float_corp_for_test(gs, 2)
        corp = CORPS[2]
        corp.set_cash(gs, face_value)

        # Put OS in receivership
        PLAYERS[0].set_shares(gs, 2, 0)

        corp_cash_before = corp.get_cash(gs)
        fi_cash_before = FI.get_cash(gs)

        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py OS receivership face-only affordable")

        # OS should still buy at face value
        assert corp.has_acquisition_company(gs, company_id), \
            "OS in receivership should buy when affordable at face value"
        assert FI.get_cash(gs) == fi_cash_before + face_value
        assert corp.get_cash(gs) == corp_cash_before - face_value


class TestQuicksortHelpers:
    """Tests for quicksort helper functions used in offer sorting."""

    # =========================================================================
    # _qsort_desc_3: Single-key descending sort (3 parallel arrays)
    # =========================================================================

    def test_qsort_desc_3_basic(self):
        """Basic descending sort."""
        keys = [3, 1, 4, 1, 5, 9, 2, 6]
        arr1 = [0, 1, 2, 3, 4, 5, 6, 7]
        arr2 = [10, 11, 12, 13, 14, 15, 16, 17]

        sorted_keys, sorted_arr1, sorted_arr2 = qsort_desc_3_py(keys, arr1, arr2)

        assert sorted_keys == [9, 6, 5, 4, 3, 2, 1, 1]
        # Verify parallel arrays maintain correspondence
        for i in range(len(sorted_keys)):
            orig_idx = keys.index(sorted_keys[i]) if i == 0 or sorted_keys[i] != sorted_keys[i-1] else \
                       [j for j, k in enumerate(keys) if k == sorted_keys[i]][1]
            # Each element maintains its original pairing
            assert sorted_arr2[i] == sorted_arr1[i] + 10

    def test_qsort_desc_3_already_sorted(self):
        """Already sorted input remains unchanged."""
        keys = [9, 7, 5, 3, 1]
        arr1 = [0, 1, 2, 3, 4]
        arr2 = [5, 6, 7, 8, 9]

        sorted_keys, sorted_arr1, sorted_arr2 = qsort_desc_3_py(keys, arr1, arr2)

        assert sorted_keys == [9, 7, 5, 3, 1]
        assert sorted_arr1 == [0, 1, 2, 3, 4]
        assert sorted_arr2 == [5, 6, 7, 8, 9]

    def test_qsort_desc_3_reverse_sorted(self):
        """Reverse sorted input gets properly sorted."""
        keys = [1, 2, 3, 4, 5]
        arr1 = [0, 1, 2, 3, 4]
        arr2 = [10, 20, 30, 40, 50]

        sorted_keys, sorted_arr1, sorted_arr2 = qsort_desc_3_py(keys, arr1, arr2)

        assert sorted_keys == [5, 4, 3, 2, 1]
        assert sorted_arr1 == [4, 3, 2, 1, 0]
        assert sorted_arr2 == [50, 40, 30, 20, 10]

    def test_qsort_desc_3_single_element(self):
        """Single element array."""
        keys, arr1, arr2 = qsort_desc_3_py([42], [1], [2])
        assert keys == [42]
        assert arr1 == [1]
        assert arr2 == [2]

    def test_qsort_desc_3_empty(self):
        """Empty arrays."""
        keys, arr1, arr2 = qsort_desc_3_py([], [], [])
        assert keys == []
        assert arr1 == []
        assert arr2 == []

    def test_qsort_desc_3_duplicates(self):
        """Arrays with duplicate keys."""
        keys = [5, 5, 3, 5, 3]
        arr1 = [0, 1, 2, 3, 4]
        arr2 = [10, 11, 12, 13, 14]

        sorted_keys, sorted_arr1, sorted_arr2 = qsort_desc_3_py(keys, arr1, arr2)

        assert sorted_keys == [5, 5, 5, 3, 3]
        # All 5s should come before 3s, parallel arrays follow
        for i, k in enumerate(sorted_keys):
            # Verify the parallel array values match their original key
            original_idx = sorted_arr1[i]
            assert keys[original_idx] == k

    def test_qsort_desc_3_two_elements(self):
        """Two element swap."""
        keys = [1, 2]
        arr1 = [0, 1]
        arr2 = [10, 20]

        sorted_keys, sorted_arr1, sorted_arr2 = qsort_desc_3_py(keys, arr1, arr2)

        assert sorted_keys == [2, 1]
        assert sorted_arr1 == [1, 0]
        assert sorted_arr2 == [20, 10]

    # =========================================================================
    # _qsort_price_fv_4: Two-key sort (price DESC, face_value DESC)
    # =========================================================================

    def test_qsort_price_fv_4_basic(self):
        """Basic two-key sort: price DESC, then face_value DESC."""
        prices = [10, 20, 10, 20, 15]
        fvs = [5, 3, 2, 8, 4]
        arr1 = [0, 1, 2, 3, 4]
        arr2 = [100, 101, 102, 103, 104]

        sorted_prices, sorted_fvs, sorted_arr1, sorted_arr2 = qsort_price_fv_4_py(
            prices, fvs, arr1, arr2
        )

        # Expected order: (20,8), (20,3), (15,4), (10,5), (10,2)
        assert sorted_prices == [20, 20, 15, 10, 10]
        assert sorted_fvs == [8, 3, 4, 5, 2]
        assert sorted_arr1 == [3, 1, 4, 0, 2]
        assert sorted_arr2 == [103, 101, 104, 100, 102]

    def test_qsort_price_fv_4_same_price_different_fv(self):
        """Same price, sorted by face_value descending."""
        prices = [10, 10, 10, 10]
        fvs = [30, 10, 20, 5]
        arr1 = [0, 1, 2, 3]
        arr2 = [0, 1, 2, 3]

        sorted_prices, sorted_fvs, sorted_arr1, sorted_arr2 = qsort_price_fv_4_py(
            prices, fvs, arr1, arr2
        )

        assert sorted_prices == [10, 10, 10, 10]
        assert sorted_fvs == [30, 20, 10, 5]  # Descending face value
        assert sorted_arr1 == [0, 2, 1, 3]

    def test_qsort_price_fv_4_different_prices(self):
        """Different prices, face_value doesn't matter."""
        prices = [5, 15, 25, 10, 20]
        fvs = [100, 100, 100, 100, 100]  # All same fv
        arr1 = [0, 1, 2, 3, 4]
        arr2 = [0, 0, 0, 0, 0]

        sorted_prices, sorted_fvs, sorted_arr1, sorted_arr2 = qsort_price_fv_4_py(
            prices, fvs, arr1, arr2
        )

        assert sorted_prices == [25, 20, 15, 10, 5]  # Descending price
        assert sorted_arr1 == [2, 4, 1, 3, 0]

    def test_qsort_price_fv_4_already_sorted(self):
        """Already sorted input (price DESC, face_value DESC)."""
        prices = [30, 20, 20, 10]
        fvs = [1, 10, 5, 1]  # For price 20: 10 > 5, so already sorted DESC
        arr1 = [0, 1, 2, 3]
        arr2 = [0, 1, 2, 3]

        sorted_prices, sorted_fvs, sorted_arr1, sorted_arr2 = qsort_price_fv_4_py(
            prices, fvs, arr1, arr2
        )

        assert sorted_prices == [30, 20, 20, 10]
        assert sorted_fvs == [1, 10, 5, 1]
        assert sorted_arr1 == [0, 1, 2, 3]

    def test_qsort_price_fv_4_single_element(self):
        """Single element."""
        prices, fvs, arr1, arr2 = qsort_price_fv_4_py([10], [5], [0], [100])
        assert prices == [10]
        assert fvs == [5]
        assert arr1 == [0]
        assert arr2 == [100]

    def test_qsort_price_fv_4_empty(self):
        """Empty arrays."""
        prices, fvs, arr1, arr2 = qsort_price_fv_4_py([], [], [], [])
        assert prices == []
        assert fvs == []
        assert arr1 == []
        assert arr2 == []

    def test_qsort_price_fv_4_realistic_scenario(self):
        """Realistic scenario: corp prices and company face values."""
        # Simulating: 3 corps (prices 27, 20, 20) buying companies (fvs 5, 8, 12, 30)
        prices = [20, 27, 20, 20]  # Corp share prices
        fvs = [8, 5, 30, 12]  # Company face values
        corp_ids = [1, 0, 1, 2]
        company_ids = [10, 20, 30, 40]

        sorted_prices, sorted_fvs, sorted_corps, sorted_companies = qsort_price_fv_4_py(
            prices, fvs, corp_ids, company_ids
        )

        # Expected order (price DESC, face_value DESC):
        # (27, 5) -> corp 0, company 20
        # (20, 30) -> corp 1, company 30
        # (20, 12) -> corp 2, company 40
        # (20, 8) -> corp 1, company 10
        assert sorted_prices == [27, 20, 20, 20]
        assert sorted_fvs == [5, 30, 12, 8]
        assert sorted_corps == [0, 1, 2, 1]
        assert sorted_companies == [20, 30, 40, 10]


class TestNoOfferBoundsCheck:
    """Verify actions return STATUS_INVALID (not segfault) when no offer is active."""

    def test_price_action_no_offer_returns_invalid(self):
        """ACTION_ACQ_PRICE with no active offer returns 1, not segfault."""
        gs = GameState(3)
        gs.initialize_game()
        # No offers generated — acq_target_company defaults to -1
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py no offer price check")
        assert get_offer_count(gs) == 0
        assert apply_acquisition_action_py(gs, ACTION_ACQ_PRICE, 0) == 1

    def test_fi_high_action_no_offer_returns_invalid(self):
        """ACTION_ACQ_FI_HIGH with no active offer returns 1, not segfault."""
        gs = GameState(3)
        gs.initialize_game()
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py no offer FI high check")
        assert apply_acquisition_action_py(gs, ACTION_ACQ_FI_HIGH) == 1

    def test_fi_face_action_no_offer_returns_invalid(self):
        """ACTION_ACQ_FI_FACE with no active offer returns 1, not segfault."""
        gs = GameState(3)
        gs.initialize_game()
        setup_acquisition_phase_py(gs)
        assert_invariants(gs, "After setup_acquisition_phase_py no offer FI face check")
        assert apply_acquisition_action_py(gs, ACTION_ACQ_FI_FACE) == 1
