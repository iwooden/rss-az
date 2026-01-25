"""Tests for ACQUISITION phase offer generation."""

import pytest
from core.state import GameState
from core.data import CORP_NAMES, GamePhases, get_company_face_value
from entities.player import PLAYERS
from entities.fi import FI
from entities.corp import CORPS
from entities.turn import TURN
from phases.acquisition import (
    generate_offers_py,
    get_offer_count,
    get_offer_at,
    setup_acquisition_phase_py
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
        # TODO: This test requires setting up complex game state with:
        # - FI owning companies
        # - Active corps with cash
        # Full implementation deferred until integration testing
        pass

    def test_os_fi_offers_first(self):
        """OFFER-02: OS->FI offers come before other corp->FI offers."""
        # TODO: Setup scenario with OS and another corp both able to buy from FI
        # Verify OS offers appear first in buffer
        pass

    def test_corp_fi_sorted_by_price(self):
        """OFFER-03: Non-OS corp->FI offers sorted by descending share price."""
        # TODO: Setup multiple corps with different share prices
        # Verify offers sorted correctly
        pass

    def test_corp_corp_offers_same_president(self):
        """OFFER-04: Corp->Corp offers only for same president."""
        # TODO: Setup player as president of multiple corps
        # Verify offers only between corps with same president
        pass

    def test_player_private_offers(self):
        """OFFER-05: Corp->Player private offers generated."""
        # TODO: Setup player with private companies and corp presidency
        # Verify offers generated
        pass


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
