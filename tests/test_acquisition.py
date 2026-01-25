"""Tests for ACQUISITION phase offer state presentation."""

import pytest
from core.state import GameState
from entities.turn import TURN
from phases.acquisition import (
    present_current_offer_py,
    advance_to_next_offer_py,
    get_offer_index
)


class TestOfferStatePresentation:
    """STATE-01 through STATE-04: Offer state management."""

    def test_no_offers_clears_state(self):
        """STATE-04: No offers sets acq_active_corp to -1."""
        gs = GameState(3)
        gs.initialize_game()

        # With no offers in buffer (count=0), presenting should clear state
        present_current_offer_py(gs)

        assert TURN.get_acq_active_corp(gs) == -1
        assert TURN.get_acq_target_company(gs) == -1
        assert not TURN.is_acq_fi_offer(gs)

    def test_offer_sets_active_corp(self):
        """STATE-01: Current offer sets acq_active_corp."""
        # Setup state with valid offer
        # Verify acq_active_corp matches offer's corp_id
        pass

    def test_offer_sets_target_company(self):
        """STATE-01: Current offer sets acq_target_company."""
        pass

    def test_fi_offer_flag_set(self):
        """STATE-01: acq_is_fi_offer true when FI owns target."""
        pass

    def test_advance_increments_index(self):
        """Advancing moves to next offer."""
        pass
