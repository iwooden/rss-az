"""Tests for the WRAP_UP phase.

Covers: player reorder by descending cash (tiebreak on ascending old turn
order), consecutive-pass counter clear, Foreign Investor face-value purchase
loop (cheapest-affordable iteration, replacement draws, empty-deck draw,
CoO-tier crossing), LOC_REVEALED → LOC_AUCTION promotion (including drawn
replacements), and transition to PHASE_ACQUISITION.

WRAP_UP is automated — tests invoke ``apply_wrap_up_py`` directly rather than
routing through the driver. See ``auto-phases.md`` for the rationale.
"""
import pytest

from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from entities.deck import DECK
from phases.wrap_up import apply_wrap_up_py

from tests.phases.conftest import (
    make_auto_phase_state,
    assert_invariants,
    assert_token_data_invariants,
    assert_post_auto,
    float_corp_for_test,
)
from tests.phases.helpers.ownership import ids_at_location


# =============================================================================
# HELPERS
# =============================================================================

PHASE_WRAP_UP = int(GamePhases.PHASE_WRAP_UP)
PHASE_ACQUISITION = int(GamePhases.PHASE_ACQUISITION)

LOC_DECK = int(CompanyLocation.LOC_DECK)
LOC_AUCTION = int(CompanyLocation.LOC_AUCTION)
LOC_REVEALED = int(CompanyLocation.LOC_REVEALED)
LOC_FI = int(CompanyLocation.LOC_FI)
LOC_REMOVED = int(CompanyLocation.LOC_REMOVED)


def _auction_ids(state):
    """Return sorted list of company IDs currently at LOC_AUCTION."""
    return ids_at_location(state, LOC_AUCTION)


def _revealed_ids(state):
    """Return sorted list of company IDs currently at LOC_REVEALED."""
    return ids_at_location(state, LOC_REVEALED)


def _fi_owned_ids(state):
    """Return sorted list of company IDs currently at LOC_FI."""
    return ids_at_location(state, LOC_FI)


# =============================================================================
# PLAYER REORDER
# =============================================================================

class TestReorderPlayers:
    """WRAP_UP step 1: rerank players by descending cash, tiebreak on old order."""

    def test_reorder_by_descending_cash(self):
        """Highest-cash player ranks first; lowest ranks last."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        PLAYERS[0].set_cash(state, 10)
        PLAYERS[1].set_cash(state, 30)
        PLAYERS[2].set_cash(state, 50)

        apply_wrap_up_py(state)

        # Player 2 (most cash) ends up at position 0; Player 0 (least) last.
        assert PLAYERS[2].get_turn_order(state) == 0
        assert PLAYERS[1].get_turn_order(state) == 1
        assert PLAYERS[0].get_turn_order(state) == 2
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_tiebreak_ascending_old_position(self):
        """Tied cash: smaller old turn_order wins the tiebreak."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # p0 lowest; p1 and p2 tied. Old turn order is 0, 1, 2.
        PLAYERS[0].set_cash(state, 20)
        PLAYERS[1].set_cash(state, 30)
        PLAYERS[2].set_cash(state, 30)

        apply_wrap_up_py(state)

        # Ties go to p1 (old pos 1 < p2's old pos 2); p0 drops to last.
        assert PLAYERS[1].get_turn_order(state) == 0
        assert PLAYERS[2].get_turn_order(state) == 1
        assert PLAYERS[0].get_turn_order(state) == 2
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_reorder_stable_when_already_sorted(self):
        """Cash already in descending order → turn_order unchanged."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        PLAYERS[0].set_cash(state, 50)
        PLAYERS[1].set_cash(state, 30)
        PLAYERS[2].set_cash(state, 10)

        apply_wrap_up_py(state)

        assert PLAYERS[0].get_turn_order(state) == 0
        assert PLAYERS[1].get_turn_order(state) == 1
        assert PLAYERS[2].get_turn_order(state) == 2
        assert_invariants(state)
        assert_token_data_invariants(state)

    @pytest.mark.parametrize("num_players", [3, 4, 5])
    def test_reorder_handles_all_player_counts(self, num_players):
        """Reorder is a permutation of [0, num_players) for any supported count."""
        state = make_auto_phase_state(num_players, PHASE_WRAP_UP)
        # Reverse the default cash ordering so reorder has real work to do.
        for p in range(num_players):
            PLAYERS[p].set_cash(state, 10 * (num_players - p))

        apply_wrap_up_py(state)

        orders = sorted(PLAYERS[p].get_turn_order(state) for p in range(num_players))
        assert orders == list(range(num_players))
        # Highest-cash player (p=0 here) lands at position 0.
        assert PLAYERS[0].get_turn_order(state) == 0
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# PASS COUNTER CLEAR
# =============================================================================

class TestPassCounterCleared:
    """WRAP_UP step 2: consecutive-passes counter always zeroed on entry."""

    def test_consecutive_passes_cleared(self):
        """A non-zero pass counter from INVEST is reset to 0."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        TURN.set_consecutive_passes(state, 3)

        apply_wrap_up_py(state)

        assert TURN.get_consecutive_passes(state) == 0
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# FI PURCHASE LOOP
# =============================================================================

class TestFIPurchaseLoop:
    """WRAP_UP step 3: FI buys cheapest affordable LOC_AUCTION company at face."""

    def test_fi_buys_cheapest_affordable(self):
        """With cash for two of three auction cards, FI buys the two cheapest."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # Seed 42, 3p: initial auction is cids {1, 2, 5} with face {2, 5, 8}.
        auction_before = _auction_ids(state)
        assert auction_before == [1, 2, 5]

        # FI cash = 7: affords cid=1 (face=2, remaining=5) and cid=2 (face=5,
        # remaining=0). Cannot afford cid=5 (face=8).
        FI.set_cash(state, 7)

        apply_wrap_up_py(state)

        fi_owned = _fi_owned_ids(state)
        assert 1 in fi_owned and 2 in fi_owned
        assert 5 not in fi_owned
        # The unaffordable card is still at auction (plus replacements).
        assert 5 in _auction_ids(state)
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_skips_when_nothing_affordable(self):
        """FI cash below the cheapest face → zero purchases, auction unchanged."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        auction_before = _auction_ids(state)
        deck_before = TURN.get_cards_remaining(state)
        FI.set_cash(state, 0)

        apply_wrap_up_py(state)

        # No purchase, no draw.
        assert _fi_owned_ids(state) == []
        assert FI.get_cash(state) == 0
        assert _auction_ids(state) == auction_before
        assert TURN.get_cards_remaining(state) == deck_before
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_buy_draws_replacement(self):
        """Each FI purchase draws a replacement card and decrements the deck."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # FI cash = 2 → buys exactly cid=1 (face=2), then cannot afford more.
        FI.set_cash(state, 2)
        deck_before = TURN.get_cards_remaining(state)

        apply_wrap_up_py(state)

        # One purchase drew one replacement.
        assert _fi_owned_ids(state) == [1]
        assert FI.get_cash(state) == 0
        assert TURN.get_cards_remaining(state) == deck_before - 1
        # Auction row stays at 3: initial {2, 5} + 1 replacement (step 4 flip).
        assert len(_auction_ids(state)) == 3
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_draw_at_empty_deck_is_legal(self):
        """FI purchase with an empty deck succeeds; no replacement drawn."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # Drain the deck. Auction cards stay at auction (not LOC_DECK),
        # and remaining deck cards flip to LOC_EXCLUDED via set_order.
        DECK.set_order(state, [])
        assert TURN.get_cards_remaining(state) == 0

        auction_before = _auction_ids(state)
        # FI cash = 2 → affords cid=1.
        FI.set_cash(state, 2)

        apply_wrap_up_py(state)

        # Purchase succeeded; no replacement drawn (deck is still empty).
        assert 1 in _fi_owned_ids(state)
        assert TURN.get_cards_remaining(state) == 0
        # Remaining auction cards are the initial ones minus cid=1; no
        # replacements since the deck was empty.
        assert _auction_ids(state) == [c for c in auction_before if c != 1]
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_draw_crossing_tier_bumps_coo(self):
        """A draw exposing a different-tier top card bumps CoO exactly once."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # Rebuild deck to bottom→top = [orange (cid=6, 2★), red (cid=0, 1★)].
        # FI buys one auction card → draws cid=0 (1★); new top cid=6 (2★)
        # crosses the tier boundary → CoO += 1.
        DECK.set_order(state, [6, 0])
        coo_before = TURN.get_coo_level(state)
        FI.set_cash(state, 2)  # Affords cid=1 (face=2); then cash=0.

        apply_wrap_up_py(state)

        # Exactly one FI purchase happened, so exactly one draw, so CoO bumps
        # by exactly one (drawn card and new top are different tiers).
        assert _fi_owned_ids(state) == [1]
        assert TURN.get_coo_level(state) == coo_before + 1
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_iterates_until_exhausted(self):
        """With enough cash, FI buys every affordable auction card in one loop."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        auction_before = _auction_ids(state)  # [1, 2, 5], faces [2, 5, 8]
        total_face = sum(COMPANIES[c].get_face_value() for c in auction_before)
        assert total_face == 15
        FI.set_cash(state, total_face)
        deck_before = TURN.get_cards_remaining(state)

        apply_wrap_up_py(state)

        # All three initial auction cards ended up at FI.
        assert set(auction_before).issubset(set(_fi_owned_ids(state)))
        # Cash fully spent.
        assert FI.get_cash(state) == 0
        # Deck drew three replacements.
        assert TURN.get_cards_remaining(state) == deck_before - 3
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# REVEALED → AUCTION PROMOTION
# =============================================================================

class TestRevealedToAuction:
    """WRAP_UP step 4: every LOC_REVEALED company is flipped to LOC_AUCTION."""

    def test_all_revealed_promoted(self):
        """Pre-existing LOC_REVEALED companies land at LOC_AUCTION after wrap-up."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # Start with a known auction row; move one entry to LOC_REVEALED to
        # simulate "revealed during an INVEST auction, not yet flipped".
        COMPANIES[1].mark_revealed(state)
        assert 1 in _revealed_ids(state)
        assert 1 not in _auction_ids(state)

        # Suppress any FI activity — we only want to test the promotion step.
        FI.set_cash(state, 0)

        apply_wrap_up_py(state)

        # Nothing left at LOC_REVEALED; the original reveal is now at auction.
        assert _revealed_ids(state) == []
        assert 1 in _auction_ids(state)
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_replacements_drawn_by_fi_also_promoted(self):
        """Replacements drawn during the FI loop end up at LOC_AUCTION by step 4."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # FI cash = 2 → one purchase → one replacement drawn.
        FI.set_cash(state, 2)
        # Top of deck (seed 42, 3p) is cid=3 — that's the replacement to trace.
        expected_replacement = DECK.peek(state)
        assert expected_replacement == 3
        # Pre-state: replacement card is still LOC_DECK, not LOC_AUCTION.
        assert COMPANIES[expected_replacement].get_location(state) == LOC_DECK

        apply_wrap_up_py(state)

        # The replacement was drawn (→ LOC_REVEALED) and then step 4 promoted
        # it to LOC_AUCTION; no cards remain at LOC_REVEALED.
        assert _revealed_ids(state) == []
        assert COMPANIES[expected_replacement].get_location(state) == LOC_AUCTION
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# TRANSITION
# =============================================================================

class TestTransition:
    """WRAP_UP step 5: hand off to PHASE_ACQUISITION with its setup run."""

    def test_transitions_to_acquisition(self):
        """Phase enum flips to ACQUISITION when an active corp exists."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        # Float a corp so ACQUISITION setup finds an active player and
        # doesn't cascade onward to CLOSING.
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)

        apply_wrap_up_py(state)

        assert_post_auto(state, PHASE_ACQUISITION)

    def test_acquisition_setup_ran(self):
        """setup_acquisition_phase ran: active_player is the corp's president."""
        state = make_auto_phase_state(3, PHASE_WRAP_UP)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)

        apply_wrap_up_py(state)

        # With only corp 0 floated under player 0, the acquisition setup
        # should seat player 0 as the active decision-maker.
        assert TURN.get_active_corp(state) == -1
        assert TURN.get_active_company(state) == -1
        assert TURN.get_active_player(state) == 0
        assert_invariants(state)
        assert_token_data_invariants(state)
