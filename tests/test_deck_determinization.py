import numpy as np

from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.deck import DECK
from entities.turn import TURN


LOC_AUCTION = int(CompanyLocation.LOC_AUCTION)
LOC_DECK = int(CompanyLocation.LOC_DECK)
LOC_EXCLUDED = int(CompanyLocation.LOC_EXCLUDED)


def _partial_red_state() -> GameState:
    state = GameState(3)
    state.initialize_game(3, seed=42)

    for cid in range(36):
        if COMPANIES[cid].get_location(state) in (
            int(CompanyLocation.LOC_AUCTION),
            int(CompanyLocation.LOC_REVEALED),
        ):
            COMPANIES[cid].exclude_from_game(state)

    # Bottom-to-top: complete future groups plus one non-MHE red card.
    DECK.set_order(
        state,
        [
            29, 30, 31, 35,
            22, 23, 24, 28,
            14, 15, 16, 21,
            6, 7, 8, 13,
            3,
        ],
    )
    for cid in (0, 1, 2):
        COMPANIES[cid].move_to_auction(state)
    TURN.set_coo_level(state, 1)
    return state


def test_determinize_remaining_forces_unseen_last_company_in_partial_group():
    state = _partial_red_state()
    assert DECK.get_order(state)[-1] == 3
    assert COMPANIES[5].get_location(state) == LOC_EXCLUDED

    det = DECK.determinize_remaining(state, np.random.default_rng(7))
    det_order = DECK.get_order(det)

    assert len(det_order) == DECK.get_remaining_count(state)
    assert det_order[-1] == 5  # MHE is forced as the last remaining red.
    assert COMPANIES[5].get_location(det) == LOC_DECK
    assert COMPANIES[3].get_location(det) == LOC_EXCLUDED

    for cid in (0, 1, 2):
        assert COMPANIES[cid].get_location(det) == LOC_AUCTION
        assert COMPANIES[cid].get_location(state) == LOC_AUCTION
    assert DECK.get_order(state)[-1] == 3


def test_determinize_remaining_rebuilds_unrevealed_future_groups_from_setup():
    state = _partial_red_state()

    det = DECK.determinize_remaining(state, np.random.default_rng(11))
    det_order = DECK.get_order(det)

    blue_group = det_order[:4]
    assert len(blue_group) == 4
    assert set(blue_group).issubset(set(range(29, 36)))
    assert 35 in blue_group  # CDG is always included for an unrevealed group.
