"""Scripted auctions replayed against Ruby: the paths the fixtures never take.

The fixtures' auctions are short: single-bid chains and one two-way closed
auction. These scripts run through Ruby's extractor as game JSONs and replay
like fixtures, compared after every action. They cover rounds of passes with
A unsold (the discount down to the free A, which can end the auction),
payouts once A is sold, passes that persist across other players' turns,
closed auctions among three bidders, a raise by a player who had passed,
purchases of the topmost above its price, off-grid amounts, a bid of all of
a player's cash, and the forced pass that follows it.
"""

from __future__ import annotations

import pytest

from tests.g1889.replay.extracts import load_game_extract
from tests.g1889.replay.test_differential_replay import replay_extract

PASS = None


def _game(label: str, num_players: int, moves: list[tuple]) -> dict:
    """A 1889 game JSON: ``moves`` are ``(seat, PASS)`` or ``(seat, private, price)``."""
    ids = [101 + i for i in range(num_players)]
    actions = []
    for i, (seat, *bid) in enumerate(moves, 1):
        kind = 'pass' if bid == [PASS] else 'bid'
        action = {'type': kind, 'entity': ids[seat], 'entity_type': 'player', 'id': i}
        if kind == 'bid':
            action.update(company=bid[0], price=bid[1])
        actions.append(action)
    return {'id': f'scripted-{label}', 'title': '1889', 'players': [{'id': p, 'name': f'Player {p}'} for p in ids],
            'settings': {'seed': 1, 'optional_rules': []}, 'actions': actions}


def _passes(*seats) -> list[tuple]:
    return [(seat, PASS) for seat in seats]


SCRIPTS = {
    # Four rounds of passes take A to ¥0: seat 2 gets it free and its chain sells
    # MF and ER. Then rounds of passes pay out, including one after a new bid.
    'discount_to_free_a': (3, [
        (0, 'MF', 35), (1, 'ER', 45), *_passes(2, 0, 1),
        *_passes(2, 0, 1), *_passes(2, 0, 1), *_passes(2, 0, 1),
        (0, 'SMR', 50), *_passes(1, 2, 0),
        (1, 'SIR', 85), *_passes(2, 0, 1),
        (2, 'DR', 60),
    ]),
    # The free A's chain sells every private: the auction ends without a purchase,
    # and priority goes to the player after A's taker.
    'free_a_ends_the_auction': (2, [
        (0, 'MF', 35), (1, 'ER', 45), (0, 'SMR', 55), (1, 'DR', 65),
        *_passes(0, 1), *_passes(0, 1), *_passes(0, 1), *_passes(0, 1),
    ]),
    # A three-way closed auction for ER, an off-grid raise, and a second closed
    # auction chained after it; then the topmost bought above its price.
    'three_way_closed_auction': (4, [
        (0, 'ER', 45), (1, 'ER', 50), (2, 'ER', 55), (3, 'MF', 35), (0, 'SMR', 55), (1, 'SMR', 60),
        (2, 'TR', 20),
        (0, 'ER', 60), (1, PASS), (2, 'ER', 65), (0, 'ER', 72), (2, PASS),
        (0, PASS),
        (3, 'DR', 63), (0, PASS), (1, 'UTF', 160), (2, 'SIR', 80),
    ]),
    # A raise clears the raiser's pass, a drop does not, and a pass persists
    # across other players' purchases and closed auctions.
    'passes_persist': (3, [
        (0, 'MF', 35), (1, 'MF', 40), (2, PASS), (0, PASS),
        (1, 'TR', 20), (0, 'MF', 45), (1, PASS),
        *_passes(2, 0, 1),
        (2, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
    ]),
    # Seat 0 commits all of its cash, so its later turns are forced passes,
    # one of which completes a round of passes.
    'all_cash_committed': (5, [
        (0, 'UTF', 155), *_passes(1, 2, 3, 4),
        (0, 'SIR', 85), *_passes(1, 2, 3, 4),
        (0, 'DR', 150), *_passes(1, 2, 3, 4),
        (0, PASS),
        (1, 'TR', 15), (2, 'MF', 30), (3, 'ER', 40), (4, PASS), (0, PASS), (1, 'SMR', 50),
    ]),
    # Six players: a three-way closed auction for UTF closes the auction.
    'six_players': (6, [
        (0, 'UTF', 155), (1, 'UTF', 160), (2, 'UTF', 165),
        (3, 'TR', 20), (4, 'MF', 30), (5, 'ER', 40), (0, 'SMR', 50), (1, 'DR', 60), (2, 'SIR', 80),
        (0, 'UTF', 170), (1, PASS), (2, 'UTF', 175), (0, PASS),
    ]),
}


@pytest.mark.parametrize('label', SCRIPTS)
def test_scripted_auction_matches_ruby(label):
    num_players, moves = SCRIPTS[label]
    extract = load_game_extract(label, _game(label, num_players, moves))
    progress = replay_extract(label, extract)
    assert progress.stop is None and progress.matched == len(extract['records']) - 1, str(progress)
    assert extract['records'][-1]['state']['game']['round'] == 'SR', 'the script should finish the auction'
