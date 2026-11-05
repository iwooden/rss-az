"""1889 self-play diagnostics (1889/model-design.md §6.6): the collector behind
``Game.new_selfplay_stats``.
"""

from __future__ import annotations

import numpy as np

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.entities.corp import CORPS
from g1889.entities.phase import PHASE
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState


def _trains_bought(state: GameState, train_type: int) -> int:
    """New trains of a type the corps have bought: the copies gone from the
    depot, where a rank's last copy sells before the rank that rusts it. D's
    depot is unlimited and D never rusts, so its count is the Ds in play. A
    purchase from the pool or another corp moves a train already counted."""
    if D.TRAIN_COUNT[train_type] < 0:
        return TRAINS.pool(state, train_type) + sum(
            TRAINS.corp_trains(state, c, train_type) for c in range(D.NUM_CORPS)
        )
    return D.TRAIN_COUNT[train_type] - TRAINS.depot(state, train_type)


class G1889SelfPlayStats:
    """One self-play game's end, length in ORs, track passes, floats, and trains.

    ``finish`` reports per-game values as means over games (denominator 1), and
    the track passes as a rate over the group's TRACK decisions, those with a
    choice: the driver plays a forced pass without a search.
    """

    def __init__(self, num_players: int) -> None:
        self.track_decisions = 0
        self.track_passes = 0
        # OR sets' lengths by SR number. Forced decisions pass unseen, and an OR
        # can hold none but them, so ORs are counted from these, not as seen.
        self.or_set_lengths: dict[int, int] = {}

    def observe(
        self,
        state: GameState,
        legal_actions: np.ndarray,
        chosen_idx: int,
        priors: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        # or_count holds a set's length from its SR's end to the next SR's.
        game_round = TURN.round(state)
        if game_round == K.ROUND_OPERATING:
            self.or_set_lengths[TURN.turn(state)] = TURN.or_count(state)
        elif game_round == K.ROUND_STOCK:
            self.or_set_lengths[TURN.turn(state) - 1] = TURN.or_count(state)
        if get_decision_phase_py(state) == K.DPHASE_TRACK:
            self.track_decisions += 1
            self.track_passes += int(legal_actions[chosen_idx] == K.TRACK_PASS)

    def finish(self, state: GameState) -> dict[str, tuple[float, float]]:
        # The game ends in an OR, of the set its SR number names. A set with no
        # search in it or the next SR has at least one OR.
        ors_played = TURN.or_num(state) + sum(
            self.or_set_lengths.get(sr, 1) for sr in range(1, TURN.turn(state))
        )
        end_reason = TURN.end_reason(state)
        last_phase = PHASE.current(state)
        stats: dict[str, tuple[float, float]] = {
            "end_bank": (float(end_reason == K.END_REASON_BANK), 1),
            "end_bankruptcy": (float(end_reason == K.END_REASON_BANKRUPT), 1),
            "ors_played": (float(ors_played), 1),
            "track_pass_share": (float(self.track_passes), float(self.track_decisions)),
            "corps_floated": (float(sum(CORPS[c].floated(state) for c in range(D.NUM_CORPS))), 1),
        }
        for phase, name in enumerate(D.PHASE_NAMES):
            stats[f"last_phase_{name}"] = (float(phase == last_phase), 1)
        for train_type, name in enumerate(D.TRAIN_NAMES):
            stats[f"trains_bought_{name}"] = (float(_trains_bought(state, train_type)), 1)
        return stats
