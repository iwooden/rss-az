"""RSS cross-president offer priors for live play and tournaments
(1889/pipeline-refactor.md §4.2).

18xx.games players make acquisition offers that self-play never produces.
Accept and reject stay legal, so ``CrossPresidentOfferPriorEvaluator`` replaces
the model's untrained priors at those offers with 50/50 and search decides them
from values alone. ``utils_18xx/live.py`` uses it for live play, and
``train/tournament.py`` for checkpoints playing under wider rules than they
trained with. Search and self-play never do.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_PASS_PY as ACTION_PASS,
    decode_action_py,
    enumerate_legal_actions_py,
)
from core.data import MAX_ACTION_SIZE, DecisionPhase, GameConstants, GamePhases
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.player import PLAYERS
from entities.turn import TURN
from games.rss import POLICY_LAYOUT


def is_cross_president_acq_offer_state(state: Any) -> bool:
    """Return whether ACQ_OFFER is deciding a non-FI cross-president offer."""
    if TURN.get_phase(state) != GamePhases.PHASE_ACQ_OFFER:
        return False

    company_id = TURN.get_active_company(state)
    if company_id < 0:
        return False

    location = COMPANIES[company_id].get_location(state)
    return location in (
        int(CompanyLocation.LOC_PLAYER),
        int(CompanyLocation.LOC_CORP),
    )


def is_acq_offer_outside_v3_limits(state: Any) -> bool:
    """Return whether ACQ_OFFER is deciding a cross-president offer v3 forbids.

    V3 players offer cross-president only at the company's high price, and
    only until one such offer is rejected in the acquisition phase. V3
    checkpoints never answered other offers; 18xx.games players can make them.
    """
    if not is_cross_president_acq_offer_state(state):
        return False

    company_id = TURN.get_active_company(state)
    proposer = CORPS[TURN.get_active_corp(state)].get_president_id(state)
    return (
        TURN.get_acq_offer_price(state) < COMPANIES[company_id].get_high_price()
        or PLAYERS[proposer].get_acq_rejections(state)
        >= int(GameConstants.ACQ_REJECTION_CAP)
    )


def _acq_offer_response_actions(state: Any) -> tuple[int, int]:
    """Return the legal ACQ_OFFER ``(pass, accept)`` action IDs."""
    buf = np.zeros(MAX_ACTION_SIZE, dtype=np.uint16)
    count = int(enumerate_legal_actions_py(state, buf))
    phase_id = int(DecisionPhase.DPHASE_ACQ_OFFER)
    by_type = {
        decode_action_py(phase_id, int(action_id)).action_type: int(action_id)
        for action_id in buf[:count]
    }
    return by_type[ACTION_PASS], by_type[ACTION_ACQ_OFFER_ACCEPT]


def _equalize_sparse_acq_offer_priors(
    state: Any,
    priors: np.ndarray,
    action_ids: np.ndarray,
) -> np.ndarray:
    response_mask = np.isin(action_ids, _acq_offer_response_actions(state))
    if int(response_mask.sum()) != 2:
        return priors

    adjusted = np.zeros_like(priors, dtype=np.float32)
    adjusted[response_mask] = 0.5
    return adjusted


def _equalize_dense_acq_offer_priors(
    state: Any,
    priors: np.ndarray,
    row_idx: int,
    action_lut_np: np.ndarray,
) -> None:
    slots = action_lut_np[
        int(DecisionPhase.DPHASE_ACQ_OFFER),
        np.array(_acq_offer_response_actions(state), dtype=np.intp),
    ]
    priors[row_idx] = 0.0
    priors[row_idx, slots] = 0.5


class CrossPresidentOfferPriorEvaluator:
    """Evaluator adapter that neutralizes cross-president offer priors.

    By default it covers every cross-president offer, for models trained with
    same-president acquisitions only. Models trained under v3's offer limits
    pass ``applies_to=is_acq_offer_outside_v3_limits`` to cover the offers
    those limits forbid. The priors for these responses are untrained, so
    search decides them from values alone.
    """

    def __init__(
        self,
        base: Any,
        *,
        num_players: int,
        max_players: int,
        applies_to: Callable[[Any], bool] = is_cross_president_acq_offer_state,
    ) -> None:
        self._base = base
        self._num_players = num_players
        self._max_players = max_players
        self._applies_to = applies_to
        self._action_lut_np = POLICY_LAYOUT.build_action_lut().numpy()
        self._scratch_state: GameState | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        priors, values, action_ids, n_legal, phase_id = self._base.evaluate(state)
        if self._applies_to(state):
            priors = _equalize_sparse_acq_offer_priors(
                state,
                priors,
                action_ids[:n_legal],
            )
        return priors, values, action_ids, n_legal, phase_id

    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        priors, values = self._base.evaluate_leaves(state_arrays, legal_mask)
        if not state_arrays:
            return priors, values

        scratch = self._scratch_state
        if scratch is None:
            scratch = GameState.from_buffer(
                state_arrays[0],
                self._num_players,
                max_players=self._max_players,
            )
            self._scratch_state = scratch

        for row_idx, state_array in enumerate(state_arrays):
            scratch.rebind(
                state_array,
                self._num_players,
                max_players=self._max_players,
            )
            if self._applies_to(scratch):
                _equalize_dense_acq_offer_priors(
                    scratch,
                    priors,
                    row_idx,
                    self._action_lut_np,
                )

        return priors, values

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        return self._base.evaluate_terminal(state)
