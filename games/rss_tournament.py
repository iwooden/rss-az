"""RSS tournament rules: checkpoints trained under different engine rules in one game.

The game uses the most permissive checkpoint rules (v3 behavior and
cross-president offers if any checkpoint was trained with them). Each
checkpoint searches under its own training rules within the game's: a model
trained with same-president acquisitions searches only those, and a legacy
(v2) model sees legacy trade-history lifetime. A model trained without
cross-president offers answers incoming ones with equal accept/reject priors,
as in live play (``CrossPresidentOfferPriorEvaluator``). ``EngineRules`` is an
RSS tournament's ``train.tournament.GameRules``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from core.data import GamePhases
from core.state import GameState
from entities.player import PLAYERS
from entities.turn import TURN
from games.rss_live import CrossPresidentOfferPriorEvaluator

if TYPE_CHECKING:
    from train.config import TrainingConfig


@dataclass(frozen=True)
class EngineRules:
    """Engine rule flags (runtime-only; not stored in the raw state array)."""
    v3_behavior: bool
    acq_same_president: bool
    # Policy/search only; game legality never depends on it.
    max_acq_price_actions: int = 0

    @classmethod
    def from_config(cls, config: TrainingConfig) -> EngineRules:
        return cls(
            config.v3_behavior, config.acq_same_president, config.max_acq_price_actions,
        )

    def describe(self) -> str:
        mode = "v3" if self.v3_behavior else "legacy"
        acq = ("same-president acquisitions" if self.acq_same_president
               else "cross-president offers")
        return f"{mode} rules, {acq}"

    def game_config(self, config: TrainingConfig) -> TrainingConfig:
        """``config`` with these rules: its adapter's ``new_game`` plays under them."""
        return dataclasses.replace(
            config, v3_behavior=self.v3_behavior,
            acq_same_president=self.acq_same_president,
            max_acq_price_actions=self.max_acq_price_actions,
        )

    def seat(
        self, config: TrainingConfig, evaluator: Any, num_players: int, max_players: int,
    ) -> tuple[Any, Callable[[GameState], GameState], str]:
        """How a checkpoint trained under ``config`` plays a game under these
        rules: its evaluator, the game's state as it searches it, and a
        description."""
        trained = EngineRules.from_config(config)
        rules = search_rules(self, trained)
        offer_priors = ""
        if trained.acq_same_president and not self.acq_same_president:
            evaluator = CrossPresidentOfferPriorEvaluator(
                evaluator, num_players=num_players, max_players=max_players,
            )
            offer_priors = "; equal cross-president offer priors"

        def view(state: GameState) -> GameState:
            return search_state(state, rules, num_players, max_players)

        price_cap = rules.max_acq_price_actions or "all"
        return evaluator, view, (
            f"searches {rules.describe()}{offer_priors}; acq prices {price_cap}"
        )


def resolve_game_rules(
    configs: list[TrainingConfig],
    v3_behavior: bool | None = None,
    acq_same_president: bool | None = None,
) -> EngineRules:
    """Game rules: explicit overrides, else the most permissive checkpoint's."""
    trained = [EngineRules.from_config(config) for config in configs]
    return EngineRules(
        v3_behavior=(any(r.v3_behavior for r in trained)
                     if v3_behavior is None else v3_behavior),
        acq_same_president=(all(r.acq_same_president for r in trained)
                            if acq_same_president is None else acq_same_president),
    )


def search_rules(game: EngineRules, trained: EngineRules) -> EngineRules:
    """Rules a checkpoint searches under: its training rules, within the game's.

    Searching same-president only restricts the game's legal actions. Legacy
    behavior drops v3's cross-president price floors, so it is used only with
    same-president search; there it differs from v3 only in trade-history
    lifetime (the rejection floor never applies to a player that makes no
    cross-president offers). Each checkpoint keeps its own ACQ price cap.
    """
    same_president = game.acq_same_president or trained.acq_same_president
    return EngineRules(
        v3_behavior=game.v3_behavior and (trained.v3_behavior or not same_president),
        acq_same_president=same_president,
        max_acq_price_actions=trained.max_acq_price_actions,
    )


# Phases where legacy rules keep trade history; they clear it when INVEST ends.
_INVEST_ROUND_PHASES = (int(GamePhases.PHASE_INVEST), int(GamePhases.PHASE_BID))


def search_state(
    state: GameState, rules: EngineRules, num_players: int, max_players: int,
) -> GameState:
    """Return ``state`` as seen under ``rules``, copying when they differ."""
    if (rules.v3_behavior == state.v3_behavior
            and rules.acq_same_president == state.acq_same_president
            and rules.max_acq_price_actions == state.max_acq_price_actions):
        return state
    view = GameState.from_array(
        state._array, num_players, max_players=max_players,
        v3_behavior=rules.v3_behavior,
    )
    view.acq_same_president = rules.acq_same_president
    view.max_acq_price_actions = rules.max_acq_price_actions
    if (state.v3_behavior and not rules.v3_behavior
            and TURN.get_phase(view) not in _INVEST_ROUND_PHASES):
        # V3 keeps trade history until the next INVEST.
        for player_id in range(num_players):
            PLAYERS[player_id].clear_roundtrip_tracking(view)
    return view
