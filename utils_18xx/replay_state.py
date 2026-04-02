"""Shared helpers for replaying 18xx.games actions through our engine."""

from __future__ import annotations

from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.actions import get_valid_action_mask
from core.data import GamePhases, get_company_low_price
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

from .action_parser import map_action, override_deck_and_offering

LOC_PLAYER = CompanyLocation.LOC_PLAYER
LOC_FI = CompanyLocation.LOC_FI
LOC_CORP = CompanyLocation.LOC_CORP


def initialize_replay_state(
    num_players: int,
    deck_order_names: list[str],
    offering_names: list[str],
    *,
    step_mode: bool = False,
    allow_closing_positive_income: bool = False,
    pause_before_acq_transition: bool = False,
) -> GameState:
    """Create a GameState configured for step-by-step 18xx replay."""
    state = GameState(num_players)
    state.initialize_game(seed=42)
    state.step_mode = step_mode
    state.allow_closing_positive_income = allow_closing_positive_income
    state.pause_before_acq_transition = pause_before_acq_transition
    override_deck_and_offering(state, deck_order_names, offering_names)
    return state


def settle_to_player_choice(state: GameState) -> int:
    """Advance deterministic phases until a player decision or game end."""
    result = STATUS_OK
    while DRIVER.is_non_player_phase(state):
        result = DRIVER.advance_phase(state)
        if result == STATUS_GAME_OVER:
            break
    return result


def apply_action_sequence(
    state: GameState,
    action_idx_or_list: int | list[int],
) -> int:
    """Apply one mapped engine action or a short action sequence."""
    action_list = (
        action_idx_or_list
        if isinstance(action_idx_or_list, list)
        else [action_idx_or_list]
    )

    result = STATUS_OK
    for idx, action_idx in enumerate(action_list):
        if idx > 0 and TURN.get_phase(state) != GamePhases.PHASE_PAR:
            break
        result = DRIVER.apply_action(state, action_idx)
        if result != STATUS_OK:
            return result
        if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return result

    return result


def align_to_action(state: GameState, action: dict, layout) -> bool:
    """Apply omitted single-option actions until *action* becomes mappable.

    Returns True when one or more forced engine actions were applied while
    catching up to the next explicit 18xx action.
    """
    advanced = False

    while True:
        settle_to_player_choice(state)
        if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return advanced

        try:
            engine_action = map_action(state, action, TURN.get_phase(state), layout)
        except (ValueError, KeyError, IndexError):
            engine_action = None
        if engine_action is not None:
            return advanced

        forced_action = _get_single_legal_action(state)
        if forced_action is None:
            return advanced

        DRIVER.apply_action(state, forced_action)
        advanced = True


def _get_single_legal_action(state: GameState) -> int | None:
    """Return the lone legal action index, or None if not forced."""
    legal_actions = [
        idx
        for idx, value in enumerate(get_valid_action_mask(state))
        if value > 0.5
    ]
    if len(legal_actions) == 1:
        return legal_actions[0]
    return None


def is_representable_acquisition_offer(
    state: GameState,
    buyer_corp_id: int,
    company_id: int,
) -> bool:
    """Return whether the current engine can represent the acquisition offer.

    Our engine only exposes same-president non-FI offers. FI offers are always
    representable.
    """
    buyer_president = CORPS[buyer_corp_id].get_president_id(state)
    owner_id = COMPANIES[company_id].get_owner_id(state)
    owner_loc = COMPANIES[company_id].get_location(state)

    if owner_loc == LOC_FI:
        return True
    if owner_loc == LOC_PLAYER:
        return buyer_president >= 0 and owner_id == buyer_president
    if owner_loc == LOC_CORP:
        seller_president = CORPS[owner_id].get_president_id(state)
        return seller_president >= 0 and seller_president == buyer_president
    return False


def apply_external_acquisition_transfer(
    state: GameState,
    buyer_corp_id: int,
    company_id: int,
    price: int,
) -> bool:
    """Apply an accepted acquisition that is outside our ACQ action space.

    This stages the company through the normal acquisition zone so the
    subsequent ACQ -> CLO transition merges it through the engine's existing
    phase logic.
    """
    seller_id = COMPANIES[company_id].get_owner_id(state)
    seller_loc = COMPANIES[company_id].get_location(state)

    if seller_loc not in (LOC_PLAYER, LOC_FI, LOC_CORP):
        return False

    CORPS[buyer_corp_id].add_cash(state, -price)

    if seller_loc == LOC_CORP:
        current = CORPS[seller_id].get_acquisition_proceeds(state)
        CORPS[seller_id].set_acquisition_proceeds(state, current + price)
    elif seller_loc == LOC_PLAYER:
        PLAYERS[seller_id].add_cash(state, price)
    else:
        FI.add_cash(state, price)

    COMPANIES[company_id].transfer_to_corp_acquisition(state, buyer_corp_id)
    return True


def replay_acquisition_offer(
    state: GameState,
    layout,
    buyer_corp_id: int,
    company_id: int,
    price: int,
    *,
    accept: bool,
    max_iterations: int = 200,
) -> bool:
    """Walk the ACQ offer buffer until the target offer is reached."""
    for _ in range(max_iterations):
        if TURN.get_phase(state) != GamePhases.PHASE_ACQUISITION:
            return False

        cur_corp = TURN.get_acq_active_corp(state)
        cur_company = TURN.get_acq_target_company(state)
        if cur_corp == buyer_corp_id and cur_company == company_id:
            if accept:
                if TURN.is_acq_fi_offer(state):
                    result = DRIVER.apply_action(state, layout.acq_fi_buy)
                else:
                    low_price = get_company_low_price(company_id)
                    result = DRIVER.apply_action(
                        state,
                        layout.acq_price_base + (price - low_price),
                    )
            else:
                result = DRIVER.apply_action(state, layout.acq_pass)

            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid ACQ replay action for corp={buyer_corp_id}, "
                    f"company={company_id}, accept={accept}"
                )
            return True

        result = DRIVER.apply_action(state, layout.acq_pass)
        if result == STATUS_INVALID:
            raise RuntimeError(
                f"Invalid ACQ pass while searching for corp={buyer_corp_id}, "
                f"company={company_id}"
            )
        if result in (STATUS_GAME_OVER, STATUS_PAUSED):
            return False

    raise RuntimeError("Exceeded ACQ replay iteration limit")


def replay_closing_offer(
    state: GameState,
    layout,
    company_id: int,
    *,
    close: bool,
    max_iterations: int = 200,
) -> bool:
    """Walk the CLO offer buffer until the target company is reached."""
    for _ in range(max_iterations):
        if TURN.get_phase(state) != GamePhases.PHASE_CLOSING:
            return False

        cur_company = TURN.get_closing_company(state)
        if cur_company == company_id:
            action_idx = layout.close_action if close else layout.close_pass
            result = DRIVER.apply_action(state, action_idx)
            if result == STATUS_INVALID:
                raise RuntimeError(
                    f"Invalid CLO replay action for company={company_id}, "
                    f"close={close}"
                )
            return True

        result = DRIVER.apply_action(state, layout.close_pass)
        if result == STATUS_INVALID:
            raise RuntimeError(
                f"Invalid CLO pass while searching for company={company_id}"
            )
        if result == STATUS_GAME_OVER:
            return False

    raise RuntimeError("Exceeded CLO replay iteration limit")


def drain_offer_phases(state: GameState, layout, max_iterations: int = 500) -> None:
    """Pass through any remaining ACQ/CLO offers after replay."""
    for _ in range(max_iterations):
        settle_to_player_choice(state)
        phase = TURN.get_phase(state)

        if phase == GamePhases.PHASE_ACQUISITION:
            result = DRIVER.apply_action(state, layout.acq_pass)
        elif phase == GamePhases.PHASE_CLOSING:
            result = DRIVER.apply_action(state, layout.close_pass)
        else:
            return

        if result == STATUS_INVALID:
            raise RuntimeError(f"Invalid replay drain action in phase={phase}")

    raise RuntimeError("Exceeded ACQ/CLO drain iteration limit")
