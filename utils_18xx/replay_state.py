"""Shared helpers for replaying 18xx.games actions through the current engine."""

from __future__ import annotations

from core.actions import (
    ACTION_ACQ_OFFER_ACCEPT_PY as ACTION_ACQ_OFFER_ACCEPT,
    ACTION_ACQ_PRICE_PY as ACTION_ACQ_PRICE,
    ACTION_CLOSE_PY as ACTION_CLOSE,
    ACTION_PASS_PY as ACTION_PASS,
)
from core.data import CorpIndices, GamePhases
from core.driver import (
    DRIVER,
    STATUS_GAME_OVER_PY as STATUS_GAME_OVER,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_OK_PY as STATUS_OK,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.state import GameState
from entities.company import COMPANIES, CompanyLocation
from entities.corp import CORPS
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN

from .action_parser import (
    PHASE_ACQ_OFFER,
    PHASE_ACQ_SELECT_COMPANY,
    PHASE_ACQ_SELECT_CORP,
    PHASE_ACQ_SELECT_PRICE,
    PHASE_CLOSING,
    find_legal_action,
    get_legal_actions,
    map_action,
    override_deck_and_offering,
)

LOC_PLAYER = CompanyLocation.LOC_PLAYER
LOC_FI = CompanyLocation.LOC_FI
LOC_CORP = CompanyLocation.LOC_CORP
ACQ_PHASES = (
    PHASE_ACQ_SELECT_CORP,
    PHASE_ACQ_SELECT_COMPANY,
    PHASE_ACQ_SELECT_PRICE,
    PHASE_ACQ_OFFER,
)


def initialize_replay_state(
    num_players: int,
    deck_order_names: list[str],
    offering_names: list[str],
    *,
    max_players: int = 0,
    cost_level: int | None = None,
    step_mode: bool = False,
    pause_before_acq_transition: bool = False,
    pause_before_closing_transition: bool = False,
) -> GameState:
    """Create a replay state on the live ACQ/CLO surface.

    ``pause_before_*`` arguments remain accepted for compatibility with older
    callers, but current replay code does not currently enable
    ``state.step_mode`` by default. The harness instead relies on normal driver
    auto-chaining plus explicit calls to ``settle_to_player_choice(...)`` rather
    than custom pause flags on ``GameState``.
    """
    del pause_before_acq_transition, pause_before_closing_transition

    if max_players:
        state = GameState(
            num_players,
            max_players=max_players,
            acq_same_president=False,
        )
        state.initialize_game(num_players, seed=42, max_players=max_players)
    else:
        state = GameState(num_players, acq_same_president=False)
        state.initialize_game(num_players, seed=42)
    state.allow_positive_income_closing = True
    state.step_mode = step_mode
    override_deck_and_offering(state, deck_order_names, offering_names)
    if cost_level is not None:
        TURN.set_coo_level(state, cost_level)
    return state


def settle_to_player_choice(state: GameState) -> int:
    """Advance deterministic work until a real decision, pause, or game end."""
    result = STATUS_OK
    while DRIVER.is_non_player_phase(state):
        result = DRIVER.advance_phase(state)
        if result != STATUS_OK:
            return result
    return result


def apply_action_sequence(
    state: GameState,
    action_idx_or_list: int | list[int],
) -> int:
    """Apply one engine action or a short explicit sequence."""
    action_list = action_idx_or_list if isinstance(action_idx_or_list, list) else [action_idx_or_list]
    result = STATUS_OK
    for action_idx in action_list:
        result = DRIVER.apply_action(state, action_idx)
        if result != STATUS_OK:
            return result
        if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return result
    return result


def align_to_action(state: GameState, action: dict, layout) -> bool:
    """Apply omitted forced actions until ``action`` becomes mappable."""
    del layout
    advanced = False

    while True:
        settle_to_player_choice(state)
        if TURN.get_phase(state) == GamePhases.PHASE_GAME_OVER:
            return advanced

        try:
            engine_action = map_action(state, action, TURN.get_phase(state), None)
        except (ValueError, KeyError, IndexError):
            engine_action = None

        if engine_action is not None:
            return advanced

        forced_action = _get_single_legal_action(state)
        if forced_action is None:
            return advanced

        result = DRIVER.apply_action(state, forced_action)
        if result == STATUS_INVALID:
            raise RuntimeError(f"Invalid forced replay action {forced_action}")
        advanced = True
        if result in (STATUS_GAME_OVER, STATUS_PAUSED):
            return advanced


def _get_single_legal_action(state: GameState) -> int | None:
    actions = get_legal_actions(state)
    if len(actions) == 1:
        return actions[0][0]
    return None


def is_representable_acquisition_offer(
    state: GameState,
    buyer_corp_id: int,
    company_id: int,
) -> bool:
    """Return whether the current live action surface can represent the offer."""
    del buyer_corp_id
    return COMPANIES[company_id].get_location(state) in (LOC_FI, LOC_PLAYER, LOC_CORP)


def apply_external_acquisition_transfer(
    state: GameState,
    buyer_corp_id: int,
    company_id: int,
    price: int,
) -> bool:
    """Fallback manual transfer for offers that cannot be replayed directly."""
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


def is_closing_transition_pending(state: GameState) -> bool:
    del state
    return False


def apply_external_close(
    state: GameState,
    company_id: int,
) -> bool:
    """Fallback manual close for cases not representable via legal actions."""
    if COMPANIES[company_id].is_removed(state):
        return False

    owner_loc = COMPANIES[company_id].get_location(state)
    owner_id = COMPANIES[company_id].get_owner_id(state)

    if owner_loc == LOC_PLAYER:
        COMPANIES[company_id].remove_from_game(state)
        return True

    if owner_loc == LOC_CORP:
        if owner_id < 0 or CORPS[owner_id].count_companies(state) < 2:
            return False
        if owner_id == CorpIndices.CORP_JS:
            CORPS[owner_id].add_cash(state, COMPANIES[company_id].get_base_income() * 2)
        COMPANIES[company_id].remove_from_game(state)
        return True

    if owner_loc == LOC_FI:
        COMPANIES[company_id].remove_from_game(state)
        return True

    return False


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
    """Replay a direct ACQ/ACQ_OFFER acquisition on the live engine surface."""
    del layout

    offer_action = {
        "type": "offer",
        "corporation": CORPS[buyer_corp_id].name,
        "company": COMPANIES[company_id].name,
        "price": price,
    }

    for _ in range(max_iterations):
        if COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id):
            return True
        if COMPANIES[company_id].is_in_corp_acquisition(state, buyer_corp_id):
            return True

        settle_to_player_choice(state)
        phase = TURN.get_phase(state)

        if phase in (PHASE_ACQ_SELECT_CORP, PHASE_ACQ_SELECT_COMPANY, PHASE_ACQ_SELECT_PRICE):
            try:
                action_id = map_action(state, offer_action, phase, None)
            except (ValueError, KeyError, IndexError):
                try:
                    action_id = find_legal_action(state, action_type=ACTION_PASS)
                except ValueError:
                    return False
                result = DRIVER.apply_action(state, action_id)
                if result == STATUS_INVALID:
                    return False
                continue

            result = DRIVER.apply_action(state, action_id)
            if result == STATUS_INVALID:
                return False
            if COMPANIES[company_id].is_owned_by_corp(state, buyer_corp_id):
                return True
            if COMPANIES[company_id].is_in_corp_acquisition(state, buyer_corp_id):
                return True
            if TURN.get_phase(state) not in ACQ_PHASES:
                return True
            continue

        if phase == PHASE_ACQ_OFFER:
            try:
                action_id = find_legal_action(
                    state,
                    action_type=ACTION_ACQ_OFFER_ACCEPT if accept else ACTION_PASS,
                )
            except ValueError:
                return False
            result = DRIVER.apply_action(state, action_id)
            return result != STATUS_INVALID

        return False

    raise RuntimeError("Exceeded ACQ replay iteration limit")


def drain_offer_phases(state: GameState, layout, max_iterations: int = 500) -> None:
    """Pass through any remaining ACQ/CLO offers after replay."""
    del layout

    for _ in range(max_iterations):
        settle_to_player_choice(state)
        phase = TURN.get_phase(state)

        if phase not in (*ACQ_PHASES, PHASE_CLOSING):
            return

        try:
            pass_id = find_legal_action(state, action_type=ACTION_PASS)
        except ValueError:
            forced_action = _get_single_legal_action(state)
            if forced_action is None:
                return
            result = DRIVER.apply_action(state, forced_action)
        else:
            result = DRIVER.apply_action(state, pass_id)

        if result == STATUS_INVALID:
            raise RuntimeError(f"Invalid replay drain action in phase={phase}")

    raise RuntimeError("Exceeded ACQ/CLO drain iteration limit")
