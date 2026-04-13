"""ACQ_OFFER phase handler.

Unified "accept/decline this acquisition?" phase. The question is always:
"Should active_corp acquire active_company at acq_offer_price?" — only
who decides (active_player) varies between FI preemption and cross-
president offers.

No setup function — ACQ_OFFER is entered mid-action from
``apply_acquisition_action`` via ``_enter_acq_offer``.
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants,
    GamePhases,
    CorpIndices,
    COMPANY_FACE_VALUE,
    COMPANY_HIGH_PRICE,
    COMPANY_LOW_PRICE,
)
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_ACQ_OFFER_ACCEPT,
)
from entities.company cimport (
    LOC_FI,
    LOC_CORP,
    LOC_PLAYER,
)

from phases.acquisition cimport (
    _clear_acquisition_context,
    _execute_fi_buy,
    _find_first_preemptor,
    _find_first_active_player,
)

from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module
from entities import fi as fi_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _return_to_acquisition(GameState state) noexcept:
    """Return to ACQUISITION phase, resuming with first non-passed player."""
    _clear_acquisition_context(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQUISITION)

    cdef int pid = _find_first_active_player(state)
    assert pid >= 0, "_return_to_acquisition: no eligible player"
    turn_module.TURN.set_active_player(state, pid)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_acq_offer_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch an ACQ_OFFER action (PASS or ACCEPT).

    Distinguishes FI preemption (company is LOC_FI) from cross-president
    offers (company is LOC_CORP or LOC_PLAYER) by checking the target
    company's current location.

    FI preemption ACCEPT: preempting corp buys from FI, return to ACQUISITION.
    FI preemption PASS: set passed flag, offer to next preemptor. If none
        remain, original corp buys. Return to ACQUISITION.
    Cross-president ACCEPT: execute transfer at acq_offer_price, return.
    Cross-president PASS: cancel acquisition, return.
    """
    cdef int active_corp = turn_module.TURN.get_active_corp(state)
    cdef int company_id = turn_module.TURN.get_active_company(state)
    cdef int original_corp = turn_module.TURN.get_acq_offer_corp(state)
    cdef int price = turn_module.TURN.get_acq_offer_price(state)

    assert active_corp >= 0, f"acq_offer: no active corp"
    assert company_id >= 0, f"acq_offer: no active company"
    assert price > 0, f"acq_offer: no offer price"

    cdef int loc = company_module.COMPANIES[company_id].get_location(state)
    cdef bint is_fi_preemption = (loc == <int>LOC_FI)

    cdef int owner_id, next_corp, next_price
    cdef int CORP_OS = <int>CorpIndices.CORP_OS

    if info.action_type == <int>ACTION_ACQ_OFFER_ACCEPT:
        if is_fi_preemption:
            # Preempting corp buys from FI
            _execute_fi_buy(state, active_corp, company_id)
        else:
            # Cross-president: execute negotiated transfer at acq_offer_price
            owner_id = company_module.COMPANIES[company_id].get_owner_id(state)
            corp_module.CORPS[active_corp].add_cash(state, -price)
            if loc == <int>LOC_CORP:
                corp_module.CORPS[owner_id].set_acquisition_proceeds(
                    state,
                    corp_module.CORPS[owner_id].get_acquisition_proceeds(state) + price,
                )
            elif loc == <int>LOC_PLAYER:
                player_module.PLAYERS[owner_id].add_cash(state, price)
            company_module.COMPANIES[company_id].transfer_to_corp_acquisition(
                state, active_corp,
            )
        _return_to_acquisition(state)

    elif info.action_type == <int>ACTION_PASS:
        if is_fi_preemption:
            # Corp declines. Set passed flag and check next preemptor.
            corp_module.CORPS[active_corp].set_passed_acq_offer(state, True)
            next_corp = _find_first_preemptor(state, original_corp, company_id)
            if next_corp >= 0:
                if next_corp == CORP_OS:
                    next_price = COMPANY_FACE_VALUE[company_id]
                else:
                    next_price = COMPANY_HIGH_PRICE[company_id]
                turn_module.TURN.set_active_corp(state, next_corp)
                turn_module.TURN.set_acq_offer_price(state, next_price)
                turn_module.TURN.set_active_player(
                    state,
                    corp_module.CORPS[next_corp].get_president_id(state),
                )
                # Stay in ACQ_OFFER
            else:
                # All declined — original corp buys
                _execute_fi_buy(state, original_corp, company_id)
                _return_to_acquisition(state)
        else:
            # Cross-president: owner declined, cancel acquisition
            _return_to_acquisition(state)


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def apply_acq_offer_action_py(GameState state, int phase_id, int action_id):
    from core.actions import decode_action_py
    info_tuple = decode_action_py(phase_id, action_id)
    cdef ActionInfo info
    info.phase = info_tuple.phase
    info.action_type = info_tuple.action_type
    info.corp_id = info_tuple.corp_id
    info.company_id = info_tuple.company_id
    info.amount = info_tuple.amount
    apply_acq_offer_action(state, &info)
