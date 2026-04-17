"""ACQ_OFFER phase handler.

Unified "accept/decline this acquisition?" phase. The question is always:
"Should active_corp acquire active_company at acq_offer_price?" — only
who decides (active_player) varies between FI preemption and cross-
president offers.

No setup function — ACQ_OFFER is entered mid-action from
``apply_acquisition_action`` via ``_enter_acq_offer``.
"""

from core.state cimport GameState
from core.data cimport GamePhases
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_ACQ_OFFER_ACCEPT,
)
from entities.company cimport (
    LOC_FI,
    LOC_CORP,
    LOC_PLAYER,
    company_location,
    company_owner_id,
)
from entities.corp cimport (
    corp_acquisition_proceeds,
    corp_is_in_receivership,
    corp_president_id,
)

from phases.acquisition cimport (
    _resume_acquisition_after_offer,
    _execute_fi_buy,
    _get_fi_purchase_price,
    _find_first_preemptor,
)

from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _return_to_acquisition(GameState state) noexcept:
    """Return to ACQUISITION after the active offer resolves."""
    cdef int original_corp = turn_module.TURN.get_acq_offer_corp(state)
    _resume_acquisition_after_offer(state, original_corp)


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

    cdef int loc = company_location(state, company_id)
    cdef bint is_fi_preemption = (loc == <int>LOC_FI)

    cdef int owner_id, next_corp, next_price, deciding_player

    if info.action_type == <int>ACTION_ACQ_OFFER_ACCEPT:
        if is_fi_preemption:
            # Preempting corp buys from FI
            _execute_fi_buy(state, active_corp, company_id)
        else:
            # Cross-president: execute negotiated transfer at acq_offer_price
            owner_id = company_owner_id(state, company_id)
            corp_module.CORPS[active_corp].add_cash(state, -price)
            if loc == <int>LOC_CORP:
                corp_module.CORPS[owner_id].set_acquisition_proceeds(
                    state,
                    corp_acquisition_proceeds(state, owner_id) + price,
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
            next_corp = _find_first_preemptor(state, company_id, original_corp)
            if next_corp >= 0 and next_corp != original_corp:
                if corp_is_in_receivership(state, next_corp):
                    _execute_fi_buy(state, next_corp, company_id)
                    _return_to_acquisition(state)
                    return

                next_price = _get_fi_purchase_price(next_corp, company_id)
                deciding_player = corp_president_id(state, next_corp)
                turn_module.TURN.enter_acq_offer(
                    state,
                    next_corp,
                    company_id,
                    next_price,
                    original_corp,
                    deciding_player,
                )
                # Stay in ACQ_OFFER
            else:
                # All higher-priority corps declined, or the original corp is
                # now first in priority order — original action goes through.
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
