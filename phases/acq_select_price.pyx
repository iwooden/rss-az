"""ACQ_SELECT_PRICE phase handler.

Final leg of the three-step ACQ flow: SELECT_CORP → SELECT_COMPANY →
SELECT_PRICE. Reads ``active_corp`` + ``active_company`` from the turn
block and executes the acquisition.

Action semantics (all decoded from DPHASE_ACQ_SELECT_PRICE):
  ACTION_ACQ_PRICE (amount ∈ [0, 50]): negotiated-price acquisition.
    Cross-president (LOC_CORP or LOC_PLAYER with foreign owner under
    ``acq_same_president=False``) enters ACQ_OFFER; otherwise transfer
    executes directly.
  ACTION_ACQ_FI_BUY: FI purchase at the fixed price, with preemption
    check. If a higher-priority player-owned corp can intervene, push
    into ACQ_OFFER for that player's accept/pass decision.

After direct execution (no ACQ_OFFER entry): clear active_corp +
active_company and walk back to PHASE_ACQ_SELECT_CORP with the active
player unchanged — the same player re-enumerates and may acquire again
or pass. The ACQ_OFFER path does its own resume via
``_resume_acquisition_after_offer``.

Reference: RULES.md Acquisition procedure. See phase-refactor.md for the
split rationale.
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants,
    GamePhases,
    COMPANY_LOW_PRICE,
)
from core.actions cimport (
    ActionInfo,
    ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_BUY,
)
from entities.company cimport (
    LOC_PLAYER,
    LOC_FI,
    LOC_CORP,
    company_location,
    company_owner_id,
)
from entities.corp cimport (
    corp_is_active,
    corp_cash,
    corp_is_in_receivership,
    corp_president_id,
    corp_acquisition_proceeds,
)
from phases.acq_select_corp cimport (
    _execute_fi_buy,
    _get_fi_purchase_price,
    _find_first_preemptor,
    _enter_acq_offer,
)

from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _clear_acq_pair(GameState state) noexcept:
    """Reset active_corp + active_company and walk back to SELECT_CORP."""
    turn_module.TURN.clear_active_corp(state)
    turn_module.TURN.clear_active_company(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_SELECT_CORP)


cdef void _handle_acq_price(GameState state, int corp_id, int company_id, int amount) noexcept:
    """Execute a negotiated-price acquisition (corp-to-corp or corp-to-player).

    If ``acq_same_president`` is False and the owner is a different player,
    enter ACQ_OFFER for owner approval instead of executing directly.
    """
    cdef int price = COMPANY_LOW_PRICE[company_id] + amount
    cdef int loc = company_location(state, company_id)
    cdef int owner_id = company_owner_id(state, company_id)
    cdef int active_player = turn_module.TURN.get_active_player(state)
    cdef int owner_player = -1

    assert corp_is_active(state, corp_id), \
        f"_handle_acq_price: corp {corp_id} not active"
    assert corp_cash(state, corp_id) >= price, \
        f"_handle_acq_price: corp {corp_id} can't afford {price}"

    # Cross-president branch: owner is a different player — ask them.
    if not state.acq_same_president:
        if loc == <int>LOC_CORP:
            assert not corp_is_in_receivership(state, owner_id), \
                f"_handle_acq_price: cannot buy company {company_id} from receivership corp {owner_id}"
            owner_player = corp_president_id(state, owner_id)
        elif loc == <int>LOC_PLAYER:
            owner_player = owner_id
        if owner_player >= 0 and owner_player != active_player:
            _enter_acq_offer(
                state, corp_id, company_id, price,
                corp_id, owner_player,
            )
            return

    # Same-president (or no foreign owner): execute directly.
    corp_module.CORPS[corp_id].add_cash(state, -price)

    if loc == <int>LOC_CORP:
        assert owner_id != corp_id, \
            f"_handle_acq_price: corp {corp_id} buying from itself"
        assert not corp_is_in_receivership(state, owner_id), \
            f"_handle_acq_price: cannot buy company {company_id} from receivership corp {owner_id}"
        corp_module.CORPS[owner_id].set_acquisition_proceeds(
            state,
            corp_acquisition_proceeds(state, owner_id) + price,
        )
    elif loc == <int>LOC_PLAYER:
        player_module.PLAYERS[owner_id].add_cash(state, price)

    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)
    _clear_acq_pair(state)


cdef void _handle_fi_buy(GameState state, int corp_id, int company_id) noexcept:
    """Execute an FI purchase, with preemption check."""
    cdef int first_preemptor, price, deciding_player

    assert company_location(state, company_id) == <int>LOC_FI, \
        f"_handle_fi_buy: company {company_id} not LOC_FI"
    assert corp_is_active(state, corp_id), \
        f"_handle_fi_buy: corp {corp_id} not active"
    assert corp_cash(state, corp_id) >= _get_fi_purchase_price(corp_id, company_id), \
        f"_handle_fi_buy: corp {corp_id} can't afford FI company {company_id}"

    # Clear per-corp passed_acq_offer flags before consulting the preemptor
    # list (matches the receivership-forced-buy prelude in SELECT_CORP).
    cdef int c
    for c in range(<int>GameConstants.NUM_CORPS):
        corp_module.CORPS[c].set_passed_acq_offer(state, False)

    first_preemptor = _find_first_preemptor(state, company_id, corp_id)
    if first_preemptor < 0 or first_preemptor == corp_id:
        _execute_fi_buy(state, corp_id, company_id)
        _clear_acq_pair(state)
        return

    # Higher-priority receivership corps have no president to ask — their
    # FI purchase is automatic. Player-controlled corps enter ACQ_OFFER.
    if corp_is_in_receivership(state, first_preemptor):
        _execute_fi_buy(state, first_preemptor, company_id)
        _clear_acq_pair(state)
        return

    price = _get_fi_purchase_price(first_preemptor, company_id)
    deciding_player = corp_president_id(state, first_preemptor)
    _enter_acq_offer(
        state, first_preemptor, company_id, price,
        corp_id, deciding_player,
    )


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_acq_select_price_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch a SELECT_PRICE action. Assumes legality (driver guarantees)."""
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    cdef int company_id = turn_module.TURN.get_active_company(state)
    assert corp_id >= 0, "apply_acq_select_price_action: active_corp unset"
    assert company_id >= 0, "apply_acq_select_price_action: active_company unset"

    if info.action_type == <int>ACTION_ACQ_PRICE:
        _handle_acq_price(state, corp_id, company_id, info.amount)
    elif info.action_type == <int>ACTION_ACQ_FI_BUY:
        _handle_fi_buy(state, corp_id, company_id)
    else:
        assert False, \
            f"apply_acq_select_price_action: unexpected type {info.action_type}"


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def apply_acq_select_price_action_py(GameState state, int phase_id, int action_id):
    from core.actions import decode_action_py
    info_tuple = decode_action_py(phase_id, action_id)
    cdef ActionInfo info
    info.phase = info_tuple.phase
    info.action_type = info_tuple.action_type
    info.corp_id = info_tuple.corp_id
    info.company_id = info_tuple.company_id
    info.amount = info_tuple.amount
    apply_acq_select_price_action(state, &info)
