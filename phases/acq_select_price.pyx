"""ACQ_SELECT_PRICE phase handler.

Final leg of the three-step ACQ flow: SELECT_CORP → SELECT_COMPANY →
SELECT_PRICE. Reads ``active_corp`` + ``active_company`` from the turn
block and executes the acquisition at the model-chosen price.

Only reached for non-FI targets. FI purchases execute directly in
SELECT_COMPANY (no price decision — OS pays face, others pay high), so
an FI company never reaches this handler.

Action semantics:
  ACTION_ACQ_PRICE (amount ∈ [0, 50]): negotiated-price acquisition.
    Cross-president (LOC_CORP or LOC_PLAYER with foreign owner under
    ``acq_same_president=False``) enters ACQ_OFFER; otherwise transfer
    executes directly.

After direct execution (no ACQ_OFFER entry): clear active_corp +
active_company and walk back to PHASE_ACQ_SELECT_CORP with the active
player unchanged — the same player re-enumerates and may acquire again
or pass. The ACQ_OFFER path does its own resume via
``_resume_acquisition_after_offer``.

Reference: RULES.md Acquisition procedure. See phase-refactor.md for the
split rationale.
"""

from core.state cimport GameState
from core.data cimport COMPANY_LOW_PRICE
from core.actions cimport (
    ActionInfo,
    ACTION_ACQ_PRICE,
)
from entities.company cimport (
    LOC_PLAYER,
    LOC_CORP,
    company_location,
    company_owner_id,
)
from entities.corp cimport (
    corp_is_active,
    corp_cash,
    corp_is_in_receivership,
    corp_president_id,
)
from phases.util.acq_common cimport (
    _clear_acq_pair,
    _execute_acq_transfer,
    _enter_acq_offer,
)

from entities import turn as turn_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

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
    _execute_acq_transfer(state, corp_id, company_id, price, loc)
    _clear_acq_pair(state)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

cdef void apply_acq_select_price_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch a SELECT_PRICE action. Assumes legality (driver guarantees)."""
    cdef int corp_id = turn_module.TURN.get_active_corp(state)
    cdef int company_id = turn_module.TURN.get_active_company(state)
    assert corp_id >= 0, "apply_acq_select_price_action: active_corp unset"
    assert company_id >= 0, "apply_acq_select_price_action: active_company unset"
    assert info.action_type == <int>ACTION_ACQ_PRICE, \
        f"apply_acq_select_price_action: unexpected type {info.action_type}"

    _handle_acq_price(state, corp_id, company_id, info.amount)


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
