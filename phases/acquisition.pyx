"""ACQUISITION phase handler.

Handles corp acquisitions of companies from FI, other corps, and players.
See ``acquisition-impl.md`` for the full design.

Two public entry points:
  - ``setup_acquisition_phase`` — called by WRAP_UP on transition.
  - ``apply_acquisition_action`` — dispatches PASS / ACQ_PRICE / FI_BUY.

Shared helpers declared in ``acquisition.pxd`` and cimported by
``acq_offer.pyx``: ``_clear_acquisition_context``,
``_resume_acquisition_after_offer``, ``_execute_fi_buy``,
``_get_fi_purchase_price``, ``_find_first_preemptor``,
``_find_first_active_player``.
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
    ACTION_ACQ_PRICE,
    ACTION_ACQ_FI_BUY,
)
from entities.company cimport (
    LOC_PLAYER,
    LOC_FI,
    LOC_CORP,
    LOC_CORP_ACQ,
)
from phases.closing cimport setup_closing_phase

from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module
from entities import fi as fi_module


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _clear_acquisition_context(GameState state) noexcept:
    """Clear all ACQ/ACQ_OFFER context fields in the turn block.

    Shared exit cleanup for both the ACQ→CLOSING transition and the
    ACQ_OFFER→ACQUISITION return path.
    """
    turn_module.TURN.clear_active_corp(state)
    turn_module.TURN.clear_active_company(state)
    turn_module.TURN.clear_acq_offer_price(state)
    turn_module.TURN.clear_acq_offer_corp(state)


cdef void _clear_acq_offer_flags(GameState state) noexcept:
    """Clear the per-corp passed_acq_offer flags."""
    cdef int c
    for c in range(<int>GameConstants.NUM_CORPS):
        corp_module.CORPS[c].set_passed_acq_offer(state, False)


cdef int _get_fi_purchase_price(int corp_id, int company_id) noexcept:
    """Return the fixed FI purchase price for this corp/company pair."""
    if corp_id == <int>CorpIndices.CORP_OS:
        return COMPANY_FACE_VALUE[company_id]
    return COMPANY_HIGH_PRICE[company_id]


cdef int _find_first_preemptor(GameState state, int company_id) noexcept:
    """Return first eligible FI buyer by priority, or -1 if none.

    Priority: OS first if it can afford the company, then descending share
    price. The only stateful exclusion is the per-corp passed_acq_offer flag;
    callers decide whether the returned corp is the original acquirer or a
    higher-priority preemptor.
    """
    cdef int face_val = COMPANY_FACE_VALUE[company_id]
    cdef int high_val = COMPANY_HIGH_PRICE[company_id]
    cdef int CORP_OS = <int>CorpIndices.CORP_OS

    # OS always highest priority
    if (corp_module.CORPS[CORP_OS].is_active(state)
            and not corp_module.CORPS[CORP_OS].has_passed_acq_offer(state)
            and corp_module.CORPS[CORP_OS].get_cash(state) >= face_val):
        return CORP_OS

    # Other corps by descending share price. Live share prices are unique in
    # ACQUISITION; $0 corps are bankrupt and $75 ends the game before this phase.
    cdef int best_id = -1
    cdef int best_price = -1
    cdef int c, sp
    for c in range(<int>GameConstants.NUM_CORPS):
        if c == CORP_OS:
            continue
        if not corp_module.CORPS[c].is_active(state):
            continue
        if corp_module.CORPS[c].has_passed_acq_offer(state):
            continue
        if corp_module.CORPS[c].get_cash(state) < high_val:
            continue
        sp = corp_module.CORPS[c].get_share_price(state)
        if sp > best_price:
            best_price = sp
            best_id = c

    return best_id


cdef int _find_first_active_player(GameState state) noexcept:
    """Return first non-passed player (in turn order) who presides over
    an active non-receivership corp. Returns -1 if none."""
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int pos, pid, c
    for pos in range(num_players):
        pid = turn_module.TURN.find_player_at_position(state, pos)
        if player_module.PLAYERS[pid].has_passed(state):
            continue
        for c in range(<int>GameConstants.NUM_CORPS):
            if (corp_module.CORPS[c].is_active(state)
                    and not corp_module.CORPS[c].is_in_receivership(state)
                    and corp_module.CORPS[c].get_president_id(state) == pid):
                return pid
    return -1


cdef void _set_first_acquisition_player_or_closing(GameState state) noexcept:
    """Select the first player decision in ACQUISITION, or exit the phase."""
    cdef int pid = _find_first_active_player(state)
    if pid >= 0:
        turn_module.TURN.set_active_player(state, pid)
        return
    _transition_to_closing(state)


cdef void _advance_to_next_player(GameState state) noexcept:
    """Advance to the next non-passed player with active non-receivership
    corps. If none found, transition to CLOSING."""
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int current_pid = turn_module.TURN.get_active_player(state)
    # Find current player's turn position
    cdef int current_pos = -1
    cdef int pos, pid, c
    for pos in range(num_players):
        if turn_module.TURN.find_player_at_position(state, pos) == current_pid:
            current_pos = pos
            break
    assert current_pos >= 0, f"_advance_to_next_player: player {current_pid} not in turn order"

    # Scan forward from next position, wrapping around
    cdef int checked = 0
    pos = (current_pos + 1) % num_players
    while checked < num_players:
        pid = turn_module.TURN.find_player_at_position(state, pos)
        if not player_module.PLAYERS[pid].has_passed(state):
            for c in range(<int>GameConstants.NUM_CORPS):
                if (corp_module.CORPS[c].is_active(state)
                        and not corp_module.CORPS[c].is_in_receivership(state)
                        and corp_module.CORPS[c].get_president_id(state) == pid):
                    turn_module.TURN.set_active_player(state, pid)
                    return
        checked += 1
        pos = (pos + 1) % num_players

    # No eligible player found — all passed or no active corps
    _transition_to_closing(state)


cdef void _handle_pass(GameState state) noexcept:
    """Mark current player as passed and advance."""
    cdef int pid = turn_module.TURN.get_active_player(state)
    player_module.PLAYERS[pid].set_has_passed(state, True)
    _advance_to_next_player(state)


cdef void _execute_fi_buy(GameState state, int corp_id, int company_id) noexcept:
    """Execute FI purchase: corp pays, FI receives, company to acq pile."""
    cdef int price = _get_fi_purchase_price(corp_id, company_id)
    corp_module.CORPS[corp_id].add_cash(state, -price)
    fi_module.FI.add_cash(state, price)
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)


cdef int _find_most_expensive_affordable_fi_company(
    GameState state, int corp_id,
) noexcept:
    """Return the most expensive FI company this corp can afford, or -1."""
    cdef int cash = corp_module.CORPS[corp_id].get_cash(state)
    cdef int company_id, price
    cdef int best_company = -1
    cdef int best_price = -1

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].get_location(state) != <int>LOC_FI:
            continue
        price = _get_fi_purchase_price(corp_id, company_id)
        if price <= cash and price > best_price:
            best_price = price
            best_company = company_id

    return best_company


cdef bint _find_receivership_forced_buy(
    GameState state, int* out_corp, int* out_company,
) noexcept:
    """Find the next automatic receivership FI buy.

    Returns True and writes ``out_corp`` / ``out_company`` when some active
    receivership corp can afford at least one FI company. OS has first
    priority; other receiverships are considered by descending share price.
    Each corp targets its most expensive affordable FI company.
    """
    cdef int CORP_OS = <int>CorpIndices.CORP_OS
    cdef int corp_id, company_id, share_price
    cdef int best_corp = -1
    cdef int best_company = -1
    cdef int best_share_price = -1

    if (corp_module.CORPS[CORP_OS].is_active(state)
            and corp_module.CORPS[CORP_OS].is_in_receivership(state)):
        company_id = _find_most_expensive_affordable_fi_company(state, CORP_OS)
        if company_id >= 0:
            out_corp[0] = CORP_OS
            out_company[0] = company_id
            return True

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_id == CORP_OS:
            continue
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        if not corp_module.CORPS[corp_id].is_in_receivership(state):
            continue
        company_id = _find_most_expensive_affordable_fi_company(state, corp_id)
        if company_id < 0:
            continue
        share_price = corp_module.CORPS[corp_id].get_share_price(state)
        if share_price > best_share_price:
            best_share_price = share_price
            best_corp = corp_id
            best_company = company_id

    if best_corp < 0:
        return False

    out_corp[0] = best_corp
    out_company[0] = best_company
    return True


cdef bint _process_receivership_forced_buys(GameState state) noexcept:
    """Run beginning-of-ACQUISITION receivership buys.

    Returns True if processing paused on an ACQ_OFFER player decision.
    Returns False once no forced receivership buy remains.
    """
    cdef int recv_corp, company_id, first_preemptor, price, deciding_player

    while _find_receivership_forced_buy(state, &recv_corp, &company_id):
        _clear_acq_offer_flags(state)
        first_preemptor = _find_first_preemptor(state, company_id)

        if first_preemptor < 0 or first_preemptor == recv_corp:
            _execute_fi_buy(state, recv_corp, company_id)
            continue

        if corp_module.CORPS[first_preemptor].is_in_receivership(state):
            _execute_fi_buy(state, first_preemptor, company_id)
            continue

        price = _get_fi_purchase_price(first_preemptor, company_id)
        deciding_player = corp_module.CORPS[first_preemptor].get_president_id(state)
        _enter_acq_offer(
            state, first_preemptor, company_id, price,
            recv_corp, deciding_player,
        )
        return True

    return False


cdef void _enter_acq_offer(
    GameState state, int offered_corp, int company_id, int price,
    int original_corp, int deciding_player,
) noexcept:
    """Push into ACQ_OFFER phase.

    Args:
        offered_corp: corp that would acquire the company (active_corp in ACQ_OFFER)
        company_id: the target company
        price: the offer price (stored in acq_offer_price)
        original_corp: the corp that initiated the acquisition (acq_offer_corp)
        deciding_player: who makes the accept/pass decision
    """
    turn_module.TURN.set_acq_offer_corp(state, original_corp)
    turn_module.TURN.set_acq_offer_price(state, price)
    turn_module.TURN.set_active_corp(state, offered_corp)
    turn_module.TURN.set_active_company(state, company_id)
    turn_module.TURN.set_active_player(state, deciding_player)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_OFFER)


cdef void _resume_acquisition_after_offer(GameState state, int original_corp) noexcept:
    """Return from ACQ_OFFER to ACQUISITION without exposing automation."""
    cdef bint resume_receivership_setup = (
        original_corp >= 0
        and corp_module.CORPS[original_corp].is_active(state)
        and corp_module.CORPS[original_corp].is_in_receivership(state)
    )
    cdef int pid

    _clear_acquisition_context(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQUISITION)

    if resume_receivership_setup:
        if _process_receivership_forced_buys(state):
            return
        _set_first_acquisition_player_or_closing(state)
        return

    if (original_corp >= 0
            and corp_module.CORPS[original_corp].is_active(state)
            and not corp_module.CORPS[original_corp].is_in_receivership(state)):
        pid = corp_module.CORPS[original_corp].get_president_id(state)
        if pid >= 0 and not player_module.PLAYERS[pid].has_passed(state):
            turn_module.TURN.set_active_player(state, pid)
            return

    _set_first_acquisition_player_or_closing(state)


cdef void _handle_acq_price(GameState state, ActionInfo* info) noexcept:
    """Execute a negotiated-price acquisition (corp-to-corp or corp-to-player).

    If acq_same_president is False and the owner is a different player,
    enters ACQ_OFFER for owner approval instead of executing directly.
    """
    cdef int corp_id = info.corp_id
    cdef int company_id = info.company_id
    cdef int price = COMPANY_LOW_PRICE[company_id] + info.amount

    cdef int loc = company_module.COMPANIES[company_id].get_location(state)
    cdef int owner_id = company_module.COMPANIES[company_id].get_owner_id(state)
    cdef int active_player = turn_module.TURN.get_active_player(state)
    cdef int owner_player = -1

    assert corp_module.CORPS[corp_id].is_active(state), \
        f"_handle_acq_price: corp {corp_id} not active"
    assert corp_module.CORPS[corp_id].get_cash(state) >= price, \
        f"_handle_acq_price: corp {corp_id} can't afford {price}"

    # Check cross-president: enter ACQ_OFFER if owner is a different player
    if not state.acq_same_president:
        if loc == <int>LOC_CORP:
            assert not corp_module.CORPS[owner_id].is_in_receivership(state), \
                f"_handle_acq_price: cannot buy company {company_id} from receivership corp {owner_id}"
            owner_player = corp_module.CORPS[owner_id].get_president_id(state)
        elif loc == <int>LOC_PLAYER:
            owner_player = owner_id
        if owner_player >= 0 and owner_player != active_player:
            _enter_acq_offer(
                state, corp_id, company_id, price,
                corp_id, owner_player,
            )
            return

    # Same-president: execute directly
    corp_module.CORPS[corp_id].add_cash(state, -price)

    if loc == <int>LOC_CORP:
        assert owner_id != corp_id, \
            f"_handle_acq_price: corp {corp_id} buying from itself"
        assert not corp_module.CORPS[owner_id].is_in_receivership(state), \
            f"_handle_acq_price: cannot buy company {company_id} from receivership corp {owner_id}"
        corp_module.CORPS[owner_id].set_acquisition_proceeds(
            state,
            corp_module.CORPS[owner_id].get_acquisition_proceeds(state) + price,
        )
    elif loc == <int>LOC_PLAYER:
        player_module.PLAYERS[owner_id].add_cash(state, price)

    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, corp_id)


cdef void _handle_fi_buy(GameState state, ActionInfo* info) noexcept:
    """Execute an FI purchase, with preemption check."""
    cdef int corp_id = info.corp_id
    cdef int company_id = info.company_id
    cdef int first_preemptor, price, deciding_player

    assert company_module.COMPANIES[company_id].get_location(state) == <int>LOC_FI, \
        f"_handle_fi_buy: company {company_id} not LOC_FI"
    assert corp_module.CORPS[corp_id].is_active(state), \
        f"_handle_fi_buy: corp {corp_id} not active"
    assert corp_module.CORPS[corp_id].get_cash(state) >= _get_fi_purchase_price(corp_id, company_id), \
        f"_handle_fi_buy: corp {corp_id} can't afford FI company {company_id}"

    # Check for preemptors
    _clear_acq_offer_flags(state)
    first_preemptor = _find_first_preemptor(state, company_id)
    if first_preemptor < 0 or first_preemptor == corp_id:
        _execute_fi_buy(state, corp_id, company_id)
        return

    # A higher-priority receivership corp has no president to ask; its FI
    # purchase is automatic. Player-controlled corps enter ACQ_OFFER.
    if corp_module.CORPS[first_preemptor].is_in_receivership(state):
        _execute_fi_buy(state, first_preemptor, company_id)
        return

    price = _get_fi_purchase_price(first_preemptor, company_id)
    deciding_player = corp_module.CORPS[first_preemptor].get_president_id(state)
    _enter_acq_offer(
        state, first_preemptor, company_id, price,
        corp_id, deciding_player,
    )


cdef void _merge_acquisition_zones(GameState state) noexcept:
    """Phase-exit cleanup: merge acq piles into owned, flush proceeds."""
    cdef int corp_id, company_id, proceeds, owner_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_module.CORPS[corp_id].is_active(state):
            continue
        proceeds = corp_module.CORPS[corp_id].get_acquisition_proceeds(state)
        if proceeds > 0:
            corp_module.CORPS[corp_id].add_cash(state, proceeds)
            corp_module.CORPS[corp_id].set_acquisition_proceeds(state, 0)

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_module.COMPANIES[company_id].get_location(state) == <int>LOC_CORP_ACQ:
            owner_id = company_module.COMPANIES[company_id].get_owner_id(state)
            company_module.COMPANIES[company_id].transfer_to_corp(state, owner_id)

    # Invariant: no acquisition-pile companies or buffered proceeds remain
    if __debug__:
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            assert company_module.COMPANIES[company_id].get_location(state) != <int>LOC_CORP_ACQ, \
                f"_merge_acquisition_zones: LOC_CORP_ACQ still present for company {company_id}"
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            if corp_module.CORPS[corp_id].is_active(state):
                assert corp_module.CORPS[corp_id].get_acquisition_proceeds(state) == 0, \
                    f"_merge_acquisition_zones: nonzero proceeds on corp {corp_id}"


cdef void _transition_to_closing(GameState state) noexcept:
    """Exit ACQUISITION: merge zones, clear context, hand off to CLOSING."""
    _merge_acquisition_zones(state)
    _clear_acquisition_context(state)
    setup_closing_phase(state)


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_acquisition_phase(GameState state) noexcept:
    """Initialize ACQUISITION phase context. Called by WRAP_UP."""
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQUISITION)
    turn_module.TURN.clear_active_corp(state)
    turn_module.TURN.clear_active_company(state)
    turn_module.TURN.clear_acq_offer_price(state)
    turn_module.TURN.clear_passed_flags(state)

    if _process_receivership_forced_buys(state):
        return
    _set_first_acquisition_player_or_closing(state)


cdef void apply_acquisition_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch an ACQUISITION action. Assumes legality (driver guarantees)."""
    if info.action_type == <int>ACTION_PASS:
        _handle_pass(state)
    elif info.action_type == <int>ACTION_ACQ_PRICE:
        _handle_acq_price(state, info)
        # Stay on same player — driver re-enumerates
    elif info.action_type == <int>ACTION_ACQ_FI_BUY:
        _handle_fi_buy(state, info)
        # Either stays in ACQUISITION (same player) or entered ACQ_OFFER


# =============================================================================
# PYTHON TEST WRAPPERS
# =============================================================================

def setup_acquisition_phase_py(GameState state):
    setup_acquisition_phase(state)

def apply_acquisition_action_py(GameState state, int phase_id, int action_id):
    from core.actions import decode_action_py
    info_tuple = decode_action_py(phase_id, action_id)
    cdef ActionInfo info
    info.phase = info_tuple.phase
    info.action_type = info_tuple.action_type
    info.corp_id = info_tuple.corp_id
    info.company_id = info_tuple.company_id
    info.amount = info_tuple.amount
    apply_acquisition_action(state, &info)
