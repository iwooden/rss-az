"""Shared ACQ helpers used across the four ACQ sub-phase handlers.

Semantically "ACQ common" — the helpers operate on the unified ACQ
turn-block context (``active_corp``, ``active_company``,
``acq_offer_*`` fields) and are cimported by
``acq_select_corp``, ``acq_select_company``, ``acq_select_price``, and
``acq_offer``. Keeping them in a dedicated module avoids any single
sub-phase owning logic the others depend on.

Reference: RULES.md Acquisition procedure; phase-refactor.md for the
SELECT_CORP / SELECT_COMPANY / SELECT_PRICE split rationale.
"""

from core.state cimport GameState
from core.data cimport (
    GameConstants,
    GamePhases,
    CorpIndices,
    COMPANY_FACE_VALUE,
    COMPANY_HIGH_PRICE,
)
from entities.company cimport (
    LOC_FI,
    LOC_CORP,
    LOC_PLAYER,
    LOC_CORP_ACQ,
    company_location,
    company_owner_id,
)
from entities.corp cimport (
    corp_is_active,
    corp_cash,
    corp_share_price,
    corp_is_in_receivership,
    corp_president_id,
    corp_has_passed_acq_offer,
    corp_acquisition_proceeds,
)
from phases.closing cimport setup_closing_phase

from entities import turn as turn_module
from entities import corp as corp_module
from entities import company as company_module
from entities import player as player_module
from entities import fi as fi_module


cdef void _clear_acquisition_context(GameState state) noexcept:
    """Clear all ACQ/ACQ_OFFER context fields in the turn block.

    Shared exit cleanup for both the SELECT_CORP→CLOSING transition and
    the ACQ_OFFER→SELECT_CORP return path.
    """
    turn_module.TURN.clear_acquisition_context(state)


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


cdef int _find_first_preemptor(GameState state, int company_id, int original_corp) noexcept:
    """Return first eligible FI buyer by priority, or -1 if none.

    Priority: OS first if it can afford the company, then descending share
    price. The only stateful exclusion is the per-corp passed_acq_offer flag;
    callers decide whether the returned corp is the original acquirer or a
    higher-priority preemptor.

    Rolling Stock / Rolling Stock Stars FI intervention has one additional
    nuance: when a player-controlled corporation proposes to buy from FI,
    higher-priority corporations owned by that same player are skipped until we
    either reach a different player's corporation or fall back to the original
    buyer. This mirrors 18xx.games' responder_list construction, which strips a
    same-president prefix so a player cannot "intervene" on their own FI buy
    with another corporation they control.

    Pass ``original_corp = -1`` when there is no player proposer (for example
    receivership auto-buys). In that case no same-president skipping occurs.
    """
    cdef int face_val = COMPANY_FACE_VALUE[company_id]
    cdef int high_val = COMPANY_HIGH_PRICE[company_id]
    cdef int CORP_OS = <int>CorpIndices.CORP_OS
    cdef int original_owner = -1

    if (original_corp >= 0
            and corp_is_active(state, original_corp)
            and not corp_is_in_receivership(state, original_corp)):
        original_owner = corp_president_id(state, original_corp)

    # OS always highest priority
    if (corp_is_active(state, CORP_OS)
            and not corp_has_passed_acq_offer(state, CORP_OS)
            and corp_cash(state, CORP_OS) >= face_val
            and (original_owner < 0
                 or CORP_OS == original_corp
                 or corp_president_id(state, CORP_OS) != original_owner)):
        return CORP_OS

    # Other corps by descending share price. Live share prices are unique in
    # ACQ; $0 corps are bankrupt and $75 ends the game before this phase.
    cdef int best_id = -1
    cdef int best_price = -1
    cdef int c, sp
    for c in range(<int>GameConstants.NUM_CORPS):
        if c == CORP_OS:
            continue
        if not corp_is_active(state, c):
            continue
        if corp_has_passed_acq_offer(state, c):
            continue
        if corp_cash(state, c) < high_val:
            continue
        if (original_owner >= 0
                and c != original_corp
                and corp_president_id(state, c) == original_owner):
            continue
        sp = corp_share_price(state, c)
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
            if (corp_is_active(state, c)
                    and not corp_is_in_receivership(state, c)
                    and corp_president_id(state, c) == pid):
                return pid
    return -1


cdef void _set_first_acquisition_player_or_closing(GameState state) noexcept:
    """Select the first player decision in SELECT_CORP, or exit the phase."""
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
    cdef int current_pos = -1
    cdef int pos, pid, c
    for pos in range(num_players):
        if turn_module.TURN.find_player_at_position(state, pos) == current_pid:
            current_pos = pos
            break
    assert current_pos >= 0, f"_advance_to_next_player: player {current_pid} not in turn order"

    cdef int checked = 0
    pos = (current_pos + 1) % num_players
    while checked < num_players:
        pid = turn_module.TURN.find_player_at_position(state, pos)
        if not player_module.PLAYERS[pid].has_passed(state):
            for c in range(<int>GameConstants.NUM_CORPS):
                if (corp_is_active(state, c)
                        and not corp_is_in_receivership(state, c)
                        and corp_president_id(state, c) == pid):
                    turn_module.TURN.set_active_player(state, pid)
                    return
        checked += 1
        pos = (pos + 1) % num_players

    # No eligible player found — all passed or no active corps
    _transition_to_closing(state)


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
    cdef int cash = corp_cash(state, corp_id)
    cdef int company_id, price
    cdef int best_company = -1
    cdef int best_price = -1

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_location(state, company_id) != <int>LOC_FI:
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

    if (corp_is_active(state, CORP_OS)
            and corp_is_in_receivership(state, CORP_OS)):
        company_id = _find_most_expensive_affordable_fi_company(state, CORP_OS)
        if company_id >= 0:
            out_corp[0] = CORP_OS
            out_company[0] = company_id
            return True

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if corp_id == CORP_OS:
            continue
        if not corp_is_active(state, corp_id):
            continue
        if not corp_is_in_receivership(state, corp_id):
            continue
        company_id = _find_most_expensive_affordable_fi_company(state, corp_id)
        if company_id < 0:
            continue
        share_price = corp_share_price(state, corp_id)
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
    """Run beginning-of-SELECT_CORP receivership buys.

    Returns True if processing paused on an ACQ_OFFER player decision.
    Returns False once no forced receivership buy remains.
    """
    cdef int recv_corp, company_id, first_preemptor, price, deciding_player

    while _find_receivership_forced_buy(state, &recv_corp, &company_id):
        _clear_acq_offer_flags(state)
        first_preemptor = _find_first_preemptor(state, company_id, -1)

        if first_preemptor < 0 or first_preemptor == recv_corp:
            _execute_fi_buy(state, recv_corp, company_id)
            continue

        if corp_is_in_receivership(state, first_preemptor):
            _execute_fi_buy(state, first_preemptor, company_id)
            continue

        price = _get_fi_purchase_price(first_preemptor, company_id)
        deciding_player = corp_president_id(state, first_preemptor)
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
    turn_module.TURN.enter_acq_offer(
        state,
        offered_corp,
        company_id,
        price,
        original_corp,
        deciding_player,
    )


cdef void _resume_acquisition_after_offer(GameState state, int original_corp) noexcept:
    """Return from ACQ_OFFER to SELECT_CORP without exposing automation."""
    cdef bint resume_receivership_setup = (
        original_corp >= 0
        and corp_is_active(state, original_corp)
        and corp_is_in_receivership(state, original_corp)
    )
    cdef int pid

    _clear_acquisition_context(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_ACQ_SELECT_CORP)

    if resume_receivership_setup:
        if _process_receivership_forced_buys(state):
            return
        _set_first_acquisition_player_or_closing(state)
        return

    if (original_corp >= 0
            and corp_is_active(state, original_corp)
            and not corp_is_in_receivership(state, original_corp)):
        pid = corp_president_id(state, original_corp)
        if pid >= 0 and not player_module.PLAYERS[pid].has_passed(state):
            turn_module.TURN.set_active_player(state, pid)
            return

    _set_first_acquisition_player_or_closing(state)


cdef void _execute_acq_transfer(
    GameState state, int buyer_corp, int company_id, int price, int loc,
) noexcept:
    """Execute an ACQ transfer at ``price`` to ``buyer_corp``'s acq pile.

    Shared by SELECT_PRICE direct-execution and ACQ_OFFER ACCEPT. The
    buyer pays; for LOC_CORP the seller corp's acquisition_proceeds is
    bumped (flushed at phase exit), for LOC_PLAYER the seller gets cash
    immediately. LOC_FI is out of scope — use ``_execute_fi_buy``.
    """
    cdef int owner_id = company_owner_id(state, company_id)
    corp_module.CORPS[buyer_corp].add_cash(state, -price)
    if loc == <int>LOC_CORP:
        assert owner_id != buyer_corp, \
            f"_execute_acq_transfer: corp {buyer_corp} buying from itself"
        assert not corp_is_in_receivership(state, owner_id), \
            f"_execute_acq_transfer: company {company_id} owned by receivership corp {owner_id}"
        corp_module.CORPS[owner_id].set_acquisition_proceeds(
            state,
            corp_acquisition_proceeds(state, owner_id) + price,
        )
    elif loc == <int>LOC_PLAYER:
        player_module.PLAYERS[owner_id].add_cash(state, price)
    company_module.COMPANIES[company_id].transfer_to_corp_acquisition(state, buyer_corp)


cdef void _merge_acquisition_zones(GameState state) noexcept:
    """Phase-exit cleanup: merge acq piles into owned, flush proceeds."""
    cdef int corp_id, company_id, proceeds, owner_id

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        proceeds = corp_acquisition_proceeds(state, corp_id)
        if proceeds > 0:
            corp_module.CORPS[corp_id].add_cash(state, proceeds)
            corp_module.CORPS[corp_id].set_acquisition_proceeds(state, 0)

    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_location(state, company_id) == <int>LOC_CORP_ACQ:
            owner_id = company_owner_id(state, company_id)
            company_module.COMPANIES[company_id].transfer_to_corp(state, owner_id)

    # Invariant: no acquisition-pile companies or buffered proceeds remain
    if __debug__:
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            assert company_location(state, company_id) != <int>LOC_CORP_ACQ, \
                f"_merge_acquisition_zones: LOC_CORP_ACQ still present for company {company_id}"
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            if corp_is_active(state, corp_id):
                assert corp_acquisition_proceeds(state, corp_id) == 0, \
                    f"_merge_acquisition_zones: nonzero proceeds on corp {corp_id}"


cdef void _transition_to_closing(GameState state) noexcept:
    """Exit SELECT_CORP: merge zones, clear context, hand off to CLOSING."""
    _merge_acquisition_zones(state)
    _clear_acquisition_context(state)
    setup_closing_phase(state)
