"""CLOSING phase handler.

Two entry points: ``setup_closing_phase`` sets the phase, runs auto-close
stages, and finds the first player for decisions; ``apply_closing_action``
dispatches player CLOSE/PASS decisions.

Three stages:
1. **Auto-close** (in ``setup_closing_phase``) — FI negative-income companies
   + receivership red/orange above cost-of-ownership thresholds.
2. **Player decisions** — each player in ascending ID order may voluntarily
   close any company they own (private) or that a non-receivership corp they
   preside over owns (subject to the corp retaining at least one company).
3. **Mandatory close** — force-close player privates (cheapest first) until
   no player would end up with negative cash after INCOME.

Most state access goes through entity-owned primitives. The Cython-only
legality predicate exported for action enumeration does cheap corp scalar
checks before doing any company-table scans.
"""

from core.state cimport GameState
from core.actions cimport (
    ActionInfo,
    ACTION_PASS,
    ACTION_CLOSE,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    CorpIndices,
    COMPANY_STARS,
    COMPANY_INCOME,
    COMPANY_FACE_VALUE,
    COST_OF_OWNERSHIP,
)
from entities.company cimport (
    LOC_PLAYER,
    LOC_CORP,
    company_adjusted_income,
    company_location,
    company_owner_id,
    company_owned_by_player,
    company_owned_by_fi,
    company_owned_by_corp,
)
from entities.corp cimport (
    count_corp_companies,
    corp_is_active,
    corp_is_in_receivership,
    corp_president_id,
)

# Late Python-level entity imports, same pattern as phases/bid.pyx.
from entities import turn as turn_module
from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module


DEF RED_STAR_TIER = 1
DEF ORANGE_STAR_TIER = 2
DEF GREEN_STAR_TIER = 3
DEF RED_RECEIVERSHIP_CLOSE_COO = 4
DEF ORANGE_RECEIVERSHIP_CLOSE_COO = 7


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

cdef void _auto_close_fi(GameState state) noexcept:
    """Close all FI-owned companies with negative adjusted income."""
    cdef int i
    for i in range(<int>GameConstants.NUM_COMPANIES):
        if (company_owned_by_fi(state, i)
                and company_adjusted_income(state, i) < 0):
            company_module.COMPANIES[i].remove_from_game(state)


cdef bint _corp_closable_by_player(GameState state, int corp_id, int player_id) noexcept nogil:
    """Return True if ``player_id`` can voluntarily close a company from ``corp_id``.

    Cheap scalar checks come first; counting owned companies scans the
    company table, so do it only after the corp could otherwise be eligible.
    """
    if not corp_is_active(state, corp_id):
        return False
    if corp_is_in_receivership(state, corp_id):
        return False
    if corp_president_id(state, corp_id) != player_id:
        return False

    return count_corp_companies(state, corp_id, False) > 1


cdef void _auto_close_receivership(GameState state) noexcept:
    """Auto-close red/orange companies for receivership corps above CoO thresholds.

    For each active receivership corp:
    - Red (1-star) companies close if CoO >= $4
    - Orange (2-star) companies close if CoO >= $7
    - The company with the highest face value is always protected.
    - If a corp has only one company, skip (it's the highest by default).
    - Close eligible companies lowest face value first.
    - JS gets 2x printed income bonus on auto-close.
    """
    cdef int corp_id, coo_level, i, count
    cdef int stars, coo_cost, face_val, max_face, max_face_idx
    cdef int close_id, close_face

    coo_level = turn_module.TURN.get_coo_level(state)

    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        if not corp_is_in_receivership(state, corp_id):
            continue

        count = count_corp_companies(state, corp_id, False)

        if count <= 1:
            continue  # Can't close the only company

        # Find highest face value among ALL corp companies (protected)
        max_face = -1
        max_face_idx = -1
        for i in range(<int>GameConstants.NUM_COMPANIES):
            if not company_owned_by_corp(state, i, corp_id):
                continue
            face_val = COMPANY_FACE_VALUE[i]
            if face_val > max_face:
                max_face = face_val
                max_face_idx = i

        # Close eligible companies lowest face value first, but never the
        # protected highest-face company or the last remaining company.
        while count > 1:
            close_id = -1
            for i in range(<int>GameConstants.NUM_COMPANIES):
                if i == max_face_idx:
                    continue
                if not company_owned_by_corp(state, i, corp_id):
                    continue
                stars = COMPANY_STARS[i]
                if stars >= GREEN_STAR_TIER:
                    continue  # Only red and orange companies are candidates.
                # COST_OF_OWNERSHIP is indexed by zero-based coo/star tiers.
                coo_cost = COST_OF_OWNERSHIP[coo_level - 1][stars - 1]
                if stars == RED_STAR_TIER:
                    if coo_cost < RED_RECEIVERSHIP_CLOSE_COO:
                        continue
                elif stars == ORANGE_STAR_TIER:
                    if coo_cost < ORANGE_RECEIVERSHIP_CLOSE_COO:
                        continue
                else:
                    continue

                face_val = COMPANY_FACE_VALUE[i]
                if close_id == -1 or face_val < close_face:
                    close_id = i
                    close_face = face_val

            if close_id == -1:
                break
            if corp_id == <int>CorpIndices.CORP_JS:
                corp_module.CORPS[corp_id].add_cash(
                    state, 2 * COMPANY_INCOME[close_id])
            company_module.COMPANIES[close_id].remove_from_game(state)
            count -= 1


cdef void _process_mandatory_close(GameState state) noexcept:
    """Force-close player privates until no player has negative income+cash.

    For each player: while income + cash < 0, close the player-owned private
    with the lowest face value. Only targets LOC_PLAYER companies, not corp
    subsidiaries. No JS bonus (mandatory close is for player privates only).
    """
    cdef int pid, i, j
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int income, cash
    cdef int cheapest_id, cheapest_face

    for pid in range(num_players):
        while True:
            income = player_module.PLAYERS[pid].get_income(state)
            cash = player_module.PLAYERS[pid].get_cash(state)
            if income + cash >= 0:
                break

            # Find the cheapest player-owned private
            cheapest_id = -1
            cheapest_face = 999
            for i in range(<int>GameConstants.NUM_COMPANIES):
                if not company_owned_by_player(state, i, pid):
                    continue
                if COMPANY_FACE_VALUE[i] < cheapest_face:
                    cheapest_face = COMPANY_FACE_VALUE[i]
                    cheapest_id = i

            # If no company to close, the player is stuck — this shouldn't
            # happen if game rules are followed, but guard against infinite loop.
            assert cheapest_id >= 0, \
                f"_process_mandatory_close: player {pid} has income+cash<0 but no companies to close"
            company_module.COMPANIES[cheapest_id].remove_from_game(state)


cdef bint _player_has_closable(GameState state, int player_id) noexcept:
    """Return True if the player has any company they can voluntarily close.

    A player can close:
    - Any player-owned private (LOC_PLAYER)
    - Any company owned by a non-receivership corp they preside, provided the
      corp retains at least 2 companies.
    No income filter — any company is eligible for voluntary close.
    """
    cdef int i, corp_id

    # Check player-owned privates
    for i in range(<int>GameConstants.NUM_COMPANIES):
        if company_owned_by_player(state, i, player_id):
            return True

    # Check corp subsidiaries
    for corp_id in range(<int>GameConstants.NUM_CORPS):
        if not corp_is_active(state, corp_id):
            continue
        if corp_is_in_receivership(state, corp_id):
            continue
        if corp_president_id(state, corp_id) != player_id:
            continue
        if count_corp_companies(state, corp_id, False) <= 1:
            continue
        # This corp has >=2 companies and the player presides — at least one
        # is closable (any except the last).
        return True

    return False


cdef void _advance_to_next_closer(GameState state) noexcept:
    """Find the next player (ascending ID) who has closable companies and hasn't passed.

    If none found, run mandatory close and transition to INCOME.
    """
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int current = turn_module.TURN.get_active_player(state)
    cdef int i, pid

    for i in range(1, num_players):
        pid = (current + i) % num_players
        if (not player_module.PLAYERS[pid].has_passed(state)
                and _player_has_closable(state, pid)):
            turn_module.TURN.set_active_player(state, pid)
            return

    # All players done
    _process_mandatory_close(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_INCOME)


# =============================================================================
# PUBLIC ENTRY POINTS
# =============================================================================

cdef void setup_closing_phase(GameState state) noexcept:
    """Initialize CLOSING phase: run auto-close stages and find first player.

    Sets the phase, runs FI and receivership auto-closes, clears passed
    flags, and sets the active player to the first with closable companies.
    If no player has closable companies, runs mandatory close and transitions
    to INCOME.
    """
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_CLOSING)
    cdef int num_players = turn_module.TURN.get_num_players(state)
    cdef int pid

    # Stage 1: FI auto-close
    _auto_close_fi(state)

    # Stage 2: Receivership auto-close
    _auto_close_receivership(state)

    # Stage 3: Clear passed flags for all players
    turn_module.TURN.clear_passed_flags(state)

    # Stage 4: Find the first player with closable companies
    for pid in range(num_players):
        if _player_has_closable(state, pid):
            turn_module.TURN.set_active_player(state, pid)
            return

    # No player has closable companies — run mandatory close and go to INCOME
    _process_mandatory_close(state)
    turn_module.TURN.set_phase(state, <int>GamePhases.PHASE_INCOME)


cdef void apply_closing_action(GameState state, ActionInfo* info) noexcept:
    """Dispatch a CLOSING-phase action (CLOSE or PASS).

    ``info`` is assumed to be a legal CLOSING action produced by
    ``decode_action(DPHASE_CLOSING, action_id)`` after the id was yielded
    by ``_enumerate_closing``. Illegal actions are a driver bug and trip
    the assertion below.
    """
    cdef int action_type = info.action_type
    cdef int company_id, owner_id, corp_id, pid, loc

    if action_type == ACTION_PASS:
        pid = turn_module.TURN.get_active_player(state)
        player_module.PLAYERS[pid].set_has_passed(state, True)
        _advance_to_next_closer(state)

    elif action_type == ACTION_CLOSE:
        company_id = info.company_id
        pid = turn_module.TURN.get_active_player(state)

        # Determine who owns the company and validate
        loc = company_location(state, company_id)
        owner_id = company_owner_id(state, company_id)

        if loc == <int>LOC_PLAYER:
            assert owner_id == pid, \
                f"apply_closing_action: company {company_id} owned by player {owner_id}, not active player {pid}"
        elif loc == <int>LOC_CORP:
            corp_id = owner_id
            assert player_module.PLAYERS[pid].is_president_of(state, corp_id), \
                f"apply_closing_action: player {pid} not president of corp {corp_id}"
            assert not corp_is_in_receivership(state, corp_id), \
                f"apply_closing_action: corp {corp_id} is in receivership"
            assert count_corp_companies(state, corp_id, False) > 1, \
                f"apply_closing_action: corp {corp_id} would lose its last company"

            # JS bonus: 2x printed income to JS cash
            if corp_id == <int>CorpIndices.CORP_JS:
                corp_module.CORPS[corp_id].add_cash(
                    state, 2 * COMPANY_INCOME[company_id])
        else:
            assert False, \
                f"apply_closing_action: company {company_id} has unexpected location {loc}"

        company_module.COMPANIES[company_id].remove_from_game(state)

        # Stay on the same player — they may close more. The driver will
        # re-enumerate; if only PASS remains, forced-action logic advances.

    else:
        assert False, \
            f"apply_closing_action: illegal action_type {action_type}"
