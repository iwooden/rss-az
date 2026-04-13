"""
Company entity implementation.

Companies live in exactly one place at a time. The compact GameState array
tracks every company's location and owner directly via two parallel int16
sub-arrays inside the companies section:

  state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.locations + cid]
      -- CompanyLocation enum
  state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.owner_ids + cid]
      -- player_id / corp_id / -1

There are no separate "for auction", "revealed", "removed", FI- or
player-side ownership flags anymore — those queries are derived from the
location enum. Transfer operations atomically update the location and
owner_id, then trigger downstream recalculation on the entities whose
income or star totals changed.

The company entity owns semantic company state only. The live deck array
is owned exclusively by the deck entity; company transitions assert if a
caller tries to move a company still marked ``LOC_DECK`` instead of
routing that change through ``DECK`` first.

Layout offsets are constants pulled from the module-level ``LAYOUT`` and
``COMPANY_OFFSETS`` on ``core.state``: there is no per-instance offset
cache and no initialize() step. The handle's only state is its
``company_id`` and ``name``.
"""

from libc.stdint cimport int16_t
from core.state cimport GameState, LAYOUT, COMPANY_OFFSETS
from entities.turn cimport TurnState
from entities.corp cimport invalidate_corp_cache
from entities.player cimport invalidate_player_cache
from core.data cimport (
    GameConstants,
    COMPANY_FACE_VALUE,
    COMPANY_LOW_PRICE,
    COMPANY_HIGH_PRICE,
    COMPANY_STARS,
    COMPANY_INCOME,
    COMPANY_LAST_IN_GROUP,
    COMPANY_SYNERGY,
)

from core.data import COMPANY_NAMES
from entities import turn as turn_module
from entities import corp as corp_module
from entities import fi as fi_module

# Lazy accessor for the TURN singleton. See the corresponding comment in
# player.pyx — caching at module init would force entities.turn to be fully
# loaded before this module finished its own init, so we defer the lookup
# to first call. After that it's a cached pointer return under the GIL
# (every callsite is in a cpdef/def context, never inside nogil).
cdef TurnState _TURN_CACHED = None

cdef TurnState _TURN():
    global _TURN_CACHED
    if _TURN_CACHED is None:
        _TURN_CACHED = <TurnState>turn_module.TURN
    return _TURN_CACHED


# =============================================================================
# COMPANIES-SECTION STORAGE ACCESS
# =============================================================================

cdef inline int _location_at(GameState state, int company_id) noexcept nogil:
    return <int>state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.locations + company_id]


cdef inline int _owner_at(GameState state, int company_id) noexcept nogil:
    return <int>state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.owner_ids + company_id]


cdef inline void _set_location_owner(GameState state, int company_id, int location, int owner_id) noexcept nogil:
    state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.locations + company_id] = <int16_t>location
    state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.owner_ids + company_id] = <int16_t>owner_id


cdef inline int _adjusted_income_at(GameState state, int company_id) noexcept nogil:
    return <int>state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.incomes + company_id]


cdef inline void _set_adjusted_income_at(GameState state, int company_id, int income) noexcept nogil:
    state._data[LAYOUT.companies_offset + COMPANY_OFFSETS.incomes + company_id] = <int16_t>income


# =============================================================================
# COMPANIES-SECTION QUERY HELPERS
# =============================================================================

cdef bint company_owned_by_player(GameState state, int company_id, int player_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_PLAYER and _owner_at(state, company_id) == player_id


cdef bint company_owned_by_fi(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_FI


cdef bint company_owned_by_corp(GameState state, int company_id, int corp_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_CORP and _owner_at(state, company_id) == corp_id


cdef bint company_in_corp_acquisition(GameState state, int company_id, int corp_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_CORP_ACQ and _owner_at(state, company_id) == corp_id


cdef int company_location(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id)


cdef int company_owner_id(GameState state, int company_id) noexcept nogil:
    return _owner_at(state, company_id)


cdef bint company_is_in_deck(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_DECK


cdef bint company_is_excluded(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_EXCLUDED


cdef bint company_is_for_auction(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_AUCTION


cdef bint company_is_revealed(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_REVEALED


cdef bint company_is_removed(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_REMOVED


cdef bint company_is_acquired(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id) == <int>LOC_CORP_ACQ


cdef int company_adjusted_income(GameState state, int company_id) noexcept nogil:
    return _adjusted_income_at(state, company_id)


cdef void set_company_adjusted_income(GameState state, int company_id, int income) noexcept nogil:
    _set_adjusted_income_at(state, company_id, income)


cdef int company_face_value(int company_id) noexcept nogil:
    return COMPANY_FACE_VALUE[company_id]


cdef int company_low_price(int company_id) noexcept nogil:
    return COMPANY_LOW_PRICE[company_id]


cdef int company_high_price(int company_id) noexcept nogil:
    return COMPANY_HIGH_PRICE[company_id]


cdef int company_stars(int company_id) noexcept nogil:
    return COMPANY_STARS[company_id]


cdef int company_base_income(int company_id) noexcept nogil:
    return COMPANY_INCOME[company_id]


cdef bint company_is_last_in_group(int company_id) noexcept nogil:
    return COMPANY_LAST_IN_GROUP[company_id] != 0


cdef int company_synergy(int company_id, int other_company_id) noexcept nogil:
    return COMPANY_SYNERGY[company_id][other_company_id]


cdef int company_sum_player_face_value(GameState state, int player_id) noexcept nogil:
    cdef int company_id
    cdef int total = 0
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_owned_by_player(state, company_id, player_id):
            total += COMPANY_FACE_VALUE[company_id]
    return total


cdef int company_sum_player_adjusted_income(GameState state, int player_id) noexcept nogil:
    cdef int company_id
    cdef int total = 0
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_owned_by_player(state, company_id, player_id):
            total += _adjusted_income_at(state, company_id)
    return total


cdef int company_sum_fi_adjusted_income(GameState state) noexcept nogil:
    cdef int company_id
    cdef int total = 0
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_owned_by_fi(state, company_id):
            total += _adjusted_income_at(state, company_id)
    return total


cdef int company_fill_corp_company_ids(GameState state, int corp_id, bint include_acquisition, int* out_ids) noexcept nogil:
    cdef int company_id
    cdef int count = 0
    for company_id in range(<int>GameConstants.NUM_COMPANIES):
        if company_owned_by_corp(state, company_id, corp_id):
            if out_ids != NULL:
                out_ids[count] = company_id
            count += 1
        elif include_acquisition and company_in_corp_acquisition(state, company_id, corp_id):
            if out_ids != NULL:
                out_ids[count] = company_id
            count += 1
    return count


# =============================================================================
# COMPANY CLASS
# =============================================================================

cdef class Company:
    """
    Entity handle for a single company.

    Instances are created once at module load with their company_id and
    name. The handle is otherwise stateless: every read computes its slot
    inline from the module-level ``LAYOUT`` constant on ``core.state``,
    so a single COMPANIES list can be reused across any number of
    GameState instances at any player count.
    """

    def __cinit__(self, int company_id, str name):
        self.company_id = company_id
        self.name = name

    # =========================================================================
    # LOCATION ACCESS (low-level)
    # =========================================================================

    cdef inline int _get_location(self, GameState state) noexcept nogil:
        return _location_at(state, self.company_id)

    cdef inline int _get_owner_id(self, GameState state) noexcept nogil:
        return _owner_at(state, self.company_id)

    cdef inline void _set_location(self, GameState state, int location, int owner_id) noexcept nogil:
        _set_location_owner(state, self.company_id, location, owner_id)

    # =========================================================================
    # LOCATION QUERIES
    # =========================================================================

    cpdef int get_location(self, GameState state):
        """Return the company's CompanyLocation enum value."""
        return company_location(state, self.company_id)

    cpdef int get_owner_id(self, GameState state):
        """Return owner ID (player_id or corp_id) for ownership locations, -1 otherwise."""
        return company_owner_id(state, self.company_id)

    cpdef bint is_in_deck(self, GameState state):
        return company_is_in_deck(state, self.company_id)

    cpdef bint is_excluded(self, GameState state):
        """True if this company was filtered out at deck setup for the player count."""
        return company_is_excluded(state, self.company_id)

    cpdef bint is_for_auction(self, GameState state):
        return company_is_for_auction(state, self.company_id)

    cpdef bint is_revealed(self, GameState state):
        return company_is_revealed(state, self.company_id)

    cpdef bint is_owned_by_player(self, GameState state, int player_id):
        assert 0 <= player_id < _TURN()._get_num_players(state), \
            f"player_id {player_id} out of range [0, {_TURN()._get_num_players(state)})"
        return company_owned_by_player(state, self.company_id, player_id)

    cpdef bint is_owned_by_fi(self, GameState state):
        return company_owned_by_fi(state, self.company_id)

    cpdef bint is_owned_by_corp(self, GameState state, int corp_id):
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return company_owned_by_corp(state, self.company_id, corp_id)

    cpdef bint is_in_corp_acquisition(self, GameState state, int corp_id):
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return company_in_corp_acquisition(state, self.company_id, corp_id)

    cpdef bint is_removed(self, GameState state):
        return company_is_removed(state, self.company_id)

    cpdef bint is_acquired(self, GameState state):
        """True if the company sits in any corp's acquisition pile this phase."""
        return company_is_acquired(state, self.company_id)

    # =========================================================================
    # TRANSFER OPERATIONS
    # =========================================================================

    cdef void _recalc_after_change(self, GameState state, int location, int owner_id):
        """Recalculate downstream entity fields after a location change.

        Called once for the (location, owner) the company is leaving and
        once for the (location, owner) it lands in. Locations that are not
        owned by an entity (DECK / AUCTION / REVEALED / REMOVED / EXCLUDED)
        are no-ops, so callers can pass them through unconditionally.

        Asserts that any LOC_CORP / LOC_CORP_ACQ owner is an active corp;
        asserts elide under ``python -O`` so this costs nothing in release.
        """
        if location == LOC_CORP or location == LOC_CORP_ACQ:
            assert corp_module.CORPS[owner_id].is_active(state), \
                f"company {self.company_id} associated with inactive corp {owner_id} (loc={location})"
            invalidate_corp_cache(state, owner_id)
        elif location == LOC_PLAYER:
            invalidate_player_cache(state, owner_id)
        elif location == LOC_FI:
            fi_module.FI.calculate_income(state)

    cdef void _move(self, GameState state, int new_loc, int new_owner):
        """Update semantic company state after deck membership is already settled.

        The deck entity owns the live deck array. Callers must route any
        company still marked ``LOC_DECK`` through ``DECK`` first (for
        example, ``DECK.draw``) and only then update the semantic company
        location here.
        """
        cdef int old_loc = self._get_location(state)
        cdef int old_owner = self._get_owner_id(state)
        assert old_loc != LOC_DECK, \
            f"company {self.company_id} still marked LOC_DECK; mutate deck via DECK first"
        self._set_location(state, new_loc, new_owner)
        if old_loc != new_loc or old_owner != new_owner:
            self._recalc_after_change(state, old_loc, old_owner)
        self._recalc_after_change(state, new_loc, new_owner)

    cpdef void transfer_to_player(self, GameState state, int player_id):
        """Transfer company to player ownership."""
        assert 0 <= player_id < _TURN()._get_num_players(state), \
            f"player_id {player_id} out of range [0, {_TURN()._get_num_players(state)})"
        self._move(state, LOC_PLAYER, player_id)

    cpdef void transfer_to_fi(self, GameState state):
        """Transfer company to Foreign Investor ownership."""
        self._move(state, LOC_FI, -1)

    cpdef void transfer_to_corp(self, GameState state, int corp_id):
        """Transfer company to corporation ownership."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        self._move(state, LOC_CORP, corp_id)

    cpdef void transfer_to_corp_acquisition(self, GameState state, int corp_id):
        """Transfer company into a corporation's acquisition pile."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        self._move(state, LOC_CORP_ACQ, corp_id)

    cpdef void move_to_auction(self, GameState state):
        """Make the company available for auction."""
        self._move(state, LOC_AUCTION, -1)

    cpdef void mark_revealed(self, GameState state):
        """Mark the company as revealed this turn (drawn but not auctionable)."""
        self._move(state, LOC_REVEALED, -1)

    cpdef void remove_from_game(self, GameState state):
        """Close the company and remove it from play."""
        self._move(state, LOC_REMOVED, -1)

    cpdef void exclude_from_game(self, GameState state):
        """Mark the company as excluded at game setup.

        LOC_EXCLUDED is distinct from LOC_REMOVED so the engine can tell
        "filtered out at setup for this player count" apart from "closed
        during play". This is set during deck setup for companies that
        do not appear in the live deck.
        """
        self._move(state, LOC_EXCLUDED, -1)

    # =========================================================================
    # STATIC COMPANY DATA
    # =========================================================================

    cpdef int get_face_value(self):
        return company_face_value(self.company_id)

    cpdef int get_low_price(self):
        return company_low_price(self.company_id)

    cpdef int get_high_price(self):
        return company_high_price(self.company_id)

    cpdef int get_stars(self):
        return company_stars(self.company_id)

    cpdef int get_base_income(self):
        return company_base_income(self.company_id)

    cpdef bint is_last_in_group(self):
        return company_is_last_in_group(self.company_id)

    cpdef int get_synergy_with(self, int other_company_id):
        assert 0 <= other_company_id < <int>GameConstants.NUM_COMPANIES, \
            f"other_company_id {other_company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return company_synergy(self.company_id, other_company_id)

    # =========================================================================
    # DYNAMIC DATA FROM STATE
    # =========================================================================

    cpdef int get_adjusted_income(self, GameState state):
        """Adjusted income (after cost of ownership). Stored as a raw int16."""
        return company_adjusted_income(state, self.company_id)

    cpdef void set_adjusted_income(self, GameState state, int income):
        set_company_adjusted_income(state, self.company_id, income)

# =============================================================================
# GLOBAL COMPANY INSTANCES
# =============================================================================

COMPANIES = [Company(i, name) for i, name in enumerate(COMPANY_NAMES)]
COMPANIES_BY_NAME = {name: COMPANIES[i] for i, name in enumerate(COMPANY_NAMES)}
