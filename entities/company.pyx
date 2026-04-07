# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Company entity implementation.

Companies live in exactly one place at a time. The compact GameState array
tracks every company's location and owner directly via two parallel int16
sub-arrays:

  state._data[LAYOUT.company_locations_offset + cid]  -- CompanyLocation enum
  state._data[LAYOUT.company_owner_ids_offset  + cid]  -- player_id / corp_id / -1

There are no separate "for auction", "revealed", "removed", FI- or
player-side ownership flags anymore — those queries are derived from the
location enum. Transfer operations atomically update the location and
owner_id, then trigger downstream recalculation on the entities whose
income or star totals changed.

Layout offsets are constants pulled from the module-level ``LAYOUT`` on
``core.state``: there is no per-instance offset cache and no initialize()
step. The handle's only state is its ``company_id`` and ``name``.
"""

from libc.stdint cimport int16_t
from core.state cimport GameState, LAYOUT
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
from entities import deck as deck_module
from entities import corp as corp_module
from entities import player as player_module
from entities import fi as fi_module


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
        return <int>state._data[LAYOUT.company_locations_offset + self.company_id]

    cdef inline int _get_owner_id(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.company_owner_ids_offset + self.company_id]

    cdef inline void _set_location(self, GameState state, int location, int owner_id) noexcept nogil:
        state._data[LAYOUT.company_locations_offset + self.company_id] = <int16_t>location
        state._data[LAYOUT.company_owner_ids_offset + self.company_id] = <int16_t>owner_id

    # =========================================================================
    # LOCATION QUERIES
    # =========================================================================

    cpdef int get_location(self, GameState state):
        """Return the company's CompanyLocation enum value."""
        return self._get_location(state)

    cpdef int get_owner_id(self, GameState state):
        """Return owner ID (player_id or corp_id) for ownership locations, -1 otherwise."""
        return self._get_owner_id(state)

    cpdef bint is_in_deck(self, GameState state):
        return self._get_location(state) == LOC_DECK

    cpdef bint is_excluded(self, GameState state):
        """True if this company was filtered out at deck setup for the player count."""
        return self._get_location(state) == LOC_EXCLUDED

    cpdef bint is_for_auction(self, GameState state):
        return self._get_location(state) == LOC_AUCTION

    cpdef bint is_revealed(self, GameState state):
        return self._get_location(state) == LOC_REVEALED

    cpdef bint is_owned_by_player(self, GameState state, int player_id):
        assert 0 <= player_id < state._num_players, \
            f"player_id {player_id} out of range [0, {state._num_players})"
        return (self._get_location(state) == LOC_PLAYER and
                self._get_owner_id(state) == player_id)

    cpdef bint is_owned_by_fi(self, GameState state):
        return self._get_location(state) == LOC_FI

    cpdef bint is_owned_by_corp(self, GameState state, int corp_id):
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return (self._get_location(state) == LOC_CORP and
                self._get_owner_id(state) == corp_id)

    cpdef bint is_in_corp_acquisition(self, GameState state, int corp_id):
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return (self._get_location(state) == LOC_CORP_ACQ and
                self._get_owner_id(state) == corp_id)

    cpdef bint is_removed(self, GameState state):
        return self._get_location(state) == LOC_REMOVED

    cpdef bint is_acquired(self, GameState state):
        """True if the company sits in any corp's acquisition pile this phase."""
        return self._get_location(state) == LOC_CORP_ACQ

    # =========================================================================
    # TRANSFER OPERATIONS
    # =========================================================================

    cdef void _remove_from_deck_if_needed(self, GameState state):
        """If the company is currently in the live deck, splice it out."""
        if self._get_location(state) == LOC_DECK:
            deck_module.DECK.remove(state, self.company_id)

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
            corp_module.CORPS[owner_id].recalculate_stars(state)
            corp_module.CORPS[owner_id].calculate_income(state)
        elif location == LOC_PLAYER:
            player_module.PLAYERS[owner_id].calculate_income(state)
        elif location == LOC_FI:
            fi_module.FI.calculate_income(state)

    cdef void _move(self, GameState state, int new_loc, int new_owner):
        """Splice out of the deck, flip location/owner, and refresh both sides."""
        cdef int old_loc = self._get_location(state)
        cdef int old_owner = self._get_owner_id(state)
        self._remove_from_deck_if_needed(state)
        self._set_location(state, new_loc, new_owner)
        if old_loc != new_loc or old_owner != new_owner:
            self._recalc_after_change(state, old_loc, old_owner)
        self._recalc_after_change(state, new_loc, new_owner)

    cpdef void transfer_to_player(self, GameState state, int player_id):
        """Transfer company to player ownership."""
        assert 0 <= player_id < state._num_players, \
            f"player_id {player_id} out of range [0, {state._num_players})"
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
        self._set_location(state, LOC_EXCLUDED, -1)

    # =========================================================================
    # STATIC COMPANY DATA
    # =========================================================================

    cpdef int get_face_value(self):
        return COMPANY_FACE_VALUE[self.company_id]

    cpdef int get_low_price(self):
        return COMPANY_LOW_PRICE[self.company_id]

    cpdef int get_high_price(self):
        return COMPANY_HIGH_PRICE[self.company_id]

    cpdef int get_stars(self):
        return COMPANY_STARS[self.company_id]

    cpdef int get_base_income(self):
        return COMPANY_INCOME[self.company_id]

    cpdef bint is_last_in_group(self):
        return COMPANY_LAST_IN_GROUP[self.company_id] != 0

    cpdef int get_synergy_with(self, int other_company_id):
        assert 0 <= other_company_id < <int>GameConstants.NUM_COMPANIES, \
            f"other_company_id {other_company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return COMPANY_SYNERGY[self.company_id][other_company_id]

    # =========================================================================
    # DYNAMIC DATA FROM STATE
    # =========================================================================

    cpdef int get_adjusted_income(self, GameState state):
        """Adjusted income (after cost of ownership). Stored as a raw int16."""
        return <int>state._data[LAYOUT.company_incomes_offset + self.company_id]

    cpdef void set_adjusted_income(self, GameState state, int income):
        state._data[LAYOUT.company_incomes_offset + self.company_id] = <int16_t>income

# =============================================================================
# GLOBAL COMPANY INSTANCES
# =============================================================================

COMPANIES = [Company(i, name) for i, name in enumerate(COMPANY_NAMES)]
COMPANIES_BY_NAME = {name: COMPANIES[i] for i, name in enumerate(COMPANY_NAMES)}
