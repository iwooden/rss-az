# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Turn state entity implementation.

Owns the turn-block fields inside the compact GameState plus the three
metadata slots that don't fit anywhere else (phase, coo_level,
turn_number). All values are raw int16 — no one-hot encoding, no
NN-side normalization, no active corp/company duplication. Most of the
"context" state the old MLP needed (active_corp, active_company,
dividend_impact, par tables, synergy previews) is gone entirely; the
transformer reconstructs that material from entity tokens at inference
time.

The methods that survived from the previous design fall into three
buckets:

1. **Single-slot scalars**: phase, coo_level, turn_number,
   end_card_flipped, consecutive_passes, cards_remaining.
2. **Auction tracking**: auction_company / price / high_bidder / starter
   (single integers with -1 sentinels) plus the per-player passed flags.
3. **Phase-remaining flag arrays**: dividend_remaining (8),
   issue_remaining (8), ipo_remaining (36) — used by the engine to track
   which corps/companies still need to act in DIV / ISSUE / IPO.

`set_coo_level` is the one method that does meaningful work beyond a
field write: it cascades the new CoO into adjusted company incomes and
re-derives corp/FI income while marking player finance caches dirty for
lazy refresh on the next read.

Layout offsets come from the module-level ``LAYOUT`` and ``TURN_OFFSETS``
constants on ``core.state``. Single-slot metadata lives at
``LAYOUT.<name>_offset``; turn-block fields live at
``LAYOUT.turn_offset + TURN_OFFSETS.<name>``. The handle has no
per-instance state at all — the singleton TURN can be reused with any
GameState at any player count.
"""

from libc.stdint cimport int16_t

from core.state cimport GameState, LAYOUT, TURN_OFFSETS
from core.data cimport (
    GameConstants,
    COMPANY_INCOME,
    COMPANY_STARS,
    COST_OF_OWNERSHIP,
)
from entities.player cimport invalidate_all_player_caches
# Late entity imports live below the class definition + ``TURN`` singleton
# creation. Peer modules (player / company / corp / fi) use a lazy
# ``_TURN()`` accessor that defers the ``turn_module.TURN`` lookup to first
# call, so import order between peers is unconstrained. The late imports
# below are still needed because TurnState methods reach into the other
# entity modules at call time (e.g. ``set_coo_level`` cascades into
# player / corp / fi income recalcs) — but they no longer dictate a
# cross-module load order.


cdef class TurnState:
    """
    Entity handle for accessing turn state.

    There is only one TurnState instance, created at module load. It has
    no per-instance state; every access reads its slot inline from the
    module-level ``LAYOUT`` / ``TURN_OFFSETS`` constants. All methods
    take a GameState as the first argument.
    """

    # =========================================================================
    # LOW-LEVEL (NOGIL) ACCESSORS
    # =========================================================================

    cdef inline int _get_phase(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]

    cdef inline int _get_coo_level(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]

    cdef inline int _get_auction_price(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price]

    cdef inline int _get_active_player(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player]

    cdef inline void _set_active_player(self, GameState state, int player_id) noexcept nogil:
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player] = <int16_t>player_id

    cdef inline int _get_num_players(self, GameState state) noexcept nogil:
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]

    # =========================================================================
    # ACTIVE PLAYER (state-level metadata, lives in the turn block)
    # =========================================================================

    cpdef int get_active_player(self, GameState state):
        """Return the active player ID."""
        return self._get_active_player(state)

    cpdef void set_active_player(self, GameState state, int player_id):
        """Set the active player ID."""
        assert 0 <= player_id < self._get_num_players(state), \
            f"player_id {player_id} out of range [0, {self._get_num_players(state)})"
        self._set_active_player(state, player_id)

    # =========================================================================
    # NUMBER OF PLAYERS
    # =========================================================================

    cpdef int get_num_players(self, GameState state):
        """Return the number of players in this game."""
        return self._get_num_players(state)

    # =========================================================================
    # PHASE
    # =========================================================================

    cpdef int get_phase(self, GameState state):
        """Return the current game phase (0-11)."""
        return self._get_phase(state)

    cpdef void set_phase(self, GameState state, int phase):
        """Set the current game phase."""
        assert 0 <= phase < <int>GameConstants.NUM_PHASES, \
            f"phase {phase} out of range [0, {<int>GameConstants.NUM_PHASES})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase] = <int16_t>phase

    # =========================================================================
    # COST OF OWNERSHIP LEVEL
    # =========================================================================

    cpdef int get_coo_level(self, GameState state):
        """Return the current cost-of-ownership level (1-7 in game terms)."""
        return self._get_coo_level(state)

    cpdef void set_coo_level(self, GameState state, int level):
        """Set the cost-of-ownership level and cascade adjusted incomes.

        Updating CoO changes every company's adjusted income, which in
        turn changes every active corp's income, every player's cached
        income, and the Foreign Investor's income. Corp/FI values are
        refreshed eagerly; player finance stays lazy behind the single
        cache-dirty bit.
        """
        assert 1 <= level <= <int>GameConstants.NUM_COO_LEVELS, \
            f"coo_level {level} out of range [1, {<int>GameConstants.NUM_COO_LEVELS}]"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level] = <int16_t>level
        self._update_all_company_incomes(state, level)
        self._update_all_corp_incomes(state)
        invalidate_all_player_caches(state)
        fi_module.FI.calculate_income(state)

    cdef void _update_all_company_incomes(self, GameState state, int coo_level):
        """Recompute all 36 adjusted company incomes for the new CoO level.

        Adjusted income = base income - cost-of-ownership at this CoO
        level for the company's star tier. Stored on the per-company
        income slot via the Company entity handle.
        """
        cdef int company_id, adjusted
        cdef int level_index = coo_level - 1
        for company_id in range(<int>GameConstants.NUM_COMPANIES):
            adjusted = (
                COMPANY_INCOME[company_id]
                - COST_OF_OWNERSHIP[level_index][COMPANY_STARS[company_id] - 1]
            )
            company_module.COMPANIES[company_id].set_adjusted_income(state, adjusted)

    cdef void _update_all_corp_incomes(self, GameState state):
        """Recalculate income for every active corporation."""
        cdef int corp_id
        for corp_id in range(<int>GameConstants.NUM_CORPS):
            if corp_module.CORPS[corp_id].is_active(state):
                corp_module.CORPS[corp_id].calculate_income(state)

    # =========================================================================
    # TURN NUMBER
    # =========================================================================

    cpdef int get_turn_number(self, GameState state):
        """Return the current turn number (1-indexed)."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.turn_number]

    cpdef void set_turn_number(self, GameState state, int turn):
        """Set the current turn number."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.turn_number] = <int16_t>turn

    # =========================================================================
    # END CARD FLIPPED
    # =========================================================================

    cpdef bint is_end_card_flipped(self, GameState state):
        """Return True if the end card has been flipped."""
        return state._data[LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped] == 1

    cpdef void set_end_card_flipped(self, GameState state, bint flipped):
        """Set whether the end card has been flipped."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped] = <int16_t>(1 if flipped else 0)

    # =========================================================================
    # CONSECUTIVE PASSES (INVEST phase)
    # =========================================================================

    cpdef int get_consecutive_passes(self, GameState state):
        """Return the consecutive-pass count for the INVEST phase."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes]

    cpdef void set_consecutive_passes(self, GameState state, int passes):
        """Set the consecutive-pass count."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes] = <int16_t>passes

    cpdef void increment_consecutive_passes(self, GameState state):
        """Increment the consecutive-pass count by one."""
        cdef int slot = LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes
        state._data[slot] = <int16_t>(<int>state._data[slot] + 1)

    cpdef void clear_consecutive_passes(self, GameState state):
        """Reset the consecutive-pass count to zero (any non-pass action)."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes] = 0

    # =========================================================================
    # CARDS REMAINING (deck mirror)
    # =========================================================================

    cpdef int get_cards_remaining(self, GameState state):
        """Return the number of cards left in the live deck."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining]

    cpdef void set_cards_remaining(self, GameState state, int count):
        """Set the cards-remaining mirror.

        The deck owns deck_top; this slot is the materialized count it
        pushes out so phases / NN tokens can read deck size without
        touching the deck array directly.
        """
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining] = <int16_t>count

    # =========================================================================
    # AUCTION STATE
    # =========================================================================

    cpdef int get_auction_company(self, GameState state):
        """Return the company currently being auctioned, or -1 if none."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_company]

    cpdef void set_auction_company(self, GameState state, int company_id):
        """Set the company currently being auctioned."""
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_company] = <int16_t>company_id

    cpdef void clear_auction_company(self, GameState state):
        """Clear the auction company sentinel."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_company] = -1

    cpdef int get_auction_price(self, GameState state):
        """Return the current auction price (0 when no auction is active)."""
        return self._get_auction_price(state)

    cpdef void set_auction_price(self, GameState state, int price):
        """Set the current auction price."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price] = <int16_t>price

    cpdef int get_auction_high_bidder(self, GameState state):
        """Return the current high bidder, or -1 if none."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder]

    cpdef void set_auction_high_bidder(self, GameState state, int player_id):
        """Set the current auction high bidder."""
        assert 0 <= player_id < self._get_num_players(state), \
            f"player_id {player_id} out of range [0, {self._get_num_players(state)})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder] = <int16_t>player_id

    cpdef void clear_auction_high_bidder(self, GameState state):
        """Clear the high-bidder sentinel."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder] = -1

    cpdef int get_auction_starter(self, GameState state):
        """Return the player who started the current auction, or -1 if none."""
        return <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_starter]

    cpdef void set_auction_starter(self, GameState state, int player_id):
        """Set the player who started the current auction."""
        assert 0 <= player_id < self._get_num_players(state), \
            f"player_id {player_id} out of range [0, {self._get_num_players(state)})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_starter] = <int16_t>player_id

    cpdef void clear_auction_starter(self, GameState state):
        """Clear the auction-starter sentinel."""
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_starter] = -1

    # =========================================================================
    # PHASE-REMAINING FLAGS
    # =========================================================================

    cpdef bint is_dividend_remaining(self, GameState state, int corp_id):
        """Return True if the corp still needs dividend processing."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return state._data[LAYOUT.turn_offset + TURN_OFFSETS.dividend_remaining + corp_id] == 1

    cpdef void set_dividend_remaining(self, GameState state, int corp_id, bint remaining):
        """Set whether the corp still needs dividend processing."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.dividend_remaining + corp_id] = <int16_t>(1 if remaining else 0)

    cpdef bint is_issue_remaining(self, GameState state, int corp_id):
        """Return True if the corp still needs issue processing."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        return state._data[LAYOUT.turn_offset + TURN_OFFSETS.issue_remaining + corp_id] == 1

    cpdef void set_issue_remaining(self, GameState state, int corp_id, bint remaining):
        """Set whether the corp still needs issue processing."""
        assert 0 <= corp_id < <int>GameConstants.NUM_CORPS, \
            f"corp_id {corp_id} out of range [0, {<int>GameConstants.NUM_CORPS})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.issue_remaining + corp_id] = <int16_t>(1 if remaining else 0)

    cpdef bint is_ipo_remaining(self, GameState state, int company_id):
        """Return True if the company still needs IPO processing."""
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        return state._data[LAYOUT.turn_offset + TURN_OFFSETS.ipo_remaining + company_id] == 1

    cpdef void set_ipo_remaining(self, GameState state, int company_id, bint remaining):
        """Set whether the company still needs IPO processing."""
        assert 0 <= company_id < <int>GameConstants.NUM_COMPANIES, \
            f"company_id {company_id} out of range [0, {<int>GameConstants.NUM_COMPANIES})"
        state._data[LAYOUT.turn_offset + TURN_OFFSETS.ipo_remaining + company_id] = <int16_t>(1 if remaining else 0)

    # =========================================================================
    # TURN ORDER NAVIGATION
    # =========================================================================

    cpdef int find_player_at_position(self, GameState state, int position):
        """Return the player_id at the given turn-order position.

        Asserts that `position` is in range and that some player matches
        — every valid position has exactly one player by construction, so
        the post-loop assert is purely defensive.
        """
        assert 0 <= position < self._get_num_players(state), \
            f"position {position} out of range [0, {self._get_num_players(state)})"
        cdef int player_id
        for player_id in range(self._get_num_players(state)):
            if player_module.PLAYERS[player_id].get_turn_order(state) == position:
                return player_id
        raise AssertionError(f"no player found at turn-order position {position}")

    cpdef void advance_to_next_bidder(self, GameState state):
        """Advance the active player to the next non-passed bidder.

        Used during auction bidding to skip players who have left. Asserts
        that at least one non-passed bidder exists — callers are expected
        to guarantee this invariant before calling.
        """
        cdef int current_player = self._get_active_player(state)
        cdef int current_position = player_module.PLAYERS[current_player].get_turn_order(state)
        cdef int next_position, candidate
        cdef int checked = 0

        while checked < self._get_num_players(state):
            next_position = (current_position + 1) % self._get_num_players(state)
            candidate = self.find_player_at_position(state, next_position)

            if not player_module.PLAYERS[candidate].has_passed_auction(state):
                self._set_active_player(state, candidate)
                return

            current_position = next_position
            checked += 1

        raise AssertionError("advance_to_next_bidder called with no active bidders")

    cpdef void set_active_player_after(self, GameState state, int player_id):
        """Set the active player to the next player after `player_id`."""
        assert 0 <= player_id < self._get_num_players(state), \
            f"player_id {player_id} out of range [0, {self._get_num_players(state)})"
        cdef int position = player_module.PLAYERS[player_id].get_turn_order(state)
        cdef int next_position = (position + 1) % self._get_num_players(state)
        cdef int next_player = self.find_player_at_position(state, next_position)
        self._set_active_player(state, next_player)


# =============================================================================
# GLOBAL TURN STATE INSTANCE
# =============================================================================

# Single TurnState instance — must be created BEFORE the entity imports below
# so peer modules (player / company / corp / fi) can take a typed cdef
# reference to it during their own initialization, even mid-cycle.
TURN = TurnState()


# =============================================================================
# LATE ENTITY IMPORTS
# =============================================================================
#
# Methods on TurnState reach into other entity modules (player, company, corp,
# fi) at *call* time, so the imports can live below the class definition and
# the singleton creation. Placing them here keeps ``TURN`` populated before
# any cyclic load of those entity modules begins.

from entities import player as player_module
from entities import company as company_module
from entities import corp as corp_module
from entities import fi as fi_module
