# cython: language_level=3
"""
Declaration file for game state (compact layout for transformer architecture).

Single contiguous int16 array — no visible/hidden split. All values are stored
as raw signed integers (no normalization). int16 is sufficient for every game
quantity (player net worth maxes around 400, share counts are single digits,
sentinels of -1 fit in the negative range). NN features extracted via
get_token_data() which is separate from state storage.
"""

from libc.stdint cimport int16_t
cimport numpy as cnp

# =============================================================================
# LAYOUT STRUCTURES
# =============================================================================

cdef struct StateLayout:
    # Strides
    int player_stride
    int corp_stride

    # NOTE: total_size is intentionally NOT in this struct. It is the only
    # value that depends on num_players, and is computed inline as
    #     LAYOUT.players_offset + LAYOUT.player_stride * num_players
    # Every other offset is a constant shared across all player counts.

    # Foreign investor section (cash + income only — ownership lives in
    # the companies section's locations / owner_ids sub-arrays)
    int fi_offset

    # Companies section (adjusted incomes + locations + owner_ids).
    # Sub-offsets live in the CompanyOffsets struct.
    int companies_offset

    # Market availability (27 flags)
    int market_offset

    # Corporation section
    int corps_offset

    # Turn state section
    int turn_offset

    # Deck section (top index + 36-slot order array). Sub-offsets live
    # in the DeckOffsets struct.
    int deck_offset

    # Player section (LAST: only num_players-dependent slice).
    # Includes share_buys / share_sells / has_passed per player.
    int players_offset


cdef struct TurnStateOffsets:
    # Metadata / phase context (folded into the turn block so StateLayout
    # only describes section offsets)
    int active_player
    int active_corp              # corp_id or -1
    int active_company           # company_id or -1
    int num_players
    int phase                    # raw integer (0-11), not one-hot
    int coo_level                # raw integer (1-7), not one-hot
    int turn_number
    # Global
    int end_card_flipped
    int consecutive_passes
    int cards_remaining
    # Auction
    int auction_price
    int auction_company          # company_id or -1
    int auction_high_bidder      # player_id or -1
    int auction_starter          # player_id or -1
    # Phase remaining tracking
    int dividend_remaining       # 8 corp flags
    int issue_remaining          # 8 corp flags
    int ipo_remaining            # 36 company flags
    # Internal dirty mask for derived player finance caches.
    # Bit i corresponds to player i (0 <= i < MAX_PLAYERS).
    int player_cache_dirty
    # Internal dirty mask for derived corporation caches.
    # Bit i corresponds to corporation i (0 <= i < NUM_CORPS).
    int corp_cache_dirty
    # Total size of the turn state block (single source of truth for layout)
    int size


cdef struct PlayerFieldOffsets:
    int cash
    int net_worth
    int liquidity
    int turn_order               # single integer (not one-hot)
    int owned_shares             # 8 raw counts
    int is_president             # 8 flags
    int round_trips
    int income
    int share_buys               # 8 per-corp buy counts (this turn)
    int share_sells              # 8 per-corp sell counts (this turn)
    int has_passed               # 1 flag (has this player passed in the current phase)
    # Total size of one player's data block
    int size


cdef struct CompanyOffsets:
    # Sub-offsets within the companies section.
    int incomes                  # 36 adjusted income slots (raw int16)
    int locations                # 36 CompanyLocation enum values
    int owner_ids                # 36 owner IDs (player_id / corp_id / -1)
    # Total size of the companies section (used by compute_layout to size it)
    int size


cdef struct DeckOffsets:
    # Sub-offsets within the deck section.
    int top                      # 1 slot: index of top card (-1 = empty)
    int order                    # 36 slots: shuffled company IDs
    # Total size of the deck section (used by compute_layout to size it)
    int size


cdef struct CorpFieldOffsets:
    int active
    int cash
    int unissued_shares
    int issued_shares
    int bank_shares
    int income
    # Stars are split into three slots so cash mutations don't require a
    # full 36-company recompute. company_stars is the sum of COMPANY_STARS
    # over the corp's owned + acquisition-zone companies; cash_stars is
    # floor(cash / 10); total_stars folds those plus the SI ability bonus.
    int total_stars
    int cash_stars
    int company_stars
    int acquisition_proceeds
    int in_receivership
    int price_index              # raw integer (0-26)
    int pending_price_move       # raw integer (index delta)
    int raw_revenue
    int synergy_income
    int coo_cost
    int ability_income
    # Total size of one corp's data block
    int size

# =============================================================================
# LAYOUT COMPUTATION FUNCTIONS
# =============================================================================

cdef StateLayout compute_layout() noexcept nogil
cdef TurnStateOffsets compute_turn_offsets() noexcept nogil
cdef PlayerFieldOffsets compute_player_field_offsets() noexcept nogil
cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil
cdef CompanyOffsets compute_company_offsets() noexcept nogil
cdef DeckOffsets compute_deck_offsets() noexcept nogil


# =============================================================================
# MODULE-LEVEL LAYOUT CONSTANTS
# =============================================================================
#
# Every offset that does not depend on num_players is constant across all
# player counts, so the layout structs can live as singletons at module
# scope. Other modules (entity handles, token extraction, etc.) cimport
# these directly and read offsets without going through a GameState
# instance. The only num_players-dependent quantity is the total buffer
# size, which is `LAYOUT.players_offset + LAYOUT.player_stride * num_players`
# and is computed at the small handful of sites that need it.

cdef StateLayout LAYOUT
cdef TurnStateOffsets TURN_OFFSETS
cdef PlayerFieldOffsets PLAYER_FIELDS
cdef CorpFieldOffsets CORP_FIELDS
cdef CompanyOffsets COMPANY_OFFSETS
cdef DeckOffsets DECK_OFFSETS

cdef class GameState:
    cdef int16_t* _data
    cdef public object _array

    # Driver config flags (Python-level, not in state array)
    cdef public bint step_mode

    # Game initialization
    cpdef void initialize_game(self, int num_players, int seed=*)
