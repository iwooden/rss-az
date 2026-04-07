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
    # Sizes
    int player_stride
    int corp_stride
    int total_size

    # Metadata offsets
    int active_player_offset
    int num_players_offset
    int phase_offset             # raw integer (0-11), not one-hot
    int coo_level_offset         # raw integer (1-7), not one-hot
    int turn_number_offset

    # Player section (includes share_buys / share_sells per player)
    int players_offset

    # Foreign investor section (cash + income only — ownership lives in
    # company_locations / company_owner_ids)
    int fi_offset

    # Company adjusted incomes (36 raw integers)
    int company_incomes_offset

    # Market availability (27 flags)
    int market_offset

    # Corporation section
    int corps_offset

    # Turn state section
    int turn_offset

    # Deck section
    int deck_top_offset
    int deck_order_offset        # 36 company IDs

    # Company tracking (36 each, enum/integer)
    int company_locations_offset
    int company_owner_ids_offset


cdef struct TurnStateOffsets:
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
    int auction_passed           # 1 flag (has this player left the current auction)
    # Total size of one player's data block (single source of truth for stride)
    int stride


cdef struct CorpFieldOffsets:
    int active
    int cash
    int unissued_shares
    int issued_shares
    int bank_shares
    int income
    int stars
    int share_price
    int acquisition_proceeds
    int in_receivership
    int price_index              # raw integer (0-26)
    int pending_price_move       # raw integer (index delta)
    int raw_revenue
    int synergy_income
    int coo_cost
    int ability_income
    # Total size of one corp's data block (single source of truth for stride)
    int stride

# =============================================================================
# LAYOUT COMPUTATION FUNCTIONS
# =============================================================================

cdef StateLayout compute_layout(int num_players) noexcept nogil
cdef TurnStateOffsets compute_turn_offsets() noexcept nogil
cdef PlayerFieldOffsets compute_player_field_offsets() noexcept nogil
cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil

cdef class GameState:
    cdef int16_t* _data
    cdef public object _array
    cdef StateLayout _layout
    cdef TurnStateOffsets _turn_offsets
    cdef PlayerFieldOffsets _player_fields
    cdef CorpFieldOffsets _corp_fields
    cdef int _num_players

    # Driver config flags (Python-level, not in state array)
    cdef public bint step_mode

    # Internal pointer access (used by entity handles)
    cdef int16_t* _player_ptr(self, int player_id) noexcept nogil
    cdef int16_t* _corp_ptr(self, int corp_id) noexcept nogil
    cdef int16_t* _turn_ptr(self) noexcept nogil
    cdef int _get_active_player(self) noexcept nogil
    cdef void _set_active_player(self, int player_id) noexcept nogil

    # State-level metadata (active player, num_players are not entity-owned)
    cpdef int get_active_player(self)
    cpdef void set_active_player(self, int player_id)
    cpdef int get_num_players(self)

    # Game initialization
    cpdef void initialize_game(self, int seed=*)
