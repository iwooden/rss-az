"""State layout and GameState for the 1889 engine (1889/engine-design.md §3).

One contiguous int16 row of raw values. Blocks are laid out in the order of
``StateLayout``; players come last, so every offset except the total width is
the same at every player capacity. The total is
``LAYOUT.players_offset + PLAYER_FIELDS.size * max_players``.

Fields with a ``# n`` comment are arrays of n slots. Supply and graph-cache
slots have no struct: tile t's copies are at ``supply_offset + t`` and the
(corp, hex) graph entry at ``graph_offset + c * NUM_HEXES + h``.
"""

from libc.stdint cimport int16_t
from libc.string cimport memcpy

cdef enum:
    ROUTE_HEX_WORDS = 4            # 16 hexes a slot: NUM_HEXES is 52


cdef struct StateLayout:
    int global_offset
    int corps_offset
    int privates_offset
    int hexes_offset
    int supply_offset
    int trains_offset
    int graph_offset
    int players_offset             # last: the only capacity-dependent block


cdef struct GlobalOffsets:
    # Game
    int num_players
    int round
    int step
    int return_step                # step a sub-phase returns to, or -1
    int game_phase
    int turn                       # SR number
    int or_num                     # the OR's number in its set; 1 outside ORs (Ruby's round_num)
    int or_count
    int active_player
    int priority_player
    # Bank
    int bank_cash
    int bank_broken                # sticky
    int end_reason                 # EndReason, set when the game ends
    # Auction
    int auction_turn
    int auction_private            # the private in a closed auction (at BID), else -1
    int auction_discount
    int auction_pass_payouts
    # Stock round
    int sr_bought
    int sr_sold_mask
    int sr_sold_then_bought
    # Operating round
    int operating_corp             # or -1
    int or_index
    int or_order                   # NUM_CORPS, -1 padded
    int route_revenue
    # Pending choice, each -1 when unset
    int pending_hex                # TILE's hex
    int pending_corp               # PAR's, SELL_AMOUNT's, or EMERGENCY_SELL_AMOUNT's corp
    int pending_private            # BID's, PRIVATE_PRICE's
    int pending_train_source       # SELECT_TRAIN's and TRAIN_PRICE's TrainSource
    int pending_train_type         # TRAIN_PRICE's train type
    int pending_seller             # ER_BONUS's player
    # Emergency money, caches
    int ebuy_sold_mask
    int ebuy_sold_any
    int market_stamp               # counter for the next market move
    int conn_dirty                 # corp mask
    int route_dirty                # corp mask: route_cache and route_hexes
    int hyp_dirty                  # corp mask: route_after_rust and route_with_next
    int size


cdef struct CorpFieldOffsets:
    int par_index                  # -1 unparred
    int floated
    int cash
    int ipo_shares                 # 10% units
    int pool_shares
    int president                  # player or -1
    int market_row                 # -1 unparred
    int market_col
    int market_stamp               # lower is higher in a market stack
    int tokens_left
    int trains                     # NUM_TRAIN_TYPES counts
    int last_revenue
    int route_cache
    int route_available
    int route_hexes                # ROUTE_HEX_WORDS: the hexes of the best routes, 16 a slot
    int route_after_rust           # the route max once the next rust has happened
    int route_with_next            # the route max with the next train bought
    int size


cdef struct PrivateFieldOffsets:
    int owner                      # an Owner code
    int flags                      # MF: port tile used
    int size


cdef struct HexFieldOffsets:
    int tile                       # 0-39 supply, 40 + h printed
    int rotation                   # canonical
    int tokens                     # corp mask
    int size


cdef struct TrainOffsets:
    int depot                      # NUM_TRAIN_TYPES; -1 unlimited
    int pool                       # NUM_TRAIN_TYPES
    int size


cdef struct PlayerFieldOffsets:
    int cash
    int shares                     # NUM_CORPS, 10% units; the President's Certificate counts 2
    int sold_mask                  # corps sold this SR
    int bids                       # NUM_PRIVATES, 0 when none
    int passed
    int bankrupt
    int size


cdef StateLayout compute_layout() noexcept nogil
cdef GlobalOffsets compute_global_offsets() noexcept nogil
cdef CorpFieldOffsets compute_corp_field_offsets() noexcept nogil
cdef PrivateFieldOffsets compute_private_field_offsets() noexcept nogil
cdef HexFieldOffsets compute_hex_field_offsets() noexcept nogil
cdef TrainOffsets compute_train_offsets() noexcept nogil
cdef PlayerFieldOffsets compute_player_field_offsets() noexcept nogil

# Module constants, computed once at import.
cdef StateLayout LAYOUT
cdef GlobalOffsets GLOBALS
cdef CorpFieldOffsets CORP_FIELDS
cdef PrivateFieldOffsets PRIVATE_FIELDS
cdef HexFieldOffsets HEX_FIELDS
cdef TrainOffsets TRAIN_FIELDS
cdef PlayerFieldOffsets PLAYER_FIELDS

cpdef int get_storage_player_capacity(int state_size_int16) except -1


cdef class GameState:
    cdef int16_t* _data
    cdef public object _array
    cdef public int max_players

    # Policy-only runtime flags (§7), outside the row. Off by default, so the
    # policy set equals the engine-legal set; the adapter sets self-play values.
    cdef public bint private_sale_same_president
    cdef public bint train_sale_same_president
    cdef int _auction_pass_payout_cap          # 0 = off; see the property

    cpdef void initialize_game(
        self, int num_players, int seed=*, int max_players=*, int priority_player=*)


# ``view`` repointed at a raw row, or a new state over it (the extraction batches).
cdef GameState view_row(GameState view, object row)


cdef inline void copy_row(GameState scratch, GameState state) noexcept nogil:
    """Copy the row, caches included, into ``scratch``, a state of its
    capacity: a hypothetical starts here, then changes the copy through the
    entity primitives, which mark the caches it makes stale."""
    memcpy(scratch._data, state._data,
           (LAYOUT.players_offset + PLAYER_FIELDS.size * state.max_players) * sizeof(int16_t))
