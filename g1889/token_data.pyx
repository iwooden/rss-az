"""Token extraction for 1889; see token_data.pxd, 1889/extraction-design.md §5,
and, for layout 2, 1889/v1-design.md §5 and token-data-1889.md.

Rows (§5.1): GLOBAL, PRICE, DIVIDEND, the corps, the privates, the train
types, the hexes, the candidates, and the players. Slot 0 of every row is its
key mask and slot 1 its ``is_selected`` (§5.2).

**Two layouts.** Layout 1 is pinned for ``nn/transformer-1889.py``
(``tests/g1889/test_layout1_parity.py``): its fills below must not change.
Layout 2 fills GLOBAL, PRICE, TRAIN, HEX, PRIVATE, and CANDIDATE as layout 1
does and then adds to them (``_fill_global2`` and the like), and has its own
fills for DIVIDEND, CORP, and PLAYER, whose slots move. Its slots are the
``*2_`` names; where a row keeps layout 1's slots it uses layout 1's names.

Per-choice features (§7) describe the decision's engine-legal choices, at
that decision only. PRICE's amounts, DIVIDEND's outcomes, and the sale prices
(CORP 15-19) are pure, and HEX 102 counts TRACK's lay candidates. CANDIDATE
rows at TILE and ER_BONUS, HEX 103-104 at TOKEN, and TRAIN 10-12 at
SELECT_TRAIN and DISCARD try each choice on the call's scratch state (§4): a
copy of the row, changed through the entity primitives, whose caches are
recomputed for what the features read. The live row changes only by its own
caches.

The decision corp is §5.2's, and the actor ``active_player``; their features
are zero at GAME_OVER, which has neither. Divisors follow §5.2: ¥1000 for
cash and train-scale money, ¥100 for prices, values, bids, and revenues,
¥7000 for the bank, and counts by their natural maximum.
"""

import numpy as np

from libc.math cimport log1p
from libc.stdint cimport int16_t, uint16_t, uint32_t
from libc.string cimport memset

from g1889.actions cimport (
    ACTION_PAR, ACTION_PRICE_POINT, ACTION_TRADE_IN, ACTION_TRAIN_TYPE, ActionInfo, LayCandidate, can_buy_private,
    can_exchange_dr, can_token, decode_action, emergency_need, emergency_sale_limit, enumerate_lay_candidates,
    enumerate_legal_actions, er_candidates, get_decision_phase, in_emergency, lay_candidates, must_buy_train,
    price_amount, sale_limit,
)
from g1889.constants cimport (
    ACTION_SIZE_BID, ACTION_SIZE_PAR, ACTION_SIZE_PRIVATE_PRICE, ACTION_SIZE_TRAIN_PRICE, DPHASE_BID,
    DPHASE_BUY_TRAIN, DPHASE_DISCARD, DPHASE_DIVIDEND, DPHASE_EMERGENCY_SELL_AMOUNT, DPHASE_ER_BONUS, DPHASE_PAR,
    DPHASE_PORT, DPHASE_PRIVATE_PRICE, DPHASE_SELECT_TRAIN, DPHASE_SELL_AMOUNT, DPHASE_STOCK, DPHASE_TILE,
    DPHASE_TOKEN, DPHASE_TRACK, DPHASE_TRAIN_PRICE, MAX_ACTION_SIZE, MAX_SALE_SHARES, NUM_DECISION_PHASES,
    OWNER_CLOSED, OWNER_CORP, OWNER_NOT_IN_PLAY, OWNER_PLAYER, OWNER_UNSOLD, ROUND_AUCTION, ROUND_GAME_OVER,
    ROUND_OPERATING, ROUND_STOCK, SHARES_PER_CORP, STEP_BUY_PRIVATE_POST, STEP_BUY_PRIVATE_PRE, TRAIN_SOURCE_CORP,
)
from g1889.data.tables cimport (
    CERT_LIMIT, CORP_HOME, D_TRAIN, ENDPOINT_PAIRS, ER_HEX, FLOAT_PERCENT, HEX_BLOCKER, HEX_EXIT_MASK,
    HEX_HOME_CORP, HEX_NEIGHBOR, LABEL_NONE, MARKET_PRICE, MAX_CITY_SLOTS, MAX_CORP_TOKENS, MAX_LAY_CANDIDATES,
    NUM_CORPS, NUM_GAME_PHASES, NUM_HEXES, NUM_PAIRS, NUM_PAR_PRICES, NUM_PRIVATES, NUM_RCS, NUM_SIDES,
    NUM_SUPPLY_TILES, NUM_TRAIN_TYPES, PAR_PRICE, PHASE_ON_TRAIN, PHASE_ORS, PHASE_TILE_COLORS,
    PHASE_TRAIN_LIMIT, PRIVATE_DR, PRIVATE_MF, PRIVATE_VALUE, RC_NONE, TERRAIN_MOUNTAIN,
    TERRAIN_WATER, TILE_BLOCKS_LAY, TILE_COLOR, TILE_COST, TILE_COUNT, TILE_EXITS, TILE_LABEL, TILE_MASK, TILE_RC,
    TILE_SLOTS, TILE_TERRAIN, TRAIN_PRICE, TRAIN_RUSTS_ON, UPG_ROT, UPG_START, UPG_TILE,
)
from g1889.entities.bank cimport bank_broken, bank_cash
from g1889.entities.corp cimport (
    corp_cash, corp_emergency_sale, corp_floated, corp_floated_mask, corp_ipo_shares, corp_par_index,
    corp_par_price, corp_pool_shares, corp_president, corp_token_price, corp_tokens_left,
)
from g1889.entities.graph cimport GRAPH_NODE, GRAPH_SIDES, graph_refresh, graph_route_available
from g1889.entities.hexmap cimport (
    hex_paths, hex_rotation, hex_tile, hex_tokens, map_blocked_for, map_free_slots, map_has_slot_for,
    map_center_bounds, map_lay_cost, map_put_tile, map_put_token, map_revenue, map_token_count, map_touch_mask,
    supply_count,
)
from g1889.entities.market cimport (
    MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, MOVE_UP, market_col, market_moved, market_precedes, market_price, market_row,
    market_sort_by_price, market_type,
)
from g1889.entities.phase cimport game_phase, next_rust_phase, phase_on_train_bought
from g1889.entities.player cimport (
    player_available, player_bid, player_cash, player_cert_count, player_passed, player_shares, player_sold_mask,
    player_value,
)
from g1889.entities.private cimport (
    MF_PORT_USED, private_bid_count, private_blocks_hex, private_flags, private_high_bid, private_min_bid,
    private_owner, private_price, private_revenue, topmost_private,
)
from g1889.entities.trains cimport (
    corp_train_count, corp_train_total, depot_count, pool_count, trains_cheapest_type, trains_depot_types,
    trains_first_crowded, trains_limit, trains_set_count,
)
from g1889.entities.turn cimport (
    turn_active_player, turn_auction_discount, turn_auction_pass_payouts, turn_auction_private,
    turn_ebuy_sold_any, turn_ebuy_sold_mask, turn_num_players, turn_operating_corp, turn_pending_corp,
    turn_pending_hex, turn_pending_private, turn_pending_train_source, turn_pending_train_type, turn_return_step,
    turn_round, turn_route_revenue, turn_sr_bought_corp, turn_sr_sold_mask, turn_turn,
)
from g1889.routes.solver cimport (
    MAX_ROUTES, route_max, route_max_after_purchase, routes_refresh_hypotheticals, solve_routes_via,
)
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, PLAYER_FIELDS, GameState, copy_row, view_row


# =============================================================================
# SLOTS AND DIVISORS
# =============================================================================

# Slots by row type, as §5's tables number them; "+ x" marks a block indexed by x.
cdef enum:
    # GLOBAL (§5.3)
    G_PHASE = 1                     # + decision phase
    G_ROUND = 20                    # + round: auction, stock, operating
    G_GAME_PHASE = 23               # + game phase
    G_FIRST_SR = 29
    G_OR_NUM = 30                   # + the OR's number in its set - 1
    G_OR_COUNT = 33
    G_BANK_CASH = 34
    G_BANK_BROKEN = 35
    G_PLAYERS = 36                  # + player count - 3
    G_PASS_PAYOUTS = 39
    G_DISCOUNT = 40
    G_TRAIN_SOURCE = 41             # + the depot, the pool, a corp

    # PRICE and DIVIDEND (§5.4)
    PR_KIND = 1                     # + BID, PAR, TRAIN_PRICE, PRIVATE_PRICE
    PR_AMOUNTS = 5                  # + phase-local id
    PR_REFERENCE = 22
    PR_FUNDS = 23
    DV_REVENUE = 1
    DV_PER_SHARE = 2
    DV_PAYOUT_PRICE = 3
    DV_WITHHOLD_PRICE = 4
    DV_TREASURY = 5
    DV_PRESIDENT = 6
    DV_POOL = 7

    # CORP (§5.5)
    C_SELECTED = 1
    C_SELLER = 2
    C_SALE = 3
    C_PARRED = 4
    C_FLOATED = 5
    C_HOME = 6
    C_PAR_PRICE = 7
    C_PRICE = 8
    C_ROW = 9
    C_COL = 10
    C_ZONE = 11                     # + MarketType
    C_SALE_PRICES = 15              # + shares sold - 1
    C_SOLD_OUT = 20
    C_CASH = 21
    C_IPO = 22
    C_POOL = 23
    C_TOKENS = 24
    C_TOKEN_PRICE = 25
    C_TRAINS = 26                   # + train type
    C_CROWDED = 32
    C_LAST_REVENUE = 33
    C_ROUTE_MAX = 34
    C_ROUTE_AVAILABLE = 35
    C_MUST_BUY = 36
    C_EMERGENCY = 37
    C_OR_POSITION = 38
    C_OPERATED = 39
    C_HEXES = 40
    C_CENTERS = 41
    C_REVENUE = 42
    C_SLOTS = 43
    C_ACTOR_SOLD = 44
    C_TURN_SOLD = 45
    C_EMERGENCY_SOLD = 46
    C_BOUGHT = 47
    C_SALE_LIMIT = 48
    C_EMERGENCY_SALE = 49
    C_PRIVATES = 50

    # PRIVATE (§5.6)
    P_SELECTED = 1
    P_TOPMOST = 2
    P_UNSOLD = 3
    P_PLAYER_OWNED = 4
    P_CORP_OWNED = 5
    P_CLOSED = 6
    P_FACE = 7
    P_REVENUE = 8
    P_PRICE = 9
    P_HIGH_BID = 10
    P_BIDS = 11
    P_MIN_BID = 12
    P_ACTOR_BID = 13
    P_BLOCKING = 14
    P_PORT_USED = 15
    P_BUYABLE = 16
    P_EXCHANGE = 17

    # TRAIN (§5.6)
    T_SELECTED = 1
    T_PRICE = 2
    T_DEPOT = 3
    T_UNLIMITED = 4
    T_POOL = 5
    T_DEPOT_SELLS = 6
    T_CHEAPEST = 7
    T_RUSTED = 8
    T_RUSTS_NEXT = 9
    T_ADD_DELTA = 10
    T_TRADE_IN_DELTA = 11
    T_DISCARD_DELTA = 12

    # The tile block (§5.7), from TILE_BLOCK
    TB_SUPPLY = 0                   # + supply tile
    TB_PRINTED = 40
    TB_PATHS = 41                   # + pair
    TB_COLOR = 62                   # + TileColor
    TB_CENTER = 68                  # + RcType - 1
    TB_REVENUE = 71
    TB_SLOTS = 72
    TB_LABEL = 73                   # + TileLabel - 1

    # HEX (§5.7)
    H_SELECTED = 1
    H_D_REVENUE = 78
    H_MOUNTAIN = 79
    H_WATER = 80
    H_TOKENS = 81
    H_FREE_SLOTS = 82
    H_RESERVED = 83
    H_COST = 84
    H_BLOCKED = 85
    H_DANGLING = 86                 # + side
    H_UPGRADES = 92
    H_UPGRADES_ANY_PHASE = 93
    H_SIDES = 94                    # + side
    H_CENTER = 100
    H_BLOCKED_FOR = 101
    H_CANDIDATES = 102
    H_TOKENABLE = 103
    H_TOKEN_DELTA = 104

    # CANDIDATE (§5.8): the tile block at TILE_BLOCK, as HEX's
    CA_ER = 1
    CA_ADDED = 78                   # + pair
    CA_COST = 99
    CA_COPIES = 100
    CA_HEXES = 101
    CA_CENTERS = 102
    CA_REVENUE = 103
    CA_ROUTE_DELTA = 104
    CA_ROUTE_AVAILABLE = 105
    CA_OTHERS = 106
    CA_OTHER_CENTERS = 107

    # PLAYER (§5.9)
    PL_SELECTED = 1
    PL_SEAT = 2                     # + seats after the actor
    PL_PRIORITY = 7
    PL_CASH = 8
    PL_VALUE = 9
    PL_CERTS_OF_LIMIT = 10
    PL_CERTS = 11
    PL_SHARES = 12                  # + corp
    PL_SOLD = 19                    # + corp
    PL_BIDS = 26                    # + private
    PL_PASSED = 33
    PL_BANKRUPT = 34
    PL_PRESIDENCIES = 35
    PL_PRIVATES = 36
    PL_EMERGENCY_SOLD = 37

# Layout 2 (1889/v1-design.md §5; token-data-1889.md). GLOBAL, PRICE, TRAIN,
# and CANDIDATE keep layout 1's slots and add these; HEX and PRIVATE have no
# new slot; DIVIDEND, CORP, and PLAYER are numbered anew.
cdef enum:
    # GLOBAL
    G2_PRE = 44
    G2_POST = 45
    G2_ORS_LEFT = 46
    G2_BANK_ORS = 47
    G2_TRAIN_LIMIT = 48
    G2_NEXT_TRAIN_LIMIT = 49
    G2_SUPPLY = 50                  # + supply tile

    # PRICE
    PR2_FLOAT_SHARES = 24           # + par index

    # DIVIDEND
    DV2_REVENUE = 1
    DV2_PAYOUT_PRICE = 2
    DV2_WITHHOLD_PRICE = 3
    DV2_TREASURY = 4
    DV2_PRESIDENT = 5
    DV2_POOL = 6
    DV2_NEXT_PRICE = 7
    DV2_WITHHOLD_SHORT = 8
    DV2_PAYOUT_SHORT = 9
    DV2_ROOM = 10
    DV2_WITH_NEXT = 11
    DV2_AFTER_RUST = 12

    # CORP: slots 1-37 are layout 1's
    C2_RANK = 38                    # + operating rank
    C2_OPERATED = 45
    C2_HEXES = 46
    C2_CENTERS = 47
    C2_REVENUE = 48
    C2_SLOTS = 49
    C2_ACTOR_SOLD = 50
    C2_TURN_SOLD = 51
    C2_EMERGENCY_SOLD = 52
    C2_BOUGHT = 53
    C2_SALE_LIMIT = 54
    C2_EMERGENCY_SALE = 55
    C2_PRIVATES = 56
    C2_OPERATING = 57
    C2_PAYOUT_PRICE = 58
    C2_WITHHOLD_PRICE = 59
    C2_SOLD_OUT_PRICE = 60
    C2_TRAINS_OF_LIMIT = 61
    C2_ROOM = 62
    C2_SURVIVING = 63
    C2_NEXT_ROOM = 64
    C2_AFTER_RUST = 65
    C2_WITH_NEXT = 66
    C2_SHORTFALL = 67
    C2_FLOAT_SHARES = 68
    C2_FLOAT_COST = 69
    C2_PRESIDENT_FLOATS = 70
    C2_ACTOR_FLOATS = 71
    C2_PRESIDENT_SOLD = 72
    C2_EMERGENCY_NEED = 73
    C2_LOG_PRICE = 74
    C2_LOG_CASH = 75
    C2_LOG_ROUTE = 76

    # TRAIN
    T2_SELLER_DELTA = 13
    T2_STARTS_PHASE = 14

    # CANDIDATE
    CA2_NEXT_DELTA = 108

    # PLAYER: slots 1-10 are layout 1's
    PL2_SHARES = 11                 # + corp
    PL2_SOLD = 18                   # + corp
    PL2_BIDS = 25                   # + private - 1: TR, the topmost, is never bid on
    PL2_PASSED = 31
    PL2_PRESIDENCIES = 32
    PL2_PRIVATES = 33
    PL2_EMERGENCY_SOLD = 34
    PL2_FLOATS_ALONE = 35
    PL2_RESUMES = 36
    PL2_INCOME = 37
    PL2_LOG_CASH = 38
    PL2_LOG_VALUE = 39

cdef float MONEY_DIV = 1000.0
cdef float PRICE_DIV = 100.0
cdef float BANK_DIV = 7000.0
cdef float LOG_DIV = 5.0
# Shares that leave the IPO to float a corp, their cost at the lowest par,
# and the largest supply of one tile.
cdef int FLOAT_SHARES = <int>FLOAT_PERCENT * <int>SHARES_PER_CORP // 100
cdef int LOWEST_PAR = min([PAR_PRICE[i] for i in range(<int>NUM_PAR_PRICES)])
cdef int LOWEST_FLOAT = FLOAT_SHARES * LOWEST_PAR
cdef float SUPPLY_DIV = max([TILE_COUNT[t] for t in range(<int>NUM_SUPPLY_TILES)])


cdef struct _Context:
    int num_players
    int phase                       # the decision phase, -1 at none
    int game_round
    int game_phase
    int actor                       # -1 at none
    int corp                        # the decision corp, -1 at none
    # Layout 2 only (``_context2``)
    int operating                   # the operating corp, -1 between turns and outside an OR
    int rank[7]                     # NUM_CORPS: each corp's operating rank, -1 for none
    int next_type                   # the next train: the cheapest on sale
    int limit                       # the train limit now, and in the next phase
    int next_limit
    int rust_phase                  # the next phase that rusts a rank, or -1
    bint emergency                  # the decision corp is in one,
    int need                        # and what its president must still raise for the next train


# A corp's reach (GRAPH): hexes with a reached side, reached centers, their
# revenue now, and those with a slot for the corp.
cdef struct _Reach:
    int hexes
    int centers
    int revenue
    int slots


cdef inline int _global(GameState state, int field) noexcept nogil:
    return <int>state._data[LAYOUT.global_offset + field]


cdef inline int _corp_field(GameState state, int corp, int field) noexcept nogil:
    return <int>state._data[LAYOUT.corps_offset + corp * CORP_FIELDS.size + field]


cdef int _decision_corp(GameState state, int phase) noexcept nogil:
    """§5.2: the pending corp at PAR, the first crowded corp at DISCARD, and the
    operating corp at every other OR decision; none in the auction and the SR."""
    if phase == DPHASE_PAR:
        return turn_pending_corp(state)
    if phase == DPHASE_DISCARD:
        return trains_first_crowded(state)
    return turn_operating_corp(state) if phase >= 0 and turn_round(state) == ROUND_OPERATING else -1


cdef inline int _bid_private(GameState state) noexcept nogil:
    """The private BID is for: a closed auction's, else the selected one."""
    cdef int p = turn_auction_private(state)
    return p if p >= 0 else turn_pending_private(state)


cdef inline int _moved_price(GameState state, int corp, int move) noexcept nogil:
    """The corp's share price one ``move`` away (``market_moved``)."""
    cdef int row, col
    market_moved(market_row(state, corp), market_col(state, corp), move, 1, &row, &col)
    return MARKET_PRICE[row][col]


cdef inline float _log_yen(int yen) noexcept nogil:
    """A log-scaled amount: ln(1 + ¥/100) / 5, which is 1.0 near ¥14,700."""
    return <float>log1p(max(yen, 0) / 100.0) / LOG_DIV


cdef inline int _payout(int shares, int revenue) noexcept nogil:
    """What ``shares`` earn of a payout of ``revenue``, rounded up, as the bank
    pays them (``_pay``, phases/dividend.pyx)."""
    return (shares * revenue + <int>SHARES_PER_CORP - 1) // <int>SHARES_PER_CORP


# =============================================================================
# PUBLIC API
# =============================================================================

cpdef int get_num_tokens(int max_players) noexcept nogil:
    """The fixed rows, then one per player."""
    return <int>ROW_PLAYER + max_players


cdef int _check_layout(int layout_version) except -1:
    if layout_version != 1 and layout_version != 2:
        raise ValueError(f"Unsupported 1889 token layout version: {layout_version}")
    return 0


cpdef int get_token_dim(int layout_version) except -1:
    _check_layout(layout_version)
    return <int>TOKEN_DIM if layout_version == 1 else <int>TOKEN_DIM_2


cpdef object get_token_widths(int max_players, int layout_version):
    """Each row's width, a uint8 array over ``get_num_tokens(max_players)`` rows."""
    _check_layout(layout_version)
    assert 3 <= max_players <= 5, f"get_token_widths: max_players must be 3-5, got {max_players}"
    by_type = ((<int>TW_GLOBAL, <int>TW_PRICE, <int>TW_DIVIDEND, <int>TW_CORP, <int>TW_PRIVATE, <int>TW_TRAIN,
                <int>TW_HEX, <int>TW_CANDIDATE, <int>TW_PLAYER) if layout_version == 1 else
               (<int>TW2_GLOBAL, <int>TW2_PRICE, <int>TW2_DIVIDEND, <int>TW2_CORP, <int>TW2_PRIVATE, <int>TW2_TRAIN,
                <int>TW2_HEX, <int>TW2_CANDIDATE, <int>TW2_PLAYER))
    counts = (1, 1, 1, <int>NUM_CORPS, <int>NUM_PRIVATES, <int>NUM_TRAIN_TYPES, <int>NUM_HEXES,
              <int>MAX_LAY_CANDIDATES, max_players)
    return np.repeat(np.array(by_type, dtype=np.uint8), counts)


cdef void extraction_prologue(GameState state) noexcept nogil:
    """Refresh every corp's GRAPH entries and route cache, writing them into
    the row as the engine's lazy getters do (§4). The fills then read the
    graph block, ``route_available``, ``route_cache``, and ``route_hexes`` raw."""
    cdef int c
    for c in range(<int>NUM_CORPS):
        graph_refresh(state, c)
        route_max(state, c)


cdef void _prologue2(GameState state, GameState scratch) noexcept nogil:
    """Layout 2's: the same, then each corp's hypothetical route maxes, which
    are solved on ``scratch`` (``_scratch_for`` gives one while any is stale)."""
    cdef int c
    extraction_prologue(state)
    if _global(state, GLOBALS.hyp_dirty):
        for c in range(<int>NUM_CORPS):
            routes_refresh_hypotheticals(state, scratch, c)


cdef int tile_candidates(GameState state, int phase, int* hex, LayCandidate* buf) noexcept nogil:
    """Write the decision's lay candidates to ``buf`` (MAX_LAY_CANDIDATES
    slots) in their ids' order, and their hex to ``hex``; return the count. At
    TILE, the operating corp's on the pending hex, candidate k being id k; at
    ER_BONUS, ER's on C4, id 1 + k (§7.1); none at other decisions."""
    if phase == DPHASE_ER_BONUS:
        hex[0] = <int>ER_HEX
        return er_candidates(state, buf)
    hex[0] = turn_pending_hex(state)
    if phase != DPHASE_TILE or hex[0] < 0 or turn_operating_corp(state) < 0:
        return 0
    return enumerate_lay_candidates(state, hex[0], buf)


cdef bint _has_hypotheticals(int phase) noexcept nogil:
    """Whether the decision's per-choice features try its choices on the scratch state."""
    return (phase == DPHASE_TILE or phase == DPHASE_ER_BONUS or phase == DPHASE_TOKEN
            or phase == DPHASE_SELECT_TRAIN or phase == DPHASE_DISCARD)


cdef bint _needs_scratch(GameState state, int layout) noexcept nogil:
    """Whether the state's fill uses the scratch state: its decision tries
    choices on it, or, in layout 2, a hypothetical route max is stale or the
    decision shows what the trains on sale would earn (TRAIN 10 and 13)."""
    cdef int phase = get_decision_phase(state)
    if _has_hypotheticals(phase):
        return True
    return layout == 2 and (_global(state, GLOBALS.hyp_dirty) != 0 or phase == DPHASE_BUY_TRAIN
                            or phase == DPHASE_DIVIDEND or phase == DPHASE_TRAIN_PRICE)


cdef GameState _scratch_for(GameState scratch, GameState state, int layout):
    """The call's scratch state for ``state`` (§4): ``scratch`` while it has the
    state's capacity or the state's fill needs none, else a new one, a copy of
    the row. One serves every hypothetical of a call."""
    if scratch is not None and scratch.max_players == state.max_players or not _needs_scratch(state, layout):
        return scratch
    return GameState.from_array(state._array, turn_num_players(state), state.max_players)


cdef int resolve_max_players(int num_players, int max_players, str func) except -1:
    if max_players == 0:
        max_players = num_players
    assert 3 <= num_players <= max_players <= 5, \
        f"{func}: need 3 <= num_players {num_players} <= max_players {max_players} <= 5"
    return max_players


cdef GameState bind_row(GameState view, object row, int max_players, str func):
    """``view`` repointed at the raw ``row``, or a new state over it if None
    (``view_row``), after the player-count checks; ``max_players`` 0 is the
    row's count."""
    view = view_row(view, row)
    resolve_max_players(turn_num_players(view), max_players, func)
    return view


# def rather than cpdef: cpdef cannot take the keyword-only layout_version.
def get_token_data(GameState state, float[:, ::1] buffer, int max_players=0, *, int layout_version):
    """Fill ``buffer``, at least ``get_num_tokens(max_players)`` rows of the
    layout's ``get_token_dim``, for ``state``; ``max_players`` 0 is the state's
    player count. The refreshed caches are written into the state's row (§4)."""
    cdef int num_players = turn_num_players(state)
    cdef GameState scratch
    cdef int dim = get_token_dim(layout_version)
    max_players = resolve_max_players(num_players, max_players, "get_token_data")
    assert buffer.shape[0] >= get_num_tokens(max_players) and buffer.shape[1] == dim, \
        f"get_token_data: buffer {(buffer.shape[0], buffer.shape[1])} for {get_num_tokens(max_players)} tokens " \
        f"of {dim}"
    scratch = _scratch_for(None, state, layout_version)
    with nogil:
        _fill_buffer(state, scratch, buffer, num_players, get_num_tokens(max_players), layout_version)


def get_token_data_batch(list state_arrays, float[:, :, ::1] buffer, int max_players=0, *, int layout_version):
    """Fill ``buffer[i]`` from ``state_arrays[i]``: writable raw rows, which may
    mix player counts. ``max_players`` 0 takes the buffer's capacity. One
    view is rebound across the rows, one scratch state serves their
    hypotheticals, and each row's prologue and fill run without the GIL."""
    cdef int n = len(state_arrays)
    cdef int num_tokens, i
    cdef GameState view = None, scratch = None
    cdef int dim = get_token_dim(layout_version)
    if n == 0:
        return
    if max_players == 0:
        max_players = <int>buffer.shape[1] - <int>ROW_PLAYER
    num_tokens = get_num_tokens(max_players)
    assert buffer.shape[0] >= n and buffer.shape[1] >= num_tokens and buffer.shape[2] == dim, \
        f"get_token_data_batch: buffer {(buffer.shape[0], buffer.shape[1], buffer.shape[2])} for {n} rows of {dim}"
    for i in range(n):
        view = bind_row(view, state_arrays[i], max_players, "get_token_data_batch")
        scratch = _scratch_for(scratch, view, layout_version)
        with nogil:
            _fill_buffer(view, scratch, buffer[i], turn_num_players(view), num_tokens, layout_version)


# =============================================================================
# FILLS
# =============================================================================

cdef void _fill_buffer(GameState state, GameState scratch, float[:, ::1] buffer, int num_players,
                       int num_tokens, int layout) noexcept nogil:
    """The layout's prologue and rows; ``scratch`` is ``_scratch_for``'s, None
    when the fill needs none."""
    cdef _Context x
    if layout == 1:
        extraction_prologue(state)
    else:
        _prologue2(state, scratch)
    memset(&buffer[0, 0], 0, num_tokens * buffer.shape[1] * sizeof(float))
    x.num_players = num_players
    x.phase = get_decision_phase(state)
    x.game_round = turn_round(state)
    x.game_phase = game_phase(state)
    x.actor = turn_active_player(state) if x.phase >= 0 else -1
    x.corp = _decision_corp(state, x.phase)
    if layout == 1:
        _fill_rows(state, scratch, buffer, &x)
    else:
        _context2(state, &x)
        _fill_rows2(state, scratch, buffer, &x)


cdef void _fill_rows(GameState state, GameState scratch, float[:, ::1] buffer, const _Context* x) noexcept nogil:
    """Layout 1's rows."""
    cdef int i
    _fill_global(state, &buffer[<int>ROW_GLOBAL, 0], x)
    if (x.phase == DPHASE_BID or x.phase == DPHASE_PAR or x.phase == DPHASE_TRAIN_PRICE
            or x.phase == DPHASE_PRIVATE_PRICE):
        _fill_price(state, &buffer[<int>ROW_PRICE, 0], x)
    elif x.phase == DPHASE_DIVIDEND:
        _fill_dividend(state, &buffer[<int>ROW_DIVIDEND, 0], x)
    for i in range(<int>NUM_CORPS):
        _fill_corp(state, &buffer[<int>ROW_CORP + i, 0], i, x)
    for i in range(<int>NUM_PRIVATES):
        _fill_private(state, &buffer[<int>ROW_PRIVATE + i, 0], i, x)
    for i in range(<int>NUM_TRAIN_TYPES):
        _fill_train(state, &buffer[<int>ROW_TRAIN + i, 0], i, x)
    for i in range(<int>NUM_HEXES):
        _fill_hex(state, &buffer[<int>ROW_HEX + i, 0], i, x)
    if x.phase == DPHASE_TILE or x.phase == DPHASE_ER_BONUS:
        _fill_candidates(state, scratch, buffer, x, 1)
    elif x.phase == DPHASE_TOKEN:
        _fill_token_outcomes(state, scratch, buffer, x)
    elif x.phase == DPHASE_SELECT_TRAIN or x.phase == DPHASE_DISCARD:
        _fill_train_outcomes(state, scratch, buffer, x)
    for i in range(x.num_players):
        _fill_player(state, &buffer[<int>ROW_PLAYER + i, 0], i, x)


cdef void _fill_global(GameState state, float* row, const _Context* x) noexcept nogil:
    row[0] = 1
    if x.phase >= 0:
        row[G_PHASE + x.phase] = 1
    if x.game_round != ROUND_GAME_OVER:
        row[G_ROUND + x.game_round] = 1
    row[G_GAME_PHASE + x.game_phase] = 1
    row[G_FIRST_SR] = turn_turn(state) <= 1
    if x.game_round == ROUND_OPERATING:
        row[G_OR_NUM + _global(state, GLOBALS.or_num) - 1] = 1
        row[G_OR_COUNT] = _global(state, GLOBALS.or_count) / 3.0
    else:
        row[G_OR_COUNT] = PHASE_ORS[x.game_phase] / 3.0
    row[G_BANK_CASH] = bank_cash(state) / BANK_DIV
    row[G_BANK_BROKEN] = bank_broken(state)
    row[G_PLAYERS + x.num_players - 3] = 1
    row[G_PASS_PAYOUTS] = turn_auction_pass_payouts(state) / 3.0
    row[G_DISCOUNT] = turn_auction_discount(state) / 20.0
    if x.phase == DPHASE_SELECT_TRAIN or x.phase == DPHASE_TRAIN_PRICE:
        row[G_TRAIN_SOURCE + min(turn_pending_train_source(state), <int>TRAIN_SOURCE_CORP)] = 1


cdef void _fill_price(GameState state, float* row, const _Context* x) noexcept nogil:
    """§5.4's PRICE row at BID, PAR, TRAIN_PRICE, and PRIVATE_PRICE: the yen of
    each legal id at PR_AMOUNTS + id (``price_amount``; at PAR the President's
    Certificate, twice the par; BID's drop has none), the reference price, and
    the payer's funds."""
    cdef uint16_t ids[46]  # MAX_ACTION_SIZE
    cdef int count = enumerate_legal_actions(state, ids)
    cdef ActionInfo info
    cdef int i
    row[0] = 1
    for i in range(count):
        info = decode_action(x.phase, ids[i])
        if info.type == ACTION_PAR:
            row[PR_AMOUNTS + ids[i]] = 2 * info.amount / MONEY_DIV
        elif info.type == ACTION_PRICE_POINT:
            row[PR_AMOUNTS + ids[i]] = price_amount(state, &info) / MONEY_DIV
    if x.phase == DPHASE_BID:
        row[PR_KIND] = 1
        row[PR_REFERENCE] = PRIVATE_VALUE[_bid_private(state)] / MONEY_DIV
        row[PR_FUNDS] = player_available(state, x.actor, _bid_private(state)) / MONEY_DIV
    elif x.phase == DPHASE_PAR:
        row[PR_KIND + 1] = 1
        row[PR_FUNDS] = player_cash(state, x.actor) / MONEY_DIV
    elif x.phase == DPHASE_TRAIN_PRICE:
        row[PR_KIND + 2] = 1
        row[PR_REFERENCE] = TRAIN_PRICE[turn_pending_train_type(state)] / MONEY_DIV
        row[PR_FUNDS] = corp_cash(state, x.corp) / MONEY_DIV
    else:
        row[PR_KIND + 3] = 1
        row[PR_REFERENCE] = PRIVATE_VALUE[turn_pending_private(state)] / MONEY_DIV
        row[PR_FUNDS] = corp_cash(state, x.corp) / MONEY_DIV


cdef void _fill_dividend(GameState state, float* row, const _Context* x) noexcept nogil:
    """§5.4's DIVIDEND row: the operating corp's revenue and its two outcomes
    (``apply_dividend``). A payout pays each share a tenth, rounded up, to its
    holder (the corp for its pool shares) and moves the price right; a
    withhold puts it in the treasury and moves the price left."""
    cdef int c = x.corp
    cdef int revenue = turn_route_revenue(state)
    row[0] = 1
    row[DV_REVENUE] = revenue / PRICE_DIV
    row[DV_PER_SHARE] = _payout(1, revenue) / PRICE_DIV
    row[DV_PAYOUT_PRICE] = _moved_price(state, c, MOVE_RIGHT) / PRICE_DIV
    row[DV_WITHHOLD_PRICE] = _moved_price(state, c, MOVE_LEFT) / PRICE_DIV
    row[DV_TREASURY] = (corp_cash(state, c) + revenue) / MONEY_DIV
    row[DV_PRESIDENT] = _payout(player_shares(state, corp_president(state, c), c), revenue) / PRICE_DIV
    row[DV_POOL] = _payout(corp_pool_shares(state, c), revenue) / PRICE_DIV


cdef _Reach _reach(GameState state, int c, int game_phase) noexcept nogil:
    """The corp's reach from its fresh graph block."""
    cdef _Reach r
    cdef int h, entry
    r.hexes = r.centers = r.revenue = r.slots = 0
    for h in range(<int>NUM_HEXES):
        entry = graph_entry(state, c, h)
        r.hexes += (entry & <int>GRAPH_SIDES) != 0
        if entry & <int>GRAPH_NODE:
            r.centers += 1
            r.revenue += map_revenue(state, h, game_phase, False)
            r.slots += map_has_slot_for(state, h, c)
    return r


cdef void _fill_corp(GameState state, float* row, int c, const _Context* x) noexcept nogil:
    """Slots 7-19 only while parred; the sale prices at the decisions that sell,
    every corp's at STOCK and BUY_TRAIN and the sold corp's at SELL_AMOUNT and
    EMERGENCY_SELL_AMOUNT."""
    cdef _Reach reach = _reach(state, c, x.game_phase)
    cdef int owned = 0
    cdef int i, market_r, market_c
    row[0] = 1
    row[C_SELECTED] = c == x.corp
    if x.phase == DPHASE_SELECT_TRAIN or x.phase == DPHASE_TRAIN_PRICE:
        row[C_SELLER] = turn_pending_train_source(state) == <int>TRAIN_SOURCE_CORP + c
    if x.phase == DPHASE_SELL_AMOUNT or x.phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        row[C_SALE] = turn_pending_corp(state) == c
    row[C_FLOATED] = corp_floated(state, c)
    row[C_HOME] = hex_tokens(state, CORP_HOME[c]) >> c & 1
    if corp_par_index(state, c) >= 0:
        row[C_PARRED] = 1
        row[C_PAR_PRICE] = corp_par_price(state, c) / PRICE_DIV
        row[C_PRICE] = market_price(state, c) / PRICE_DIV
        row[C_ROW] = market_row(state, c) / 10.0
        row[C_COL] = market_col(state, c) / 14.0
        row[C_ZONE + market_type(state, c)] = 1
        if x.phase == DPHASE_STOCK or x.phase == DPHASE_BUY_TRAIN or row[C_SALE] != 0:
            market_r, market_c = market_row(state, c), market_col(state, c)
            for i in range(<int>MAX_SALE_SHARES):  # a row down per share sold
                market_moved(market_r, market_c, MOVE_DOWN, 1, &market_r, &market_c)
                row[C_SALE_PRICES + i] = MARKET_PRICE[market_r][market_c] / PRICE_DIV
    row[C_SOLD_OUT] = corp_ipo_shares(state, c) == 0 and corp_pool_shares(state, c) == 0
    row[C_CASH] = corp_cash(state, c) / MONEY_DIV
    row[C_IPO] = corp_ipo_shares(state, c) / <float>SHARES_PER_CORP
    row[C_POOL] = corp_pool_shares(state, c) / <float>SHARES_PER_CORP
    row[C_TOKENS] = corp_tokens_left(state, c) / <float>MAX_CORP_TOKENS
    row[C_TOKEN_PRICE] = corp_token_price(state, c) / PRICE_DIV
    for i in range(<int>NUM_TRAIN_TYPES):
        row[C_TRAINS + i] = corp_train_count(state, c, i) / <float>MAX_ROUTES
    row[C_CROWDED] = corp_train_total(state, c) > trains_limit(state)
    row[C_LAST_REVENUE] = _corp_field(state, c, CORP_FIELDS.last_revenue) / PRICE_DIV
    row[C_ROUTE_MAX] = _corp_field(state, c, CORP_FIELDS.route_cache) / PRICE_DIV
    row[C_ROUTE_AVAILABLE] = _corp_field(state, c, CORP_FIELDS.route_available) != 0
    row[C_MUST_BUY] = must_buy_train(state, c)
    row[C_EMERGENCY] = c == x.corp and in_emergency(state, c)
    for i in range(<int>NUM_CORPS):
        if _global(state, GLOBALS.or_order + i) == c:
            row[C_OR_POSITION] = (i + 1) / <float>NUM_CORPS
            row[C_OPERATED] = i < _global(state, GLOBALS.or_index)
    row[C_HEXES] = reach.hexes / <float>NUM_HEXES
    row[C_CENTERS] = reach.centers / <float>NUM_RCS
    row[C_REVENUE] = reach.revenue / PRICE_DIV
    row[C_SLOTS] = reach.slots / 5.0
    if x.actor >= 0:
        row[C_ACTOR_SOLD] = player_sold_mask(state, x.actor) >> c & 1
        if x.game_round == ROUND_STOCK:
            row[C_SALE_LIMIT] = sale_limit(state, x.actor, c) / <float>MAX_SALE_SHARES
        row[C_EMERGENCY_SALE] = corp_emergency_sale(state, x.actor, c) / <float>MAX_SALE_SHARES
    row[C_TURN_SOLD] = turn_sr_sold_mask(state) >> c & 1
    row[C_EMERGENCY_SOLD] = turn_ebuy_sold_mask(state) >> c & 1
    row[C_BOUGHT] = turn_sr_bought_corp(state) == c
    for i in range(<int>NUM_PRIVATES):
        owned += private_owner(state, i) == <int>OWNER_CORP + c
    row[C_PRIVATES] = owned / 3.0


cdef void _fill_private(GameState state, float* row, int p, const _Context* x) noexcept nogil:
    """Hidden while not in play; the price and the auction's slots only while unsold."""
    cdef int owner = private_owner(state, p)
    cdef int h
    if owner == <int>OWNER_NOT_IN_PLAY:
        return
    row[0] = 1
    row[P_SELECTED] = (x.phase == DPHASE_BID and _bid_private(state) == p
                       or x.phase == DPHASE_PRIVATE_PRICE and turn_pending_private(state) == p)
    row[P_TOPMOST] = topmost_private(state) == p
    if owner == <int>OWNER_UNSOLD:
        row[P_UNSOLD] = 1
    elif owner == <int>OWNER_CLOSED:
        row[P_CLOSED] = 1
    else:
        row[P_PLAYER_OWNED if owner < <int>OWNER_CORP else P_CORP_OWNED] = 1
    row[P_FACE] = PRIVATE_VALUE[p] / PRICE_DIV
    if owner != <int>OWNER_CLOSED:
        row[P_REVENUE] = private_revenue(state, p) / PRICE_DIV
    if owner == <int>OWNER_UNSOLD:
        row[P_PRICE] = private_price(state, p) / PRICE_DIV
        row[P_HIGH_BID] = private_high_bid(state, p) / PRICE_DIV
        row[P_BIDS] = private_bid_count(state, p) / 5.0
        row[P_MIN_BID] = private_min_bid(state, p) / PRICE_DIV
        if x.actor >= 0:
            row[P_ACTOR_BID] = player_bid(state, x.actor, p) / PRICE_DIV
    for h in range(<int>NUM_HEXES):
        if HEX_BLOCKER[h] == p:
            row[P_BLOCKING] = private_blocks_hex(state, h)
    row[P_PORT_USED] = (private_flags(state, p) & <int>MF_PORT_USED) != 0
    if x.corp >= 0:
        row[P_BUYABLE] = can_buy_private(state, x.corp, p)
    row[P_EXCHANGE] = p == <int>PRIVATE_DR and x.actor >= 0 and can_exchange_dr(state, x.actor)


cdef int _rust_phase(int train_type) noexcept nogil:
    """The game phase whose start rusts ``train_type``'s trains, or -1 for never."""
    cdef int rusts_on = TRAIN_RUSTS_ON[train_type]
    cdef int phase
    for phase in range(<int>NUM_GAME_PHASES):
        if rusts_on >= 0 and PHASE_ON_TRAIN[phase] == rusts_on:
            return phase
    return -1


cdef void _fill_train(GameState state, float* row, int t, const _Context* x) noexcept nogil:
    cdef int depot = depot_count(state, t)
    cdef int rusts = _rust_phase(t)
    row[0] = 1
    row[T_SELECTED] = x.phase == DPHASE_TRAIN_PRICE and turn_pending_train_type(state) == t
    row[T_PRICE] = TRAIN_PRICE[t] / MONEY_DIV
    row[T_DEPOT] = max(depot, 0) / 6.0
    row[T_UNLIMITED] = depot < 0
    row[T_POOL] = pool_count(state, t) / 4.0
    row[T_DEPOT_SELLS] = trains_depot_types(state) >> t & 1
    row[T_CHEAPEST] = trains_cheapest_type(state) == t
    row[T_RUSTED] = 0 <= rusts <= x.game_phase
    row[T_RUSTS_NEXT] = rusts == x.game_phase + 1


cdef void _fill_tile_block(float* block, int tile, int rotation, int revenue) noexcept nogil:
    """The tile block of ``tile`` at ``rotation``, whose center earns ``revenue`` now."""
    cdef uint32_t paths = TILE_MASK[tile][rotation]
    cdef int i
    block[TB_SUPPLY + tile if tile < <int>NUM_SUPPLY_TILES else TB_PRINTED] = 1
    for i in range(<int>NUM_PAIRS):
        block[TB_PATHS + i] = paths >> i & 1
    block[TB_COLOR + TILE_COLOR[tile]] = 1
    if TILE_RC[tile] != <int>RC_NONE:
        block[TB_CENTER + TILE_RC[tile] - 1] = 1
    block[TB_REVENUE] = revenue / PRICE_DIV
    block[TB_SLOTS] = TILE_SLOTS[tile] / <float>MAX_CITY_SLOTS
    if TILE_LABEL[tile] != <int>LABEL_NONE:
        block[TB_LABEL + TILE_LABEL[tile] - 1] = 1


cdef void _fill_hex(GameState state, float* row, int h, const _Context* x) noexcept nogil:
    cdef int tile = hex_tile(state, h)
    cdef int rotation = hex_rotation(state, h)
    cdef int k = tile * <int>NUM_SIDES + rotation
    cdef int home = HEX_HOME_CORP[h]
    cdef int terrain = TILE_TERRAIN[<int>NUM_SUPPLY_TILES + h]
    cdef int colors = PHASE_TILE_COLORS[x.game_phase]
    cdef uint32_t paths = hex_paths(state, h)
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int upgrades = 0, upgrades_any_phase = 0
    cdef int s, i, target, entry
    row[0] = 1
    row[H_SELECTED] = (x.phase == DPHASE_TILE and turn_pending_hex(state) == h
                       or x.phase == DPHASE_ER_BONUS and h == <int>ER_HEX)
    _fill_tile_block(row + <int>TILE_BLOCK, tile, rotation, map_revenue(state, h, x.game_phase, False))
    row[H_D_REVENUE] = map_revenue(state, h, x.game_phase, True) / PRICE_DIV
    row[H_MOUNTAIN] = (terrain & <int>TERRAIN_MOUNTAIN) != 0
    row[H_WATER] = (terrain & <int>TERRAIN_WATER) != 0
    row[H_TOKENS] = map_token_count(state, h) / <float>MAX_CITY_SLOTS
    row[H_FREE_SLOTS] = map_free_slots(state, h) / <float>MAX_CITY_SLOTS
    row[H_RESERVED] = home >= 0 and not (hex_tokens(state, h) >> home & 1)
    row[H_COST] = TILE_COST[tile] / PRICE_DIV
    row[H_BLOCKED] = private_blocks_hex(state, h)
    # Track at an open side, and none facing back from the hex across.
    for s in range(<int>NUM_SIDES):
        row[H_DANGLING + s] = ((paths & ENDPOINT_PAIRS[s]) != 0 and HEX_EXIT_MASK[h] >> s & 1
                               and not (hex_paths(state, HEX_NEIGHBOR[h][s])
                                        & ENDPOINT_PAIRS[(s + 3) % <int>NUM_SIDES]))
    # The upgrade table's targets in supply whose exits stay on open sides,
    # as _lay_scan filters them before connectivity, cost, and the track rule.
    for i in range(UPG_START[k], UPG_START[k + 1]):
        target = UPG_TILE[i]
        if (TILE_BLOCKS_LAY[target] or supply_count(state, target) <= 0
                or TILE_EXITS[target][UPG_ROT[i]] & ~HEX_EXIT_MASK[h]):
            continue
        upgrades_any_phase += 1
        upgrades += colors >> TILE_COLOR[target] & 1
    row[H_UPGRADES] = upgrades / <float>MAX_LAY_CANDIDATES
    row[H_UPGRADES_ANY_PHASE] = upgrades_any_phase / <float>MAX_LAY_CANDIDATES
    if x.corp >= 0:
        entry = graph_entry(state, x.corp, h)
        for s in range(<int>NUM_SIDES):
            row[H_SIDES + s] = entry >> s & 1
        row[H_CENTER] = (entry & <int>GRAPH_NODE) != 0
        row[H_BLOCKED_FOR] = map_blocked_for(state, h, x.corp)
        if x.phase == DPHASE_TRACK:
            row[H_CANDIDATES] = lay_candidates(state, x.corp, h, candidates) / <float>MAX_LAY_CANDIDATES


cdef void _fill_candidates(GameState state, GameState scratch, float[:, ::1] buffer, const _Context* x,
                           int layout) noexcept nogil:
    """§5.8's rows at TILE and ER_BONUS, row k for candidate k, each lay tried
    on the scratch state (§7.1). The deltas are the decision corp's, the
    buyer at ER_BONUS. Another corp is refreshed only if its walks reach a
    side of the hex, since a lay on a hex they never enter cannot change its
    reach, and only if the lay may have gained it a center
    (``map_center_bounds``). Layout 2 divides the other corps' centers by 50
    and adds CANDIDATE 108: the lay's Δ route max were the corp to buy the
    next train (``route_with_next``'s purchase, tried on the laid scratch
    state; one forest serves both solves unless the purchase starts a phase)."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef _Reach before[7]             # NUM_CORPS
    cdef _Reach after
    cdef int hex, k, i, c, tile, rotation, route, grown, gained
    cdef int count = tile_candidates(state, x.phase, &hex, candidates)
    cdef uint32_t old_paths, added
    cdef float cost
    cdef float* row
    cdef int bounds[7]                # NUM_CORPS
    cdef int touch, route_after, next_after, with_next
    cdef bint room, starts
    if count == 0:
        return
    touch = map_touch_mask(state, hex)  # one search serves every candidate's copy
    room = layout == 2 and x.next_type >= 0 and corp_train_total(state, x.corp) < x.limit
    starts = room and x.game_phase + 1 < <int>NUM_GAME_PHASES and PHASE_ON_TRAIN[x.game_phase + 1] == x.next_type
    old_paths = hex_paths(state, hex)
    cost = 0 if x.phase == DPHASE_ER_BONUS else map_lay_cost(state, hex, x.corp) / PRICE_DIV
    route = _corp_field(state, x.corp, CORP_FIELDS.route_cache)
    with_next = _corp_field(state, x.corp, CORP_FIELDS.route_with_next) if room else route
    for c in range(<int>NUM_CORPS):
        if c == x.corp or graph_entry(state, c, hex):
            before[c] = _reach(state, c, x.game_phase)
    for k in range(count):
        tile = candidates[k].tile
        rotation = candidates[k].rotation
        row = &buffer[<int>ROW_CANDIDATE + k, 0]
        copy_row(scratch, state)
        map_put_tile(scratch, hex, tile, rotation, touch)
        row[0] = 1
        row[CA_ER] = x.phase == DPHASE_ER_BONUS
        _fill_tile_block(row + <int>TILE_BLOCK, tile, rotation, map_revenue(scratch, hex, x.game_phase, False))
        added = TILE_MASK[tile][rotation] & ~old_paths
        for i in range(<int>NUM_PAIRS):
            row[CA_ADDED + i] = added >> i & 1
        row[CA_COST] = cost
        row[CA_COPIES] = supply_count(scratch, tile) / <float>TILE_COUNT[tile]
        grown = gained = 0
        map_center_bounds(scratch, bounds)
        for c in range(<int>NUM_CORPS):
            if c != x.corp and (not graph_entry(state, c, hex) or bounds[c] <= before[c].centers):
                continue
            graph_refresh(scratch, c)
            after = _reach(scratch, c, x.game_phase)
            if c == x.corp:
                row[CA_HEXES] = (after.hexes - before[c].hexes) / <float>NUM_HEXES
                row[CA_CENTERS] = (after.centers - before[c].centers) / <float>NUM_RCS
                row[CA_REVENUE] = (after.revenue - before[c].revenue) / PRICE_DIV
                # A lay only adds routes, and those pass its hex: search them alone.
                if room and not starts:
                    route_after = solve_routes_via(scratch, c, hex, route, x.next_type, with_next, &next_after)
                else:
                    route_after = next_after = solve_routes_via(scratch, c, hex, route, -1, 0, NULL)
                row[CA_ROUTE_DELTA] = (route_after - route) / PRICE_DIV
                row[CA_ROUTE_AVAILABLE] = graph_route_available(scratch, c)
            elif after.centers > before[c].centers:
                grown += 1
                gained += after.centers - before[c].centers
        row[CA_OTHERS] = grown / <float>(NUM_CORPS - 1)
        row[CA_OTHER_CENTERS] = gained / <float>NUM_RCS
        if layout == 2:
            row[CA_OTHER_CENTERS] = gained / 50.0
            if starts:  # last: the purchase starts a phase on the scratch state
                trains_set_count(scratch, x.corp, x.next_type, corp_train_count(scratch, x.corp, x.next_type) + 1)
                phase_on_train_bought(scratch, x.next_type)
                next_after = solve_routes_via(scratch, x.corp, hex, with_next, -1, 0, NULL)
            row[CA2_NEXT_DELTA] = (next_after - with_next) / PRICE_DIV  # without room, the trains it holds


cdef inline float _route_delta(GameState scratch, GameState state, int corp) noexcept nogil:
    """The corp's route max on the changed scratch state less the row's /100."""
    return (route_max(scratch, corp) - _corp_field(state, corp, CORP_FIELDS.route_cache)) / PRICE_DIV


cdef void _fill_token_outcomes(GameState state, GameState scratch, float[:, ::1] buffer,
                               const _Context* x) noexcept nogil:
    """HEX 103-104 at TOKEN (§5.7): each city the decision corp may token,
    TOKEN's hexes, and its Δ route max with the token there, tried on the
    scratch state (``solve_routes_via``)."""
    cdef int route = _corp_field(state, x.corp, CORP_FIELDS.route_cache)
    cdef int h
    for h in range(<int>NUM_HEXES):
        if can_token(state, x.corp, h):
            copy_row(scratch, state)
            map_put_token(scratch, h, x.corp, 0)  # the scratch caches are not read
            buffer[<int>ROW_HEX + h, H_TOKENABLE] = 1
            # The corp's own token only adds routes, and those pass its city.
            buffer[<int>ROW_HEX + h, H_TOKEN_DELTA] = \
                (solve_routes_via(scratch, x.corp, h, route, -1, 0, NULL) - route) / PRICE_DIV


cdef void _fill_train_outcomes(GameState state, GameState scratch, float[:, ::1] buffer,
                               const _Context* x) noexcept nogil:
    """TRAIN 10-12 (§7.2): the decision corp's Δ route max after each legal
    choice, tried on the scratch state: at SELECT_TRAIN one more train of type
    t (slot 10) or a D for a traded t (slot 11), and at DISCARD one fewer
    (slot 12). What a purchase does to the phase is left out, and no train is
    added past MAX_ROUTES, where the solver aborts (§4)."""
    cdef uint16_t ids[46]  # MAX_ACTION_SIZE
    cdef int count = enumerate_legal_actions(state, ids)
    cdef int c = x.corp
    cdef ActionInfo info
    cdef int i, t, slot
    for i in range(count):
        info = decode_action(x.phase, ids[i])
        t = info.train_type
        if info.type == ACTION_TRAIN_TYPE and corp_train_total(state, c) >= <int>MAX_ROUTES:
            continue
        copy_row(scratch, state)
        if info.type == ACTION_TRAIN_TYPE:
            slot = T_ADD_DELTA
            trains_set_count(scratch, c, t, corp_train_count(state, c, t) + 1)
        else:  # a trade-in or a discard: one fewer of type t
            slot = T_TRADE_IN_DELTA if info.type == ACTION_TRADE_IN else T_DISCARD_DELTA
            trains_set_count(scratch, c, t, corp_train_count(state, c, t) - 1)
            if info.type == ACTION_TRADE_IN:
                trains_set_count(scratch, c, <int>D_TRAIN, corp_train_count(state, c, <int>D_TRAIN) + 1)
        buffer[<int>ROW_TRAIN + t, slot] = _route_delta(scratch, state, c)


cdef void _fill_player(GameState state, float* row, int i, const _Context* x) noexcept nogil:
    cdef int n = x.num_players
    cdef int certs = player_cert_count(state, i)
    cdef int operating = turn_operating_corp(state)
    cdef int presidencies = 0, privates = 0
    cdef int c, p
    row[0] = 1
    row[PL_SELECTED] = i == x.actor
    if x.actor >= 0:
        row[PL_SEAT + (i - x.actor + n) % n] = 1
    row[PL_PRIORITY] = i == _global(state, GLOBALS.priority_player)
    row[PL_CASH] = player_cash(state, i) / MONEY_DIV
    row[PL_VALUE] = player_value(state, i) / MONEY_DIV
    row[PL_CERTS_OF_LIMIT] = certs / <float>CERT_LIMIT[n]
    row[PL_CERTS] = certs / 25.0
    for c in range(<int>NUM_CORPS):
        row[PL_SHARES + c] = player_shares(state, i, c) / <float>SHARES_PER_CORP
        row[PL_SOLD + c] = player_sold_mask(state, i) >> c & 1
        presidencies += corp_president(state, c) == i
    for p in range(<int>NUM_PRIVATES):
        row[PL_BIDS + p] = player_bid(state, i, p) / PRICE_DIV
        privates += private_owner(state, p) == <int>OWNER_PLAYER + i
    row[PL_PASSED] = player_passed(state, i)
    row[PL_BANKRUPT] = state._data[LAYOUT.players_offset + i * PLAYER_FIELDS.size + PLAYER_FIELDS.bankrupt] != 0
    row[PL_PRESIDENCIES] = presidencies / <float>NUM_CORPS
    row[PL_PRIVATES] = privates / 3.0
    # The operating corp's president: an emergency sale since the turn's last purchase.
    row[PL_EMERGENCY_SOLD] = operating >= 0 and corp_president(state, operating) == i and turn_ebuy_sold_any(state)


# =============================================================================
# LAYOUT 2
# =============================================================================

cdef void _context2(GameState state, _Context* x) noexcept nogil:
    """Layout 2's context. The operating rank is the corp's place in the
    OR's order during an OR. In an SR and the auction it is the place the
    next OR would give it: the floated corps in stock price order, market
    stack included, and for a parred corp yet to float the place it would
    take among them."""
    cdef int order[7]  # NUM_CORPS
    cdef int floated, count, c, i
    x.operating = turn_operating_corp(state)
    x.next_type = trains_cheapest_type(state)
    x.limit = trains_limit(state)
    x.next_limit = PHASE_TRAIN_LIMIT[min(x.game_phase + 1, <int>NUM_GAME_PHASES - 1)]
    x.rust_phase = next_rust_phase(x.game_phase)
    x.emergency = x.corp >= 0 and x.game_round == ROUND_OPERATING and in_emergency(state, x.corp)
    x.need = max(emergency_need(state, x.corp), 0) if x.emergency else 0
    for c in range(<int>NUM_CORPS):
        x.rank[c] = -1
    if x.game_round == ROUND_OPERATING:
        for i in range(<int>NUM_CORPS):
            c = _global(state, GLOBALS.or_order + i)
            if c >= 0:
                x.rank[c] = i
    elif x.game_round != ROUND_GAME_OVER:
        floated = corp_floated_mask(state)
        count = market_sort_by_price(state, floated, order)
        for i in range(count):
            x.rank[order[i]] = i
        for c in range(<int>NUM_CORPS):
            if corp_par_index(state, c) >= 0 and not (floated >> c & 1):
                x.rank[c] = 0
                for i in range(count):
                    x.rank[c] += market_precedes(state, order[i], c)


cdef void _fill_rows2(GameState state, GameState scratch, float[:, ::1] buffer, const _Context* x) noexcept nogil:
    """Layout 2's rows."""
    cdef int i
    _fill_global2(state, &buffer[<int>ROW_GLOBAL, 0], x)
    if (x.phase == DPHASE_BID or x.phase == DPHASE_PAR or x.phase == DPHASE_TRAIN_PRICE
            or x.phase == DPHASE_PRIVATE_PRICE):
        _fill_price2(state, &buffer[<int>ROW_PRICE, 0], x)
    elif x.phase == DPHASE_DIVIDEND:
        _fill_dividend2(state, &buffer[<int>ROW_DIVIDEND, 0], x)
    for i in range(<int>NUM_CORPS):
        _fill_corp2(state, &buffer[<int>ROW_CORP + i, 0], i, x)
    for i in range(<int>NUM_PRIVATES):
        _fill_private(state, &buffer[<int>ROW_PRIVATE + i, 0], i, x)
        if i != <int>PRIVATE_MF:  # the port tile is MF's
            buffer[<int>ROW_PRIVATE + i, P_PORT_USED] = 0
    for i in range(<int>NUM_TRAIN_TYPES):
        _fill_train(state, &buffer[<int>ROW_TRAIN + i, 0], i, x)
        buffer[<int>ROW_TRAIN + i, T2_STARTS_PHASE] = (x.game_phase + 1 < <int>NUM_GAME_PHASES
                                                       and PHASE_ON_TRAIN[x.game_phase + 1] == i)
    for i in range(<int>NUM_HEXES):
        _fill_hex(state, &buffer[<int>ROW_HEX + i, 0], i, x)
    if x.phase == DPHASE_TILE or x.phase == DPHASE_ER_BONUS:
        _fill_candidates(state, scratch, buffer, x, 2)
    elif x.phase == DPHASE_TOKEN:
        _fill_token_outcomes(state, scratch, buffer, x)
    elif (x.phase == DPHASE_SELECT_TRAIN or x.phase == DPHASE_DISCARD or x.phase == DPHASE_TRAIN_PRICE
          or x.phase == DPHASE_BUY_TRAIN or x.phase == DPHASE_DIVIDEND):
        _fill_train_outcomes2(state, scratch, buffer, x)
    for i in range(x.num_players):
        _fill_player2(state, &buffer[<int>ROW_PLAYER + i, 0], i, x)


cdef void _fill_global2(GameState state, float* row, const _Context* x) noexcept nogil:
    """Layout 1's GLOBAL, with the auction's slots cleared once it ends, and
    slots 44-89: the step a private's sub-phase interrupts, the ORs left in
    the set, the ORs until the bank breaks if every floated corp pays out its
    route max each OR, the train limits, and the tile supply."""
    cdef int cash = bank_cash(state)
    cdef int drain = 0  # yen an OR, times SHARES_PER_CORP
    cdef int c, t, step
    _fill_global(state, row, x)
    if x.game_round != ROUND_AUCTION:
        row[G_PASS_PAYOUTS] = row[G_DISCOUNT] = 0
    if x.phase == DPHASE_PRIVATE_PRICE or x.phase == DPHASE_ER_BONUS or x.phase == DPHASE_PORT:
        step = turn_return_step(state)
        row[G2_PRE] = step == STEP_BUY_PRIVATE_PRE
        row[G2_POST] = step == STEP_BUY_PRIVATE_POST
    if x.game_round == ROUND_OPERATING:
        row[G2_ORS_LEFT] = (_global(state, GLOBALS.or_count) - _global(state, GLOBALS.or_num)) / 2.0
    for c in range(<int>NUM_CORPS):
        if corp_floated(state, c):
            drain += _corp_field(state, c, CORP_FIELDS.route_cache) * (<int>SHARES_PER_CORP - corp_ipo_shares(state, c))
    if cash > 0:
        row[G2_BANK_ORS] = min(cash / (drain / <float>SHARES_PER_CORP) / 10.0, 1.0) if drain else 1.0
    row[G2_TRAIN_LIMIT] = x.limit / <float>MAX_ROUTES
    row[G2_NEXT_TRAIN_LIMIT] = x.next_limit / <float>MAX_ROUTES
    for t in range(<int>NUM_SUPPLY_TILES):
        row[G2_SUPPLY + t] = supply_count(state, t) / SUPPLY_DIV


cdef void _fill_price2(GameState state, float* row, const _Context* x) noexcept nogil:
    """Layout 1's PRICE, and at PAR slots 24-29: of the five shares that
    must leave the IPO to float the corp, how many are out once the actor has
    bought all their cash buys at each legal par (DR's exchange may have
    taken one already)."""
    cdef uint16_t ids[46]  # MAX_ACTION_SIZE
    cdef ActionInfo info
    cdef int count, i, out
    _fill_price(state, row, x)
    if x.phase != DPHASE_PAR:
        return
    out = <int>SHARES_PER_CORP - corp_ipo_shares(state, x.corp)
    count = enumerate_legal_actions(state, ids)
    for i in range(count):
        info = decode_action(x.phase, ids[i])
        if info.type == ACTION_PAR:
            row[PR2_FLOAT_SHARES + ids[i]] = min(out + player_cash(state, x.actor) // info.amount, FLOAT_SHARES) \
                / <float>FLOAT_SHARES


cdef void _fill_dividend2(GameState state, float* row, const _Context* x) noexcept nogil:
    """The DIVIDEND row: layout 1's outcomes (``_fill_dividend``) without the
    per-share payout, then what each leaves toward the next train: its price
    and the shortfalls while the corp has room for it, the room, and the
    corp's two hypothetical route maxes."""
    cdef int c = x.corp
    cdef int revenue = turn_route_revenue(state)
    cdef int cash = corp_cash(state, c)
    cdef int pool = _payout(corp_pool_shares(state, c), revenue)
    cdef int room = x.limit - corp_train_total(state, c)
    cdef int price
    row[0] = 1
    row[DV2_REVENUE] = revenue / PRICE_DIV
    row[DV2_PAYOUT_PRICE] = _moved_price(state, c, MOVE_RIGHT) / PRICE_DIV
    row[DV2_WITHHOLD_PRICE] = _moved_price(state, c, MOVE_LEFT) / PRICE_DIV
    row[DV2_TREASURY] = (cash + revenue) / MONEY_DIV
    row[DV2_PRESIDENT] = _payout(player_shares(state, corp_president(state, c), c), revenue) / PRICE_DIV
    row[DV2_POOL] = pool / PRICE_DIV
    if room > 0 and x.next_type >= 0:
        price = TRAIN_PRICE[x.next_type]
        row[DV2_NEXT_PRICE] = price / MONEY_DIV
        row[DV2_WITHHOLD_SHORT] = max(price - cash - revenue, 0) / MONEY_DIV
        row[DV2_PAYOUT_SHORT] = max(price - cash - pool, 0) / MONEY_DIV
    row[DV2_ROOM] = room / <float>MAX_ROUTES
    row[DV2_WITH_NEXT] = _corp_field(state, c, CORP_FIELDS.route_with_next) / PRICE_DIV
    row[DV2_AFTER_RUST] = _corp_field(state, c, CORP_FIELDS.route_after_rust) / PRICE_DIV


cdef void _fill_corp2(GameState state, float* row, int c, const _Context* x) noexcept nogil:
    """Slots 7-19, 58-60, and 74 only while parred. Slots 1-37 are layout 1's
    (``_fill_corp``), but the sale prices are filled at every decision."""
    cdef _Reach reach = _reach(state, c, x.game_phase)
    cdef int cash = corp_cash(state, c)
    cdef int route = _corp_field(state, c, CORP_FIELDS.route_cache)
    cdef int president = corp_president(state, c)
    cdef int held = corp_train_total(state, c)
    cdef int owned = 0, rusting = 0, rusting_next = 0
    cdef int i, market_r, market_c, to_float, cost
    row[0] = 1
    row[C_SELECTED] = c == x.corp
    if x.phase == DPHASE_SELECT_TRAIN or x.phase == DPHASE_TRAIN_PRICE:
        row[C_SELLER] = turn_pending_train_source(state) == <int>TRAIN_SOURCE_CORP + c
    if x.phase == DPHASE_SELL_AMOUNT or x.phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        row[C_SALE] = turn_pending_corp(state) == c
    row[C_FLOATED] = corp_floated(state, c)
    row[C_HOME] = hex_tokens(state, CORP_HOME[c]) >> c & 1
    if corp_par_index(state, c) >= 0:
        row[C_PARRED] = 1
        row[C_PAR_PRICE] = corp_par_price(state, c) / PRICE_DIV
        row[C_PRICE] = market_price(state, c) / PRICE_DIV
        row[C_ROW] = market_row(state, c) / 10.0
        row[C_COL] = market_col(state, c) / 14.0
        row[C_ZONE + market_type(state, c)] = 1
        market_r, market_c = market_row(state, c), market_col(state, c)
        for i in range(<int>MAX_SALE_SHARES):  # a row down per share sold; the bottom repeats
            market_moved(market_r, market_c, MOVE_DOWN, 1, &market_r, &market_c)
            row[C_SALE_PRICES + i] = MARKET_PRICE[market_r][market_c] / PRICE_DIV
        row[C2_PAYOUT_PRICE] = _moved_price(state, c, MOVE_RIGHT) / PRICE_DIV
        row[C2_WITHHOLD_PRICE] = _moved_price(state, c, MOVE_LEFT) / PRICE_DIV
        row[C2_SOLD_OUT_PRICE] = _moved_price(state, c, MOVE_UP) / PRICE_DIV
        row[C2_LOG_PRICE] = _log_yen(market_price(state, c))
    row[C_SOLD_OUT] = corp_ipo_shares(state, c) == 0 and corp_pool_shares(state, c) == 0
    row[C_CASH] = cash / MONEY_DIV
    row[C_IPO] = corp_ipo_shares(state, c) / <float>SHARES_PER_CORP
    row[C_POOL] = corp_pool_shares(state, c) / <float>SHARES_PER_CORP
    row[C_TOKENS] = corp_tokens_left(state, c) / <float>MAX_CORP_TOKENS
    row[C_TOKEN_PRICE] = corp_token_price(state, c) / PRICE_DIV
    for i in range(<int>NUM_TRAIN_TYPES):
        row[C_TRAINS + i] = corp_train_count(state, c, i) / <float>MAX_ROUTES
        rusting += corp_train_count(state, c, i) if 0 <= x.rust_phase == _rust_phase(i) else 0
        rusting_next += corp_train_count(state, c, i) if _rust_phase(i) == x.game_phase + 1 else 0
    row[C_CROWDED] = held > x.limit
    row[C_LAST_REVENUE] = _corp_field(state, c, CORP_FIELDS.last_revenue) / PRICE_DIV
    row[C_ROUTE_MAX] = route / PRICE_DIV
    row[C_ROUTE_AVAILABLE] = _corp_field(state, c, CORP_FIELDS.route_available) != 0
    row[C_MUST_BUY] = must_buy_train(state, c)
    row[C_EMERGENCY] = c == x.corp and x.emergency
    if x.rank[c] >= 0:
        row[C2_RANK + x.rank[c]] = 1
    if x.game_round == ROUND_OPERATING:
        row[C2_OPERATED] = 0 <= x.rank[c] < _global(state, GLOBALS.or_index)
    row[C2_HEXES] = reach.hexes / <float>NUM_HEXES
    row[C2_CENTERS] = reach.centers / <float>NUM_RCS
    row[C2_REVENUE] = reach.revenue / PRICE_DIV
    row[C2_SLOTS] = reach.slots / 15.0
    if x.actor >= 0:
        row[C2_ACTOR_SOLD] = player_sold_mask(state, x.actor) >> c & 1
        if x.game_round == ROUND_STOCK:
            row[C2_SALE_LIMIT] = sale_limit(state, x.actor, c) / <float>MAX_SALE_SHARES
    if x.emergency:
        row[C2_EMERGENCY_SALE] = emergency_sale_limit(state, x.corp, c, x.need) / <float>MAX_SALE_SHARES
        if c == x.corp:
            row[C2_EMERGENCY_NEED] = x.need / MONEY_DIV
    row[C2_TURN_SOLD] = turn_sr_sold_mask(state) >> c & 1
    row[C2_EMERGENCY_SOLD] = turn_ebuy_sold_mask(state) >> c & 1
    row[C2_BOUGHT] = turn_sr_bought_corp(state) == c
    for i in range(<int>NUM_PRIVATES):
        owned += private_owner(state, i) == <int>OWNER_CORP + c
    row[C2_PRIVATES] = owned / <float>NUM_PRIVATES
    row[C2_OPERATING] = c == x.operating
    row[C2_TRAINS_OF_LIMIT] = held / <float>x.limit
    row[C2_ROOM] = (x.limit - held) / <float>MAX_ROUTES
    row[C2_SURVIVING] = (held - rusting) / <float>MAX_ROUTES
    row[C2_NEXT_ROOM] = (x.next_limit - (held - rusting_next)) / <float>MAX_ROUTES
    row[C2_AFTER_RUST] = _corp_field(state, c, CORP_FIELDS.route_after_rust) / PRICE_DIV
    row[C2_WITH_NEXT] = _corp_field(state, c, CORP_FIELDS.route_with_next) / PRICE_DIV
    if x.next_type >= 0:
        row[C2_SHORTFALL] = max(TRAIN_PRICE[x.next_type] - cash, 0) / MONEY_DIV
    # The shares still to leave the IPO before the corp floats, at par; at the lowest par while unparred.
    to_float = max(corp_ipo_shares(state, c) - (<int>SHARES_PER_CORP - FLOAT_SHARES), 0)
    if to_float and not corp_floated(state, c):
        cost = to_float * (corp_par_price(state, c) if corp_par_index(state, c) >= 0 else LOWEST_PAR)
        row[C2_FLOAT_SHARES] = to_float / <float>FLOAT_SHARES
        row[C2_FLOAT_COST] = cost / MONEY_DIV
        row[C2_PRESIDENT_FLOATS] = president >= 0 and player_cash(state, president) >= cost
        row[C2_ACTOR_FLOATS] = x.actor >= 0 and player_cash(state, x.actor) >= cost
    row[C2_PRESIDENT_SOLD] = president >= 0 and (player_sold_mask(state, president) >> c & 1) != 0
    row[C2_LOG_CASH] = _log_yen(cash)
    row[C2_LOG_ROUTE] = _log_yen(route)


cdef void _fill_train_outcomes2(GameState state, GameState scratch, float[:, ::1] buffer,
                                const _Context* x) noexcept nogil:
    """TRAIN 10-13: what a train changes for the decision corp's route max,
    each purchase tried on the scratch state with the phase it starts
    (``route_max_after_purchase``), and no train added past MAX_ROUTES.

    - SELECT_TRAIN, each legal choice: one more of type t (slot 10) or a D
      for a traded t (slot 11); from a corp, also the seller's Δ (slot 13).
    - TRAIN_PRICE, the pending type: slots 10 and 13, as at SELECT_TRAIN.
    - BUY_TRAIN and DIVIDEND, while the corp has room under the limit: slot
      10 for each type the depot sells or the pool holds, whatever its price.
    - DISCARD, each legal choice: one fewer of type t (slot 12)."""
    cdef uint16_t ids[46]  # MAX_ACTION_SIZE
    cdef int c = x.corp
    cdef int route = _corp_field(state, c, CORP_FIELDS.route_cache)
    cdef int held = corp_train_total(state, c)
    cdef int seller = turn_pending_train_source(state) - <int>TRAIN_SOURCE_CORP
    cdef ActionInfo info
    cdef int count, i, t, types, value
    if x.phase == DPHASE_BUY_TRAIN or x.phase == DPHASE_DIVIDEND:
        if held >= x.limit:
            return
        types = trains_depot_types(state)
        for t in range(<int>NUM_TRAIN_TYPES):
            if types >> t & 1 or pool_count(state, t):
                value = (_corp_field(state, c, CORP_FIELDS.route_with_next) if t == x.next_type
                         else route_max_after_purchase(state, scratch, c, t, -1))
                buffer[<int>ROW_TRAIN + t, T_ADD_DELTA] = (value - route) / PRICE_DIV
        return
    if x.phase == DPHASE_TRAIN_PRICE:
        if held < <int>MAX_ROUTES and seller >= 0:
            _fill_corp_sale(state, scratch, &buffer[<int>ROW_TRAIN + turn_pending_train_type(state), 0], c, seller,
                            turn_pending_train_type(state))
        return
    count = enumerate_legal_actions(state, ids)
    for i in range(count):
        info = decode_action(x.phase, ids[i])
        t = info.train_type
        if info.type == ACTION_TRAIN_TYPE:
            if held >= <int>MAX_ROUTES:
                continue
            if seller >= 0:
                _fill_corp_sale(state, scratch, &buffer[<int>ROW_TRAIN + t, 0], c, seller, t)
            else:
                buffer[<int>ROW_TRAIN + t, T_ADD_DELTA] = \
                    (route_max_after_purchase(state, scratch, c, t, -1) - route) / PRICE_DIV
        elif info.type == ACTION_TRADE_IN:
            buffer[<int>ROW_TRAIN + t, T_TRADE_IN_DELTA] = \
                (route_max_after_purchase(state, scratch, c, <int>D_TRAIN, t) - route) / PRICE_DIV
        else:  # a discard: one fewer of type t
            copy_row(scratch, state)
            trains_set_count(scratch, c, t, corp_train_count(state, c, t) - 1)
            buffer[<int>ROW_TRAIN + t, T_DISCARD_DELTA] = _route_delta(scratch, state, c)


cdef void _fill_corp_sale(GameState state, GameState scratch, float* row, int buyer, int seller, int t) noexcept nogil:
    """TRAIN 10 and 13 on type t's row for one of ``seller``'s passing to
    ``buyer``: a corp's train starts no phase."""
    copy_row(scratch, state)
    trains_set_count(scratch, buyer, t, corp_train_count(state, buyer, t) + 1)
    trains_set_count(scratch, seller, t, corp_train_count(state, seller, t) - 1)
    row[T_ADD_DELTA] = _route_delta(scratch, state, buyer)
    row[T2_SELLER_DELTA] = _route_delta(scratch, state, seller)


cdef void _fill_player2(GameState state, float* row, int i, const _Context* x) noexcept nogil:
    """Slots 1-10 are layout 1's (``_fill_player``)."""
    cdef int n = x.num_players
    cdef int cash = player_cash(state, i)
    cdef int presidencies = 0, privates = 0, income = 0
    cdef int c, p
    row[0] = 1
    row[PL_SELECTED] = i == x.actor
    if x.actor >= 0:
        row[PL_SEAT + (i - x.actor + n) % n] = 1
    row[PL_PRIORITY] = i == _global(state, GLOBALS.priority_player)
    row[PL_CASH] = cash / MONEY_DIV
    row[PL_VALUE] = player_value(state, i) / MONEY_DIV
    row[PL_CERTS_OF_LIMIT] = player_cert_count(state, i) / <float>CERT_LIMIT[n]
    for c in range(<int>NUM_CORPS):
        row[PL2_SHARES + c] = player_shares(state, i, c) / <float>SHARES_PER_CORP
        row[PL2_SOLD + c] = player_sold_mask(state, i) >> c & 1
        presidencies += corp_president(state, c) == i
        if corp_floated(state, c):
            income += _payout(player_shares(state, i, c), _corp_field(state, c, CORP_FIELDS.route_cache))
    for p in range(<int>NUM_PRIVATES):
        if p:
            row[PL2_BIDS + p - 1] = player_bid(state, i, p) / PRICE_DIV
        if private_owner(state, p) == <int>OWNER_PLAYER + i:
            privates += 1
            income += private_revenue(state, p)
    row[PL2_PASSED] = player_passed(state, i)
    row[PL2_PRESIDENCIES] = presidencies / <float>NUM_CORPS
    row[PL2_PRIVATES] = privates / <float>NUM_PRIVATES
    # The operating corp's president: an emergency sale since the turn's last purchase.
    row[PL2_EMERGENCY_SOLD] = x.operating >= 0 and corp_president(state, x.operating) == i and turn_ebuy_sold_any(state)
    row[PL2_FLOATS_ALONE] = cash >= LOWEST_FLOAT
    row[PL2_RESUMES] = (x.phase == DPHASE_BID and turn_auction_private(state) >= 0
                        and i == _global(state, GLOBALS.auction_turn))
    row[PL2_INCOME] = income / MONEY_DIV
    row[PL2_LOG_CASH] = _log_yen(cash)
    row[PL2_LOG_VALUE] = _log_yen(player_value(state, i))


# The row map against the tables, and each row type's last slot against its width.
assert ROW_PRIVATE - ROW_CORP == NUM_CORPS and ROW_TRAIN - ROW_PRIVATE == NUM_PRIVATES
assert ROW_HEX - ROW_TRAIN == NUM_TRAIN_TYPES and ROW_CANDIDATE - ROW_HEX == NUM_HEXES
assert ROW_PLAYER - ROW_CANDIDATE == MAX_LAY_CANDIDATES
assert G_PHASE + NUM_DECISION_PHASES == G_ROUND and G_TRAIN_SOURCE + 3 == TW_GLOBAL
assert PR_KIND + 4 == PR_AMOUNTS and PR_AMOUNTS + ACTION_SIZE_BID == PR_REFERENCE and PR_FUNDS + 1 == TW_PRICE
assert max(<int>ACTION_SIZE_PAR, <int>ACTION_SIZE_TRAIN_PRICE, <int>ACTION_SIZE_PRIVATE_PRICE) <= ACTION_SIZE_BID
assert DV_POOL + 1 == TW_DIVIDEND
assert C_ZONE + 4 == C_SALE_PRICES and C_SALE_PRICES + MAX_SALE_SHARES == C_SOLD_OUT and C_PRIVATES + 1 == TW_CORP
assert P_EXCHANGE + 1 == TW_PRIVATE and T_DISCARD_DELTA + 1 == TW_TRAIN and H_TOKEN_DELTA + 1 == TW_HEX
assert TB_LABEL + 3 == TW_TILE_BLOCK and TILE_BLOCK + TW_TILE_BLOCK == H_D_REVENUE == CA_ADDED
assert CA_ADDED + NUM_PAIRS == CA_COST and CA_OTHER_CENTERS + 1 == TW_CANDIDATE
assert PL_SEAT + 5 == PL_PRIORITY and PL_EMERGENCY_SOLD + 1 == TW_PLAYER
assert <int>TOKEN_DIM == max(TW_GLOBAL, TW_PRICE, TW_DIVIDEND, TW_CORP, TW_PRIVATE, TW_TRAIN, TW_HEX, TW_CANDIDATE,
                             TW_PLAYER)
# Layout 2: the same, and the rows it shares with layout 1 are no narrower.
assert <int>G2_PRE == <int>TW_GLOBAL and G2_SUPPLY + NUM_SUPPLY_TILES == TW2_GLOBAL
assert <int>PR2_FLOAT_SHARES == <int>TW_PRICE and PR2_FLOAT_SHARES + NUM_PAR_PRICES == TW2_PRICE
assert DV2_AFTER_RUST + 1 == TW2_DIVIDEND
assert <int>C2_RANK == <int>C_OR_POSITION and C2_RANK + NUM_CORPS == C2_OPERATED and C2_LOG_ROUTE + 1 == TW2_CORP
assert <int>TW2_PRIVATE == <int>TW_PRIVATE and <int>TW2_HEX == <int>TW_HEX
assert <int>T2_SELLER_DELTA == <int>TW_TRAIN and T2_STARTS_PHASE + 1 == TW2_TRAIN
assert <int>CA2_NEXT_DELTA == <int>TW_CANDIDATE and CA2_NEXT_DELTA + 1 == TW2_CANDIDATE
assert <int>PL2_SHARES == <int>PL_CERTS and PL2_SHARES + NUM_CORPS == PL2_SOLD and PL2_SOLD + NUM_CORPS == PL2_BIDS
assert PL2_BIDS + NUM_PRIVATES - 1 == PL2_PASSED and PL2_LOG_VALUE + 1 == TW2_PLAYER
assert <int>TOKEN_DIM_2 == max(TW2_GLOBAL, TW2_PRICE, TW2_DIVIDEND, TW2_CORP, TW2_PRIVATE, TW2_TRAIN, TW2_HEX,
                               TW2_CANDIDATE, TW2_PLAYER)
assert MAX_ROUTES == 4 and FLOAT_SHARES == 5  # the divisors token-data-1889.md states
# The fills' buffers are sized with literals.
assert MAX_LAY_CANDIDATES == 15 and NUM_CORPS == 7 and MAX_ACTION_SIZE == 46
