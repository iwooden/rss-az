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

**Layout 3** (1889/v2-design.md; token-data-1889.md, "Layout 3") is one
packed vector a state, filled by ``fill_inputs``: rows that reuse layout 2's
fills through slot maps, dense pair blocks, and a choice table of what each
of the decision's actions does, applied alone on a scratch state
(``apply_alone``). Unlike layouts 1 and 2, its fill holds the GIL. See its
sections below.
"""

from collections import namedtuple

import numpy as np

from libc.math cimport log1p
from libc.stdint cimport int16_t, uint16_t, uint32_t
from libc.string cimport memset

from g1889.actions cimport (
    ACTION_BANKRUPT, ACTION_BUY_IPO, ACTION_BUY_PRIVATE, ACTION_BUY_TOPMOST, ACTION_DR_EXCHANGE, ACTION_MF_PORT,
    ACTION_PAR, ACTION_PASS, ACTION_PAYOUT, ACTION_PRICE_POINT, ACTION_SELECT_BID, ACTION_SELECT_PAR,
    ACTION_SELECT_SALE, ACTION_TRADE_IN, ACTION_TRAIN_TYPE, ActionInfo, LayCandidate, can_buy_private,
    can_exchange_dr, can_token, decode_action, emergency_need, emergency_sale_limit, enumerate_lay_candidates,
    enumerate_legal_actions, er_candidates, get_decision_phase, in_emergency, lay_candidates, must_buy_train,
    price_amount, sale_limit,
)
from g1889.constants cimport (
    ACTION_SIZE_BID, ACTION_SIZE_PAR, ACTION_SIZE_PRIVATE_PRICE, ACTION_SIZE_TRAIN_PRICE, AUCTION_BID, AUCTION_BUY,
    BUY_PRIVATE_POST_DR, BUY_PRIVATE_POST_PRIVATE, BUY_PRIVATE_PRE_DR, BUY_PRIVATE_PRE_MF, BUY_PRIVATE_PRE_PRIVATE,
    BUY_TRAIN_CORP, BUY_TRAIN_SELL, DPHASE_AUCTION, DPHASE_BID, DPHASE_BUY_PRIVATE_POST, DPHASE_BUY_PRIVATE_PRE,
    DPHASE_BUY_TRAIN, DPHASE_DISCARD, DPHASE_DIVIDEND, DPHASE_EMERGENCY_SELL_AMOUNT, DPHASE_ER_BONUS, DPHASE_PAR,
    DPHASE_PORT, DPHASE_PRIVATE_PRICE, DPHASE_SELECT_TRAIN, DPHASE_SELL_AMOUNT, DPHASE_STOCK, DPHASE_TILE,
    DPHASE_TOKEN, DPHASE_TRACK, DPHASE_TRAIN_PRICE, ER_BONUS_CANDIDATE, MAX_ACTION_SIZE, MAX_SALE_SHARES,
    NUM_DECISION_PHASES, OWNER_CLOSED, OWNER_CORP, OWNER_NOT_IN_PLAY, OWNER_PLAYER, OWNER_UNSOLD, PHASE_ACTION_SIZE,
    PORT_LAY, ROUND_AUCTION, ROUND_GAME_OVER, ROUND_OPERATING, ROUND_STOCK, SHARES_PER_CORP, STEP_BUY_PRIVATE_POST,
    STEP_BUY_PRIVATE_PRE, STOCK_DR, STOCK_IPO, STOCK_MF, STOCK_PAR, STOCK_POOL, STOCK_SELL, TOKEN_HEX, TRACK_HEX,
    TRACK_PASS, TRAIN_SOURCE_CORP, UNIFIED_ACTION_DIM,
)
from g1889.data.tables cimport (
    CERT_LIMIT, CORP_HOME, DR_EXCHANGE_CORP, D_TRAIN, ENDPOINT_PAIRS, ER_HEX, FLOAT_PERCENT, HEX_BLOCKER,
    HEX_EXIT_MASK, HEX_HOME_CORP, HEX_NEIGHBOR, HEX_RC, LABEL_NONE, MARKET_PRICE, MAX_CITY_SLOTS, MAX_CORP_TOKENS,
    MAX_LAY_CANDIDATES, NUM_CORPS, NUM_GAME_PHASES, NUM_HEXES, NUM_PAIRS, NUM_PAR_PRICES, NUM_PORT_LAYS, NUM_PRIVATES,
    NUM_RCS, NUM_SIDES, NUM_SUPPLY_TILES, NUM_TILES, NUM_TOKEN_HEXES, NUM_TRACK_HEXES, NUM_TRAIN_TYPES, PAR_PRICE,
    PHASE_ON_TRAIN, PHASE_ORS, PHASE_TILE_COLORS, PHASE_TRAIN_LIMIT, PORT_LAY_HEX, PORT_LAY_ROT, PORT_TILE, PRIVATE_DR,
    PRIVATE_MF, PRIVATE_VALUE, RC_HEX, RC_NONE, TERRAIN_MOUNTAIN, TERRAIN_WATER, TILE_BLOCKS_LAY, TILE_COLOR,
    TILE_COST, TILE_COUNT, TILE_EXITS, TILE_LABEL, TILE_MASK, TILE_RC, TILE_REVENUE, TILE_SLOTS, TILE_TERRAIN,
    TOKEN_HEXES, TRACK_HEXES, TRAIN_PRICE, TRAIN_RUSTS_ON, UPG_ROT, UPG_START, UPG_TILE,
)
from g1889.entities.bank cimport bank_broken, bank_cash
from g1889.entities.corp cimport (
    PRESIDENT_SHARES, corp_cash, corp_emergency_sale, corp_floated, corp_floated_mask, corp_ipo_shares, corp_par_index,
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
    corp_train_count, corp_train_total, depot_count, pool_count, trade_in_price, trains_cheapest_type,
    trains_depot_types, trains_first_crowded, trains_limit, trains_set_count,
)
from g1889.entities.turn cimport (
    turn_active_player, turn_auction_discount, turn_auction_pass_payouts, turn_auction_private,
    turn_ebuy_sold_any, turn_ebuy_sold_mask, turn_num_players, turn_operating_corp, turn_pending_corp,
    turn_pending_hex, turn_pending_private, turn_pending_train_source, turn_pending_train_type, turn_return_step,
    turn_round, turn_route_revenue, turn_sr_bought_corp, turn_sr_sold_mask, turn_turn,
)
from g1889.routes.solver cimport (
    MAX_ROUTES, route_hex, route_max, route_max_after_purchase, routes_refresh_hypotheticals, solve_routes_via,
)
from g1889.state cimport CORP_FIELDS, GLOBALS, LAYOUT, PLAYER_FIELDS, GameState, copy_row, view_row
from g1889.driver cimport apply_alone


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
    decision shows what the trains on sale would earn (TRAIN 10 and 13).
    Layout 3 keeps its own scratch states (``_scratch3``)."""
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


# What a lay on one hex is measured against (``_lay_base``), and what one
# candidate's lay gives (``_lay_outcome``): layouts 1 and 2's CANDIDATE rows,
# and layout 3's TILE, ER_BONUS, and TRACK rows.
cdef struct _LayBase:
    int hex
    int corp                        # the decision corp, ER's buyer at ER_BONUS
    int game_phase
    int touch                       # map_touch_mask: one search serves every candidate's copy
    uint32_t old_paths
    int route                       # the corp's route max now,
    int with_next                   # and with the next train (its route max now without room)
    int next_type
    bint room                       # the corp has room for the next train, and the purchase
    bint starts                     # starts a phase
    _Reach before[7]                # NUM_CORPS: the reach of each corp whose walks enter the hex

cdef struct _LayOutcome:
    int revenue                     # the laid tile's revenue now
    int copies                      # its copies left in supply
    int hexes                       # Δ of the corp's reach
    int centers
    int reach_revenue
    int route_after                 # its route max after the lay,
    int next_after                  # and with the next train
    bint route_available
    int grown                       # other corps whose reached centers grow,
    int gained                      # and the centers they gain


cdef void _lay_base(GameState state, const _Context* x, int hex, int corp, bint next_train, const _Reach* reaches,
                    _LayBase* b) noexcept nogil:
    """Lays on ``hex`` by ``corp``, measured with the next train if
    ``next_train`` (layouts 2 and 3). ``reaches`` holds every corp's reach
    now, or is NULL to compute those the lays need."""
    cdef int c
    b.hex = hex
    b.corp = corp
    b.game_phase = x.game_phase
    b.touch = map_touch_mask(state, hex)
    b.old_paths = hex_paths(state, hex)
    b.next_type = x.next_type
    b.room = next_train and x.next_type >= 0 and corp_train_total(state, corp) < x.limit
    b.starts = b.room and _starts_phase(x.game_phase, x.next_type)
    b.route = _corp_field(state, corp, CORP_FIELDS.route_cache)
    b.with_next = _corp_field(state, corp, CORP_FIELDS.route_with_next) if b.room else b.route
    for c in range(<int>NUM_CORPS):
        if c == corp or graph_entry(state, c, hex):
            b.before[c] = reaches[c] if reaches != NULL else _reach(state, c, x.game_phase)


cdef void _lay_outcome(GameState state, GameState scratch, const _LayBase* b, int tile, int rotation, bint others,
                       _LayOutcome* o) noexcept nogil:
    """One lay tried on the scratch state (§7.1), which it leaves changed.
    Another corp is refreshed only if ``others``, if its walks reach a side
    of the hex, since a lay on a hex they never enter cannot change its
    reach, and only if the lay may have gained it a center
    (``map_center_bounds``). With room, the route max with the next train
    comes from the same forest, unless the purchase starts a phase, which is
    then tried last on the laid state."""
    cdef _Reach after
    cdef int bounds[7]              # NUM_CORPS
    cdef int c
    copy_row(scratch, state)
    map_put_tile(scratch, b.hex, tile, rotation, b.touch)
    o.revenue = map_revenue(scratch, b.hex, b.game_phase, False)
    o.copies = supply_count(scratch, tile)
    o.grown = o.gained = 0
    if others:
        map_center_bounds(scratch, bounds)
    for c in range(<int>NUM_CORPS):
        if c != b.corp and (not others or not graph_entry(state, c, b.hex) or bounds[c] <= b.before[c].centers):
            continue
        graph_refresh(scratch, c)
        after = _reach(scratch, c, b.game_phase)
        if c == b.corp:
            o.hexes = after.hexes - b.before[c].hexes
            o.centers = after.centers - b.before[c].centers
            o.reach_revenue = after.revenue - b.before[c].revenue
            # A lay only adds routes, and those pass its hex: search them alone.
            if b.room and not b.starts:
                o.route_after = solve_routes_via(scratch, c, b.hex, b.route, b.next_type, b.with_next, &o.next_after)
            else:
                o.route_after = o.next_after = solve_routes_via(scratch, c, b.hex, b.route, -1, 0, NULL)
            o.route_available = graph_route_available(scratch, c)
        elif after.centers > b.before[c].centers:
            o.grown += 1
            o.gained += after.centers - b.before[c].centers
    if b.starts:  # last: the purchase starts a phase on the scratch state
        trains_set_count(scratch, b.corp, b.next_type, corp_train_count(scratch, b.corp, b.next_type) + 1)
        phase_on_train_bought(scratch, b.next_type)
        o.next_after = solve_routes_via(scratch, b.corp, b.hex, b.with_next, -1, 0, NULL)


cdef void _fill_candidates(GameState state, GameState scratch, float[:, ::1] buffer, const _Context* x,
                           int layout) noexcept nogil:
    """§5.8's rows at TILE and ER_BONUS, row k for candidate k, each lay tried
    on the scratch state (``_lay_outcome``). The deltas are the decision
    corp's, the buyer at ER_BONUS. Layout 2 divides the other corps' centers
    by 50 and adds CANDIDATE 108: the lay's Δ route max were the corp to buy
    the next train (``route_with_next``'s purchase, tried on the laid scratch
    state)."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef _LayBase base
    cdef _LayOutcome o
    cdef int hex, k, i, tile, rotation
    cdef int count = tile_candidates(state, x.phase, &hex, candidates)
    cdef uint32_t added
    cdef float cost
    cdef float* row
    if count == 0:
        return
    _lay_base(state, x, hex, x.corp, layout == 2, NULL, &base)
    cost = 0 if x.phase == DPHASE_ER_BONUS else map_lay_cost(state, hex, x.corp) / PRICE_DIV
    for k in range(count):
        tile = candidates[k].tile
        rotation = candidates[k].rotation
        row = &buffer[<int>ROW_CANDIDATE + k, 0]
        _lay_outcome(state, scratch, &base, tile, rotation, True, &o)
        row[0] = 1
        row[CA_ER] = x.phase == DPHASE_ER_BONUS
        _fill_tile_block(row + <int>TILE_BLOCK, tile, rotation, o.revenue)
        added = TILE_MASK[tile][rotation] & ~base.old_paths
        for i in range(<int>NUM_PAIRS):
            row[CA_ADDED + i] = added >> i & 1
        row[CA_COST] = cost
        row[CA_COPIES] = o.copies / <float>TILE_COUNT[tile]
        row[CA_HEXES] = o.hexes / <float>NUM_HEXES
        row[CA_CENTERS] = o.centers / <float>NUM_RCS
        row[CA_REVENUE] = o.reach_revenue / PRICE_DIV
        row[CA_ROUTE_DELTA] = (o.route_after - base.route) / PRICE_DIV
        row[CA_ROUTE_AVAILABLE] = o.route_available
        row[CA_OTHERS] = o.grown / <float>(NUM_CORPS - 1)
        row[CA_OTHER_CENTERS] = o.gained / <float>NUM_RCS
        if layout == 2:
            row[CA_OTHER_CENTERS] = o.gained / 50.0
            row[CA2_NEXT_DELTA] = (o.next_after - base.with_next) / PRICE_DIV  # without room, the trains it holds


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

cdef inline bint _starts_phase(int game_phase, int train_type) noexcept nogil:
    """Whether the next depot purchase of ``train_type`` starts a new game
    phase: it is the next phase's train (``PHASE_ON_TRAIN``)."""
    return game_phase + 1 < <int>NUM_GAME_PHASES and PHASE_ON_TRAIN[game_phase + 1] == train_type


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
        buffer[<int>ROW_TRAIN + i, T2_STARTS_PHASE] = _starts_phase(x.game_phase, i)
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


# =============================================================================
# LAYOUT 3
# =============================================================================
#
# One packed vector a state (token-data-1889.md, "Layout 3"): the rows, the
# pair blocks, and the choice table at the InputOffset3 offsets. The rows
# keep layout 2's slots where token-data-1889.md says so: each is filled by
# layout 2's fill into a scratch row and copied through a slot map, so the
# two layouts cannot drift apart. Build distance's slots and channels are
# zero until it lands (v2-design §10, step 6).

cdef enum:
    # GLOBAL: 1-89 are layout 2's
    G3_TRAINS = 90                  # + G3_TRAIN_SLOTS × train type + one of the T3_ slots
    G3_TRAIN_SLOTS = 7
    G3_REFERENCE = 132
    G3_FUNDS = 133
    G3_REVENUE = 134

    # GLOBAL's per-type block, from the TRAIN row
    T3_DEPOT = 0
    T3_POOL = 1
    T3_DEPOT_SELLS = 2
    T3_CHEAPEST = 3
    T3_RUSTED = 4
    T3_RUSTS_NEXT = 5
    T3_STARTS_PHASE = 6

    # CORP: 0-69 are layout 2's (CORP3_FROM2), and 70-75 build distance's
    C3_KEPT = 70

    # CENTER: its hex's layout-2 slots 1 and 78-93, its tile, and its revenue now
    CE3_SELECTED = 1
    CE3_TILE = 2
    CE3_ROTATION = 3
    CE3_REVENUE = 4
    CE3_HEX = 5                     # + HEX slot - H_D_REVENUE

    # Pair blocks' channels
    PC3_SHARES = 0                  # PLAYER x CORP
    PC3_PRESIDENT = 1
    PC3_SOLD = 2
    PP3_OWNS = 0                    # PLAYER x PRIVATE
    PP3_BID = 1
    CC3_TOKEN = 0                   # CORP x CENTER; 3 is build distance's
    CC3_REACHES = 1
    CC3_ROUTE = 2

# Layout 2's slot behind each layout 3 slot of the rows that keep them.
cdef int CORP3_FROM2[70]            # C3_KEPT: CORP 0-14, 20-53, 56-76 (15-19, 54, 55 leave the row)
cdef int PRIVATE3_FROM2[15]         # TW3_PRIVATE: PRIVATE 0-5, 7-15 (6, closed, is a hidden row; 16, 17 legality)
cdef int PLAYER3_FROM2[20]          # TW3_PLAYER: PLAYER 0-10, 31-39 (11-30 are pair blocks)
_CORP3_FROM2 = (*range(0, 15), *range(20, 54), *range(56, 77))
_PRIVATE3_FROM2 = (*range(0, 6), *range(7, 16))
_PLAYER3_FROM2 = (*range(0, 11), *range(31, 40))
for _i, _slot in enumerate(_CORP3_FROM2):
    CORP3_FROM2[_i] = _slot
for _i, _slot in enumerate(_PRIVATE3_FROM2):
    PRIVATE3_FROM2[_i] = _slot
for _i, _slot in enumerate(_PLAYER3_FROM2):
    PLAYER3_FROM2[_i] = _slot


cdef int input_size3(int max_players) noexcept nogil:
    """The packed vector's length at capacity ``max_players``."""
    return <int>IN3_PLAYERS + <int>IN3_PER_PLAYER * max_players


cdef inline void _copy_slots(float* row, const float* source, const int* slots, int count) noexcept nogil:
    cdef int i
    for i in range(count):
        row[i] = source[slots[i]]


# Layout 3's two scratch states by capacity, made on first use: one for the
# applied action and the lays, one for the solves on it. The fill holds the
# GIL throughout, so nothing else uses them meanwhile.
cdef list _SCRATCH3 = [None] * 6    # indexed by capacity, 3-5
# Timing only (``_set_profile3``): parts of the choice table to skip.
cdef int _PROFILE3 = 0


cdef GameState _scratch3(GameState state, int which):
    cdef object pair = _SCRATCH3[state.max_players]
    if pair is None:
        pair = (GameState.from_array(state._array, turn_num_players(state), state.max_players),
                GameState.from_array(state._array, turn_num_players(state), state.max_players))
        _SCRATCH3[state.max_players] = pair
    return <GameState>pair[which]


def _set_profile3(int skip):
    """Timing only: skip parts of layout 3's choice table, for the profile
    scripts; 0, the default, fills everything. Bits: 1 the whole table, 2
    TRACK's best over candidates (17-25 stay but 18-25), 4 TOKEN's other
    corps' solves (0-3), 8 BUY_TRAIN's source previews, 16 TRACK 24-25 (other
    corps' reach), 32 TRACK 19 (the next train)."""
    global _PROFILE3
    _PROFILE3 = skip


cdef int _fill_inputs3(GameState state, float* out, int num_players, int max_players) except -1:
    """Layout 3's packed vector: the parts, then the choice table. Holds the GIL."""
    cdef GameState a = _scratch3(state, 0)
    cdef GameState b = _scratch3(state, 1)
    cdef _Context x
    _fill_parts3(state, a, out, num_players, max_players, &x)
    return _fill_choices3(state, a, b, out + <int>IN3_CHOICES, &x, _PROFILE3)


cdef void _fill_parts3(GameState state, GameState scratch, float* out, int num_players, int max_players,
                       _Context* x) noexcept nogil:
    """Layout 3's prologue (layout 2's), the context ``x``, and every part
    but the choice table, which is left zero."""
    cdef _Context plain
    cdef float source[109]          # TOKEN_DIM_2: one layout 2 row
    cdef int c, p, r, i
    _prologue2(state, scratch)
    memset(out, 0, input_size3(max_players) * sizeof(float))
    x.num_players = num_players
    x.phase = get_decision_phase(state)
    x.game_round = turn_round(state)
    x.game_phase = game_phase(state)
    x.actor = turn_active_player(state) if x.phase >= 0 else -1
    x.corp = _decision_corp(state, x.phase)
    _context2(state, x)
    _fill_global3(state, out + <int>IN3_GLOBAL, x)
    for c in range(<int>NUM_CORPS):
        memset(source, 0, sizeof(source))
        _fill_corp2(state, source, c, x)
        _copy_slots(out + <int>IN3_CORP + c * <int>TW3_CORP, source, CORP3_FROM2, <int>C3_KEPT)
    for p in range(<int>NUM_PRIVATES):
        if private_owner(state, p) == <int>OWNER_CLOSED:
            continue                # hidden, as a private out of play is (_fill_private)
        memset(source, 0, sizeof(source))
        _fill_private(state, source, p, x)
        if p != <int>PRIVATE_MF:    # the port tile is MF's, as in layout 2
            source[P_PORT_USED] = 0
        _copy_slots(out + <int>IN3_PRIVATE + p * <int>TW3_PRIVATE, source, PRIVATE3_FROM2, <int>TW3_PRIVATE)
    plain = x[0]
    plain.corp = -1                 # CENTER rows hold nothing of the decision corp's: no HEX 94-102
    for r in range(<int>NUM_RCS):
        memset(source, 0, sizeof(source))
        _fill_center3(state, out + <int>IN3_CENTER + r * <int>TW3_CENTER, source, r, &plain)
    _fill_pairs3(state, out, num_players, max_players)
    for i in range(num_players):
        memset(source, 0, sizeof(source))
        _fill_player2(state, source, i, x)
        _copy_slots(out + <int>IN3_PLAYERS + i * <int>TW3_PLAYER, source, PLAYER3_FROM2, <int>TW3_PLAYER)


cdef void _fill_global3(GameState state, float* row, const _Context* x) noexcept nogil:
    """Layout 2's GLOBAL (slots 1-89), then each train type's TRAIN 3, 5-9,
    and 14, PRICE 22 and 23 at the decisions with a PRICE row, and
    DIVIDEND 1."""
    cdef float source[24]           # TW_PRICE; a TRAIN row's 15 fit
    cdef float* block
    cdef int t
    _fill_global2(state, row, x)
    for t in range(<int>NUM_TRAIN_TYPES):
        memset(source, 0, sizeof(source))
        _fill_train(state, source, t, x)
        block = row + <int>G3_TRAINS + <int>G3_TRAIN_SLOTS * t
        block[T3_DEPOT] = source[T_DEPOT]
        block[T3_POOL] = source[T_POOL]
        block[T3_DEPOT_SELLS] = source[T_DEPOT_SELLS]
        block[T3_CHEAPEST] = source[T_CHEAPEST]
        block[T3_RUSTED] = source[T_RUSTED]
        block[T3_RUSTS_NEXT] = source[T_RUSTS_NEXT]
        block[T3_STARTS_PHASE] = _starts_phase(x.game_phase, t)
    if (x.phase == DPHASE_BID or x.phase == DPHASE_PAR or x.phase == DPHASE_TRAIN_PRICE
            or x.phase == DPHASE_PRIVATE_PRICE):
        memset(source, 0, sizeof(source))
        _fill_price(state, source, x)
        row[G3_REFERENCE] = source[PR_REFERENCE]
        row[G3_FUNDS] = source[PR_FUNDS]
    elif x.phase == DPHASE_DIVIDEND:
        row[G3_REVENUE] = turn_route_revenue(state) / PRICE_DIV


cdef void _fill_center3(GameState state, float* row, float* hex_row, int r, const _Context* x) noexcept nogil:
    """Revenue center r's row, from its hex's layout-2 row (``_fill_hex``
    into the zeroed ``hex_row``, with ``x`` naming no decision corp): HEX 1,
    the tile as its code and rotation, the tile block's revenue now, and
    HEX 78-93."""
    cdef int h = RC_HEX[r]
    cdef int i
    _fill_hex(state, hex_row, h, x)
    row[0] = 1
    row[CE3_SELECTED] = hex_row[H_SELECTED]
    row[CE3_TILE] = hex_tile(state, h) + 1
    row[CE3_ROTATION] = hex_rotation(state, h)
    row[CE3_REVENUE] = hex_row[<int>TILE_BLOCK + <int>TB_REVENUE]
    for i in range(<int>H_UPGRADES_ANY_PHASE + 1 - <int>H_D_REVENUE):
        row[CE3_HEX + i] = hex_row[H_D_REVENUE + i]


cdef void _fill_pairs3(GameState state, float* out, int num_players, int max_players) noexcept nogil:
    """The pair blocks, [first type][second type][channel]. A padded seat's
    entries and a hidden private's (out of play or closed) are zero, and so
    are build distance's channels. Reads the caches raw, after the prologue."""
    cdef float* player_corp = out + <int>IN3_PLAYERS + max_players * <int>TW3_PLAYER
    cdef float* player_private = player_corp + max_players * <int>NUM_CORPS * <int>CH3_PLAYER_CORP
    cdef float* cell
    cdef int i, c, p, r, h, owner
    for i in range(num_players):
        for c in range(<int>NUM_CORPS):
            cell = player_corp + (i * <int>NUM_CORPS + c) * <int>CH3_PLAYER_CORP
            cell[PC3_SHARES] = player_shares(state, i, c) / <float>SHARES_PER_CORP
            cell[PC3_PRESIDENT] = corp_president(state, c) == i
            cell[PC3_SOLD] = player_sold_mask(state, i) >> c & 1
    for p in range(<int>NUM_PRIVATES):
        owner = private_owner(state, p)
        if owner == <int>OWNER_NOT_IN_PLAY or owner == <int>OWNER_CLOSED:
            continue
        if owner >= <int>OWNER_CORP:
            out[<int>IN3_CORP_PRIVATE + ((owner - <int>OWNER_CORP) * <int>NUM_PRIVATES + p) * <int>CH3_CORP_PRIVATE] = 1
        for i in range(num_players):
            cell = player_private + (i * <int>NUM_PRIVATES + p) * <int>CH3_PLAYER_PRIVATE
            cell[PP3_OWNS] = owner == <int>OWNER_PLAYER + i
            cell[PP3_BID] = player_bid(state, i, p) / PRICE_DIV
    for c in range(<int>NUM_CORPS):
        for r in range(<int>NUM_RCS):
            h = RC_HEX[r]
            cell = out + <int>IN3_CORP_CENTER + (c * <int>NUM_RCS + r) * <int>CH3_CORP_CENTER
            cell[CC3_TOKEN] = hex_tokens(state, h) >> c & 1
            cell[CC3_REACHES] = (graph_entry(state, c, h) & <int>GRAPH_NODE) != 0
            cell[CC3_ROUTE] = route_hex(state, c, h)


# =============================================================================
# LAYOUT 3: THE CHOICE TABLE
# =============================================================================
#
# Row k describes the decision's phase-local action k (token-data-1889.md,
# "The choice table"): the integer columns, the shared block S0-S6, and the
# phase block. An applied row's shared block is its action applied alone on
# scratch state a (``apply_alone``) less the row; route solves on it use
# scratch state b. A row that only opens a sub-decision has a zero shared
# block and previews the sub-decision in its phase block. Illegal rows stay
# zero. Lays use the CANDIDATE rows' fast path (``_lay_outcome``), and the
# shared block's other slots there are closed forms.

cdef enum:
    # The shared block, from CHOICE3_SHARED
    SH_CASH = 0
    SH_TREASURY = 1
    SH_PRICE = 2
    SH_CERTS = 3
    SH_ROUTE = 4
    SH_NEXT = 5
    SH_COST = 6

    # Phase blocks, from CHOICE3_PHASE. AUCTION and BID:
    AB_PRICE = 0
    AB_MIN_BID = 1
    AB_HIGH_BID = 2
    AB_LEFT = 3
    # STOCK
    SK_KIND = 0                     # + IPO, pool, par, sell
    SK_PRESIDENT = 4
    SK_FLOATS = 5
    SK_TO_FLOAT = 6
    SK_SALE = 7
    SK_SALE_PRICE = 8
    # PAR
    PA_PAR = 0
    PA_FLOAT = 1
    PA_TREASURY = 2
    # SELL_AMOUNT and EMERGENCY_SELL_AMOUNT
    SA_SHARES = 0
    SA_PRICE = 1
    SA_PRESIDENT = 2
    SA_NEED = 3
    # BUY_PRIVATE_PRE, BUY_PRIVATE_POST, and PRIVATE_PRICE
    BP_PRICE = 0
    BP_LEAST = 1
    BP_FACE = 2
    BP_REVENUE = 3
    BP_LEFT = 4
    # TRACK: 0-15 are HEX 86-101
    TK_HEX_SLOTS = 16
    TK_CANDIDATES = 16
    TK_COST = 17
    TK_ROUTE = 18
    TK_NEXT = 19
    TK_HEXES = 20
    TK_CENTERS = 21
    TK_REVENUE = 22
    TK_AVAILABLE = 23
    TK_OTHERS = 24
    TK_OTHER_CENTERS = 25
    # TILE and ER_BONUS: 0-20 are the paths added
    TL_COPIES = 21
    TL_HEXES = 22
    TL_CENTERS = 23
    TL_REVENUE = 24
    TL_AVAILABLE = 25
    TL_OTHERS = 26
    TL_OTHER_CENTERS = 27
    # TOKEN
    TO_LOST = 0
    TO_LARGEST = 1
    TO_NEXT_LOST = 2
    TO_NEXT_LARGEST = 3
    TO_FREE = 4
    # DIVIDEND
    DI_PER_SHARE = 0
    DI_TOTAL = 1
    DI_SHORTFALL = 2
    DI_RIVALS = 3                   # + seats after the actor - 1
    # BUY_TRAIN, SELECT_TRAIN, and TRAIN_PRICE
    TN_TYPE = 0                     # + train type
    TN_PRICE = 6
    TN_GAIN = 7
    TN_STARTS = 8
    TN_RUSTED = 9
    TN_OTHERS_RUSTED = 10
    TN_SELLER = 11
    TN_CONTRIBUTION = 12
    TN_BLOCK = 13
    TN_BLOCK_PRICE = 14
    TN_NEED = 15

    # _set_profile3's bits
    P3_NO_CHOICES = 1
    P3_NO_TRACK_BEST = 2
    P3_NO_TOKEN_OTHERS = 4
    P3_NO_BUY_PREVIEWS = 8
    P3_TRACK_NO_OTHERS = 16
    P3_TRACK_NO_NEXT = 32

    GAIN_UNSET = -32768             # a SELECT_TRAIN choice's gain not yet solved


# A train a SELECT_TRAIN choice buys, as the train block shows it.
cdef struct _TrainOption:
    int type                        # the train bought: a D at a trade-in
    int price                       # 0 for a corp's train before TRAIN_PRICE
    int gain                        # the buyer's Δ route max
    bint starts                     # the purchase starts a phase,
    int rusted                      # which rusts these of the buyer's trains,
    int others_rusted               # and these of the other corps'
    int seller                      # a corp's train: the seller's Δ route max
    int contribution                # the president's, in an emergency


cdef inline int _route_now(GameState state, int corp) noexcept nogil:
    return _corp_field(state, corp, CORP_FIELDS.route_cache)


cdef inline int _route_next(GameState state, int corp) noexcept nogil:
    """The corp's route max with the next train, fresh after the prologue."""
    return _corp_field(state, corp, CORP_FIELDS.route_with_next)


cdef int _with_next3(GameState state, GameState scratch, int corp, int now) noexcept nogil:
    """``route_with_next`` of ``state``, solved on ``scratch``: with the next
    train bought while the corp has room, its route max ``now`` without."""
    cdef int next_type = trains_cheapest_type(state)
    if next_type >= 0 and corp_train_total(state, corp) < trains_limit(state):
        return route_max_after_purchase(state, scratch, corp, next_type, -1)
    return now


cdef inline float _to_float(int ipo, bint floated) noexcept nogil:
    """CORP 61: the shares that must still leave the IPO before the corp floats /5."""
    return 0 if floated else max(ipo - (<int>SHARES_PER_CORP - FLOAT_SHARES), 0) / <float>FLOAT_SHARES


cdef inline float _presidency(GameState state, GameState after, int corp, int player) noexcept nogil:
    """1 if ``player`` is the corp's president after and not before, -1 the reverse."""
    return (corp_president(after, corp) == player) - (corp_president(state, corp) == player)


cdef inline int _price_after_sale(GameState state, int corp, int shares) noexcept nogil:
    """The corp's share price after ``shares`` are sold: a row down each."""
    cdef int row = market_row(state, corp)
    cdef int col = market_col(state, corp)
    cdef int i
    for i in range(shares):
        market_moved(row, col, MOVE_DOWN, 1, &row, &col)
    return MARKET_PRICE[row][col]


cdef void _diff3(GameState state, GameState after, float* row, const _Context* x, int corp) noexcept nogil:
    """S0-S3: the actor's cash and certificates, and ``corp``'s treasury and
    share price, applied less on the row."""
    cdef float* shared = row + <int>CHOICE3_SHARED
    if x.actor >= 0:
        shared[SH_CASH] = (player_cash(after, x.actor) - player_cash(state, x.actor)) / MONEY_DIV
        shared[SH_CERTS] = ((player_cert_count(after, x.actor) - player_cert_count(state, x.actor))
                            / <float>CERT_LIMIT[x.num_players])
    if corp >= 0:
        shared[SH_TREASURY] = (corp_cash(after, corp) - corp_cash(state, corp)) / MONEY_DIV
        if corp_par_index(state, corp) >= 0 and corp_par_index(after, corp) >= 0:
            shared[SH_PRICE] = (market_price(after, corp) - market_price(state, corp)) / PRICE_DIV


cdef int _applied3(GameState state, GameState after, float* row, const _Context* x, int action_id,
                   int corp) except -1:
    """Apply the row's action alone on ``after``, a copy of the row, and
    write S0-S3 for ``corp``, the row's corp, or -1."""
    copy_row(after, state)
    apply_alone(after, action_id)
    _diff3(state, after, row, x, corp)
    return 0


cdef void _routes3(GameState state, GameState after, GameState scratch, float* row, int corp) noexcept nogil:
    """S4 and S5: the corp's route max, and with the next train, on the
    applied state ``after`` (a full solve), less on the row."""
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef int route = route_max(after, corp)
    shared[SH_ROUTE] = (route - _route_now(state, corp)) / PRICE_DIV
    shared[SH_NEXT] = (_with_next3(after, scratch, corp, route) - _route_next(state, corp)) / PRICE_DIV


cdef int _fill_choices3(GameState state, GameState a, GameState b, float* table, const _Context* x,
                        int skip) except -1:
    """Every engine-legal row of the decision; nothing without one."""
    cdef uint16_t ids[46]           # MAX_ACTION_SIZE
    cdef int gains[9]               # ACTION_SIZE_SELECT_TRAIN: each choice's gain, solved once a decision
    cdef int shared_route[3]        # TRAIN_PRICE's: the buyer after, the seller after, the buyer with the next
    cdef int count, i
    if x.phase < 0 or skip & P3_NO_CHOICES:
        return 0
    shared_route[0] = shared_route[1] = shared_route[2] = 0
    count = enumerate_legal_actions(state, ids)
    for i in range(9):
        gains[i] = GAIN_UNSET
    if x.phase == DPHASE_TRACK:
        _fill_track3(state, a, table, x, ids, count, skip)
    elif x.phase == DPHASE_TILE or x.phase == DPHASE_ER_BONUS:
        _fill_lays3(state, a, table, x)
    elif x.phase == DPHASE_TRAIN_PRICE:
        _train_price_routes3(state, a, b, x, shared_route)
    for i in range(count):
        _fill_row3(state, a, b, table + ids[i] * <int>CHOICE3_WIDTH, x, ids[i], gains, shared_route, skip)
    return 0


cdef int _fill_row3(GameState state, GameState a, GameState b, float* row, const _Context* x, int k, int* gains,
                    const int* shared_route, int skip) except -1:
    """Row k, but TRACK's hexes and the lays, which ``_fill_choices3`` has filled."""
    cdef ActionInfo info = decode_action(x.phase, k)
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef float* block = row + <int>CHOICE3_PHASE
    cdef int phase = x.phase
    cdef int c = x.corp
    cdef int p, amount, n
    if phase == DPHASE_AUCTION:
        if info.type == ACTION_SELECT_BID:  # BID's minimum bid, previewed
            p = info.private
            block[AB_PRICE] = private_price(state, p) / PRICE_DIV
            block[AB_MIN_BID] = private_min_bid(state, p) / PRICE_DIV
            block[AB_HIGH_BID] = private_high_bid(state, p) / PRICE_DIV
            block[AB_LEFT] = (player_available(state, x.actor, p) - private_min_bid(state, p)) / MONEY_DIV
            return 0
        _applied3(state, a, row, x, k, -1)
        if info.type == ACTION_BUY_TOPMOST:
            p = topmost_private(state)
            amount = private_price(state, p)
            shared[SH_COST] = amount / MONEY_DIV
            block[AB_PRICE] = amount / PRICE_DIV
            block[AB_LEFT] = (player_available(state, x.actor, p) - amount) / MONEY_DIV
    elif phase == DPHASE_BID:
        _applied3(state, a, row, x, k, -1)
        if info.type != ACTION_PASS:
            p = _bid_private(state)
            amount = price_amount(state, &info)
            shared[SH_COST] = amount / MONEY_DIV
            block[AB_PRICE] = amount / PRICE_DIV
            block[AB_MIN_BID] = private_min_bid(state, p) / PRICE_DIV
            block[AB_HIGH_BID] = private_high_bid(state, p) / PRICE_DIV
            block[AB_LEFT] = (player_available(state, x.actor, p) - amount) / MONEY_DIV
    elif phase == DPHASE_STOCK:
        _fill_stock3(state, a, row, x, k, &info)
    elif phase == DPHASE_PAR:
        amount = PAR_PRICE[k]
        _applied3(state, a, row, x, k, c)
        shared[SH_COST] = <int>PRESIDENT_SHARES * amount / MONEY_DIV
        block[PA_PAR] = amount / PRICE_DIV
        block[PA_FLOAT] = (min(<int>SHARES_PER_CORP - corp_ipo_shares(state, c) + player_cash(state, x.actor) // amount,
                               FLOAT_SHARES) / <float>FLOAT_SHARES)
        block[PA_TREASURY] = <int>SHARES_PER_CORP * amount / MONEY_DIV
    elif phase == DPHASE_SELL_AMOUNT or phase == DPHASE_EMERGENCY_SELL_AMOUNT:
        p = turn_pending_corp(state)  # the corp sold
        _applied3(state, a, row, x, k, p)
        block[SA_SHARES] = info.count / <float>MAX_SALE_SHARES
        block[SA_PRICE] = market_price(a, p) / PRICE_DIV
        if phase == DPHASE_SELL_AMOUNT:
            block[SA_PRESIDENT] = _presidency(state, a, p, x.actor)
        else:
            block[SA_NEED] = max(emergency_need(state, c) - info.count * market_price(state, p), 0) / MONEY_DIV
    elif phase == DPHASE_BUY_PRIVATE_PRE or phase == DPHASE_BUY_PRIVATE_POST or phase == DPHASE_PRIVATE_PRICE:
        if phase == DPHASE_PRIVATE_PRICE:
            amount = price_amount(state, &info)
            _applied3(state, a, row, x, k, c)
            shared[SH_COST] = amount / MONEY_DIV
            _fill_private_block3(state, block, c, turn_pending_private(state), amount)
        elif info.type == ACTION_BUY_PRIVATE:  # PRIVATE_PRICE's last point, previewed
            _fill_private_block3(state, block, c, info.private, min(2 * PRIVATE_VALUE[info.private], corp_cash(state, c)))
        elif info.type != ACTION_MF_PORT:  # done, or DR's exchange, for IR
            _applied3(state, a, row, x, k, <int>DR_EXCHANGE_CORP if info.type == ACTION_DR_EXCHANGE else c)
    elif phase == DPHASE_TRACK or phase == DPHASE_ER_BONUS:
        if info.type == ACTION_PASS:  # TRACK's hexes and ER's lays are filled
            _applied3(state, a, row, x, k, c)
    elif phase == DPHASE_TOKEN:
        if info.type == ACTION_PASS:
            _applied3(state, a, row, x, k, c)
        else:
            _fill_token3(state, a, b, row, x, k, skip)
    elif phase == DPHASE_DIVIDEND:
        _applied3(state, a, row, x, k, c)
        _fill_dividend3(state, block, x, info.type == ACTION_PAYOUT)
    elif phase == DPHASE_DISCARD:
        _applied3(state, a, row, x, k, c)
        _routes3(state, a, b, row, c)
    elif phase == DPHASE_BUY_TRAIN:
        if info.type == ACTION_PASS or info.type == ACTION_BANKRUPT:
            _applied3(state, a, row, x, k, c)
        elif info.type == ACTION_SELECT_SALE:  # EMERGENCY_SELL_AMOUNT's largest block, previewed
            amount = max(emergency_need(state, c), 0)
            n = emergency_sale_limit(state, c, info.corp, amount)
            block[TN_BLOCK] = n / <float>MAX_SALE_SHARES
            block[TN_BLOCK_PRICE] = _price_after_sale(state, info.corp, n) / PRICE_DIV
            block[TN_NEED] = max(amount - n * market_price(state, info.corp), 0) / MONEY_DIV
        elif not (skip & P3_NO_BUY_PREVIEWS):
            _fill_buy_preview3(state, a, b, block, x, k, gains)
    elif phase == DPHASE_SELECT_TRAIN:
        _fill_select3(state, a, b, row, x, k, gains)
    elif phase == DPHASE_TRAIN_PRICE:
        _fill_train_price3(state, a, row, x, k, &info, shared_route)
    elif phase == DPHASE_PORT:
        _applied3(state, a, row, x, k, c)  # none at an SR's
        row[<int>CHOICE3_HEX] = PORT_LAY_HEX[k - <int>PORT_LAY] + 1
        row[<int>CHOICE3_TILE] = <int>PORT_TILE + 1
        row[<int>CHOICE3_ROTATION] = PORT_LAY_ROT[k - <int>PORT_LAY]
        if c >= 0:
            _routes3(state, a, b, row, c)
    return 0


cdef int _fill_stock3(GameState state, GameState a, float* row, const _Context* x, int k,
                      const ActionInfo* info) except -1:
    """STOCK's row k: buys and DR's exchange applied; a par and a sale
    previewed, the sale by playing its largest block on the scratch state."""
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef float* block = row + <int>CHOICE3_PHASE
    cdef int c = info.corp
    cdef int limit
    if info.type == ACTION_PASS:
        return _applied3(state, a, row, x, k, -1)
    if info.type == ACTION_MF_PORT:
        return 0
    if info.type == ACTION_SELECT_PAR:
        block[SK_KIND + 2] = 1
        block[SK_PRESIDENT] = 1
        block[SK_TO_FLOAT] = _to_float(corp_ipo_shares(state, c) - <int>PRESIDENT_SHARES, corp_floated(state, c))
        return 0
    if info.type == ACTION_SELECT_SALE:
        limit = sale_limit(state, x.actor, c)
        block[SK_KIND + 3] = 1
        block[SK_TO_FLOAT] = _to_float(corp_ipo_shares(state, c), corp_floated(state, c))
        block[SK_SALE] = limit / <float>MAX_SALE_SHARES
        block[SK_SALE_PRICE] = _price_after_sale(state, c, limit) / PRICE_DIV
        copy_row(a, state)
        apply_alone(a, k)
        apply_alone(a, limit - 1)  # SELL_AMOUNT's largest block
        block[SK_PRESIDENT] = _presidency(state, a, c, x.actor)
        return 0
    if info.type == ACTION_DR_EXCHANGE:
        c = <int>DR_EXCHANGE_CORP
    elif info.type == ACTION_BUY_IPO:
        block[SK_KIND] = 1
        shared[SH_COST] = corp_par_price(state, c) / MONEY_DIV
    else:  # a pool share
        block[SK_KIND + 1] = 1
        shared[SH_COST] = market_price(state, c) / MONEY_DIV
    _applied3(state, a, row, x, k, c)
    block[SK_PRESIDENT] = _presidency(state, a, c, x.actor)
    block[SK_FLOATS] = corp_floated(a, c) and not corp_floated(state, c)
    block[SK_TO_FLOAT] = _to_float(corp_ipo_shares(a, c), corp_floated(a, c))
    return 0


cdef void _fill_private_block3(GameState state, float* block, int corp, int p, int price) noexcept nogil:
    block[BP_PRICE] = price / PRICE_DIV
    block[BP_LEAST] = (PRIVATE_VALUE[p] + 1) // 2 / PRICE_DIV
    block[BP_FACE] = PRIVATE_VALUE[p] / PRICE_DIV
    block[BP_REVENUE] = private_revenue(state, p) / PRICE_DIV
    block[BP_LEFT] = (corp_cash(state, corp) - price) / MONEY_DIV


cdef void _fill_dividend3(GameState state, float* block, const _Context* x, bint payout) noexcept nogil:
    """DIVIDEND's outcome for the operating corp's revenue r: what a payout
    pays (_fill_dividend's rounding) and what either leaves toward the next
    train."""
    cdef int c = x.corp
    cdef int revenue = turn_route_revenue(state)
    cdef int pool = _payout(corp_pool_shares(state, c), revenue)
    cdef int total = pool
    cdef int i
    if payout:
        block[DI_PER_SHARE] = _payout(1, revenue) / PRICE_DIV
        for i in range(x.num_players):
            total += _payout(player_shares(state, i, c), revenue)
        block[DI_TOTAL] = total / MONEY_DIV
        for i in range(1, x.num_players):
            block[DI_RIVALS + i - 1] = _payout(player_shares(state, (x.actor + i) % x.num_players, c), revenue) / PRICE_DIV
    if x.limit - corp_train_total(state, c) > 0 and x.next_type >= 0:
        block[DI_SHORTFALL] = max(TRAIN_PRICE[x.next_type] - corp_cash(state, c) - (pool if payout else revenue),
                                  0) / MONEY_DIV


cdef void _fill_track3(GameState state, GameState scratch, float* table, const _Context* x, const uint16_t* ids,
                       int count, int skip) noexcept nogil:
    """TRACK's hex rows: the hex's slots (``_fill_hex``'s HEX 86-101, with no
    selection and no candidate scan), its candidates' count and lay cost, and
    each of 18-25 the best of its own over the candidates' TILE outcomes."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef _Reach reaches[7]          # NUM_CORPS
    cdef float hex_row[109]         # TOKEN_DIM_2
    cdef _Context plain = x[0]
    cdef _LayBase base
    cdef _LayOutcome o
    cdef float* row
    cdef float* block
    cdef int i, j, k, h, n, c
    cdef int route = 0, next_ = 0, hexes = 0, centers = 0, revenue = 0, grown = 0, gained = 0
    cdef bint available = False
    plain.phase = -1
    for c in range(<int>NUM_CORPS):
        reaches[c] = _reach(state, c, x.game_phase)
    for i in range(count):
        k = ids[i]
        if k == <int>TRACK_PASS:
            continue
        h = TRACK_HEXES[k - <int>TRACK_HEX]
        row = table + k * <int>CHOICE3_WIDTH
        block = row + <int>CHOICE3_PHASE
        row[<int>CHOICE3_HEX] = h + 1
        row[<int>CHOICE3_TILE] = hex_tile(state, h) + 1
        row[<int>CHOICE3_ROTATION] = hex_rotation(state, h)
        memset(hex_row, 0, sizeof(hex_row))
        _fill_hex(state, hex_row, h, &plain)
        for j in range(<int>TK_HEX_SLOTS):
            block[j] = hex_row[H_DANGLING + j]
        n = lay_candidates(state, x.corp, h, candidates)
        block[TK_CANDIDATES] = n / <float>MAX_LAY_CANDIDATES
        block[TK_COST] = map_lay_cost(state, h, x.corp) / PRICE_DIV
        if skip & P3_NO_TRACK_BEST:
            continue
        _lay_base(state, x, h, x.corp, not (skip & P3_TRACK_NO_NEXT), reaches, &base)
        for j in range(n):
            _lay_outcome(state, scratch, &base, candidates[j].tile, candidates[j].rotation,
                         not (skip & P3_TRACK_NO_OTHERS), &o)
            if j == 0:
                route, next_, hexes, centers = o.route_after, o.next_after, o.hexes, o.centers
                revenue, available, grown, gained = o.reach_revenue, o.route_available, o.grown, o.gained
            else:
                route, next_ = max(route, o.route_after), max(next_, o.next_after)
                hexes, centers, revenue = max(hexes, o.hexes), max(centers, o.centers), max(revenue, o.reach_revenue)
                available = available or o.route_available
                grown, gained = min(grown, o.grown), min(gained, o.gained)
        block[TK_ROUTE] = (route - base.route) / PRICE_DIV
        block[TK_NEXT] = (next_ - base.with_next) / PRICE_DIV
        block[TK_HEXES] = hexes / <float>NUM_HEXES
        block[TK_CENTERS] = centers / <float>NUM_RCS
        block[TK_REVENUE] = revenue / PRICE_DIV
        block[TK_AVAILABLE] = available
        block[TK_OTHERS] = grown / <float>(NUM_CORPS - 1)
        block[TK_OTHER_CENTERS] = gained / 50.0


cdef void _fill_lays3(GameState state, GameState scratch, float* table, const _Context* x) noexcept nogil:
    """TILE's rows and ER_BONUS's candidate rows, each lay tried on the
    scratch state (``_lay_outcome``), as layout 2's CANDIDATE rows. The corp
    pays the lay cost at TILE; ER's seller pays nothing."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef _LayBase base
    cdef _LayOutcome o
    cdef int hex, k, i, tile, rotation, cost
    cdef int count = tile_candidates(state, x.phase, &hex, candidates)
    cdef int first = <int>ER_BONUS_CANDIDATE if x.phase == DPHASE_ER_BONUS else 0
    cdef uint32_t added
    cdef float* row
    cdef float* shared
    cdef float* block
    if count == 0:
        return
    _lay_base(state, x, hex, x.corp, True, NULL, &base)
    cost = 0 if x.phase == DPHASE_ER_BONUS else map_lay_cost(state, hex, x.corp)
    for k in range(count):
        tile = candidates[k].tile
        rotation = candidates[k].rotation
        _lay_outcome(state, scratch, &base, tile, rotation, True, &o)
        row = table + (first + k) * <int>CHOICE3_WIDTH
        shared = row + <int>CHOICE3_SHARED
        block = row + <int>CHOICE3_PHASE
        row[<int>CHOICE3_HEX] = hex + 1
        row[<int>CHOICE3_TILE] = tile + 1
        row[<int>CHOICE3_ROTATION] = rotation
        shared[SH_TREASURY] = -cost / MONEY_DIV
        shared[SH_ROUTE] = (o.route_after - base.route) / PRICE_DIV
        shared[SH_NEXT] = (o.next_after - base.with_next) / PRICE_DIV
        shared[SH_COST] = cost / MONEY_DIV
        added = TILE_MASK[tile][rotation] & ~base.old_paths
        for i in range(<int>NUM_PAIRS):
            block[i] = added >> i & 1
        block[TL_COPIES] = o.copies / <float>TILE_COUNT[tile]
        block[TL_HEXES] = o.hexes / <float>NUM_HEXES
        block[TL_CENTERS] = o.centers / <float>NUM_RCS
        block[TL_REVENUE] = o.reach_revenue / PRICE_DIV
        block[TL_AVAILABLE] = o.route_available
        block[TL_OTHERS] = o.grown / <float>(NUM_CORPS - 1)
        block[TL_OTHER_CENTERS] = o.gained / 50.0


cdef int _fill_token3(GameState state, GameState a, GameState b, float* row, const _Context* x, int k,
                      int skip) except -1:
    """A token's row, applied on ``a``. The corp's own token only adds
    routes, through its city, so its route maxes come from
    ``solve_routes_via``, as ``_fill_token_outcomes``' do; with room, the next
    train's from the same forest, unless the purchase starts a phase, which
    is tried last on ``a``. Other corps lose routes only where the token
    fills the city's last slot."""
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef int c = x.corp
    cdef int h = TOKEN_HEXES[k - <int>TOKEN_HEX]
    cdef int route = _route_now(state, c)
    cdef bint room = x.next_type >= 0 and corp_train_total(state, c) < x.limit
    cdef bint starts = room and _starts_phase(x.game_phase, x.next_type)
    cdef int with_next = _route_next(state, c) if room else route
    cdef int route_after, next_after
    shared[SH_COST] = corp_token_price(state, c) / MONEY_DIV
    _applied3(state, a, row, x, k, c)
    row[<int>CHOICE3_HEX] = h + 1
    if room and not starts:
        route_after = solve_routes_via(a, c, h, route, x.next_type, with_next, &next_after)
    else:
        route_after = next_after = solve_routes_via(a, c, h, route, -1, 0, NULL)
    if not (skip & P3_NO_TOKEN_OTHERS):
        _token_losses3(state, a, b, row + <int>CHOICE3_PHASE, c, h)
    row[<int>CHOICE3_PHASE + TO_FREE] = map_free_slots(a, h) / <float>MAX_CITY_SLOTS
    if starts:  # last: the purchase changes ``a``
        trains_set_count(a, c, x.next_type, corp_train_count(a, c, x.next_type) + 1)
        phase_on_train_bought(a, x.next_type)
        next_after = solve_routes_via(a, c, h, with_next, -1, 0, NULL)
    shared[SH_ROUTE] = (route_after - route) / PRICE_DIV
    shared[SH_NEXT] = (next_after - with_next) / PRICE_DIV
    return 0


cdef void _token_losses3(GameState state, GameState after, GameState scratch, float* block, int corp,
                         int hex) noexcept nogil:
    """TOKEN 0-3: what the other corps lose, now and with their next train,
    where the token makes the city block them and their networks touch it;
    each such corp costs a full solve and a purchase solve on ``after``."""
    cdef int touch = -1
    cdef int o, now, lost, next_lost
    cdef int total = 0, largest = 0, next_total = 0, next_largest = 0
    for o in range(<int>NUM_CORPS):
        if o == corp or not map_blocked_for(after, hex, o) or map_blocked_for(state, hex, o):
            continue
        if touch < 0:
            touch = map_touch_mask(state, hex)
        if not (touch >> o & 1):
            continue
        now = route_max(after, o)
        lost = _route_now(state, o) - now
        next_lost = _route_next(state, o) - _with_next3(after, scratch, o, now)
        total += lost
        largest = max(largest, lost)
        next_total += next_lost
        next_largest = max(next_largest, next_lost)
    block[TO_LOST] = total / PRICE_DIV
    block[TO_LARGEST] = largest / PRICE_DIV
    block[TO_NEXT_LOST] = next_total / PRICE_DIV
    block[TO_NEXT_LARGEST] = next_largest / PRICE_DIV


cdef int _train_gain(GameState state, GameState scratch, int buyer, int action_id, int* gains) noexcept nogil:
    """SELECT_TRAIN choice ``action_id``'s buyer's Δ route max, solved once a
    decision: a choice's gain does not depend on its source, since a train
    that starts a phase is only ever the depot's."""
    cdef ActionInfo info
    if gains[action_id] == <int>GAIN_UNSET:
        info = decode_action(DPHASE_SELECT_TRAIN, action_id)
        if info.type == ACTION_TRADE_IN:
            gains[action_id] = route_max_after_purchase(state, scratch, buyer, <int>D_TRAIN, info.train_type)
        elif corp_train_total(state, buyer) >= <int>MAX_ROUTES:  # never legal; the solver takes no fifth train
            gains[action_id] = _route_now(state, buyer)
        else:
            gains[action_id] = route_max_after_purchase(state, scratch, buyer, info.train_type, -1)
        gains[action_id] -= _route_now(state, buyer)
    return gains[action_id]


cdef void _train_option(GameState state, GameState scratch, const _Context* x, int buyer, int source, int action_id,
                        int* gains, _TrainOption* o) noexcept nogil:
    """The train SELECT_TRAIN's ``action_id`` buys from ``source``; the
    purchase's phase start and rust as closed forms."""
    cdef ActionInfo info = decode_action(DPHASE_SELECT_TRAIN, action_id)
    cdef int seller = source - <int>TRAIN_SOURCE_CORP
    cdef int traded = info.train_type if info.type == ACTION_TRADE_IN else -1
    cdef int t, other
    o.type = <int>D_TRAIN if traded >= 0 else info.train_type
    o.price = trade_in_price(traded) if traded >= 0 else (TRAIN_PRICE[o.type] if seller < 0 else 0)
    o.gain = _train_gain(state, scratch, buyer, action_id, gains)
    o.starts = seller < 0 and _starts_phase(x.game_phase, o.type)
    o.rusted = o.others_rusted = 0
    if o.starts:
        for t in range(<int>NUM_TRAIN_TYPES):
            if TRAIN_RUSTS_ON[t] != o.type:
                continue
            o.rusted += corp_train_count(state, buyer, t) - (t == traded)
            for other in range(<int>NUM_CORPS):
                if other != buyer:
                    o.others_rusted += corp_train_count(state, other, t)
    o.seller = 0
    if seller >= 0:
        copy_row(scratch, state)
        trains_set_count(scratch, seller, info.train_type, corp_train_count(state, seller, info.train_type) - 1)
        o.seller = route_max(scratch, seller) - _route_now(state, seller)
    o.contribution = max(o.price - corp_cash(state, buyer), 0) if seller < 0 and traded < 0 else 0


cdef void _fill_train_block3(float* block, const _TrainOption* o) noexcept nogil:
    block[TN_TYPE + o.type] = 1
    block[TN_PRICE] = o.price / MONEY_DIV
    block[TN_GAIN] = o.gain / PRICE_DIV
    block[TN_STARTS] = o.starts
    block[TN_RUSTED] = o.rusted / <float>MAX_ROUTES
    block[TN_OTHERS_RUSTED] = o.others_rusted / 6.0
    block[TN_SELLER] = o.seller / PRICE_DIV
    block[TN_CONTRIBUTION] = o.contribution / MONEY_DIV


cdef int _fill_buy_preview3(GameState state, GameState a, GameState b, float* block, const _Context* x, int k,
                            int* gains) except -1:
    """A BUY_TRAIN source: opened on ``a``, the SELECT_TRAIN choice there with
    the largest gain, ties to the lower id; its train block."""
    cdef uint16_t ids[46]           # MAX_ACTION_SIZE
    cdef _TrainOption o
    cdef int count, i, gain
    cdef int best = -1, best_gain = 0
    copy_row(a, state)
    apply_alone(a, k)
    count = enumerate_legal_actions(a, ids)
    for i in range(count):
        gain = _train_gain(state, b, x.corp, ids[i], gains)
        if best < 0 or gain > best_gain:
            best, best_gain = ids[i], gain
    _train_option(state, b, x, x.corp, turn_pending_train_source(a), best, gains, &o)
    _fill_train_block3(block, &o)
    return 0


cdef int _fill_select3(GameState state, GameState a, GameState b, float* row, const _Context* x, int k,
                       int* gains) except -1:
    """SELECT_TRAIN's row k: a depot or pool train, or a trade-in, applied,
    with the gain from a full solve of the applied state; a corp's train
    previews TRAIN_PRICE."""
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef int c = x.corp
    cdef int source = turn_pending_train_source(state)
    cdef int route
    cdef _TrainOption o
    if source < <int>TRAIN_SOURCE_CORP:
        _applied3(state, a, row, x, k, c)
        route = route_max(a, c)
        gains[k] = route - _route_now(state, c)
        shared[SH_ROUTE] = gains[k] / PRICE_DIV
        shared[SH_NEXT] = (_with_next3(a, b, c, route) - _route_next(state, c)) / PRICE_DIV
    _train_option(state, b, x, c, source, k, gains, &o)
    if source < <int>TRAIN_SOURCE_CORP:
        shared[SH_COST] = o.price / MONEY_DIV
    _fill_train_block3(row + <int>CHOICE3_PHASE, &o)
    return 0


cdef void _train_price_routes3(GameState state, GameState a, GameState b, const _Context* x,
                               int* routes) noexcept nogil:
    """TRAIN_PRICE's route maxes, one solve each for every price: the buyer's
    and the seller's after the train passes, and the buyer's with the next
    train after it."""
    cdef int t = turn_pending_train_type(state)
    cdef int seller = turn_pending_train_source(state) - <int>TRAIN_SOURCE_CORP
    copy_row(b, state)
    trains_set_count(b, x.corp, t, corp_train_count(state, x.corp, t) + 1)
    trains_set_count(b, seller, t, corp_train_count(state, seller, t) - 1)
    routes[0] = route_max(b, x.corp)
    routes[1] = route_max(b, seller)
    routes[2] = _with_next3(b, a, x.corp, routes[0])


cdef int _fill_train_price3(GameState state, GameState a, float* row, const _Context* x, int k,
                            const ActionInfo* info, const int* routes) except -1:
    cdef float* shared = row + <int>CHOICE3_SHARED
    cdef float* block = row + <int>CHOICE3_PHASE
    cdef int seller = turn_pending_train_source(state) - <int>TRAIN_SOURCE_CORP
    cdef int amount = price_amount(state, info)
    _applied3(state, a, row, x, k, x.corp)
    shared[SH_ROUTE] = (routes[0] - _route_now(state, x.corp)) / PRICE_DIV
    shared[SH_NEXT] = (routes[2] - _route_next(state, x.corp)) / PRICE_DIV
    shared[SH_COST] = amount / MONEY_DIV
    block[TN_TYPE + turn_pending_train_type(state)] = 1
    block[TN_PRICE] = amount / MONEY_DIV
    block[TN_GAIN] = shared[SH_ROUTE]
    block[TN_SELLER] = (routes[1] - _route_now(state, seller)) / PRICE_DIV
    return 0


cdef int _check_input_layout(int layout_version) except -1:
    if layout_version != 3:
        raise ValueError(f"Unsupported 1889 packed input layout version: {layout_version} (layout 3 is packed; "
                         f"layouts 1 and 2 are token buffers)")
    return 0


cdef int _check_capacity(int max_players, str func) except -1:
    assert 3 <= max_players <= 5, f"{func}: max_players must be 3-5, got {max_players}"
    return 0


def get_input_size(int max_players, *, int layout_version):
    """The packed vector's length at capacity ``max_players`` (3-5):
    ``IN3_PLAYERS + IN3_PER_PLAYER * max_players``."""
    _check_input_layout(layout_version)
    _check_capacity(max_players, "get_input_size")
    return input_size3(max_players)


def get_num_trunk_rows(int max_players, *, int layout_version):
    """The model's trunk rows: GLOBAL, the corps, the privates, the revenue
    centers, then ``max_players`` players."""
    _check_input_layout(layout_version)
    _check_capacity(max_players, "get_num_trunk_rows")
    return <int>TRUNK3_PLAYER + max_players


InputPart = namedtuple('InputPart', ('name', 'offset', 'shape', 'rows'))
InputPart.__doc__ = """One part of the packed vector: ``shape`` values from ``offset``, row-major
with the last axis fastest. ``rows`` gives each leading axis's trunk rows as
``(first, count)``: one for a row part, two for a pair block, none for the
choice table."""


def get_input_parts(int max_players, *, int layout_version):
    """The packed vector's parts in order, as ``InputPart``s, at capacity
    ``max_players`` (3-5). The offsets chain, and the last part ends at
    ``get_input_size``."""
    _check_input_layout(layout_version)
    _check_capacity(max_players, "get_input_parts")
    corps = (<int>TRUNK3_CORP, <int>NUM_CORPS)
    privates = (<int>TRUNK3_PRIVATE, <int>NUM_PRIVATES)
    centers = (<int>TRUNK3_CENTER, <int>NUM_RCS)
    players = (<int>TRUNK3_PLAYER, max_players)
    player_corp = <int>IN3_PLAYERS + max_players * <int>TW3_PLAYER
    player_private = player_corp + max_players * <int>NUM_CORPS * <int>CH3_PLAYER_CORP
    return (
        InputPart('GLOBAL', <int>IN3_GLOBAL, (1, <int>TW3_GLOBAL), ((<int>TRUNK3_GLOBAL, 1),)),
        InputPart('CORP', <int>IN3_CORP, (<int>NUM_CORPS, <int>TW3_CORP), (corps,)),
        InputPart('PRIVATE', <int>IN3_PRIVATE, (<int>NUM_PRIVATES, <int>TW3_PRIVATE), (privates,)),
        InputPart('CENTER', <int>IN3_CENTER, (<int>NUM_RCS, <int>TW3_CENTER), (centers,)),
        InputPart('CORP_PRIVATE', <int>IN3_CORP_PRIVATE, (<int>NUM_CORPS, <int>NUM_PRIVATES, <int>CH3_CORP_PRIVATE),
                  (corps, privates)),
        InputPart('CORP_CENTER', <int>IN3_CORP_CENTER, (<int>NUM_CORPS, <int>NUM_RCS, <int>CH3_CORP_CENTER),
                  (corps, centers)),
        InputPart('CENTER_CENTER', <int>IN3_CENTER_CENTER, (<int>NUM_RCS, <int>NUM_RCS, <int>CH3_CENTER_CENTER),
                  (centers, centers)),
        InputPart('CHOICES', <int>IN3_CHOICES, (<int>CHOICE3_ROWS, <int>CHOICE3_WIDTH), ()),
        InputPart('PLAYER', <int>IN3_PLAYERS, (max_players, <int>TW3_PLAYER), (players,)),
        InputPart('PLAYER_CORP', player_corp, (max_players, <int>NUM_CORPS, <int>CH3_PLAYER_CORP), (players, corps)),
        InputPart('PLAYER_PRIVATE', player_private, (max_players, <int>NUM_PRIVATES, <int>CH3_PLAYER_PRIVATE),
                  (players, privates)),
    )


def fill_inputs(GameState state, float[::1] buffer, int max_players=0, *, int layout_version):
    """Fill ``buffer``, exactly ``get_input_size(max_players)`` long, with
    ``state``'s packed vector; ``max_players`` 0 is the state's player count.
    The refreshed caches are written into the state's row, as the token
    fills write them."""
    cdef int num_players = turn_num_players(state)
    _check_input_layout(layout_version)
    max_players = resolve_max_players(num_players, max_players, "fill_inputs")
    assert buffer.shape[0] == input_size3(max_players), \
        f"fill_inputs: buffer of {buffer.shape[0]} for {input_size3(max_players)} values"
    _fill_inputs3(state, &buffer[0], num_players, max_players)


def fill_inputs_batch(list state_arrays, float[:, ::1] buffer, int max_players=0, *, int layout_version):
    """Fill ``buffer[i]`` from ``state_arrays[i]``: writable raw rows, which
    may mix player counts. ``max_players`` 0 takes the capacity the buffer's
    width gives. One view is rebound across the rows, as in
    ``get_token_data_batch``, and layout 3's scratch states serve them."""
    cdef int n = len(state_arrays)
    cdef int i
    cdef GameState view = None
    _check_input_layout(layout_version)
    if n == 0:
        return
    if max_players == 0:
        max_players = (<int>buffer.shape[1] - <int>IN3_PLAYERS) // <int>IN3_PER_PLAYER
    _check_capacity(max_players, "fill_inputs_batch")
    assert buffer.shape[0] >= n and buffer.shape[1] == input_size3(max_players), \
        f"fill_inputs_batch: buffer {(buffer.shape[0], buffer.shape[1])} for {n} rows of {input_size3(max_players)}"
    for i in range(n):
        view = bind_row(view, state_arrays[i], max_players, "fill_inputs_batch")
        _fill_inputs3(view, &buffer[i, 0], turn_num_players(view), max_players)


def get_tile_table():
    """The model's tile table, ``(NUM_TILES + 1, NUM_SIDES, TW_TILE_BLOCK)``
    float32: entry [code, rotation] is the tile block of tile ``code - 1`` at
    ``rotation`` (``_fill_tile_block``), with the tile's own revenue
    (``TILE_REVENUE``: 0 for an offboard, whose CENTER row has its revenue by
    phase); code 0, no tile, is zero. Printed tiles occur at rotation 0 only."""
    out = np.zeros((<int>NUM_TILES + 1, <int>NUM_SIDES, <int>TW_TILE_BLOCK), np.float32)
    cdef float[:, :, ::1] table = out
    cdef int tile, rotation
    for tile in range(<int>NUM_TILES):
        for rotation in range(<int>NUM_SIDES):
            _fill_tile_block(&table[tile + 1, rotation, 0], tile, rotation, TILE_REVENUE[tile])
    return out


def get_choice_slots():
    """``(NUM_DECISION_PHASES, CHOICE3_ROWS)`` int16: the unified policy slot
    of choice row k at decision phase d, the phase's first slot + k, or -1
    past its size."""
    out = np.full((<int>NUM_DECISION_PHASES, <int>CHOICE3_ROWS), -1, np.int16)
    cdef int first = 0
    cdef int phase, k
    for phase in range(<int>NUM_DECISION_PHASES):
        for k in range(PHASE_ACTION_SIZE[phase]):
            out[phase, k] = first + k
        first += PHASE_ACTION_SIZE[phase]
    assert first == <int>UNIFIED_ACTION_DIM
    return out


def get_named_rows():
    """``(UNIFIED_ACTION_DIM,)`` int16: the trunk row each unified slot names
    (token-data-1889.md, "Rows and what they name"), ``NAMED3_NONE`` for
    none, or ``NAMED3_TOPMOST`` for AUCTION's buy, which names the PRIVATE
    row PRIVATE 2 selects. Static: no slot names a player."""
    slots = get_choice_slots()
    out = np.full(<int>UNIFIED_ACTION_DIM, <int>NAMED3_NONE, np.int16)

    def name(int phase, int action, int row):
        out[slots[phase, action]] = row

    cdef int c, p, i, rc
    name(DPHASE_AUCTION, AUCTION_BUY, NAMED3_TOPMOST)
    for p in range(1, <int>NUM_PRIVATES):  # private 0 has no bid slot
        name(DPHASE_AUCTION, AUCTION_BID + p - 1, <int>TRUNK3_PRIVATE + p)
    for c in range(<int>NUM_CORPS):
        for first in (STOCK_IPO, STOCK_POOL, STOCK_PAR, STOCK_SELL):
            name(DPHASE_STOCK, first + c, <int>TRUNK3_CORP + c)
        name(DPHASE_BUY_TRAIN, BUY_TRAIN_CORP + c, <int>TRUNK3_CORP + c)
        name(DPHASE_BUY_TRAIN, BUY_TRAIN_SELL + c, <int>TRUNK3_CORP + c)
    name(DPHASE_STOCK, STOCK_DR, <int>TRUNK3_PRIVATE + <int>PRIVATE_DR)
    name(DPHASE_STOCK, STOCK_MF, <int>TRUNK3_PRIVATE + <int>PRIVATE_MF)
    for p in range(<int>NUM_PRIVATES):
        name(DPHASE_BUY_PRIVATE_PRE, BUY_PRIVATE_PRE_PRIVATE + p, <int>TRUNK3_PRIVATE + p)
        name(DPHASE_BUY_PRIVATE_POST, BUY_PRIVATE_POST_PRIVATE + p, <int>TRUNK3_PRIVATE + p)
    name(DPHASE_BUY_PRIVATE_PRE, BUY_PRIVATE_PRE_DR, <int>TRUNK3_PRIVATE + <int>PRIVATE_DR)
    name(DPHASE_BUY_PRIVATE_PRE, BUY_PRIVATE_PRE_MF, <int>TRUNK3_PRIVATE + <int>PRIVATE_MF)
    name(DPHASE_BUY_PRIVATE_POST, BUY_PRIVATE_POST_DR, <int>TRUNK3_PRIVATE + <int>PRIVATE_DR)
    for i in range(<int>NUM_TRACK_HEXES):  # a plain hex has no row
        rc = HEX_RC[TRACK_HEXES[i]]
        if rc >= 0:
            name(DPHASE_TRACK, TRACK_HEX + i, <int>TRUNK3_CENTER + rc)
    for i in range(<int>NUM_TOKEN_HEXES):  # every token hex is a city
        name(DPHASE_TOKEN, TOKEN_HEX + i, <int>TRUNK3_CENTER + HEX_RC[TOKEN_HEXES[i]])
    for i in range(<int>NUM_PORT_LAYS):  # every port hex is a town
        name(DPHASE_PORT, PORT_LAY + i, <int>TRUNK3_CENTER + HEX_RC[PORT_LAY_HEX[i]])
    return out


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
# Layout 3: the parts chain, the trunk rows follow the tables, and each row's
# slots end at its width.
assert <int>IN3_GLOBAL == 0 and IN3_CORP == IN3_GLOBAL + TW3_GLOBAL and IN3_PRIVATE == IN3_CORP + NUM_CORPS * TW3_CORP
assert IN3_CENTER == IN3_PRIVATE + NUM_PRIVATES * TW3_PRIVATE and IN3_CORP_PRIVATE == IN3_CENTER + NUM_RCS * TW3_CENTER
assert IN3_CORP_CENTER == IN3_CORP_PRIVATE + NUM_CORPS * NUM_PRIVATES * CH3_CORP_PRIVATE
assert IN3_CENTER_CENTER == IN3_CORP_CENTER + NUM_CORPS * NUM_RCS * CH3_CORP_CENTER
assert IN3_CHOICES == IN3_CENTER_CENTER + NUM_RCS * NUM_RCS * CH3_CENTER_CENTER
assert IN3_PLAYERS == IN3_CHOICES + CHOICE3_ROWS * CHOICE3_WIDTH
assert IN3_PER_PLAYER == TW3_PLAYER + NUM_CORPS * CH3_PLAYER_CORP + NUM_PRIVATES * CH3_PLAYER_PRIVATE
assert <int>TRUNK3_GLOBAL == 0 and TRUNK3_CORP == 1 and TRUNK3_PRIVATE == TRUNK3_CORP + NUM_CORPS
assert TRUNK3_CENTER == TRUNK3_PRIVATE + NUM_PRIVATES and TRUNK3_PLAYER == TRUNK3_CENTER + NUM_RCS
assert <int>CHOICE3_ROWS == <int>MAX_ACTION_SIZE and CHOICE3_SHARED == CHOICE3_ROTATION + 1
assert CHOICE3_PHASE == CHOICE3_SHARED + 7 and CHOICE3_PHASE + CHOICE3_PHASE_WIDTH == CHOICE3_WIDTH
assert <int>G3_TRAINS == <int>TW2_GLOBAL and G3_TRAINS + G3_TRAIN_SLOTS * NUM_TRAIN_TYPES == G3_REFERENCE
assert T3_STARTS_PHASE + 1 == G3_TRAIN_SLOTS and G3_REVENUE + 1 == TW3_GLOBAL
assert len(_CORP3_FROM2) == C3_KEPT and C3_KEPT + 6 == TW3_CORP and max(_CORP3_FROM2) + 1 == TW2_CORP
assert len(_PRIVATE3_FROM2) == TW3_PRIVATE and max(_PRIVATE3_FROM2) == P_PORT_USED
assert len(_PLAYER3_FROM2) == TW3_PLAYER and max(_PLAYER3_FROM2) + 1 == TW2_PLAYER
assert CE3_HEX + H_UPGRADES_ANY_PHASE + 1 - H_D_REVENUE == TW3_CENTER
# The fills' buffers are sized with literals.
assert MAX_LAY_CANDIDATES == 15 and NUM_CORPS == 7 and MAX_ACTION_SIZE == 46
assert <int>TOKEN_DIM_2 == 109 and max(<int>TW_PRICE, <int>TW2_TRAIN) <= 24
