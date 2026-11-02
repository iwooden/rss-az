"""Stock round handlers; see stock.pxd.

Ruby's ``Round::Stock`` and ``BuySellParShares`` are the oracle. Turns go
clockwise from the priority holder. On a turn the player buys at most one
certificate and sells blocks (``actions.pyx`` has the legality); slot 0 then
ends the turn. Before acting it is a pass, which sets the player's ``passed``
flag; after acting it is "done", which clears it (``pass!``). Ruby ends a
turn by itself when the player has nothing left to do; the engine offers only
slot 0 there, and the driver's forced rule applies it. The round ends when
every flag is set, that is, once everyone has passed in succession.

Fields:

- ``active_player`` is the turn player;
- ``sr_bought``, ``sr_sold_mask``, and ``sr_sold_then_bought`` are this turn's
  purchase, its sales, and whether a purchase followed a sale;
- each player's ``sold_mask`` holds the corps they sold this SR, which they
  may not buy back;
- ``pending_corp`` is the corp PAR pars or SELL_AMOUNT sells, while it is open;
- ``priority_player`` is Ruby's ``priority_deal_player``: the player after
  the last to buy or sell, else the SR's first player. SR_END moves it to the
  player after the last passer, which is the same player.
"""

from g1889.actions cimport (
    ACTION_BUY_IPO, ACTION_BUY_POOL, ACTION_PASS, ACTION_SELECT_PAR, ACTION_SELECT_SALE, ActionInfo,
)
from g1889.constants cimport (
    ROUND_OPERATING, ROUND_STOCK, STEP_OR_START, STEP_PAR, STEP_SELL_AMOUNT, STEP_SR_END, STEP_STOCK,
)
from g1889.data.tables cimport NUM_CORPS, PHASE_ORS
from g1889.entities.corp cimport Corporation, corp_floated
from g1889.entities.market cimport Market, market_sort_by_price
from g1889.entities.phase cimport game_phase
from g1889.entities.player cimport Player, all_passed, clear_passed, player_sold_mask
from g1889.entities.turn cimport (
    TurnState, turn_active_player, turn_num_players, turn_pending_corp, turn_sr_bought, turn_sr_sold_mask,
)
from g1889.state cimport GameState

from g1889.entities import corp as _corp, market as _market, player as _player, turn as _turn

cdef list CORPS = _corp.CORPS
cdef list PLAYERS = _player.PLAYERS
cdef Market MARKET = _market.MARKET
cdef TurnState TURN = _turn.TURN


cdef inline Player _player_at(int player):
    return <Player>PLAYERS[player]


cdef inline Corporation _corp_at(int corp):
    return <Corporation>CORPS[corp]


# =============================================================================
# TURNS
# =============================================================================

cdef void _start_turn(GameState state, int player) except *:
    """``start_entity``: the player's turn opens with nothing bought or sold."""
    TURN.set_active_player(state, player)
    TURN.set_sr_bought(state, 0)
    TURN.set_sr_sold_mask(state, 0)
    TURN.set_sr_sold_then_bought(state, 0)
    TURN.set_step(state, STEP_STOCK)


cdef void start_stock_round(GameState state) except *:
    """Open an SR at the priority player's turn. The passes and SR sales are
    clear already: the auction's end and SR_END clear them."""
    TURN.set_round(state, ROUND_STOCK)
    _start_turn(state, TURN.priority_player(state))


cdef void _acted(GameState state, int player, bint bought) except *:
    """Record a purchase or a sale (``track_action``): priority moves to the
    next player, and a purchase after a sale ends the turn's selling."""
    if bought:
        TURN.set_sr_bought(state, 1)
        if turn_sr_sold_mask(state):
            TURN.set_sr_sold_then_bought(state, 1)
    TURN.set_priority_player(state, TURN.next_player(state, player))


# =============================================================================
# HANDLERS
# =============================================================================

cdef void apply_stock(GameState state, const ActionInfo* info) except *:
    """Pass or done, buy a share, or select a corp to par (PAR follows) or to
    sell (SELL_AMOUNT follows)."""
    cdef int player = turn_active_player(state)
    cdef int c = info.corp
    if info.type == ACTION_PASS:
        _player_at(player).set_passed(state, not (turn_sr_bought(state) or turn_sr_sold_mask(state)))
        if all_passed(state):
            TURN.set_step(state, STEP_SR_END)
        else:
            _start_turn(state, TURN.next_player(state, player))
    elif info.type == ACTION_BUY_IPO or info.type == ACTION_BUY_POOL:
        _corp_at(c).buy_share(state, player, info.type == ACTION_BUY_POOL)
        _acted(state, player, True)
    else:
        assert info.type == ACTION_SELECT_PAR or info.type == ACTION_SELECT_SALE, \
            f"apply_stock: action type {info.type}"
        TURN.set_pending_corp(state, c)
        TURN.set_step(state, STEP_PAR if info.type == ACTION_SELECT_PAR else STEP_SELL_AMOUNT)


cdef void apply_par(GameState state, const ActionInfo* info) except *:
    """Par the selected corp at par index ``count``; the turn goes on in STOCK."""
    cdef int player = turn_active_player(state)
    cdef int c = turn_pending_corp(state)
    TURN.set_pending_corp(state, -1)
    _corp_at(c).par(state, player, info.count)
    _acted(state, player, True)
    TURN.set_step(state, STEP_STOCK)


cdef void apply_sell_amount(GameState state, const ActionInfo* info) except *:
    """Sell ``count`` shares of the selected corp as one block, which the
    player may not buy back this SR; the turn goes on in STOCK."""
    cdef int player = turn_active_player(state)
    cdef int c = turn_pending_corp(state)
    TURN.set_pending_corp(state, -1)
    _corp_at(c).sell_shares(state, player, info.count)
    _player_at(player).set_sold_mask(state, player_sold_mask(state, player) | 1 << c)
    TURN.set_sr_sold_mask(state, turn_sr_sold_mask(state) | 1 << c)
    _acted(state, player, False)
    TURN.set_step(state, STEP_STOCK)


cdef void run_sr_end(GameState state) except *:
    """Every player has passed (``Round::Stock#finish_round``, ``next_round!``):
    priority goes to the player after the last passer; sold-out corps move up
    one row in stock price order, taken before any moves; the coming OR set's
    length is fixed by the phase. The passes and SR sales clear, and the OR
    begins."""
    cdef int[7] order  # NUM_CORPS
    cdef int floated = 0
    cdef int c, i, count
    TURN.set_priority_player(state, TURN.next_player(state, turn_active_player(state)))
    for c in range(<int>NUM_CORPS):
        floated |= corp_floated(state, c) << c
    count = market_sort_by_price(state, floated, order)
    for i in range(count):
        if _corp_at(order[i]).sold_out(state):
            MARKET.move_up(state, order[i])
    clear_passed(state)
    for i in range(turn_num_players(state)):
        _player_at(i).set_sold_mask(state, 0)
    TURN.set_sr_bought(state, 0)
    TURN.set_sr_sold_mask(state, 0)
    TURN.set_sr_sold_then_bought(state, 0)
    TURN.set_or_count(state, PHASE_ORS[game_phase(state)])
    TURN.set_or_num(state, 1)
    TURN.set_round(state, ROUND_OPERATING)
    TURN.set_step(state, STEP_OR_START)


# =============================================================================
# PYTHON TEST ENTRY POINTS
# =============================================================================

def setup_stock_round_py(GameState state, int first_player):
    """Open an SR at ``first_player``'s turn with priority, on holdings set up
    by hand; the SR number (``TURN.turn``) is the caller's."""
    TURN.set_priority_player(state, first_player)
    start_stock_round(state)


def run_sr_end_py(GameState state):
    """Run SR_END on a state whose last passer is the active player."""
    run_sr_end(state)


# market_sort_by_price's buffer above is sized with the literal 7.
assert <int>NUM_CORPS == 7
