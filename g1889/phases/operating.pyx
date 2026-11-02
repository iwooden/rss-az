"""Operating round handlers; see operating.pxd.

Ruby's ``Round::Operating`` (``round/operating.rb``) and the 1889 OR steps
(``game/g_1889/game.rb``: Track, Token, Route, Dividend, DiscardTrain,
BuyTrain, then a blocking BuyCompany) are the oracle. An OR runs:

- ``OR_START`` (``Round::Operating#setup``): every owned private pays its
  revenue, the floated corps place their home tokens
  (``HOME_TOKEN_TIMING = :operating_round``), and the floated corps are put
  in stock price order (``operating_order``). The order is fixed then, so a
  corp that floats during the OR waits for the next.
- ``TURN_START`` (``start_operating``): the next corp operates, its
  president acting.
- TRACK (with TILE for a chosen hex), TOKEN, the automated ROUTE, DIVIDEND
  (``dividend.pyx``), BUY_TRAIN with SELECT_TRAIN, TRAIN_PRICE,
  EMERGENCY_SELL_AMOUNT, and DISCARD (``buy_train.pyx``), and in phases
  3-4 BUY_PRIVATE (``privates.pyx``,
  with the slots that buy privates and use DR and MF at the turn's
  decisions). Where Ruby skips a step because there is nothing to choose,
  the engine offers one action, which the driver's forced rule applies: a
  pass, or the withhold at zero revenue.
- ``TURN_END`` (``next_entity!``): the next corp, or ``OR_END``.
- ``OR_END`` (``next_round!``): the set's next OR, or else the next SR,
  with the SR number up and ``or_num`` back to 1; after the set's last OR,
  the game's end if the bank has broken (``GAME_END_CHECK``, ``:full_or``).

Fields:

- ``or_order`` holds the OR's corps, padded with -1, and ``or_index`` the
  operating corp's place in it (Ruby's ``entity_index``);
- ``operating_corp`` is the corp operating, -1 between turns;
- ``pending_hex`` is TILE's hex while TILE is open;
- ``route_revenue`` is the turn's revenue, from ROUTE until DIVIDEND;
- ``ebuy_sold_mask`` and ``ebuy_sold_any`` are the turn's emergency sales
  (``buy_train.pyx``), cleared at TURN_END.

The step implies the turn's lay and token use (engine-design.md §3). The
game ends at OR_END after a broken bank's set, or at once when a president
goes bankrupt (``buy_train.pyx``); either way the row keeps the last round's
fields, as Ruby keeps its round, and ``end_reason`` tells which.
"""

from g1889.actions cimport (
    ACTION_PASS, ACTION_PLACE_TOKEN, ACTION_SELECT_HEX, ActionInfo, LayCandidate, enumerate_lay_candidates,
)
from g1889.constants cimport (
    END_REASON_BANK, ROUND_GAME_OVER, ROUND_OPERATING, STEP_DIVIDEND, STEP_GAME_OVER, STEP_OR_END, STEP_OR_START,
    STEP_ROUTE, STEP_TILE, STEP_TOKEN, STEP_TRACK, STEP_TURN_START,
)
from g1889.data.tables cimport CORP_HOME, NUM_CORPS
from g1889.entities.bank cimport bank_broken
from g1889.entities.corp cimport corp_floated, corp_president
from g1889.entities.hexmap cimport HexMap, hex_tokens
from g1889.entities.market cimport market_sort_by_price
from g1889.entities.private cimport pay_revenue
from g1889.entities.turn cimport TurnState, turn_operating_corp, turn_pending_hex
from g1889.phases.stock cimport start_stock_round
from g1889.routes.solver cimport route_max
from g1889.state cimport GameState

from g1889.entities import hexmap as _hexmap, turn as _turn

cdef HexMap MAP = _hexmap.MAP
cdef TurnState TURN = _turn.TURN


cdef void _clear_order(GameState state) except *:
    cdef int i
    for i in range(<int>NUM_CORPS):
        TURN.set_or_order(state, i, -1)
    TURN.set_or_index(state, 0)


# =============================================================================
# ROUND AND TURNS
# =============================================================================

cdef void run_or_start(GameState state) except *:
    """``Round::Operating#setup``: the privates pay out (``payout_companies``),
    the floated corps place their home tokens free on their reserved slots
    (``place_home_token``), and the OR's order is the floated corps in stock
    price order. Ruby pays the privates player by player, but the bank only
    pays out, so the order cannot change the result."""
    cdef int[7] order  # NUM_CORPS
    cdef int floated = 0
    cdef int c, i, count
    pay_revenue(state)
    for c in range(<int>NUM_CORPS):
        floated |= corp_floated(state, c) << c
    count = market_sort_by_price(state, floated, order)
    _clear_order(state)
    for i in range(count):
        c = order[i]
        TURN.set_or_order(state, i, c)
        if not (hex_tokens(state, CORP_HOME[c]) >> c & 1):
            MAP.place_token(state, CORP_HOME[c], c)
    TURN.set_step(state, STEP_TURN_START if count else STEP_OR_END)


cdef void run_turn_start(GameState state) except *:
    """``start_operating``: the corp at ``or_index`` operates, its president acting."""
    cdef int corp = TURN.or_order(state, TURN.or_index(state))
    TURN.set_operating_corp(state, corp)
    TURN.set_active_player(state, corp_president(state, corp))
    TURN.set_step(state, STEP_TRACK)


cdef void run_turn_end(GameState state) except *:
    """``next_entity!``: the next corp in the order operates, or the OR ends.
    The turn's emergency sales clear (Ruby's BuyTrain ``setup`` clears them
    as the next corp's turn starts)."""
    cdef int index = TURN.or_index(state) + 1
    TURN.set_operating_corp(state, -1)
    TURN.set_ebuy_sold_mask(state, 0)
    TURN.set_ebuy_sold_any(state, 0)
    if index < <int>NUM_CORPS and TURN.or_order(state, index) >= 0:
        TURN.set_or_index(state, index)
        TURN.set_step(state, STEP_TURN_START)
    else:
        TURN.set_step(state, STEP_OR_END)


cdef void run_or_end(GameState state) except *:
    """``next_round!`` after an OR: the set's next OR, else the next SR, which
    the priority player opens (Ruby reorders the players only after an SR).
    After the set's last OR a broken bank ends the game instead, wherever it
    broke, in an SR or an OR (``end_now?`` with ``:full_or``)."""
    cdef int or_num = TURN.or_num(state)
    if or_num >= TURN.or_count(state) and bank_broken(state):
        end_game(state, <int>END_REASON_BANK)
        return
    _clear_order(state)
    if or_num < TURN.or_count(state):
        TURN.set_or_num(state, or_num + 1)
        TURN.set_step(state, STEP_OR_START)
        return
    TURN.set_turn(state, TURN.turn(state) + 1)
    TURN.set_or_num(state, 1)
    start_stock_round(state)


cdef void resort_order(GameState state) except *:
    """``recalculate_order`` (``round/operating.rb:97``), after an emergency
    sale or a bankruptcy: the corps yet to run this OR go back into stock
    price order, behind those that have run, which keep their places."""
    cdef int[7] order  # NUM_CORPS
    cdef int index = TURN.or_index(state)
    cdef int mask = 0
    cdef int c, i, count
    for i in range(index + 1, <int>NUM_CORPS):
        c = TURN.or_order(state, i)
        if c >= 0:
            mask |= 1 << c
    count = market_sort_by_price(state, mask, order)
    for i in range(count):
        TURN.set_or_order(state, index + 1 + i, order[i])


cdef void end_game(GameState state, int reason) except *:
    """``end_game!``: the game is over, for ``reason`` (an EndReason). Ruby
    keeps the last round's fields, and so does the row: the OR's order, and
    after a bankruptcy its operating corp and the turn's emergency sales."""
    TURN.set_end_reason(state, reason)
    TURN.set_round(state, ROUND_GAME_OVER)
    TURN.set_step(state, STEP_GAME_OVER)


# =============================================================================
# TRACK, TILE, TOKEN
# =============================================================================

cdef void apply_track(GameState state, const ActionInfo* info) except *:
    """Pass the lay, or choose its hex (TILE follows)."""
    if info.type == ACTION_PASS:
        TURN.set_step(state, STEP_TOKEN)
        return
    assert info.type == ACTION_SELECT_HEX, f"apply_track: action type {info.type}"
    TURN.set_pending_hex(state, info.hex)
    TURN.set_step(state, STEP_TILE)


cdef void apply_tile(GameState state, const ActionInfo* info) except *:
    """Lay candidate ``count`` on the pending hex, paying its cost
    (``lay_tile``). A turn has one lay, so TOKEN follows."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int hex = turn_pending_hex(state)
    cdef int count = enumerate_lay_candidates(state, hex, candidates)
    assert 0 <= info.count < count, f"apply_tile: candidate {info.count} of {count}"
    TURN.set_pending_hex(state, -1)
    MAP.lay_track(state, turn_operating_corp(state), hex, candidates[info.count].tile,
                  candidates[info.count].rotation)
    TURN.set_step(state, STEP_TOKEN)


cdef void apply_token(GameState state, const ActionInfo* info) except *:
    """Pass, or buy the corp's next token for the hex's city (``Tokener#place_token``).
    A turn has one token, so ROUTE follows."""
    if info.type == ACTION_PLACE_TOKEN:
        MAP.buy_token(state, info.hex, turn_operating_corp(state))
    else:
        assert info.type == ACTION_PASS, f"apply_token: action type {info.type}"
    TURN.set_step(state, STEP_ROUTE)


# =============================================================================
# ROUTE
# =============================================================================

cdef void run_route(GameState state) except *:
    """Ruby's Route step: the corp runs its trains for the most they can earn
    (``routes/solver.pyx``, cached in ``route_cache``), the turn's revenue;
    a corp without trains or a route earns 0. Replay sets a human's revenue
    with ``DRIVER.override_route_revenue`` at DIVIDEND."""
    TURN.set_route_revenue(state, route_max(state, turn_operating_corp(state)))
    TURN.set_step(state, STEP_DIVIDEND)


# =============================================================================
# PYTHON TEST ENTRY POINTS
# =============================================================================

def setup_operating_round_py(GameState state, int or_num=1):
    """Enter OR ``or_num`` of the set at OR_START, on holdings and a map set up
    by hand; ``DRIVER.advance`` runs it. The set's length (``TURN.or_count``)
    and the SR number are the caller's."""
    TURN.set_round(state, ROUND_OPERATING)
    TURN.set_or_num(state, or_num)
    TURN.set_step(state, STEP_OR_START)


# The buffers for market_sort_by_price are sized with the literal 7.
assert <int>NUM_CORPS == 7
