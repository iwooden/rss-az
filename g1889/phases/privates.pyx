"""Private handlers; see privates.pxd.

Ruby's non-blocking OR and SR steps for privates are the oracle:
``BuyCompany`` (``step/buy_company.rb``), ``Exchange``
(``step/exchange.rb``), and 1889's ``SpecialTrack``
(``game/g_1889/step/special_track.rb`` with
``step/track_lay_when_company_sold.rb``), with the blocking BuyCompany after
BuyTrain. ``actions.pyx`` has the legality.

- **Buy private p** at BUY_PRIVATE_PRE or BUY_PRIVATE_POST (phases 3-4):
  ``pending_private`` holds p and ``return_step`` the step, and
  PRIVATE_PRICE names the price. The sale pays the seller from the corp's
  treasury (``PRIVATES[p].sell_to_corp``). A sale of ER then opens ER_BONUS
  for its seller (``pending_seller``), who may lay a green city tile on C4
  free, on top of the corp's own lay; then the step resumes, its president
  acting again.
- **DR's exchange** at the active player's STOCK, BUY_PRIVATE_PRE, or
  BUY_PRIVATE_POST: DR closes for an IR share, and the decision goes on.
  Ruby's Exchange step tracks no SR action (``current_actions``), so it
  neither ends the turn nor resets the passes.
- **MF's port tile** at its owner's STOCK or BUY_PRIVATE_PRE: PORT lays it,
  and the step resumes. Ruby's SpecialTrack uses no lay of the corp's, so
  TRACK still offers the corp's own (962 action 39).

``DRIVER.override_private_slot`` enters the same handlers for replay at the
turn's other decisions.
"""

from g1889.actions cimport (
    ACTION_BUY_PRIVATE, ACTION_DR_EXCHANGE, ACTION_LAY_CANDIDATE, ACTION_MF_PORT, ACTION_PASS, ActionInfo,
    LayCandidate, er_candidates, price_amount,
)
from g1889.constants cimport (
    OWNER_PLAYER, STEP_ER_BONUS, STEP_PORT, STEP_PRIVATE_PRICE, STEP_TRACK, STEP_TURN_END,
)
from g1889.data.tables cimport (
    ER_HEX, MAX_LAY_CANDIDATES, MF_PORT_HEX, NUM_PORTS, NUM_SUPPLY_TILES, PORT_TILE, PRIVATE_DR, PRIVATE_ER,
    PRIVATE_MF, TILE_COST,
)
from g1889.entities.corp cimport corp_president
from g1889.entities.hexmap cimport HexMap
from g1889.entities.private cimport MF_PORT_USED, Private, private_owner
from g1889.entities.turn cimport (
    TurnState, turn_operating_corp, turn_pending_private, turn_step,
)
from g1889.state cimport GameState

from g1889.entities import hexmap as _hexmap, private as _private, turn as _turn

cdef HexMap MAP = _hexmap.MAP
cdef list PRIVATES = _private.PRIVATES
cdef TurnState TURN = _turn.TURN


cdef void apply_interjection(GameState state, const ActionInfo* info) except *:
    """Choose a private to buy (PRIVATE_PRICE follows), exchange DR, or use
    MF (PORT follows), from any decision that offers the slot or takes
    replay's override."""
    if info.type == ACTION_BUY_PRIVATE:
        TURN.set_pending_private(state, info.private)
        TURN.set_return_step(state, turn_step(state))
        TURN.set_step(state, STEP_PRIVATE_PRICE)
    elif info.type == ACTION_DR_EXCHANGE:
        (<Private>PRIVATES[<int>PRIVATE_DR]).exchange(state)
    else:
        assert info.type == ACTION_MF_PORT, f"apply_interjection: action type {info.type}"
        TURN.set_return_step(state, turn_step(state))
        TURN.set_step(state, STEP_PORT)


cdef void apply_buy_private_pre(GameState state, const ActionInfo* info) except *:
    """Done: on to the corp's lay. BUY_PRIVATE_PRE's slots are interjections
    (``apply_interjection``) that return here."""
    assert info.type == ACTION_PASS, f"apply_buy_private_pre: action type {info.type}"
    TURN.set_step(state, STEP_TRACK)


cdef void apply_buy_private_post(GameState state, const ActionInfo* info) except *:
    """Done: the corp's turn ends. BUY_PRIVATE_POST's slots are interjections
    (``apply_interjection``) that return here."""
    assert info.type == ACTION_PASS, f"apply_buy_private_post: action type {info.type}"
    TURN.set_step(state, STEP_TURN_END)


cdef void apply_private_price(GameState state, const ActionInfo* info) except *:
    """The operating corp buys the pending private for the chosen price
    (``BuyCompany#buy_company``). After ER, its seller may lay on C4
    (``TrackLayWhenCompanySold``: Ruby's ``active_players`` are the seller);
    otherwise the step resumes."""
    cdef int p = turn_pending_private(state)
    cdef int seller = private_owner(state, p) - <int>OWNER_PLAYER
    cdef int price = price_amount(state, info)
    (<Private>PRIVATES[p]).sell_to_corp(state, turn_operating_corp(state), price)
    TURN.set_pending_private(state, -1)
    if p != <int>PRIVATE_ER:
        TURN.resume(state)
        return
    TURN.set_pending_seller(state, seller)
    TURN.set_active_player(state, seller)
    TURN.set_step(state, STEP_ER_BONUS)


cdef void apply_er_bonus(GameState state, const ActionInfo* info) except *:
    """ER's seller passes or lays candidate ``count`` on C4. Ruby charges the
    seller C4's terrain cost, which is nothing. The corp's president acts
    again where the sale came from."""
    cdef LayCandidate candidates[15]  # MAX_LAY_CANDIDATES
    cdef int count
    if info.type == ACTION_LAY_CANDIDATE:
        count = er_candidates(state, candidates)
        assert 0 <= info.count < count, f"apply_er_bonus: candidate {info.count} of {count}"
        MAP.lay(state, <int>ER_HEX, candidates[info.count].tile, candidates[info.count].rotation)
    else:
        assert info.type == ACTION_PASS, f"apply_er_bonus: action type {info.type}"
    TURN.set_pending_seller(state, -1)
    TURN.set_active_player(state, corp_president(state, turn_operating_corp(state)))
    TURN.resume(state)


cdef void apply_port(GameState state, const ActionInfo* info) except *:
    """MF's owner lays the port tile on ``hex`` at rotation ``count``, free,
    and MF's one lay is used; MF stays open."""
    cdef Private mf = <Private>PRIVATES[<int>PRIVATE_MF]
    MAP.lay(state, info.hex, <int>PORT_TILE, info.count)
    mf.set_flags(state, mf.flags(state) | <int>MF_PORT_USED)
    TURN.resume(state)


# The ER and port lays charge nothing: C4 and the port hexes have no terrain.
assert TILE_COST[<int>NUM_SUPPLY_TILES + <int>ER_HEX] == 0
assert all(TILE_COST[<int>NUM_SUPPLY_TILES + MF_PORT_HEX[i]] == 0 for i in range(<int>NUM_PORTS))

# er_candidates' buffer is sized with the literal 15.
assert <int>MAX_LAY_CANDIDATES == 15
