"""The proofs behind the action slots (1889/engine-design.md §12 item 8).

The slot tables of ``g1889.data.tables`` hold only what can ever be legal. A
slot left out needs an argument from the rules, the map, or the tables, never
just that random play did not reach it, so each test here states one: walks of
the upgrade table, the train schedule, and the entity rules, against the
tables' sets rather than by rebuilding them. ``test_data_vs_ruby.py`` holds the
same tables against Ruby's own checks, and ``test_private_oracles.py`` the
engine's legal sets against Ruby at every fixture snapshot.

What each test relies on: a hex's tile changes only by a lay, and every lay is
a row of the upgrade table (``_lay_scan`` a corp's, ``er_candidates`` ER's,
``port_lays`` MF's), so the walk from the printed tile, which ignores phase
colors, supply, open sides, and connectivity, covers every tile a hex can hold.
"""

import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_OK
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    HEX, TILE, apply_and_verify, at_or, choose_train, enter_operating_round, enter_phase, give_train, hold, lay,
    legal_ids, new_game, opened, past_pre, set_cash, trade, train_choices,
)
from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_100 = 5
GRAY, RED = D.COLOR_GRAY, D.COLOR_RED


def printed(h):
    return D.NUM_SUPPLY_TILES + h


def targets(tile, rotation):
    k = tile * D.NUM_SIDES + rotation
    return [(D.UPG_TILE[i], D.UPG_ROT[i]) for i in range(D.UPG_START[k], D.UPG_START[k + 1])]


def reachable(h):
    """Every (tile, rotation) hex ``h`` can ever hold."""
    seen, stack = {(printed(h), 0)}, [(printed(h), 0)]
    while stack:
        for target in targets(*stack.pop()):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def open_target(h, tile, rotation):
    return D.TILE_EXITS[tile][rotation] & ~D.HEX_EXIT_MASK[h] == 0


def test_the_tables_are_what_the_slots_index():
    assert D.TRACK_HEXES == [h for h in range(D.NUM_HEXES) if D.HEX_TRACK_SLOT[h] >= 0]
    assert D.TOKEN_HEXES == [h for h in range(D.NUM_HEXES) if D.HEX_TOKEN_SLOT[h] >= 0]
    assert [D.TRACK_HEXES[D.HEX_TRACK_SLOT[h]] for h in D.TRACK_HEXES] == D.TRACK_HEXES
    assert [D.TOKEN_HEXES[D.HEX_TOKEN_SLOT[h]] for h in D.TOKEN_HEXES] == D.TOKEN_HEXES
    assert [D.DISCARD_TYPES[D.TRAIN_DISCARD_SLOT[t]] for t in D.DISCARD_TYPES] == D.DISCARD_TYPES
    lays = list(zip(D.PORT_LAY_HEX, D.PORT_LAY_ROT))
    assert [lays[D.PORT_LAY_SLOT[D.MF_PORT_HEX.index(h)][r]] for h, r in lays] == lays


# =============================================================================
# TRACK AND TOKEN
# =============================================================================

def test_a_hex_without_a_track_slot_is_never_laid_on():
    """Its printed tile, gray or red, has no upgrade target at all, and no
    private's lay names it, so its tile never changes."""
    outside = [h for h in range(D.NUM_HEXES) if D.HEX_TRACK_SLOT[h] < 0]
    assert [D.HEX_IDS[h] for h in outside] == ['B3', 'B7', 'F1', 'G14', 'J1', 'J7', 'L7']
    for h in outside:
        assert D.TILE_COLOR[printed(h)] in (GRAY, RED), D.HEX_IDS[h]
        assert reachable(h) == {(printed(h), 0)}, D.HEX_IDS[h]
        assert h != D.ER_HEX and h not in D.MF_PORT_HEX


def test_a_hex_with_a_track_slot_has_a_lay_on_the_map():
    for h in D.TRACK_HEXES:
        assert any(open_target(h, *target) and not D.TILE_BLOCKS_LAY[target[0]] for target in targets(printed(h), 0)), \
            D.HEX_IDS[h]


def test_a_hex_holds_a_city_ever_exactly_when_it_has_a_token_slot():
    """No upgrade turns a plain hex or a town into a city, the port tile and
    ER's tiles included; every other hex is a city from its printed tile on.
    TOKEN needs a city (``map_has_slot_for``)."""
    for h in range(D.NUM_HEXES):
        tiles = {tile for tile, _ in reachable(h)}
        if D.HEX_TOKEN_SLOT[h] < 0:
            assert not any(D.TILE_RC[t] == D.RC_CITY or D.TILE_SLOTS[t] for t in tiles), D.HEX_IDS[h]
        else:
            assert all(D.TILE_RC[t] == D.RC_CITY for t in tiles), D.HEX_IDS[h]
    # The privates' lays are in those walks: MF's on towns, ER's on C4, a city.
    assert D.TILE_RC[D.PORT_TILE] == D.RC_TOWN and D.HEX_TOKEN_SLOT[D.ER_HEX] >= 0
    for h, r in zip(D.PORT_LAY_HEX, D.PORT_LAY_ROT):
        assert (D.PORT_TILE, r) in reachable(h) and D.HEX_TOKEN_SLOT[h] < 0
    assert set(D.ER_TILE) <= {tile for tile, _ in reachable(D.ER_HEX)}


def _ar_at_j7(tile, rotation):
    """AR at TRACK, its K8 reaching J7, on which a hand-placed tile corrupts
    the state: J7 is gray, with neither a lay nor a token slot."""
    state = at_or(3)
    enter_phase(state, D.GAME_PHASE_3)
    hold(state, AR, {0: 6}, par=PAR_100)
    lay(state, 'K8', '6', 0)
    enter_operating_round(state)
    past_pre(state)
    assert TURN.operating_corp(state) == AR and TURN.step(state) == K.STEP_TRACK
    MAP.set_tile(state, HEX['J7'], tile, rotation)
    return state


def test_a_lay_without_a_slot_fails_loudly():
    state = _ar_at_j7(TILE['8'], 0)  # a yellow tile, which takes green
    with pytest.raises(AssertionError, match='no slot'):
        legal_ids(state)


def test_a_token_without_a_slot_fails_loudly():
    state = _ar_at_j7(TILE['5'], 5)  # a yellow city with an exit toward K8
    TURN.set_step(state, K.STEP_TOKEN)
    with pytest.raises(AssertionError, match='no slot'):
        legal_ids(state)


# =============================================================================
# ER_BONUS
# =============================================================================

def test_er_blocks_c4_until_its_sale_so_the_bonus_finds_the_printed_tile():
    """Every lay on C4 waits for the sale: ER blocks a corp's while a player owns
    it, no other private's lay names C4, and a corp never sells ER on, so
    the bonus fires once, on the printed tile. The auction sells ER before any
    OR, at every player count."""
    assert D.HEX_BLOCKER[D.ER_HEX] == D.PRIVATE_ER and D.ER_HEX not in D.MF_PORT_HEX
    assert all(D.PRIVATES_IN_PLAY[n] >> D.PRIVATE_ER & 1 for n in range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
    state = at_or(3)
    hold(state, UR, {0: 6}, par=PAR_100)
    MAP.place_token(state, D.CORP_HOME[UR], UR)
    lay(state, 'B5', '9', 0)
    assert GRAPH.sides_reached(state, UR, D.ER_HEX) and PRIVATES[D.PRIVATE_ER].blocks_hex(state, D.ER_HEX)
    assert D.ER_HEX not in A.lay_hexes_py(state, UR)
    enter_phase(state, D.GAME_PHASE_3)
    PRIVATES[D.PRIVATE_ER].sell_to_corp(state, UR, 40)
    assert not PRIVATES[D.PRIVATE_ER].blocks_hex(state, D.ER_HEX) and D.ER_HEX in A.lay_hexes_py(state, UR)
    assert A.purchasable_privates_py(state, UR).count(D.PRIVATE_ER) == 0


@pytest.mark.parametrize('phase', [D.GAME_PHASE_3, D.GAME_PHASE_4])
def test_the_printed_c4_gives_the_most_candidates_at_full_supply(phase):
    """The purchase window's phases allow green, and a thinner supply only
    removes candidates, so full supply on the printed tile is the bound."""
    state = at_or(3)
    enter_phase(state, phase)
    bound = [t for t in targets(printed(D.ER_HEX), 0) if t[0] in D.ER_TILE and open_target(D.ER_HEX, *t)]
    assert sorted(A.er_candidates_py(state)) == sorted(bound) and len(bound) == D.NUM_ER_CANDIDATES == 8
    assert not [p for p, can_buy in enumerate(D.PHASE_CAN_BUY_PRIVATES) if can_buy
                and not D.PHASE_TILE_COLORS[p] >> D.TILE_COLOR[D.ER_TILE[0]] & 1]
    MAP.set_supply(state, D.ER_TILE[0], 0)
    assert len(A.er_candidates_py(state)) < len(bound)


# =============================================================================
# PORT
# =============================================================================

def test_only_a_printed_town_takes_the_port_tile():
    """The upgrade table reaches the port tile only from a printed tile, whose
    rotation is 0: a lay on a port hex can only remove its port lays."""
    sources = {(tile, rotation) for tile in range(D.NUM_TILES) for rotation in range(D.NUM_SIDES)
               if any(new == D.PORT_TILE for new, _ in targets(tile, rotation))}
    assert sources and all(tile >= D.NUM_SUPPLY_TILES and rotation == 0 for tile, rotation in sources)
    assert {(printed(h), 0) for h in D.MF_PORT_HEX} <= sources


def test_the_port_lays_are_the_rotations_on_the_map():
    """Each port hex's printed tile takes the port tile at every rotation, and
    the map closes the rest: the pairs with no slot leave the board."""
    lays = list(zip(D.PORT_LAY_HEX, D.PORT_LAY_ROT))
    for h in D.MF_PORT_HEX:
        for r in range(D.NUM_SIDES):
            assert (D.PORT_TILE, r) in targets(printed(h), 0)
            assert ((h, r) in lays) == open_target(h, D.PORT_TILE, r), (D.HEX_IDS[h], r)


def test_port_legality_reads_the_state_only_to_remove_lays():
    """With MF a player's and every port hex printed, ``port_lays`` is all of
    the slots; a port hex with a tile has none."""
    state = at_or(3)
    lays = sorted(zip(D.PORT_LAY_HEX, D.PORT_LAY_ROT))
    assert sorted(A.port_lays_py(state)) == lays
    lay(state, 'G10', '3', 2)
    assert sorted(A.port_lays_py(state)) == [(h, r) for h, r in lays if h != HEX['G10']]
    PRIVATES[D.PRIVATE_MF].set_flags(state, 1)
    assert A.port_lays_py(state) == []


# =============================================================================
# DISCARD
# =============================================================================

def test_the_limit_drops_only_as_4s_and_5s_arrive():
    """A corp goes over its limit only as the limit drops, and that happens on
    the first 4 and the first 5. It never rises."""
    limits = D.PHASE_TRAIN_LIMIT
    assert all(after <= before for before, after in zip(limits, limits[1:]))
    drops = [D.PHASE_ON_TRAIN[p] for p in range(1, D.NUM_GAME_PHASES) if limits[p] < limits[p - 1]]
    assert drops == [D.TRAIN_4, D.TRAIN_5]


def test_no_discard_slot_is_missing_at_either_drop():
    """A 2 rusts on the very 4 that first drops the limit, so it is gone by
    either drop; and a 6 or a D ranks above the 5 that last drops it, and the
    depot sells by rank, so neither exists yet. 3s, 4s, and the buyer's 5 stay."""
    assert D.TRAIN_RUSTS_ON[D.TRAIN_2] == D.TRAIN_4
    assert D.TRAIN_RUSTS_ON[D.TRAIN_3] == D.TRAIN_6 and D.TRAIN_RUSTS_ON[D.TRAIN_4] == D.TRAIN_D
    assert D.TRAIN_5 < D.TRAIN_6 < D.TRAIN_D and D.TRAIN_RUSTS_ON[D.TRAIN_5] < 0
    assert D.DISCARD_TYPES == [D.TRAIN_3, D.TRAIN_4, D.TRAIN_5]
    state = at_or(3)
    enter_phase(state, D.GAME_PHASE_4)
    TRAINS.set_pool(state, D.TRAIN_4, TRAINS.pool(state, D.TRAIN_4) + TRAINS.depot(state, D.TRAIN_4))
    TRAINS.set_depot(state, D.TRAIN_4, 0)
    assert TRAINS.depot_types(state) == 1 << D.TRAIN_5  # with a 5 left, no 6 or D, as a phase 4 corp may buy
    TRAINS.set_depot(state, D.TRAIN_5, 0)
    assert TRAINS.depot_types(state) == 1 << D.TRAIN_6  # and the D waits for phase 6


@pytest.mark.parametrize('phase, held', [
    (D.GAME_PHASE_3, {D.TRAIN_3: 4}),
    (D.GAME_PHASE_4, {D.TRAIN_3: 2, D.TRAIN_4: 1}),
    (D.GAME_PHASE_5, {D.TRAIN_4: 1, D.TRAIN_5: 1}),
    (D.GAME_PHASE_6, {D.TRAIN_5: 1, D.TRAIN_6: 1}),
    (D.GAME_PHASE_D, {D.TRAIN_6: 1, D.TRAIN_D: 1}),
])
def test_a_corp_at_its_limit_can_only_trade_in(phase, held):
    """No purchase takes a corp past a limit that has not dropped: at the limit
    every source but a trade-in closes, from the depot, the pool, or another
    corp; and a trade-in leaves the count as it was."""
    state = at_or(3)
    enter_phase(state, phase)
    hold(state, IR, {1: 6})
    hold(state, KO, {0: 6}, par=PAR_100)
    give_train(state, IR, D.TRAIN_5 if phase >= D.GAME_PHASE_5 else D.TRAIN_3)
    for train, count in held.items():
        give_train(state, KO, train, count)
    set_cash(state, KO, 3000)
    enter_operating_round(state)
    assert TURN.operating_corp(state) == KO
    TURN.set_step(state, K.STEP_BUY_TRAIN)  # the driver would force the pass where nothing else is offered
    assert TRAINS.corp_total(state, KO) == D.PHASE_TRAIN_LIMIT[phase]
    choices = train_choices(state)
    assert all(choice == K.BUY_TRAIN_PASS or choice[0] == 'trade' for choice in choices), choices
    trades = [choice for choice in choices if choice != K.BUY_TRAIN_PASS]
    assert bool(trades) == (phase >= D.GAME_PHASE_6)
    if trades:
        choose_train(state, trade(trades[0][1]), expected_status=None)
        assert TRAINS.corp_total(state, KO) == D.PHASE_TRAIN_LIMIT[phase] and TRAINS.first_crowded(state) < 0


@pytest.mark.parametrize('train', [D.TRAIN_2, D.TRAIN_6, D.TRAIN_D])
def test_a_discard_without_a_slot_fails_loudly(train):
    state = at_or(3)
    hold(state, KO, {0: 6}, par=PAR_100)
    TRAINS.set_corp_trains(state, KO, train, D.PHASE_TRAIN_LIMIT[PHASE.current(state)] + 1)
    TURN.set_step(state, K.STEP_DISCARD)
    with pytest.raises(AssertionError, match='no slot'):
        legal_ids(state)


# =============================================================================
# AUCTION
# =============================================================================

def test_private_0_is_in_play_at_every_count_and_the_topmost_while_unsold():
    """So a bid on it is always the purchase above its price that buying at
    the price dominates, which the policy has always dropped; ``topmost`` is
    the first unsold private, and nothing sells a later one first."""
    assert all(D.PRIVATES_IN_PLAY[n] & 1 for n in range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
    assert D.PRIVATE_VALUE == sorted(D.PRIVATE_VALUE)
    for n in range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1):
        state = new_game(n)
        assert topmost(state) == 0
        apply_and_verify(state, K.AUCTION_BID)  # a bid on the next private leaves private 0 unsold and topmost
        if TURN.step(state) == K.STEP_BID:
            apply_and_verify(state, K.BID_POINT)
        assert topmost(state) == 0 and PRIVATES[0].bid_count(state) == 0


@pytest.mark.parametrize('fixture', FIXTURES)
def test_ruby_keeps_private_0_first_while_it_is_unsold(fixture):
    seen = 0
    for record in load_fixture_extract(fixture)['records']:
        auction = record['state']['auction']
        if auction is not None and D.PRIVATE_SYMS[0] in auction['companies']:
            assert auction['companies'][0] == D.PRIVATE_SYMS[0] and auction['auctioning'] != D.PRIVATE_SYMS[0]
            seen += 1
    assert seen


def test_no_later_private_loses_its_bid_slot_when_it_becomes_topmost():
    state = new_game(3)
    apply_and_verify(state, K.AUCTION_BUY)
    mf = K.AUCTION_BID + D.PRIVATE_MF - 1
    assert mf in legal_ids(state) and topmost(state) == D.PRIVATE_MF
    result = opened(state, mf)
    assert TURN.step(result) == K.STEP_BID and TURN.pending_private(result) == D.PRIVATE_MF


def test_the_override_reaches_the_purchase_ruby_takes():
    state = new_game(3)
    assert DRIVER.override_topmost_bid(state) == STATUS_OK
    apply_and_verify(state, K.BID_POINT + 3)
    assert PRIVATES[0].owner(state) == K.OWNER_PLAYER and TURN.step(state) == K.STEP_AUCTION
    assert topmost(state) == D.PRIVATE_MF


# =============================================================================
# THE SIZES
# =============================================================================

def test_each_size_is_its_table():
    assert (K.ACTION_SIZE_TRACK, K.ACTION_SIZE_TOKEN, K.ACTION_SIZE_DISCARD, K.ACTION_SIZE_ER_BONUS,
            K.ACTION_SIZE_PORT, K.ACTION_SIZE_AUCTION) == (
        1 + len(D.TRACK_HEXES), 1 + len(D.TOKEN_HEXES), len(D.DISCARD_TYPES), 1 + D.NUM_ER_CANDIDATES,
        len(D.PORT_LAY_HEX), 2 + D.NUM_PRIVATES - 1)
    assert K.MAX_ACTION_SIZE == K.ACTION_SIZE_TRACK == max(K.PHASE_ACTION_SIZES)
