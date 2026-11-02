"""MAP mutations, GRAPH connectivity, and lay and token legality on hand-built
maps (1889/engine-design.md §4, §5.4; rules §7, §8.4-8.5). The fixture and
random-map oracles in tests/g1889/replay/test_network_oracles.py check the
same code against Ruby at scale; these pin the individual rules.

Sides: 0 S, 1 SW, 2 NW, 3 N, 4 NE, 5 SE. Most cases start from KO's home,
Takamatsu (K4, a printed yellow T city with track to sides 0, 1, and 2):
side 0 faces K6, side 1 J5, and side 2 J3.
"""

import numpy as np
import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.phase import PHASE
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout

HEX = {hex_id: h for h, hex_id in enumerate(D.HEX_IDS)}
TILE = {name: t for t, name in enumerate(D.TILE_NAMES)}
ALL_CORPS = (1 << D.NUM_CORPS) - 1
KO, AR, IR, SR, TR = D.CORP_KO, D.CORP_AR, D.CORP_IR, D.CORP_SR, D.CORP_TR
ROUTE_DIRTY_SLOT = get_layout(D.MIN_PLAYERS).global_offset + get_global_fields().route_dirty


@pytest.fixture
def state():
    s = GameState(4)
    s.initialize_game(4, seed=1, priority_player=0)
    return s


def fund(state, corp, cash):
    BANK.transfer(state, BANK_ACCOUNT, K.OWNER_CORP + corp, cash)


def lay(state, hex_id, tile, rotation):
    MAP.lay(state, HEX[hex_id], TILE[tile], rotation)


def place(state, corp, hex_id):
    MAP.place_token(state, HEX[hex_id], corp)


def sides(state, corp, hex_id):
    mask = GRAPH.sides_reached(state, corp, HEX[hex_id])
    return {s for s in range(D.NUM_SIDES) if mask >> s & 1}


def candidates(state, corp, hex_id):
    return {(D.TILE_NAMES[t], r) for t, r in A.lay_candidates_py(state, HEX[hex_id], corp)}


def exits(tile, rotation):
    return {s for s in range(D.NUM_SIDES) if D.TILE_EXITS[TILE[tile]][rotation] >> s & 1}


def takamatsu(state, cash=0):
    """KO with its home token on K4."""
    fund(state, KO, cash)
    place(state, KO, 'K4')
    return state


# =============================================================================
# MAP
# =============================================================================

def test_lay_takes_from_and_returns_to_the_supply(state):
    nine, green = TILE['9'], TILE['24']
    lay(state, 'J3', '9', 5)
    assert (MAP.tile(state, HEX['J3']), MAP.rotation(state, HEX['J3'])) == (nine, 2)  # a straight: 5 is 2
    assert MAP.supply(state, nine) == D.TILE_COUNT[nine] - 1
    lay(state, 'J3', '24', 5)  # paths 5-1 and 5-2 keep the straight
    assert MAP.supply(state, nine) == D.TILE_COUNT[nine]
    assert MAP.supply(state, green) == D.TILE_COUNT[green] - 1
    # A printed tile never returns: upgrading K4 leaves the supply as it was, less the new tile.
    before = [MAP.supply(state, t) for t in range(D.NUM_SUPPLY_TILES)]
    lay(state, 'K4', '440', 0)
    after = [MAP.supply(state, t) for t in range(D.NUM_SUPPLY_TILES)]
    assert [b - a for b, a in zip(before, after)] == [int(t == TILE['440']) for t in range(D.NUM_SUPPLY_TILES)]


def test_lay_keeps_tokens_and_invalidates_every_corp(state):
    takamatsu(state)
    for c in range(D.NUM_CORPS):
        GRAPH.refresh(state, c)
    state._array[ROUTE_DIRTY_SLOT] = 0
    assert MAP.conn_dirty(state) == 0
    lay(state, 'K4', '440', 0)
    assert MAP.tokens(state, HEX['K4']) == 1 << KO
    assert MAP.conn_dirty(state) == ALL_CORPS and all(c.route_dirty(state) for c in CORPS)


def test_lay_rejects_what_the_upgrade_table_does_not_hold(state):
    with pytest.raises(AssertionError, match='does not upgrade'):
        lay(state, 'J3', '24', 0)  # green on an empty hex
    with pytest.raises(AssertionError, match='does not upgrade'):
        lay(state, 'K4', '440', 3)  # loses K4's track
    MAP.set_supply(state, TILE['9'], 0)
    with pytest.raises(AssertionError, match='not in supply'):
        lay(state, 'J3', '9', 2)


def test_lay_track_pays_the_terrain_cost(state):
    fund(state, IR, 100)
    MAP.lay_track(state, IR, HEX['E4'], TILE['9'], 0)
    assert (CORPS[IR].cash(state), BANK.cash(state)) == (20, D.BANK_CASH - 4 * D.START_CASH[4] - 20)


def test_place_and_buy_token(state):
    fund(state, KO, 50)
    MAP.buy_token(state, HEX['K4'], KO)  # the home token is free
    assert (CORPS[KO].tokens_left(state), CORPS[KO].cash(state)) == (1, 50)
    assert MAP.tokens(state, HEX['K4']) == 1 << KO and MAP.conn_dirty(state) == ALL_CORPS
    assert CORPS[KO].token_price(state) == D.CORP_TOKEN_PRICE[KO][1] == 40
    lay(state, 'J5', '58', 2)
    lay(state, 'I4', '438', 0)  # Kotohira: KO's track does not reach it, which buy_token does not check
    MAP.buy_token(state, HEX['I4'], KO)
    assert (CORPS[KO].tokens_left(state), CORPS[KO].cash(state), CORPS[KO].token_price(state)) == (0, 10, 0)
    with pytest.raises(AssertionError, match='no tokens left'):
        MAP.place_token(state, HEX['F3'], KO)


def test_place_token_needs_a_slot(state):
    with pytest.raises(AssertionError, match='cannot token'):
        place(state, KO, 'J3')  # no city
    with pytest.raises(AssertionError, match='cannot token'):
        place(state, KO, 'E2')  # IR's home reservation holds its one slot
    place(state, IR, 'E2')
    with pytest.raises(AssertionError, match='cannot token'):
        place(state, KO, 'E2')  # full


def test_slots_reservations_and_blocking(state):
    k8 = HEX['K8']  # AR's home: one slot, reserved
    assert (MAP.free_slots(state, k8), MAP.has_slot_for(state, k8, AR), MAP.has_slot_for(state, k8, KO)) == \
        (0, True, False)
    assert not MAP.blocked_for(state, k8, KO)  # a reservation never blocks
    place(state, AR, 'K8')
    assert (MAP.free_slots(state, k8), MAP.has_slot_for(state, k8, KO), MAP.has_slot_for(state, k8, AR)) == \
        (0, False, False)
    assert MAP.blocked_for(state, k8, KO) and not MAP.blocked_for(state, k8, AR)
    lay(state, 'K8', '6', 1)
    lay(state, 'K8', '15', 1)  # two slots
    assert (MAP.free_slots(state, k8), MAP.has_slot_for(state, k8, KO)) == (1, True)
    assert not MAP.blocked_for(state, k8, KO)
    # Before AR's home token, an upgraded home has one slot for others.
    e2 = HEX['E2']
    lay(state, 'E2', '57', 1)
    lay(state, 'E2', '15', 4)
    assert (MAP.free_slots(state, e2), MAP.has_slot_for(state, e2, KO)) == (1, True)
    assert not MAP.blocked_for(state, HEX['J5'], KO) and MAP.free_slots(state, HEX['J5']) == 0  # a town


def test_revenue(state):
    assert MAP.revenue(state, HEX['K8'], D.GAME_PHASE_2) == 0  # an empty printed city
    assert MAP.revenue(state, HEX['K4'], D.GAME_PHASE_2) == 30
    lay(state, 'K8', '6', 1)
    assert MAP.revenue(state, HEX['K8'], D.GAME_PHASE_5) == 20
    f1 = HEX['F1']
    assert [MAP.revenue(state, f1, p) for p in range(D.NUM_GAME_PHASES)] == [30, 30, 30, 60, 60, 60]
    assert MAP.revenue(state, f1, D.GAME_PHASE_D, True) == 100


def test_lay_cost_terrain_and_smr(state):
    smr = PRIVATES[D.PRIVATE_SMR]
    costs = {hex_id: MAP.lay_cost(state, HEX[hex_id], IR) for hex_id in ('E4', 'K6', 'H5', 'I4', 'F9', 'D3')}
    assert costs == {'E4': 80, 'K6': 80, 'H5': 80, 'I4': 80, 'F9': 80, 'D3': 0}
    smr.set_owner(state, K.OWNER_PLAYER + 1)
    assert MAP.lay_cost(state, HEX['E4'], IR) == 80  # SMR helps only the corp that owns it
    smr.set_owner(state, K.OWNER_CORP + IR)
    costs = {hex_id: MAP.lay_cost(state, HEX[hex_id], IR) for hex_id in ('E4', 'K6', 'H5', 'I4', 'F9')}
    assert costs == {'E4': 0, 'K6': 80, 'H5': 80, 'I4': 80, 'F9': 80}  # pure mountain only
    assert MAP.lay_cost(state, HEX['E4'], SR) == 80
    lay(state, 'E4', '9', 0)
    lay(state, 'I4', '438', 0)
    assert (MAP.lay_cost(state, HEX['E4'], SR), MAP.lay_cost(state, HEX['I4'], SR)) == (0, 80)  # #438 carries ¥80


# =============================================================================
# GRAPH
# =============================================================================

def test_a_token_reaches_its_hex_and_its_track(state):
    takamatsu(state)
    assert GRAPH.dirty(state, KO)
    assert {hex_id: sides(state, KO, hex_id) for hex_id in D.HEX_IDS if sides(state, KO, hex_id)} == {
        'K4': {0, 1, 2}, 'K6': {3}, 'J5': {4}, 'J3': {5}}
    assert [hex_id for hex_id in D.HEX_IDS if GRAPH.node_reached(state, KO, HEX[hex_id])] == ['K4']
    assert not GRAPH.dirty(state, KO)
    assert not any(sides(state, AR, hex_id) for hex_id in D.HEX_IDS)  # other corps reach nothing


def test_a_white_home_reaches_its_open_sides(state):
    place(state, IR, 'E2')  # no track yet
    assert {hex_id: sides(state, IR, hex_id) for hex_id in D.HEX_IDS if sides(state, IR, hex_id)} == {
        'E2': {0, 1, 4, 5}}
    assert not GRAPH.route_available(state, IR)


def test_walks_pass_towns(state):
    takamatsu(state)
    lay(state, 'J5', '58', 2)  # town on sides 2 and 4
    assert sides(state, KO, 'J5') == {2, 4} and sides(state, KO, 'I4') == {5}
    assert GRAPH.node_reached(state, KO, HEX['J5'])


def test_walks_enter_but_never_pass_offboards(state):
    takamatsu(state)
    lay(state, 'J3', '8', 3)  # K4 to Okayama (J1)
    assert GRAPH.node_reached(state, KO, HEX['J1'])
    assert sides(state, KO, 'J1') == {0} and sides(state, KO, 'I2') == set()


def test_walks_enter_but_never_pass_blocked_cities(state):
    takamatsu(state)
    lay(state, 'K6', '9', 0)
    lay(state, 'K8', '6', 1)  # AR's home, sides 1 and 3
    assert sides(state, KO, 'K8') == {1, 3} and sides(state, KO, 'J9') == {4}  # reserved, not blocking
    place(state, AR, 'K8')
    assert GRAPH.node_reached(state, KO, HEX['K8'])
    assert sides(state, KO, 'K8') == {3} and sides(state, KO, 'J9') == set()
    assert sides(state, AR, 'J9') == {4}  # AR's own token never blocks it


def test_walks_never_reverse_at_a_junction(state):
    takamatsu(state)
    MAP.set_tile(state, HEX['K6'], TILE['23'], 5)  # branches 5-2 and 5-3 meet at side 5
    assert sides(state, KO, 'K6') == {3, 5} and sides(state, KO, 'J5') == {4}  # the 5-2 branch is not taken
    MAP.set_tile(state, HEX['K6'], TILE['24'], 3)  # branches 3-5 and 3-0 split at the side KO enters by
    assert sides(state, KO, 'K6') == {0, 3, 5} and sides(state, KO, 'K8') == {3}


def test_route_available_needs_two_revenue_centers_on_one_walk(state):
    takamatsu(state)
    place(state, KO, 'A10')  # an isolated second token without track
    assert not GRAPH.route_available(state, KO)  # one center each
    lay(state, 'J5', '58', 2)
    assert GRAPH.route_available(state, KO)
    other = GameState(4)
    other.initialize_game(4, priority_player=0)
    takamatsu(other)
    lay(other, 'J3', '8', 3)
    assert GRAPH.route_available(other, KO)  # K4 and Okayama
    lay(other, 'J5', '58', 2)
    place(other, AR, 'K8')
    assert not GRAPH.route_available(other, AR)  # an empty printed city has no track to a second center


# =============================================================================
# LAYS
# =============================================================================

def test_lays_need_a_reached_exit_on_open_sides(state):
    takamatsu(state)
    lays = candidates(state, KO, 'J3')
    assert lays == {('7', 5), ('8', 3), ('8', 5), ('9', 2)}
    for tile, rotation in lays:
        assert 5 in exits(tile, rotation) and 4 not in exits(tile, rotation)  # J3's side 4 faces the sea
    assert candidates(state, KO, 'J7') == set()  # gray
    assert candidates(state, KO, 'I8') == set()  # not reached
    assert A.lay_hexes_py(state, KO) == [HEX['J3'], HEX['J5']]  # K6 is water: ¥80, and KO has none
    fund(state, KO, 80)
    assert A.lay_hexes_py(state, KO) == [HEX['J3'], HEX['J5'], HEX['K6']]
    assert candidates(state, KO, 'K6') == {('7', 2), ('8', 3), ('9', 0)}


def test_terrain_must_be_affordable_unless_smr_pays(state):
    place(state, IR, 'E2')
    assert A.lay_hexes_py(state, IR) == [HEX['E2']]  # a token without track reaches only its own hex
    lay(state, 'E2', '6', 4)  # Matsuyama's track to sides 0 (E4, a mountain) and 4 (Imabari)
    fund(state, IR, 79)
    assert candidates(state, IR, 'E4') == set()
    mountain = {('7', 2), ('7', 3), ('8', 1), ('8', 3), ('9', 0)}
    PRIVATES[D.PRIVATE_SMR].set_owner(state, K.OWNER_CORP + IR)
    assert candidates(state, IR, 'E4') == mountain
    PRIVATES[D.PRIVATE_SMR].set_owner(state, K.OWNER_PLAYER)
    assert candidates(state, IR, 'E4') == set()
    fund(state, IR, 1)
    assert candidates(state, IR, 'E4') == mountain


def test_lays_follow_the_phase_colors_and_labels(state):
    takamatsu(state)
    assert candidates(state, KO, 'K4') == set()  # yellow only in phase 2
    PHASE.set_current(state, D.GAME_PHASE_3)
    assert {tile for tile, _ in candidates(state, KO, 'K4')} == {'440'}  # T takes only T
    PHASE.set_current(state, D.GAME_PHASE_5)
    assert {tile for tile, _ in candidates(state, KO, 'K4')} == {'440'}  # brown only over green
    place(state, SR, 'I2')
    assert {tile for tile, _ in candidates(state, SR, 'I2')} == {'5', '6', '57'}  # no labels on plain cities
    lay(state, 'I2', '5', 0)  # Marugame's track to side 0, Kotohira (I4)
    fund(state, SR, 80)
    assert {tile for tile, _ in candidates(state, SR, 'I4')} == {'438'}  # H takes only H
    place(state, TR, 'F9')
    fund(state, TR, 80)
    assert {tile for tile, _ in candidates(state, TR, 'F9')} == {'465'}  # K's only upgrade


def test_private_blockers_hold_their_hex_while_a_player_owns_them(state):
    takamatsu(state)
    PHASE.set_current(state, D.GAME_PHASE_3)
    tr, er = PRIVATES[D.PRIVATE_TR], PRIVATES[D.PRIVATE_ER]
    assert candidates(state, KO, 'K4')  # unsold blocks nothing
    tr.set_owner(state, K.OWNER_PLAYER + 2)
    assert candidates(state, KO, 'K4') == set()
    assert tr.blocks_hex(state, HEX['K4']) and not tr.blocks_hex(state, HEX['C4'])
    tr.set_owner(state, K.OWNER_CORP + KO)
    assert candidates(state, KO, 'K4')
    tr.set_owner(state, K.OWNER_CLOSED)
    assert candidates(state, KO, 'K4')
    er.set_owner(state, K.OWNER_PLAYER)
    assert er.blocks_hex(state, HEX['C4']) and not er.blocks_hex(state, HEX['K4'])


def test_lays_need_the_tile_in_supply(state):
    takamatsu(state)
    PHASE.set_current(state, D.GAME_PHASE_3)
    MAP.set_supply(state, TILE['440'], 0)
    assert candidates(state, KO, 'K4') == set()
    for tile in ('7', '8'):
        MAP.set_supply(state, TILE[tile], 0)
    assert candidates(state, KO, 'J3') == {('9', 2)}


def test_semi_restrictive_rule_needs_reachable_new_track(state):
    """J3 holds a straight from side 5, where KO arrives, to side 2. A green
    upgrade must add a branch KO can reach: one at side 5 does, one at side 2
    or a crossing does not, since KO would have to reverse to use it."""
    takamatsu(state)
    lay(state, 'J3', '9', 2)
    PHASE.set_current(state, D.GAME_PHASE_3)
    lays = candidates(state, KO, 'J3')
    straight = D.TILE_MASK[TILE['9']][2]
    for t, r in A.lay_candidates_py(state, HEX['J3'], KO):
        new = D.TILE_MASK[t][r] & ~straight
        assert new & D.ENDPOINT_PAIRS[5], (D.TILE_NAMES[t], r)  # a branch at the side KO arrives by
    assert lays == {('23', 5), ('24', 5), ('27', 5)}
    # Each green tile that keeps the straight and stays on open sides, but adds
    # track only at side 2 or apart from it, is out.
    k = D.NUM_SIDES * TILE['9'] + 2
    upgrades = {(D.TILE_NAMES[D.UPG_TILE[i]], D.UPG_ROT[i]) for i in range(D.UPG_START[k], D.UPG_START[k + 1])
                if D.TILE_EXITS[D.UPG_TILE[i]][D.UPG_ROT[i]] & ~D.HEX_EXIT_MASK[HEX['J3']] == 0}
    assert upgrades - lays == {('19', 5), ('20', 2), ('23', 2), ('26', 2), ('27', 2)}
    # A city upgrade changes the city's value, which passes the rule without new track.
    PHASE.set_current(state, D.GAME_PHASE_5)
    lay(state, 'K4', '440', 0)
    assert {tile for tile, _ in candidates(state, KO, 'K4')} == {'466'}


# =============================================================================
# TOKENS
# =============================================================================

def test_tokens_need_a_reached_city_with_a_slot(state):
    takamatsu(state, cash=100)
    lay(state, 'K6', '9', 0)
    assert A.token_hexes_py(state, KO) == []  # K8 has no track to its city yet
    lay(state, 'K8', '6', 1)
    assert A.token_hexes_py(state, KO) == []  # reserved for AR
    lay(state, 'K8', '15', 1)
    assert A.token_hexes_py(state, KO) == [HEX['K8']]  # a second slot
    place(state, AR, 'K8')
    assert A.token_hexes_py(state, KO) == [HEX['K8']]
    place(state, IR, 'K8')
    assert A.token_hexes_py(state, KO) == []  # full
    assert A.token_hexes_py(state, IR) == []  # no track of its own there, and a token already


def test_tokens_cost_the_next_charter_price(state):
    takamatsu(state, cash=39)
    lay(state, 'K6', '9', 0)
    lay(state, 'K8', '6', 1)
    lay(state, 'K8', '15', 1)
    assert A.token_hexes_py(state, KO) == []  # ¥40
    fund(state, KO, 1)
    assert A.token_hexes_py(state, KO) == [HEX['K8']]
    place(state, KO, 'K8')
    assert A.token_hexes_py(state, KO) == [] and CORPS[KO].tokens_left(state) == 0


def test_track_tile_and_token_ids(state):
    takamatsu(state, cash=100)
    TURN.set_round(state, K.ROUND_OPERATING)
    TURN.set_operating_corp(state, KO)
    ids = np.zeros(K.MAX_ACTION_SIZE, dtype=np.uint16)

    def legal():
        return [int(i) for i in ids[:A.enumerate_legal_actions_py(state, ids)]]

    TURN.set_step(state, K.STEP_TRACK)
    assert legal() == [K.TRACK_PASS] + [K.TRACK_HEX + HEX[h] for h in ('J3', 'J5', 'K6')]
    assert A.enumerate_policy_actions_py(state, ids) == 4
    TURN.set_step(state, K.STEP_TILE)
    TURN.set_pending_hex(state, HEX['J3'])
    assert legal() == [0, 1, 2, 3]
    assert A.lay_candidates_py(state, HEX['J3']) == [(TILE['7'], 5), (TILE['8'], 3), (TILE['8'], 5), (TILE['9'], 2)]
    assert A.is_legal_action_py(state, 3) and not A.is_legal_action_py(state, 4)
    TURN.set_step(state, K.STEP_TOKEN)
    assert legal() == [K.TOKEN_PASS]
    lay(state, 'K6', '9', 0)
    lay(state, 'K8', '6', 1)
    lay(state, 'K8', '15', 1)
    assert legal() == [K.TOKEN_PASS, K.TOKEN_HEX + HEX['K8']]
