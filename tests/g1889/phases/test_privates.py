"""Private purchases, PRIVATE_PRICE, ER_BONUS, PORT, and DR's exchange, as
6.8a builds them (1889/engine-design.md §5.1-5.3, §12 items 4 and 6; rules
§10; Ruby's ``step/buy_company.rb``, ``step/exchange.rb``,
``step/track_lay_when_company_sold.rb``, ``game/g_1889/step/special_track.rb``,
``company.rb``, and ``game/base.rb``'s ``purchasable_companies`` and ability
times).

Covers the purchase window (phases 3-4, a player's private, half its face in
the treasury) and BUY_PRIVATE_PRE and BUY_PRIVATE_POST, the decisions that
offer it; ``DRIVER.override_private_slot``, replay's way to the turn's other
decisions; PRIVATE_PRICE's grid, exact amounts, and its forced single price; a
sale's effects (the seller's cash and certificates, TR's and ER's blocks,
SMR's discount, UTF's revenue and closing); ER's bonus lay on C4 for its
seller, on top of the corp's lay; MF's port tile on its owner's SR turn and at
the start of the turn of a corp its owner is president of, without using the
corp's lay; DR's exchange (before IR is parred, the presidency, the 60% limit,
the certificate limit, and a float mid-OR); and the
``private_sale_same_president`` flag. Everything runs through the driver;
``tests/g1889/replay/`` checks the same rules against Ruby at every fixture
snapshot.

KO (president seat 0) operates from Takamatsu (K4); the privates belong to
seat 2 unless a test says otherwise. J3's tile 8 at rotation 3 joins K4 to
Sakaide & Okayama, a route of ¥50 for a 2-train in phases 2-4.
"""

import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID, STATUS_OK
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.phases.operating import setup_operating_round_py

from tests.g1889.phases.conftest import (
    HEX, TILE, apply_and_verify, at_or, choose_train, depot, end_auction, enter_operating_round, enter_phase,
    enter_stock_round, exact_id, give_train, hold, lay, legal_ids, new_game, past_pre, policy_ids, set_cash,
    train_choices,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
P_TR, P_MF, P_ER, P_SMR, P_DR, P_SIR, P_UTF = range(D.NUM_PRIVATES)
PAR_65, PAR_100 = 0, 5
T2, T3 = D.TRAIN_2, D.TRAIN_3
PHASE_2, PHASE_3, PHASE_4, PHASE_5 = D.GAME_PHASE_2, D.GAME_PHASE_3, D.GAME_PHASE_4, D.GAME_PHASE_5
IN_PLAY_3 = [P_TR, P_MF, P_ER, P_SMR, P_DR, P_SIR]  # UTF needs four players


def at_turn(phase=PHASE_3, owners=(), num_players=3, cash=None, route=False):
    """KO at its turn's first choice in ``phase``; ``route`` gives it a 2-train
    and J3's track, so it earns ¥50."""
    state = at_or(num_players, owners)
    enter_phase(state, phase)
    hold(state, KO, {0: 6}, par=PAR_100)
    if route:
        give_train(state, KO, T2 if phase < PHASE_4 else T3)
        lay(state, 'J3', '8', 3)
    if cash is not None:
        set_cash(state, KO, cash)
    enter_operating_round(state)
    assert TURN.operating_corp(state) == KO
    return state


def at_pre(*args, **kwargs):
    """KO at BUY_PRIVATE_PRE, a private's slot legal."""
    state = at_turn(*args, **kwargs)
    assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
    return state


def at_track(*args, **kwargs):
    """KO at its first TRACK choice, past BUY_PRIVATE_PRE if it opens."""
    state = at_turn(*args, **kwargs)
    past_pre(state)
    assert TURN.step(state) == K.STEP_TRACK
    return state


def buy_slot(first, privates):
    return [first + p for p in privates]


def buy(state, slot, private, price_point=None, exact=None):
    """Select ``private`` at the current decision's slot, then pay a grid point or an exact price."""
    apply_and_verify(state, slot + private)
    assert TURN.step(state) == K.STEP_PRIVATE_PRICE and TURN.pending_private(state) == private
    return apply_and_verify(state, exact_id(state, exact) if exact is not None else price_point)


def prices(state):
    """PRIVATE_PRICE's grid as (id, yen)."""
    return [(i, A.price_amount_py(state, i)) for i in legal_ids(state)]


# =============================================================================
# THE WINDOW
# =============================================================================

class TestWindow:
    @pytest.mark.parametrize('phase, open_', [(PHASE_2, False), (PHASE_3, True), (PHASE_4, True)])
    def test_phases_3_and_4_open_the_window(self, phase, open_):
        state = at_turn(phase)
        assert A.purchasable_privates_py(state) == (IN_PLAY_3 if open_ else [])
        assert TURN.step(state) == (K.STEP_BUY_PRIVATE_PRE if open_ else K.STEP_TRACK)
        if open_:
            assert legal_ids(state) == [K.BUY_PRIVATE_PRE_DONE] + buy_slot(K.BUY_PRIVATE_PRE_PRIVATE, IN_PLAY_3)

    def test_the_privates_close_at_phase_5(self):
        state = at_turn(PHASE_5)
        assert all(PRIVATES[p].owner(state) == K.OWNER_CLOSED for p in IN_PLAY_3)
        assert A.purchasable_privates_py(state) == [] and TURN.step(state) == K.STEP_TRACK

    def test_the_first_5_closes_the_window_before_post(self):
        """engine-design §5.1's limitation: PRE is past, the 5 closes the
        privates, and the turn ends with no POST, where the withhold's cash
        could have bought one."""
        state = at_or(3)
        enter_phase(state, PHASE_4)
        hold(state, KO, {0: 6}, par=PAR_100)
        TRAINS.set_pool(state, D.TRAIN_4, TRAINS.pool(state, D.TRAIN_4) + TRAINS.depot(state, D.TRAIN_4))
        TRAINS.set_depot(state, D.TRAIN_4, 0)
        enter_operating_round(state)
        past_pre(state)
        apply_and_verify(state, K.TRACK_PASS)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and A.purchasable_privates_py(state) == IN_PLAY_3
        choose_train(state, depot(D.TRAIN_5))
        assert PHASE.current(state) == PHASE_5 and A.purchasable_privates_py(state) == []
        result = apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert K.STEP_TURN_END in result.steps and not [phase for phase, _ in result.forced
                                                        if phase == K.DPHASE_BUY_PRIVATE_POST]

    def test_a_private_needs_half_its_face_in_the_treasury(self):
        """``min_price`` is ⌈face/2⌉ (TR ¥10, MF ¥15, ER ¥20, SMR ¥25, DR ¥30, SIR ¥40)."""
        state = at_pre(cash=29)
        assert A.purchasable_privates_py(state) == [P_TR, P_MF, P_ER, P_SMR]
        set_cash(state, KO, 9)
        assert A.purchasable_privates_py(state) == []

    def test_only_a_players_private_is_for_sale(self):
        state = at_pre()
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_SIR, price_point=0)
        assert PRIVATES[P_SIR].owner(state) == K.OWNER_CORP + KO and TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
        assert P_SIR not in A.purchasable_privates_py(state)
        assert P_SIR not in A.purchasable_privates_py(state, IR)  # nor to another corp (``company_sellable``)

    def test_post_asks_when_the_turns_decisions_are_forced(self):
        """KO cannot token and earns nothing, so Ruby skips Token and Dividend,
        and with nothing to buy it skips BuyTrain: the driver forces each,
        whatever the privates, and BUY_PRIVATE_POST asks."""
        state = at_track(cash=50)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced == [(K.DPHASE_TOKEN, K.TOKEN_PASS), (K.DPHASE_DIVIDEND, K.DIVIDEND_WITHHOLD),
                                 (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)]
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST
        assert legal_ids(state) == [K.BUY_PRIVATE_POST_DONE] + buy_slot(K.BUY_PRIVATE_POST_PRIVATE, IN_PLAY_3)

    def test_track_token_dividend_and_buy_train_offer_no_slot(self):
        """Each offers only its own choices, in the window with every private buyable."""
        state = at_track(route=True)
        lay(state, 'K6', '9', 0)  # K8, AR's home, has a slot KO may token
        lay(state, 'K8', '6', 1)
        lay(state, 'K8', '15', 1)
        assert A.purchasable_privates_py(state) == IN_PLAY_3
        assert all(K.TRACK_HEX <= i for i in legal_ids(state)[1:]) and legal_ids(state)[0] == K.TRACK_PASS
        apply_and_verify(state, K.TRACK_PASS)
        assert legal_ids(state) == [K.TOKEN_PASS, K.TOKEN_HEX + HEX['K8']]
        apply_and_verify(state, K.TOKEN_PASS)
        assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) > 0
        assert legal_ids(state) == [K.DIVIDEND_WITHHOLD, K.DIVIDEND_PAYOUT]
        apply_and_verify(state, K.DIVIDEND_WITHHOLD)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and depot(T3) in train_choices(state)
        assert A.purchasable_privates_py(state) == IN_PLAY_3
        assert all(not isinstance(choice, int) or choice == K.BUY_TRAIN_PASS for choice in train_choices(state))

    def test_the_same_president_flag_keeps_other_players_privates_out_of_the_policy(self):
        state = at_or(3, owners=[0, 2])  # seat 0, KO's president, owns TR, ER, DR
        state.private_sale_same_president = True
        enter_phase(state, PHASE_3)
        hold(state, KO, {0: 6}, par=PAR_100)
        enter_operating_round(state)
        slots = buy_slot(K.BUY_PRIVATE_PRE_PRIVATE, IN_PLAY_3)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
        assert [i for i in legal_ids(state) if i in slots] == slots
        assert [i for i in policy_ids(state) if i in slots] == buy_slot(K.BUY_PRIVATE_PRE_PRIVATE, [P_TR, P_ER, P_DR])
        state.private_sale_same_president = False
        assert policy_ids(state) == legal_ids(state)


# =============================================================================
# BUY_PRIVATE_PRE
# =============================================================================

class TestPre:
    def test_it_stops_only_with_a_slot_besides_done(self):
        """A turn with nothing to buy, exchange, or lay goes straight to TRACK: PRE is not in its history."""
        state = at_or(3)
        enter_phase(state, PHASE_3)
        hold(state, KO, {0: 6}, par=PAR_100)
        set_cash(state, KO, 9)  # below every private's minimum, seat 2's DR and MF not KO's president's
        history = []
        TURN.set_or_count(state, 1)
        setup_operating_round_py(state)
        DRIVER.advance(state, history)
        assert TURN.step(state) == K.STEP_TRACK
        assert not [phase for _, phase, _ in history if phase == K.DPHASE_BUY_PRIVATE_PRE]

    def test_done_opens_track(self):
        state = at_pre()
        hexes = A.lay_hexes_py(state)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_DONE)
        assert TURN.step(state) == K.STEP_TRACK
        assert legal_ids(state) == [K.TRACK_PASS] + [K.TRACK_HEX + h for h in hexes]

    def test_it_repeats_after_a_purchase_until_done(self):
        state = at_pre()
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_TR, price_point=0)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_SMR, price_point=0)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
        assert legal_ids(state) == [K.BUY_PRIVATE_PRE_DONE] + buy_slot(K.BUY_PRIVATE_PRE_PRIVATE,
                                                                       [P_MF, P_ER, P_DR, P_SIR])

    def test_a_purchase_of_the_last_private_it_can_afford_forces_done(self):
        """¥50: SIR at its ¥40 minimum leaves ¥10, for TR alone; then nothing is left."""
        state = at_pre(cash=50)
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_SIR, price_point=0)
        assert A.purchasable_privates_py(state) == [P_TR] and TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
        result = apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_TR)  # its one price, then "done", are forced
        assert result.forced[:2] == [(K.DPHASE_PRIVATE_PRICE, 0), (K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DONE)]
        assert TURN.step(state) != K.STEP_BUY_PRIVATE_PRE

    def test_the_flags_force_done_where_only_rivals_own_privates(self):
        """Engine-legal purchases open PRE, but the policy leaves "done", which costs no search."""
        state = at_or(3)  # seat 2 owns every private; seat 0 presides KO
        state.private_sale_same_president = True
        enter_phase(state, PHASE_3)
        hold(state, KO, {0: 6}, par=PAR_100)
        TURN.set_or_count(state, 1)
        setup_operating_round_py(state)
        history = []
        DRIVER.advance(state, history)
        forced = [(phase, action_id) for _, phase, action_id in history if phase >= 0]
        assert (K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DONE) in forced
        assert TURN.step(state) != K.STEP_BUY_PRIVATE_PRE

    @pytest.mark.parametrize('phase', [PHASE_2, PHASE_3, PHASE_4])
    def test_dr_and_mf_open_it_in_their_owners_turn(self, phase):
        """In phase 2 too, where nothing can be bought; only they are offered there."""
        state = at_pre(phase, owners=[0])
        own = [K.BUY_PRIVATE_PRE_DR, K.BUY_PRIVATE_PRE_MF]
        if phase == PHASE_2:
            assert legal_ids(state) == [K.BUY_PRIVATE_PRE_DONE] + own
        else:
            assert legal_ids(state) == [K.BUY_PRIVATE_PRE_DONE] + buy_slot(K.BUY_PRIVATE_PRE_PRIVATE, IN_PLAY_3) + own

    def test_the_decision_is_the_presidents(self):
        state = at_pre()
        assert TURN.active_player(state) == CORPS[KO].president(state) == 0 and TURN.operating_corp(state) == KO


# =============================================================================
# REPLAY'S OVERRIDE
# =============================================================================

class TestOverride:
    def _at(self, step, **kwargs):
        """KO at a decision of its turn, in the window: ``step`` is one of TRACK, DIVIDEND, or BUY_TRAIN."""
        state = at_track(route=True, **kwargs)
        if step != K.STEP_TRACK:
            apply_and_verify(state, K.TRACK_PASS)
        if step == K.STEP_BUY_TRAIN:
            apply_and_verify(state, K.DIVIDEND_WITHHOLD)
        assert TURN.step(state) == step
        return state

    @pytest.mark.parametrize('step', [K.STEP_TRACK, K.STEP_DIVIDEND, K.STEP_BUY_TRAIN])
    def test_a_purchase_returns_to_the_decision_it_interrupted(self, step):
        state = self._at(step)
        before = legal_ids(state)
        cash, seller = CORPS[KO].cash(state), PLAYERS[2].cash(state)
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR) == STATUS_OK
        assert TURN.step(state) == K.STEP_PRIVATE_PRICE and TURN.return_step(state) == step
        assert apply_and_verify(state, exact_id(state, 123)).status == STATUS_OK
        assert PRIVATES[P_SIR].owner(state) == K.OWNER_CORP + KO
        assert CORPS[KO].cash(state) == cash - 123 and PLAYERS[2].cash(state) == seller + 123
        assert TURN.step(state) == step and legal_ids(state) == before and TURN.return_step(state) == -1

    def test_the_revenue_waits_through_it_at_dividend(self):
        state = self._at(K.STEP_DIVIDEND)
        revenue = TURN.route_revenue(state)
        assert revenue > 0 and DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_TR) == STATUS_OK
        apply_and_verify(state, 0)
        assert TURN.step(state) == K.STEP_DIVIDEND and TURN.route_revenue(state) == revenue

    def test_er_chains_through_its_bonus(self):
        state = self._at(K.STEP_BUY_TRAIN)
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_ER) == STATUS_OK
        apply_and_verify(state, 0)
        assert TURN.step(state) == K.STEP_ER_BONUS and TURN.active_player(state) == 2 == TURN.pending_seller(state)
        apply_and_verify(state, K.ER_BONUS_PASS)
        assert TURN.step(state) == K.STEP_BUY_TRAIN and TURN.active_player(state) == 0

    def test_dr_exchanges_in_place(self):
        state = self._at(K.STEP_DIVIDEND, owners=[0])
        before = legal_ids(state)
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_DR) == STATUS_OK
        assert PRIVATES[P_DR].owner(state) == K.OWNER_CLOSED and PLAYERS[0].shares(state, IR) == 1
        assert TURN.step(state) == K.STEP_DIVIDEND and legal_ids(state) == before

    def test_mf_lays_its_port_tile_and_returns(self):
        state = self._at(K.STEP_TRACK, owners=[0])
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_MF) == STATUS_OK
        assert TURN.step(state) == K.STEP_PORT and TURN.return_step(state) == K.STEP_TRACK
        apply_and_verify(state, K.PORT_LAY + D.NUM_SIDES * D.MF_PORT_HEX.index(HEX['G10']))
        assert TURN.step(state) == K.STEP_TRACK and PRIVATES[P_MF].flags(state) & 1

    def test_history_gets_the_auto_chain_not_the_override(self):
        """¥40 in the treasury: SIR has one price, which the driver forces."""
        state = at_track(route=True)
        set_cash(state, KO, 40)
        history = []
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR, history) == STATUS_OK
        assert [(phase, action_id) for _, phase, action_id in history][0] == (K.DPHASE_PRIVATE_PRICE, 0)
        assert CORPS[KO].cash(state) == 0

    def test_it_is_invalid_where_it_does_not_apply(self):
        state = self._at(K.STEP_BUY_TRAIN)
        row = state._array.copy()
        for action_id in (K.BUY_PRIVATE_PRE_DONE,  # not a move
                          K.BUY_PRIVATE_PRE_DR, K.BUY_PRIVATE_PRE_MF,  # seat 2's, not the president's
                          K.ACTION_SIZE_BUY_PRIVATE_PRE, -1):
            assert DRIVER.override_private_slot(state, action_id) == STATUS_INVALID, action_id
        assert (state._array == row).all()
        PRIVATES[P_SIR].close(state)  # no longer a player's
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR) == STATUS_INVALID

    def test_it_is_invalid_at_the_decisions_with_their_own_slots_and_outside_a_turn(self):
        slot = K.BUY_PRIVATE_PRE_PRIVATE + P_SIR
        state = at_pre(cash=50)  # too little for a train, so POST follows the lay
        assert DRIVER.override_private_slot(state, slot) == STATUS_INVALID  # PRE offers it
        apply_and_verify(state, K.BUY_PRIVATE_PRE_DONE)
        apply_and_verify(state, K.TRACK_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST
        assert DRIVER.override_private_slot(state, slot) == STATUS_INVALID
        apply_and_verify(state, K.BUY_PRIVATE_POST_PRIVATE + P_SIR)  # PRIVATE_PRICE, a sub-phase
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_TR) == STATUS_INVALID
        state = new_game(3)
        end_auction(state, [0])
        enter_stock_round(state, 0)  # DR is usable, but not through the override in an SR
        assert K.STOCK_DR in legal_ids(state)
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_DR) == STATUS_INVALID

    def test_it_is_invalid_past_the_window(self):
        state = at_track(PHASE_5)
        assert DRIVER.override_private_slot(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR) == STATUS_INVALID


# =============================================================================
# PRIVATE_PRICE AND THE SALE
# =============================================================================

class TestPrivatePrice:
    def test_the_grid_runs_from_half_to_twice_face(self):
        state = at_pre()
        apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR)
        assert prices(state) == [(0, 40), (1, 80), (2, 120), (3, 160)]
        assert A.is_legal_action_py(state, exact_id(state, 41)) and A.is_legal_action_py(state, exact_id(state, 160))
        assert not A.is_legal_action_py(state, exact_id(state, 39))
        assert not A.is_legal_action_py(state, exact_id(state, 161))

    def test_the_treasury_caps_the_price(self):
        """¥100 drops 1½·face (¥120), and the top point is the treasury."""
        state = at_pre(cash=100)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR)
        assert prices(state) == [(0, 40), (1, 80), (3, 100)]
        assert not A.is_legal_action_py(state, exact_id(state, 101))

    def test_equal_points_are_offered_once(self):
        """TR at ¥20 with ¥20: ⌈face/2⌉ ¥10, face ¥20, and the cap ¥20 again."""
        state = at_pre(cash=20)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_TR)
        assert prices(state) == [(0, 10), (1, 20)]

    def test_a_single_price_is_forced(self):
        state = at_pre(cash=40)
        result = apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_SIR)
        assert result.forced[0] == (K.DPHASE_PRIVATE_PRICE, 0) and CORPS[KO].cash(state) == 0
        assert PRIVATES[P_SIR].owner(state) == K.OWNER_CORP + KO

    def test_a_sale_pays_the_seller_from_the_treasury(self):
        state = at_pre()
        seller, ko, certs = PLAYERS[2].cash(state), CORPS[KO].cash(state), PLAYERS[2].cert_count(state)
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_SIR, exact=123)
        assert PLAYERS[2].cash(state) == seller + 123 and CORPS[KO].cash(state) == ko - 123
        assert PLAYERS[2].cert_count(state) == certs - 1  # privates count toward the limit
        assert TURN.return_step(state) == -1 and TURN.active_player(state) == 0

    def test_tr_blocks_k4_until_a_corp_owns_it(self):
        state = at_pre()
        assert PRIVATES[P_TR].blocks_hex(state, HEX['K4']) and HEX['K4'] not in A.lay_hexes_py(state)
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_TR, price_point=0)
        assert not PRIVATES[P_TR].blocks_hex(state, HEX['K4']) and HEX['K4'] in A.lay_hexes_py(state)

    def test_smr_discounts_its_corps_pure_mountain(self):
        state = at_pre()
        assert MAP.lay_cost(state, HEX['E4'], KO) == 80
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_SMR, price_point=0)
        assert MAP.lay_cost(state, HEX['E4'], KO) == 0 and MAP.lay_cost(state, HEX['E4'], IR) == 80
        assert MAP.lay_cost(state, HEX['H5'], KO) == 80  # river and mountain

    def test_utf_pays_its_corp_and_closes_with_it_at_phase_5(self):
        """A player's UTF earns ¥50 from phase 5 and stays open; a corp's earns
        ¥30 and closes (``close`` binds only for a player owner)."""
        state = at_pre(num_players=4)
        assert P_UTF in A.purchasable_privates_py(state)
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_UTF, price_point=0)
        assert PRIVATES[P_UTF].revenue(state) == 30
        PHASE.advance(state)  # to phase 4
        PHASE.advance(state)  # to phase 5
        assert PHASE.current(state) == PHASE_5 and PRIVATES[P_UTF].owner(state) == K.OWNER_CLOSED


# =============================================================================
# ER_BONUS
# =============================================================================

ER_LAYS = [('12', 0), ('13', 0), ('14', 1), ('15', 5), ('205', 1), ('205', 5), ('206', 2), ('206', 5)]


class TestErBonus:
    def test_selling_er_lets_its_seller_lay_on_c4(self):
        state = at_pre()
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_ER, price_point=0)
        assert TURN.step(state) == K.STEP_ER_BONUS and TURN.active_player(state) == 2 == TURN.pending_seller(state)
        assert not PRIVATES[P_ER].blocks_hex(state, HEX['C4'])
        assert A.er_candidates_py(state) == [(TILE[name], r) for name, r in ER_LAYS]
        assert legal_ids(state) == list(range(1 + len(ER_LAYS)))
        seller = PLAYERS[2].cash(state)
        apply_and_verify(state, K.ER_BONUS_CANDIDATE + ER_LAYS.index(('14', 1)))
        assert (MAP.tile(state, HEX['C4']), MAP.rotation(state, HEX['C4'])) == (TILE['14'], 1)
        assert PLAYERS[2].cash(state) == seller  # C4 costs nothing
        # Back at KO's PRE, its own lay still to come.
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE and TURN.active_player(state) == 0 and A.lay_hexes_py(state)
        assert TURN.pending_seller(state) == -1 and TURN.return_step(state) == -1

    def test_the_seller_may_pass(self):
        state = at_pre()
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_ER, price_point=0)
        apply_and_verify(state, K.ER_BONUS_PASS)
        assert MAP.tile(state, HEX['C4']) == D.NUM_SUPPLY_TILES + HEX['C4']
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE

    def test_without_a_tile_the_pass_is_forced(self):
        """Ruby still asks (``TrackLayWhenCompanySold``); the driver applies the pass."""
        state = at_pre()
        for name, _ in ER_LAYS:
            MAP.set_supply(state, TILE[name], 0)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_PRIVATE + P_ER)
        result = apply_and_verify(state, 0)
        assert result.forced == [(K.DPHASE_ER_BONUS, K.ER_BONUS_PASS)] and TURN.step(state) == K.STEP_BUY_PRIVATE_PRE

    def test_the_bonus_returns_to_the_purchase_step(self):
        state = at_track(cash=200)
        apply_and_verify(state, K.TRACK_PASS)
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST
        buy(state, K.BUY_PRIVATE_POST_PRIVATE, P_ER, price_point=0)
        apply_and_verify(state, K.ER_BONUS_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST and TURN.active_player(state) == 0


# =============================================================================
# PORT
# =============================================================================

def port_ids(state):
    """PORT's ids as (hex id, rotation)."""
    return [(D.HEX_IDS[D.MF_PORT_HEX[i // D.NUM_SIDES]], i % D.NUM_SIDES) for i in legal_ids(state)]


class TestPort:
    def test_mf_lays_the_port_tile_on_its_owners_sr_turn(self):
        """Any empty port hex, at each rotation whose exits stay on the map;
        the turn goes on, and nothing counts as bought."""
        state = new_game(3)
        end_auction(state, [0])
        hold(state, IR, {1: 2})
        enter_stock_round(state, 0)
        assert K.STOCK_MF in legal_ids(state)
        apply_and_verify(state, K.STOCK_MF)
        assert TURN.step(state) == K.STEP_PORT
        expected = [(D.HEX_IDS[h], r) for h, r in A.port_lays_py(state)]
        assert port_ids(state) == expected and ('B11', 2) in expected and ('B11', 0) not in expected
        assert all(D.TILE_EXITS[D.PORT_TILE][r] & ~D.HEX_EXIT_MASK[HEX[h]] == 0 for h, r in expected)
        apply_and_verify(state, K.PORT_LAY + D.NUM_SIDES * D.MF_PORT_HEX.index(HEX['B11']) + 2)
        assert MAP.tile(state, HEX['B11']) == D.PORT_TILE and PRIVATES[P_MF].flags(state) & 1
        assert MAP.supply(state, D.PORT_TILE) == 0 and PRIVATES[P_MF].owner(state) == K.OWNER_PLAYER  # stays open
        assert (TURN.step(state), TURN.active_player(state), TURN.sr_bought(state)) == (K.STEP_STOCK, 0, 0)
        assert K.STOCK_MF not in legal_ids(state)

    def test_mf_lays_at_its_owners_pre(self):
        """962 action 39: the port tile uses no lay of the corp's."""
        state = at_pre(PHASE_2, owners=[0])
        lays = A.lay_hexes_py(state)
        assert K.BUY_PRIVATE_PRE_MF in legal_ids(state)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_MF)
        apply_and_verify(state, K.PORT_LAY + D.NUM_SIDES * D.MF_PORT_HEX.index(HEX['G10']))
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE and K.BUY_PRIVATE_PRE_MF not in legal_ids(state)
        assert A.lay_hexes_py(state) == lays and TURN.return_step(state) == -1

    def test_another_players_mf_waits(self):
        state = at_turn(PHASE_2)  # seat 2 owns MF
        assert TURN.step(state) == K.STEP_TRACK and A.port_lays_py(state)

    def test_a_port_hex_with_a_tile_is_not_offered(self):
        state = at_pre(PHASE_2, owners=[0])
        lay(state, 'B11', '3', 2)
        lay(state, 'G10', '3', 2)
        lay(state, 'I12', '58', 1)
        assert {D.HEX_IDS[h] for h, _ in A.port_lays_py(state)} == {'J9'}
        lay(state, 'J9', '58', 0)
        assert A.port_lays_py(state) == [] and K.BUY_PRIVATE_PRE_MF not in legal_ids(state)

    def test_mf_loses_its_ability_to_a_corp(self):
        state = at_pre(owners=[0])
        buy(state, K.BUY_PRIVATE_PRE_PRIVATE, P_MF, price_point=0)
        assert A.port_lays_py(state) == [] and K.BUY_PRIVATE_PRE_MF not in legal_ids(state)


# =============================================================================
# DR
# =============================================================================

class TestDrExchange:
    def test_dr_exchanges_for_an_unparred_ir_share(self):
        """Before anyone pars IR, the share makes nobody president, and IR
        floats once two more shares follow the President's Certificate."""
        state = new_game(3)
        end_auction(state, [0])
        enter_stock_round(state, 0, turn=1)
        assert K.STOCK_DR in legal_ids(state)
        apply_and_verify(state, K.STOCK_DR)
        ir = CORPS[IR]
        assert PRIVATES[P_DR].owner(state) == K.OWNER_CLOSED and PLAYERS[0].shares(state, IR) == 1
        assert (ir.ipo_shares(state), ir.president(state), ir.par_index(state)) == (9, -1, -1)
        assert (TURN.step(state), TURN.active_player(state), TURN.sr_bought(state)) == (K.STEP_STOCK, 0, 0)
        assert K.STOCK_DR not in legal_ids(state) and A.ACTION_SELECT_SALE not in [
            A.decode_action_py(K.DPHASE_STOCK, i).type for i in legal_ids(state)]  # no price, no sale
        apply_and_verify(state, K.STOCK_PASS)
        apply_and_verify(state, K.STOCK_PAR + IR)
        apply_and_verify(state, PAR_65)
        assert ir.president(state) == 1 and ir.ipo_shares(state) == 7 and not ir.floated(state)
        apply_and_verify(state, K.STOCK_IPO + IR)
        apply_and_verify(state, K.STOCK_IPO + IR)
        assert ir.ipo_shares(state) == 5 and ir.floated(state)

    def test_the_exchange_can_take_the_presidency(self):
        state = at_or(3, owners=[0])
        hold(state, IR, {0: 2, 1: 2}, president=1)
        enter_stock_round(state, 0)
        apply_and_verify(state, K.STOCK_DR)
        assert CORPS[IR].president(state) == 0

    @pytest.mark.parametrize('cell, allowed', [(None, False), ((8, 0), True)])
    def test_the_exchange_keeps_within_sixty_percent_outside_orange(self, cell, allowed):
        state = at_or(3, owners=[0])
        hold(state, IR, {0: 6, 1: 2}, cell=cell)
        enter_stock_round(state, 0)
        assert MARKET.unlimited(state, IR) == allowed
        assert A.can_exchange_dr_py(state, 0) == allowed and (K.STOCK_DR in legal_ids(state)) == allowed

    def test_the_exchange_skips_the_certificate_limit(self):
        """One certificate goes for the one that comes (``can_gain?`` with ``exchange``)."""
        state = at_or(3, owners=[0])
        for c, n in ((AR, 6), (SR, 6), (KO, 4)):
            hold(state, c, {0: n, 1: 2})
        assert PLAYERS[0].cert_count(state) == D.CERT_LIMIT[3]
        assert not PLAYERS[0].can_gain(state, IR) and A.can_exchange_dr_py(state, 0)

    def test_dr_joins_pre_and_post_of_its_owner(self):
        state = at_pre(route=True, owners=[0])
        assert K.BUY_PRIVATE_PRE_DR in legal_ids(state)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_DONE)
        apply_and_verify(state, K.TRACK_PASS)
        assert TURN.step(state) == K.STEP_DIVIDEND
        apply_and_verify(state, K.DIVIDEND_WITHHOLD)
        apply_and_verify(state, K.BUY_TRAIN_PASS)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST and K.BUY_PRIVATE_POST_DR in legal_ids(state)
        apply_and_verify(state, K.BUY_PRIVATE_POST_DR)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_POST and PLAYERS[0].shares(state, IR) == 1

    def test_the_exchange_stays_at_pre(self):
        state = at_pre(owners=[0])
        apply_and_verify(state, K.BUY_PRIVATE_PRE_DR)
        assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE and PLAYERS[0].shares(state, IR) == 1
        assert K.BUY_PRIVATE_PRE_DR not in legal_ids(state)

    def test_an_exchange_floats_ir_for_the_next_or(self):
        """KO's president exchanges at KO's PRE, the fifth IR share out:
        IR floats with its capital, but OR 1's order stays (rules §12.10)."""
        state = at_or(3, owners=[0])
        hold(state, IR, {1: 2, 2: 2})
        hold(state, KO, {0: 6}, par=PAR_100)
        enter_operating_round(state, or_count=2)
        bank = CORPS[IR].cash(state)
        apply_and_verify(state, K.BUY_PRIVATE_PRE_DR)
        assert CORPS[IR].floated(state) and CORPS[IR].cash(state) == bank + 10 * D.PAR_PRICE[PAR_65]
        assert [TURN.or_order(state, i) for i in range(2)] == [KO, -1]
        while TURN.or_num(state) == 1:
            apply_and_verify(state, 0)
        assert [TURN.or_order(state, i) for i in range(2)] == [KO, IR]  # stock price order: ¥100 first
