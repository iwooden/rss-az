"""Emergency money, bankruptcy, and the game's end, as 6.8b builds them
(1889/engine-design.md §6.2, §9; rules §8.8, §11; Ruby's ``step/train.rb``,
``step/buy_train.rb``, ``step/emergency_money.rb``, ``step/bankrupt.rb``,
``game/base.rb``'s ``can_go_bankrupt?`` and ``game_end_check``, and
``player.rb``'s ``value``).

Covers a must-buy whose treasury is short of the cheapest depot or pool
train: the president's contribution toward that train only (the pool's when
the depot does not sell its type, the depot's on a tie), a depot buy that
changes the phase, other corps' trains for ¥1 to the treasury until a share
is sold, the president's sales (BUY_TRAIN's corp, then EMERGENCY_SELL_AMOUNT's
block: no presidency changes, the pool's room, each corp once, never more
than needed, the operating corp's own shares, the price drops, the corps yet
to run re-sorted), privates still for sale, and
bankruptcy (when it is legal, the shares it sells, the cash to the bank, the
priority deal, the game's end); and the scores. ``test_operating_round.py``
covers the game's end after a broken bank's set, and ``tests/g1889/replay/``
checks the same rules against Ruby at every fixture BuyTrain snapshot and
both fixture endings.

KO (president seat 0, par ¥100) operates first from Takamatsu (K4); J3's
tile 8 at rotation 3 gives it a route, so it must buy. The privates belong
to seat 2. KO's zero withhold moves its price left, to ¥90, before
BUY_TRAIN.
"""

import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, at_or, choose_train, depot, enter_operating_round, enter_phase, from_corp, give_train, hold,
    lay, legal_ids, pool, sale, set_cash, train_choices, train_decisions,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_65, PAR_70, PAR_75, PAR_80, PAR_100 = 0, 1, 2, 3, 5
T2, T3, T4 = D.TRAIN_2, D.TRAIN_3, D.TRAIN_4
PHASE_3, PHASE_4 = D.GAME_PHASE_3, D.GAME_PHASE_4


def private_buys(state):
    return [K.BUY_TRAIN_BUY_PRIVATE + p for p in A.purchasable_privates_py(state)]


def set_player_cash(state, player, cash):
    delta = PLAYERS[player].cash(state) - cash
    src, dst = (K.OWNER_PLAYER + player, BANK_ACCOUNT) if delta > 0 else (BANK_ACCOUNT, K.OWNER_PLAYER + player)
    BANK.transfer(state, src, dst, abs(delta))


def emergency(ko_cash, president_cash, phase=None, ko_holdings=None, setup=None):
    """KO with ``ko_cash`` and its president with ``president_cash``, at its
    first TRACK; ``setup`` adds the other corps and trains first."""
    state = at_or(3)
    if phase is not None:
        enter_phase(state, phase)
    lay(state, 'J3', '8', 3)
    hold(state, KO, ko_holdings or {0: 6}, par=PAR_100)
    if setup is not None:
        setup(state)
    set_cash(state, KO, ko_cash)
    set_player_cash(state, 0, president_cash)
    enter_operating_round(state)
    assert TURN.operating_corp(state) == KO and TURN.step(state) == K.STEP_TRACK
    return state


def to_buy_train(state):
    """Pass KO's TRACK: the token pass and the zero withhold are forced."""
    result = apply_and_verify(state, K.TRACK_PASS)
    assert TURN.step(state) == K.STEP_BUY_TRAIN and A.in_emergency_py(state)
    assert MARKET.price(state, KO) == 90
    return result


def ir_with_a_train(train=T2):
    def setup(state):
        hold(state, IR, {1: 6}, par=PAR_80)
        give_train(state, IR, train)
    return setup


# =============================================================================
# TRAINS
# =============================================================================

class TestTrains:
    def test_the_president_pays_what_the_treasury_lacks(self):
        """No pass; the cheapest train, the depot's 2, with the president's
        ¥30, or IR's 2. The purchase spends the treasury, so the pass that
        ends BUY_TRAIN is forced."""
        state = emergency(50, 420, setup=ir_with_a_train())
        to_buy_train(state)
        assert train_choices(state) == [depot(T2), from_corp(IR, T2)]
        bank = BANK.cash(state)
        result = choose_train(state, depot(T2))
        assert result.forced == [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_PASS)] and TURN.operating_corp(state) == IR
        assert TRAINS.corp_trains(state, KO, T2) == 1 and CORPS[KO].cash(state) == 0
        assert PLAYERS[0].cash(state) == 390 and BANK.cash(state) == bank + 80

    def test_a_single_choice_is_forced(self):
        state = emergency(79, 420)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced[-3:-1] == train_decisions(depot(T2)) and PLAYERS[0].cash(state) == 419

    def test_only_the_cheapest_train_the_pools_when_the_depot_does_not_sell_its_type(self):
        """Phase 3: the depot sells 3s and the pool holds 2s and a 3. Only the
        pool's 2 is offered (``min_depot_train``), and the privates are still
        for sale (Ruby's BuyCompany ignores the emergency)."""
        state = emergency(50, 420, phase=PHASE_3, setup=ir_with_a_train())
        to_buy_train(state)
        assert TRAINS.cheapest_type(state) == T2 and TRAINS.cheapest_in_pool(state)
        assert private_buys(state) and train_choices(state) == [pool(T2), from_corp(IR, T2), *private_buys(state)]
        choose_train(state, pool(T2))
        assert TRAINS.corp_trains(state, KO, T2) == 1 and PLAYERS[0].cash(state) == 390

    def test_on_a_tie_the_depots_train(self):
        """Phase 4 with every 3 held: the pool's 4 and the depot's 4 cost the
        same, and Ruby lists the depot's first."""
        def setup(state):
            hold(state, IR, {1: 6}, par=PAR_80)
            hold(state, SR, {2: 6}, par=PAR_75)
            give_train(state, IR, T3, 3)
            give_train(state, SR, T3, 2)
        state = emergency(100, 420, phase=PHASE_4, setup=setup)
        to_buy_train(state)
        assert TRAINS.pool(state, T4) == 1 and TRAINS.cheapest_type(state) == T4
        assert not TRAINS.cheapest_in_pool(state)
        assert train_choices(state) == [depot(T4), from_corp(IR, T3), from_corp(SR, T3), *private_buys(state)]

    def test_a_depot_buy_still_changes_the_phase(self):
        """Every 2 is held, so the cheapest train is the depot's first 3, which
        starts phase 3."""
        def setup(state):
            hold(state, IR, {1: 6}, par=PAR_80)
            hold(state, SR, {2: 6}, par=PAR_75)
            give_train(state, IR, T2, 4)
            give_train(state, SR, T2, 2)
        state = emergency(100, 420, setup=setup)
        to_buy_train(state)
        assert train_choices(state) == [depot(T3), from_corp(IR, T2), from_corp(SR, T2)]
        choose_train(state, depot(T3))
        assert PHASE.current(state) == PHASE_3 and TRAINS.corp_trains(state, KO, T3) == 1
        assert PLAYERS[0].cash(state) == 340 and CORPS[KO].cash(state) == 0

    def test_another_corps_train_costs_one_yen_to_the_treasury(self):
        """The president cannot cover the 2 (``spend_minmax``: ¥1 to the
        treasury; TRAIN_PRICE's grid drops the ¥80 list price)."""
        state = emergency(30, 20, setup=ir_with_a_train())
        to_buy_train(state)
        assert train_choices(state) == [from_corp(IR, T2), sale(KO, 1)]
        choose_train(state, from_corp(IR, T2))
        assert TURN.step(state) == K.STEP_TRAIN_PRICE and legal_ids(state) == [0, 2, 3, 4]
        ir = CORPS[IR].cash(state)
        apply_and_verify(state, 4)
        assert TRAINS.corp_trains(state, KO, T2) == 1 and CORPS[IR].cash(state) == ir + 30
        assert PLAYERS[0].cash(state) == 20

    def test_no_train_of_another_corp_after_a_sale(self):
        """Ruby's ``EBUY_FROM_OTHERS = :never``: once the president has sold a
        share, other corps' trains are out; and each corp sells once, so AR's
        other two shares stay, and the president goes bankrupt or KO buys a
        private."""
        def setup(state):
            ir_with_a_train(T3)(state)
            hold(state, AR, {1: 4, 0: 3}, par=PAR_65)
            hold(state, SR, {2: 4, 0: 1}, par=PAR_65)
        state = emergency(10, 0, phase=PHASE_4, ko_holdings={0: 4, 2: 4}, setup=setup)
        to_buy_train(state)
        assert TRAINS.cheapest_price(state) == 180  # the pool's 3s: the need is ¥170
        assert train_choices(state) == [from_corp(IR, T3), sale(AR, 1), sale(AR, 2), sale(AR, 3), sale(SR, 1),
                                    *private_buys(state)]
        choose_train(state, sale(SR, 1))
        assert TURN.ebuy_sold_any(state) and TURN.ebuy_sold_mask(state) == 1 << SR
        assert train_choices(state) == [sale(AR, 1), sale(AR, 2), *private_buys(state)]  # ¥105 short: not 3
        choose_train(state, sale(AR, 1))
        assert PLAYERS[0].shares(state, AR) == 2
        assert train_choices(state) == [K.BUY_TRAIN_BANKRUPT, *private_buys(state)]


# =============================================================================
# SALES
# =============================================================================

class TestSales:
    def test_no_presidency_changes_and_the_pools_room(self):
        """``corp_emergency_sale``: a president keeps the most another holds,
        and the President's Certificate; anyone stops at the pool's room. An
        SR sale may pass the presidency (``corp_max_sale``)."""
        state = at_or(3)
        hold(state, AR, {0: 5, 1: 3}, par=PAR_65)
        hold(state, IR, {0: 4, 1: 1}, par=PAR_65)
        hold(state, SR, {0: 2, 1: 1, 2: 2}, par=PAR_65, president=0)
        hold(state, KO, {1: 4, 0: 3}, par=PAR_65, pool=3)
        assert [CORPS[c].emergency_sale(state, 0) for c in (AR, IR, SR, KO)] == [2, 2, 0, 2]
        assert [CORPS[c].max_sale(state, 0) for c in (AR, IR, SR, KO)] == [5, 2, 2, 2]
        assert CORPS[UR].emergency_sale(state, 0) == 0  # unparred

    def test_never_more_than_needed(self):
        """¥70 short with three AR shares at ¥65: one or two sell, as the next
        smaller block would still fall short; two drop AR two rows. AR is
        BUY_TRAIN's one choice, so the driver picks it, and
        EMERGENCY_SELL_AMOUNT waits for the block."""
        def setup(state):
            hold(state, AR, {1: 4, 0: 3}, par=PAR_65)
        state = emergency(0, 10, ko_holdings={0: 4, 2: 4}, setup=setup)
        result = apply_and_verify(state, K.TRACK_PASS)
        assert result.forced[-1] == (K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_SELL + AR)
        assert TURN.step(state) == K.STEP_EMERGENCY_SELL_AMOUNT and TURN.pending_corp(state) == AR
        assert legal_ids(state) == [0, 1]  # one or two shares
        result = apply_and_verify(state, 1)
        assert result.forced[:2] == train_decisions(depot(T2))  # the one choice left
        assert TURN.pending_corp(state) == -1
        assert CORPS[AR].pool_shares(state) == 2 and MARKET.price(state, AR) == 55
        assert PLAYERS[0].cash(state) == 10 + 130 - 80 and TRAINS.corp_trains(state, KO, T2) == 1

    def test_the_operating_corps_own_shares(self):
        state = emergency(5, 10, ko_holdings={0: 6, 1: 2}, setup=ir_with_a_train())
        to_buy_train(state)
        assert CORPS[KO].emergency_sale(state, 0) == 4 and train_choices(state) == [from_corp(IR, T2), sale(KO, 1)]
        choose_train(state, sale(KO, 1))
        assert CORPS[KO].president(state) == 0 and TRAINS.corp_trains(state, KO, T2) == 1
        assert MARKET.price(state, KO) == 80

    def test_a_sale_re_sorts_the_corps_yet_to_run(self):
        """``recalculate_order``: IR's ¥80 drops to ¥75 at the foot of SR's
        stack, so SR runs before it; the turn's end clears the sales."""
        def setup(state):
            hold(state, IR, {1: 4, 0: 2}, par=PAR_80)
            hold(state, SR, {2: 6}, par=PAR_75)
            hold(state, AR, {1: 4, 0: 1}, par=PAR_65)
        state = emergency(0, 0, ko_holdings={0: 4, 2: 4}, setup=setup)
        assert [TURN.or_order(state, i) for i in range(4)] == [KO, IR, SR, AR]
        to_buy_train(state)
        assert train_choices(state) == [sale(AR, 1), sale(IR, 1)]
        choose_train(state, sale(IR, 1))
        assert [TURN.or_order(state, i) for i in range(4)] == [KO, SR, IR, AR] and TURN.operating_corp(state) == SR
        assert TURN.ebuy_sold_mask(state) == 0 and not TURN.ebuy_sold_any(state)
        assert PLAYERS[0].cash(state) == 0 and TRAINS.corp_trains(state, KO, T2) == 1


# =============================================================================
# BANKRUPTCY
# =============================================================================

def bankrupt_setup(state):
    """Seat 0 can sell one AR share (¥65); its KO shares keep it president
    beside seat 2's four, its SR share has no room in the pool, and its IR
    shares are the President's Certificate. AR has a 2."""
    hold(state, AR, {1: 4, 0: 1}, par=PAR_65)
    hold(state, SR, {1: 4, 0: 1}, par=PAR_65, pool=5)
    hold(state, IR, {0: 2, 1: 2, 2: 1}, par=PAR_70)
    give_train(state, AR, T2)


class TestBankruptcy:
    def test_bankruptcy_when_everything_falls_short(self):
        """¥70 short, with ¥65 to sell (``can_go_bankrupt?``); an empty
        treasury buys no train of another corp. ¥65 short, the sale covers it."""
        state = emergency(0, 10, ko_holdings={0: 4, 2: 4}, setup=bankrupt_setup)
        to_buy_train(state)
        assert A.can_go_bankrupt_py(state) and train_choices(state) == [sale(AR, 1), K.BUY_TRAIN_BANKRUPT]
        state = emergency(5, 10, ko_holdings={0: 4, 2: 4}, setup=bankrupt_setup)
        to_buy_train(state)
        assert not A.can_go_bankrupt_py(state) and train_choices(state) == [from_corp(AR, T2), sale(AR, 1)]

    def test_bankruptcy_sells_what_it_can_and_ends_the_game(self):
        """``process_bankrupt``: AR's share is sold, the cash goes to the
        bank, the corps yet to run re-sort (AR falls below SR), and the game
        ends at once. Seat 0 keeps what could not be sold, which scores."""
        state = emergency(0, 10, ko_holdings={0: 4, 2: 4}, setup=bankrupt_setup)
        assert [TURN.or_order(state, i) for i in range(4)] == [KO, IR, AR, SR] and TURN.priority_player(state) == 0
        to_buy_train(state)
        bank = BANK.cash(state)
        apply_and_verify(state, K.BUY_TRAIN_BANKRUPT, expected_status=STATUS_GAME_OVER)
        assert A.is_terminal_py(state) and TURN.end_reason(state) == K.END_REASON_BANKRUPT
        assert PLAYERS[0].bankrupt(state) and PLAYERS[0].cash(state) == 0 and BANK.cash(state) == bank + 10
        assert [PLAYERS[0].shares(state, c) for c in (KO, AR, SR, IR)] == [4, 0, 1, 2]
        assert MARKET.price(state, AR) == 60 and [TURN.or_order(state, i) for i in range(4)] == [KO, IR, SR, AR]
        assert TURN.priority_player(state) == 1  # Ruby's priority deal skips a bankrupt player
        assert TURN.operating_corp(state) == KO and TURN.ebuy_sold_mask(state) == 0  # Ruby keeps its round
        assert A.final_scores_py(state)[0] == 4 * 90 + 65 + 2 * 70
        assert DRIVER.apply_action(state, K.BUY_TRAIN_PASS) == STATUS_GAME_OVER

    def test_a_sale_can_leave_bankruptcy_the_one_choice(self):
        """Sold first, AR stays sold, and ¥5 short with nothing left to sell the
        president's bankruptcy is forced."""
        state = emergency(0, 10, ko_holdings={0: 4, 2: 4}, setup=bankrupt_setup)
        to_buy_train(state)
        result = choose_train(state, sale(AR, 1), expected_status=STATUS_GAME_OVER)
        assert result.forced == [(K.DPHASE_BUY_TRAIN, K.BUY_TRAIN_BANKRUPT)]
        assert TURN.ebuy_sold_mask(state) == 1 << AR and TURN.ebuy_sold_any(state)
        assert PLAYERS[0].bankrupt(state) and PLAYERS[0].shares(state, AR) == 0


# =============================================================================
# SCORES
# =============================================================================

class TestScores:
    @pytest.mark.parametrize('num_players', [2, 3, 4, 5, 6])
    def test_a_new_game_scores_cash_and_privates_at_face(self, num_players):
        state = at_or(num_players, owners=[0])
        faces = sum(D.PRIVATE_VALUE[p] for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_PLAYER)
        assert A.final_scores_py(state) == [D.START_CASH[num_players] + (faces if i == 0 else 0)
                                            for i in range(num_players)]
        assert not A.is_terminal_py(state)

    def test_shares_count_at_market_the_certificate_twice_unparred_shares_nothing(self):
        state = at_or(3)
        hold(state, KO, {0: 6, 1: 2}, par=PAR_100)
        MARKET.move_down(state, KO)
        PLAYERS[1].set_shares(state, IR, 1)  # DR's exchange before IR is parred
        CORPS[IR].set_ipo_shares(state, 9)
        cash = [PLAYERS[i].cash(state) for i in range(3)]
        faces = sum(D.PRIVATE_VALUE[p] for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_PLAYER + 2)
        assert A.final_scores_py(state) == [cash[0] + 6 * 90, cash[1] + 2 * 90, cash[2] + faces]
