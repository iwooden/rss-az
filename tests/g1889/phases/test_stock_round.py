"""The stock round: STOCK, PAR, SELL_AMOUNT, and SR_END (1889/engine-design.md
§5.2-5.3, §6.1; rules §6; Ruby's ``step/buy_sell_par_shares.rb`` and
``round/stock.rb``).

Covers the turn (pass and done, one purchase, sell/buy order, blocks: STOCK
picks the corp and SELL_AMOUNT the block, forced when it is the only one), the
first SR's ban on sales, no buying back, the pool cap, the President's
Certificate, presidency changes both ways, floats, the certificate limit at
every player count, the 60% limit, the yellow and orange zones, must-sell
turns, players with nothing to do, PAR, market edges and stacking, and
SR_END's order and priority. It also covers the driver's forced rule for
amounts. ``tests/g1889/replay/test_stock_scripts.py`` and the fixtures' SRs
check the same paths against Ruby.
"""

import pytest

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.market import MARKET
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN

from tests.g1889.phases.conftest import (
    apply_and_verify, apply_factored, end_auction, enter_auction, enter_stock_round, exact_id, hold, legal_ids,
    new_game, opened, policy_ids,
)

AR, IR, SR, KO, TR, KU, UR = range(D.NUM_CORPS)
PAR_65, PAR_70, PAR_100 = 0, 1, 5
PASS = K.STOCK_PASS


def ipo(c):
    return K.STOCK_IPO + c


def pool(c):
    return K.STOCK_POOL + c


def par(c):
    return K.STOCK_PAR + c


def sell(state, c, n):
    """Sell ``n`` shares of ``c``: STOCK's corp, then SELL_AMOUNT's block."""
    return apply_factored(state, K.STOCK_SELL + c, K.DPHASE_SELL_AMOUNT, n - 1, f'sell {n} of corp {c}')


def sales(state, c):
    """The block sizes of ``c`` the turn player may sell: SELL_AMOUNT's, if STOCK offers ``c``."""
    if K.STOCK_SELL + c not in legal_ids(state):
        return []
    return [n + 1 for n in legal_ids(opened(state, K.STOCK_SELL + c))]


def shares(state, c, n=None):
    n = n or TURN.num_players(state)
    return [PLAYERS[i].shares(state, c) for i in range(n)]


def cash(state):
    return [PLAYERS[i].cash(state) for i in range(TURN.num_players(state))]


def passed(state):
    return [PLAYERS[i].passed(state) for i in range(TURN.num_players(state))]


def at_sr(num_players=3, owners=()):
    """A game past the auction: the privates in play went free to ``owners``
    (the last seat by default). Holdings and ``enter_stock_round`` follow."""
    state = new_game(num_players)
    end_auction(state, owners)
    return state


def first_sr(num_players=3, first=0):
    state = at_sr(num_players)
    enter_stock_round(state, first, turn=1)
    return state


# =============================================================================
# TURNS
# =============================================================================

class TestTurns:
    def test_a_pass_sets_the_flag_and_done_after_acting_clears_it(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 1: 1})
        enter_stock_round(state, 0)
        apply_and_verify(state, PASS)
        assert passed(state) == [True, False, False] and TURN.active_player(state) == 1
        apply_and_verify(state, PASS)
        apply_and_verify(state, ipo(IR))  # player 2 buys; it may still sell, so the turn goes on
        assert TURN.active_player(state) == 2 and TURN.sr_bought(state)
        apply_and_verify(state, PASS)  # done
        assert passed(state) == [True, True, False] and TURN.active_player(state) == 0

    def test_the_round_ends_when_everyone_passes_in_succession(self):
        state = first_sr(3)
        apply_and_verify(state, PASS)  # player 0
        apply_and_verify(state, par(KO))  # player 1, then PAR
        result = apply_and_verify(state, PAR_65)
        # The first SR sells nothing, so the purchase leaves only "done", which the driver applies.
        assert result.forced == [(K.DPHASE_STOCK, PASS)] and TURN.active_player(state) == 2
        apply_and_verify(state, PASS)  # player 2
        apply_and_verify(state, PASS)  # player 0 again: player 1 acted since, so the SR goes on
        assert TURN.round(state) == K.ROUND_STOCK and passed(state) == [True, False, True]
        result = apply_and_verify(state, PASS)  # player 1 completes the run
        # KO has not floated, so the OR has nobody to operate, and SR 2 follows.
        assert result.steps == [K.STEP_SR_END, K.STEP_OR_START, K.STEP_OR_END]
        assert (TURN.round(state), TURN.step(state), TURN.turn(state)) == (K.ROUND_STOCK, K.STEP_STOCK, 2)

    def test_priority_goes_to_the_player_after_the_last_to_act(self):
        state = first_sr(3)
        assert TURN.priority_player(state) == 0
        apply_and_verify(state, PASS)
        assert TURN.priority_player(state) == 0  # a pass is no action
        apply_and_verify(state, par(KO))
        apply_and_verify(state, PAR_65)
        assert TURN.priority_player(state) == 2
        for _ in range(3):
            apply_and_verify(state, PASS)
        # The OR is empty (KO has not floated), and player 2 opens SR 2.
        assert (TURN.turn(state), TURN.priority_player(state), TURN.active_player(state)) == (2, 2, 2)

    def test_one_purchase_per_turn(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 1: 1})
        hold(state, KO, {1: 2})
        enter_stock_round(state, 0)
        assert {ipo(IR), ipo(KO), par(AR)} <= set(legal_ids(state))
        apply_and_verify(state, ipo(KO))
        # Sales stay open, but IR's president cannot sell: nobody else holds two shares.
        assert legal_ids(state) == [PASS, K.STOCK_SELL + KO] and sales(state, KO) == [1]

    def test_selling_then_buying_ends_the_turns_sales(self):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 2})
        hold(state, KO, {0: 3, 1: 2})
        enter_stock_round(state, 0)
        sell(state, IR, 1)
        assert sales(state, IR) == [] and sales(state, KO) == [1, 2, 3]  # one block per corp per turn
        assert ipo(IR) not in legal_ids(state) and ipo(KO) in legal_ids(state)
        result = apply_and_verify(state, ipo(KO))
        # Nothing is left: no second purchase and no sale after a purchase that followed one.
        assert result.forced == [(K.DPHASE_STOCK, PASS)] and TURN.active_player(state) == 1

    def test_buying_then_selling_keeps_selling_open(self):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 2})
        hold(state, KO, {0: 3, 1: 2})
        enter_stock_round(state, 0)
        apply_and_verify(state, ipo(IR))
        assert sales(state, IR) == [1, 2, 3, 4, 5] and sales(state, KO) == [1, 2, 3]
        sell(state, IR, 2)
        assert sales(state, KO) == [1, 2, 3] and not TURN.sr_sold_then_bought(state)
        result = sell(state, KO, 1)  # buy, sell, sell: both corps are sold for the turn
        assert result.forced == [(K.DPHASE_STOCK, PASS)] and TURN.active_player(state) == 1

    def test_sold_shares_are_not_bought_back_this_round(self):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 2})
        enter_stock_round(state, 0)
        sell(state, IR, 1)
        apply_and_verify(state, PASS)
        assert pool(IR) in legal_ids(state)  # player 1 may buy it
        apply_and_verify(state, PASS)
        apply_and_verify(state, PASS)
        assert TURN.active_player(state) == 0
        assert ipo(IR) not in legal_ids(state) and pool(IR) not in legal_ids(state)
        assert PLAYERS[0].sold_mask(state) == 1 << IR

    def test_nothing_is_sold_in_the_first_round(self):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 2})
        enter_stock_round(state, 0, turn=1)
        assert sales(state, IR) == []
        TURN.set_turn(state, 2)
        assert sales(state, IR) == [1, 2, 3, 4]

    def test_a_player_with_nothing_to_do_is_passed_by_the_driver(self):
        """Ruby's SR skips such a player; the engine offers only the pass."""
        state = first_sr(3)
        BANK.transfer(state, 1, BANK_ACCOUNT, PLAYERS[1].cash(state) - 60)  # under any par or share
        result = apply_and_verify(state, PASS)
        assert result.forced == [(K.DPHASE_STOCK, PASS)]
        assert TURN.active_player(state) == 2 and passed(state) == [True, True, False]

    @pytest.mark.parametrize('private, slot', [(D.PRIVATE_DR, K.STOCK_DR), (D.PRIVATE_MF, K.STOCK_MF)])
    def test_a_broke_owner_of_dr_or_mf_still_chooses(self, private, slot):
        """Ruby skips a player with nothing to buy or sell, but its DR
        exchanges at any time (``when: 'any'``) and MF lays at any moment of an
        SR, so the engine offers the private on the owner's own turn, even with
        nothing else (§12 item 4). Either choice ends the turn: after the
        exchange or the lay only the pass is left, which the driver applies."""
        def broke_owner():
            """SR 1 after player 0's pass: player 1 has ¥0 and only ``private``."""
            state = new_game(3)
            end_auction(state, [1 if p == private else 2 for p in range(D.NUM_PRIVATES - 1)])  # no UTF at 3
            BANK.transfer(state, 1, BANK_ACCOUNT, PLAYERS[1].cash(state))
            enter_stock_round(state, 0, turn=1)
            apply_and_verify(state, PASS)
            assert TURN.active_player(state) == 1 and PRIVATES[private].owner(state) == K.OWNER_PLAYER + 1
            assert legal_ids(state) == policy_ids(state) == [PASS, slot]
            return state

        state = broke_owner()
        apply_and_verify(state, PASS)
        assert TURN.active_player(state) == 2 and passed(state) == [True, True, False]
        assert PRIVATES[private].owner(state) == K.OWNER_PLAYER + 1 and not PRIVATES[private].flags(state)

        state = broke_owner()
        result = apply_and_verify(state, slot)
        if private == D.PRIVATE_MF:
            assert TURN.step(state) == K.STEP_PORT
            result = apply_and_verify(state, legal_ids(state)[0])
            assert PRIVATES[private].flags(state) and MAP.supply(state, D.PORT_TILE) == 0
        else:
            assert PRIVATES[private].owner(state) == K.OWNER_CLOSED and PLAYERS[1].shares(state, IR) == 1
        assert result.forced == [(K.DPHASE_STOCK, PASS)]
        assert TURN.active_player(state) == 2 and passed(state) == [True, True, False]

    def test_the_round_ends_by_itself_when_nobody_can_act(self):
        state = at_sr(2)
        hold(state, IR, {0: 6})  # floated, so the OR that follows has a turn to rest in
        for i in range(2):
            BANK.transfer(state, i, BANK_ACCOUNT, PLAYERS[i].cash(state))
        enter_stock_round(state, 1, turn=1)
        result = apply_and_verify(state, PASS)
        assert [entry[1:] for entry in result.history] == [
            (K.DPHASE_STOCK, PASS), (K.DPHASE_STOCK, PASS), (-1, K.STEP_SR_END), (-1, K.STEP_OR_START),
            (-1, K.STEP_TURN_START)]
        assert TURN.step(state) == K.STEP_TRACK and TURN.priority_player(state) == 1

    def test_dr_and_mf_join_their_owners_turn(self):
        """``test_privates.py`` covers the exchange and the port tile."""
        state = at_sr(3, owners=[0])  # player 0 owns DR and MF
        hold(state, IR, {1: 2})
        enter_stock_round(state, 0)
        assert K.STOCK_DR in legal_ids(state) and K.STOCK_MF in legal_ids(state)
        apply_and_verify(state, PASS)
        assert K.STOCK_DR not in legal_ids(state) and K.STOCK_MF not in legal_ids(state)


# =============================================================================
# PAR
# =============================================================================

class TestPar:
    def test_par_prices_follow_cash(self):
        state = first_sr(3)
        BANK.transfer(state, 0, BANK_ACCOUNT, PLAYERS[0].cash(state) - 160)
        apply_and_verify(state, par(UR))
        assert legal_ids(state) == [0, 1, 2, 3]  # ¥65-80: twice the par fits ¥160
        apply_and_verify(state, PAR_70)
        corp = CORPS[UR]
        assert (corp.par_price(state), corp.president(state), PLAYERS[0].shares(state, UR)) == (70, 0, 2)
        assert (MARKET.row(state, UR), MARKET.col(state, UR)) == (D.PAR_ROW[1], D.PAR_COL[1])
        assert PLAYERS[0].cash(state) == 20 and corp.ipo_shares(state) == 8 and not corp.floated(state)

    def test_one_affordable_par_is_applied_by_the_driver(self):
        state = first_sr(3)
        BANK.transfer(state, 0, BANK_ACCOUNT, PLAYERS[0].cash(state) - 135)
        result = apply_and_verify(state, par(SR))
        assert result.forced[0] == (K.DPHASE_PAR, PAR_65) and CORPS[SR].par_price(state) == 65

    def test_par_needs_the_lowest_certificate(self):
        state = first_sr(3)
        BANK.transfer(state, 0, BANK_ACCOUNT, PLAYERS[0].cash(state) - 129)
        assert legal_ids(state) == [PASS]

    def test_a_par_goes_to_the_bottom_of_its_cell(self):
        state = first_sr(3)
        apply_and_verify(state, par(KO))
        apply_and_verify(state, PAR_65)
        apply_and_verify(state, par(IR))
        apply_and_verify(state, PAR_65)
        assert MARKET.sort_by_price(state, 0x7F) == [KO, IR]


# =============================================================================
# SALES, THE POOL, AND THE PRESIDENT'S CERTIFICATE
# =============================================================================

class TestSales:
    def test_a_sale_pays_the_price_then_drops_a_row_per_share(self):
        state = at_sr(3)
        hold(state, KO, {0: 4, 1: 2}, par=PAR_70)  # ¥70 at (4, 3)
        enter_stock_round(state, 0)
        before = cash(state)
        sell(state, KO, 2)
        assert cash(state)[0] - before[0] == 140 and CORPS[KO].pool_shares(state) == 2
        assert (MARKET.row(state, KO), MARKET.price(state, KO)) == (6, 60)

    @pytest.mark.parametrize('pool_shares, blocks', [(0, [1, 2, 3]), (3, [1, 2]), (4, [1]), (5, [])])
    def test_the_pool_holds_at_most_half(self, pool_shares, blocks):
        state = at_sr(3)
        hold(state, KO, {0: 3, 2: 2}, pool=pool_shares)
        enter_stock_round(state, 0)
        assert sales(state, KO) == blocks

    def test_a_sale_of_one_possible_block_is_forced(self):
        """STOCK picks the corp and SELL_AMOUNT the block; with room for one
        share in the pool, the driver sells it."""
        state = at_sr(3)
        hold(state, KO, {0: 3, 2: 2}, pool=4)
        enter_stock_round(state, 0)
        result = apply_and_verify(state, K.STOCK_SELL + KO)
        assert result.decisions[:2] == [(K.DPHASE_STOCK, K.STOCK_SELL + KO), (K.DPHASE_SELL_AMOUNT, 0)]
        assert CORPS[KO].pool_shares(state) == 5 and TURN.pending_corp(state) == -1
        assert TURN.sr_sold_mask(state) == PLAYERS[0].sold_mask(state) == 1 << KO

    @pytest.mark.parametrize('other, blocks', [(1, [1]), (2, [1, 2, 3])])
    def test_the_presidents_certificate_needs_a_successor(self, other, blocks):
        state = at_sr(3)
        hold(state, IR, {0: 3, 1: other})
        enter_stock_round(state, 0)
        assert sales(state, IR) == blocks  # without a two-share holder, the certificate stays

    @pytest.mark.parametrize('count, president', [(1, 0), (2, 1)])
    def test_a_president_selling_below_another_hands_over_the_corp(self, count, president):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 3})
        enter_stock_round(state, 0)
        sell(state, IR, count)  # one share leaves a tie, which keeps the presidency
        # Holdings stay as they are: Ruby swaps certificates, not shares.
        assert CORPS[IR].president(state) == president and shares(state, IR) == [4 - count, 3, 0]

    def test_selling_into_the_certificate_passes_it_clockwise(self):
        state = at_sr(4)
        hold(state, IR, {1: 3, 0: 2, 3: 2}, president=1)
        enter_stock_round(state, 1)
        sell(state, IR, 2)  # seats 3 and 0 tie at two: seat 3 is nearer clockwise from 1
        assert CORPS[IR].president(state) == 3 and shares(state, IR) == [2, 1, 0, 2]
        assert CORPS[IR].pool_shares(state) == 2

    def test_selling_the_whole_certificate(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 2: 2})
        enter_stock_round(state, 0)
        sell(state, IR, 2)
        assert CORPS[IR].president(state) == 2 and shares(state, IR) == [0, 0, 2]


class TestPurchases:
    def test_ipo_shares_cost_par_and_pool_shares_the_market(self):
        state = at_sr(3)
        hold(state, KO, {1: 2}, par=PAR_70, cell=(6, 3), pool=1)  # par ¥70, market ¥60
        enter_stock_round(state, 0)
        bank = BANK.cash(state)
        apply_and_verify(state, pool(KO))
        apply_and_verify(state, PASS)
        apply_and_verify(state, ipo(KO))
        assert cash(state)[:2] == [420 - 60, 420 - 70] and BANK.cash(state) == bank + 130
        assert CORPS[KO].cash(state) == 0  # every payment goes to the bank

    def test_a_purchase_beyond_the_president_takes_the_corp(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 1: 2})
        enter_stock_round(state, 1, turn=1)
        apply_and_verify(state, ipo(IR))
        assert CORPS[IR].president(state) == 1 and shares(state, IR) == [2, 3, 0]

    def test_a_purchase_that_only_ties_changes_nothing(self):
        state = at_sr(3)
        hold(state, IR, {0: 3, 1: 2})
        enter_stock_round(state, 1, turn=1)
        apply_and_verify(state, ipo(IR))
        assert CORPS[IR].president(state) == 0

    def test_a_corp_floats_when_half_its_shares_are_out(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 1: 2}, par=PAR_100)
        enter_stock_round(state, 2, turn=1)
        assert not CORPS[IR].floated(state)
        apply_and_verify(state, ipo(IR))
        assert CORPS[IR].floated(state) and CORPS[IR].cash(state) == 1000

    def test_pool_shares_do_not_float_a_corp(self):
        state = at_sr(3)
        hold(state, IR, {0: 2, 1: 1}, pool=2)  # IPO 5 counts, the pool does not
        assert CORPS[IR].floated(state)
        state = at_sr(3)
        hold(state, IR, {0: 2}, pool=2)
        enter_stock_round(state, 1)
        apply_and_verify(state, pool(IR))
        assert not CORPS[IR].floated(state)


# =============================================================================
# LIMITS AND ZONES
# =============================================================================

def _at_cert_limit(num_players, extra=0):
    """Player 0 owns every private and presides over corps up to the
    certificate limit plus ``extra``; UR is open for purchase."""
    state = at_sr(num_players, owners=[0])
    privates = sum(PRIVATES[p].owner(state) == K.OWNER_PLAYER for p in range(D.NUM_PRIVATES))
    left = D.CERT_LIMIT[num_players] + extra - privates
    for c in (AR, IR, SR, KO, TR):
        if left <= 0:
            break
        certs = min(left, 5)
        hold(state, c, {0: certs + 1, 1: 1})
        left -= certs
    assert left == 0
    hold(state, UR, {1: 2})
    return state


@pytest.mark.parametrize('num_players', range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
class TestCertificateLimit:
    def test_no_purchase_at_the_limit(self, num_players):
        state = _at_cert_limit(num_players)
        enter_stock_round(state, 0)
        assert PLAYERS[0].cert_count(state) == D.CERT_LIMIT[num_players]
        assert not {ipo(UR), par(KU)} & set(legal_ids(state))

    def test_purchases_below_the_limit(self, num_players):
        state = _at_cert_limit(num_players, extra=-1)
        enter_stock_round(state, 0)
        assert {ipo(UR), par(KU)} <= set(legal_ids(state))

    def test_over_the_limit_a_turn_offers_only_sales(self, num_players):
        state = _at_cert_limit(num_players, extra=1)
        hold(state, KU, {0: 3, 1: 2}, cell=(7, 1))  # yellow: uncounted, and keeps the turn open
        enter_stock_round(state, 0)
        legal = legal_ids(state)
        assert legal and PASS not in legal and all(i >= K.STOCK_SELL for i in legal)
        sell(state, AR, 1)  # back to the limit: the turn opens up
        legal = legal_ids(state)
        assert legal[0] == PASS and sales(state, AR) == [] and sales(state, KU) == [1, 2, 3]
        assert ipo(KU) in legal and ipo(UR) not in legal  # only uncounted shares may be bought


class TestZones:
    def test_yellow_shares_skip_the_limit(self):
        state = _at_cert_limit(3)
        hold(state, UR, {1: 2}, cell=(7, 1))  # ¥45, yellow
        enter_stock_round(state, 0)
        assert ipo(UR) in legal_ids(state)
        apply_and_verify(state, ipo(UR))
        assert PLAYERS[0].cert_count(state) == D.CERT_LIMIT[3]

    def test_orange_shares_skip_the_sixty_percent_limit(self):
        state = at_sr(3)
        hold(state, UR, {0: 6, 1: 2}, cell=(9, 1))  # ¥30, orange
        enter_stock_round(state, 0)
        assert ipo(UR) in legal_ids(state)
        hold(state, UR, {0: 6, 1: 2}, cell=(7, 1))  # yellow keeps the 60% limit
        assert ipo(UR) not in legal_ids(state)

    def test_above_sixty_percent_outside_orange_must_sell(self):
        state = at_sr(3)
        hold(state, UR, {0: 7, 1: 2}, cell=(7, 1))
        enter_stock_round(state, 0)
        assert legal_ids(state) == [K.STOCK_SELL + UR] and sales(state, UR) == [1, 2, 3, 4, 5]
        sell(state, UR, 1)
        assert legal_ids(state)[0] == PASS

    def test_a_must_sell_turn_offers_dr_and_mf(self):
        """DR's exchange skips the certificate limit (``can_gain?`` with
        ``exchange``, ``share_buying.rb:49``) and MF's lay buys nothing, so their
        owner may use them before selling."""
        state = at_sr(3, owners=[0])
        hold(state, UR, {0: 7, 1: 2}, cell=(7, 1))
        enter_stock_round(state, 0)
        assert legal_ids(state) == [K.STOCK_SELL + UR, K.STOCK_DR, K.STOCK_MF]
        apply_and_verify(state, K.STOCK_DR)
        assert legal_ids(state) == [K.STOCK_SELL + UR, K.STOCK_MF]  # still 70% of UR

    def test_must_sell_needs_a_sale_of_the_corp_over_sixty_percent(self):
        """After one UR block the player still holds 70%, but UR cannot be sold
        again this turn, so the turn may end (``must_sell_corporations``)."""
        state = at_sr(3)
        hold(state, UR, {0: 8, 1: 2}, cell=(7, 1))
        hold(state, KO, {0: 3, 1: 2})
        enter_stock_round(state, 0)
        assert PASS not in legal_ids(state)
        sell(state, UR, 1)
        assert PLAYERS[0].shares(state, UR) == 7 and legal_ids(state)[0] == PASS
        assert sales(state, KO) == [1, 2, 3]

    def test_a_sale_into_the_yellow_zone_can_end_a_must_sell_turn(self):
        """KO's three certificates put the player over the limit; selling one KO
        share moves KO into the yellow zone, so the other two stop counting."""
        state = _at_cert_limit(3)
        hold(state, KO, {0: 4, 1: 1}, cell=(7, 3))  # ¥55; (8, 3) is yellow
        enter_stock_round(state, 0)
        assert PLAYERS[0].cert_count(state) == D.CERT_LIMIT[3] + 3 and PASS not in legal_ids(state)
        sell(state, KO, 1)
        assert MARKET.type(state, KO) == D.MARKET_NO_CERT_LIMIT
        assert PLAYERS[0].cert_count(state) == D.CERT_LIMIT[3] and legal_ids(state)[0] == PASS


# =============================================================================
# SR_END
# =============================================================================

def _end_round(state):
    """Pass for everyone from the turn player on; return the state at the SR's
    end, before the OR that follows begins."""
    result = apply_and_verify(state, PASS)
    while K.STEP_SR_END not in result.steps:
        result = apply_and_verify(state, PASS)
    return result.state_before(K.STEP_OR_START)


class TestRoundEnd:
    def test_sold_out_corps_move_up_in_price_order_beneath_the_cell(self):
        state = at_sr(4)
        hold(state, AR, {0: 2}, par=PAR_70)  # (4, 3), not sold out
        hold(state, KO, {0: 4, 1: 3, 2: 3})  # (5, 3), on top
        hold(state, IR, {1: 4, 2: 3, 3: 3})  # (5, 3), beneath KO
        enter_stock_round(state, 0)
        state = _end_round(state)
        assert [(MARKET.row(state, c), MARKET.col(state, c)) for c in (AR, KO, IR)] == [(4, 3)] * 3
        assert MARKET.sort_by_price(state, 0x7F) == [AR, KO, IR]

    def test_the_stack_order_carries_over(self):
        state = at_sr(4)
        hold(state, IR, {1: 4, 2: 3, 3: 3})
        hold(state, KO, {0: 4, 1: 3, 2: 3})
        enter_stock_round(state, 0)
        state = _end_round(state)
        assert MARKET.sort_by_price(state, 0x7F) == [IR, KO]

    def test_the_top_row_and_unsold_corps_stay(self):
        state = at_sr(4, owners=[1, 2, 3])
        hold(state, KO, {0: 4, 1: 3, 2: 3}, cell=(0, 5))
        hold(state, IR, {0: 4, 1: 3, 2: 2})  # one IPO share left
        hold(state, AR, {0: 4, 1: 3, 2: 2}, pool=1)  # one in the pool
        enter_stock_round(state, 0)
        stamps = [MARKET.stamp(state, c) for c in (KO, IR, AR)]
        state = _end_round(state)
        assert [MARKET.stamp(state, c) for c in (KO, IR, AR)] == stamps

    def test_priority_ors_and_clearing(self):
        state = at_sr(3)
        hold(state, IR, {0: 4, 1: 2})
        PHASE.set_current(state, D.GAME_PHASE_3)
        enter_stock_round(state, 1)
        apply_and_verify(state, PASS)  # 1
        apply_and_verify(state, PASS)  # 2
        sell(state, IR, 1)  # 0 sells, then is done
        apply_and_verify(state, PASS)
        apply_and_verify(state, PASS)  # 1
        apply_and_verify(state, PASS)  # 2
        assert TURN.round(state) == K.ROUND_STOCK
        apply_and_verify(state, PASS)  # 0: the last passer
        assert TURN.round(state) == K.ROUND_OPERATING and TURN.priority_player(state) == 1
        assert TURN.or_count(state) == D.PHASE_ORS[D.GAME_PHASE_3] == 2 and TURN.or_num(state) == 1
        assert passed(state) == [False] * 3 and PLAYERS[0].sold_mask(state) == 0


# =============================================================================
# THE DRIVER
# =============================================================================

class TestDriver:
    def test_a_narrow_off_grid_bid_range_is_not_forced(self):
        """A human's off-grid ¥373 on SIR leaves ¥47 for ER: one grid point
        (¥45), and the exact ¥46 and ¥47 above it, so BID waits."""
        state = new_game(3)
        enter_auction(state, 0, bids=[(0, D.PRIVATE_SIR, 373)])
        apply_and_verify(state, K.AUCTION_BID + D.PRIVATE_ER)
        assert TURN.step(state) == K.STEP_BID and policy_ids(state) == legal_ids(state) == [K.BID_POINT]
        assert DRIVER.forced_action_py(state) == -1
        assert A.is_legal_action_py(state, exact_id(state, 47)) and not A.is_legal_action_py(state, exact_id(state, 48))
        apply_and_verify(state, exact_id(state, 47))
        assert PLAYERS[0].bid(state, D.PRIVATE_ER) == 47

    def test_a_one_value_bid_range_is_forced(self):
        state = new_game(3)
        enter_auction(state, 0, bids=[(0, D.PRIVATE_SIR, 375)])
        result = apply_and_verify(state, K.AUCTION_BID + D.PRIVATE_ER)
        assert result.forced == [(K.DPHASE_BID, K.BID_POINT)]

    def test_a_capped_bid_with_one_policy_point_is_forced(self):
        """The pass cap keeps ¥45 alone (¥47 less A's ¥20 leaves room for ¥2
        more); the legal grid has five points, and self-play never bids exact
        amounts, so the driver applies the point."""
        state = new_game(3, auction_pass_payout_cap=1)
        enter_auction(state, 0, bids=[(0, D.PRIVATE_SIR, 353)])
        TURN.set_auction_pass_payouts(state, 1)
        result = apply_and_verify(state, K.AUCTION_BID + D.PRIVATE_ER)
        assert result.forced == [(K.DPHASE_BID, K.BID_POINT)] and PLAYERS[0].bid(state, D.PRIVATE_ER) == 45
