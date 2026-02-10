"""Comprehensive tests for player net worth calculation, storage, and freshness.

Net worth = cash + acquisition_proceeds + sum(company face values) + sum(shares * share_price)

Tests verify:
1. The calculate_net_worth formula includes all components
2. Net worth is refreshed (stored == calculated) before every model decision
3. Key game actions (buy/sell/auction/bankruptcy) keep NW fresh
4. IPO is net-worth-neutral for the founding player
"""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.data import (
    GamePhases, get_company_face_value,
    get_par_price, get_par_index_for_slot, COMPANY_NAME_TO_ID,
)
from core.actions import get_valid_action_mask, get_action_layout
from entities.player import PLAYERS, update_all_net_worths
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.turn import TURN
from entities.deck import DECK
from phases.dividends import setup_dividends_phase_py
from phases.issue import setup_issue_phase_py
from phases.acquisition import setup_acquisition_phase_py
from phases.closing import apply_closing_auto_py
from phases.ipo import setup_ipo_phase_py, process_ipo_py
from tests.phases.conftest import (
    apply_and_verify_all, float_corp_for_test,
)


# =============================================================================
# HELPERS
# =============================================================================

def assert_net_worth_fresh(state, msg=""):
    """Assert stored net worth equals calculated net worth for all players."""
    for p in range(state.get_num_players()):
        stored = PLAYERS[p].get_net_worth(state)
        calculated = PLAYERS[p].calculate_net_worth(state)
        assert stored == calculated, (
            f"{msg}\nPlayer {p} net worth stale: "
            f"stored={stored}, calculated={calculated}"
        )


def inject_stale_net_worth(state, value=99999):
    """Set all players' stored NW to a bogus value."""
    for p in range(state.get_num_players()):
        PLAYERS[p].set_net_worth(state, value)


def apply_action_raw(state, action_idx):
    """Apply action via driver without invariant checks on intermediate states.

    Used for stale-injection tests where the pre-action state deliberately
    violates the freshness invariant.
    """
    mask = get_valid_action_mask(state)
    assert mask[action_idx] == 1.0, f"Action {action_idx} not valid in current mask"
    status = DRIVER.apply_action(state, action_idx)
    return status


# =============================================================================
# 1. NET WORTH FORMULA
# =============================================================================

class TestNetWorthFormula:
    """Unit tests of calculate_net_worth."""

    def test_cash_only(self):
        """NW = cash when player has no companies or shares."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 75)
        nw = PLAYERS[0].calculate_net_worth(state)
        assert nw == 75

    def test_cash_plus_company(self):
        """NW includes company face value."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 50)
        cid = DECK.draw(state)
        COMPANIES[cid].transfer_to_player(state, 0)
        fv = get_company_face_value(cid)

        nw = PLAYERS[0].calculate_net_worth(state)
        assert nw == 50 + fv

    def test_cash_plus_shares(self):
        """NW includes shares x price for active corp."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 30)
        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        share_price = CORPS[0].get_share_price(state)

        # calculate_net_worth counts player-owned privates, not corp subsidiaries
        nw = PLAYERS[0].calculate_net_worth(state)
        assert nw == 30 + (2 * share_price)

    def test_all_components(self):
        """NW = cash + acq_proceeds + companies + shares."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 40)
        PLAYERS[0].set_acquisition_proceeds(state, 15)

        # Give player a private company
        cid = DECK.draw(state)
        COMPANIES[cid].transfer_to_player(state, 0)
        fv = get_company_face_value(cid)

        # Float a corp so player has shares
        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=1)
        share_price = CORPS[0].get_share_price(state)

        nw = PLAYERS[0].calculate_net_worth(state)
        assert nw == 40 + 15 + fv + (1 * share_price)

    def test_acquisition_proceeds_included(self):
        """Set proceeds=100, clear, verify NW drops by 100."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 50)
        PLAYERS[0].set_acquisition_proceeds(state, 100)
        nw_with = PLAYERS[0].calculate_net_worth(state)

        PLAYERS[0].clear_acquisition_proceeds(state)
        nw_without = PLAYERS[0].calculate_net_worth(state)

        assert nw_with - nw_without == 100

    def test_inactive_corp_shares_excluded(self):
        """Shares in inactive corp don't count toward NW."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 50)
        float_corp_for_test(state, corp_id=0, par_index=1, float_shares=2)

        # Directly deactivate corp (simulate bankruptcy aftermath)
        PLAYERS[0].set_shares(state, 0, 0)
        CORPS[0].set_active(state, False)

        nw_after = PLAYERS[0].calculate_net_worth(state)
        assert nw_after == 50  # Only cash remains

    def test_update_stores_calculated_value(self):
        """update_net_worth stores same as calculate_net_worth."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 75)
        calculated = PLAYERS[0].calculate_net_worth(state)
        PLAYERS[0].update_net_worth(state)
        stored = PLAYERS[0].get_net_worth(state)
        assert stored == calculated


# =============================================================================
# 2. NET WORTH FRESH AFTER INVEST
# =============================================================================

class TestNetWorthFreshAfterInvest:
    """Buy/sell share refreshes NW."""

    @pytest.fixture
    def trade_state(self):
        """State with active corp for buy/sell testing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)
        PLAYERS[0].set_cash(state, 100)
        update_all_net_worths(state)
        return state

    def test_fresh_after_buy_share(self, trade_state):
        """Inject stale -> buy -> assert fresh."""
        inject_stale_net_worth(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_action_raw(trade_state, buy_idx)

        assert_net_worth_fresh(trade_state, "After buy share")

    def test_all_players_fresh_after_buy(self, trade_state):
        """Inject stale for all -> buy -> all fresh."""
        PLAYERS[1].set_shares(trade_state, 0, 1)
        PLAYERS[1].set_cash(trade_state, 50)

        inject_stale_net_worth(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_action_raw(trade_state, buy_idx)

        assert_net_worth_fresh(trade_state, "All players after buy")

    def test_price_change_reflected_after_buy(self, trade_state):
        """P1 has shares, P0 buys -> P1 NW increases."""
        PLAYERS[1].set_shares(trade_state, 0, 1)
        PLAYERS[1].set_cash(trade_state, 50)
        update_all_net_worths(trade_state)

        initial_nw_p1 = PLAYERS[1].get_net_worth(trade_state)
        initial_price = CORPS[0].get_share_price(trade_state)

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_and_verify_all(trade_state, buy_idx)

        new_price = CORPS[0].get_share_price(trade_state)
        new_nw_p1 = PLAYERS[1].get_net_worth(trade_state)

        assert new_price > initial_price
        assert new_nw_p1 > initial_nw_p1

    def test_fresh_after_sell_share(self, trade_state):
        """Inject stale -> sell -> assert fresh."""
        inject_stale_net_worth(trade_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_action_raw(trade_state, sell_idx)

        assert_net_worth_fresh(trade_state, "After sell share")


# =============================================================================
# 3. NET WORTH FRESH AFTER AUCTION WIN
# =============================================================================

class TestNetWorthFreshAfterAuctionWin:
    """Auction resolution updates winner."""

    @pytest.fixture
    def resolved_auction(self):
        """Run auction to completion and return (state, winner, company, bid_price, initial_nw)."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Start auction
        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        winner_id = TURN.get_auction_high_bidder(state)
        company_id = TURN.get_auction_company(state)
        bid_price = TURN.get_auction_price(state)
        initial_nw = PLAYERS[winner_id].get_net_worth(state)
        face_value = get_company_face_value(company_id)

        # All others leave to resolve
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_and_verify_all(state, layout['leave_auction'])

        return state, winner_id, company_id, bid_price, initial_nw, face_value

    def test_winner_net_worth_correct_after_auction(self, resolved_auction):
        """NW change = face_value - bid_price."""
        state, winner_id, company_id, bid_price, initial_nw, face_value = resolved_auction

        final_nw = PLAYERS[winner_id].get_net_worth(state)
        expected_change = face_value - bid_price
        assert final_nw == initial_nw + expected_change

    def test_winner_net_worth_fresh_after_auction(self):
        """Inject stale for winner -> resolve -> winner's NW is fresh."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        mask = get_valid_action_mask(state)
        layout = get_action_layout(3)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_and_verify_all(state, i)
                break

        winner_id = TURN.get_auction_high_bidder(state)

        # Inject stale NW after entering auction (before resolution)
        inject_stale_net_worth(state)

        # Resolve auction using raw driver (stale intermediate state is expected)
        for _ in range(2):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action_raw(state, layout['leave_auction'])

        # Winner's NW is refreshed by auction resolution
        stored = PLAYERS[winner_id].get_net_worth(state)
        calculated = PLAYERS[winner_id].calculate_net_worth(state)
        assert stored == calculated, (
            f"Winner (player {winner_id}) NW stale after auction: "
            f"stored={stored}, calculated={calculated}"
        )


# =============================================================================
# 4. NET WORTH FRESH AFTER BANKRUPTCY
# =============================================================================

class TestNetWorthFreshAfterBankruptcy:
    """go_bankrupt updates all players."""

    @pytest.fixture
    def bankruptcy_state(self):
        """State where one sell triggers bankruptcy."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        float_corp_for_test(state, corp_id=0, company_id=3, par_index=1, float_shares=2)
        PLAYERS[0].set_cash(state, 100)
        update_all_net_worths(state)
        return state

    def test_bankruptcy_updates_all_shareholders(self, bankruptcy_state):
        """P1 NW drops by share_value after bankruptcy."""
        corp = CORPS[0]
        PLAYERS[1].set_shares(bankruptcy_state, 0, 1)
        PLAYERS[2].set_shares(bankruptcy_state, 0, 1)
        PLAYERS[1].set_cash(bankruptcy_state, 50)
        PLAYERS[2].set_cash(bankruptcy_state, 50)
        update_all_net_worths(bankruptcy_state)

        initial_nw_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        share_value = corp.get_share_price(bankruptcy_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_and_verify_all(bankruptcy_state, sell_idx)

        assert PLAYERS[1].get_shares(bankruptcy_state, 0) == 0
        new_nw_p1 = PLAYERS[1].get_net_worth(bankruptcy_state)
        assert new_nw_p1 == initial_nw_p1 - share_value

    def test_bankruptcy_net_worth_fresh(self, bankruptcy_state):
        """Inject stale -> bankruptcy -> all players fresh."""
        PLAYERS[1].set_shares(bankruptcy_state, 0, 1)
        PLAYERS[1].set_cash(bankruptcy_state, 50)

        inject_stale_net_worth(bankruptcy_state)

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_action_raw(bankruptcy_state, sell_idx)

        assert_net_worth_fresh(bankruptcy_state, "After bankruptcy")


# =============================================================================
# 5. NET WORTH FRESH BEFORE DIVIDEND DECISION
# =============================================================================

class TestNetWorthFreshBeforeDividendDecision:
    """setup_dividends_phase calls update_all_net_worths before player decision."""

    def test_fresh_before_dividend_decision(self):
        """Inject stale -> setup_dividends_phase -> assert fresh."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=1)
        CORPS[0].set_cash(state, 50)
        CORPS[0].set_stars(state, 5)

        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)

        inject_stale_net_worth(state)
        setup_dividends_phase_py(state)

        # If the phase found a player-controlled corp, NW should be fresh
        if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
            assert_net_worth_fresh(state, "Before dividend decision")


# =============================================================================
# 6. NET WORTH FRESH BEFORE ISSUE DECISION
# =============================================================================

class TestNetWorthFreshBeforeIssueDecision:
    """setup_issue_phase calls update_all_net_worths before player decision."""

    def test_fresh_before_issue_decision(self):
        """Inject stale -> setup_issue_phase -> assert fresh."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=10, float_shares=1)
        CORPS[0].set_cash(state, 50)

        TURN.set_phase(state, GamePhases.PHASE_ISSUE_SHARES)

        inject_stale_net_worth(state)
        setup_issue_phase_py(state)

        if state.get_phase() == GamePhases.PHASE_ISSUE_SHARES:
            assert_net_worth_fresh(state, "Before issue decision")


# =============================================================================
# 7. NET WORTH FRESH BEFORE ACQUISITION OFFER
# =============================================================================

class TestNetWorthFreshBeforeAcquisitionOffer:
    """setup_acquisition_phase calls update_all_net_worths before player offer."""

    def test_fresh_before_acquisition_offer(self):
        """Float corp with cash, FI owns company -> inject stale -> setup -> fresh."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        float_corp_for_test(state, corp_id=0, par_index=15, float_shares=1)
        CORPS[0].set_cash(state, 200)

        # Give FI a company for the corp to potentially acquire
        fi_cid = DECK.draw(state)
        COMPANIES[fi_cid].transfer_to_fi(state)

        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)

        inject_stale_net_worth(state)
        setup_acquisition_phase_py(state)

        if state.get_phase() == GamePhases.PHASE_ACQUISITION:
            assert_net_worth_fresh(state, "Before acquisition offer")


# =============================================================================
# 8. NET WORTH FRESH BEFORE CLOSING OFFER
# =============================================================================

class TestNetWorthFreshBeforeClosingOffer:
    """apply_closing_auto calls update_all_net_worths before player offer."""

    def test_fresh_before_closing_offer(self):
        """Float corp with negative-income company, high CoO -> inject stale -> fresh."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # High CoO so companies have negative income
        TURN.set_coo_level(state, 6)

        # Give player a private company with negative income at this CoO
        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_player(state, 0)

        TURN.set_phase(state, GamePhases.PHASE_CLOSING)

        inject_stale_net_worth(state)
        apply_closing_auto_py(state)

        if state.get_phase() == GamePhases.PHASE_CLOSING:
            assert_net_worth_fresh(state, "Before closing offer")


# =============================================================================
# 9. ACQUISITION PROCEEDS IN NET WORTH
# =============================================================================

class TestAcquisitionProceedsInNetWorth:
    """Verify acquisition_proceeds is part of net worth calculation."""

    def test_proceeds_included_in_calculation(self):
        """Set acq_proceeds=50, verify calculate_net_worth includes it."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 100)
        PLAYERS[0].set_acquisition_proceeds(state, 50)

        nw = PLAYERS[0].calculate_net_worth(state)
        assert nw == 150

    def test_proceeds_cleared_reduces_net_worth(self):
        """Set proceeds, calculate, clear, recalculate — verify drop."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        PLAYERS[0].set_cash(state, 100)
        PLAYERS[0].set_acquisition_proceeds(state, 50)
        nw_before = PLAYERS[0].calculate_net_worth(state)

        PLAYERS[0].clear_acquisition_proceeds(state)
        nw_after = PLAYERS[0].calculate_net_worth(state)

        assert nw_before - nw_after == 50


# =============================================================================
# 10. IPO NET WORTH NEUTRAL
# =============================================================================

class TestIpoNetWorthNeutral:
    """IPO should be net-worth-neutral for the founding player.

    Formula proof:
    NW_after = (cash - player_payment) + float_shares * par
             = (cash - (float_shares * par - FV)) + float_shares * par
             = cash + FV
             = NW_before
    """

    def test_ipo_neutral_fv_leq_par(self):
        """FV <= par (1 share each): player NW unchanged after IPO."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Red company (stars=1), FV=6
        cid = 3
        fv = get_company_face_value(cid)
        assert fv <= 10  # Sanity: FV <= lowest par

        COMPANIES[cid].transfer_to_player(state, 0)
        PLAYERS[0].set_cash(state, 50)

        nw_before = PLAYERS[0].calculate_net_worth(state)
        assert nw_before == 50 + fv

        # IPO: par_slot=0 for star tier 1 -> float_shares = 1
        setup_ipo_phase_py(state)
        process_ipo_py(state, 0, 0)

        PLAYERS[0].update_net_worth(state)
        nw_after = PLAYERS[0].calculate_net_worth(state)
        assert nw_after == nw_before, (
            f"IPO should be NW-neutral: before={nw_before}, after={nw_after}"
        )

    def test_ipo_neutral_fv_gt_par(self):
        """FV > par (2 shares each): player NW unchanged after IPO."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Orange company (stars=2), FV=19
        cid = 13
        fv = get_company_face_value(cid)
        star_tier = 2

        par_index = get_par_index_for_slot(star_tier, 0)
        par_price = get_par_price(par_index)
        assert fv > par_price, f"Need FV({fv}) > par({par_price}) for 2-share float"

        COMPANIES[cid].transfer_to_player(state, 0)
        PLAYERS[0].set_cash(state, 100)

        nw_before = PLAYERS[0].calculate_net_worth(state)
        assert nw_before == 100 + fv

        # IPO: FV > par -> float_shares = 2
        setup_ipo_phase_py(state)
        process_ipo_py(state, 0, 0)

        PLAYERS[0].update_net_worth(state)
        nw_after = PLAYERS[0].calculate_net_worth(state)
        assert nw_after == nw_before, (
            f"IPO should be NW-neutral: before={nw_before}, after={nw_after}"
        )
