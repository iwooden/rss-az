"""Tests for the DIVIDENDS phase.

Covers: dividend payments to shareholders, max dividend constraints (price
card and affordability), share price adjustment after dividends, receivership
auto-pay, corp processing order (descending share price), phase transitions
to END_CARD, and bankruptcy on price slide to 0.
"""
from core.actions import (
    ACTION_DIVIDEND_PY as ACTION_DIVIDEND,
)
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from phases.dividends import setup_dividends_phase_py

from tests.phases.conftest import (
    apply_and_verify,
    get_legal_actions,
    find_legal_action,
    float_corp_for_test,
    setup_receivership_corp,
)


# =============================================================================
# HELPERS
# =============================================================================

# Company IDs with known star counts (static properties — printed on the card).
# BME: 1 star, face=1.  Used for "too few stars" scenarios.
CO_1STAR = 0
# CDG: 5 stars, face=60. Used for "excess stars" scenarios.
CO_5STAR = 35
# BSE: 1 star, face=2.  Second 1-star company for multi-company setups.
CO_1STAR_B = 1


def _enter_dividends(state):
    """Transition state into DIVIDENDS phase.

    If no active player-controlled corp exists, setup may immediately
    transition to END_CARD (all receivership or no corps).
    """
    setup_dividends_phase_py(state)


# =============================================================================
# DIVIDEND PAYMENT TESTS
# =============================================================================

class TestDividendPayment:
    """Dividend cash flows: corp pays, players receive."""

    def test_single_shareholder_receives_dividend(self, game_state):
        """Player holding shares receives dividend * shares."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10, float_shares=1)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert TURN.get_active_corp(game_state) == 0
        p0_cash_before = PLAYERS[0].get_cash(game_state)
        corp_cash_before = CORPS[0].get_cash(game_state)
        issued = CORPS[0].get_issued_shares(game_state)
        p0_shares = PLAYERS[0].get_shares(game_state, 0)

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=3)
        apply_and_verify(game_state, aid)

        assert PLAYERS[0].get_cash(game_state) == p0_cash_before + 3 * p0_shares
        assert CORPS[0].get_cash(game_state) == corp_cash_before - 3 * issued

    def test_multiple_shareholders_receive_proportional(self, game_state):
        """Each player receives dividend * their shares independently."""
        num_players = TURN.get_num_players(game_state)
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10, float_shares=2)
        CORPS[0].set_cash(game_state, 200)

        if num_players >= 2:
            PLAYERS[1].set_shares(game_state, 0, 1)

        _enter_dividends(game_state)

        cash_before = [PLAYERS[p].get_cash(game_state) for p in range(num_players)]
        shares = [PLAYERS[p].get_shares(game_state, 0) for p in range(num_players)]

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=2)
        apply_and_verify(game_state, aid)

        for p in range(num_players):
            assert PLAYERS[p].get_cash(game_state) == cash_before[p] + 2 * shares[p]

    def test_zero_dividend_no_cash_change(self, game_state):
        """Zero dividend moves no cash at all."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        num_players = TURN.get_num_players(game_state)
        cash_before = [PLAYERS[p].get_cash(game_state) for p in range(num_players)]
        corp_cash_before = CORPS[0].get_cash(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        for p in range(num_players):
            assert PLAYERS[p].get_cash(game_state) == cash_before[p]
        assert CORPS[0].get_cash(game_state) == corp_cash_before

    def test_bank_shares_deduct_but_not_paid(self, game_state):
        """Corp cash decreases by dividend * issued_shares, even for bank shares."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10, float_shares=1)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        issued = CORPS[0].get_issued_shares(game_state)
        bank_shares = CORPS[0].get_bank_shares(game_state)
        assert bank_shares > 0, "Need bank shares to test this path"

        corp_cash_before = CORPS[0].get_cash(game_state)
        p0_shares = PLAYERS[0].get_shares(game_state, 0)
        p0_cash_before = PLAYERS[0].get_cash(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=4)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_cash(game_state) == corp_cash_before - 4 * issued
        assert PLAYERS[0].get_cash(game_state) == p0_cash_before + 4 * p0_shares


# =============================================================================
# LEGAL ACTION ENUMERATION TESTS
# =============================================================================

class TestLegalActions:
    """Legal dividend amounts: 0 to min(price//3, cash//issued, 25)."""

    def test_max_limited_by_price_card(self, game_state):
        """Max dividend is price // 3 when corp has ample cash."""
        # par_index 10 => price = $14, max = 14 // 3 = 4
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 500)
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        amounts = sorted(info.amount for _, info in actions)
        assert amounts == [0, 1, 2, 3, 4]

    def test_max_limited_by_affordability(self, game_state):
        """Max dividend is cash // issued when corp is cash-constrained."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=20)
        # par_index 20 => price = $41, max by price = 41//3 = 13
        issued = CORPS[0].get_issued_shares(game_state)
        CORPS[0].set_cash(game_state, issued * 2 + 1)  # afford 2 per share
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        amounts = sorted(info.amount for _, info in actions)
        assert amounts == [0, 1, 2]

    def test_max_at_highest_claimable_price(self, game_state):
        """Max dividend at highest claimable price index (25, $68)."""
        # par_index 25 => price $68, max = 68//3 = 22. The action-space
        # cap of 25 can only bind at index 26 ($75), which is a boundary
        # space and can't be floated into.
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=25)
        CORPS[0].set_cash(game_state, 10000)
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        amounts = sorted(info.amount for _, info in actions)
        assert amounts == list(range(23))  # 0..22

    def test_zero_cash_only_zero_dividend(self, game_state):
        """Corp with no cash can only pay 0."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 0)
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        assert len(actions) == 1
        assert actions[0][1].amount == 0

    def test_no_pass_action_all_dividend_type(self, game_state):
        """DIVIDENDS has no explicit pass — all actions are ACTION_DIVIDEND."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        action_types = {info.action_type for _, info in actions}
        assert action_types == {ACTION_DIVIDEND}


# =============================================================================
# SHARE PRICE ADJUSTMENT TESTS
# =============================================================================

class TestSharePriceAdjustment:
    """Price moves based on owned stars vs required stars after dividend."""

    def test_price_increases_when_stars_exceed_required(self, game_state):
        """Price index increases when corp has more stars than required.

        At index 1 ($5) with issued=2: required_stars = round(2*5/10) = 1.
        CDG (5 stars) + cash_stars(0) = 5 total. diff=+4, move=+2.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_5STAR, par_index=1)
        CORPS[0].set_cash(game_state, 0)

        price_before = CORPS[0].get_price_index(game_state)
        pending_move = CORPS[0].get_pending_price_move(game_state)
        assert pending_move > 0, (
            f"Expected positive move: stars={CORPS[0].get_total_stars(game_state)}, "
            f"move={pending_move}"
        )

        _enter_dividends(game_state)
        # Cash=0 so only amount=0 is legal
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        if CORPS[0].is_active(game_state):
            assert CORPS[0].get_price_index(game_state) > price_before

    def test_price_decreases_when_stars_below_required(self, game_state):
        """Price index decreases when corp has fewer stars than required.

        At index 15 ($24) with issued=2: required_stars = round(2*24/10) = 5.
        BME (1 star) + cash_stars(0) = 1 total. diff=-4, move=-2.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=15)
        CORPS[0].set_cash(game_state, 0)

        price_before = CORPS[0].get_price_index(game_state)
        pending_move = CORPS[0].get_pending_price_move(game_state)
        assert pending_move < 0, (
            f"Expected negative move: stars={CORPS[0].get_total_stars(game_state)}, "
            f"move={pending_move}"
        )

        _enter_dividends(game_state)
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        if CORPS[0].is_active(game_state):
            assert CORPS[0].get_price_index(game_state) < price_before


# =============================================================================
# PROCESSING ORDER TESTS
# =============================================================================

class TestProcessingOrder:
    """Corps processed in descending share-price order."""

    def test_highest_price_corp_processed_first(self, game_state):
        """The corp with the highest share price is offered first."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_1STAR_B, par_index=15)
        CORPS[0].set_cash(game_state, 100)
        CORPS[1].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert TURN.get_active_corp(game_state) == 1

    def test_second_corp_after_first_resolved(self, game_state):
        """After first corp pays, second corp becomes active."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_1STAR_B, par_index=15)
        CORPS[0].set_cash(game_state, 100)
        CORPS[1].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert TURN.get_active_corp(game_state) == 1
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        if TURN.get_phase(game_state) == GamePhases.PHASE_DIVIDENDS:
            assert TURN.get_active_corp(game_state) == 0

    def test_three_corps_processed_in_descending_price(self, game_state):
        """Three corps at different prices are processed high to low."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 3:
            return

        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=0, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=1,
                            company_id=1, par_index=15)
        float_corp_for_test(game_state, corp_id=2, player_id=2,
                            company_id=2, par_index=10)
        for c in [0, 1, 2]:
            CORPS[c].set_cash(game_state, 200)
        _enter_dividends(game_state)

        expected_order = [1, 2, 0]
        for expected_corp in expected_order:
            if TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS:
                break
            assert TURN.get_active_corp(game_state) == expected_corp, (
                f"Expected corp {expected_corp}, got {TURN.get_active_corp(game_state)}"
            )
            aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
            apply_and_verify(game_state, aid)

    def test_active_player_is_president_of_active_corp(self, game_state):
        """Active player is the president of the currently active corp."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 2:
            return

        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=1,
                            company_id=CO_1STAR_B, par_index=15)
        CORPS[0].set_cash(game_state, 100)
        CORPS[1].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert TURN.get_active_corp(game_state) == 1
        assert TURN.get_active_player(game_state) == 1


# =============================================================================
# RECEIVERSHIP TESTS
# =============================================================================

class TestReceivership:
    """Receivership corps auto-pay 0, skipped with no player decision."""

    def test_receivership_corp_auto_skipped(self, game_state):
        """Receivership corp is auto-processed, player-controlled corp gets decision."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        CORPS[0].set_cash(game_state, 100)

        # Corp 1: receivership at higher price (par_index=15)
        setup_receivership_corp(game_state, corp_id=1,
                                company_ids=[CO_1STAR_B], par_index=15)
        CORPS[1].set_cash(game_state, 50)

        _enter_dividends(game_state)

        # Corp 1 has higher price but is in receivership — auto-processed.
        assert TURN.get_phase(game_state) == GamePhases.PHASE_DIVIDENDS
        assert TURN.get_active_corp(game_state) == 0

    def test_receivership_corp_no_dividend_paid(self, game_state):
        """Receivership corps pay 0 dividend — no cash changes for players."""
        setup_receivership_corp(game_state, corp_id=0, company_ids=[CO_1STAR])
        CORPS[0].set_cash(game_state, 50)

        num_players = TURN.get_num_players(game_state)
        cash_before = [PLAYERS[p].get_cash(game_state) for p in range(num_players)]
        corp_cash_before = CORPS[0].get_cash(game_state)

        _enter_dividends(game_state)

        # All receivership => transitions past DIVIDENDS
        assert TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS

        for p in range(num_players):
            assert PLAYERS[p].get_cash(game_state) == cash_before[p]
        assert CORPS[0].get_cash(game_state) == corp_cash_before

    def test_receivership_price_still_adjusts(self, game_state):
        """Receivership corps still get share price adjusted (with 0 dividend).

        BME (1 star) at par_index=10 ($14), issued=2: required=3, total=1.
        move=-2 guaranteed. Price should decrease.
        """
        setup_receivership_corp(game_state, corp_id=0, company_ids=[CO_1STAR])
        CORPS[0].set_cash(game_state, 0)

        price_before = CORPS[0].get_price_index(game_state)
        pending_move = CORPS[0].get_pending_price_move(game_state)
        assert pending_move != 0, f"Expected non-zero move, got {pending_move}"

        _enter_dividends(game_state)

        if CORPS[0].is_active(game_state):
            assert CORPS[0].get_price_index(game_state) != price_before

    def test_mixed_receivership_and_player_corps(self, game_state):
        """Mix of receivership and player corps: only player corps need decisions."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 2:
            return

        # Corp 0: player-controlled at price 5
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        CORPS[0].set_cash(game_state, 100)

        # Corp 1: receivership at price 20 (highest — will be auto-skipped)
        setup_receivership_corp(game_state, corp_id=1,
                                company_ids=[CO_1STAR_B], par_index=20)
        CORPS[1].set_cash(game_state, 50)

        _enter_dividends(game_state)

        # Receivership corp 1 was auto-processed, player gets corp 0
        assert TURN.get_active_corp(game_state) == 0


# =============================================================================
# PHASE TRANSITION TESTS
# =============================================================================

class TestPhaseTransition:
    """Transitions from DIVIDENDS to END_CARD."""

    def test_single_corp_transitions_after_dividend(self, game_state):
        """After single corp's dividend, phase transitions past DIVIDENDS."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS

    def test_no_active_corps_immediate_transition(self, game_state):
        """If no active corps, setup immediately transitions past DIVIDENDS."""
        _enter_dividends(game_state)
        assert TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS

    def test_all_receivership_transitions_past_dividends(self, game_state):
        """When all active corps are in receivership, phase auto-completes."""
        setup_receivership_corp(game_state, corp_id=0, company_ids=[CO_1STAR])
        CORPS[0].set_cash(game_state, 50)

        _enter_dividends(game_state)
        assert TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS

    def test_two_corps_transitions_after_both(self, game_state):
        """After both corps pay, transitions past DIVIDENDS."""
        num_players = TURN.get_num_players(game_state)
        if num_players < 2:
            return

        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=1,
                            company_id=CO_1STAR_B, par_index=15)
        CORPS[0].set_cash(game_state, 100)
        CORPS[1].set_cash(game_state, 100)
        _enter_dividends(game_state)

        # First corp (1, higher price)
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        if TURN.get_phase(game_state) == GamePhases.PHASE_DIVIDENDS:
            aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
            apply_and_verify(game_state, aid)

        assert TURN.get_phase(game_state) != GamePhases.PHASE_DIVIDENDS


# =============================================================================
# BANKRUPTCY TESTS
# =============================================================================

class TestBankruptcy:
    """Bankruptcy when price slides to index 0."""

    def test_corp_goes_bankrupt_on_price_zero(self, game_state):
        """Corp at index 1 with negative move goes bankrupt.

        BME (1 star) at index 1 ($5), issued=3 (float_shares=2):
        required_stars = round(3*5/10) = 2, total=1, move=-1.
        Target = 1 + (-1) = 0 => bankruptcy.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=1, float_shares=2)
        CORPS[0].set_cash(game_state, 0)

        pending = CORPS[0].get_pending_price_move(game_state)
        assert pending < 0, f"Expected negative move, got {pending}"

        _enter_dividends(game_state)
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert not CORPS[0].is_active(game_state)

    def test_bankrupt_corp_companies_removed(self, game_state):
        """Bankrupt corp's companies are removed from game."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=1, float_shares=2)
        CORPS[0].set_cash(game_state, 0)

        assert CORPS[0].count_companies(game_state) == 1

        _enter_dividends(game_state)
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert not CORPS[0].is_active(game_state)
        assert CORPS[0].count_companies(game_state) == 0


# =============================================================================
# MARKET SPACE SLIDE TESTS
# =============================================================================

class TestMarketSlide:
    """Price slide through occupied spaces."""

    def test_price_slides_past_occupied_space_up(self, game_state):
        """Rising price slides past occupied target.

        Corp 0: BME (1 star) at index 5 ($9), issued=2, cash=20 (cash_stars=2).
        required=round(2*9/10)=2. total=1+2=3. move=+1. Target=6.
        Corp 1 occupies index 6. Slide should continue to index 7.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=5)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_1STAR_B, par_index=6)
        CORPS[0].set_cash(game_state, 20)
        # Corp 1 must stay put so index 6 remains occupied when corp 0 targets it.
        # BSE (1 star) at index 6 ($10), issued=2, required=2. cash=10 => cash_stars=1,
        # total=2, move=0.
        CORPS[1].set_cash(game_state, 10)

        assert CORPS[0].get_pending_price_move(game_state) == 1
        assert CORPS[1].get_pending_price_move(game_state) == 0
        assert not MARKET.is_space_available(game_state, 6)

        _enter_dividends(game_state)
        # Corp 1 processes first (higher price), stays at 6.
        assert TURN.get_active_corp(game_state) == 1
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        # Corp 0 processes. Target=6 (still occupied), slides to 7.
        assert TURN.get_active_corp(game_state) == 0
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_price_index(game_state) == 7

    def test_price_slides_past_occupied_space_down(self, game_state):
        """Falling price slides past occupied target.

        Corp 0: BME (1 star) at index 7 ($11), issued=2, cash=0.
        required=round(2*11/10)=2. total=1. move=-1. Target=6.
        Corp 1 occupies index 6. Slide should continue down to index 5.
        """
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=7)
        float_corp_for_test(game_state, corp_id=1, player_id=0,
                            company_id=CO_1STAR_B, par_index=6)
        CORPS[0].set_cash(game_state, 0)
        CORPS[1].set_cash(game_state, 100)

        assert CORPS[0].get_pending_price_move(game_state) == -1
        assert not MARKET.is_space_available(game_state, 6)

        _enter_dividends(game_state)
        # Corp 0 has higher price — processed first. Target=6 (occupied), slides to 5.
        assert TURN.get_active_corp(game_state) == 0
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_price_index(game_state) == 5


# =============================================================================
# DIVIDEND REMAINING FLAG TESTS
# =============================================================================

class TestDividendRemaining:
    """Dividend remaining flags track which corps still need processing."""

    def test_remaining_cleared_after_processing(self, game_state):
        """After corp pays dividend, its remaining flag is cleared."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert TURN.is_dividend_remaining(game_state, 0)
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=0)
        apply_and_verify(game_state, aid)

        assert not TURN.is_dividend_remaining(game_state, 0)

    def test_inactive_corps_not_remaining(self, game_state):
        """Inactive corps have remaining flag set to False."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        assert not TURN.is_dividend_remaining(game_state, 1)

    def test_only_active_corps_marked_remaining(self, game_state):
        """Only active corps are marked as remaining at setup."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        float_corp_for_test(game_state, corp_id=2, player_id=0,
                            company_id=CO_1STAR_B, par_index=5)
        CORPS[0].set_cash(game_state, 100)
        CORPS[2].set_cash(game_state, 100)
        _enter_dividends(game_state)

        for c in range(int(GameConstants.NUM_CORPS)):
            if CORPS[c].is_active(game_state):
                assert TURN.is_dividend_remaining(game_state, c)
            else:
                assert not TURN.is_dividend_remaining(game_state, c)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Boundary and edge case scenarios."""

    def test_max_dividend_exact_cash(self, game_state):
        """Corp with exactly enough cash for max dividend can pay it."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        # price=14, max_by_price = 14//3 = 4
        issued = CORPS[0].get_issued_shares(game_state)
        CORPS[0].set_cash(game_state, 4 * issued)
        _enter_dividends(game_state)

        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=4)
        corp_cash_before = CORPS[0].get_cash(game_state)
        apply_and_verify(game_state, aid)

        assert CORPS[0].get_cash(game_state) == corp_cash_before - 4 * issued

    def test_dividend_one_less_than_max(self, game_state):
        """Corp can pay one less than the maximum."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_1STAR, par_index=10)
        CORPS[0].set_cash(game_state, 200)
        _enter_dividends(game_state)

        # Max is 4 (price=14, 14//3=4), pay 3
        aid = find_legal_action(game_state, action_type=ACTION_DIVIDEND, amount=3)
        apply_and_verify(game_state, aid)

    def test_low_price_low_max_dividend(self, game_state):
        """At lowest non-bankrupt price (index 1, $5), max is 5//3=1."""
        float_corp_for_test(game_state, corp_id=0, player_id=0,
                            company_id=CO_5STAR, par_index=1)
        CORPS[0].set_cash(game_state, 100)
        _enter_dividends(game_state)

        actions = get_legal_actions(game_state)
        amounts = sorted(info.amount for _, info in actions)
        assert amounts == [0, 1]
