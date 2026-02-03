"""Integration tests verifying invariants throughout multi-action sequences.

This file consolidates integration-style tests from per-phase test files.
Tests here verify that invariants hold across phase transitions and
multi-action sequences, not just individual action correctness.

Add new integration tests here as phases are implemented.
"""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN


# =============================================================================
# INVEST PHASE INTEGRATION
# =============================================================================

class TestInvestIntegration:
    """Integration tests verifying invariants after every action."""

    def test_multiple_passes_maintains_invariants(self, game_state):
        """Multiple pass actions maintain game invariants throughout."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(game_state, "Initial state")

        layout = get_action_layout(3)
        # Pass twice (not enough for WRAP_UP)
        for i in range(2):
            apply_action_and_verify(game_state, layout['pass_invest'], f"Pass {i+1}")

        assert game_state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(game_state, "After two passes")

    def test_auction_cycle_maintains_invariants(self, game_state):
        """Starting auction and completing it maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants, assert_valid_mask

        assert_invariants(game_state, "Initial state")
        assert_valid_mask(game_state, msg="Initial mask valid")

        # Find and start auction
        mask = get_valid_action_mask(game_state)
        layout = get_action_layout(3)
        auction_idx = None
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                auction_idx = i
                break

        if auction_idx is not None:
            apply_action_and_verify(game_state, auction_idx, "Start auction")
            assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
            assert_invariants(game_state, "After auction start")

    def test_buy_share_maintains_invariants(self, trade_state):
        """Buy share action maintains all game invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(trade_state, "Initial trade state")

        layout = get_action_layout(3)
        buy_idx = layout['buy_share_base'] + 0
        apply_action_and_verify(trade_state, buy_idx, "Buy share")

        assert_invariants(trade_state, "After buy share")

    def test_sell_share_maintains_invariants(self, trade_state):
        """Sell share action maintains all game invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        assert_invariants(trade_state, "Initial trade state")

        layout = get_action_layout(3)
        sell_idx = layout['sell_share_base'] + 0
        apply_action_and_verify(trade_state, sell_idx, "Sell share")

        assert_invariants(trade_state, "After sell share")

    def test_multiple_trades_maintain_invariants(self, trade_state):
        """Multiple buy actions in sequence maintain invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        layout = get_action_layout(3)

        assert_invariants(trade_state, "Initial state")

        # Buy twice (trade_state has 2 bank shares available)
        buy_idx = layout['buy_share_base'] + 0

        apply_action_and_verify(trade_state, buy_idx, "First buy")
        apply_action_and_verify(trade_state, buy_idx, "Second buy")

        assert_invariants(trade_state, "After two buys")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_wrap_up_transition_maintains_invariants(self, num_players):
        """Phase transition through WRAP_UP maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        layout = get_action_layout(num_players)
        # Apply all passes with verify helper
        for i in range(num_players):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")

        # After all passes, should be back in INVEST (after WRAP_UP -> ACQUISITION)
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After WRAP_UP -> ACQUISITION -> INVEST transition")


# =============================================================================
# BID PHASE INTEGRATION
# =============================================================================

class TestBidIntegration:
    """Integration tests verifying invariants throughout auction cycles."""

    def test_full_auction_maintains_invariants(self):
        """Complete auction cycle (start -> bids -> resolution) maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants, assert_valid_mask

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")
        assert_valid_mask(state, msg="Initial INVEST mask")

        layout = get_action_layout(3)

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Start auction")
                break

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        assert_invariants(state, "After auction start")
        assert_valid_mask(state, msg="BID phase mask")

        # First player leaves
        apply_action_and_verify(state, layout['leave_auction'], "First leave")
        assert_invariants(state, "After first leave")

        # Second player leaves - triggers resolution
        if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action_and_verify(state, layout['leave_auction'], "Second leave")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(state, "After auction resolution")
        assert_valid_mask(state, msg="Post-auction INVEST mask")

    def test_auction_with_raises_maintains_invariants(self):
        """Auction with multiple raise bids maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Start auction")
                break

        assert_invariants(state, "After auction start")

        # First player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "First raise")
                break

        assert_invariants(state, "After first raise")

        # Second player raises
        mask = get_valid_action_mask(state)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Second raise")
                break

        assert_invariants(state, "After second raise")

        # Resolve via leaves
        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action_and_verify(state, layout['leave_auction'], "Leave to resolve")

        assert_invariants(state, "After resolution with raises")

    def test_multiple_auctions_maintain_invariants(self):
        """Multiple auction cycles in sequence maintain invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        for auction_num in range(2):  # Two auction cycles
            assert_invariants(state, f"Before auction {auction_num + 1}")

            # Start auction
            mask = get_valid_action_mask(state)
            auction_started = False
            for i in range(layout['auction_base'], layout['buy_share_base']):
                if mask[i] == 1.0:
                    apply_action_and_verify(state, i, f"Start auction {auction_num + 1}")
                    auction_started = True
                    break

            if not auction_started:
                break  # No more auctions available

            # Complete auction via leaves
            while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action_and_verify(state, layout['leave_auction'], f"Leave auction {auction_num + 1}")

            assert_invariants(state, f"After auction {auction_num + 1}")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_auction_maintains_invariants_all_player_counts(self, num_players):
        """Auction cycle maintains invariants for all player counts."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        layout = get_action_layout(num_players)

        assert_invariants(state, f"Initial {num_players}p")

        # Start auction
        mask = get_valid_action_mask(state)
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, f"Start auction {num_players}p")
                break

        # All but one leave
        for _ in range(num_players - 1):
            if state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action_and_verify(state, layout['leave_auction'], f"Leave {num_players}p")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert_invariants(state, f"After auction {num_players}p")


# =============================================================================
# ACQUISITION PHASE INTEGRATION
# =============================================================================

class TestAcquisitionIntegration:
    """Integration tests verifying ACQUISITION phase maintains invariants throughout."""

    def test_wrap_up_to_acquisition_maintains_invariants(self):
        """INVEST->WRAP_UP->ACQUISITION phase transition maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        layout = get_action_layout(3)

        # Pass all players to trigger WRAP_UP
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")

        # After all passes, should be back in INVEST (WRAP_UP -> ACQUISITION -> INVEST)
        # In a fresh game, ACQUISITION has no offers, so it completes immediately
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After WRAP_UP->ACQUISITION->INVEST transition")

    def test_acquisition_to_invest_new_turn(self):
        """ACQUISITION phase completes and transitions to CLOSING then INVEST with new turn."""
        from tests.phases.conftest import assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up ACQUISITION phase (no offers in fresh game)
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        assert state.get_phase() == GamePhases.PHASE_ACQUISITION
        initial_turn = TURN.get_turn_number(state)

        assert_invariants(state, "Before transition")

        # Transition to CLOSING phase
        transition_to_closing_py(state)

        # Should be in CLOSING (auto-close executes next)
        assert state.get_phase() == GamePhases.PHASE_CLOSING
        assert TURN.get_turn_number(state) == initial_turn  # Turn not incremented yet
        assert_invariants(state, "After transition to CLOSING")

        # Execute auto-close (transitions to INCOME)
        apply_closing_auto_py(state)

        # Should now be in INCOME (CLOSING transitions to INCOME now)
        # Turn number incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == initial_turn  # Not incremented yet
        assert_invariants(state, "After CLOSING to INCOME")

    def test_full_turn_cycle_with_acquisition(self):
        """Full turn cycle (INVEST->WRAP_UP->ACQUISITION->INVEST) maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        assert_invariants(state, "Initial state")
        assert TURN.get_turn_number(state) == 1

        # Turn 1: Pass all players
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Turn 1 Pass {i+1}")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After turn 1")

        # Turn 2: Pass all players again
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Turn 2 Pass {i+1}")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 3
        assert_invariants(state, "After turn 2")

    def test_acquisition_accept_maintains_invariants(self):
        """Accept action in ACQUISITION maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS
        from phases.acquisition import setup_acquisition_phase_py, get_offer_count
        from core.data import CORP_NAMES

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up valid acquisition offer: Player 0's private company -> Corp 0
        COMPANIES[0].transfer_to_player(state, 0)  # Give company 0 to player 0
        CORPS[0].set_active(state, True)  # Activate corp 0
        CORPS[0].set_cash(state, 50000)  # Give corp cash
        PLAYERS[0].set_president_of(state, 0, True)  # Player 0 is president of corp 0

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        setup_acquisition_phase_py(state)

        # Should have at least one offer
        offer_count = get_offer_count(state)
        assert offer_count > 0, "Should have generated at least one offer"

        assert_invariants(state, "Before accept")

        # Find and apply acquisition accept action
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)

        # Find a valid acquisition action (ACQ_PRICE actions start at acquisition_start)
        action_idx = None
        for i in range(layout['acquisition_start'], len(mask)):
            if mask[i] == 1.0:
                action_idx = i
                break

        assert action_idx is not None, "Should have at least one valid acquisition action"

        # Apply action and verify invariants
        apply_action_and_verify(state, action_idx, "Accept acquisition")

        assert_invariants(state, "After accept")

    def test_acquisition_pass_maintains_invariants(self):
        """Pass action in ACQUISITION maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS
        from phases.acquisition import setup_acquisition_phase_py, get_offer_count
        from core.data import CORP_NAMES

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up valid acquisition offer: Player 0's private company -> Corp 0
        COMPANIES[0].transfer_to_player(state, 0)
        CORPS[0].set_active(state, True)
        CORPS[0].set_cash(state, 50000)
        PLAYERS[0].set_president_of(state, 0, True)

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        setup_acquisition_phase_py(state)

        assert get_offer_count(state) > 0
        assert_invariants(state, "Before pass")

        # Apply pass action (acq_pass is the pass action for ACQUISITION phase)
        layout = get_action_layout(3)
        apply_action_and_verify(state, layout['acq_pass'], "Pass acquisition")

        assert_invariants(state, "After pass")

    def test_multiple_acquisitions_maintain_invariants(self):
        """Multiple acquisition actions maintain invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS
        from phases.acquisition import setup_acquisition_phase_py, get_offer_count
        from core.data import CORP_NAMES

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up multiple offers: Player 0's companies -> Corp 0
        COMPANIES[0].transfer_to_player(state, 0)
        COMPANIES[1].transfer_to_player(state, 0)
        CORPS[0].set_active(state, True)
        CORPS[0].set_cash(state, 100000)  # Enough for multiple
        PLAYERS[0].set_president_of(state, 0, True)

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        setup_acquisition_phase_py(state)

        assert get_offer_count(state) >= 2
        assert_invariants(state, "Before acquisitions")

        # Apply accept and pass actions
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)

        # Accept first offer
        for i in range(layout['acquisition_start'], len(mask)):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Accept first")
                assert_invariants(state, "After first accept")
                break

        # Pass or accept remaining offers until phase completes
        max_iterations = 10
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ACQUISITION and iterations < max_iterations:
            mask = get_valid_action_mask(state)
            # Try to pass (use acq_pass for ACQUISITION phase)
            apply_action_and_verify(state, layout['acq_pass'], f"Action {iterations+1}")
            assert_invariants(state, f"After action {iterations+1}")
            iterations += 1

        # Process through ISSUE_SHARES phase (pass on all corps)
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ISSUE_SHARES and iterations < max_iterations:
            apply_action_and_verify(state, layout['issue_pass'], f"Issue pass {iterations+1}")
            assert_invariants(state, f"After issue pass {iterations+1}")
            iterations += 1

        # Should eventually complete and transition
        assert state.get_phase() == GamePhases.PHASE_INVEST

    def test_zone_merge_at_phase_transition(self):
        """Zone merging happens correctly at ACQUISITION phase transition."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS
        from phases.acquisition import transition_to_closing_py
        from core.data import CORP_NAMES

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up acquisition proceeds and acquisition zone companies
        player = PLAYERS[0]
        corp = CORPS[0]

        corp.set_active(state, True)

        initial_player_cash = player.get_cash(state)
        initial_corp_cash = corp.get_cash(state)

        # Add proceeds from selling
        player.add_acquisition_proceeds(state, 35)
        corp.set_acquisition_proceeds(state, 45)

        # Add company to acquisition zone
        COMPANIES[0].transfer_to_corp_acquisition(state, 0)

        # Verify setup
        assert player.get_acquisition_proceeds(state) == 35
        assert corp.get_acquisition_proceeds(state) == 45
        # Company should be in corp 0's acquisition zone
        assert COMPANIES[0].is_in_corp_acquisition(state, 0)

        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        assert_invariants(state, "Before transition")

        # Transition to next phase
        transition_to_closing_py(state)

        # Verify proceeds merged to cash
        assert player.get_cash(state) == initial_player_cash + 35, "Player proceeds not merged"
        assert corp.get_cash(state) == initial_corp_cash + 45, "Corp proceeds not merged"

        # Verify proceeds cleared
        assert player.get_acquisition_proceeds(state) == 0, "Player proceeds not cleared"
        assert corp.get_acquisition_proceeds(state) == 0, "Corp proceeds not cleared"

        # Verify company merged to owned
        assert not COMPANIES[0].is_in_corp_acquisition(state, 0), "Company still in acquisition zone"
        assert COMPANIES[0].get_owner_id(state) == 0, "Company ownership changed"
        assert corp.owns_company(state, 0), "Corp doesn't own company after merge"

        assert_invariants(state, "After transition with merges")


# =============================================================================
# INCOME PHASE INTEGRATION
# =============================================================================

class TestIncomeIntegration:
    """Integration tests verifying INCOME phase maintains invariants throughout."""

    def test_closing_to_income_transition_maintains_invariants(self):
        """CLOSING -> INCOME phase transition maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py
        from phases.income import apply_income_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        # Enter ACQUISITION phase, then transition to CLOSING
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        transition_to_closing_py(state)

        assert state.get_phase() == GamePhases.PHASE_CLOSING
        assert_invariants(state, "After transition to CLOSING")

        # Execute auto-close (transitions to INCOME in fresh game with no negative-income companies)
        apply_closing_auto_py(state)

        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert_invariants(state, "After CLOSING -> INCOME")

        # Apply income (transitions to DIVIDENDS, then END_CARD since no active corps)
        apply_income_py(state)

        # No active corps -> DIVIDENDS -> END_CARD immediately
        assert state.get_phase() == GamePhases.PHASE_END_CARD
        assert_invariants(state, "After INCOME -> DIVIDENDS -> END_CARD")

    def test_income_with_active_corp_maintains_invariants(self):
        """INCOME with active corps maintains invariants through DIVIDENDS."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.income import apply_income_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 5)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, "Before INCOME")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        # Should be in DIVIDENDS with active corp
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        assert TURN.get_dividend_corp(state) == 0
        assert_invariants(state, "After INCOME -> DIVIDENDS")

    def test_income_bankruptcy_maintains_invariants(self):
        """Corp bankruptcy during INCOME maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.market import MARKET
        from phases.income import apply_income_py
        from core.data import COMPANY_NAME_TO_ID

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set high CoO for negative income
        TURN.set_coo_level(state, 6)

        # Set up corp with negative income (will go bankrupt)
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 1)  # Not enough to cover negative income
        corp.set_price_index(state, 10)
        MARKET.set_space_available(state, 10, False)

        # KK: income=5, stars=3. At CoO level 6, 3-star CoO=7. Adjusted = -2
        kk = COMPANY_NAME_TO_ID["KK"]
        COMPANIES[kk].transfer_to_corp(state, 0)
        corp.set_owns_company(state, kk, True)

        assert_invariants(state, "Before INCOME with bankruptcy setup")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        # Corp should be bankrupt
        assert not corp.is_active(state), "Corp should be bankrupt"
        assert_invariants(state, "After INCOME with bankruptcy")

    def test_income_multiple_bankruptcies_maintain_invariants(self):
        """Multiple corp bankruptcies during INCOME maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.market import MARKET
        from phases.income import apply_income_py
        from core.data import COMPANY_NAME_TO_ID

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set high CoO for negative income
        TURN.set_coo_level(state, 6)

        # Set up multiple corps with negative income
        companies = [COMPANY_NAME_TO_ID["KK"], COMPANY_NAME_TO_ID["DR"], COMPANY_NAME_TO_ID["BY"]]
        for corp_id, cid in zip([0, 1, 2], companies):
            corp = CORPS[corp_id]
            corp.set_active(state, True)
            corp.set_cash(state, 0)  # Will go bankrupt
            corp.set_price_index(state, 10 + corp_id)
            MARKET.set_space_available(state, 10 + corp_id, False)

            COMPANIES[cid].transfer_to_corp(state, corp_id)
            corp.set_owns_company(state, cid, True)

        assert_invariants(state, "Before INCOME with multiple bankruptcies")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        # All corps should be bankrupt
        for corp_id in [0, 1, 2]:
            assert not CORPS[corp_id].is_active(state), f"Corp {corp_id} should be bankrupt"

        assert_invariants(state, "After INCOME with multiple bankruptcies")

    def test_full_turn_cycle_through_income(self):
        """Full turn cycle: INVEST -> WRAP_UP -> ACQUISITION -> CLOSING -> INCOME -> DIVIDENDS."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp for full dividend cycle
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 5)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, "Initial state")
        assert TURN.get_turn_number(state) == 1

        layout = get_action_layout(3)

        # INVEST: all players pass to trigger WRAP_UP
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")
            assert_invariants(state, f"After pass {i+1}")

        # After all passes: WRAP_UP -> ACQUISITION
        # With active corp, ACQUISITION may have offers to process
        # Pass through any ACQUISITION offers
        max_iterations = 20
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ACQUISITION and iterations < max_iterations:
            apply_action_and_verify(state, layout['acq_pass'], f"Pass acquisition {iterations+1}")
            assert_invariants(state, f"After acquisition pass {iterations+1}")
            iterations += 1

        # After ACQUISITION completes: CLOSING (auto-executes) -> INCOME (auto-executes) -> DIVIDENDS
        # Since no negative-income companies: CLOSING has no offers and auto-transitions
        # INCOME is non-player phase that auto-executes
        # With active corp, should stop at DIVIDENDS waiting for dividend decision
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS, f"Expected DIVIDENDS, got phase {state.get_phase()}"
        assert TURN.get_dividend_corp(state) == 0
        assert_invariants(state, "After full turn cycle to DIVIDENDS")

    def test_income_applies_to_all_entities(self):
        """INCOME applies income to corps, FI, and players; maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.fi import FI
        from entities.market import MARKET
        from phases.income import apply_income_py
        from core.data import COMPANY_NAME_TO_ID

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up corp with company
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 5)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        cdg = COMPANY_NAME_TO_ID["CDG"]
        COMPANIES[cdg].transfer_to_corp(state, 0)
        corp.set_owns_company(state, cdg, True)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        # Set up player with company
        mad = COMPANY_NAME_TO_ID["MAD"]
        COMPANIES[mad].transfer_to_player(state, 1)
        PLAYERS[1].set_owns_company(state, mad, True)

        # Set up FI with company
        fra = COMPANY_NAME_TO_ID["FRA"]
        COMPANIES[fra].transfer_to_fi(state)
        FI.set_owns_company(state, fra, True)

        # Record starting values
        corp_cash_before = corp.get_cash(state)
        corp_income = corp.calculate_income(state)
        player_cash_before = PLAYERS[1].get_cash(state)
        player_income = PLAYERS[1].get_income(state)
        fi_cash_before = FI.get_cash(state)
        fi_income = FI.calculate_income(state)

        assert_invariants(state, "Before INCOME")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        # Verify income was applied
        assert corp.get_cash(state) == corp_cash_before + corp_income, "Corp income not applied"
        assert PLAYERS[1].get_cash(state) == player_cash_before + player_income, "Player income not applied"
        assert FI.get_cash(state) == fi_cash_before + fi_income, "FI income not applied"

        assert_invariants(state, "After INCOME")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_income_integration_player_counts(self, num_players):
        """INCOME phase works correctly for all player counts."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.income import apply_income_py

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set up active corp
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 5)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, f"Before INCOME ({num_players}p)")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        assert_invariants(state, f"After INCOME ({num_players}p)")

    def test_income_after_closing_with_offers_maintains_invariants(self):
        """CLOSING with offers -> INCOME transition maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.market import MARKET
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py, apply_closing_action_py, get_close_offer_count_py
        from phases.income import apply_income_py
        from core.actions import ACTION_CLOSE_PY

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set high CoO for negative income companies
        TURN.set_coo_level(state, 6)

        # Give player 0 a negative income company
        PLAYERS[0].set_owns_company(state, 0, True)  # Company 0: 1 star, negative adjusted income

        # Set up active corp for DIVIDENDS phase
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 5)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, "Initial setup")

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)

        # Transition to CLOSING
        transition_to_closing_py(state)
        assert state.get_phase() == GamePhases.PHASE_CLOSING
        assert_invariants(state, "After transition to CLOSING")

        # Execute auto-close (generates offers for player's negative-income company)
        apply_closing_auto_py(state)

        # Should have close offers
        offer_count = get_close_offer_count_py(state)
        assert offer_count > 0, "Should have close offers"
        assert_invariants(state, "In CLOSING with offers")

        # Accept close offer
        result = apply_closing_action_py(state, ACTION_CLOSE_PY)
        assert result == 0, "Close action should succeed"

        # Should be in INCOME phase
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert_invariants(state, "After CLOSING -> INCOME")

        # Apply income
        apply_income_py(state)

        # Should be in DIVIDENDS with active corp
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        assert TURN.get_dividend_corp(state) == 0
        assert_invariants(state, "After INCOME -> DIVIDENDS")


# =============================================================================
# CLOSING PHASE INTEGRATION
# =============================================================================

class TestClosingIntegration:
    """Integration tests verifying CLOSING phase maintains invariants throughout."""

    def test_closing_with_no_offers_flow(self):
        """ACQUISITION->CLOSING->INVEST flow with no close offers (all positive income)."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        layout = get_action_layout(3)

        # Pass all players in INVEST to trigger WRAP_UP -> ACQUISITION
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")

        # After all passes, fresh game goes WRAP_UP -> ACQUISITION -> CLOSING -> INVEST
        # No negative-income companies in fresh game = no close offers = direct transition
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After ACQUISITION->CLOSING->INVEST with no offers")

    def test_closing_with_accept_flow(self):
        """CLOSING flow with accept: player closes negative-income company."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py, apply_closing_action_py, get_close_offer_count_py
        from core.actions import ACTION_CLOSE_PY
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up state with negative-income company owned by player
        # High CoO level makes companies have negative adjusted income
        TURN.set_coo_level(state, 6)  # Level 6: Red=$6, Orange=$4 CoO

        # Give player 0 a red company (company 0 has 1 star = red)
        # Company 0: income $2, stars 1 (red), face value $1
        # At CoO level 6: Red CoO = $6, so adjusted = $2 - $6 = -$4
        PLAYERS[0].set_owns_company(state, 0, True)

        # Set up corp 0 (Junkyard Scrappers) as active to test JS bonus
        CORPS[0].set_active(state, True)
        CORPS[0].set_cash(state, 100)
        initial_js_cash = CORPS[0].get_cash(state)

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        initial_turn = TURN.get_turn_number(state)

        assert_invariants(state, "Before transition to CLOSING")

        # Transition to CLOSING
        transition_to_closing_py(state)
        assert state.get_phase() == GamePhases.PHASE_CLOSING

        # Execute auto-close phase entry (FI/receivership auto-close + offer generation)
        apply_closing_auto_py(state)

        # Verify we have close offers (player owns negative-income company)
        assert get_close_offer_count_py(state) > 0, "Should have at least one close offer"
        assert TURN.get_closing_company(state) >= 0, "Should have an active close offer"

        assert_invariants(state, "In CLOSING with active offer")

        # Accept the close offer
        result = apply_closing_action_py(state, ACTION_CLOSE_PY)
        assert result == 0, "Close action should succeed"

        # Should transition to INCOME (no more offers after closing the only one)
        # Note: CLOSING now transitions to INCOME, not directly to INVEST
        # Turn number is incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == initial_turn  # Not incremented yet

        # Verify company was closed
        assert not PLAYERS[0].owns_company(state, 0), "Company should be closed"
        assert COMPANIES[0].is_removed(state), "Company should be removed from game"

        # Verify Junkyard Scrappers does NOT get bonus when player closes their own company
        # JS only gets bonus when JS itself closes one of its own companies
        assert CORPS[0].get_cash(state) == initial_js_cash, f"JS cash should be unchanged: {CORPS[0].get_cash(state)} != {initial_js_cash}"

        assert_invariants(state, "After CLOSING->INVEST with accept")

    def test_closing_with_pass_and_mandatory_close_flow(self):
        """CLOSING flow with pass triggers mandatory close for player with negative income+cash."""
        from tests.phases.conftest import assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py, apply_closing_action_py, get_close_offer_count_py
        from core.actions import ACTION_PASS_PY
        from entities.player import PLAYERS
        from entities.company import COMPANIES

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up state with negative-income company owned by player with low cash
        TURN.set_coo_level(state, 6)  # Level 6: Red=$6 CoO

        # Give player 0 a red company with negative adjusted income
        # Company 0: income $2, stars 1 (red) -> adjusted = $2 - $6 = -$4
        PLAYERS[0].set_owns_company(state, 0, True)

        # Set player cash low enough that income + cash < 0
        # Player income from company 0: -$4 (negative), so need cash < $4 for mandatory close
        PLAYERS[0].set_cash(state, 2)  # income (-$4) + cash ($2) = -$2 < 0

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        initial_turn = TURN.get_turn_number(state)

        assert_invariants(state, "Before transition to CLOSING")

        # Transition to CLOSING and execute auto-close
        transition_to_closing_py(state)
        apply_closing_auto_py(state)

        # Verify we have close offers
        offer_count = get_close_offer_count_py(state)
        assert offer_count > 0, "Should have at least one close offer"
        assert state.get_phase() == GamePhases.PHASE_CLOSING

        assert_invariants(state, "In CLOSING with offer")

        # Pass on the close offer - this should trigger mandatory close
        result = apply_closing_action_py(state, ACTION_PASS_PY)
        assert result == 0, "Pass action should succeed"

        # Should transition to INCOME after mandatory close
        # Note: CLOSING now transitions to INCOME, not directly to INVEST
        # Turn number is incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == initial_turn  # Not incremented yet

        # Verify mandatory close happened - company was forcibly closed
        assert not PLAYERS[0].owns_company(state, 0), "Company should be mandatorily closed"
        assert COMPANIES[0].is_removed(state), "Company should be removed from game"

        assert_invariants(state, "After CLOSING->INVEST with mandatory close")

    def test_full_turn_cycle_with_closing_offers(self):
        """Full turn cycle: INVEST->WRAP_UP->ACQUISITION->CLOSING(accept)->INVEST."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from phases.closing import apply_closing_action_py, get_close_offer_count_py
        from core.actions import ACTION_CLOSE_PY
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")
        assert TURN.get_turn_number(state) == 1

        # Set up negative-income scenario: give player 0 a company with negative adjusted income
        # Use a different company (company 1) so it doesn't create acquisition offers
        TURN.set_coo_level(state, 6)  # Level 6: Red=$6 CoO
        # Use transfer_to_player to properly update all ownership state
        COMPANIES[1].transfer_to_player(state, 0)  # Company 1: 1 star, $1 income -> -$5 adjusted

        layout = get_action_layout(3)

        # INVEST: all players pass to trigger WRAP_UP
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"INVEST pass {i+1}")
            assert_invariants(state, f"After pass {i+1}")

        # After all passes: WRAP_UP -> ACQUISITION (no offers) -> CLOSING with offer
        assert state.get_phase() == GamePhases.PHASE_CLOSING
        assert get_close_offer_count_py(state) > 0, "Should have close offer"
        assert TURN.get_closing_company(state) >= 0, "Should have active close offer"

        # Accept the close offer
        result = apply_closing_action_py(state, ACTION_CLOSE_PY)
        assert result == 0, "Close accept should succeed"

        # Verify: CLOSING -> INCOME (turn NOT yet incremented - that happens in TEMP_END_TURN)
        # Note: When using apply_closing_action_py directly, we get INCOME
        # Turn is incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == 1, "Turn not yet incremented"
        assert not PLAYERS[0].owns_company(state, 1), "Company should be closed"

        assert_invariants(state, "After full turn cycle with closing")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_closing_integration_player_counts(self, num_players):
        """ACQUISITION->CLOSING->INVEST flow works for all player counts."""
        from tests.phases.conftest import assert_invariants
        from phases.acquisition import transition_to_closing_py
        from phases.closing import apply_closing_auto_py, apply_closing_action_py, get_close_offer_count_py
        from core.actions import ACTION_CLOSE_PY
        from entities.player import PLAYERS
        from entities.company import COMPANIES

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        assert_invariants(state, f"Initial state ({num_players}p)")

        # Set up negative-income scenario
        TURN.set_coo_level(state, 6)  # Level 6: Red=$6 CoO
        PLAYERS[0].set_owns_company(state, 0, True)  # Company 0: 1 star, $1 income

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        initial_turn = TURN.get_turn_number(state)

        assert_invariants(state, f"Before transition ({num_players}p)")

        # Transition to CLOSING
        transition_to_closing_py(state)
        assert state.get_phase() == GamePhases.PHASE_CLOSING

        # Execute auto-close
        apply_closing_auto_py(state)

        # Verify offers generated
        assert get_close_offer_count_py(state) > 0, "Should have close offer"
        assert_invariants(state, f"In CLOSING ({num_players}p)")

        # Accept close offer
        result = apply_closing_action_py(state, ACTION_CLOSE_PY)
        assert result == 0, "Close action should succeed"

        # Verify transition: CLOSING -> INCOME (not directly to INVEST)
        # Turn number incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == initial_turn  # Not incremented yet
        assert not PLAYERS[0].owns_company(state, 0), "Company should be closed"

        assert_invariants(state, f"After CLOSING ({num_players}p)")

    def test_acquisition_accept_then_closing_accept(self):
        """Accept acquisition offer, then accept close offer in same turn."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from phases.acquisition import setup_acquisition_phase_py, get_offer_count
        from phases.closing import apply_closing_action_py, get_close_offer_count_py
        from core.actions import ACTION_CLOSE_PY
        from entities.player import PLAYERS
        from entities.company import COMPANIES
        from entities.corp import CORPS

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert_invariants(state, "Initial state")

        # Set up state for BOTH acquisition offer AND close offer:
        # 1. Player 0 owns company 0 (for acquisition)
        # 2. Player 0 owns company 1 with negative adjusted income (for close offer)
        # 3. Corp 0 is active with president player 0 (to make acquisition offer)

        COMPANIES[0].transfer_to_player(state, 0)  # Company 0 for acquisition
        PLAYERS[0].set_owns_company(state, 1, True)  # Company 1 for close offer

        CORPS[0].set_active(state, True)
        CORPS[0].set_cash(state, 50000)
        PLAYERS[0].set_president_of(state, 0, True)

        # Set high CoO level so company 1 has negative income
        TURN.set_coo_level(state, 6)  # Level 6: Red=$6 CoO

        # Enter ACQUISITION phase
        TURN.set_phase(state, GamePhases.PHASE_ACQUISITION)
        setup_acquisition_phase_py(state)

        initial_turn = TURN.get_turn_number(state)

        # Verify acquisition offers exist
        offer_count = get_offer_count(state)
        assert offer_count > 0, "Should have acquisition offers"
        assert_invariants(state, "Before acquisition action")

        # Find and accept acquisition offer
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)

        action_idx = None
        for i in range(layout['acquisition_start'], len(mask)):
            if mask[i] == 1.0:
                action_idx = i
                break

        assert action_idx is not None, "Should have valid acquisition action"
        apply_action_and_verify(state, action_idx, "Accept acquisition")

        # Process remaining offers by passing until CLOSING phase
        max_iterations = 20
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ACQUISITION and iterations < max_iterations:
            apply_action_and_verify(state, layout['acq_pass'], f"Pass acquisition {iterations+1}")
            iterations += 1

        # Should now be in CLOSING phase
        assert state.get_phase() == GamePhases.PHASE_CLOSING

        # Verify close offers exist (company 1 has negative income)
        assert get_close_offer_count_py(state) > 0, "Should have close offers"
        assert TURN.get_closing_company(state) >= 0, "Should have active close offer"

        assert_invariants(state, "In CLOSING after acquisition")

        # Accept close offer
        result = apply_closing_action_py(state, ACTION_CLOSE_PY)
        assert result == 0, "Close action should succeed"

        # Verify final state: INCOME phase (CLOSING now transitions to INCOME)
        # Turn number incremented in TEMP_END_TURN phase (not reached yet)
        assert state.get_phase() == GamePhases.PHASE_INCOME
        assert TURN.get_turn_number(state) == initial_turn  # Not incremented yet

        # Company 1 should be closed
        assert not PLAYERS[0].owns_company(state, 1), "Company 1 should be closed"

        assert_invariants(state, "After both acquisition and closing accepts")


# =============================================================================
# DIVIDENDS PHASE INTEGRATION
# =============================================================================

class TestDividendsIntegration:
    """Integration tests verifying DIVIDENDS phase maintains invariants throughout."""

    def test_income_to_dividends_maintains_invariants(self):
        """INCOME -> DIVIDENDS phase transition maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.income import apply_income_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp for dividends
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 8)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, "Initial state")

        # Enter INCOME phase
        TURN.set_phase(state, GamePhases.PHASE_INCOME)
        apply_income_py(state)

        # Should be in DIVIDENDS
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        assert TURN.get_dividend_corp(state) == 0
        assert_invariants(state, "After INCOME -> DIVIDENDS")

    def test_dividend_action_maintains_invariants(self):
        """Dividend payment action maintains invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 10)
        corp.set_unissued_shares(state, 3)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[1].set_shares(state, 0, 2)
        PLAYERS[0].set_president_of(state, 0, True)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before dividend action")

        # Apply dividend action through driver
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)

        # Find valid dividend action
        dividend_idx = None
        for i in range(layout['dividend_base'], layout['dividend_base'] + 26):
            if mask[i] == 1.0:
                dividend_idx = i
                break

        assert dividend_idx is not None, "Should have valid dividend action"
        apply_action_and_verify(state, dividend_idx, "Dividend action")

        assert_invariants(state, "After dividend action")

    def test_multiple_corps_maintain_invariants(self):
        """Processing multiple corps maintains invariants throughout."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py
        from core.data import get_corp_share_count

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up three active corps at different prices
        # Corp 0, 1 have 7 shares; Corp 2 has 6 shares
        for corp_id in range(3):
            corp = CORPS[corp_id]
            total_shares = get_corp_share_count(corp_id)
            corp.set_active(state, True)
            corp.set_cash(state, 100)
            corp.set_price_index(state, 15 - corp_id * 5)  # 15, 10, 5
            corp.set_stars(state, 10)
            corp.set_unissued_shares(state, total_shares - 4)  # Leave 4 for player
            corp.set_bank_shares(state, 0)
            corp.set_issued_shares(state, 4)
            MARKET.set_space_available(state, 15 - corp_id * 5, False)

            PLAYERS[corp_id].set_shares(state, corp_id, 4)
            PLAYERS[corp_id].set_president_of(state, corp_id, True)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before dividend cycle")

        layout = get_action_layout(3)

        # Process all three corps (highest price first: corp 0, then 1, then 2)
        corps_processed = []
        max_iterations = 10
        iterations = 0

        while state.get_phase() == GamePhases.PHASE_DIVIDENDS and iterations < max_iterations:
            current_corp = TURN.get_dividend_corp(state)
            corps_processed.append(current_corp)

            # Pay dividend
            mask = get_valid_action_mask(state)
            for i in range(layout['dividend_base'], layout['dividend_base'] + 26):
                if mask[i] == 1.0:
                    apply_action_and_verify(state, i, f"Dividend for corp {current_corp}")
                    break

            assert_invariants(state, f"After corp {current_corp} dividend")
            iterations += 1

        # Should have processed all three corps in price order
        assert corps_processed == [0, 1, 2], f"Expected [0, 1, 2], got {corps_processed}"

        # Process through ISSUE_SHARES phase (pass on all corps)
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ISSUE_SHARES and iterations < max_iterations:
            apply_action_and_verify(state, layout['issue_pass'], f"Issue pass {iterations+1}")
            assert_invariants(state, f"After issue pass {iterations+1}")
            iterations += 1

        # TEMP_END_TURN is auto-processed by driver -> returns to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2

    def test_receivership_auto_process_maintains_invariants(self):
        """Receivership corps auto-process while maintaining invariants."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up receivership corp at highest price (corp 0 has 7 shares)
        recv_corp = CORPS[0]
        recv_corp.set_active(state, True)
        recv_corp.set_in_receivership(state, True)
        recv_corp.set_cash(state, 100)
        recv_corp.set_price_index(state, 15)
        recv_corp.set_stars(state, 10)
        recv_corp.set_unissued_shares(state, 3)
        recv_corp.set_bank_shares(state, 4)  # All issued shares in bank
        recv_corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 15, False)

        # Set up player-controlled corp at lower price (corp 1 has 7 shares)
        player_corp = CORPS[1]
        player_corp.set_active(state, True)
        player_corp.set_cash(state, 100)
        player_corp.set_price_index(state, 10)
        player_corp.set_stars(state, 8)
        player_corp.set_unissued_shares(state, 3)
        player_corp.set_bank_shares(state, 0)
        player_corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 1, 4)
        PLAYERS[0].set_president_of(state, 1, True)

        initial_recv_cash = recv_corp.get_cash(state)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "After setup with receivership auto-process")

        # Receivership corp should have been auto-processed (skipped)
        # Active corp should be the player-controlled one
        assert TURN.get_dividend_corp(state) == 1, "Should skip receivership corp"

        # Receivership corp cash unchanged (paid $0)
        assert recv_corp.get_cash(state) == initial_recv_cash

        # Process player corp
        layout = get_action_layout(3)
        mask = get_valid_action_mask(state)
        for i in range(layout['dividend_base'], layout['dividend_base'] + 26):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Player corp dividend")
                break

        assert_invariants(state, "After all corps processed")

        # Process through ISSUE_SHARES phase (pass on all corps, receivership auto-issues)
        layout = get_action_layout(3)
        max_iterations = 10
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ISSUE_SHARES and iterations < max_iterations:
            apply_action_and_verify(state, layout['issue_pass'], f"Issue pass {iterations+1}")
            assert_invariants(state, f"After issue pass {iterations+1}")
            iterations += 1

        # TEMP_END_TURN is auto-processed by driver -> returns to INVEST
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2

    def test_bankruptcy_during_dividends_maintains_invariants(self):
        """Corp bankruptcy during price adjustment maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.company import COMPANIES
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up corp that will go bankrupt (low price, no stars)
        # Corp 0 has 7 total shares
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_price_index(state, 1)  # Very low, will drop to 0
        corp.set_stars(state, 0)  # No stars -> will drop
        corp.set_cash(state, 10)
        corp.set_unissued_shares(state, 4)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 3)
        MARKET.set_space_available(state, 1, False)

        # Give corp a company (bankruptcy needs to liquidate)
        COMPANIES[0].transfer_to_corp(state, 0)
        corp.set_owns_company(state, 0, True)

        PLAYERS[0].set_shares(state, 0, 3)
        PLAYERS[0].set_president_of(state, 0, True)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before bankruptcy")

        # Pay $0 dividend (triggers price adjustment -> bankruptcy)
        apply_dividend_action_py(state, 0)

        # Corp should be bankrupt
        assert not corp.is_active(state), "Corp should be bankrupt"
        assert_invariants(state, "After bankruptcy")

    def test_full_turn_cycle_through_dividends(self):
        """Full turn cycle: INVEST->WRAP_UP->ACQUISITION->CLOSING->INCOME->DIVIDENDS->TEMP_END_TURN."""
        from tests.phases.conftest import apply_action_and_verify, assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp for full cycle (corp 0 has 7 total shares)
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 10)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, "Initial state")
        assert TURN.get_turn_number(state) == 1

        layout = get_action_layout(3)

        # INVEST: all players pass
        for i in range(3):
            apply_action_and_verify(state, layout['pass_invest'], f"Pass {i+1}")

        # Pass through any ACQUISITION offers
        max_iterations = 20
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ACQUISITION and iterations < max_iterations:
            apply_action_and_verify(state, layout['acq_pass'], f"Acq pass {iterations+1}")
            iterations += 1

        # Should reach DIVIDENDS (CLOSING auto-transitions through INCOME)
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS, f"Expected DIVIDENDS, got {state.get_phase()}"
        assert_invariants(state, "At DIVIDENDS phase")

        # Pay dividend
        mask = get_valid_action_mask(state)
        for i in range(layout['dividend_base'], layout['dividend_base'] + 26):
            if mask[i] == 1.0:
                apply_action_and_verify(state, i, "Dividend action")
                break

        # Process through ISSUE_SHARES phase (pass on all corps)
        max_iterations = 10
        iterations = 0
        while state.get_phase() == GamePhases.PHASE_ISSUE_SHARES and iterations < max_iterations:
            apply_action_and_verify(state, layout['issue_pass'], f"Issue pass {iterations+1}")
            assert_invariants(state, f"After issue pass {iterations+1}")
            iterations += 1

        # TEMP_END_TURN is auto-processed by driver -> returns to INVEST with turn incremented
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        assert_invariants(state, "After full turn cycle through DIVIDENDS")

    def test_dividend_payments_correct(self):
        """Dividend payments are correctly distributed to shareholders."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up corp with known share distribution (corp 0 has 7 total shares)
        # unissued=2, bank=1, P0=2, P1=1, P2=1 = 7
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 15)  # Higher price for larger max dividend
        corp.set_stars(state, 15)
        corp.set_unissued_shares(state, 2)
        corp.set_bank_shares(state, 1)
        corp.set_issued_shares(state, 5)
        MARKET.set_space_available(state, 15, False)

        # P0: 2 shares, P1: 1 share, P2: 1 share, bank: 1 share
        PLAYERS[0].set_shares(state, 0, 2)
        PLAYERS[1].set_shares(state, 0, 1)
        PLAYERS[2].set_shares(state, 0, 1)
        PLAYERS[0].set_president_of(state, 0, True)

        initial_cash = [PLAYERS[i].get_cash(state) for i in range(3)]
        initial_corp_cash = corp.get_cash(state)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before dividend")

        # Pay $3 per share
        dividend_amount = 3
        apply_dividend_action_py(state, dividend_amount)

        # Verify payments
        assert PLAYERS[0].get_cash(state) == initial_cash[0] + (2 * dividend_amount)  # 2 shares
        assert PLAYERS[1].get_cash(state) == initial_cash[1] + (1 * dividend_amount)  # 1 share
        assert PLAYERS[2].get_cash(state) == initial_cash[2] + (1 * dividend_amount)  # 1 share

        # Corp pays for all issued shares (including bank)
        total_payout = 5 * dividend_amount
        assert corp.get_cash(state) == initial_corp_cash - total_payout

        assert_invariants(state, "After dividend payment")

    def test_price_adjustment_maintains_invariants(self):
        """Share price adjustment during dividends maintains invariants."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up corp with stars that will cause price increase (corp 0 has 7 shares)
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 20)  # Many stars -> price should rise
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)
        MARKET.set_space_available(state, 11, True)
        MARKET.set_space_available(state, 12, True)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        initial_price = corp.get_price_index(state)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before price adjustment")

        # Pay $0 (to not affect cash bonus stars)
        apply_dividend_action_py(state, 0)

        # Price should have increased
        new_price = corp.get_price_index(state)
        assert new_price > initial_price, f"Price should increase: {new_price} > {initial_price}"

        # Old space freed, new space occupied
        assert MARKET.is_space_available(state, initial_price)
        assert not MARKET.is_space_available(state, new_price)

        assert_invariants(state, "After price adjustment")

    def test_dividends_to_game_over_when_end_card(self):
        """DIVIDENDS -> GAME_OVER when end card is flipped."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Set up active corp (corp 0 has 7 shares)
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 10)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        # Set end card as flipped
        TURN.set_end_card_flipped(state, True)

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert_invariants(state, "Before final dividend")

        # Pay dividend
        apply_dividend_action_py(state, 0)

        # Should transition to END_CARD (which will handle GAME_OVER when executed)
        assert state.get_phase() == GamePhases.PHASE_END_CARD
        assert_invariants(state, "After DIVIDENDS -> END_CARD")

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_dividends_integration_player_counts(self, num_players):
        """DIVIDENDS phase works correctly for all player counts."""
        from tests.phases.conftest import assert_invariants
        from entities.player import PLAYERS
        from entities.corp import CORPS
        from entities.market import MARKET
        from phases.dividends import setup_dividends_phase_py, apply_dividend_action_py

        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Set up active corp (corp 0 has 7 shares)
        corp = CORPS[0]
        corp.set_active(state, True)
        corp.set_cash(state, 100)
        corp.set_price_index(state, 10)
        corp.set_stars(state, 10)
        corp.set_unissued_shares(state, 3)
        corp.set_bank_shares(state, 0)
        corp.set_issued_shares(state, 4)
        MARKET.set_space_available(state, 10, False)

        PLAYERS[0].set_shares(state, 0, 4)
        PLAYERS[0].set_president_of(state, 0, True)

        assert_invariants(state, f"Before DIVIDENDS ({num_players}p)")

        # Enter DIVIDENDS phase
        TURN.set_phase(state, GamePhases.PHASE_DIVIDENDS)
        setup_dividends_phase_py(state)

        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        assert_invariants(state, f"In DIVIDENDS ({num_players}p)")

        # Pay dividend
        apply_dividend_action_py(state, 2)

        assert state.get_phase() == GamePhases.PHASE_END_CARD
        assert_invariants(state, f"After DIVIDENDS ({num_players}p)")
