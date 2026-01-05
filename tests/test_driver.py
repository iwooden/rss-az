"""
Tests for the game driver module.
"""
import pytest
from state import GameState
from driver import GameDriver, get_driver, apply_action
from actions import get_action_layout, decode_action_py


class TestDriverCreation:
    """Test driver instantiation."""

    def test_create_driver_3_players(self):
        """Driver can be created for 3 players."""
        driver = GameDriver(3)
        assert driver is not None

    def test_create_driver_6_players(self):
        """Driver can be created for 6 players."""
        driver = GameDriver(6)
        assert driver is not None

    def test_get_driver_caches(self):
        """get_driver returns cached instance."""
        driver1 = get_driver(3)
        driver2 = get_driver(3)
        assert driver1 is driver2

    def test_get_driver_different_counts(self):
        """Different player counts get different drivers."""
        driver3 = get_driver(3)
        driver4 = get_driver(4)
        assert driver3 is not driver4


class TestInvestPhaseActions:
    """Test INVEST phase action dispatch."""

    def test_pass_action(self):
        """Pass action increments consecutive passes."""
        state = GameState(3)
        state.phase = 0  # INVEST
        driver = get_driver(3)

        initial_passes = state.consecutive_passes
        layout = get_action_layout(3)

        # Action 0 is PASS in INVEST phase
        driver.apply_action(state, layout['pass_invest'])

        # Consecutive passes should have incremented
        assert state.consecutive_passes == initial_passes + 1

    def test_pass_advances_player(self):
        """Pass action advances to next player."""
        state = GameState(3)
        state.phase = 0  # INVEST
        driver = get_driver(3)

        initial_player = state.active_player
        layout = get_action_layout(3)

        driver.apply_action(state, layout['pass_invest'])

        # Active player should have advanced
        assert state.active_player == (initial_player + 1) % 3


class TestAutomaticPhaseTransitions:
    """Test automatic phase handling."""

    def test_wrap_up_runs_automatically(self):
        """WRAP_UP phase runs automatically after all pass in INVEST."""
        state = GameState(3)
        state.phase = 0  # INVEST

        driver = get_driver(3)
        layout = get_action_layout(3)

        # All players pass
        for _ in range(3):
            driver.apply_action(state, layout['pass_invest'])

        # After all pass, game cycles through all automatic phases
        # (WRAP_UP, ACQUISITION, CLOSING, INCOME, DIVIDENDS, END_CARD, ISSUE, IPO)
        # and returns to INVEST phase with consecutive_passes cleared
        assert state.phase == 0  # Back to INVEST
        assert state.consecutive_passes == 0  # Cleared after cycle


class TestModuleLevelApplyAction:
    """Test the module-level apply_action function."""

    def test_apply_action_function(self):
        """Module-level apply_action works correctly."""
        state = GameState(3)
        state.phase = 0  # INVEST

        initial_passes = state.consecutive_passes

        # Use module-level function
        apply_action(state, 0)  # Pass action

        assert state.consecutive_passes == initial_passes + 1


class TestActionDecoding:
    """Test that action decoding aligns with dispatch."""

    def test_all_invest_actions_decode_correctly(self):
        """All INVEST phase actions decode to expected types."""
        layout = get_action_layout(3)

        # Pass
        phase, action_type, _, _, _ = decode_action_py(0, 3)
        assert phase == 0  # INVEST
        assert action_type == 0  # PASS

        # Auction
        phase, action_type, slot, _, bid = decode_action_py(1, 3)
        assert phase == 0  # INVEST
        assert action_type == 1  # AUCTION
        assert slot == 0
        assert bid == 0

        # Buy share
        buy_base = layout['buy_share_base']
        phase, action_type, _, corp_id, _ = decode_action_py(buy_base, 3)
        assert phase == 0  # INVEST
        assert action_type == 2  # BUY_SHARE
        assert corp_id == 0

        # Sell share
        sell_base = layout['sell_share_base']
        phase, action_type, _, corp_id, _ = decode_action_py(sell_base, 3)
        assert phase == 0  # INVEST
        assert action_type == 3  # SELL_SHARE
        assert corp_id == 0

    def test_all_phases_decode_correctly(self):
        """Actions in different phases decode to correct phase."""
        layout = get_action_layout(3)

        # BID_IN_AUCTION
        phase, _, _, _, _ = decode_action_py(layout['leave_auction'], 3)
        assert phase == 1

        # ACQUISITION
        phase, _, _, _, _ = decode_action_py(layout['acq_price_base'], 3)
        assert phase == 3

        # CLOSING
        phase, _, _, _, _ = decode_action_py(layout['close_action'], 3)
        assert phase == 4

        # DIVIDENDS
        phase, _, _, _, _ = decode_action_py(layout['dividend_base'], 3)
        assert phase == 6

        # ISSUE
        phase, _, _, _, _ = decode_action_py(layout['issue_pass'], 3)
        assert phase == 8

        # IPO
        phase, _, _, _, _ = decode_action_py(layout['ipo_pass'], 3)
        assert phase == 9
