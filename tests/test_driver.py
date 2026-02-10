"""Tests for GameDriver action dispatch and validation."""

import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER, STATUS_OK_PY as STATUS_OK, STATUS_INVALID_PY as STATUS_INVALID, STATUS_GAME_OVER_PY as STATUS_GAME_OVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.company import COMPANIES
from entities.corp import CORPS
from entities.player import PLAYERS
from tests.phases.conftest import apply_and_verify_all, float_corp_for_test


class TestGetLegalMoves:
    """Test GameDriver.get_legal_moves() method."""

    @pytest.fixture
    def game_state(self):
        """Create initialized game state for testing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        return state

    def test_get_legal_moves_returns_numpy_array(self, game_state):
        """get_legal_moves should return numpy float32 array."""
        mask = DRIVER.get_legal_moves(game_state)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.float32

    def test_get_legal_moves_matches_action_mask(self, game_state):
        """get_legal_moves should return same result as get_valid_action_mask."""
        driver_mask = DRIVER.get_legal_moves(game_state)
        direct_mask = get_valid_action_mask(game_state)
        np.testing.assert_array_equal(driver_mask, direct_mask)

    def test_get_legal_moves_has_valid_actions(self, game_state):
        """Initial state should have at least one valid action."""
        mask = DRIVER.get_legal_moves(game_state)
        assert np.sum(mask) > 0, "Should have at least one valid action"


class TestApplyActionValidation:
    """Test action validation in apply_action."""

    @pytest.fixture
    def game_state(self):
        """Create initialized game state for testing."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        return state

    def test_invalid_action_index_negative(self, game_state):
        """Negative action index should return STATUS_INVALID."""
        result = DRIVER.apply_action(game_state, -1)
        assert result == STATUS_INVALID

    def test_invalid_action_index_too_large(self, game_state):
        """Action index beyond total_size should return STATUS_INVALID."""
        layout = get_action_layout(3)
        result = DRIVER.apply_action(game_state, layout['total_size'])
        assert result == STATUS_INVALID

    def test_invalid_action_not_in_mask(self, game_state):
        """Action with mask[idx]=0 should return STATUS_INVALID."""
        mask = DRIVER.get_legal_moves(game_state)
        # Find an invalid action (mask = 0)
        invalid_idx = None
        for i in range(len(mask)):
            if mask[i] == 0.0:
                invalid_idx = i
                break
        assert invalid_idx is not None, "Test requires at least one invalid action"
        result = DRIVER.apply_action(game_state, invalid_idx)
        assert result == STATUS_INVALID


class TestApplyActionInvestPhase:
    """Test apply_action dispatch for INVEST phase."""

    @pytest.fixture
    def invest_state(self):
        """Create game state in INVEST phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        assert state.get_phase() == GamePhases.PHASE_INVEST
        return state

    def test_valid_auction_action_returns_ok(self, invest_state):
        """Valid auction action should return STATUS_OK."""
        mask = DRIVER.get_legal_moves(invest_state)
        layout = get_action_layout(3)
        # Find first valid auction action
        auction_idx = None
        for i in range(layout['auction_base'], layout['buy_share_base']):
            if mask[i] == 1.0:
                auction_idx = i
                break
        if auction_idx is not None:
            apply_and_verify_all(invest_state, auction_idx)


class TestApplyActionBidPhase:
    """Test apply_action dispatch for BID_IN_AUCTION phase."""

    @pytest.fixture
    def bid_state(self):
        """Create game state in BID_IN_AUCTION phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        TURN.set_phase(state, GamePhases.PHASE_BID_IN_AUCTION)
        for company_id in range(36):
            if state.is_company_for_auction(company_id):
                TURN.set_auction_company(state, company_id)
                TURN.set_auction_price(state, 1)  # Face value
                break
        return state

    def test_leave_auction_returns_ok(self, bid_state):
        """Leave auction action should return STATUS_OK."""
        layout = get_action_layout(3)
        leave_idx = layout['leave_auction']
        mask = DRIVER.get_legal_moves(bid_state)
        if mask[leave_idx] == 1.0:
            apply_and_verify_all(bid_state, leave_idx)

    def test_valid_raise_bid_returns_ok(self, bid_state):
        """Valid raise bid action should return STATUS_OK."""
        mask = DRIVER.get_legal_moves(bid_state)
        layout = get_action_layout(3)
        for i in range(layout['raise_bid_base'], layout['acquisition_start']):
            if mask[i] == 1.0:
                apply_and_verify_all(bid_state, i)
                break


class TestMultiplePlayerCounts:
    """Test driver works correctly for different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_get_legal_moves_correct_size(self, num_players):
        """Mask size should be correct for each player count."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)
        mask = DRIVER.get_legal_moves(state)
        layout = get_action_layout(num_players)
        assert len(mask) == layout['total_size']

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_pass_action_works(self, num_players):
        """Pass action should return STATUS_OK."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)
        layout = get_action_layout(num_players)
        apply_and_verify_all(state, layout['pass_invest'])


# =============================================================================
# FORCED ACTION AUTO-APPLY TESTS
# =============================================================================

class TestForcedActionAutoApply:
    """Tests for GameDriver forced action auto-apply mechanism.

    The driver auto-applies actions when exactly 1 legal action exists,
    and auto-executes non-player phases (WRAP_UP, INCOME, etc.) when
    0 legal actions exist. These tests verify the mechanism itself.
    """

    def test_no_auto_apply_when_choice_exists(self):
        """History has exactly 1 entry when multiple legal actions remain.

        A single pass in a 3-player game doesn't exhaust all players, so
        the next player still has choices. No forced actions should fire.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        layout = get_action_layout(3)

        result = apply_and_verify_all(state, layout['pass_invest'])
        assert result.applied_count == 1, (
            f"Expected 1 history entry (no auto-apply), got {result.applied_count}"
        )
        assert result.get_action_at(0) == layout['pass_invest']

    @pytest.mark.parametrize("num_players", [3, 6])
    def test_non_player_phase_chain_produces_sentinels(self, num_players):
        """All players passing triggers WRAP_UP/ACQUISITION sentinels in history.

        When the last player passes, the driver auto-executes non-player phases
        (count=0 path). Sentinels (-100 for WRAP_UP, -101 for ACQUISITION) should
        appear in history alongside the player's pass action.
        """
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)
        layout = get_action_layout(num_players)
        pass_idx = layout['pass_invest']

        for _ in range(num_players - 1):
            apply_and_verify_all(state, pass_idx)

        result = apply_and_verify_all(state, pass_idx)
        assert state.get_phase() == GamePhases.PHASE_INVEST

        assert result.applied_count >= 3
        action_values = [result.get_action_at(i) for i in range(result.applied_count)]
        assert action_values[0] == pass_idx, "First action should be the player's pass"
        assert -100 in action_values, "WRAP_UP sentinel (-100) missing from history"
        assert -101 in action_values, "ACQUISITION sentinel (-101) missing from history"

    def test_history_structure_and_progression(self):
        """History entries have correct types, valid sentinels, and progressing state.

        Verifies the complete history contract for a forced action chain:
        - Each entry is (float32 ndarray, int)
        - Player actions are non-negative, sentinels are negative and known
        - State snapshots differ between steps (chain modifies state)
        - Turn number advances after WRAP_UP chain
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']
        assert TURN.get_turn_number(state) == 1

        for _ in range(2):
            apply_and_verify_all(state, pass_idx)
        result = apply_and_verify_all(state, pass_idx)

        # Entry types: (float32 ndarray, int)
        for i in range(result.applied_count):
            entry_state = result.history[i][0]
            entry_action = result.history[i][1]
            assert isinstance(entry_state, np.ndarray), f"Entry {i}: state is not ndarray"
            assert entry_state.dtype == np.float32, f"Entry {i}: state not float32"
            assert isinstance(entry_action, int), f"Entry {i}: action is not int"

        # Player action non-negative, sentinels negative and from known set
        assert result.get_action_at(0) >= 0, "Player action should be non-negative"
        sentinels = [
            result.get_action_at(i)
            for i in range(result.applied_count)
            if result.get_action_at(i) < 0
        ]
        assert len(sentinels) >= 2, f"Expected at least 2 sentinels, got {len(sentinels)}"
        for s in sentinels:
            assert s in (-100, -101, -102, -103, -105), f"Unknown sentinel value: {s}"

        # State snapshots progress (not identical across steps)
        first_state = result.history[0][0]
        second_state = result.history[1][0]
        assert not np.array_equal(first_state, second_state), (
            "State snapshots should differ between chain steps"
        )

        # Turn advanced
        assert TURN.get_turn_number(state) == 2

    def test_game_over_during_non_player_phase(self):
        """STATUS_GAME_OVER returned when END_CARD phase triggers game end.

        If the end card has already been flipped, the END_CARD non-player phase
        sets PHASE_GAME_OVER. The driver should detect this and return
        STATUS_GAME_OVER immediately.

        The corp is put in receivership so DIVIDENDS auto-processes (at $0)
        without needing player input, allowing the phase chain to reach END_CARD.

        Note: Uses raw DRIVER calls because the test setup (receivership without
        proper share cleanup) creates states that violate invariants intentionally.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Float a corp so the game doesn't end from _is_game_terminal
        float_corp_for_test(state, corp_id=0, par_index=10)

        # Put corp in receivership so DIVIDENDS auto-processes at $0
        # (otherwise DIVIDENDS needs player input for dividend amount)
        CORPS[0].set_in_receivership(state, True)
        PLAYERS[0].set_shares(state, 0, 0)  # Remove player shares for receivership

        # Flip the end card - next time END_CARD phase executes, game ends
        TURN.set_end_card_flipped(state, True)

        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']

        for _ in range(2):
            status = DRIVER.apply_action(state, pass_idx)
            assert status == STATUS_OK

        # Last pass should trigger the full chain including END_CARD
        history = []
        status = DRIVER.apply_action(state, pass_idx, history=history)
        assert status == STATUS_GAME_OVER, (
            f"Expected STATUS_GAME_OVER, got {status}; phase={state.get_phase()}"
        )
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER

        action_values = [entry[1] for entry in history]
        assert -105 in action_values, "END_CARD sentinel (-105) should be in history"

    def test_game_over_from_terminal_state_during_chain(self):
        """STATUS_GAME_OVER returned when _is_game_terminal triggers during chain.

        If no auction companies and no active corps exist, closing/acquisition
        sets PHASE_GAME_OVER. With all companies removed, PASS is the only legal
        action for every player, so one pass triggers a full chain of forced
        passes + non-player phases ending in GAME_OVER.

        Note: Uses raw DRIVER calls because removing all companies without
        updating deck count creates states that violate invariants intentionally.
        """
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Remove all companies from the game
        for cid in range(36):
            COMPANIES[cid].remove_from_game(state)

        # No active corps exist by default (none floated)
        # PASS is now the only legal action for all players, so the first
        # pass auto-applies the remaining passes and the full phase chain.

        layout = get_action_layout(3)
        pass_idx = layout['pass_invest']

        history = []
        status = DRIVER.apply_action(state, pass_idx, history=history)
        assert status == STATUS_GAME_OVER, (
            f"Expected STATUS_GAME_OVER from terminal state, got {status}"
        )
        assert state.get_phase() == GamePhases.PHASE_GAME_OVER

        # History should contain: player pass + forced passes + phase sentinels
        action_values = [entry[1] for entry in history]
        pass_count = sum(1 for a in action_values if a == pass_idx)
        assert pass_count == 3, f"Expected 3 passes (1 player + 2 forced), got {pass_count}"
        assert -100 in action_values, "WRAP_UP sentinel missing"
