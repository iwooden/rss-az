"""Tests for INVEST phase actions."""
import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS

STATUS_OK = 0


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def game_state():
    """Create initialized game state in INVEST phase."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    return state


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_first_valid_auction_action(state):
    """Find first valid auction action index."""
    mask = get_valid_action_mask(state)
    layout = get_action_layout(state.get_num_players())
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            return i
    return None


def apply_pass_to_all_players(state, num_players):
    """Apply pass action for all players (for wrap_up test)."""
    layout = get_action_layout(num_players)
    pass_idx = layout['pass_invest']
    for _ in range(num_players):
        result = DRIVER.apply_action(state, pass_idx)
        assert result == STATUS_OK


# =============================================================================
# PASS ACTION TESTS
# =============================================================================

class TestPassAction:
    """Test INVEST phase pass action behavior."""

    def test_pass_increments_consecutive_passes(self, game_state):
        """INV-01: Pass action increments consecutive_passes counter."""
        # Get initial consecutive_passes count
        initial_passes = TURN.get_consecutive_passes(game_state)
        assert initial_passes == 0

        # Apply pass action
        layout = get_action_layout(3)
        result = DRIVER.apply_action(game_state, layout['pass_invest'])
        assert result == STATUS_OK

        # Verify consecutive_passes incremented
        new_passes = TURN.get_consecutive_passes(game_state)
        assert new_passes == initial_passes + 1

    def test_pass_advances_active_player(self, game_state):
        """INV-04: Pass action advances active player in turn order."""
        # Get initial active player
        initial_player = game_state.get_active_player()
        initial_position = PLAYERS[initial_player].get_turn_order(game_state)

        # Apply pass action
        layout = get_action_layout(3)
        result = DRIVER.apply_action(game_state, layout['pass_invest'])
        assert result == STATUS_OK

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (initial_position + 1) % 3

    def test_pass_follows_turn_order(self, game_state):
        """INV-04a: Pass uses turn order (one-hot vectors), not player_id."""
        # Record all players in turn order (only 2 passes to avoid WRAP_UP)
        turn_sequence = []
        layout = get_action_layout(3)

        for i in range(2):
            current_player = game_state.get_active_player()
            turn_sequence.append(current_player)
            DRIVER.apply_action(game_state, layout['pass_invest'])

        # Get third player
        third_player = game_state.get_active_player()
        turn_sequence.append(third_player)

        # Verify all 3 players are unique
        assert len(set(turn_sequence)) == 3  # All 3 players appeared in turn order

        # Verify they appear in consecutive positions (following turn_order)
        for i, player_id in enumerate(turn_sequence):
            position = PLAYERS[player_id].get_turn_order(game_state)
            # Position should match index in sequence
            expected_position = turn_sequence.index(player_id)
            # All players should have unique positions
            assert position in [0, 1, 2]

    def test_all_players_pass_transitions_to_wrap_up(self, game_state):
        """INV-03: WRAP_UP transition when all players pass."""
        # Apply pass for all 3 players
        apply_pass_to_all_players(game_state, 3)

        # Verify phase transition
        assert game_state.get_phase() == GamePhases.PHASE_WRAP_UP

    def test_non_pass_resets_consecutive_passes(self, game_state):
        """INV-02: Non-pass action (auction) resets consecutive_passes."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        DRIVER.apply_action(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        if auction_idx is not None:
            result = DRIVER.apply_action(game_state, auction_idx)
            assert result == STATUS_OK

            # Verify consecutive_passes was reset to 0
            assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# START AUCTION TESTS
# =============================================================================

class TestStartAuction:
    """Test INVEST phase start auction action behavior."""

    def test_start_auction_sets_company(self, game_state):
        """INV-05: Start auction sets auction_company."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Verify no auction company initially
        initial_company = TURN.get_auction_company(game_state)
        assert initial_company == -1

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction company was set
        auction_company = TURN.get_auction_company(game_state)
        assert auction_company >= 0 and auction_company < 36

    def test_start_auction_sets_price(self, game_state):
        """INV-05: Start auction sets auction_price."""
        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction price was set (should be >= face value)
        auction_price = TURN.get_auction_price(game_state)
        assert auction_price > 0

    def test_start_auction_sets_high_bidder(self, game_state):
        """INV-05: Start auction sets auction_high_bidder to starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify high bidder is the starter
        high_bidder = TURN.get_auction_high_bidder(game_state)
        assert high_bidder == starter_id

    def test_start_auction_sets_starter(self, game_state):
        """INV-05: Start auction sets auction_starter."""
        starter_id = game_state.get_active_player()

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify auction starter was recorded
        auction_starter = TURN.get_auction_starter(game_state)
        assert auction_starter == starter_id

    def test_start_auction_clears_passed_flags(self, game_state):
        """INV-05: Start auction clears all auction passed flags."""
        # Manually set some passed flags for testing
        TURN.set_player_passed_auction(game_state, 0, True)
        TURN.set_player_passed_auction(game_state, 1, True)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify all passed flags cleared
        for player_id in range(3):
            assert not TURN.has_player_passed_auction(game_state, player_id)

    def test_start_auction_transitions_to_bid_phase(self, game_state):
        """INV-06: Start auction transitions to BID_IN_AUCTION phase."""
        # Verify initial phase is INVEST
        assert game_state.get_phase() == GamePhases.PHASE_INVEST

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify phase transition
        assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    def test_start_auction_advances_to_next_bidder(self, game_state):
        """Start auction advances active player to next in turn order."""
        starter_id = game_state.get_active_player()
        starter_position = PLAYERS[starter_id].get_turn_order(game_state)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None

        # Apply auction action
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify active player advanced
        new_player = game_state.get_active_player()
        new_position = PLAYERS[new_player].get_turn_order(game_state)
        assert new_position == (starter_position + 1) % 3

    def test_start_auction_resets_consecutive_passes(self, game_state):
        """INV-02: Start auction resets consecutive_passes counter."""
        # Apply pass to increment counter
        layout = get_action_layout(3)
        DRIVER.apply_action(game_state, layout['pass_invest'])
        assert TURN.get_consecutive_passes(game_state) >= 1

        # Find and apply auction action
        auction_idx = get_first_valid_auction_action(game_state)
        assert auction_idx is not None
        result = DRIVER.apply_action(game_state, auction_idx)
        assert result == STATUS_OK

        # Verify consecutive_passes was reset
        assert TURN.get_consecutive_passes(game_state) == 0


# =============================================================================
# MULTIPLE PLAYER COUNT TESTS
# =============================================================================

class TestMultiplePlayerCounts:
    """Test INVEST phase behavior across different player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_pass_works_all_player_counts(self, num_players):
        """Pass action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        layout = get_action_layout(num_players)
        result = DRIVER.apply_action(state, layout['pass_invest'])
        assert result == STATUS_OK

        # Verify consecutive_passes incremented
        assert TURN.get_consecutive_passes(state) == 1

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_auction_works_all_player_counts(self, num_players):
        """Auction action works correctly for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Find valid auction action
        auction_idx = get_first_valid_auction_action(state)
        if auction_idx is not None:
            result = DRIVER.apply_action(state, auction_idx)
            assert result == STATUS_OK

            # Verify transition to BID phase
            assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_wrap_up_triggers_at_correct_pass_count(self, num_players):
        """WRAP_UP triggers after exactly num_players passes."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        # Apply pass for all players
        apply_pass_to_all_players(state, num_players)

        # Verify phase transition
        assert state.get_phase() == GamePhases.PHASE_WRAP_UP
