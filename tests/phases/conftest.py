"""Shared fixtures and assertion helpers for phase tests.

Test Categorization for Auto-Apply (v2.1):
=========================================

Category 1: No Changes Needed
- Tests that only check final state outcomes
- Tests using apply_action_and_verify (already handles status)
- Examples: most buy/sell/bankruptcy tests

Category 2: Updated with Explicit Assertions
- Tests checking intermediate states now use apply_and_track()
- Assert `len(result.history) == 1` when no auto-apply expected
- Documents intent and catches regressions

Category 3: New Edge Case Tests
- TestAutoApplyEdgeCases in test_invest.py
- TestAutoApplyBehavior in test_bid_in_auction.py
- Cover: phase transitions, forced chains, error guards

Fixture Usage Guide:
-------------------
- apply_action_and_verify(state, action): Standard action with invariant checks
- apply_and_track(state, action): Returns ApplyTrackResult with history access
- Use apply_and_track when you need to:
  - Verify no auto-apply occurred (history length == 1)
  - Inspect intermediate states in a forced action chain
  - Verify specific action sequence in history
"""
import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases, CORP_NAMES, get_corp_share_count
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.market import MARKET
from entities.company import COMPANIES

from core.driver import STATUS_OK_PY as STATUS_OK, STATUS_INVALID_PY as STATUS_INVALID, STATUS_GAME_OVER_PY as STATUS_GAME_OVER


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_valid_mask(state, expected_actions=None, msg=""):
    """
    Assert action mask validity.

    Args:
        state: GameState to check
        expected_actions: Set of action indices that should be valid, or None for any
        msg: Additional context for assertion failure
    """
    mask = get_valid_action_mask(state)

    if expected_actions is not None:
        actual_valid = set(i for i in range(len(mask)) if mask[i] == 1.0)
        assert actual_valid == expected_actions, f"{msg}\nExpected: {expected_actions}\nActual: {actual_valid}"
    else:
        # Just verify at least one valid action exists
        assert np.sum(mask) > 0, f"{msg}\nNo valid actions in mask"


def assert_invariants(state, msg=""):
    """
    Assert game state invariants are maintained.

    Checks:
    - Total shares per corp = unissued + bank + all players
    - Player cash >= 0
    - Corp cash >= 0
    - Net worths >= 0
    - Auction row size <= num_players
    """
    num_players = state.get_num_players()

    # Share conservation for active corps
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            total = corp.get_unissued_shares(state) + corp.get_bank_shares(state)
            for p in range(num_players):
                total += PLAYERS[p].get_shares(state, corp_id)
            expected = get_corp_share_count(corp_id)
            assert total == expected, f"{msg}\nCorp {corp_id} share count: {total} != {expected}"

    # Player cash non-negative
    for p in range(num_players):
        cash = PLAYERS[p].get_cash(state)
        assert cash >= 0, f"{msg}\nPlayer {p} cash negative: {cash}"

    # Player net worth non-negative
    for p in range(num_players):
        net_worth = PLAYERS[p].get_net_worth(state)
        assert net_worth >= 0, f"{msg}\nPlayer {p} net worth negative: {net_worth}"

    # Corp cash non-negative for active corps
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            cash = corp.get_cash(state)
            assert cash >= 0, f"{msg}\nCorp {corp_id} cash negative: {cash}"

    # Auction row size check
    auction_count = sum(
        1 for cid in range(36)
        if state.is_company_for_auction(cid)
    )
    assert auction_count <= num_players, f"{msg}\nAuction row size {auction_count} > {num_players}"

    # Market boundary spaces must always be available
    # - Index 0 ($0): Bankruptcy space - corps that land here go bankrupt and leave
    # - Index 26 ($75): Maximum price - multiple corps can share ("no card" state)
    assert MARKET.is_space_available(state, 0), f"{msg}\nMarket space 0 ($0 bankruptcy) must always be available"
    assert MARKET.is_space_available(state, 26), f"{msg}\nMarket space 26 ($75 max) must always be available"


def apply_action_and_verify(state, action_idx, msg=""):
    """
    Apply action and verify invariants + mask validity.

    Returns the result status from DRIVER.apply_action.
    """
    # Verify action is valid before applying
    mask = get_valid_action_mask(state)
    assert mask[action_idx] == 1.0, f"{msg}\nAction {action_idx} not valid in current mask"

    result = DRIVER.apply_action(state, action_idx)
    assert result == STATUS_OK, f"{msg}\nAction {action_idx} failed with status {result}"

    assert_invariants(state, f"{msg}\nAfter action {action_idx}")

    # Don't check for valid actions in terminal phases (WRAP_UP, GAME_OVER have no actions)
    phase = state.get_phase()
    if phase not in [GamePhases.PHASE_WRAP_UP, GamePhases.PHASE_GAME_OVER]:
        assert np.sum(get_valid_action_mask(state)) > 0, f"{msg}\nNo valid actions after {action_idx}"

    return result


class ApplyTrackResult:
    """Result wrapper for apply_and_track() fixture."""

    def __init__(self, state, history, status, num_players):
        self.state = state              # Final state after all actions
        self.history = history          # List of (state_array, action_idx) tuples
        self.status = status            # Return status from apply_action
        self.applied_count = len(history)
        self._num_players = num_players

    def get_state_at(self, index):
        """Get state snapshot at position (supports negative indexing)."""
        return GameState.from_array(self.history[index][0], self._num_players)

    def get_action_at(self, index):
        """Get action at position (supports negative indexing)."""
        return self.history[index][1]

    @property
    def last_action(self):
        """Last action applied (convenience property)."""
        return self.history[-1][1] if self.history else None


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def game_state():
    """Base initialized game state in INVEST phase."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)
    assert state.get_phase() == GamePhases.PHASE_INVEST
    return state


@pytest.fixture
def invest_state(game_state):
    """Alias for clarity - game starts in INVEST."""
    return game_state


@pytest.fixture
def bid_state(game_state):
    """State with active auction in BID_IN_AUCTION phase."""
    # Find and apply first valid auction action to enter BID phase
    mask = get_valid_action_mask(game_state)
    layout = get_action_layout(3)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            DRIVER.apply_action(game_state, i)
            break
    assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
    assert TURN.get_auction_company(game_state) >= 0
    return game_state


@pytest.fixture
def trade_state():
    """State with active corp for buy/sell testing."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Manually activate corp 0 (JS) with tradeable shares
    # Corp 0 has 7 total shares: unissued(3) + bank(2) + player(2) = 7
    corp = CORPS[0]
    corp.set_active(state, True)
    corp.set_price_index(state, 10)
    corp.set_unissued_shares(state, 3)
    corp.set_bank_shares(state, 2)
    corp.set_issued_shares(state, 4)

    PLAYERS[0].set_shares(state, 0, 2)
    PLAYERS[0].set_cash(state, 100)
    PLAYERS[0].set_president_of(state, 0, True)

    MARKET.set_space_available(state, 10, False)

    return state


@pytest.fixture
def apply_and_track():
    """Fixture providing action application with full history tracking.

    Usage:
        result = apply_and_track(state, action_idx)
        assert result.applied_count >= 1
        intermediate = result.get_state_at(0)  # State before first action
    """
    def _apply(state, action_idx):
        history = []
        status = DRIVER.apply_action(state, action_idx, history=history)
        return ApplyTrackResult(state, history, status, state.get_num_players())
    return _apply


@pytest.fixture
def closing_offer_state():
    """Create game state with companies ready for close offers."""
    gs = GameState(num_players=3)
    gs.initialize_game(seed=42)

    # Set high CoO level so companies have negative income
    # Level 6: Red=$6, Orange=$4 CoO
    TURN.set_coo_level(gs, 6)

    return gs
