"""Shared fixtures and assertion helpers for phase tests.

Fixture Usage Guide:
-------------------
- apply_and_verify_all(state, action): Apply action, check invariants on all
  intermediate states (auto-applied forced actions). Returns ApplyTrackResult.
- assert_invariants(state, msg): Check state invariants directly (for tests
  that call phase-internal _py functions instead of the driver).
"""
import pytest
import numpy as np
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases, CORP_NAMES, CorpIndices, get_corp_share_count, get_company_stars
from entities.turn import TURN
from entities.player import PLAYERS, update_all_net_worths
from entities.corp import CORPS
from entities.market import MARKET
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from entities.deck import DECK

from core.driver import STATUS_OK_PY as STATUS_OK, STATUS_INVALID_PY as STATUS_INVALID, STATUS_GAME_OVER_PY as STATUS_GAME_OVER


# =============================================================================
# TEST UTILITIES
# =============================================================================

def float_corp_for_test(state, corp_id, company_id=None, player_id=0, par_index=10, float_shares=1):
    """
    Float a corporation for testing purposes.

    This is a convenience function that handles the common pattern of:
    1. Finding/assigning a company to a player
    2. Calling Corp.float_corp() to activate the corporation

    Args:
        state: GameState instance
        corp_id: Corporation ID to float (required)
        company_id: Company ID to use for floating. If None, draws from
                   the deck (properly updating deck tracking).
        player_id: Player who becomes president (default 0)
        par_index: Market price index for starting share price (default 10)
        float_shares: Shares each for player and bank (default 1)

    Returns:
        The company_id that was used (useful when company_id was None)
    """
    # Draw a company from the deck if none specified
    if company_id is None:
        company_id = DECK.draw(state)
        if company_id < 0:
            raise ValueError("Deck is empty, cannot draw company for floating")

    # Transfer company to player and float the corp
    COMPANIES[company_id].transfer_to_player(state, player_id)
    CORPS[corp_id].float_corp(state, player_id, company_id, par_index, float_shares)

    # Keep net worth fresh for invariant checks
    update_all_net_worths(state)

    return company_id


def setup_receivership_corp(state, corp_id, company_ids):
    """Float a corp and put it into receivership with the given companies.

    Args:
        state: GameState instance
        corp_id: Corporation ID to float
        company_ids: List of company IDs to assign. First is used for floating,
                    rest are transferred after.
    """
    float_corp_for_test(state, corp_id=corp_id, company_id=company_ids[0])
    PLAYERS[0].set_shares(state, corp_id, 0)
    for cid in company_ids[1:]:
        COMPANIES[cid].transfer_to_corp(state, corp_id)


def setup_player_private_offer(state, player_id, company_id, corp_id, corp_cash):
    """Set up a player-private -> corp acquisition offer.

    Transfers the company to the player, floats the corp with the player as
    president, sets corp cash, and generates acquisition offers.

    Args:
        state: GameState instance
        player_id: Player who owns the private company
        company_id: Company to transfer to the player
        corp_id: Corporation that will buy
        corp_cash: Cash to give the corp
    """
    from phases.acquisition import setup_acquisition_phase_py
    COMPANIES[company_id].transfer_to_player(state, player_id)
    float_corp_for_test(state, corp_id, player_id=player_id)
    CORPS[corp_id].set_cash(state, corp_cash)
    setup_acquisition_phase_py(state)
    assert_invariants(state, "After setup_player_private_offer")


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
    - Share conservation: unissued + bank + players = total per corp
    - Issued shares = bank + players per corp
    - Player cash >= 0, net worth >= 0
    - Corp cash >= 0, stars consistent, price index in [0, 26]
    - Active corp has >= 1 company
    - President has >= 1 share (non-receivership), no president (receivership)
    - FI cash >= 0
    - Auction row size <= num_players
    - Market boundary spaces available
    - Company locations valid, deck count consistent
    - Company ownership: player/corp-owned companies have valid owner
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

    # Corp stars consistency: stored stars must match computed value
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            stored = corp.get_stars(state)
            computed = sum(
                get_company_stars(cid) for cid in range(36)
                if corp.owns_company(state, cid) or corp.has_acquisition_company(state, cid)
            )
            cash = corp.get_cash(state)
            if cash > 0:
                computed += cash // 10
            if corp_id == CorpIndices.CORP_SI:
                computed += 2
            assert stored == computed, (
                f"{msg}\nCorp {corp_id} stars mismatch: "
                f"stored={stored} != computed={computed}"
            )

    # Corp income consistency: stored income must match calculated value
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            stored = corp.get_income(state)
            computed = corp.calculate_income(state)
            assert stored == computed, (
                f"{msg}\nCorp {corp_id} income mismatch: "
                f"stored={stored} != computed={computed}"
            )

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

    # FI cash non-negative
    fi_cash = FI.get_cash(state)
    assert fi_cash >= 0, f"{msg}\nForeign Investor cash negative: {fi_cash}"

    # Company location validity - every company has a known location
    for cid in range(36):
        loc = COMPANIES[cid].get_location(state)
        assert 0 <= loc <= 7, f"{msg}\nCompany {cid} has invalid location: {loc}"

    # Deck count matches companies with LOC_DECK location
    deck_remaining = DECK.get_remaining_count(state)
    deck_loc_count = sum(
        1 for cid in range(36)
        if COMPANIES[cid].get_location(state) == CompanyLocation.LOC_DECK
    )
    assert deck_remaining == deck_loc_count, (
        f"{msg}\nDeck count mismatch: deck entity says {deck_remaining} "
        f"but {deck_loc_count} companies have LOC_DECK"
    )

    # Issued shares = bank + player shares (for active corps)
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            issued = corp.get_issued_shares(state)
            bank = corp.get_bank_shares(state)
            player_held = sum(PLAYERS[p].get_shares(state, corp_id) for p in range(num_players))
            assert issued == bank + player_held, (
                f"{msg}\nCorp {corp_id} issued shares mismatch: "
                f"issued({issued}) != bank({bank}) + players({player_held})"
            )

    # President invariants for active, non-receivership corps
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state) and not corp.is_in_receivership(state):
            pres_id = corp.get_president_id(state)
            assert pres_id >= 0, (
                f"{msg}\nCorp {corp_id} active and not in receivership but has no president"
            )
            pres_shares = PLAYERS[pres_id].get_shares(state, corp_id)
            assert pres_shares >= 1, (
                f"{msg}\nCorp {corp_id} president (player {pres_id}) holds "
                f"{pres_shares} shares (must be >= 1)"
            )

    # Receivership means no president
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state) and corp.is_in_receivership(state):
            pres_id = corp.get_president_id(state)
            assert pres_id == -1, (
                f"{msg}\nCorp {corp_id} in receivership but has president: player {pres_id}"
            )

    # Active corp must have >= 1 company
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            company_count = corp.count_companies(state, include_acquisition=True)
            assert company_count >= 1, (
                f"{msg}\nCorp {corp_id} is active but has {company_count} companies"
            )

    # Corp price index in valid range for active corps
    for corp_id in range(8):
        corp = CORPS[corp_id]
        if corp.is_active(state):
            price_idx = corp.get_price_index(state)
            assert 0 <= price_idx <= 26, (
                f"{msg}\nCorp {corp_id} price index out of range: {price_idx}"
            )

    # Ghost deck entries must not have LOC_DECK location
    # Slots past deck_top are already-drawn or removed cards; their companies
    # should have been moved to another location (LOC_PLAYER, LOC_FI, etc.)
    for slot_idx, cid in DECK.get_ghost_entries(state):
        loc = COMPANIES[cid].get_location(state)
        assert loc != CompanyLocation.LOC_DECK, (
            f"{msg}\nCompany {cid} in ghost deck slot {slot_idx} "
            f"still has LOC_DECK location"
        )

    # Company ownership consistency: player/corp-owned companies must have
    # a valid owner, and that owner must actually list the company
    for cid in range(36):
        company = COMPANIES[cid]
        loc = company.get_location(state)
        owner_id = company.get_owner_id(state)

        if loc == CompanyLocation.LOC_PLAYER:
            assert 0 <= owner_id < num_players, (
                f"{msg}\nCompany {cid} at LOC_PLAYER has invalid owner_id: {owner_id}"
            )
            assert PLAYERS[owner_id].owns_company(state, cid), (
                f"{msg}\nCompany {cid} says LOC_PLAYER owner={owner_id} "
                f"but player doesn't list it"
            )
        elif loc == CompanyLocation.LOC_CORP:
            assert 0 <= owner_id < 8, (
                f"{msg}\nCompany {cid} at LOC_CORP has invalid owner_id: {owner_id}"
            )
            assert CORPS[owner_id].owns_company(state, cid), (
                f"{msg}\nCompany {cid} says LOC_CORP owner={owner_id} "
                f"but corp doesn't list it"
            )
        elif loc == CompanyLocation.LOC_CORP_ACQ:
            assert 0 <= owner_id < 8, (
                f"{msg}\nCompany {cid} at LOC_CORP_ACQ has invalid owner_id: {owner_id}"
            )


class ApplyTrackResult:
    """Result wrapper for apply_and_verify_all()."""

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
def bid_state(game_state):
    """State with active auction in BID_IN_AUCTION phase."""
    # Find and apply first valid auction action to enter BID phase
    mask = get_valid_action_mask(game_state)
    layout = get_action_layout(3)
    for i in range(layout['auction_base'], layout['buy_share_base']):
        if mask[i] == 1.0:
            apply_and_verify_all(game_state, i)
            break
    assert game_state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
    assert TURN.get_auction_company(game_state) >= 0
    return game_state


@pytest.fixture
def trade_state():
    """State with active corp for buy/sell testing."""
    state = GameState(num_players=3)
    state.initialize_game(seed=42)

    # Float corp 0 (JS) with 2 shares each to player and bank
    # This sets up: unissued(3), bank(2), issued(4), player 0 has 2 shares, price index 10
    float_corp_for_test(state, corp_id=0, par_index=10, float_shares=2)

    # Set up player cash for trading
    PLAYERS[0].set_cash(state, 100)
    update_all_net_worths(state)

    return state


def apply_and_verify_all(state, action_idx, msg="", expected_status=STATUS_OK):
    """Apply action and verify invariants on every intermediate state.

    Checks invariants on all intermediate states from the driver's
    auto-apply loop, not just the final state. Returns ApplyTrackResult
    for history inspection.

    Args:
        expected_status: Expected return status (default STATUS_OK).
            Use STATUS_GAME_OVER for game-ending actions.
    """
    # Verify action is valid before applying
    mask = get_valid_action_mask(state)
    assert mask[action_idx] == 1.0, f"{msg}\nAction {action_idx} not valid in current mask"

    history = []
    status = DRIVER.apply_action(state, action_idx, history=history)
    assert status == expected_status, f"{msg}\nAction {action_idx} returned status {status}, expected {expected_status}"

    # Check invariants on every intermediate state (captured BEFORE each action)
    for i, (state_array, action_id) in enumerate(history):
        intermediate = GameState.from_array(state_array, state.get_num_players())
        assert_invariants(intermediate,
            f"{msg}\nIntermediate state {i}/{len(history)}, "
            f"before action {action_id}")

    # Check invariants on final state (AFTER all actions)
    assert_invariants(state, f"{msg}\nFinal state after action chain")

    # Verify valid actions exist in non-terminal phases
    phase = state.get_phase()
    if phase not in [GamePhases.PHASE_WRAP_UP, GamePhases.PHASE_GAME_OVER]:
        assert np.sum(get_valid_action_mask(state)) > 0, f"{msg}\nNo valid actions after {action_idx}"

    return ApplyTrackResult(state, history, status, state.get_num_players())


@pytest.fixture
def closing_offer_state():
    """Create game state with companies ready for close offers."""
    gs = GameState(num_players=3)
    gs.initialize_game(seed=42)

    # Set high CoO level so companies have negative income
    # Level 6: Red=$6, Orange=$4 CoO
    TURN.set_coo_level(gs, 6)

    return gs
