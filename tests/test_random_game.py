"""
Random game tests with invariant checking.

Plays complete games using random valid actions, checking game invariants
after each action to catch bugs in the Cython game engine.
"""
import pytest
import numpy as np
from state import GameState
from driver import apply_action
from actions import get_valid_action_mask
from data import py_get_corp_share_count

# Constants
PHASE_GAME_OVER = 10
PHASE_BID_IN_AUCTION = 1
NUM_CORPS = 8
NUM_COMPANIES = 36
NUM_MARKET_SPACES = 27

# Corp share counts (JS=7, S=7, OS=6, SM=6, PR=5, DA=5, VM=4, SI=4)
CORP_SHARE_COUNTS = [py_get_corp_share_count(i) for i in range(NUM_CORPS)]


# =============================================================================
# INVARIANT CHECKING FUNCTIONS
# =============================================================================

def check_player_invariants(state, context=""):
    """Check all player-related invariants."""
    num_players = state.num_players

    for player_id in range(num_players):
        # Player cash >= 0
        cash = state.get_player_cash_py(player_id)
        assert cash >= 0, f"Player {player_id} has negative cash: {cash}. {context}"

        # Player net worth >= 0 (net worth = cash + face value of companies + share value)
        # This should never be negative since cash >= 0, face values >= 0, share values >= 0
        net_worth = state.get_player_net_worth_py(player_id)
        assert net_worth >= 0, f"Player {player_id} has negative net worth: {net_worth}. {context}"

        # Player shares >= 0 for all corps
        for corp_id in range(NUM_CORPS):
            shares = state.get_player_shares_py(player_id, corp_id)
            assert shares >= 0, f"Player {player_id} has negative shares in corp {corp_id}: {shares}. {context}"


def check_corp_invariants(state, context=""):
    """Check all corporation-related invariants."""
    num_players = state.num_players

    for corp_id in range(NUM_CORPS):
        is_active = state.is_corp_active_py(corp_id)
        in_receivership = state.is_corp_in_receivership_py(corp_id)

        # Check for president
        has_president = False
        for player_id in range(num_players):
            if state.is_player_president_py(player_id, corp_id):
                has_president = True
                break

        # Active corp cannot be in receivership AND have a president
        if is_active:
            assert not (in_receivership and has_president), \
                f"Corp {corp_id} is both in receivership and has a president. {context}"

        # Corp cash >= 0 for active corps
        if is_active:
            cash = state.get_corp_cash_py(corp_id)
            assert cash >= 0, f"Corp {corp_id} has negative cash: {cash}. {context}"

        # Share accounting: issued_shares = bank_shares + sum(player_shares)
        if is_active:
            issued = state.get_corp_issued_shares_py(corp_id)
            bank = state.get_corp_bank_shares_py(corp_id)
            player_shares = sum(state.get_player_shares_py(p, corp_id) for p in range(num_players))

            assert issued == bank + player_shares, \
                f"Corp {corp_id} share accounting mismatch: issued={issued}, bank={bank}, player_shares={player_shares}. {context}"

        # Total shares must equal the corp's share count (4-7 depending on corp)
        # This only applies to active corps - inactive corps have all shares reset
        if is_active:
            issued = state.get_corp_issued_shares_py(corp_id)
            unissued = state.get_corp_unissued_shares_py(corp_id)
            expected_total = CORP_SHARE_COUNTS[corp_id]
            assert issued + unissued == expected_total, \
                f"Corp {corp_id} share count mismatch: issued={issued} + unissued={unissued} = {issued + unissued}, expected {expected_total}. {context}"

        # Price index validity
        price_index = state.get_corp_price_index_py(corp_id)
        if is_active:
            # Active corps must have a valid price index 1-26
            # Index 0 = inactive/bankruptcy, Index 26 = price $75
            assert 1 <= price_index < NUM_MARKET_SPACES, \
                f"Active corp {corp_id} has invalid price index: {price_index}. {context}"

        # Acquisition proceeds >= 0
        proceeds = state.get_corp_acquisition_proceeds_py(corp_id)
        assert proceeds >= 0, f"Corp {corp_id} has negative acquisition proceeds: {proceeds}. {context}"

        # Active corps must own at least 1 company (from rules: "Owns: 1+ companies (never zero)")
        if is_active:
            company_count = sum(1 for c in range(NUM_COMPANIES) if state.corp_owns_company_py(corp_id, c))
            assert company_count >= 1, \
                f"Active corp {corp_id} owns no companies. {context}"

        # If corp is not in receivership, it must have a president
        if is_active and not in_receivership:
            assert has_president, \
                f"Active non-receivership corp {corp_id} has no president. {context}"

        # If corp is in receivership, all issued shares must be bank shares
        if is_active and in_receivership:
            issued = state.get_corp_issued_shares_py(corp_id)
            bank = state.get_corp_bank_shares_py(corp_id)
            assert issued == bank, \
                f"Receivership corp {corp_id} has player-owned shares: issued={issued}, bank={bank}. {context}"


def check_market_invariants(state, context=""):
    """Check market space consistency."""
    active_corps = []

    for corp_id in range(NUM_CORPS):
        if state.is_corp_active_py(corp_id):
            price_index = state.get_corp_price_index_py(corp_id)
            active_corps.append((corp_id, price_index))

    # Check that no two active corps share the same market space
    # Exception: multiple corps can be at index 0 (inactive - shouldn't happen for active)
    # or index 26 (price $75, can be on or off market)
    exclusive_indices = [pi for _, pi in active_corps if 0 < pi < NUM_MARKET_SPACES - 1]
    unique_indices = set(exclusive_indices)

    assert len(exclusive_indices) == len(unique_indices), \
        f"Multiple corps share exclusive market spaces (1-25): {active_corps}. {context}"

    # Verify market array consistency with corp price indices
    # Only indices 1-25 should be marked as taken; 0 and 26 allow multiple corps
    for corp_id, price_index in active_corps:
        if 0 < price_index < NUM_MARKET_SPACES - 1:
            # Market space should be marked as taken (0.0)
            is_available = state.is_market_space_available_py(price_index)
            assert not is_available, \
                f"Corp {corp_id} at price index {price_index} but market space is available. {context}"


def check_company_invariants(state, context=""):
    """Check that each company has exactly one location."""
    num_players = state.num_players

    for company_id in range(NUM_COMPANIES):
        locations = 0
        location_names = []

        # Check auction (available for auction)
        if state.is_company_for_auction_py(company_id):
            locations += 1
            location_names.append("auction")

        # Check revealed (unavailable, drawn this turn)
        if state.is_company_revealed_py(company_id):
            locations += 1
            location_names.append("revealed")

        # Check player ownership
        for player_id in range(num_players):
            if state.player_owns_company_py(player_id, company_id):
                locations += 1
                location_names.append(f"player_{player_id}")

        # Check corp ownership
        for corp_id in range(NUM_CORPS):
            if state.corp_owns_company_py(corp_id, company_id):
                locations += 1
                location_names.append(f"corp_{corp_id}")

        # Check corp acquisition pile (during acquisition phase)
        for corp_id in range(NUM_CORPS):
            if state.corp_has_acquisition_company_py(corp_id, company_id):
                locations += 1
                location_names.append(f"corp_{corp_id}_acquisition")

        # Check FI ownership
        if state.fi_owns_company_py(company_id):
            locations += 1
            location_names.append("fi")

        # Check removed
        if state.is_company_removed_py(company_id):
            locations += 1
            location_names.append("removed")

        # Check if in deck (hidden state)
        if state.is_company_in_deck_py(company_id):
            locations += 1
            location_names.append("deck")

        # A company must have exactly one location
        assert locations == 1, \
            f"Company {company_id} has {locations} locations: {location_names}. {context}"


def check_fi_invariants(state, context=""):
    """Check Foreign Investor invariants."""
    fi_cash = state.get_fi_cash_py()
    assert fi_cash >= 0, f"FI has negative cash: {fi_cash}. {context}"


def check_auction_invariants(state, context=""):
    """Check auction state consistency."""
    phase = state.phase

    if phase == PHASE_BID_IN_AUCTION:
        # During auction, should have valid auction state
        auction_company = state.get_auction_company_py()
        auction_price = state.get_auction_price_py()
        auction_bidder = state.get_auction_high_bidder_py()
        auction_starter = state.get_auction_starter_py()

        assert auction_company >= 0, \
            f"In auction phase but no auction company: {auction_company}. {context}"
        assert auction_price > 0, \
            f"In auction phase but auction price is 0 or negative: {auction_price}. {context}"
        # High bidder should always be set (starts as the auction starter)
        assert auction_bidder >= 0, \
            f"In auction phase but no high bidder: {auction_bidder}. {context}"
        assert auction_starter >= 0, \
            f"In auction phase but no auction starter: {auction_starter}. {context}"
        # High bidder must be a valid player
        assert auction_bidder < state.num_players, \
            f"In auction phase but high bidder is invalid: {auction_bidder}. {context}"


def check_game_flow_invariants(state, prev_turn, context=""):
    """Check game flow invariants."""
    current_turn = state.turn_number

    # Turn number should not decrease
    assert current_turn >= prev_turn, \
        f"Turn number decreased from {prev_turn} to {current_turn}. {context}"

    return current_turn


def check_invariants(state, prev_turn=0, context=""):
    """Run all invariant checks."""
    check_player_invariants(state, context)
    check_corp_invariants(state, context)
    check_market_invariants(state, context)
    check_company_invariants(state, context)
    check_fi_invariants(state, context)
    check_auction_invariants(state, context)
    return check_game_flow_invariants(state, prev_turn, context)


# =============================================================================
# RANDOM GAME TESTS
# =============================================================================

def play_random_game(num_players, seed=None, max_steps=10000):
    """
    Play a random game to completion.

    Returns tuple of (steps_taken, final_state).
    Raises AssertionError if any invariant is violated.
    """
    if seed is not None:
        np.random.seed(seed)

    state = GameState(num_players)
    state.setup_new_game(shuffle_seed=seed)
    step = 0
    prev_turn = 0

    while state.phase != PHASE_GAME_OVER and step < max_steps:
        # Get valid actions
        mask = get_valid_action_mask(state)
        valid_indices = np.where(mask == 1.0)[0]

        # Must have at least one valid action unless game over
        context = f"Step {step}, Phase {state.phase}, Turn {state.turn_number}"
        assert len(valid_indices) > 0, f"No valid actions but game not over. {context}"

        # Choose random action
        action = np.random.choice(valid_indices)

        # Apply action
        apply_action(state, action)
        step += 1

        # Check invariants
        context = f"Step {step}, Phase {state.phase}, Turn {state.turn_number}, Action {action}"
        prev_turn = check_invariants(state, prev_turn, context)

    if step >= max_steps:
        raise RuntimeError(f"Game did not complete within {max_steps} steps")

    return step, state


class TestRandomGame3Players:
    """Test random games with 3 players."""

    def test_single_game_completes(self):
        """A single random game completes without invariant violations."""
        steps, state = play_random_game(3, seed=42)
        assert steps > 0
        assert state.phase == PHASE_GAME_OVER

    def test_100_games_complete(self):
        """100 random games complete without invariant violations."""
        for i in range(100):
            steps, state = play_random_game(3, seed=i)
            assert steps > 0
            assert state.phase == PHASE_GAME_OVER


class TestRandomGame6Players:
    """Test random games with 6 players."""

    def test_single_game_completes(self):
        """A single random game completes without invariant violations."""
        steps, state = play_random_game(6, seed=42)
        assert steps > 0
        assert state.phase == PHASE_GAME_OVER

    def test_50_games_complete(self):
        """50 random games complete without invariant violations."""
        for i in range(50):
            steps, state = play_random_game(6, seed=i)
            assert steps > 0
            assert state.phase == PHASE_GAME_OVER


class TestRandomGameVariousPlayerCounts:
    """Test random games with various player counts."""

    @pytest.mark.parametrize("num_players", [3, 4, 5, 6])
    def test_10_games_per_player_count(self, num_players):
        """10 random games complete for each player count."""
        for i in range(10):
            steps, state = play_random_game(num_players, seed=i * 100 + num_players)
            assert steps > 0
            assert state.phase == PHASE_GAME_OVER
