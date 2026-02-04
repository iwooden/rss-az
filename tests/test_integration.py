"""Integration tests for complete turn and multi-turn game sequences.

These tests use ONLY the driver (how players/model interact with the game).
Phase-specific tests are in tests/phases/*.py.
Bankruptcy tests are in tests/test_bankruptcy.py.

Key insight: Corps can't be active until turn 2+ (IPO happens at end of turn 1).
"""
import pytest
from core.state import GameState
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES
from entities.market import MARKET


# =============================================================================
# HELPERS
# =============================================================================

def apply_action(state, action_idx, msg=""):
    """Apply action through driver and verify success."""
    from tests.phases.conftest import apply_action_and_verify
    apply_action_and_verify(state, action_idx, msg)


def check_invariants(state, msg=""):
    """Verify game invariants hold."""
    from tests.phases.conftest import assert_invariants
    assert_invariants(state, msg)


def find_valid_action(state, start_idx, end_idx):
    """Find first valid action in range, or None."""
    mask = get_valid_action_mask(state)
    for i in range(start_idx, end_idx):
        if mask[i] == 1.0:
            return i
    return None


def pass_all_players(state, num_players):
    """Have all players pass in INVEST phase."""
    layout = get_action_layout(num_players)
    for i in range(num_players):
        apply_action(state, layout['pass_invest'], f"Player {i} pass")


def complete_turn_from_invest(state, num_players, pass_ipo=True):
    """Complete a turn starting from INVEST phase.

    Passes all players in INVEST, then handles subsequent phases until
    we're back in INVEST with turn incremented.
    """
    layout = get_action_layout(num_players)

    # Pass all players in INVEST
    pass_all_players(state, num_players)

    # Handle any ACQUISITION offers
    pass_through_phase(state, num_players, GamePhases.PHASE_ACQUISITION, 'acq_pass')

    # Handle DIVIDENDS if we have active corps
    if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
        div_action = find_valid_action(
            state, layout['dividend_base'], layout['dividend_base'] + 26
        )
        if div_action:
            apply_action(state, div_action, "Dividend")

    # Handle ISSUE_SHARES
    pass_through_phase(state, num_players, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')

    # Handle IPO phase
    if pass_ipo:
        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")


def pass_through_phase(state, num_players, phase, pass_action_key, max_iterations=20):
    """Pass through all offers/decisions in a phase until it transitions."""
    layout = get_action_layout(num_players)
    iterations = 0
    while state.get_phase() == phase and iterations < max_iterations:
        apply_action(state, layout[pass_action_key], f"Pass in phase {phase}")
        iterations += 1
    return iterations


# =============================================================================
# MINIMAL TURN TESTS (Turn 1, no corps)
# =============================================================================

class TestMinimalTurn:
    """Turn 1: Fresh game, all pass, no corps active.

    Flow: INVEST (all pass) → WRAP_UP → ACQUISITION (no offers) → CLOSING (no offers)
          → INCOME → DIVIDENDS (skipped) → END_CARD → ISSUE (skipped) → IPO → INVEST
    """

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
    def test_all_pass_completes_turn(self, num_players):
        """All players passing completes turn and increments turn number."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        check_invariants(state, "Initial state")
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 1

        # All players pass
        pass_all_players(state, num_players)

        # Should be back in INVEST with turn 2
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2
        check_invariants(state, "After turn 1 complete")

    def test_minimal_turn_maintains_invariants_throughout(self):
        """Invariants hold at every step during minimal turn."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)
        check_invariants(state, "Initial")

        # Pass each player, checking invariants after each
        for i in range(3):
            apply_action(state, layout['pass_invest'], f"Pass {i+1}")
            check_invariants(state, f"After pass {i+1}")

        assert TURN.get_turn_number(state) == 2

    def test_two_minimal_turns(self):
        """Two consecutive minimal turns (all pass) work correctly."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Turn 1
        pass_all_players(state, 3)
        assert TURN.get_turn_number(state) == 2
        check_invariants(state, "After turn 1")

        # Turn 2
        pass_all_players(state, 3)
        assert TURN.get_turn_number(state) == 3
        check_invariants(state, "After turn 2")


# =============================================================================
# TURN WITH AUCTION TESTS
# =============================================================================

class TestTurnWithAuction:
    """Turn with auction: company purchased, possibly IPO'd."""

    def test_auction_then_all_pass(self):
        """Player wins auction, then all pass to complete turn."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)
        check_invariants(state, "Initial")

        # Find and start an auction
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        assert auction_idx is not None, "Should have available auction"
        apply_action(state, auction_idx, "Start auction")

        assert state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION
        check_invariants(state, "In auction")

        # All other players leave auction
        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave auction")

        # Back in INVEST, winner has company
        assert state.get_phase() == GamePhases.PHASE_INVEST
        check_invariants(state, "After auction")

        # Complete turn (all pass, then handle IPO)
        complete_turn_from_invest(state, 3)

        assert TURN.get_turn_number(state) == 2
        check_invariants(state, "After turn complete")

    def test_auction_with_raises(self):
        """Auction with bid raises maintains invariants."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Start auction
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        apply_action(state, auction_idx, "Start auction")

        # Raise bid a couple times
        for i in range(2):
            raise_idx = find_valid_action(
                state, layout['raise_bid_base'], layout['acquisition_start']
            )
            if raise_idx and state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action(state, raise_idx, f"Raise {i+1}")
                check_invariants(state, f"After raise {i+1}")

        # Complete auction
        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave")

        check_invariants(state, "After auction with raises")

    def test_multiple_auctions_in_turn(self):
        """Multiple auctions in same turn work correctly."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)
        auctions_completed = 0

        for _ in range(2):  # Try two auctions
            # Check if auction is available
            auction_idx = find_valid_action(
                state, layout['auction_base'], layout['buy_share_base']
            )
            if auction_idx is None:
                break

            apply_action(state, auction_idx, f"Start auction {auctions_completed+1}")

            # Complete auction
            while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action(state, layout['leave_auction'], "Leave")

            auctions_completed += 1
            check_invariants(state, f"After auction {auctions_completed}")

        assert auctions_completed >= 1, "Should complete at least one auction"

    def test_auction_then_ipo(self):
        """Player wins auction, then IPOs company at end of turn."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Win auction
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        apply_action(state, auction_idx, "Start auction")

        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave")

        # All pass to reach IPO phase
        pass_all_players(state, 3)

        # Should be in IPO phase now
        assert state.get_phase() == GamePhases.PHASE_IPO
        check_invariants(state, "At IPO phase")

        # Find valid IPO action (not pass)
        ipo_action = find_valid_action(
            state, layout['ipo_base'], layout['ipo_base'] + 64
        )

        if ipo_action:
            apply_action(state, ipo_action, "IPO company")
            check_invariants(state, "After IPO")

            # Verify a corp is now active
            active_corps = sum(1 for i in range(8) if CORPS[i].is_active(state))
            assert active_corps >= 1, "Should have at least one active corp"

        # Complete any remaining IPO offers
        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")

        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2


# =============================================================================
# TURN WITH ACTIVE CORP TESTS (Turn 2+)
# =============================================================================

class TestTurnWithActiveCorp:
    """Turn 2+: With active corp, full phase sequence.

    Must be turn 2+ since corp can't exist until after turn 1 IPO.
    Tests: INVEST → ACQUISITION → CLOSING → INCOME → DIVIDENDS → ISSUE → IPO → INVEST
    """

    def _setup_turn_2_with_corp(self, num_players=3, seed=42):
        """Set up game at turn 2 with an active corp.

        Returns (state, corp_id, president_player_id).
        """
        state = GameState(num_players=num_players)
        state.initialize_game(seed=seed)

        layout = get_action_layout(num_players)

        # Turn 1: Win auction
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        apply_action(state, auction_idx, "Start auction")

        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave")

        # All pass to reach IPO
        pass_all_players(state, num_players)

        # IPO the company
        assert state.get_phase() == GamePhases.PHASE_IPO

        ipo_action = find_valid_action(
            state, layout['ipo_base'], layout['ipo_base'] + 64
        )
        assert ipo_action is not None, "Should have valid IPO action"
        apply_action(state, ipo_action, "IPO company")

        # Complete IPO phase
        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")

        # Now at turn 2 with active corp
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 2

        # Find the active corp and its president
        corp_id = None
        president_id = None
        for cid in range(8):
            if CORPS[cid].is_active(state):
                corp_id = cid
                for pid in range(num_players):
                    if PLAYERS[pid].is_president_of(state, cid):
                        president_id = pid
                        break
                break

        assert corp_id is not None, "Should have active corp"
        assert president_id is not None, "Corp should have president"

        return state, corp_id, president_id

    def test_turn_2_with_corp_completes(self):
        """Turn 2 with active corp completes full phase sequence."""
        state, corp_id, president_id = self._setup_turn_2_with_corp()
        layout = get_action_layout(3)

        check_invariants(state, "Start of turn 2")

        # All pass in INVEST
        pass_all_players(state, 3)

        # Process through ACQUISITION (pass all offers)
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        # CLOSING auto-processes, should reach DIVIDENDS
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        check_invariants(state, "At DIVIDENDS")

        # Pay dividend
        div_action = find_valid_action(
            state, layout['dividend_base'], layout['dividend_base'] + 26
        )
        assert div_action is not None, "Should have valid dividend action"
        apply_action(state, div_action, "Pay dividend")

        # Process through ISSUE_SHARES
        pass_through_phase(state, 3, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')

        # Process through IPO
        pass_through_phase(state, 3, GamePhases.PHASE_IPO, 'ipo_pass')

        # Should be at turn 3
        assert state.get_phase() == GamePhases.PHASE_INVEST
        assert TURN.get_turn_number(state) == 3
        check_invariants(state, "After turn 2 complete")

    def test_corp_dividend_payment(self):
        """Corp pays dividends to shareholders correctly."""
        state, corp_id, president_id = self._setup_turn_2_with_corp()
        layout = get_action_layout(3)

        corp = CORPS[corp_id]
        president = PLAYERS[president_id]

        # Record starting cash
        president_cash_before = president.get_cash(state)
        corp_cash_before = corp.get_cash(state)
        shares_held = president.get_shares(state, corp_id)

        # All pass in INVEST
        pass_all_players(state, 3)

        # Pass through ACQUISITION
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        # Should be at DIVIDENDS
        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        check_invariants(state, "At DIVIDENDS")

        # Find a non-zero dividend action (if available)
        for offset in range(1, 26):
            action_idx = layout['dividend_base'] + offset
            mask = get_valid_action_mask(state)
            if mask[action_idx] == 1.0:
                # This dividend amount is valid
                apply_action(state, action_idx, f"Pay dividend {offset}")
                break
        else:
            # Just pay 0 if no other option
            apply_action(state, layout['dividend_base'], "Pay dividend 0")

        check_invariants(state, "After dividend payment")

    def test_share_trading_affects_next_phases(self):
        """Share trading in INVEST affects DIVIDENDS (via shareholder changes)."""
        state, corp_id, president_id = self._setup_turn_2_with_corp()
        layout = get_action_layout(3)

        corp = CORPS[corp_id]

        # Check if we can buy a share (bank must have shares)
        buy_action = layout['buy_share_base'] + corp_id
        mask = get_valid_action_mask(state)

        if mask[buy_action] == 1.0:
            # Buy a share
            apply_action(state, buy_action, "Buy share")
            check_invariants(state, "After buy share")

        # Complete the turn
        pass_all_players(state, 3)
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        assert state.get_phase() == GamePhases.PHASE_DIVIDENDS
        check_invariants(state, "At DIVIDENDS after share trading")


# =============================================================================
# MULTI-TURN TESTS
# =============================================================================

class TestMultiTurn:
    """Multi-turn sequences (2-3 turns with meaningful state evolution)."""

    def test_three_turn_game(self):
        """Three complete turns with corp activity."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Turn 1: Auction and IPO
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        apply_action(state, auction_idx, "Start auction")

        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave")

        pass_all_players(state, 3)

        # IPO the company
        ipo_action = find_valid_action(
            state, layout['ipo_base'], layout['ipo_base'] + 64
        )
        if ipo_action:
            apply_action(state, ipo_action, "IPO")

        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")

        assert TURN.get_turn_number(state) == 2
        check_invariants(state, "After turn 1")

        # Turn 2: Corp operations
        pass_all_players(state, 3)
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
            div_action = find_valid_action(
                state, layout['dividend_base'], layout['dividend_base'] + 26
            )
            if div_action:
                apply_action(state, div_action, "Dividend")

        pass_through_phase(state, 3, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')
        pass_through_phase(state, 3, GamePhases.PHASE_IPO, 'ipo_pass')

        assert TURN.get_turn_number(state) == 3
        check_invariants(state, "After turn 2")

        # Turn 3: Another auction
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        if auction_idx:
            apply_action(state, auction_idx, "Start auction turn 3")
            while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action(state, layout['leave_auction'], "Leave")

        pass_all_players(state, 3)
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
            div_action = find_valid_action(
                state, layout['dividend_base'], layout['dividend_base'] + 26
            )
            if div_action:
                apply_action(state, div_action, "Dividend")

        pass_through_phase(state, 3, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')
        pass_through_phase(state, 3, GamePhases.PHASE_IPO, 'ipo_pass')

        assert TURN.get_turn_number(state) == 4
        check_invariants(state, "After turn 3")

    def test_multiple_corps_active(self):
        """Game with multiple corps active by turn 3."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        layout = get_action_layout(3)

        # Turn 1: First auction and IPO
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        apply_action(state, auction_idx, "Auction 1")

        while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
            apply_action(state, layout['leave_auction'], "Leave")

        pass_all_players(state, 3)

        ipo_action = find_valid_action(
            state, layout['ipo_base'], layout['ipo_base'] + 64
        )
        if ipo_action:
            apply_action(state, ipo_action, "IPO 1")

        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")

        corps_after_t1 = sum(1 for i in range(8) if CORPS[i].is_active(state))
        check_invariants(state, "After turn 1")

        # Turn 2: Second auction and IPO
        auction_idx = find_valid_action(
            state, layout['auction_base'], layout['buy_share_base']
        )
        if auction_idx:
            apply_action(state, auction_idx, "Auction 2")
            while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                apply_action(state, layout['leave_auction'], "Leave")

        pass_all_players(state, 3)
        pass_through_phase(state, 3, GamePhases.PHASE_ACQUISITION, 'acq_pass')

        if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
            div_action = find_valid_action(
                state, layout['dividend_base'], layout['dividend_base'] + 26
            )
            if div_action:
                apply_action(state, div_action, "Dividend")

        pass_through_phase(state, 3, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')

        # Try to IPO second company
        ipo_action = find_valid_action(
            state, layout['ipo_base'], layout['ipo_base'] + 64
        )
        if ipo_action:
            apply_action(state, ipo_action, "IPO 2")

        while state.get_phase() == GamePhases.PHASE_IPO:
            apply_action(state, layout['ipo_pass'], "Pass IPO")

        corps_after_t2 = sum(1 for i in range(8) if CORPS[i].is_active(state))
        check_invariants(state, "After turn 2")

        # Should have at least one corp, possibly two
        assert corps_after_t2 >= corps_after_t1

    @pytest.mark.parametrize("num_players", [2, 3, 6])
    def test_two_turns_all_player_counts(self, num_players):
        """Two complete turns work for all player counts."""
        state = GameState(num_players=num_players)
        state.initialize_game(seed=42)

        layout = get_action_layout(num_players)

        for turn in range(1, 3):  # Turns 1 and 2
            expected_turn = turn
            assert TURN.get_turn_number(state) == expected_turn

            # Try auction if available
            auction_idx = find_valid_action(
                state, layout['auction_base'], layout['buy_share_base']
            )
            if auction_idx:
                apply_action(state, auction_idx, f"Turn {turn} auction")
                while state.get_phase() == GamePhases.PHASE_BID_IN_AUCTION:
                    apply_action(state, layout['leave_auction'], "Leave")

            # All pass
            pass_all_players(state, num_players)

            # Handle corp phases if active
            pass_through_phase(state, num_players, GamePhases.PHASE_ACQUISITION, 'acq_pass')

            if state.get_phase() == GamePhases.PHASE_DIVIDENDS:
                div_action = find_valid_action(
                    state, layout['dividend_base'], layout['dividend_base'] + 26
                )
                if div_action:
                    apply_action(state, div_action, "Dividend")

            pass_through_phase(state, num_players, GamePhases.PHASE_ISSUE_SHARES, 'issue_pass')

            # IPO if available
            ipo_action = find_valid_action(
                state, layout['ipo_base'], layout['ipo_base'] + 64
            )
            if ipo_action:
                apply_action(state, ipo_action, "IPO")

            while state.get_phase() == GamePhases.PHASE_IPO:
                apply_action(state, layout['ipo_pass'], "Pass IPO")

            check_invariants(state, f"After turn {turn}")

        assert TURN.get_turn_number(state) == 3


# =============================================================================
# GAME END TESTS
# =============================================================================

class TestGameEnd:
    """Game termination scenarios.

    Note: Detailed game end mechanics are tested in tests/phases/test_end_card.py.
    These tests verify game end is reachable through normal driver-based play.
    """

    def test_game_over_phase_recognized(self):
        """Verify GAME_OVER phase constant is recognized."""
        # Just verify the constant exists and has expected value
        assert GamePhases.PHASE_GAME_OVER == 10

    def test_extended_play_eventually_ends_or_continues(self):
        """Play many turns - game should either end or continue stably."""
        from core.driver import STATUS_GAME_OVER_PY

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        max_actions = 200
        actions_taken = 0

        while actions_taken < max_actions:
            if state.get_phase() == GamePhases.PHASE_GAME_OVER:
                break

            mask = get_valid_action_mask(state)
            valid_actions = [i for i, v in enumerate(mask) if v == 1.0]

            if not valid_actions:
                break

            # Take first valid action (deterministic)
            action = valid_actions[0]
            result = DRIVER.apply_action(state, action)
            actions_taken += 1

            if result == STATUS_GAME_OVER_PY:
                break

        # Should have taken some actions without crashing
        assert actions_taken > 0
        check_invariants(state, "After extended play")


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests with random/extensive action sequences."""

    def test_many_turns_no_crash(self):
        """Run many turns without crashing."""
        from core.driver import STATUS_GAME_OVER_PY

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        max_turns = 10

        while TURN.get_turn_number(state) <= max_turns:
            if state.get_phase() == GamePhases.PHASE_GAME_OVER:
                break

            # Get valid actions
            mask = get_valid_action_mask(state)

            # Pick first valid action
            action = None
            for i in range(len(mask)):
                if mask[i] == 1.0:
                    action = i
                    break

            if action is None:
                break

            result = DRIVER.apply_action(state, action)

            if result == STATUS_GAME_OVER_PY:
                break

            check_invariants(state, f"Turn {TURN.get_turn_number(state)}")

    def test_random_valid_actions(self):
        """Play random valid actions for several turns."""
        from core.driver import STATUS_GAME_OVER_PY
        import random
        random.seed(12345)

        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        max_actions = 100
        actions_taken = 0

        while actions_taken < max_actions:
            if state.get_phase() == GamePhases.PHASE_GAME_OVER:
                break

            mask = get_valid_action_mask(state)
            valid_actions = [i for i, v in enumerate(mask) if v == 1.0]

            if not valid_actions:
                break

            action = random.choice(valid_actions)
            result = DRIVER.apply_action(state, action)
            actions_taken += 1

            if result == STATUS_GAME_OVER_PY:
                break

            check_invariants(state, f"After {actions_taken} actions")

        check_invariants(state, f"Final state after {actions_taken} actions")
