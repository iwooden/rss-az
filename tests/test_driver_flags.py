"""Tests for driver config flags and paused auto-transition behavior."""
import numpy as np
from core.state import GameState
from core.driver import (
    DRIVER,
    STATUS_OK_PY as STATUS_OK,
    STATUS_INVALID_PY as STATUS_INVALID,
    STATUS_PAUSED_PY as STATUS_PAUSED,
)
from core.actions import get_valid_action_mask, get_action_layout
from core.data import GamePhases
from entities.turn import TURN
from entities.corp import CORPS
from entities.company import COMPANIES
from tests.phases.conftest import float_corp_for_test


PHASE_INVEST = GamePhases.PHASE_INVEST
PHASE_BID = GamePhases.PHASE_BID_IN_AUCTION
PHASE_WRAP_UP = GamePhases.PHASE_WRAP_UP
PHASE_ACQ = GamePhases.PHASE_ACQUISITION
PHASE_CLOSING = GamePhases.PHASE_CLOSING
PHASE_INCOME = GamePhases.PHASE_INCOME
PHASE_DIVIDENDS = GamePhases.PHASE_DIVIDENDS
PHASE_ISSUE = GamePhases.PHASE_ISSUE_SHARES
PHASE_IPO = GamePhases.PHASE_IPO
PHASE_PAR = GamePhases.PHASE_PAR
PHASE_GAME_OVER = GamePhases.PHASE_GAME_OVER


class TestStepModeFlag:
    """Tests for GameState.step_mode flag."""

    def test_default_is_false(self):
        """step_mode defaults to False."""
        state = GameState(num_players=3)
        assert state.step_mode is False

    def test_set_step_mode(self):
        """step_mode can be set to True."""
        state = GameState(num_players=3)
        state.step_mode = True
        assert state.step_mode is True

    def test_from_array_preserves_default(self):
        """from_array creates a new GameState with default step_mode=False."""
        state = GameState(num_players=3)
        state.step_mode = True
        state2 = GameState.from_array(state._array, 3)
        assert state2.step_mode is False

    def test_normal_mode_auto_applies_forced(self):
        """Without step_mode, driver auto-applies forced actions."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # Get initial phase and apply a pass in INVEST
        assert TURN.get_phase(state) == PHASE_INVEST

        # Apply passes until all players pass — driver should auto-advance
        # through WRAP_UP/ACQ/CLO/INCOME/etc.
        for _ in range(10):
            phase = TURN.get_phase(state)
            if phase != PHASE_INVEST:
                break
            mask = get_valid_action_mask(state)
            # Apply first valid action (pass)
            for i in range(len(mask)):
                if mask[i] > 0.5:
                    DRIVER.apply_action(state, i)
                    break

        # Normal mode should have auto-advanced past INVEST
        # (through WRAP_UP, etc.)
        # After 3 passes, we go through automated phases and come back to INVEST
        # The key point: we should NOT be stuck in a non-player phase
        phase = TURN.get_phase(state)
        assert phase not in (PHASE_WRAP_UP, PHASE_INCOME, PHASE_ACQ, PHASE_CLOSING)

    def test_step_mode_no_auto_apply(self):
        """In step_mode, driver applies exactly one action and returns."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        assert TURN.get_phase(state) == PHASE_INVEST
        layout = get_action_layout(3)

        # Apply a pass action — should NOT auto-advance through phases
        mask = get_valid_action_mask(state)
        pass_action = layout['pass_invest']
        assert mask[pass_action] > 0.5

        result = DRIVER.apply_action(state, pass_action)
        assert result == STATUS_OK

        # Still in INVEST — just advanced to next player, no auto-cascade
        assert TURN.get_phase(state) == PHASE_INVEST

    def test_step_mode_stops_at_non_player_phase(self):
        """In step_mode, after all players pass in INVEST, engine stops at WRAP_UP."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        layout = get_action_layout(3)
        pass_action = layout['pass_invest']

        # Apply 3 passes (all players) — should transition to WRAP_UP but not execute it
        for i in range(3):
            assert TURN.get_phase(state) == PHASE_INVEST, f"Expected INVEST at pass {i}"
            result = DRIVER.apply_action(state, pass_action)
            assert result == STATUS_OK

        # After 3 passes, should be in WRAP_UP (non-player phase), not auto-advanced
        assert TURN.get_phase(state) == PHASE_WRAP_UP

    def test_step_mode_does_not_auto_apply_forced_action(self):
        """In step_mode, forced actions (1 legal move) are NOT auto-applied."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        # Set up a dividends situation with a corp that has 0 cash → forced dividend 0
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=5)
        CORPS[0].set_cash(state, 0)
        TURN.set_phase(state, PHASE_DIVIDENDS)
        TURN.set_dividend_corp(state, 0)

        # In normal mode, dividend 0 would be the only legal action and auto-applied.
        # In step mode, we should be able to see it and apply it manually.
        mask = get_valid_action_mask(state)
        legal_count = sum(1 for v in mask if v > 0.5)
        assert legal_count == 1, "Should have exactly 1 legal action (forced dividend 0)"

        # Find the forced action
        forced_action = next(i for i, v in enumerate(mask) if v > 0.5)
        result = DRIVER.apply_action(state, forced_action)
        assert result == STATUS_OK

    def test_advance_phase_executes_non_player_phase(self):
        """advance_phase() executes one non-player phase."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        layout = get_action_layout(3)
        pass_action = layout['pass_invest']

        # Get to WRAP_UP
        for _ in range(3):
            DRIVER.apply_action(state, pass_action)

        assert TURN.get_phase(state) == PHASE_WRAP_UP
        assert DRIVER.is_non_player_phase(state) is True

        # Advance through WRAP_UP
        result = DRIVER.advance_phase(state)
        assert result == STATUS_OK

        # Should have moved to next phase (ACQ or beyond)
        phase = TURN.get_phase(state)
        assert phase != PHASE_WRAP_UP

    def test_advance_phase_invalid_on_player_phase(self):
        """advance_phase() returns STATUS_INVALID on player phases."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        assert TURN.get_phase(state) == PHASE_INVEST
        assert DRIVER.is_non_player_phase(state) is False

        result = DRIVER.advance_phase(state)
        assert result == STATUS_INVALID

    def test_is_non_player_phase(self):
        """is_non_player_phase() correctly identifies non-player phases."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)

        # INVEST is a player phase
        TURN.set_phase(state, PHASE_INVEST)
        assert DRIVER.is_non_player_phase(state) is False

        # WRAP_UP is always non-player
        TURN.set_phase(state, PHASE_WRAP_UP)
        assert DRIVER.is_non_player_phase(state) is True

        # INCOME is always non-player
        TURN.set_phase(state, PHASE_INCOME)
        assert DRIVER.is_non_player_phase(state) is True

    def test_step_mode_manual_cascade(self):
        """In step_mode, caller can manually cascade through non-player phases."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        layout = get_action_layout(3)
        pass_action = layout['pass_invest']

        # Pass for all players
        for _ in range(3):
            DRIVER.apply_action(state, pass_action)

        # Now manually advance through non-player phases
        max_iterations = 20
        for _ in range(max_iterations):
            phase = TURN.get_phase(state)
            if phase == PHASE_GAME_OVER:
                break
            if DRIVER.is_non_player_phase(state):
                DRIVER.advance_phase(state)
            else:
                # Hit a player phase — stop
                break

        # Should have reached a player phase (INVEST, DIVIDENDS, etc.)
        phase = TURN.get_phase(state)
        assert not DRIVER.is_non_player_phase(state) or phase == PHASE_GAME_OVER

    def test_step_mode_history_records_single_action(self):
        """In step_mode, history records exactly the one action applied."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        layout = get_action_layout(3)
        pass_action = layout['pass_invest']

        history = []
        DRIVER.apply_action(state, pass_action, history)

        # Should record exactly 1 entry (the pass action)
        assert len(history) == 1
        assert history[0][1] == pass_action

    def test_step_mode_advance_phase_records_history(self):
        """advance_phase() records sentinel to history."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.step_mode = True

        layout = get_action_layout(3)
        pass_action = layout['pass_invest']

        # Get to WRAP_UP
        for _ in range(3):
            DRIVER.apply_action(state, pass_action)

        history = []
        DRIVER.advance_phase(state, history)

        # Should have recorded the non-player phase execution
        assert len(history) == 1
        assert history[0][1] < 0  # Sentinel values are negative


class TestPauseBeforeAcqTransition:
    """Tests for GameState.pause_before_acq_transition flag."""

    def test_default_is_false(self):
        """pause_before_acq_transition defaults to False."""
        state = GameState(num_players=3)
        assert state.pause_before_acq_transition is False

    def test_set_flag(self):
        """pause_before_acq_transition can be enabled."""
        state = GameState(num_players=3)
        state.pause_before_acq_transition = True
        assert state.pause_before_acq_transition is True

    def test_apply_action_pauses_before_acq_transition(self):
        """Driver returns STATUS_PAUSED before auto-transitioning out of ACQ."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.pause_before_acq_transition = True

        layout = get_action_layout(3)
        pass_action = layout["pass_invest"]

        for _ in range(2):
            result = DRIVER.apply_action(state, pass_action)
            assert result == STATUS_OK

        history = []
        result = DRIVER.apply_action(state, pass_action, history=history)
        assert result == STATUS_PAUSED
        assert TURN.get_phase(state) == PHASE_ACQ
        assert TURN.get_acq_active_corp(state) == -1
        assert DRIVER.is_non_player_phase(state) is True

        action_values = [entry[1] for entry in history]
        assert pass_action in action_values
        assert -100 in action_values, "WRAP_UP sentinel should be in history"
        assert -101 not in action_values, "ACQ transition should not have executed"

    def test_advance_phase_resumes_after_pause(self):
        """advance_phase() continues ACQ transition after a paused return."""
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.pause_before_acq_transition = True

        layout = get_action_layout(3)
        pass_action = layout["pass_invest"]
        for _ in range(2):
            DRIVER.apply_action(state, pass_action)

        assert DRIVER.apply_action(state, pass_action) == STATUS_PAUSED
        assert TURN.get_phase(state) == PHASE_ACQ

        history = []
        result = DRIVER.advance_phase(state, history=history)
        assert result == STATUS_OK
        assert TURN.get_phase(state) == PHASE_CLOSING
        assert len(history) == 1
        assert history[0][1] == -101


class TestPauseBeforeClosingTransition:
    """Tests for GameState.pause_before_closing_transition flag."""

    def _make_single_offer_closing_state(self) -> tuple[GameState, int]:
        state = GameState(num_players=3)
        state.initialize_game(seed=42)
        state.pause_before_closing_transition = True
        TURN.set_coo_level(state, 7)

        company_id = None
        for cid in range(36):
            if COMPANIES[cid].get_location(state) != 0:
                continue
            if COMPANIES[cid].get_adjusted_income(state) < 0:
                company_id = cid
                break

        assert company_id is not None
        COMPANIES[company_id].transfer_to_player(state, 0)
        TURN.set_phase(state, PHASE_CLOSING)
        return state, company_id

    def test_default_is_false(self):
        """pause_before_closing_transition defaults to False."""
        state = GameState(num_players=3)
        assert state.pause_before_closing_transition is False

    def test_set_flag(self):
        """pause_before_closing_transition can be enabled."""
        state = GameState(num_players=3)
        state.pause_before_closing_transition = True
        assert state.pause_before_closing_transition is True

    def test_apply_action_pauses_before_closing_transition(self):
        """Driver returns STATUS_PAUSED before CLO finalizes into INCOME."""
        state, company_id = self._make_single_offer_closing_state()
        layout = get_action_layout(3)

        history = []
        assert DRIVER.advance_phase(state, history=history) == STATUS_OK
        assert TURN.get_phase(state) == PHASE_CLOSING
        assert TURN.get_closing_company(state) == company_id

        result = DRIVER.apply_action(state, layout["close_action"], history=history)
        assert result == STATUS_PAUSED
        assert TURN.get_phase(state) == PHASE_CLOSING
        assert TURN.get_closing_company(state) == -1
        assert DRIVER.is_non_player_phase(state) is True

        action_values = [entry[1] for entry in history]
        assert action_values.count(-102) == 1
        assert action_values[-1] == layout["close_action"]

    def test_advance_phase_resumes_after_pause(self):
        """advance_phase() continues CLO transition after a paused return."""
        state, _ = self._make_single_offer_closing_state()
        layout = get_action_layout(3)

        assert DRIVER.advance_phase(state) == STATUS_OK
        assert DRIVER.apply_action(state, layout["close_action"]) == STATUS_PAUSED
        assert TURN.get_phase(state) == PHASE_CLOSING

        history = []
        result = DRIVER.advance_phase(state, history=history)
        assert result == STATUS_OK
        assert TURN.get_phase(state) == PHASE_INCOME
        assert len(history) == 1
        assert history[0][1] == -102


class TestFlagsCombined:
    """Tests for the remaining driver control flags."""

    def test_both_flags_independent(self):
        """Driver control flags can be set independently."""
        state = GameState(num_players=3)
        state.step_mode = True
        state.pause_before_acq_transition = True
        state.pause_before_closing_transition = True
        assert state.step_mode is True
        assert state.pause_before_acq_transition is True
        assert state.pause_before_closing_transition is True

        state.step_mode = False
        assert state.step_mode is False
        assert state.pause_before_acq_transition is True
        assert state.pause_before_closing_transition is True

    def test_flags_do_not_affect_state_array(self):
        """Flags are Python-level only, don't change the float array."""
        state1 = GameState(num_players=3)
        state1.initialize_game(seed=42)

        state2 = GameState(num_players=3)
        state2.initialize_game(seed=42)
        state2.step_mode = True
        state2.pause_before_acq_transition = True
        state2.pause_before_closing_transition = True

        # Arrays should be identical
        np.testing.assert_array_equal(state1._array, state2._array)
