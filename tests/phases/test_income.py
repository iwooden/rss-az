"""Tests for the INCOME phase.

Covers: player income application from privately-owned companies, corp income
(positive add, negative with reserve survives, negative insolvent → bankruptcy,
inactive corps skipped), FI income application including the +5 base bonus,
and transition to PHASE_DIVIDENDS with its setup.

INCOME is automated — tests invoke ``apply_income_py`` directly rather than
routing through the driver. See ``auto-phases.md`` for the rationale.
"""
from core.data import GamePhases, GameConstants
from entities.turn import TURN
from entities.player import PLAYERS
from entities.corp import CORPS
from entities.company import COMPANIES, CompanyLocation
from entities.fi import FI
from phases.income import apply_income_py

from tests.phases.conftest import (
    make_auto_phase_state,
    assert_invariants,
    assert_token_data_invariants,
    assert_post_auto,
    draw_to_player,
    draw_to_fi,
    float_corp_for_test,
)


# =============================================================================
# HELPERS
# =============================================================================

PHASE_INCOME = int(GamePhases.PHASE_INCOME)
PHASE_DIVIDENDS = int(GamePhases.PHASE_DIVIDENDS)


def _prime_corp_income(state, corp_id, income):
    """Force a corp's cached income to ``income`` without re-dirtying the cache.

    The income cache is dirtied by any setter that invalidates
    ``corp_cache_dirty`` (e.g. ``set_cash``, ``set_price_index``). Callers
    wanting to stage a synthetic income for a floated corp must:
      1. First land all dirty-bit-invalidating mutations (set_cash, etc.).
      2. Read ``get_income`` once to refresh and clear the dirty bit.
      3. Finally write ``set_income(income)`` to overwrite just the income
         slot with the synthetic value.
    """
    CORPS[corp_id].get_income(state)  # Refresh cache → clears dirty bit.
    CORPS[corp_id].set_income(state, income)


# =============================================================================
# PLAYER INCOME
# =============================================================================

class TestPlayerIncome:
    """INCOME step 1: each player's cached income is added to their cash."""

    def test_player_income_adds_to_cash(self):
        """Player with one owned company gains its adjusted income in cash."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        cid = draw_to_player(state, player_id=0)

        # Player income auto-refreshes from the companies cache on read.
        expected_income = COMPANIES[cid].get_adjusted_income(state)
        assert PLAYERS[0].get_income(state) == expected_income
        cash_before = PLAYERS[0].get_cash(state)

        apply_income_py(state)

        assert PLAYERS[0].get_cash(state) == cash_before + expected_income
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_player_no_companies_no_change(self):
        """Player with no owned companies has zero income → cash unchanged."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        assert PLAYERS[1].get_income(state) == 0
        cash_before = PLAYERS[1].get_cash(state)

        apply_income_py(state)

        assert PLAYERS[1].get_cash(state) == cash_before
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_multiple_players_independent(self):
        """Different players with different companies each get exactly their own."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        cid_a = draw_to_player(state, player_id=0)
        cid_b = draw_to_player(state, player_id=2)

        inc_a = COMPANIES[cid_a].get_adjusted_income(state)
        inc_b = COMPANIES[cid_b].get_adjusted_income(state)
        cash0 = PLAYERS[0].get_cash(state)
        cash1 = PLAYERS[1].get_cash(state)
        cash2 = PLAYERS[2].get_cash(state)

        apply_income_py(state)

        assert PLAYERS[0].get_cash(state) == cash0 + inc_a
        assert PLAYERS[1].get_cash(state) == cash1  # Owns nothing.
        assert PLAYERS[2].get_cash(state) == cash2 + inc_b
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# CORP INCOME
# =============================================================================

class TestCorpIncome:
    """INCOME step 2: corp income with bankruptcy-on-negative-and-insolvent."""

    def test_positive_income_adds_to_cash(self):
        """Floated corp with positive income gains it in cash; stays active."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)

        CORPS[0].set_cash(state, 20)
        _prime_corp_income(state, 0, 7)
        assert CORPS[0].get_income(state) == 7

        apply_income_py(state)

        assert CORPS[0].is_active(state)
        assert CORPS[0].get_cash(state) == 27
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_negative_income_with_reserve_survives(self):
        """Corp with negative income but sufficient cash survives (no bankruptcy)."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)

        CORPS[0].set_cash(state, 10)
        _prime_corp_income(state, 0, -3)

        apply_income_py(state)

        # income < 0 but cash_after = 7 >= 0 → no bankruptcy branch.
        assert CORPS[0].is_active(state)
        assert CORPS[0].get_cash(state) == 7
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_negative_income_insolvent_goes_bankrupt(self):
        """income < 0 AND cash_after < 0 triggers go_bankrupt side effects."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        cid = float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)

        CORPS[0].set_cash(state, 1)
        _prime_corp_income(state, 0, -5)

        apply_income_py(state)

        # go_bankrupt contract: corp inactive, cash zeroed, no companies,
        # price_index reset, owned company returned to LOC_REMOVED.
        assert not CORPS[0].is_active(state)
        assert CORPS[0].get_cash(state) == 0
        assert CORPS[0].count_companies(state, include_acquisition=True) == 0
        assert CORPS[0].get_price_index(state) == 0
        assert COMPANIES[cid].get_location(state) == int(CompanyLocation.LOC_REMOVED)
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_inactive_corp_skipped(self):
        """Inactive corps don't receive income and aren't bankruptcy-checked."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        # Float corp 0 only; corps 1..7 remain inactive.
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        _prime_corp_income(state, 0, 0)

        # Stage a sentinel: write a non-zero cash and non-zero income to an
        # inactive corp's slot. The handler must NOT read/modify either.
        CORPS[3].set_income(state, -99)  # Would trigger bankruptcy if applied.
        # set_cash would normally dirty the cache; on an inactive corp the
        # income slot is incidental since the handler skips it outright.
        cash3_before = CORPS[3].get_cash(state)

        apply_income_py(state)

        assert not CORPS[3].is_active(state)
        assert CORPS[3].get_cash(state) == cash3_before
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# FI INCOME
# =============================================================================

class TestFIIncome:
    """INCOME step 3: FI's stored income (incl. +5 bonus) lands in FI cash."""

    def test_fi_bonus_always_applied(self):
        """FI with zero companies has stored income == 5 → cash += 5."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        # initialize_game seeds FI income to 5 (the base bonus).
        assert FI.get_income(state) == 5
        cash_before = FI.get_cash(state)

        apply_income_py(state)

        assert FI.get_cash(state) == cash_before + 5
        assert_invariants(state)
        assert_token_data_invariants(state)

    def test_fi_income_with_companies(self):
        """FI cash gains +5 plus sum of FI-owned company adjusted incomes."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        cid_a = draw_to_fi(state)
        cid_b = draw_to_fi(state)

        inc_a = COMPANIES[cid_a].get_adjusted_income(state)
        inc_b = COMPANIES[cid_b].get_adjusted_income(state)
        # transfer_to_fi cascades FI.calculate_income, so stored income is
        # already 5 + inc_a + inc_b by now.
        expected_income = 5 + inc_a + inc_b
        assert FI.get_income(state) == expected_income
        cash_before = FI.get_cash(state)

        apply_income_py(state)

        assert FI.get_cash(state) == cash_before + expected_income
        assert_invariants(state)
        assert_token_data_invariants(state)


# =============================================================================
# TRANSITION
# =============================================================================

class TestTransition:
    """INCOME step 4: hand off to PHASE_DIVIDENDS with its setup run."""

    def test_transitions_to_dividends(self):
        """Phase enum flips to DIVIDENDS when at least one active corp exists."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        # Need an active corp so setup doesn't cascade past DIVIDENDS.
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        _prime_corp_income(state, 0, 0)

        apply_income_py(state)

        assert_post_auto(state, PHASE_DIVIDENDS)

    def test_dividends_setup_ran(self):
        """setup_dividends_phase sets dividend_remaining for active corps."""
        state = make_auto_phase_state(3, PHASE_INCOME)
        float_corp_for_test(state, corp_id=0, player_id=0, par_index=10)
        _prime_corp_income(state, 0, 0)
        # A second, higher-priced active corp verifies the "highest price
        # first" active_corp selection done by setup.
        float_corp_for_test(state, corp_id=1, player_id=0, par_index=13)
        _prime_corp_income(state, 1, 0)

        apply_income_py(state)

        # Active corps have dividend_remaining True; inactives have False.
        for c in range(int(GameConstants.NUM_CORPS)):
            expected = CORPS[c].is_active(state)
            assert TURN.is_dividend_remaining(state, c) == expected, (
                f"dividend_remaining[{c}] = {TURN.is_dividend_remaining(state, c)}, "
                f"expected {expected}"
            )
        # Highest-price active corp goes first.
        assert TURN.get_active_corp(state) == 1
        assert TURN.get_active_player(state) == 0
        assert_invariants(state)
        assert_token_data_invariants(state)
