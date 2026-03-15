"""Tests for GameState layout sizes and static data.

These tests verify that documented sizes in VECTORS.md and CLAUDE.md match actual
computed sizes. Uses core.state.get_layout() as the single source of truth.
"""
import numpy as np
import pytest
from core.state import GameState, get_layout
from core.data import (
    get_company_stars, get_company_face_value,
    get_company_low_price, get_company_high_price,
    get_company_synergy, PY_STAR_DIVISOR, PY_CASH_DIVISOR,
    GameConstants,
)


class TestStateLayoutSizes:
    """Verify state layout sizes match documentation."""

    # Expected sizes - these MUST match VECTORS.md and CLAUDE.md
    # If these tests fail, update the documentation to match!
    EXPECTED_SIZES = {
        2: {'visible': 2943, 'hidden': 1184, 'total': 4127},
        3: {'visible': 3023, 'hidden': 1184, 'total': 4207},
        4: {'visible': 3105, 'hidden': 1184, 'total': 4289},
        5: {'visible': 3189, 'hidden': 1184, 'total': 4373},
        6: {'visible': 3275, 'hidden': 1184, 'total': 4459},
    }

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
    def test_total_size_matches_actual(self, num_players):
        """Total computed size matches actual GameState array size."""
        gs = GameState(num_players)
        actual_total = gs._array.shape[0]
        layout = get_layout(num_players)

        assert layout.total_size == actual_total, (
            f"{num_players} players: computed {layout.total_size} != actual {actual_total}"
        )

    @pytest.mark.parametrize("num_players", [2, 3, 4, 5, 6])
    def test_sizes_match_expected(self, num_players):
        """Computed sizes match expected documented values."""
        expected = self.EXPECTED_SIZES[num_players]
        layout = get_layout(num_players)

        assert layout.visible_size == expected['visible'], (
            f"{num_players} players visible: {layout.visible_size} != expected {expected['visible']}"
        )
        assert layout.hidden_size == expected['hidden'], (
            f"{num_players} players hidden: {layout.hidden_size} != expected {expected['hidden']}"
        )
        assert layout.total_size == expected['total'], (
            f"{num_players} players total: {layout.total_size} != expected {expected['total']}"
        )


class TestComponentSizes:
    """Verify individual component sizes."""

    def test_player_stride_formula(self):
        """Player stride = 72 + num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_layout(num_players)
            assert layout.player_stride == 72 + num_players, (
                f"{num_players} players: stride {layout.player_stride} != 72 + {num_players}"
            )

    def test_corp_stride_fixed(self):
        """Corp stride = 109 (fixed for all player counts)."""
        layout = get_layout(3)
        assert layout.corp_stride == 109

    def test_turn_size_formula(self):
        """Turn size = 251 + 3*num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_layout(num_players)
            expected = 251 + 3 * num_players
            assert layout.turn_size == expected, (
                f"{num_players} players: turn size {layout.turn_size} != {expected}"
            )

    def test_hidden_size_fixed(self):
        """Hidden size = 1184 (fixed for all player counts)."""
        layout = get_layout(3)
        assert layout.hidden_size == 1184

    def test_static_size(self):
        """Static company data = 36 * 40 = 1440."""
        layout = get_layout(3)
        assert layout.static_size == 1440


NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)
STATIC_STRIDE = 4 + NUM_COMPANIES


class TestStaticCompanyData:
    """Verify the static company data block is populated correctly."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_static_block_not_all_zeros(self, state):
        """Static block must contain nonzero data after initialization."""
        layout = get_layout(3)
        block = state._array[layout.static_offset:layout.static_offset + layout.static_size]
        assert np.count_nonzero(block) > 0

    @pytest.mark.parametrize("company_id", range(NUM_COMPANIES))
    def test_company_stars(self, state, company_id):
        layout = get_layout(3)
        offset = layout.static_offset + company_id * STATIC_STRIDE
        expected = get_company_stars(company_id) / PY_STAR_DIVISOR
        assert abs(state._array[offset + 0] - expected) < 1e-6

    @pytest.mark.parametrize("company_id", range(NUM_COMPANIES))
    def test_company_low_price(self, state, company_id):
        layout = get_layout(3)
        offset = layout.static_offset + company_id * STATIC_STRIDE
        expected = get_company_low_price(company_id) / PY_CASH_DIVISOR
        assert abs(state._array[offset + 1] - expected) < 1e-6

    @pytest.mark.parametrize("company_id", range(NUM_COMPANIES))
    def test_company_face_value(self, state, company_id):
        layout = get_layout(3)
        offset = layout.static_offset + company_id * STATIC_STRIDE
        expected = get_company_face_value(company_id) / PY_CASH_DIVISOR
        assert abs(state._array[offset + 2] - expected) < 1e-6

    @pytest.mark.parametrize("company_id", range(NUM_COMPANIES))
    def test_company_high_price(self, state, company_id):
        layout = get_layout(3)
        offset = layout.static_offset + company_id * STATIC_STRIDE
        expected = get_company_high_price(company_id) / PY_CASH_DIVISOR
        assert abs(state._array[offset + 3] - expected) < 1e-6

    @pytest.mark.parametrize("company_id", range(NUM_COMPANIES))
    def test_company_synergy_flags(self, state, company_id):
        layout = get_layout(3)
        offset = layout.static_offset + company_id * STATIC_STRIDE + 4
        for target_id in range(NUM_COMPANIES):
            expected = 1.0 if get_company_synergy(company_id, target_id) != 0 else 0.0
            actual = state._array[offset + target_id]
            assert abs(actual - expected) < 1e-6, (
                f"Company {company_id} synergy[{target_id}]: {actual} != {expected}"
            )

    def test_static_data_identical_across_seeds(self):
        """Static data should be the same regardless of game seed."""
        s1 = GameState(3)
        s1.initialize_game(seed=1)
        s2 = GameState(3)
        s2.initialize_game(seed=999)
        layout = get_layout(3)
        block1 = s1._array[layout.static_offset:layout.static_offset + layout.static_size]
        block2 = s2._array[layout.static_offset:layout.static_offset + layout.static_size]
        np.testing.assert_array_equal(block1, block2)

    def test_static_data_identical_across_player_counts(self):
        """Static data content should be the same for different player counts."""
        layouts = {}
        blocks = {}
        for n in [2, 3, 6]:
            gs = GameState(n)
            gs.initialize_game(seed=42)
            layouts[n] = get_layout(n)
            blocks[n] = gs._array[layouts[n].static_offset:layouts[n].static_offset + layouts[n].static_size]
        np.testing.assert_array_equal(blocks[2], blocks[3])
        np.testing.assert_array_equal(blocks[3], blocks[6])
