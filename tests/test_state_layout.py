"""Tests for GameState layout sizes.

These tests verify that documented sizes in VECTORS.md and CLAUDE.md match actual
computed sizes. Uses core.state.get_layout() as the single source of truth.
"""
import pytest
from core.state import GameState, get_layout


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
