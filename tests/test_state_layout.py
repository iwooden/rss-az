"""Tests for GameState layout sizes, auction slot info, active company, and active corp.

These tests verify that documented sizes in VECTORS.md and CLAUDE.md match actual
computed sizes. Uses core.state.get_layout() as the single source of truth.
"""
import numpy as np
import pytest
from core.state import GameState, get_layout
from core.data import (
    get_company_stars, get_company_face_value,
    get_company_low_price, get_company_high_price,
    get_adjusted_company_income,
    PY_STAR_DIVISOR, PY_CASH_DIVISOR,
)
from entities.company import COMPANIES, get_auction_company_for_slot_py
from entities.deck import DECK
from entities.fi import FI
from entities.player import PLAYERS
from entities.turn import TURN


class TestStateLayoutSizes:
    """Verify state layout sizes match documentation."""

    # Expected sizes - these MUST match VECTORS.md and CLAUDE.md
    # If these tests fail, update the documentation to match!
    EXPECTED_SIZES = {
        2: {'visible': 1473, 'hidden': 1184, 'total': 2657},
        3: {'visible': 1559, 'hidden': 1184, 'total': 2743},
        4: {'visible': 1647, 'hidden': 1184, 'total': 2831},
        5: {'visible': 1737, 'hidden': 1184, 'total': 2921},
        6: {'visible': 1829, 'hidden': 1184, 'total': 3013},
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
        """Player stride = 73 + num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_layout(num_players)
            assert layout.player_stride == 73 + num_players, (
                f"{num_players} players: stride {layout.player_stride} != 73 + {num_players}"
            )

    def test_corp_stride_fixed(self):
        """Corp stride = 109 (fixed for all player counts)."""
        layout = get_layout(3)
        assert layout.corp_stride == 109

    def test_turn_size_formula(self):
        """Turn size = 208 + 3*num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_layout(num_players)
            expected = 208 + 3 * num_players
            assert layout.turn_size == expected, (
                f"{num_players} players: turn size {layout.turn_size} != {expected}"
            )

    def test_hidden_size_fixed(self):
        """Hidden size = 1184 (fixed for all player counts)."""
        layout = get_layout(3)
        assert layout.hidden_size == 1184

    def test_auction_slot_info_size(self):
        """Auction slot info = 5 * num_players."""
        for num_players in [2, 3, 4, 5, 6]:
            layout = get_layout(num_players)
            assert layout.auction_slot_info_size == 5 * num_players, (
                f"{num_players} players: auction slot info {layout.auction_slot_info_size} != {5 * num_players}"
            )


class TestAuctionSlotInfo:
    """Verify the auction slot info block is populated correctly."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_auction_slots_populated_on_init(self, state):
        """Auction slot info block must contain nonzero data after initialization."""
        layout = get_layout(3)
        block = state._array[layout.auction_slot_info_offset:layout.auction_slot_info_offset + layout.auction_slot_info_size]
        assert np.count_nonzero(block) > 0

    def test_auction_slot_data_matches_companies(self, state):
        """Each slot's data should match the actual auction company's static data."""
        layout = get_layout(3)
        coo_level = TURN.get_coo_level(state)
        for slot in range(3):
            company_id = get_auction_company_for_slot_py(state, slot)
            if company_id < 0:
                continue
            base = layout.auction_slot_info_offset + slot * 5
            assert abs(state._array[base + 0] - get_company_stars(company_id) / PY_STAR_DIVISOR) < 1e-6
            assert abs(state._array[base + 1] - get_company_low_price(company_id) / PY_CASH_DIVISOR) < 1e-6
            assert abs(state._array[base + 2] - get_company_face_value(company_id) / PY_CASH_DIVISOR) < 1e-6
            assert abs(state._array[base + 3] - get_company_high_price(company_id) / PY_CASH_DIVISOR) < 1e-6
            expected_income = get_adjusted_company_income(company_id, coo_level) / PY_CASH_DIVISOR
            assert abs(state._array[base + 4] - expected_income) < 1e-6

    def test_empty_slots_are_zero(self, state):
        """When fewer companies available than slots, remaining slots are zero."""
        layout = get_layout(3)

        # Remove all auction companies so all slots become empty
        for cid in range(36):
            if COMPANIES[cid].is_for_auction(state):
                state.set_company_for_auction(cid, False)

        # Re-populate auction slot info
        state._populate_auction_slot_info()

        # All 3 slots should now be zero
        for slot in range(3):
            base = layout.auction_slot_info_offset + slot * 5
            for i in range(5):
                assert state._array[base + i] == 0.0, (
                    f"slot {slot} field {i} should be 0 when no auction companies"
                )


class TestActiveCompany:
    """Verify the active company block in turn state."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_active_company_zeroed_on_init(self, state):
        """Active company one-hot and info should be zero after initialization."""
        layout = get_layout(3)
        oh_base = layout.active_company_offset
        info_base = layout.active_company_info_offset
        for i in range(36):
            assert state._array[oh_base + i] == 0.0, f"active_company_oh[{i}] != 0.0 at init"
        for i in range(5):
            assert state._array[info_base + i] == 0.0, f"active_company_info[{i}] != 0.0 at init"

    def test_set_active_company(self, state):
        """set_active_company should populate the 5 info scalars correctly."""
        layout = get_layout(3)
        company_id = 5  # An arbitrary company
        coo_level = TURN.get_coo_level(state)

        state.set_active_company(company_id)
        base = layout.active_company_info_offset
        assert abs(state._array[base + 0] - get_company_stars(company_id) / PY_STAR_DIVISOR) < 1e-6
        assert abs(state._array[base + 1] - get_company_low_price(company_id) / PY_CASH_DIVISOR) < 1e-6
        assert abs(state._array[base + 2] - get_company_face_value(company_id) / PY_CASH_DIVISOR) < 1e-6
        assert abs(state._array[base + 3] - get_company_high_price(company_id) / PY_CASH_DIVISOR) < 1e-6
        expected_income = get_adjusted_company_income(company_id, coo_level) / PY_CASH_DIVISOR
        assert abs(state._array[base + 4] - expected_income) < 1e-6

    def test_clear_active_company(self, state):
        """clear_active_company should zero out the 5 info scalars."""
        layout = get_layout(3)
        state.set_active_company(10)  # Set first
        state.clear_active_company()  # Then clear
        base = layout.active_company_info_offset
        for i in range(5):
            assert state._array[base + i] == 0.0, f"active_company_info[{i}] != 0.0 after clear"


class TestActiveCorp:
    """Verify the active corp block in turn state."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_active_corp_zeroed_on_init(self, state):
        """Active corp one-hot, info, and companies should be zero after init."""
        layout = get_layout(3)
        oh_base = layout.active_corp_offset
        info_base = layout.active_corp_info_offset
        co_base = layout.active_corp_companies_offset
        for i in range(8):
            assert state._array[oh_base + i] == 0.0, f"active_corp_oh[{i}] != 0.0 at init"
        for i in range(3):
            assert state._array[info_base + i] == 0.0, f"active_corp_info[{i}] != 0.0 at init"
        for i in range(36):
            assert state._array[co_base + i] == 0.0, f"active_corp_companies[{i}] != 0.0 at init"

    def test_set_active_corp(self, state):
        """set_active_corp should populate info and owned companies from corp data."""
        layout = get_layout(3)
        # Float a corp first so it has meaningful data
        from tests.phases.conftest import float_corp_for_test
        float_corp_for_test(state, corp_id=0, par_index=10, player_id=0)

        state.set_active_corp(0)
        info_base = layout.active_corp_info_offset
        co_base = layout.active_corp_companies_offset

        # Info scalars: income, stars, share_price (already normalized in corp data)
        corp_ptr_income = state._array[layout.corps_offset + 0 * layout.corp_stride + 5]  # income offset=5
        corp_ptr_stars = state._array[layout.corps_offset + 0 * layout.corp_stride + 6]  # stars offset=6
        corp_ptr_price = state._array[layout.corps_offset + 0 * layout.corp_stride + 7]  # share_price offset=7
        assert abs(state._array[info_base + 0] - corp_ptr_income) < 1e-6
        assert abs(state._array[info_base + 1] - corp_ptr_stars) < 1e-6
        assert abs(state._array[info_base + 2] - corp_ptr_price) < 1e-6

        # Owned companies should match corp's owned_companies block
        corp_owned_offset = layout.corps_offset + 0 * layout.corp_stride + 37  # owned_companies offset=37
        for i in range(36):
            assert state._array[co_base + i] == state._array[corp_owned_offset + i], (
                f"active_corp_companies[{i}] doesn't match corp owned_companies"
            )

    def test_clear_active_corp(self, state):
        """clear_active_corp should zero out info and owned companies."""
        layout = get_layout(3)
        from tests.phases.conftest import float_corp_for_test
        float_corp_for_test(state, corp_id=0, par_index=10, player_id=0)
        state.set_active_corp(0)
        state.clear_active_corp()

        info_base = layout.active_corp_info_offset
        co_base = layout.active_corp_companies_offset
        for i in range(3):
            assert state._array[info_base + i] == 0.0, f"active_corp_info[{i}] != 0.0 after clear"
        for i in range(36):
            assert state._array[co_base + i] == 0.0, f"active_corp_companies[{i}] != 0.0 after clear"


class TestPlayerIncome:
    """Verify the player income field in visible state."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_player_income_zero_at_init(self, state):
        """Players have no companies at init, income should be 0."""
        for p in range(3):
            assert PLAYERS[p].get_income(state) == 0

    def test_player_income_after_company_transfer(self, state):
        """Transfer a company to player, income should update."""
        # Transfer company 5 (1-star, red) to player 0
        COMPANIES[5].transfer_to_player(state, 0)
        income = PLAYERS[0].get_income(state)
        expected = get_adjusted_company_income(5, TURN.get_coo_level(state))
        assert income == expected

    def test_player_income_cleared_on_remove(self, state):
        """Removing a player's company should update income."""
        COMPANIES[5].transfer_to_player(state, 0)
        assert PLAYERS[0].get_income(state) != 0
        COMPANIES[5].remove_from_game(state)
        assert PLAYERS[0].get_income(state) == 0


class TestFIIncome:
    """Verify the FI income field in visible state."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_fi_income_at_init(self, state):
        """FI has no companies at init but gets +5 base bonus."""
        assert FI.get_income(state) == 5

    def test_fi_income_after_company_transfer(self, state):
        """Transfer a company to FI, income should update."""
        COMPANIES[5].transfer_to_fi(state)
        expected = get_adjusted_company_income(5, TURN.get_coo_level(state)) + 5
        assert FI.get_income(state) == expected

    def test_fi_income_cleared_on_remove(self, state):
        """Removing FI's company should update income back to base."""
        COMPANIES[5].transfer_to_fi(state)
        assert FI.get_income(state) != 5
        COMPANIES[5].remove_from_game(state)
        assert FI.get_income(state) == 5


class TestCardsRemaining:
    """Verify the cards remaining field in turn state."""

    @pytest.fixture
    def state(self):
        gs = GameState(3)
        gs.initialize_game(seed=42)
        return gs

    def test_cards_remaining_at_init(self, state):
        """Cards remaining should reflect deck after initial draws."""
        remaining = DECK.get_remaining_count(state)
        layout = get_layout(3)
        # cards_remaining is the last field in turn state
        cr_offset = layout.turn_offset + layout.turn_size - 1
        stored = state._array[cr_offset]
        expected = remaining / 36.0
        assert abs(stored - expected) < 1e-6

    def test_cards_remaining_decreases_on_draw(self, state):
        """Drawing a card should decrease cards_remaining."""
        before = DECK.get_remaining_count(state)
        DECK.draw(state)
        after = DECK.get_remaining_count(state)
        assert after == before - 1
        layout = get_layout(3)
        cr_offset = layout.turn_offset + layout.turn_size - 1
        stored = state._array[cr_offset]
        assert abs(stored - after / 36.0) < 1e-6
