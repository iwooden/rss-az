"""Tests for player-slot permutation augmentation.

Validates that apply_player_permutation correctly shuffles the 4 groups of
player-indexed state data and value targets while leaving everything else
untouched. Mistakes here could silently destabilize training, so coverage
is deliberately thorough.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from core.state import GameState, get_layout
from core.driver import DRIVER
from core.actions import get_valid_action_mask, get_total_action_count
from train.augment import apply_player_permutation, random_player_permutation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=[3, 4, 6], ids=["3p", "4p", "6p"])
def num_players(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def layout(num_players: int):
    return get_layout(num_players)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_state_batch(
    num_players: int,
    layout,
    rng: np.random.Generator,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a batch with distinctive per-player values.

    Returns (states, legal_masks, policy_targets, value_targets).
    """
    vis = layout.visible_size
    action_dim = get_total_action_count(num_players)

    states = torch.from_numpy(rng.standard_normal((batch_size, vis)).astype(np.float32))
    legal_masks = torch.zeros(batch_size, action_dim)
    policy_targets = torch.zeros(batch_size, action_dim)
    value_targets = torch.from_numpy(
        rng.standard_normal((batch_size, num_players)).astype(np.float32)
    )

    # Write distinctive values per player slot so we can track permutations.
    stride = layout.player_stride
    off = layout.players_offset
    for p in range(num_players):
        start = off + p * stride
        # Fill each player block with (p + 1) * 0.1 so they're distinguishable
        states[:, start : start + stride] = (p + 1) * 0.1

    # Write distinctive per-player turn fields
    for field_offset in (
        layout.auction_high_bidder_offset,
        layout.auction_starter_offset,
        layout.auction_passed_offset,
    ):
        for p in range(num_players):
            states[:, field_offset + p] = (p + 1) * 0.01

    return states, legal_masks, policy_targets, value_targets


def _identity_perm(num_players: int) -> torch.Tensor:
    return torch.arange(num_players)


def _all_inactive_perms(num_players: int) -> list[torch.Tensor]:
    """All permutations that fix slot 0 and permute 1..N-1."""
    inactive = list(range(1, num_players))
    perms = []
    for p in itertools.permutations(inactive):
        perm = torch.tensor([0] + list(p))
        perms.append(perm)
    return perms


def _inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a permutation."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    return inv


# ---------------------------------------------------------------------------
# Test 1: Identity permutation is a no-op
# ---------------------------------------------------------------------------


class TestIdentityNoOp:
    def test_identity_leaves_state_unchanged(self, num_players, layout, rng):
        states, _, _, value_targets = _make_state_batch(num_players, layout, rng)
        orig_states = states.clone()
        orig_values = value_targets.clone()

        perm = _identity_perm(num_players)
        apply_player_permutation(states, value_targets, perm, layout)

        torch.testing.assert_close(states, orig_states)
        torch.testing.assert_close(value_targets, orig_values)


# ---------------------------------------------------------------------------
# Test 2: Round-trip / cycle recovery
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_perm_then_inverse_recovers_original(self, num_players, layout, rng):
        states, _, _, value_targets = _make_state_batch(num_players, layout, rng)
        orig_states = states.clone()
        orig_values = value_targets.clone()

        perm = random_player_permutation(num_players, torch.device("cpu"))
        inv = _inverse_perm(perm)

        apply_player_permutation(states, value_targets, perm, layout)
        apply_player_permutation(states, value_targets, inv, layout)

        torch.testing.assert_close(states, orig_states)
        torch.testing.assert_close(value_targets, orig_values)

    def test_swap_applied_twice_recovers_original(self, layout, rng):
        """A transposition (swap) has order 2: apply twice = identity."""
        num_players = 3
        states, _, _, value_targets = _make_state_batch(num_players, layout, rng)
        if layout.num_players != 3:
            pytest.skip("swap test only for 3p")

        orig_states = states.clone()
        orig_values = value_targets.clone()

        swap = torch.tensor([0, 2, 1])
        apply_player_permutation(states, value_targets, swap, layout)
        apply_player_permutation(states, value_targets, swap, layout)

        torch.testing.assert_close(states, orig_states)
        torch.testing.assert_close(value_targets, orig_values)

    def test_cyclic_perm_order_recovery(self, num_players, layout, rng):
        """A cyclic permutation of N-1 elements has order N-1."""
        n = num_players - 1
        if n < 2:
            pytest.skip("need at least 2 inactive players")

        states, _, _, value_targets = _make_state_batch(num_players, layout, rng)
        orig_states = states.clone()
        orig_values = value_targets.clone()

        # Cyclic perm: 0 -> [0, 2, 3, ..., N-1, 1]
        cycle = torch.tensor([0] + list(range(2, num_players)) + [1])

        for _ in range(n):
            apply_player_permutation(states, value_targets, cycle, layout)

        torch.testing.assert_close(states, orig_states)
        torch.testing.assert_close(value_targets, orig_values)


# ---------------------------------------------------------------------------
# Test 3: Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_sequential_equals_composed(self, num_players, layout, rng):
        """apply(A) then apply(B) == apply(B[A])."""
        states_seq, _, _, values_seq = _make_state_batch(num_players, layout, rng)
        states_comp = states_seq.clone()
        values_comp = values_seq.clone()

        gen = torch.Generator().manual_seed(123)
        perm_a = random_player_permutation(num_players, torch.device("cpu"), gen)
        gen = torch.Generator().manual_seed(456)
        perm_b = random_player_permutation(num_players, torch.device("cpu"), gen)

        # Sequential: A then B
        apply_player_permutation(states_seq, values_seq, perm_a, layout)
        apply_player_permutation(states_seq, values_seq, perm_b, layout)

        # Composed: A then B = A[B] (gather is right-to-left composition)
        composed = perm_a[perm_b]
        apply_player_permutation(states_comp, values_comp, composed, layout)

        torch.testing.assert_close(states_seq, states_comp)
        torch.testing.assert_close(values_seq, values_comp)


# ---------------------------------------------------------------------------
# Test 4: Known-value spot checks
# ---------------------------------------------------------------------------


class TestKnownValues:
    def test_player_blocks_swap_correctly(self, num_players, layout, rng):
        """Verify specific player block values land in expected slots."""
        states, _, _, _ = _make_state_batch(num_players, layout, rng)
        stride = layout.player_stride
        off = layout.players_offset

        # Build a swap: slot 1 <-> slot N-1 (if N > 2), else slot 1 <-> 2
        perm = _identity_perm(num_players).clone()
        perm[1] = num_players - 1
        perm[num_players - 1] = 1

        orig_block_1 = states[:, off + stride : off + 2 * stride].clone()
        orig_block_last = states[
            :, off + (num_players - 1) * stride : off + num_players * stride
        ].clone()

        value_targets = torch.zeros(states.shape[0], num_players)
        apply_player_permutation(states, value_targets, perm, layout)

        # Block at slot 1 should now contain what was in slot N-1
        torch.testing.assert_close(
            states[:, off + stride : off + 2 * stride], orig_block_last
        )
        # Block at slot N-1 should now contain what was in slot 1
        torch.testing.assert_close(
            states[
                :, off + (num_players - 1) * stride : off + num_players * stride
            ],
            orig_block_1,
        )

    def test_turn_fields_swap_correctly(self, num_players, layout, rng):
        """Verify per-player turn fields are permuted."""
        states, _, _, _ = _make_state_batch(num_players, layout, rng)

        perm = _identity_perm(num_players).clone()
        perm[1] = num_players - 1
        perm[num_players - 1] = 1

        orig_fields = {}
        for name, field_offset in [
            ("high_bidder", layout.auction_high_bidder_offset),
            ("starter", layout.auction_starter_offset),
            ("passed", layout.auction_passed_offset),
        ]:
            orig_fields[name] = states[
                :, field_offset : field_offset + num_players
            ].clone()

        value_targets = torch.zeros(states.shape[0], num_players)
        apply_player_permutation(states, value_targets, perm, layout)

        for name, field_offset in [
            ("high_bidder", layout.auction_high_bidder_offset),
            ("starter", layout.auction_starter_offset),
            ("passed", layout.auction_passed_offset),
        ]:
            result = states[:, field_offset : field_offset + num_players]
            # Slot 1 should have what was in slot N-1
            torch.testing.assert_close(result[:, 1], orig_fields[name][:, num_players - 1])
            # Slot N-1 should have what was in slot 1
            torch.testing.assert_close(result[:, num_players - 1], orig_fields[name][:, 1])

    def test_value_targets_permuted(self, num_players, layout, rng):
        states, _, _, value_targets = _make_state_batch(num_players, layout, rng)

        perm = _identity_perm(num_players).clone()
        perm[1] = num_players - 1
        perm[num_players - 1] = 1

        orig_values = value_targets.clone()
        apply_player_permutation(states, value_targets, perm, layout)

        torch.testing.assert_close(value_targets[:, 0], orig_values[:, 0])
        torch.testing.assert_close(value_targets[:, 1], orig_values[:, num_players - 1])
        torch.testing.assert_close(value_targets[:, num_players - 1], orig_values[:, 1])


# ---------------------------------------------------------------------------
# Test 5: Slot 0 (active player) invariance
# ---------------------------------------------------------------------------


class TestSlot0Invariance:
    def test_active_player_block_unchanged(self, num_players, layout, rng):
        """Slot 0 player data must be identical across all permutations."""
        states, _, _, _ = _make_state_batch(num_players, layout, rng)
        off = layout.players_offset
        stride = layout.player_stride
        orig_slot0 = states[:, off : off + stride].clone()

        for perm in _all_inactive_perms(num_players):
            test_states = states.clone()
            values = torch.zeros(states.shape[0], num_players)
            apply_player_permutation(test_states, values, perm, layout)
            torch.testing.assert_close(
                test_states[:, off : off + stride], orig_slot0
            )

    def test_active_player_value_unchanged(self, num_players, layout, rng):
        """value_targets[:, 0] must be identical across all permutations."""
        _, _, _, value_targets = _make_state_batch(num_players, layout, rng)
        orig_v0 = value_targets[:, 0].clone()

        for perm in _all_inactive_perms(num_players):
            test_values = value_targets.clone()
            # Need a dummy state
            states = torch.zeros(value_targets.shape[0], layout.visible_size)
            apply_player_permutation(states, test_values, perm, layout)
            torch.testing.assert_close(test_values[:, 0], orig_v0)

    def test_active_player_turn_fields_unchanged(self, num_players, layout, rng):
        """Turn field index 0 must be identical across all permutations."""
        states, _, _, _ = _make_state_batch(num_players, layout, rng)

        orig_vals = {}
        for name, off in [
            ("hb", layout.auction_high_bidder_offset),
            ("st", layout.auction_starter_offset),
            ("pa", layout.auction_passed_offset),
        ]:
            orig_vals[name] = states[:, off].clone()

        for perm in _all_inactive_perms(num_players):
            test_states = states.clone()
            values = torch.zeros(states.shape[0], num_players)
            apply_player_permutation(test_states, values, perm, layout)
            for name, off in [
                ("hb", layout.auction_high_bidder_offset),
                ("st", layout.auction_starter_offset),
                ("pa", layout.auction_passed_offset),
            ]:
                torch.testing.assert_close(test_states[:, off], orig_vals[name])


# ---------------------------------------------------------------------------
# Test 6: Policy targets and legal masks invariance
# ---------------------------------------------------------------------------


class TestPolicyMaskInvariance:
    def test_policy_and_mask_untouched(self, num_players, layout, rng):
        """apply_player_permutation should not modify policy or legal masks."""
        states, legal_masks, policy_targets, value_targets = _make_state_batch(
            num_players, layout, rng
        )
        orig_masks = legal_masks.clone()
        orig_policy = policy_targets.clone()

        perm = random_player_permutation(num_players, torch.device("cpu"))
        # The function signature doesn't take these, so they can't be modified.
        # But verify the API contract: only states and value_targets are touched.
        apply_player_permutation(states, value_targets, perm, layout)

        torch.testing.assert_close(legal_masks, orig_masks)
        torch.testing.assert_close(policy_targets, orig_policy)


# ---------------------------------------------------------------------------
# Test 7: Non-player-field invariance
# ---------------------------------------------------------------------------


class TestNonPlayerInvariance:
    def _player_indices(self, layout) -> set[int]:
        """Return all state indices that belong to player-specific fields."""
        indices: set[int] = set()
        num_players = layout.num_players
        stride = layout.player_stride
        off = layout.players_offset

        # Player data blocks
        for p in range(num_players):
            start = off + p * stride
            indices.update(range(start, start + stride))

        # Per-player turn fields
        for field_offset in (
            layout.auction_high_bidder_offset,
            layout.auction_starter_offset,
            layout.auction_passed_offset,
        ):
            indices.update(range(field_offset, field_offset + num_players))

        return indices

    def test_non_player_fields_untouched(self, num_players, layout, rng):
        """Everything outside the 4 player-specific regions must be unchanged."""
        states, _, _, _ = _make_state_batch(num_players, layout, rng)
        orig_states = states.clone()

        player_idx = self._player_indices(layout)
        non_player_idx = sorted(set(range(layout.visible_size)) - player_idx)
        non_player_idx_t = torch.tensor(non_player_idx)

        for perm in _all_inactive_perms(num_players):
            test_states = states.clone()
            values = torch.zeros(states.shape[0], num_players)
            apply_player_permutation(test_states, values, perm, layout)
            torch.testing.assert_close(
                test_states[:, non_player_idx_t],
                orig_states[:, non_player_idx_t],
            )


# ---------------------------------------------------------------------------
# Test 8: Real game state smoke test
# ---------------------------------------------------------------------------


class TestRealGameState:
    def test_permuted_bid_state_consistent(self):
        """Permute a real bid state and verify one-hot fields stay valid."""
        num_players = 3
        layout = get_layout(num_players)
        gs = GameState(num_players)
        gs.initialize_game(seed=42)

        # Advance to BID phase by playing some actions
        rng_local = np.random.RandomState(42)
        for _ in range(200):
            mask = get_valid_action_mask(gs)
            legal = np.flatnonzero(mask)
            if len(legal) == 0:
                break
            # Check if we're in BID phase (phase index 1, visible index 1)
            vis = gs._array[:layout.visible_size]
            if vis[1] == 1.0:  # BID phase one-hot
                break
            action = legal[rng_local.randint(len(legal))]
            DRIVER.apply_action(gs, action, None)

        vis = gs._array[:layout.visible_size].copy()
        states = torch.from_numpy(vis).unsqueeze(0)
        value_targets = torch.tensor([[0.5, 0.3, -0.8]])

        swap = torch.tensor([0, 2, 1])
        apply_player_permutation(states, value_targets, swap, layout)

        # Verify auction_high_bidder is still a valid one-hot or all-zeros
        hb_off = layout.auction_high_bidder_offset
        hb = states[0, hb_off : hb_off + num_players]
        hb_sum = hb.sum().item()
        assert hb_sum == 0.0 or hb_sum == 1.0, f"invalid one-hot sum: {hb_sum}"

        # Verify auction_starter is still a valid one-hot or all-zeros
        st_off = layout.auction_starter_offset
        st = states[0, st_off : st_off + num_players]
        st_sum = st.sum().item()
        assert st_sum == 0.0 or st_sum == 1.0, f"invalid one-hot sum: {st_sum}"

        # Verify value targets swapped correctly
        assert value_targets[0, 0].item() == 0.5  # active player unchanged
        assert value_targets[0, 1].item() == pytest.approx(-0.8)
        assert value_targets[0, 2].item() == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Test: random_player_permutation
# ---------------------------------------------------------------------------


class TestRandomPermutation:
    def test_slot_0_always_fixed(self, num_players):
        """Slot 0 must always be 0 regardless of random seed."""
        for seed in range(100):
            gen = torch.Generator().manual_seed(seed)
            perm = random_player_permutation(num_players, torch.device("cpu"), gen)
            assert perm[0].item() == 0

    def test_is_valid_permutation(self, num_players):
        """Output must be a valid permutation of [0, N)."""
        for seed in range(50):
            gen = torch.Generator().manual_seed(seed)
            perm = random_player_permutation(num_players, torch.device("cpu"), gen)
            assert sorted(perm.tolist()) == list(range(num_players))

    def test_not_always_identity(self):
        """For 3+ players, should produce non-identity permutations sometimes."""
        saw_non_identity = False
        identity = list(range(3))
        for seed in range(100):
            gen = torch.Generator().manual_seed(seed)
            perm = random_player_permutation(3, torch.device("cpu"), gen)
            if perm.tolist() != identity:
                saw_non_identity = True
                break
        assert saw_non_identity, "never produced a non-identity permutation"

    def test_exhaustive_coverage_3p(self):
        """For 3p, both permutations ([0,1,2] and [0,2,1]) should appear."""
        seen: set[tuple[int, ...]] = set()
        for seed in range(200):
            gen = torch.Generator().manual_seed(seed)
            perm = random_player_permutation(3, torch.device("cpu"), gen)
            seen.add(tuple(perm.tolist()))
        assert (0, 1, 2) in seen
        assert (0, 2, 1) in seen
