"""Player-slot permutation augmentation for training.

After rotation puts the active player at slot 0, the ordering of slots 1..N-1
is arbitrary. Randomly permuting these inactive slots acts as data augmentation
(regularization), teaching the NN slot-order invariance.

The permutation is applied to 4 groups of player-indexed data:
  1. Player data blocks (contiguous, player_stride floats each)
  2. auction_high_bidder (one-hot, num_players floats)
  3. auction_starter (one-hot, num_players floats)
  4. auction_passed (flags, num_players floats)

Plus the corresponding columns of the value targets.
Policy targets and legal masks are invariant (no action references a player slot).
"""

from __future__ import annotations

import torch

from core.state import LayoutInfo


def apply_player_permutation(
    states: torch.Tensor,
    value_targets: torch.Tensor,
    perm: torch.Tensor,
    layout: LayoutInfo,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a permutation of inactive player slots to a batch.

    Args:
        states: [batch, state_size] float32 state vectors (modified in-place).
        value_targets: [batch, num_players] float32 value targets (modified in-place).
        perm: 1-D int64 tensor of length num_players, mapping destination slot
              to source slot. perm[0] MUST be 0 (active player stays fixed).
              Example for 3p swap: [0, 2, 1].
        layout: LayoutInfo from get_layout(num_players).

    Returns:
        (states, value_targets) — the same tensors, mutated in-place.
    """
    num_players = layout.num_players
    player_stride = layout.player_stride
    players_offset = layout.players_offset

    # --- Player data blocks ---
    # Gather all player blocks into [batch, num_players, player_stride],
    # permute along the player axis, write back.
    block_start = players_offset
    block_end = players_offset + num_players * player_stride
    player_blocks = states[:, block_start:block_end].view(-1, num_players, player_stride)
    states[:, block_start:block_end] = player_blocks[:, perm].reshape(-1, num_players * player_stride)

    # --- Per-player turn state fields (3 fields, each num_players floats) ---
    for field_offset in (
        layout.auction_high_bidder_offset,
        layout.auction_starter_offset,
        layout.auction_passed_offset,
    ):
        field_end = field_offset + num_players
        states[:, field_offset:field_end] = states[:, field_offset:field_end][:, perm]

    # --- Value targets ---
    value_targets[:] = value_targets[:, perm]

    return states, value_targets


def random_player_permutation(
    num_players: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a random permutation that fixes slot 0 and shuffles 1..N-1.

    Returns:
        1-D int64 tensor of length num_players.
    """
    perm = torch.arange(num_players, device=device)
    # Permute only the inactive slots [1, num_players)
    inactive = torch.randperm(num_players - 1, device=device, generator=generator) + 1
    perm[1:] = inactive
    return perm
