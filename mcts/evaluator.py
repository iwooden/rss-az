"""NN evaluator for MCTS leaf evaluation.

Handles state rotation (active player → slot 0), legal action masking,
neural network inference, and value un-rotation to canonical player order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from core.data import GameConstants

NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)
NUM_CORPS = int(GameConstants.NUM_CORPS)
NUM_MARKET_SPACES = int(GameConstants.NUM_MARKET_SPACES)
NUM_PHASES = int(GameConstants.NUM_PHASES)
NUM_COO_LEVELS = int(GameConstants.NUM_COO_LEVELS)
MAX_DIVIDEND = int(GameConstants.MAX_DIVIDEND)

# Static company data: stars(1) + low_price(1) + face_value(1) + high_price(1) + synergies(36)
STATIC_COMPANY_SIZE = 4 + NUM_COMPANIES  # 40


@dataclass(frozen=True)
class VisibleLayout:
    """Python-computed visible state layout offsets (mirrors Cython StateLayout)."""

    num_players: int
    player_stride: int
    players_offset: int
    players_size: int
    fi_offset: int
    fi_size: int
    turn_offset: int
    turn_size: int
    visible_size: int
    # Per-player turn field offsets (absolute, within visible state)
    auction_high_bidder_offset: int
    auction_starter_offset: int
    auction_passed_offset: int


def compute_visible_layout(num_players: int) -> VisibleLayout:
    """Compute visible state layout offsets for a given player count.

    Mirrors the logic in core/state.pyx compute_layout() and
    compute_turn_offsets().
    """
    offset = 0

    # Phase one-hot
    offset += NUM_PHASES  # 11

    # CoO one-hot
    offset += NUM_COO_LEVELS  # 7

    # Players
    players_offset = offset
    player_stride = (
        1  # cash
        + 1  # net_worth
        + num_players  # turn_order one-hot
        + 1  # is_auction_high_bidder
        + NUM_COMPANIES  # owned_companies
        + NUM_CORPS  # owned_shares
        + NUM_CORPS  # is_president
        + NUM_CORPS  # share_buys
        + NUM_CORPS  # share_sells
        + 1  # acquisition_proceeds
    )
    players_size = player_stride * num_players
    offset += players_size

    # Foreign Investor
    fi_offset = offset
    fi_size = 1 + NUM_COMPANIES  # cash + owned companies
    offset += fi_size

    # Company locations (auction, revealed, removed)
    offset += NUM_COMPANIES * 3

    # Company adjusted incomes
    offset += NUM_COMPANIES

    # Market availability
    offset += NUM_MARKET_SPACES

    # Corporations
    corp_stride = (
        1  # active
        + 1  # cash
        + 1  # unissued_shares
        + 1  # issued_shares
        + 1  # bank_shares
        + 1  # income
        + 1  # stars
        + 1  # share_price
        + 1  # acquisition_proceeds
        + 1  # in_receivership
        + NUM_MARKET_SPACES  # price_index one-hot
        + NUM_COMPANIES  # owned_companies
        + NUM_COMPANIES  # acquisition_companies
    )
    offset += corp_stride * NUM_CORPS

    # Turn state
    turn_offset = offset
    turn_inner = 0
    # turn_number, end_card_flipped, consecutive_passes
    turn_inner += 3
    # auction_company (36 one-hot)
    turn_inner += NUM_COMPANIES
    # auction_price
    turn_inner += 1
    # Per-player turn fields
    auction_high_bidder_rel = turn_inner
    turn_inner += num_players
    auction_starter_rel = turn_inner
    turn_inner += num_players
    auction_passed_rel = turn_inner
    turn_inner += num_players
    # dividend_corp, dividend_impact, dividend_remaining
    turn_inner += NUM_CORPS + MAX_DIVIDEND + NUM_CORPS
    # issue_corp, issue_remaining
    turn_inner += NUM_CORPS + NUM_CORPS
    # ipo_company, ipo_remaining
    turn_inner += NUM_COMPANIES + NUM_COMPANIES
    # acq_active_corp, acq_target_company, acq_is_fi_offer
    turn_inner += NUM_CORPS + NUM_COMPANIES + 1
    # closing_company
    turn_inner += NUM_COMPANIES
    turn_size = turn_inner
    offset += turn_size

    # Static company data
    offset += STATIC_COMPANY_SIZE * NUM_COMPANIES

    visible_size = offset

    return VisibleLayout(
        num_players=num_players,
        player_stride=player_stride,
        players_offset=players_offset,
        players_size=players_size,
        fi_offset=fi_offset,
        fi_size=fi_size,
        turn_offset=turn_offset,
        turn_size=turn_size,
        visible_size=visible_size,
        auction_high_bidder_offset=turn_offset + auction_high_bidder_rel,
        auction_starter_offset=turn_offset + auction_starter_rel,
        auction_passed_offset=turn_offset + auction_passed_rel,
    )


# Cache layouts per player count
_layout_cache: dict[int, VisibleLayout] = {}


def get_layout(num_players: int) -> VisibleLayout:
    """Get cached layout for a given player count."""
    if num_players not in _layout_cache:
        _layout_cache[num_players] = compute_visible_layout(num_players)
    return _layout_cache[num_players]


def rotate_visible_state(state_array: np.ndarray, active_player_id: int,
                         num_players: int) -> np.ndarray:
    """Return a copy of the visible state with player data rotated.

    After rotation, the active player's data is at slot 0 (what the NN expects).

    Rotates:
    - Player data blocks (contiguous, each player_stride floats)
    - Turn state per-player fields: auction_high_bidder, auction_starter,
      auction_passed (each num_players floats)

    Args:
        state_array: Full state array (visible + hidden).
        active_player_id: Canonical player ID (0 to num_players-1).
        num_players: Number of players in the game.

    Returns:
        Copy of visible state portion with rotation applied.
    """
    layout = get_layout(num_players)
    visible = state_array[:layout.visible_size].copy()

    if active_player_id == 0:
        return visible  # No rotation needed

    # Rotate player data blocks
    p_off = layout.players_offset
    stride = layout.player_stride
    # Extract all player blocks, then roll
    player_data = visible[p_off:p_off + layout.players_size].copy()
    player_blocks = player_data.reshape(num_players, stride)
    rotated_blocks = np.roll(player_blocks, -active_player_id, axis=0)
    visible[p_off:p_off + layout.players_size] = rotated_blocks.ravel()

    # Rotate per-player turn state fields
    for field_offset in (layout.auction_high_bidder_offset,
                         layout.auction_starter_offset,
                         layout.auction_passed_offset):
        field = visible[field_offset:field_offset + num_players].copy()
        visible[field_offset:field_offset + num_players] = np.roll(
            field, -active_player_id
        )

    return visible


def unrotate_values(values: np.ndarray, active_player_id: int) -> np.ndarray:
    """Convert NN value output (active player at index 0) to canonical order.

    The NN outputs values where index 0 = active player, index 1 = next player, etc.
    This rotates them back so index 0 = player 0, index 1 = player 1, etc.

    Args:
        values: Per-player values from NN, shape (num_players,).
        active_player_id: Canonical player ID of the active player.

    Returns:
        Values in canonical player order.
    """
    return np.roll(values, active_player_id)


def compute_terminal_values(net_worths: list[int], num_players: int) -> np.ndarray:
    """Compute canonical reward values for a terminal game state.

    Ranks players by net worth. Rewards are evenly distributed from
    +1.0 (1st place) to -1.0 (last place). Ties receive averaged rewards.

    For 3 players: 1st=1.0, 2nd=0.0, 3rd=-1.0

    Args:
        net_worths: List of net worth values per player (canonical order).
        num_players: Number of players.

    Returns:
        np.ndarray of shape (num_players,) with reward values per player.
    """
    # Rank rewards: evenly distributed from +1.0 to -1.0
    rank_rewards = np.linspace(1.0, -1.0, num_players)

    # Sort players by net worth descending, stable sort for tie consistency
    sorted_indices = np.argsort(net_worths)[::-1]  # descending

    values = np.zeros(num_players, dtype=np.float32)

    i = 0
    while i < num_players:
        # Find group of tied players
        j = i + 1
        while j < num_players and net_worths[sorted_indices[j]] == net_worths[sorted_indices[i]]:
            j += 1

        # Average reward for tied positions
        avg_reward = float(np.mean(rank_rewards[i:j]))
        for k in range(i, j):
            values[sorted_indices[k]] = avg_reward

        i = j

    return values


class NNEvaluator:
    """Wraps a neural network model for MCTS leaf evaluation.

    Handles state rotation, legal action masking, inference,
    and value un-rotation to canonical player order.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_players: int = 3) -> None:
        self.model = model
        self.device = device
        self.num_players = num_players
        self.layout = get_layout(num_players)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a game state with the neural network.

        Args:
            state: GameState object.

        Returns:
            Tuple of (policy_probs, canonical_values):
            - policy_probs: shape (action_dim,), softmax over legal actions
            - canonical_values: shape (num_players,), per-player values in
              canonical order (index 0 = player 0, etc.)
        """
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()

        # Rotate visible state so active player is at slot 0
        rotated_visible = rotate_visible_state(
            state._array, active_player, self.num_players
        )

        # Get legal action mask
        mask_np = get_valid_action_mask(state)

        # Convert to tensors
        x = torch.from_numpy(rotated_visible).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

        # Forward pass
        policy_logits, value_output = self.model(x, legal_action_mask=mask)

        # Policy: softmax over masked logits
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        # Value: already has tanh applied in the model, un-rotate to canonical
        values_rotated = value_output.squeeze(0).cpu().numpy()
        canonical_values = unrotate_values(values_rotated, active_player)

        return policy_probs, canonical_values

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values from a game-over state.

        Args:
            state: GameState in GAME_OVER phase.

        Returns:
            Canonical values, shape (num_players,).
        """
        net_worths = [state.get_player_net_worth(i) for i in range(self.num_players)]
        return compute_terminal_values(net_worths, self.num_players)
