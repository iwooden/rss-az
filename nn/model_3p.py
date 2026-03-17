"""AlphaZero-style PyTorch model for Rolling Stock Stars (3 players).

This module implements a residual MLP architecture sized around ~26M parameters:
- Input (visible state): 1763 floats (3-player)
- Policy output: 246 logits (all actions for 3-player layout)
- Value output: 3 scalars in [-1, 1] (per-player expected outcomes via tanh)

The value head outputs per-player expected outcomes [v_active, v_next, v_next_next],
where the active player is always at index 0, and subsequent players follow in
turn order. Values are bounded to [-1, 1] by a tanh activation, where 1.0 maps
to a first-place finish and -1.0 maps to a last-place finish.

The model expects legal-action masking to be provided by the Cython engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RSSModelConfig:
    """Configuration for the residual MLP trunk and heads."""

    input_dim: int = 1763
    action_dim: int = 246
    value_dim: int = 3  # Per-player expected outcomes: [v_active, v_next, v_next_next]
    hidden_dim: int = 768
    num_blocks: int = 10
    expansion: int = 2
    dropout: float = 0.0


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block with expansion factor."""

    def __init__(self, hidden_dim: int, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return residual + x


class RSSAlphaZeroNet(nn.Module):
    """Residual MLP model with policy and value heads for RSS."""

    def __init__(self, cfg: RSSModelConfig = RSSModelConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim, cfg.expansion, cfg.dropout) for _ in range(cfg.num_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 3),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 3, cfg.action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim // 4, cfg.value_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers; stable defaults for LayerNorm.

        Residual block fc2 layers are zero-initialized so blocks start as
        identity functions, improving training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Zero-init last linear in each residual block so blocks start as identity
        for block in self.blocks:
            assert isinstance(block, ResidualMLPBlock)
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: shape [batch, input_dim], float32 visible-state vectors.

        Returns:
            policy_logits: shape [batch, action_dim], raw (unmasked) logits.
            values: shape [batch, value_dim], per-player expected outcomes in [-1, 1].
                Index 0 = active player, subsequent indices follow turn order.
        """
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.trunk_norm(h)

        policy_logits = self.policy_head(h)
        values = self.value_head(h)
        return policy_logits, values


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = RSSModelConfig(
        input_dim=1763,
        action_dim=246,
        value_dim=3,
        hidden_dim=768,
        num_blocks=10,
        expansion=2,
        dropout=0.0,
    )
    model = RSSAlphaZeroNet(cfg)
    print(f"Trainable parameters: {count_parameters(model):,}")

    batch_size = 4
    x = torch.randn(batch_size, cfg.input_dim)
    legal_mask = torch.ones(batch_size, cfg.action_dim)
    legal_mask[:, -5:] = 0.0

    policy_logits, values = model(x)
    # Mask and softmax applied outside the model
    policy_logits.masked_fill_(legal_mask <= 0, -1e9)
    policy = torch.softmax(policy_logits, dim=-1)

    print("policy_logits:", tuple(policy_logits.shape))
    print("values:", tuple(values.shape))
    print("policy row sum:", policy[0].sum().item())
    print("values (first sample):", values[0].tolist())
    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("all values in [-1, 1]: ok")
