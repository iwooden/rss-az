"""AlphaZero-style PyTorch model for Rolling Stock Stars (3 players) — v2.

Architecture changes from v1 (model_3p.py) based on interpretability analysis:
- Two-layer input preprocessing (input → 2*hidden → hidden) instead of a single
  linear projection. V1's block 0 did 92% of the work because one linear layer
  was too bottlenecked for the input (v1 had 1296 static synergy flags, now removed).
- 10 residual blocks (restored from 6). Probing showed effective rank still growing
  at block 5 and all blocks active with flat contribution profile (0.07-0.11),
  indicating the model can use more depth.
- hidden_dim=384 (down from 768). Effective rank was 150-192 across all layers;
  384 gives ~2x headroom. Multiple of 64 for GPU tensor core alignment.
- Asymmetric heads informed by probing analysis:
  - Policy head: 2 hidden layers (hidden→2*hidden→hidden→action_dim). Probing showed
    trunk activations are linearly poor for policy (0.58 acc) but good for value (R²=0.97),
    meaning the policy head needs more nonlinear capacity.
  - Value head: 1 hidden layer (hidden→hidden→value_dim). Trunk already computes value
    almost linearly; extra layers would be wasted capacity.

~8.2M parameters (down from ~26.6M).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RSSModelConfig2:
    """Configuration for the v2 residual MLP trunk and heads."""

    input_dim: int = 1559  # get_layout(3).visible_size; always pass explicitly
    action_dim: int = 246
    value_dim: int = 3
    hidden_dim: int = 384
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


class RSSAlphaZeroNet2(nn.Module):
    """V2 residual MLP with multi-layer input preprocessing."""

    def __init__(self, cfg: RSSModelConfig2 = RSSModelConfig2()) -> None:
        super().__init__()
        self.cfg = cfg

        # Two-layer input preprocessing:
        # input_dim → 2*hidden_dim → hidden_dim
        self.input_preprocess = nn.Sequential(
            nn.Linear(cfg.input_dim, 2 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim, cfg.expansion, cfg.dropout) for _ in range(cfg.num_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)

        self.policy_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.value_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for linear layers; zero-init residual block fc2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
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
        """
        h = self.input_preprocess(x)
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
    from core.state import get_layout
    _layout = get_layout(3)
    cfg = RSSModelConfig2(input_dim=_layout.visible_size, action_dim=186 + 3 * 20)
    model = RSSAlphaZeroNet2(cfg)
    total = count_parameters(model)
    print(f"Trainable parameters: {total:,}")

    # Parameter breakdown
    preprocess_params = sum(p.numel() for p in model.input_preprocess.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    norm_params = sum(p.numel() for p in model.trunk_norm.parameters())
    policy_params = sum(p.numel() for p in model.policy_head.parameters())
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print(f"\nParameter breakdown:")
    print(f"  Input preprocess: {preprocess_params:>12,}  ({preprocess_params/total*100:.1f}%)")
    print(f"  Residual blocks:  {block_params:>12,}  ({block_params/total*100:.1f}%)")
    print(f"  Trunk norm:       {norm_params:>12,}  ({norm_params/total*100:.1f}%)")
    print(f"  Policy head:      {policy_params:>12,}  ({policy_params/total*100:.1f}%)")
    print(f"  Value head:       {value_params:>12,}  ({value_params/total*100:.1f}%)")

    # Smoke test
    batch_size = 4
    x = torch.randn(batch_size, cfg.input_dim)
    policy_logits, values = model(x)

    print(f"\npolicy_logits: {tuple(policy_logits.shape)}")
    print(f"values: {tuple(values.shape)}")
    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("all values in [-1, 1]: ok")
