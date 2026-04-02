"""AlphaZero-style PyTorch model for Rolling Stock Stars (4 players).

Architecture informed by interpretability analysis:
- Three-layer input preprocessing (input -> 3*hidden -> 2*hidden -> hidden) with
  LayerNorm at the output. SVD/probing analysis at epoch 375 showed the old two-layer
  768->256 linear compression losing ~50% of signal (median attenuation 0.502) across
  57/59 feature groups. Adding a nonlinear intermediate (768->512->256 with two GELUs)
  gives the network a richer compression path. LayerNorm stabilizes the residual stream
  entry point (matching pre-LN inside each block).
- 10 residual blocks with expansion=1 (no inner expansion). SVD analysis showed
  expanded widths at 13-16% utilization — removing the expansion halves trunk
  parameters with no representational loss. Extra depth for capacity experiments.
- hidden_dim=256. Multiple of 64 for GPU tensor core alignment.
- Asymmetric heads informed by probing + conductance analysis:
  - Policy: 8 phase-specific heads, each 3 hidden layers at hidden_dim wide
    (hidden->hidden->hidden->hidden->phase_action_count). Replaces a single shared
    head where 63% of neurons were dominated by acq_price and 27.3% were dead.
    Phase dispatch reads the 8-wide phase one-hot from the input state.
  - Value head: 3 hidden layers (hidden->hidden->hidden->hidden->value_dim), matching
    per-phase policy head shape. Deeper value head for capacity experiments.

~4.6M parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RSSModelConfig:
    """Configuration for the residual MLP trunk and heads."""

    input_dim: int  # get_layout(num_players).visible_size
    action_dim: int  # get_total_action_count(num_players)
    value_dim: int  # num_players (per-player expected outcomes)
    hidden_dim: int = 384
    num_blocks: int = 10


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block (no inner expansion)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class RSSAlphaZeroNet(nn.Module):
    """Residual MLP with multi-layer input preprocessing."""

    def __init__(self, cfg: RSSModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Three-layer input preprocessing with LayerNorm:
        # input_dim -> 3*hidden_dim -> 2*hidden_dim -> hidden_dim -> LN
        self.input_preprocess = nn.Sequential(
            nn.Linear(cfg.input_dim, 3 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(3 * cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim) for _ in range(cfg.num_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)

        # Phase-specific policy heads: each outputs only its phase's action count.
        # Eliminates cross-phase interference (epoch 375: 63% of shared-head neurons
        # dominated by acq_price, 27.3% dead, phase identity 99%->83% through head).
        from core.actions import get_action_layout

        num_players = cfg.value_dim
        action_layout = get_action_layout(num_players)
        phase_start_keys = [
            'invest_start', 'bid_start', 'acquisition_start', 'closing_start',
            'dividends_start', 'issue_start', 'ipo_start', 'par_start',
        ]
        phase_starts = [action_layout[k] for k in phase_start_keys]
        phase_ends = phase_starts[1:] + [cfg.action_dim]

        # Store as plain Python lists — these are constants, no need for GPU
        # tensors (which would incur .item() sync overhead in the forward loop).
        self._phase_starts: list[int] = phase_starts
        self._phase_ends: list[int] = phase_ends

        self.phase_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, end - start),
            )
            for start, end in zip(phase_starts, phase_ends)
        ])

        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.value_dim),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming init for linear layers (GELU); zero-init residual block fc2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
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

        # Phase dispatch: sort batch by phase for contiguous head inputs,
        # pad each head's output to full action width, cat, then unsort.
        # No in-place ops — clean autograd graph for training backward pass.
        phases = x[:, :8].argmax(dim=-1)
        order = phases.argsort()
        sorted_h = h[order]
        counts = phases.bincount(minlength=8).tolist()

        phase_logits: list[torch.Tensor] = []
        offset = 0
        for phase_idx, head in enumerate(self.phase_heads):
            n = counts[phase_idx]
            if n == 0:
                continue
            start = self._phase_starts[phase_idx]
            end = self._phase_ends[phase_idx]
            phase_out = head(sorted_h[offset:offset + n])
            padded = F.pad(phase_out, (start, self.cfg.action_dim - end), value=-1e9)
            phase_logits.append(padded)
            offset += n

        policy_logits = torch.cat(phase_logits, dim=0)[order.argsort()]

        values = self.value_head(h)
        return policy_logits, values


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from core.actions import get_action_layout, get_total_action_count
    from core.state import get_layout

    _layout = get_layout(4)
    cfg = RSSModelConfig(input_dim=_layout.visible_size, action_dim=get_total_action_count(4), value_dim=4)
    model = RSSAlphaZeroNet(cfg)
    total = count_parameters(model)
    print(f"Trainable parameters: {total:,}")

    # Parameter breakdown
    preprocess_params = sum(p.numel() for p in model.input_preprocess.parameters())
    block_params = sum(p.numel() for p in model.blocks.parameters())
    norm_params = sum(p.numel() for p in model.trunk_norm.parameters())
    policy_params = sum(p.numel() for p in model.phase_heads.parameters())
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print(f"\nParameter breakdown:")
    print(f"  Input preprocess: {preprocess_params:>12,}  ({preprocess_params/total*100:.1f}%)")
    print(f"  Residual blocks:  {block_params:>12,}  ({block_params/total*100:.1f}%)")
    print(f"  Trunk norm:       {norm_params:>12,}  ({norm_params/total*100:.1f}%)")
    print(f"  Phase heads:      {policy_params:>12,}  ({policy_params/total*100:.1f}%)")
    print(f"  Value head:       {value_params:>12,}  ({value_params/total*100:.1f}%)")

    _phase_names = ["INVEST", "BID", "ACQ", "CLOSE", "DIV", "ISSUE", "IPO", "PAR"]
    for i, (name, head) in enumerate(zip(_phase_names, model.phase_heads)):
        hp = sum(p.numel() for p in head.parameters())
        start = model._phase_starts[i]
        end = model._phase_ends[i]
        print(f"    {name:>6} ({end - start:>2} actions): {hp:>10,}")

    # Smoke test with phase-aware input
    batch_size = 8
    x = torch.randn(batch_size, cfg.input_dim)
    x[:, :8] = 0
    for i in range(batch_size):
        x[i, i % 8] = 1.0

    policy_logits, values = model(x)

    print(f"\npolicy_logits: {tuple(policy_logits.shape)}")
    print(f"values: {tuple(values.shape)}")
    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("all values in [-1, 1]: ok")

    # Verify phase dispatch: only the correct phase slice has real logits
    al = get_action_layout(4)
    starts = [al['invest_start'], al['bid_start'], al['acquisition_start'],
              al['closing_start'], al['dividends_start'], al['issue_start'],
              al['ipo_start'], al['par_start']]
    ends = starts[1:] + [cfg.action_dim]
    for i in range(batch_size):
        phase = i % 8
        for j in range(8):
            if j == phase:
                assert policy_logits[i, starts[j]:ends[j]].max() > -1e8
            else:
                assert (policy_logits[i, starts[j]:ends[j]] <= -1e8).all()
    print("phase dispatch verified: ok")
