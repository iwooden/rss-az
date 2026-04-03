"""Shared AlphaZero model template for Rolling Stock Stars.

Residual MLP architecture with phase-specific policy heads and per-player
value output. All player-count variants share this implementation and differ
only in config defaults (hidden_dim, num_blocks, value_hidden_layers).

Architecture:
  - Three-layer input preprocessing (input -> 3*H -> 2*H -> H + LayerNorm)
  - N pre-LN residual blocks (H -> H, no inner expansion, GELU, zero-init fc2)
  - 8 phase-specific policy heads (3 hidden layers each, dispatched by phase one-hot)
  - Value head (configurable depth, tanh output in [-1, 1])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RSSModelConfig:
    """Configuration for the residual MLP trunk and heads.

    Per-model defaults are set in each model file (e.g. model_3p.py).
    The factory in nn/__init__.py instantiates with input_dim/action_dim/value_dim
    and relies on the per-model defaults for everything else.
    """

    input_dim: int  # get_layout(num_players).visible_size
    action_dim: int  # get_total_action_count(num_players)
    value_dim: int  # num_players (per-player expected outcomes)
    hidden_dim: int = 256
    num_blocks: int = 8
    value_hidden_layers: int = 1


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
    """Residual MLP with phase-specific policy heads and per-player value output."""

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
            self._make_head(cfg.hidden_dim, end - start, 3)
            for start, end in zip(phase_starts, phase_ends)
        ])

        self.value_head = nn.Sequential(
            *self._make_head(cfg.hidden_dim, cfg.value_dim, cfg.value_hidden_layers),
            nn.Tanh(),
        )

        self._init_weights()

    @staticmethod
    def _make_head(hidden_dim: int, output_dim: int, num_hidden: int) -> nn.Sequential:
        """Build a head with num_hidden hidden layers (each hidden_dim wide)."""
        layers: list[nn.Module] = []
        for _ in range(num_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

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

        # Phase dispatch by gathering examples for each phase directly on-device.
        phases = x[:, :8].argmax(dim=-1)
        policy_logits: torch.Tensor | None = None
        for phase_idx, head in enumerate(self.phase_heads):
            phase_rows = torch.nonzero(phases == phase_idx, as_tuple=True)[0]
            if phase_rows.numel() == 0:
                continue
            start = self._phase_starts[phase_idx]
            end = self._phase_ends[phase_idx]
            phase_out = head(h.index_select(0, phase_rows))
            if policy_logits is None:
                # Lazy init from phase_out so dtype matches under autocast.
                policy_logits = phase_out.new_full(
                    (h.shape[0], self.cfg.action_dim), -1e9,
                )
            policy_logits[phase_rows, start:end] = phase_out  # pyright: ignore[reportOptionalSubscript]
        assert policy_logits is not None

        values = self.value_head(h)
        return policy_logits, values


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_smoke_test(num_players: int, config_cls: type[RSSModelConfig]) -> None:
    """Standalone smoke test for a model variant. Run via `python -m nn.model_Xp`."""
    from core.actions import get_action_layout, get_total_action_count
    from core.state import get_layout

    layout = get_layout(num_players)
    cfg = config_cls(
        input_dim=layout.visible_size,
        action_dim=get_total_action_count(num_players),
        value_dim=num_players,
    )
    model = RSSAlphaZeroNet(cfg)
    total = count_parameters(model)
    print(f"Model: {num_players}p  (hidden={cfg.hidden_dim}, blocks={cfg.num_blocks}, "
          f"value_layers={cfg.value_hidden_layers})")
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

    phase_names = ["INVEST", "BID", "ACQ", "CLOSE", "DIV", "ISSUE", "IPO", "PAR"]
    for i, (name, head) in enumerate(zip(phase_names, model.phase_heads)):
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
    al = get_action_layout(num_players)
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
