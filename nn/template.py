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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # Phase-specific policy heads with full-width outputs for grouped execution.
        # Each head outputs the full action_dim; non-phase logits are masked out by
        # the legal action mask before softmax. Uniform shape enables torch.bmm.
        from core.actions import get_action_layout

        num_players = cfg.value_dim
        action_layout = get_action_layout(num_players)
        phase_start_keys = [
            'invest_start', 'bid_start', 'acquisition_start', 'closing_start',
            'dividends_start', 'issue_start', 'ipo_start', 'par_start',
        ]
        phase_starts = [action_layout[k] for k in phase_start_keys]
        phase_ends = phase_starts[1:] + [cfg.action_dim]

        # Phase action boundaries — used by checkpoint migration hook.
        self._phase_starts: list[int] = phase_starts
        self._phase_ends: list[int] = phase_ends

        self.phase_heads = nn.ModuleList([
            self._make_head(cfg.hidden_dim, cfg.action_dim, 3)
            for _ in range(len(phase_starts))
        ])

        self.value_head = nn.Sequential(
            *self._make_head(cfg.hidden_dim, cfg.value_dim, cfg.value_hidden_layers),
            nn.Tanh(),
        )

        self._init_weights()
        self.register_load_state_dict_pre_hook(self._migrate_phase_heads_hook)

    def _migrate_phase_heads_hook(
        self,
        _module: nn.Module,
        state_dict: Mapping[str, Any],
        prefix: str,
        *_args: object,
        **_kwargs: object,
    ) -> None:
        """Expand narrow per-phase output layers to full action_dim in-place."""
        for i, (start, end) in enumerate(
            zip(self._phase_starts, self._phase_ends)
        ):
            key_w = f"{prefix}phase_heads.{i}.6.weight"
            key_b = f"{prefix}phase_heads.{i}.6.bias"
            old_w = state_dict.get(key_w)
            if old_w is None or not isinstance(old_w, torch.Tensor):
                continue
            old_width = end - start
            if old_w.shape[0] == old_width and old_width < self.cfg.action_dim:
                new_w = torch.zeros(
                    self.cfg.action_dim, old_w.shape[1], dtype=old_w.dtype,
                )
                new_w[start:end] = old_w
                state_dict[key_w] = new_w  # type: ignore[index]
                old_b = state_dict[key_b]
                assert isinstance(old_b, torch.Tensor)
                new_b = torch.zeros(self.cfg.action_dim, dtype=old_b.dtype)
                new_b[start:end] = old_b
                state_dict[key_b] = new_b  # type: ignore[index]

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

        phases = x[:, :8].argmax(dim=-1)
        policy_logits = self._grouped_policy_forward(h, phases)

        values = self.value_head(h)
        return policy_logits, values

    def _grouped_policy_forward(
        self, h: torch.Tensor, phases: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped bmm policy forward. Packs by phase, runs 4 batched matmuls."""
        B, H = h.shape
        P = len(self.phase_heads)
        A = self.cfg.action_dim
        heads = [self.phase_heads[p] for p in range(P)]

        # Stack weights from individual heads for batched matmul.
        w1 = torch.stack([heads[p][0].weight for p in range(P)])  # type: ignore[index]
        b1 = torch.stack([heads[p][0].bias for p in range(P)])  # type: ignore[index]
        w2 = torch.stack([heads[p][2].weight for p in range(P)])  # type: ignore[index]
        b2 = torch.stack([heads[p][2].bias for p in range(P)])  # type: ignore[index]
        w3 = torch.stack([heads[p][4].weight for p in range(P)])  # type: ignore[index]
        b3 = torch.stack([heads[p][4].bias for p in range(P)])  # type: ignore[index]
        w4 = torch.stack([heads[p][6].weight for p in range(P)])  # type: ignore[index]
        b4 = torch.stack([heads[p][6].bias for p in range(P)])  # type: ignore[index]

        # Sort by phase for contiguous packing.
        sorted_idx = torch.argsort(phases, stable=True)
        sorted_phases = phases[sorted_idx]
        sorted_h = h[sorted_idx]

        # Count per phase, compute max bucket size.
        counts = torch.bincount(sorted_phases, minlength=P)
        M = int(counts.max().item())

        if M == 0:
            return h.new_full((B, A), -1e9)

        # Pack into (P, M, H) — zero-padded for phases with fewer samples.
        packed = sorted_h.new_zeros(P, M, H)
        offsets = counts.cumsum(0).roll(1)
        offsets[0] = 0
        within_phase = torch.arange(B, device=h.device) - offsets[sorted_phases]
        packed[sorted_phases, within_phase] = sorted_h

        # 4 layers via batched matmul + GELU.
        packed = torch.bmm(packed, w1.transpose(1, 2)) + b1.unsqueeze(1)
        packed = F.gelu(packed)
        packed = torch.bmm(packed, w2.transpose(1, 2)) + b2.unsqueeze(1)
        packed = F.gelu(packed)
        packed = torch.bmm(packed, w3.transpose(1, 2)) + b3.unsqueeze(1)
        packed = F.gelu(packed)
        packed = torch.bmm(packed, w4.transpose(1, 2)) + b4.unsqueeze(1)

        # Unpack back to (B, A). Init from packed so dtype follows autocast.
        policy_logits = packed.new_full((B, A), -1e9)
        policy_logits[sorted_idx] = packed[sorted_phases, within_phase]
        return policy_logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_smoke_test(num_players: int, config_cls: type[RSSModelConfig]) -> None:
    """Standalone smoke test for a model variant. Run via `python -m nn.model_Xp`."""
    from core.actions import get_total_action_count
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

    assert torch.isfinite(policy_logits).all(), "non-finite logits"
    print("all logits finite: ok")
