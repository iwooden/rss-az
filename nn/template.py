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

_GELU_APPROXIMATE = "tanh"


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
        self.act = nn.GELU(approximate=_GELU_APPROXIMATE)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return residual + x


class PackedPhaseHeads(nn.Module):
    """Packed phase-specific policy heads for grouped GPU execution.

    Stores the 8 phase heads as batched weight tensors so forward does not need
    to rebuild stacked weights on every call.
    """

    def __init__(self, num_heads: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.w1 = nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim))
        self.b2 = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim))
        self.b3 = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.w4 = nn.Parameter(torch.empty(num_heads, action_dim, hidden_dim))
        self.b4 = nn.Parameter(torch.empty(num_heads, action_dim))

    def __len__(self) -> int:
        return self.num_heads

    @property
    def params_per_head(self) -> int:
        """Trainable parameter count for one phase head."""
        return (
            self.w1[0].numel() + self.b1[0].numel()
            + self.w2[0].numel() + self.b2[0].numel()
            + self.w3[0].numel() + self.b3[0].numel()
            + self.w4[0].numel() + self.b4[0].numel()
        )

    @staticmethod
    def _batched_linear(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    ) -> torch.Tensor:
        out_features = weight.shape[1]
        bias_3d = bias.unsqueeze(1).expand(-1, x.shape[1], out_features)
        return torch.baddbmm(bias_3d, x, weight.transpose(1, 2))

    def forward_packed(self, packed: torch.Tensor) -> torch.Tensor:
        """Run the packed (P, M, H) phase buckets through all 4 layers."""
        packed = self._batched_linear(packed, self.w1, self.b1)
        packed = F.gelu(packed, approximate=_GELU_APPROXIMATE)
        packed = self._batched_linear(packed, self.w2, self.b2)
        packed = F.gelu(packed, approximate=_GELU_APPROXIMATE)
        packed = self._batched_linear(packed, self.w3, self.b3)
        packed = F.gelu(packed, approximate=_GELU_APPROXIMATE)
        return self._batched_linear(packed, self.w4, self.b4)


class RSSAlphaZeroNet(nn.Module):
    """Residual MLP with phase-specific policy heads and per-player value output."""

    def __init__(self, cfg: RSSModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Three-layer input preprocessing with LayerNorm:
        # input_dim -> 3*hidden_dim -> 2*hidden_dim -> hidden_dim -> LN
        self.input_preprocess = nn.Sequential(
            nn.Linear(cfg.input_dim, 3 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROXIMATE),
            nn.Linear(3 * cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROXIMATE),
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

        self.phase_heads = PackedPhaseHeads(
            num_heads=len(phase_starts),
            hidden_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
        )

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
        """Migrate old per-phase Sequential heads into packed head tensors."""
        packed_key = f"{prefix}phase_heads.w1"
        if packed_key in state_dict:
            return

        layer_specs = [
            ("w1", "b1", 0, self.cfg.hidden_dim),
            ("w2", "b2", 2, self.cfg.hidden_dim),
            ("w3", "b3", 4, self.cfg.hidden_dim),
            ("w4", "b4", 6, self.cfg.action_dim),
        ]
        phase_count = len(self._phase_starts)

        for new_w_name, new_b_name, layer_idx, output_dim in layer_specs:
            weights: list[torch.Tensor] = []
            biases: list[torch.Tensor] = []
            old_keys: list[str] = []
            for i, (start, end) in enumerate(zip(self._phase_starts, self._phase_ends)):
                key_w = f"{prefix}phase_heads.{i}.{layer_idx}.weight"
                key_b = f"{prefix}phase_heads.{i}.{layer_idx}.bias"
                old_w = state_dict.get(key_w)
                old_b = state_dict.get(key_b)
                if old_w is None or old_b is None:
                    weights = []
                    biases = []
                    break
                assert isinstance(old_w, torch.Tensor)
                assert isinstance(old_b, torch.Tensor)

                if layer_idx == 6:
                    old_width = end - start
                    if old_w.shape[0] == old_width and old_width < output_dim:
                        expanded_w = torch.zeros(
                            output_dim, old_w.shape[1], dtype=old_w.dtype,
                        )
                        expanded_w[start:end] = old_w
                        old_w = expanded_w

                        expanded_b = torch.zeros(output_dim, dtype=old_b.dtype)
                        expanded_b[start:end] = old_b
                        old_b = expanded_b

                weights.append(old_w)
                biases.append(old_b)
                old_keys.extend([key_w, key_b])

            if len(weights) != phase_count:
                continue

            state_dict[f"{prefix}phase_heads.{new_w_name}"] = torch.stack(weights)  # type: ignore[index]
            state_dict[f"{prefix}phase_heads.{new_b_name}"] = torch.stack(biases)  # type: ignore[index]
            for key in old_keys:
                del state_dict[key]  # type: ignore[misc]

    @staticmethod
    def _make_head(hidden_dim: int, output_dim: int, num_hidden: int) -> nn.Sequential:
        """Build a head with num_hidden hidden layers (each hidden_dim wide)."""
        layers: list[nn.Module] = []
        for _ in range(num_hidden):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(approximate=_GELU_APPROXIMATE),
            ]
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
        for weight in (
            self.phase_heads.w1, self.phase_heads.w2,
            self.phase_heads.w3, self.phase_heads.w4,
        ):
            nn.init.kaiming_uniform_(weight, nonlinearity="relu")
        for bias in (
            self.phase_heads.b1, self.phase_heads.b2,
            self.phase_heads.b3, self.phase_heads.b4,
        ):
            nn.init.zeros_(bias)

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
        """Grouped policy forward using packed per-layer head tensors."""
        B, H = h.shape
        P = len(self.phase_heads)
        A = self.cfg.action_dim

        # Sort by phase for contiguous packing.
        sorted_idx = torch.argsort(phases, stable=True)
        sorted_phases = phases[sorted_idx]
        sorted_h = h[sorted_idx]

        # Count per phase, compute max bucket size.
        counts = torch.bincount(sorted_phases, minlength=P)
        M = torch.sym_int(counts.max())

        # Pack into (P, M, H) — zero-padded for phases with fewer samples.
        packed = sorted_h.new_zeros(P, M, H)
        offsets = counts.cumsum(0).roll(1)
        offsets[0] = 0
        within_phase = torch.arange(B, device=h.device) - offsets[sorted_phases]
        packed[sorted_phases, within_phase] = sorted_h

        packed = self.phase_heads.forward_packed(packed)

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
    for i, name in enumerate(phase_names):
        hp = model.phase_heads.params_per_head
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
