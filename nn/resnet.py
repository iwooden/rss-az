"""Residual MLP model for Rolling Stock Stars AlphaZero training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data import PHASE_ACTION_SIZES
from nn.transformer import PHASES_WITH_PASS_HEAD, UNIFIED_LOGIT_DIM

_GELU_APPROX = "tanh"

_PHASE_OFFSETS: list[int] = [0]
for _size in PHASE_ACTION_SIZES:
    _PHASE_OFFSETS.append(_PHASE_OFFSETS[-1] + int(_size))


@dataclass(frozen=True)
class RSSResNetConfig:
    """Configuration for the residual MLP trunk and dense heads."""

    num_players: int
    input_dim: int
    hidden_dim: int = 256
    num_blocks: int = 8
    value_hidden_layers: int = 1
    input_norm: bool = True

    def __post_init__(self) -> None:
        assert 3 <= self.num_players <= 5, (
            f"num_players must be 3-5, got {self.num_players}"
        )
        assert self.input_dim > 0, f"input_dim must be positive, got {self.input_dim}"
        assert self.hidden_dim > 0, (
            f"hidden_dim must be positive, got {self.hidden_dim}"
        )
        assert self.num_blocks >= 0, (
            f"num_blocks must be >= 0, got {self.num_blocks}"
        )
        assert self.value_hidden_layers >= 0, (
            "value_hidden_layers must be >= 0, "
            f"got {self.value_hidden_layers}"
        )
        assert isinstance(self.input_norm, bool), (
            f"input_norm must be bool, got {self.input_norm!r}"
        )


class ResidualMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h, approximate=_GELU_APPROX)
        h = self.fc2(h)
        return x + h


class RSSResNet(nn.Module):
    """Residual MLP with dense unified policy logits and per-player values."""

    def __init__(self, cfg: RSSResNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        pre_layers: list[nn.Module] = []
        if cfg.input_norm:
            pre_layers.append(nn.LayerNorm(cfg.input_dim))
        pre_layers.extend([
            nn.Linear(cfg.input_dim, 3 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(3 * cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        ])
        self.input_preprocess = nn.Sequential(*pre_layers)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim) for _ in range(cfg.num_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)
        self.policy_head = nn.Linear(cfg.hidden_dim, int(UNIFIED_LOGIT_DIM))
        self.value_head = nn.Sequential(
            *self._make_head(
                cfg.hidden_dim,
                cfg.num_players,
                cfg.value_hidden_layers,
            ),
            nn.Tanh(),
        )

        self._init_weights()

    @staticmethod
    def _make_head(
        hidden_dim: int,
        output_dim: int,
        num_hidden: int,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for _ in range(num_hidden):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(approximate=_GELU_APPROX),
            ])
        layers.append(nn.Linear(hidden_dim, output_dim))
        return layers

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ResNet.

        Args:
            x: ``(batch, input_dim)`` dense ResNet vectors.
            legal_mask: ``(batch, UNIFIED_LOGIT_DIM)`` bool or uint8 dense
                unified legal mask.
            relations: Ignored; accepted so generic call sites can share the
                transformer forward shape.
        """
        del relations
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        if x.ndim != 2:
            raise AssertionError(
                f"ResNet input must be rank 2 or rank 3, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.cfg.input_dim:
            raise AssertionError(
                f"ResNet input width must be {self.cfg.input_dim}, got {x.shape[1]}"
            )
        if legal_mask.shape != (x.shape[0], int(UNIFIED_LOGIT_DIM)):
            raise AssertionError(
                f"legal_mask shape must be {(x.shape[0], int(UNIFIED_LOGIT_DIM))}, "
                f"got {tuple(legal_mask.shape)}"
            )
        if legal_mask.dtype not in (torch.bool, torch.uint8):
            raise AssertionError(
                f"legal_mask must be bool or uint8, got {legal_mask.dtype}"
            )

        h = self.input_preprocess(x)
        for block in self.blocks:
            h = block(h)
        h = self.trunk_norm(h)

        policy_logits = self.policy_head(h).to(torch.float32)
        policy_logits = policy_logits.masked_fill(~legal_mask.to(torch.bool), -1e9)
        values = self.value_head(h)
        return policy_logits, values

    def pass_action_logit_abs(
        self,
        policy_logits: torch.Tensor,
        legal_mask: torch.Tensor,
        phase_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the same pass/action logit-scale diagnostic as the transformer."""
        abs_logits = policy_logits.detach().abs()
        stats: list[torch.Tensor] = []
        for phase in PHASES_WITH_PASS_HEAD:
            offset = _PHASE_OFFSETS[phase]
            size = int(PHASE_ACTION_SIZES[phase])
            pass_slot = offset
            action_start = offset + 1
            action_stop = offset + size

            rows = phase_ids == phase
            pass_legal = legal_mask[:, pass_slot] & rows
            action_legal = legal_mask[:, action_start:action_stop] & rows.unsqueeze(1)

            pass_count = pass_legal.sum().clamp_min(1).to(abs_logits.dtype)
            action_count = action_legal.sum().clamp_min(1).to(abs_logits.dtype)

            pass_sum = (abs_logits[:, pass_slot] * pass_legal.to(abs_logits.dtype)).sum()
            action_sum = (
                abs_logits[:, action_start:action_stop]
                * action_legal.to(abs_logits.dtype)
            ).sum()

            stats.append(pass_sum / pass_count)
            stats.append(action_sum / action_count)
        return torch.stack(stats)

    def phase_mod_diagnostics(self) -> dict[str, float]:
        """ResNet has no phase-modulation parameters."""
        return {}

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.fc2.weight)
            nn.init.zeros_(block.fc2.bias)
