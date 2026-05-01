"""Residual MLP model for Rolling Stock Stars AlphaZero training."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.data import PHASE_ACTION_SIZES, DecisionPhase
from core.resnet_data import get_resnet_vector_size
from nn.transformer import PHASES_WITH_PASS_HEAD, UNIFIED_LOGIT_DIM, build_action_lut

_GELU_APPROX = "tanh"
NUM_PHASES = len(DecisionPhase)
_PHASE_NAMES: list[str] = [
    "INVEST", "BID", "ACQ_SELECT_CORP", "ACQ_OFFER",
    "CLOSING", "DIVIDENDS", "ISSUE", "IPO", "PAR",
    "ACQ_SELECT_COMPANY", "ACQ_SELECT_PRICE",
]

_PHASE_OFFSETS: list[int] = [0]
for _size in PHASE_ACTION_SIZES:
    _PHASE_OFFSETS.append(_PHASE_OFFSETS[-1] + int(_size))


@dataclass(frozen=True)
class RSSResNetConfig:
    """Configuration for the residual MLP trunk and dense heads."""

    num_players: int
    input_dim: int
    hidden_dim: int = 256
    num_blocks: int = 10

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

        self.input_preprocess = nn.Sequential(
            nn.Linear(cfg.input_dim, 3 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(3 * cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(cfg.hidden_dim) for _ in range(cfg.num_blocks)]
        )
        self.trunk_norm = nn.LayerNorm(cfg.hidden_dim)
        self.policy_heads = nn.ModuleList(
            [
                nn.Sequential(*self._make_head(cfg.hidden_dim, int(phase_size)))
                for phase_size in PHASE_ACTION_SIZES
            ]
        )
        self.value_head = nn.Sequential(
            *self._make_head(cfg.hidden_dim, cfg.num_players),
            nn.Tanh(),
        )

        self._validate_policy_layout()
        self._init_weights()

    @staticmethod
    def _make_head(
        hidden_dim: int,
        output_dim: int,
    ) -> list[nn.Module]:
        return [
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(hidden_dim, output_dim),
        ]

    @staticmethod
    def _head_output_width(head: nn.Module) -> int:
        if not isinstance(head, nn.Sequential):
            raise AssertionError(f"policy head must be nn.Sequential, got {type(head)}")
        final = head[-1]
        if not isinstance(final, nn.Linear):
            raise AssertionError(
                f"policy head final layer must be nn.Linear, got {type(final)}"
            )
        return final.out_features

    def _validate_policy_layout(self) -> None:
        """Validate per-phase policy head widths against the unified layout.

        The ResNet runs every phase head for every row, then concatenates their
        outputs in ``DecisionPhase`` order. The legal mask determines which
        phase-local block is live for each state, matching the transformer's
        static unified-logit contract.
        """
        if len(PHASE_ACTION_SIZES) != NUM_PHASES:
            raise AssertionError(
                f"PHASE_ACTION_SIZES has {len(PHASE_ACTION_SIZES)} entries, "
                f"expected {NUM_PHASES}"
            )
        if len(self.policy_heads) != NUM_PHASES:
            raise AssertionError(
                f"policy_heads has {len(self.policy_heads)} entries, "
                f"expected {NUM_PHASES}"
            )

        widths = [self._head_output_width(head) for head in self.policy_heads]
        expected = [int(size) for size in PHASE_ACTION_SIZES]
        if widths != expected:
            raise AssertionError(
                f"policy head widths {widths} do not match "
                f"PHASE_ACTION_SIZES {expected}"
            )
        if sum(widths) != int(UNIFIED_LOGIT_DIM):
            raise AssertionError(
                f"policy head total {sum(widths)} != UNIFIED_LOGIT_DIM "
                f"{int(UNIFIED_LOGIT_DIM)}"
            )

    def _build_unified_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Run all phase heads and concatenate their phase-local logits."""
        return torch.cat([head(h) for head in self.policy_heads], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        relations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ResNet.

        Args:
            x: ``(batch, input_dim)`` dense ResNet vectors.
            legal_mask: ``(batch, UNIFIED_LOGIT_DIM)`` bool dense unified
                legal mask.
            relations: Ignored; accepted so generic call sites can share the
                transformer forward shape.
        """
        del relations
        if x.ndim != 2:
            raise AssertionError(
                f"ResNet input must be rank 2, got shape {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise AssertionError(f"x must be floating-point features, got {x.dtype}")
        if x.shape[1] != self.cfg.input_dim:
            raise AssertionError(
                f"ResNet input width must be {self.cfg.input_dim}, got {x.shape[1]}"
            )
        if legal_mask.dtype != torch.bool:
            raise AssertionError(f"legal_mask must be bool, got {legal_mask.dtype}")
        if legal_mask.ndim != 2:
            raise AssertionError(
                f"legal_mask must be rank 2, got shape {tuple(legal_mask.shape)}"
            )
        if legal_mask.shape[1] != int(UNIFIED_LOGIT_DIM):
            raise AssertionError(
                f"legal_mask width must be {int(UNIFIED_LOGIT_DIM)}, "
                f"got {legal_mask.shape[1]}"
            )
        if (
            not torch.compiler.is_compiling()
            and legal_mask.shape[0] != x.shape[0]
        ):
            raise AssertionError(
                f"legal_mask batch must be {x.shape[0]}, got {legal_mask.shape[0]}"
            )
        if legal_mask.device != x.device:
            raise AssertionError(
                f"legal_mask device must match x device; got "
                f"{legal_mask.device} vs {x.device}"
            )

        h = self.input_preprocess(x)
        for block in self.blocks:
            h = block(h)
        h = self.trunk_norm(h)

        policy_logits = self._build_unified_logits(h).to(torch.float32)
        policy_logits = policy_logits.masked_fill(~legal_mask, -1e9)
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


def count_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _module_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_breakdown(model: RSSResNet) -> dict[str, int]:
    """Return trainable ResNet parameters grouped by architectural component."""
    return {
        "Input preprocessing": _module_parameters(model.input_preprocess),
        "Residual blocks": _module_parameters(model.blocks),
        "Trunk norm": _module_parameters(model.trunk_norm),
        "Policy heads": _module_parameters(model.policy_heads),
        "Value head": _module_parameters(model.value_head),
    }


def policy_head_breakdown(model: RSSResNet) -> dict[str, int]:
    """Return trainable policy-head parameters by decision phase."""
    return {
        name: _module_parameters(head)
        for name, head in zip(_PHASE_NAMES, model.policy_heads, strict=True)
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = RSSResNetConfig(
        num_players=3,
        input_dim=get_resnet_vector_size(3),
    )
    model = RSSResNet(cfg)
    model.eval()

    total = count_parameters(model)
    print("RSSResNet")
    print(f"  num_players={cfg.num_players}")
    print(
        f"  input_dim={cfg.input_dim}, hidden_dim={cfg.hidden_dim}, "
        f"blocks={cfg.num_blocks}"
    )
    print(f"  Trainable parameters: {total:,}")
    print()

    print("Parameter breakdown:")
    for name, count in parameter_breakdown(model).items():
        print(f"  {name + ':':22s} {count:>10,}  ({count / total * 100:.1f}%)")

    print()
    print("Policy head breakdown:")
    for name, count in policy_head_breakdown(model).items():
        print(f"  {name + ':':22s} {count:>10,}  ({count / total * 100:.1f}%)")

    print()
    for name, size in zip(_PHASE_NAMES, PHASE_ACTION_SIZES, strict=True):
        print(f"  {name:>18s}: {int(size):>3d} actions")

    # --- Smoke test ---
    print()
    batch_size = NUM_PHASES
    x = torch.randn(batch_size, cfg.input_dim)

    lut = build_action_lut()
    legal_mask = torch.zeros(batch_size, int(UNIFIED_LOGIT_DIM), dtype=torch.bool)
    for i in range(NUM_PHASES):
        n = int(PHASE_ACTION_SIZES[i])
        legal_mask[i, lut[i, :n]] = True

    policy_logits, values = model(x, legal_mask)

    print(f"policy_logits: {tuple(policy_logits.shape)}")
    print(f"values:        {tuple(values.shape)}")

    assert policy_logits.shape == (batch_size, int(UNIFIED_LOGIT_DIM))
    assert values.shape == (batch_size, cfg.num_players)

    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("values in [-1, 1]: ok")

    for i in range(NUM_PHASES):
        legal = policy_logits[i][legal_mask[i]]
        illegal = policy_logits[i][~legal_mask[i]]
        assert torch.isfinite(legal).all(), (
            f"{_PHASE_NAMES[i]}: non-finite legal logits"
        )
        if illegal.numel() > 0:
            assert (illegal == -1e9).all(), (
                f"{_PHASE_NAMES[i]}: leak into illegal slots"
            )
    print("per-row legal mask: ok")

    print("\nSmoke test passed.")
