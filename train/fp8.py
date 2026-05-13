"""Experimental FP8 conversion for transformer trunk matmuls."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FP8TrunkStats:
    """Summary of the transformer trunk modules selected for FP8 training."""

    modules: int
    parameters: int


def _is_fp8_trunk_linear(module: nn.Module, fqn: str) -> bool:
    """Return True for tensor-core-friendly transformer-block Linear layers."""
    if not isinstance(module, nn.Linear):
        return False
    if not fqn.startswith("blocks."):
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def _fp8_trunk_stats(model: nn.Module) -> FP8TrunkStats:
    modules = 0
    parameters = 0
    for name, module in model.named_modules():
        if not _is_fp8_trunk_linear(module, name):
            continue
        modules += 1
        parameters += sum(param.numel() for param in module.parameters())
    return FP8TrunkStats(modules=modules, parameters=parameters)


def convert_transformer_trunk_to_fp8(model: nn.Module) -> FP8TrunkStats:
    """Convert large transformer-block Linear matmuls to TorchAO FP8 training.

    The trainable parameters remain high precision. TorchAO's Float8Linear
    dynamically casts eligible inputs/weights/grad outputs to FP8 for the GEMMs
    while leaving token projections, embeddings, relation bias parameters,
    policy heads, value heads, norms, residual adds, and softmax paths in their
    normal precision.
    """
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise RuntimeError(
            "The experimental fp8 branch requires torchao to be installed in "
            "the active environment."
        ) from exc

    stats = _fp8_trunk_stats(model)
    if stats.modules == 0:
        raise RuntimeError(
            "No transformer trunk Linear layers matched the FP8 filter "
            "(expected modules named like 'blocks.0.qkv_proj')."
        )

    fp8_config = Float8LinearConfig.from_recipe_name("rowwise")
    convert_to_float8_training(
        model,
        config=fp8_config,
        module_filter_fn=_is_fp8_trunk_linear,
    )
    return stats
