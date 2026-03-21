"""Neural network models for Rolling Stock Stars AlphaZero training."""

import torch.nn as nn

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig, count_parameters
from nn.model_3p_2 import RSSAlphaZeroNet2, RSSModelConfig2

__all__ = [
    "RSSAlphaZeroNet", "RSSModelConfig",
    "RSSAlphaZeroNet2", "RSSModelConfig2",
    "count_parameters", "create_model",
]


def create_model(
    arch: str,
    input_dim: int,
    action_dim: int,
    value_dim: int,
) -> nn.Module:
    """Factory: instantiate a model by architecture name.

    Args:
        arch: "v1" (model_3p, ~26.6M params) or "v2" (model_3p_2, ~6.6M params).
        input_dim: Visible state size (e.g. 1763 for 3 players).
        action_dim: Action space size (e.g. 226 for 3 players).
        value_dim: Number of players (e.g. 3).
    """
    if arch == "v1":
        cfg = RSSModelConfig(
            input_dim=input_dim, action_dim=action_dim, value_dim=value_dim,
        )
        return RSSAlphaZeroNet(cfg)
    if arch == "v2":
        cfg2 = RSSModelConfig2(
            input_dim=input_dim, action_dim=action_dim, value_dim=value_dim,
        )
        return RSSAlphaZeroNet2(cfg2)
    raise ValueError(f"Unknown model arch: {arch!r}. Must be 'v1' or 'v2'.")
