"""Neural network models for Rolling Stock Stars AlphaZero training."""

import importlib

import torch.nn as nn

from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig, count_parameters

__all__ = [
    "RSSAlphaZeroNet", "RSSModelConfig",
    "count_parameters", "create_model",
]

# Default model module path (used by --model-path CLI arg)
DEFAULT_MODEL_PATH = "nn.model_3p"


def create_model(
    model_path: str,
    input_dim: int,
    action_dim: int,
    value_dim: int,
) -> nn.Module:
    """Factory: instantiate a model by module path.

    The module must export ``RSSModelConfig`` (a dataclass accepting at least
    ``input_dim``, ``action_dim``, ``value_dim``) and ``RSSAlphaZeroNet``
    (an ``nn.Module`` subclass that takes the config).

    Args:
        model_path: Dotted Python module path (e.g. "nn.model_3p").
        input_dim: Visible state size (e.g. 1018 for 3 players).
        action_dim: Action space size (from get_total_action_count).
        value_dim: Number of players (e.g. 3).
    """
    mod = importlib.import_module(model_path)

    config_cls = getattr(mod, "RSSModelConfig", None)
    if config_cls is None:
        raise ValueError(
            f"Module {model_path!r} does not export 'RSSModelConfig'"
        )

    net_cls = getattr(mod, "RSSAlphaZeroNet", None)
    if net_cls is None:
        raise ValueError(
            f"Module {model_path!r} does not export 'RSSAlphaZeroNet'"
        )

    cfg = config_cls(input_dim=input_dim, action_dim=action_dim, value_dim=value_dim)
    return net_cls(cfg)
