"""Neural network models for Rolling Stock Stars AlphaZero training."""

import importlib

import torch.nn as nn

__all__ = ["create_model", "default_model_path"]


def default_model_path(num_players: int) -> str:
    """Return the default model module path for a given player count."""
    return f"nn.model_{num_players}p"


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
        model_path: Dotted Python module path (e.g. "nn.model_3p", "nn.model_4p").
        input_dim: Visible state size (from get_layout).
        action_dim: Action space size (from get_total_action_count).
        value_dim: Number of players.
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
