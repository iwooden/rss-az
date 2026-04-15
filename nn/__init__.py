"""Neural network models for Rolling Stock Stars AlphaZero training."""

import torch.nn as nn

from nn.transformer import RSSTransformerNet, TransformerConfig

__all__ = ["create_model"]


def create_model(num_players: int) -> nn.Module:
    """Instantiate the transformer for the given player count.

    Post-refactor, a single transformer architecture serves every supported
    player count via ``TransformerConfig.num_players``. Any other config
    fields keep their declared defaults — override by subclassing if needed.

    Args:
        num_players: Number of players, 3-5.
    """
    return RSSTransformerNet(TransformerConfig(num_players=num_players))
