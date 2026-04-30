"""Neural network models for Rolling Stock Stars AlphaZero training."""

import torch.nn as nn

from nn.transformer import RSSTransformerNet, TransformerConfig

__all__ = ["create_model"]


def create_model(
    num_players: int,
    *,
    price_slot_fourier_bands: int = 4,
    price_slot_residual_scale: float = 1.0,
) -> nn.Module:
    """Instantiate the transformer for the given player count.

    Post-refactor, a single transformer architecture serves every supported
    player count via ``TransformerConfig.num_players``. Price-slot key
    hyperparameters are exposed here so training configs can sweep the
    Fourier/residual slot-identity mix without editing model code.

    Args:
        num_players: Number of players, 3-5.
    """
    return RSSTransformerNet(
        TransformerConfig(
            num_players=num_players,
            price_slot_fourier_bands=price_slot_fourier_bands,
            price_slot_residual_scale=price_slot_residual_scale,
        )
    )
