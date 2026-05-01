"""Neural network models for Rolling Stock Stars AlphaZero training."""

from __future__ import annotations

import torch.nn as nn

from core.resnet_data import get_resnet_vector_size
from core.token_data import TokenDataSize, get_num_tokens
from nn.model_contract import (
    ModelInputSpec,
    ModelKind,
    canonical_player_for_relative,
    normalize_model_type,
    relative_slot_for_canonical,
    rotate_values_to_relative,
    unrotate_values_to_canonical,
)
from nn.resnet import RSSResNet, RSSResNetConfig
from nn.transformer import RSSTransformerNet, TransformerConfig, UNIFIED_LOGIT_DIM

__all__ = [
    "ModelInputSpec",
    "ModelKind",
    "RSSResNet",
    "RSSResNetConfig",
    "RSSTransformerNet",
    "TransformerConfig",
    "canonical_player_for_relative",
    "create_model",
    "get_model_input_spec",
    "relative_slot_for_canonical",
    "rotate_values_to_relative",
    "unrotate_values_to_canonical",
]


def _config_value(config: object, name: str, default: object) -> object:
    return getattr(config, name, default)


def _resnet_input_dim(num_players: int) -> int:
    return get_resnet_vector_size(num_players)


def get_model_input_spec(config: object) -> ModelInputSpec:
    """Resolve model input/output dimensions from a training config."""
    num_players = int(_config_value(config, "num_players", 3))
    model_kind = normalize_model_type(
        str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
    )

    if model_kind is ModelKind.TRANSFORMER:
        return ModelInputSpec(
            model_type=model_kind.value,
            num_players=num_players,
            policy_dim=int(UNIFIED_LOGIT_DIM),
            value_dim=num_players,
            input_dim=None,
            num_tokens=get_num_tokens(num_players),
            token_dim=int(TokenDataSize.TOKEN_DIM),
            uses_relations=True,
            values_are_active_relative=False,
        )

    return ModelInputSpec(
        model_type=model_kind.value,
        num_players=num_players,
        policy_dim=int(UNIFIED_LOGIT_DIM),
        value_dim=num_players,
        input_dim=_resnet_input_dim(num_players),
        num_tokens=None,
        token_dim=None,
        uses_relations=False,
        values_are_active_relative=True,
    )


def create_model(
    config: object | int | None = None,
    *,
    num_players: int | None = None,
    phase_conditioning: bool = True,
    price_slot_fourier_bands: int = 4,
    price_slot_residual_scale: float = 1.0,
) -> nn.Module:
    """Instantiate the configured model family.

    Preferred usage is ``create_model(training_config)``. The legacy
    ``create_model(num_players=..., ...)`` path is retained for targeted tests
    and utility callers that still construct the transformer directly.
    """
    if isinstance(config, int):
        if num_players is not None:
            raise TypeError("Pass num_players either positionally or by keyword, not both")
        num_players = config
        config = None

    if config is None:
        if num_players is None:
            raise TypeError("create_model requires a TrainingConfig or num_players")
        return RSSTransformerNet(
            TransformerConfig(
                num_players=num_players,
                phase_conditioning=phase_conditioning,
                price_slot_fourier_bands=price_slot_fourier_bands,
                price_slot_residual_scale=price_slot_residual_scale,
            )
        )

    cfg_num_players = int(_config_value(config, "num_players", 3))
    model_kind = normalize_model_type(
        str(_config_value(config, "model_type", ModelKind.TRANSFORMER.value))
    )

    if model_kind is ModelKind.TRANSFORMER:
        return RSSTransformerNet(
            TransformerConfig(
                num_players=cfg_num_players,
                phase_conditioning=bool(
                    _config_value(config, "phase_conditioning", True)
                ),
                price_slot_fourier_bands=int(
                    _config_value(config, "price_slot_fourier_bands", 4)
                ),
                price_slot_residual_scale=float(
                    _config_value(config, "price_slot_residual_scale", 1.0)
                ),
            )
        )

    return RSSResNet(
        RSSResNetConfig(
            num_players=cfg_num_players,
            input_dim=_resnet_input_dim(cfg_num_players),
            hidden_dim=int(_config_value(config, "resnet_hidden_dim", 256)),
            num_blocks=int(_config_value(config, "resnet_num_blocks", 8)),
        )
    )
