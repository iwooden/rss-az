"""5-player model: hidden_dim=448, 15 blocks, 3-layer value head (~14.9M params).

Trainable parameters: 14,873,818
"""

from __future__ import annotations

from dataclasses import dataclass

from nn.template import RSSAlphaZeroNet, count_parameters, run_smoke_test
from nn.template import RSSModelConfig as _BaseConfig

__all__ = ["RSSModelConfig", "RSSAlphaZeroNet", "count_parameters"]


@dataclass(frozen=True)
class RSSModelConfig(_BaseConfig):
    hidden_dim: int = 448
    num_blocks: int = 15
    value_hidden_layers: int = 3


if __name__ == "__main__":
    run_smoke_test(num_players=5, config_cls=RSSModelConfig)
