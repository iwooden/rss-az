"""3-player model: hidden_dim=256, 8 blocks, 1-layer value head (~4.1M params).

Trainable parameters: 4,127,930
"""

from __future__ import annotations

from dataclasses import dataclass

from nn.template import RSSAlphaZeroNet, count_parameters, run_smoke_test
from nn.template import RSSModelConfig as _BaseConfig

__all__ = ["RSSModelConfig", "RSSAlphaZeroNet", "count_parameters"]


@dataclass(frozen=True)
class RSSModelConfig(_BaseConfig):
    hidden_dim: int = 256
    num_blocks: int = 8
    value_hidden_layers: int = 1


if __name__ == "__main__":
    run_smoke_test(num_players=3, config_cls=RSSModelConfig)
