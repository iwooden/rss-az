"""3-player extended model: hidden_dim=256, 10 blocks, 3-layer value head (~4.5M params).

Trainable parameters: 4,523,706
"""

from __future__ import annotations

from dataclasses import dataclass

from nn.template import RSSAlphaZeroNet, count_parameters, run_smoke_test
from nn.template import RSSModelConfig as _BaseConfig

__all__ = ["RSSModelConfig", "RSSAlphaZeroNet", "count_parameters"]


@dataclass(frozen=True)
class RSSModelConfig(_BaseConfig):
    hidden_dim: int = 256
    num_blocks: int = 10
    value_hidden_layers: int = 3


if __name__ == "__main__":
    run_smoke_test(num_players=3, config_cls=RSSModelConfig)
